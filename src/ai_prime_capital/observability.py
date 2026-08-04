from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LOGGER = logging.getLogger("ai_prime_capital.http")
REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_DEFAULT_REQUEST_BYTES = 50 * 1024 * 1024
_DEFAULT_QUESTION_CHARS = 4_000
_JSON_ENVELOPE_BYTES = 1_024
_MAX_JSON_ESCAPED_CHAR_BYTES = 12
_MULTIPART_OVERHEAD_BYTES = 1024 * 1024


class RequestBodyTooLarge(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="request body exceeds the configured limit",
        )


def _safe_request_id(value: str | None) -> str:
    if value is not None and _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return uuid4().hex


def get_request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if isinstance(request_id, str) and _REQUEST_ID_PATTERN.fullmatch(request_id):
        return request_id
    request_id = _safe_request_id(request.headers.get(REQUEST_ID_HEADER))
    request.state.request_id = request_id
    return request_id


def _request_limit(request: Request) -> int:
    resources = getattr(request.app.state, "resources", None)
    settings = getattr(resources, "settings", None)
    configured = getattr(settings, "max_request_bytes", _DEFAULT_REQUEST_BYTES)
    base_limit = (
        configured if isinstance(configured, int) and configured > 0 else _DEFAULT_REQUEST_BYTES
    )

    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        # max_request_bytes owns the aggregate file-content budget. Permit a
        # bounded amount for multipart boundaries and per-part headers while
        # retaining a transport-level ceiling before Starlette parses files.
        return base_limit + _MULTIPART_OVERHEAD_BYTES

    if request.url.path == "/api/v1/questions":
        question_chars = getattr(settings, "max_question_chars", _DEFAULT_QUESTION_CHARS)
        if not isinstance(question_chars, int) or question_chars < 1:
            question_chars = _DEFAULT_QUESTION_CHARS
        return min(
            base_limit,
            question_chars * _MAX_JSON_ESCAPED_CHAR_BYTES + _JSON_ENVELOPE_BYTES,
        )

    return base_limit


def _request_limit_problem(
    request: Request,
    *,
    status_code: int,
    title: str,
    detail: str,
    code: str,
    request_id: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "about:blank",
            "title": title,
            "status": status_code,
            "detail": detail,
            "instance": request.url.path,
            "code": code,
            "request_id": request_id,
        },
        media_type="application/problem+json",
        headers={REQUEST_ID_HEADER: request_id, "Connection": "close"},
    )


class RequestBodyLimitMiddleware:
    """Bound declared and streamed HTTP bodies before route-level parsing."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        request_id = get_request_id(request)
        limit = _request_limit(request)
        content_length = request.headers.get("content-length")

        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                await self._send_problem(
                    request,
                    scope,
                    receive,
                    send,
                    status_code=status.HTTP_400_BAD_REQUEST,
                    title="Invalid request",
                    detail="The request could not be accepted.",
                    code="request.invalid_content_length",
                    request_id=request_id,
                )
                return
            if declared_bytes < 0 or declared_bytes > limit:
                await self._send_too_large(request, scope, receive, send, request_id)
                return

        received_bytes = 0

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > limit:
                    raise RequestBodyTooLarge
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _send_too_large(
        request: Request,
        scope: Scope,
        receive: Receive,
        send: Send,
        request_id: str,
    ) -> None:
        await RequestBodyLimitMiddleware._send_problem(
            request,
            scope,
            receive,
            send,
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            title="Request exceeds limits",
            detail="The request body exceeds the configured limit.",
            code="request.body_too_large",
            request_id=request_id,
        )

    @staticmethod
    async def _send_problem(
        request: Request,
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        status_code: int,
        title: str,
        detail: str,
        code: str,
        request_id: str,
    ) -> None:
        response = _request_limit_problem(
            request,
            status_code=status_code,
            title=title,
            detail=detail,
            code=code,
            request_id=request_id,
        )
        await response(scope, receive, send)


def install_observability(app: FastAPI) -> None:
    app.add_middleware(RequestBodyLimitMiddleware)

    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = _safe_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        started = perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            LOGGER.error(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "operation": request.url.path,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                    "error_type": "unhandled",
                },
            )
            raise

        response.headers.setdefault(REQUEST_ID_HEADER, request_id)
        LOGGER.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "operation": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        return response
