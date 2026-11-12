from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger("ai_prime_capital.http")
REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_DEFAULT_REQUEST_BYTES = 50 * 1024 * 1024


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
    limit = getattr(settings, "max_request_bytes", _DEFAULT_REQUEST_BYTES)
    return limit if isinstance(limit, int) and limit > 0 else _DEFAULT_REQUEST_BYTES


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
        headers={REQUEST_ID_HEADER: request_id},
    )


def install_observability(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = _safe_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        started = perf_counter()

        try:
            content_length = request.headers.get("content-length")
            response: Response | None = None
            if content_length is not None:
                try:
                    declared_bytes = int(content_length)
                except ValueError:
                    response = _request_limit_problem(
                        request,
                        status_code=status.HTTP_400_BAD_REQUEST,
                        title="Invalid request",
                        detail="The request could not be accepted.",
                        code="request.invalid_content_length",
                        request_id=request_id,
                    )
                else:
                    if declared_bytes < 0 or declared_bytes > _request_limit(request):
                        response = _request_limit_problem(
                            request,
                            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                            title="Request exceeds limits",
                            detail="The request body exceeds the configured limit.",
                            code="request.body_too_large",
                            request_id=request_id,
                        )
            if response is None:
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
