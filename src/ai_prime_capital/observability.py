from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response

LOGGER = logging.getLogger("ai_prime_capital.http")
REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


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
