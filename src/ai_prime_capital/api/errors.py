from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..application.errors import AdapterError, AdapterErrorCode
from ..infrastructure.extractors.base import UploadErrorCode, UploadRejected
from ..observability import RequestBodyTooLarge, get_request_id
from .schemas import ProblemDetails

LOGGER = logging.getLogger("ai_prime_capital.api")


_UPLOAD_STATUS: dict[UploadErrorCode, int] = {
    UploadErrorCode.UNSUPPORTED_TYPE: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    UploadErrorCode.MEDIA_TYPE_MISMATCH: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    UploadErrorCode.TOO_MANY_FILES: status.HTTP_413_CONTENT_TOO_LARGE,
    UploadErrorCode.FILE_TOO_LARGE: status.HTTP_413_CONTENT_TOO_LARGE,
    UploadErrorCode.REQUEST_TOO_LARGE: status.HTTP_413_CONTENT_TOO_LARGE,
    UploadErrorCode.PACKAGE_TOO_MANY_ENTRIES: status.HTTP_413_CONTENT_TOO_LARGE,
    UploadErrorCode.PACKAGE_TOO_LARGE: status.HTTP_413_CONTENT_TOO_LARGE,
    UploadErrorCode.PACKAGE_COMPRESSION_RATIO: status.HTTP_413_CONTENT_TOO_LARGE,
    UploadErrorCode.EXTRACTED_TEXT_TOO_LARGE: status.HTTP_413_CONTENT_TOO_LARGE,
}

_ADAPTER_STATUS: dict[AdapterErrorCode, int] = {
    AdapterErrorCode.EMBEDDING_INVALID_INPUT: status.HTTP_422_UNPROCESSABLE_CONTENT,
    AdapterErrorCode.QUESTION_INVALID_INPUT: status.HTTP_422_UNPROCESSABLE_CONTENT,
    AdapterErrorCode.GENERATION_INVALID_INPUT: status.HTTP_422_UNPROCESSABLE_CONTENT,
    AdapterErrorCode.VECTOR_INVALID_INPUT: status.HTTP_422_UNPROCESSABLE_CONTENT,
    AdapterErrorCode.INGESTION_INVALID_INPUT: status.HTTP_422_UNPROCESSABLE_CONTENT,
    AdapterErrorCode.EMBEDDING_UPSTREAM: status.HTTP_503_SERVICE_UNAVAILABLE,
    AdapterErrorCode.GENERATION_UPSTREAM: status.HTTP_503_SERVICE_UNAVAILABLE,
    AdapterErrorCode.VECTOR_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    AdapterErrorCode.VECTOR_INCOMPATIBLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    AdapterErrorCode.EMBEDDING_DIMENSION: status.HTTP_502_BAD_GATEWAY,
    AdapterErrorCode.EMBEDDING_RESPONSE: status.HTTP_502_BAD_GATEWAY,
    AdapterErrorCode.GENERATION_RESPONSE: status.HTTP_502_BAD_GATEWAY,
    AdapterErrorCode.VECTOR_RESPONSE: status.HTTP_502_BAD_GATEWAY,
    AdapterErrorCode.VECTOR_WRITE: status.HTTP_502_BAD_GATEWAY,
    AdapterErrorCode.INGESTION_RESPONSE: status.HTTP_502_BAD_GATEWAY,
}


def _problem(
    request: Request,
    *,
    status_code: int,
    title: str,
    detail: str,
    code: str,
) -> JSONResponse:
    request_id = get_request_id(request)
    body = ProblemDetails(
        status=status_code,
        title=title,
        detail=detail,
        instance=request.url.path,
        code=code,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        media_type="application/problem+json",
        headers={"X-Request-ID": request_id},
    )


def _upload_message(code: UploadErrorCode, status_code: int) -> tuple[str, str]:
    if status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE:
        return "Unsupported upload type", "The uploaded file type is not supported."
    if status_code == status.HTTP_413_CONTENT_TOO_LARGE:
        return "Upload exceeds limits", "The uploaded content exceeds the configured limits."
    if code is UploadErrorCode.NO_FILES:
        return "No files supplied", "At least one supported file is required."
    return "Invalid upload", "The uploaded file could not be accepted."


def _adapter_message(code: AdapterErrorCode, status_code: int) -> tuple[str, str]:
    if status_code == status.HTTP_422_UNPROCESSABLE_CONTENT:
        return "Invalid request", "The request could not be accepted."
    if status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        return "Dependency unavailable", "A required dependency is unavailable."
    return "Upstream service failure", "A required upstream service returned an invalid response."


async def handle_upload_rejected(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, UploadRejected):
        return await handle_unexpected_error(request, exc)
    status_code = _UPLOAD_STATUS.get(exc.code, status.HTTP_422_UNPROCESSABLE_CONTENT)
    title, detail = _upload_message(exc.code, status_code)
    return _problem(
        request,
        status_code=status_code,
        title=title,
        detail=detail,
        code=str(exc.code),
    )


async def handle_adapter_error(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, AdapterError):
        return await handle_unexpected_error(request, exc)
    status_code = _ADAPTER_STATUS.get(exc.code, status.HTTP_503_SERVICE_UNAVAILABLE)
    title, detail = _adapter_message(exc.code, status_code)
    return _problem(
        request,
        status_code=status_code,
        title=title,
        detail=detail,
        code=str(exc.code),
    )


async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        return await handle_unexpected_error(request, exc)
    return _problem(
        request,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        title="Request validation failed",
        detail="The request could not be validated.",
        code="request.invalid",
    )


async def handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, HTTPException):
        return await handle_unexpected_error(request, exc)
    if isinstance(exc, RequestBodyTooLarge):
        return _problem(
            request,
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            title="Request exceeds limits",
            detail="The request body exceeds the configured limit.",
            code="request.body_too_large",
        )
    status_code = exc.status_code if 400 <= exc.status_code <= 599 else 500
    return _problem(
        request,
        status_code=status_code,
        title="HTTP request failed",
        detail="The HTTP request could not be completed.",
        code=f"http.{status_code}",
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # Log only classification and safe request context; exception text can carry
    # document content, prompts, or credentials supplied by an adapter.
    LOGGER.error(
        "unexpected_request_error",
        extra={
            "request_id": get_request_id(request),
            "operation": request.url.path,
            "error_type": type(exc).__name__,
        },
    )
    return _problem(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        title="Internal server error",
        detail="The server could not complete the request.",
        code="internal.error",
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(UploadRejected, handle_upload_rejected)
    app.add_exception_handler(AdapterError, handle_adapter_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(HTTPException, handle_http_exception)
    app.add_exception_handler(Exception, handle_unexpected_error)
