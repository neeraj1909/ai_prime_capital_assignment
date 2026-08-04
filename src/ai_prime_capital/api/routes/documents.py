from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, File, UploadFile, status

from ...application.ingest_documents import RawUpload
from ...infrastructure.extractors.base import IncomingUpload
from ..dependencies import IngestDependency
from ..schemas import (
    DOCUMENT_UPLOAD_EXAMPLE,
    PROBLEM_EXAMPLE,
    DocumentUploadResponse,
    problem_response_examples,
    problem_response_spec,
)

router = APIRouter(prefix="/api/v1", tags=["documents"])


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index documents",
    responses={
        201: {
            "description": "Documents were validated, extracted, embedded, and indexed.",
            "content": {"application/json": {"example": DOCUMENT_UPLOAD_EXAMPLE}},
        },
        413: problem_response_examples(
            {
                "request_body_too_large": {
                    **PROBLEM_EXAMPLE,
                    "status": 413,
                    "title": "Request exceeds limits",
                    "detail": "The request body exceeds the configured limit.",
                    "instance": "/api/v1/documents",
                    "code": "request.body_too_large",
                },
                "uploaded_file_too_large": {
                    **PROBLEM_EXAMPLE,
                    "status": 413,
                    "title": "Upload exceeds limits",
                    "detail": "The uploaded content exceeds the configured limits.",
                    "instance": "/api/v1/documents",
                    "code": "upload.file_too_large",
                },
            }
        ),
        415: problem_response_spec(
            PROBLEM_EXAMPLE | {"status": 415, "code": "upload.unsupported_type"}
        ),
        422: problem_response_spec(PROBLEM_EXAMPLE | {"code": "upload.invalid_document"}),
        503: problem_response_spec(
            PROBLEM_EXAMPLE
            | {"status": 503, "title": "Dependency unavailable", "code": "vector.unavailable"}
        ),
    },
)
async def upload_documents(
    files: Annotated[
        list[UploadFile],
        File(description="One or more PDF, DOCX, or UTF-8 TXT files."),
    ],
    service: IngestDependency,
) -> DocumentUploadResponse:
    uploads = tuple(
        IncomingUpload(
            client_name=file.filename or "",
            declared_media_type=file.content_type,
            stream=file.file,
        )
        for file in files
    )
    result = await service.execute(cast(tuple[RawUpload, ...], uploads))
    return DocumentUploadResponse.from_domain(result)
