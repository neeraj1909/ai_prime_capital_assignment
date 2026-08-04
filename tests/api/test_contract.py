from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ai_prime_capital.application.errors import AdapterError, AdapterErrorCode
from ai_prime_capital.application.ingest_documents import (
    IndexedDocument,
    IngestionResult,
)
from ai_prime_capital.bootstrap import Readiness, RuntimeResources
from ai_prime_capital.config import Settings
from ai_prime_capital.domain.grounding import NO_RELEVANT_INFORMATION
from ai_prime_capital.domain.models import GroundedAnswer, SourceLocator, SourceReference
from ai_prime_capital.infrastructure.extractors.base import UploadErrorCode, UploadRejected
from ai_prime_capital.main import create_app

pytestmark = pytest.mark.api


def build_settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-openai-secret",
        openai_chat_model="test-chat-model",
        embedding_dimensions=3,
        **overrides,
    )


class RecordingIngest:
    def __init__(self, result: IngestionResult | Exception) -> None:
        self.result = result
        self.calls: list[tuple[object, ...]] = []

    async def execute(self, uploads: Sequence[object]) -> IngestionResult:
        self.calls.append(tuple(uploads))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class RecordingAnswer:
    def __init__(self, result: GroundedAnswer | Exception) -> None:
        self.result = result
        self.calls: list[tuple[str, int | None]] = []

    async def execute(self, question: str, *, top_k: int | None = None) -> GroundedAnswer:
        self.calls.append((question, top_k))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def indexed_result() -> IngestionResult:
    return IngestionResult(
        documents=(
            IndexedDocument(
                document_id="doc-1",
                file_name="facts.txt",
                media_type="text/plain",
                chunk_count=2,
            ),
        )
    )


def grounded_result() -> GroundedAnswer:
    return GroundedAnswer(
        answer="Revenue increased.",
        grounded=True,
        sources=(
            SourceReference(
                document_id="doc-1",
                file_name="facts.txt",
                chunk_id="chunk-1",
                snippet="Revenue increased.",
                score=0.91,
                locator=SourceLocator(start_offset=0, end_offset=19),
            ),
        ),
    )


def fallback_result() -> GroundedAnswer:
    return GroundedAnswer(
        answer=NO_RELEVANT_INFORMATION,
        grounded=False,
        sources=(),
    )


def make_client(
    *,
    ingest: RecordingIngest | None = None,
    answer: RecordingAnswer | None = None,
) -> tuple[TestClient, RecordingIngest, RecordingAnswer]:
    selected_ingest = ingest or RecordingIngest(indexed_result())
    selected_answer = answer or RecordingAnswer(grounded_result())
    resources = RuntimeResources(
        settings=build_settings(),
        readiness=Readiness(ready=True),
        ingest_documents=selected_ingest,  # type: ignore[arg-type]
        answer_question=selected_answer,  # type: ignore[arg-type]
    )

    async def factory(_settings: Settings) -> RuntimeResources:
        return resources

    return (
        TestClient(create_app(settings=build_settings(), factory=factory)),
        selected_ingest,
        selected_answer,
    )


def test_document_upload_invokes_one_use_case_and_returns_201() -> None:
    client, ingest, _ = make_client()
    with client:
        response = client.post(
            "/api/v1/documents",
            files=[
                ("files", ("facts.txt", b"Revenue increased.", "text/plain")),
                ("files", ("notes.txt", b"Costs decreased.", "text/plain")),
            ],
        )

    assert response.status_code == 201
    assert response.json() == {
        "documents": [
            {
                "document_id": "doc-1",
                "file_name": "facts.txt",
                "media_type": "text/plain",
                "chunk_count": 2,
                "status": "indexed",
            }
        ]
    }
    assert len(ingest.calls) == 1
    assert len(ingest.calls[0]) == 2


def test_question_route_returns_server_owned_sources() -> None:
    client, _, answer = make_client()
    with client:
        response = client.post(
            "/api/v1/questions",
            json={"question": "  What happened to revenue?  ", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Revenue increased.",
        "grounded": True,
        "sources": [
            {
                "document_id": "doc-1",
                "file_name": "facts.txt",
                "chunk_id": "chunk-1",
                "snippet": "Revenue increased.",
                "score": 0.91,
                "locator": {"start_offset": 0, "end_offset": 19},
            }
        ],
    }
    assert answer.calls == [("  What happened to revenue?  ", 3)]


def test_unanswerable_question_preserves_exact_fallback_contract() -> None:
    client, _, answer = make_client(answer=RecordingAnswer(fallback_result()))
    with client:
        response = client.post("/api/v1/questions", json={"question": "Unknown fact?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": NO_RELEVANT_INFORMATION,
        "grounded": False,
        "sources": [],
    }
    assert answer.calls == [("Unknown fact?", None)]


def test_validation_errors_are_problem_json_without_request_content() -> None:
    client, _, answer = make_client()
    with client:
        response = client.post(
            "/api/v1/questions",
            headers={"X-Request-ID": "req-test-123"},
            json={"question": ""},
        )

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["x-request-id"] == "req-test-123"
    body = response.json()
    assert body["status"] == 422
    assert body["code"] == "request.invalid"
    assert "question" not in body["detail"]
    assert answer.calls == []


def test_upload_failures_map_to_safe_problem_json() -> None:
    error = UploadRejected(UploadErrorCode.UNSUPPORTED_TYPE, "secret document content")
    client, ingest, _ = make_client(ingest=RecordingIngest(error))
    with client:
        response = client.post(
            "/api/v1/documents",
            files=[("files", ("facts.exe", b"secret", "application/octet-stream"))],
        )

    assert response.status_code == 415
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == "upload.unsupported_type"
    assert "secret" not in response.text
    assert len(ingest.calls) == 1


def test_upstream_failures_map_to_503_and_redact_details() -> None:
    error = AdapterError(AdapterErrorCode.VECTOR_UNAVAILABLE, "OPENAI_KEY=secret")
    client, _, _ = make_client(answer=RecordingAnswer(error))
    with client:
        response = client.post("/api/v1/questions", json={"question": "facts?"})

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == "vector.unavailable"
    assert "OPENAI_KEY" not in response.text
    assert "secret" not in response.text


def test_openapi_and_docs_expose_public_contract_examples() -> None:
    client, _, _ = make_client()
    with client:
        openapi = client.get("/openapi.json")
        docs = client.get("/docs")

    assert docs.status_code == 200
    assert openapi.status_code == 200
    paths = openapi.json()["paths"]
    assert "/api/v1/documents" in paths
    assert "/api/v1/questions" in paths
    question_post = paths["/api/v1/questions"]["post"]
    examples = question_post["responses"]["200"]["content"]["application/json"]["examples"]
    assert examples["grounded"]["value"]["grounded"] is True
    assert (
        question_post["responses"]["422"]["content"]["application/problem+json"]["example"]["code"]
        == "request.invalid"
    )


def test_generated_request_id_is_returned_when_header_is_absent() -> None:
    client, _, _ = make_client()
    with client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.headers["x-request-id"]
    assert len(response.headers["x-request-id"]) <= 128


def test_logs_contain_request_metadata_but_not_error_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = AdapterError(AdapterErrorCode.GENERATION_UPSTREAM, "question=secret OPENAI_KEY=secret")
    client, _, _ = make_client(answer=RecordingAnswer(error))
    caplog.set_level("INFO", logger="ai_prime_capital")

    with client:
        response = client.post("/api/v1/questions", json={"question": "secret question"})

    assert response.status_code == 503
    assert "request_completed" in caplog.text
    assert "secret" not in caplog.text
    assert "OPENAI_KEY" not in caplog.text
