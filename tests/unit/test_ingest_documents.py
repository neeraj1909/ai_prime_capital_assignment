from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import pytest

from ai_prime_capital.application.errors import AdapterError, AdapterErrorCode
from ai_prime_capital.application.ingest_documents import (
    IndexedDocument,
    IngestDocuments,
    IngestionResult,
)
from ai_prime_capital.config import Settings
from ai_prime_capital.domain.models import Chunk, ExtractedBlock, RetrievedCandidate, SourceLocator
from ai_prime_capital.infrastructure.extractors.base import UploadErrorCode, UploadRejected

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class RawUpload:
    client_name: str
    declared_media_type: str | None
    stream: BinaryIO


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    display_name: str
    media_type: str
    content: bytes


def build_settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-openai-secret",
        openai_chat_model="test-chat-model",
        embedding_dimensions=3,
        chunk_max_tokens=64,
        chunk_overlap_tokens=0,
    )


def raw_uploads() -> tuple[RawUpload, RawUpload]:
    return (
        RawUpload("first.txt", "text/plain", BytesIO(b"first")),
        RawUpload("second.txt", "text/plain", BytesIO(b"second")),
    )


def validated_uploads() -> tuple[ValidatedUpload, ValidatedUpload]:
    return (
        ValidatedUpload("first.txt", "text/plain", b"first"),
        ValidatedUpload("second.txt", "text/plain", b"second"),
    )


def duplicate_validated_uploads() -> tuple[ValidatedUpload, ValidatedUpload]:
    return (
        ValidatedUpload("first.txt", "text/plain", b"same bytes"),
        ValidatedUpload("renamed.txt", "text/plain", b"same bytes"),
    )


class SpyValidator:
    def __init__(self, result: Sequence[ValidatedUpload] | Exception) -> None:
        self.result = result
        self.calls: list[tuple[RawUpload, ...]] = []

    def __call__(self, uploads: Sequence[RawUpload]) -> tuple[ValidatedUpload, ...]:
        self.calls.append(tuple(uploads))
        if isinstance(self.result, Exception):
            raise self.result
        return tuple(self.result)


class SpyExtractor:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.fail_with: Exception | None = None
        self.return_empty = False
        self.text_by_name = {
            "first.txt": "Revenue is stable.",
            "second.txt": "Costs are lower.",
        }

    async def extract(self, content: bytes, *, file_name: str) -> tuple[ExtractedBlock, ...]:
        del content
        self.events.append(f"extract:{file_name}")
        if self.fail_with is not None:
            raise self.fail_with
        if self.return_empty:
            return ()
        return (
            ExtractedBlock(
                text=self.text_by_name[file_name],
                locator=SourceLocator(),
            ),
        )


class SpyEmbedder:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.inputs: list[tuple[str, ...]] = []
        self.result_override: tuple[tuple[float, ...], ...] | None = None
        self.fail_with: Exception | None = None

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.events.append("embed")
        self.inputs.append(tuple(texts))
        if self.fail_with is not None:
            raise self.fail_with
        if self.result_override is not None:
            return self.result_override
        return tuple((1.0, 0.0, 0.0) for _ in texts)


class SpyVectorStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.upserts: list[tuple[tuple[Chunk, ...], tuple[tuple[float, ...], ...]]] = []
        self.fail_with: Exception | None = None

    async def ensure_compatible(self) -> None:
        self.events.append("ensure")

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        self.events.append("upsert")
        self.upserts.append((tuple(chunks), tuple(tuple(vector) for vector in vectors)))
        if self.fail_with is not None:
            raise self.fail_with

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
    ) -> tuple[RetrievedCandidate, ...]:
        del query_vector, limit
        return ()


def make_service(
    validated: Sequence[ValidatedUpload] | Exception,
) -> tuple[
    IngestDocuments,
    SpyValidator,
    SpyExtractor,
    SpyEmbedder,
    SpyVectorStore,
    list[str],
]:
    events: list[str] = []
    validator = SpyValidator(validated)
    extractor = SpyExtractor(events)
    embedder = SpyEmbedder(events)
    vector_store = SpyVectorStore(events)
    service = IngestDocuments(
        settings=build_settings(),
        validate=validator,
        extractor=extractor,
        embedder=embedder,
        vector_store=vector_store,
    )
    return service, validator, extractor, embedder, vector_store, events


async def test_validates_every_upload_then_extracts_embeds_and_upserts_once() -> None:
    service, validator, _, embedder, vector_store, events = make_service(validated_uploads())

    result = await service.execute(raw_uploads())

    assert len(validator.calls) == 1
    assert events == ["extract:first.txt", "extract:second.txt", "embed", "upsert"]
    assert embedder.inputs == [("Revenue is stable.", "Costs are lower.")]
    assert len(vector_store.upserts) == 1
    assert result == IngestionResult(
        documents=(
            IndexedDocument(
                document_id="doc_a7937b64b8caa58f03721bb6bacf5c78cb235febe0e70b1b84cd99541461a08e",
                file_name="first.txt",
                media_type="text/plain",
                chunk_count=1,
            ),
            IndexedDocument(
                document_id="doc_16367aacb67a4a017c8da8ab95682ccb390863780f7114dda0a0e0c55644c7c4",
                file_name="second.txt",
                media_type="text/plain",
                chunk_count=1,
            ),
        )
    )


async def test_validation_failure_happens_before_any_side_effect() -> None:
    error = UploadRejected(UploadErrorCode.FILE_TOO_LARGE, "too large")
    service, validator, extractor, embedder, vector_store, events = make_service(error)

    with pytest.raises(UploadRejected) as raised:
        await service.execute(raw_uploads())

    assert raised.value is error
    assert len(validator.calls) == 1
    assert events == []
    assert embedder.inputs == []
    assert vector_store.upserts == []
    assert extractor.fail_with is None


async def test_validator_cannot_drop_a_file_without_failing_before_side_effects() -> None:
    service, validator, _, embedder, vector_store, events = make_service((validated_uploads()[0],))

    with pytest.raises(AdapterError) as error:
        await service.execute(raw_uploads())

    assert error.value.code is AdapterErrorCode.INGESTION_INVALID_INPUT
    assert len(validator.calls) == 1
    assert events == []
    assert embedder.inputs == []
    assert vector_store.upserts == []


async def test_duplicate_content_is_rejected_before_extraction_or_indexing() -> None:
    service, _, _, embedder, vector_store, events = make_service(duplicate_validated_uploads())

    with pytest.raises(AdapterError) as error:
        await service.execute(raw_uploads())

    assert error.value.code is AdapterErrorCode.INGESTION_INVALID_INPUT
    assert "duplicate" in error.value.detail
    assert events == []
    assert embedder.inputs == []
    assert vector_store.upserts == []


async def test_extraction_failure_prevents_embedding_and_upsert() -> None:
    service, _, extractor, embedder, vector_store, events = make_service(validated_uploads())
    extractor.fail_with = UploadRejected(UploadErrorCode.NO_EXTRACTABLE_TEXT, "empty")

    with pytest.raises(UploadRejected):
        await service.execute(raw_uploads())

    assert events == ["extract:first.txt"]
    assert embedder.inputs == []
    assert vector_store.upserts == []


async def test_empty_extraction_result_prevents_embedding_and_upsert() -> None:
    service, _, extractor, embedder, vector_store, events = make_service(validated_uploads())
    extractor.return_empty = True

    with pytest.raises(AdapterError) as error:
        await service.execute(raw_uploads())

    assert error.value.code is AdapterErrorCode.INGESTION_INVALID_INPUT
    assert events == ["extract:first.txt", "extract:second.txt"]
    assert embedder.inputs == []
    assert vector_store.upserts == []


async def test_embedding_count_mismatch_prevents_upsert() -> None:
    service, _, _, embedder, vector_store, events = make_service(validated_uploads())
    embedder.result_override = ((1.0, 0.0, 0.0),)

    with pytest.raises(AdapterError) as error:
        await service.execute(raw_uploads())

    assert error.value.code is AdapterErrorCode.INGESTION_RESPONSE
    assert events == ["extract:first.txt", "extract:second.txt", "embed"]
    assert vector_store.upserts == []


async def test_upstream_embedding_failure_is_not_converted_to_success() -> None:
    service, _, _, embedder, vector_store, events = make_service(validated_uploads())
    expected = AdapterError(AdapterErrorCode.EMBEDDING_UPSTREAM, "provider unavailable")
    embedder.fail_with = expected

    with pytest.raises(AdapterError) as error:
        await service.execute(raw_uploads())

    assert error.value is expected
    assert events == ["extract:first.txt", "extract:second.txt", "embed"]
    assert vector_store.upserts == []


async def test_vector_write_failure_is_propagated_after_one_write_attempt() -> None:
    service, _, _, _, vector_store, events = make_service(validated_uploads())
    expected = AdapterError(AdapterErrorCode.VECTOR_WRITE, "write failed")
    vector_store.fail_with = expected

    with pytest.raises(AdapterError) as error:
        await service.execute(raw_uploads())

    assert error.value is expected
    assert events == ["extract:first.txt", "extract:second.txt", "embed", "upsert"]
    assert len(vector_store.upserts) == 1


async def test_repeating_the_same_request_reuses_deterministic_document_and_chunk_ids() -> None:
    service, _, _, _, vector_store, _ = make_service(validated_uploads())

    first = await service.execute(raw_uploads())
    second = await service.execute(raw_uploads())

    assert first == second
    assert [chunk.chunk_id for chunk in vector_store.upserts[0][0]] == [
        chunk.chunk_id for chunk in vector_store.upserts[1][0]
    ]
