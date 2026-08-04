from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import BinaryIO, Literal, Protocol

from ..config import Settings
from ..domain.chunking import chunk_blocks
from ..domain.identity import document_id
from ..domain.models import Chunk
from .errors import AdapterError, AdapterErrorCode
from .ports import EmbedderPort, ExtractorPort, VectorStorePort


class RawUpload(Protocol):
    """The small upload shape required from an HTTP or CLI boundary."""

    client_name: str
    declared_media_type: str | None
    stream: BinaryIO


class ValidatedUpload(Protocol):
    """Validated bytes and provenance supplied by the upload boundary."""

    display_name: str
    media_type: str
    content: bytes


UploadValidator = Callable[
    [Sequence[RawUpload]],
    Sequence[ValidatedUpload],
]


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    document_id: str
    file_name: str
    media_type: str
    chunk_count: int
    status: Literal["indexed"] = "indexed"


@dataclass(frozen=True, slots=True)
class IngestionResult:
    documents: tuple[IndexedDocument, ...]


class IngestDocuments:
    """Coordinate one complete, deterministic upload-to-index operation."""

    def __init__(
        self,
        *,
        settings: Settings,
        validate: UploadValidator,
        extractor: ExtractorPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
    ) -> None:
        self._settings = settings
        self._validate = validate
        self._extractor = extractor
        self._embedder = embedder
        self._vector_store = vector_store

    async def execute(
        self,
        uploads: Sequence[RawUpload],
    ) -> IngestionResult:
        # Validation must consume the complete request before any parser or
        # vector-store side effect can occur.
        validated = tuple(self._validate(tuple(uploads)))
        if len(validated) != len(uploads):
            raise AdapterError(
                AdapterErrorCode.INGESTION_INVALID_INPUT,
                "upload validation changed the request cardinality",
            )

        prepared: list[tuple[str, ValidatedUpload, tuple[Chunk, ...]]] = []
        all_chunks: list[Chunk] = []

        # Extract and chunk every document before embedding any document.
        for upload in validated:
            current_document_id = document_id(upload.content)
            blocks = await self._extractor.extract(
                upload.content,
                file_name=upload.display_name,
            )
            chunks = chunk_blocks(
                blocks,
                document_id_value=current_document_id,
                file_name=upload.display_name,
                embedding_model=self._settings.openai_embedding_model,
                schema_version=self._settings.chunk_schema_version,
                max_tokens=self._settings.chunk_max_tokens,
                overlap_tokens=self._settings.chunk_overlap_tokens,
            )
            prepared.append((current_document_id, upload, chunks))
            all_chunks.extend(chunks)

        if not all_chunks:
            raise AdapterError(
                AdapterErrorCode.INGESTION_INVALID_INPUT,
                "validated uploads produced no chunks",
            )

        vectors = tuple(await self._embedder.embed(tuple(chunk.text for chunk in all_chunks)))
        if len(vectors) != len(all_chunks):
            raise AdapterError(
                AdapterErrorCode.INGESTION_RESPONSE,
                "embedding count does not match chunk count",
            )

        # The vector adapter owns acknowledgement and retry-safe point writes.
        # This service deliberately does not retry ambiguous writes or claim
        # rollback after a network failure.
        await self._vector_store.upsert(all_chunks, vectors)

        return IngestionResult(
            documents=tuple(
                IndexedDocument(
                    document_id=current_document_id,
                    file_name=upload.display_name,
                    media_type=upload.media_type,
                    chunk_count=len(chunks),
                )
                for current_document_id, upload, chunks in prepared
            )
        )
