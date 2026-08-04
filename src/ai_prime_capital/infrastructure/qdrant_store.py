from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import ceil, isfinite
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import CollectionInfo, Distance, PointStruct, ScoredPoint, VectorParams

from ..application.errors import AdapterError, AdapterErrorCode
from ..application.ports import VectorStorePort
from ..config import Settings
from ..domain.models import Chunk, RetrievedCandidate, SourceLocator

DISTANCES = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclid": Distance.EUCLID,
}


class QdrantVectorStore(VectorStorePort):
    """Translate Qdrant collection, point, and query APIs into application types."""

    def __init__(
        self,
        settings: Settings,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or AsyncQdrantClient(
            url=str(settings.qdrant_url),
            api_key=(
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            ),
            timeout=ceil(settings.qdrant_timeout_seconds),
        )

    async def ensure_compatible(self) -> None:
        info = await self._get_collection()
        if info is None:
            try:
                await self._client.create_collection(
                    collection_name=self._settings.qdrant_collection,
                    vectors_config={
                        self._settings.vector_name: VectorParams(
                            size=self._settings.embedding_dimensions,
                            distance=self._distance,
                        )
                    },
                    metadata=self._expected_metadata(),
                )
            except UnexpectedResponse as exc:
                if exc.status_code != 409:
                    raise AdapterError(
                        AdapterErrorCode.VECTOR_UNAVAILABLE,
                        "Qdrant collection creation failed",
                    ) from exc
                # Another process may have created it between lookup and create.
                info = await self._get_collection()
                if info is None:
                    raise AdapterError(
                        AdapterErrorCode.VECTOR_UNAVAILABLE,
                        "Qdrant collection disappeared after creation conflict",
                    ) from exc
            except Exception as exc:
                raise AdapterError(
                    AdapterErrorCode.VECTOR_UNAVAILABLE,
                    "Qdrant collection creation failed",
                ) from exc

            if info is None:
                return

        self._assert_compatible(info)

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise AdapterError(
                AdapterErrorCode.VECTOR_INVALID_INPUT,
                "chunk and vector counts must match",
            )
        if not chunks:
            return

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            try:
                values = tuple(float(value) for value in vector)
            except (TypeError, ValueError) as exc:
                raise AdapterError(
                    AdapterErrorCode.VECTOR_INVALID_INPUT,
                    "vector contains a non-numeric value",
                ) from exc
            if len(values) != self._settings.embedding_dimensions:
                raise AdapterError(
                    AdapterErrorCode.VECTOR_INVALID_INPUT,
                    "vector dimension does not match settings",
                )
            if not all(isfinite(value) for value in values):
                raise AdapterError(
                    AdapterErrorCode.VECTOR_INVALID_INPUT,
                    "vector contains a non-finite value",
                )
            try:
                point_id = str(UUID(chunk.chunk_id))
            except (ValueError, AttributeError) as exc:
                raise AdapterError(
                    AdapterErrorCode.VECTOR_INVALID_INPUT,
                    "chunk identifier must be a UUID",
                ) from exc
            if point_id != chunk.chunk_id:
                raise AdapterError(
                    AdapterErrorCode.VECTOR_INVALID_INPUT,
                    "chunk identifier must use canonical UUID text",
                )
            points.append(
                PointStruct(
                    id=point_id,
                    vector={self._settings.vector_name: list(values)},
                    payload=self._payload(chunk),
                )
            )

        await self.ensure_compatible()
        try:
            await self._client.upsert(
                collection_name=self._settings.qdrant_collection,
                points=points,
                wait=True,
            )
        except UnexpectedResponse as exc:
            raise AdapterError(
                AdapterErrorCode.VECTOR_WRITE,
                "Qdrant vector write failed",
            ) from exc
        except Exception as exc:
            raise AdapterError(
                AdapterErrorCode.VECTOR_WRITE,
                "Qdrant vector write failed",
            ) from exc

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
    ) -> tuple[RetrievedCandidate, ...]:
        if limit < 1:
            raise AdapterError(
                AdapterErrorCode.VECTOR_INVALID_INPUT,
                "search limit must be positive",
            )

        try:
            values = tuple(float(value) for value in query_vector)
        except (TypeError, ValueError) as exc:
            raise AdapterError(
                AdapterErrorCode.VECTOR_INVALID_INPUT,
                "query vector contains a non-numeric value",
            ) from exc
        if len(values) != self._settings.embedding_dimensions or not all(
            isfinite(value) for value in values
        ):
            raise AdapterError(
                AdapterErrorCode.VECTOR_INVALID_INPUT,
                "query vector does not match settings",
            )

        await self.ensure_compatible()
        try:
            response = await self._client.query_points(
                collection_name=self._settings.qdrant_collection,
                query=list(values),
                using=self._settings.vector_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as exc:
            raise AdapterError(
                AdapterErrorCode.VECTOR_UNAVAILABLE,
                "Qdrant vector query failed",
            ) from exc
        except Exception as exc:
            raise AdapterError(
                AdapterErrorCode.VECTOR_UNAVAILABLE,
                "Qdrant vector query failed",
            ) from exc

        try:
            points = response.points
            return tuple(
                RetrievedCandidate(
                    chunk=self._chunk_from_point(point),
                    score=float(point.score),
                )
                for point in points
            )
        except AdapterError:
            raise
        except (TypeError, ValueError) as exc:
            raise AdapterError(
                AdapterErrorCode.VECTOR_RESPONSE,
                "Qdrant returned an invalid point payload",
            ) from exc

    async def close(self) -> None:
        await self._client.close()

    @property
    def _distance(self) -> Distance:
        return DISTANCES[self._settings.qdrant_distance]

    async def _get_collection(self) -> CollectionInfo | None:
        try:
            return await self._client.get_collection(self._settings.qdrant_collection)
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                return None
            raise AdapterError(
                AdapterErrorCode.VECTOR_UNAVAILABLE,
                "Qdrant collection lookup failed",
            ) from exc
        except ValueError as exc:
            # The embedded Qdrant test client reports a missing collection as
            # ValueError rather than the remote API's 404 response.
            if f"Collection {self._settings.qdrant_collection} not found" in str(exc):
                return None
            raise AdapterError(
                AdapterErrorCode.VECTOR_UNAVAILABLE,
                "Qdrant collection lookup failed",
            ) from exc
        except Exception as exc:
            raise AdapterError(
                AdapterErrorCode.VECTOR_UNAVAILABLE,
                "Qdrant collection lookup failed",
            ) from exc

    def _assert_compatible(self, info: CollectionInfo) -> None:
        try:
            vectors = info.config.params.vectors
            actual = (
                vectors.get(self._settings.vector_name) if isinstance(vectors, Mapping) else None
            )
            metadata = info.config.metadata or {}
            if not isinstance(metadata, Mapping):
                raise TypeError("collection metadata must be a mapping")
        except (AttributeError, TypeError) as exc:
            raise AdapterError(
                AdapterErrorCode.VECTOR_INCOMPATIBLE,
                "Qdrant collection schema is unreadable",
            ) from exc

        if actual is None or not isinstance(actual, VectorParams):
            raise AdapterError(
                AdapterErrorCode.VECTOR_INCOMPATIBLE,
                "Qdrant named vector is incompatible",
            )
        if actual.size != self._settings.embedding_dimensions or actual.distance != self._distance:
            raise AdapterError(
                AdapterErrorCode.VECTOR_INCOMPATIBLE,
                "Qdrant vector dimensions or distance are incompatible",
            )

        expected = self._expected_metadata()
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise AdapterError(
                AdapterErrorCode.VECTOR_INCOMPATIBLE,
                "Qdrant vector fingerprint is incompatible",
            )

    def _expected_metadata(self) -> dict[str, object]:
        return {
            "embedding_model": self._settings.openai_embedding_model,
            "embedding_dimensions": self._settings.embedding_dimensions,
            "distance": self._settings.qdrant_distance,
            "vector_fingerprint": self._settings.vector_fingerprint,
            "chunk_schema_version": self._settings.chunk_schema_version,
        }

    @staticmethod
    def _payload(chunk: Chunk) -> dict[str, object]:
        payload: dict[str, object] = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "file_name": chunk.file_name,
            "text": chunk.text,
            "locator": {
                key: value for key, value in asdict(chunk.locator).items() if value is not None
            },
        }
        if chunk.char_count is not None:
            payload["char_count"] = chunk.char_count
        if chunk.token_count is not None:
            payload["token_count"] = chunk.token_count
        return payload

    @classmethod
    def _chunk_from_point(cls, point: ScoredPoint) -> Chunk:
        payload = point.payload
        if not isinstance(payload, Mapping):
            raise AdapterError(
                AdapterErrorCode.VECTOR_RESPONSE,
                "Qdrant point is missing a payload",
            )

        chunk_id = cls._required_string(payload, "chunk_id")
        if str(point.id) != chunk_id:
            raise AdapterError(
                AdapterErrorCode.VECTOR_RESPONSE,
                "Qdrant point identity does not match its payload",
            )
        document_id = cls._required_string(payload, "document_id")
        file_name = cls._required_string(payload, "file_name")
        text = cls._required_string(payload, "text")
        locator_value = payload.get("locator")
        if not isinstance(locator_value, Mapping):
            raise AdapterError(
                AdapterErrorCode.VECTOR_RESPONSE,
                "Qdrant point is missing a source locator",
            )

        try:
            locator = SourceLocator(
                page_number=cls._optional_int(locator_value, "page_number"),
                section=cls._optional_string(locator_value, "section"),
                paragraph_index=cls._optional_int(locator_value, "paragraph_index"),
                table_index=cls._optional_int(locator_value, "table_index"),
                start_offset=cls._optional_int(locator_value, "start_offset"),
                end_offset=cls._optional_int(locator_value, "end_offset"),
            )
            return Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                file_name=file_name,
                text=text,
                locator=locator,
                char_count=cls._optional_int(payload, "char_count"),
                token_count=cls._optional_int(payload, "token_count"),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise AdapterError(
                AdapterErrorCode.VECTOR_RESPONSE,
                "Qdrant point contains invalid chunk metadata",
            ) from exc

    @staticmethod
    def _required_string(payload: Mapping[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise AdapterError(
                AdapterErrorCode.VECTOR_RESPONSE,
                "Qdrant point contains invalid text metadata",
            )
        return value

    @staticmethod
    def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{key} must be a string")
        return value

    @staticmethod
    def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
        value = payload.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be an integer")
        return value
