from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..domain.models import (
    Chunk,
    ExtractedBlock,
    ModelDecision,
    RetrievedCandidate,
)


class ExtractorPort(Protocol):
    async def extract(
        self,
        content: bytes,
        *,
        file_name: str,
    ) -> tuple[ExtractedBlock, ...]: ...


class EmbedderPort(Protocol):
    async def embed(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]: ...


class GeneratorPort(Protocol):
    async def generate(
        self,
        *,
        question: str,
        candidates: Sequence[RetrievedCandidate],
    ) -> ModelDecision: ...


class VectorStorePort(Protocol):
    async def ensure_compatible(self) -> None: ...

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
    ) -> tuple[RetrievedCandidate, ...]: ...


class ClosablePort(Protocol):
    async def close(self) -> None: ...
