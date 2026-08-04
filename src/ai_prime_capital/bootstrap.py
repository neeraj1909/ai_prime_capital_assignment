from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import cast

from fastapi import FastAPI
from pydantic import ValidationError

from .application.answer_question import AnswerQuestion
from .application.errors import AdapterError
from .application.ingest_documents import IngestDocuments, RawUpload
from .application.ingest_documents import ValidatedUpload as ApplicationValidatedUpload
from .application.ports import (
    ClosablePort,
    EmbedderPort,
    ExtractorPort,
    GeneratorPort,
    VectorStorePort,
)
from .config import Settings
from .infrastructure.extractors.base import IncomingUpload
from .infrastructure.extractors.registry import ExtractorRegistry, validate_uploads
from .infrastructure.openai_gateway import OpenAIAnswerGateway, OpenAIEmbeddingGateway
from .infrastructure.qdrant_store import QdrantVectorStore


@dataclass(frozen=True, slots=True)
class Readiness:
    ready: bool
    reason: str | None = None


@dataclass(slots=True)
class RuntimeResources:
    settings: Settings | None
    readiness: Readiness
    extractor: ExtractorPort | None = None
    embedder: EmbedderPort | None = None
    generator: GeneratorPort | None = None
    vector_store: VectorStorePort | None = None
    ingest_documents: IngestDocuments | None = None
    answer_question: AnswerQuestion | None = None
    closers: tuple[ClosablePort, ...] = ()

    @classmethod
    def unavailable(
        cls,
        *,
        settings: Settings | None,
        reason: str,
    ) -> RuntimeResources:
        return cls(
            settings=settings,
            readiness=Readiness(ready=False, reason=reason),
        )

    async def close(self) -> None:
        for resource in reversed(self.closers):
            await resource.close()


class ResourceInitializationError(RuntimeError):
    """Expected dependency/configuration failure during application startup."""


ResourceFactory = Callable[[Settings], Awaitable[RuntimeResources]]


async def build_unconfigured_resources(
    settings: Settings,
) -> RuntimeResources:
    """Return a deliberately unavailable resource set for explicit dry runs."""

    return RuntimeResources.unavailable(
        settings=settings,
        reason="integration adapters are not configured",
    )


async def build_configured_resources(settings: Settings) -> RuntimeResources:
    """Compose the production adapters and application use cases once per app."""

    extractor = ExtractorRegistry(settings)
    embedder = OpenAIEmbeddingGateway(settings)
    generator = OpenAIAnswerGateway(settings)
    vector_store = QdrantVectorStore(settings)

    try:
        await vector_store.ensure_compatible()
    except AdapterError as exc:
        for resource in (vector_store, generator, embedder):
            try:
                await resource.close()
            except Exception:
                # Startup is already failing; do not mask the readiness reason
                # with a best-effort client-close failure.
                pass
        raise ResourceInitializationError("required dependencies are unavailable") from exc

    def validate(uploads: Sequence[RawUpload]) -> Sequence[ApplicationValidatedUpload]:
        incoming = cast(tuple[IncomingUpload, ...], tuple(uploads))
        validated = validate_uploads(incoming, settings)
        return cast(Sequence[ApplicationValidatedUpload], validated)

    ingest_documents = IngestDocuments(
        settings=settings,
        validate=validate,
        extractor=extractor,
        embedder=embedder,
        vector_store=vector_store,
    )
    answer_question = AnswerQuestion(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
    )

    return RuntimeResources(
        settings=settings,
        readiness=Readiness(ready=True),
        extractor=extractor,
        embedder=embedder,
        generator=generator,
        vector_store=vector_store,
        ingest_documents=ingest_documents,
        answer_question=answer_question,
        closers=(vector_store, generator, embedder),
    )


async def _initialize_resources(
    settings: Settings | None,
    factory: ResourceFactory,
) -> RuntimeResources:
    try:
        # Required settings are intentionally resolved at lifespan startup, not
        # while importing the ASGI module. This keeps `/health/live` callable
        # during configuration failures and makes tests able to inject settings.
        resolved_settings = settings if settings is not None else Settings()  # type: ignore[call-arg]
    except ValidationError:
        return RuntimeResources.unavailable(
            settings=None,
            reason="configuration is invalid",
        )

    try:
        return await factory(resolved_settings)
    except ResourceInitializationError:
        return RuntimeResources.unavailable(
            settings=resolved_settings,
            reason="required dependencies are unavailable",
        )


def make_lifespan(
    *,
    settings: Settings | None,
    factory: ResourceFactory,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resources = await _initialize_resources(settings, factory)
        app.state.resources = resources

        try:
            yield
        finally:
            await resources.close()

    return lifespan
