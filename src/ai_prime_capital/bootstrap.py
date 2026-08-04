from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from pydantic import ValidationError

from .application.ports import (
    ClosablePort,
    EmbedderPort,
    ExtractorPort,
    GeneratorPort,
    VectorStorePort,
)
from .config import Settings


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
    """Temporary foundation state until provider adapters exist."""

    return RuntimeResources.unavailable(
        settings=settings,
        reason="integration adapters are not configured",
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
