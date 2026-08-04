from __future__ import annotations

from fastapi import FastAPI

from .api.routes.health import router as health_router
from .bootstrap import (
    ResourceFactory,
    build_unconfigured_resources,
    make_lifespan,
)
from .config import Settings


def create_app(
    *,
    settings: Settings | None = None,
    factory: ResourceFactory | None = None,
) -> FastAPI:
    selected_factory = factory if factory is not None else build_unconfigured_resources

    app = FastAPI(
        title="AI Prime Capital Document-Grounded RAG API",
        version="0.1.0",
        lifespan=make_lifespan(
            settings=settings,
            factory=selected_factory,
        ),
    )

    app.include_router(health_router)

    return app


app = create_app()
