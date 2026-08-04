from __future__ import annotations

from fastapi import FastAPI

from .api.errors import register_exception_handlers
from .api.routes.documents import router as documents_router
from .api.routes.health import router as health_router
from .api.routes.questions import router as questions_router
from .bootstrap import (
    ResourceFactory,
    build_configured_resources,
    make_lifespan,
)
from .config import Settings
from .observability import install_observability


def create_app(
    *,
    settings: Settings | None = None,
    factory: ResourceFactory | None = None,
) -> FastAPI:
    selected_factory = factory if factory is not None else build_configured_resources

    app = FastAPI(
        title="AI Prime Capital Document-Grounded RAG API",
        version="0.1.0",
        lifespan=make_lifespan(
            settings=settings,
            factory=selected_factory,
        ),
    )

    install_observability(app)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(questions_router)

    return app


app = create_app()
