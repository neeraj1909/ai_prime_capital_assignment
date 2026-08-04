from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from ..dependencies import ResourcesDependency


class HealthResponse(BaseModel):
    status: Literal["ok", "ready", "not_ready"]
    reason: str | None = None


router = APIRouter(tags=["health"])


@router.get(
    "/health/live",
    response_model=HealthResponse,
    response_model_exclude_none=True,
)
def live() -> HealthResponse:
    # Liveness must not depend on OpenAI, Qdrant, or any shared resource.
    return HealthResponse(status="ok")


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    responses={503: {"model": HealthResponse}},
)
def ready(
    response: Response,
    resources: ResourcesDependency,
) -> HealthResponse:
    if resources.readiness.ready:
        return HealthResponse(status="ready")

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="not_ready",
        reason=resources.readiness.reason,
    )
