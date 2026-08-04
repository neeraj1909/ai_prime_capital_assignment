from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai_prime_capital.api.dependencies import get_resources
from ai_prime_capital.bootstrap import (
    Readiness,
    ResourceInitializationError,
    RuntimeResources,
)
from ai_prime_capital.config import Settings
from ai_prime_capital.main import create_app

pytestmark = pytest.mark.api


def build_settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-secret",
        openai_chat_model="test-chat-model",
    )


class RecordingCloser:
    def __init__(self) -> None:
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1


class RecordingFactory:
    def __init__(self, *, ready: bool, reason: str | None = None) -> None:
        self.ready = ready
        self.reason = reason
        self.create_count = 0
        self.settings: Settings | None = None
        self.closer = RecordingCloser()

    async def __call__(self, settings: Settings) -> RuntimeResources:
        self.create_count += 1
        self.settings = settings
        return RuntimeResources(
            settings=settings,
            readiness=Readiness(ready=self.ready, reason=self.reason),
            closers=(self.closer,),
        )


class FailingFactory:
    async def __call__(self, _settings: Settings) -> RuntimeResources:
        raise ResourceInitializationError("test dependency unavailable")


def test_lifespan_creates_and_closes_resources() -> None:
    factory = RecordingFactory(ready=True)
    app = create_app(settings=build_settings(), factory=factory)

    assert factory.create_count == 0

    with TestClient(app) as client:
        assert factory.create_count == 1
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").json() == {"status": "ready"}

    assert factory.closer.close_count == 1


def test_liveness_is_independent_of_dependency_readiness() -> None:
    factory = RecordingFactory(
        ready=False,
        reason="vector store is unavailable",
    )
    app = create_app(settings=build_settings(), factory=factory)

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "ok"}
    assert ready.status_code == 503
    assert ready.json() == {
        "status": "not_ready",
        "reason": "vector store is unavailable",
    }


def test_initialization_failure_keeps_liveness_and_marks_not_ready() -> None:
    app = create_app(settings=build_settings(), factory=FailingFactory())

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503


def test_dependency_override_is_supported() -> None:
    factory = RecordingFactory(ready=False)
    app = create_app(settings=build_settings(), factory=factory)
    override_resources = RuntimeResources(
        settings=build_settings(),
        readiness=Readiness(ready=True),
    )
    app.dependency_overrides[get_resources] = lambda: override_resources

    try:
        with TestClient(app) as client:
            response = client.get("/health/ready")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_health_routes_are_in_openapi() -> None:
    app = create_app(settings=build_settings(), factory=RecordingFactory(ready=True))

    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    assert "/health/live" in schema["paths"]
    assert "/health/ready" in schema["paths"]
