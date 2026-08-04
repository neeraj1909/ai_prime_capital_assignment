from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import ai_prime_capital.bootstrap as bootstrap
from ai_prime_capital.api.dependencies import get_resources
from ai_prime_capital.bootstrap import (
    Readiness,
    ResourceInitializationError,
    RuntimeResources,
)
from ai_prime_capital.config import Settings
from ai_prime_capital.main import create_app

pytestmark = pytest.mark.api


def build_settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-secret",
        openai_chat_model="test-chat-model",
        **overrides,
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


class RecoveringFactory:
    def __init__(self, *, unexpected_retry_failure: bool = False) -> None:
        self.create_count = 0
        self.closer = RecordingCloser()
        self.allow_retry = threading.Event()
        self.unexpected_retry_failure = unexpected_retry_failure

    async def __call__(self, settings: Settings) -> RuntimeResources:
        self.create_count += 1
        if self.create_count == 1:
            raise ResourceInitializationError("transient dependency failure")
        await asyncio.to_thread(self.allow_retry.wait)
        if self.unexpected_retry_failure and self.create_count == 2:
            raise RuntimeError("unexpected retry failure")
        return RuntimeResources(
            settings=settings,
            readiness=Readiness(ready=True),
            closers=(self.closer,),
        )


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


def test_initialization_retries_and_recovers_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bootstrap, "_RESOURCE_RETRY_INTERVAL_SECONDS", 0.01, raising=False)
    factory = RecoveringFactory()
    app = create_app(settings=build_settings(), factory=factory)

    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 503
        factory.allow_retry.set()
        deadline = time.monotonic() + 1
        response = client.get("/health/ready")
        while response.status_code != 200 and time.monotonic() < deadline:
            time.sleep(0.01)
            response = client.get("/health/ready")

        assert response.status_code == 200
        assert response.json() == {"status": "ready"}
        assert factory.create_count == 2

    assert factory.closer.close_count == 1


def test_initialization_retry_survives_unexpected_factory_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bootstrap, "_RESOURCE_RETRY_INTERVAL_SECONDS", 0.01, raising=False)
    factory = RecoveringFactory(unexpected_retry_failure=True)
    app = create_app(settings=build_settings(), factory=factory)

    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 503
        factory.allow_retry.set()
        deadline = time.monotonic() + 1
        response = client.get("/health/ready")
        while response.status_code != 200 and time.monotonic() < deadline:
            time.sleep(0.01)
            response = client.get("/health/ready")

        assert response.status_code == 200
        assert factory.create_count == 3

    assert factory.closer.close_count == 1


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


def test_content_length_limit_rejects_large_requests_before_route_parsing() -> None:
    app = create_app(settings=build_settings(), factory=RecordingFactory(ready=True))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/questions",
            content=b"{}",
            headers={"Content-Length": str(50 * 1024 * 1024 + 1)},
        )

    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.headers["connection"] == "close"
    assert response.json()["code"] == "request.body_too_large"


def test_streamed_body_without_content_length_is_still_bounded() -> None:
    settings = build_settings(max_file_bytes=16, max_request_bytes=16)
    app = create_app(settings=settings, factory=RecordingFactory(ready=True))

    def body_chunks() -> Iterator[bytes]:
        yield b'{"question":"'
        yield b"x" * 32
        yield b'"}'

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/questions",
            content=body_chunks(),
            headers={
                "Content-Type": "application/json",
                "Transfer-Encoding": "chunked",
            },
        )

    assert response.status_code == 413
    assert response.json()["code"] == "request.body_too_large"


def test_max_length_json_escaped_question_is_not_rejected_by_transport_limit() -> None:
    settings = build_settings()
    app = create_app(settings=settings, factory=RecordingFactory(ready=True))
    body = json.dumps({"question": "😀" * settings.max_question_chars}).encode()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/questions",
            content=body,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 503
    assert response.json()["code"] == "vector.unavailable"


def test_configured_log_level_is_applied_at_startup() -> None:
    logger = logging.getLogger("ai_prime_capital")
    previous_level = logger.level
    app = create_app(
        settings=build_settings(log_level="ERROR"),
        factory=RecordingFactory(ready=True),
    )

    try:
        with TestClient(app):
            assert logger.getEffectiveLevel() == logging.ERROR
    finally:
        logger.setLevel(previous_level)
