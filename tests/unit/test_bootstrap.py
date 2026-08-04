from __future__ import annotations

from typing import Any

import pytest

import ai_prime_capital.bootstrap as bootstrap
from ai_prime_capital.application.errors import AdapterError, AdapterErrorCode
from ai_prime_capital.bootstrap import ResourceInitializationError
from ai_prime_capital.config import Settings

pytestmark = pytest.mark.unit


def build_settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-openai-secret",
        openai_chat_model="test-chat-model",
    )


class Probe:
    def __init__(self, name: str, events: list[str], failure: Exception | None = None) -> None:
        self.name = name
        self.events = events
        self.failure = failure

    async def ensure_available(self) -> None:
        self.events.append(f"probe:{self.name}")
        if self.failure is not None:
            raise self.failure

    async def ensure_compatible(self) -> None:
        self.events.append(f"probe:{self.name}")
        if self.failure is not None:
            raise self.failure

    async def close(self) -> None:
        self.events.append(f"close:{self.name}")


async def test_configured_resources_probe_all_runtime_dependencies(monkeypatch: Any) -> None:
    events: list[str] = []
    embedder = Probe("embedding", events)
    generator = Probe("answer", events)
    vector_store = Probe("vector", events)

    monkeypatch.setattr(bootstrap, "OpenAIEmbeddingGateway", lambda _settings: embedder)
    monkeypatch.setattr(bootstrap, "OpenAIAnswerGateway", lambda _settings: generator)
    monkeypatch.setattr(bootstrap, "QdrantVectorStore", lambda _settings: vector_store)

    resources = await bootstrap.build_configured_resources(build_settings())

    assert resources.readiness.ready
    assert events[:3] == ["probe:embedding", "probe:answer", "probe:vector"]
    await resources.close()
    assert events[-3:] == ["close:embedding", "close:answer", "close:vector"]


async def test_provider_probe_failure_closes_all_clients_and_fails_startup(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    failure = AdapterError(AdapterErrorCode.EMBEDDING_UPSTREAM, "provider unavailable")
    embedder = Probe("embedding", events, failure)
    generator = Probe("answer", events)
    vector_store = Probe("vector", events)

    monkeypatch.setattr(bootstrap, "OpenAIEmbeddingGateway", lambda _settings: embedder)
    monkeypatch.setattr(bootstrap, "OpenAIAnswerGateway", lambda _settings: generator)
    monkeypatch.setattr(bootstrap, "QdrantVectorStore", lambda _settings: vector_store)

    with pytest.raises(ResourceInitializationError):
        await bootstrap.build_configured_resources(build_settings())

    assert events == [
        "probe:embedding",
        "close:vector",
        "close:answer",
        "close:embedding",
    ]
