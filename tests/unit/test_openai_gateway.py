from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import Request, Response
from openai import APIStatusError, APITimeoutError

from ai_prime_capital.application.errors import AdapterError, AdapterErrorCode
from ai_prime_capital.config import Settings
from ai_prime_capital.infrastructure.openai_gateway import OpenAIEmbeddingGateway

pytestmark = pytest.mark.unit


REQUIRED_SETTINGS = {
    "openai_api_key": "test-openai-secret",
    "openai_chat_model": "test-chat-model",
}


def build_settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        **(REQUIRED_SETTINGS | {"embedding_dimensions": 3} | overrides),
    )


class FakeEmbeddings:
    def __init__(self, responses: Sequence[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClient:
    def __init__(self, responses: Sequence[Any]) -> None:
        self.embeddings = FakeEmbeddings(responses)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def response(*vectors: Sequence[float], indexes: Sequence[int] | None = None) -> Any:
    item_indexes = indexes or range(len(vectors))
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=list(vector))
            for index, vector in zip(item_indexes, vectors, strict=True)
        ]
    )


async def test_embed_batches_requests_and_restores_input_order() -> None:
    client = FakeClient(
        (
            response((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), indexes=(1, 0)),
            response((0.7, 0.8, 0.9)),
        )
    )
    settings = build_settings(embedding_batch_size=2)
    gateway = OpenAIEmbeddingGateway(settings, client=client)  # type: ignore[arg-type]

    vectors = await gateway.embed(("first", "second", "third"))

    assert vectors == (
        (0.4, 0.5, 0.6),
        (0.1, 0.2, 0.3),
        (0.7, 0.8, 0.9),
    )
    assert [call["input"] for call in client.embeddings.calls] == [
        ["first", "second"],
        ["third"],
    ]
    assert all(call["model"] == settings.openai_embedding_model for call in client.embeddings.calls)
    assert all(
        call["dimensions"] == settings.embedding_dimensions for call in client.embeddings.calls
    )


async def test_empty_input_is_a_noop() -> None:
    client = FakeClient(())
    gateway = OpenAIEmbeddingGateway(build_settings(), client=client)  # type: ignore[arg-type]

    assert await gateway.embed(()) == ()
    assert client.embeddings.calls == []


async def test_blank_input_is_rejected_with_stable_application_error() -> None:
    gateway = OpenAIEmbeddingGateway(build_settings(), client=FakeClient(()))  # type: ignore[arg-type]

    with pytest.raises(AdapterError) as error:
        await gateway.embed(("valid", "  "))

    assert error.value.code is AdapterErrorCode.EMBEDDING_INVALID_INPUT


async def test_timeout_is_retried_with_a_bounded_attempt_count() -> None:
    client = FakeClient(
        (
            APITimeoutError(request=Request("POST", "https://api.openai.test")),
            response((0.1, 0.2, 0.3)),
        )
    )
    gateway = OpenAIEmbeddingGateway(
        build_settings(embedding_max_retries=1),
        client=client,  # type: ignore[arg-type]
    )

    assert await gateway.embed(("text",)) == ((0.1, 0.2, 0.3),)
    assert len(client.embeddings.calls) == 2


async def test_non_retryable_provider_error_is_translated_without_retry() -> None:
    response_error = APIStatusError(
        "bad request",
        response=Response(400, request=Request("POST", "https://api.openai.test")),
        body={"error": "bad request"},
    )
    client = FakeClient((response_error,))
    gateway = OpenAIEmbeddingGateway(
        build_settings(embedding_max_retries=3),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(AdapterError) as error:
        await gateway.embed(("text",))

    assert error.value.code is AdapterErrorCode.EMBEDDING_UPSTREAM
    assert len(client.embeddings.calls) == 1


async def test_invalid_provider_dimension_is_rejected() -> None:
    client = FakeClient((response((0.1, 0.2)),))
    gateway = OpenAIEmbeddingGateway(build_settings(), client=client)  # type: ignore[arg-type]

    with pytest.raises(AdapterError) as error:
        await gateway.embed(("text",))

    assert error.value.code is AdapterErrorCode.EMBEDDING_DIMENSION


async def test_close_closes_the_injected_client() -> None:
    client = FakeClient(())
    gateway = OpenAIEmbeddingGateway(build_settings(), client=client)  # type: ignore[arg-type]

    await gateway.close()

    assert client.closed
