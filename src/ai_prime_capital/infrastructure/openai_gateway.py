from __future__ import annotations

import asyncio
from collections.abc import Sequence
from itertools import batched
from math import isfinite

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse

from ..application.errors import AdapterError, AdapterErrorCode
from ..application.ports import EmbedderPort
from ..config import Settings


class OpenAIEmbeddingGateway(EmbedderPort):
    """Translate the async OpenAI embedding API into the application port."""

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self._settings = settings
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_seconds,
            max_retries=0,
        )

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        values = tuple(texts)
        if not values:
            return ()

        if any(not text.strip() for text in values):
            raise AdapterError(
                AdapterErrorCode.EMBEDDING_INVALID_INPUT,
                "embedding input must not contain blank text",
            )

        output: list[tuple[float, ...]] = []
        for batch in batched(values, self._settings.embedding_batch_size):
            response = await self._create_embedding(batch)
            try:
                data = sorted(response.data, key=lambda item: item.index)
            except (AttributeError, TypeError, ValueError) as exc:
                raise AdapterError(
                    AdapterErrorCode.EMBEDDING_RESPONSE,
                    "embedding provider returned an invalid response",
                ) from exc

            if len(data) != len(batch) or [item.index for item in data] != list(range(len(batch))):
                raise AdapterError(
                    AdapterErrorCode.EMBEDDING_RESPONSE,
                    "embedding provider returned an invalid response",
                )

            for item in data:
                try:
                    vector = tuple(float(value) for value in item.embedding)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise AdapterError(
                        AdapterErrorCode.EMBEDDING_RESPONSE,
                        "embedding provider returned a non-numeric vector",
                    ) from exc

                if len(vector) != self._settings.embedding_dimensions:
                    raise AdapterError(
                        AdapterErrorCode.EMBEDDING_DIMENSION,
                        "embedding provider returned an unexpected vector dimension",
                    )
                if not all(isfinite(value) for value in vector):
                    raise AdapterError(
                        AdapterErrorCode.EMBEDDING_RESPONSE,
                        "embedding provider returned a non-finite vector",
                    )
                output.append(vector)

        return tuple(output)

    async def _create_embedding(self, batch: Sequence[str]) -> CreateEmbeddingResponse:
        for attempt in range(self._settings.embedding_max_retries + 1):
            try:
                return await self._client.embeddings.create(
                    input=list(batch),
                    model=self._settings.openai_embedding_model,
                    dimensions=self._settings.embedding_dimensions,
                )
            except (APITimeoutError, APIConnectionError) as exc:
                if attempt == self._settings.embedding_max_retries:
                    raise AdapterError(
                        AdapterErrorCode.EMBEDDING_UPSTREAM,
                        "embedding provider unavailable",
                    ) from exc
            except APIStatusError as exc:
                retryable = exc.status_code == 429 or exc.status_code >= 500
                if not retryable or attempt == self._settings.embedding_max_retries:
                    raise AdapterError(
                        AdapterErrorCode.EMBEDDING_UPSTREAM,
                        "embedding provider rejected the request",
                    ) from exc

            await asyncio.sleep(min(2.0, 0.25 * (2**attempt)))

        raise AssertionError("retry loop must return or raise")

    async def close(self) -> None:
        await self._client.close()
