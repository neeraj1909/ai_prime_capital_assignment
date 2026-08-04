from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from ai_prime_capital.application.errors import AdapterError, AdapterErrorCode
from ai_prime_capital.config import Settings
from ai_prime_capital.domain.models import Chunk, SourceLocator
from ai_prime_capital.infrastructure.qdrant_store import QdrantVectorStore

pytestmark = pytest.mark.integration


REQUIRED_SETTINGS = {
    "openai_api_key": "test-openai-secret",
    "openai_chat_model": "test-chat-model",
}


def build_settings(*, collection: str, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        **(
            REQUIRED_SETTINGS
            | {
                "embedding_dimensions": 3,
                "qdrant_url": os.environ.get("QDRANT_TEST_URL", "http://localhost:6333"),
                "qdrant_collection": collection,
                "qdrant_distance": "dot",
                "retrieval_min_score": 0.8,
                "retrieval_top_k": 2,
            }
            | overrides
        ),
    )


def make_chunks() -> tuple[Chunk, ...]:
    records = (
        (
            "00000000-0000-0000-0000-000000000101",
            "Revenue increased.",
        ),
        (
            "00000000-0000-0000-0000-000000000102",
            "Revenue remained stable.",
        ),
        (
            "00000000-0000-0000-0000-000000000103",
            "Costs decreased.",
        ),
    )
    chunks: list[Chunk] = []
    offset = 0
    for chunk_id, text in records:
        end_offset = offset + len(text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id="document-1",
                file_name="facts.txt",
                text=text,
                locator=SourceLocator(start_offset=offset, end_offset=end_offset),
                char_count=len(text),
                token_count=len(text.split()),
            )
        )
        offset = end_offset + 1
    return tuple(chunks)


@pytest.fixture
async def qdrant_fixture() -> AsyncIterator[tuple[QdrantVectorStore, AsyncQdrantClient, Settings]]:
    collection = f"test_task_21_{uuid4().hex}"
    settings = build_settings(collection=collection)
    client = AsyncQdrantClient(url=str(settings.qdrant_url))
    try:
        await client.get_collections()
    except Exception as exc:  # pragma: no cover - only exercised without a server
        await client.close()
        pytest.skip(f"Qdrant integration prerequisite unavailable at {settings.qdrant_url}: {exc}")

    store = QdrantVectorStore(settings, client=client)
    try:
        yield store, client, settings
    finally:
        try:
            await client.delete_collection(collection_name=collection)
        finally:
            await store.close()


async def test_search_returns_ranked_payloads_above_threshold_and_honors_top_k(
    qdrant_fixture: tuple[QdrantVectorStore, AsyncQdrantClient, Settings],
) -> None:
    store, _, settings = qdrant_fixture
    chunks = make_chunks()
    vectors = (
        (1.0, 0.0, 0.0),
        (0.81, 0.6, 0.0),
        (0.2, 0.98, 0.0),
    )

    await store.upsert(chunks, vectors)

    results = await store.search((1.0, 0.0, 0.0), limit=50)

    assert len(results) == settings.retrieval_top_k
    assert [candidate.chunk.chunk_id for candidate in results] == [
        chunks[0].chunk_id,
        chunks[1].chunk_id,
    ]
    assert [candidate.score for candidate in results] == pytest.approx([1.0, 0.81], abs=1e-5)
    assert all(candidate.score >= settings.retrieval_min_score for candidate in results)
    assert results[0].chunk == chunks[0]
    assert results[1].chunk.locator.start_offset == chunks[1].locator.start_offset
    assert results[1].chunk.locator.end_offset == chunks[1].locator.end_offset


async def test_search_empty_collection_returns_no_candidates(
    qdrant_fixture: tuple[QdrantVectorStore, AsyncQdrantClient, Settings],
) -> None:
    store, _, _ = qdrant_fixture

    await store.ensure_compatible()

    assert await store.search((1.0, 0.0, 0.0), limit=50) == ()


async def test_search_rejects_malformed_payload_as_vector_response(
    qdrant_fixture: tuple[QdrantVectorStore, AsyncQdrantClient, Settings],
) -> None:
    store, client, settings = qdrant_fixture
    point_id = "00000000-0000-0000-0000-000000000199"

    await store.ensure_compatible()
    await client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            PointStruct(
                id=point_id,
                vector={settings.vector_name: [1.0, 0.0, 0.0]},
                payload={"chunk_id": point_id},
            )
        ],
        wait=True,
    )

    with pytest.raises(AdapterError) as error:
        await store.search((1.0, 0.0, 0.0), limit=1)

    assert error.value.code is AdapterErrorCode.VECTOR_RESPONSE
