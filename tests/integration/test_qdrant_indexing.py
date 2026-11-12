from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

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
            }
            | overrides
        ),
    )


def make_chunks() -> tuple[Chunk, Chunk]:
    return (
        Chunk(
            chunk_id="00000000-0000-0000-0000-000000000001",
            document_id="document-1",
            file_name="facts.txt",
            text="Revenue increased.",
            locator=SourceLocator(start_offset=0, end_offset=18),
            char_count=18,
            token_count=3,
        ),
        Chunk(
            chunk_id="00000000-0000-0000-0000-000000000002",
            document_id="document-1",
            file_name="facts.txt",
            text="Costs decreased.",
            locator=SourceLocator(start_offset=19, end_offset=35),
            char_count=16,
            token_count=3,
        ),
    )


@pytest.fixture
async def qdrant_fixture() -> AsyncIterator[tuple[QdrantVectorStore, AsyncQdrantClient, Settings]]:
    collection = f"test_task_14_{uuid4().hex}"
    settings = build_settings(collection=collection)
    client = AsyncQdrantClient(url=str(settings.qdrant_url))
    try:
        await client.get_collections()
    except Exception as exc:  # pragma: no cover - only exercised without a server
        await client.close()
        if os.environ.get("REQUIRE_QDRANT_INTEGRATION") == "1":
            pytest.fail(
                f"Qdrant integration prerequisite unavailable at {settings.qdrant_url}: {exc}"
            )
        pytest.skip(f"Qdrant integration prerequisite unavailable at {settings.qdrant_url}: {exc}")

    store = QdrantVectorStore(settings, client=client)
    try:
        yield store, client, settings
    finally:
        try:
            await client.delete_collection(collection_name=collection)
        finally:
            await store.close()


async def test_create_upsert_reupload_and_search_are_idempotent(
    qdrant_fixture: tuple[QdrantVectorStore, AsyncQdrantClient, Settings],
) -> None:
    store, client, settings = qdrant_fixture
    chunks = make_chunks()
    vectors = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    await store.ensure_compatible()
    await store.upsert(chunks, vectors)
    await store.upsert(chunks, vectors)

    count = await client.count(settings.qdrant_collection, exact=True)
    results = await store.search((1.0, 0.0, 0.0), limit=2)

    assert count.count == 2
    assert results[0].chunk.chunk_id == "00000000-0000-0000-0000-000000000001"
    assert results[0].chunk.locator.start_offset == 0
    assert results[0].score == pytest.approx(1.0)
    assert results[0].chunk.text == "Revenue increased."


@pytest.mark.parametrize(
    ("vector_size", "distance", "metadata_overrides", "expected_code"),
    [
        (2, Distance.COSINE, {}, AdapterErrorCode.VECTOR_INCOMPATIBLE),
        (3, Distance.DOT, {}, AdapterErrorCode.VECTOR_INCOMPATIBLE),
        (3, Distance.COSINE, {"vector_fingerprint": "wrong"}, AdapterErrorCode.VECTOR_INCOMPATIBLE),
    ],
)
async def test_incompatible_collection_fails_closed_without_deletion(
    qdrant_fixture: tuple[QdrantVectorStore, AsyncQdrantClient, Settings],
    vector_size: int,
    distance: Distance,
    metadata_overrides: dict[str, object],
    expected_code: AdapterErrorCode,
) -> None:
    store, client, settings = qdrant_fixture
    metadata = {
        "embedding_model": settings.openai_embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "distance": settings.qdrant_distance,
        "vector_fingerprint": settings.vector_fingerprint,
        "chunk_schema_version": settings.chunk_schema_version,
    } | metadata_overrides
    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            settings.vector_name: VectorParams(size=vector_size, distance=distance),
        },
        metadata=metadata,
    )

    with pytest.raises(AdapterError) as error:
        await store.ensure_compatible()

    assert error.value.code is expected_code
    assert await client.collection_exists(settings.qdrant_collection)
