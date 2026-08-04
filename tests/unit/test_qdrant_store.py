from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, ScoredPoint, VectorParams

from ai_prime_capital.application.errors import AdapterError, AdapterErrorCode
from ai_prime_capital.config import Settings
from ai_prime_capital.domain.models import Chunk, SourceLocator
from ai_prime_capital.infrastructure.qdrant_store import QdrantVectorStore

pytestmark = pytest.mark.unit


CHUNK_ID = "00000000-0000-0000-0000-000000000001"


def build_settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-openai-secret",
        openai_chat_model="test-chat-model",
        embedding_dimensions=3,
        qdrant_collection=f"unit_task_14_{uuid4().hex}",
        **overrides,
    )


def make_chunk(chunk_id: str = CHUNK_ID) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="document-1",
        file_name="facts.txt",
        text="Revenue increased.",
        locator=SourceLocator(start_offset=0, end_offset=18),
        char_count=18,
        token_count=3,
    )


@pytest.fixture
async def local_store() -> AsyncIterator[tuple[QdrantVectorStore, AsyncQdrantClient, Settings]]:
    settings = build_settings()
    client = AsyncQdrantClient(location=":memory:")
    store = QdrantVectorStore(settings, client=client)
    try:
        yield store, client, settings
    finally:
        await store.close()


async def test_local_collection_upsert_search_and_reupload_are_idempotent(
    local_store: tuple[QdrantVectorStore, AsyncQdrantClient, Settings],
) -> None:
    store, client, settings = local_store
    chunk = make_chunk()

    await store.upsert((chunk,), ((1.0, 0.0, 0.0),))
    await store.upsert((chunk,), ((1.0, 0.0, 0.0),))
    results = await store.search((1.0, 0.0, 0.0), limit=1)

    assert (await client.count(settings.qdrant_collection, exact=True)).count == 1
    assert results[0].chunk == chunk
    assert results[0].score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("chunks", "vectors", "expected_code"),
    [
        ((make_chunk(),), (), AdapterErrorCode.VECTOR_INVALID_INPUT),
        ((make_chunk("not-a-uuid"),), ((1.0, 0.0, 0.0),), AdapterErrorCode.VECTOR_INVALID_INPUT),
        ((make_chunk(),), ((1.0, 0.0),), AdapterErrorCode.VECTOR_INVALID_INPUT),
        ((make_chunk(),), ((1.0, float("nan"), 0.0),), AdapterErrorCode.VECTOR_INVALID_INPUT),
        ((make_chunk(),), (("bad", 0.0, 0.0),), AdapterErrorCode.VECTOR_INVALID_INPUT),
    ],
)
async def test_upsert_rejects_invalid_inputs(
    local_store: tuple[QdrantVectorStore, AsyncQdrantClient, Settings],
    chunks: tuple[Chunk, ...],
    vectors: tuple[tuple[object, ...], ...] | tuple[()],
    expected_code: AdapterErrorCode,
) -> None:
    store, _, _ = local_store

    with pytest.raises(AdapterError) as error:
        await store.upsert(chunks, vectors)  # type: ignore[arg-type]

    assert error.value.code is expected_code


@pytest.mark.parametrize(
    ("query_vector", "limit"),
    [((1.0, 0.0), 1), ((1.0, float("inf"), 0.0), 1), (("bad", 0.0, 0.0), 1), ((1.0, 0.0, 0.0), 0)],
)
async def test_search_rejects_invalid_queries(
    local_store: tuple[QdrantVectorStore, AsyncQdrantClient, Settings],
    query_vector: tuple[object, ...],
    limit: int,
) -> None:
    store, _, _ = local_store

    with pytest.raises(AdapterError) as error:
        await store.search(query_vector, limit=limit)  # type: ignore[arg-type]

    assert error.value.code is AdapterErrorCode.VECTOR_INVALID_INPUT


async def test_incompatible_local_collection_fails_closed(
    local_store: tuple[QdrantVectorStore, AsyncQdrantClient, Settings],
) -> None:
    store, client, settings = local_store
    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={settings.vector_name: VectorParams(size=2, distance=Distance.COSINE)},
        metadata=store._expected_metadata(),
    )

    with pytest.raises(AdapterError) as error:
        await store.ensure_compatible()

    assert error.value.code is AdapterErrorCode.VECTOR_INCOMPATIBLE


def test_malformed_point_payload_is_rejected() -> None:
    point = ScoredPoint(
        id=CHUNK_ID,
        version=0,
        score=1.0,
        payload={"chunk_id": CHUNK_ID},
    )

    with pytest.raises(AdapterError) as error:
        QdrantVectorStore._chunk_from_point(point)

    assert error.value.code is AdapterErrorCode.VECTOR_RESPONSE
