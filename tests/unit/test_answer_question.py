from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from ai_prime_capital.application.answer_question import AnswerQuestion
from ai_prime_capital.application.errors import AdapterError
from ai_prime_capital.config import Settings
from ai_prime_capital.domain.grounding import NO_RELEVANT_INFORMATION
from ai_prime_capital.domain.models import (
    Chunk,
    GroundedAnswer,
    ModelDecision,
    RetrievedCandidate,
    SourceLocator,
)

pytestmark = pytest.mark.unit


def build_settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-openai-secret",
        openai_chat_model="test-chat-model",
        embedding_dimensions=3,
        retrieval_min_score=0.8,
        max_question_chars=64,
        **overrides,
    )


def candidate(*, chunk_id: str = "chunk-a", score: float = 0.9) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="document-1",
            file_name="facts.txt",
            text="Revenue increased.",
            locator=SourceLocator(page_number=1),
        ),
        score=score,
    )


class SpyEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.result: tuple[tuple[float, ...], ...] = ((1.0, 0.0, 0.0),)

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(tuple(texts))
        return self.result


class SpyVectorStore:
    def __init__(self, candidates: Sequence[RetrievedCandidate]) -> None:
        self.candidates = tuple(candidates)
        self.calls: list[tuple[tuple[float, ...], int]] = []

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
    ) -> tuple[RetrievedCandidate, ...]:
        self.calls.append((tuple(query_vector), limit))
        return self.candidates


class SpyGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[RetrievedCandidate, ...]]] = []
        self.decision = ModelDecision(
            answerable=True,
            answer="Revenue increased.",
            citations=("chunk-a",),
        )

    async def generate(
        self,
        *,
        question: str,
        candidates: Sequence[RetrievedCandidate],
    ) -> ModelDecision:
        self.calls.append((question, tuple(candidates)))
        return self.decision


def make_service(
    candidates: Sequence[RetrievedCandidate],
    *,
    settings: Settings | None = None,
) -> tuple[AnswerQuestion, SpyEmbedder, SpyVectorStore, SpyGenerator]:
    embedder = SpyEmbedder()
    vector_store = SpyVectorStore(candidates)
    generator = SpyGenerator()
    service = AnswerQuestion(
        settings=settings or build_settings(),
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
    )
    return service, embedder, vector_store, generator


def assert_no_relevant_information(answer: GroundedAnswer) -> None:
    assert answer == GroundedAnswer(
        answer=NO_RELEVANT_INFORMATION,
        grounded=False,
        sources=(),
    )


async def test_empty_retrieval_returns_exact_fallback_without_generation() -> None:
    service, embedder, vector_store, generator = make_service(())

    answer = await service.execute("Which revenue fact is documented?")

    assert_no_relevant_information(answer)
    assert embedder.calls == [("Which revenue fact is documented?",)]
    assert vector_store.calls == [((1.0, 0.0, 0.0), 5)]
    assert generator.calls == []


async def test_below_threshold_retrieval_returns_fallback_without_generation() -> None:
    service, _, _, generator = make_service((candidate(score=0.79),))

    answer = await service.execute("Which revenue fact is documented?")

    assert_no_relevant_information(answer)
    assert generator.calls == []


async def test_threshold_equality_is_qualifying_and_reaches_generator() -> None:
    service, _, _, generator = make_service((candidate(score=0.8),))

    answer = await service.execute("Which revenue fact is documented?")

    assert answer.grounded is True
    assert answer.answer == "Revenue increased."
    assert tuple(source.chunk_id for source in answer.sources) == ("chunk-a",)
    assert len(generator.calls) == 1
    assert generator.calls[0][0] == "Which revenue fact is documented?"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
async def test_blank_question_is_rejected_before_any_downstream_call(question: str) -> None:
    service, embedder, vector_store, generator = make_service(())

    with pytest.raises(AdapterError) as error:
        await service.execute(question)

    assert error.value.code.value == "question.invalid_input"
    assert embedder.calls == []
    assert vector_store.calls == []
    assert generator.calls == []


async def test_question_is_trimmed_before_embedding_and_generation() -> None:
    service, embedder, _, generator = make_service((candidate(),))

    await service.execute("  Which revenue fact is documented?  ")

    assert embedder.calls == [("Which revenue fact is documented?",)]
    assert generator.calls[0][0] == "Which revenue fact is documented?"


async def test_question_over_maximum_length_is_rejected_before_embedding() -> None:
    service, embedder, vector_store, generator = make_service(())

    with pytest.raises(AdapterError) as error:
        await service.execute("x" * 65)

    assert error.value.code.value == "question.invalid_input"
    assert embedder.calls == []
    assert vector_store.calls == []
    assert generator.calls == []


async def test_requested_top_k_is_bounded_by_configured_retrieval_limit() -> None:
    service, _, vector_store, _ = make_service(())

    await service.execute("Which revenue fact is documented?", top_k=20)

    assert vector_store.calls == [((1.0, 0.0, 0.0), 5)]
