from math import inf, nan

import pytest

from ai_prime_capital.domain.grounding import (
    MAX_SNIPPET_CHARS,
    NO_RELEVANT_INFORMATION,
    build_grounded_answer,
    qualify_candidates,
)
from ai_prime_capital.domain.models import (
    Chunk,
    ExtractedBlock,
    GroundedAnswer,
    ModelDecision,
    RetrievedCandidate,
    SourceLocator,
    SourceReference,
)

pytestmark = pytest.mark.unit


def candidate(
    chunk_id: str,
    *,
    score: float = 0.8,
    document_id: str = "doc_abc",
    file_name: str = "facts.txt",
    text: str | None = None,
    locator: SourceLocator | None = None,
) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            file_name=file_name,
            text=text or f"Evidence for {chunk_id}",
            locator=locator or SourceLocator(),
        ),
        score=score,
    )


def assert_no_relevant_information(answer: GroundedAnswer) -> None:
    assert answer == GroundedAnswer(
        answer=NO_RELEVANT_INFORMATION,
        grounded=False,
        sources=(),
    )


def test_qualify_candidates_includes_threshold_equality_and_preserves_order() -> None:
    first = candidate("chunk-a", score=0.8)
    below = candidate("chunk-b", score=0.79)
    third = candidate("chunk-c", score=0.9)

    result = qualify_candidates((first, below, third), minimum_score=0.8)

    assert result == (first, third)


@pytest.mark.parametrize("minimum_score", [nan, inf, -inf])
def test_qualify_candidates_rejects_non_finite_thresholds(minimum_score: float) -> None:
    with pytest.raises(ValueError, match="minimum_score must be finite"):
        qualify_candidates((), minimum_score)


@pytest.mark.parametrize("score", [nan, inf, -inf])
def test_retrieved_candidate_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="score must be finite"):
        candidate("chunk-a", score=score)


@pytest.mark.parametrize(
    "decision",
    [
        None,
        ModelDecision(answerable=False),
        ModelDecision(
            answerable=True,
            answer="answer",
            citations=("chunk-a",),
            refused=True,
        ),
        ModelDecision(answerable=True, answer="   ", citations=("chunk-a",)),
        ModelDecision(answerable=True, answer="answer", citations=()),
    ],
)
def test_invalid_or_negative_model_decisions_fail_closed(
    decision: ModelDecision | None,
) -> None:
    answer = build_grounded_answer(decision, (candidate("chunk-a"),))

    assert_no_relevant_information(answer)


def test_empty_retrieval_fails_closed() -> None:
    decision = ModelDecision(
        answerable=True,
        answer="answer",
        citations=("chunk-a",),
    )

    assert_no_relevant_information(build_grounded_answer(decision, ()))


@pytest.mark.parametrize(
    "citations",
    [
        ("unknown",),
        ("chunk-a", "unknown"),
    ],
)
def test_unknown_or_mixed_citations_fail_closed(citations: tuple[str, ...]) -> None:
    decision = ModelDecision(
        answerable=True,
        answer="answer",
        citations=citations,
    )

    answer = build_grounded_answer(decision, (candidate("chunk-a"),))

    assert_no_relevant_information(answer)


def test_duplicate_citations_are_deduplicated_in_retrieval_order() -> None:
    first = candidate("chunk-a")
    second = candidate("chunk-b")
    decision = ModelDecision(
        answerable=True,
        answer="answer",
        citations=("chunk-b", "chunk-a", "chunk-b"),
    )

    answer = build_grounded_answer(decision, (first, second))

    assert answer.grounded is True
    assert tuple(source.chunk_id for source in answer.sources) == (
        "chunk-a",
        "chunk-b",
    )


def test_duplicate_retrieved_chunk_ids_fail_closed() -> None:
    decision = ModelDecision(
        answerable=True,
        answer="answer",
        citations=("chunk-a",),
    )

    answer = build_grounded_answer(
        decision,
        (
            candidate("chunk-a", document_id="doc_one"),
            candidate("chunk-a", document_id="doc_two"),
        ),
    )

    assert_no_relevant_information(answer)


def test_positive_answer_uses_only_server_owned_candidate_metadata() -> None:
    locator = SourceLocator(page_number=2, section="Returns")
    retrieved = candidate(
        "chunk-a",
        score=0.83,
        document_id="doc_policy",
        file_name="policy.pdf",
        text="Returns are accepted within 30 days.",
        locator=locator,
    )
    decision = ModelDecision(
        answerable=True,
        answer="  The return period is 30 days.  ",
        citations=("chunk-a",),
    )

    answer = build_grounded_answer(decision, (retrieved,))

    assert answer == GroundedAnswer(
        answer="The return period is 30 days.",
        grounded=True,
        sources=(
            SourceReference(
                document_id="doc_policy",
                file_name="policy.pdf",
                chunk_id="chunk-a",
                snippet="Returns are accepted within 30 days.",
                score=0.83,
                locator=locator,
            ),
        ),
    )


def test_source_snippets_are_normalized_and_bounded() -> None:
    retrieved = candidate(
        "chunk-a",
        text="  Revenue   increased.\n\nThe quarter\twas strong.  ",
    )
    answer = build_grounded_answer(
        ModelDecision(answerable=True, answer="Revenue increased.", citations=("chunk-a",)),
        (retrieved,),
    )

    assert answer.grounded is True
    assert answer.sources[0].snippet == "Revenue increased. The quarter was strong."

    long_text = "word " * MAX_SNIPPET_CHARS
    long_answer = build_grounded_answer(
        ModelDecision(answerable=True, answer="answer", citations=("chunk-a",)),
        (candidate("chunk-a", text=long_text),),
    )

    snippet = long_answer.sources[0].snippet
    assert len(snippet) == MAX_SNIPPET_CHARS
    assert snippet.endswith("...")
    assert snippet.startswith("word word")


@pytest.mark.parametrize(
    "decision",
    [
        object(),
        ModelDecision(answerable=True, answer=None, citations=("chunk-a",)),  # type: ignore[arg-type]
        ModelDecision(answerable=True, answer="answer", citations=None),  # type: ignore[arg-type]
        ModelDecision(answerable=True, answer="answer", citations=(["chunk-a"],)),  # type: ignore[arg-type]
    ],
)
def test_malformed_model_decisions_fail_closed_without_leaking_exceptions(
    decision: object,
) -> None:
    answer = build_grounded_answer(decision, (candidate("chunk-a"),))  # type: ignore[arg-type]

    assert_no_relevant_information(answer)


@pytest.mark.parametrize(
    "locator_kwargs",
    [
        {"page_number": 0},
        {"paragraph_index": -1},
        {"table_index": -1},
        {"start_offset": -1, "end_offset": 1},
        {"start_offset": 1},
        {"end_offset": 1},
        {"start_offset": 2, "end_offset": 1},
        {"section": "   "},
    ],
)
def test_source_locator_rejects_invalid_values(
    locator_kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        SourceLocator(**locator_kwargs)  # type: ignore[arg-type]


def test_extracted_block_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="text must not be blank"):
        ExtractedBlock(text="   ", locator=SourceLocator())


@pytest.mark.parametrize(
    "field_name",
    ["chunk_id", "document_id", "file_name", "text"],
)
def test_chunk_rejects_blank_required_fields(field_name: str) -> None:
    values = {
        "chunk_id": "chunk-a",
        "document_id": "doc_abc",
        "file_name": "facts.txt",
        "text": "evidence",
    }
    values[field_name] = "   "

    with pytest.raises(ValueError):
        Chunk(**values, locator=SourceLocator())


def test_grounded_answer_rejects_inconsistent_grounding_state() -> None:
    source = SourceReference(
        document_id="doc_abc",
        file_name="facts.txt",
        chunk_id="chunk-a",
        snippet="evidence",
        score=0.8,
        locator=SourceLocator(),
    )

    with pytest.raises(ValueError, match="grounded must match"):
        GroundedAnswer(answer="answer", grounded=True, sources=())

    with pytest.raises(ValueError, match="grounded must match"):
        GroundedAnswer(answer="answer", grounded=False, sources=(source,))
