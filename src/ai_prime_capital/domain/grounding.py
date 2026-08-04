from collections.abc import Iterable, Sequence
from math import isfinite

from .models import (
    GroundedAnswer,
    ModelDecision,
    RetrievedCandidate,
    SourceReference,
)

NO_RELEVANT_INFORMATION = "No relevant information was found in the uploaded documents."


def qualify_candidates(
    candidates: Iterable[RetrievedCandidate],
    minimum_score: float,
) -> tuple[RetrievedCandidate, ...]:
    if not isfinite(minimum_score):
        raise ValueError("minimum_score must be finite")

    return tuple(candidate for candidate in candidates if candidate.score >= minimum_score)


def _fallback() -> GroundedAnswer:
    return GroundedAnswer(
        answer=NO_RELEVANT_INFORMATION,
        grounded=False,
        sources=(),
    )


def build_grounded_answer(
    decision: ModelDecision | None,
    candidates: Sequence[RetrievedCandidate],
) -> GroundedAnswer:
    if (
        decision is None
        or decision.refused
        or not decision.answerable
        or not decision.answer.strip()
        or not candidates
    ):
        return _fallback()

    candidate_ids = tuple(candidate.chunk.chunk_id for candidate in candidates)

    # Duplicate retrieval records are ambiguous and should fail closed.
    if len(candidate_ids) != len(set(candidate_ids)):
        return _fallback()

    citation_ids = set(decision.citations)
    if not citation_ids or not citation_ids.issubset(candidate_ids):
        return _fallback()

    # Model ordering and duplicate citations are ignored. Sources are deduplicated
    # and returned in server-owned retrieval order.
    sources = tuple(
        SourceReference(
            document_id=candidate.chunk.document_id,
            file_name=candidate.chunk.file_name,
            chunk_id=candidate.chunk.chunk_id,
            snippet=candidate.chunk.text,
            score=candidate.score,
            locator=candidate.chunk.locator,
        )
        for candidate in candidates
        if candidate.chunk.chunk_id in citation_ids
    )

    return GroundedAnswer(
        answer=decision.answer.strip(),
        grounded=True,
        sources=sources,
    )
