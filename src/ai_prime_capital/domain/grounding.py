from collections.abc import Iterable, Sequence
from math import isfinite

from .models import (
    GroundedAnswer,
    ModelDecision,
    RetrievedCandidate,
    SourceReference,
)

NO_RELEVANT_INFORMATION = "No relevant information was found in the uploaded documents."
MAX_SNIPPET_CHARS = 400


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


def normalize_snippet(text: str) -> str:
    """Return a bounded, transport-safe source preview owned by the server."""

    normalized = " ".join(text.split())
    if len(normalized) <= MAX_SNIPPET_CHARS:
        return normalized
    return f"{normalized[: MAX_SNIPPET_CHARS - 3].rstrip()}..."


def build_grounded_answer(
    decision: ModelDecision | None,
    candidates: Sequence[RetrievedCandidate],
) -> GroundedAnswer:
    if (
        not isinstance(decision, ModelDecision)
        or decision.refused
        or not isinstance(decision.answerable, bool)
        or not decision.answerable
        or not isinstance(decision.answer, str)
        or not decision.answer.strip()
        or not candidates
    ):
        return _fallback()

    try:
        candidate_ids = tuple(candidate.chunk.chunk_id for candidate in candidates)
    except (AttributeError, TypeError, ValueError):
        return _fallback()

    if any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in candidate_ids):
        return _fallback()

    # Duplicate retrieval records are ambiguous and should fail closed.
    if len(candidate_ids) != len(set(candidate_ids)):
        return _fallback()

    if not isinstance(decision.citations, tuple) or not decision.citations:
        return _fallback()
    if any(
        not isinstance(citation, str) or not citation.strip() or citation != citation.strip()
        for citation in decision.citations
    ):
        return _fallback()

    citation_ids = set(decision.citations)
    if not citation_ids or not citation_ids.issubset(candidate_ids):
        return _fallback()

    # Model ordering and duplicate citations are ignored. Sources are deduplicated
    # and returned in server-owned retrieval order.
    try:
        sources = tuple(
            SourceReference(
                document_id=candidate.chunk.document_id,
                file_name=candidate.chunk.file_name,
                chunk_id=candidate.chunk.chunk_id,
                snippet=normalize_snippet(candidate.chunk.text),
                score=candidate.score,
                locator=candidate.chunk.locator,
            )
            for candidate in candidates
            if candidate.chunk.chunk_id in citation_ids
        )
    except (AttributeError, TypeError, ValueError):
        return _fallback()
    if not sources or any(not source.snippet for source in sources):
        return _fallback()

    try:
        return GroundedAnswer(
            answer=decision.answer.strip(),
            grounded=True,
            sources=sources,
        )
    except (AttributeError, TypeError, ValueError):
        return _fallback()
