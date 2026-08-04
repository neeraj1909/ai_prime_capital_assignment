from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ai_prime_capital.config import Settings
from ai_prime_capital.domain.grounding import build_grounded_answer
from ai_prime_capital.domain.models import (
    Chunk,
    ModelDecision,
    RetrievedCandidate,
    SourceLocator,
)

pytestmark = pytest.mark.evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "documents"
CASES_PATH = Path(__file__).with_name("cases.yaml")

TOP_K = 5
MINIMUM_SCORE = 0.35
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "did",
        "documented",
        "does",
        "for",
        "how",
        "in",
        "is",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "with",
        "recorded",
    }
)


@dataclass(frozen=True, slots=True)
class EvaluationChunk:
    chunk_id: str
    document_id: str
    file_name: str
    text: str


CORPUS = (
    EvaluationChunk(
        "eval-revenue",
        "doc-revenue",
        "facts.pdf",
        (FIXTURE_ROOT / "evaluation_revenue.txt").read_text(encoding="utf-8"),
    ),
    EvaluationChunk(
        "eval-returns",
        "doc-returns",
        "returns.docx",
        (FIXTURE_ROOT / "evaluation_returns.txt").read_text(encoding="utf-8"),
    ),
    EvaluationChunk(
        "eval-policy",
        "doc-policy",
        "policy.txt",
        (FIXTURE_ROOT / "evaluation_policy.txt").read_text(encoding="utf-8"),
    ),
    EvaluationChunk(
        "eval-allocation",
        "doc-allocation",
        "allocation.txt",
        (FIXTURE_ROOT / "evaluation_allocation.txt").read_text(encoding="utf-8"),
    ),
    EvaluationChunk(
        "eval-adversarial",
        "doc-adversarial",
        "adversarial_instructions.txt",
        (FIXTURE_ROOT / "adversarial_instructions.txt").read_text(encoding="utf-8"),
    ),
)


def _normalize_token(token: str) -> str:
    token = {
        "increased": "increase",
        "increasing": "increase",
        "percentage": "percent",
    }.get(token, token)
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _terms(value: str) -> frozenset[str]:
    return frozenset(
        _normalize_token(token)
        for token in TOKEN_PATTERN.findall(value.lower())
        if token not in STOP_WORDS and len(token) > 2
    )


def _retrieved_candidate(chunk: EvaluationChunk, score: float) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk=Chunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            file_name=chunk.file_name,
            text=chunk.text,
            locator=SourceLocator(),
        ),
        score=score,
    )


def retrieve(question: str) -> tuple[RetrievedCandidate, ...]:
    query_terms = _terms(question)
    ranked = sorted(
        (
            _retrieved_candidate(
                chunk,
                len(query_terms & _terms(chunk.text)) / len(query_terms) if query_terms else 0.0,
            )
            for chunk in CORPUS
        ),
        key=lambda candidate: (-candidate.score, candidate.chunk.chunk_id),
    )
    return tuple(candidate for candidate in ranked[:TOP_K] if candidate.score >= MINIMUM_SCORE)


def load_cases() -> tuple[dict[str, Any], ...]:
    # The file intentionally uses JSON syntax, which is a strict YAML subset,
    # so evaluation has no undeclared parser dependency.
    raw = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(case, dict) for case in raw):
        raise AssertionError("evaluation cases must be a list of mappings")
    return tuple(raw)


def test_case_labels_are_explicit_and_reference_only_known_chunks() -> None:
    known_ids = {chunk.chunk_id for chunk in CORPUS}
    cases = load_cases()

    assert len(cases) >= 10
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert isinstance(case["question"], str) and case["question"].strip()
        assert isinstance(case["answerable"], bool)
        relevant_ids = case["relevant_chunk_ids"]
        citations = case["citations"]
        assert isinstance(relevant_ids, list)
        assert isinstance(citations, list)
        assert set(relevant_ids) <= known_ids
        assert set(citations) <= set(relevant_ids)
        if case["answerable"]:
            assert relevant_ids
            assert citations
        else:
            assert relevant_ids == []
            assert citations == []


def test_calibration_limits_match_runtime_retrieval_defaults() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-openai-secret",
        openai_chat_model="test-chat-model",
    )

    assert settings.retrieval_top_k == TOP_K
    assert settings.retrieval_min_score == MINIMUM_SCORE
    assert settings.chunk_max_tokens == 600
    assert settings.chunk_overlap_tokens == 100


def test_retrieval_quality_meets_calibration_gates(capsys: pytest.CaptureFixture[str]) -> None:
    cases = load_cases()
    answerable = [case for case in cases if case["answerable"]]
    unanswerable = [case for case in cases if not case["answerable"]]
    recalled = 0
    false_positives = 0
    citation_valid = 0
    case_lines: list[str] = []

    for case in cases:
        candidates = retrieve(case["question"])
        candidate_ids = {candidate.chunk.chunk_id for candidate in candidates}
        expected_ids = set(case["relevant_chunk_ids"])
        if case["answerable"] and expected_ids <= candidate_ids:
            recalled += 1
            decision = ModelDecision(
                answerable=True,
                answer="Evidence-supported answer.",
                citations=tuple(case["citations"]),
            )
            grounded = build_grounded_answer(decision, candidates)
            if grounded.grounded and set(case["citations"]) == {
                source.chunk_id for source in grounded.sources
            }:
                citation_valid += 1
        elif not case["answerable"] and candidates:
            false_positives += 1

        case_lines.append(
            f"{case['id']}: candidates={tuple(sorted(candidate_ids))} "
            f"expected={tuple(sorted(expected_ids))}"
        )

    recall_at_5 = recalled / len(answerable)
    false_positive_rate = false_positives / len(unanswerable)
    citation_validity = citation_valid / len(answerable)
    print("\n".join(case_lines))
    print(
        "metrics: "
        f"recall@5={recall_at_5:.3f} "
        f"false_positive_rate={false_positive_rate:.3f} "
        f"citation_validity={citation_validity:.3f}"
    )
    captured = capsys.readouterr().out

    assert "metrics:" in captured
    assert recall_at_5 >= 0.90
    assert false_positive_rate <= 0.10
    assert citation_validity == 1.0


def test_prompt_injection_case_preserves_grounding_contract() -> None:
    case = next(case for case in load_cases() if case["category"] == "prompt-injection")
    candidates = retrieve(case["question"])

    assert [candidate.chunk.chunk_id for candidate in candidates] == ["eval-adversarial"]
    answer = build_grounded_answer(
        ModelDecision(
            answerable=True,
            answer="Capital reserves are 10 million dollars.",
            citations=("eval-adversarial",),
        ),
        candidates,
    )

    assert answer.grounded is True
    assert answer.answer == "Capital reserves are 10 million dollars."
    assert answer.sources[0].document_id == "doc-adversarial"
    assert "IGNORE ALL PREVIOUS" in answer.sources[0].snippet
