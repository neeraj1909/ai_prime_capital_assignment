from __future__ import annotations

from ..config import Settings
from ..domain.grounding import build_grounded_answer, qualify_candidates
from ..domain.models import GroundedAnswer
from .errors import AdapterError, AdapterErrorCode
from .ports import EmbedderPort, GeneratorPort, VectorStorePort


class AnswerQuestion:
    """Coordinate bounded question retrieval and the evidence gate."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        generator: GeneratorPort,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._generator = generator

    async def execute(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> GroundedAnswer:
        normalized_question = self._validate_question(question)
        limit = self._resolve_limit(top_k)

        try:
            vectors = tuple(await self._embedder.embed((normalized_question,)))
        except AdapterError:
            raise
        except (TypeError, ValueError) as exc:
            raise AdapterError(
                AdapterErrorCode.EMBEDDING_RESPONSE,
                "embedding provider returned an invalid response",
            ) from exc

        if len(vectors) != 1:
            raise AdapterError(
                AdapterErrorCode.EMBEDDING_RESPONSE,
                "embedding provider returned an unexpected vector count",
            )

        candidates = await self._vector_store.search(vectors[0], limit=limit)
        qualifying_candidates = qualify_candidates(
            candidates,
            self._settings.retrieval_min_score,
        )

        # This is the hard evidence gate: no qualifying evidence means no model
        # call and the exact deterministic fallback.
        if not qualifying_candidates:
            return build_grounded_answer(None, ())

        decision = await self._generator.generate(
            question=normalized_question,
            candidates=qualifying_candidates,
        )
        return build_grounded_answer(decision, qualifying_candidates)

    def _validate_question(self, question: str) -> str:
        if not isinstance(question, str):
            raise AdapterError(
                AdapterErrorCode.QUESTION_INVALID_INPUT,
                "question must be text",
            )

        normalized = question.strip()
        if not normalized:
            raise AdapterError(
                AdapterErrorCode.QUESTION_INVALID_INPUT,
                "question must not be blank",
            )
        if len(normalized) > self._settings.max_question_chars:
            raise AdapterError(
                AdapterErrorCode.QUESTION_INVALID_INPUT,
                "question exceeds the configured character limit",
            )
        return normalized

    def _resolve_limit(self, top_k: int | None) -> int:
        requested: object = self._settings.retrieval_top_k if top_k is None else top_k
        if (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or not 1 <= requested <= 20
        ):
            raise AdapterError(
                AdapterErrorCode.QUESTION_INVALID_INPUT,
                "top_k must be an integer between 1 and 20",
            )
        return min(requested, self._settings.retrieval_top_k)
