from __future__ import annotations

from fastapi import APIRouter, status

from ..dependencies import AnswerDependency
from ..schemas import (
    ANSWER_EXAMPLE,
    NO_ANSWER_EXAMPLE,
    PROBLEM_EXAMPLE,
    AnswerResponse,
    QuestionRequest,
    problem_response_spec,
)

router = APIRouter(prefix="/api/v1", tags=["questions"])


@router.post(
    "/questions",
    response_model=AnswerResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary="Answer a question from indexed documents",
    responses={
        200: {
            "description": "Grounded answer or the exact deterministic no-answer fallback.",
            "content": {
                "application/json": {
                    "examples": {
                        "grounded": {"value": ANSWER_EXAMPLE},
                        "no_relevant_information": {"value": NO_ANSWER_EXAMPLE},
                    }
                }
            },
        },
        422: problem_response_spec(PROBLEM_EXAMPLE),
        502: problem_response_spec(
            PROBLEM_EXAMPLE
            | {"status": 502, "title": "Upstream service failure", "code": "vector.response"}
        ),
        503: problem_response_spec(
            PROBLEM_EXAMPLE
            | {"status": 503, "title": "Dependency unavailable", "code": "vector.unavailable"}
        ),
    },
)
async def ask_question(
    payload: QuestionRequest,
    service: AnswerDependency,
) -> AnswerResponse:
    result = await service.execute(payload.question, top_k=payload.top_k)
    return AnswerResponse.from_domain(result)
