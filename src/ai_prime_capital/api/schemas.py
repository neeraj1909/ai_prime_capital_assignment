from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from ..application.ingest_documents import IngestionResult
from ..domain.models import GroundedAnswer, SourceReference

MAX_API_QUESTION_CHARS = 4_000


class SourceLocatorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int | None = Field(default=None, ge=1)
    section: str | None = None
    paragraph_index: int | None = Field(default=None, ge=0)
    table_index: int | None = Field(default=None, ge=0)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)


class SourceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: StrictStr = Field(min_length=1)
    file_name: StrictStr = Field(min_length=1)
    chunk_id: StrictStr = Field(min_length=1)
    snippet: StrictStr = Field(min_length=1)
    score: float
    locator: SourceLocatorResponse

    @classmethod
    def from_domain(cls, source: SourceReference) -> SourceResponse:
        return cls(
            document_id=source.document_id,
            file_name=source.file_name,
            chunk_id=source.chunk_id,
            snippet=source.snippet,
            score=source.score,
            locator=SourceLocatorResponse.model_validate(source.locator, from_attributes=True),
        )


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: StrictStr = Field(min_length=1)
    grounded: StrictBool
    sources: list[SourceResponse]

    @model_validator(mode="after")
    def validate_grounding_contract(self) -> AnswerResponse:
        if self.grounded != bool(self.sources):
            raise ValueError("grounded must match whether sources are present")
        return self

    @classmethod
    def from_domain(cls, answer: GroundedAnswer) -> AnswerResponse:
        return cls(
            answer=answer.answer,
            grounded=answer.grounded,
            sources=[SourceResponse.from_domain(source) for source in answer.sources],
        )


class IndexedDocumentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: StrictStr = Field(min_length=1)
    file_name: StrictStr = Field(min_length=1)
    media_type: StrictStr = Field(min_length=1)
    chunk_count: StrictInt = Field(ge=1)
    status: Literal["indexed"]


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[IndexedDocumentResponse] = Field(min_length=1)

    @classmethod
    def from_domain(cls, result: IngestionResult) -> DocumentUploadResponse:
        return cls(
            documents=[
                IndexedDocumentResponse.model_validate(document, from_attributes=True)
                for document in result.documents
            ]
        )


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        StrictStr,
        Field(
            min_length=1,
            max_length=MAX_API_QUESTION_CHARS,
            examples=["What revenue fact is stated in the uploaded documents?"],
        ),
    ]
    top_k: StrictInt | None = Field(default=None, ge=1, le=20, examples=[5])


class ProblemDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: StrictStr = "about:blank"
    title: StrictStr
    status: StrictInt
    detail: StrictStr
    instance: StrictStr
    code: StrictStr
    request_id: StrictStr = Field(min_length=1)


ANSWER_EXAMPLE = {
    "answer": "Revenue increased.",
    "grounded": True,
    "sources": [
        {
            "document_id": "doc_example",
            "file_name": "facts.txt",
            "chunk_id": "chunk_example",
            "snippet": "Revenue increased by 12%.",
            "score": 0.91,
            "locator": {"start_offset": 0, "end_offset": 25},
        }
    ],
}

NO_ANSWER_EXAMPLE = {
    "answer": "No relevant information was found in the uploaded documents.",
    "grounded": False,
    "sources": [],
}

DOCUMENT_UPLOAD_EXAMPLE = {
    "documents": [
        {
            "document_id": "doc_example",
            "file_name": "facts.txt",
            "media_type": "text/plain",
            "chunk_count": 3,
            "status": "indexed",
        }
    ]
}

PROBLEM_EXAMPLE = {
    "type": "about:blank",
    "title": "Request validation failed",
    "status": 422,
    "detail": "The request could not be validated.",
    "instance": "/api/v1/questions",
    "code": "request.invalid",
    "request_id": "req_example",
}


def problem_response_spec(example: dict[str, object]) -> dict[str, object]:
    return {
        "model": ProblemDetails,
        "content": {
            "application/problem+json": {
                "example": example,
            }
        },
    }


def problem_response_examples(
    examples: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "model": ProblemDetails,
        "content": {
            "application/problem+json": {
                "examples": {name: {"value": example} for name, example in examples.items()},
            }
        },
    }
