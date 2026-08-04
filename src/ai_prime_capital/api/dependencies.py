from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from ..application.answer_question import AnswerQuestion
from ..application.errors import AdapterError, AdapterErrorCode
from ..application.ingest_documents import IngestDocuments
from ..bootstrap import RuntimeResources
from ..config import Settings


def get_resources(request: Request) -> RuntimeResources:
    resources = getattr(request.app.state, "resources", None)

    if not isinstance(resources, RuntimeResources):
        raise RuntimeError("application resources are not initialized")

    return resources


ResourcesDependency = Annotated[
    RuntimeResources,
    Depends(get_resources),
]


def get_settings(resources: ResourcesDependency) -> Settings:
    if resources.settings is None:
        raise RuntimeError("settings are unavailable")

    return resources.settings


SettingsDependency = Annotated[
    Settings,
    Depends(get_settings),
]


def get_ingest_documents(resources: ResourcesDependency) -> IngestDocuments:
    service = resources.ingest_documents
    if service is None:
        raise AdapterError(
            AdapterErrorCode.VECTOR_UNAVAILABLE,
            "document ingestion service is unavailable",
        )
    return service


IngestDependency = Annotated[
    IngestDocuments,
    Depends(get_ingest_documents),
]


def get_answer_question(resources: ResourcesDependency) -> AnswerQuestion:
    service = resources.answer_question
    if service is None:
        raise AdapterError(
            AdapterErrorCode.VECTOR_UNAVAILABLE,
            "question answering service is unavailable",
        )
    return service


AnswerDependency = Annotated[
    AnswerQuestion,
    Depends(get_answer_question),
]
