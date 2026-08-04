from __future__ import annotations

from enum import StrEnum


class AdapterErrorCode(StrEnum):
    """Stable, transport-independent failures raised by infrastructure adapters."""

    EMBEDDING_INVALID_INPUT = "embedding.invalid_input"
    EMBEDDING_UPSTREAM = "embedding.upstream"
    EMBEDDING_DIMENSION = "embedding.dimension"
    EMBEDDING_RESPONSE = "embedding.response"
    VECTOR_INVALID_INPUT = "vector.invalid_input"
    VECTOR_UNAVAILABLE = "vector.unavailable"
    VECTOR_INCOMPATIBLE = "vector.incompatible"
    VECTOR_RESPONSE = "vector.response"
    VECTOR_WRITE = "vector.write"
    INGESTION_INVALID_INPUT = "ingestion.invalid_input"
    INGESTION_RESPONSE = "ingestion.response"


class AdapterError(RuntimeError):
    """Safe application error for provider and vector-store boundary failures."""

    def __init__(self, code: AdapterErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail)
