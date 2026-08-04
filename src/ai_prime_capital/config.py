from __future__ import annotations

import json
from hashlib import sha256
from typing import Literal, Self

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Luna Max

MIB = 1024 * 1024


class Settings(BaseSettings):
    """Validated runtime configuration owned by the application boundary"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="forbid",
        validate_default=True,
        frozen=True,
    )

    # OpenAI
    openai_api_key: SecretStr
    openai_chat_model: str

    # Explicit embedding hypothesis. Changing these changes vector identity.
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0)
    openai_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        allow_inf_nan=False,
    )

    # Qdrant
    qdrant_url: AnyHttpUrl = AnyHttpUrl("http://localhost:6333")
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(
        default="ai_prime_capital_documents", pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
    )
    qdrant_distance: Literal["cosine", "dot", "euclid"] = "cosine"
    qdrant_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        allow_inf_nan=False,
    )

    # Upload and extraction safety
    max_files_per_request: int = Field(default=10, ge=1, le=100)
    max_file_bytes: int = Field(default=10 * MIB, gt=0)
    max_request_bytes: int = Field(default=50 * MIB, gt=0)
    max_extracted_chars: int = Field(default=5_000_000, gt=0)
    docx_max_entries: int = Field(default=1_000, gt=0)
    docx_max_expanded_bytes: int = Field(default=50 * MIB, gt=0)
    docx_max_compression_ratio: float = Field(
        default=100.0,
        gt=0,
        allow_inf_nan=False,
    )

    # Chunking
    chunk_max_tokens: int = Field(default=600, ge=64)
    chunk_overlap_tokens: int = Field(default=100, ge=0)

    # Bump when chunk boundaries or chunking semantics become incompatible.
    chunk_schema_version: str = "v1"

    # Retrieval and generation
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_min_score: float = Field(
        default=0.35,
        allow_inf_nan=False,
    )
    max_question_chars: int = Field(default=4_000, gt=0)
    max_answer_tokens: int = Field(default=800, gt=0)

    # Logging
    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    @field_validator(
        "openai_chat_model",
        "openai_embedding_model",
        "chunk_schema_version",
    )
    @classmethod
    def require_clean_nonblank_text(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("must be non-empty and have no surrounding whitespace")
        return value

    @field_validator("openai_api_key")
    @classmethod
    def require_clean_openai_secret(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()

        if not secret or secret != secret.strip():
            raise ValueError("must be non-empty and have no surrounding whiespaces")

        return value

    @field_validator("qdrant_api_key")
    @classmethod
    def validate_optional_qdrant_secret(
        cls,
        value: SecretStr | None,
    ) -> SecretStr | None:
        if value is None:
            return None

        secret = value.get_secret_value()

        if not secret or secret != secret.strip():
            raise ValueError("must be non-empty and have no surrounding whitespace")

        return value

    @model_validator(mode="after")
    def validate_related_limits(self) -> Self:
        if self.max_request_bytes < self.max_file_bytes:
            raise ValueError("max_request_bytes must be greater than or equal to max_file_bytes")

        if self.chunk_overlap_tokens >= self.chunk_max_tokens:
            raise ValueError("chunk_overlap_tokens must be less than chunk_max_tokens")

        return self

    @property
    def vector_fingerprint(self) -> str:
        """Stable identifier for an embedding/chunk compatibility contract."""

        payload = json.dumps(
            {
                "chunk_schema_version": self.chunk_schema_version,
                "dimensions": self.embedding_dimensions,
                "distance": self.qdrant_distance,
                "model": self.openai_embedding_model,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def vector_name(self) -> str:
        return f"text_{self.vector_fingerprint}"
