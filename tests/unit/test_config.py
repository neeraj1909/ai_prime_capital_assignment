from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai_prime_capital.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_SETTINGS = {
    "openai_api_key": "test-openai-secret",
    "openai_chat_model": "test-chat-model",
}

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep settings tests independent of a developer's shell environment."""

    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)


def build_settings(**overrides: Any) -> Settings:
    values = REQUIRED_SETTINGS | overrides
    return Settings(_env_file=None, **values)


def set_required_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-secret")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "test-chat-model")


def test_provider_secret_and_chat_model_are_required() -> None:
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)

    locations = {item["loc"] for item in error.value.errors()}

    assert ("openai_api_key",) in locations
    assert ("openai_chat_model",) in locations


def test_settings_are_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "384")

    settings = Settings(_env_file=None)

    assert settings.openai_chat_model == "test-chat-model"
    assert settings.openai_embedding_model == "test-embedding-model"
    assert settings.embedding_dimensions == 384
    assert settings.openai_api_key.get_secret_value() == "test-openai-secret"


@pytest.mark.parametrize(
    ("environment_name", "invalid_value"),
    [
        ("OPENAI_API_KEY", " "),
        ("OPENAI_CHAT_MODEL", " "),
        ("EMBEDDING_DIMENSIONS", "0"),
        ("QDRANT_URL", "ftp://localhost"),
        ("QDRANT_COLLECTION", "invalid/name"),
        ("QDRANT_DISTANCE", "manhattan"),
        ("MAX_FILES_PER_REQUEST", "0"),
        ("MAX_FILE_BYTES", "0"),
        ("MAX_REQUEST_BYTES", "0"),
        ("CHUNK_MAX_TOKENS", "63"),
        ("CHUNK_OVERLAP_TOKENS", "-1"),
        ("RETRIEVAL_TOP_K", "0"),
        ("RETRIEVAL_MIN_SCORE", "nan"),
        ("MAX_ANSWER_TOKENS", "0"),
        ("OPENAI_TIMEOUT_SECONDS", "0"),
        ("QDRANT_TIMEOUT_SECONDS", "inf"),
        ("LOG_LEVEL", "verbose"),
    ],
)
def test_invalid_environment_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    invalid_value: str,
) -> None:
    set_required_environment(monkeypatch)
    monkeypatch.setenv(environment_name, invalid_value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_file_bytes": 2, "max_request_bytes": 1},
        {"chunk_max_tokens": 600, "chunk_overlap_tokens": 600},
    ],
)
def test_inconsistent_limits_are_rejected(
    overrides: dict[str, int],
) -> None:
    with pytest.raises(ValidationError):
        build_settings(**overrides)


def test_secrets_are_masked() -> None:
    settings = build_settings(qdrant_api_key="test-qdrant-secret")
    rendered = repr(settings)

    assert "test-openai-secret" not in rendered
    assert "test-qdrant-secret" not in rendered
    assert str(settings.openai_api_key) == "**********"


def test_embedding_hypothesis_is_explicit() -> None:
    settings = build_settings()

    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536


def test_vector_fingerprint_is_stable() -> None:
    first = build_settings()
    second = build_settings()

    assert first.vector_fingerprint == second.vector_fingerprint
    assert first.vector_fingerprint == "be91d8d1633f79b6"
    assert first.vector_name == "text_be91d8d1633f79b6"


@pytest.mark.parametrize(
    "overrides",
    [
        {"openai_embedding_model": "another-embedding-model"},
        {"embedding_dimensions": 3072},
        {"qdrant_distance": "dot"},
        {"chunk_schema_version": "v2"},
    ],
)
def test_vector_fingerprint_changes_with_compatibility_inputs(
    overrides: dict[str, object],
) -> None:
    baseline = build_settings()
    changed = build_settings(**overrides)

    assert changed.vector_fingerprint != baseline.vector_fingerprint


def test_unknown_dotenv_settings_are_rejected(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=test-secret\nOPENAI_CHAT_MODEL=test-chat\nMISSPELLED_SETTING=value\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)


def test_env_example_lists_every_setting_without_credentials() -> None:
    content = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    entries = {
        line.split("=", maxsplit=1)[0]: line.split("=", maxsplit=1)[1]
        for line in content.splitlines()
        if line and not line.startswith("#") and "=" in line
    }

    expected_names = {name.upper() for name in Settings.model_fields}

    assert expected_names <= entries.keys()
    assert entries["OPENAI_API_KEY"] == ""
    assert entries["QDRANT_API_KEY"] == ""
    assert "sk-" not in content.lower()
