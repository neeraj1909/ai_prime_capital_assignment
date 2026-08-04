from hashlib import sha256
from uuid import UUID

import pytest

from ai_prime_capital.domain.identity import chunk_id, document_id

pytestmark = pytest.mark.unit


def test_document_id_uses_the_complete_sha256_digest() -> None:
    content = b"document contents"

    assert document_id(content) == f"doc_{sha256(content).hexdigest()}"


def test_document_id_is_deterministic_and_content_sensitive() -> None:
    content = b"same document"

    assert document_id(content) == document_id(content)
    assert document_id(content) != document_id(b"different document")


def test_document_id_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="content must not be empty"):
        document_id(b"")


def test_chunk_id_is_a_stable_uuid5() -> None:
    first = chunk_id("doc_abc", "v1", 0)
    second = chunk_id("doc_abc", "v1", 0)

    assert first == second
    assert UUID(first).version == 5


def test_chunk_id_changes_when_any_identity_input_changes() -> None:
    identities = {
        chunk_id("doc_abc", "v1", 0),
        chunk_id("doc_def", "v1", 0),
        chunk_id("doc_abc", "v2", 0),
        chunk_id("doc_abc", "v1", 1),
    }

    assert len(identities) == 4


@pytest.mark.parametrize(
    ("document_id_value", "schema_version", "chunk_index"),
    [
        ("", "v1", 0),
        ("   ", "v1", 0),
        ("doc_abc", "", 0),
        ("doc_abc", "   ", 0),
        ("doc_abc", "v1", -1),
    ],
)
def test_chunk_id_rejects_invalid_identity_inputs(
    document_id_value: str,
    schema_version: str,
    chunk_index: int,
) -> None:
    with pytest.raises(ValueError):
        chunk_id(document_id_value, schema_version, chunk_index)
