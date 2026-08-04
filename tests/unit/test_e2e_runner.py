from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ai_prime_capital_e2e_runner",
    Path(__file__).parents[2] / "scripts" / "e2e.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ContractFailure = _MODULE.ContractFailure
_multipart = _MODULE._multipart
_validate_sources = _MODULE._validate_sources

pytestmark = pytest.mark.unit


def _source(file_name: str = "facts.pdf") -> dict[str, object]:
    return {
        "document_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
        "file_name": file_name,
        "chunk_id": "7b3d2f5a-8d1e-4c62-9a0f-4b3f4e9a4e10",
        "snippet": "Annual revenue increased by 12 percent.",
        "score": 0.91,
        "locator": {"page_number": 1},
    }


def test_multipart_builder_preserves_repeated_file_fields() -> None:
    body, content_type = _multipart(
        (
            ("files", "facts.txt", "text/plain", b"facts"),
            ("files", "notes.txt", "text/plain", b"notes"),
        )
    )

    assert content_type.startswith("multipart/form-data; boundary=")
    assert body.count(b'name="files"') == 2
    assert b'filename="facts.txt"' in body
    assert b'filename="notes.txt"' in body


def test_source_validation_requires_server_owned_locator_and_expected_file() -> None:
    payload = {"answer": "Revenue increased.", "grounded": True, "sources": [_source()]}

    assert _validate_sources(payload, expected_files={"facts.pdf"}) == {"facts.pdf"}

    invalid = {**payload, "sources": [{**_source(), "locator": {}}]}
    with pytest.raises(ContractFailure, match="locator"):
        _validate_sources(invalid, expected_files={"facts.pdf"})


def test_source_validation_enforces_cross_format_evidence() -> None:
    payload = {
        "answer": "Two facts.",
        "grounded": True,
        "sources": [_source("facts.pdf")],
    }

    with pytest.raises(ContractFailure, match="multiple samples"):
        _validate_sources(
            payload,
            expected_files={"facts.pdf", "notes.txt"},
            require_multiple_files=True,
        )
