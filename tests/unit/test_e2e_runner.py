from __future__ import annotations

import importlib.util
import json
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
ContractRunner = _MODULE.ContractRunner
Response = _MODULE.Response
_multipart = _MODULE._multipart
_require_document_id = _MODULE._require_document_id
_safe_problem_summary = _MODULE._safe_problem_summary
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


def test_source_validation_rejects_cross_wired_file_identity() -> None:
    payload = {
        "answer": "Revenue increased by 12 percent.",
        "grounded": True,
        "sources": [_source()],
    }

    with pytest.raises(ContractFailure, match="persisted document identities"):
        _validate_sources(
            payload,
            expected_files={"facts.pdf"},
            expected_documents={"facts.pdf": "doc_saved_identity"},
        )


def test_source_validation_requires_expected_answer_concepts() -> None:
    payload = {
        "answer": "Revenue increased.",
        "grounded": True,
        "sources": [_source()],
    }

    with pytest.raises(ContractFailure, match="expected fact"):
        _validate_sources(
            payload,
            expected_files={"facts.pdf"},
            expected_answer_concepts=(("12 percent", "12%"),),
        )


def test_source_validation_rejects_answer_fact_missing_from_evidence() -> None:
    source = {**_source(), "snippet": "Revenue information is unavailable."}
    payload = {
        "answer": "Revenue increased by 12 percent in 2024.",
        "grounded": True,
        "sources": [source],
    }

    with pytest.raises(ContractFailure, match="source evidence"):
        _validate_sources(
            payload,
            expected_files={"facts.pdf"},
            expected_documents={"facts.pdf": source["document_id"]},
            expected_evidence_concepts={
                "facts.pdf": (("12 percent", "12%"), ("2024",)),
            },
        )


def test_source_validation_accepts_expected_facts_and_persisted_identity() -> None:
    source = _source()
    source["snippet"] = "Annual revenue increased by 12 percent during 2024."
    payload = {
        "answer": "Revenue increased by 12% during 2024.",
        "grounded": True,
        "sources": [source],
    }

    assert _validate_sources(
        payload,
        expected_files={"facts.pdf"},
        expected_documents={"facts.pdf": source["document_id"]},
        expected_answer_concepts=(("12 percent", "12%"), ("2024",)),
        expected_evidence_concepts={
            "facts.pdf": (("12 percent", "12%"), ("2024",)),
        },
    ) == {"facts.pdf"}


def test_upload_validation_rejects_duplicate_files_and_invalid_chunk_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ContractRunner(base_url="http://test", timeout=1, state_file=tmp_path / "state")
    duplicate = {
        "documents": [
            {
                "document_id": f"doc-{index}",
                "file_name": "facts.pdf",
                "media_type": "application/pdf",
                "status": "indexed",
                "chunk_count": True,
            }
            for index in range(3)
        ]
    }
    monkeypatch.setattr(
        runner,
        "request",
        lambda *_args, **_kwargs: Response(201, json.dumps(duplicate).encode()),
    )
    monkeypatch.setattr(runner, "_sample_parts", lambda: ())

    with pytest.raises(ContractFailure, match=r"invalid index result|duplicate"):
        runner.upload_samples(label="upload.test")


def test_safe_problem_summary_reports_code_and_request_id_only() -> None:
    response = Response(
        503,
        (
            b'{"status":503,"title":"Dependency unavailable",'
            b'"detail":"secret prompt content",'
            b'"code":"embedding.upstream","request_id":"req-123"}'
        ),
    )

    summary = _safe_problem_summary(response)

    assert summary == "code=embedding.upstream, request_id=req-123"
    assert "secret prompt content" not in summary


def test_safe_problem_summary_ignores_malformed_or_unsafe_payloads() -> None:
    malformed = Response(503, b"provider failure")
    unsafe = Response(
        503,
        b'{"code":"embedding.upstream\nleak", "request_id":"req-123"}',
    )

    assert _safe_problem_summary(malformed) == ""
    assert _safe_problem_summary(unsafe) == ""


def test_document_identity_validation_accepts_content_addressed_ids() -> None:
    value = "doc_" + "a" * 64

    assert _require_document_id(value, "document_id") == value


@pytest.mark.parametrize("value", ["", "   ", None, 42])
def test_document_identity_validation_rejects_invalid_ids(value: object) -> None:
    with pytest.raises(ContractFailure, match="document_id"):
        _require_document_id(value, "document_id")
