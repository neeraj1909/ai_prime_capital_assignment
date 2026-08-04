from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from ai_prime_capital.config import Settings
from ai_prime_capital.infrastructure.extractors.base import IncomingUpload
from ai_prime_capital.infrastructure.extractors.registry import (
    ExtractorRegistry,
    validate_uploads,
)

pytestmark = pytest.mark.evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_ROOT = PROJECT_ROOT / "samples"
MANIFEST_PATH = SAMPLES_ROOT / "manifest.json"


def load_manifest() -> dict[str, Any]:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_sample_manifest_hashes_and_labels_are_reproducible() -> None:
    manifest = load_manifest()
    documents = manifest["documents"]

    assert manifest["version"] == 1
    assert {document["file"] for document in documents} == {
        "facts.pdf",
        "policy.docx",
        "notes.txt",
    }
    for document in documents:
        path = SAMPLES_ROOT / document["file"]
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == document["sha256"]
        assert document["media_type"]
        assert document["facts"]
        assert document["expected_locator"]

    assert manifest["absent_facts"] == ["moonshot office", "Antarctica headcount"]


@pytest.mark.asyncio
async def test_sample_documents_extract_facts_and_provenance() -> None:
    manifest = load_manifest()
    settings = Settings(
        _env_file=None,
        openai_api_key="test-openai-secret",
        openai_chat_model="test-chat-model",
    )
    registry = ExtractorRegistry(settings)
    extracted_text: list[str] = []

    for document in manifest["documents"]:
        path = SAMPLES_ROOT / document["file"]
        content = path.read_bytes()
        validated = validate_uploads(
            (
                IncomingUpload(
                    client_name=path.name,
                    declared_media_type=document["media_type"],
                    stream=BytesIO(content),
                ),
            ),
            settings,
        )
        blocks = await registry.extract(content, file_name=path.name)
        document_text = "\n".join(block.text for block in blocks)
        extracted_text.append(document_text)

        assert blocks
        assert all(fact in document_text for fact in document["facts"])
        locator = blocks[0].locator
        for field, expected in document["expected_locator"].items():
            assert getattr(locator, field) == expected
        assert validated[0].display_name == path.name

    combined = "\n".join(extracted_text).lower()
    assert all(absent.lower() not in combined for absent in manifest["absent_facts"])
