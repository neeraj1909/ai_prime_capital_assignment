from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import pytest
import tiktoken

from ai_prime_capital.domain.chunking import chunk_blocks
from ai_prime_capital.domain.identity import chunk_id
from ai_prime_capital.domain.models import ExtractedBlock, SourceLocator

pytestmark = pytest.mark.unit

MODEL = "text-embedding-3-small"
ENCODER = tiktoken.encoding_for_model(MODEL)


def block(text: str, locator: SourceLocator | None = None) -> ExtractedBlock:
    return ExtractedBlock(text=text, locator=locator or SourceLocator())


def chunks(
    blocks: Sequence[ExtractedBlock],
    *,
    max_tokens: int = 12,
    overlap_tokens: int = 3,
    schema_version: str = "v1",
):
    return chunk_blocks(
        blocks,
        document_id_value="doc_test",
        file_name="facts.txt",
        embedding_model=MODEL,
        schema_version=schema_version,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def token_count(text: str) -> int:
    return len(ENCODER.encode(text, disallowed_special=()))


def suffix_prefix_overlap(left: str, right: str) -> int:
    left_tokens = ENCODER.encode(left, disallowed_special=())
    right_tokens = ENCODER.encode(right, disallowed_special=())
    return max(
        (
            size
            for size in range(1, min(len(left_tokens), len(right_tokens)) + 1)
            if left_tokens[-size:] == right_tokens[:size]
        ),
        default=0,
    )


def test_short_block_preserves_text_locator_and_counts() -> None:
    locator = SourceLocator(page_number=2, section="Summary")

    result = chunks((block("Revenue is stable.", locator),), max_tokens=32)

    assert len(result) == 1
    assert result[0].text == "Revenue is stable."
    assert result[0].locator == locator
    assert result[0].char_count == len(result[0].text)
    assert result[0].token_count == token_count(result[0].text)
    assert result[0].chunk_id == chunk_id("doc_test", "v1", 0)


def test_exact_token_limit_is_not_split() -> None:
    text = " ".join(f"word-{index}" for index in range(20))
    limit = token_count(text)

    result = chunks((block(text),), max_tokens=limit, overlap_tokens=0)

    assert len(result) == 1
    assert result[0].token_count == limit


def test_long_text_falls_back_to_token_bounded_chunks() -> None:
    text = " ".join(f"word-{index}" for index in range(120))

    result = chunks((block(text),))

    assert len(result) > 1
    assert all(0 < item.token_count <= 12 for item in result)
    assert result[0].text.startswith("word-0")
    assert "word-119" in result[-1].text


def test_overlap_never_exceeds_configured_bound() -> None:
    text = " ".join(f"word-{index}" for index in range(120))

    result = chunks((block(text),), max_tokens=12, overlap_tokens=3)

    assert all(
        suffix_prefix_overlap(left.text, right.text) <= 3 for left, right in pairwise(result)
    )


def test_source_order_and_locators_survive_multiple_blocks() -> None:
    blocks = (
        block("Page one fact", SourceLocator(page_number=1)),
        block("Table fact", SourceLocator(table_index=0)),
        block("Page two fact", SourceLocator(page_number=2)),
    )

    result = chunks(blocks, max_tokens=32)

    assert [item.text for item in result] == [item.text for item in blocks]
    assert [item.locator for item in result] == [item.locator for item in blocks]
    assert [item.chunk_id for item in result] == [
        chunk_id("doc_test", "v1", index) for index in range(3)
    ]


def test_unicode_text_is_bounded_and_ordered() -> None:
    text = "भारत में निवेश सुरक्षित है. Café growth remains strong."

    result = chunks((block(text),), max_tokens=8, overlap_tokens=2)

    assert all(0 < item.token_count <= 8 for item in result)
    assert "भारत" in "".join(item.text for item in result)
    assert "strong" in "".join(item.text for item in result)


def test_repeated_input_is_deterministic_and_schema_version_changes_ids() -> None:
    source = (block("A stable source with enough words to split."),)

    first = chunks(source)
    second = chunks(source)
    changed_schema = chunks(source, schema_version="v2")

    assert first == second
    assert [item.chunk_id for item in first] != [item.chunk_id for item in changed_schema]


def test_empty_input_produces_no_chunks() -> None:
    assert chunks(()) == ()


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens"),
    [(0, 0), (12, -1), (12, 12)],
)
def test_invalid_chunk_limits_are_rejected(max_tokens: int, overlap_tokens: int) -> None:
    with pytest.raises(ValueError):
        chunks((block("text"),), max_tokens=max_tokens, overlap_tokens=overlap_tokens)
