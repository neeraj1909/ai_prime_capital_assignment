from __future__ import annotations

from collections.abc import Sequence

import tiktoken

from .identity import chunk_id
from .models import Chunk, ExtractedBlock

_SEPARATORS = ("\n\n", "\n", ". ", " ")


def _encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _split_preserving_separator(text: str, separator: str) -> tuple[str, ...]:
    parts = text.split(separator)
    result: list[str] = []

    for index, part in enumerate(parts):
        if index < len(parts) - 1:
            result.append(part + separator)
        elif part:
            result.append(part)

    return tuple(result)


def _recursive_split(
    text: str,
    encoder: tiktoken.Encoding,
    max_tokens: int,
) -> tuple[str, ...]:
    tokens = encoder.encode(text, disallowed_special=())

    if len(tokens) <= max_tokens:
        return (text,)

    for separator in _SEPARATORS:
        pieces = _split_preserving_separator(text, separator)
        if len(pieces) <= 1:
            continue

        return tuple(
            part for piece in pieces for part in _recursive_split(piece, encoder, max_tokens)
        )

    return tuple(
        encoder.decode(tokens[start : start + max_tokens])
        for start in range(0, len(tokens), max_tokens)
    )


def _pack_with_overlap(
    units: Sequence[str],
    encoder: tiktoken.Encoding,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[str, ...]:
    chunks: list[str] = []
    current: list[int] = []

    for unit in units:
        unit_tokens = encoder.encode(unit, disallowed_special=())

        if len(unit_tokens) > max_tokens:
            raise ValueError("recursive splitter emitted an oversized unit")

        if current and len(current) + len(unit_tokens) > max_tokens:
            chunks.append(encoder.decode(current))

            overlap = min(overlap_tokens, len(current))
            current = current[-overlap:] if overlap else []

            if len(current) + len(unit_tokens) > max_tokens:
                current = []

        current.extend(unit_tokens)

    if current:
        chunks.append(encoder.decode(current))

    return tuple(chunk for chunk in chunks if chunk.strip())


def chunk_blocks(
    blocks: Sequence[ExtractedBlock],
    *,
    document_id_value: str,
    file_name: str,
    embedding_model: str,
    schema_version: str,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[Chunk, ...]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be less than max_tokens")

    encoder = _encoding(embedding_model)
    result: list[Chunk] = []

    # Keep each ExtractedBlock separate so its locator remains truthful.
    for block in blocks:
        units = _recursive_split(block.text, encoder, max_tokens)
        texts = _pack_with_overlap(
            units,
            encoder,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        for text in texts:
            result.append(
                Chunk(
                    chunk_id=chunk_id(
                        document_id_value,
                        schema_version,
                        len(result),
                    ),
                    document_id=document_id_value,
                    file_name=file_name,
                    text=text,
                    locator=block.locator,
                    char_count=len(text),
                    token_count=len(encoder.encode(text, disallowed_special=())),
                )
            )

    return tuple(result)
