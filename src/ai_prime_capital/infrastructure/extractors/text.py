from __future__ import annotations

import asyncio
import re
import unicodedata

from ...config import Settings
from ...domain.models import ExtractedBlock, SourceLocator
from .base import UploadErrorCode, UploadRejected, ensure_extracted_text_limit


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    return value.replace("\xa0", " ").strip()


class TextExtractor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def extract(
        self,
        content: bytes,
        *,
        file_name: str,
    ) -> tuple[ExtractedBlock, ...]:
        return await asyncio.to_thread(self._extract_sync, content, file_name)

    def _extract_sync(
        self,
        content: bytes,
        _file_name: str,
    ) -> tuple[ExtractedBlock, ...]:
        try:
            raw = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise UploadRejected(
                UploadErrorCode.INVALID_TEXT,
                "text file is not valid UTF-8",
            ) from error

        ensure_extracted_text_limit(raw, self.settings)

        blocks: list[ExtractedBlock] = []
        extracted_chars = 0
        for match in re.finditer(r"[^\r\n]+", raw):
            text = normalize_text(match.group())
            if not text:
                continue

            extracted_chars += len(text)
            if extracted_chars > self.settings.max_extracted_chars:
                raise UploadRejected(
                    UploadErrorCode.EXTRACTED_TEXT_TOO_LARGE,
                    "extracted text exceeds the configured character limit",
                )

            blocks.append(
                ExtractedBlock(
                    text=text,
                    locator=SourceLocator(
                        start_offset=match.start(),
                        end_offset=match.end(),
                    ),
                )
            )

        if not blocks:
            raise UploadRejected(
                UploadErrorCode.NO_EXTRACTABLE_TEXT,
                "text file contains no extractable text",
            )

        return tuple(blocks)
