from __future__ import annotations

import asyncio
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ...config import Settings
from ...domain.models import ExtractedBlock, SourceLocator
from .base import UploadErrorCode, UploadRejected
from .text import normalize_text


class PdfExtractor:
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
        file_name: str,
    ) -> tuple[ExtractedBlock, ...]:
        try:
            return self._extract_sync_impl(content, file_name)
        except UploadRejected:
            raise
        except Exception as error:
            # Keep parser-specific failures inside the stable upload contract.
            raise UploadRejected(
                UploadErrorCode.INVALID_DOCUMENT,
                "PDF document is invalid",
            ) from error

    def _extract_sync_impl(
        self,
        content: bytes,
        _file_name: str,
    ) -> tuple[ExtractedBlock, ...]:
        try:
            reader = PdfReader(BytesIO(content), strict=False)
        except (PdfReadError, OSError, ValueError) as error:
            raise UploadRejected(
                UploadErrorCode.INVALID_DOCUMENT,
                "PDF document is invalid",
            ) from error

        if reader.is_encrypted:
            raise UploadRejected(
                UploadErrorCode.ENCRYPTED_DOCUMENT,
                "encrypted PDF documents are not supported",
            )

        blocks: list[ExtractedBlock] = []
        extracted_chars = 0

        try:
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = normalize_text(page.extract_text() or "")
                if not page_text:
                    continue

                extracted_chars += len(page_text)
                if extracted_chars > self.settings.max_extracted_chars:
                    raise UploadRejected(
                        UploadErrorCode.EXTRACTED_TEXT_TOO_LARGE,
                        "extracted text exceeds the configured character limit",
                    )

                blocks.append(
                    ExtractedBlock(
                        text=page_text,
                        locator=SourceLocator(page_number=page_number),
                    )
                )
        except UploadRejected:
            raise
        except (PdfReadError, OSError, ValueError) as error:
            raise UploadRejected(
                UploadErrorCode.INVALID_DOCUMENT,
                "PDF page extraction failed",
            ) from error

        if not blocks:
            raise UploadRejected(
                UploadErrorCode.NO_EXTRACTABLE_TEXT,
                "PDF contains no extractable text",
            )

        return tuple(blocks)
