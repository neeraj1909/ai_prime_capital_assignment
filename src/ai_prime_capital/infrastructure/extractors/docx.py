from __future__ import annotations

import asyncio
from collections.abc import Iterator
from io import BytesIO
from zipfile import BadZipFile

from docx import Document
from docx.document import Document as DocumentType
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from ...config import Settings
from ...domain.models import ExtractedBlock, SourceLocator
from .base import UploadErrorCode, UploadRejected, ensure_extracted_text_limit
from .text import normalize_text


def iter_body_blocks(document: DocumentType) -> Iterator[Paragraph | Table]:
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


class DocxExtractor:
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
            # python-docx/lxml can reject malformed XML after the initial ZIP
            # guard. Keep parser failures inside the upload error contract so a
            # hostile package cannot escape as an opaque API 500.
            raise UploadRejected(
                UploadErrorCode.INVALID_DOCUMENT,
                "DOCX document is invalid",
            ) from error

    def _extract_sync_impl(
        self,
        content: bytes,
        _file_name: str,
    ) -> tuple[ExtractedBlock, ...]:
        try:
            document = Document(BytesIO(content))
        except (BadZipFile, PackageNotFoundError, OSError, ValueError, KeyError) as error:
            raise UploadRejected(
                UploadErrorCode.INVALID_DOCUMENT,
                "DOCX document is invalid",
            ) from error

        blocks: list[ExtractedBlock] = []
        paragraph_index = 0
        table_index = 0

        for item in iter_body_blocks(document):
            if isinstance(item, Paragraph):
                text = normalize_text(item.text)
                locator = SourceLocator(paragraph_index=paragraph_index)
                paragraph_index += 1

                if text:
                    blocks.append(ExtractedBlock(text=text, locator=locator))
                continue

            if isinstance(item, Table):
                rows: list[str] = []

                for row in item.rows:
                    cells: list[str] = []
                    for cell in row.cells:
                        if cell.tables:
                            raise UploadRejected(
                                UploadErrorCode.UNSUPPORTED_NESTED_CONTENT,
                                "nested DOCX tables are unsupported",
                            )

                        text = normalize_text(
                            "\n".join(paragraph.text for paragraph in cell.paragraphs)
                        )
                        if text:
                            cells.append(text)

                    if cells:
                        rows.append(" | ".join(cells))

                table_text = "\n".join(rows)
                locator = SourceLocator(table_index=table_index)
                table_index += 1

                if table_text:
                    blocks.append(ExtractedBlock(text=table_text, locator=locator))

        ensure_extracted_text_limit(
            "".join(block.text for block in blocks),
            self.settings,
        )

        if not blocks:
            raise UploadRejected(
                UploadErrorCode.NO_EXTRACTABLE_TEXT,
                "DOCX contains no extractable text",
            )

        return tuple(blocks)
