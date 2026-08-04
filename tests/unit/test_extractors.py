from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from docx import Document
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import ai_prime_capital.infrastructure.extractors.pdf as pdf_module
from ai_prime_capital.config import Settings
from ai_prime_capital.domain.models import ExtractedBlock
from ai_prime_capital.infrastructure.extractors.base import (
    UploadErrorCode,
    UploadRejected,
)
from ai_prime_capital.infrastructure.extractors.docx import DocxExtractor
from ai_prime_capital.infrastructure.extractors.pdf import PdfExtractor
from ai_prime_capital.infrastructure.extractors.registry import ExtractionRegistry
from ai_prime_capital.infrastructure.extractors.text import TextExtractor

pytestmark = pytest.mark.unit

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "documents"


def build_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "max_extracted_chars": 5_000_000,
        "docx_max_entries": 1_000,
        "docx_max_expanded_bytes": 50 * 1024 * 1024,
        "docx_max_compression_ratio": 100.0,
    }
    values.update(overrides)
    return Settings(
        _env_file=None,
        openai_api_key="test-secret",
        openai_chat_model="test-model",
        **values,
    )


def pdf_bytes(pages: tuple[str | None, ...]) -> bytes:
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        if text is not None:
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/F1"): writer._add_object(font)}
                    )
                }
            )
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode())
            page[NameObject("/Contents")] = writer._add_object(stream)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def encrypted_pdf_bytes() -> bytes:
    reader = PdfReader(BytesIO(pdf_bytes(("secret text",))))
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.encrypt("test-password")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def docx_bytes(*, nested_table: bool = False, textless: bool = False) -> bytes:
    document = Document()
    if not textless:
        document.add_paragraph("  First section\u00a0fact  ")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Revenue"
        table.cell(0, 1).text = "$10M"
        table.cell(1, 0).text = "Risk"
        table.cell(1, 1).text = "Low"
        if nested_table:
            nested = table.cell(0, 0).add_table(rows=1, cols=1)
            nested.cell(0, 0).text = "Nested"
        document.add_paragraph("Final conclusion")

    output = BytesIO()
    document.save(output)
    return output.getvalue()


def malformed_docx_xml_bytes() -> bytes:
    source = BytesIO(docx_bytes())
    output = BytesIO()
    with ZipFile(source) as archive, ZipFile(output, "w", compression=ZIP_DEFLATED) as rebuilt:
        for info in archive.infolist():
            content = (
                b"<w:document><w:body"
                if info.filename == "word/document.xml"
                else archive.read(info)
            )
            rebuilt.writestr(info, content)
    return output.getvalue()


def assert_rejected(error: pytest.ExceptionInfo[UploadRejected], code: UploadErrorCode) -> None:
    assert error.value.code == code


async def extract_text(content: bytes, **settings: object) -> tuple[ExtractedBlock, ...]:
    return await TextExtractor(build_settings(**settings)).extract(
        content,
        file_name="facts.txt",
    )


async def extract_pdf(content: bytes, **settings: object) -> tuple[ExtractedBlock, ...]:
    return await PdfExtractor(build_settings(**settings)).extract(
        content,
        file_name="facts.pdf",
    )


async def extract_docx(content: bytes, **settings: object) -> tuple[ExtractedBlock, ...]:
    return await DocxExtractor(build_settings(**settings)).extract(
        content,
        file_name="facts.docx",
    )


@pytest.mark.asyncio
async def test_text_extractor_preserves_order_and_character_offsets() -> None:
    raw = "\n  First\u00a0fact  \n\nSecond fact\n"

    blocks = await extract_text(raw.encode("utf-8"))

    assert [block.text for block in blocks] == ["First fact", "Second fact"]
    assert blocks[0].locator.start_offset == raw.index("  First")
    assert blocks[0].locator.end_offset == raw.index("  First") + len("  First\u00a0fact  ")
    assert blocks[1].locator.start_offset == raw.index("Second")
    assert blocks[1].locator.end_offset == raw.index("Second") + len("Second fact")


@pytest.mark.asyncio
async def test_text_extractor_accepts_utf8_bom() -> None:
    blocks = await extract_text("\ufeffalpha\nbeta".encode("utf-8"))

    assert [block.text for block in blocks] == ["alpha", "beta"]
    assert blocks[0].locator.start_offset == 0
    assert blocks[1].locator.start_offset == 6


@pytest.mark.asyncio
async def test_text_extractor_rejects_unsupported_encoding() -> None:
    with pytest.raises(UploadRejected) as error:
        await extract_text(b"valid\xfftext")

    assert_rejected(error, UploadErrorCode.INVALID_TEXT)


@pytest.mark.asyncio
async def test_text_extractor_rejects_empty_normalized_document() -> None:
    with pytest.raises(UploadRejected) as error:
        await extract_text(b" \t\n\r\n")

    assert_rejected(error, UploadErrorCode.NO_EXTRACTABLE_TEXT)


@pytest.mark.asyncio
async def test_text_extractor_enforces_extracted_character_limit() -> None:
    with pytest.raises(UploadRejected) as error:
        await extract_text(b"hello", max_extracted_chars=4)

    assert_rejected(error, UploadErrorCode.EXTRACTED_TEXT_TOO_LARGE)


@pytest.mark.asyncio
async def test_pdf_extractor_emits_one_block_per_text_page() -> None:
    blocks = await extract_pdf(pdf_bytes(("Page one fact", None, "Page three fact")))

    assert [block.text for block in blocks] == ["Page one fact", "Page three fact"]
    assert [block.locator.page_number for block in blocks] == [1, 3]


@pytest.mark.asyncio
async def test_pdf_extractor_rejects_malformed_document() -> None:
    with pytest.raises(UploadRejected) as error:
        await extract_pdf(b"%PDF-1.7\nnot a valid PDF")

    assert_rejected(error, UploadErrorCode.INVALID_DOCUMENT)


@pytest.mark.asyncio
async def test_pdf_extractor_maps_unexpected_parser_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_parser_error(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("malformed parser state")

    monkeypatch.setattr(pdf_module, "PdfReader", raise_parser_error)

    with pytest.raises(UploadRejected) as error:
        await extract_pdf(b"%PDF-1.7\ncontent")

    assert_rejected(error, UploadErrorCode.INVALID_DOCUMENT)


@pytest.mark.asyncio
async def test_pdf_extractor_rejects_encrypted_document() -> None:
    with pytest.raises(UploadRejected) as error:
        await extract_pdf(encrypted_pdf_bytes())

    assert_rejected(error, UploadErrorCode.ENCRYPTED_DOCUMENT)


@pytest.mark.asyncio
async def test_pdf_extractor_rejects_textless_document() -> None:
    with pytest.raises(UploadRejected) as error:
        await extract_pdf(pdf_bytes((None, None)))

    assert_rejected(error, UploadErrorCode.NO_EXTRACTABLE_TEXT)


@pytest.mark.asyncio
async def test_pdf_extractor_enforces_extracted_character_limit() -> None:
    with pytest.raises(UploadRejected) as error:
        await extract_pdf(pdf_bytes(("12345",)), max_extracted_chars=4)

    assert_rejected(error, UploadErrorCode.EXTRACTED_TEXT_TOO_LARGE)


@pytest.mark.asyncio
async def test_docx_extractor_preserves_paragraph_and_table_order() -> None:
    blocks = await extract_docx(docx_bytes())

    assert [block.text for block in blocks] == [
        "First section fact",
        "Revenue | $10M\nRisk | Low",
        "Final conclusion",
    ]
    assert blocks[0].locator.paragraph_index == 0
    assert blocks[1].locator.table_index == 0
    assert blocks[2].locator.paragraph_index == 1


@pytest.mark.asyncio
async def test_docx_extractor_rejects_malformed_document() -> None:
    with pytest.raises(UploadRejected) as error:
        await extract_docx(b"not a DOCX package")

    assert_rejected(error, UploadErrorCode.INVALID_DOCUMENT)


@pytest.mark.asyncio
async def test_docx_extractor_rejects_malformed_xml_after_zip_validation() -> None:
    with pytest.raises(UploadRejected) as error:
        await extract_docx(malformed_docx_xml_bytes())

    assert_rejected(error, UploadErrorCode.INVALID_DOCUMENT)


@pytest.mark.asyncio
async def test_docx_extractor_rejects_textless_document() -> None:
    with pytest.raises(UploadRejected) as error:
        await extract_docx(docx_bytes(textless=True))

    assert_rejected(error, UploadErrorCode.NO_EXTRACTABLE_TEXT)


@pytest.mark.asyncio
async def test_docx_extractor_rejects_nested_tables_explicitly() -> None:
    with pytest.raises(UploadRejected) as error:
        await extract_docx(docx_bytes(nested_table=True))

    assert_rejected(error, UploadErrorCode.UNSUPPORTED_NESTED_CONTENT)


@pytest.mark.asyncio
async def test_docx_extractor_enforces_extracted_character_limit() -> None:
    with pytest.raises(UploadRejected) as error:
        await extract_docx(docx_bytes(), max_extracted_chars=5)

    assert_rejected(error, UploadErrorCode.EXTRACTED_TEXT_TOO_LARGE)


@pytest.mark.asyncio
async def test_registry_dispatches_by_extension_and_rejects_unknown_types() -> None:
    registry = ExtractionRegistry(build_settings())

    blocks = await registry.extract(b"registry text", file_name="notes.TXT")

    assert [block.text for block in blocks] == ["registry text"]

    with pytest.raises(UploadRejected) as error:
        await registry.extract(b"content", file_name="notes.csv")

    assert_rejected(error, UploadErrorCode.UNSUPPORTED_TYPE)


def test_text_fixture_is_available_for_review() -> None:
    fixture = FIXTURE_ROOT / "facts.txt"

    assert fixture.read_text(encoding="utf-8").strip() == "Capital is a fact."
