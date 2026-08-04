from __future__ import annotations

from io import BytesIO
from pathlib import Path
from re import fullmatch
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from ai_prime_capital.config import Settings
from ai_prime_capital.infrastructure.extractors.base import (
    IncomingUpload,
    UploadErrorCode,
    UploadRejected,
)
from ai_prime_capital.infrastructure.extractors.registry import validate_uploads

pytestmark = pytest.mark.unit

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "documents"
DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def build_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "max_files_per_request": 2,
        "max_file_bytes": 1 * 1024 * 1024,
        "max_request_bytes": 2 * 1024 * 1024,
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


def upload(
    name: str,
    media_type: str | None,
    content: bytes,
    *,
    stream: object | None = None,
) -> IncomingUpload:
    return IncomingUpload(
        client_name=name,
        declared_media_type=media_type,
        stream=stream or BytesIO(content),
    )  # type: ignore[arg-type]


def docx_bytes(
    entries: dict[str, bytes],
    *,
    compression: int = ZIP_DEFLATED,
) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def valid_docx_bytes() -> bytes:
    return docx_bytes(
        {
            "[Content_Types].xml": b"<Types />",
            "word/document.xml": b"<document />",
        }
    )


def assert_rejected(
    error: pytest.ExceptionInfo[UploadRejected],
    code: UploadErrorCode,
) -> None:
    assert error.value.code == code


class RecordingStream:
    def __init__(self, content: bytes) -> None:
        self._stream = BytesIO(content)
        self.requested_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.requested_sizes.append(size)
        return self._stream.read(size)


def test_valid_pdf_gets_generated_internal_name() -> None:
    result = validate_uploads(
        (upload("facts.PDF", "application/pdf", b"%PDF-1.7\n"),),
        build_settings(),
    )

    assert result[0].display_name == "facts.PDF"
    assert result[0].extension == ".pdf"
    assert result[0].media_type == "application/pdf"
    assert fullmatch(r"[0-9a-f]{32}\.pdf", result[0].internal_name)
    assert result[0].content == b"%PDF-1.7\n"


def test_valid_docx_is_accepted_without_extracting_the_archive() -> None:
    result = validate_uploads(
        (upload("policy.docx", DOCX_MEDIA_TYPE, valid_docx_bytes()),),
        build_settings(),
    )

    assert result[0].extension == ".docx"
    assert result[0].media_type == DOCX_MEDIA_TYPE


def test_valid_utf8_text_accepts_a_media_type_parameter() -> None:
    result = validate_uploads(
        (upload("notes.txt", "text/plain; charset=utf-8", b"hello\n"),),
        build_settings(),
    )

    assert result[0].content == b"hello\n"


def test_empty_request_is_rejected() -> None:
    with pytest.raises(UploadRejected) as error:
        validate_uploads((), build_settings())

    assert_rejected(error, UploadErrorCode.NO_FILES)


@pytest.mark.parametrize(
    ("name", "media_type", "content", "code"),
    [
        ("facts.exe", "text/plain", b"hello", UploadErrorCode.UNSUPPORTED_TYPE),
        (
            "facts.pdf",
            "text/plain",
            b"%PDF-1.7",
            UploadErrorCode.MEDIA_TYPE_MISMATCH,
        ),
        (
            "facts.pdf",
            "application/pdf",
            b"not-pdf",
            UploadErrorCode.SIGNATURE_MISMATCH,
        ),
        (
            "facts.txt",
            "text/plain",
            b"\xff\xfe",
            UploadErrorCode.INVALID_TEXT,
        ),
        (
            "facts.txt",
            "text/plain",
            b"hello\x00world",
            UploadErrorCode.BINARY_TEXT,
        ),
        ("facts.txt", None, b"hello", UploadErrorCode.MEDIA_TYPE_MISMATCH),
        ("", "text/plain", b"hello", UploadErrorCode.UNSAFE_FILENAME),
        (" facts.txt", "text/plain", b"hello", UploadErrorCode.UNSAFE_FILENAME),
        ("facts.txt ", "text/plain", b"hello", UploadErrorCode.UNSAFE_FILENAME),
        ("../facts.txt", "text/plain", b"hello", UploadErrorCode.UNSAFE_FILENAME),
        (r"..\facts.txt", "text/plain", b"hello", UploadErrorCode.UNSAFE_FILENAME),
        ("/tmp/facts.txt", "text/plain", b"hello", UploadErrorCode.UNSAFE_FILENAME),
        (r"C:\\tmp\\facts.txt", "text/plain", b"hello", UploadErrorCode.UNSAFE_FILENAME),
    ],
)
def test_invalid_uploads_fail_with_stable_codes(
    name: str,
    media_type: str | None,
    content: bytes,
    code: UploadErrorCode,
) -> None:
    with pytest.raises(UploadRejected) as error:
        validate_uploads((upload(name, media_type, content),), build_settings())

    assert_rejected(error, code)


def test_fixture_with_a_pdf_extension_is_rejected_by_signature() -> None:
    content = (FIXTURE_ROOT / "not-a-pdf.pdf").read_bytes()

    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (upload("not-a-pdf.pdf", "application/pdf", content),),
            build_settings(),
        )

    assert_rejected(error, UploadErrorCode.SIGNATURE_MISMATCH)


def test_empty_file_is_rejected() -> None:
    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (upload("empty.txt", "text/plain", b""),),
            build_settings(),
        )

    assert_rejected(error, UploadErrorCode.EMPTY_FILE)


def test_file_limit_is_enforced_at_the_boundary() -> None:
    result = validate_uploads(
        (upload("notes.txt", "text/plain", b"12345"),),
        build_settings(max_file_bytes=5, max_request_bytes=5),
    )

    assert result[0].content == b"12345"

    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (upload("notes.txt", "text/plain", b"123456"),),
            build_settings(max_file_bytes=5, max_request_bytes=10),
        )

    assert_rejected(error, UploadErrorCode.FILE_TOO_LARGE)


def test_file_count_limit_accepts_the_boundary() -> None:
    result = validate_uploads(
        (
            upload("one.txt", "text/plain", b"one"),
            upload("two.txt", "text/plain", b"two"),
        ),
        build_settings(max_files_per_request=2),
    )

    assert len(result) == 2


def test_request_limit_is_enforced_across_files() -> None:
    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (
                upload("one.txt", "text/plain", b"12345"),
                upload("two.txt", "text/plain", b"67890"),
            ),
            build_settings(
                max_files_per_request=2,
                max_file_bytes=9,
                max_request_bytes=9,
            ),
        )

    assert_rejected(error, UploadErrorCode.REQUEST_TOO_LARGE)


def test_request_limit_accepts_the_boundary() -> None:
    result = validate_uploads(
        (
            upload("one.txt", "text/plain", b"12345"),
            upload("two.txt", "text/plain", b"67890"),
        ),
        build_settings(
            max_files_per_request=2,
            max_file_bytes=9,
            max_request_bytes=10,
        ),
    )

    assert sum(len(item.content) for item in result) == 10


def test_file_count_is_checked_before_reading_any_stream() -> None:
    first = RecordingStream(b"one")
    second = RecordingStream(b"two")
    third = RecordingStream(b"three")

    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (
                upload("one.txt", "text/plain", b"one", stream=first),
                upload("two.txt", "text/plain", b"two", stream=second),
                upload("three.txt", "text/plain", b"three", stream=third),
            ),
            build_settings(max_files_per_request=2),
        )

    assert_rejected(error, UploadErrorCode.TOO_MANY_FILES)
    assert first.requested_sizes == []
    assert second.requested_sizes == []
    assert third.requested_sizes == []


def test_file_read_is_bounded_to_one_byte_over_the_limit() -> None:
    stream = RecordingStream(b"x" * 100)

    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (upload("notes.txt", "text/plain", b"", stream=stream),),
            build_settings(max_file_bytes=8, max_request_bytes=16),
        )

    assert_rejected(error, UploadErrorCode.FILE_TOO_LARGE)
    assert max(stream.requested_sizes) <= 9


def test_file_read_is_also_bounded_by_remaining_request_budget() -> None:
    first = RecordingStream(b"12345")
    second = RecordingStream(b"x" * 100)

    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (
                upload("one.txt", "text/plain", b"", stream=first),
                upload("two.txt", "text/plain", b"", stream=second),
            ),
            build_settings(
                max_files_per_request=2,
                max_file_bytes=9,
                max_request_bytes=9,
            ),
        )

    assert_rejected(error, UploadErrorCode.REQUEST_TOO_LARGE)
    assert max(second.requested_sizes) <= 5


def test_extracted_text_limit_is_enforced_for_utf8_text() -> None:
    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (upload("notes.txt", "text/plain", b"12345"),),
            build_settings(max_extracted_chars=4),
        )

    assert_rejected(error, UploadErrorCode.EXTRACTED_TEXT_TOO_LARGE)


def test_extracted_text_limit_accepts_the_boundary() -> None:
    result = validate_uploads(
        (upload("notes.txt", "text/plain", b"12345"),),
        build_settings(max_extracted_chars=5),
    )

    assert result[0].content == b"12345"


def test_invalid_docx_fixture_is_rejected() -> None:
    content = (FIXTURE_ROOT / "not-a-docx.docx").read_bytes()

    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (upload("not-a-docx.docx", DOCX_MEDIA_TYPE, content),),
            build_settings(),
        )

    assert_rejected(error, UploadErrorCode.SIGNATURE_MISMATCH)


def test_docx_missing_required_parts_is_rejected() -> None:
    content = docx_bytes({"word/document.xml": b"<document />"})

    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (upload("policy.docx", DOCX_MEDIA_TYPE, content),),
            build_settings(),
        )

    assert_rejected(error, UploadErrorCode.INVALID_PACKAGE)


def test_docx_archive_path_traversal_is_rejected() -> None:
    content = docx_bytes(
        {
            "[Content_Types].xml": b"<Types />",
            "word/document.xml": b"<document />",
            "../escape.txt": b"bad",
        }
    )

    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (upload("policy.docx", DOCX_MEDIA_TYPE, content),),
            build_settings(),
        )

    assert_rejected(error, UploadErrorCode.PACKAGE_PATH)


def test_docx_entry_count_is_bounded() -> None:
    content = docx_bytes(
        {
            "[Content_Types].xml": b"<Types />",
            "word/document.xml": b"<document />",
            "word/extra.xml": b"extra",
        }
    )

    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (upload("policy.docx", DOCX_MEDIA_TYPE, content),),
            build_settings(docx_max_entries=2),
        )

    assert_rejected(error, UploadErrorCode.PACKAGE_TOO_MANY_ENTRIES)


def test_docx_entry_count_accepts_the_boundary() -> None:
    result = validate_uploads(
        (upload("policy.docx", DOCX_MEDIA_TYPE, valid_docx_bytes()),),
        build_settings(docx_max_entries=2),
    )

    assert result[0].extension == ".docx"


def test_docx_expanded_size_is_bounded() -> None:
    content = docx_bytes(
        {
            "[Content_Types].xml": b"<Types />",
            "word/document.xml": b"x" * 100,
        }
    )

    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (upload("policy.docx", DOCX_MEDIA_TYPE, content),),
            build_settings(docx_max_expanded_bytes=50),
        )

    assert_rejected(error, UploadErrorCode.PACKAGE_TOO_LARGE)


def test_docx_expanded_size_accepts_the_boundary() -> None:
    content = valid_docx_bytes()
    with ZipFile(BytesIO(content)) as archive:
        expanded_size = sum(entry.file_size for entry in archive.infolist())

    result = validate_uploads(
        (upload("policy.docx", DOCX_MEDIA_TYPE, content),),
        build_settings(docx_max_expanded_bytes=expanded_size),
    )

    assert result[0].content == content


def test_docx_compression_ratio_is_bounded() -> None:
    content = docx_bytes(
        {
            "[Content_Types].xml": b"<Types />",
            "word/document.xml": b"A" * 10_000,
        }
    )

    with pytest.raises(UploadRejected) as error:
        validate_uploads(
            (upload("policy.docx", DOCX_MEDIA_TYPE, content),),
            build_settings(
                max_file_bytes=1 * 1024 * 1024,
                max_request_bytes=2 * 1024 * 1024,
                docx_max_expanded_bytes=20_000,
                docx_max_compression_ratio=10.0,
            ),
        )

    assert_rejected(error, UploadErrorCode.PACKAGE_COMPRESSION_RATIO)


def test_docx_compression_ratio_accepts_the_boundary() -> None:
    content = valid_docx_bytes()
    with ZipFile(BytesIO(content)) as archive:
        ratios = [
            entry.file_size / entry.compress_size
            for entry in archive.infolist()
            if entry.file_size and entry.compress_size
        ]

    result = validate_uploads(
        (upload("policy.docx", DOCX_MEDIA_TYPE, content),),
        build_settings(docx_max_compression_ratio=max(ratios)),
    )

    assert result[0].content == content
