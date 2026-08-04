from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import BinaryIO

from ...config import Settings


class UploadErrorCode(StrEnum):
    NO_FILES = "upload.no_files"
    TOO_MANY_FILES = "upload.too_many_files"
    UNSAFE_FILENAME = "upload.unsafe_filename"
    UNSUPPORTED_TYPE = "upload.unsupported_type"
    MEDIA_TYPE_MISMATCH = "upload.media_type_mismatch"
    EMPTY_FILE = "upload.empty_file"
    FILE_TOO_LARGE = "upload.file_too_large"
    REQUEST_TOO_LARGE = "upload.request_too_large"
    SIGNATURE_MISMATCH = "upload.signature_mismatch"
    INVALID_TEXT = "upload.invalid_text"
    BINARY_TEXT = "upload.binary_text"
    INVALID_PACKAGE = "upload.invalid_package"
    PACKAGE_PATH = "upload.package_path"
    PACKAGE_TOO_MANY_ENTRIES = "upload.package_too_many_entries"
    PACKAGE_TOO_LARGE = "upload.package_too_large"
    PACKAGE_COMPRESSION_RATIO = "upload.package_compression_ratio"
    EXTRACTED_TEXT_TOO_LARGE = "upload.extracted_text_too_large"
    INVALID_STREAM = "upload.invalid_stream"
    INVALID_DOCUMENT = "upload.invalid_document"
    ENCRYPTED_DOCUMENT = "upload.encrypted_document"
    NO_EXTRACTABLE_TEXT = "upload.no_extractable_text"
    UNSUPPORTED_NESTED_CONTENT = "upload.unsupported_nested_content"


class UploadRejected(ValueError):
    def __init__(self, code: UploadErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class IncomingUpload:
    client_name: str
    declared_media_type: str | None
    stream: BinaryIO


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    display_name: str
    internal_name: str
    extension: str
    media_type: str
    content: bytes


def read_bounded(
    stream: BinaryIO,
    *,
    per_file_limit: int,
    request_seen: int,
    request_limit: int,
) -> tuple[bytes, int]:
    chunks: list[bytes] = []
    size = 0
    request_remaining = request_limit - request_seen
    if request_remaining < 0:
        raise UploadRejected(
            UploadErrorCode.REQUEST_TOO_LARGE,
            "request exceeds the configured byte limit",
        )

    read_limit = min(per_file_limit + 1, request_remaining + 1)

    while size < read_limit:
        remaining = read_limit - size
        block = stream.read(min(64 * 1024, remaining))

        if block == b"":
            break

        if not isinstance(block, bytes):
            raise UploadRejected(
                UploadErrorCode.INVALID_STREAM,
                "upload stream did not return bytes",
            )

        chunks.append(block)
        size += len(block)

        if size > per_file_limit:
            raise UploadRejected(
                UploadErrorCode.FILE_TOO_LARGE,
                "file exceeds the configured byte limit",
            )

    total_seen = request_seen + size
    if total_seen > request_limit:
        raise UploadRejected(
            UploadErrorCode.REQUEST_TOO_LARGE,
            "request exceeds the configured byte limit",
        )

    return b"".join(chunks), total_seen


def ensure_extracted_text_limit(text: str, settings: Settings) -> None:
    if len(text) > settings.max_extracted_chars:
        raise UploadRejected(
            UploadErrorCode.EXTRACTED_TEXT_TOO_LARGE,
            "extracted text exceeds the configured character limit",
        )
