from __future__ import annotations

import stat
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath, PureWindowsPath
from typing import NoReturn
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from ...application.ports import ExtractorPort
from ...config import Settings
from ...domain.models import ExtractedBlock
from .base import (
    IncomingUpload,
    UploadErrorCode,
    UploadRejected,
    ValidatedUpload,
    ensure_extracted_text_limit,
    read_bounded,
)
from .docx import DocxExtractor
from .pdf import PdfExtractor
from .text import TextExtractor


@dataclass(frozen=True, slots=True)
class FormatSpec:
    extension: str
    media_type: str


FORMATS = {
    ".pdf": FormatSpec(".pdf", "application/pdf"),
    ".docx": FormatSpec(
        ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    ".txt": FormatSpec(".txt", "text/plain"),
}


def _reject(code: UploadErrorCode, detail: str) -> NoReturn:
    raise UploadRejected(code, detail)


def _format_for_name(name: str) -> FormatSpec:
    if not name or name != name.strip():
        _reject(UploadErrorCode.UNSAFE_FILENAME, "filename is blank or padded")

    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        _reject(UploadErrorCode.UNSAFE_FILENAME, "filename contains control characters")

    normalized = name.replace("\\", "/")
    posix_name = PurePosixPath(normalized)
    windows_name = PureWindowsPath(name)

    if (
        "/" in name
        or "\\" in name
        or posix_name.is_absolute()
        or windows_name.is_absolute()
        or bool(windows_name.drive)
        or ".." in posix_name.parts
    ):
        _reject(UploadErrorCode.UNSAFE_FILENAME, "path components are not allowed")

    extension = posix_name.suffix.lower()
    try:
        return FORMATS[extension]
    except KeyError:
        _reject(UploadErrorCode.UNSUPPORTED_TYPE, "file extension is not supported")


def _validate_media_type(upload: IncomingUpload, spec: FormatSpec) -> str:
    if upload.declared_media_type is None:
        _reject(UploadErrorCode.MEDIA_TYPE_MISMATCH, "declared media type is required")

    media_type = upload.declared_media_type.partition(";")[0].strip().lower()
    if media_type != spec.media_type:
        _reject(UploadErrorCode.MEDIA_TYPE_MISMATCH, "declared media type is invalid")

    return media_type


def _validate_text(content: bytes, settings: Settings) -> None:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise UploadRejected(
            UploadErrorCode.INVALID_TEXT,
            "text file is not valid UTF-8",
        ) from error

    if any((ord(char) < 32 and char not in "\t\n\r\f") or ord(char) == 127 for char in text):
        _reject(UploadErrorCode.BINARY_TEXT, "text file contains binary control data")

    ensure_extracted_text_limit(text, settings)


def _validate_archive_path(name: str) -> None:
    normalized = name.replace("\\", "/")
    posix_name = PurePosixPath(normalized)
    windows_name = PureWindowsPath(name)

    if (
        not name
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
        or posix_name.is_absolute()
        or windows_name.is_absolute()
        or bool(windows_name.drive)
        or ".." in posix_name.parts
    ):
        _reject(UploadErrorCode.PACKAGE_PATH, "DOCX contains an unsafe archive path")


def _validate_docx_package(content: bytes, settings: Settings) -> None:
    if not content.startswith(b"PK"):
        _reject(UploadErrorCode.SIGNATURE_MISMATCH, "DOCX is not a ZIP package")

    try:
        with ZipFile(BytesIO(content)) as archive:
            entries = archive.infolist()
    except (BadZipFile, OSError, ValueError) as error:
        raise UploadRejected(
            UploadErrorCode.INVALID_PACKAGE,
            "DOCX package is invalid",
        ) from error

    if len(entries) > settings.docx_max_entries:
        _reject(
            UploadErrorCode.PACKAGE_TOO_MANY_ENTRIES,
            "DOCX contains too many entries",
        )

    expanded_size = 0
    names: set[str] = set()

    for entry in entries:
        _validate_archive_path(entry.filename)

        mode = (entry.external_attr >> 16) & 0o170000
        if stat.S_ISLNK(mode):
            _reject(UploadErrorCode.PACKAGE_PATH, "DOCX contains a symbolic link")

        expanded_size += entry.file_size
        if expanded_size > settings.docx_max_expanded_bytes:
            _reject(
                UploadErrorCode.PACKAGE_TOO_LARGE,
                "DOCX expands beyond the configured limit",
            )

        if entry.file_size:
            ratio = (
                float("inf") if entry.compress_size == 0 else entry.file_size / entry.compress_size
            )
            if ratio > settings.docx_max_compression_ratio:
                _reject(
                    UploadErrorCode.PACKAGE_COMPRESSION_RATIO,
                    "DOCX compression ratio is unsafe",
                )

        names.add(entry.filename.replace("\\", "/"))

    if not {"[Content_Types].xml", "word/document.xml"}.issubset(names):
        _reject(
            UploadErrorCode.INVALID_PACKAGE,
            "DOCX package is missing required OOXML parts",
        )


def _validate_signature(
    content: bytes,
    spec: FormatSpec,
    settings: Settings,
) -> None:
    if not content:
        _reject(UploadErrorCode.EMPTY_FILE, "file is empty")

    if spec.extension == ".pdf" and not content.startswith(b"%PDF-"):
        _reject(UploadErrorCode.SIGNATURE_MISMATCH, "PDF signature is invalid")
    elif spec.extension == ".docx":
        _validate_docx_package(content, settings)
    elif spec.extension == ".txt":
        _validate_text(content, settings)


def validate_uploads(
    uploads: Sequence[IncomingUpload],
    settings: Settings,
) -> tuple[ValidatedUpload, ...]:
    if not uploads:
        _reject(UploadErrorCode.NO_FILES, "at least one file is required")

    if len(uploads) > settings.max_files_per_request:
        _reject(UploadErrorCode.TOO_MANY_FILES, "request contains too many files")

    validated: list[ValidatedUpload] = []
    request_seen = 0

    for upload in uploads:
        spec = _format_for_name(upload.client_name)
        media_type = _validate_media_type(upload, spec)
        content, request_seen = read_bounded(
            upload.stream,
            per_file_limit=settings.max_file_bytes,
            request_seen=request_seen,
            request_limit=settings.max_request_bytes,
        )
        _validate_signature(content, spec, settings)

        validated.append(
            ValidatedUpload(
                display_name=upload.client_name,
                internal_name=f"{uuid4().hex}{spec.extension}",
                extension=spec.extension,
                media_type=media_type,
                content=content,
            )
        )

    return tuple(validated)


class ExtractorRegistry:
    def __init__(self, settings: Settings) -> None:
        self._extractors: dict[str, ExtractorPort] = {
            ".pdf": PdfExtractor(settings),
            ".docx": DocxExtractor(settings),
            ".txt": TextExtractor(settings),
        }

    async def extract(
        self,
        content: bytes,
        *,
        file_name: str,
    ) -> tuple[ExtractedBlock, ...]:
        extension = PurePosixPath(file_name).suffix.lower()

        try:
            extractor = self._extractors[extension]
        except KeyError as error:
            raise UploadRejected(
                UploadErrorCode.UNSUPPORTED_TYPE,
                "no extractor is registered for this file type",
            ) from error

        return await extractor.extract(content, file_name=file_name)


ExtractionRegistry = ExtractorRegistry
