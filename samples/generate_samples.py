from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_STORED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parent

PDF_TEXT = "Annual revenue increased by 12 percent in fiscal year 2024, driven by enterprise sales."
DOCX_TEXT = "Customers may return unopened products within 30 days of delivery."


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(text: str) -> bytes:
    stream = (f"BT\n/F1 12 Tf\n72 720 Td\n({_pdf_escape(text)}) Tj\nET\n").encode("latin-1")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"endstream",
    )
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, value in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(value)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def _zip_entry(name: str, content: bytes) -> tuple[str, bytes]:
    return name, content


def build_docx(text: str) -> bytes:
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>
""".encode()
    content_types = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        b'  <Default Extension="rels" '
        b'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        b'  <Default Extension="xml" ContentType="application/xml"/>\n'
        b'  <Override PartName="/word/document.xml" '
        b'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>\n'
        b"</Types>\n"
    )
    root_rels = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        b'  <Relationship Id="rId1" '
        b'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        b'relationships/officeDocument" '
        b'Target="word/document.xml"/>\n'
        b"</Relationships>\n"
    )
    entries = (
        _zip_entry("[Content_Types].xml", content_types),
        _zip_entry("_rels/.rels", root_rels),
        _zip_entry("word/document.xml", document_xml),
    )
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_STORED) as archive:
        for name, content in entries:
            info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    return output.getvalue()


def expected_files() -> dict[str, bytes]:
    return {
        "facts.pdf": build_pdf(PDF_TEXT),
        "policy.docx": build_docx(DOCX_TEXT),
        "notes.txt": (ROOT / "notes.txt").read_bytes(),
    }


def expected_manifest(files: dict[str, bytes]) -> dict[str, object]:
    return {
        "version": 1,
        "documents": [
            {
                "file": "facts.pdf",
                "media_type": "application/pdf",
                "sha256": hashlib.sha256(files["facts.pdf"]).hexdigest(),
                "facts": [PDF_TEXT],
                "expected_locator": {"page_number": 1},
            },
            {
                "file": "policy.docx",
                "media_type": (
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ),
                "sha256": hashlib.sha256(files["policy.docx"]).hexdigest(),
                "facts": [DOCX_TEXT],
                "expected_locator": {"paragraph_index": 0},
            },
            {
                "file": "notes.txt",
                "media_type": "text/plain",
                "sha256": hashlib.sha256(files["notes.txt"]).hexdigest(),
                "facts": [
                    "The investment policy expires on December 31, 2027.",
                    "Capital reserves are reported as 10 million dollars.",
                ],
                "expected_locator": {"start_offset": 0},
            },
        ],
        "absent_facts": ["moonshot office", "Antarctica headcount"],
    }


def check() -> None:
    files = expected_files()
    for name, expected in files.items():
        path = ROOT / name
        if not path.is_file() or path.read_bytes() != expected:
            raise SystemExit(f"sample is missing or differs: {path}")
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest != expected_manifest(files):
        raise SystemExit("samples/manifest.json differs from deterministic inputs")


def generate() -> None:
    files = expected_files()
    for name, content in files.items():
        (ROOT / name).write_bytes(content)
    (ROOT / "manifest.json").write_text(
        json.dumps(expected_manifest(files), indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated samples")
    args = parser.parse_args()
    check() if args.check else generate()


if __name__ == "__main__":
    main()
