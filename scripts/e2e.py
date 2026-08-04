"""Run the deployed public-contract checks without printing sensitive bodies."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_ROOT = PROJECT_ROOT / "samples"
NO_ANSWER = "No relevant information was found in the uploaded documents."
SAMPLE_MEDIA_TYPES = {
    "facts.pdf": "application/pdf",
    "policy.docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "notes.txt": "text/plain",
}


class ContractFailure(RuntimeError):
    """A deployed API contract check failed."""


class TransportFailure(RuntimeError):
    """The API could not be contacted yet."""


@dataclass(frozen=True, slots=True)
class Response:
    status: int
    body: bytes

    def json(self) -> Any:
        try:
            return json.loads(self.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractFailure("API returned a non-JSON response") from exc


_SAFE_ERROR_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def _safe_problem_summary(response: Response) -> str:
    """Return stable, non-sensitive fields from an RFC 9457-style error."""

    if not response.body:
        return ""

    try:
        payload = json.loads(response.body)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""

    fields: list[str] = []
    for field in ("code", "request_id"):
        value = payload.get(field)
        if isinstance(value, str) and _SAFE_ERROR_TOKEN.fullmatch(value):
            fields.append(f"{field}={value}")
    return ", ".join(fields)


def _multipart(
    files: tuple[tuple[str, str, str, bytes], ...],
) -> tuple[bytes, str]:
    boundary = f"----ai-prime-capital-{uuid4().hex}"
    body = bytearray()
    for field, filename, media_type, content in files:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode()
        )
        body.extend(f"Content-Type: {media_type}\r\n\r\n".encode())
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _require_mapping(value: Any, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractFailure(message)
    return value


def _require_uuid(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractFailure(f"{field} is missing")
    try:
        UUID(value)
    except ValueError as exc:
        raise ContractFailure(f"{field} is not a UUID") from exc
    return value


def _require_document_id(value: Any, field: str = "document_id") -> str:
    """Validate the API's opaque, stable document identity.

    Document identities are content-addressed (``doc_<sha256>``) by the
    domain layer, rather than UUIDs.  Keep this check intentionally opaque so
    the E2E runner validates the public contract without coupling itself to a
    particular identity encoding.
    """

    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractFailure(f"{field} is missing")
    return value


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", value.casefold()).strip()


def _contains_concept(text: str, alternatives: tuple[str, ...]) -> bool:
    normalized_text = f" {_normalized_text(text)} "
    normalized_alternatives = tuple(_normalized_text(value) for value in alternatives)
    return bool(normalized_alternatives) and any(
        alternative and f" {alternative} " in normalized_text
        for alternative in normalized_alternatives
    )


def _validate_sources(
    payload: dict[str, Any],
    *,
    expected_files: set[str],
    require_multiple_files: bool = False,
    expected_documents: dict[str, str] | None = None,
    expected_answer_concepts: tuple[tuple[str, ...], ...] = (),
    expected_evidence_concepts: dict[str, tuple[tuple[str, ...], ...]] | None = None,
) -> set[str]:
    if payload.get("grounded") is not True:
        raise ContractFailure("answerable response is not grounded")
    if not isinstance(payload.get("answer"), str) or not payload["answer"].strip():
        raise ContractFailure("answerable response has no answer")
    for alternatives in expected_answer_concepts:
        if not _contains_concept(payload["answer"], alternatives):
            raise ContractFailure("answerable response is missing an expected fact")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ContractFailure("grounded response has no sources")

    file_names: set[str] = set()
    snippets_by_file: dict[str, list[str]] = {}
    for source in sources:
        source_map = _require_mapping(source, "source is not an object")
        document_id = _require_document_id(source_map.get("document_id"))
        _require_uuid(source_map.get("chunk_id"), "chunk_id")
        file_name = source_map.get("file_name")
        snippet = source_map.get("snippet")
        locator = source_map.get("locator")
        score = source_map.get("score")
        if not isinstance(file_name, str) or not file_name:
            raise ContractFailure("source file_name is missing")
        if not isinstance(snippet, str) or not snippet.strip():
            raise ContractFailure("source snippet is missing")
        if not isinstance(locator, dict) or not any(
            value is not None for value in locator.values()
        ):
            raise ContractFailure("source locator is empty")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ContractFailure("source score is invalid")
        if expected_documents is not None and expected_documents.get(file_name) != document_id:
            raise ContractFailure("sources do not match the persisted document identities")
        file_names.add(file_name)
        snippets_by_file.setdefault(file_name, []).append(snippet)

    if not file_names or not file_names <= expected_files:
        raise ContractFailure("sources do not identify an expected sample")
    if require_multiple_files and len(file_names & expected_files) < 2:
        raise ContractFailure("cross-format answer did not cite multiple samples")
    for file_name, concept_groups in (expected_evidence_concepts or {}).items():
        evidence = " ".join(snippets_by_file.get(file_name, ()))
        if any(not _contains_concept(evidence, alternatives) for alternatives in concept_groups):
            raise ContractFailure("source evidence is missing an expected fact")
    return file_names


class ContractRunner:
    def __init__(self, *, base_url: str, timeout: float, state_file: Path) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.state_file = state_file

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> Response:
        headers = {"User-Agent": "ai-prime-capital-e2e/1"}
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return Response(response.status, response.read())
        except HTTPError as exc:
            # Retain the bounded body for structured, redacted diagnostics. Never
            # print the raw body: it may contain user-controlled data.
            try:
                body = exc.read(16_384)
            finally:
                exc.close()
            return Response(exc.code, body)
        except (TimeoutError, URLError, OSError) as exc:
            raise TransportFailure(f"{method} {path} could not reach the API") from exc

    @staticmethod
    def passed(step: str, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        print(f"PASS {step}{suffix}")

    def wait_for_ready(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = self.request("GET", "/health/ready")
            except TransportFailure:
                time.sleep(1)
                continue
            if response.status == 200:
                payload = _require_mapping(response.json(), "ready response is not an object")
                if payload.get("status") == "ready":
                    self.passed("health.ready", "200")
                    return
            elif response.status != 503:
                raise ContractFailure(f"health.ready returned unexpected status {response.status}")
            time.sleep(1)
        raise ContractFailure("health.ready did not become ready before the timeout")

    def verify_discovery(self) -> None:
        live = self.request("GET", "/health/live")
        if (
            live.status != 200
            or _require_mapping(live.json(), "live response is invalid").get("status") != "ok"
        ):
            raise ContractFailure("health.live contract failed")
        self.passed("health.live", "200")

        docs = self.request("GET", "/docs")
        if docs.status != 200 or b"swagger" not in docs.body.lower():
            raise ContractFailure("Swagger UI contract failed")
        self.passed("docs", "200")

        openapi = self.request("GET", "/openapi.json")
        schema = _require_mapping(openapi.json(), "OpenAPI response is invalid")
        paths = schema.get("paths")
        if openapi.status != 200 or not isinstance(paths, dict):
            raise ContractFailure("OpenAPI contract failed")
        required_paths = {"/api/v1/documents", "/api/v1/questions", "/health/ready"}
        if not required_paths <= set(paths):
            raise ContractFailure("OpenAPI is missing a public route")
        self.passed("openapi", "200")

    def _sample_parts(self) -> tuple[tuple[str, str, str, bytes], ...]:
        parts = []
        for name, media_type in SAMPLE_MEDIA_TYPES.items():
            path = SAMPLES_ROOT / name
            if not path.is_file():
                raise ContractFailure(f"sample is missing: {name}")
            parts.append(("files", name, media_type, path.read_bytes()))
        return tuple(parts)

    def upload_samples(self, *, label: str) -> dict[str, str]:
        body, content_type = _multipart(self._sample_parts())
        response = self.request("POST", "/api/v1/documents", body=body, content_type=content_type)
        if response.status != 201:
            summary = _safe_problem_summary(response)
            suffix = f" ({summary})" if summary else ""
            raise ContractFailure(f"{label} returned unexpected status {response.status}{suffix}")
        payload = _require_mapping(response.json(), f"{label} response is invalid")
        documents = payload.get("documents")
        if not isinstance(documents, list) or len(documents) != len(SAMPLE_MEDIA_TYPES):
            raise ContractFailure(f"{label} returned an invalid document count")
        indexed_documents: dict[str, str] = {}
        for document in documents:
            document_map = _require_mapping(document, f"{label} document is invalid")
            document_id = _require_document_id(document_map.get("document_id"))
            file_name = document_map.get("file_name")
            if not isinstance(file_name, str) or file_name not in SAMPLE_MEDIA_TYPES:
                raise ContractFailure(f"{label} returned an unexpected file")
            if file_name in indexed_documents:
                raise ContractFailure(f"{label} returned a duplicate file")
            chunk_count = document_map.get("chunk_count")
            if (
                document_map.get("media_type") != SAMPLE_MEDIA_TYPES[file_name]
                or document_map.get("status") != "indexed"
                or not isinstance(chunk_count, int)
                or isinstance(chunk_count, bool)
                or chunk_count < 1
            ):
                raise ContractFailure(f"{label} returned an invalid index result")
            indexed_documents[file_name] = document_id
        if set(indexed_documents) != set(SAMPLE_MEDIA_TYPES):
            raise ContractFailure(f"{label} did not return every sample")
        self.passed(label, f"{len(indexed_documents)} documents")
        return indexed_documents

    def answer(self, question: str) -> dict[str, Any]:
        body = json.dumps({"question": question, "top_k": 5}).encode()
        response = self.request(
            "POST",
            "/api/v1/questions",
            body=body,
            content_type="application/json",
        )
        if response.status != 200:
            raise ContractFailure(f"question returned unexpected status {response.status}")
        return _require_mapping(response.json(), "question response is invalid")

    def verify_positive_questions(self, *, expected_documents: dict[str, str]) -> None:
        revenue = self.answer("What revenue increase is documented?")
        _validate_sources(
            revenue,
            expected_files={"facts.pdf"},
            expected_documents=expected_documents,
            expected_answer_concepts=(("12 percent", "12%"), ("2024",)),
            expected_evidence_concepts={
                "facts.pdf": (("12 percent", "12%"), ("2024",)),
            },
        )
        self.passed("question.revenue", "grounded")

        cross_format = self.answer(
            "What is the investment policy expiry date and the return window for unopened products?"
        )
        _validate_sources(
            cross_format,
            expected_files={"notes.txt", "policy.docx"},
            require_multiple_files=True,
            expected_documents=expected_documents,
            expected_answer_concepts=(
                ("December 31 2027", "31 December 2027", "2027 12 31"),
                ("30 days", "30 day"),
            ),
            expected_evidence_concepts={
                "notes.txt": (("December 31 2027", "31 December 2027", "2027 12 31"),),
                "policy.docx": (("30 days", "30 day"),),
            },
        )
        self.passed("question.cross_format", "multiple sources")

    def verify_no_answer(self) -> None:
        payload = self.answer("Which office won the moonshot award?")
        if payload != {"answer": NO_ANSWER, "grounded": False, "sources": []}:
            raise ContractFailure("absent-fact response did not match the exact fallback")
        self.passed("question.absent", "exact fallback")

    def verify_rejection(
        self, *, filename: str, media_type: str, content: bytes, expected: int
    ) -> None:
        body, content_type = _multipart((("files", filename, media_type, content),))
        response = self.request("POST", "/api/v1/documents", body=body, content_type=content_type)
        if response.status != expected:
            raise ContractFailure(f"{filename} returned unexpected status {response.status}")
        self.passed(f"upload.{filename}", str(expected))

    def verify_upload_rejections(self) -> None:
        self.verify_rejection(
            filename="unsupported.exe",
            media_type="application/octet-stream",
            content=b"not a supported document",
            expected=415,
        )
        self.verify_rejection(
            filename="oversized.txt",
            media_type="text/plain",
            content=b"x" * (10 * 1024 * 1024 + 1),
            expected=413,
        )

    def run(self, *, verify_existing: bool, ready_timeout: float) -> None:
        self.verify_discovery()
        self.wait_for_ready(timeout=ready_timeout)

        if verify_existing:
            if not self.state_file.is_file():
                raise ContractFailure("--verify-existing requires a previous state file")
            state = _require_mapping(
                json.loads(self.state_file.read_text(encoding="utf-8")),
                "E2E state file is invalid",
            )
            documents = state.get("documents")
            if not isinstance(documents, dict) or set(documents) != set(SAMPLE_MEDIA_TYPES):
                raise ContractFailure("E2E state file has no document identities")
            indexed_documents: dict[str, str] = {}
            for file_name, document_id in documents.items():
                if not isinstance(file_name, str):
                    raise ContractFailure("E2E state file has an invalid file name")
                _require_document_id(document_id)
                indexed_documents[file_name] = document_id
            self.passed("upload.verify_existing", "no re-upload")
        else:
            first_ids = self.upload_samples(label="upload.initial")
            second_ids = self.upload_samples(label="upload.repeat")
            if first_ids != second_ids:
                raise ContractFailure("repeat upload changed document identities")
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(
                json.dumps({"documents": first_ids}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            indexed_documents = first_ids
            self.passed("upload.idempotent", "stable document identities")

        self.verify_positive_questions(expected_documents=indexed_documents)
        self.verify_no_answer()
        self.verify_upload_rejections()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--ready-timeout", type=float, default=90.0)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("/tmp/ai-prime-capital-e2e-state.json"),
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="skip uploads and verify indexed data after a restart",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        ContractRunner(
            base_url=args.base_url,
            timeout=args.timeout,
            state_file=args.state_file,
        ).run(verify_existing=args.verify_existing, ready_timeout=args.ready_timeout)
    except (ContractFailure, TransportFailure) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
