# AI Prime Capital — document-grounded RAG API

This repository implements the mandatory vertical slice from the AI Developer
Technical Assessment: upload PDF, DOCX, or UTF-8 TXT documents, index their
text in Qdrant, and answer questions only when retrieved evidence supports the
answer.

## Current status

The local foundation, ingestion pipeline, retrieval/evidence gate, structured
answer adapter, citation validation, API contract, sample corpus, offline
evaluation, pinned container build, health-gated Compose deployment, and real
Qdrant integration gate are implemented. The live E2E runner reaches the
deployed API but the supplied OpenAI credential currently returns `401`, so no
live answer-quality claim is made.

## Architecture

```text
                         HTTP / JSON / multipart
                                  |
                     API delivery layer (FastAPI)
                   routes · schemas · safe errors
                                  |
                         application use cases
                  IngestDocuments · AnswerQuestion
                         /                 \
              deterministic rules      typed ports
             identity · chunking ·     ExtractorPort
             grounding · provenance    EmbedderPort
                                      GeneratorPort
                                     VectorStorePort
                                           ^
                                           |
                               infrastructure adapters
                       PDF/DOCX/TXT · OpenAI · Qdrant

Composition root: config.py + bootstrap.py + main.py
```

Dependencies point inward: the API and infrastructure depend on the application
contracts and domain types; the application does not import FastAPI, OpenAI, or
Qdrant. Vendor SDK objects and transport models therefore stop at their adapter
boundaries.

| Area | Responsibility | Deliberately excluded |
|---|---|---|
| `api/` | HTTP routes, request/response schemas, dependency providers, OpenAPI, request IDs, and safe problem responses | extraction, retrieval, prompting, and grounding policy |
| `application/` | Coordinates the two use cases and defines the narrow extractor, embedding, generation, and vector-store ports | HTTP and vendor SDK types |
| `domain/` | Deterministic identities, chunk/provenance models, evidence qualification, citation validation, and the exact no-answer result | FastAPI, OpenAI, and Qdrant concerns |
| `infrastructure/` | Bounded PDF/DOCX/TXT extraction plus OpenAI Responses/embedding and Qdrant adapters | public API contracts and business decisions |
| `bootstrap.py` | Composition root and lifecycle ownership; constructs shared clients, probes readiness, and closes resources | request-specific behavior |

### Request flows

Document ingestion follows one ordered pipeline:

```text
multipart upload
  -> validate count, size, type, signature, and archive safety
  -> extract text with provenance
  -> create deterministic token-bounded chunks and IDs
  -> request embeddings
  -> idempotently upsert named vectors and payloads into Qdrant
  -> return stable document IDs and chunk counts
```

Question answering has an explicit evidence gate before generation:

```text
question
  -> validate and embed
  -> retrieve threshold-qualified Qdrant candidates
  -> no candidates: return the exact fallback without calling the answer model
  -> candidates: request a strict structured OpenAI Responses decision
  -> validate every cited ID against the retrieved set
  -> build answer sources only from server-owned chunk metadata
```

This separation keeps the highest-risk rules—threshold qualification, the
no-answer decision, citation membership, and returned provenance—deterministic
and testable outside the model. `bootstrap.py` creates shared adapters once per
FastAPI lifespan; readiness stays false when provider access or the Qdrant
collection contract is unavailable or incompatible.

## Technology choices

Python 3.12 and FastAPI provide typed async application boundaries and generated
Swagger/OpenAPI. `pypdf`, `python-docx`, and `tiktoken` handle bounded text
extraction and token-aware chunks. OpenAI supplies embeddings and structured
Responses output; Qdrant stores named vectors and provenance payloads. `uv.lock`
is the dependency source of truth; `requirements.txt` is the assessment-facing
production export.

## Setup

```bash
uv sync --locked
cp .env.example .env
```

Set at least `OPENAI_API_KEY` and `OPENAI_CHAT_MODEL` in `.env`. Compose reads
those two values from the project `.env` file (exported shell variables are not
forwarded), so recreate the API after changing the file. The default
runtime also needs a reachable Qdrant instance at `QDRANT_URL` (default:
`http://localhost:6333`). Do not commit `.env` or credentials.

Start the API:

```bash
uv run --locked uvicorn ai_prime_capital.main:app --host 127.0.0.1 --port 8000
```

Liveness and OpenAPI remain available even when readiness is `503` because a
provider or Qdrant prerequisite is missing:

```bash
curl -i http://127.0.0.1:8000/health/live
curl -i http://127.0.0.1:8000/health/ready
```

Open `http://127.0.0.1:8000/docs` in a browser for Swagger UI.

If `/health/ready` returns `503`, inspect its problem response: the usual
causes are a missing `OPENAI_API_KEY`, an unavailable Qdrant URL, or a vector
collection whose dimensions/metric do not match the configured embedding
model. Liveness and OpenAPI remain useful for diagnosing those cases. The
Compose API health check remains unhealthy until Qdrant compatibility succeeds;
it never recreates an incompatible collection.

## Docker deployment and E2E

The default Compose deployment publishes only the API on port `8000`; Qdrant is
on an internal network with a named volume. Set `OPENAI_API_KEY` and
`OPENAI_CHAT_MODEL` in `.env` before starting it:

```bash
docker compose config --quiet
docker compose build --pull
docker compose up -d --wait
uv run --locked python scripts/e2e.py --base-url http://localhost:8000
```

After a restart, verify that the samples remain queryable without re-uploading:

```bash
docker compose restart api qdrant
docker compose up -d --wait
uv run --locked python scripts/e2e.py --base-url http://localhost:8000 --verify-existing
docker compose ps
docker compose logs --no-color api qdrant
```

The disposable integration database is available only under the `test` profile:

```bash
docker compose --profile test up -d --wait qdrant-test
```

The E2E runner prints only step names, status codes, counts, and contract
labels. It never prints response bodies, prompts, document text, or secrets.

## API usage

Upload one or more documents as repeated `files` parts:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/documents \
  -F 'files=@samples/facts.pdf;type=application/pdf' \
  -F 'files=@samples/policy.docx;type=application/vnd.openxmlformats-officedocument.wordprocessingml.document' \
  -F 'files=@samples/notes.txt;type=text/plain'
```

Ask a question:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/questions \
  -H 'content-type: application/json' \
  -d '{"question":"What revenue increase is documented?","top_k":5}'
```

An answerable response has `grounded: true` and server-owned `sources`:

```json
{
  "answer": "Revenue increased by 12 percent in fiscal year 2024.",
  "grounded": true,
  "sources": [{
    "document_id": "…",
    "file_name": "facts.pdf",
    "chunk_id": "…",
    "snippet": "Annual revenue increased by 12 percent…",
    "score": 0.91,
    "locator": {"page_number": 1}
  }]
}
```

When no qualifying evidence exists, the API returns exactly:

```json
{
  "answer": "No relevant information was found in the uploaded documents.",
  "grounded": false,
  "sources": []
}
```

Expected client/provider failures use `application/problem+json` with stable
`status`, `code`, `instance`, and `request_id` fields. Details never echo raw
uploads, prompts, questions, answers, secrets, or stack traces.

## Samples and evaluation

The assessment-owned samples and manifest are in [`samples/`](samples/README.md).
Verify deterministic PDF/DOCX/TXT bytes and hashes with:

```bash
uv run --locked python samples/generate_samples.py --check
```

The offline calibration corpus and labels are in `tests/evaluation/`. It covers
exact, semantic, paraphrase, cross-format, boundary, distractor, absent-fact,
and prompt-injection cases. It is a deterministic calibration oracle, not proof
of universal factuality or live-provider quality.

## Validation

```bash
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked python -m compileall -q src tests
uv lock --check
timeout 60 uv run --locked pytest -m "unit or api" --no-cov
timeout 60 uv run --locked pytest --no-cov
timeout 60 uv run --locked pytest -m evaluation --no-cov -s
uv run --locked python scripts/e2e.py --help
```

If Qdrant is already available, run the integration gate separately:

```bash
REQUIRE_QDRANT_INTEGRATION=1 timeout 60 uv run --locked pytest -m integration --no-cov
```

The `REQUIRE_QDRANT_INTEGRATION=1` guard makes an unavailable Qdrant service a
failed gate instead of a green run with skipped tests.

Regenerate the reviewer-facing dependency export without hashes/annotations:

```bash
uv export --locked --no-dev --no-emit-project --no-hashes --no-annotate \
  --no-header --format requirements.txt --output-file requirements.txt
```

## Assumptions and limitations

- This is a single-tenant local/demo API. Authentication, per-user isolation,
  conversation history, streaming, and UI are deferred.
- PDFs must contain extractable text; OCR for scanned/image-only PDFs is out of
  scope. Raw upload bytes are not persisted by the ingestion service.
- Upload extension, declared media type, signatures, archive paths, size limits,
  and extracted-text limits are validated before vector upsert.
- Retrieved document text is untrusted data. It cannot override system or
  application instructions. Positive answers require server-validated citations.
- OpenAI calls and Qdrant are external dependencies. Without credentials or a
  reachable compatible collection, readiness fails closed; this is not a fake
  success response.
- Retrieval threshold/chunk settings are calibrated starting values. Small
  offline corpora do not eliminate hallucination risk or guarantee general-world
  factuality.

## Evidence map

See [`docs/requirement-test-matrix.md`](docs/requirement-test-matrix.md) for the
requirement-to-test boundary matrix and the distinction between local evidence,
Qdrant evidence, live-provider evidence, and Docker E2E evidence.
