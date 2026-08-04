# Requirement-to-test matrix

This matrix keeps lower-rung local evidence separate from the Qdrant, live
provider, and Docker evidence owners declared in the assessment plan.

| Requirement | Local executable evidence | Named boundary still required |
|---|---|---|
| F-001 upload documents | `tests/api/test_contract.py::test_document_upload_invokes_one_use_case_and_returns_201`; `tests/unit/test_upload_guards.py` | Docker E2E with PDF/DOCX/TXT and stable indexed counts |
| F-002 extraction/provenance | `tests/unit/test_extractors.py`; `tests/evaluation/test_samples.py` | Integration coverage against persistent indexed metadata |
| F-003 idempotent indexing | `tests/unit/test_identity.py`; `tests/unit/test_ingest_documents.py`; `tests/integration/test_qdrant_indexing.py` | Dedicated Qdrant run when `localhost:6333` is available |
| F-004 retrieval | `tests/unit/test_qdrant_store.py`; `tests/evaluation/test_retrieval.py` | Qdrant integration plus calibrated evaluation corpus |
| F-005 grounded answer/sources | `tests/unit/test_answer_question.py`; `tests/api/test_contract.py::test_question_route_returns_server_owned_sources` | Live-provider Docker E2E |
| F-006 deterministic refusal | `tests/unit/test_answer_question.py`; `tests/api/test_contract.py::test_unanswerable_question_preserves_exact_fallback_contract` | Docker E2E proof of the complete route/provider path |
| F-007 health/discovery | `tests/api/test_health.py`; `tests/api/test_contract.py::test_openapi_and_docs_expose_public_contract_examples` | Docker E2E readiness and deployment check |
| F-008 representative samples | `samples/manifest.json`; `tests/evaluation/test_samples.py` | Repository review and E2E runner use |
| NFR-001 upload safety | `tests/unit/test_upload_guards.py`; extractor tests; API error cases | API/integration evidence for every deployment boundary |
| NFR-002 grounding integrity | `tests/unit/test_grounding.py`; `tests/evaluation/test_retrieval.py` | Live evaluation for provider behavior |
| NFR-003 boundaries | `tests/unit/test_architecture.py`; `uv run --locked mypy src` | Static/CI enforcement at release boundary |
| NFR-004 reproducibility | `uv.lock`, `requirements.txt`, export drift command, `uv lock --check` | Clean checkout and Docker proof |
| NFR-005 evaluation quality | `tests/evaluation/test_retrieval.py` | Re-run with the declared live/provider evidence owner |
| NFR-006 privacy | `tests/unit/test_openai_answer_gateway.py`; API redaction/log tests | Integration log capture and provider boundary |
| NFR-007 lifecycle/errors | `tests/api/test_health.py`; API error tests; default-startup smoke | Docker lifecycle/log evidence |
| NFR-008 observability | `tests/api/test_contract.py::test_logs_contain_request_metadata_but_not_error_details` | Integration log assertions with deployment telemetry |
| D-001 source code | Import, Ruff, mypy, compileall, unit/API suite | Level 1–3 clean run |
| D-002 README | Root `README.md` and documented command review | Clean-reader review plus E2E |
| D-003 requirements | `requirements.txt` matches locked export command | Export drift gate |
| D-004 OpenAPI | API contract tests and `/openapi.json` examples | Docker route discovery |
| D-005 samples | `samples/manifest.json`, generator `--check`, sample extraction tests | E2E runner use |

An unavailable provider or Qdrant service is recorded as missing evidence; it is
never converted into a passing local fake claim.
