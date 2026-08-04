# Retrieval calibration

The committed evaluation corpus and labels live in
`tests/evaluation/cases.yaml`. The file is JSON syntax deliberately kept inside
the YAML extension so the suite can use the standard library and does not add a
second dependency solely for test data.

The offline harness uses deterministic token overlap only as a calibration
oracle for the current fixture set. It is not a substitute for OpenAI
embeddings or a live Qdrant run. Production retrieval remains the Qdrant adapter
with the configured named vector, `retrieval_top_k=5`, and starting threshold
`retrieval_min_score=0.35`.

The cases cover exact, semantic, paraphrase, cross-format metadata, boundary,
distractor, absent-information, and embedded-instruction questions. Expected
labels and relevant chunk IDs are versioned inputs; tuning the scorer or runtime
configuration must never edit labels to make a result pass.

The evaluation gate reports recall@5, unanswerable false-positive rate, and
citation validity. The current deterministic fixture gate requires recall@5 of
at least 90%, false-positive rate of at most 10%, and 100% citation validity.
Run it with:

```bash
uv run --locked pytest -m evaluation --no-cov -s
```

The adversarial fixture contains an instruction-like paragraph beside a factual
statement. The expected contract is still source-ID citation plus server-owned
metadata; the instruction text never becomes policy or metadata.
