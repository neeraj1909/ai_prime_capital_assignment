# Reviewer samples

These three small documents are created for this assessment and contain no
private or third-party material:

- `facts.pdf` — one-page revenue fact; expected provenance is page 1.
- `policy.docx` — one paragraph about the return window; expected provenance is
  paragraph 0.
- `notes.txt` — UTF-8 notes containing the policy expiry and capital reserves;
  expected provenance starts at offset 0.

`manifest.json` records SHA-256 hashes, media types, known facts, expected
locators, and deliberately absent facts. The binary files are generated
deterministically by `generate_samples.py` using only the Python standard
library, so a reviewer can reproduce or verify them:

```bash
uv run python samples/generate_samples.py --check
```

The absent facts (`moonshot office` and `Antarctica headcount`) are not present
in any sample. Their labels are negative evaluation inputs, not model-generated
claims.
