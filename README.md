# privaite-bench

PII detection benchmark suite for [PrivAiTe](https://github.com/crp4222/PrivAiTe).

Measures detection rate, false positive rate, and latency across languages and entity types.

## Datasets

- `datasets/pii_samples.json` — 12 texts with annotated PII (FR, EN, DE, mixed)
- `datasets/clean_samples.json` — 14 texts with zero PII (should trigger no detections)

## Run

```bash
cd privaite-bench
python bench.py
```

Results are printed to stdout and saved to `results/report.json`.

## What it measures

| Metric | Description |
|--------|-------------|
| Detection rate | % of known PII correctly anonymized |
| Miss rate | % of known PII that leaked through |
| False positive rate | % of clean texts where something was incorrectly flagged |
| Latency | ms per request, PII vs clean |
| By language | Detection rate per language (FR, EN, DE) |
| By entity type | Detection rate per type (PERSON, EMAIL, PHONE, etc.) |

## Adding test cases

Add entries to `datasets/pii_samples.json`:

```json
{
  "id": "my_test_case",
  "lang": "fr",
  "text": "The input text with PII",
  "expected": {
    "the PII value": "ENTITY_TYPE"
  }
}
```

For clean texts (false positive testing), add to `datasets/clean_samples.json` with `"expected": {}`.
