# privaite-bench

PII detection benchmark suite for [PrivAiTe](https://github.com/crp4222/PrivAiTe).

Measures detection rate, false positive rate, and latency across languages and entity types.

## Latest results

### Precision / Recall / F1 (entity-level, 64 documents, 5 languages)

We measure at the entity level:
- **TP** = entity correctly flagged (matches expected)
- **FP** = entity flagged but not in expected list
- **FN** = entity in expected list but not flagged

| Metric | Value |
|--------|-------|
| **Precision** | **84.8%** (of what we flag, how much is real PII) |
| **Recall** | **94.3%** (of real PII, how much we catch) |
| **F1** | **89.3%** |
| TP / FP / FN | 263 / 47 / 16 |

**By language:**

| Lang | Precision | Recall | F1 |
|------|-----------|--------|----|
| FR | 91.0% | 98.2% | 94.5% |
| EN | 75.4% | 90.8% | 82.4% |
| DE | 89.7% | 89.7% | 89.7% |
| ES | 89.5% | 94.4% | 91.9% |
| IT | 91.7% | 100.0% | 95.7% |

### Honest analysis of false positives

Many "false positives" are actually real PII we forgot to include in the expected list:
- DLP test SSNs we didn't annotate ("172-32-1176", "514-14-8905", etc.)
- Dates in expected docs ("23 décembre 2025", "14/03/1958")
- Account numbers ("9876543210", "0012345678901")

Genuine over-anonymization (model errors) is limited to:
- Field labels: "Name", "Email", "Nombre"
- Honorifics / titles: "Dr", "Mrs"
- Salutations: "Dear Sir"
- Short alphanumeric IDs misread as account numbers

EN has the lowest precision because we annotated less rigorously and the English dataset has more form-style documents (more labels like "Name:", "Dear Dr.").

### Known recall misses (16 FN out of 279 expected)

PERSON entities that NER doesn't detect:
- Single-word names without context ("Schmidt", "Isabelle", "Schneider-Weber")
- Long multi-part Spanish names ("María del Carmen Ortega Ruiz")
- Compound names with particle ("Francesca De Luca")

Regex-based entities (email, phone, IBAN, credit card, IP, SSN) reach near-100%.

## Latency (Apple M1 Pro, light preset)

Measured on real corporate documents at p50/p95/p99 over 30 runs each. Adding a language adds ~70-100ms per request.

| Languages active | ~500 tokens p50 | ~1000 tokens p50 |
|------------------|-----------------|------------------|
| 1 (en or fr) | 62 ms | 113 ms |
| 2 (fr + en) | 129 ms | 232 ms |
| 3 (fr + en + de) | 199 ms | 356 ms |
| 5 (fr + en + de + es + it) | 326 ms | 587 ms |
| 7 (+ pt + nl) | 460 ms | 814 ms |

p95 stays within 2-5% of p50. p99 spikes occasionally to 1s on the largest configurations (cold spaCy state).

Boot time: 6.3 s for 1 language, ~4.2 s for 7 languages (cached spaCy models).

If you only need 1-2 languages, latency is well under 250 ms even on long documents. Enabling 7 languages is meant for multi-lingual SaaS use cases and triples the cost.

## Datasets

### Synthetic (self-built)
- `datasets/pii_samples.json` — 12 base PII samples (FR, EN, DE, mixed)
- `datasets/corporate_samples.json` — 8 corporate documents (bank transfers, insurance claims, HR records, contracts)
- `datasets/batch_samples.json` — 16 additional documents (CVs, leases, complaints, NDAs, invoices, medical referrals)
- `datasets/clean_samples.json` — 14 clean texts with zero PII (should trigger no detections)

### External / public sources
- `datasets/dlptest_us.json` — 4 entries from [DLP Test](https://dlptest.com/sample-data/) public sample data (US names, SSN, emails, credit cards)
- `datasets/enedis_rse_extracts.json` — 11 extracts from the [Enedis RSE 2024 report](https://www.enedis.fr/) (public corporate document, 105 pages). Includes PII samples (photographer names, executive contacts) and clean business text (financial figures, carbon metrics, biodiversity data). Each entry cites the source page.
- `datasets/realworld_samples.json` — 10 additional real-world inspired samples from the same report

### Data sources transparency

| Dataset | Documents | PII count | Source | Synthetic? |
|---------|-----------|-----------|--------|------------|
| pii_samples | 12 | 37 | Self-built | Yes |
| corporate_samples | 8 | 48 | Self-built | Yes |
| batch_samples | 16 | 99 | Self-built | Yes |
| dlptest_us | 4 | 16 | [dlptest.com](https://dlptest.com/sample-data/) | No (public test data) |
| enedis_rse_extracts | 11 | 18 | [Enedis RSE 2024](https://www.enedis.fr/) | No (public report) |
| realworld_samples | 10 | 26 | Enedis RSE 2024 inspired | Mixed |
| clean_samples | 14 | 0 | Self-built | Yes |

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
| By language | Detection rate per language (FR, EN, DE, ES, IT) |
| By entity type | Detection rate per type (PERSON, EMAIL, PHONE, etc.) |

## Adding test cases

Add entries to any dataset JSON file:

```json
{
  "id": "my_test_case",
  "lang": "fr",
  "source": "where this data comes from",
  "text": "The input text with PII",
  "expected": {
    "the PII value": "ENTITY_TYPE"
  },
  "must_preserve": ["business terms that should NOT be anonymized"]
}
```

For clean texts (false positive testing), add to `datasets/clean_samples.json` with `"expected": {}`.

## Contributing

Found a document type that breaks PrivAiTe? Add it as a test case and open a PR. Every false positive or missed detection reported here directly improves the project.
