# privaite-bench

PII detection benchmark suite for [PrivAiTe](https://github.com/crp4222/PrivAiTe).

Measures detection rate, false positive rate, and latency across languages and entity types.

## Latest results

**61 documents, 244 PII entities, 5 languages, 8 entity types.**

| Metric | Result |
|--------|--------|
| Detection rate | 97.1% (237/244) |
| False positives | 1/14 clean texts (Kubernetes flagged as LOCATION) |
| EMAIL | 100% (59/59) |
| PHONE | 100% (52/52) |
| IBAN | 100% (9/9) |
| CREDIT_CARD | 100% (7/7) |
| US_SSN | 100% (3/3) |
| IP_ADDRESS | 100% (2/2) |
| DATE_TIME | 100% (11/11) |
| PERSON | 93% (94/101) — weakest, see Known misses below |
| FR | 98% |
| EN | 100% |
| DE | 93% |
| ES | 89% |
| IT | 91% |

### Known misses

All 7 misses are PERSON entities that spaCy NER doesn't detect:
- Single-word names without context ("Schmidt", "Isabelle", "Schneider-Weber")
- Long multi-part Spanish names ("María del Carmen Ortega Ruiz", "Fernando José Méndez Castillo")
- Compound name with particle ("Francesca De Luca")

Regex-based entities (email, phone, IBAN, credit card, IP, SSN) are 100%.

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
