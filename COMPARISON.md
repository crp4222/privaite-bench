# Comparative benchmark

Corpus: 120 real AI4Privacy documents (458 PII items labeled by 10 independent auditor agents) across de, en, fr, it, plus 14 clean documents for false positives.

Scoring is substring based: a PII item is caught when it no longer appears in the solution's output.

## Ground truth

The labels are produced by 10 independent auditor agents and cross-checked against AI4Privacy's own sensitive mask. The agents independently recovered 93.1% of the dataset's 554 sensitive spans, and 93.4% of the agent labels overlap the dataset mask (the rest are incidental PII the dataset did not tag). High overlap means the ground truth is precise.

## Headline

| Solution | Recall | False positives | Tool-call protection | Tool-call leak | Multimodal leak | Latency |
|---|---|---|---|---|---|---|
| privaite-light | 34.5% | 0 on 14 | 100.0% | 65.5% | 65.5% | 99.1ms |
| privaite-onnx | 84.5% | 2 on 14 | 100.0% | 15.5% | 15.5% | 749.0ms |
| presidio-baseline | 70.3% | 3 on 14 | 0.6% | 99.1% | 100.0% | 11.5ms |

Tool-call protection is, of the PII a solution catches in plain text, how much it also removes from a tool-call argument (higher is better). Tool-call leak and multimodal leak are the share of all PII that survives inside a tool-call argument or a multimodal text part (lower is better).

## Recall by language

| Solution | de | en | fr | it |
|---|---|---|---|---|
| privaite-light | 36.6% | 34.7% | 29.5% | 37.1% |
| privaite-onnx | 82.1% | 76.3% | 89.3% | 90.5% |
| presidio-baseline | 64.3% | 81.4% | 69.6% | 65.5% |

## Recall by entity type

| Solution | CREDIT_CARD | DATE_TIME | EMAIL_ADDRESS | FINANCIAL | IBAN_CODE | IP_ADDRESS | LOCATION | ORGANIZATION | PERSON | PHONE_NUMBER | SECRET | URL | US_SSN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| privaite-light | 11.1% | 59.3% | 100.0% | 6.5% | 85.7% | 76.5% | 13.2% | 0.0% | 33.9% | 50.0% | 0.0% | 0.0% | 50.0% |
| privaite-onnx | 100.0% | 89.8% | 100.0% | 87.1% | 100.0% | 100.0% | 73.7% | 61.1% | 86.6% | 100.0% | 71.4% | 42.1% | 100.0% |
| presidio-baseline | 11.1% | 78.0% | 100.0% | 37.1% | 100.0% | 100.0% | 69.7% | 72.2% | 81.2% | 50.0% | 28.6% | 100.0% | 12.5% |

Reproduce: `python solutions/ai4privacy_loader.py && python solutions/compare.py`
