# Comparative benchmark

Corpus: 120 real AI4Privacy documents (458 PII items labeled by 10 independent auditor agents) across de, en, fr, it, plus 14 clean documents for false positives. Methodology and caveats are at the end.

## Bottom line

`privaite-onnx` has the highest recall (84.5% span / 79.9% strict) with fewer false positives than the Presidio baseline (2 vs 3 on 14 clean docs), and it is the only solution that also strips PII from tool-call arguments and multimodal content (100.0% tool-call protection vs the flat-text baseline's 0.6%). The `light` preset trades recall for near-zero latency. (`privaite-light` is the crippled 9-entity-allowlist config; `privaite-light-all` is the real light preset.)

## Headline

| Solution | Recall | Recall (strict) | False positives | Tool-call protection | Tool-call leak | Multimodal leak | Latency |
|---|---|---|---|---|---|---|---|
| privaite-light | 34.5% | 33.2% | 0 on 14 | 100.0% | 65.5% | 65.5% | 98.0ms |
| privaite-light-all | 62.4% | 57.6% | 3 on 14 | 100.0% | 37.6% | 37.6% | 67.2ms |
| privaite-onnx | 84.5% | 79.9% | 2 on 14 | 100.0% | 15.5% | 15.5% | 603.5ms |
| presidio-baseline | 70.3% | 65.3% | 3 on 14 | 0.6% | 99.1% | 100.0% | 10.2ms |

Tool-call protection is, of the PII a solution catches in plain text, how much it also removes from a tool-call argument (higher is better). Tool-call leak and multimodal leak are the share of all PII that survives inside a tool-call argument or a multimodal text part (lower is better).

## Recall by language

| Solution | de | en | fr | it |
|---|---|---|---|---|
| privaite-light | 36.6% | 34.7% | 29.5% | 37.1% |
| privaite-light-all | 60.7% | 68.6% | 64.3% | 56.0% |
| privaite-onnx | 82.1% | 76.3% | 89.3% | 90.5% |
| presidio-baseline | 64.3% | 81.4% | 69.6% | 65.5% |

## Recall by entity type

| Solution | CREDIT_CARD | DATE_TIME | EMAIL_ADDRESS | FINANCIAL | IBAN_CODE | IP_ADDRESS | LOCATION | ORGANIZATION | PERSON | PHONE_NUMBER | SECRET | URL | US_SSN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| privaite-light | 11.1% | 59.3% | 100.0% | 6.5% | 85.7% | 76.5% | 13.2% | 0.0% | 33.9% | 50.0% | 0.0% | 0.0% | 50.0% |
| privaite-light-all | 11.1% | 64.4% | 100.0% | 38.7% | 100.0% | 100.0% | 67.1% | 44.4% | 58.9% | 50.0% | 28.6% | 100.0% | 50.0% |
| privaite-onnx | 100.0% | 89.8% | 100.0% | 87.1% | 100.0% | 100.0% | 73.7% | 61.1% | 86.6% | 100.0% | 71.4% | 42.1% | 100.0% |
| presidio-baseline | 11.1% | 78.0% | 100.0% | 37.1% | 100.0% | 100.0% | 69.7% | 72.2% | 81.2% | 50.0% | 28.6% | 100.0% | 12.5% |

## Methodology and caveats

Scoring is substring based: a PII item is caught when it no longer appears in the solution's output. Two recall columns are reported. **Recall** is span-level: a multi-token span (e.g. a full name or a street address) counts as caught when its exact full string disappears, so a partial redaction is credited as a full catch and this is an upper bound. **Recall (strict)** is token-level: every >=4-char token of the span must be removed. The truth is between the two; the gap (~1-5pp, roughly uniform across solutions) does not change the ranking.

**Ground truth.** The labels are produced by 10 independent auditor agents and cross-checked against AI4Privacy's own sensitive mask (loose substring overlap, so these are an upper bound). The agents independently recovered 93.1% of the dataset's 554 sensitive spans, and 93.4% of the agent labels overlap the dataset mask (the rest are incidental PII the dataset did not tag). The labels are independent of any solution under test: if they were a product's own detections, that product would score 100% recall, and none does.

**Baseline.** `presidio-baseline` is vanilla Microsoft Presidio run on flat message text (full entity set, default threshold). It is the common flat-text approach behind most drop-in PII proxies, NOT the strongest possible competitor integration: it does not look inside tool-call arguments or multimodal content by design, which is why its tool-call/multimodal numbers are a floor. A head-to-head against competitors' own structured-aware integrations (e.g. LiteLLM's Presidio guardrail output parsing) is future work; read the structured columns as 'structured-aware vs the flat-text approach', not 'vs every competitor'.

**Latency** is hardware-dependent and not reproducible run-to-run (ONNX in particular varies with CoreML/CPU warmup); treat it as indicative, not exact.

Reproduce: `python solutions/ai4privacy_loader.py && python solutions/compare.py`
