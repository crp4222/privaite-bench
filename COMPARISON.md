# Comparative benchmark

Corpus: 120 real documents from the open [AI4Privacy `pii-masking-200k`](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) dataset (Hugging Face), 458 PII items labeled by 10 independent auditor agents, across de, en, fr, it, plus 14 clean documents for false positives. Methodology, dataset licensing, and caveats are at the end.

## Bottom line

`privaite-onnx` (the default full ONNX preset) has the highest recall (84.9% span / 81.0% strict) and is the only solution that also strips PII from tool-call arguments: it removes 100.0% of the PII it catches from a tool-call argument, while LiteLLM's Presidio guardrail and LLM Guard remove 0.0% and 0.0%. Those tools scan message text (LiteLLM's guardrail also scrubs multimodal text parts, so its multimodal leak is 29.7%) but never parse the tool-call JSON, so 100.0% and 100.0% of all PII survives inside a tool call. `privaite-onnx` also keeps false positives low (2 on 14 clean docs). (`privaite-light-all` is the fast Presidio-only preset; `privaite-light` is the crippled 9-entity-allowlist config, shown for reference.)

## Headline

| Solution | Recall | Recall (strict) | False positives | Tool-call protection | Tool-call leak | Multimodal leak | Latency |
|---|---|---|---|---|---|---|---|
| privaite-onnx | 84.9% | 81.0% | 2 on 14 | 100.0% | 15.1% | 15.1% | 671.9ms |
| privaite-light-all | 62.7% | 58.1% | 3 on 14 | 100.0% | 37.3% | 37.3% | 108.9ms |
| privaite-light | 36.7% | 35.4% | 0 on 14 | 100.0% | 63.3% | 63.3% | 97.3ms |
| litellm-presidio | 70.3% | 65.3% | 3 on 14 | 0.0% | 100.0% | 29.7% | 27.2ms |
| llm-guard | 76.9% | 74.9% | 5 on 14 | 0.0% | 100.0% | 100.0% | 88.5ms (offline) |

Tool-call protection is, of the PII a solution catches in plain text, how much it also removes from a tool-call argument (higher is better). Tool-call leak and multimodal leak are the share of all PII that survives inside a tool-call argument or a multimodal text part (lower is better).

## Recall by language

| Solution | de | en | fr | it |
|---|---|---|---|---|
| privaite-onnx | 82.1% | 76.3% | 91.1% | 90.5% |
| privaite-light-all | 60.7% | 68.6% | 64.3% | 56.9% |
| privaite-light | 36.6% | 34.7% | 37.5% | 37.9% |
| litellm-presidio | 64.3% | 81.4% | 69.6% | 65.5% |
| llm-guard | 83.9% | 71.2% | 73.2% | 79.3% |

## Recall by entity type

| Solution | CREDIT_CARD | DATE_TIME | EMAIL_ADDRESS | FINANCIAL | IBAN_CODE | IP_ADDRESS | LOCATION | ORGANIZATION | PERSON | PHONE_NUMBER | SECRET | URL | US_SSN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| privaite-onnx | 100.0% | 89.8% | 100.0% | 87.1% | 100.0% | 100.0% | 76.3% | 61.1% | 86.6% | 100.0% | 71.4% | 42.1% | 100.0% |
| privaite-light-all | 11.1% | 64.4% | 100.0% | 38.7% | 100.0% | 100.0% | 67.1% | 44.4% | 58.9% | 50.0% | 35.7% | 100.0% | 50.0% |
| privaite-light | 11.1% | 59.3% | 100.0% | 9.7% | 85.7% | 76.5% | 19.7% | 0.0% | 35.7% | 50.0% | 7.1% | 0.0% | 50.0% |
| litellm-presidio | 11.1% | 78.0% | 100.0% | 37.1% | 100.0% | 100.0% | 69.7% | 72.2% | 81.2% | 50.0% | 28.6% | 100.0% | 12.5% |
| llm-guard | 33.3% | 93.2% | 100.0% | 40.3% | 85.7% | 76.5% | 88.2% | 11.1% | 88.4% | 100.0% | 0.0% | 100.0% | 75.0% |

## Methodology and caveats

Scoring is substring based: a PII item is caught when it no longer appears in the solution's output. Two recall columns are reported. **Recall** is span-level: a multi-token span (e.g. a full name or a street address) counts as caught when its exact full string disappears, so a partial redaction is credited as a full catch and this is an upper bound. **Recall (strict)** is token-level: every >=4-char token of the span must be removed. The truth is between the two; the gap (~1-5pp, roughly uniform across solutions) does not change the ranking.

**Ground truth.** The labels are produced by 10 independent auditor agents and cross-checked against AI4Privacy's own sensitive mask (loose substring overlap, so these are an upper bound). The agents independently recovered 93.1% of the dataset's 554 sensitive spans, and 93.4% of the agent labels overlap the dataset mask (the rest are incidental PII the dataset did not tag). The labels are independent of any solution under test: if they were a product's own detections, that product would score 100% recall, and none does.

**Competitors.** The two competitor rows are faithful integrations of the real tools, configured at their genuine best, not strawmen:

- `litellm-presidio` reproduces LiteLLM's built-in Presidio guardrail. LiteLLM POSTs message text to a Presidio analyzer/anonymizer service; we run the same Presidio engine in-process (per-document language, full default entity set, default threshold, `<ENTITY_TYPE>` replacement). Its request path scrubs message string content AND multimodal text parts (so its multimodal leak is low), but it never sends tool-call arguments to Presidio, so those leak. Its detection therefore equals Presidio; the differentiator is architectural. In production its latency also includes an HTTP sidecar hop, not counted here.
- `llm-guard` is protectai/llm-guard's Anonymize scanner (its own DeBERTa AI4Privacy v2 model + Presidio + regex), run in an isolated environment and cached (`scripts/build_llm_guard_cache.py`). Its language is set to `en` (its supported language), but the DeBERTa model is multilingual and the regex recognizers are language-agnostic, so it detects well across all four corpus languages (see the per-language table) and actually out-recalls the Presidio guardrail on flat text. Its blind spot is structural, not linguistic: as a flat-string scanner with no message/multimodal/tool-call awareness, everything structured leaks.

**How each competitor is configured** (so "you crippled it" cannot be argued):

| Row | Engine | Language | Entities | Threshold | Structured handling |
|---|---|---|---|---|---|
| litellm-presidio | Presidio analyzer + anonymizer | per-doc (en/fr/de/it) | all default recognizers | Presidio default | message text + multimodal text parts; tool-call JSON NOT parsed |
| llm-guard | DeBERTa AI4Privacy v2 + Presidio + regex | en (its supported language) | its default list expanded with DATE_TIME/LOCATION/URL | 0.5 (its default) | flat string only |
| privaite-onnx | Presidio + ONNX privacy-filter | per-doc | full (onnx preset) | 0.4 | message text + multimodal + tool-call JSON leaf values |

The structured columns are an ARCHITECTURAL claim about how each component handles a request payload, scoped to that component, not a verdict on those products overall. The `_structured_payload` probe is identical for every solution: the same document text placed in a multimodal text part and a `save_record` tool-call argument.

**Contamination and home-field advantage.** The corpus is AI4Privacy pii-masking-200k, and the pii-masking 65k/200k/300k/400k releases are one synthetic series (shared pipeline, subjects, taxonomy), so any model fine-tuned on any of them has train/test overlap here. Of the scored rows, exactly one does: the `llm-guard` row uses `Isotonic/deberta-v3-base_finetuned_ai4privacy_v2`, fine-tuned directly on pii-masking-200k, so its recall is an optimistic upper bound, not generalization. PrivAiTe's own default, `openai/privacy-filter` (behind `privaite-onnx`), is by contrast independent: OpenAI's model card (§7.2.1) states it did not train on the PII-Masking training data and only evaluated on the held-out test split, so `privaite-onnx`'s recall is a genuine number (with the honest asterisk that it is open-weights not open-data, and its label taxonomy is format-aligned to AI4Privacy). Net effect: the reported gap in PrivAiTe's favor is conservative, not flattered by contamination. An out-of-distribution cross-check on a non-AI4Privacy corpus (Gretel finance PII) confirms it: `privaite-onnx` holds ~84% recall off-distribution while the AI4Privacy-fine-tuned model drops to ~62% (see `OOD_COMPARISON.md`).

**Latency** is hardware-dependent and not reproducible run-to-run (ONNX in particular varies with CoreML/CPU warmup); treat it as indicative, not exact. `llm-guard` is marked `(offline)`: it cannot share this environment, so its number is its real inference latency measured in the isolated venv, not a live in-process call; `litellm-presidio` is in-process here but adds an HTTP sidecar hop in production.

**Dataset & licensing.** The document text comes from the open [AI4Privacy `pii-masking-200k`](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) dataset on Hugging Face. That dataset declares no explicit license, so this repo does NOT redistribute its raw text: only our derived per-document labels are committed, and `solutions/ai4privacy_loader.py` fetches the source text on demand to reproduce the corpus. See the bench README's 'Data sources transparency' section for the other (public) sources.

Reproduce: `python solutions/ai4privacy_loader.py && python -m solutions.compare`. The `llm-guard` row additionally needs its isolated-venv cache (`scripts/build_llm_guard_cache.py`, which stores raw corpus text and is not committed); without that cache the runner omits the row.
