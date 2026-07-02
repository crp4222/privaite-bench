# Out-of-distribution generalization cross-check

Same detectors as `COMPARISON.md`, scored on a NON-AI4Privacy corpus: 120 documents (502 core-PII spans, de, en, it) from [Gretel `synthetic_pii_finance_multilingual`](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual), which ships exact char-level PII spans and is independent of both AI4Privacy and the GLiNER training data. This measures which detectors actually generalize versus which look good only on the distribution they were trained on.

## Result

| Detector / combination | Recall (core PII) | Precision | de | en | it |
|---|---|---|---|---|---|
| presidio | 66.7% | 50.6% | 56.5% | 73.9% | 69.3% |
| openai/privacy-filter | 73.7% | 75.6% | 69.6% | 80.0% | 71.6% |
| ai4privacy-mdeberta (llm-guard) | 62.4% | 48.1% | 54.0% | 67.9% | 64.8% |
| gliner (independent) | 71.1% | 66.2% | 69.6% | 68.5% | 75.0% |
| privaite-onnx = PF + presidio | 84.3% | 59.0% | 76.4% | 92.7% | 83.5% |
| PF + gliner | 83.9% | 70.1% | 80.1% | 91.5% | 80.1% |
| PF + presidio + gliner | 88.6% | 61.3% | 82.0% | 95.2% | 88.6% |
| PF + presidio + mdeberta | 86.7% | 54.8% | 78.9% | 94.5% | 86.4% |
| all four | 89.8% | 57.3% | 83.2% | 97.0% | 89.2% |

## What it shows

- **`privaite-onnx` generalizes.** Its off-distribution recall here is close to its AI4Privacy recall in `COMPARISON.md`, because its default model `openai/privacy-filter` was not trained on AI4Privacy. The score is not a home-field score.
- **The AI4Privacy fine-tune does not.** `ai4privacy-mdeberta` (the model behind the `llm-guard` row) drops sharply off-distribution, which is the empirical signature of train/test overlap on the main bench: its AI4Privacy recall is an optimistic upper bound, not generalization.
- **Combining an INDEPENDENT model helps, modestly and honestly.** Adding GLiNER (independent training data) raises recall a few points off-distribution, at a precision cost (more false positives). Adding the AI4Privacy fine-tune does not help off-distribution and hurts precision, so the large union gains seen on the AI4Privacy corpus were contamination, not a real ceiling.

## Methodology and caveats

Scoring is span-overlap: a ground-truth PII span counts as caught when any detected span overlaps it. **Recall** is over CORE PII only (company/date/time excluded: organization and generic temporals are treated inconsistently across taxonomies). **Precision** counts a detection as correct when it overlaps ANY labeled span (including company/date/time), so a detector is not penalised for correctly finding those. GLiNER is run with a fixed standard PII label set (not the Gretel taxonomy) at threshold 0.5; mDeBERTa at its defaults. Both transformer models truncate long inputs (~512 / ~384 tokens) while `openai/privacy-filter` does not; restricting to short documents leaves the ranking unchanged. The corpus is synthetic finance-domain text, one OOD slice, not a universal benchmark; the raw Gretel text is not redistributed (only integer span offsets are committed).

Reproduce: `python scripts/ood/build_gretel_corpus.py`, then `run_privaite_detectors.py` (PrivAiTe venv), `run_transformer_models.py mdeberta` (llm-guard venv) and `... gliner` (gliner venv), then `score_ood.py`.
