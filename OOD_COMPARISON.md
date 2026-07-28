# Out-of-distribution generalization cross-check

Same detectors as `COMPARISON.md`, scored on a NON-AI4Privacy corpus: 120 documents (502 core-PII spans, de, en, it) from [Gretel `synthetic_pii_finance_multilingual`](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual), which ships exact char-level PII spans and is independent of both AI4Privacy and the GLiNER training data. This measures which detectors actually generalize versus which look good only on the distribution they were trained on.

Two independent cross-checks live here: the Gretel detector-stack comparison directly below, and a second dataset (nvidia/Nemotron-PII), scored value-based through the shipped presets, at the [end of this file](#second-dataset-nvidia-nemotron-pii-shipped-presets-value-based). Two sources and two scoring methods reaching the same conclusion is the point.

## Result

| Detector / combination | Recall (core PII) | Precision | de | en | it |
|---|---|---|---|---|---|
| presidio | 66.7% | 50.6% | 56.5% | 73.9% | 69.3% |
| openai/privacy-filter | 73.7% | 75.6% | 69.6% | 80.0% | 71.6% |
| ai4privacy-mdeberta (llm-guard) | 62.4% | 48.1% | 54.0% | 67.9% | 64.8% |
| gliner (independent) | 71.1% | 66.2% | 69.6% | 68.5% | 75.0% |
| PF + Presidio (PrivAiTe onnx stack) | 84.3% | 59.0% | 76.4% | 92.7% | 83.5% |
| PF + gliner | 83.9% | 70.1% | 80.1% | 91.5% | 80.1% |
| PF + Presidio + GLiNER (PrivAiTe max stack) | 88.6% | 61.3% | 82.0% | 95.2% | 88.6% |
| PF + presidio + mdeberta | 86.7% | 54.8% | 78.9% | 94.5% | 86.4% |
| all four | 89.8% | 57.3% | 83.2% | 97.0% | 89.2% |

## What it shows

- **The PrivAiTe onnx stack (`openai/privacy-filter` + Presidio) generalizes.** Its off-distribution recall here is close to its AI4Privacy recall in `COMPARISON.md`, because `openai/privacy-filter` was not trained on AI4Privacy. The score is not a home-field score.
- **The AI4Privacy fine-tune does not.** `ai4privacy-mdeberta` (the model behind the `llm-guard` row) drops sharply off-distribution, which is the empirical signature of train/test overlap on the main bench: its AI4Privacy recall is an optimistic upper bound, not generalization.
- **Combining an INDEPENDENT model helps, modestly and honestly.** Adding GLiNER (independent training data) raises recall a few points off-distribution, at a precision cost (more false positives). Adding the AI4Privacy fine-tune does not help off-distribution and hurts precision, so the large union gains seen on the AI4Privacy corpus were contamination, not a real ceiling.

## Methodology and caveats

Scoring is span-overlap: a ground-truth PII span counts as caught when any detected span overlaps it. **Recall** is over CORE PII only (company/date/time excluded: organization and generic temporals are treated inconsistently across taxonomies). **Precision** counts a detection as correct when it overlaps ANY labeled span (including company/date/time), so a detector is not penalised for correctly finding those. GLiNER is run with a fixed standard PII label set (not the Gretel taxonomy) at threshold 0.5; mDeBERTa at its defaults. Both transformer models truncate long inputs (~512 / ~384 tokens) while `openai/privacy-filter` does not; restricting to short documents leaves the ranking unchanged. The PrivAiTe rows here run Presidio with its full default recognizers; the shipped `onnx`/`max` presets pin Presidio to a 9-type allowlist (names/addresses/secrets come from the ML models), trading a little of this recall for higher precision, so read these as the detector stacks, not the exact preset configs. The corpus is synthetic finance-domain text, one OOD slice, not a universal benchmark; the raw Gretel text is not redistributed (only integer span offsets are committed).

Reproduce: `python -m scripts.ood.build_gretel_corpus`, then `-m scripts.ood.run_privaite_detectors` (PrivAiTe venv), `-m scripts.ood.run_transformer_models mdeberta` (llm-guard venv) and `... gliner` (gliner venv), then `-m scripts.ood.score_ood`.

---

## Second dataset: nvidia/Nemotron-PII (shipped presets, value-based)

A second fully independent corpus, scored a different way. [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) is NVIDIA's synthetic PII set, unrelated to AI4Privacy, to Gretel and to the GLiNER training data. Here the SHIPPED presets run end to end through the real PrivAiTe engine (`compare.evaluate` + `PrivAiTeSolution`, the same value-based recall as the headline in `COMPARISON.md`), not the raw detector stacks. So this table is directly comparable to the main-bench recall, and it re-tests the "onnx generalizes" claim on a second source AND a second scoring method.

292 documents, 1190 targeted PII values (English us-locale head of the train split).

| Preset (shipped) | Recall | Token-level recall | Caught |
|---|---|---|---|
| privaite-light (Presidio, 9-type allowlist) | 57.3% | 53.8% | 682/1190 |
| privaite-onnx (default: PF + Presidio) | 73.4% | 70.8% | 873/1190 |
| privaite-max (onnx + GLiNER) | 81.9% | 79.5% | 975/1190 |

### What it shows

- **The default `onnx` preset holds up on a second independent dataset.** 73.4% recall over a broad standard-PII set, no distribution collapse, on data from a different author than both its training set and AI4Privacy. Consistent with the Gretel result; the gap to the 84.9% AI4Privacy headline is a broader, harder type set (see caveats), not a home-field effect.
- **Independent GLiNER adds recall again (+8.5pp).** `max` reaches 81.9%: the same "add an independent model, gain a few points honestly" pattern the Gretel table shows, on a completely separate corpus.
- **Presidio-only (`light`) is the floor.** 57.3%: on out-of-distribution text the ML models are what carry recall.

### Methodology and caveats

Recall is value-based, identical to `compare.evaluate`: a labeled PII string counts as caught when it no longer appears verbatim in the anonymized output. It is measured over the PII types the presets target, mapped from Nemotron's labels: person (first/last/name), email, phone (phone + fax), url, location (street/city/state/country/county/postcode/coordinate), date and time, SSN, credit card, IP (v4/v6). Excluded, because no compared tool targets them by default and counting them would measure scope rather than detection: company, occupation, education and employment status, demographics (race, religion, gender, sexuality, political view, age, blood type, language), opaque secrets and domain identifiers (passwords, API keys, PINs, CVV, MAC and device IDs, medical, employee, customer and unique IDs, license, vehicle and certificate numbers, biometrics, tax IDs), and format-less financial account numbers (account, routing, SWIFT). The exact excluded labels and their counts are written to `results/nemotron_ood_report.json`. Two honest caveats: the train-split head is entirely us-locale English, so this is an English slice, not a multilingual result; and the type set is broader than the AI4Privacy corpus emphasises (it counts bare times, URLs, fax numbers and GPS coordinates, all hard), which is why onnx sits below its 84.9% headline here, so read 73.4% as a floor. Raw Nemotron text is not redistributed; only integer and label-derived stats are committed.

Reproduce: `python -m scripts.ood.build_nemotron_corpus`, then (from the repo root) `python -m solutions.nemotron_crosscheck --preset onnx --preset light` (PrivAiTe venv) and `python -m solutions.nemotron_crosscheck --preset max` (gliner venv).
