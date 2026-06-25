# Comparative benchmark (multiple solutions)

`compare.py` scores several PII-anonymization solutions on the **same** corpus, so
the numbers are apples to apples. This is the multi-solution part of the suite:
adding another tool or repo is one subclass.

## Corpus

Real documents from the public [AI4Privacy pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
dataset (en, fr, de, it). The dataset declares no license, so its raw text is
**not** redistributed here: `ai4privacy_loader.py` downloads it on demand and
caches it locally, and this repo commits only the derived ground-truth labels
(`datasets/comparative_labels.json`, keyed by row id) plus the results.

Ground truth: 10 independent auditor agents labeled every piece of PII a privacy
proxy must hide. The labels were then cross-checked against the dataset's own
sensitive mask (the agents independently recovered ~93% of it, and ~93% of the
agent labels overlap it), so the ground truth is precise rather than asserted.

## Solutions compared

- **privaite-light**: PrivAiTe, Presidio-only preset. Fast, zero false positives, classic PII.
- **privaite-onnx**: PrivAiTe, full ONNX Privacy Filter preset (the default). Detects secrets and addresses too.
- **presidio-baseline**: vanilla Microsoft Presidio on message text. This is the engine behind LiteLLM's Presidio guardrail and most drop-in PII proxies; it does not look inside tool-call arguments or multimodal content.

## Metrics

- **Recall**: of the labeled PII, how much each solution removes (per language, per type).
- **False positives**: things anonymized in clean text.
- **Tool-call protection**: of the PII a solution catches in plain text, how much it *also* removes from a tool-call argument. This isolates structured handling from raw detection quality.
- **Tool-call leak / multimodal leak**: share of all PII that survives inside a tool-call argument or a multimodal text part.
- **Latency**.

## Results

See [../COMPARISON.md](../COMPARISON.md). Headline: `privaite-onnx` leads on recall
and removes PII from tool-call arguments (100% protection) where the flat-text
baseline leaks about 99%.

## Reproduce

```bash
python solutions/ai4privacy_loader.py   # download + sample the corpus
python -m solutions.compare             # run all solutions, write COMPARISON.md
```

Needs PrivAiTe importable next to this repo (`../PrivAiTe`) and the spaCy models
for en, fr, de, it.

## Add another solution (another repo)

Subclass `Solution` in `solutions.py` (implement `anonymize_text` and
`anonymize_payload`), add it to `all_solutions()`, and re-run. That is how a new
tool or repo enters the comparison.
