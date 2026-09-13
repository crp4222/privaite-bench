# Privy and Kiji evaluation

This experiment adds a protocol-trace corpus and a **benchmark-only** Kiji
adapter. It does not install a new PrivAiTe preset or change the default model.
Detection and anonymization run locally; there is no LLM-provider request.

## Reproduce

Use PrivAiTe's Python environment (or this repository's pinned dependencies and
the EN/FR/DE/IT spaCy models). No `datasets`, PyTorch, PyArrow, or remote Python
loader is needed.

```bash
# Downloads the public ZIP; reads test-large.json directly without extraction.
python -m scripts.privy.build_corpus

# Download only the model/tokenizer data needed by the adapter, once.
python -c 'import asyncio; from solutions.kiji import KijiDetector; asyncio.run(KijiDetector(offline=False).initialize())'

# Each row gets its own process; run sequentially to compare latency and peak RSS.
for solution in light onnx kiji kiji-argmax kiji-presidio onnx-kiji; do
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -m scripts.privy.evaluate --solution "$solution"
done
```

The generated corpus is gitignored. The committed selection manifest contains
the pinned dataset revision, archive/corpus hashes, row IDs, counts and seed.
Per-solution reports contain only numeric measurements, labels, IDs and software
provenance; no original or anonymized document text is redistributed.

## Corpus and sampling

[Privy](https://huggingface.co/datasets/beki/privy) generates synthetic protocol
traces from API schemas. This evaluation reads the publisher's **test-large**
split at revision `dc137a6a976f6b5bb8768e9bb51ec58df930ccd1` and checks every
gold offset against its annotated value. Invalid annotations are excluded and
counted; none were found in the initial traversal.

A fixed-seed reservoir selects 60 distinct templates per format: JSON, HTML,
XML, SQL with backticks, and other SQL. The last two categories are inferred
from syntax, not ground-truth database labels. XML bytes representations are
kept exactly as supplied. All 120,574 test rows are visited; selection uses
neither detector outputs nor preferred entity types. Empty-gold documents stay
in the sample. This produces 300 documents, 491 annotated spans across 24
types, and 78 documents with no annotated sensitive span.

The same run also uses the existing 120-document AI4Privacy corpus and 14 clean
documents, plus the generated log/URI regression fixtures. This is an external
corpus check, not proof that either model has never encountered related data
during training. Privy's card declares MIT in its metadata and loading script;
the prose licensing section is incomplete. Raw data is not vendored here.

## Candidate definitions

| Row | Detectors | Threshold |
|---|---|---|
| `light` | Current full Presidio preset, including PrivAiTe rules | Current defaults |
| `onnx` | Current Presidio allowlist + Privacy Filter | Current defaults |
| `kiji` | Kiji ONNX alone | Mean span confidence ≥ 0.5 |
| `kiji-argmax` | Kiji ONNX alone, publisher-style argmax | No confidence cutoff |
| `kiji-presidio` | Same Presidio configuration as `onnx`, Kiji replaces Privacy Filter | Kiji ≥ 0.5 |
| `onnx-kiji` | Current `onnx` stack plus Kiji | Kiji ≥ 0.5 |

These thresholds were specified before measuring the evaluation corpus. This
is not threshold tuning or a trained ensemble. All rows use the same PrivAiTe
engine, `SECRET: redact`, `CREDIT_CARD: mask`, reversible placeholders for other
types, and the current overlap policy. Presidio uses the document language plus
English fallback; the English log/URI replay uses EN only. The detection cache
is off. Current request-local Privacy Filter window reuse remains enabled.

Kiji uses its published
[ONNX artifact](https://huggingface.co/DataikuNLP/kiji-pii-model-onnx)
at revision `a0c9dd2ff9023b63378f67bafa50e33925663219`, on CPU. The adapter
reads the supplied 53 BIO labels, validates the graph output contract, uses
512-token windows with 64-token overlap, and stitches predictions before span
decoding. No tail of a long tool output is discarded. The coreference head is
unused; no cross-request identity mapping is added.

The ONNX card describes DistilBERT, whereas the
[source-model card](https://huggingface.co/DataikuNLP/kiji-pii-model) describes
DeBERTa with a CRF. The pinned ONNX artifact actually loads with the supplied
DistilBERT tokenizer (28,996 vocabulary entries) and exposes 53 PII outputs and
7 coreference outputs. Its actual SHA-256 also differs from the bundled
`model_manifest.json`; the adapter checks the observed hash of the pinned
artifact and reports hashes for every file used. Results apply to this artifact,
not to the separately described DeBERTa/CRF model.

## Metrics and limitations

- Full-span recall requires **all non-whitespace characters** of a gold span
  to be removed. A partially hidden password is a miss.
- Character recall and precision expose partial coverage and over-redaction.
  Non-gold removal measures how much other text, including syntax, was removed.
  These metrics are relative to dataset labels; unannotated sensitive data may
  make the apparent false-positive rate pessimistic.
- Literal and strict-token recall preserve the original comparison's criteria.
  They are reported separately from full character coverage.
- Per-type, per-format and per-language results retain their denominators.
  Coverage is type-agnostic; taxonomy differences do not get counted as leaks
  when all characters were removed. Reversibility is checked separately on the
  planted-secret replay.
- Every twentieth document also traverses multimodal text and parsed tool-call
  JSON; outputs must match the flat-text scrub. This tests payload handling,
  not the intelligence of a coding agent.
- Latencies cover actual engine scrubbing, with one neutral warm-up per engine.
  Initialization is recorded separately. Corpus inputs run once; the long-log
  and URI cases run three times. Workers run sequentially in separate processes,
  and peak RSS includes Python, models and benchmark input data.

The benchmark does not establish production concurrency, network timeout
behavior, malicious-agent containment, or universal detection. Run a local
agent session separately before claiming end-to-end agent performance.
