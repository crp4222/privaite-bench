# Structured-secret regression replay

Measured locally on 2026-09-13, comparing PrivAiTe 0.4.3 source (`8d4ee3c`)
with the unreleased [structured-secret and overlap-policy changes](https://github.com/crp4222/PrivAiTe/commit/7a66b372eda2dd9f4d02b839ed58de84899ddce0). This is an
offline engine replay, not a new Claude Code or OpenCode session. No request
is sent to a model provider.

The original [live-session results](RESULTS_BIG.md) remain historical evidence.
This replay targets the two log fields behind those misses and a synthetic
connection URI password. It is a regression fixture, not an independent
generalization benchmark.

## Results

Each cell shows surviving **full credential occurrences per request**. Results
were identical across three runs. The full log contains five occurrences of
each of two credentials; the excerpt contains one of each.

| Input | Bytes | `light` before → after | `onnx` before → after |
|---|---:|---:|---:|
| Seven-line log excerpt, with preceding log context | 1,006 | 2/2 → 0/2 | 1/2 → 0/2 |
| Full ingest log | 69,068 | 10/10 → 0/10 | 9/10 → 0/10 |
| Connection URI | 78 | 1/1 → 0/1 | 1/1 → 0/1 |

After the change, no contiguous eight-character fragment of these credentials
survived in any output, and no full credential was present in the reversible
mapping. This says nothing about shorter fragments or other, untested secrets.

Median processing time over three runs, with the detection cache disabled:

| Input | `light` before → after | `onnx` before → after |
|---|---:|---:|
| Log excerpt | 0.1063 → 0.1048 s | 0.5646 → 0.5451 s |
| Full ingest log | 7.1639 → 7.1948 s | 24.7641 → 24.6672 s |
| Connection URI | 0.0351 → 0.0350 s | 0.2925 → 0.2918 s |

The added coverage did not produce a material latency change in this run.
The large log still takes about 25 seconds with `onnx`; this is not a speedup
claim. Three repeated runs on one host are insufficient to establish a small
performance difference.

## Setup and reproduction

- Apple M1 Pro, 16 GiB RAM; CPU inference.
- `light` and `onnx` presets, default EN/FR language configuration.
- `SECRET: redact`, `CREDIT_CARD: mask`; other types use placeholders.
- Detection cache disabled. Request-local ONNX window deduplication enabled.
- One neutral warm-up request after engine initialization; initialization is
  excluded from the reported timings. Three repetitions per input and preset.
- Baseline and modified versions run sequentially using the same Python
  environment and cached model weights.

From the benchmark repository, using a Python environment with PrivAiTe's
dependencies and language models installed:

```bash
PRIVAITE_PATH=/path/to/baseline-checkout \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m agent_workflow.bench_structured_secrets \
  --source-label baseline --output /tmp/secret-baseline.json

PRIVAITE_PATH=/path/to/modified-checkout \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m agent_workflow.bench_structured_secrets \
  --source-label modified --output /tmp/secret-modified.json
```

The runner generates the existing big fixture in a temporary directory. It
removes that directory at exit and writes only counts and timings, never the
generated credential values. The
[recorded numeric results](../results/structured_secrets_20260913.json) contain
every repetition.

## What this does not validate

The new rules recognize explicit assignment names, URI userinfo and plaintext
bearer headers. They cannot guarantee detection of arbitrary field names,
encoded values, incomplete input or bare JSON values without field context.
NLP still runs on the entire input. No regex hit causes an early return.

The overlap-policy fix is covered separately by PrivAiTe's regression tests:
blocked types outrank irreversible types, which outrank reversible types, with
the existing confidence/span rules breaking ties. The whole detected union
remains covered, so useful surrounding text can still be removed. This replay
does not measure all boundary errors, agent answer quality, network timeouts
or alternative detectors such as Kiji.
