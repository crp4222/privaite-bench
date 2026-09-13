# Privy protocol traces and Kiji ONNX

Measured on 2026-09-13: six configurations attempted, 6 completed, with identical inputs, CPU inference, and the detection cache disabled. Results are from local engine requests, not a live agent/provider session. [Method and reproduction](scripts/privy/README.md).

## Decision

Keep the current `onnx` default. Kiji is a benchmark-only candidate: replacing Privacy Filter with this ONNX artifact trades away too much detection and precision on protocol traces. Kiji alone also leaves nine of ten planted credential occurrences intact in the long log. The deterministic rules retained in the Presidio combinations cover those specific formats, but do not restore broad password coverage: `onnx` catches 12/15 Privy passwords, `kiji-presidio` 2/15.

The default also has substantial misses: 233/491 Privy spans are not fully covered. Its three missed passwords occur in SQL traces. Prioritize a separate SQL/field-context evaluation with exact value boundaries; do not tune on this test sample and report the tuned score as independent evidence. Adding entity categories or passing a planted-secret replay is insufficient by itself.

## Runtime failure retained

The first `onnx-kiji` attempt was terminated after more than ten minutes without completion. Its last document-progress marker was 400/420; a regression temporary directory was active. A one-second native stack sample showed Privacy Filter's `QMoECPU` inside nested ONNX Runtime thread-pool parallel sections, spinning in `EndParallelSectionInternal`. This is evidence of a local runtime stall, not an established root cause or a network timeout. The failed attempt has no completed scores and is never averaged into latency.

An identical fresh-process retry ended with status **complete**, after 326.7 seconds, under a 600-second watchdog. The `onnx-kiji` table entries come from that retry; one successful retry does not establish runtime reliability.

[Initial failure record](results/kiji_privy/onnx-kiji-initial-failure.json) · [Retry status](results/kiji_privy/onnx-kiji-retry-status.json). Isolate model/session lifecycle and thread-pool settings before considering the combined configuration for production. This experiment did not change PrivAiTe's runtime settings.

## Adding Kiji: measurable gain, unresolved tradeoffs

The combined stack fully covers 283/491 Privy spans versus 258/491 for `onnx`. On AI4Privacy, full-span recall rises from 74.67% to 84.28%. That is a useful experimental signal, but Privy password coverage remains 12/15 and non-gold character removal rises from 8.46% to 14.84%. More clean documents change, and the long log takes longer. Keep this option in the benchmark until runtime reliability and unwanted redaction are addressed and independently re-evaluated.

## Privy: 300 traces, 491 annotated spans

A full span is caught only when all its non-whitespace characters are removed. Character precision is relative to the supplied annotations. Non-gold removal counts other characters removed, including useful syntax; lower is better.
These traces are short: 29–2,156 bytes, median 201 bytes, p95 695 bytes. The long-log replay below measures a separate input scale.

| Configuration | Full-span recall | Character recall | Character precision | Non-gold removal | Mean / p95 |
|---|---:|---:|---:|---:|---:|
| `light` | 31.77% | 45.76% | 50.50% | 3.25% | 29.03 / 41.49 ms |
| `onnx` | 52.55% | 63.33% | 35.15% | 8.46% | 384.61 / 501.87 ms |
| `kiji` | 13.65% | 28.94% | 17.65% | 9.78% | 19.38 / 52.39 ms |
| `kiji-argmax` | 14.87% | 32.29% | 18.32% | 10.42% | 19.10 / 53.52 ms |
| `kiji-presidio` | 29.94% | 48.16% | 24.66% | 10.65% | 28.09 / 54.56 ms |
| `onnx-kiji` | 57.64% | 72.63% | 26.16% | 14.84% | 406.91 / 553.34 ms |

## AI4Privacy cross-check: 120 documents

The first two recall columns use the original comparison's literal and strict-token criteria. They are different from full character coverage, reported separately.

| Configuration | Literal recall | Strict-token recall | Full-span recall | Mean / p95 |
|---|---:|---:|---:|---:|
| `light` | 62.66% | 58.08% | 51.97% | 51.98 / 68.70 ms |
| `onnx` | 84.93% | 81.00% | 74.67% | 382.24 / 545.24 ms |
| `kiji` | 70.96% | 59.61% | 48.25% | 12.04 / 17.91 ms |
| `kiji-argmax` | 77.73% | 68.12% | 50.87% | 11.77 / 16.13 ms |
| `kiji-presidio` | 77.51% | 70.52% | 61.14% | 42.75 / 58.77 ms |
| `onnx-kiji` | 95.63% | 92.36% | 84.28% | 417.23 / 567.09 ms |

## False-positive controls and memory

The 78 Privy documents here contain no **annotated** PII. Incidental PII may be unannotated; these counts should not be interpreted as independently audited precision. The separate 14-document clean corpus is unchanged.

| Configuration | Privy empty-gold docs changed / 78 | Clean docs changed / 14 | Clean spans removed | Peak RSS |
|---|---:|---:|---:|---:|
| `light` | 30 | 2 | 3 | 2240.34 MiB |
| `onnx` | 38 | 1 | 2 | 4821.16 MiB |
| `kiji` | 72 | 5 | 5 | 338.58 MiB |
| `kiji-argmax` | 73 | 5 | 9 | 338.06 MiB |
| `kiji-presidio` | 74 | 5 | 5 | 2485.56 MiB |
| `onnx-kiji` | 76 | 5 | 6 | 4960.89 MiB |

## Long-log regression

Same synthetic 69,068-byte log, two planted credentials with five occurrences each. Timings are medians of three runs. EN-only Presidio configuration; do not compare the absolute time with the earlier EN/FR replay as a speedup.

| Configuration | Median | Surviving occurrences / 10, each run | Any eight-character fragment survives | Any full secret reversible |
|---|---:|---:|---|---|
| `light` | 4.016 s | 0, 0, 0 | no | no |
| `onnx` | 24.141 s | 0, 0, 0 | no | no |
| `kiji` | 3.731 s | 9, 9, 9 | yes | no |
| `kiji-argmax` | 3.746 s | 9, 9, 9 | yes | no |
| `kiji-presidio` | 4.411 s | 0, 0, 0 | no | no |
| `onnx-kiji` | 29.146 s | 0, 0, 0 | no | no |

The seven-line excerpt and connection-URI results, all individual timings and initialization costs are in the [numeric reports](results/kiji_privy/). All 126 completed structured probes matched their corresponding flat-text output for both multimodal text and parsed tool-call JSON.

## Privy recall by type

| Type | Gold spans | `onnx` | `kiji-presidio` | `onnx-kiji` |
|---|---:|---:|---:|---:|
| AGE | 5 | 0.00% | 20.00% | 20.00% |
| COORDINATE | 22 | 31.82% | 0.00% | 31.82% |
| CREDIT_CARD | 7 | 100.00% | 71.43% | 100.00% |
| DATE_TIME | 46 | 60.87% | 45.65% | 60.87% |
| EMAIL_ADDRESS | 6 | 100.00% | 100.00% | 100.00% |
| FINANCIAL | 19 | 42.11% | 57.89% | 63.16% |
| IBAN_CODE | 9 | 100.00% | 100.00% | 100.00% |
| IMEI | 9 | 88.89% | 11.11% | 88.89% |
| IP_ADDRESS | 7 | 100.00% | 100.00% | 100.00% |
| LOCATION | 93 | 30.11% | 11.83% | 37.63% |
| MAC_ADDRESS | 6 | 66.67% | 0.00% | 66.67% |
| NRP | 35 | 2.86% | 2.86% | 5.71% |
| ORGANIZATION | 9 | 22.22% | 33.33% | 44.44% |
| PASSWORD | 15 | 80.00% | 13.33% | 80.00% |
| PERSON | 77 | 79.22% | 37.66% | 84.42% |
| PHONE_NUMBER | 7 | 100.00% | 100.00% | 100.00% |
| TITLE | 41 | 29.27% | 0.00% | 29.27% |
| URL | 18 | 11.11% | 22.22% | 33.33% |
| US_BANK_NUMBER | 7 | 100.00% | 28.57% | 100.00% |
| US_DRIVER_LICENSE | 12 | 91.67% | 50.00% | 91.67% |
| US_ITIN | 7 | 71.43% | 42.86% | 100.00% |
| US_LICENSE_PLATE | 12 | 58.33% | 16.67% | 58.33% |
| US_PASSPORT | 12 | 75.00% | 50.00% | 75.00% |
| US_SSN | 10 | 100.00% | 100.00% | 100.00% |

## Privy recall by format

SQL groups are inferred from quote style. The XML group keeps the source's bytes-string representation intact.

| Format | Documents | Gold spans | `onnx` | `kiji-presidio` | `onnx-kiji` |
|---|---:|---:|---:|---:|---:|
| sql_backticks | 60 | 51 | 13.73% | 19.61% | 19.61% |
| html | 60 | 126 | 59.52% | 25.40% | 61.11% |
| sql_other | 60 | 58 | 37.93% | 41.38% | 53.45% |
| xml | 60 | 155 | 56.77% | 21.29% | 58.06% |
| json | 60 | 101 | 65.35% | 47.52% | 74.26% |

## AI4Privacy recall by language

Full-span recall, with 30 documents per language. Italian is outside the six languages listed by the Kiji ONNX card; it stays in the cross-check because it is part of PrivAiTe's existing corpus. These small samples do not establish general multilingual quality.

| Language | Gold spans | `onnx` | `kiji` | `kiji-presidio` | `onnx-kiji` |
|---|---:|---:|---:|---:|---:|
| de | 112 | 75.89% | 50.00% | 61.61% | 83.93% |
| en | 118 | 66.95% | 45.76% | 59.32% | 82.20% |
| fr | 112 | 75.89% | 45.54% | 58.93% | 83.04% |
| it | 116 | 80.17% | 51.72% | 64.66% | 87.93% |

## Provenance and limits

- PrivAiTe source: `7a66b372eda2dd9f4d02b839ed58de84899ddce0` (unreleased detector fixes on 0.4.3).
- Benchmark source: `d9239d5ee5fde9c6b1c7bb9f65b238669c48c0de`. Each result also stores hashes of its scoring code.
- Hardware: Apple M1 Pro, 16 GiB RAM. Separate sequential processes; initialization excluded from per-request latency, included in peak RSS.
- [Privy dataset and selection manifest](results/privy_manifest.json): fixed revision, fixed seed, 60 documents per format, no detector-dependent selection.
- [Kiji adapter and artifact caveats](scripts/privy/README.md#candidate-definitions): published ONNX artifact only, 512/64 windows, supplied BIO labels, CPU, no remote code or coreference mapping.
- Small denominators matter: the Privy sample contains only 15 password spans. These results do not establish broad secret-scanning guarantees.
- We did not train models or tune thresholds on this evaluation sample; overlap with the models' original training data is not established. No claim of statistical significance is made from the single corpus pass and three long-log repetitions.
- These are detection, redaction and local-latency measurements. They do not test network timeouts, concurrent production traffic or a live agent's answers.
