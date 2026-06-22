# Changelog

## 2026-06-22

### Added
- `bench_structured.py`: parity and round-trip benchmark for structured inputs.
  It re-embeds the existing PII corpus into tool-call and multimodal shapes and
  checks that detection matches the flat baseline (0 regressions) and that
  tool-call arguments de-anonymize losslessly. Includes a `--selftest` mode that
  validates the harness with a fake detector, no spaCy models required.
- `datasets/structured_samples.json`: 5 hand-written function-call and multimodal
  samples (13 PII).
- `PRIVAITE_PATH` environment override so the suite can target any PrivAiTe
  checkout instead of the hardcoded `../PrivAiTe` sibling.

### Results
- Structured carriers reach full parity with flat text on the 64-document corpus
  (263/279 anonymized, 0 regressions across all three carriers, tool-call
  round-trip 64/64). See the README for the breakdown.
