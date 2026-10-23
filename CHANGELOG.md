# Changelog

## 2026-10-23

The suite is rebuilt as one benchmark (`pbench/`) that scores every system on the
same public corpora. The earlier runners, their self-written datasets and the
reports built on them (COMPARISON, OOD_COMPARISON, KIJI_PRIVY, ONNX_VARIANTS) are
removed; they remain in history up to a56a91c.

### Added
- 8 public corpora selected with one seed from pinned dataset revisions: TAB,
  MAPA and PIILO (real text, human labels), OpenPII, Nemotron-PII, Gretel EN,
  Gretel finance and Privy (synthetic). No document is written or labeled here,
  and none that was used while PrivAiTe was developed is selected
  (`pbench/dev_exposure.json`).
- 22 systems, each in its own pinned environment and worker process: PrivAiTe
  (onnx, light, max), LiteLLM's Presidio guardrail and content filter through the
  real proxy, NeMo Guardrails through `LLMRails`, Kiji Privacy Proxy, agentgateway
  and the occludra gateway as their own processes, LangChain `PIIMiddleware`,
  DataFog and its LiteLLM guardrail, Presidio, Guardrails AI GuardrailsPII,
  LLM Guard, scrubadub, OpenMed, openai/privacy-filter, GLiNER multi PII,
  NVIDIA GLiNER-PII, GLiNER2-PII, Piiranha and Kiji.
- A structured suite placing the same records in user messages, multimodal
  parts, system prompts, tool-call arguments (flat and nested) and tool results,
  with a request integrity check (roles, tool calls and ids kept).
- Label-agnostic character scoring, document-level bootstrap intervals, paired
  comparison with the leader, contamination marks, and a real-text-only view.
- A "failed closed" outcome for requests a product answers with its own error
  without calling the provider (NeMo Guardrails on multimodal content): no leak,
  no service, shown next to blocks; places a system cannot serve sort as leaked.
- METHODOLOGY.md, RESULTS.md (generated), MIT license.

### Fixed before the first published run
- Presidio rows load the NER block of Presidio's shipped `conf/default.yaml`;
  the multi-language model list used before dropped it and returned spaCy
  organizations as PII.
- Piiranha windows are cut to the 256-token context its card states.
- occludra's per-key limit of 10 requests a second is raised; it had turned
  fast sequential requests into 429s.
- F2 is 0, not undefined, when a system has neither precision nor recall.

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
