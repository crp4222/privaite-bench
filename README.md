# privaite-bench

An open, reproducible benchmark of PII redaction for LLM traffic. 25 systems
(proxies, gateways, guardrails, agent middleware, libraries and standalone
models) on 8
public corpora, three of them real human-written documents labeled by humans,
plus the places agents actually put personal data: tool-call arguments, tool
results, system prompts and multimodal parts.

Maintained by the author of [PrivAiTe](https://github.com/crp4222/PrivAiTe),
which is one of the systems measured: every choice that could favor it is listed
in [METHODOLOGY.md](METHODOLOGY.md#conflict-of-interest), and everything needed to
rerun and audit the numbers is in this repo.

## Leaderboard: text

Macro average over the 8 corpora (each corpus weighs the same). Ranked by the
share of **direct identifiers** (names, emails, phone numbers, ID, card and
account numbers, secrets, addresses...) whose every character was masked, with
its 95% bootstrap interval. `=` marks a system statistically tied with #1.

<!-- HEADLINE:START -->
| # | System | Kind | Direct identifiers fully masked | Real text only | Char recall | Precision | F2 | Latency p50 |
|---:|---|---|---|---|---|---|---|---|
| 1 | [PrivAiTe 0.8.0 (minilm, opt-in)](https://github.com/crp4222/PrivAiTe) | request integration | 90.6% <sub>[89%, 92%]</sub> | 83.3% <sub>[81%, 86%]</sub> | 90.6% | 86.0% | 89.8% | 37 ms |
| 2 | [PrivAiTe MiniLM (model alone, ONNX)](https://huggingface.co/crp4222/privaite-pii-minilm) | model | 90.4% <sub>[89%, 92%]</sub> | 83.2% <sub>[81%, 86%]</sub> | 89.8% | 91.2% | 90.1% | 34 ms |
| 3 | [PrivAiTe 0.8.0 (ultrafast, opt-in)](https://github.com/crp4222/PrivAiTe) | request integration | 90.3% <sub>[89%, 91%]</sub> | 83.0% <sub>[80%, 85%]</sub> | 89.9% | 89.4% | 89.8% | 10 ms |
| 4 | [PrivAiTe 0.7.2 (max)](https://github.com/crp4222/PrivAiTe) | request integration | 87.5% <sub>[86%, 89%]</sub> | 80.7% <sub>[78%, 83%]</sub> | 74.2% | 65.9% | 70.2% | 335 ms |
| 5 | [NVIDIA GLiNER-PII](https://huggingface.co/nvidia/gliner-PII) | model | 80.2% <sub>[79%, 82%]</sub> | 83.3% <sub>[81%, 86%]</sub> | 79.5% | 71.9% | 76.1% | 413 ms |
| 6 | [GLiNER multi PII v1](https://huggingface.co/urchade/gliner_multi_pii-v1) | model | 78.0% <sub>[76%, 80%]</sub> | 79.0% <sub>[76%, 81%]</sub> | 77.8% | 63.7% | 72.1% | 185 ms |
| 7 | [GLiNER2 PII (fastino)](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) | model | 75.5% <sub>[74%, 77%]</sub> | 67.9% <sub>[65%, 71%]</sub> | 67.2% | 78.7% | 67.0% | 218 ms |
| 8 | [PrivAiTe 0.7.2 (onnx, default)](https://github.com/crp4222/PrivAiTe) | request integration | 72.3% <sub>[70%, 75%]</sub> | 56.3% <sub>[51%, 61%]</sub> | 67.0% | 75.3% | 66.3% | 121 ms |
| 9 | [openai/privacy-filter (model alone)](https://huggingface.co/openai/privacy-filter) | model | 66.6% <sub>[64%, 69%]</sub> | 47.7% <sub>[43%, 52%]</sub> | 56.9% | 78.5% | 56.7% | 135 ms |
| 10 | [Microsoft Presidio 2.2.364](https://github.com/microsoft/presidio) | library | 61.6% <sub>[60%, 63%]</sub> | 71.3% <sub>[68%, 74%]</sub> | 61.5% | 70.4% | 59.5% | 28 ms |
| 11 | [Guardrails AI GuardrailsPII 1.0.8](https://github.com/guardrails-ai/guardrails_pii) | library | 60.6% <sub>[59%, 63%]</sub> | 47.3% <sub>[43%, 52%]</sub> | 52.5% | 68.2% | 50.8% | 137 ms |
| 12 | [OpenMed 3.0 (privacy-filter-multilingual)](https://huggingface.co/OpenMed/privacy-filter-multilingual) | library | 54.7% <sub>[53%, 56%]</sub> | 58.1% <sub>[55%, 61%]</sub> | 61.6% | 77.4% | 61.0% | 3610 ms |
| 13 | [Piiranha v1](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information) | model | 52.9% <sub>[51%, 55%]</sub> | 48.6% <sub>[45%, 53%]</sub> | 39.3% | 86.1% | 42.2% | 113 ms |
| 14 | [occludra/AISG gateway (e948e23)](https://github.com/occludra/gateway) | request integration | 50.8% <sub>[49%, 53%]</sub> | 64.5% <sub>[61%, 68%]</sub> | 36.7% | 59.0% | 32.9% | 32 ms |
| 15 | [Kiji Privacy Proxy 1.6.4 (Dataiku)](https://github.com/dataiku/kiji-proxy) | request integration | 49.6% <sub>[48%, 51%]</sub> | 30.2% <sub>[27%, 34%]</sub> | 43.5% | 82.3% | 47.4% | 351 ms |
| 16 | [PrivAiTe 0.7.2 (light)](https://github.com/crp4222/PrivAiTe) | request integration | 47.7% <sub>[46%, 50%]</sub> | 48.3% <sub>[44%, 53%]</sub> | 61.3% | 63.1% | 60.4% | 30 ms |
| 17 | [LLM Guard 0.3.16 Anonymize](https://github.com/protectai/llm-guard) | library | 43.3% <sub>[41%, 46%]</sub> | 38.5% <sub>[33%, 44%]</sub> | 33.0% | 60.6% | 30.4% | 145 ms |
| 18 | [Kiji PII (Dataiku, ONNX)](https://huggingface.co/DataikuNLP/kiji-pii-model-onnx) | model | 32.9% <sub>[31%, 35%]</sub> | 27.6% <sub>[24%, 31%]</sub> | 40.9% | 68.6% | 40.0% | 37 ms |
| 19 | [DataFog 4.9 (+ its LiteLLM guardrail)](https://github.com/DataFog/datafog-python) | request integration | 15.7% <sub>[15%, 17%]</sub> | 0.7% <sub>[0%, 1%]</sub> | 15.7% | 87.9% | 20.3% | 0.2 ms |
| 20 | [scrubadub 2.0.1](https://github.com/LeapBeyond/scrubadub) | library | 9.4% <sub>[8%, 10%]</sub> | 2.6% <sub>[1%, 4%]</sub> | 12.7% | 65.9% | 13.5% | 0.3 ms |
| 21 | [LangChain 1.4 PIIMiddleware](https://docs.langchain.com/oss/python/langchain/middleware) | request integration | 7.0% <sub>[6%, 8%]</sub> | 2.5% <sub>[1%, 4%]</sub> | 11.5% | 72.4% | 12.0% | 0.1 ms |
<!-- HEADLINE:END -->

**Char recall**: share of direct and quasi identifier characters masked.
**Precision**: share of masked characters that belong to a labeled entity, on the
7 exhaustively labeled corpora. **F2**: weights recall four times precision.
**Latency**: median per document, CPU only (Apple M1 Pro).
Per-corpus, per-type and per-language tables: [RESULTS.md](RESULTS.md).

## Agent traffic: where the record sits in the request

Share of direct identifiers still present in what reaches the model provider,
when the same public record sits in each part of an OpenAI-style request (lower
is better). **Requests kept intact** guards against the cheap way to score well:
a gateway that drops tool calls leaks nothing from them but breaks the agent.
Such a cell reads "(breaks it)", and "fails closed" when the system answered
every request there with its own error; rows sort those places as fully leaked.
Only systems that ship a request-level integration can be scored here;
detection libraries and models have no notion of a request.

<!-- STRUCTURED:START -->
| System | User message | Multimodal part | System prompt | Tool-call arguments | Nested arguments | Tool result | Requests kept intact | Blocked | Failed closed |
|---|---|---|---|---|---|---|---|---|---|
| [PrivAiTe 0.8.0 (minilm, opt-in)](https://github.com/crp4222/PrivAiTe) | 2.1% | 2.1% | 1.9% | 2.5% | 2.2% | 2.1% | 100.0% | 0 | 0 |
| [PrivAiTe 0.8.0 (ultrafast, opt-in)](https://github.com/crp4222/PrivAiTe) | 2.2% | 2.2% | 2.1% | 2.8% | 2.4% | 2.2% | 100.0% | 0 | 0 |
| [PrivAiTe 0.7.2 (max)](https://github.com/crp4222/PrivAiTe) | 5.6% | 5.6% | 4.8% | 6.1% | 5.3% | 5.6% | 100.0% | 0 | 0 |
| [PrivAiTe 0.7.2 (onnx, default)](https://github.com/crp4222/PrivAiTe) | 10.3% | 10.3% | 9.2% | 10.7% | 10.3% | 10.3% | 100.0% | 0 | 0 |
| [PrivAiTe 0.7.2 (light)](https://github.com/crp4222/PrivAiTe) | 38.5% | 38.5% | 38.8% | 38.2% | 37.0% | 38.5% | 100.0% | 0 | 0 |
| [LiteLLM 1.104 Presidio guardrail](https://docs.litellm.ai/docs/proxy/guardrails/pii_masking_v2) | 27.0% | 27.0% | 26.7% | 100.0% | 99.9% | 27.0% | 100.0% | 0 | 0 |
| [Kiji Privacy Proxy 1.6.4 (Dataiku)](https://github.com/dataiku/kiji-proxy) | 22.3% | 100.0% | 22.4% | 100.0% | 99.9% | 22.3% | 100.0% | 0 | 0 |
| [LiteLLM 1.104 content filter](https://docs.litellm.ai/docs/proxy/guardrails/litellm_content_filter) | 64.6% | 64.6% | 64.6% | 64.6% | 64.5% | 64.6% | 100.0% | 0 | 0 |
| [DataFog 4.9 (+ its LiteLLM guardrail)](https://github.com/DataFog/datafog-python) | 68.9% | 68.9% | 68.9% | 100.0% | 99.9% | 68.9% | 100.0% | 0 | 0 |
| [agentgateway 1.6.0 (prompt guard)](https://github.com/agentgateway/agentgateway) | 63.9% | 63.9% | 63.9% | 100.0% | 99.9% | 100.0% | 100.0% | 0 | 0 |
| [occludra/AISG gateway (e948e23)](https://github.com/occludra/gateway) | 48.9% | 48.9% (breaks it) | 48.6% | 0.0% (breaks it) | 0.0% (breaks it) | 48.9% (breaks it) | 36.1% | 0 | 0 |
| [NeMo Guardrails 0.24 (sensitive data)](https://github.com/NVIDIA/NeMo-Guardrails) | 40.4% | fails closed | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% | 0 | 212 |
| [LangChain 1.4 PIIMiddleware](https://docs.langchain.com/oss/python/langchain/middleware) | 89.2% | 89.2% | 100.0% | 100.0% | 99.9% | 89.2% | 100.0% | 0 | 0 |
<!-- STRUCTURED:END -->

## What stands out

PrivAiTe's own model, [privaite-pii-minilm](https://huggingface.co/crp4222/privaite-pii-minilm),
masks the most direct identifiers, alone (90.4%) and in PrivAiTe 0.8.0's opt-in
`minilm` and `ultrafast` presets (90.6% and 90.3%), ahead of PrivAiTe's max
preset and NVIDIA GLiNER-PII, at 10 to 37 ms per document. Read those rows with
their contamination: the author of this benchmark trained that model on the
train splits of five of the eight corpora (their test splits are scored here,
with every scored document removed from the training data). On the three it
never saw (TAB, PIILO, Privy), the two presets mask 82.9% and 82.7% of direct
identifiers against 72.7% for PrivAiTe's default preset; the model card adds
test sets from other sources. PrivAiTe's default preset sits behind three
GLiNER models.
Pattern-based layers (DataFog, LangChain's middleware, scrubadub) mask under 16%
of direct identifiers, and almost nothing in real text.

In agent traffic, most request integrations never look at tool calls: LiteLLM's
Presidio guardrail, DataFog, agentgateway, LangChain, NeMo Guardrails and Kiji
forward tool-call arguments untouched. occludra leaks none of them because it
drops the tool calls, which breaks the agent. NeMo Guardrails answers every
multimodal request with an internal error. Only PrivAiTe and LiteLLM's content
filter give every part of the request the same protection.

## What is measured

- **Corpora**: [TAB](https://huggingface.co/datasets/ildpil/text-anonymization-benchmark)
  (European Court of Human Rights judgments), [MAPA](https://huggingface.co/datasets/joelniklaus/mapa)
  (EU court decisions in 7 languages), [PIILO](https://www.kaggle.com/datasets/lburleigh/piilo-dataset)
  (student essays): real text, human labels. [OpenPII](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m),
  [Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII), [Gretel EN](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1),
  [Gretel finance](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual),
  [Privy](https://huggingface.co/datasets/beki/privy): synthetic, from four
  independent generators. 1,607 documents, 2.84M characters, en fr de es it pt nl.
  Nothing was written or labeled for this benchmark.
- **Systems** run at their documented defaults, each in its own pinned
  environment, with two changes for all of them: the document's language is
  passed in, and inference runs on CPU. Request integrations get every surface
  they offer switched on. Proxies run for real (LiteLLM behind `litellm
  --config`, Kiji, agentgateway and occludra as their own processes, NeMo through
  `LLMRails`), in front of a fake provider that records what it receives.
- **Scoring** is label-agnostic and character-based: a name half masked is a
  leak. Intervals resample documents, not spans. Models trained on a corpus's
  generator family are marked in RESULTS.md.

Full definitions, corpus handling, configuration of every system and known
limits: [METHODOLOGY.md](METHODOLOGY.md).

## Reproduce

```sh
envs/build.sh harness privaite presidio litellm llmguard langchain scrubadub \
  datafog nemo models openmed guardrails occludra
envs/tools.sh                                     # agentgateway, occludra and Kiji, pinned
.venvs/harness/bin/python -m pbench.sources      # download pinned corpora, select documents
.venvs/harness/bin/python -m pbench.structured   # build the agent-traffic requests
.venvs/harness/bin/python -m pbench.run          # every system, one at a time (hours on CPU)
.venvs/harness/bin/python -m pbench.report       # results/summary.json, RESULTS.md, the tables above
.venvs/harness/bin/python -m pytest tests
```

PIILO is distributed through Kaggle only: download it with
`kaggle datasets download lburleigh/piilo-dataset` into
`~/.cache/privaite-bench/raw/piilo/` first. `results/runs/` holds every
system's raw outcome as offsets and booleans (no corpus text), so
`pbench.report` reproduces every number without rerunning a model.

To add a system: one module in `pbench/systems/` (see its `__init__.py` for the
four methods), one entry in `pbench/registry.py`, one `envs/<name>.txt`.

## Other studies in this repo

[`agent_workflow/`](agent_workflow/) runs real coding agents (Claude Code, Codex)
over a repository holding planted secrets, with and without the PrivAiTe gateway,
and counts what reaches the provider on the wire.

## License

Code: MIT. The corpora keep their own licenses (MIT, CC BY 4.0, Apache 2.0) and
are downloaded from their publishers, never redistributed here.
