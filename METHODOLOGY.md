# Methodology

This document says exactly what the numbers in [README.md](README.md) and
[RESULTS.md](RESULTS.md) mean, how they were produced, and where they could be
wrong. Everything described here is implemented in `pbench/`; file references
point to the code that does it.

## The question

When a person's data goes into an LLM request, how much of it does each privacy
tool keep from reaching the model provider? Two parts:

1. **Detection**: given a document, which characters does the tool hide?
   Every tool that can take text is scored on the same documents.
2. **Coverage of the request**: agents and chat clients put data in more places
   than the user message: tool-call arguments, tool results, system prompts,
   multimodal parts. A perfect detector that never looks at tool calls still
   leaks everything a tool returns. Only tools that ship a request-level
   integration (a proxy, a guardrail, an agent middleware) can be scored here.

## Corpora

### Selection

No document was written or labeled for this benchmark. Eight public datasets
are used as published, with their own labels (`pbench/sources.py`):

| Corpus | Text | Labels | License | Selection |
|---|---|---|---|---|
| TAB, test split | real ECHR judgments | human, two-level (identifier type + masking decision) | MIT | all 127 judgments, one annotator each (below) |
| MAPA EUR-Lex, test split | real EU court decisions and acts | human (MAPA project) | CC BY 4.0 | every test document in en fr de es it pt nl, cut into chunks of at most 4,000 characters at sentence boundaries (190 chunks) |
| PIILO | real student essays, PII replaced by surrogates by the publisher | human | CC BY 4.0 | 200 of the 945 essays that contain at least one label |
| Ai4Privacy OpenPII 1M, validation | synthetic | generated with the text | CC BY 4.0 | 30 per language (en fr de es it pt nl), 200 to 4,000 characters |
| NVIDIA Nemotron-PII, test | synthetic | generated with the text | CC BY 4.0 | 50 per cell of locale (us, intl) x format (structured, unstructured), at most 6,000 characters |
| Gretel PII masking EN v1, test | synthetic | generated with the text | Apache 2.0 | 200 |
| Gretel synthetic PII finance multilingual, test | synthetic | generated with the text | Apache 2.0 | 30 per language (en fr de es it nl), at most 6,000 characters |
| Privy, test-large | synthetic API payloads | generated with the payload | MIT | 60 per format (JSON, XML, HTML, two SQL styles), one per template |

Every file is downloaded at a pinned dataset commit and checked against a
pinned sha256 before use. Sampling uses one fixed seed (20261023) over the whole
eligible split (never a prefix: OpenPII's file is sorted by language, Privy's by
template). `results/corpora.json` lists every selected document's id and the
sha256 of its text, so the selection can be verified without the text.

The three real corpora answer the "is it all synthetic" objection; the five
synthetic ones come from four independent generators (Ai4Privacy, NVIDIA,
Gretel, Privy), so no single generator's style decides the ranking.

### Source-specific handling

- **TAB**: several law students annotated some judgments. Each judgment is
  scored once, against the quality-checked annotator when there is one (else the
  first by id), so a judgment annotated ten times does not count ten times. TAB's
  identifier type decides the kind: DIRECT is direct, QUASI is quasi, NO_MASK
  (annotators judged it safe to keep) is neutral.
- **MAPA**: the source keeps tokens but no whitespace, so tokens are joined with
  one space ("( 1 )"). Detectors see slightly unnatural spacing around
  punctuation; every system sees the same text. MAPA's ADDRESS class is almost
  always a country or city in these texts, so it is quasi. Its coarse PERSON
  spans include honorifics ("Herr Gerard Cotter"); the tokens its fine layer
  tags TITLE are split off as neutral spans (see honorifics below). MAPA's PERSON
  class also covers parties named in case citations ("Simmenthal", "Costanzo")
  and the Advocate General: that is the project's anonymization policy for
  court texts, kept as published.
- **PIILO**: offsets are rebuilt from tokens and their trailing-whitespace flags,
  and the rebuild must reproduce the dataset's `full_text` exactly or the build
  stops. PIILO labels **student** names only: cited authors and teachers are left
  unlabeled on purpose, so masking them is correct but would look like a false
  positive. PIILO is therefore scored on recall only.
- **Honorifics, all corpora**: TAB writes "Mr Ruslan Kurkaev" as one PERSON span
  (225 of its 240 direct names), Gretel and Privy sometimes "Dr. Kaitlin Wong",
  the other corpora label the honorific apart or not at all. A leading honorific
  (Mr, Mrs, Dr, Herr, Mme, Sig., Sr., Dhr. and their equivalents) is split off
  every PERSON span into a neutral span, so a tool that masks the name and keeps
  "Mr" has protected the person in every corpus alike.
- **Gretel EN** ships labeled strings without offsets: every occurrence of each
  string is gold; a labeled string absent from its text is dropped and counted.
- **Gretel finance**: 8 of 5,594 test rows have spans past the end of their text;
  they are excluded before sampling rather than clipped.
- **Privy and Gretel finance documents already used while PrivAiTe was
  developed** are excluded, see [Conflict of interest](#conflict-of-interest).

### Taxonomy

Each source label is mapped (tables in `pbench/sources.py`, an unmapped label
stops the build) to:

- a **kind**. *Direct* identifiers identify a person on their own: names,
  usernames, emails, phone and fax numbers, street addresses, government IDs,
  financial account and card numbers, secrets (passwords, PINs, CVVs, API keys,
  cookies), network and device identifiers, record numbers (patient, customer,
  employee, policy, plate, VIN), dates of birth. *Quasi* identifiers identify in
  combination: other dates, city/region/country/postcode, employer, age,
  demographic attributes. *Neutral* spans are labeled by the source but are not
  personal data under this policy (honorifics, amounts, TAB's NO_MASK).
- a **type** among 15, used only for breakdowns.

Scoring never looks at the type a system assigns. A privacy proxy is judged on
what it hides, and type vocabularies differ between every pair of tools.

## Systems and configuration rules

**Which systems.** Every open-source tool found that ships PII redaction for
LLM traffic or PII detection as a library or model, runs on a local machine, and
had at least 25 GitHub stars (or, for a model, a public release by its
publisher) when the benchmark was built: proxies and gateways (PrivAiTe, Kiji
Privacy Proxy, LiteLLM, agentgateway, occludra/AISG), guardrail frameworks (NeMo
Guardrails, Guardrails AI, LLM Guard), agent middleware (LangChain), libraries
(Presidio, DataFog, scrubadub, OpenMed) and models (openai/privacy-filter, GLiNER
multi PII, NVIDIA GLiNER-PII, GLiNER2-PII, Piiranha, Kiji). No row uses a
general-purpose LLM as the detector: in this setup the LLM is the provider the
redaction layer protects data from. What was left out, and why, is listed at
the end of RESULTS.md.

The registry is `pbench/registry.py`; each adapter is one module in
`pbench/systems/` whose docstring states its configuration. Rules applied to
every system:

1. **Documented defaults.** Model, threshold, entity list and decoding are the
   ones the product or the model card ships, unless rule 2 or 3 says otherwise,
   and every change is recorded in the system's `info.json`.
2. **Language.** Each system receives the document's language when it accepts
   one, and only that language (PrivAiTe and Presidio run one NER pass per
   configured language; giving one of them English as a second language would
   give it a second pass the others do not get). LLM Guard accepts only "en" and
   "zh" and NeMo Guardrails builds its analyzer for English only, so they run in
   English: that is their behavior.
3. **Request integrations get every surface they offer.** LangChain's
   middleware is enabled on input, output and tool results (the default is input
   only). DataFog's guardrail gets every type its regex engine supports (the
   default is four) and `fail_policy: closed`. LiteLLM's content filter gets every
   prebuilt identifier pattern. NeMo gets its input and output masking rails. A
   surface the product does not offer is not added: writing that code would
   measure us, not the product.
4. **CPU only**, one system at a time, after an untimed warm-up call in every
   language. `OMP_NUM_THREADS=8`, and torch-based adapters set 8 intra-op
   threads; libraries with their own thread pools (onnxruntime, spaCy) keep
   their defaults.
5. **Pinned environments.** Each system runs in its own virtualenv built from
   `envs/<name>.txt`, where the system's own package is pinned to a release at
   least two weeks old at the time of the run; its dependencies are resolved by
   uv at build time and recorded exactly in `envs/<name>.lock`. Model weights are
   pinned by revision wherever the library takes one (the OpenMed SDK does not:
   the revision it loaded is recorded in its `info.json`).
6. **Real request paths.** LiteLLM is run as the actual proxy (`litellm
   --config`) with its Presidio guardrail calling a Presidio HTTP service, and
   NeMo Guardrails through `LLMRails` in passthrough mode. Both forward to a local
   fake provider that records the request body: what it records is what a
   provider would receive. PrivAiTe is run through `PIIEngine.process_request`,
   the function its proxy and gateway call. LangChain's middleware is replayed the
   way an agent loop runs it: `before_model` when a human or tool message
   arrives, `after_model` when the model answers.
7. **Standalone models** are run through the library their card documents
   (`gliner`, `gliner2`, Transformers `pipeline`, the OpenMed SDK). Their
   libraries truncate long inputs silently, so they get the minimal windowing an
   integrator would add (1,000 characters, 150 overlap, cut on whitespace).
   Products keep their own long-input handling, whatever it is, because that is
   part of what they are. A model whose card states a shorter context gets it:
   Piiranha's card gives 256 tokens, which a 1,000-character window often
   exceeds (three windows in four on MAPA), so the pipeline's own stride
   chunking cuts each window into 256-token pieces (tested).
8. **Both GLiNER rows get the same plain-English label list** (25 labels, one per
   kind of identifier in the taxonomy) so neither is favored by a vocabulary it
   was trained on. GLiNER2-PII uses the 42 labels its own card lists.

Per-system notes that go beyond the rules:

- **Kiji Privacy Proxy** runs from Dataiku's signed macOS release (1.6.4; 1.6.7
  ships no macOS build), headless, with a temporary HOME and database and its
  transparent MITM proxy off, so neither the machine's proxy settings nor any
  Kiji data are touched. Its detection API returns UTF-8 byte offsets; they are
  converted to character offsets (tested), without which every accented document
  would be scored wrong.
- **occludra/AISG** pins `httpx~=0.28` next to `litellm~=1.56`, which requires
  `httpx<0.28`: its published requirements do not resolve. httpx is relaxed to
  0.27, the only change.
- **Guardrails AI** exports OpenTelemetry traces to its hosted endpoint unless
  told otherwise; metrics are switched off in-process. Its validator is installed
  from its Apache-2.0 source at a pinned commit because the Hub requires an
  account; the code is the same.
- **Presidio** (the library row and the HTTP service behind LiteLLM's guardrail)
  loads the NER block of its shipped `conf/default.yaml`, which
  ignores spaCy organizations and numeric labels, with one spaCy model per
  language. Presidio's multi-language example passes a bare model list instead;
  that drops the block and brings organizations back as PII (tested in
  `tests/test_adapters.py`). NeMo Guardrails builds its own English analyzer from
  the same file.
- **No telemetry**: every adapter sets the opt-out switches its library offers
  (`HF_HUB_DISABLE_TELEMETRY`, `DATAFOG_NO_TELEMETRY`, `DO_NOT_TRACK`, Guardrails
  metrics off; Kiji, LiteLLM and DataFog are opt-in already).

## Metrics

Implemented in `pbench/score.py`, tested in `tests/test_score.py`.

- A character is **masked** when it lies inside any span the system returned.
  **Whitespace is ignored** everywhere: masking "Jane" and "Doe" separately
  protects "Jane Doe".
- A gold span is **protected** when every one of its characters is masked. Half a
  name, the last four digits of a card or the domain of an email still leak.
- **Direct identifiers fully masked** (headline) = protected direct spans /
  direct spans.
- **Char recall** = masked gold characters / gold characters, direct and quasi.
- **Precision** = masked characters inside any labeled span (neutral included,
  so masking an honorific is never a false positive) / masked characters. Only on
  the seven corpora whose labels are exhaustive. A system that masks nothing has
  precision 1 by convention so that it is ranked by its recall.
- **F2** = 5PR / (4P + R): missing PII costs more than over-masking.
- **Macro average**: each corpus weighs the same, whatever its size, so the 2,000
  spans of OpenPII do not drown the 320 of PIILO. "Real text only" is the same
  average over TAB, MAPA and PIILO.
- **Errors fail open.** A document a system raised on, crashed on or spent more
  than 15 minutes on counts as nothing masked; such a request counts as every
  value leaked. A privacy layer that fails open leaks. A system is ranked only
  when every document and request has an outcome.

### Uncertainty

Spans inside one document are not independent (same writer, same template), so
intervals resample **documents**, with replacement, 2,000 times, within each
corpus; the 95% interval is the 2.5th to 97.5th percentile. The macro interval
resamples each corpus independently and averages. "Tied with #1" compares each
system with the leader on the **same** resamples (paired bootstrap) and holds
when the interval of the difference contains zero.

## Structured suite

`pbench/structured.py`. 212 documents are drawn from the short, record-shaped
corpora (Nemotron structured and free text, Gretel EN, Gretel finance, OpenPII,
Privy JSON payloads), and each is placed verbatim in six request shapes: a user
message, a text part next to an image, a system prompt, the JSON arguments of an
assistant tool call (Privy's JSON payloads are passed as the arguments as-is),
the same record split into nested objects and arrays, and a `role: tool` result.
The wrapper strings around the record contain no personal data.

A gold value **leaks** when it still appears in any string of the request the
system forwards, tool-call arguments JSON-decoded first (so `é` escaping
cannot hide a leak). Values shorter than 4 characters are not scored: "12" or
"Al" match by accident in any request. A request a system blocks by policy (a PII or
prompt-injection block it answers with an explicit refusal) leaks nothing and is
counted in its own column. So is a request it **fails closed** on: the product
answers the client with its own documented error reply and calls no provider
(NeMo Guardrails 0.24.1 does this on every multimodal request, its masking action
raising on the list of parts). Any other refusal or missing provider call is a
failure, scored as every value leaked.

Leaks alone would reward breaking the request: a gateway that drops the
assistant's `tool_calls` leaks none of their arguments, and the provider then
rejects the conversation or the agent loses its call. Two integrity measures sit
next to the leak rates. **Requests kept intact**: the forwarded request has the
original's shape (same messages and roles in order, same tool calls with the
same ids and names, tool results still tied to their call, same multimodal
parts). **Tool JSON still valid**: every tool-call `arguments` string still
parses after redaction.

The test `test_every_carrier_carries_every_gold_value_before_any_system_runs`
pins that every value is present in every request before any system touches it.

## Latency

Wall time inside the worker process around the system's call only (no process
start, model load or pipe overhead), one document at a time, on one Apple M1 Pro
(16 GB), CPU only. Request integrations that run behind a real proxy include its
HTTP hops (LiteLLM: client to proxy, proxy to Presidio, proxy to the fake
provider). Latency depends on the machine; the ranking between systems is what
transfers, not the milliseconds.

## Contamination

A learned component evaluated on data from the generator family it was trained
on has a home-field advantage. From the model cards:

| Component | Trained on | Marked in-distribution on |
|---|---|---|
| LLM Guard default NER (Isotonic/deberta-v3-base_finetuned_ai4privacy_v2) | Ai4Privacy pii-masking-200k | OpenPII |
| Piiranha v1 | Ai4Privacy pii-masking-400k | OpenPII |
| OpenMed privacy-filter-multilingual | Ai4Privacy 200k, 400k, open-pii-masking-500k | OpenPII |
| NVIDIA GLiNER-PII | Nemotron-PII | Nemotron |
| openai/privacy-filter | not disclosed in its Hugging Face card | none |
| GLiNER multi PII v1 | urchade/synthetic-pii-ner-mistral-v1 | none |
| GLiNER2-PII | the publisher's own synthetic corpus | none |
| Kiji | DataikuNLP/kiji-pii-training-data (synthetic) | none |

RESULTS.md marks those cells with †. The real-text-only column contains no
in-distribution cell for any system.

## Conflict of interest

This benchmark is maintained by the author of PrivAiTe, which it measures.
What could favor PrivAiTe, and what was done about it:

- **PrivAiTe was developed against samples of AI4Privacy 200k, Gretel finance,
  Nemotron (train split) and Privy.** Those exact documents are excluded:
  `pbench/dev_exposure.json` lists the 300 Privy rows (and their templates) and
  the 120 Gretel finance rows used during development, and the selection skips
  them; Nemotron is drawn from the test split; OpenPII is a different release
  from the 200k set used. A text-prefix check found no selected document in any
  earlier evaluation set. What cannot be excluded is familiarity with those
  generators' styles: the **real text only** column (TAB, MAPA, PIILO, never used
  in PrivAiTe's development) is the view free of that.
- **Taxonomy and kinds** were decided by the same author. The mapping tables are
  in `pbench/sources.py`, applied identically to every system, and scoring is
  label-agnostic, so a system is never helped by sharing PrivAiTe's type names.
- **Two rows reuse PrivAiTe code.** The openai/privacy-filter row runs the model
  through PrivAiTe's ONNX detector: the q4 export, PrivAiTe's windowing and its
  BIOES span decoder, which is not the constrained Viterbi decoder the model card
  describes. The row isolates the model inside PrivAiTe's default preset; it is
  not the best possible openai/privacy-filter integration. The Kiji row decodes the published ONNX artifact with
  PrivAiTe's BIO decoder because the publisher ships no Python inference.
- **Configuration** of every competitor follows the rules above, and each
  adapter states its settings; reports of a better documented configuration for
  any system are welcome and will be rerun.

## Limits

- One machine, one run. Latency on other hardware will differ.
- Five of eight corpora are synthetic, and English dominates (TAB, PIILO,
  Nemotron, Gretel EN and Privy are English only).
- Quasi identifiers are context dependent; the kinds follow each dataset's own
  notion where it has one (TAB) and the taxonomy above otherwise.
- The structured suite checks exact survival of a gold string. A system that
  rewrites a value into a different but still identifying form (e.g. drops a
  space inside an IBAN) is credited.
- Substitution quality, reversibility (restoring values in responses) and
  streaming are not measured here.

## Reproduce and audit

See the README. `results/runs/<system>/` holds, per document, the masked spans as
offsets, the time, and any error, and per request which values leaked; no corpus
text. `pbench.report` recomputes every number from those files against the gold
spans and refuses to score a document whose text hash changed.
