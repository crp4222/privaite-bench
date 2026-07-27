# Agent workflow leak benchmark

The other benchmarks in this repo grade PrivAiTe's detector on curated text.
This one measures the end-to-end questions that actually matter for gateway
mode: **when a real coding agent reads a repo containing secrets and PII, does
that data reach the model provider, does PrivAiTe stop it, and what does that
protection cost in latency?**

## How it works

- `fixtures/acme_support/` is a small fake helpdesk project: two docs and a
  README with person names, emails, phones and a postal address, a
  `customers.json` with IBANs, a credit card and an SSN, and a `.env` with
  fake secrets (API key, DB password, JWT, SMTP password).
- `gen_fixture.py` generates the `.env` at runtime from a fixed seed (so no
  realistic-looking secret is ever committed; the file is gitignored) and
  writes a ground-truth manifest of every planted value. It also verifies the
  static values still appear in the committed fixture files.
- `recorder.py` is a stdlib recording proxy placed between the client and the
  real provider: it records every forwarded request body (plus the response
  stream and its timing), relays the client's auth headers verbatim, and
  streams the response back, so it observes exactly what the provider receives
  without changing behavior.
- `run.py` drives each agent (Claude Code, Codex) through the matrix
  agent x arm x preset:
  - `direct`: agent -> recorder -> provider (baseline, what leaks today)
  - `privaite`: agent -> timing tap -> PrivAiTe gateway -> recorder ->
    provider, once per PII preset (`light`, `full`, `onnx` by default; the
    `onnx-auto` and `onnx-auto-cache` variants run the onnx preset with
    `device: auto`, without and with the opt-in detection cache)

  The timing tap is a second recorder instance in front of the gateway, so
  per-request gateway latency is the difference between the same request's
  timestamps on both sides of the gateway, measured rather than inferred.
  After each privaite cell the harness snapshots the gateway's `/stats`
  endpoint (per-entity-type detection counts; the gateway process is fresh per
  cell). Every captured body is then scanned for each ground-truth value and
  the leak counts, timing, context growth and cache analysis are written to
  [`RESULTS.md`](RESULTS.md) plus `results/agent_workflow/report.json`.

A value counts as leaked only if its exact planted string appears in a body
the provider received. This is a wire-level measurement, independent of what
the agent chose to display.

## Validity guard

A leak number for a privaite cell is only meaningful if the traffic provably
went through the gateway, so the harness enforces that in two places:

- Pre-flight probe: before launching the agent on a privaite cell, `run.py`
  sends one synthetic request (marked with an `X-Bench-Probe` header) through
  the full chain, agent entry point -> tap -> gateway -> recorder. The tap
  forwards it, the recorder absorbs it without contacting the real provider,
  and the harness requires the probe to show up in the gateway's own
  handled-request log and in the capture. If it does not, the agent is never
  launched and the cell is recorded as invalid with the reason. This proves
  the protocol route is genuinely served end to end (a 200 on `/health` does
  not).
- Post-cell check: the gateway logs one line per request it handled; the scan
  compares that count (probes excluded) against the request bodies the
  recorder captured behind it. Any shortfall, or a cell with no captured
  traffic at all, marks the cell invalid. Invalid cells render as "invalid,
  not measured" in `RESULTS.md` and carry `invalid`/`invalid_reason` in
  `report.json`; they never publish a leak count.

Both checks fail closed and are covered by `--selftest`.

## Run

```bash
cd privaite-bench
python3 agent_workflow/run.py                          # full matrix
python3 agent_workflow/run.py --agents claude          # subset
python3 agent_workflow/run.py --presets onnx           # one preset
python3 agent_workflow/run.py --rescan                 # re-analyze captures on
                                                       # disk, no live runs
python3 agent_workflow/run.py --selftest               # measurement math only
```

Requirements: the PrivAiTe checkout as a sibling (`../PrivAiTe`, override with
`PRIVAITE_PATH`) with its venv at `.venv`, and the `claude` and `codex` CLIs
logged in (both relay your own subscription OAuth through the local proxy, and
the fixture content goes to the real providers on the `direct` arm, which is
the point of the baseline).

Each cell has a per-agent timeout (default 600 s, override with
`AGENT_WORKFLOW_TIMEOUT` or `--timeout`). Hitting it is not an error: the cell
is recorded with `timed_out: true`, the partial capture is scanned and
reported, and the run continues with the next cell. Slowness is data here, not
a failure mode.

Raw captures land in `results/agent_workflow/` (gitignored: they contain the
generated fake secrets and full request bodies). The big-session variant
writes to `results/agent_workflow_big/` (also gitignored) via the env
overrides shown in [`RESULTS_BIG.md`](RESULTS_BIG.md).

## What the runs found

Two results documents carry the live tables, both from 2026-07-17 runs with
the real CLIs, the real providers, and the gateway from the local PrivAiTe
checkout; that gateway shipped in PrivAiTe 0.4.0:

- [`RESULTS.md`](RESULTS.md): the small-fixture matrix (5 files, ~3 KB).
- [`RESULTS_BIG.md`](RESULTS_BIG.md): a realistic big session (11 files,
  ~73 KB including a 69 KB ingest log, 20 to 40 provider round-trips). Build
  that fixture with [`gen_big_fixture.py`](gen_big_fixture.py); it
  regenerates byte-identically from fixed seeds.

Headline, stated carefully:

- Small fixture: Claude Code `direct` put 24/24 planted values on the wire
  and Codex `direct` 20/24; through the gateway with the onnx presets both
  agents drop to **0/24**, detection cache on and off, every cell
  validity-guarded.
- Big session: every gateway cell leaked **2/24**, the same two secrets in
  key=value log lines. Offline reproduction with the run's exact config shows
  the detector catches those values on their own, in `.env` assignment form
  and as an isolated log line, and misses them once roughly one preceding line
  of log-shaped context sits in front of them: a 7-line, ~1 KB excerpt of that
  log already reproduces the miss (the API key survives 5 of 5 occurrences
  there, the SMTP password 4 of 5), and 41-line windows leak 4 of 5 and 3 of 5.
  The effect is order dependent, since text appended after the line never
  triggers it, and it is a property of the detector rather than of the gateway,
  so every surface that runs the engine (the OpenAI-compatible proxy, the
  Open WebUI filter, the LiteLLM guardrail) is affected the same way. Not a
  routing bug. The honest agent-CLI claim is therefore "0/24 small, 2/24 big",
  never a blanket zero, and the 2 is a floor rather than a ceiling. Details and
  evidence in `RESULTS_BIG.md`.
- `device: auto` survived 300 KB+ bodies live (no process death, peak RSS
  roughly 3 to 4 GB), which is exactly what the `onnx-auto*` cells verify.

### The 2026-07-16 pre-fix Codex finding (kept for the record)

The first run (2026-07-16, pre-fix build) had Codex leak 20/24 THROUGH the
gateway. Claude Code speaks the Anthropic Messages API (`/v1/messages`): the
file contents it reads come back as `tool_result` blocks inside `messages`,
which the gateway traverses and scrubs, so everything was caught. Codex
speaks the Responses API (`/v1/responses`) and reads files through its own
custom shell tool, so the file contents arrive as `custom_tool_call_output`
items. The pre-fix Responses scrubber rewrote the `message` items but left
`custom_tool_call_output` untouched, so every raw name, email, phone,
address, IBAN, card and SSN that Codex had read flowed straight through. The
4 secrets did not reach the provider only because Codex read `.env` by
counting assignments and never emitted the raw values: agent self-censoring,
not a gateway catch.

PrivAiTe then gained a fix that scrubs tool-output carriers
(`custom_tool_call_output`, `function_call_output`) the same way it scrubs
`message` content, and the 2026-07-17 rerun validated it live: Codex via the
gateway now measures 0/24 on the small fixture. One caveat stands: the Codex
(Responses API) path has had fewer live hours against the gateway than the
Claude Code path; treat it as the less battle-tested of the two even though
its current numbers match.

## Timing: the cost of protection

Scrubbing an agent conversation is not free, and this harness reports the
cost instead of hiding it. Agents resend the whole growing conversation on
every turn, and without the detection cache the gateway rescans each request
statelessly, so per-turn scrub cost grows with context size (roughly O(n) per
turn, O(n^2) over a session). Measured at the timing tap (median per-request
scrub, `device: auto`):

| Session | Agent | Cache off | Cache on |
|---|---|---|---|
| small | claude | 4.91 s | 3.62 s |
| small | codex | 15.87 s | 1.04 s |
| big | claude | 15.22 s | 1.02 s |
| big | codex | 24.68 s | 0.98 s |

Without the cache the big session degrades to ~50 s per request by the time
bodies reach 330 KB; with the cache it stays around 1 s per request, and the
leak counts are identical either way. The practical guidance is to enable
the detection cache for agent sessions. RESULTS carries the per-turn tables
(body size vs gap) so the growth is visible, plus the provider cache
counters extracted from the recorded responses (`cache_read_input_tokens`,
`cached_tokens`), which keep climbing through the gateway: placeholder
rewriting does not break the provider's prompt cache.
