# Agent workflow leak benchmark: big session results

Companion to [`RESULTS.md`](RESULTS.md) (the small-fixture matrix). This run
answers the question the small fixture cannot: what happens on a **realistic
long agent session**, with a large log file in context (a single tool output
far above 4k detector tokens), 20 to 40 provider round-trips, and request
bodies growing past 300 KB. Same harness (`run.py`), same validity guards,
same 24 planted ground-truth values; only the fixture, prompt and timeout
differ.

All numbers below come from `results/agent_workflow_big/report.json`,
generated from live captures on 2026-07-17 (harness logs
`results/big_live_0717.log` and `results/big_live_0717_codex_finish.log`).
Real Claude Code and Codex CLIs, real providers, the PrivAiTe gateway from the
local sibling checkout (this gateway is not part of a released PrivAiTe
package yet; these numbers describe that local build). PrivAiTe presets under
test: `onnx-auto` (onnx preset, `device: auto`, detection cache off) and
`onnx-auto-cache` (same plus the opt-in detection cache). No cell in this run
failed the validity guard; every privaite leak count below is backed by a
gateway handled-request count that matches the captured request count
(24/24, 22/22, 33/33, 41/41).

## Headline, stated honestly

**Every privaite cell leaked 2 of the 24 planted values: the same two
secrets, in both agents, with the detection cache on and off.** The gateway
reduced the leak from 24/24 (claude direct) and 23/24 (codex direct) to 2/24,
and the 2 is a detection-recall gap at full-log scale, not a routing bug and
not a log-line problem (evidence below). The honest agent-CLI headline for a realistic session
is "2/24", not the small fixture's "0/24".

## Setup

- Fixture: `acme_support_big`, 11 files, 73,408 bytes: the standard
  `acme_support` files plus `docs/oncall.md`, `docs/runbook.md`, a 69,068-byte
  `logs/ingest_batch.log` (465 lines) and three `tickets/*.md`. Rebuild it
  byte-identically with `python3 agent_workflow/gen_big_fixture.py OUTDIR`
  (fixed seeds; verified to reproduce the exact fixture this run used).
- Ground truth: the same 24 planted values as the small fixture (4 secret,
  6 email, 6 person, 3 phone, 1 address, 2 iban, 1 credit_card, 1 ssn). The
  log repeats several of them, including two secrets in key=value log-line
  form, so the leak scan covers the same values in more carriers.
- Prompt: a 10-step audit (list files, read every doc, read `.env` variable
  names, read the full log with the file-reading tool, read every ticket,
  cross-reference ticket ids, 12-line summary). Full text in
  `results/agent_workflow_big/report.json`. Per-cell timeout 1800 s.
- Run command:

  ```bash
  python3 agent_workflow/gen_big_fixture.py /path/to/acme_support_big
  AGENT_WORKFLOW_FIXTURE=/path/to/acme_support_big \
  AGENT_WORKFLOW_OUT=results/agent_workflow_big \
  AGENT_WORKFLOW_RESULTS=results/agent_workflow_big/RESULTS.md \
  AGENT_WORKFLOW_TIMEOUT=1800 \
  python3 agent_workflow/run.py --presets onnx-auto,onnx-auto-cache
  ```

Measurement and validity guards are identical to the small run (see
[`README.md`](README.md)): wire-level substring scan of every captured
provider-bound body, pre-flight probe through the full chain before each
privaite cell, per-cell fresh gateway process, post-cell comparison of the
gateway's handled-request log against the recorder's captures, invalid cells
render as invalid and never publish a leak count.

## Values that reached the provider (leaked/planted)

| Agent | Arm | Preset | secret | email | person | phone | address | iban | credit_card | ssn | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude | direct | - | 4/4 | 6/6 | 6/6 | 3/3 | 1/1 | 2/2 | 1/1 | 1/1 | **24/24** |
| claude | privaite | onnx-auto | 2/4 | 0/6 | 0/6 | 0/3 | 0/1 | 0/2 | 0/1 | 0/1 | **2/24** |
| claude | privaite | onnx-auto-cache | 2/4 | 0/6 | 0/6 | 0/3 | 0/1 | 0/2 | 0/1 | 0/1 | **2/24** |
| codex | direct | - | 3/4 | 6/6 | 6/6 | 3/3 | 1/1 | 2/2 | 1/1 | 1/1 | **23/24** |
| codex | privaite | onnx-auto | 2/4 | 0/6 | 0/6 | 0/3 | 0/1 | 0/2 | 0/1 | 0/1 | **2/24** |
| codex | privaite | onnx-auto-cache | 2/4 | 0/6 | 0/6 | 0/3 | 0/1 | 0/2 | 0/1 | 0/1 | **2/24** |

The two leaked values are the secrets labeled `ACME_API_KEY` and
`SMTP_PASSWORD` (labels are the `.env` variable names; the generated values
are never printed anywhere in this repo). Codex direct leaked 23/24 rather
than 24/24 because it never emitted the JWT signature, agent behavior, not
protection.

## Why 2/24, verified in the captures

- Every leaked occurrence in all four privaite cells sits inside the log's
  `event=key_rotation_failed presented_key=... smtp_secret=...` lines. A
  redacted context scan of the captures confirms this: 100% of the
  occurrences of both values, in all four cells, are in that log-line form,
  and zero occurrences of the other two secrets appear anywhere.
- The same log lines arrive at the provider with `<DATE_TIME_n>` placeholders
  already substituted in them: the gateway traversed and scrubbed exactly
  these lines, and the detector flagged the timestamps but not the two
  key=value secrets. This rules out a routing/carrier bug (the class of bug
  the 2026-07-16 Codex fix addressed).
- The same two secret values in `.env` assignment form were scrubbed in every
  cell (and all 4 secrets are caught in the small-fixture run, 0/4 leaked).
- The miss reproduces offline, deterministically, with the engine loaded from
  this run's exact `gateway.yaml`: the raw `.env`, a single
  `key_rotation_failed` log line, and a 40-line log window are all fully
  scrubbed, but on the full 69,068-byte log the same two values survive the
  scrub. So the gap is detector recall at full-log scale (a single input far
  above the model's inference window, scanned in multiple windows), not the
  log-line shape per se, and not a gateway routing bug or a cache regression
  (identical 2/24 cache on and off, in both agents). The onnx preset's
  measured SECRET recall, 71.4% on the independent comparison corpus (see
  [`COMPARISON.md`](../COMPARISON.md)), already says secret detection is the
  weakest entity type; this is where that weakness lands in practice.

Treat this as the calibration for any claim about agent traffic: the gateway
removes what the detector detects, at the request level, reliably (validity
guarded); the residual risk is the detector's recall on unusual shapes.

## Timing

Exact per-request gateway latency from the timing tap (second recorder in
front of the gateway, same-clock difference per request):

| Agent | Preset | Median scrub (s) | Max scrub (s) | Median restore (s) |
|---|---|---|---|---|
| claude | onnx-auto | 15.22 | 53.58 | 0.0 |
| claude | onnx-auto-cache | 1.02 | 21.23 | 0.0 |
| codex | onnx-auto | 24.68 | 53.37 | 0.0 |
| codex | onnx-auto-cache | 0.98 | 14.59 | 0.0 |

Per-turn gap (time between consecutive provider-bound requests, bundling
provider generation, agent work and the scrub):

| Agent | Preset | Direct median gap (s) | PrivAiTe median gap (s) | Added per turn (s) |
|---|---|---|---|---|
| claude | onnx-auto | 3.7 | 26.8 | +23.1 |
| claude | onnx-auto-cache | 3.7 | 6.7 | +3.0 |
| codex | onnx-auto | 11.3 | 38.1 | +26.8 |
| codex | onnx-auto-cache | 11.3 | 14.0 | +2.7 |

Wire span (first to last captured provider-bound request of the cell,
including retries/restarts where they occurred):

| Agent | Arm/Preset | Requests captured | Wire span (s) | Last body (KB) |
|---|---|---|---|---|
| claude | direct | 42 | 290.9 | 322.8 |
| claude | privaite onnx-auto | 24 | 819.1 | 334.7 |
| claude | privaite onnx-auto-cache | 22 | 339.1 | 358.6 |
| codex | direct | 23 | 281.2 | 294.4 |
| codex | privaite onnx-auto | 33 (2 attempts) | 1374.4 | 190.0 |
| codex | privaite onnx-auto-cache | 41 (2 attempts) | 645.3 | 162.3 |

### The cache on/off story

Without the detection cache the gateway rescans the whole resent
conversation every turn, so per-request scrub cost grows with the context:
by the end of these sessions it reached the low 50s of seconds per request
(claude at ~330 KB bodies, codex at ~190 KB). That is unbounded in session
length. With the opt-in detection cache the median scrub drops to about 1 s
per request for both agents (1.02 s claude, 0.98 s codex), because the
unchanged prefix hits the cache (codex cell: 2,755 hits vs 116 misses).
Claude's end-to-end wire span dropped from 819.1 s to 339.1 s, codex's from
1374.4 s to 645.3 s.

**Guidance: enable the detection cache for agent sessions.** The leak counts
are identical cache on and off (2/24 in all four cells), so the cache costs
nothing in protection here while removing the quadratic scrub cost.

## Gateway process and `device: auto`

- No gateway process died in any cell (every privaite cell passed the
  handled-vs-captured check to the last request). This run exercised
  `device: auto` on 300 KB+ bodies, the exact scenario that used to SIGKILL
  the process when auto selected CoreML; on the fixed PrivAiTe build it is
  stable.
- Peak RSS, sampled from outside the process: 3,133.8 MB (codex onnx-auto)
  and 4,036.2 MB (codex onnx-auto-cache). RSS was not sampled for the two
  claude cells of this run (see caveats); the small-fixture run's four
  onnx-auto cells ranged 2,874.5 to 3,515.0 MB. Plan for roughly 3 to 4 GB.
- Detection cache counters (codex onnx-auto-cache cell): hits=2755,
  misses=116, entries=116. Counts only, the cache stores salted hashes and
  span metadata, never text.

## Session robustness

- Both codex privaite cells' first attempt ended with the Codex CLI exiting 1
  mid-session (after 495.3 s and 395.7 s); the harness's automatic second
  attempt completed both cells. The gateway itself stayed up and its
  handled-request count matches the captures, so the failures were on the
  client side of the chain. With the cache the completed attempt took 409.2 s
  end to end; without it, 1008.9 s.
- The claude captures contain client-side conversation restarts (one in
  direct, one in onnx-auto; none in onnx-auto-cache). They occur on the
  direct arm too, so they are agent behavior, not gateway-induced.

## Provider prompt cache

The providers' own cache counters, extracted from the captured response
streams, keep tracking the growing context through the gateway:
`cache_read_input_tokens` climbs to 142,132 by the last turn of the claude
onnx-auto-cache cell (and to 131,662 on onnx-auto), and `cached_tokens`
climbs to 74,496 before the codex onnx-auto-cache cell's client restart.
Placeholder rewriting is consistent enough between turns not to destroy the
provider-side cache. (Raw byte-prefix stability looks low on the claude arm,
14.6% and 31.2%, but it is equally low patterned on claude direct sections;
the divergence point is the `cache_control` breakpoint marker the client
moves between turns, so the token-level counters above are the authoritative
signal, as in the small run.)

## Caveats

- One live run per cell; live agents are nondeterministic in file-reading
  order, turn count and duration.
- The report for this run was regenerated with `--rescan` after the codex
  cells finished, which preserved every wire-level number but dropped
  process-side metadata for the four cells of the first phase: agent
  wall-clock, peak RSS and `/stats` snapshots are therefore only reported for
  the codex privaite cells. Nothing wire-derived (leaks, gaps, tap latency,
  cache counters) is affected.
- The codex path (Responses API) has seen fewer live hours than the Claude
  Code path (Anthropic Messages API) overall; treat it as the less
  battle-tested integration even though its numbers here match.
- The leak scan is an exact substring match on the planted values;
  paraphrased or partially masked values do not count as leaks.
- Detection is best-effort. This very page documents a real miss class
  (a detector recall gap at full-log scale on a single very large input).
  Treat the privaite arm as a strong,
  measured reduction, never as a guarantee.
- Raw captures (`results/agent_workflow_big/`, gitignored) contain the
  generated fake secrets and full request bodies; only counts, types and
  variable-name labels appear in this document.
