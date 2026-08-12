# Agent workflow leak benchmark: big session results

Companion to [`RESULTS.md`](RESULTS.md) (the small-fixture matrix). This run
answers the question the small fixture cannot: what happens on a **realistic
long agent session**, with a large log file in context (a single tool output
far above 4k detector tokens), a 10-step audit prompt, and request bodies
growing past 300 KB. Same harness (`run.py`), same validity guards, same 24
planted ground-truth values; only the fixture, prompt and timeout differ.

All numbers below come from `results/agent_workflow_big/report.json`,
generated from live captures on 2026-08-12. Real Claude Code (`claude-opus-5`)
and Codex (`gpt-5.6-terra`) CLIs, real providers, the PrivAiTe gateway from the
local sibling checkout (this gateway shipped in PrivAiTe 0.4.1; these numbers
describe that build). PrivAiTe presets under test: `onnx-auto` (onnx preset,
`device: auto`, detection cache off) and `onnx-auto-cache` (same plus the
opt-in detection cache).

## Headline, stated honestly

**Four of the five comparable gateway measurements leaked 2 of the 24 planted
values: the same two secrets, in both agents, with the detection cache on and
off.** The gateway reduced the leak from 23/24 (both direct arms) to 2/24, and
the 2 is a detection miss, not a routing bug: the gateway scrubbed the very
lines those values sit on. What defeats the detector is the log-shaped context
in front of the value, not the size of the input, and the effect is a property
of the detector rather than of the gateway (evidence below). The honest
agent-CLI headline for a realistic session is "2/24", not the small fixture's
"0/24", and 2/24 is a **floor**, not a ceiling: see "Why 2/24 is a floor".

The sixth cell (codex, `onnx-auto-cache`) reported 0/24 but is **not a
protection measurement**: that run put only 5 of the 11 fixture files on the
wire and never sent a single `key_rotation_failed` log line, so the two values
that are known to slip through were never in play. It is reported as measured,
with its coverage, and excluded from the headline. Codex reached its monthly
usage limit before the cell could be re-run, so it stays as is rather than
being quietly dropped.

## Setup

- Fixture: `acme_support_big`, 11 files, 73,408 bytes: the standard
  `acme_support` files plus `docs/oncall.md`, `docs/runbook.md`, a 69,068-byte
  `logs/ingest_batch.log` (465 lines) and three `tickets/*.md`. Rebuild it
  byte-identically with `python3 agent_workflow/gen_big_fixture.py OUTDIR`
  (fixed seeds).
- Ground truth: the same 24 planted values as the small fixture (4 secret,
  6 email, 6 person, 3 phone, 1 address, 2 iban, 1 credit_card, 1 ssn). The
  log repeats several of them, including two secrets in key=value log-line
  form, so the leak scan covers the same values in more carriers.
- Coverage markers: 11, one per fixture file, generated next to the fixture as
  `acme_support_big.markers.json`. The six big-fixture files (log, tickets,
  extra docs) were added after an earlier run scored a full 5/5 coverage while
  the agent had never sent the 69 KB log the whole fixture exists to exercise.
- Prompt: the 10-step audit in [`prompt_big.txt`](prompt_big.txt) (list files,
  read every doc, read `.env` variable names, read the full log with the
  file-reading tool, read every ticket, cross-reference ticket ids, 12-line
  summary). It is committed so a run is reproducible from the repo alone; the
  original wording lived only in the gitignored report and was lost, so this
  is a rewrite from its description, not the byte-identical July text.
  Per-cell timeout 1800 s.
- Run command:

  ```bash
  python3 agent_workflow/gen_big_fixture.py /path/to/acme_support_big
  AGENT_WORKFLOW_FIXTURE=/path/to/acme_support_big \
  AGENT_WORKFLOW_OUT=results/agent_workflow_big \
  AGENT_WORKFLOW_RESULTS=results/agent_workflow_big/RESULTS.md \
  AGENT_WORKFLOW_TIMEOUT=1800 \
  AGENT_WORKFLOW_PROMPT="$(cat agent_workflow/prompt_big.txt)" \
  python3 agent_workflow/run.py --presets onnx-auto,onnx-auto-cache
  ```

Measurement and validity guards are identical to the small run (see
[`README.md`](README.md)): wire-level substring scan of every captured
provider-bound body, one trivial agent turn before the matrix so an agent that
cannot run is skipped rather than filling its row with empty cells, a
pre-flight probe through the full chain before each privaite cell, per-cell
fresh gateway process, post-cell comparison of the gateway's handled-request
log against the recorder's captures, and a coverage check that no cell
publishes a leak count without the agent having put the fixture on the wire.
Every privaite cell here matched exactly: 18/18, 13/13, 15/15 and 12/12
handled versus captured.

## Values that reached the provider (leaked/planted)

| Agent | Arm | Preset | secret | email | person | phone | address | iban | credit_card | ssn | Files on wire | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude | direct | - | 3/4 | 6/6 | 6/6 | 3/3 | 1/1 | 2/2 | 1/1 | 1/1 | 11/11 | **23/24** |
| claude | privaite | onnx-auto | 2/4 | 0/6 | 0/6 | 0/3 | 0/1 | 0/2 | 0/1 | 0/1 | 11/11 | **2/24** |
| claude | privaite | onnx-auto-cache | 2/4 | 0/6 | 0/6 | 0/3 | 0/1 | 0/2 | 0/1 | 0/1 | 11/11 | **2/24** |
| codex | direct | - | 3/4 | 6/6 | 6/6 | 3/3 | 1/1 | 2/2 | 1/1 | 1/1 | 11/11 | **23/24** |
| codex | privaite | onnx-auto | 2/4 | 0/6 | 0/6 | 0/3 | 0/1 | 0/2 | 0/1 | 0/1 | 10/11 | **2/24** |
| codex | privaite | onnx-auto-cache | 0/4 | 0/6 | 0/6 | 0/3 | 0/1 | 0/2 | 0/1 | 0/1 | 5/11 | 0/24, see below |

The two leaked values are the secrets labeled `ACME_API_KEY` and
`SMTP_PASSWORD` (labels are the `.env` variable names; the generated values
are never printed anywhere in this repo). Both direct arms leaked 23/24 rather
than 24/24 because neither agent ever emitted the JWT signature: agent
behavior, not protection.

The codex `onnx-auto-cache` cell is the one to read with its coverage column.
It sent 5 of 11 files and zero `key_rotation_failed` log lines, against 7 such
lines in the codex `onnx-auto` cell and 41 in claude's. Its 0/24 measures an
agent that skipped the log, not a gateway that removed more.

## Why 2/24, verified in the captures

- Every leaked occurrence in the four comparable privaite cells sits inside the
  log's `event=key_rotation_failed presented_key=... smtp_secret=...` lines.
- The same log lines arrive at the provider with `<DATE_TIME_n>` placeholders
  already substituted in them: the gateway traversed and scrubbed exactly
  these lines, and the detector flagged the timestamps but not the two
  key=value secrets. This rules out a routing/carrier bug (the class of bug
  the 2026-07-16 Codex fix addressed).
- The same two secret values in `.env` assignment form were scrubbed in every
  cell (and all 4 secrets are caught in the small-fixture run, 0/4 leaked).
- The miss reproduces offline, deterministically, with the engine loaded from
  this run's exact `gateway.yaml`, and the offline sweep locates the trigger
  precisely. It is **not** input size:
  - The raw `.env` assignment form is fully scrubbed, and so is a single
    `key_rotation_failed` log line on its own. Both values are detected when
    they stand alone.
  - Roughly **one preceding line of log-shaped context is enough to break
    it**. A 7-line, ~1 KB excerpt of the same log already reproduces the miss:
    the API key survives 5 of its 5 occurrences in that excerpt, the SMTP
    password 4 of 5. 41-line windows leak 4 of 5 and 3 of 5.
  - The effect is **order dependent**: text appended *after* the line never
    triggers it. Only text placed in front of the value does.

  So this is not a gateway routing bug and not a cache regression (identical
  2/24 cache on and off on the agent that exercised the log in both cells): it
  is a detector property, which means **every surface that runs the PrivAiTe
  engine leaks these values on this input**, the OpenAI-compatible proxy and
  the Open WebUI filter and the LiteLLM guardrail alike, not only the agent CLI
  gateway. The onnx preset's measured SECRET recall, 71.4% on the independent
  comparison corpus (see [`COMPARISON.md`](../COMPARISON.md)), already says
  secret detection is the weakest entity type; this is where that weakness
  lands in practice.

## Why 2/24 is a floor, not a ceiling

2/24 is the best-case reading of this run, and it still is on 0.4.1. One of the
four occurrences of the database-URL password is held back only by a **false
positive**: Presidio's `EMAIL_ADDRESS` recognizer scores 1.0 across the whole
userinfo-plus-host span of the connection URI and wins the overlap, so that
occurrence is removed as an email rather than as a secret. (Re-checked against
this build: on the fixture's own `DATABASE_URL` line the covering span is
`EMAIL_ADDRESS` at score 1.00, with `SECRET` scoring 0.67 on a fragment.) Fix
that recognizer's precision, as it should be fixed, and the count on this
fixture becomes 3/24 with no change in detector recall.

That same false positive has a second consequence worth stating plainly: under
the shipped configs, `SECRET` is redacted (irreversible) while `EMAIL_ADDRESS`
gets a reversible placeholder. A database password typed as an email therefore
leaves the machine as a reversible placeholder and is restored in the reply,
which is not the handling an operator who redacts secrets is expecting.

Treat this as the calibration for any claim about agent traffic: the gateway
removes what the detector detects, at the request level, reliably (validity
guarded); the residual risk is the detector's recall, and the count published
here is a floor.

## Timing

Exact per-request gateway latency from the timing tap (second recorder in
front of the gateway, same-clock difference per request):

| Agent | Preset | Median scrub (s) | Max scrub (s) | Median restore (s) |
|---|---|---|---|---|
| claude | onnx-auto | 12.08 | 41.80 | 0.00 |
| claude | onnx-auto-cache | 2.26 | 14.65 | 0.00 |
| codex | onnx-auto | 52.16 | 71.52 | 0.01 |
| codex | onnx-auto-cache | 1.45 | 13.74 | 0.01 |

Per-turn gap (time between consecutive provider-bound requests, bundling
provider generation, agent work and the scrub):

| Agent | Preset | Direct median gap (s) | PrivAiTe median gap (s) | Added per turn (s) |
|---|---|---|---|---|
| claude | onnx-auto | 6.9 | 28.0 | +21.1 |
| claude | onnx-auto-cache | 6.9 | 8.4 | +1.5 |
| codex | onnx-auto | 14.4 | 59.9 | +45.5 |
| codex | onnx-auto-cache | 14.4 | 12.2 | -2.2 |

The codex cache row is negative because that cell skipped most of the fixture
and never carried the large log; it is the same cell excluded from the leak
headline, and it is not evidence that the gateway is free.

Wire span (first to last captured provider-bound request of the cell,
including retries where they occurred):

| Agent | Arm/Preset | Requests captured | Wire span (s) | Last body (KB) |
|---|---|---|---|---|
| claude | direct | 19 (2 attempts) | 349.2 | 338.2 |
| claude | privaite onnx-auto | 18 | 475.6 | 350.4 |
| claude | privaite onnx-auto-cache | 13 | 171.2 | 347.6 |
| codex | direct | 8 | 94.4 | 191.2 |
| codex | privaite onnx-auto | 15 (2 attempts) | 798.6 | 68.3 |
| codex | privaite onnx-auto-cache | 12 | 160.4 | 193.5 |

### The cache on/off story

Without the detection cache the gateway rescans the whole resent conversation
every turn, so per-request scrub cost grows with the context: on these sessions
the maximum reached 41.8 s (claude, ~350 KB bodies) and 71.5 s (codex). That is
unbounded in session length. With the opt-in detection cache the median scrub
drops to 2.26 s (claude) and 1.45 s (codex), because the unchanged prefix hits
the cache (claude cell: 217 hits vs 58 misses; codex cell: 463 vs 44), and
claude's end-to-end wire span drops from 475.6 s to 171.2 s.

**Guidance: enable the detection cache for agent sessions.** On the agent that
exercised the full fixture in both cells, the leak counts are identical cache
on and off (2/24), so the cache costs nothing in protection here while removing
the quadratic scrub cost.

## Gateway process and `device: auto`

- No gateway process died in any cell (every privaite cell passed the
  handled-vs-captured check to the last request). This run exercised
  `device: auto` on 300 KB+ bodies, the exact scenario that used to SIGKILL
  the process when auto selected CoreML; on the fixed PrivAiTe build it is
  stable.
- Peak RSS, sampled from outside the process: 2,983.5 MB and 3,307.6 MB
  (claude, cache off and on), 3,443.7 MB and 4,043.4 MB (codex). Plan for
  roughly 3 to 4 GB.
- Detection cache counters: claude hits=217, misses=58, entries=58; codex
  hits=463, misses=44, entries=44. Counts only, the cache stores salted hashes
  and span metadata, never text.

## Session robustness

- The claude direct cell and the codex `onnx-auto` cell each needed the
  harness's second attempt after the CLI exited 1 mid-session. The gateway
  stayed up and its handled-request count matches the captures, so the failure
  was on the client side of the chain.
- The codex `onnx-auto` cell's first attempt was retried because it exited 1,
  and a separate earlier attempt of the same cell was discarded entirely by the
  coverage guard: Codex's model-catalog refresh timed out, the CLI stopped
  trusting its own workspace and searched the filesystem instead of reading the
  fixture. That run would previously have published a 0/24.

## Provider prompt cache

The providers' own cache counters, extracted from the captured response
streams, keep tracking the growing context through the gateway:
`cache_read_input_tokens` climbs to 142,252 by the last turn of the claude
`onnx-auto` cell (and 138,910 on `onnx-auto-cache`), and `cached_tokens`
climbs to 54,016 on the codex `onnx-auto` cell. Placeholder rewriting is
consistent enough between turns not to destroy the provider-side cache. (Raw
byte-prefix stability is low on the claude arm, 4.8% to 10.3%, but it is
equally low on claude direct at 6.6%; the divergence point is the
`cache_control` breakpoint marker the client moves between turns, so the
token-level counters above are the authoritative signal, as in the small run.
On codex, where the client does not move a marker, prefix stability is 93.4%
direct and 95.8% to 97.1% through the gateway.)

## Caveats

- One live run per cell; live agents are nondeterministic in file-reading
  order, turn count and duration. This run is a good illustration: the same
  prompt produced 8 to 19 provider turns depending on the cell.
- The codex `onnx-auto-cache` cell covers 5 of 11 files and is excluded from
  every comparison in this document. Codex hit its monthly usage limit before
  it could be re-run.
- Leak counts are not portable across models. Both agents are pinned by the
  harness (`claude-opus-5`, `gpt-5.6-terra`) so the arms of a comparison match,
  but a different model reads different files and sends different carriers.
- The codex path (Responses API) has seen fewer live hours than the Claude
  Code path (Anthropic Messages API) overall; treat it as the less
  battle-tested integration even though its numbers here match.
- The leak scan is an exact substring match on the planted values;
  paraphrased or partially masked values do not count as leaks.
- Detection is best-effort. This very page documents a real miss class
  (a secret on a log line, missed once about one preceding line of log-shaped
  context sits in front of it, on every surface that runs the engine). Treat
  the privaite arm as a strong, measured reduction, never as a guarantee, and
  read the leak count as a floor.
- Raw captures (`results/agent_workflow_big/`, gitignored) contain the
  generated fake secrets and full request bodies; only counts, types and
  variable-name labels appear in this document.
