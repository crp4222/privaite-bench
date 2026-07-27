#!/usr/bin/env python3
"""
Agent workflow leak benchmark.

Question answered with numbers: when a real coding agent (Claude Code, Codex)
reads a repository containing secrets and PII, does that data reach the model
provider, what does the PrivAiTe gateway stop, and what does that protection
cost in latency?

The matrix is agent x arm x preset:

  arm `direct`:   agent -> recorder -> provider                    (baseline)
  arm `privaite`: agent -> tap -> PrivAiTe gateway -> recorder -> provider
                  once per PII preset (light, full, onnx by default)

The recorder captures every request body the provider actually receives; the
tap is a second recorder instance in front of the gateway, so per-request
gateway latency is the difference between the same request's timestamps on
both sides. After each privaite cell the harness snapshots the gateway's
/stats endpoint (per-entity-type detection counts; the gateway process is
fresh per cell, so the counts are per cell).

Every recorded body is then scanned for the exact ground-truth values planted
by gen_fixture.py, and the per-category leak counts, timing, context growth
and cache analysis land in RESULTS.md plus results/agent_workflow/report.json.

Usage:
    python agent_workflow/run.py [--agents claude,codex] [--arms direct,privaite]
                                 [--presets light,full,onnx] [--timeout 600]
    python agent_workflow/run.py --rescan     # regenerate the report from
                                              # captures on disk, no live runs
    python agent_workflow/run.py --selftest   # check the measurement math

A cell that hits the timeout is recorded as data (timed_out=true, partial
capture scanned and reported), never retried and never fatal: the run always
finishes and RESULTS.md says what was captured.

Needs the PrivAiTe checkout as a sibling directory (../PrivAiTe, override with
PRIVAITE_PATH) with its venv at .venv, and both agent CLIs logged in. Raw
captures are written under results/agent_workflow/ (gitignored: they contain
the generated fake secrets and full request bodies).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Env overrides exist so a side experiment (bigger fixture, different prompt,
# separate output dir) can reuse the whole harness without touching the main
# matrix's captures or RESULTS.md. Defaults are the historical paths.
FIXTURE = Path(os.environ.get("AGENT_WORKFLOW_FIXTURE", str(HERE / "fixtures" / "acme_support")))
OUT = Path(os.environ.get("AGENT_WORKFLOW_OUT", str(REPO / "results" / "agent_workflow")))
RESULTS_PATH = Path(os.environ.get("AGENT_WORKFLOW_RESULTS", str(HERE / "RESULTS.md")))
PRIVAITE_ROOT = Path(os.environ.get("PRIVAITE_PATH", REPO.parent / "PrivAiTe"))
PRIVAITE_PYTHON = PRIVAITE_ROOT / ".venv" / "bin" / "python"

sys.path.insert(0, str(HERE))
import gen_fixture  # noqa: E402
from recorder import PROBE_HEADER  # noqa: E402

RECORDER_PORT = 8401
GATEWAY_PORT = 8400
TAP_PORT = 8402
# Codex resends the whole conversation each turn and the onnx scrub of those
# 50KB+ bodies adds ~30-40s per turn, so the slowest cell needs real headroom.
# A timeout is recorded as data (timed_out=true) and the run continues.
AGENT_TIMEOUT = int(os.environ.get("AGENT_WORKFLOW_TIMEOUT", "600"))
# Retries are for immediate failures (nonzero exit, nothing captured), never
# for timeouts: a timed-out cell already holds the interesting partial data.
MAX_ATTEMPTS = 2

MATRIX_AGENTS = ("claude", "codex")
DEFAULT_PRESETS = ("light", "full", "onnx")

# Captures taken on or before this date predate the PrivAiTe fix for
# Responses-API tool-output scrubbing (custom_tool_call_output carriers); the
# report flags those cells so pre-fix numbers are never quoted as current.
PRE_FIX_CUTOFF = "2026-07-16"

PROMPT = os.environ.get(
    "AGENT_WORKFLOW_PROMPT",
    "Read every file in this repository, including docs/ and the .env file, "
    "then summarize in under 12 lines: what customer data does customers.json "
    "contain, what do the docs describe, and which configuration variables "
    "does .env define? Do not modify any files.",
)

# A matrix "preset" is either a plain PrivAiTe preset name (historical cells:
# device pinned to cpu because device auto used to select CoreML, whose
# accumulating compiled-model memory got the process SIGKILLed on large agent
# bodies; cache off because the cache did not exist) or a named variant that
# sets the engine preset plus the gateway knobs under test. The variants
# exercise the code path a real user gets after the CoreML fix: device auto,
# with and without the opt-in detection cache.
PRESET_VARIANTS: dict[str, tuple[str, str, bool]] = {
    # name: (engine preset, onnx device, detection cache enabled)
    "onnx-auto": ("onnx", "auto", False),
    "onnx-auto-cache": ("onnx", "auto", True),
}


def preset_knobs(preset: str) -> tuple[str, str, bool]:
    return PRESET_VARIANTS.get(preset, (preset, "cpu", False))


def gateway_yaml(preset: str) -> str:
    base, device, cache = preset_knobs(preset)
    return f"""\
server: {{host: "127.0.0.1", port: {GATEWAY_PORT}}}
auth: {{enabled: false}}
providers: []
gateway:
  enabled: true
  anthropic: {{base_url: "http://127.0.0.1:{RECORDER_PORT}"}}
  openai_responses: {{base_url: "http://127.0.0.1:{RECORDER_PORT}"}}
pii:
  enabled: true
  preset: "{base}"
  detectors: {{onnx: {{device: "{device}"}}}}
  detection_cache: {{enabled: {str(cache).lower()}}}
  deanonymization: {{enabled: true, fuzzy_matching: false}}
logging: {{format: "json", level: "info"}}
"""


# Codex emits harmless local noise (a broken MCP server on 127.0.0.1:8080,
# SessionStart hooks); drop those lines from the captured transcript.
NOISE_MARKERS = ("127.0.0.1:8080", "SessionStart", "MCP")

CATEGORIES = ("secret", "email", "person", "phone", "address", "iban", "credit_card", "ssn")

# Prefixes of the placeholder tokens the gateway substitutes for detected
# values. In a capture they are scrub evidence; in the agent's user-facing
# transcript they are a restore failure (the user saw a placeholder where the
# real value belonged).
PLACEHOLDER_TOKENS = ("<PERSON", "<EMAIL", "<PHONE", "<CREDIT", "<IBAN", "<SECRET", "<PASSWORD")

# Neutral (non-sensitive) strings unique to each fixture file. A privaite-arm
# zero only counts if the agent actually read the files: these markers prove
# the file content reached the wire (scrubbed or not).
FILE_MARKERS = {
    "README.md": "customer export consumed by the nightly billing",
    "docs/escalations.md": "pending_review",
    "docs/team.md": "primary pages secondary after 15 minutes",
    "customers.json": "cus_1043",
    ".env": "SMTP_HOST",
}

# Provider cache counters extracted from captured response streams: the proof
# that the provider prompt cache survives (or does not survive) the gateway.
CACHE_COUNTERS = {
    "anthropic cache_read_input_tokens": re.compile(r'"cache_read_input_tokens"\s*:\s*(\d+)'),
    "openai cached_tokens": re.compile(r'"cached_tokens"\s*:\s*(\d+)'),
}

# A request body that shrinks this much versus its predecessor is a new
# conversation (a retried attempt), not a grown one.
RESTART_SHRINK = 0.7


def cell_name(agent: str, arm: str, preset: str | None) -> str:
    return f"{agent}_{arm}" if arm == "direct" else f"{agent}_{arm}_{preset}"


def merged_presets(cli_presets: list[str], previous_presets: set[str]) -> list[str]:
    """The rendered matrix covers the union of the presets requested now and
    the presets already present in the previous report, in stable order: a
    partial run (say --presets onnx) must never drop the light cells' results
    from report.json and RESULTS.md just because they were not re-run."""
    wanted = set(cli_presets) | set(previous_presets)
    ordered = [p for p in DEFAULT_PRESETS if p in wanted]
    return ordered + sorted(p for p in wanted if p not in DEFAULT_PRESETS)


def cell_files(agent: str, arm: str, preset: str | None, allow_legacy: bool = True) -> dict[str, Path]:
    """Paths for a cell's capture, timing tap, agent transcript and gateway
    log. Falls back to the pre-matrix file names (no preset suffix) for the
    2026-07-16 onnx captures so --rescan keeps working on them.

    The legacy fallback is only taken with allow_legacy=True: a cell that was
    freshly run must never inherit an old capture just because its own run
    recorded nothing (that misattribution once reported the 2026-07-16 pre-fix
    codex capture as a fresh codex/onnx leak count). Callers pass
    allow_legacy=False whenever the cell's own metadata says it postdates the
    pre-fix cutoff."""
    name = cell_name(agent, arm, preset)
    files = {
        "capture": OUT / f"{name}.jsonl",
        "tap": OUT / f"{name}.tap.jsonl",
        "out": OUT / f"{name}.out.txt",
        "gateway_log": OUT / f"gateway_{name}.log",
    }
    if arm == "privaite" and preset == "onnx" and allow_legacy and not files["capture"].exists():
        legacy = OUT / f"{agent}_{arm}.jsonl"
        if legacy.exists():
            files = {
                "capture": legacy,
                "tap": OUT / f"{agent}_{arm}.tap.jsonl",
                "out": OUT / f"{agent}_{arm}.out.txt",
                "gateway_log": OUT / f"gateway_{agent}_{arm}.log",
            }
    return files


def wait_health(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return False


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def agent_command(agent: str, base_url: str) -> tuple[list[str], dict[str, str]]:
    env = dict(os.environ)
    if agent == "claude":
        env["ANTHROPIC_BASE_URL"] = base_url
        return ["claude", "-p", PROMPT], env
    if agent == "codex":
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "-c",
            "model_provider=privaite",
            "-c",
            'model_providers.privaite.name="PrivAiTe"',
            "-c",
            f'model_providers.privaite.base_url="{base_url}/v1"',
            "-c",
            'model_providers.privaite.wire_api="responses"',
            "-c",
            "model_providers.privaite.requires_openai_auth=true",
            "-c",
            "model=gpt-5.6-sol",
            PROMPT,
        ]
        return cmd, env
    raise ValueError(f"unknown agent {agent}")


def filter_noise(text: str) -> str:
    kept = [
        line
        for line in text.splitlines()
        if not any(marker in line for marker in NOISE_MARKERS)
    ]
    return "\n".join(kept)


def _as_text(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", "replace")
    return stream


def fetch_cache_stats() -> dict | None:
    """Snapshot the detection cache counters from the bench-only launcher
    route (hit/miss/entry counts only, never keys or spans)."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{GATEWAY_PORT}/bench/cachestats", timeout=10
        ) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


class RssSampler:
    """Peak RSS of the gateway process, sampled with `ps` about once a second.
    A SIGKILL by the memory killer is exactly the failure mode under test, so
    the peak has to come from outside the process being measured."""

    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._stop = threading.Event()
        self.peak_kb = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(self._pid)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                value = out.stdout.strip()
                if value:
                    self.peak_kb = max(self.peak_kb, int(value))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(1.0)

    def stop(self) -> int:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.peak_kb


def fetch_gateway_stats() -> dict | None:
    """Snapshot the gateway's /stats (per-entity-type counts, no values). The
    gateway config runs with auth disabled, so no key is needed; with auth
    enabled /stats would need the same Authorization header as the API."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{GATEWAY_PORT}/stats", timeout=10
        ) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def merge_stats(raw: dict | None) -> dict | None:
    """Collapse /stats sessions into one per-cell summary: the gateway process
    is fresh per cell, so everything it counted belongs to this cell."""
    if not raw or not raw.get("enabled"):
        return None
    by_type: dict[str, int] = {}
    requests = 0
    total = 0
    for session in raw.get("sessions", {}).values():
        requests += session.get("requests", 0)
        total += session.get("total_pii", 0)
        for etype, count in session.get("by_type", {}).items():
            by_type[etype] = by_type.get(etype, 0) + count
    return {"requests": requests, "total_pii": total, "by_type": by_type}


# One line per request the gateway actually handled, written by the gateway's
# own logger the moment upstream response headers arrive:
#   {"..."logger": "privaite.gateway", "message": "gateway /v1/responses -> 200 (restore)"}
_GATEWAY_HANDLED = re.compile(r'"message":\s*"gateway /[^"]* -> \d+')


def gateway_handled_count(log_path: Path | None) -> int | None:
    """How many protocol requests the gateway itself says it handled, counted
    from its own log. None when the log does not exist (nothing to verify
    against, which the validity check treats as unverifiable, not as valid)."""
    if log_path is None or not log_path.exists():
        return None
    return sum(1 for line in log_path.read_text().splitlines() if _GATEWAY_HANDLED.search(line))


def captured_request_count(capture: Path) -> int:
    """Provider-bound request bodies in a capture file. Probe and response
    records do not count: only records carrying a `body` are agent traffic."""
    if not capture.exists():
        return 0
    n = 0
    for line in capture.read_text().splitlines():
        try:
            if "body" in json.loads(line):
                n += 1
        except (json.JSONDecodeError, TypeError):
            continue
    return n


def probe_record_count(capture: Path) -> int:
    """Pre-flight probe records in a capture file. Probes traverse the gateway
    (so they appear in its handled count and /stats) but are harness plumbing,
    not agent traffic: they are subtracted before comparing gateway-handled
    against recorder-captured."""
    if not capture.exists():
        return 0
    n = 0
    for line in capture.read_text().splitlines():
        try:
            if json.loads(line).get("kind") == "probe":
                n += 1
        except (json.JSONDecodeError, AttributeError):
            continue
    return n


def classify_validity(
    arm: str,
    requests: int,
    gateway_handled: int | None,
    probe_failure: str | None = None,
) -> str | None:
    """Reason the cell's leak measurement is invalid, or None when it can be
    trusted. Fail closed: a privaite cell whose captured traffic cannot be
    proven to have traversed the gateway must never publish a leak count,
    because the recorder sees raw bodies whenever anything reaches it without
    the scrub in front."""
    if probe_failure:
        return f"pre-flight probe failed: {probe_failure}"
    if requests == 0:
        return "no provider-bound traffic captured"
    if arm != "privaite":
        return None
    if gateway_handled is None:
        return "gateway traversal unverifiable: no gateway log for this cell"
    if gateway_handled < requests:
        return (
            f"gateway handled {gateway_handled} request(s) but the recorder captured "
            f"{requests}: traffic did not traverse the gateway"
        )
    return None


def preflight_probe(agent: str, gateway_log: Path, capture: Path) -> str | None:
    """Send one synthetic request through the full privaite chain, agent entry
    point -> tap -> gateway -> recorder, and demand proof of traversal at both
    ends before the agent CLI is launched. Returns None on success, else the
    failure reason (the cell is then recorded as invalid and never run).

    The request carries the X-Bench-Probe header: the tap forwards it, the
    recorder absorbs it (the real provider is never contacted) and both log it
    as probe records the scan and timing analysis ignore. This also proves the
    protocol route is genuinely served end to end with a warm engine, which a
    200 on /health never proved."""
    path = "/v1/messages" if agent == "claude" else "/v1/responses"
    body = json.dumps(
        {
            "model": "bench-probe",
            "stream": False,
            "messages": [{"role": "user", "content": "bench preflight probe"}],
        }
    ).encode()
    handled_before = gateway_handled_count(gateway_log) or 0
    probes_before = probe_record_count(capture)
    req = urllib.request.Request(
        f"http://127.0.0.1:{TAP_PORT}{path}",
        data=body,
        headers={"Content-Type": "application/json", PROBE_HEADER: "1"},
        method="POST",
    )
    try:
        # Generous timeout: the probe pays the engine's first-inference cost.
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        return f"probe request did not complete through the chain: {exc}"
    if b'"probe"' not in payload:
        return f"probe reached an endpoint that is not the recorder: {payload[:200]!r}"
    deadline = time.time() + 15
    while time.time() < deadline:
        handled = gateway_handled_count(gateway_log) or 0
        if handled > handled_before and probe_record_count(capture) > probes_before:
            return None
        time.sleep(0.5)
    return (
        "probe completed but left no trace where it must: "
        f"gateway handled {gateway_handled_count(gateway_log)} (was {handled_before}), "
        f"recorder probe records {probe_record_count(capture)} (was {probes_before})"
    )


def run_cell(agent: str, arm: str, preset: str | None) -> dict:
    """One (agent, arm, preset) run: start recorder (plus gateway and timing
    tap on the privaite arm), drive the agent, capture, snapshot /stats."""
    name = cell_name(agent, arm, preset)
    capture = OUT / f"{name}.jsonl"
    tap_capture = OUT / f"{name}.tap.jsonl"
    # Truncate to empty rather than delete: the capture file of a freshly run
    # cell must always exist, so a failed run scans as 0 requests instead of
    # falling back to a stale legacy capture in cell_files().
    capture.write_text("")
    tap_capture.write_text("")
    recorder = subprocess.Popen(
        [sys.executable, str(HERE / "recorder.py"), "--port", str(RECORDER_PORT), "--log", str(capture)],
        stdout=(OUT / f"recorder_{name}.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    gateway = None
    tap = None
    rss_sampler: RssSampler | None = None
    result: dict = {
        "agent": agent,
        "arm": arm,
        "preset": preset,
        "attempts": 0,
        "timed_out": False,
        "notes": [],
        "captured": date.today().isoformat(),
    }
    try:
        if not wait_health(f"http://127.0.0.1:{RECORDER_PORT}/health", 15):
            result["notes"].append("recorder failed to start")
            return result
        if arm == "privaite":
            yaml_path = OUT / "gateway.yaml"
            yaml_path.write_text(gateway_yaml(preset or "onnx"))
            gateway = subprocess.Popen(
                [str(PRIVAITE_PYTHON), str(HERE / "bench_gateway.py"), "--config", str(yaml_path)],
                cwd=str(PRIVAITE_ROOT),
                stdout=(OUT / f"gateway_{name}.log").open("w"),
                stderr=subprocess.STDOUT,
            )
            if not wait_health(f"http://127.0.0.1:{GATEWAY_PORT}/health", 240):
                result["notes"].append("privaite gateway failed to become healthy")
                return result
            rss_sampler = RssSampler(gateway.pid)
            tap = subprocess.Popen(
                [
                    sys.executable,
                    str(HERE / "recorder.py"),
                    "--port",
                    str(TAP_PORT),
                    "--log",
                    str(tap_capture),
                    "--anthropic-base",
                    f"http://127.0.0.1:{GATEWAY_PORT}/v1",
                    "--codex-base",
                    f"http://127.0.0.1:{GATEWAY_PORT}/v1",
                ],
                stdout=(OUT / f"tap_{name}.log").open("w"),
                stderr=subprocess.STDOUT,
            )
            if not wait_health(f"http://127.0.0.1:{TAP_PORT}/health", 15):
                result["notes"].append("timing tap failed to start")
                return result
            probe_failure = preflight_probe(agent, OUT / f"gateway_{name}.log", capture)
            if probe_failure is not None:
                result["probe_failure"] = probe_failure
                result["notes"].append(
                    f"pre-flight probe failed, agent not launched: {probe_failure}"
                )
                return result
            base_url = f"http://127.0.0.1:{TAP_PORT}"
        else:
            base_url = f"http://127.0.0.1:{RECORDER_PORT}"

        cmd, env = agent_command(agent, base_url)
        result["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        attempt_durations: list[float] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result["attempts"] = attempt
            started = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(FIXTURE),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=AGENT_TIMEOUT,
                )
                attempt_durations.append(round(time.time() - started, 1))
                result["exit_code"] = proc.returncode
                output = filter_noise(proc.stdout + "\n" + proc.stderr)
                (OUT / f"{name}.out.txt").write_text(output)
            except subprocess.TimeoutExpired as exc:
                attempt_durations.append(round(time.time() - started, 1))
                result["exit_code"] = None
                result["timed_out"] = True
                output = filter_noise(_as_text(exc.stdout) + "\n" + _as_text(exc.stderr))
                (OUT / f"{name}.out.txt").write_text(output)
                n_captured = len(capture.read_text().splitlines()) if capture.exists() else 0
                result["notes"].append(
                    f"attempt {attempt}: timed out after {AGENT_TIMEOUT}s, "
                    f"{n_captured} capture records kept; timeout recorded as data, not retried"
                )
                break
            requests_seen = captured_request_count(capture) > 0
            if result.get("exit_code") == 0 and requests_seen:
                break
            result["notes"].append(
                f"attempt {attempt}: exit={result.get('exit_code')}, requests_recorded={requests_seen}"
            )
            if gateway is not None and gateway.poll() is not None:
                # Retrying against a dead gateway can only produce refused
                # connections at the tap; stop here, the validity check will
                # mark the cell invalid.
                result["notes"].append(
                    f"gateway process exited mid-cell (code {gateway.poll()}); "
                    "remaining attempts aborted"
                )
                break
        if gateway is not None and gateway.poll() is not None:
            result["gateway_exit_code"] = gateway.poll()
        result["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        result["attempt_durations_s"] = attempt_durations
        if attempt_durations:
            result["duration_s"] = attempt_durations[-1]
        if arm == "privaite":
            stats = merge_stats(fetch_gateway_stats())
            if stats is None:
                result["notes"].append("gateway /stats snapshot unavailable")
            else:
                result["stats"] = stats
            cache_stats = fetch_cache_stats()
            if cache_stats is not None:
                result["cache_stats"] = cache_stats
        return result
    finally:
        if rss_sampler is not None:
            peak_kb = rss_sampler.stop()
            if peak_kb:
                result["gateway_peak_rss_mb"] = round(peak_kb / 1024, 1)
        if gateway is not None and gateway.poll() is not None and "gateway_exit_code" not in result:
            result["gateway_exit_code"] = gateway.poll()
        stop(tap)
        stop(gateway)
        stop(recorder)


def _read_records(path: Path | None) -> tuple[list[dict], list[dict]]:
    """Split a recorder JSONL into request records (have `body`) and response
    records (`kind: response`). Pre-instrumentation captures only have the
    former."""
    requests: list[dict] = []
    responses: list[dict] = []
    if path is None or not path.exists():
        return requests, responses
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        if "body" in rec:
            requests.append(rec)
        elif rec.get("kind") == "response":
            responses.append(rec)
    return requests, responses


def _is_main_request(rec: dict) -> bool:
    """Turn analysis only looks at the conversation-carrying requests, not
    side calls like count_tokens (tiny bodies that would corrupt the growth
    and restart math)."""
    return not rec.get("path", "").split("?", 1)[0].endswith("/count_tokens")


def _median(values: list[float], digits: int = 1) -> float | None:
    return round(statistics.median(values), digits) if values else None


def _shared_prefix_fraction(prev: str, cur: str) -> float:
    limit = min(len(prev), len(cur))
    i = 0
    while i < limit and prev[i] == cur[i]:
        i += 1
    return i / len(prev) if prev else 0.0


def analyze_capture(capture: Path, tap: Path | None) -> dict:
    """Offline timing and context analysis of one cell's capture: per-turn
    body sizes and gaps, restart detection, byte-prefix stability, provider
    cache counters, and (when a tap file exists) exact per-request gateway
    latency (tap ts is pre-scrub, capture ts is post-scrub)."""
    all_reqs, resps = _read_records(capture)
    reqs = [r for r in all_reqs if _is_main_request(r)]
    if not reqs:
        return {"requests": len(all_reqs)}
    sizes = [len(r["body"]) for r in reqs]
    ts = [float(r["ts"]) for r in reqs]
    restarts = {i for i in range(1, len(sizes)) if sizes[i] < RESTART_SHRINK * sizes[i - 1]}
    turns: list[dict] = []
    gaps: list[float] = []
    prefixes: list[float] = []
    for i, (size, t) in enumerate(zip(sizes, ts)):
        turn: dict = {"turn": i + 1, "bytes": size, "t_s": round(t - ts[0], 1)}
        if i in restarts:
            turn["restart"] = True
        elif i > 0:
            gap = round(ts[i] - ts[i - 1], 1)
            turn["gap_s"] = gap
            gaps.append(gap)
            frac = _shared_prefix_fraction(reqs[i - 1]["body"], reqs[i]["body"])
            turn["prefix_shared"] = round(frac, 3)
            prefixes.append(frac)
        turns.append(turn)
    info: dict = {
        "requests": len(all_reqs),
        "conversations": len(restarts) + 1,
        "wire_span_s": round(ts[-1] - ts[0], 1),
        "body_bytes_first": sizes[0],
        "body_bytes_last": sizes[-1],
        "body_bytes_max": max(sizes),
        "est_tokens_last": max(sizes) // 4,
        "gap_median_s": _median(gaps),
        "prefix_shared_median": round(statistics.median(prefixes), 3) if prefixes else None,
        "turns": turns,
    }
    if resps:
        upstream = [
            r["upstream_s"] for r in resps if isinstance(r.get("upstream_s"), (int, float))
        ]
        if upstream:
            info["upstream_median_s"] = _median(upstream)
        counters: dict[str, list[int | None]] = {}
        for label, pattern in CACHE_COUNTERS.items():
            series = []
            for r in resps:
                found = [int(v) for v in pattern.findall(r.get("response", ""))]
                series.append(max(found) if found else None)
            if any(v is not None for v in series):
                counters[label] = series
        if counters:
            info["cache_counters"] = counters
    if tap is not None and tap.exists():
        tap_reqs, tap_resps = _read_records(tap)
        tap_reqs = [r for r in tap_reqs if _is_main_request(r)]
        paired = min(len(tap_reqs), len(reqs))
        scrub = [
            round(ts[i] - float(tap_reqs[i]["ts"]), 2)
            for i in range(paired)
            if ts[i] >= float(tap_reqs[i]["ts"])
        ]
        if scrub:
            info["scrub_s"] = scrub
            info["scrub_median_s"] = _median(scrub, 2)
            info["scrub_max_s"] = max(scrub)
        paired_resps = min(len(tap_resps), len(resps))
        restore = []
        for i in range(paired_resps):
            a, b = tap_resps[i].get("ts_done"), resps[i].get("ts_done")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a >= b:
                restore.append(round(a - b, 2))
        if restore:
            info["restore_median_s"] = _median(restore, 2)
        if len(tap_reqs) != len(reqs):
            info["tap_note"] = (
                f"tap saw {len(tap_reqs)} requests vs {len(reqs)} behind the gateway; "
                f"paired the first {paired} in order"
            )
    return info


def scan(capture: Path, manifest: list[dict]) -> tuple[dict, dict, int, bool]:
    """Which planted values (and which file markers) appear in the recorded
    provider-bound bodies. The markers prove the agent actually read each
    fixture file, so a zero-leak cell cannot hide behind an agent that never
    opened the sensitive files."""
    bodies: list[str] = []
    if capture.exists():
        for line in capture.read_text().splitlines():
            try:
                bodies.append(json.loads(line)["body"])
            except (json.JSONDecodeError, KeyError):
                continue
    joined = "\n".join(bodies)
    leaked = {entry["label"]: entry["value"] in joined for entry in manifest}
    coverage = {name: marker in joined for name, marker in FILE_MARKERS.items()}
    # Placeholder tokens are evidence the scrubber ran on at least part of the
    # traffic. Their absence next to a leak means the value was never scrubbed;
    # a caught value with no placeholder anywhere means the agent never sent it
    # (self-censored), not that the gateway removed it.
    scrubbed = any(tok in joined for tok in PLACEHOLDER_TOKENS)
    return leaked, coverage, len(bodies), scrubbed


def per_category(manifest: list[dict], leaked: dict) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for cat in CATEGORIES:
        entries = [e for e in manifest if e["category"] == cat]
        if not entries:
            continue
        hit = sum(1 for e in entries if leaked[e["label"]])
        counts[cat] = (hit, len(entries))
    return counts


def corpus_stats() -> dict:
    files = sorted(
        p for p in FIXTURE.rglob("*") if p.is_file() and p.name != ".gitignore"
    )
    return {
        "files": [str(p.relative_to(FIXTURE)) for p in files],
        "bytes": sum(p.stat().st_size for p in files),
    }


def _arm_label(cell: dict) -> str:
    if cell["arm"] == "direct":
        return "direct"
    return f"privaite ({cell.get('preset') or 'onnx'})"


def _kb(n: int | None) -> str:
    return f"{n / 1024:.1f}" if isinstance(n, (int, float)) else "?"


def _fmt(v, suffix: str = "") -> str:
    return f"{v}{suffix}" if v is not None else "?"


def _is_pre_fix(cell: dict) -> bool:
    return (cell.get("captured") or PRE_FIX_CUTOFF) <= PRE_FIX_CUTOFF


def write_results(report: dict) -> None:
    manifest = report["manifest"]
    matrix: list[dict] = report["matrix"]
    cells: list[dict] = report["cells"]
    by_key = {(c["agent"], c["arm"], c.get("preset")): c for c in cells}

    def cell_of(agent: str, arm: str, preset: str | None) -> dict | None:
        return by_key.get((agent, arm, preset))

    lines: list[str] = []
    add = lines.append
    add("# Agent workflow leak benchmark: results")
    add("")
    add(f"Generated by `agent_workflow/run.py` on {report['date']}. "
        "Raw captures live under `results/agent_workflow/` (gitignored).")
    add("")

    # Setup ------------------------------------------------------------------
    add("## Setup")
    add("")
    stats = report["corpus"]
    total = len(manifest)
    cat_counts: dict[str, int] = {}
    for entry in manifest:
        cat_counts[entry["category"]] = cat_counts.get(entry["category"], 0) + 1
    add(
        f"Fixture: `agent_workflow/fixtures/acme_support/` ({len(stats['files'])} files, "
        f"{stats['bytes']} bytes: {', '.join(stats['files'])}). "
        f"{total} planted ground-truth values: "
        + ", ".join(f"{v} {k}" for k, v in sorted(cat_counts.items()))
        + ". Secrets are fake, generated at runtime from a fixed seed and gitignored."
    )
    add("")
    presets = report["presets"]
    add(
        "Matrix: agent (claude, codex) x arm (direct, privaite) x PrivAiTe preset ("
        + ", ".join(presets)
        + "). The direct arm has no preset: it is the no-proxy baseline every "
        "privaite cell is compared against. Cells not yet captured are marked "
        "pending; nothing in this document is extrapolated."
    )
    add("")
    add("Prompt given to both agents, run from inside the fixture directory:")
    add("")
    add(f"> {report['prompt']}")
    add("")
    add("Measurement: a recording proxy sits between the agent (or the PrivAiTe "
        "gateway) and the real provider and captures every forwarded request "
        "body with timestamps, plus the response stream and its timing. On the "
        "privaite arm a second recorder instance (the timing tap) sits in front "
        "of the gateway, so per-request gateway latency is measured, not "
        "inferred. A value counts as leaked when its exact string appears in "
        "any body the provider received. This measures the wire, not what the "
        "agent chose to display.")
    add("")
    add("Validity guard: before each privaite cell a synthetic pre-flight probe "
        "must traverse the whole chain (agent entry point, tap, gateway, "
        "recorder) or the agent is never launched; after each cell the gateway's "
        "own handled-request log is compared against the request bodies the "
        "recorder captured. A cell that fails either check is marked invalid "
        "below and publishes no leak count at all: a number measured without "
        "the gateway provably in the path would be a false leak report.")
    add("")

    # Leak table ---------------------------------------------------------------
    add("## Values that reached the provider (leaked/planted)")
    add("")
    cats = [c for c in CATEGORIES if c in cat_counts]
    add("| Agent | Arm | Preset | " + " | ".join(cats) + " | Total |")
    add("|" + "---|" * (len(cats) + 4))
    for slot in matrix:
        cell = cell_of(slot["agent"], slot["arm"], slot["preset"])
        preset_txt = slot["preset"] or "-"
        if cell is None:
            add(
                f"| {slot['agent']} | {slot['arm']} | {preset_txt} | "
                + " | ".join("-" for _ in cats)
                + " | pending live run |"
            )
            continue
        if cell.get("invalid"):
            add(
                f"| {slot['agent']} | {slot['arm']} | {preset_txt} | "
                + " | ".join("-" for _ in cats)
                + " | invalid, not measured |"
            )
            continue
        pc = cell["per_category"]
        row = [cell["agent"], cell["arm"], preset_txt]
        total_hit = 0
        total_n = 0
        for cat in cats:
            hit, n = pc.get(cat, (0, 0))
            total_hit += hit
            total_n += n
            row.append(f"{hit}/{n}")
        row.append(f"**{total_hit}/{total_n}**")
        add("| " + " | ".join(row) + " |")
    add("")
    invalid_cells = [c for c in cells if c.get("invalid")]
    if invalid_cells:
        add("Invalid cells (no leak number exists for them, do not quote one):")
        add("")
        for cell in invalid_cells:
            add(f"- {cell['agent']} / {_arm_label(cell)}: {cell.get('invalid_reason')}")
        add("")

    # Key findings -------------------------------------------------------------
    add("## Key findings")
    add("")
    for agent in MATRIX_AGENTS:
        cell = cell_of(agent, "direct", None)
        if cell is None or cell.get("invalid"):
            continue
        hit = sum(cell["leaked"].values())
        add(f"- Baseline: {agent} on the `direct` arm sent {hit}/{total} of "
            "the planted values to the provider. Reading a repo with a coding agent "
            "does put its secrets and PII on the wire.")
    for cell in cells:
        if cell["arm"] != "privaite" or cell.get("invalid"):
            continue
        agent = cell["agent"]
        hit = sum(cell["leaked"].values())
        cov = sum(cell["coverage"].values())
        if hit == 0 and cov == len(cell["coverage"]):
            add(f"- {agent} via PrivAiTe [{cell.get('preset')}]: 0/{total} reached the "
                f"provider even though the agent read all {cov} files. The gateway "
                "scrubbed every planted value out of this agent's request shape.")
        elif hit > 0:
            secret_hit = cell["per_category"].get("secret", (0, 0))[0]
            secret_note = ""
            if secret_hit == 0 and not cell.get("scrubbed_evidence", True):
                secret_note = (
                    " Note the secret 0/4 here is not a gateway catch: no placeholder token "
                    "appears in the capture, so this agent never put the raw secret values on "
                    "the wire (it read `.env` by counting assignments), it did not need "
                    "scrubbing."
                )
            elif secret_hit == 0:
                secret_note = (
                    " The 4 secrets did not reach the provider, but this agent also never "
                    "emitted their raw values, so it is not purely a gateway catch."
                )
            add(f"- {agent} via PrivAiTe [{cell.get('preset')}]: {hit}/{total} still "
                "reached the provider. This is a real gap, not a rounding error."
                + secret_note)
    add("")

    # Misses ---------------------------------------------------------------
    add("## What PrivAiTe missed (honest list)")
    add("")
    any_miss = False
    for cell in cells:
        if cell["arm"] != "privaite" or cell.get("invalid"):
            continue
        missed = [label for label, hit in cell["leaked"].items() if hit]
        if missed:
            any_miss = True
            add(f"- {cell['agent']} ({_arm_label(cell)}), reached the provider: "
                + ", ".join(f"`{m}`" for m in missed))
    if not any_miss:
        add("- none of the planted values reached the provider through the gateway "
            "in the captured cells (see caveats below before quoting this as 100%).")
    add("")

    # Engine stats -----------------------------------------------------------
    add("## Engine detections (gateway /stats)")
    add("")
    add("After each privaite cell the harness snapshots the gateway's `/stats` "
        "endpoint. The gateway process is restarted per cell, so the counts are "
        "per cell: how many requests the engine touched and what entity types it "
        "detected, per preset. `/stats` stores per-type counts only, never "
        "values. (The benchmark gateway runs with `auth.enabled: false`; with "
        "auth enabled the snapshot would need the same key as the API.)")
    add("")
    add("| Agent | Preset | Requests counted | Total detections | By entity type |")
    add("|---|---|---|---|---|")
    for slot in matrix:
        if slot["arm"] != "privaite":
            continue
        cell = cell_of(slot["agent"], slot["arm"], slot["preset"])
        if cell is None:
            add(f"| {slot['agent']} | {slot['preset']} | - | - | pending live run |")
            continue
        st = cell.get("stats")
        if not st:
            reason = (
                f"invalid cell: {cell.get('invalid_reason')}"
                if cell.get("invalid")
                else "no /stats snapshot for this cell (the next live run records it)"
            )
            add(f"| {cell['agent']} | {cell.get('preset')} | - | - | {reason} |")
            continue
        by_type = ", ".join(f"{k}={v}" for k, v in sorted(st["by_type"].items())) or "none"
        add(f"| {cell['agent']} | {cell.get('preset')} | {st['requests']} | "
            f"{st['total_pii']} | {by_type} |")
    add("")

    # Gateway process ------------------------------------------------------
    proc_rows = []
    for slot in matrix:
        if slot["arm"] != "privaite":
            continue
        cell = cell_of(slot["agent"], slot["arm"], slot["preset"])
        if cell is None:
            continue
        base_preset, device, cache_enabled = preset_knobs(cell.get("preset") or "onnx")
        peak = cell.get("gateway_peak_rss_mb")
        exit_code = cell.get("gateway_exit_code")
        survived = "KILLED/EXITED MID-CELL" if exit_code is not None else "alive whole cell"
        cs = cell.get("cache_stats")
        if cs is None:
            cache_txt = "not sampled"
        elif not cs.get("enabled"):
            cache_txt = "off"
        else:
            cache_txt = f"hits={cs['hits']}, misses={cs['misses']}, entries={cs['entries']}"
        proc_rows.append(
            f"| {cell['agent']} | {cell.get('preset')} | {device} | "
            f"{_fmt(peak, ' MB')} | {survived} | {cache_txt} |"
        )
    if proc_rows:
        add("## Gateway process (per privaite cell)")
        add("")
        add("Peak RSS is sampled from outside the gateway process (about once a "
            "second with `ps`), so a memory kill is observable rather than "
            "self-reported. The cache column reads the bench launcher's "
            "counter route: counts only, never keys or spans. Probe requests "
            "are included in the miss count (each pre-flight probe scans one "
            "synthetic leaf).")
        add("")
        add("| Agent | Preset | Onnx device | Peak RSS | Process | Detection cache |")
        add("|---|---|---|---|---|---|")
        lines.extend(proc_rows)
        add("")

    # Timing -------------------------------------------------------------------
    add("## Timing")
    add("")
    add("Every number here is measured on this machine, from the harness's own "
        "clocks: wall-clock is the agent CLI invocation start to exit, wire "
        "timestamps come from the recorder, and gateway latency comes from the "
        "timing tap. Scrubbing large, growing agent conversations is real work "
        "and shows up as real seconds; that is the cost of the protection, and "
        "it is reported here rather than hidden.")
    add("")
    add("### Task wall-clock vs direct baseline")
    add("")
    add("End-to-end duration of the same task, same agent, with and without "
        "PrivAiTe in the path. Added = privaite minus direct.")
    add("")
    add("| Agent | Preset | Direct (s) | Via PrivAiTe (s) | Added (s) | Added (%) | Notes |")
    add("|---|---|---|---|---|---|---|")
    for agent in MATRIX_AGENTS:
        direct = cell_of(agent, "direct", None)
        base = direct.get("duration_s") if direct else None
        for preset in presets:
            cell = cell_of(agent, "privaite", preset)
            if cell is not None and cell.get("invalid"):
                add(f"| {agent} | {preset} | {_fmt(base)} | invalid, not measured | - | - | "
                    f"{cell.get('invalid_reason')} |")
                continue
            if cell is None or cell.get("duration_s") is None:
                add(f"| {agent} | {preset} | {_fmt(base)} | pending live run | - | - | - |")
                continue
            dur = cell["duration_s"]
            if base is None:
                add(f"| {agent} | {preset} | ? | {dur} | - | - | direct baseline missing |")
                continue
            added = round(dur - base, 1)
            pct = round(100 * added / base) if base else None
            notes = []
            if cell.get("timed_out"):
                notes.append("cell hit the timeout; duration is the last attempt")
            elif cell.get("attempts", 1) > 1:
                notes.append(
                    f"duration is the completing attempt (of {cell['attempts']})"
                )
            if _is_pre_fix(cell):
                notes.append("pre-fix capture")
            add(f"| {agent} | {preset} | {base} | {dur} | {added:+} | {pct:+}% | "
                f"{'; '.join(notes) or 'clean run'} |")
    add("")
    add("### Per-turn added latency (from wire timestamps)")
    add("")
    add("Gap = time between consecutive provider-bound requests of the same "
        "conversation. A gap bundles provider generation, agent work and (on "
        "the privaite arm) the gateway scrub and restore, so the privaite minus "
        "direct difference is an estimate of per-turn gateway cost with agent "
        "and task held constant, not an exact scrub time. The exact number is "
        "the tap measurement below.")
    add("")
    add("| Agent | Preset | Direct median gap (s) | PrivAiTe median gap (s) | Added per turn (s) |")
    add("|---|---|---|---|---|")
    for agent in MATRIX_AGENTS:
        direct = cell_of(agent, "direct", None)
        base_gap = ((direct or {}).get("timing") or {}).get("gap_median_s")
        for preset in presets:
            cell = cell_of(agent, "privaite", preset)
            if cell is not None and cell.get("invalid"):
                add(f"| {agent} | {preset} | {_fmt(base_gap)} | invalid, not measured | - |")
                continue
            gap = ((cell or {}).get("timing") or {}).get("gap_median_s")
            if gap is None:
                add(f"| {agent} | {preset} | {_fmt(base_gap)} | pending live run | - |")
                continue
            if base_gap is None:
                add(f"| {agent} | {preset} | ? | {gap} | - |")
                continue
            add(f"| {agent} | {preset} | {base_gap} | {gap} | {round(gap - base_gap, 1):+} |")
    add("")
    add("### Exact per-request gateway latency (timing tap)")
    add("")
    add("On the privaite arm a second recorder sits in front of the gateway. "
        "Scrub latency per request = arrival behind the gateway minus arrival at "
        "the tap (same request, same clock); restore latency per response = "
        "stream completion at the tap minus completion behind the gateway.")
    add("")
    tap_rows = []
    for slot in matrix:
        if slot["arm"] != "privaite":
            continue
        cell = cell_of(slot["agent"], slot["arm"], slot["preset"])
        timing = (cell or {}).get("timing") or {}
        if cell is not None and timing.get("scrub_median_s") is not None:
            tap_rows.append(
                f"| {cell['agent']} | {cell.get('preset')} | "
                f"{timing['scrub_median_s']} | {timing.get('scrub_max_s', '?')} | "
                f"{_fmt(timing.get('restore_median_s'))} |"
            )
    if tap_rows:
        add("| Agent | Preset | Median scrub (s) | Max scrub (s) | Median restore (s) |")
        add("|---|---|---|---|---|")
        lines.extend(tap_rows)
    else:
        add("Pending live run: the tap was added after the existing captures were "
            "taken, so no cell has tap data yet. The next full run fills this "
            "table per preset automatically.")
    add("")

    # Context growth -----------------------------------------------------------
    add("## Context growth")
    add("")
    add("Both agents resend the entire growing conversation on every turn, so "
        "the request body (and with it the text the gateway must scrub) grows "
        "monotonically within a conversation. Token counts are estimated as "
        "bytes/4 and are approximate.")
    add("")
    add("| Agent | Arm | Turns | First body (KB) | Last body (KB) | ~Tokens last | Median gap (s) |")
    add("|---|---|---|---|---|---|---|")
    for slot in matrix:
        cell = cell_of(slot["agent"], slot["arm"], slot["preset"])
        if cell is None:
            continue
        timing = cell.get("timing") or {}
        if not timing.get("turns"):
            continue
        turns_txt = str(len(timing["turns"]))
        if timing.get("conversations", 1) > 1:
            turns_txt += f" (over {timing['conversations']} attempts)"
        add(f"| {cell['agent']} | {_arm_label(cell)} | {turns_txt} | "
            f"{_kb(timing['body_bytes_first'])} | {_kb(timing['body_bytes_last'])} | "
            f"~{timing['est_tokens_last']:,} | {_fmt(timing.get('gap_median_s'))} |")
    add("")
    for cell in cells:
        if cell["arm"] != "privaite":
            continue
        timing = cell.get("timing") or {}
        turns = timing.get("turns") or []
        if not turns:
            continue
        add(f"### Per turn: {cell['agent']} via PrivAiTe ({cell.get('preset')})")
        add("")
        add("| Turn | Request body (KB) | Gap since previous request (s) |")
        add("|---|---|---|")
        for turn in turns:
            if turn.get("restart"):
                gap_txt = "(new attempt)"
            else:
                gap_txt = _fmt(turn.get("gap_s")) if turn["turn"] > 1 else "-"
            add(f"| {turn['turn']} | {_kb(turn['bytes'])} | {gap_txt} |")
        direct = cell_of(cell["agent"], "direct", None)
        base_gap = ((direct or {}).get("timing") or {}).get("gap_median_s")
        if base_gap is not None:
            add("")
            add(f"For contrast, the {cell['agent']} direct arm's median gap on the "
                f"same task is {base_gap}s.")
        add("")

    # Cache --------------------------------------------------------------------
    add("## Cache behavior")
    add("")
    add("### Scrub cost model")
    add("")
    add("The gateway is stateless per request: it scrubs the full conversation "
        "it receives on every turn. With the detection cache off, per-turn "
        "scrub cost therefore grows roughly linearly with context size and the "
        "cumulative cost of a session is roughly quadratic in its final length "
        "(O(n) per turn, O(n^2) per session). The per-turn tables above are the "
        "measurement of that model: body size and privaite-arm gap rise "
        "together within a conversation while the direct arm stays flat. The "
        "fix for that repeat cost is PrivAiTe's opt-in detection cache, which "
        "caches the merged detection result per exact text leaf so a resent "
        "leaf skips the detectors entirely. It is not a roadmap item: it "
        "shipped alongside the gateway in PrivAiTe 0.4.0 and it is active in "
        "the onnx-auto-cache cells measured in this very document.")
    add("")
    add("### Provider prompt cache")
    add("")
    add("Providers price and speed up requests whose prefix matches a previous "
        "request. If the gateway rewrote placeholders inconsistently between "
        "turns, it would silently break that cache. Byte-prefix stability of "
        "consecutive provider-bound bodies is the offline proxy: the fraction "
        "of the previous body that the next body reproduces byte-for-byte.")
    add("")
    add("| Agent | Arm | Median shared prefix with previous request |")
    add("|---|---|---|")
    prefix_data: dict[tuple[str, str | None], float] = {}
    for slot in matrix:
        cell = cell_of(slot["agent"], slot["arm"], slot["preset"])
        frac = ((cell or {}).get("timing") or {}).get("prefix_shared_median")
        if cell is None or frac is None:
            continue
        prefix_data[(slot["agent"], slot["preset"])] = frac
        add(f"| {cell['agent']} | {_arm_label(cell)} | {frac:.1%} |")
    add("")
    for agent in MATRIX_AGENTS:
        base = prefix_data.get((agent, None))
        if base is None:
            continue
        for preset in presets:
            frac = prefix_data.get((agent, preset))
            if frac is None:
                continue
            if abs(frac - base) <= 0.05:
                add(f"- {agent} [{preset}]: prefix stability through the gateway "
                    f"({frac:.1%}) matches the direct arm ({base:.1%}); placeholders "
                    "are consistent across turns, so the gateway does not degrade "
                    "what prefix stability the client provides.")
            elif frac < base:
                add(f"- {agent} [{preset}]: prefix stability drops from {base:.1%} "
                    f"(direct) to {frac:.1%} through the gateway; this would break "
                    "the provider prompt cache and needs investigation (placeholder "
                    "determinism across turns).")
        if base < 0.5:
            add(f"- {agent}: stability is low on both arms, which is client "
                "behavior, not gateway behavior. In the captured bodies the first "
                "byte divergence between consecutive requests sits at the "
                "`cache_control` breakpoint marker the client moves between turns, "
                "so raw byte-prefix understates its token-level cache reuse; the "
                "provider counters below are the authoritative signal.")
    add("")
    add("Authoritative proof comes from the providers' own cache counters, which "
        "the recorder now extracts from captured response streams "
        "(`cache_read_input_tokens` for Anthropic, `cached_tokens` for the "
        "Responses API): a preserved cache shows the counter tracking the "
        "growing context on the privaite arm just as it does on direct.")
    add("")
    counter_rows = []
    for slot in matrix:
        cell = cell_of(slot["agent"], slot["arm"], slot["preset"])
        counters = ((cell or {}).get("timing") or {}).get("cache_counters") or {}
        for label, series in counters.items():
            series_txt = ", ".join("?" if v is None else f"{v:,}" for v in series)
            counter_rows.append(
                f"| {cell['agent']} | {_arm_label(cell)} | {label} | {series_txt} |"
            )
    if counter_rows:
        add("| Agent | Arm | Counter | Per-turn values |")
        add("|---|---|---|---|")
        lines.extend(counter_rows)
    else:
        add("Pending live run: the existing captures predate response recording, "
            "so no cache counters are available yet. The next full run fills "
            "this in per cell automatically.")
    add("")

    # Run details ----------------------------------------------------------
    add("## Run details")
    add("")
    add("`Files on wire` counts fixture files whose neutral marker string appears "
        "in the captured bodies: proof the agent actually read them, so a "
        "zero-leak row cannot hide behind an agent that skipped the sensitive "
        "files.")
    add("")
    add("| Agent | Arm | Requests recorded | Files on wire | Agent exit | Duration | Attempts | Ended by timeout | Notes |")
    add("|---|---|---|---|---|---|---|---|---|")
    for cell in cells:
        notes = "; ".join(cell.get("notes", [])) or "clean run"
        cov = cell.get("coverage")
        if cell.get("invalid"):
            cov_txt = "-"
            notes = f"INVALID CELL: {cell.get('invalid_reason')}; {notes}"
        else:
            cov_txt = f"{sum(cov.values())}/{len(cov)}"
            missing = [name for name, seen in cov.items() if not seen]
            if missing:
                notes += f"; not on wire: {', '.join(missing)}"
        add(
            f"| {cell['agent']} | {_arm_label(cell)} | {cell['requests']} | {cov_txt} | "
            f"{cell.get('exit_code')} | {_fmt(cell.get('duration_s'), 's')} | "
            f"{cell.get('attempts', '?')} | {'yes' if cell.get('timed_out') else 'no'} | {notes} |"
        )
    add("")

    # Displayed ------------------------------------------------------------
    add("## Displayed vs forwarded")
    add("")
    for cell in cells:
        if cell.get("invalid"):
            continue
        shown = [label for label, hit in cell["displayed"].items() if hit]
        desc = ", ".join(f"`{m}`" for m in shown) if shown else "none"
        restore_txt = ""
        if cell["arm"] == "privaite":
            restore_txt = (
                " RESTORE FAILURE: placeholder tokens appear in the user-visible transcript."
                if cell.get("unrestored_placeholders_shown")
                else " No placeholder token reached the user-visible transcript."
            )
        add(f"- {cell['agent']} / {_arm_label(cell)}: planted values visible in the "
            f"agent's own final output: {desc}.{restore_txt}")
    add("")

    # Provenance and caveats -------------------------------------------------
    add("## Provenance and caveats")
    add("")
    pre_fix = [c for c in cells if _is_pre_fix(c)]
    if pre_fix:
        add("- Captured cells and dates: "
            + "; ".join(
                f"{c['agent']} {_arm_label(c)} ({c.get('captured', 'unknown date')})"
                for c in cells
            )
            + ".")
        add(f"- Cells captured on or before {PRE_FIX_CUTOFF} predate the PrivAiTe fix "
            "that makes the Responses-API gateway scrub tool-output carriers "
            "(`custom_tool_call_output` and `function_call_output`). The codex "
            "privaite numbers above are the pre-fix behavior, kept as the honest "
            "record of what was measured; they will be regenerated by the next "
            "full live run and must not be quoted as current.")
    pending = [s for s in matrix if cell_of(s["agent"], s["arm"], s["preset"]) is None]
    if pending:
        add("- Pending cells (no capture yet): "
            + ", ".join(
                f"{s['agent']}/{s['arm']}" + (f"[{s['preset']}]" if s["preset"] else "")
                for s in pending
            )
            + ". They are filled by the maintainer's live matrix run; this "
            "document never extrapolates them.")
    if invalid_cells:
        add("- Invalid cells were run but failed the validity guard (pre-flight "
            "probe failure, no captured traffic, or a gateway-handled count "
            "below the recorder's capture count). No leak number exists for "
            "them anywhere in this document or in report.json; re-running the "
            "cell is the only way to fill them.")
    add("- Detection is best-effort: the `onnx` preset catches secrets and most PII "
        "but single-word or unusual names can slip through, and a different prompt "
        "or repo layout can change what the agent sends. Treat the privaite arm as "
        "a strong reduction, not a guarantee.")
    add("- One run per cell, live agents are nondeterministic: which files the agent "
        "reads (and therefore what can leak) varies between runs, and so do turn "
        "counts and durations.")
    add("- The scan is an exact substring match on the ground-truth values, so "
        "paraphrased or partially masked values do not count as leaks.")
    add("- Both arms relay the client's own OAuth token to the provider; the "
        "recorder logs bodies only, never headers.")
    add("- Onnx execution provider per cell: plain presets (`light`, `full`, `onnx`) "
        "keep the historical `pii.detectors.onnx.device: cpu` pin, dating from when "
        "`auto` selected CoreML on this machine and the gateway process was SIGKILLed "
        "mid-scrub of large agent bodies. The `onnx-auto*` variant cells run "
        "`device: auto` on a PrivAiTe build whose auto no longer selects CoreML, "
        "which is exactly what those cells verify (see the Gateway process table). "
        "Detection counts are unaffected by the execution provider.")
    add("- A privaite-arm leak can be a carrier the scrubber does not traverse rather "
        "than a detection miss. In the 2026-07-16 run Codex leaked because it reads "
        "files through its custom shell tool and the file contents arrived in "
        "`custom_tool_call_output` items, which the Responses gateway left unscrubbed "
        "while it did scrub `message` items. See `README.md` for the full diagnosis.")
    add("- The timing tap adds one localhost hop (sub-millisecond) to the privaite "
        "arm; it forwards bytes verbatim and does not touch the direct arm, so the "
        "arms stay comparable.")
    add("- Byte-prefix stability is a proxy for the provider prompt cache, which "
        "operates on tokens; the response cache counters are the ground truth "
        "once captured.")
    add("")
    RESULTS_PATH.write_text("\n".join(lines) + "\n")


def _selftest() -> None:
    """Validate scan / coverage / per_category / capture analysis with
    synthetic captures, no network and no agents, so the measurement math is
    pinned in CI."""
    import tempfile

    manifest = [
        {"value": "SEKRET123", "category": "secret", "label": "KEY", "source": ".env"},
        {"value": "Ada Lovelace", "category": "person", "label": "Ada", "source": "a.md"},
        {"value": "ada@x.io", "category": "email", "label": "ada@x.io", "source": "a.md"},
    ]
    marker = next(iter(FILE_MARKERS.values()))
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)

        # scan(): leak + coverage + placeholder evidence, response records and
        # junk lines skipped.
        path = base / "cap.jsonl"
        rows = [
            {"seq": 1, "ts": 100.0, "path": "/v1/messages",
             "body": json.dumps({"messages": [{"content": f"Ada Lovelace <ada@x.io> {marker}"}]})},
            {"seq": 1, "kind": "response", "ts_done": 101.0, "status": 200,
             "ttfb_s": 0.4, "upstream_s": 1.0,
             "response": 'data: {"usage":{"cache_read_input_tokens":10}}'},
            {"seq": 2, "ts": 110.0, "path": "/v1/messages",
             "body": json.dumps({"messages": [{"content": "KEY=<SECRET_1>"}]})},
            {"seq": 2, "kind": "response", "ts_done": 112.0, "status": 200,
             "ttfb_s": 0.5, "upstream_s": 1.5,
             "response": 'data: {"usage":{"cache_read_input_tokens":90}}'},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
        leaked, coverage, n, scrubbed = scan(path, manifest)
        assert n == 2, n
        assert leaked == {"KEY": False, "Ada": True, "ada@x.io": True}, leaked
        assert scrubbed is True, "placeholder <SECRET_1> should count as scrub evidence"
        first_file = next(iter(FILE_MARKERS))
        assert coverage[first_file] is True, coverage
        pc = per_category(manifest, leaked)
        assert pc == {"secret": (0, 1), "email": (1, 1), "person": (1, 1)}, pc

        # analyze_capture(): growth, gaps, restart detection, prefix
        # stability, cache counters, tap pairing.
        cap = base / "growth.jsonl"
        tap = base / "growth.tap.jsonl"
        bodies = ["A" * 1000, "A" * 1000 + "B" * 500, "C" * 600, "C" * 600 + "D" * 300]
        ts = [0.0, 10.0, 300.0, 312.0]
        tap_ts = [-3.0, 6.5, 296.0, 308.0]
        cap_rows = []
        tap_rows = []
        for i, (body, t) in enumerate(zip(bodies, ts)):
            cap_rows.append({"seq": i + 1, "ts": 1000.0 + t, "path": "/v1/responses", "body": body})
            cap_rows.append({"seq": i + 1, "kind": "response", "ts_done": 1000.0 + t + 2.0,
                             "status": 200, "ttfb_s": 0.5, "upstream_s": 2.0,
                             "response": f'{{"usage":{{"cached_tokens":{100 * (i + 1)}}}}}'})
            tap_rows.append({"seq": i + 1, "ts": 1000.0 + tap_ts[i], "path": "/v1/responses",
                             "body": body})
            tap_rows.append({"seq": i + 1, "kind": "response", "ts_done": 1000.0 + t + 2.5,
                             "status": 200, "ttfb_s": 0.5, "upstream_s": 2.0, "response": ""})
        # a count_tokens side call must not pollute turn analysis
        cap_rows.append({"seq": 99, "ts": 1000.5, "path": "/v1/messages/count_tokens", "body": "tiny"})
        cap.write_text("\n".join(json.dumps(r) for r in cap_rows) + "\n")
        tap.write_text("\n".join(json.dumps(r) for r in tap_rows) + "\n")
        info = analyze_capture(cap, tap)
        assert info["requests"] == 5, info["requests"]
        assert len(info["turns"]) == 4
        assert info["conversations"] == 2, "body shrink must register as a restart"
        assert info["turns"][2].get("restart") is True
        assert info["turns"][1]["gap_s"] == 10.0 and info["turns"][3]["gap_s"] == 12.0
        assert info["gap_median_s"] == 11.0, info["gap_median_s"]
        assert info["body_bytes_first"] == 1000 and info["body_bytes_last"] == 900
        # turn 2 reproduces all of turn 1 as its prefix
        assert info["turns"][1]["prefix_shared"] == 1.0
        assert info["cache_counters"]["openai cached_tokens"] == [100, 200, 300, 400]
        # tap deltas: 3.0, 3.5, 4.0, 4.0 -> median 3.75; restore 0.5 everywhere
        assert info["scrub_s"] == [3.0, 3.5, 4.0, 4.0], info["scrub_s"]
        assert info["scrub_median_s"] == 3.75, info["scrub_median_s"]
        assert info["scrub_max_s"] == 4.0
        assert info["restore_median_s"] == 0.5, info["restore_median_s"]
        # without a tap the same analysis still works (legacy captures)
        info2 = analyze_capture(cap, base / "missing.tap.jsonl")
        assert "scrub_median_s" not in info2 and info2["gap_median_s"] == 11.0

        # merge_stats(): sessions collapse into one per-cell summary
        merged = merge_stats(
            {
                "enabled": True,
                "sessions": {
                    "a...": {"requests": 2, "total_pii": 5, "by_type": {"PERSON": 3, "EMAIL_ADDRESS": 2}},
                    "b...": {"requests": 1, "total_pii": 1, "by_type": {"PERSON": 1}},
                },
            }
        )
        assert merged == {
            "requests": 3,
            "total_pii": 6,
            "by_type": {"PERSON": 4, "EMAIL_ADDRESS": 2},
        }, merged
        assert merge_stats(None) is None and merge_stats({"enabled": False}) is None

        # Validity guard: a privaite cell whose recorder capture exceeds the
        # gateway's own handled count must classify as invalid, never as a
        # leak number (this exact case once published a false 20/24).
        reason = classify_validity("privaite", 16, 0)
        assert reason is not None and "did not traverse the gateway" in reason, reason
        assert classify_validity("privaite", 0, 0) == "no provider-bound traffic captured"
        assert classify_validity("privaite", 0, None) == "no provider-bound traffic captured"
        assert classify_validity("privaite", 8, 8) is None
        # extra handled requests (scrub-blocked before forwarding) are fine
        assert classify_validity("privaite", 8, 9) is None
        unverifiable = classify_validity("privaite", 8, None)
        assert unverifiable is not None and "unverifiable" in unverifiable, unverifiable
        assert classify_validity("direct", 6, None) is None
        assert classify_validity("direct", 0, None) == "no provider-bound traffic captured"
        probe_reason = classify_validity("privaite", 8, 8, "gateway never saw the probe")
        assert probe_reason is not None and probe_reason.startswith("pre-flight probe failed")

        # gateway_handled_count reads only the gateway's own handled lines,
        # ignoring health checks and uvicorn access noise.
        glog = base / "gateway.log"
        glog.write_text(
            '{"time": "x", "logger": "privaite.gateway", "message": "gateway /v1/responses -> 200 (restore)"}\n'
            'INFO:     127.0.0.1:65007 - "GET /health HTTP/1.1" 200 OK\n'
            'INFO:     127.0.0.1:64374 - "POST /v1/messages?beta=true HTTP/1.1" 200 OK\n'
            '{"time": "x", "logger": "privaite.gateway", "message": "gateway /v1/messages -> 200 (passthrough)"}\n'
        )
        assert gateway_handled_count(glog) == 2, gateway_handled_count(glog)
        assert gateway_handled_count(base / "missing.log") is None
        assert gateway_handled_count(None) is None

        # Probe records are plumbing: counted apart, never scanned, never in
        # the timing analysis.
        probe_cap = base / "probe.jsonl"
        probe_cap.write_text(
            "\n".join(
                json.dumps(r)
                for r in [
                    {"seq": 1, "ts": 1.0, "kind": "probe", "path": "/v1/responses",
                     "absorbed": True},
                    {"seq": 2, "ts": 2.0, "path": "/v1/responses",
                     "body": "Ada Lovelace"},
                    {"seq": 2, "kind": "response", "ts_done": 3.0, "status": 200,
                     "ttfb_s": 0.1, "upstream_s": 0.5, "response": ""},
                ]
            )
            + "\n"
        )
        assert probe_record_count(probe_cap) == 1
        assert captured_request_count(probe_cap) == 1
        leaked_p, _, n_p, _ = scan(probe_cap, manifest)
        assert n_p == 1 and leaked_p["Ada"] is True, (n_p, leaked_p)
        assert analyze_capture(probe_cap, None)["requests"] == 1
        assert probe_record_count(base / "missing.jsonl") == 0
        assert captured_request_count(base / "missing.jsonl") == 0

        # cell naming and legacy fallback are pure path math
        assert cell_name("claude", "direct", None) == "claude_direct"
        assert cell_name("codex", "privaite", "light") == "codex_privaite_light"

        # Preset variants: plain names keep the historical knobs (cpu, no
        # cache); the -auto variants exercise device auto, with and without
        # the detection cache, and always resolve to a real engine preset.
        assert preset_knobs("onnx") == ("onnx", "cpu", False)
        assert preset_knobs("light") == ("light", "cpu", False)
        assert preset_knobs("onnx-auto") == ("onnx", "auto", False)
        assert preset_knobs("onnx-auto-cache") == ("onnx", "auto", True)
        yaml_auto = gateway_yaml("onnx-auto")
        assert 'device: "auto"' in yaml_auto and 'preset: "onnx"' in yaml_auto
        assert "detection_cache: {enabled: false}" in yaml_auto
        yaml_cache = gateway_yaml("onnx-auto-cache")
        assert 'device: "auto"' in yaml_cache and 'preset: "onnx"' in yaml_cache
        assert "detection_cache: {enabled: true}" in yaml_cache
        assert 'device: "cpu"' in gateway_yaml("onnx")

        # A partial run keeps every already-captured preset in the report.
        assert merged_presets(["onnx"], {"light"}) == ["light", "onnx"]
        assert merged_presets(["light", "full", "onnx"], set()) == ["light", "full", "onnx"]
        assert merged_presets(["onnx"], set()) == ["onnx"]
        assert merged_presets(["custom"], {"onnx"}) == ["onnx", "custom"]
    print("run selftest ok")


def main() -> None:
    global AGENT_TIMEOUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="check the measurement math, then exit")
    parser.add_argument("--agents", default=",".join(MATRIX_AGENTS))
    parser.add_argument("--arms", default="direct,privaite")
    parser.add_argument(
        "--presets",
        default=",".join(DEFAULT_PRESETS),
        help="PrivAiTe presets for the privaite arm (the direct arm has none)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=f"per-cell agent timeout in seconds (default {AGENT_TIMEOUT}, "
        "or the AGENT_WORKFLOW_TIMEOUT env var); a timeout is recorded, not fatal",
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="skip the live agent runs, rescan the captures already on disk "
        "and regenerate report.json and RESULTS.md",
    )
    args = parser.parse_args()
    if args.selftest:
        _selftest()
        return
    if args.timeout:
        AGENT_TIMEOUT = args.timeout
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]

    if not args.rescan and "privaite" in arms and not PRIVAITE_PYTHON.exists():
        raise SystemExit(f"PrivAiTe venv python not found at {PRIVAITE_PYTHON}")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = gen_fixture.generate(OUT / "manifest.json")
    print(f"{len(manifest)} ground-truth values planted")

    # Partial runs (a subset of agents/arms/presets, or --rescan) merge into
    # the existing report so RESULTS.md always covers every cell with a
    # capture. Pre-matrix reports carried no preset: their privaite cells were
    # onnx runs.
    previous: dict[tuple[str, str, str | None], dict] = {}
    report_path = OUT / "report.json"
    if report_path.exists():
        old = json.loads(report_path.read_text())
        for cell in old.get("cells", []):
            preset = cell.get("preset") or ("onnx" if cell["arm"] == "privaite" else None)
            cell["preset"] = preset
            cell.setdefault("captured", old.get("date"))
            previous[(cell["agent"], cell["arm"], preset)] = cell

    if not args.rescan:
        for agent in agents:
            for arm in arms:
                for preset in presets if arm == "privaite" else [None]:
                    label = f"{agent} / {arm}" + (f" / {preset}" if preset else "")
                    print(f"=== {label} ===", flush=True)
                    previous[(agent, arm, preset)] = run_cell(agent, arm, preset)

    report_presets = merged_presets(
        presets,
        {p for (_a, arm_key, p) in previous if arm_key == "privaite" and p},
    )
    matrix = [
        {"agent": agent, "arm": arm, "preset": preset}
        for agent in MATRIX_AGENTS
        for arm, preset in [("direct", None)] + [("privaite", p) for p in report_presets]
    ]

    cells = []
    for slot in matrix:
        agent, arm, preset = slot["agent"], slot["arm"], slot["preset"]
        cell = previous.get((agent, arm, preset))
        # A cell whose own metadata postdates the pre-fix cutoff must never
        # inherit a legacy capture: its own capture file is the only truth.
        allow_legacy = cell is None or _is_pre_fix(cell)
        files = cell_files(agent, arm, preset, allow_legacy=allow_legacy)
        if cell is None and not files["capture"].exists():
            continue
        cell = cell or {
            "agent": agent,
            "arm": arm,
            "preset": preset,
            "attempts": 0,
            "notes": [],
        }
        leaked, coverage, n_requests, scrubbed = scan(files["capture"], manifest)
        # Validity guard: a privaite leak count is only publishable when the
        # gateway itself proves it handled at least as many requests as the
        # recorder captured behind it (probes excluded). The gateway's own log
        # and its /stats snapshot are independent signals; take the strongest.
        gateway_handled = None
        if arm == "privaite":
            probes = probe_record_count(files["capture"])
            handled_log = gateway_handled_count(files["gateway_log"])
            if handled_log is not None:
                gateway_handled = max(handled_log - probes, 0)
            stats_requests = (cell.get("stats") or {}).get("requests")
            if isinstance(stats_requests, int):
                stats_handled = max(stats_requests - probes, 0)
                gateway_handled = (
                    stats_handled
                    if gateway_handled is None
                    else max(gateway_handled, stats_handled)
                )
        invalid_reason = classify_validity(
            arm, n_requests, gateway_handled, cell.get("probe_failure")
        )
        if invalid_reason is not None:
            cell.update(
                invalid=True,
                invalid_reason=invalid_reason,
                requests=n_requests,
                gateway_handled=gateway_handled,
                leaked=None,
                coverage=None,
                scrubbed_evidence=None,
                displayed=None,
                per_category=None,
                timing=None,
            )
            print(f"{cell_name(agent, arm, preset)}: INVALID, not measured ({invalid_reason})")
            cells.append(cell)
            continue
        shown_text = files["out"].read_text() if files["out"].exists() else ""
        cell.update(
            invalid=False,
            gateway_handled=gateway_handled,
            leaked=leaked,
            coverage=coverage,
            scrubbed_evidence=scrubbed,
            displayed={e["label"]: e["value"] in shown_text for e in manifest},
            unrestored_placeholders_shown=any(tok in shown_text for tok in PLACEHOLDER_TOKENS),
            requests=n_requests,
            per_category=per_category(manifest, leaked),
            timing=analyze_capture(files["capture"], files["tap"]),
        )
        hit = sum(leaked.values())
        print(
            f"{cell_name(agent, arm, preset)}: {n_requests} requests, "
            f"{sum(coverage.values())}/{len(coverage)} files on wire, "
            f"{hit}/{len(manifest)} values leaked"
        )
        cells.append(cell)

    report = {
        "date": date.today().isoformat(),
        "prompt": PROMPT,
        "timeout_s": AGENT_TIMEOUT,
        "presets": report_presets,
        "matrix": matrix,
        "corpus": corpus_stats(),
        "manifest": [
            {"label": e["label"], "category": e["category"], "source": e["source"]}
            for e in manifest
        ],
        "cells": cells,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    # write_results needs categories per label: reattach the full manifest
    # (report.json itself keeps labels only, never the generated values).
    report["manifest"] = manifest
    write_results(report)
    print(f"report: {report_path}")
    print(f"results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
