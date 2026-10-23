"""Run systems over corpora and store raw outcomes (offsets and booleans only).

    .venvs/harness/bin/python -m pbench.run [--systems a,b] [--corpora x,y]
                                            [--suite text|structured|all] [--limit N]

Each system runs in its own worker process (`pbench.worker`) inside its own
venv, one system at a time so latencies never compete for the CPU. Outcomes go to
results/runs/<system>/: `<corpus>.jsonl` (per document: id, sha256 of the text,
masked spans as offsets, milliseconds, error) and `structured.jsonl` (per request:
which gold values leaked, whether tool-call JSON still parses). No corpus text is
written there, so the directory is committed and the report can be rebuilt from
it. A run resumes: documents already scored with the same text hash are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import select
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pbench.corpus import read_jsonl
from pbench.registry import BY_NAME, SYSTEMS
from pbench.sources import BUILDERS, CORPORA
from pbench.structured import args_valid, item_sha, leaks, not_forwarded, structure_kept

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"
TIMEOUT_S = 900  # one document; a system that hangs longer is restarted


class Worker:
    def __init__(self, system: str) -> None:
        self.system = system
        # Kept open across worker restarts; closed in close().
        self.log = open(Path.home() / ".cache/privaite-bench/logs" / f"worker_{system}.log", "a")  # noqa: SIM115
        self.start()

    def start(self) -> None:
        env_name = BY_NAME[self.system].env
        python = REPO / ".venvs" / env_name / "bin" / "python"
        env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen([str(python), "-m", "pbench.worker", self.system],
                                     cwd=REPO, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True,
                                     encoding="utf-8")
        ready = self._read(timeout=1800)
        if not ready or not ready.get("ready"):
            raise RuntimeError(f"{self.system}: worker failed to start (see {self.log.name})")

    def _read(self, timeout: float) -> dict | None:
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            return None
        line = self.proc.stdout.readline()
        return json.loads(line) if line else None

    def ask(self, request: dict) -> dict:
        try:
            self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            reply = self._read(TIMEOUT_S)
        except (BrokenPipeError, OSError, ValueError):
            reply = None  # the worker died before or while answering
        if reply is None:
            self.proc.kill()
            self.proc.wait()
            self.start()
            return {"error": f"worker crashed or exceeded {TIMEOUT_S}s; restarted"}
        return reply

    def close(self) -> None:
        try:
            self.proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=60)
        except Exception:  # noqa: BLE001  (any shutdown failure ends in a kill)
            self.proc.kill()
        self.log.close()


def _done(path: Path, required: str | None = None) -> dict[str, str]:
    """id -> fingerprint of the outcomes that need no rerun: not failed, and
    carrying `required` when an older run did not record that field yet."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if "error" not in rec and (required is None or required in rec):
            out[rec["id"]] = rec.get("sha256", "")
    return out


def machine() -> dict:
    def sysctl(key: str) -> str:
        try:
            return subprocess.run(["sysctl", "-n", key], check=False, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    return {"platform": platform.platform(), "cpu": sysctl("machdep.cpu.brand_string"),
            "memory_gb": round(int(sysctl("hw.memsize") or 0) / 2**30),
            "python": platform.python_version()}


def _drive(worker: Worker, todo: list, request, record, f, label: str, every: int) -> None:
    """Send each item to the worker in order and write its record as soon as it
    is back, so an interrupted run keeps everything done so far."""
    t0 = time.time()
    for i, item in enumerate(todo, 1):
        f.write(json.dumps(record(item, worker.ask(request(item)))) + "\n")
        f.flush()
        if i % every == 0 or i == len(todo):
            print(f"  {label}: {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


def run_text(worker: Worker, system: str, corpus: str, limit: int | None) -> None:
    all_docs = read_jsonl(CORPORA / f"{corpus}.jsonl")
    docs = all_docs[:limit]
    path = RUNS / system / f"{corpus}.jsonl"
    done = _done(path)
    todo = [d for d in docs if done.get(d.id) != d.sha256]
    current = {d.id for d in all_docs}
    lines = path.read_text().splitlines() if path.exists() else []
    stale = [ln for ln in lines if json.loads(ln)["id"] not in current]
    if not todo and not stale:
        return
    # Rewrite without the records about to be redone, nor records of documents
    # that are no longer in the corpus (a reselection must not leave orphans).
    keep = [ln for ln in lines if json.loads(ln)["id"] in current - {d.id for d in todo}]

    def record(d, reply: dict) -> dict:
        rec = {"id": d.id, "sha256": d.sha256}
        if "error" in reply:
            rec["error"] = reply["error"]
        else:
            rec["spans"] = reply["spans"]
            rec["ms"] = round(reply["ms"], 3)
        return rec

    with path.open("w") as f:
        for ln in keep:
            f.write(ln + "\n")
        _drive(worker, todo, lambda d: {"op": "text", "text": d.text, "lang": d.lang}, record,
               f, f"{system} {corpus}", 25)


def run_structured(worker: Worker, system: str, limit: int | None) -> None:
    all_items = [json.loads(ln) for ln in (CORPORA / "structured.jsonl").read_text().splitlines()]
    items = all_items[:limit]
    path = RUNS / system / "structured.jsonl"
    done = _done(path, required="structure_kept")
    todo = [it for it in items if done.get(it["id"]) != item_sha(it)]
    current = {it["id"] for it in all_items}
    lines = path.read_text().splitlines() if path.exists() else []
    stale = [ln for ln in lines if json.loads(ln)["id"] not in current]
    if not todo and not stale:
        return
    keep = [ln for ln in lines if json.loads(ln)["id"] in current - {it["id"] for it in todo}]

    def record(it: dict, reply: dict) -> dict:
        rec = {"id": it["id"], "sha256": item_sha(it)}
        if "error" in reply:
            rec["error"] = reply["error"]
        else:
            out = reply["messages"]
            rec["blocked"] = isinstance(out, dict) and "blocked" in out
            rec["failed_closed"] = isinstance(out, dict) and "failed_closed" in out
            rec["leaked"] = leaks(it, out)
            rec["args_valid"] = None if not_forwarded(out) else args_valid(out)
            rec["structure_kept"] = structure_kept(it["messages"], out)
            rec["ms"] = round(reply["ms"], 3)
        return rec

    with path.open("w") as f:
        for ln in keep:
            f.write(ln + "\n")
        _drive(worker, todo, lambda it: {"op": "payload", "messages": it["messages"], "lang": it["lang"]},
               record, f, f"{system} structured", 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", default=",".join(s.name for s in SYSTEMS))
    parser.add_argument("--corpora", default=",".join(BUILDERS))
    parser.add_argument("--suite", choices=("text", "structured", "all"), default="all")
    parser.add_argument("--limit", type=int, default=None, help="first N items only (smoke test)")
    args = parser.parse_args()
    for name in args.systems.split(","):
        spec = BY_NAME[name]
        want_text = spec.text and args.suite in ("text", "all")
        want_struct = spec.payload and args.suite in ("structured", "all")
        if not (want_text or want_struct):
            continue
        (RUNS / name).mkdir(parents=True, exist_ok=True)
        print(f"{name}: starting worker ({spec.env})", flush=True)
        worker = Worker(name)
        try:
            info = worker.ask({"op": "info"}).get("info", {})
            (RUNS / name / "info.json").write_text(json.dumps({
                "system": name, "title": spec.title, "started": datetime.now(UTC).isoformat(timespec="seconds"),
                "machine": machine(), **info}, indent=1, ensure_ascii=False) + "\n")
            if want_text:
                for corpus in args.corpora.split(","):
                    run_text(worker, name, corpus, args.limit)
            if want_struct:
                run_structured(worker, name, args.limit)
        finally:
            worker.close()


if __name__ == "__main__":
    sys.exit(main())
