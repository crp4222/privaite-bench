"""Replay synthetic log/URI regressions locally, without an agent or provider.

Run from the bench root with PrivAiTe's venv. PRIVAITE_PATH can point to an
earlier checkout to run the exact same cases against a baseline. Only counts
and timings are written; the generated fixture and credential values stay in
a temporary directory which is removed at exit.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import re
import statistics
import tempfile
import time
from pathlib import Path

from benchlib.paths import ensure_privaite_path

ensure_privaite_path()

from privaite.config.schema import PIIConfig
from privaite.pii.engine import PIIEngine

from agent_workflow.gen_big_fixture import build


def cases(root: Path) -> list[tuple[str, str, dict[str, str]]]:
    with contextlib.redirect_stdout(io.StringIO()):
        build(root / "fixture", root / "manifest.json")
    log = (root / "fixture/logs/ingest_batch.log").read_text()
    lines = log.splitlines()
    first = next(i for i, line in enumerate(lines) if "presented_key=" in line)
    secrets = {}
    for field in ("presented_key", "smtp_secret"):
        match = re.search(field + r"=([^\s]+)", log)
        assert match is not None
        secrets[field] = match.group(1)
    password = "SyntheticPass_OnlyForTesting!"
    return [
        ("preceded_log_excerpt", "\n".join(lines[first - 1:first + 6]), secrets),
        ("full_ingest_log", log, secrets),
        ("connection_uri", f"postgresql://service:{password}@db.example.invalid:5432/app",
         {"uri_password": password}),
    ]


async def evaluate(preset: str, inputs: list, repeats: int) -> list[dict]:
    engine = PIIEngine(PIIConfig(
        preset=preset,
        anonymization={"entity_overrides": {
            "SECRET": {"method": "redact"}, "CREDIT_CARD": {"method": "mask"},
        }},
        detection_cache={"enabled": False},
    ))
    await engine.initialize()
    rows = []
    try:
        await engine.process_request([{"role": "user", "content": "Warm up the local detector."}])
        for name, text, secrets in inputs:
            runs = []
            for _ in range(repeats):
                start = time.perf_counter()
                out, mapping = await engine.process_request([{"role": "tool", "content": text}])
                elapsed = time.perf_counter() - start
                output = out[0]["content"]
                originals = list(mapping.get_all_fakes().values())
                runs.append({
                    "seconds": round(elapsed, 4),
                    "remaining_occurrences": {k: output.count(v) for k, v in secrets.items()},
                    "remaining_eight_char_fragments": {
                        k: sum(v[i:i + 8] in output for i in range(len(v) - 7))
                        for k, v in secrets.items()
                    },
                    "reversible_secrets": {
                        k: any(v in original for original in originals) for k, v in secrets.items()
                    },
                })
            rows.append({
                "preset": preset, "case": name, "input_bytes": len(text.encode()),
                "original_occurrences": {k: text.count(v) for k, v in secrets.items()},
                "median_seconds": round(statistics.median(r["seconds"] for r in runs), 4),
                "runs": runs,
            })
    finally:
        await engine.shutdown()
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    logging.disable(logging.CRITICAL)
    with tempfile.TemporaryDirectory(prefix="privaite-secret-bench-") as directory:
        inputs = cases(Path(directory))
        rows = []
        for preset in ("light", "onnx"):
            rows.extend(await evaluate(preset, inputs, args.repeats))
    report = {
        "source": args.source_label, "repeats": args.repeats,
        "detection_cache": False, "request_local_window_deduplication": True,
        "upstream_requests": 0, "measurements": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        leaks = sum(sum(r["remaining_occurrences"].values()) for r in row["runs"])
        print(f"{row['preset']} {row['case']}: {row['median_seconds']}s median; "
              f"{leaks} surviving credential occurrences across {args.repeats} runs")


if __name__ == "__main__":
    asyncio.run(main())
