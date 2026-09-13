"""Run each candidate offline in a fresh process with a wall-clock deadline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.render_kiji_privy import ORDER


def run_one(command: list[str], report: Path, timeout: float) -> dict:
    """Keep failure distinct from zero leaks; discard this run's partial report."""
    if report.exists():
        raise FileExistsError("Refusing to reuse an existing report")
    env = os.environ | {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    }
    start = time.perf_counter()
    with report.with_suffix(".log").open("w") as log:
        try:
            subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )
            if not report.exists():
                status = "missing_report"
            else:
                json.loads(report.read_text())
                status = "complete"
        except subprocess.TimeoutExpired:
            status = "timed_out"
        except subprocess.CalledProcessError as error:
            status = f"failed_exit_{error.returncode}"
        except json.JSONDecodeError:
            status = "invalid_report"
    if status != "complete":
        report.unlink(missing_ok=True)
    return {
        "status": status,
        "watchdog_seconds": timeout,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solutions", choices=ORDER, nargs="+", default=list(ORDER))
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/kiji_privy_replay")
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if len(set(args.solutions)) != len(args.solutions):
        parser.error("--solutions must not contain duplicates")
    # An existing directory could contain stale successful reports from a failed retry.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outcomes = []
    for solution in args.solutions:
        print(f"Starting {solution} (deadline: {args.timeout:g}s)", flush=True)
        report = args.output_dir / f"{solution}.json"
        outcome = run_one(
            [
                sys.executable,
                "-m",
                "scripts.privy.evaluate",
                "--solution",
                solution,
                "--output",
                str(report),
            ],
            report,
            args.timeout,
        )
        outcomes.append({"solution": solution, **outcome})
        (args.output_dir / "run_status.json").write_text(
            json.dumps(outcomes, indent=2) + "\n"
        )
        print(f"{solution}: {outcome['status']}", flush=True)
    if any(row["status"] != "complete" for row in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
