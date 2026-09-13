"""Compare Kiji candidates with the current PrivAiTe engine on identical inputs.

Run one solution per process so peak RSS and model lifetime are comparable.
All requests stay local. Serialized reports contain no input/output text.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import platform
import resource
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from benchlib.paths import RESULTS, SOLUTIONS, bootstrap, privaite_path

bootstrap()

from privaite.config.schema import PIIConfig
from privaite.pii.engine import PIIEngine

from agent_workflow.bench_structured_secrets import cases as regression_cases
from scripts.privy.metrics import coverage, summarize, text_recall
from solutions.compare import load_clean, load_corpus
from solutions.kiji import KijiDetector

CHOICES = ("light", "onnx", "kiji", "kiji-argmax", "kiji-presidio", "onnx-kiji")


class RecordingEngine(PIIEngine):
    """Observe merged spans during the actual scrub, without a second detection."""

    last_spans: list

    async def _detect_all(self, text: str, language: str = "en"):
        entities = await super()._detect_all(text, language)
        self.last_spans = [[e.start, e.end, e.entity_type] for e in entities]
        return entities


class EngineRunner:
    def __init__(self, solution: str):
        self.solution = solution
        self.engine = None
        self.language = None
        self.initialization_seconds = 0.0
        self.kiji_artifacts = None

    async def get(self, lang: str) -> RecordingEngine:
        if self.engine is not None and self.language == lang:
            return self.engine
        await self.close()
        gc.collect()
        start = time.perf_counter()
        is_kiji_only = self.solution in ("kiji", "kiji-argmax")
        cfg = PIIConfig(
            preset="light" if self.solution == "light" else "onnx",
            anonymization={
                "entity_overrides": {
                    "SECRET": {"method": "redact"},
                    "CREDIT_CARD": {"method": "mask"},
                }
            },
            detection_cache={"enabled": False},
        )
        cfg.detectors.presidio.languages = [lang] if lang == "en" else [lang, "en"]
        cfg.detectors.presidio.enabled = not is_kiji_only
        cfg.detectors.onnx.enabled = self.solution in ("onnx", "onnx-kiji")
        cfg.detectors.onnx.device = "cpu"
        engine = RecordingEngine(cfg)
        if not is_kiji_only:
            await engine.initialize()
        if "kiji" in self.solution:
            detector = KijiDetector(
                threshold=0.0 if self.solution == "kiji-argmax" else 0.5
            )
            await detector.initialize()
            engine.detectors.append(detector)
            self.kiji_artifacts = detector.artifacts
        # The raw-model row uses the same scrubbing API but is deliberately a
        # benchmark injection, not a supported zero-detector product config.
        engine._ready = True
        self.initialization_seconds += time.perf_counter() - start
        await engine.process_request(
            [{"role": "user", "content": "Review the incident report."}]
        )
        self.engine, self.language = engine, lang
        return engine

    async def close(self) -> None:
        if self.engine is not None:
            await self.engine.shutdown()
            self.engine = None


def inputs() -> list[dict]:
    privy_path = SOLUTIONS / "_privy_corpus.json"
    payload = privy_path.read_bytes()
    manifest = json.loads((RESULTS / "privy_manifest.json").read_text())
    if hashlib.sha256(payload).hexdigest() != manifest["corpus_sha256"]:
        raise ValueError("Privy corpus differs from its selection manifest")
    docs = [{**d, "corpus": "privy"} for d in json.loads(payload)]
    for doc in load_corpus():
        spans = []
        for value, entity_type in doc["expected"].items():
            start = doc["text"].find(value)
            if start < 0 or not value:
                raise ValueError("AI4Privacy label absent from source text")
            while start >= 0:
                spans.append([start, start + len(value), entity_type])
                start = doc["text"].find(value, start + len(value))
        docs.append({**doc, "corpus": "ai4privacy", "format": "text", "spans": spans})
    return sorted(docs, key=lambda d: (d["lang"], d["corpus"], str(d["id"])))


def latency_stats(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "requests": len(values),
        "mean_ms": round(statistics.mean(values) * 1000, 2),
        "median_ms": round(statistics.median(values) * 1000, 2),
        "p95_ms": round(ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)] * 1000, 2),
    }


def source_provenance() -> dict:
    return {
        "privaite_source": subprocess.check_output(
            ["git", "-C", privaite_path(), "rev-parse", "HEAD"], text=True
        ).strip(),
        "benchmark_source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "benchmark_worktree_dirty": bool(
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    "scripts/privy",
                    "solutions/kiji.py",
                ],
                text=True,
            )
        ),
        "benchmark_files_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "scripts/privy/build_corpus.py",
                "scripts/privy/metrics.py",
                "scripts/privy/evaluate.py",
                "solutions/kiji.py",
            )
        },
    }


async def run(solution: str, output: Path, repeats: int) -> None:
    logging.disable(logging.CRITICAL)
    runner = EngineRunner(solution)
    docs = inputs()
    rows, clean, probes, regressions = [], [], [], []
    try:
        for index, doc in enumerate(docs):
            engine = await runner.get(doc["lang"])
            start = time.perf_counter()
            result, _ = await engine.process_request(
                [{"role": "tool", "content": doc["text"]}]
            )
            elapsed = time.perf_counter() - start
            text_out = result[0]["content"]
            rows.append(
                {
                    "id": doc["id"],
                    "corpus": doc["corpus"],
                    "format": doc["format"],
                    "lang": doc["lang"],
                    "bytes": len(doc["text"].encode()),
                    "seconds": round(elapsed, 6),
                    "changed": text_out != doc["text"],
                    "coverage": coverage(doc["text"], doc["spans"], engine.last_spans),
                    "literal": text_recall(doc["text"], text_out, doc["spans"]),
                }
            )
            # Deterministic payload probe, independent of detector success.
            if index % 20 == 0:
                body = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": doc["text"]}],
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "save",
                                    "arguments": json.dumps({"note": doc["text"]}),
                                },
                            }
                        ],
                    },
                ]
                scrubbed, _ = await engine.process_request(body)
                mm = scrubbed[0]["content"][0]["text"]
                tool = json.loads(
                    scrubbed[1]["tool_calls"][0]["function"]["arguments"]
                )["note"]
                probes.append(
                    {
                        "id": doc["id"],
                        "multimodal_matches_flat": mm == text_out,
                        "tool_matches_flat": tool == text_out,
                    }
                )
            if (index + 1) % 50 == 0:
                print(f"{solution}: {index + 1}/{len(docs)} documents", flush=True)
        for index, doc in enumerate(
            sorted(load_clean(), key=lambda d: d.get("lang", "en"))
        ):
            engine = await runner.get(doc.get("lang", "en"))
            result, _ = await engine.process_request(
                [{"role": "tool", "content": doc["text"]}]
            )
            clean.append(
                {
                    "index": index,
                    "changed": result[0]["content"] != doc["text"],
                    "spans": len(engine.last_spans),
                    "coverage": coverage(doc["text"], [], engine.last_spans),
                }
            )
        engine = await runner.get("en")
        with tempfile.TemporaryDirectory(prefix="privy-regressions-") as temp:
            for name, text, secrets in regression_cases(Path(temp)):
                samples = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    result, mapping = await engine.process_request(
                        [{"role": "tool", "content": text}]
                    )
                    elapsed = time.perf_counter() - start
                    text_out = result[0]["content"]
                    originals = mapping.get_all_fakes().values()
                    samples.append(
                        {
                            "seconds": round(elapsed, 6),
                            "remaining_occurrences": {
                                k: text_out.count(v) for k, v in secrets.items()
                            },
                            "remaining_eight_char_fragments": {
                                k: sum(
                                    v[i : i + 8] in text_out for i in range(len(v) - 7)
                                )
                                for k, v in secrets.items()
                            },
                            "reversible_secrets": {
                                k: any(v in o for o in originals)
                                for k, v in secrets.items()
                            },
                        }
                    )
                regressions.append(
                    {
                        "case": name,
                        "bytes": len(text.encode()),
                        "original_occurrences": {
                            k: text.count(v) for k, v in secrets.items()
                        },
                        "latency": latency_stats([s["seconds"] for s in samples]),
                        "runs": samples,
                    }
                )
    finally:
        await runner.close()
    corpora = {}
    for corpus in ("privy", "ai4privacy"):
        selected = [r for r in rows if r["corpus"] == corpus]
        dimensions = {}
        for dimension in ("format", "lang"):
            grouped = defaultdict(list)
            for row in selected:
                grouped[row[dimension]].append(row)
            dimensions[dimension] = {
                key: summarize(value) for key, value in grouped.items()
            }
        corpora[corpus] = {
            **summarize(selected),
            "breakdown": dimensions,
            "latency": latency_stats([r["seconds"] for r in selected]),
        }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "solution": solution,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        **await asyncio.to_thread(source_provenance),
        "dependencies": {
            name: version(name)
            for name in (
                "onnxruntime",
                "transformers",
                "presidio-analyzer",
                "numpy",
                "tokenizers",
            )
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "peak_rss_mib": round(
            peak / (1024**2 if platform.system() == "Darwin" else 1024), 2
        ),
        "initialization_seconds": round(runner.initialization_seconds, 3),
        "detection_cache": False,
        "upstream_requests": 0,
        "kiji_artifacts": runner.kiji_artifacts,
        "corpora": corpora,
        "clean": clean,
        "structured_probes": probes,
        "regressions": regressions,
        "documents": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{solution}: complete; Privy full-span recall "
        f"{corpora['privy']['full_span_recall_pct']}%, mean "
        f"{corpora['privy']['latency']['mean_ms']} ms",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", choices=CHOICES, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    asyncio.run(
        run(
            args.solution,
            args.output or RESULTS / "kiji_privy" / f"{args.solution}.json",
            args.repeats,
        )
    )


if __name__ == "__main__":
    main()
