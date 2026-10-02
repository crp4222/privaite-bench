"""Run PrivAiTe's onnx preset with one ONNX export of openai/privacy-filter and
record what it detects and how long it takes, for ONNX_VARIANTS.md.

One variant per process, so the peak memory recorded is that variant's. Scoring
is the comparative benchmark's own (`solutions.compare.evaluate`), plus the Gretel
out-of-distribution corpus, two long agent-shaped inputs, and the cost of one ONNX
pass per window. Anonymized outputs and the labelled values a variant leaves in place are stored as
hashes only: the corpora are not redistributed. Run from the repo root with the PrivAiTe venv:
    <privaite venv>/bin/python -m scripts.onnx_variants.run q4f16
    <privaite venv>/bin/python -m scripts.onnx_variants.run q4
    <privaite venv>/bin/python -m scripts.onnx_variants.run quantized
then `python -m scripts.onnx_variants.report`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import privaite
from benchlib.data import load_json
from benchlib.paths import RESULTS, SOLUTIONS
from solutions.compare import evaluate, load_clean, load_corpus
from solutions.solutions import PrivAiTeSolution

VARIANT = sys.argv[1]
OUT = RESULTS / "onnx_variants"


def digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


class VariantSolution(PrivAiTeSolution):
    """The shipped onnx preset with only `onnx_variant` changed."""

    def __init__(self) -> None:
        super().__init__("onnx")
        self.name = f"privaite-onnx-{VARIANT}"
        self.outputs: dict[str, str] = {}

    def _config(self, lang):
        config = super()._config(lang)
        config.detectors.onnx.onnx_variant = VARIANT
        return config

    async def anonymize_text(self, text, lang):
        anon, originals = await super().anonymize_text(text, lang)
        self.outputs[digest(text)] = anon
        return anon, originals

    async def teardown(self) -> None:
        # Keep the engines: the Gretel and long-input passes reuse them.
        pass


def long_inputs() -> dict[str, str]:
    """Agent-shaped inputs that need several ONNX windows: a source file the
    agent read, and a service log that repeats the same failing lines."""
    source = (Path(privaite.__file__).parent / "pii" / "detector_onnx.py").read_text()
    lines = [
        "2026-09-27 22:14:02 INFO  report-job started by jules.moreau@example.com",
        "2026-09-27 22:14:02 INFO  connecting url=postgres://app_reader:Tr0ub4dor-demo@10.20.30.40:55432/crm",
        "2026-09-27 22:14:03 ERROR query failed: relation \"client\" does not exist",
        "2026-09-27 22:14:03 INFO  report-job exited 1, paging Camille Bernard on +33 6 12 34 56 78",
    ]
    return {"source file (detector_onnx.py)": source, "service log, 40 repeats": "\n".join(lines * 40)}


def window_costs() -> dict[str, float]:
    """Median seconds of one ONNX pass at 512 and 1,024 tokens (the window size)."""
    import glob

    import onnxruntime as ort
    from huggingface_hub import constants

    from privaite.config.schema import OnnxDetectorConfig

    revision = OnnxDetectorConfig().revision
    snapshot = glob.glob(f"{constants.HF_HUB_CACHE}/models--openai--privacy-filter/snapshots/{revision}/onnx")[0]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(f"{snapshot}/model_{VARIANT}.onnx", opts, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    costs = {}
    for n in (512, 1024):
        feed = {
            i.name: np.ones((1, n), np.int64) if "mask" in i.name else rng.integers(1000, 199000, (1, n), dtype=np.int64)
            for i in session.get_inputs()
        }
        session.run(None, feed)
        runs = []
        for _ in range(7):
            t0 = time.perf_counter()
            session.run(None, feed)
            runs.append(time.perf_counter() - t0)
        costs[str(n)] = round(statistics.median(runs), 3)
    return costs


async def main() -> None:
    sol = VariantSolution()
    corpus, clean = load_corpus(), load_clean()

    t0 = time.perf_counter()
    bench = await evaluate(sol, corpus, clean)
    bench_seconds = time.perf_counter() - t0

    leaked: dict[str, list[str]] = {}
    for doc in corpus:
        out = sol.outputs[digest(doc["text"])]
        leaked[digest(doc["text"])] = sorted(digest(v) for v in doc["expected"] if v in out)

    gretel = load_json(SOLUTIONS / "_gretel_ood_corpus.json")
    g_total = g_caught = 0
    t0 = time.perf_counter()
    for doc in gretel:
        anon, _ = await sol.anonymize_text(doc["text"], doc.get("lang", "en"))
        values = [doc["text"][s:e] for s, e, _ in doc["spans"]]
        g_total += len(values)
        g_caught += sum(v not in anon for v in values)
        leaked[digest(doc["text"])] = sorted(digest(v) for v in values if v in anon)
    gretel_seconds = time.perf_counter() - t0

    longs = {}
    for name, text in long_inputs().items():
        t0 = time.perf_counter()
        _, originals = await sol.anonymize_text(text, "en")
        longs[name] = {"chars": len(text), "seconds": round(time.perf_counter() - t0, 2), "values": len(originals)}

    result = {
        "variant": VARIANT,
        "privaite": privaite.__version__,
        "bench": {k: bench[k] for k in ("recall", "recall_token_level", "caught", "total",
                                        "false_positives", "clean_docs")},
        "bench_seconds": round(bench_seconds, 1),
        "gretel": {"recall": round(100 * g_caught / g_total, 1), "caught": g_caught, "total": g_total,
                   "seconds": round(gretel_seconds, 1)},
        "long_inputs": longs,
        "window_seconds": window_costs(),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6), 2),
        "output_digests": {k: digest(v) for k, v in sol.outputs.items()},
        "leaked_labelled_value_digests": leaked,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{VARIANT}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("output_digests", "leaked_labelled_values")},
                     indent=1))


if __name__ == "__main__":
    asyncio.run(main())
