"""Cost of one ONNX pass per window, variants measured in alternation.

run.py times each variant in its own process, so a load spike on the machine
lands on one variant and not the other. Here the sessions live in one process
and the calls alternate, so ambient load hits both the same way and the ratio
holds even on a busy machine. Writes results/onnx_variants/window_ratio.json,
which report.py picks up. Run from the repo root with the PrivAiTe venv:
    <privaite venv>/bin/python -m scripts.onnx_variants.window_ratio
"""

from __future__ import annotations

import glob
import json
import statistics
import time

import numpy as np
import onnxruntime as ort
from huggingface_hub import constants

from benchlib.paths import RESULTS
from privaite.config.schema import OnnxDetectorConfig

VARIANTS = ["q4", "q4f16", "quantized"]
WINDOWS = (512, 1024)
ROUNDS = 15


def main() -> None:
    revision = OnnxDetectorConfig().revision
    folder = glob.glob(f"{constants.HF_HUB_CACHE}/models--openai--privacy-filter/snapshots/{revision}/onnx")[0]
    sessions = {}
    for variant in VARIANTS:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sessions[variant] = ort.InferenceSession(
            f"{folder}/model_{variant}.onnx", opts, providers=["CPUExecutionProvider"]
        )

    rng = np.random.default_rng(0)
    result: dict[str, dict[str, dict[str, float]]] = {v: {} for v in VARIANTS}
    for n in WINDOWS:
        feeds = {
            v: {
                i.name: np.ones((1, n), np.int64)
                if "mask" in i.name
                else rng.integers(1000, 199000, (1, n), dtype=np.int64)
                for i in s.get_inputs()
            }
            for v, s in sessions.items()
        }
        for v, s in sessions.items():
            s.run(None, feeds[v])
        timings: dict[str, list[float]] = {v: [] for v in VARIANTS}
        for _ in range(ROUNDS):
            for v, s in sessions.items():
                t0 = time.perf_counter()
                s.run(None, feeds[v])
                timings[v].append(time.perf_counter() - t0)
        for v in VARIANTS:
            result[v][str(n)] = {
                "median_s": round(statistics.median(timings[v]), 3),
                "ratio_vs_q4f16": round(statistics.median(timings["q4f16"]) / statistics.median(timings[v]), 2),
            }
    out = RESULTS / "onnx_variants" / "window_ratio.json"
    out.write_text(json.dumps({"rounds": ROUNDS, "variants": result}, indent=1) + "\n")
    for v in VARIANTS:
        print(v, {n: (r["median_s"], f"x{r['ratio_vs_q4f16']}") for n, r in result[v].items()})


if __name__ == "__main__":
    main()
