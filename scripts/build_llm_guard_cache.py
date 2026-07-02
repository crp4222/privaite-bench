"""Generate results/llm_guard_cache.json for the comparative benchmark.

LLM Guard (github.com/protectai/llm-guard) hard-conflicts with the PrivAiTe
environment (it pins transformers 4.51 and requires torch, while PrivAiTe is
ONNX-only on transformers 5). So it is run ONCE here, in an isolated venv, and its
per-document output is cached for solutions/solutions.py::LLMGuardSolution to read.

Run with the isolated venv, from the repo root:
    python3.12 -m venv ~/.venvs/llm-guard
    ~/.venvs/llm-guard/bin/pip install 'llm-guard==0.3.16'
    ~/.venvs/llm-guard/bin/python scripts/build_llm_guard_cache.py

Fairness (see COMPARISON.md "How each competitor is configured"):
  - Anonymize scanner at its own default threshold (0.5) and supported language
    ("en"; LLM Guard supports en/zh only, so its non-English recall is honestly
    lower and shown in the per-language table).
  - entity_types EXPANDED beyond its default to also request DATE_TIME, LOCATION
    and URL (types present in the corpus that its Presidio engine can detect), so
    the row is not weakened by a narrow default list.
  - It is a flat-string scanner with no message/multimodal/tool-call awareness;
    the bench models that honestly (structured fields leak).
"""

import json
import time
from pathlib import Path

from llm_guard.input_scanners import Anonymize
from llm_guard.input_scanners.anonymize import DEFAULT_ENTITY_TYPES
from llm_guard.vault import Vault

BENCH = Path(__file__).resolve().parents[1]

# Expand coverage to every corpus type LLM Guard's engine can plausibly detect
# (its default list omits DATE_TIME / LOCATION / URL). SECRET and FINANCIAL are
# PrivAiTe-specific types LLM Guard has no recognizer for; not fabricated here.
ENTITY_TYPES = list(dict.fromkeys(DEFAULT_ENTITY_TYPES + ["DATE_TIME", "LOCATION", "URL"]))

scanner = Anonymize(Vault(), language="en", threshold=0.5, entity_types=ENTITY_TYPES)


_latencies: list[float] = []


def scan_one(text: str) -> dict:
    scanner._vault = Vault()  # fresh vault so detections are per-document
    t0 = time.perf_counter()
    sanitized, is_valid, risk = scanner.scan(text)
    _latencies.append((time.perf_counter() - t0) * 1000)
    detected = [original for (_placeholder, original) in scanner._vault.get()]
    return {"sanitized": sanitized, "is_valid": is_valid, "risk": risk, "detected": detected}


def main() -> None:
    corpus = json.loads((BENCH / "solutions" / "_corpus.json").read_text("utf-8"))
    clean = json.loads((BENCH / "datasets" / "clean_samples.json").read_text("utf-8"))

    out: dict[str, dict] = {}
    for i, doc in enumerate(corpus):
        rec = scan_one(doc["text"])
        rec.update(text=doc["text"], lang=doc["lang"])
        out[doc["id"]] = rec
        print(f"corpus {i + 1}/{len(corpus)}  {doc['id']}  detected={len(rec['detected'])}")
    for doc in clean:
        rec = scan_one(doc["text"])
        rec.update(text=doc["text"], lang=doc.get("lang", "en"))
        out[f"clean::{doc['id']}"] = rec
        print(f"clean {doc['id']}  flagged={len(rec['detected'])}")

    # Real inference latency, measured HERE (isolated venv). The bench itself reads
    # this cache in ~0ms, so it stores the true offline number for the latency
    # column (marked "offline" since it is measured in a separate environment).
    mean_ms = round(sum(_latencies) / len(_latencies), 1) if _latencies else 0.0
    out["__meta__"] = {"mean_scan_ms": mean_ms, "entity_types": ENTITY_TYPES}

    (BENCH / "results").mkdir(exist_ok=True)
    (BENCH / "results" / "llm_guard_cache.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote results/llm_guard_cache.json  ({len(out) - 1} records)")
    print(f"mean scan latency: {mean_ms}ms")
    print(f"entity_types requested: {ENTITY_TYPES}")


if __name__ == "__main__":
    main()
