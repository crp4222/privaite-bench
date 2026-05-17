"""
Latency benchmark for long texts and multi-language configurations.

Measures p50/p95/p99 latency on realistic long documents (~1000 tokens)
across different language configurations (1, 3, 5, 7 languages).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from statistics import median

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent / "PrivAiTe"))

from privaite.config.schema import (
    AnonymizationConfig,
    DeanonymizationConfig,
    DetectorsConfig,
    PIIConfig,
    PresidioDetectorConfig,
)
from privaite.pii.engine import PIIEngine


def make_config(languages: list[str]) -> PIIConfig:
    return PIIConfig(
        enabled=True,
        preset=None,
        detectors=DetectorsConfig(
            presidio=PresidioDetectorConfig(
                enabled=True,
                languages=languages,
                score_threshold=0.4,
                entities=[
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                    "IBAN_CODE", "IP_ADDRESS", "DATE_TIME", "US_SSN", "UK_NHS",
                ],
            ),
        ),
        anonymization=AnonymizationConfig(method="placeholder", faker_locale=["en_US"]),
        deanonymization=DeanonymizationConfig(enabled=True),
    )


def estimate_tokens(text: str) -> int:
    return len(text.split()) * 4 // 3


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


async def measure_latency(engine: PIIEngine, text: str, runs: int = 30) -> dict:
    await engine.process_request([{"role": "user", "content": text}])

    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        await engine.process_request([{"role": "user", "content": text}])
        timings.append((time.perf_counter() - t0) * 1000)

    return {
        "min": min(timings),
        "p50": median(timings),
        "p95": percentile(timings, 95),
        "p99": percentile(timings, 99),
        "max": max(timings),
        "mean": sum(timings) / len(timings),
    }


async def main():
    base = Path(__file__).parent

    with open(base / "datasets" / "long_texts.json") as f:
        long_texts = json.load(f)

    with open(base / "datasets" / "long_1000_tokens.json") as f:
        long_texts.extend(json.load(f))

    print("=" * 80)
    print("LATENCY BENCHMARK — long texts × multi-language configurations")
    print("=" * 80)
    print()

    print("Document sizes:")
    for doc in long_texts:
        chars = len(doc["text"])
        tokens = estimate_tokens(doc["text"])
        print(f"  {doc['id']:25} {chars:5} chars ~{tokens:4} tokens ({doc['lang']})")
    print()

    lang_configs = [
        ("1 language", ["fr"]),
        ("2 languages", ["fr", "en"]),
        ("3 languages", ["fr", "en", "de"]),
        ("5 languages", ["fr", "en", "de", "es", "it"]),
        ("7 languages", ["fr", "en", "de", "es", "it", "pt", "nl"]),
    ]

    results = {}

    for label, langs in lang_configs:
        print(f"## {label}: {langs}")
        engine = PIIEngine(make_config(langs))
        boot_t0 = time.perf_counter()
        await engine.initialize()
        boot_ms = (time.perf_counter() - boot_t0) * 1000
        print(f"  Boot: {boot_ms:.0f}ms")

        results[label] = {"boot_ms": boot_ms, "docs": {}}

        for doc in long_texts:
            primary_lang = doc["lang"]
            if primary_lang not in langs:
                continue

            test_langs = [primary_lang] + [l for l in langs if l != primary_lang]
            test_engine = PIIEngine(make_config(test_langs))
            await test_engine.initialize()

            stats = await measure_latency(test_engine, doc["text"], runs=30)
            tokens = estimate_tokens(doc["text"])
            print(f"  [{doc['id']:25}] ~{tokens}t  "
                  f"p50={stats['p50']:.0f}ms  "
                  f"p95={stats['p95']:.0f}ms  "
                  f"p99={stats['p99']:.0f}ms  "
                  f"mean={stats['mean']:.0f}ms")

            results[label]["docs"][doc["id"]] = {
                "tokens": tokens,
                **stats,
            }

            await test_engine.shutdown()

        await engine.shutdown()
        print()

    output = base / "results" / "latency_report.json"
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {output}")


if __name__ == "__main__":
    asyncio.run(main())
