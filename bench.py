#!/usr/bin/env python3
"""
PrivAiTe PII Detection Benchmark

Runs datasets against PrivAiTe's detection engine with different presets,
measures detection rate, false positive rate, and latency.

Usage:
    python bench.py [--presets light,standard] [--output results/report.json]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

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


PRESETS = {
    "light": PIIConfig(
        enabled=True,
        preset=None,
        detectors=DetectorsConfig(
            presidio=PresidioDetectorConfig(
                enabled=True,
                languages=["fr", "en"],
                score_threshold=0.4,
                entities=[
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                    "IBAN_CODE", "IP_ADDRESS", "DATE_TIME",
                    "US_SSN", "UK_NHS",
                ],
            ),
        ),
        anonymization=AnonymizationConfig(
            method="placeholder", faker_locale=["fr_FR", "en_US"]
        ),
        deanonymization=DeanonymizationConfig(enabled=True),
    ),
    "onnx": PIIConfig(
        enabled=True,
        preset="onnx",
        anonymization=AnonymizationConfig(
            method="placeholder", faker_locale=["fr_FR", "en_US"]
        ),
        deanonymization=DeanonymizationConfig(enabled=True),
    ),
}


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _config_for_lang(base: PIIConfig, lang: str) -> PIIConfig:
    langs = [lang]
    if lang != "en":
        langs.append("en")
    return PIIConfig(
        enabled=base.enabled,
        preset=base.preset,
        detectors=DetectorsConfig(
            presidio=PresidioDetectorConfig(
                enabled=base.detectors.presidio.enabled,
                languages=langs,
                score_threshold=base.detectors.presidio.score_threshold,
                entities=base.detectors.presidio.entities,
            ),
        ),
        anonymization=base.anonymization,
        deanonymization=base.deanonymization,
    )


async def run_preset(preset_name: str, config: PIIConfig, samples: list[dict]):
    engines: dict[str, PIIEngine] = {}

    results = []
    for sample in samples:
        lang = sample.get("lang", "fr")
        if lang not in engines:
            lang_config = _config_for_lang(config, lang)
            eng = PIIEngine(lang_config)
            await eng.initialize()
            engines[lang] = eng
        engine = engines[lang]
        expected = sample.get("expected", {})

        t0 = time.perf_counter()
        msgs = [{"role": "user", "content": sample["text"]}]
        anon_msgs, mapping = await engine.process_request(msgs)
        latency = (time.perf_counter() - t0) * 1000

        anon_text = anon_msgs[0]["content"]

        detected = {}
        missed = {}
        for pii_text, pii_type in expected.items():
            if pii_text not in anon_text:
                detected[pii_text] = pii_type
            else:
                missed[pii_text] = pii_type

        false_positives = []
        if not expected:
            for orig in mapping._original_to_fake:
                false_positives.append({
                    "text": orig,
                    "type": mapping.get_entity_type(orig),
                })

        results.append({
            "id": sample["id"],
            "lang": sample.get("lang", "?"),
            "total_expected": len(expected),
            "detected": len(detected),
            "missed": len(missed),
            "missed_items": missed,
            "false_positives": false_positives,
            "latency_ms": round(latency, 1),
        })

    for eng in engines.values():
        await eng.shutdown()
    return results


def print_report(preset: str, pii_results: list[dict], clean_results: list[dict]):
    total_expected = sum(r["total_expected"] for r in pii_results)
    total_detected = sum(r["detected"] for r in pii_results)
    total_missed = sum(r["missed"] for r in pii_results)
    total_fp = sum(len(r["false_positives"]) for r in clean_results)
    total_clean = len(clean_results)

    det_rate = total_detected / total_expected * 100 if total_expected else 0
    miss_rate = total_missed / total_expected * 100 if total_expected else 0
    fp_rate = total_fp / total_clean * 100 if total_clean else 0

    avg_lat_pii = sum(r["latency_ms"] for r in pii_results) / len(pii_results)
    avg_lat_clean = sum(r["latency_ms"] for r in clean_results) / len(clean_results)

    print(f"\n{'=' * 60}")
    print(f"  PRESET: {preset}")
    print(f"{'=' * 60}")
    print(f"  Detection rate:     {total_detected}/{total_expected} ({det_rate:.1f}%)")
    print(f"  Miss rate:          {total_missed}/{total_expected} ({miss_rate:.1f}%)")
    print(f"  False positives:    {total_fp}/{total_clean} clean texts ({fp_rate:.1f}%)")
    print(f"  Avg latency (PII):  {avg_lat_pii:.1f}ms")
    print(f"  Avg latency (clean):{avg_lat_clean:.1f}ms")

    if total_missed > 0:
        print(f"\n  MISSED PII:")
        for r in pii_results:
            for text, ptype in r["missed_items"].items():
                print(f"    [{r['id']}] {ptype}: \"{text}\"")

    if total_fp > 0:
        print(f"\n  FALSE POSITIVES:")
        for r in clean_results:
            for fp in r["false_positives"]:
                print(f"    [{r['id']}] {fp['type']}: \"{fp['text']}\"")

    print()

    by_lang = {}
    for r in pii_results:
        lang = r["lang"]
        if lang not in by_lang:
            by_lang[lang] = {"expected": 0, "detected": 0}
        by_lang[lang]["expected"] += r["total_expected"]
        by_lang[lang]["detected"] += r["detected"]

    print("  BY LANGUAGE:")
    for lang, stats in sorted(by_lang.items()):
        rate = stats["detected"] / stats["expected"] * 100 if stats["expected"] else 0
        print(f"    {lang}: {stats['detected']}/{stats['expected']} ({rate:.0f}%)")

    by_type = {}
    for r in pii_results:
        for text, ptype in {**dict.fromkeys(
            [t for t in r.get("missed_items", {}).values()], "missed"
        )}.items():
            pass
    for r in pii_results:
        sample = next(s for s in pii_samples if s["id"] == r["id"])
        for text, ptype in sample.get("expected", {}).items():
            if ptype not in by_type:
                by_type[ptype] = {"total": 0, "detected": 0}
            by_type[ptype]["total"] += 1
            if text not in r.get("missed_items", {}):
                by_type[ptype]["detected"] += 1

    print("\n  BY ENTITY TYPE:")
    for ptype, stats in sorted(by_type.items()):
        rate = stats["detected"] / stats["total"] * 100 if stats["total"] else 0
        status = "ok" if rate == 100 else "MISS" if rate < 100 else "ok"
        print(f"    {ptype:20} {stats['detected']}/{stats['total']} ({rate:.0f}%) {status}")


async def main():
    global pii_samples
    base = Path(__file__).parent

    pii_samples = load_dataset(base / "datasets" / "pii_samples.json")
    clean_samples = load_dataset(base / "datasets" / "clean_samples.json")

    for extra in [
        "corporate_samples.json", "batch_samples.json",
        "realworld_samples.json", "dlptest_us.json",
        "enedis_rse_extracts.json",
    ]:
        path = base / "datasets" / extra
        if path.exists():
            extra_data = load_dataset(path)
            pii_samples = pii_samples + extra_data

    print(f"Loaded {len(pii_samples)} PII samples, {len(clean_samples)} clean samples")

    all_results = {}

    for preset_name, config in PRESETS.items():
        print(f"\nRunning preset: {preset_name}...")

        pii_results = await run_preset(preset_name, config, pii_samples)
        clean_results = await run_preset(preset_name, config, clean_samples)

        print_report(preset_name, pii_results, clean_results)

        all_results[preset_name] = {
            "pii": pii_results,
            "clean": clean_results,
        }

    output = base / "results" / "report.json"
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Full results saved to {output}")


if __name__ == "__main__":
    asyncio.run(main())
