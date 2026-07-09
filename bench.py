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
import time

from benchlib.paths import RESULTS, bootstrap

bootstrap()

from benchlib.config import LIGHT_ENTITIES, langs_for, pii_config  # noqa: E402
from benchlib.data import load_dataset  # noqa: E402
from benchlib.report import rule  # noqa: E402
from privaite.config.schema import (  # noqa: E402
    AnonymizationConfig,
    DeanonymizationConfig,
    DetectorsConfig,
    PIIConfig,
    PresidioDetectorConfig,
)
from privaite.pii.engine import PIIEngine  # noqa: E402


PRESETS = {
    # Presidio pinned to the 9-entity allowlist. onnx keeps the preset expansion
    # (its detector set comes from preset="onnx"), so it stays an explicit config.
    "light": pii_config(
        languages=["fr", "en"],
        entities=LIGHT_ENTITIES,
        faker_locale=["fr_FR", "en_US"],
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


def _config_for_lang(base: PIIConfig, lang: str) -> PIIConfig:
    return PIIConfig(
        enabled=base.enabled,
        preset=base.preset,
        detectors=DetectorsConfig(
            presidio=PresidioDetectorConfig(
                enabled=base.detectors.presidio.enabled,
                languages=langs_for(lang),
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


def _print_totals(preset, pii_results, clean_results):
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

    print(f"\n{rule(60)}")
    print(f"  PRESET: {preset}")
    print(f"{rule(60)}")
    print(f"  Detection rate:     {total_detected}/{total_expected} ({det_rate:.1f}%)")
    print(f"  Miss rate:          {total_missed}/{total_expected} ({miss_rate:.1f}%)")
    print(f"  False positives:    {total_fp}/{total_clean} clean texts ({fp_rate:.1f}%)")
    print(f"  Avg latency (PII):  {avg_lat_pii:.1f}ms")
    print(f"  Avg latency (clean):{avg_lat_clean:.1f}ms")
    return total_missed, total_fp


def _print_missed(pii_results):
    print("\n  MISSED PII:")
    for r in pii_results:
        for text, ptype in r["missed_items"].items():
            print(f"    [{r['id']}] {ptype}: \"{text}\"")


def _print_false_positives(clean_results):
    print("\n  FALSE POSITIVES:")
    for r in clean_results:
        for fp in r["false_positives"]:
            print(f"    [{r['id']}] {fp['type']}: \"{fp['text']}\"")


def _print_by_language(pii_results):
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


def _print_by_type(pii_results, pii_samples):
    by_type = {}
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


def print_report(preset, pii_results, clean_results, pii_samples):
    total_missed, total_fp = _print_totals(preset, pii_results, clean_results)
    if total_missed > 0:
        _print_missed(pii_results)
    if total_fp > 0:
        _print_false_positives(clean_results)
    print()
    _print_by_language(pii_results)
    _print_by_type(pii_results, pii_samples)


async def main():
    pii_samples = load_dataset("pii_samples.json")
    clean_samples = load_dataset("clean_samples.json")

    for extra in [
        "corporate_samples.json", "batch_samples.json",
        "realworld_samples.json", "dlptest_us.json",
        "enedis_rse_extracts.json",
    ]:
        pii_samples = pii_samples + load_dataset(extra)

    print(f"Loaded {len(pii_samples)} PII samples, {len(clean_samples)} clean samples")

    all_results = {}

    for preset_name, config in PRESETS.items():
        print(f"\nRunning preset: {preset_name}...")

        pii_results = await run_preset(preset_name, config, pii_samples)
        clean_results = await run_preset(preset_name, config, clean_samples)

        print_report(preset_name, pii_results, clean_results, pii_samples)

        all_results[preset_name] = {
            "pii": pii_results,
            "clean": clean_results,
        }

    output = RESULTS / "report.json"
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Full results saved to {output}")


if __name__ == "__main__":
    asyncio.run(main())
