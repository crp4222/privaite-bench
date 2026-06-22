#!/usr/bin/env python3
"""Structured-input PII benchmark.

bench.py and bench_precision_recall.py send every sample as one flat user
message. Real OpenAI-compatible clients also put user data inside multimodal
content parts and tool-call arguments (a JSON string, often nested). This script
re-embeds the same PII corpus into those shapes and checks that none of it
bypasses anonymization.

Two metrics:

  parity      For every PII string the flat baseline anonymizes, each structured
              carrier (multimodal, tool_call, tool_call_nested) must anonymize it
              too. A value caught flat but leaked through a carrier is a
              regression. Target: zero.
  round-trip  Tool-call arguments must de-anonymize back to the original on the
              response side, without loss.

It also runs datasets/structured_samples.json: hand-written function-call and
multimodal payloads that are natively structured, not re-wrapped.

Usage:
    python bench_structured.py                      real run (needs spaCy models)
    python bench_structured.py --selftest           harness check, fake detector
    PRIVAITE_PATH=/path/to/checkout python bench_structured.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The other scripts hardcode ../PrivAiTe; allow an override so the suite can run
# against any checkout (e.g. a feature branch) without moving directories.
_PRIVAITE_PATH = os.environ.get("PRIVAITE_PATH") or str(
    Path(__file__).parent.parent / "PrivAiTe"
)
sys.path.insert(0, _PRIVAITE_PATH)

from privaite.config.schema import (  # noqa: E402
    AnonymizationConfig,
    DeanonymizationConfig,
    DetectorsConfig,
    PIIConfig,
    PresidioDetectorConfig,
)
from privaite.pii.engine import PIIEngine  # noqa: E402

CARRIERS = ["flat", "multimodal", "tool_call", "tool_call_nested"]

DATASETS = [
    "pii_samples.json",
    "corporate_samples.json",
    "batch_samples.json",
    "dlptest_us.json",
    "enedis_rse_extracts.json",
    "realworld_samples.json",
    "long_texts.json",
]


def wrap(text: str, carrier: str) -> list[dict]:
    """Embed a flat text into the message shape for a given carrier."""
    if carrier == "flat":
        return [{"role": "user", "content": text}]
    if carrier == "multimodal":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/scan.png"},
                    },
                ],
            }
        ]
    if carrier == "tool_call":
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "save_note",
                            "arguments": json.dumps({"note": text}, ensure_ascii=False),
                        },
                    }
                ],
            }
        ]
    if carrier == "tool_call_nested":
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "save_profile",
                            "arguments": json.dumps(
                                {"user": {"profile": {"bio": text}}, "tags": [text]},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }
        ]
    raise ValueError(f"unknown carrier: {carrier}")


def flatten_messages(messages: list[dict]) -> str:
    """Collect every string a provider would actually receive, across shapes."""
    chunks: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                chunks.append(function["arguments"])
    return "\n".join(chunks)


def score_leaks(expected: dict, flattened: str) -> tuple[list[str], list[str]]:
    """Split expected PII into (anonymized, leaked) by presence in the output."""
    anonymized = [pii for pii in expected if pii not in flattened]
    leaked = [pii for pii in expected if pii in flattened]
    return anonymized, leaked


def _collect_tool_calls(messages: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            calls.append(call)
    return calls


async def _restore_ok(engine: PIIEngine, anon_messages: list[dict], mapping, want: list[str]) -> bool | None:
    """De-anonymize tool-call arguments and check every `want` PII reappears."""
    calls = _collect_tool_calls(anon_messages)
    if not calls:
        return None
    restored = await engine.process_response_tool_calls(calls, mapping)
    restored_flat = flatten_messages([{"role": "assistant", "tool_calls": restored}])
    return all(pii in restored_flat for pii in want)


async def eval_rewrapped(engine: PIIEngine, sample: dict) -> dict:
    expected = sample.get("expected", {})
    carriers: dict[str, dict] = {}
    for carrier in CARRIERS:
        messages = wrap(sample["text"], carrier)
        anon, mapping = await engine.process_request(messages)
        flat = flatten_messages(anon)
        anonymized, leaked = score_leaks(expected, flat)
        carriers[carrier] = {
            "anonymized": anonymized,
            "leaked": leaked,
            "roundtrip_ok": await _restore_ok(engine, anon, mapping, anonymized),
        }
    return {"id": sample["id"], "lang": sample.get("lang", "?"), "carriers": carriers}


async def eval_native(engine: PIIEngine, sample: dict) -> dict:
    expected = sample.get("expected", {})
    messages = sample["messages"]
    anon, mapping = await engine.process_request(messages)
    flat = flatten_messages(anon)
    anonymized, leaked = score_leaks(expected, flat)
    return {
        "id": sample["id"],
        "lang": sample.get("lang", "?"),
        "expected_total": len(expected),
        "anonymized": anonymized,
        "leaked": leaked,
        "roundtrip_ok": await _restore_ok(engine, anon, mapping, anonymized),
    }


def report_rewrapped(results: list[dict]) -> dict:
    print("=" * 72)
    print("  STRUCTURED CARRIERS vs FLAT BASELINE  (parity on re-wrapped corpus)")
    print("=" * 72)

    flat_total = sum(len(r["carriers"]["flat"]["anonymized"]) for r in results)
    summary: dict[str, dict] = {}

    for carrier in CARRIERS:
        anonymized = sum(len(r["carriers"][carrier]["anonymized"]) for r in results)
        leaked = sum(len(r["carriers"][carrier]["leaked"]) for r in results)
        # Regression = caught flat, leaked here.
        regressions = []
        for r in results:
            flat_ok = set(r["carriers"]["flat"]["anonymized"])
            here_leaked = set(r["carriers"][carrier]["leaked"])
            for pii in flat_ok & here_leaked:
                regressions.append([r["id"], pii])
        rt = [r["carriers"][carrier]["roundtrip_ok"] for r in results]
        rt_checked = [x for x in rt if x is not None]
        summary[carrier] = {
            "anonymized": anonymized,
            "leaked": leaked,
            "regressions": regressions,
            "roundtrip_failures": sum(1 for x in rt_checked if x is False),
            "roundtrip_checked": len(rt_checked),
        }
        flag = "OK" if not regressions else f"!! {len(regressions)} REGRESSIONS"
        print(f"  {carrier:18} anonymized={anonymized:4}  leaked={leaked:3}  vs flat={flat_total:4}  {flag}")
        if rt_checked:
            print(
                f"  {'':18} round-trip restored "
                f"{len(rt_checked) - summary[carrier]['roundtrip_failures']}/{len(rt_checked)}"
            )

    total_regressions = sum(len(s["regressions"]) for s in summary.values())
    print()
    if total_regressions == 0:
        print(f"  PARITY OK: 0 regressions across {len(CARRIERS) - 1} structured carriers "
              f"on {flat_total} flat-detected PII.")
    else:
        print(f"  PARITY FAILED — {total_regressions} regression(s):")
        for carrier, s in summary.items():
            for doc_id, pii in s["regressions"]:
                print(f"    [{carrier}] {doc_id}: {pii!r}")
    print()
    return summary


def report_native(results: list[dict]) -> dict:
    if not results:
        return {}
    print("=" * 72)
    print("  NATIVE STRUCTURED SAMPLES  (hand-written function-call / multimodal)")
    print("=" * 72)
    total = sum(r["expected_total"] for r in results)
    anonymized = sum(len(r["anonymized"]) for r in results)
    leaked = sum(len(r["leaked"]) for r in results)
    rt = [r["roundtrip_ok"] for r in results]
    rt_checked = [x for x in rt if x is not None]
    rate = anonymized / total * 100 if total else 0.0
    print(f"  Anonymized: {anonymized}/{total} ({rate:.1f}%)   Leaked: {leaked}")
    print(f"  Round-trip restored: {sum(1 for x in rt_checked if x)}/{len(rt_checked)}")
    if leaked:
        print("\n  LEAKED:")
        for r in results:
            for pii in r["leaked"]:
                print(f"    [{r['id']}] {pii!r}")
    print()
    return {"total": total, "anonymized": anonymized, "leaked": leaked}


# Multi-language light preset, same detector config as bench_precision_recall.py.
def _real_config() -> PIIConfig:
    return PIIConfig(
        enabled=True,
        preset=None,
        detectors=DetectorsConfig(
            presidio=PresidioDetectorConfig(
                enabled=True,
                languages=["fr", "en", "de", "es", "it"],
                score_threshold=0.4,
                entities=[
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                    "IBAN_CODE", "IP_ADDRESS", "DATE_TIME", "US_SSN", "UK_NHS",
                ],
            ),
        ),
        anonymization=AnonymizationConfig(method="placeholder", faker_locale=["en_US"]),
        deanonymization=DeanonymizationConfig(enabled=True, fuzzy_matching=False),
    )


# light + the OpenAI privacy-filter ONNX model (needs `pip install transformers onnxruntime`).
def _onnx_config() -> PIIConfig:
    config = _real_config()
    config.detectors.onnx.enabled = True
    return config


def _load(base: Path, name: str) -> list[dict]:
    path = base / "datasets" / name
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


async def main() -> None:
    base = Path(__file__).parent

    corpus: list[dict] = []
    for name in DATASETS:
        corpus.extend(_load(base, name))
    native = _load(base, "structured_samples.json")

    print(f"Loaded {len(corpus)} flat samples (re-wrapped into {len(CARRIERS)} carriers), "
          f"{len(native)} native structured samples")

    report: dict = {}
    for preset_name, config in (("light", _real_config()), ("onnx", _onnx_config())):
        print(f"\n{'#' * 72}\n# PRESET: {preset_name}\n{'#' * 72}")
        engine = PIIEngine(config)
        await engine.initialize()

        rewrapped = [await eval_rewrapped(engine, s) for s in corpus]
        native_results = [await eval_native(engine, s) for s in native]

        await engine.shutdown()

        rewrapped_summary = report_rewrapped(rewrapped)
        native_summary = report_native(native_results)
        report[preset_name] = {
            "rewrapped": rewrapped_summary,
            "native": native_summary,
            "by_doc": rewrapped,
            "native_by_doc": native_results,
        }

    out = base / "results" / "structured_report.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Full results saved to {out}")


# Validate the harness with a deterministic fake detector (no spaCy needed).
async def selftest() -> int:
    from privaite.pii.detector_base import PIIDetector
    from privaite.pii.entity import PIIEntity

    class FakeDetector(PIIDetector):
        def __init__(self, terms: dict[str, str]) -> None:
            self.terms = terms

        @property
        def name(self) -> str:
            return "fake"

        async def initialize(self) -> None:
            pass

        async def detect(self, text: str, language: str = "en") -> list[PIIEntity]:
            ents: list[PIIEntity] = []
            for term, etype in self.terms.items():
                start = 0
                while (idx := text.find(term, start)) >= 0:
                    ents.append(PIIEntity(etype, term, idx, idx + len(term), 0.99, "fake"))
                    start = idx + len(term)
            return ents

    base = Path(__file__).parent
    native = _load(base, "structured_samples.json")
    corpus = [
        {
            "id": "st_inline",
            "lang": "fr",
            "text": "Contact Marie Dupont at marie@acme.com or +33 6 11 22 33 44.",
            "expected": {
                "Marie Dupont": "PERSON",
                "marie@acme.com": "EMAIL_ADDRESS",
                "+33 6 11 22 33 44": "PHONE_NUMBER",
            },
        }
    ]

    # Fake detector knows every expected PII string across both sample sets.
    terms: dict[str, str] = {}
    for sample in corpus:
        terms.update(sample["expected"])
    for sample in native:
        terms.update(sample.get("expected", {}))

    config = PIIConfig(
        enabled=True,
        detectors=DetectorsConfig(presidio=PresidioDetectorConfig(enabled=False)),
        anonymization=AnonymizationConfig(method="placeholder"),
        deanonymization=DeanonymizationConfig(enabled=True, fuzzy_matching=False),
    )
    engine = PIIEngine(config)
    engine.detectors = [FakeDetector(terms)]
    engine._ready = True

    failures = 0

    for sample in corpus:
        result = await eval_rewrapped(engine, sample)
        flat_ok = set(result["carriers"]["flat"]["anonymized"])
        assert flat_ok == set(sample["expected"]), f"flat baseline incomplete: {result}"
        for carrier in CARRIERS:
            c = result["carriers"][carrier]
            if c["leaked"]:
                print(f"  FAIL [{carrier}] leaked {c['leaked']}")
                failures += 1
            if carrier.startswith("tool_call") and c["roundtrip_ok"] is not True:
                print(f"  FAIL [{carrier}] round-trip not restored")
                failures += 1

    for sample in native:
        result = await eval_native(engine, sample)
        if result["leaked"]:
            print(f"  FAIL [native {result['id']}] leaked {result['leaked']}")
            failures += 1
        if result["roundtrip_ok"] is False:
            print(f"  FAIL [native {result['id']}] round-trip not restored")
            failures += 1

    if failures == 0:
        print(f"SELFTEST PASSED: {len(corpus)} re-wrapped x {len(CARRIERS)} carriers + "
              f"{len(native)} native samples, 0 leaks, round-trip lossless.")
    else:
        print(f"SELFTEST FAILED — {failures} failure(s).")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="Validate harness with a fake detector (no spaCy).")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(1 if asyncio.run(selftest()) else 0)
    asyncio.run(main())
