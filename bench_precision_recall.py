"""
Precision / recall benchmark.

Counts true positives, false positives, false negatives at the entity level:
- TP = entity correctly flagged (matches expected)
- FP = entity flagged but not in expected list (over-anonymization)
- FN = entity in expected list but not flagged (leak)

precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * P * R / (P + R)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
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


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _spans_overlap(a_text: str, b_text: str) -> bool:
    a, b = a_text.strip().lower(), b_text.strip().lower()
    return a in b or b in a


async def evaluate_sample(engine: PIIEngine, sample: dict) -> dict:
    msgs = [{"role": "user", "content": sample["text"]}]
    _, mapping = await engine.process_request(msgs)

    detected = list(mapping._original_to_fake.keys())
    expected = set(sample.get("expected", {}).keys())

    tp = 0
    matched_expected = set()
    fp_items = []

    for det in detected:
        matched = False
        for exp in expected:
            if exp in matched_expected:
                continue
            if _spans_overlap(det, exp):
                tp += 1
                matched_expected.add(exp)
                matched = True
                break
        if not matched:
            fp_items.append(det)

    fn_items = list(expected - matched_expected)

    return {
        "id": sample["id"],
        "lang": sample.get("lang", "?"),
        "tp": tp,
        "fp": len(fp_items),
        "fn": len(fn_items),
        "expected_total": len(expected),
        "detected_total": len(detected),
        "fp_items": fp_items,
        "fn_items": fn_items,
    }


def aggregate(results: list[dict]) -> dict:
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


async def main():
    base = Path(__file__).parent

    datasets = []
    for name in ["pii_samples.json", "corporate_samples.json", "batch_samples.json",
                 "dlptest_us.json", "enedis_rse_extracts.json", "realworld_samples.json",
                 "long_texts.json"]:
        path = base / "datasets" / name
        if path.exists():
            datasets.extend(load_dataset(path))

    print(f"Loaded {len(datasets)} PII samples")
    print()

    config = PIIConfig(
        enabled=True, preset=None,
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
        deanonymization=DeanonymizationConfig(enabled=True),
    )

    engine = PIIEngine(config)
    await engine.initialize()

    results = []
    for sample in datasets:
        result = await evaluate_sample(engine, sample)
        results.append(result)

    print("=" * 80)
    print("PRECISION / RECALL — entity-level evaluation")
    print("=" * 80)
    print()

    agg = aggregate(results)
    print(f"  True positives (TP):  {agg['tp']}")
    print(f"  False positives (FP): {agg['fp']}")
    print(f"  False negatives (FN): {agg['fn']}")
    print()
    print(f"  Precision: {agg['precision']*100:.1f}%  (of what we flag, how much is real PII)")
    print(f"  Recall:    {agg['recall']*100:.1f}%  (of real PII, how much we catch)")
    print(f"  F1:        {agg['f1']*100:.1f}%")
    print()

    by_lang = {}
    for r in results:
        lang = r["lang"]
        if lang not in by_lang:
            by_lang[lang] = []
        by_lang[lang].append(r)

    print("BY LANGUAGE:")
    for lang in sorted(by_lang.keys()):
        a = aggregate(by_lang[lang])
        print(f"  {lang}: P={a['precision']*100:.1f}% R={a['recall']*100:.1f}% F1={a['f1']*100:.1f}% "
              f"(TP={a['tp']} FP={a['fp']} FN={a['fn']})")
    print()

    fp_by_doc = [(r["id"], r["fp_items"]) for r in results if r["fp"] > 0]
    if fp_by_doc:
        print("FALSE POSITIVES (entities flagged that weren't in expected list):")
        for doc_id, fps in fp_by_doc:
            print(f"  [{doc_id}]")
            for fp in fps:
                print(f"    + {fp!r}")
    print()

    fn_by_doc = [(r["id"], r["fn_items"]) for r in results if r["fn"] > 0]
    if fn_by_doc:
        print("FALSE NEGATIVES (PII that leaked):")
        for doc_id, fns in fn_by_doc:
            print(f"  [{doc_id}]")
            for fn in fns:
                print(f"    - {fn!r}")
    print()

    output = base / "results" / "precision_recall_report.json"
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump({"aggregate": agg, "by_doc": results}, f, indent=2, ensure_ascii=False)
    print(f"Full results saved to {output}")

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
