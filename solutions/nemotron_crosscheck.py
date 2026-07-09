"""Score PrivAiTe's shipped presets on the Nemotron-PII OOD corpus.

Reuses compare.evaluate (the SAME value-based recall the headline uses) so the result
is directly comparable to the main-bench recall, not to the span-overlap OOD table.
Recall is over the PII types PrivAiTe's presets target; Nemotron labels that no
compared PII tool detects by default are excluded, with the split fixed below and the
excluded counts written to the report for transparency.

Run per preset (onnx/light in the PrivAiTe venv, max in the gliner venv), each run
merges its row into results/nemotron_ood_report.json (integer counts only, no text).
Run as a module from the repo root (running the file directly fails because
solutions/solutions.py shadows the package on sys.path, same as compare.py):
    <privaite venv>/bin/python -m solutions.nemotron_crosscheck --preset onnx --preset light
    <gliner venv>/bin/python  -m solutions.nemotron_crosscheck --preset max
Build the corpus first: python scripts/ood/build_nemotron_corpus.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from solutions.compare import evaluate
from solutions.solutions import PrivAiTeSolution

# Nemotron label -> PrivAiTe type. INCLUDED: the contact / identity / temporal /
# network / carded-financial PII the presets are built to detect. Everything else
# (demographics, occupation/company, opaque secrets and domain identifiers, and
# format-less financial account numbers) is EXCLUDED: no compared tool targets it,
# so counting it would measure scope, not detection. The excluded labels and their
# counts are recorded in the report.
MAP = {
    "first_name": "PERSON", "last_name": "PERSON", "name": "PERSON",
    "email": "EMAIL_ADDRESS",
    "phone_number": "PHONE_NUMBER", "fax_number": "PHONE_NUMBER",
    "url": "URL",
    "street_address": "LOCATION", "city": "LOCATION", "state": "LOCATION",
    "country": "LOCATION", "county": "LOCATION", "postcode": "LOCATION",
    "coordinate": "LOCATION",
    "date": "DATE_TIME", "date_of_birth": "DATE_TIME", "date_time": "DATE_TIME",
    "time": "DATE_TIME",
    "ssn": "US_SSN",
    "credit_debit_card": "CREDIT_CARD",
    "ipv4": "IP_ADDRESS", "ipv6": "IP_ADDRESS",
}

CORPUS = Path(__file__).resolve().parent / "_nemotron_ood_corpus.json"
REPORT = Path(__file__).resolve().parents[1] / "results" / "nemotron_ood_report.json"


def load_eval_corpus() -> tuple[list[dict], dict]:
    raw = json.loads(CORPUS.read_text("utf-8"))
    corpus, excluded = [], {}
    for doc in raw:
        expected = {}
        for value, label in doc["expected_raw"].items():
            if label in MAP:
                expected[value] = MAP[label]
            else:
                excluded[label] = excluded.get(label, 0) + 1
        if expected:
            corpus.append({"id": doc["id"], "lang": doc["lang"],
                           "text": doc["text"], "expected": expected})
    stats = {
        "docs": len(corpus),
        "entities": sum(len(d["expected"]) for d in corpus),
        "excluded_labels": dict(sorted(excluded.items(), key=lambda kv: -kv[1])),
    }
    return corpus, stats


async def main(presets: list[str]) -> None:
    corpus, stats = load_eval_corpus()
    print(f"Nemotron OOD corpus: {stats['docs']} docs, {stats['entities']} targeted PII "
          f"entities ({len(stats['excluded_labels'])} label types excluded)")

    report = json.loads(REPORT.read_text("utf-8")) if REPORT.exists() else {}
    report["corpus"] = stats
    for preset in presets:
        sol = PrivAiTeSolution(preset)
        r = await evaluate(sol, corpus, [])
        await sol.teardown()
        row = {
            "recall": r["recall"],
            "recall_token_level": r["recall_token_level"],
            "caught": r["caught"],
            "total": r["total"],
        }
        report.setdefault("presets", {})[preset] = row
        print(f"  privaite-{preset:9} recall={row['recall']:5}%  "
              f"token-level={row['recall_token_level']:5}%  "
              f"caught={row['caught']}/{row['total']}", flush=True)

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print("wrote", REPORT.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", action="append", dest="presets",
                    choices=["light", "onnx", "max", "standard", "full"],
                    help="repeatable; default: onnx light")
    args = ap.parse_args()
    asyncio.run(main(args.presets or ["onnx", "light"]))
