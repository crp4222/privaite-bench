"""
Comparative benchmark: multiple PII-anonymization solutions on the same corpus.

The corpus is real AI4Privacy documents whose ground-truth PII was labeled by ten
independent auditor agents (see solutions/ai4privacy_loader.py and the labeling
workflow). Scoring is substring based, exactly like bench.py: a PII item counts as
caught if the solution's output no longer contains it.

For each solution we measure, on the same documents:
  - recall on flat message text (per language and per entity type)
  - false positives on clean text
  - leakage inside tool-call arguments and multimodal parts (the structured gap)
  - average latency

Run:  python solutions/compare.py
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from solutions.solutions import all_solutions

BENCH = Path(__file__).resolve().parents[1]


def load_corpus() -> list[dict]:
    """Prefer the local corpus (with text); else rebuild from committed labels."""
    local = BENCH / "solutions" / "_corpus.json"
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    data = json.loads((BENCH / "datasets" / "comparative_labels.json").read_text("utf-8"))
    labels = data["labels"] if isinstance(data, dict) else data
    from solutions.ai4privacy_loader import load_text_by_id

    index = load_text_by_id()
    corpus = []
    for entry in labels:
        doc = index.get(entry["id"])
        if doc:
            corpus.append({
                "id": entry["id"], "lang": entry["lang"],
                "text": doc["text"], "expected": entry["expected"],
            })
    return corpus


def load_clean() -> list[dict]:
    return json.loads((BENCH / "datasets" / "clean_samples.json").read_text("utf-8"))


def ground_truth_quality(corpus: list[dict]) -> dict | None:
    """Cross-check the agent ground truth against AI4Privacy's own sensitive mask.

    High overlap means the agent labels are precise; the agent-only items are
    incidental PII the dataset mask did not tag.
    """
    sample_path = BENCH / "solutions" / "_ai4privacy_sample.json"
    if not sample_path.exists():
        return None
    gold_by_id = {d["id"]: d.get("gold", []) for d in json.loads(sample_path.read_text("utf-8"))}

    def overlaps(a: str, b: str) -> bool:
        return a in b or b in a

    gold_total = gold_caught = agent_total = agent_in_gold = 0
    for doc in corpus:
        gold_vals = [g["value"] for g in gold_by_id.get(doc["id"], [])]
        agent_keys = list(doc["expected"].keys())
        for gv in gold_vals:
            gold_total += 1
            if any(overlaps(gv, ak) for ak in agent_keys):
                gold_caught += 1
        for ak in agent_keys:
            agent_total += 1
            if any(overlaps(ak, gv) for gv in gold_vals):
                agent_in_gold += 1
    return {
        "dataset_sensitive_spans": gold_total,
        "agent_recovered_dataset_pct": round(gold_caught / gold_total * 100, 1) if gold_total else 0.0,
        "agent_labels": agent_total,
        "agent_overlap_with_dataset_pct": round(agent_in_gold / agent_total * 100, 1) if agent_total else 0.0,
    }


def _structured_payload(text: str) -> list[dict]:
    """Same PII placed in a multimodal text part and a tool-call argument."""
    return [
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "https://example.com/scan.png"}},
        ]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "save_record",
                "arguments": json.dumps({"note": text}),
            }},
        ]},
    ]


def _tool_args(messages: list[dict]) -> str:
    for msg in messages:
        for call in (msg.get("tool_calls") or []):
            return call.get("function", {}).get("arguments", "")
    return ""


def _multimodal_text(messages: list[dict]) -> str:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


async def evaluate(sol, corpus: list[dict], clean: list[dict]) -> dict:
    await sol.setup()
    by_lang = defaultdict(lambda: {"total": 0, "caught": 0})
    by_type = defaultdict(lambda: {"total": 0, "caught": 0})
    total = caught = 0
    tool_total = tool_leaked = 0
    mm_total = mm_leaked = 0
    prot_total = prot_removed = 0
    latencies: list[float] = []

    for doc in corpus:
        text, lang, expected = doc["text"], doc["lang"], doc["expected"]
        t0 = time.perf_counter()
        anon_text, _ = await sol.anonymize_text(text, lang)
        latencies.append((time.perf_counter() - t0) * 1000)

        caught_in_flat = set()
        for pii, ptype in expected.items():
            total += 1
            by_lang[lang]["total"] += 1
            by_type[ptype]["total"] += 1
            if pii not in anon_text:
                caught += 1
                caught_in_flat.add(pii)
                by_lang[lang]["caught"] += 1
                by_type[ptype]["caught"] += 1

        anon_payload = await sol.anonymize_payload(_structured_payload(text), lang)
        args_out = _tool_args(anon_payload)
        mm_out = _multimodal_text(anon_payload)
        for pii in expected:
            tool_total += 1
            mm_total += 1
            if pii in args_out:
                tool_leaked += 1
            if pii in mm_out:
                mm_leaked += 1
            # Of the PII this solution catches in flat text, does it also remove
            # it from the tool-call argument? That isolates structural handling.
            if pii in caught_in_flat:
                prot_total += 1
                if pii not in args_out:
                    prot_removed += 1

    false_positives = 0
    for doc in clean:
        _, originals = await sol.anonymize_text(doc["text"], doc.get("lang", "en"))
        false_positives += len(originals)

    await sol.teardown()

    return {
        "solution": sol.name,
        "recall": round(caught / total * 100, 1) if total else 0.0,
        "caught": caught, "total": total,
        "false_positives": false_positives, "clean_docs": len(clean),
        "tool_call_leak_pct": round(tool_leaked / tool_total * 100, 1) if tool_total else 0.0,
        "multimodal_leak_pct": round(mm_leaked / mm_total * 100, 1) if mm_total else 0.0,
        "tool_call_protection_pct": round(prot_removed / prot_total * 100, 1) if prot_total else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "by_lang": {k: round(v["caught"] / v["total"] * 100, 1) if v["total"] else 0.0
                    for k, v in sorted(by_lang.items())},
        "by_type": {k: round(v["caught"] / v["total"] * 100, 1) if v["total"] else 0.0
                    for k, v in sorted(by_type.items())},
    }


def render_markdown(report: dict) -> str:
    rows = report["solutions"]
    types = sorted({t for r in rows for t in r["by_type"]})
    langs = sorted({lng for r in rows for lng in r["by_lang"]})
    lines = []
    lines.append("# Comparative benchmark")
    lines.append("")
    lines.append(f"Corpus: {report['corpus']['pii_docs']} real AI4Privacy documents "
                 f"({report['corpus']['pii_items']} PII items labeled by 10 independent "
                 f"auditor agents) across {', '.join(langs)}, plus "
                 f"{report['corpus']['clean_docs']} clean documents for false positives.")
    lines.append("")
    lines.append("Scoring is substring based: a PII item is caught when it no longer "
                 "appears in the solution's output.")
    lines.append("")
    gt = report.get("ground_truth")
    if gt:
        lines.append("## Ground truth")
        lines.append("")
        lines.append("The labels are produced by 10 independent auditor agents and "
                     "cross-checked against AI4Privacy's own sensitive mask. The agents "
                     f"independently recovered {gt['agent_recovered_dataset_pct']}% of the "
                     f"dataset's {gt['dataset_sensitive_spans']} sensitive spans, and "
                     f"{gt['agent_overlap_with_dataset_pct']}% of the agent labels overlap "
                     "the dataset mask (the rest are incidental PII the dataset did not "
                     "tag). High overlap means the ground truth is precise.")
        lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| Solution | Recall | False positives | Tool-call protection | Tool-call leak | Multimodal leak | Latency |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['solution']} | {r['recall']}% | "
                     f"{r['false_positives']} on {r['clean_docs']} | "
                     f"{r['tool_call_protection_pct']}% | "
                     f"{r['tool_call_leak_pct']}% | {r['multimodal_leak_pct']}% | "
                     f"{r['avg_latency_ms']}ms |")
    lines.append("")
    lines.append("Tool-call protection is, of the PII a solution catches in plain text, "
                 "how much it also removes from a tool-call argument (higher is better). "
                 "Tool-call leak and multimodal leak are the share of all PII that "
                 "survives inside a tool-call argument or a multimodal text part (lower "
                 "is better).")
    lines.append("")
    lines.append("## Recall by language")
    lines.append("")
    lines.append("| Solution | " + " | ".join(langs) + " |")
    lines.append("|---" * (len(langs) + 1) + "|")
    for r in rows:
        lines.append(f"| {r['solution']} | " +
                     " | ".join(f"{r['by_lang'].get(lng, 0.0)}%" for lng in langs) + " |")
    lines.append("")
    lines.append("## Recall by entity type")
    lines.append("")
    lines.append("| Solution | " + " | ".join(types) + " |")
    lines.append("|---" * (len(types) + 1) + "|")
    for r in rows:
        lines.append(f"| {r['solution']} | " +
                     " | ".join(f"{r['by_type'].get(t, 0.0)}%" for t in types) + " |")
    lines.append("")
    lines.append("Reproduce: `python solutions/ai4privacy_loader.py && python solutions/compare.py`")
    lines.append("")
    return "\n".join(lines)


async def main() -> None:
    corpus = load_corpus()
    clean = load_clean()
    pii_items = sum(len(d["expected"]) for d in corpus)
    print(f"corpus: {len(corpus)} docs, {pii_items} PII items; {len(clean)} clean docs\n")

    rows = []
    for sol in all_solutions():
        print(f"running {sol.name} ...")
        rows.append(await evaluate(sol, corpus, clean))

    report = {
        "corpus": {"pii_docs": len(corpus), "pii_items": pii_items, "clean_docs": len(clean)},
        "ground_truth": ground_truth_quality(corpus),
        "solutions": rows,
    }
    (BENCH / "results").mkdir(exist_ok=True)
    (BENCH / "results" / "comparison_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (BENCH / "COMPARISON.md").write_text(render_markdown(report), encoding="utf-8")

    print()
    for r in rows:
        print(f"{r['solution']:20} recall={r['recall']:5}%  FP={r['false_positives']:3}  "
              f"toolcall_protect={r['tool_call_protection_pct']:5}%  "
              f"toolcall_leak={r['tool_call_leak_pct']:5}%  mm_leak={r['multimodal_leak_pct']:5}%  "
              f"lat={r['avg_latency_ms']}ms")
    print("\nwrote results/comparison_report.json and COMPARISON.md")


if __name__ == "__main__":
    asyncio.run(main())
