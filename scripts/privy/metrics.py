"""Character coverage makes partial leaks and excessive redaction visible.

Metrics are type-agnostic: removing a password as EMAIL still removes its
characters, although policy/reversibility is tested separately. Precision is
relative to the supplied gold annotations, not a claim that they are exhaustive.
"""

from __future__ import annotations

import re
from collections import Counter


def coverage(text: str, gold: list, predicted: list) -> dict:
    nonspace = {i for i, char in enumerate(text) if not char.isspace()}
    gold_chars: set[int] = set()
    predicted_chars: set[int] = set()
    for start, end, *_ in gold:
        if not 0 <= start < end <= len(text):
            raise ValueError("Invalid gold offsets")
        gold_chars.update(range(start, end))
    for start, end, *_ in predicted:
        if not 0 <= start < end <= len(text):
            raise ValueError("Invalid predicted offsets")
        predicted_chars.update(range(start, end))
    gold_chars &= nonspace
    predicted_chars &= nonspace
    types: dict[str, Counter] = {}
    complete = partial = 0
    for start, end, entity_type in gold:
        positions = set(range(start, end)) & nonspace
        if not positions:
            continue
        hit = len(positions & predicted_chars)
        full = hit == len(positions)
        complete += full
        partial += 0 < hit < len(positions)
        stats = types.setdefault(entity_type, Counter())
        stats.update(
            spans=1, complete=int(full), gold_chars=len(positions), covered_chars=hit
        )
    return {
        "gold_spans": sum(c["spans"] for c in types.values()),
        "complete_spans": complete,
        "partial_spans": partial,
        "gold_chars": len(gold_chars),
        "predicted_chars": len(predicted_chars),
        "covered_chars": len(gold_chars & predicted_chars),
        "non_gold_chars_removed": len(predicted_chars - gold_chars),
        "non_gold_chars": len(nonspace - gold_chars),
        "by_type": {key: dict(counts) for key, counts in types.items()},
    }


def text_recall(text: str, output: str, gold: list) -> dict:
    """Keep the existing comparison's literal and >=4-char-token criteria."""
    values = {text[start:end] for start, end, *_ in gold}
    strict = 0
    for value in values:
        tokens = [token for token in re.findall(r"\w+", value) if len(token) >= 4]
        strict += (
            all(token not in output for token in tokens)
            if tokens
            else value not in output
        )
    return {
        "values": len(values),
        "absent_values": sum(v not in output for v in values),
        "strict_absent_values": strict,
    }


def percentage(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 2) if denominator else None


def summarize(rows: list[dict]) -> dict:
    counts = Counter()
    types: dict[str, Counter] = {}
    for row in rows:
        metrics = row["coverage"]
        counts.update(
            {key: value for key, value in metrics.items() if key != "by_type"}
        )
        counts.update(row["literal"])
        for name, value in metrics["by_type"].items():
            types.setdefault(name, Counter()).update(value)
    return {
        "documents": len(rows),
        "counts": dict(counts),
        "full_span_recall_pct": percentage(
            counts["complete_spans"], counts["gold_spans"]
        ),
        "character_recall_pct": percentage(
            counts["covered_chars"], counts["gold_chars"]
        ),
        "character_precision_pct": percentage(
            counts["covered_chars"], counts["predicted_chars"]
        ),
        "non_gold_removal_pct": percentage(
            counts["non_gold_chars_removed"], counts["non_gold_chars"]
        ),
        "literal_recall_pct": percentage(counts["absent_values"], counts["values"]),
        "strict_literal_recall_pct": percentage(
            counts["strict_absent_values"], counts["values"]
        ),
        "zero_gold_docs_changed": sum(
            r["changed"] for r in rows if not r["coverage"]["gold_spans"]
        ),
        "zero_gold_docs": sum(not r["coverage"]["gold_spans"] for r in rows),
        "by_type": {
            name: {
                **dict(value),
                "full_span_recall_pct": percentage(value["complete"], value["spans"]),
            }
            for name, value in sorted(types.items())
        },
    }
