"""Build the second out-of-distribution corpus: nvidia/Nemotron-PII.

OOD_COMPARISON.md's first cross-check uses Gretel finance text. This adds an
independent SECOND source so "the shipped presets generalize" does not rest on one
dataset. Nemotron-PII is NVIDIA's synthetic PII set, unrelated to AI4Privacy, to
Gretel and to the GLiNER training data; each row ships exact char-level spans
(start/end/text/label) over a clean `text` field.

The corpus is scored value-based by solutions/nemotron_crosscheck.py, the same way
the main bench scores recall (compare.evaluate), so the numbers are directly
comparable to the headline recall rather than to the span-overlap OOD table.

Output solutions/_nemotron_ood_corpus.json (raw third-party text -> gitignored, like
the AI4Privacy and Gretel corpora). Only the build script and integer/label-derived
stats are committed. Run from the repo root (no PrivAiTe needed):
    python -m scripts.ood.build_nemotron_corpus
"""

import ast
import json
from collections import Counter

from benchlib.hf import paginate, rows_url
from benchlib.paths import SOLUTIONS

BASE = rows_url("nvidia/Nemotron-PII")
# Nemotron locales are regional (us, gb, ...); map to a Presidio language. The
# public train head is entirely us-locale English, so this stays en in practice.
LOCALE_LANG = {"us": "en", "gb": "en", "ca": "en", "au": "en"}
WANT_DOCS = 300


def main() -> None:
    corpus = []
    labels: Counter = Counter()
    for entry in paginate(BASE, max_offset=4000):
        if len(corpus) >= WANT_DOCS:
            break
        row = entry["row"]
        lang = LOCALE_LANG.get(row.get("locale"))
        if not lang:
            continue
        text = row["text"]
        spans = ast.literal_eval(row["spans"]) if isinstance(row["spans"], str) else row["spans"]
        # value-based expected map, same shape the main bench consumes: keep the
        # first label seen for a given surface string (dict collapses repeats).
        expected_raw: dict[str, str] = {}
        for s in spans:
            labels[s["label"]] += 1
            expected_raw.setdefault(s["text"], s["label"])
        if not expected_raw:
            continue
        corpus.append({
            "id": f"nemotron-{row['uid']}", "lang": lang,
            "text": text, "expected_raw": expected_raw,
        })

    out = SOLUTIONS / "_nemotron_ood_corpus.json"
    out.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    print("docs:", len(corpus),
          "| raw labeled values:", sum(len(d["expected_raw"]) for d in corpus))
    print("distinct labels:", len(labels))
    print("top labels:", dict(labels.most_common(12)))
    print("wrote", out.name, "(gitignored: raw third-party text not redistributed)")


if __name__ == "__main__":
    main()
