"""Build the out-of-distribution (OOD) corpus for the generalization cross-check.

The main bench uses AI4Privacy pii-masking-200k. Models fine-tuned on that family
(the llm-guard DeBERTa, Piiranha) get a home-field advantage there. To measure real
generalization we score the SAME detectors on a corpus from a different source:
Gretel's synthetic_pii_finance_multilingual, which is independent of AI4Privacy and
of the GLiNER training data, and ships exact char-level PII spans.

Output solutions/_gretel_ood_corpus.json (raw third-party text -> gitignored, like
the AI4Privacy corpus). Run from the repo root:
    python -m scripts.ood.build_gretel_corpus
"""

import json
from collections import Counter

from benchlib.hf import paginate, rows_url
from benchlib.paths import SOLUTIONS

LANG_MAP = {"English": "en", "French": "fr", "German": "de", "Italian": "it"}
WANT = {"en": 40, "fr": 40, "de": 40, "it": 40}
BASE = rows_url("gretelai/synthetic_pii_finance_multilingual", split="test")


def main() -> None:
    got: dict[str, list] = {k: [] for k in WANT}
    for entry in paginate(BASE, max_offset=3000, timeout=30):
        row = entry["row"]
        lang = LANG_MAP.get(row.get("language"))
        if not lang or len(got[lang]) >= WANT[lang]:
            continue
        text, spans = row["generated_text"], row["pii_spans"]
        if isinstance(spans, str):
            spans = json.loads(spans)
        if not spans or len(text) > 4000:
            continue
        got[lang].append({
            "id": f"gretel-{row['index']}", "lang": lang, "text": text,
            "spans": [[s["start"], s["end"], s["label"]] for s in spans],
        })
        if all(len(got[k]) >= WANT[k] for k in WANT):
            break

    corpus = [d for k in got for d in got[k]]
    (SOLUTIONS / "_gretel_ood_corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    print("docs:", len(corpus), "by lang:", dict(Counter(d["lang"] for d in corpus)))
    print("PII spans:", sum(len(d["spans"]) for d in corpus))


if __name__ == "__main__":
    main()
