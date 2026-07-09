"""Run the transformer PII models on the OOD corpus, in their isolated venvs.

Two competitor/candidate detectors, each needing torch (so NOT the PrivAiTe env):
  - Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2 : the model behind llm-guard's
    Anonymize scanner, fine-tuned ON AI4Privacy -> here it is OFF-distribution, which
    is the whole point (does a contaminated-on-AI4Privacy model generalize?).
  - urchade/gliner_multi_pii-v1 : GLiNER, trained on independent Mistral-generated
    synthetic PII (NOT AI4Privacy) -> a clean generalization candidate to ADD.

Writes results/ood_spans_mdeberta.json and results/ood_spans_gliner.json (char spans,
no PII text -> committable). Run each (from the repo root) with the matching venv:
    ~/.venvs/llm-guard/bin/python -m scripts.ood.run_transformer_models mdeberta
    ~/.venvs/gliner/bin/python    -m scripts.ood.run_transformer_models gliner

Both models truncate long inputs (mDeBERTa ~512 tok, GLiNER ~384 tok); PrivAiTe's
onnx detector does not. That truncation is their real behavior and is noted in
OOD_COMPARISON.md; the ranking is unchanged when scoring only short docs.
"""

import json
import sys

from benchlib.data import load_json
from benchlib.paths import RESULTS, SOLUTIONS

GLINER_LABELS = [
    "person", "first name", "last name", "email", "phone number", "address",
    "social security number", "iban", "swift bic code", "credit card number",
    "date of birth", "password", "organization", "date", "time", "employee id",
    "passport number", "ip address", "job title",
]


def run_mdeberta(corpus: list) -> dict:
    from transformers import pipeline

    clf = pipeline("token-classification",
                   model="Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2",
                   aggregation_strategy="simple")
    out = {}
    for doc in corpus:
        try:
            res = clf(doc["text"][:2000])  # ~ its real 512-token context window
            out[doc["id"]] = [[int(e["start"]), int(e["end"])] for e in res]
        except Exception:
            out[doc["id"]] = []
    return out


def run_gliner(corpus: list) -> dict:
    from gliner import GLiNER

    model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
    out = {}
    for doc in corpus:
        try:
            ents = model.predict_entities(doc["text"], GLINER_LABELS, threshold=0.5)
            out[doc["id"]] = [[int(e["start"]), int(e["end"])] for e in ents]
        except Exception:
            out[doc["id"]] = []
    return out


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    corpus = load_json(SOLUTIONS / "_gretel_ood_corpus.json")
    runner = {"mdeberta": run_mdeberta, "gliner": run_gliner}.get(which)
    if runner is None:
        raise SystemExit("usage: python -m scripts.ood.run_transformer_models {mdeberta|gliner}")
    out = runner(corpus)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"ood_spans_{which}.json").write_text(json.dumps(out), "utf-8")
    print(f"{which} spans:", sum(len(v) for v in out.values()))


if __name__ == "__main__":
    main()
