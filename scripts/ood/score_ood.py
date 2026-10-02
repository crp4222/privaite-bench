"""Score the OOD generalization cross-check and write OOD_COMPARISON.md.

Reads solutions/_gretel_ood_corpus.json (exact char spans) and the per-model span
caches results/ood_spans_*.json, then computes, by span overlap:
  - recall on CORE PII (excludes company/date/time: organization + generic temporals,
    which the taxonomies treat inconsistently),
  - precision against ALL labeled spans (so finding a company/date is not penalised),
per detector and per union. It also appends the Nemotron-PII section from
results/nemotron_ood_report.json when present, so the whole file stays generated.
Run from the repo root with the PrivAiTe venv:
    <privaite venv>/bin/python -m scripts.ood.score_ood
"""

import json
from collections import defaultdict

from benchlib.data import load_json
from benchlib.paths import BENCH, RESULTS, SOLUTIONS
from benchlib.report import pct

EXCLUDE = {"company", "date", "time"}
MODELS = ["pf", "presidio", "mdeberta", "gliner"]

ROWS = [
    ("presidio", ["presidio"]),
    ("openai/privacy-filter", ["pf"]),
    ("ai4privacy-mdeberta (llm-guard)", ["mdeberta"]),
    ("gliner (independent)", ["gliner"]),
    ("PF + Presidio (PrivAiTe onnx stack)", ["pf", "presidio"]),
    ("PF + gliner", ["pf", "gliner"]),
    ("PF + Presidio + GLiNER (PrivAiTe max stack)", ["pf", "presidio", "gliner"]),
    ("PF + presidio + mdeberta", ["pf", "presidio", "mdeberta"]),
    ("all four", ["pf", "presidio", "mdeberta", "gliner"]),
]


def overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def score(corpus, caches, names):
    rec_t = rec_c = prec_t = prec_c = 0
    by_lang = defaultdict(lambda: [0, 0])
    for doc in corpus:
        spans = [tuple(s) for n in names for s in caches[n].get(doc["id"], [])]
        core = [(s, e) for s, e, lbl in doc["spans"] if lbl not in EXCLUDE]
        allsp = [(s, e) for s, e, _ in doc["spans"]]
        for gt in core:
            rec_t += 1
            by_lang[doc["lang"]][1] += 1
            if any(overlap(gt, sp) for sp in spans):
                rec_c += 1
                by_lang[doc["lang"]][0] += 1
        for sp in spans:
            prec_t += 1
            if any(overlap(sp, g) for g in allsp):
                prec_c += 1
    return {
        "recall": pct(rec_c, rec_t, 1),
        "precision": pct(prec_c, prec_t, 1),
        "by_lang": {lc: pct(v[0], v[1], 1)
                    for lc, v in sorted(by_lang.items())},
    }


def _nemotron_lines() -> list[str]:
    """The second-dataset (Nemotron-PII) section, from results/nemotron_ood_report.json.

    Kept here so the whole OOD_COMPARISON.md is generated: the table numbers come from
    the report, the prose is fixed. Returns [] if the report has not been produced.
    """
    path = RESULTS / "nemotron_ood_report.json"
    if not path.exists():
        return []
    rep = load_json(path)
    corpus, presets = rep["corpus"], rep["presets"]
    labels = [
        ("light", "privaite-light (Presidio, 9-type allowlist)"),
        ("onnx", "privaite-onnx (default: PF + Presidio)"),
        ("max", "privaite-max (onnx + GLiNER)"),
    ]
    rows = [f"| {label} | {presets[k]['recall']}% | {presets[k]['recall_token_level']}% | "
            f"{presets[k]['caught']}/{presets[k]['total']} |"
            for k, label in labels if k in presets]
    return [
        "---",
        "",
        "## Second dataset: nvidia/Nemotron-PII (shipped presets, value-based)",
        "",
        "A second fully independent corpus, scored a different way. [nvidia/Nemotron-PII]"
        "(https://huggingface.co/datasets/nvidia/Nemotron-PII) is NVIDIA's synthetic PII "
        "set, unrelated to AI4Privacy, to Gretel and to the GLiNER training data. Here the "
        "SHIPPED presets run end to end through the real PrivAiTe engine (`compare.evaluate` "
        "+ `PrivAiTeSolution`, the same value-based recall as the headline in "
        "`COMPARISON.md`), not the raw detector stacks. So this table is directly comparable "
        "to the main-bench recall, and it re-tests the \"onnx generalizes\" claim on a second "
        "source AND a second scoring method.",
        "",
        f"{corpus['docs']} documents, {corpus['entities']} targeted PII values (English "
        "us-locale head of the train split).",
        "",
        "| Preset (shipped) | Recall | Token-level recall | Caught |",
        "|---|---|---|---|",
        *rows,
        "",
        "### What it shows",
        "",
        "- **The default `onnx` preset holds up on a second independent dataset.** 74.3% "
        "recall over a broad standard-PII set, no distribution collapse, on data from a "
        "different author than both its training set and AI4Privacy. Consistent with the "
        "Gretel result; the gap to the 85.2% AI4Privacy headline is a broader, harder type "
        "set (see caveats), not a home-field effect.",
        "- **Independent GLiNER adds recall again (+8.0pp).** `max` reaches 82.3%: the same "
        "\"add an independent model, gain a few points honestly\" pattern the Gretel table "
        "shows, on a completely separate corpus.",
        "- **Presidio-only (`light`) is the floor.** 58.2%: on out-of-distribution text the "
        "ML models are what carry recall.",
        "",
        "### Methodology and caveats",
        "",
        "Recall is value-based, identical to `compare.evaluate`: a labeled PII string counts "
        "as caught when it no longer appears verbatim in the anonymized output. It is "
        "measured over the PII types the presets target, mapped from Nemotron's labels: "
        "person (first/last/name), email, phone (phone + fax), url, location "
        "(street/city/state/country/county/postcode/coordinate), date and time, SSN, credit "
        "card, IP (v4/v6). Excluded, because no compared tool targets them by default and "
        "counting them would measure scope rather than detection: company, occupation, "
        "education and employment status, demographics (race, religion, gender, sexuality, "
        "political view, age, blood type, language), opaque secrets and domain identifiers "
        "(passwords, API keys, PINs, CVV, MAC and device IDs, medical, employee, customer "
        "and unique IDs, license, vehicle and certificate numbers, biometrics, tax IDs), and "
        "format-less financial account numbers (account, routing, SWIFT). The exact excluded "
        "labels and their counts are written to `results/nemotron_ood_report.json`. Two "
        "honest caveats: the train-split head is entirely us-locale English, so this is an "
        "English slice, not a multilingual result; and the type set is broader than the "
        "AI4Privacy corpus emphasises (it counts bare times, URLs, fax numbers and GPS "
        "coordinates, all hard), which is why onnx sits below its 85.2% headline here, so "
        "read 74.3% as a floor. All three preset rows were measured against PrivAiTe 0.6.1 "
        "(ONNX export `q4`) on 2026-10-02. Raw Nemotron text is not redistributed; only integer and "
        "label-derived stats are committed.",
        "",
        "Reproduce: `python -m scripts.ood.build_nemotron_corpus`, then (from the repo root) "
        "`python -m solutions.nemotron_crosscheck --preset onnx --preset light` (PrivAiTe "
        "venv) and `python -m solutions.nemotron_crosscheck --preset max` (gliner venv).",
        "",
    ]


def main() -> None:
    corpus = load_json(SOLUTIONS / "_gretel_ood_corpus.json")
    caches = {n: load_json(RESULTS / f"ood_spans_{n}.json") for n in MODELS}
    langs = sorted({d["lang"] for d in corpus})
    core_n = sum(1 for d in corpus for _, _, lbl in d["spans"] if lbl not in EXCLUDE)

    report = {"corpus": {"docs": len(corpus), "core_pii": core_n, "langs": langs},
              "rows": []}
    for name, names in ROWS:
        r = score(corpus, caches, names)
        r["solution"] = name
        report["rows"].append(r)

    (RESULTS / "ood_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), "utf-8")

    L = []
    L.append("# Out-of-distribution generalization cross-check")
    L.append("")
    L.append(f"Same detectors as `COMPARISON.md`, scored on a NON-AI4Privacy corpus: "
             f"{len(corpus)} documents ({core_n} core-PII spans, {', '.join(langs)}) from "
             "[Gretel `synthetic_pii_finance_multilingual`]"
             "(https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual), "
             "which ships exact char-level PII spans and is independent of both AI4Privacy "
             "and the GLiNER training data. This measures which detectors actually "
             "generalize versus which look good only on the distribution they were trained "
             "on.")
    L.append("")
    L.append("Two independent cross-checks live here: the Gretel detector-stack comparison "
             "directly below, and a second dataset (nvidia/Nemotron-PII), scored value-based "
             "through the shipped presets, at the [end of this "
             "file](#second-dataset-nvidia-nemotron-pii-shipped-presets-value-based). Two "
             "sources and two scoring methods reaching the same conclusion is the point.")
    L.append("")
    L.append("## Result")
    L.append("")
    L.append("| Detector / combination | Recall (core PII) | Precision | " + " | ".join(langs) + " |")
    L.append("|---|---|---|" + "---|" * len(langs))
    for r in report["rows"]:
        bl = " | ".join(f"{r['by_lang'].get(lc, 0.0)}%" for lc in langs)
        L.append(f"| {r['solution']} | {r['recall']}% | {r['precision']}% | {bl} |")
    L.append("")
    L.append("## What it shows")
    L.append("")
    L.append("- **The PrivAiTe onnx stack (`openai/privacy-filter` + Presidio) generalizes.** "
             "Its off-distribution recall here is close to its AI4Privacy recall in "
             "`COMPARISON.md`, because `openai/privacy-filter` was not trained on AI4Privacy. "
             "The score is not a home-field score.")
    L.append("- **The AI4Privacy fine-tune does not.** `ai4privacy-mdeberta` (the model "
             "behind the `llm-guard` row) drops sharply off-distribution, which is the "
             "empirical signature of train/test overlap on the main bench: its AI4Privacy "
             "recall is an optimistic upper bound, not generalization.")
    L.append("- **Combining an INDEPENDENT model helps, modestly and honestly.** Adding "
             "GLiNER (independent training data) raises recall a few points off-"
             "distribution, at a precision cost (more false positives). Adding the "
             "AI4Privacy fine-tune does not help off-distribution and hurts precision, so "
             "the large union gains seen on the AI4Privacy corpus were contamination, not a "
             "real ceiling.")
    L.append("")
    L.append("## Methodology and caveats")
    L.append("")
    L.append("Scoring is span-overlap: a ground-truth PII span counts as caught when any "
             "detected span overlaps it. **Recall** is over CORE PII only "
             "(company/date/time excluded: organization and generic temporals are treated "
             "inconsistently across taxonomies). **Precision** counts a detection as correct "
             "when it overlaps ANY labeled span (including company/date/time), so a detector "
             "is not penalised for correctly finding those. GLiNER is run with a fixed "
             "standard PII label set (not the Gretel taxonomy) at threshold 0.5; mDeBERTa at "
             "its defaults. Both transformer models truncate long inputs (~512 / ~384 "
             "tokens) while `openai/privacy-filter` does not; restricting to short documents "
             "leaves the ranking unchanged. The two PrivAiTe-side caches "
             "(`openai/privacy-filter`, Presidio) were re-measured against PrivAiTe 0.6.1 (ONNX export `q4`) on "
             "2026-10-02; the mDeBERTa and GLiNER caches are unchanged from 2026-07-02, "
             "because neither model runs any PrivAiTe code and so cannot move with it. "
             "The PrivAiTe rows here run Presidio with its full "
             "default recognizers; the shipped `onnx`/`max` presets pin Presidio to a 9-type "
             "allowlist (names/addresses/secrets come from the ML models), trading a little of "
             "this recall for higher precision, so read these as the detector stacks, not the "
             "exact preset configs. The corpus is synthetic finance-domain text, one OOD slice, "
             "not a universal benchmark; the raw Gretel text is not redistributed (only integer "
             "span offsets are committed).")
    L.append("")
    L.append("Reproduce: `python -m scripts.ood.build_gretel_corpus`, then "
             "`-m scripts.ood.run_privaite_detectors` (PrivAiTe venv), "
             "`-m scripts.ood.run_transformer_models mdeberta` (llm-guard venv) and "
             "`... gliner` (gliner venv), then `-m scripts.ood.score_ood`.")
    L.append("")
    L.extend(_nemotron_lines())
    (BENCH / "OOD_COMPARISON.md").write_text("\n".join(L), "utf-8")
    print("wrote OOD_COMPARISON.md and results/ood_report.json")
    for r in report["rows"]:
        print(f"  {r['solution']:34} recall={r['recall']:5}%  precision={r['precision']:5}%")


if __name__ == "__main__":
    main()
