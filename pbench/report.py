"""Turn results/runs into results/summary.json, RESULTS.md and the README tables.

    .venvs/harness/bin/python -m pbench.report

Needs the selected corpora in ~/.cache/privaite-bench/corpora (run
`python -m pbench.sources` and `python -m pbench.structured` first): scores are
recomputed from the stored offsets against the gold spans, and every document's
text hash is checked against the one stored with its outcome.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from pbench.corpus import read_jsonl
from pbench.registry import BY_NAME, CONTAMINATION, NOT_TESTED, SYSTEMS
from pbench.score import BOOT, SEED, as_matrix, corpus_metrics, metrics_from_sums, score_doc
from pbench.sources import BUILDERS, CORPORA, EXHAUSTIVE, INFO
from pbench.structured import CARRIERS, item_sha

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"
CORPUS_ORDER = list(BUILDERS)
REAL = [c for c in CORPUS_ORDER if INFO[c]["real"]]


def _records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def macro_metrics(mats: dict, idx: dict, corpora: list[str]) -> dict:
    """Mean over `corpora` of each corpus's metric, every corpus weighing the same
    whatever its size; precision and F2 only over exhaustively labeled corpora.
    The interval resamples documents within each corpus independently and keeps
    the resampled means (`_boot`) for paired comparisons."""
    point = {k: [] for k in ("protected_direct", "char_recall", "precision", "f2")}
    boot = {k: [] for k in point}
    for corpus in corpora:
        mat = mats[corpus]
        p = metrics_from_sums(mat.sum(axis=0))
        b = metrics_from_sums(mat[idx[corpus]].sum(axis=1))
        for k, values in point.items():
            if k in ("precision", "f2") and corpus not in EXHAUSTIVE:
                continue
            values.append(p[k])
            boot[k].append(b[k])
    out = {}
    for k, values in point.items():
        stack = np.nanmean(np.vstack(boot[k]), axis=0)
        out[k] = {"value": float(np.nanmean(values)), "lo": float(np.percentile(stack, 2.5)),
                  "hi": float(np.percentile(stack, 97.5)), "_boot": stack}
    return out


def text_results(docs_by_corpus: dict) -> dict:
    rng = np.random.default_rng(SEED)
    idx = {c: rng.integers(0, len(d), size=(BOOT, len(d))) for c, d in docs_by_corpus.items()}
    out, incomplete = {}, {}
    for spec in SYSTEMS:
        if not spec.text:
            continue
        per_corpus, mats, per_type, per_lang, ms, chars, errors, missing = {}, {}, defaultdict(list), defaultdict(list), [], [], 0, 0
        for corpus, docs in docs_by_corpus.items():
            recs = _records(RUNS / spec.name / f"{corpus}.jsonl")
            scores = []
            for d in docs:
                rec = recs.get(d.id)
                if rec is None:
                    missing += 1
                    spans = None
                elif rec.get("sha256") != d.sha256:
                    raise ValueError(f"{spec.name}/{corpus}/{d.id}: text hash changed, rerun")
                else:
                    spans = rec.get("spans")
                    if "error" in rec:
                        errors += 1
                    else:
                        ms.append(rec["ms"])
                        chars.append(len(d.text))
                ds, types = score_doc(d, spans)
                scores.append(ds)
                for t, p in types:
                    per_type[t].append(p)
                if ds.direct_spans:
                    per_lang[d.lang].append((ds.direct_protected, ds.direct_spans))
            mats[corpus] = as_matrix(scores)
            per_corpus[corpus] = corpus_metrics(mats[corpus], idx[corpus])
        if missing:
            # Not (fully) run yet: never ranked on a partial run. A document the
            # system was given and failed on is an error, not missing, and stays
            # in the score as nothing masked.
            incomplete[spec.name] = missing
            continue
        macro = macro_metrics(mats, idx, CORPUS_ORDER)
        macro_real = macro_metrics(mats, idx, REAL)
        ms_a, ch_a = np.array(ms), np.array(chars)
        out[spec.name] = {
            "macro": macro,
            "macro_real": macro_real,
            "per_corpus": per_corpus,
            "per_type": {t: {"protected": float(np.mean(v)), "n": len(v)} for t, v in sorted(per_type.items())},
            "per_lang": {lg: {"protected": sum(a for a, _ in v) / sum(b for _, b in v),
                              "n": sum(b for _, b in v)} for lg, v in sorted(per_lang.items())},
            "latency_ms": {"p50": float(np.median(ms_a)) if ms else None,
                           "p95": float(np.percentile(ms_a, 95)) if ms else None,
                           "ms_per_1k_chars": float(np.median(ms_a / np.maximum(ch_a, 1) * 1000)) if ms else None},
            "errors": errors,
        }
    # Paired comparison with the leader on the headline: same resamples for both.
    if out:
        best = max(out, key=lambda s: out[s]["macro"]["protected_direct"]["value"])
        for name, r in out.items():
            diff = out[best]["macro"]["protected_direct"]["_boot"] - r["macro"]["protected_direct"]["_boot"]
            r["tied_with_best"] = bool(np.percentile(diff, 2.5) <= 0 <= np.percentile(diff, 97.5)) or name == best
        for r in out.values():
            for block in (r["macro"], r["macro_real"]):
                for m in block.values():
                    m.pop("_boot")
    if incomplete:
        print("not ranked, outcomes missing:", incomplete)
    return out


def structured_results() -> dict:
    items = {i["id"]: i for i in map(json.loads, (CORPORA / "structured.jsonl").read_text().splitlines())}
    docs = sorted({i["doc"] for i in items.values()})
    doc_index = {d: k for k, d in enumerate(docs)}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(docs), size=(BOOT, len(docs)))
    out = {}
    for spec in SYSTEMS:
        if not spec.payload:
            continue
        recs = _records(RUNS / spec.name / "structured.jsonl")
        if set(recs) != set(items):
            if recs:
                print(f"not ranked, {len(set(items) - set(recs))} structured outcomes missing: {spec.name}")
            continue
        per_carrier, valid, kept, blocked, failed_closed, errors, ms = {}, [], {}, 0, 0, 0, []
        for carrier in CARRIERS:
            # per document: leaked direct values, direct values
            leak = np.zeros(len(docs))
            total = np.zeros(len(docs))
            n_items = n_failed_closed = 0
            for iid, it in items.items():
                if it["carrier"] != carrier:
                    continue
                k = doc_index[it["doc"]]
                rec = recs.get(iid)
                if rec.get("sha256") != item_sha(it):
                    raise ValueError(f"{spec.name}/structured/{iid}: request or gold changed, rerun")
                direct = [v["kind"] == "direct" for v in it["values"]]
                total[k] += sum(direct)
                n_items += 1
                if "error" in rec:
                    # Failing open is leaking: a request the system errored on
                    # counts every value in it as leaked.
                    errors += 1
                    leak[k] += sum(direct)
                    continue
                blocked += rec["blocked"]
                failed_closed += rec.get("failed_closed", False)
                n_failed_closed += rec.get("failed_closed", False)
                ms.append(rec["ms"])
                leak[k] += sum(lk for lk, d in zip(rec["leaked"], direct) if d)
                if rec["args_valid"] is not None:
                    valid.append(rec["args_valid"])
                if rec.get("structure_kept") is not None:
                    kept.setdefault(carrier, []).append(rec["structure_kept"])
            rate = leak.sum() / total.sum()
            b = leak[idx].sum(axis=1) / total[idx].sum(axis=1)
            per_carrier[carrier] = {"leak_direct": float(rate), "lo": float(np.percentile(b, 2.5)),
                                    "hi": float(np.percentile(b, 97.5)), "values": int(total.sum()),
                                    "requests": n_items, "failed_closed": n_failed_closed}
        out[spec.name] = {
            "per_carrier": per_carrier,
            "worst_carrier": max(per_carrier, key=lambda c: per_carrier[c]["leak_direct"]),
            "tool_args_json_valid": float(np.mean(valid)) if valid else None,
            "structure_kept": float(np.mean([k for v in kept.values() for k in v])) if kept else None,
            "structure_kept_per_carrier": {c: float(np.mean(v)) for c, v in kept.items()},
            "blocked": blocked, "failed_closed": failed_closed, "errors": errors,
            "latency_ms_p50": float(np.median(ms)) if ms else None,
            "requests": len(items),
        }
    return out


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}%"


def ci(m: dict) -> str:
    return f"{pct(m['value'])} <sub>[{pct(m['lo'], 0)}, {pct(m['hi'], 0)}]</sub>"


def ms(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.0f} ms" if x >= 10 else f"{x:.1f} ms"


KIND = {"integration": "request integration", "library": "library", "model": "model"}


def headline_table(text: dict) -> list[str]:
    rows = sorted(text.items(), key=lambda kv: -kv[1]["macro"]["protected_direct"]["value"])
    lines = ["| # | System | Kind | Direct identifiers fully masked | Real text only | Char recall | Precision | F2 | Latency p50 |",
             "|---:|---|---|---|---|---|---|---|---|"]
    for rank, (name, r) in enumerate(rows, 1):
        spec = BY_NAME[name]
        m = r["macro"]
        tie = " =" if r["tied_with_best"] and rank > 1 else ""
        lines.append(f"| {rank}{tie} | [{spec.title}]({spec.url}) | {KIND[spec.kind]} | {ci(m['protected_direct'])} | "
                     f"{ci(r['macro_real']['protected_direct'])} | {pct(m['char_recall']['value'])} | {pct(m['precision']['value'])} | "
                     f"{pct(m['f2']['value'])} | {ms(r['latency_ms']['p50'])} |")
    return lines


CARRIER_TITLES = {"user_text": "User message", "multimodal_text": "Multimodal part",
                  "system_prompt": "System prompt", "tool_call_args": "Tool-call arguments",
                  "tool_call_nested": "Nested arguments", "tool_result": "Tool result"}


def unusable(r: dict, carrier: str) -> str | None:
    """Why a carrier's leak rate is not protection: the system failed closed on
    every request there, or broke most of the requests it forwarded."""
    c = r["per_carrier"][carrier]
    if c.get("requests") and c.get("failed_closed") == c["requests"]:
        return "fails closed"
    kept = r.get("structure_kept_per_carrier", {}).get(carrier)
    if kept is not None and kept < 0.5:
        return "breaks it"
    return None


def structured_order(r: dict) -> float:
    """Mean leak over the carriers, a carrier the system cannot serve counting
    as fully leaked: dropping a tool call must not sort as protecting it."""
    return float(np.mean([1.0 if unusable(r, c) else r["per_carrier"][c]["leak_direct"] for c in CARRIERS]))


def leak_cell(r: dict, carrier: str, interval: bool = False) -> str:
    c = r["per_carrier"][carrier]
    why = unusable(r, carrier)
    if why == "fails closed":
        return why
    cell = pct(c["leak_direct"])
    if interval:
        cell += f" <sub>[{pct(c['lo'], 0)}, {pct(c['hi'], 0)}]</sub>"
    return f"{cell} ({why})" if why else cell


def structured_table(st: dict) -> list[str]:
    rows = sorted(st.items(), key=lambda kv: structured_order(kv[1]))
    head = ("| System | " + " | ".join(CARRIER_TITLES[c] for c in CARRIERS)
            + " | Requests kept intact | Blocked | Failed closed |")
    lines = [head, "|---|" + "---|" * (len(CARRIERS) + 3)]
    for name, r in rows:
        cells = [leak_cell(r, c) for c in CARRIERS]
        lines.append(f"| [{BY_NAME[name].title}]({BY_NAME[name].url}) | " + " | ".join(cells)
                     + f" | {pct(r['structure_kept'])} | {r['blocked']} | {r.get('failed_closed', 0)} |")
    return lines


def corpus_table(manifest: dict) -> list[str]:
    lines = ["| Corpus | Real text | Labels | Docs | Langs | Direct spans | Quasi spans | License |",
             "|---|---|---|---:|---|---:|---:|---|"]
    for c in CORPUS_ORDER:
        m = manifest[c]
        lines.append(f"| [{m['title']}]({m['url']}) | {'yes' if m['real'] else 'synthetic'} | "
                     f"{m['annotation']}{'' if m['exhaustive_labels'] else ' (partial: recall only)'} | "
                     f"{m['docs']} | {' '.join(m['by_lang'])} | {m['spans_by_kind']['direct']} | "
                     f"{m['spans_by_kind']['quasi']} | {m['license']} |")
    return lines


def per_corpus_tables(text: dict) -> list[str]:
    names = sorted(text, key=lambda s: -text[s]["macro"]["protected_direct"]["value"])
    lines = []
    for metric, title in (("protected_direct", "Direct identifiers fully masked"),
                          ("char_recall", "Character recall (direct + quasi)"),
                          ("precision", "Character precision (exhaustively labeled corpora)"),
                          ("protected_quasi", "Quasi identifiers fully masked")):
        cols = [c for c in CORPUS_ORDER if metric != "precision" or c in EXHAUSTIVE]
        lines += [f"### {title}", "", "| System | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for n in names:
            cells = []
            for c in cols:
                m = text[n]["per_corpus"][c][metric]
                cell = ci(m) if metric == "protected_direct" else pct(m["value"])
                if c in CONTAMINATION.get(n, set()):
                    cell += " †"
                cells.append(cell)
            lines.append(f"| {BY_NAME[n].title} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("† trained on data from the same generator family as this corpus (in-distribution).")
    return lines


def breakdown_table(text: dict, key: str, title: str) -> list[str]:
    names = sorted(text, key=lambda s: -text[s]["macro"]["protected_direct"]["value"])
    cols = sorted({k for n in names for k in text[n][key]})
    n0 = text[names[0]][key]
    lines = [f"### {title}", "", "| System | " + " | ".join(f"{c} <sub>n={n0.get(c, {}).get('n', 0)}</sub>" for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for n in names:
        lines.append(f"| {BY_NAME[n].title} | " + " | ".join(
            pct(text[n][key][c]["protected"], 0) if c in text[n][key] else "n/a" for c in cols) + " |")
    return lines + [""]


def latency_table(text: dict, st: dict) -> list[str]:
    lines = ["| System | p50 per document | p95 per document | ms per 1,000 chars (median) | Structured request p50 |",
             "|---|---|---|---|---|"]
    for n in sorted(text, key=lambda s: text[s]["latency_ms"]["p50"] or 1e9):
        lt = text[n]["latency_ms"]
        lines.append(f"| {BY_NAME[n].title} | {ms(lt['p50'])} | {ms(lt['p95'])} | {ms(lt['ms_per_1k_chars'])} | "
                     f"{ms(st[n]['latency_ms_p50']) if n in st else 'n/a'} |")
    for n in st:
        if n not in text:
            lines.append(f"| {BY_NAME[n].title} | n/a | n/a | n/a | {ms(st[n]['latency_ms_p50'])} |")
    return lines


def system_table() -> list[str]:
    lines = ["| System | Kind | Environment | Scored on | Notes |", "|---|---|---|---|---|"]
    for s in SYSTEMS:
        suites = " + ".join(x for x, ok in (("text", s.text), ("requests", s.payload)) if ok)
        lines.append(f"| [{s.title}]({s.url}) | {KIND[s.kind]} | `envs/{s.env}.txt` | {suites} | {s.note} |")
    return lines


def write_results_md(text: dict, st: dict, manifest: dict) -> None:
    lines = [
        "# Detailed results",
        "",
        (f"Generated by `python -m pbench.report` on {date.today().isoformat()}. Every number below is "
         "recomputed from `results/runs/` (offsets and booleans, no text) against the gold spans of the "
         "public corpora. Bracketed ranges are 95% intervals from 2,000 document-level bootstrap resamples. "
         "See [METHODOLOGY.md](METHODOLOGY.md) for definitions."),
        "",
        "## Corpora", "", *corpus_table(manifest), "",
        "## Systems", "", *system_table(), "",
        "## Text suite, per corpus", "", *per_corpus_tables(text), "",
        "## Breakdowns (all corpora pooled, direct identifiers)", "",
        *breakdown_table(text, "per_type", "By identifier type"),
        *breakdown_table(text, "per_lang", "By language"),
        "## Structured suite", "",
        ("Share of direct-identifier values still present in the request the system forwards, per place "
         "the record sits (lower is better), with 95% intervals. A request the system blocks by policy, or "
         "answers with its own error without forwarding it (failed closed), leaks nothing and is not "
         "served either: both are counted next to the leak rates. \"breaks it\": most requests reached the "
         "provider without their original shape (see Request integrity), so the low rate is not protection. "
         "Rows are sorted by mean leak, a place the system cannot serve counting as fully leaked."), "",
    ]
    names = sorted(st, key=lambda s: structured_order(st[s]))
    lines += ["| System | " + " | ".join(CARRIER_TITLES[c] for c in CARRIERS)
              + " | Blocked | Failed closed | Errors |",
              "|---|" + "---|" * (len(CARRIERS) + 3)]
    for n in names:
        r = st[n]
        cells = [leak_cell(r, c, interval=True) for c in CARRIERS]
        lines.append(f"| {BY_NAME[n].title} | " + " | ".join(cells)
                     + f" | {r['blocked']} | {r.get('failed_closed', 0)} | {r['errors']} |")
    lines += ["", "### Request integrity", "",
              ("Share of forwarded requests that kept the original's shape (same messages and roles, same "
               "tool calls with their ids, tool results tied to their call, same multimodal parts), and "
               "share of tool-call arguments that still parse as JSON. A dropped tool call leaks nothing "
               "but breaks the agent."), "",
              "| System | " + " | ".join(CARRIER_TITLES[c] for c in CARRIERS) + " | Tool JSON still valid |",
              "|---|" + "---|" * (len(CARRIERS) + 1)]
    for n in names:
        r = st[n]
        cells = [pct(r["structure_kept_per_carrier"].get(c), 0) for c in CARRIERS]
        lines.append(f"| {BY_NAME[n].title} | " + " | ".join(cells) + f" | {pct(r['tool_args_json_valid'])} |")
    lines += ["", "## Latency", "", ("Single process, CPU only, one document at a time, after a warm-up call "
                                     "in each language. Request integrations that run behind a real proxy "
                                     "(LiteLLM) include its HTTP hops."), "", *latency_table(text, st), "",
              "## Errors and missing outcomes", "",
              "| System | Text documents failed | Requests failed |", "|---|---:|---:|"]
    for n in sorted(set(text) | set(st)):
        lines.append(f"| {BY_NAME[n].title} | {text.get(n, {}).get('errors', 'n/a')} | "
                     f"{st.get(n, {}).get('errors', 'n/a')} |")
    lines += ["", ("A failure (exception, crash, or more than 15 minutes on one document) counts as nothing "
                   "masked (text) or every value leaked (requests): a privacy layer that fails open leaks. A "
                   "system is ranked only once every document and request has an outcome."), "",
              "## Not tested", "", "| Product | Why |", "|---|---|",
              *[f"| {a} | {b} |" for a, b in NOT_TESTED], ""]
    (REPO / "RESULTS.md").write_text("\n".join(lines))


def write_readme_tables(text: dict, st: dict) -> None:
    readme = REPO / "README.md"
    body = readme.read_text()
    blocks = {
        "HEADLINE": "\n".join(headline_table(text)),
        "STRUCTURED": "\n".join(structured_table(st)),
    }
    for key, table in blocks.items():
        start, end = f"<!-- {key}:START -->", f"<!-- {key}:END -->"
        a, b = body.index(start) + len(start), body.index(end)
        body = body[:a] + "\n" + table + "\n" + body[b:]
    readme.write_text(body)


def main() -> None:
    manifest = json.loads((REPO / "results" / "corpora.json").read_text())
    docs = {c: read_jsonl(CORPORA / f"{c}.jsonl") for c in CORPUS_ORDER}
    text = text_results(docs)
    st = structured_results()
    (REPO / "results" / "summary.json").write_text(json.dumps(
        {"text": text, "structured": st}, indent=1, default=float) + "\n")
    write_results_md(text, st, manifest)
    write_readme_tables(text, st)
    print("wrote results/summary.json, RESULTS.md, README.md tables")


if __name__ == "__main__":
    main()
