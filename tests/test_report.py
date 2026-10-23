import json

import pytest

from pbench import report
from pbench.corpus import Doc, Span
from pbench.registry import System
from pbench.structured import CARRIERS, item_sha


def docs(corpus, n):
    return [Doc(f"{corpus}-{i}", corpus, "en", f"Jane Doe {i}", [Span(0, 8, "n", "PERSON", "direct")])
            for i in range(n)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    good = System("good", "Good", "x", "library", True, True, "https://good")
    half = System("half", "Half", "x", "model", True, False, "https://half")
    monkeypatch.setattr(report, "SYSTEMS", [good, half])
    monkeypatch.setitem(report.BY_NAME, "good", good)
    monkeypatch.setitem(report.BY_NAME, "half", half)
    monkeypatch.setattr(report, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(report, "CORPORA", tmp_path / "corpora")
    monkeypatch.setattr(report, "CORPUS_ORDER", ["a", "b"])
    monkeypatch.setattr(report, "REAL", ["a"])
    monkeypatch.setattr(report, "EXHAUSTIVE", {"a", "b"})
    corpora = {"a": docs("a", 40), "b": docs("b", 40)}
    for name in ("good", "half"):
        (tmp_path / "runs" / name).mkdir(parents=True)
    for c, ds in corpora.items():
        with open(tmp_path / "runs" / "good" / f"{c}.jsonl", "w") as f:
            f.writelines(json.dumps({"id": d.id, "sha256": d.sha256, "spans": [[0, 8, "P"]], "ms": 2.0}) + "\n" for d in ds)
        with open(tmp_path / "runs" / "half" / f"{c}.jsonl", "w") as f:
            for k, d in enumerate(ds[:-1]):
                spans = [[0, 8, "P"]] if k % 2 else [[0, 4, "P"]]
                f.write(json.dumps({"id": d.id, "sha256": d.sha256, "spans": spans, "ms": 1.0}) + "\n")
            last = ds[-1]  # the system errored on the last document
            f.write(json.dumps({"id": last.id, "sha256": last.sha256, "error": "boom"}) + "\n")
    return corpora, tmp_path


def test_scores_ranking_ties_and_missing_outcomes(env):
    corpora, _ = env
    text = report.text_results(corpora)
    assert text["good"]["macro"]["protected_direct"]["value"] == 1.0
    assert text["good"]["tied_with_best"] is True
    half = text["half"]
    # 19 of 39 fully masked per corpus (odd k); the error document masks nothing.
    assert half["macro"]["protected_direct"]["value"] == pytest.approx(19 / 40)
    assert half["errors"] == 2 and half["tied_with_best"] is False
    assert half["macro_real"]["protected_direct"]["value"] == pytest.approx(19 / 40)
    lines = report.headline_table(text)
    assert lines[2].startswith("| 1 | [Good](https://good)") and lines[3].startswith("| 2 | [Half]")
    assert half["latency_ms"]["p50"] == 1.0


def test_a_system_with_missing_outcomes_is_not_ranked(env):
    corpora, tmp = env
    path = tmp / "runs" / "good" / "a.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    assert "good" not in report.text_results(corpora)


def test_changed_text_refuses_to_score(env):
    corpora, _ = env
    corpora["a"][0].text = "Jane Doe changed"
    with pytest.raises(ValueError, match="text hash changed"):
        report.text_results(corpora)


def test_structured_errors_count_as_leaks_blocks_do_not_and_partial_runs_are_not_ranked(env):
    _, tmp = env
    (tmp / "corpora").mkdir()
    values = [{"value": "Jane Doe", "type": "PERSON", "kind": "direct"},
              {"value": "Paris", "type": "LOCATION", "kind": "quasi"}]
    items = [{"id": f"d{k}#{c}", "doc": f"d{k}", "carrier": c, "values": values, "messages": [c]}
             for k in range(2) for c in CARRIERS]
    sha = {i["id"]: item_sha(i) for i in items}
    (tmp / "corpora" / "structured.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    clean = {"blocked": False, "leaked": [False, False], "args_valid": None, "structure_kept": True, "ms": 3}
    recs = {i["id"]: clean for i in items}
    recs.update({
        "d0#user_text": {"blocked": False, "leaked": [False, True], "args_valid": None, "ms": 3},
        "d1#user_text": {"blocked": True, "leaked": [False, False], "args_valid": None, "ms": 3},
        "d0#multimodal_text": {"blocked": False, "failed_closed": True, "leaked": [False, False],
                               "args_valid": None, "structure_kept": None, "ms": 3},
        "d0#tool_call_args": {"error": "boom"},
        "d1#tool_call_args": {"blocked": False, "leaked": [True, True], "args_valid": False,
                              "structure_kept": False, "ms": 3},
    })
    path = tmp / "runs" / "good" / "structured.jsonl"
    path.write_text("".join(json.dumps({"id": iid, "sha256": sha[iid], **r}) + "\n" for iid, r in recs.items()))
    st = report.structured_results()["good"]
    assert st["per_carrier"]["user_text"]["leak_direct"] == 0.0  # quasi leak not in headline
    assert st["per_carrier"]["tool_call_args"]["leak_direct"] == 1.0
    assert st["per_carrier"]["tool_result"]["leak_direct"] == 0.0
    assert st["blocked"] == 1 and st["errors"] == 1 and st["tool_args_json_valid"] == 0.0
    assert st["failed_closed"] == 1 and st["per_carrier"]["multimodal_text"]["leak_direct"] == 0.0
    assert report.structured_table({"good": st})[2].endswith("| 1 | 1 |")
    assert st["structure_kept_per_carrier"]["tool_call_args"] == 0.0
    assert st["structure_kept_per_carrier"]["tool_result"] == 1.0
    path.write_text("".join(json.dumps({"id": iid, "sha256": sha[iid], **r}) + "\n"
                            for iid, r in list(recs.items())[:-1]))
    assert "good" not in report.structured_results()
    items[0]["values"] = values[:1]  # gold changed after the run
    (tmp / "corpora" / "structured.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    path.write_text("".join(json.dumps({"id": iid, "sha256": sha[iid], **r}) + "\n" for iid, r in recs.items()))
    with pytest.raises(ValueError, match="rerun"):
        report.structured_results()


def test_readme_blocks_are_replaced_in_place(tmp_path, monkeypatch, env):
    corpora, _ = env
    monkeypatch.setattr(report, "REPO", tmp_path)
    (tmp_path / "README.md").write_text(
        "intro\n<!-- HEADLINE:START -->\nold\n<!-- HEADLINE:END -->\nmiddle\n"
        "<!-- STRUCTURED:START -->\n<!-- STRUCTURED:END -->\nend\n")
    report.write_readme_tables(report.text_results(corpora), {})
    body = (tmp_path / "README.md").read_text()
    assert body.startswith("intro\n<!-- HEADLINE:START -->\n| # | System")
    assert "old" not in body and body.endswith("<!-- STRUCTURED:END -->\nend\n")
    report.write_readme_tables(report.text_results(corpora), {})
    assert (tmp_path / "README.md").read_text() == body  # idempotent


def test_formatting():
    assert report.pct(0.12345) == "12.3%" and report.pct(None) == "n/a"
    assert report.pct(float("nan")) == "n/a"
    assert report.ms(3.14159) == "3.1 ms" and report.ms(1234.5) == "1234 ms"


def test_results_markdown_has_every_section(env, tmp_path, monkeypatch):
    corpora, _ = env
    monkeypatch.setattr(report, "REPO", tmp_path)
    monkeypatch.setattr(report, "INFO", {c: {"real": c == "a"} for c in ("a", "b")})
    manifest = {c: {"title": c.upper(), "url": "u", "real": c == "a", "annotation": "human",
                    "exhaustive_labels": True, "docs": 40, "by_lang": {"en": 40},
                    "spans_by_kind": {"direct": 40, "quasi": 0}, "license": "MIT"} for c in ("a", "b")}
    st = {"good": {"per_carrier": {c: {"leak_direct": 0.0, "lo": 0.0, "hi": 0.0, "values": 1}
                                   for c in CARRIERS},
                   "tool_args_json_valid": 1.0, "structure_kept": 1.0,
                   "structure_kept_per_carrier": {c: 1.0 for c in CARRIERS}, "blocked": 0, "errors": 0,
                   "latency_ms_p50": 5.0}}
    report.write_results_md(report.text_results(corpora), st, manifest)
    body = (tmp_path / "RESULTS.md").read_text()
    for heading in ("## Corpora", "## Systems", "## Text suite, per corpus", "### By language",
                    "## Structured suite", "## Latency", "## Errors and missing outcomes", "## Not tested"):
        assert heading in body
    assert "| Half | 2 | n/a |" in body


def test_carriers_a_system_cannot_serve_are_marked_and_sorted_as_leaked(env):
    def system(leak, kept, failed=0):
        return {"per_carrier": {c: {"leak_direct": leak, "lo": 0.0, "hi": 0.0, "values": 4, "requests": 2,
                                    "failed_closed": failed if c == "multimodal_text" else 0} for c in CARRIERS},
                "structure_kept_per_carrier": {c: kept for c in CARRIERS if not (failed and c == "multimodal_text")},
                "structure_kept": kept, "blocked": 0, "failed_closed": failed, "errors": 0}

    dropper = system(0.0, 0.0)        # leaks nothing because it drops what it cannot handle
    honest = system(0.3, 1.0)
    crasher = system(0.0, 1.0, failed=2)
    assert report.leak_cell(dropper, "tool_call_args") == "0.0% (breaks it)"
    assert report.leak_cell(crasher, "multimodal_text") == "fails closed"
    assert report.leak_cell(honest, "user_text", interval=True).startswith("30.0% <sub>")
    assert report.structured_order(dropper) == 1.0 and report.structured_order(honest) == 0.3
    assert report.structured_order(crasher) == pytest.approx(1 / len(CARRIERS))
    rows = report.structured_table({"half": dropper, "good": honest})
    assert rows[2].startswith("| [Good]") and rows[3].startswith("| [Half]")  # the dropper sorts last
    assert "(breaks it)" in rows[3] and "(breaks it)" not in rows[2]
