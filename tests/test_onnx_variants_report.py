"""The ONNX variants report turns per-variant result files into published numbers.

Agreement is the part that can quietly go wrong: it compares output digests and
the digests of the labelled values each variant leaves in place, against the
`q4f16` reference. These tests pin that arithmetic on small synthetic runs.
"""

from __future__ import annotations

import json

import pytest

from scripts.onnx_variants import report


def _run(variant: str, outputs: dict[str, str], leaked: dict[str, list[str]], recall: float) -> dict:
    return {
        "variant": variant,
        "privaite": "9.9.9",
        "bench": {"recall": recall, "recall_token_level": recall - 3, "false_positives": 2, "clean_docs": 14},
        "bench_seconds": 100.0,
        "gretel": {"recall": 60.0, "seconds": 40.0},
        "long_inputs": {"a source file": {"chars": 1234, "seconds": 3.5, "values": 4}},
        "window_seconds": {"512": 0.4, "1024": 0.8},
        "peak_rss_gb": 6.0,
        "output_digests": outputs,
        "leaked_labelled_value_digests": leaked,
    }


@pytest.fixture
def written(tmp_path, monkeypatch):
    folder = tmp_path / "results" / "onnx_variants"
    folder.mkdir(parents=True)
    # Three documents. q4 differs on doc c only; there it catches value "v2"
    # that the reference missed and misses nothing the reference caught.
    ref = _run("q4f16", {"a": "1", "b": "2", "c": "3"}, {"a": [], "b": ["v1"], "c": ["v2"]}, 84.9)
    q4 = _run("q4", {"a": "1", "b": "2", "c": "X"}, {"a": [], "b": ["v1"], "c": []}, 85.2)
    # int8 differs everywhere: like q4 it catches "v2" on doc c, and it leaves
    # "w1" in place on doc a, which the reference had caught.
    int8 = _run("quantized", {"a": "Y", "b": "Z", "c": "X"}, {"a": ["w1"], "b": ["v1"], "c": []}, 84.3)
    for name, data in (("q4f16", ref), ("q4", q4), ("quantized", int8)):
        (folder / f"{name}.json").write_text(json.dumps(data))
    monkeypatch.setattr(report, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(report, "BENCH", tmp_path)
    report.main()
    return (tmp_path / "ONNX_VARIANTS.md").read_text()


def _row(text: str, section: str, variant: str) -> list[str]:
    block = text.split(section, 1)[1]
    line = next(row for row in block.splitlines() if row.startswith(f"| `{variant}`"))
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_agreement_counts_identical_outputs_and_value_differences(written) -> None:
    section = "## Agreement with `q4f16`"
    assert _row(written, section, "q4")[1:] == ["2 / 3", "1", "0"]
    assert _row(written, section, "quantized")[1:] == ["0 / 3", "1", "1"]


def test_result_table_reports_each_variant_as_measured(written) -> None:
    row = _row(written, "## Result", "q4")
    assert row[1:4] == ["85.2%", "82.2%", "2 / 14"]
    assert _row(written, "## Result", "q4f16")[1] == "84.9%"


def test_the_reference_is_not_compared_with_itself(written) -> None:
    agreement = written.split("## Agreement with `q4f16`", 1)[1].split("##", 1)[0]
    assert "`q4f16`" not in agreement.split("|---|---|---|---|", 1)[1]


def test_provenance_names_the_release_the_measured_engine_shipped_as() -> None:
    measured_before_bump = {"privaite": "0.6.1", "engine_released_as": "0.7.0"}
    line = report.provenance(measured_before_bump)
    assert "released as privaite 0.7.0" in line and "still read 0.6.1" in line
    assert report.provenance({"privaite": "0.7.0"}).endswith("with privaite 0.7.0.")
