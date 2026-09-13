from copy import deepcopy

import pytest

from scripts.render_kiji_privy import ORDER, validate, validate_matrix


@pytest.fixture
def reports():
    baseline = {
        "documents": [{"corpus": "fixture", "id": i, "bytes": 50} for i in range(420)],
        "privaite_source": "core-revision",
        "benchmark_source": "benchmark-revision",
        "benchmark_worktree_dirty": False,
        "benchmark_files_sha256": {"metrics.py": "pinned-hash"},
        "clean": [{} for _ in range(14)],
        "structured_probes": [
            {"tool_matches_flat": True, "multimodal_matches_flat": True}
            for _ in range(21)
        ],
        "detection_cache": False,
        "upstream_requests": 0,
        "regressions": [{"runs": [{}, {}, {}]} for _ in range(3)],
    }
    return {"onnx": baseline, "kiji": deepcopy(baseline)}


def test_complete_comparable_reports_are_accepted(reports):
    validate(reports)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("privaite_source", "another-core"),
        ("benchmark_source", "another-benchmark"),
        ("benchmark_worktree_dirty", True),
        ("benchmark_files_sha256", {"metrics.py": "changed-hash"}),
        ("detection_cache", True),
        ("upstream_requests", 1),
        ("documents", []),
        ("clean", []),
        ("structured_probes", []),
        ("regressions", []),
    ],
)
def test_incomparable_or_incomplete_results_cannot_be_published(reports, field, value):
    reports["kiji"][field] = value
    with pytest.raises(ValueError):
        validate(reports)


@pytest.mark.parametrize("carrier", ["tool_matches_flat", "multimodal_matches_flat"])
def test_a_single_structured_bypass_blocks_the_report(reports, carrier):
    reports["kiji"]["structured_probes"][-1][carrier] = False
    with pytest.raises(ValueError, match="Structured payload mismatch"):
        validate(reports)


def test_three_cases_with_one_missing_repetition_are_incomplete(reports):
    reports["kiji"]["regressions"][-1]["runs"].pop()
    with pytest.raises(ValueError, match="Incomplete regression"):
        validate(reports)


@pytest.mark.parametrize("status", [None, "running", "complete"])
def test_missing_candidate_requires_an_actual_failure_record(status):
    reports = {name: {} for name in ORDER if name != "onnx-kiji"}
    retry = {"solution": "onnx-kiji", "status": status} if status else None
    with pytest.raises(ValueError, match="Missing measurements"):
        validate_matrix(reports, retry)


def test_failed_candidate_is_explicitly_missing_not_scored_as_zero():
    reports = {name: {} for name in ORDER if name != "onnx-kiji"}
    validate_matrix(reports, {"solution": "onnx-kiji", "status": "timed_out"})
