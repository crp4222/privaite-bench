import json
import os
import sys

import pytest

from scripts.run_kiji_privy import run_one


def test_timeout_kills_worker_and_discards_its_partial_report(tmp_path):
    report = tmp_path / "result.json"
    pid_file = tmp_path / "worker.pid"
    code = (
        "import os,sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text('{}'); "
        "Path(sys.argv[2]).write_text(str(os.getpid())); time.sleep(30)"
    )
    result = run_one(
        [sys.executable, "-c", code, str(report), str(pid_file)], report, 1
    )
    assert result["status"] == "timed_out"
    assert not report.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("Path(sys.argv[1]).write_text('{}'); sys.exit(7)", "failed_exit_7"),
        ("Path(sys.argv[1]).write_text('{')", "invalid_report"),
        ("pass", "missing_report"),
    ],
)
def test_unsuccessful_worker_never_leaves_a_publishable_report(tmp_path, code, status):
    report = tmp_path / "result.json"
    command = [
        sys.executable,
        "-c",
        "import sys; from pathlib import Path; " + code,
        str(report),
    ]
    assert run_one(command, report, 5)["status"] == status
    assert not report.exists()


def test_successful_worker_is_offline_and_preserves_its_result(tmp_path):
    report = tmp_path / "result.json"
    code = (
        "import os,sys; from pathlib import Path; "
        "assert os.environ['HF_HUB_OFFLINE']=='1'; "
        "assert os.environ['TRANSFORMERS_OFFLINE']=='1'; "
        "Path(sys.argv[1]).write_text('{\"valid\":true}')"
    )
    assert (
        run_one([sys.executable, "-c", code, str(report)], report, 5)["status"]
        == "complete"
    )
    assert json.loads(report.read_text()) == {"valid": True}


def test_existing_report_cannot_be_mistaken_for_a_successful_retry(tmp_path):
    report = tmp_path / "result.json"
    report.write_text("{}")
    with pytest.raises(FileExistsError):
        run_one([sys.executable, "-c", "raise SystemExit(1)"], report, 5)
    assert report.read_text() == "{}"
