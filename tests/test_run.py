import json

import pytest

from pbench import run
from pbench.corpus import Doc, Span, write_jsonl
from pbench.registry import System
from pbench.structured import item_sha


@pytest.fixture
def fake(tmp_path, monkeypatch):
    spec = System("tests.fake_system", "Fake", "harness", "library", True, True, "")
    monkeypatch.setitem(run.BY_NAME, spec.name, spec)
    monkeypatch.setattr(run, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(run, "CORPORA", tmp_path / "corpora")
    logs = tmp_path / "home" / ".cache/privaite-bench/logs"
    logs.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # worker logs go under it
    (run.RUNS / spec.name).mkdir(parents=True)
    docs = [Doc(f"d{i}", "c", "en", f"Jane number {i}", [Span(0, 4, "n", "PERSON", "direct")])
            for i in range(3)]
    write_jsonl(run.CORPORA / "c.jsonl", docs)
    return spec, docs


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_text_writes_offsets_only_and_resumes(fake):
    spec, docs = fake
    worker = run.Worker(spec.name)
    try:
        run.run_text(worker, spec.name, "c", None)
        path = run.RUNS / spec.name / "c.jsonl"
        recs = records(path)
        assert [r["id"] for r in recs] == ["d0", "d1", "d2"]
        assert recs[0]["spans"] == [[0, 4, "PERSON"]] and "text" not in recs[0]
        assert recs[0]["sha256"] == docs[0].sha256
        # A second run has nothing to do and leaves the file untouched.
        before = path.read_text()
        run.run_text(worker, spec.name, "c", None)
        assert path.read_text() == before
        # A changed document text is rescored, the others are kept.
        docs[1].text = "Jane changed"
        write_jsonl(run.CORPORA / "c.jsonl", docs)
        run.run_text(worker, spec.name, "c", None)
        recs = {r["id"]: r for r in records(path)}
        assert recs["d1"]["sha256"] == docs[1].sha256 and len(recs) == 3
    finally:
        worker.close()


def test_a_failed_document_is_recorded_and_retried_next_run(fake):
    spec, docs = fake
    docs[2].text = "boom"
    docs[2].spans = []
    write_jsonl(run.CORPORA / "c.jsonl", docs)
    worker = run.Worker(spec.name)
    try:
        run.run_text(worker, spec.name, "c", None)
        path = run.RUNS / spec.name / "c.jsonl"
        assert "error" in records(path)[2]
        assert set(run._done(path)) == {"d0", "d1"}
    finally:
        worker.close()


def test_a_crashed_worker_is_restarted(fake):
    spec, _ = fake
    worker = run.Worker(spec.name)
    try:
        worker.proc.kill()
        worker.proc.wait()
        reply = worker.ask({"op": "text", "text": "Jane", "lang": "en"})
        assert "restarted" in reply["error"]
        assert worker.ask({"op": "text", "text": "Jane", "lang": "en"})["spans"] == [[0, 4, "PERSON"]]
    finally:
        worker.close()


def test_a_hanging_document_times_out_and_the_next_one_runs(fake, monkeypatch):
    spec, _ = fake
    monkeypatch.setattr(run, "TIMEOUT_S", 1)
    worker = run.Worker(spec.name)
    try:
        assert "restarted" in worker.ask({"op": "text", "text": "hang", "lang": "en"})["error"]
        assert worker.ask({"op": "text", "text": "Jane", "lang": "en"})["spans"] == [[0, 4, "PERSON"]]
    finally:
        worker.close()


def test_records_of_documents_dropped_from_a_corpus_are_purged(fake):
    spec, docs = fake
    worker = run.Worker(spec.name)
    try:
        run.run_text(worker, spec.name, "c", None)
        write_jsonl(run.CORPORA / "c.jsonl", docs[:2])
        run.run_text(worker, spec.name, "c", None)
        assert [r["id"] for r in records(run.RUNS / spec.name / "c.jsonl")] == ["d0", "d1"]
    finally:
        worker.close()


def test_run_structured_records_leaks_and_json_validity(fake):
    spec, _ = fake
    items = [
        {"id": "x#user_text", "doc": "x", "lang": "en", "carrier": "user_text",
         "messages": [{"role": "user", "content": "Jane wrote"}],
         "values": [{"value": "Jane", "kind": "direct"}]},
        {"id": "x#tool_result", "doc": "x", "lang": "en", "carrier": "tool_result",
         "messages": [{"role": "tool", "tool_call_id": "c", "content": "boom"}],
         "values": [{"value": "boom", "kind": "direct"}]},
    ]
    (run.CORPORA / "structured.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    worker = run.Worker(spec.name)
    try:
        run.run_structured(worker, spec.name, None)
    finally:
        worker.close()
    recs = records(run.RUNS / spec.name / "structured.jsonl")
    assert recs[0] == {"id": "x#user_text", "sha256": item_sha(items[0]), "blocked": False,
                       "failed_closed": False, "leaked": [False], "args_valid": None, "structure_kept": True,
                       "ms": recs[0]["ms"]}
    assert recs[1]["leaked"] == [True]
    # A changed gold list makes the outcome stale: it is redone, not reused.
    items[0]["values"] = [{"value": "wrote", "kind": "direct"}]
    (run.CORPORA / "structured.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    worker = run.Worker(spec.name)
    try:
        run.run_structured(worker, spec.name, None)
    finally:
        worker.close()
    assert records(run.RUNS / spec.name / "structured.jsonl")[1]["leaked"] == [True]
    assert {r["id"]: r for r in records(run.RUNS / spec.name / "structured.jsonl")}["x#user_text"]["leaked"] == [True]


def test_main_writes_info_and_skips_suites_a_system_does_not_have(fake, monkeypatch):
    spec, _ = fake
    text_only = System("tests.fake_system", "Fake", "harness", "library", True, False, "")
    monkeypatch.setitem(run.BY_NAME, spec.name, text_only)
    monkeypatch.setattr("sys.argv", ["run", "--systems", spec.name, "--corpora", "c"])
    run.main()
    info = json.loads((run.RUNS / spec.name / "info.json").read_text())
    assert info["title"] == "Fake" and "machine" in info and "calls" in info
    assert not (run.RUNS / spec.name / "structured.jsonl").exists()
    assert len(records(run.RUNS / spec.name / "c.jsonl")) == 3


def test_a_request_the_system_fails_closed_on_is_recorded_as_such(fake):
    spec, _ = fake
    item = {"id": "y#user_text", "doc": "y", "lang": "en", "carrier": "user_text",
            "messages": [{"role": "user", "content": "fail closed"}],
            "values": [{"value": "fail closed", "kind": "direct"}]}
    (run.CORPORA / "structured.jsonl").write_text(json.dumps(item))
    worker = run.Worker(spec.name)
    try:
        run.run_structured(worker, spec.name, None)
    finally:
        worker.close()
    rec = records(run.RUNS / spec.name / "structured.jsonl")[0]
    assert rec["failed_closed"] is True and rec["blocked"] is False
    assert rec["leaked"] == [False] and rec["structure_kept"] is None and rec["args_valid"] is None
