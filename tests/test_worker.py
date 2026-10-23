import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pbench.worker import WARMUP_TEXT

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def worker():
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    proc = subprocess.Popen([sys.executable, "-m", "pbench.worker", "tests.fake_system"],
                            cwd=REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    def ask(req: dict) -> dict:
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())

    assert json.loads(proc.stdout.readline()) == {"ready": True}
    yield ask, proc
    proc.kill()
    proc.wait()


def test_text_request_returns_spans_and_time(worker):
    ask, _ = worker
    reply = ask({"op": "text", "text": "Jane met Jane", "lang": "en"})
    assert reply["spans"] == [[0, 4, "PERSON"], [9, 13, "PERSON"]]
    assert reply["ms"] >= 0


def test_stdout_prints_from_the_system_never_reach_the_protocol(worker):
    ask, _ = worker
    # setup() and detect() both print; every reply must still parse as JSON.
    for _ in range(3):
        assert "spans" in ask({"op": "text", "text": "Jane", "lang": "en"})


def test_first_request_in_a_language_is_preceded_by_an_untimed_warm_up(worker):
    ask, _ = worker
    ask({"op": "text", "text": "Jane", "lang": "fr"})
    ask({"op": "text", "text": "Jane again", "lang": "fr"})
    calls = [tuple(c) for c in ask({"op": "info"})["info"]["calls"]]
    assert calls[0] == ("detect", WARMUP_TEXT, "en")
    fr = [c for c in calls if c[2] == "fr"]
    assert fr[0] == ("detect", WARMUP_TEXT, "fr")
    assert fr[1][0] == "payload"  # the warm-up also exercises the payload path
    assert [c[1] for c in fr if c[0] == "detect"] == [WARMUP_TEXT, "Jane", "Jane again"]


def test_a_failing_document_is_reported_and_the_worker_keeps_serving(worker):
    ask, _ = worker
    reply = ask({"op": "text", "text": "boom", "lang": "en"})
    assert reply["error"].startswith("RuntimeError")
    assert ask({"op": "text", "text": "Jane", "lang": "en"})["spans"] == [[0, 4, "PERSON"]]


def test_payload_and_unknown_ops(worker):
    ask, _ = worker
    reply = ask({"op": "payload", "lang": "en", "messages": [{"role": "user", "content": "hi Jane"}]})
    assert reply["messages"] == [{"role": "user", "content": "hi <P>"}]
    assert "unknown op" in ask({"op": "nope"})["error"]
