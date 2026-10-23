"""Contract tests for every system adapter, each run in its own virtualenv.

These start the real worker of every system (skipped when its env is not
built) and check what the scoring relies on:

- spans are within the text and the system finds an obvious email;
- a request with no personal data comes back with exactly the same strings in
  every carrier. An adapter that dropped or rewrote a field would make the
  structured suite count a value as protected when the adapter, not the
  product, removed it;
- `info` reports versions.

They load models, so they are slow (minutes) and marked `adapters`:
    .venvs/harness/bin/python -m pytest -m adapters tests/test_adapters.py
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from pbench.registry import SYSTEMS
from pbench.structured import carriers, strings, structure_kept

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.adapters

# No date, number or name: dates and version strings are personal data to some
# detectors (Presidio reads "v2.3.1" as a driver's license), so the request
# must be one no system has a reason to touch.
CLEAN = ("Please summarize the notes below and suggest improvements to the onboarding "
         "checklist for new team members.")
CONTACT_TEXT = "Please write to jane.doe@example.org or call 415-555-0199 about the invoice."
CONTACT_VALUES = ("jane.doe@example.org", "415-555-0199")

# Measured product behavior on clean requests, asserted as such so that a change
# in either product shows up here: NeMo fails closed on multimodal content, and
# occludra drops assistant tool calls and the tool_call_id of tool results.
PRODUCT_DEVIATIONS = {
    ("nemo_guardrails", "multimodal_text"): "failed_closed",
    ("occludra", "tool_call_args"): "reshaped",
    ("occludra", "tool_call_nested"): "reshaped",
    ("occludra", "tool_result"): "reshaped",
}


def covered(spans, text: str, value: str) -> bool:
    """Every non-space character of `value` masked by the union of spans, the
    way the scoring counts it (adjacent pieces protect the whole)."""
    start = text.index(value)
    masked = set()
    for s, e, _ in spans:
        masked.update(range(s, e))
    return all(i in masked for i in range(start, start + len(value)) if not text[i].isspace())


def _worker(system):
    python = REPO / ".venvs" / system.env / "bin" / "python"
    if not python.exists():
        pytest.skip(f"env {system.env} not built")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    proc = subprocess.Popen([str(python), "-m", "pbench.worker", system.name], cwd=REPO, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    assert json.loads(proc.stdout.readline()) == {"ready": True}

    def ask(req):
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())

    return proc, ask


def leaves(messages) -> list[str]:
    """Message text, tool-call argument values and ids, ignoring the raw JSON
    form of arguments (a proxy may re-serialize it with other spacing) and the
    null content of assistant tool-call messages."""
    out = []
    for s in strings(messages):
        try:
            if isinstance(json.loads(s), (dict, list)):
                continue
        except ValueError:
            pass
        out.append(s)
    return out


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_adapter_contract(system):
    proc, ask = _worker(system)
    try:
        info = ask({"op": "info"})["info"]
        assert info.get("versions") or info.get("guardrail") or info.get("rails")
        if system.text:
            # Wiring, not quality: offsets land inside the text and cover at
            # least one of two obvious values.
            spans = ask({"op": "text", "text": CONTACT_TEXT, "lang": "en"})["spans"]
            assert all(0 <= s < e <= len(CONTACT_TEXT) for s, e, _ in spans)
            assert any(covered(spans, CONTACT_TEXT, v) for v in CONTACT_VALUES), spans
        if system.payload:
            for carrier, messages in carriers(CLEAN).items():
                reply = ask({"op": "payload", "messages": messages, "lang": "en"})
                assert "error" not in reply, (carrier, reply)
                out = reply["messages"]
                deviation = PRODUCT_DEVIATIONS.get((system.name, carrier))
                if deviation == "failed_closed":
                    assert isinstance(out, dict) and "failed_closed" in out, carrier
                    continue
                if deviation == "reshaped":
                    assert structure_kept(messages, out) is False, carrier
                    continue
                assert Counter(leaves(out)) == Counter(leaves(messages)), carrier
                assert [m["role"] for m in out] == [m["role"] for m in messages], carrier
    finally:
        proc.kill()
        proc.wait()


ORG_TEXT = "Jane Doe joined Microsoft in Paris last year."


def test_presidio_rows_keep_the_shipped_ner_configuration():
    """conf/default.yaml ignores spaCy organizations; a bare multi-language
    model list would bring them back. Checked on the library row and on the
    HTTP service behind LiteLLM's guardrail."""
    from pbench.registry import BY_NAME
    from pbench.systems._capture import free_port, wait_http

    proc, ask = _worker(BY_NAME["presidio"])
    try:
        spans = ask({"op": "text", "text": ORG_TEXT, "lang": "en"})["spans"]
    finally:
        proc.kill()
        proc.wait()
    org = ORG_TEXT.index("Microsoft")
    assert any(label == "PERSON" for _, _, label in spans)
    assert not any(s < org + len("Microsoft") and e > org for s, e, _ in spans)

    port = free_port()
    service = subprocess.Popen([str(REPO / ".venvs/presidio/bin/python"), "-m", "pbench.presidio_service",
                                str(port)], cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                               stderr=subprocess.DEVNULL)
    try:
        wait_http(f"http://127.0.0.1:{port}/health")
        import urllib.request
        req = urllib.request.Request(f"http://127.0.0.1:{port}/analyze", method="POST",
                                     data=json.dumps({"text": ORG_TEXT, "language": "en"}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            found = {x["entity_type"] for x in json.loads(r.read())}
    finally:
        service.kill()
        service.wait()
    assert "PERSON" in found and "ORGANIZATION" not in found


def test_occludra_does_not_rate_limit_the_benchmark_client():
    """Its shipped limit (10 requests a second per key) would turn fast
    sequential requests into 429s, scored as failures that say nothing about PII."""
    from pbench.registry import BY_NAME

    proc, ask = _worker(BY_NAME["occludra"])
    try:
        messages = carriers(CLEAN)["user_text"]
        replies = [ask({"op": "payload", "messages": messages, "lang": "en"}) for _ in range(40)]
    finally:
        proc.kill()
        proc.wait()
    assert [r.get("error") for r in replies] == [None] * 40


def test_piiranha_never_sees_more_than_its_256_token_context():
    """Its card states a 256-token context; a PII past that point in one window
    must still be found, from a model call no longer than 256 tokens."""
    python = REPO / ".venvs/models/bin/python"
    if not python.exists():
        pytest.skip("env models not built")
    code = """
import json
from pbench.systems.piiranha import System
s = System(); s.setup()
seen = []
fwd = s.pipe.model.forward
def spy(*a, **k):
    seen.append(int(k["input_ids"].shape[-1]))
    return fwd(*a, **k)
s.pipe.model.forward = spy
text = ("See Art. 12(3)(b), Reg. 2016/679, ref. KX-4821/B-77; " * 20)[:900] + " Call 415-555-0199."
window_tokens = len(s.pipe.tokenizer(text)["input_ids"])
print(json.dumps({"window_tokens": window_tokens, "spans": s.detect(text, "en"),
                  "phone": text.index("415"), "calls": seen}))
"""
    out = subprocess.run([str(python), "-c", code], cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    res = json.loads(out)
    assert res["window_tokens"] > 256  # the case the cap exists for
    assert res["calls"] and max(res["calls"]) <= 256
    assert any(s <= res["phone"] < e for s, e, _ in res["spans"])
