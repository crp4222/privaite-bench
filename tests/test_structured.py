import json

from pbench.corpus import Doc, Span
from pbench.structured import CARRIERS, args_valid, carriers, gold_values, leaks, not_forwarded, strings

TEXT = "Ticket: Jane Doe <jane@corp.com>\nIBAN DE89370400440532013000\nRef 12"


def test_every_carrier_carries_every_gold_value_before_any_system_runs():
    doc = Doc("d", "c", "en", TEXT, [Span(8, 16, "n", "PERSON", "direct"),
                                     Span(18, 31, "e", "CONTACT", "direct"),
                                     Span(38, 60, "i", "FINANCIAL", "direct")])
    values = gold_values(doc)
    for carrier, messages in carriers(TEXT).items():
        item = {"values": values}
        assert all(leaks(item, messages)), carrier
    assert set(carriers(TEXT)) == set(CARRIERS)


def test_short_values_and_neutral_spans_are_not_scored():
    doc = Doc("d", "c", "en", TEXT, [Span(65, 67, "n", "RECORD_ID", "direct"),
                                     Span(0, 6, "t", "OTHER", "neutral")])
    assert gold_values(doc) == []


def test_json_payloads_pass_through_as_tool_arguments():
    payload = '{"customer": {"name": "Jane Doe"}}'
    args = carriers(payload)["tool_call_args"][1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args) == {"customer": {"name": "Jane Doe"}}


def test_strings_decode_escaped_tool_arguments():
    escaped = json.dumps({"to": 'Ann "Nan" O’Neil'}, ensure_ascii=True)
    msgs = [{"role": "assistant", "tool_calls": [{"function": {"arguments": escaped}}]}]
    assert any('Ann "Nan" O’Neil' == s for s in strings(msgs))


def test_leaks_blocked_and_json_validity():
    item = {"values": [{"value": "Jane Doe"}, {"value": "jane@corp.com"}]}
    out = [{"role": "user", "content": "<PERSON> wrote from jane@corp.com"}]
    assert leaks(item, out) == [False, True]
    assert leaks(item, {"blocked": 400}) == [False, False]
    ok = [{"tool_calls": [{"function": {"arguments": '{"a": 1}'}}]}]
    broken = [{"tool_calls": [{"function": {"arguments": '{"a": <REDACTED>'}}]}]
    assert args_valid(ok) is True and args_valid(broken) is False
    assert args_valid([{"role": "user", "content": "x"}]) is None


def test_build_draws_from_the_plan_and_main_writes_the_suite(tmp_path, monkeypatch):
    from pbench import structured
    from pbench.corpus import write_jsonl

    for corpus, meta in (("nemotron", {"format": "structured"}), ("gretel_en", {}), ("gretel_fin", {}),
                         ("openpii", {}), ("privy", {"format": "json"})):
        docs = [Doc(f"{corpus}-{i}", corpus, "en", f'{{"name": "Jane Doe {i}"}}',
                    [Span(10, 18, "n", "PERSON", "direct")], meta) for i in range(3)]
        write_jsonl(tmp_path / f"{corpus}.jsonl", docs)
    monkeypatch.setattr(structured, "CORPORA", tmp_path)
    items = structured.build()
    assert len(items) == 5 * 3 * len(CARRIERS)
    assert {i["values"][0]["value"] for i in items} == {"Jane Doe"}
    assert len({i["doc"] for i in items}) == 15
    structured.main()
    assert len((tmp_path / "structured.jsonl").read_text().splitlines()) == len(items)


def test_structure_kept_catches_dropped_tool_calls_and_collapsed_parts():
    from pbench.structured import structure_kept

    msgs = carriers(TEXT)["tool_call_args"]
    redacted = json.loads(json.dumps(msgs))
    redacted[1]["tool_calls"][0]["function"]["arguments"] = '{"record": "<REDACTED>"}'
    assert structure_kept(msgs, redacted) is True  # values may change, shape may not
    dropped = json.loads(json.dumps(msgs))
    del dropped[1]["tool_calls"]
    assert structure_kept(msgs, dropped) is False
    unlinked = json.loads(json.dumps(msgs))
    del unlinked[2]["tool_call_id"]
    assert structure_kept(msgs, unlinked) is False
    mm = carriers(TEXT)["multimodal_text"]
    collapsed = [{"role": "user", "content": "text only"}]
    assert structure_kept(mm, collapsed) is False
    assert structure_kept(mm, mm[:0]) is False
    assert structure_kept(mm, {"blocked": 400}) is None


def test_a_request_answered_by_the_system_itself_leaks_nothing_and_has_no_shape():
    from pbench.structured import structure_kept

    item = {"values": [{"value": "Jane Doe", "kind": "direct"}],
            "messages": [{"role": "user", "content": "Jane Doe"}]}
    for out in ({"blocked": 400}, {"failed_closed": "internal error"}):
        assert not_forwarded(out)
        assert leaks(item, out) == [False]
        assert structure_kept(item["messages"], out) is None
    assert not not_forwarded(item["messages"]) and not not_forwarded({"other": 1})
