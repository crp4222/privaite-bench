import io
import json

import pytest

from pbench.privy import iter_json_array, normalize_row, select_rows, trace_format


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 65536])
def test_json_stream_handles_unicode_escapes_and_nested_records(chunk_size):
    rows = [{"text": 'é☀️ and "quotes"', "spans": [[1, 3], [4, 9]]}, {"data": {"x": []}}]
    encoded = json.dumps(rows, ensure_ascii=False)
    assert list(iter_json_array(io.StringIO(encoded), chunk_size)) == rows
    assert list(iter_json_array(io.StringIO("[]"), chunk_size)) == []


@pytest.mark.parametrize("text", ["[{}", "[{},]", "[{}{}]", "[{}] false", '{"a":1}', "[true]"])
def test_json_stream_rejects_invalid_or_partial_inputs(text):
    with pytest.raises((ValueError, TypeError)):
        list(iter_json_array(io.StringIO(text), 1))


def test_normalization_validates_original_offsets():
    row = {"full_text": '{"person":"Jane"}', "template_id": 1, "spans": [
        {"entity_type": "PERSON", "entity_value": "Jane", "start_position": 11, "end_position": 15}]}
    assert normalize_row(row, 4)["spans"] == [(11, 15, "PERSON")]
    row["spans"][0]["start_position"] = 10
    with pytest.raises(ValueError):
        normalize_row(row, 4)


def test_protocol_classification_keeps_bytes_repr_xml_intact():
    assert trace_format("b'<?xml version=\"1.0\"?><x/>'") == "xml"
    assert trace_format('<table border="1">') == "html"
    assert trace_format("SELECT `name` FROM `people`") == "sql_backticks"
    assert trace_format('INSERT INTO "people" VALUES (1)') == "sql_other"


def test_sampling_balances_formats_and_limits_duplicate_templates():
    examples = ['{"value":"x"}', "<table><td>x</td></table>", "b'<?xml?><x/>'", "SELECT `x`", 'SELECT "x"']
    rows = [{"full_text": t, "template_id": i, "spans": []} for i in range(8) for t in examples]
    selected, stats = select_rows(iter(rows + rows), 3, 42)
    assert len(selected) == 15
    assert stats["excluded"]["repeated_template"] == 40
    assert len({(d["format"], d["template_id"]) for d in selected}) == 15
    assert selected == select_rows(iter(rows + rows), 3, 42)[0]


def test_rows_and_templates_seen_in_development_are_never_selected():
    examples = ['{"value":"x"}', "<table><td>x</td></table>", "b'<?xml?><x/>'", "SELECT `x`", 'SELECT "x"']
    rows = [{"full_text": t, "template_id": i, "spans": []} for t in examples for i in range(6)]
    # Row 0 is JSON template 0; JSON template 3 is excluded wherever it appears.
    selected, stats = select_rows(iter(rows), 4, 1, exclude_rows=frozenset({0}),
                                  exclude_templates=frozenset({("json", 3)}))
    assert sorted(d["template_id"] for d in selected if d["format"] == "json") == [1, 2, 4, 5]
    assert stats["excluded"]["seen_in_development"] == 2
