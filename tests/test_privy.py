from __future__ import annotations

import io
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.privy.build_corpus import (
    iter_json_array,
    normalize_row,
    select_rows,
    trace_format,
)
from scripts.privy.metrics import coverage, text_recall
from solutions.kiji import KijiDetector


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 65536])
def test_json_stream_handles_unicode_escapes_and_nested_records(chunk_size):
    rows = [{"text": 'é☀️ and "quotes"', "spans": [[1, 3], [4, 9]]}, {"data": {"x": []}}]
    encoded = json.dumps(rows, ensure_ascii=False)
    assert list(iter_json_array(io.StringIO(encoded), chunk_size)) == rows
    assert list(iter_json_array(io.StringIO("[]"), chunk_size)) == []


@pytest.mark.parametrize(
    "text", ["[{}", "[{},]", "[{}{}]", "[{}] false", '{"a":1}', "[true]"]
)
def test_json_stream_rejects_invalid_or_partial_inputs(text):
    with pytest.raises((ValueError, TypeError)):
        list(iter_json_array(io.StringIO(text), 1))


def test_full_span_recall_does_not_credit_partial_redaction():
    text = "Name: Jane Doe; id=123456"
    gold = [[6, 14, "PERSON"], [19, 25, "ID"]]
    result = coverage(text, gold, [[6, 10, "PERSON"], [19, 22, "ID"]])
    assert result["complete_spans"] == 0
    assert result["partial_spans"] == 2
    assert result["covered_chars"] == 7


def test_precision_penalizes_swallowed_labels_and_quotes():
    text = 'email="abc@example.test"'
    result = coverage(text, [[7, 23, "EMAIL"]], [[0, len(text), "SECRET"]])
    assert result["complete_spans"] == 1
    assert result["non_gold_chars_removed"] == 8


def test_gold_overlap_is_not_double_counted_in_character_metrics():
    result = coverage("abcde", [[0, 4, "A"], [2, 5, "B"]], [[0, 5, "X"]])
    assert result["gold_chars"] == result["covered_chars"] == 5
    assert result["complete_spans"] == 2


def test_literal_scoring_checks_all_occurrences_and_keeps_short_values():
    text = "Jane Doe; Jane Doe; id=AB"
    result = text_recall(
        text, "<PERSON>; Jane Doe; id=AB", [[0, 8, "PERSON"], [22, 24, "ID"]]
    )
    assert result["absent_values"] == result["strict_absent_values"] == 0


def test_normalization_validates_original_offsets_before_scoring():
    row = {
        "full_text": '{"person":"Jane"}',
        "template_id": 1,
        "spans": [
            {
                "entity_type": "PERSON",
                "entity_value": "Jane",
                "start_position": 11,
                "end_position": 15,
            },
        ],
    }
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
    examples = [
        '{"value":"x"}',
        "<table><td>x</td></table>",
        "b'<?xml?><x/>'",
        "SELECT `x`",
        'SELECT "x"',
    ]
    rows = [
        {"full_text": text, "template_id": i, "spans": []}
        for i in range(8)
        for text in examples
    ]
    selected, stats = select_rows(iter(rows + rows), 3, 42)
    assert len(selected) == 15
    assert stats["excluded"]["repeated_template"] == 40
    assert len({(d["format"], d["template_id"]) for d in selected}) == 15
    assert selected == select_rows(iter(rows + rows), 3, 42)[0]


class WindowTokenizer:
    """Fake classifier input: the only sensitive token is past window one."""

    def __call__(self, text, **kwargs):
        words = [(m.group(), m.span()) for m in re.finditer(r"\S+", text)]
        rows = []
        size = kwargs["max_length"] - 2
        step = size - kwargs["stride"]
        for start in range(0, len(words), step):
            part = words[start : start + size]
            ids = [0] + [1 if word == "canary" else 0 for word, _ in part] + [0]
            offsets = [(0, 0)] + [offset for _, offset in part] + [(0, 0)]
            rows.append((ids, offsets))
            if not kwargs.get("return_overflowing_tokens") or start + size >= len(
                words
            ):
                break
        return {
            "input_ids": [r[0] for r in rows],
            "attention_mask": [[1] * len(r[0]) for r in rows],
            "offset_mapping": [r[1] for r in rows],
        }


class TokenSession:
    def get_inputs(self):
        return [
            SimpleNamespace(name="input_ids"),
            SimpleNamespace(name="attention_mask"),
        ]

    def run(self, names, feed):
        tokens = feed["input_ids"][0]
        logits = np.full((1, len(tokens), 53), -20.0)
        for i, token in enumerate(tokens):
            logits[0, i, int(token)] = 20.0
        return [logits]


@pytest.mark.asyncio
async def test_kiji_scans_after_512_tokens_with_absolute_unicode_offsets():
    detector = KijiDetector()
    detector.tokenizer, detector.session = WindowTokenizer(), TokenSession()
    detector.labels = {0: "O", 1: "B-PASSWORD"}
    text = "É☀️ " + "word " * 1100 + "canary"
    spans = await detector.detect(text)
    assert len(spans) == 1
    assert spans[0].entity_type == "SECRET"
    assert spans[0].start == text.index("canary") and spans[0].end == len(text)


@pytest.mark.asyncio
async def test_kiji_uninitialized_does_not_return_empty_detections():
    with pytest.raises(RuntimeError, match="not initialized"):
        await KijiDetector().detect("canary")
