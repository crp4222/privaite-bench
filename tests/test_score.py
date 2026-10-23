import numpy as np
import pytest

from pbench.corpus import Doc, Span
from pbench.score import as_matrix, corpus_metrics, metrics_from_sums, score_doc


def doc(text: str, spans: list[Span]) -> Doc:
    return Doc("d", "c", "en", text, spans)


TEXT = "Call Jane Doe at 555-0100 today."
NAME = Span(5, 13, "NAME", "PERSON", "direct")
PHONE = Span(17, 25, "PHONE", "CONTACT", "direct")
DAY = Span(26, 31, "DATE", "DATE", "quasi")


def test_partial_masking_is_not_protection():
    ds, types = score_doc(doc(TEXT, [NAME]), [(5, 9, "PERSON")])  # "Jane" only
    assert (ds.direct_spans, ds.direct_protected) == (1, 0)
    assert ds.direct_masked == 4 and ds.direct_chars == 7  # whitespace not counted
    assert types == [("PERSON", False)]


def test_whitespace_inside_a_span_is_neither_needed_nor_counted():
    # Two separate masks around the space still protect the full name.
    ds, _ = score_doc(doc(TEXT, [NAME]), [(5, 9, "X"), (10, 13, "X")])
    assert ds.direct_protected == 1
    assert ds.masked_chars == 7


def test_masks_outside_labels_lower_precision_and_neutral_spans_are_free():
    neutral = Span(0, 4, "TITLE", "OTHER", "neutral")  # "Call"
    ds, _ = score_doc(doc(TEXT, [NAME, neutral]), [(0, 4, "X"), (26, 31, "X")])
    assert ds.masked_chars == 9
    assert ds.masked_in_labels == 4  # "Call" is labeled (neutral), "today" is not
    assert ds.gold_chars == 7 and ds.gold_masked == 0  # neutral is not gold


def test_an_errored_document_masks_nothing():
    ds, _ = score_doc(doc(TEXT, [NAME, PHONE]), None)
    assert ds.error and ds.direct_protected == 0 and ds.masked_chars == 0


def test_spans_are_clipped_to_the_text():
    ds, _ = score_doc(doc(TEXT, [DAY]), [(26, 999, "X"), (-5, 2, "X")])
    assert ds.quasi_protected == 1


def test_metrics_formulas():
    ds, _ = score_doc(doc(TEXT, [NAME, PHONE, DAY]), [(5, 13, "X"), (26, 31, "X"), (0, 4, "X")])
    m = metrics_from_sums(as_matrix([ds])[0])
    assert m["protected_direct"] == pytest.approx(0.5)
    p, r = 12 / 16, 12 / 20  # masked in labels / masked ; gold masked / gold chars
    assert m["precision"] == pytest.approx(p)
    assert m["char_recall"] == pytest.approx(r)
    assert m["f2"] == pytest.approx(5 * p * r / (4 * p + r))


def test_masking_nothing_has_precision_one_and_recall_zero():
    ds, _ = score_doc(doc(TEXT, [NAME]), [])
    m = metrics_from_sums(as_matrix([ds])[0])
    assert m["precision"] == 1.0 and m["char_recall"] == 0.0 and m["f2"] == 0.0


def test_masking_only_unlabeled_text_has_f2_zero_not_undefined():
    ds, _ = score_doc(doc(TEXT, [NAME]), [(0, 4, "X")])
    m = metrics_from_sums(as_matrix([ds])[0])
    assert m["precision"] == 0.0 and m["char_recall"] == 0.0 and m["f2"] == 0.0


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    scores = []
    for i in range(30):
        spans = [(5, 13, "X")] if i % 3 else []
        scores.append(score_doc(doc(TEXT, [NAME]), spans)[0])
    mat = as_matrix(scores)
    idx = rng.integers(0, len(scores), size=(500, len(scores)))
    m = corpus_metrics(mat, idx)["protected_direct"]
    assert m["value"] == pytest.approx(20 / 30)
    assert m["lo"] < m["value"] < m["hi"]
