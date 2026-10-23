import pytest

from pbench.corpus import Doc, Span, check_spans, read_jsonl, write_jsonl
from pbench.sources import (
    MAPA,
    _spans,
    bio_spans,
    mapa_file_docs,
    piilo_doc,
    tab_doc,
    tokens_to_text,
)


def test_tokens_to_text_offsets():
    text, offs = tokens_to_text(["Hi", ",", "Jane", "\n\n", "x"], [False, True, False, False, False])
    assert text == "Hi, Jane\n\nx"
    assert [text[s:e] for s, e in offs] == ["Hi", ",", "Jane", "\n\n", "x"]


def test_bio_spans_join_continuations_and_split_on_label_change():
    offs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    tags = ["B-PERSON", "I-PERSON", "I-DATE", "O", "I-PERSON"]
    assert bio_spans(offs, tags) == [(0, 3, "PERSON"), (4, 5, "DATE"), (8, 9, "PERSON")]


def test_unmapped_source_labels_fail_loudly():
    with pytest.raises(KeyError):
        _spans("abc", [(0, 1, "NOT_A_LABEL")], MAPA)


def test_span_order_does_not_depend_on_hashing():
    raw = [(0, 3, "AMOUNT"), (0, 3, "PERSON"), (0, 3, "DATE")]
    assert [s.label for s in _spans("abc", raw, MAPA)] == ["AMOUNT", "DATE", "PERSON"]


def _mapa_row(n, tokens, tags, fine=None):
    return {"language": "fr", "file_name": "f", "sentence_number": n, "tokens": tokens,
            "coarse_grained": tags, "fine_grained": fine or ["O"] * len(tokens)}


def test_mapa_chunks_at_sentence_boundaries_and_keeps_offsets():
    rows = [_mapa_row(1, ["M.", "Dupont", "arrive"], ["O", "B-PERSON", "O"]),
            _mapa_row(0, ["Le", "12", "mai"], ["O", "B-DATE", "I-DATE"]),
            _mapa_row(2, ["Fin"], ["O"])]
    docs = mapa_file_docs("f", rows, max_chars=28)
    assert [d.text for d in docs] == ["Le 12 mai\nM. Dupont arrive", "Fin"]
    first = docs[0]
    assert [(first.text[s.start:s.end], s.kind) for s in first.spans] == [("12 mai", "quasi"),
                                                                       ("Dupont", "direct")]
    assert [d.id for d in docs] == ["mapa-f-0", "mapa-f-1"]


def _tab_ann(annotator, qc, mentions):
    return {"doc_id": "001", "annotator_id": annotator, "quality_checked": qc,
            "text": "Mr Galip Yalman lives in Ankara.", "entity_mentions": mentions}


def test_tab_prefers_the_quality_checked_annotator():
    m = lambda s, e, t, k: {"start_offset": s, "end_offset": e, "entity_type": t, "identifier_type": k}
    a1 = _tab_ann("annotator1", [], [m(3, 15, "PERSON", "DIRECT")])
    a2 = _tab_ann("annotator2", ["annotator2"], [m(3, 15, "PERSON", "DIRECT"), m(25, 31, "LOC", "QUASI")])
    d = tab_doc([a1, a2])
    assert d.meta == {"annotator": "annotator2", "quality_checked": True}
    assert [(s.type, s.kind) for s in d.spans] == [("PERSON", "direct"), ("LOCATION", "quasi")]
    assert tab_doc([a1]).meta["annotator"] == "annotator1"


def test_piilo_rebuild_must_reproduce_full_text():
    row = {"document": 7, "tokens": ["I", "am", "Ann", "Lee", "."],
           "trailing_whitespace": [True, True, True, False, False],
           "labels": ["O", "O", "B-NAME_STUDENT", "I-NAME_STUDENT", "O"],
           "full_text": "I am Ann Lee."}
    d = piilo_doc(row)
    assert [(d.text[s.start:s.end], s.type) for s in d.spans] == [("Ann Lee", "PERSON")]
    row["full_text"] = "I am Ann Lee!"
    with pytest.raises(ValueError):
        piilo_doc(row)


def test_doc_roundtrip_and_span_checks(tmp_path):
    d = Doc("x", "c", "en", "Jane", [Span(0, 4, "N", "PERSON", "direct")], {"k": 1})
    write_jsonl(tmp_path / "c.jsonl", [d])
    assert read_jsonl(tmp_path / "c.jsonl") == [d]
    check_spans(d)
    with pytest.raises(ValueError):
        check_spans(Doc("x", "c", "en", "Jane", [Span(0, 9, "N", "PERSON", "direct")]))
    with pytest.raises(ValueError):
        check_spans(Doc("x", "c", "en", "Jane", [Span(0, 4, "N", "NAME", "direct")]))


def test_row_converters_map_labels_and_offsets():
    from pbench.sources import gretel_en_doc, gretel_fin_doc, gretel_fin_valid, nemotron_doc, openpii_doc

    d = openpii_doc({"uid": "u1", "language": "fr", "region": "FR", "source_text": "Mme Ronce, née le 2/3/1990",
                     "privacy_mask": [{"start": 0, "end": 3, "label": "TITLE"},
                                      {"start": 4, "end": 9, "label": "SURNAME"},
                                      {"start": 18, "end": 26, "label": "DATE"}]})
    assert [(d.text[s.start:s.end], s.kind) for s in d.spans] == [("Mme", "neutral"), ("Ronce", "direct"),
                                                                  ("2/3/1990", "quasi")]
    n = nemotron_doc({"uid": "n1", "text": "Brian, SSN 123-45-6789", "locale": "us",
                      "document_format": "unstructured", "domain": "Life",
                      "spans": "[{'start': 0, 'end': 5, 'label': 'first_name'}, "
                               "{'start': 11, 'end': 22, 'label': 'ssn'}]"})
    assert [(s.type, s.kind) for s in n.spans] == [("PERSON", "direct"), ("GOV_ID", "direct")]
    assert n.meta["format"] == "unstructured"
    g, missing = gretel_en_doc({"uid": "g1", "text": "Ann Lee met Ann.", "domain": "x", "document_type": "y",
                                "entities": "[{'entity': 'Ann', 'types': ['first_name']}, "
                                            "{'entity': 'Bob', 'types': ['first_name']}]"})
    assert [(s.start, s.end) for s in g.spans] == [(0, 3), (12, 15)] and missing == 1
    row = {"index": 7, "language": "German", "generated_text": "IBAN DE89370400440532013000",
           "document_type": "x", "pii_spans": '[{"start": 5, "end": 27, "label": "iban"}]'}
    assert gretel_fin_valid(row) and gretel_fin_doc(row).lang == "de"
    assert not gretel_fin_valid({**row, "pii_spans": '[{"start": 5, "end": 99, "label": "iban"}]'})


def test_mapa_titles_are_neutral_and_split_from_the_name():
    row = _mapa_row(0, ["Herr", "Gerard", "Cotter", "kam"], ["B-PERSON", "I-PERSON", "I-PERSON", "O"],
                    ["B-TITLE", "B-GIVEN NAME", "B-FAMILY NAME", "O"])
    d = mapa_file_docs("f", [row])[0]
    assert [(d.text[s.start:s.end], s.kind) for s in d.spans] == [("Herr", "neutral"),
                                                                  ("Gerard Cotter", "direct")]


def test_leading_honorifics_become_neutral_in_every_corpus():
    from pbench.sources import split_honorifics

    text = "Mr Ruslan Kurkaev met Dr. Wong, Mrs and Ms."
    spans = [Span(0, 17, "PERSON", "PERSON", "direct"), Span(22, 30, "n", "PERSON", "direct"),
             Span(32, 35, "n", "PERSON", "direct"), Span(0, 2, "t", "OTHER", "neutral")]
    out = split_honorifics(text, spans)
    got = [(text[s.start:s.end], s.kind) for s in out]
    assert ("Mr", "neutral") in got and ("Ruslan Kurkaev", "direct") in got
    assert ("Dr.", "neutral") in got and ("Wong", "direct") in got
    assert ("Mrs", "direct") in got  # a lone honorific-like word is left alone
