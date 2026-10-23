import pytest

from pbench.systems._common import dedupe, occurrences, windows


@pytest.mark.parametrize("n", [0, 5, 999, 1000, 1001, 4321])
def test_windows_cover_the_text_with_correct_offsets(n):
    text = " ".join(f"w{i}" for i in range(n))
    covered = [False] * len(text)
    for off, chunk in windows(text, size=1000, overlap=150):
        assert text[off:off + len(chunk)] == chunk
        assert len(chunk) <= 1000
        for i in range(off, off + len(chunk)):
            covered[i] = True
    assert all(covered)


def test_windows_overlap_so_a_value_on_a_cut_is_seen_whole_once():
    text = "x " * 600 + "Jane Doe " + "y " * 600
    hits = [off + chunk.find("Jane Doe") for off, chunk in windows(text) if "Jane Doe" in chunk]
    assert hits and all(text[h:h + 8] == "Jane Doe" for h in hits)


def test_windows_without_spaces_still_progress():
    text = "a" * 2500
    chunks = list(windows(text))
    assert chunks[-1][0] + len(chunks[-1][1]) == 2500


def test_dedupe_and_occurrences():
    assert dedupe([(1, 2, "A"), (1, 2, "A"), (0, 1, "B")]) == [(0, 1, "B"), (1, 2, "A")]
    assert occurrences("abab", "ab") == [(0, 2), (2, 4)]
    assert occurrences("abc", "") == []


def test_kiji_byte_offsets_become_character_offsets():
    from pbench.systems.kiji_proxy import byte_to_char_offsets

    text = "Émilie à Zürich 😀 écrit à jürgen@exämple.de"
    value = "jürgen@exämple.de"
    raw = text.encode()
    b_start = raw.index(value.encode())
    s, e = byte_to_char_offsets(text, b_start, b_start + len(value.encode()))
    assert text[s:e] == value
