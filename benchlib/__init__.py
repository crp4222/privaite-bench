"""Shared helpers for the privaite-bench runners.

One home for what every benchmark script used to re-implement: locating the
PrivAiTe checkout, the datasets-server fetch, loading `datasets/*.json`, and the
console report formatting. Import from here rather than copying the idiom.
"""

from __future__ import annotations

from benchlib.hf import fetch_rows, paginate, rows_url
from benchlib.paths import BENCH, DATASETS, RESULTS, SOLUTIONS, ensure_privaite_path

__all__ = [
    "BENCH", "DATASETS", "RESULTS", "SOLUTIONS", "ensure_privaite_path",
    "fetch_rows", "paginate", "rows_url",
]
