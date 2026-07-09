"""Thin client for the Hugging Face datasets-server rows API.

The corpus builders (`build_gretel_corpus`, `build_nemotron_corpus`,
`ai4privacy_loader`) each inlined the same paginated `urlopen(...offset=...&length=...)`
fetch. This is that fetch, once. Each caller keeps its own accumulation/stop logic
and just pulls rows from `paginate`.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator

ROWS_API = "https://datasets-server.huggingface.co/rows"


def rows_url(dataset: str, config: str = "default", split: str = "train") -> str:
    """Base rows-API URL for a dataset; append `&offset=&length=` per page."""
    return f"{ROWS_API}?dataset={dataset}&config={config}&split={split}"


def fetch_rows(base_url: str, offset: int, length: int = 100, timeout: int = 60) -> list[dict]:
    """One page of rows (the raw `{"row": {...}}` entries) from the rows API."""
    with urllib.request.urlopen(f"{base_url}&offset={offset}&length={length}", timeout=timeout) as r:
        return json.load(r).get("rows", [])


def paginate(base_url: str, length: int = 100, max_offset: int = 4000,
             timeout: int = 60) -> Iterator[dict]:
    """Yield row entries page by page until an empty page or `max_offset`.

    Lazy: the next page is fetched only when the caller asks for the next item, so
    a caller that `break`s once it has enough never triggers an extra request.
    """
    offset = 0
    while offset < max_offset:
        page = fetch_rows(base_url, offset, length, timeout)
        if not page:
            return
        yield from page
        offset += length
