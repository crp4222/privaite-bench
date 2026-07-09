"""Loading `datasets/*.json` and the corpus file lists.

Unifies the two inline idioms (`json.load(open(...))` and
`json.loads(Path.read_text(...))`, which return the same thing) and the
skip-missing behavior. Callers pass their own file list: the P/R and structured
runners share PR_DATASETS, but bench.py deliberately loads a different set.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from benchlib.paths import DATASETS

# The 7-file corpus shared verbatim by bench_precision_recall.py and
# bench_structured.py (same files, same order). bench.py uses a different set.
PR_DATASETS = [
    "pii_samples.json",
    "corporate_samples.json",
    "batch_samples.json",
    "dlptest_us.json",
    "enedis_rse_extracts.json",
    "realworld_samples.json",
    "long_texts.json",
]


def load_json(path: str | Path) -> list | dict:
    """Parse a JSON file (utf-8)."""
    return json.loads(Path(path).read_text("utf-8"))


def load_dataset(name: str, base: Path = DATASETS) -> list[dict]:
    """One `datasets/<name>` file, or [] if it is missing."""
    path = Path(base) / name
    return load_json(path) if path.exists() else []  # type: ignore[return-value]


def load_datasets(names: Sequence[str], base: Path = DATASETS) -> list[dict]:
    """Concatenate several dataset files in order, skipping any that are absent."""
    corpus: list[dict] = []
    for name in names:
        corpus.extend(load_dataset(name, base))
    return corpus
