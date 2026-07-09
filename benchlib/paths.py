"""Repo paths and the single place that puts the PrivAiTe checkout on sys.path.

Every runner used to inline `sys.path.insert(0, .../PrivAiTe)` with its own
`parents[N]` arithmetic; they all resolved to `<git-parent>/PrivAiTe` (or the
`PRIVAITE_PATH` override). This centralizes that so there is one source of truth.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# benchlib/ lives directly under the repo root.
BENCH = Path(__file__).resolve().parents[1]
DATASETS = BENCH / "datasets"
RESULTS = BENCH / "results"
SOLUTIONS = BENCH / "solutions"


def privaite_path() -> str:
    """The PrivAiTe checkout to import: the PRIVAITE_PATH override, else the
    sibling `../PrivAiTe` next to this repo (what every runner assumed)."""
    return os.environ.get("PRIVAITE_PATH") or str(BENCH.parent / "PrivAiTe")


def ensure_privaite_path() -> None:
    """Prepend the PrivAiTe checkout to sys.path once, so `import privaite` works."""
    path = privaite_path()
    if path not in sys.path:
        sys.path.insert(0, path)


def bootstrap() -> None:
    """Top-of-module setup every runner needs before importing `privaite`: quiet the
    tokenizers fork warning and put the PrivAiTe checkout on the path. Call it before
    any `from privaite...` import."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ensure_privaite_path()
