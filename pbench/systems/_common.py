"""Stdlib helpers shared by system adapters (imported inside every env)."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version

# Every system runs on CPU with the same thread budget so latency compares like
# for like on the benchmark machine.
THREADS = int(os.environ.get("PBENCH_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# No benchmark process phones home: DataFog and others honor these switches.
os.environ.setdefault("DATAFOG_NO_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")


def versions(*packages: str) -> dict[str, str]:
    out = {}
    for p in packages:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not installed"
    return out


def occurrences(text: str, value: str) -> list[tuple[int, int]]:
    out, i = [], text.find(value) if value else -1
    while i != -1:
        out.append((i, i + len(value)))
        i = text.find(value, i + len(value))
    return out


SPACY_MODELS = {
    "en": "en_core_web_lg", "fr": "fr_core_news_md", "de": "de_core_news_md",
    "es": "es_core_news_md", "it": "it_core_news_md", "pt": "pt_core_news_md",
    "nl": "nl_core_news_md",
}


def presidio_nlp_engine():
    """Presidio's spaCy engine as the package ships it (conf/default.yaml, whose
    NER block ignores organizations and numeric labels), with one model per
    benchmark language where the shipped file loads English only. Presidio's
    multi-language snippet passes a bare model list instead, which silently drops
    that block and makes spaCy organizations come back as PII."""
    from pathlib import Path

    import presidio_analyzer
    import yaml
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    conf = yaml.safe_load((Path(presidio_analyzer.__file__).parent / "conf" / "default.yaml").read_text())
    conf["models"] = [{"lang_code": k, "model_name": v} for k, v in SPACY_MODELS.items()]
    return NlpEngineProvider(nlp_configuration=conf).create_engine()


def windows(text: str, size: int = 1000, overlap: int = 150):
    """(offset, chunk) windows of at most `size` chars, cut on whitespace when
    possible, consecutive windows overlapping by about `overlap` chars.

    Used only for the standalone models, whose libraries silently truncate long
    inputs (GLiNER at 384 tokens, DeBERTa at 512): a raw model row measures the
    model, so it gets the minimal windowing any integrator would add. Products
    keep their own handling of long inputs, whatever it is."""
    if len(text) <= size:
        yield 0, text
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > start else end
        yield start, text[start:end]
        if end >= len(text):
            return
        nxt = text.rfind(" ", start, end - overlap)
        start = nxt + 1 if nxt > start else end - overlap


def dedupe(spans):
    """Drop exact duplicates produced by overlapping windows."""
    return sorted(set(spans))
