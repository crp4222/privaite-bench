"""Scoring: pure functions from (gold spans, masked spans) to metrics, plus the
document-clustered bootstrap used for every confidence interval.

Definitions (all label-agnostic: a system is judged on which characters it
hides, never on the type name it gives them):

- A character is *masked* when it falls inside any span the system returned.
  Whitespace is ignored everywhere, so a system is neither credited nor blamed
  for the space between a first and a last name.
- A gold span is *protected* when every one of its characters is masked. A
  partially masked name or IBAN is not protected: the remainder still leaks.
- *Protected (direct)* = protected direct spans / direct spans. The headline.
- *Char recall* = masked gold characters / gold characters (direct + quasi).
- *Precision* = masked characters inside any gold span (including neutral spans)
  / all masked characters. Only on corpora whose labels are exhaustive.
- *F2* = 5PR / (4P + R) on char precision and char recall: recall weighted
  four times precision, because a leak costs more than an over-mask.

Uncertainty: documents, not spans, are the independent units (spans in one
document share its style and its formatting), so intervals come from resampling
documents with replacement (2,000 resamples, fixed seed), percentile method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pbench.corpus import Doc

BOOT = 2000
SEED = 20261023


def masked_mask(text: str, spans: list) -> np.ndarray:
    m = np.zeros(len(text), dtype=bool)
    for s, e, *_ in spans:
        s, e = max(0, int(s)), min(len(text), int(e))
        if e > s:
            m[s:e] = True
    return m


@dataclass
class DocScore:
    """Additive counts for one document; every metric is a ratio of sums."""
    direct_spans: int = 0
    direct_protected: int = 0
    quasi_spans: int = 0
    quasi_protected: int = 0
    gold_chars: int = 0          # direct + quasi
    gold_masked: int = 0
    direct_chars: int = 0
    direct_masked: int = 0
    masked_chars: int = 0        # all masked non-whitespace characters
    masked_in_labels: int = 0    # of those, inside any labeled span (any kind)
    error: bool = False


def score_doc(doc: Doc, spans: list | None) -> tuple[DocScore, list[tuple[str, bool]]]:
    """Counts for one document, plus (type, protected) per direct span for the
    per-type breakdown. `spans=None` means the system errored: nothing masked."""
    text = doc.text
    nonspace = ~np.char.isspace(np.array(list(text))) if text else np.zeros(0, bool)
    masked = masked_mask(text, spans or []) & nonspace
    labeled = np.zeros(len(text), dtype=bool)
    ds = DocScore(error=spans is None)
    per_type = []
    for sp in doc.spans:
        labeled[sp.start:sp.end] = True
        if sp.kind == "neutral":
            continue
        chars = nonspace[sp.start:sp.end]
        n = int(chars.sum())
        if n == 0:
            continue
        hit = int((masked[sp.start:sp.end] & chars).sum())
        protected = hit == n
        ds.gold_chars += n
        ds.gold_masked += hit
        if sp.kind == "direct":
            ds.direct_spans += 1
            ds.direct_protected += protected
            ds.direct_chars += n
            ds.direct_masked += hit
            per_type.append((sp.type, protected))
        else:
            ds.quasi_spans += 1
            ds.quasi_protected += protected
    ds.masked_chars = int(masked.sum())
    ds.masked_in_labels = int((masked & labeled).sum())
    return ds, per_type


FIELDS = ("direct_spans", "direct_protected", "quasi_spans", "quasi_protected", "gold_chars",
          "gold_masked", "direct_chars", "direct_masked", "masked_chars", "masked_in_labels")


def as_matrix(scores: list[DocScore]) -> np.ndarray:
    return np.array([[getattr(s, f) for f in FIELDS] for s in scores], dtype=float).reshape(-1, len(FIELDS))


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def metrics_from_sums(sums: np.ndarray) -> dict[str, np.ndarray]:
    """sums[..., field] -> metric arrays (works on one row or a bootstrap stack)."""
    f = {name: sums[..., i] for i, name in enumerate(FIELDS)}
    p = _ratio(f["masked_in_labels"], f["masked_chars"])
    # A system that masks nothing has no wrong masks: precision 1 by convention,
    # so it is ranked by its (zero) recall rather than dropped as undefined.
    p = np.where(f["masked_chars"] > 0, p, 1.0)
    r = _ratio(f["gold_masked"], f["gold_chars"])
    # Masking only unlabeled text with zero recall is an F2 of 0, not 0/0.
    f2 = np.where(4 * p + r == 0, 0.0, _ratio(5 * p * r, 4 * p + r))
    return {
        "protected_direct": _ratio(f["direct_protected"], f["direct_spans"]),
        "protected_quasi": _ratio(f["quasi_protected"], f["quasi_spans"]),
        "char_recall": r,
        "char_recall_direct": _ratio(f["direct_masked"], f["direct_chars"]),
        "precision": p,
        "f2": f2,
    }


def corpus_metrics(mat: np.ndarray, idx: np.ndarray) -> dict[str, dict]:
    """Point estimate and 95% interval for every metric of one corpus."""
    point = metrics_from_sums(mat.sum(axis=0))
    boot = metrics_from_sums(mat[idx].sum(axis=1))
    out = {}
    for k, v in point.items():
        b = boot[k][~np.isnan(boot[k])]
        lo, hi = (np.percentile(b, [2.5, 97.5]) if b.size else (np.nan, np.nan))
        out[k] = {"value": float(v), "lo": float(lo), "hi": float(hi)}
    return out
