"""Corpus model and the unified taxonomy every public dataset is mapped onto.

Every corpus in the benchmark is a public, licensed dataset with its own labels.
Nothing here is written by us: the loaders in `pbench.sources` only select
documents with a fixed seed and translate each source label into

- a **kind**: `direct` (identifies a person on its own: name, email, ID number,
  card, secret...), `quasi` (identifies in combination: date, city, employer,
  age...), or `neutral` (labeled by the source but not personal data under our
  policy: honorifics, amounts). Masking a neutral span is never a false positive
  and never a catch.
- a **type** from a short shared list, used only for per-type breakdowns.

Scoring is label-agnostic: a system is credited for masking the characters of a
gold span whatever type it assigns, the way a privacy proxy is judged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

KINDS = ("direct", "quasi", "neutral")

TYPES = (
    "PERSON", "CONTACT", "ADDRESS", "GOV_ID", "FINANCIAL", "SECRET", "NETWORK_ID",
    "RECORD_ID", "DATE_OF_BIRTH", "DATE", "LOCATION", "ORGANIZATION", "AGE",
    "DEMOGRAPHIC", "OTHER",
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str   # the source dataset's own label, kept for audit
    type: str    # one of TYPES
    kind: str    # one of KINDS


@dataclass
class Doc:
    id: str
    corpus: str
    lang: str
    text: str
    spans: list[Span]
    meta: dict = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def to_json(self) -> dict:
        out = asdict(self)
        out["spans"] = [[s.start, s.end, s.label, s.type, s.kind] for s in self.spans]
        return out

    @classmethod
    def from_json(cls, raw: dict) -> Doc:
        return cls(
            id=raw["id"], corpus=raw["corpus"], lang=raw["lang"], text=raw["text"],
            spans=[Span(*s) for s in raw["spans"]], meta=raw.get("meta", {}),
        )


def check_spans(doc: Doc) -> None:
    """Reject spans that fall outside the text or carry an unknown type/kind, so
    a loader bug fails loudly instead of skewing a score."""
    for s in doc.spans:
        if not 0 <= s.start < s.end <= len(doc.text):
            raise ValueError(f"{doc.id}: span {s} outside text of length {len(doc.text)}")
        if s.type not in TYPES or s.kind not in KINDS:
            raise ValueError(f"{doc.id}: span {s} has unknown type or kind")


def write_jsonl(path: Path, docs: list[Doc]) -> None:
    """Written to a temporary file and renamed, so a run reading the corpus never
    sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_json(), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[Doc]:
    with Path(path).open(encoding="utf-8") as f:
        return [Doc.from_json(json.loads(line)) for line in f if line.strip()]
