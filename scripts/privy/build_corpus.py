"""Select a reproducible, format-balanced sample from Privy's test-large split.

The publisher's dataset loader is read as documentation, never executed. Read
the JSON member directly without extracting the archive. Raw texts stay in an
ignored local cache; the committed manifest contains row IDs and counts only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from benchlib.paths import RESULTS, SOLUTIONS

DATASET = "beki/privy"
REVISION = "dc137a6a976f6b5bb8768e9bb51ec58df930ccd1"
MEMBER = "test-large.json"
FORMATS = ("json", "html", "xml", "sql_backticks", "sql_other")
SEED = 20260913


def iter_json_array(stream: TextIO, chunk_size: int = 65536) -> Iterator[dict]:
    """Read a large JSON array without keeping the 700 MB member in memory."""
    decoder = json.JSONDecoder()
    buffer = ""
    eof = False

    def fill() -> bool:
        nonlocal buffer, eof
        chunk = stream.read(chunk_size)
        eof = not chunk
        buffer += chunk
        if len(buffer) > 16 * 1024 * 1024:
            raise ValueError("A Privy record exceeds the 16 MiB parser limit")
        return bool(chunk)

    def trim() -> None:
        nonlocal buffer
        buffer = buffer.lstrip()
        while not buffer and not eof:
            fill()
            buffer = buffer.lstrip()

    trim()
    if not buffer.startswith("["):
        raise ValueError("Expected a JSON array")
    buffer = buffer[1:]
    trim()
    if buffer.startswith("]"):
        buffer = buffer[1:]
    else:
        while True:
            trim()
            while True:
                try:
                    row, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if not fill():
                        raise ValueError("Truncated or invalid Privy JSON") from None
            if not isinstance(row, dict):
                raise TypeError("Expected a Privy object")
            yield row
            buffer = buffer[end:]
            trim()
            if buffer.startswith("]"):
                buffer = buffer[1:]
                break
            if not buffer.startswith(","):
                raise ValueError("Missing JSON array separator")
            buffer = buffer[1:]
    trim()
    if buffer:
        raise ValueError("Unexpected trailing JSON content")


def trace_format(text: str) -> str | None:
    prefix = text.lstrip()
    if prefix.startswith(("{", "[")):
        return "json"
    if prefix.lower().startswith(("<table", "<html", "<!doctype")):
        return "html"
    if prefix.startswith(("<?xml", "b'<?xml", 'b"<?xml')):
        return "xml"
    if prefix.upper().startswith(("SELECT ", "INSERT ", "UPDATE ", "DELETE ")):
        return "sql_backticks" if "`" in prefix else "sql_other"
    return None


def normalize_row(row: dict, index: int) -> dict:
    text = row["full_text"]
    spans = []
    for entry in row["spans"]:
        span = json.loads(entry) if isinstance(entry, str) else entry
        if span["entity_type"] == "O":
            continue
        start, end = span["start_position"], span["end_position"]
        if not 0 <= start < end <= len(text) or text[start:end] != span["entity_value"]:
            raise ValueError("Invalid ground-truth offsets")
        spans.append([start, end, span["entity_type"]])
    return {
        "id": f"privy-test-large-{index}",
        "row_index": index,
        "lang": "en",
        "format": trace_format(text),
        "template_id": row["template_id"],
        "text": text,
        "spans": sorted({tuple(s) for s in spans}),
    }


def select_rows(
    rows: Iterator[dict], per_format: int, seed: int
) -> tuple[list[dict], dict]:
    """Reservoir sample without inspecting any detector output or entity type.

    One candidate per (format, template_id) limits repeated templates. Traverse
    the whole split so the sample is not a convenient prefix of the archive.
    """
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {name: [] for name in FORMATS}
    candidates: Counter = Counter()
    excluded: Counter = Counter()
    templates: set[tuple[str, int]] = set()
    total = 0
    for index, row in enumerate(rows):
        total += 1
        try:
            doc = normalize_row(row, index)
        except (ValueError, KeyError, TypeError):
            excluded["invalid_annotations"] += 1
            continue
        kind = doc["format"]
        if kind is None:
            excluded["unknown_format"] += 1
            continue
        key = (kind, doc["template_id"])
        if key in templates:
            excluded["repeated_template"] += 1
            continue
        templates.add(key)
        candidates[kind] += 1
        pool = pools[kind]
        if len(pool) < per_format:
            pool.append(doc)
        else:
            slot = rng.randrange(candidates[kind])
            if slot < per_format:
                pool[slot] = doc
    if any(len(pool) != per_format for pool in pools.values()):
        raise ValueError("Not enough valid distinct templates for every format")
    selected = [
        doc for kind in FORMATS for doc in sorted(pools[kind], key=lambda d: d["id"])
    ]
    stats = {
        "rows_read": total,
        "eligible_distinct_templates": dict(candidates),
        "excluded": dict(excluded),
        "docs": len(selected),
        "by_format": dict(Counter(d["format"] for d in selected)),
        "span_types": dict(Counter(s[2] for d in selected for s in d["spans"])),
        "zero_gold_docs": sum(not d["spans"] for d in selected),
    }
    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-format", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--archive", type=Path, help="Use an already downloaded archive"
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.per_format < 1:
        parser.error("--per-format must be positive")
    if args.archive:
        archive = args.archive
    else:
        from huggingface_hub import hf_hub_download

        archive = Path(
            hf_hub_download(
                DATASET,
                "privy-dataset.zip",
                repo_type="dataset",
                revision=REVISION,
                token=False,
                local_files_only=args.offline,
            )
        )
    with archive.open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    with (
        zipfile.ZipFile(archive) as compressed,
        io.TextIOWrapper(compressed.open(MEMBER), encoding="utf-8") as stream,
    ):
        docs, stats = select_rows(iter_json_array(stream), args.per_format, args.seed)
    payload = json.dumps(docs, ensure_ascii=False)
    (SOLUTIONS / "_privy_corpus.json").write_text(payload, encoding="utf-8")
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "archive_sha256": checksum,
        "member": MEMBER,
        "seed": args.seed,
        "per_format": args.per_format,
        "corpus_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "statistics": stats,
        "selection": [
            {k: d[k] for k in ("id", "row_index", "format", "template_id")}
            for d in docs
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "privy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
