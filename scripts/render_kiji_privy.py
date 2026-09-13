"""Render the Kiji/Privy tables from recorded numeric reports, with validity gates."""

from __future__ import annotations

import json

from benchlib.paths import BENCH, RESULTS

ORDER = ("light", "onnx", "kiji", "kiji-argmax", "kiji-presidio", "onnx-kiji")


def validate(reports: dict) -> None:
    reference = reports["onnx"]
    expected = {(d["corpus"], str(d["id"]), d["bytes"]) for d in reference["documents"]}
    if len(expected) != 420:
        raise ValueError("Expected exactly 420 distinct documents")
    for name, report in reports.items():
        observed = {
            (d["corpus"], str(d["id"]), d["bytes"]) for d in report["documents"]
        }
        if observed != expected or len(report["documents"]) != 420:
            raise ValueError(f"Different input set for {name}")
        if report["privaite_source"] != reference["privaite_source"]:
            raise ValueError("Different PrivAiTe source revisions")
        if report["benchmark_source"] != reference["benchmark_source"]:
            raise ValueError("Different benchmark source revisions")
        if report["benchmark_worktree_dirty"]:
            raise ValueError("Uncommitted benchmark implementation")
        if report["benchmark_files_sha256"] != reference["benchmark_files_sha256"]:
            raise ValueError("Different benchmark implementations")
        if len(report["clean"]) != 14 or len(report["structured_probes"]) != 21:
            raise ValueError(f"Incomplete controls for {name}")
        if report["detection_cache"] or report["upstream_requests"]:
            raise ValueError("Unexpected cache or upstream traffic")
        if any(
            not p["tool_matches_flat"] or not p["multimodal_matches_flat"]
            for p in report["structured_probes"]
        ):
            raise ValueError(f"Structured payload mismatch for {name}")
        if len(report["regressions"]) != 3 or any(
            len(r["runs"]) != 3 for r in report["regressions"]
        ):
            raise ValueError(f"Incomplete regression repetitions for {name}")


def number(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def validate_matrix(reports: dict, retry: dict | None) -> None:
    missing = set(ORDER) - set(reports)
    if not missing:
        return
    if missing != {"onnx-kiji"} or retry is None:
        raise ValueError("Missing measurements without a recorded runtime failure")
    if retry.get("solution") != "onnx-kiji" or retry.get("status") not in (
        "timed_out",
        "failed_exit_1",
    ):
        raise ValueError("Missing measurements without a completed failure record")


def breakdown_cell(reports: dict, solution: str, corpus: str, *keys: str) -> str:
    if solution not in reports:
        return "n/a"
    result = reports[solution]["corpora"][corpus]
    for key in keys:
        result = result[key]
    value = result["full_span_recall_pct"]
    return "n/a" if value is None else f"{value:.2f}%"


def render(
    reports: dict, retry: dict | None = None, failure: dict | None = None
) -> str:
    validate_matrix(reports, retry)
    validate(reports)
    successful = [name for name in ORDER if name in reports]
    lines = [
        "# Privy protocol traces and Kiji ONNX",
        "",
        (
            f"Measured on 2026-09-13: six configurations attempted, {len(successful)} completed, "
            "with identical inputs, CPU inference, "
            "and the detection cache disabled. Results are from local engine requests, not a "
            "live agent/provider session. [Method and reproduction](scripts/privy/README.md)."
        ),
        "",
        "## Decision",
        "",
        (
            "Keep the current `onnx` default. Kiji is a benchmark-only candidate: replacing "
            "Privacy Filter with this ONNX artifact trades away too much detection and "
            "precision on protocol traces. Kiji alone also leaves nine of ten planted "
            "credential occurrences intact in the long log. The deterministic rules retained "
            "in the Presidio combinations cover those specific formats, but do not restore "
            "broad password coverage: `onnx` catches 12/15 Privy passwords, `kiji-presidio` 2/15."
        ),
        "",
        (
            "The default also has substantial misses: 233/491 Privy spans are not fully "
            "covered. Its three missed passwords occur in SQL traces. Prioritize a separate "
            "SQL/field-context evaluation with exact value boundaries; do not tune on this "
            "test sample and report the tuned score as independent evidence. Adding entity "
            "categories or passing a planted-secret replay is insufficient by itself."
        ),
        "",
    ]
    if failure:
        lines += [
            "## Runtime failure retained",
            "",
            (
                "The first `onnx-kiji` attempt was terminated after more than ten minutes "
                "without completion. Its last document-progress marker was 400/420; a "
                "regression temporary directory was active. A one-second native stack sample "
                "showed Privacy Filter's `QMoECPU` inside nested ONNX Runtime thread-pool "
                "parallel sections, spinning in `EndParallelSectionInternal`. This is evidence "
                "of a local runtime stall, not an established root cause or a network timeout. "
                "The failed attempt has no completed scores and is never averaged into latency."
            ),
            "",
            (
                f"An identical fresh-process retry ended with status **{retry['status']}**, "
                f"after {retry['elapsed_seconds']:.1f} seconds, under a "
                f"{retry['watchdog_seconds']}-second watchdog. "
                + (
                    "The `onnx-kiji` table entries come from that retry; one successful retry "
                    "does not establish runtime reliability."
                    if "onnx-kiji" in reports
                    else "The tables contain only the five completed configurations; `n/a` "
                    "means missing measurements, not zero recall or zero latency."
                )
            ),
            "",
            (
                "[Initial failure record](results/kiji_privy/onnx-kiji-initial-failure.json) · "
                "[Retry status](results/kiji_privy/onnx-kiji-retry-status.json). "
                "Isolate model/session lifecycle and thread-pool settings before considering "
                "the combined configuration for production. This experiment did not change "
                "PrivAiTe's runtime settings."
            ),
            "",
        ]
    if "onnx-kiji" in reports:
        current = reports["onnx"]["corpora"]["privy"]
        hybrid = reports["onnx-kiji"]["corpora"]["privy"]
        lines += [
            "## Adding Kiji: measurable gain, unresolved tradeoffs",
            "",
            (
                f"The combined stack fully covers {hybrid['counts']['complete_spans']}/491 "
                f"Privy spans versus {current['counts']['complete_spans']}/491 for `onnx`. "
                f"On AI4Privacy, full-span recall rises from "
                f"{reports['onnx']['corpora']['ai4privacy']['full_span_recall_pct']:.2f}% to "
                f"{reports['onnx-kiji']['corpora']['ai4privacy']['full_span_recall_pct']:.2f}%. "
                "That is a useful experimental signal, but Privy password coverage remains "
                f"12/15 and non-gold character removal rises from "
                f"{current['non_gold_removal_pct']:.2f}% to {hybrid['non_gold_removal_pct']:.2f}%. "
                "More clean documents change, and the long log takes longer. Keep this "
                "option in the benchmark until runtime reliability and unwanted redaction "
                "are addressed and independently re-evaluated."
            ),
            "",
        ]
    lines += [
        "## Privy: 300 traces, 491 annotated spans",
        "",
        (
            "A full span is caught only when all its non-whitespace characters are removed. "
            "Character precision is relative to the supplied annotations. Non-gold removal "
            "counts other characters removed, including useful syntax; lower is better."
        ),
        "These traces are short: 29–2,156 bytes, median 201 bytes, p95 695 bytes. The long-log replay below measures a separate input scale.",
        "",
        "| Configuration | Full-span recall | Character recall | Character precision | Non-gold removal | Mean / p95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in successful:
        r = reports[name]["corpora"]["privy"]
        lines.append(
            f"| `{name}` | {number(r['full_span_recall_pct'])}% | "
            f"{number(r['character_recall_pct'])}% | {number(r['character_precision_pct'])}% | "
            f"{number(r['non_gold_removal_pct'])}% | {number(r['latency']['mean_ms'])} / "
            f"{number(r['latency']['p95_ms'])} ms |"
        )
    lines += [
        "",
        "## AI4Privacy cross-check: 120 documents",
        "",
        (
            "The first two recall columns use the original comparison's literal and strict-token "
            "criteria. They are different from full character coverage, reported separately."
        ),
        "",
        "| Configuration | Literal recall | Strict-token recall | Full-span recall | Mean / p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in successful:
        r = reports[name]["corpora"]["ai4privacy"]
        lines.append(
            f"| `{name}` | {number(r['literal_recall_pct'])}% | "
            f"{number(r['strict_literal_recall_pct'])}% | {number(r['full_span_recall_pct'])}% | "
            f"{number(r['latency']['mean_ms'])} / {number(r['latency']['p95_ms'])} ms |"
        )
    lines += [
        "",
        "## False-positive controls and memory",
        "",
        (
            "The 78 Privy documents here contain no **annotated** PII. Incidental PII "
            "may be unannotated; these counts should not be interpreted as independently "
            "audited precision. The separate 14-document clean corpus is unchanged."
        ),
        "",
        "| Configuration | Privy empty-gold docs changed / 78 | Clean docs changed / 14 | Clean spans removed | Peak RSS |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in successful:
        r = reports[name]
        lines.append(
            f"| `{name}` | {r['corpora']['privy']['zero_gold_docs_changed']} | "
            f"{sum(c['changed'] for c in r['clean'])} | {sum(c['spans'] for c in r['clean'])} | "
            f"{number(r['peak_rss_mib'])} MiB |"
        )
    lines += [
        "",
        "## Long-log regression",
        "",
        (
            "Same synthetic 69,068-byte log, two planted credentials with five occurrences "
            "each. Timings are medians of three runs. EN-only Presidio configuration; "
            "do not compare the absolute time with the earlier EN/FR replay as a speedup."
        ),
        "",
        "| Configuration | Median | Surviving occurrences / 10, each run | Any eight-character fragment survives | Any full secret reversible |",
        "|---|---:|---:|---|---|",
    ]
    for name in successful:
        row = next(
            r for r in reports[name]["regressions"] if r["case"] == "full_ingest_log"
        )
        counts = ", ".join(
            str(sum(r["remaining_occurrences"].values())) for r in row["runs"]
        )
        fragments = any(
            any(r["remaining_eight_char_fragments"].values()) for r in row["runs"]
        )
        reversible = any(any(r["reversible_secrets"].values()) for r in row["runs"])
        lines.append(
            f"| `{name}` | {row['latency']['median_ms'] / 1000:.3f} s | "
            f"{counts} | {'yes' if fragments else 'no'} | {'yes' if reversible else 'no'} |"
        )
    lines += [
        "",
        (
            "The seven-line excerpt and connection-URI results, all individual timings "
            "and initialization costs are in the [numeric reports](results/kiji_privy/). "
            f"All {sum(len(r['structured_probes']) for r in reports.values())} completed "
            "structured probes matched their corresponding flat-text output "
            "for both multimodal text and parsed tool-call JSON."
        ),
        "",
        "## Privy recall by type",
        "",
        "| Type | Gold spans | `onnx` | `kiji-presidio` | `onnx-kiji` |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, counts in reports["onnx"]["corpora"]["privy"]["by_type"].items():
        values = [
            breakdown_cell(reports, s, "privy", "by_type", name)
            for s in ("onnx", "kiji-presidio", "onnx-kiji")
        ]
        lines.append(f"| {name} | {counts['spans']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Privy recall by format",
        "",
        (
            "SQL groups are inferred from quote style. The XML group keeps the source's "
            "bytes-string representation intact."
        ),
        "",
        "| Format | Documents | Gold spans | `onnx` | `kiji-presidio` | `onnx-kiji` |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, counts in reports["onnx"]["corpora"]["privy"]["breakdown"][
        "format"
    ].items():
        values = [
            breakdown_cell(reports, s, "privy", "breakdown", "format", name)
            for s in ("onnx", "kiji-presidio", "onnx-kiji")
        ]
        lines.append(
            f"| {name} | {counts['documents']} | {counts['counts']['gold_spans']} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "## AI4Privacy recall by language",
        "",
        (
            "Full-span recall, with 30 documents per language. Italian is outside the "
            "six languages listed by the Kiji ONNX card; it stays in the cross-check "
            "because it is part of PrivAiTe's existing corpus. These small samples "
            "do not establish general multilingual quality."
        ),
        "",
        "| Language | Gold spans | `onnx` | `kiji` | `kiji-presidio` | `onnx-kiji` |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for lang, counts in reports["onnx"]["corpora"]["ai4privacy"]["breakdown"][
        "lang"
    ].items():
        values = [
            breakdown_cell(reports, s, "ai4privacy", "breakdown", "lang", lang)
            for s in ("onnx", "kiji", "kiji-presidio", "onnx-kiji")
        ]
        lines.append(
            f"| {lang} | {counts['counts']['gold_spans']} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "## Provenance and limits",
        "",
        f"- PrivAiTe source: `{reports['onnx']['privaite_source']}` (unreleased detector fixes on 0.4.3).",
        f"- Benchmark source: `{reports['onnx']['benchmark_source']}`. Each result also stores hashes of its scoring code.",
        "- Hardware: Apple M1 Pro, 16 GiB RAM. Separate sequential processes; initialization excluded from per-request latency, included in peak RSS.",
        "- [Privy dataset and selection manifest](results/privy_manifest.json): fixed revision, fixed seed, 60 documents per format, no detector-dependent selection.",
        "- [Kiji adapter and artifact caveats](scripts/privy/README.md#candidate-definitions): published ONNX artifact only, 512/64 windows, supplied BIO labels, CPU, no remote code or coreference mapping.",
        "- Small denominators matter: the Privy sample contains only 15 password spans. These results do not establish broad secret-scanning guarantees.",
        "- We did not train models or tune thresholds on this evaluation sample; overlap with the models' original training data is not established. No claim of statistical significance is made from the single corpus pass and three long-log repetitions.",
        "- These are detection, redaction and local-latency measurements. They do not test network timeouts, concurrent production traffic or a live agent's answers.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    directory = RESULTS / "kiji_privy"
    reports = {
        name: json.loads((directory / f"{name}.json").read_text())
        for name in ORDER
        if (directory / f"{name}.json").exists()
    }
    retry_path = directory / "onnx-kiji-retry-status.json"
    failure_path = directory / "onnx-kiji-initial-failure.json"
    retry = json.loads(retry_path.read_text()) if retry_path.exists() else None
    failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
    path = BENCH / "KIJI_PRIVY.md"
    path.write_text(render(reports, retry, failure))
    print(f"Wrote {path.name}; all input/provenance/structured-probe guards passed.")


if __name__ == "__main__":
    main()
