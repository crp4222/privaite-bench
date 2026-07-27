#!/usr/bin/env python3
"""
Big-session fixture builder for the agent workflow leak benchmark.

Produces `acme_support_big`: the standard `acme_support` fixture plus a large
ingest log (a single tool output far above 4k tokens, exercising multi-window
inference in the detector) and extra docs/tickets that force a 20-40 turn
agent session. This is the fixture behind `RESULTS_BIG.md`.

The planted ground truth is exactly the same 24 values as the small fixture:
static PII comes from `gen_fixture.STATIC_VALUES`, the 4 secrets are generated
into the fixture `.env` by `gen_fixture.generate` (fixed seed, never
committed). The log additionally repeats two of the secrets in key=value
log-line form (`presented_key=...`, `smtp_secret=...`); the onnx SECRET
detector catches those two on their own but misses them once roughly one
preceding line of log-shaped context sits in front of them (see
RESULTS_BIG.md), so keeping this shape in the fixture keeps that gap
measurable. Nothing sensitive is printed to stdout.

Usage:
    python3 agent_workflow/gen_big_fixture.py OUTDIR [--manifest PATH]

Then point the harness at it:
    AGENT_WORKFLOW_FIXTURE=OUTDIR \
    AGENT_WORKFLOW_OUT=results/agent_workflow_big \
    AGENT_WORKFLOW_RESULTS=results/agent_workflow_big/RESULTS.md \
    AGENT_WORKFLOW_TIMEOUT=1800 \
    python3 agent_workflow/run.py --presets onnx-auto,onnx-auto-cache
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_fixture  # noqa: E402

# Committed files of the base fixture (never copy a local .env or caches).
BASE_FILES = (
    ".gitignore",
    "README.md",
    "customers.json",
    "docs/team.md",
    "docs/escalations.md",
)

LOG_SEED = 20260717
LOG_ITERATIONS = 430

EVENTS = [
    "sync_ok", "sync_ok", "sync_ok", "sync_retry", "webhook_delivered",
    "webhook_timeout", "invoice_generated", "payment_captured", "kyc_check",
    "export_started", "export_finished", "escalation_opened", "escalation_closed",
]
NOTES = [
    "batch window checkpoint reached",
    "nightly billing export consumed the customer export",
    "queue depth nominal, no operator action required",
    "retry budget at 40 percent, monitoring",
    "schema validation passed for all rows",
    "dedupe pass removed 0 rows",
    "provider webhook acknowledged within SLA",
    "cold cache, latency above p95 target",
]


def build(outdir: Path, manifest_path: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for rel in BASE_FILES:
        dst = outdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(gen_fixture.FIXTURE / rel, dst)

    # Generate .env + ground-truth manifest directly into the big fixture.
    # Same fixed seed as the small fixture, so the planted secret values are
    # identical across both fixtures.
    original = gen_fixture.FIXTURE
    try:
        gen_fixture.FIXTURE = outdir
        gen_fixture.generate(manifest_path)
    finally:
        gen_fixture.FIXTURE = original

    static: dict[str, list[str]] = {}
    for needle, category, _label, _source in gen_fixture.STATIC_VALUES:
        static.setdefault(category, []).append(needle)

    env_vals = {}
    for line in (outdir / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, val = line.partition("=")
            env_vals[key] = val
    api_key = env_vals["ACME_API_KEY"]
    db_url = env_vals["DATABASE_URL"]
    smtp_password = env_vals["SMTP_PASSWORD"]

    rng = random.Random(LOG_SEED)
    people = static["person"]
    emails = static["email"]
    phones = static["phone"]
    address = static["address"][0]
    ibans = static["iban"]
    card = static["credit_card"][0]
    ssn = static["ssn"][0]

    lines = []
    ticket_ids = [f"T-{10000 + i}" for i in range(60)]
    minute = 0
    for i in range(LOG_ITERATIONS):
        minute += rng.randint(1, 3)
        ts = f"2026-07-11T{9 + minute // 60:02d}:{minute % 60:02d}:{rng.randint(0, 59):02d}Z"
        ticket = rng.choice(ticket_ids)
        event = rng.choice(EVENTS)
        note = rng.choice(NOTES)
        lines.append(
            f"{ts} INFO ingest ticket={ticket} customer=cus_{1000 + rng.randint(1, 60)} "
            f'event={event} latency_ms={rng.randint(40, 900)} note="{note}"'
        )
        # Sprinkle real planted values through the log so the big session
        # carries the same ground truth the leak scan checks for.
        if i % 23 == 4:
            who = rng.choice(people)
            mail = rng.choice(emails)
            lines.append(
                f"{ts} WARN ingest ticket={ticket} event=contact_update "
                f'agent="{who}" reply_to={mail} phone="{rng.choice(phones)}"'
            )
        if i % 61 == 17:
            lines.append(
                f"{ts} WARN billing ticket={ticket} event=mandate_review "
                f"iban={rng.choice(ibans)} card_on_file={card} ssn_on_file={ssn} "
                f'mailing_address="{address}"'
            )
        if i % 97 == 31:
            lines.append(
                f"{ts} ERROR auth ticket={ticket} event=key_rotation_failed "
                f"presented_key={api_key} smtp_secret={smtp_password}"
            )
        if i % 113 == 45:
            lines.append(f"{ts} ERROR db ticket={ticket} event=conn_fail dsn={db_url}")

    log_dir = outdir / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "ingest_batch.log").write_text("\n".join(lines) + "\n")

    (outdir / "docs" / "runbook.md").write_text(
        "# Ingest runbook\n\n"
        "When `webhook_timeout` exceeds 5 per hour, page the on-call listed in\n"
        "docs/team.md. Primary contact for billing mandates is "
        f"{people[3]} ({emails[3]}).\n\n"
        "Escalation path: open a ticket, attach the relevant slice of\n"
        "`logs/ingest_batch.log`, and notify the requester by phone "
        f"({phones[0]}).\n\n"
        "Never restart the exporter during the nightly billing window.\n"
    )
    (outdir / "docs" / "oncall.md").write_text(
        "# On-call rotation (July)\n\n"
        f"- Week 28: {people[4]} ({emails[4]})\n"
        f"- Week 29: {people[0]} ({emails[0]}, {phones[0]})\n"
        "- Week 30: TBD\n\n"
        "Handover notes live in tickets/.\n"
    )
    tickets = outdir / "tickets"
    tickets.mkdir(exist_ok=True)
    (tickets / "t-10412.md").write_text(
        "# T-10412: recurring webhook timeouts\n\n"
        f"Reporter: {people[1]} ({emails[1]})\n"
        "Status: pending_review\n\n"
        "Timeouts cluster around the nightly export. See logs/ingest_batch.log.\n"
    )
    (tickets / "t-10555.md").write_text(
        "# T-10555: mandate verification\n\n"
        f"Customer: {people[5]} ({emails[5]})\n"
        f"IBAN on file: {ibans[0]}\n"
        "Status: resolved\n\n"
        "Verified against the billing export, no action needed.\n"
    )
    (tickets / "t-10600.md").write_text(
        "# T-10600: address correction\n\n"
        f"Requested by: {people[2]} ({emails[2]}, {phones[1]})\n"
        f"New mailing address: {address}\n"
        "Status: pending_review\n"
    )

    files = [p for p in outdir.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    log_size = (log_dir / "ingest_batch.log").stat().st_size
    print(
        f"big fixture ready at {outdir}: {len(files)} files, {total} bytes total, "
        f"log {log_size} bytes, {len(lines)} log lines"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path, help="directory to build the big fixture in")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE.parent / "results" / "agent_workflow_big" / "manifest.json",
    )
    args = parser.parse_args()
    build(args.outdir, args.manifest)


if __name__ == "__main__":
    main()
