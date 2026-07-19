# Acme Support Desk

Internal tooling for the Acme helpdesk team: ticket triage notes and the
customer export consumed by the nightly billing reconciliation job.

Maintainer: Rachel Donnelly <rachel.donnelly@acmesupport.io>, desk phone
+1-415-555-0134.

## Layout

- `docs/escalations.md`: open P1/P2 escalations and who owns them
- `docs/team.md`: on-call roster for the current rotation
- `customers.json`: sample rows from the customer export (billing reconciliation input)
- `.env`: local runtime configuration (gitignored, never commit)
