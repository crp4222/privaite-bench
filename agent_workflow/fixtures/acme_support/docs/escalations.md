# Open escalations

## ACME-4821 (P1): refund not processed after chargeback reversal

Customer: Tomas Alvarez <tomas.alvarez84@gmail.com>, phone +1-415-555-0188.
Refund of 249.00 USD stuck in `pending_review` since the chargeback was
reversed. Billing says the reconciliation job skipped his row because the
IBAN on file failed checksum validation on the previous export version.

Replacement card reader ships to his office:
1742 Willow Crescent, Suite 210, Oakland, CA 94607.

Owner: Priya Natarajan. Next update due at the Thursday sync.

## ACME-4835 (P2): duplicate welcome emails

Customer: Ingrid Bergstrom <ingrid.bergstrom@nordmail.se> reports receiving
the onboarding sequence twice. Suspected double-enqueue in the SMTP worker
when `SMTP_PASSWORD` rotation restarts the queue consumer mid-batch.

Owner: Marcus Weaver. Reproduced in staging, fix in review.
