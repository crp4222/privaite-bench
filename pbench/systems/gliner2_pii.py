"""fastino/gliner2-privacy-filter-PII-multi through the `gliner2` library's
`extract_entities`, with the 42 labels its model card lists and the card's
threshold 0.5."""

from __future__ import annotations

from pbench.systems._common import THREADS, dedupe, versions, windows

MODEL = "fastino/gliner2-privacy-filter-PII-multi"
REVISION = "1cb4166094dc58fa8d836429f060d6c95f62b495"
THRESHOLD = 0.5
LABELS = [
    "person", "full_name", "first_name", "middle_name", "last_name", "date_of_birth",
    "email", "phone_number", "address", "street_address", "city", "state_or_region",
    "postal_code", "country", "government_id", "national_id_number", "passport_number",
    "drivers_license_number", "license_number", "tax_id", "tax_number", "bank_account",
    "account_number", "routing_number", "iban", "payment_card", "card_number",
    "card_expiry", "card_cvv", "username", "ip_address", "account_id",
    "sensitive_account_id", "password", "secret", "api_key", "access_token",
    "recovery_code", "sensitive_date", "document_date", "expiration_date",
    "transaction_date",
]
assert len(LABELS) == 42


class System:
    supports_payload = False

    def setup(self) -> None:
        import torch
        from gliner2 import GLiNER2

        torch.set_num_threads(THREADS)
        from huggingface_hub import snapshot_download

        path = snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors"])
        self.model = GLiNER2.from_pretrained(path, map_location="cpu")
        self.model.eval()

    def detect(self, text: str, lang: str):
        import torch

        spans = []
        for off, chunk in windows(text):
            with torch.inference_mode():
                res = self.model.extract_entities(chunk, LABELS, threshold=THRESHOLD,
                                                  include_confidence=True, include_spans=True)
            for label, items in (res.get("entities") or {}).items():
                for it in items or []:
                    spans.append((int(it["start"]) + off, int(it["end"]) + off, label))
        return dedupe(spans)

    def info(self) -> dict:
        return {"versions": versions("gliner2", "torch", "transformers"),
                "model": f"{MODEL}@{REVISION}", "labels": LABELS, "threshold": THRESHOLD,
                "window_chars": 1000}
