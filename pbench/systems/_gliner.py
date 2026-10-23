"""GLiNER-family zero-shot models, through the `gliner` library's documented
`predict_entities(text, labels, threshold)`. Both GLiNER rows get the same plain
English label list (one label per kind of identifier in the benchmark taxonomy),
so neither is favored by a label vocabulary it was trained on, and the card
default threshold 0.5."""

from __future__ import annotations

from pbench.systems._common import THREADS, dedupe, versions, windows

LABELS = [
    "person", "email", "phone number", "street address", "city", "postal code",
    "date of birth", "date", "social security number", "national id number",
    "passport number", "driver's license number", "tax id", "credit card number",
    "bank account number", "iban", "ip address", "username", "password", "api key",
    "medical record number", "customer id", "organization", "age", "url",
]
THRESHOLD = 0.5


class GlinerSystem:
    supports_payload = False
    model_id = ""
    revision = ""

    def setup(self) -> None:
        import torch
        from gliner import GLiNER

        torch.set_num_threads(THREADS)
        self.model = GLiNER.from_pretrained(self.model_id, revision=self.revision)
        self.model.eval()

    def detect(self, text: str, lang: str):
        spans = []
        for off, chunk in windows(text):
            for e in self.model.predict_entities(chunk, LABELS, threshold=THRESHOLD):
                spans.append((e["start"] + off, e["end"] + off, e["label"]))
        return dedupe(spans)

    def info(self) -> dict:
        return {"versions": versions("gliner", "torch", "transformers"),
                "model": f"{self.model_id}@{self.revision}", "labels": LABELS,
                "threshold": THRESHOLD, "window_chars": 1000}
