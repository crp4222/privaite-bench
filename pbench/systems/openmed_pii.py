"""OpenMed SDK `extract_pii` with OpenMed/privacy-filter-multilingual (a
fine-tune of openai/privacy-filter on the AI4Privacy series), the call its model
card documents, with the document language and `preserve_whitespace=True` so
offsets refer to the exact input. The SDK handles long inputs itself.
Trained on AI4Privacy data, so its OpenPII score is in-distribution."""

from __future__ import annotations

from pbench.systems._common import THREADS, versions

MODEL = "OpenMed/privacy-filter-multilingual"


class System:
    supports_payload = False

    def setup(self) -> None:
        import torch

        torch.set_num_threads(THREADS)
        from openmed import ModelLoader

        # The SDK reloads the pipeline on every call unless given a loader; a
        # shared loader is its documented way to reuse a warm model.
        self.loader = ModelLoader()

    def detect(self, text: str, lang: str):
        from openmed import extract_pii

        res = extract_pii(text, model_name=MODEL, lang=lang, preserve_whitespace=True,
                          loader=self.loader)
        return [(e.start, e.end, e.label) for e in res.entities]

    def info(self) -> dict:
        from huggingface_hub import model_info

        return {"versions": versions("openmed", "torch", "transformers"),
                "model": f"{MODEL}@{model_info(MODEL).sha}",
                "confidence_threshold": 0.5,
                "changed_from_default": {"lang": "document language", "preserve_whitespace": True}}
