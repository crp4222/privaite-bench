"""iiiorg/piiranha-v1 (mDeBERTa fine-tuned on ai4privacy pii-masking-400k),
through the Transformers token-classification pipeline with
aggregation_strategy="simple". The card documents no API beyond a quickstart
notebook that takes the argmax label of every token and merges adjacent PII
tokens; the pipeline gives the same token labels and merges the same way. The
card states a 256-token context ("if your text is longer than that, just split
it up"), so every window is also cut into 256-token chunks (32 overlap) by the
pipeline's own stride chunking: a 1,000-character window often runs past 256
tokens (three windows in four on MAPA). Trained on the AI4Privacy family, so its
OpenPII score is in-distribution (see the contamination table)."""

from __future__ import annotations

from pbench.systems._common import THREADS, dedupe, versions, windows

MODEL = "iiiorg/piiranha-v1-detect-personal-information"
REVISION = "255acde67a2f34cf452eb42e365b24d2957352fc"
CONTEXT_TOKENS = 256  # from the model card
STRIDE_TOKENS = 32


class System:
    supports_payload = False

    def setup(self) -> None:
        import torch
        from transformers import AutoTokenizer, pipeline

        torch.set_num_threads(THREADS)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, model_max_length=CONTEXT_TOKENS)
        self.pipe = pipeline("token-classification", model=MODEL, revision=REVISION, tokenizer=tokenizer,
                             aggregation_strategy="simple", stride=STRIDE_TOKENS, device="cpu")

    def detect(self, text: str, lang: str):
        spans = []
        for off, chunk in windows(text):
            for e in self.pipe(chunk):
                if e["entity_group"] != "O":
                    spans.append((int(e["start"]) + off, int(e["end"]) + off, e["entity_group"]))
        return dedupe(spans)

    def info(self) -> dict:
        return {"versions": versions("transformers", "torch"), "model": f"{MODEL}@{REVISION}",
                "aggregation": "simple", "window_chars": 1000,
                "context_tokens": CONTEXT_TOKENS, "stride_tokens": STRIDE_TOKENS}
