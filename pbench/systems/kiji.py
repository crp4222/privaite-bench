"""DataikuNLP/kiji-pii-model-onnx, the published ONNX artifact (pinned revision
and sha256). The publisher ships no Python inference code (its proxy is in Go),
so the token outputs are decoded with PrivAiTe's BIO/BIOES decoder, 512-token
windows with 64 overlap, threshold 0.5; see _kiji_detector.py."""

from __future__ import annotations

from pbench.systems._common import versions
from pbench.systems._kiji_detector import MODEL, REVISION, KijiDetector


class System:
    supports_payload = False

    async def setup(self) -> None:
        self.detector = KijiDetector(offline=False)
        await self.detector.initialize()

    async def detect(self, text: str, lang: str):
        return [(e.start, e.end, e.entity_type) for e in await self.detector.detect(text, lang)]

    def info(self) -> dict:
        return {"versions": versions("onnxruntime", "transformers", "privaite"),
                "model": f"{MODEL}@{REVISION}", "threshold": self.detector.threshold,
                "decoder": "PrivAiTe decode_bioes_spans + refine_boundary"}
