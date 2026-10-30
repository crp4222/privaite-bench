"""crp4222/privaite-pii-minilm alone (PrivAiTe's model, multilingual MiniLM, 15
types), through the detector PrivAiTe 0.8.0 ships (`MiniLMDetector`): ONNX fp32
on CPU, windows of 256 tokens with 64 of overlap, spans extended to whole words.
No Presidio, no recognizers. Trained on the train splits of Nemotron-PII,
OpenPII, the two Gretel sets and MAPA (see the contamination table)."""

from __future__ import annotations

from pbench.systems._common import THREADS, versions


class System:
    supports_payload = False

    async def setup(self) -> None:
        import logging

        from privaite.config.schema import MiniLMDetectorConfig
        from privaite.pii.detector_minilm import MiniLMDetector

        logging.disable(logging.WARNING)
        self.config = MiniLMDetectorConfig(enabled=True, backend="onnx")
        self.detector = MiniLMDetector(self.config)
        await self.detector.initialize()

    async def detect(self, text: str, lang: str):
        return [(e.start, e.end, e.entity_type) for e in await self.detector.detect(text, lang)]

    def info(self) -> dict:
        return {
            "versions": versions("privaite", "onnxruntime", "transformers"),
            "model": f"{self.config.model_name}@{self.config.revision}",
            "backend": self.config.backend,
            "threads": THREADS,
        }

    async def teardown(self) -> None:
        await self.detector.shutdown()
