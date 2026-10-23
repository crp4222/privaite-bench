"""openai/privacy-filter on its own (the model behind PrivAiTe's default preset,
without Presidio, PrivAiTe's recognizers or merging), through PrivAiTe's ONNX
detector: the q4 export, the 1024-token windows and the BIOES decoding PrivAiTe
ships. It isolates how much of the onnx preset is the model."""

from __future__ import annotations

from pbench.systems._common import versions


class System:
    supports_payload = False

    async def setup(self) -> None:
        from privaite.config.schema import PIIConfig
        from privaite.pii.detector_onnx import OnnxPrivacyFilterDetector

        self.config = PIIConfig(preset="onnx").detectors.onnx
        self.config.device = "cpu"
        self.detector = OnnxPrivacyFilterDetector(self.config)
        await self.detector.initialize()

    async def detect(self, text: str, lang: str):
        return [(e.start, e.end, e.entity_type) for e in await self.detector.detect(text, lang)]

    def info(self) -> dict:
        return {"versions": versions("privaite", "onnxruntime", "transformers"),
                "model": f"{self.config.model_name}@{self.config.revision}",
                "variant": self.config.onnx_variant, "threshold": self.config.score_threshold}
