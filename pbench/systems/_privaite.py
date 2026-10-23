"""PrivAiTe, as released on PyPI, through its own engine.

`detect` is `PIIEngine.inspect_text`, the merged entity list the request path
replaces. `payload` is `PIIEngine.process_request`, the function the proxy and the
gateway call on every request (message text, multimodal parts, tool-call
arguments, tool results). Configuration is the shipped preset with two changes
applied to every system in this benchmark: the document's language is passed in
(as the only Presidio language, exactly what the Presidio row gets; PrivAiTe runs
every configured language on every request, so adding English would give it a
second NER pass the others do not get), and inference runs on CPU.
"""

from __future__ import annotations

from pbench.systems._common import THREADS, versions


class PrivAiTeSystem:
    preset = "onnx"
    supports_payload = True

    def __init__(self) -> None:
        self.engines = {}

    async def setup(self) -> None:
        import logging

        logging.disable(logging.WARNING)

    async def _engine(self, lang: str):
        if lang not in self.engines:
            from privaite.config.schema import PIIConfig
            from privaite.pii.engine import PIIEngine

            config = PIIConfig(preset=self.preset)
            config.detectors.presidio.languages = [lang]
            config.detectors.onnx.device = "cpu"
            config.detectors.gliner.device = "cpu"
            engine = PIIEngine(config)
            await engine.initialize()
            self.engines[lang] = engine
        return self.engines[lang]

    async def detect(self, text: str, lang: str):
        engine = await self._engine(lang)
        return [(e.start, e.end, e.entity_type) for e in await engine.inspect_text(text)]

    async def payload(self, messages: list, lang: str):
        engine = await self._engine(lang)
        out, _ = await engine.process_request(messages)
        return out

    def info(self) -> dict:
        from privaite.config.schema import PIIConfig

        config = PIIConfig(preset=self.preset)
        return {
            "versions": versions("privaite", "presidio-analyzer", "onnxruntime", "spacy",
                                 "transformers", "gliner", "torch"),
            "preset": self.preset,
            "onnx_model": f"{config.detectors.onnx.model_name}@{config.detectors.onnx.revision}"
            if config.detectors.onnx.enabled else None,
            "onnx_variant": config.detectors.onnx.onnx_variant,
            "changed_from_default": {"presidio.languages": "document language only",
                                     "onnx.device": "cpu", "gliner.device": "cpu"},
            "threads": THREADS,
        }

    async def teardown(self) -> None:
        for engine in self.engines.values():
            await engine.shutdown()
