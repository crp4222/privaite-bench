"""
Solution adapters for the comparative benchmark.

Every solution exposes the same interface so the runner can score them apples to
apples:

    await sol.setup()
    anon_text, anonymized = await sol.anonymize_text(text, lang)
    anon_messages          = await sol.anonymize_payload(messages, lang)
    await sol.teardown()

`anonymize_text` returns the scrubbed text plus the set of original substrings the
solution chose to anonymize (used to count false positives on clean text).
`anonymize_payload` takes an OpenAI-style messages list so we can measure leakage
inside tool-call arguments and multimodal parts, not just message text.

Adding another solution (another repo) is just another subclass here.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "PrivAiTe"))

from privaite.config.schema import (  # noqa: E402
    AnonymizationConfig,
    DeanonymizationConfig,
    DetectorsConfig,
    PIIConfig,
    PresidioDetectorConfig,
)
from privaite.pii.engine import PIIEngine  # noqa: E402

LIGHT_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE",
    "IP_ADDRESS", "DATE_TIME", "US_SSN", "UK_NHS",
]


def _langs(lang: str) -> list[str]:
    return [lang] if lang == "en" else [lang, "en"]


class Solution:
    name = "base"

    async def setup(self) -> None: ...
    async def teardown(self) -> None: ...

    async def anonymize_text(self, text: str, lang: str) -> tuple[str, set[str]]:
        raise NotImplementedError

    async def anonymize_payload(self, messages: list[dict], lang: str) -> list[dict]:
        raise NotImplementedError


class PrivAiTeSolution(Solution):
    """PrivAiTe engine. Handles message text, tool-call arguments and multimodal."""

    def __init__(self, preset: str) -> None:
        self.name = f"privaite-{preset}"
        self._preset = preset
        self._engines: dict[str, PIIEngine] = {}

    def _config(self, lang: str) -> PIIConfig:
        # "light"     = Presidio restricted to a 9-entity allowlist (the old shipped
        #               example config: low recall).
        # "light-all" = the PRODUCT's actual preset:light (full Presidio, no pin).
        # "onnx"      = full ONNX suite (default).
        entities = list(LIGHT_ENTITIES) if self._preset == "light" else None
        presidio = PresidioDetectorConfig(
            enabled=True, languages=_langs(lang), score_threshold=0.4,
            entities=entities,
        )
        return PIIConfig(
            enabled=True,
            preset="onnx" if self._preset == "onnx" else None,
            detectors=DetectorsConfig(presidio=presidio),
            anonymization=AnonymizationConfig(method="placeholder", faker_locale=["en_US"]),
            deanonymization=DeanonymizationConfig(enabled=True),
        )

    async def _engine(self, lang: str) -> PIIEngine:
        if lang not in self._engines:
            eng = PIIEngine(self._config(lang))
            await eng.initialize()
            self._engines[lang] = eng
        return self._engines[lang]

    async def anonymize_text(self, text: str, lang: str) -> tuple[str, set[str]]:
        eng = await self._engine(lang)
        anon, mapping = await eng.process_request([{"role": "user", "content": text}])
        return anon[0]["content"], set(mapping.get_all_fakes().values())

    async def anonymize_payload(self, messages: list[dict], lang: str) -> list[dict]:
        eng = await self._engine(lang)
        anon, _ = await eng.process_request(copy.deepcopy(messages))
        return anon

    async def teardown(self) -> None:
        for eng in self._engines.values():
            await eng.shutdown()


class PresidioBaselineSolution(Solution):
    """Vanilla Microsoft Presidio on flat message text.

    This is the engine behind LiteLLM's Presidio guardrail and most drop-in PII
    proxies. It scrubs the string content of a message; it does not look inside
    tool-call arguments or multimodal content (the documented gap).
    """

    name = "presidio-baseline"

    def __init__(self) -> None:
        self._analyzer = None
        self._anonymizer = None

    async def setup(self) -> None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        models = [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "fr", "model_name": "fr_core_news_md"},
            {"lang_code": "de", "model_name": "de_core_news_md"},
            {"lang_code": "it", "model_name": "it_core_news_md"},
            {"lang_code": "es", "model_name": "es_core_news_md"},
        ]
        nlp_engine = NlpEngineProvider(
            nlp_configuration={"nlp_engine_name": "spacy", "models": models}
        ).create_engine()
        self._analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=[m["lang_code"] for m in models],
        )
        self._anonymizer = AnonymizerEngine()

    def _scrub(self, text: str, lang: str) -> tuple[str, set[str]]:
        results = self._analyzer.analyze(text=text, language=lang)
        originals = {text[r.start:r.end] for r in results}
        anon = self._anonymizer.anonymize(text=text, analyzer_results=results).text
        return anon, originals

    async def anonymize_text(self, text: str, lang: str) -> tuple[str, set[str]]:
        return self._scrub(text, lang)

    async def anonymize_payload(self, messages: list[dict], lang: str) -> list[dict]:
        out = copy.deepcopy(messages)
        for msg in out:
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = self._scrub(content, lang)[0]
            # list (multimodal) content and tool_calls are left untouched, which
            # is exactly how a flat-text scrubber behaves: the PII inside them leaks.
        return out


def all_solutions() -> list[Solution]:
    return [
        PrivAiTeSolution("light"),
        PrivAiTeSolution("light-all"),
        PrivAiTeSolution("onnx"),
        PresidioBaselineSolution(),
    ]
