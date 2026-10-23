"""Microsoft Presidio, vanilla: AnalyzerEngine with the shipped NER
configuration, one spaCy pipeline per language and every predefined recognizer
Presidio registers for it, default score threshold. This is the detector behind LiteLLM's and NeMo Guardrails'
Presidio integrations and most drop-in PII proxies. A library, not a request
integration, so it is scored on text only.
"""

from __future__ import annotations

from pbench.systems._common import SPACY_MODELS, presidio_nlp_engine, versions


class System:
    supports_payload = False

    def setup(self) -> None:
        from presidio_analyzer import AnalyzerEngine

        self.analyzer = AnalyzerEngine(nlp_engine=presidio_nlp_engine(), supported_languages=list(SPACY_MODELS))

    def detect(self, text: str, lang: str):
        return [(r.start, r.end, r.entity_type) for r in self.analyzer.analyze(text=text, language=lang)]

    def info(self) -> dict:
        return {
            "versions": versions("presidio-analyzer", "spacy", *SPACY_MODELS.values()),
            "recognizers": {lang: sorted({r.name for r in self.analyzer.registry.get_recognizers(lang, all_fields=True)})
                            for lang in SPACY_MODELS},
            "changed_from_default": {"languages": "document language, spaCy model per language "
                                                 "(conf/default.yaml NER block kept)"},
        }
