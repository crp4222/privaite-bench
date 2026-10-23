"""LLM Guard (protectai/llm-guard) Anonymize input scanner, defaults.

`detect` runs the exact steps of `Anonymize.scan` up to the replacement
(analyze, conflict removal, whitespace merge) and returns the spans that `scan`
replaces. LLM Guard scans one prompt string and has no notion of messages, tool
calls or tool results, so it has no request-level integration to score.
Its language parameter accepts only "en" and "zh", so it runs in English on every
document; its DeBERTa model is multilingual and its regexes language-agnostic.
"""

from __future__ import annotations

from pbench.systems._common import THREADS, versions


class System:
    supports_payload = False

    def setup(self) -> None:
        import torch
        from llm_guard.input_scanners import Anonymize
        from llm_guard.vault import Vault

        torch.set_num_threads(THREADS)
        self.scanner = Anonymize(Vault())

    def detect(self, text: str, lang: str):
        s = self.scanner
        if not text.strip():
            return []
        cleaned = s.remove_single_quotes(text)
        assert len(cleaned) == len(text)
        results = s._analyzer.analyze(text=cleaned, language=s._language, entities=s._entity_types,
                                      allow_list=s._allowed_names, score_threshold=s._threshold)
        results = s._remove_conflicts_and_get_text_manipulation_data(results)
        merged = s._merge_entities_with_whitespace_between(text, results)
        return [(r.start, r.end, r.entity_type) for r in merged]

    def info(self) -> dict:
        s = self.scanner
        return {
            "versions": versions("llm-guard", "presidio-analyzer", "transformers", "torch"),
            "language": s._language,
            "entity_types": s._entity_types,
            "threshold": s._threshold,
            "recognizer_conf": "default (DeBERTa AI4Privacy v2)",
            "changed_from_default": {},
        }
