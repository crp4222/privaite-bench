"""Guardrails AI `GuardrailsPII` validator (Presidio + GLiNER small v2.1), run
locally (`use_local=True`) with both of its entity sets, "pii" and "spi", the
broadest it offers. `detect` is the validator's own `anonymize`, whose error
spans are what it replaces. The validator hard-codes English. Installed from its
Apache-2.0 source at a pinned commit (envs/guardrails.txt) because the Hub
install requires an account token; the code is the same.

Guardrails AI exports OpenTelemetry traces to its hosted endpoint by default
(`enable_metrics` is true unless ~/.guardrailsrc says otherwise); it is switched
off in-process here, the equivalent of `guardrails configure --disable-metrics`.
"""

from __future__ import annotations

from pbench.systems._common import THREADS, versions


class System:
    supports_payload = False

    def setup(self) -> None:
        import os

        os.environ.setdefault("OTEL_SDK_DISABLED", "true")
        import torch
        from guardrails.settings import settings

        settings.rc.enable_metrics = False
        from validator import GuardrailsPII

        torch.set_num_threads(THREADS)
        entities = GuardrailsPII.PII_ENTITIES_MAP["pii"] + GuardrailsPII.PII_ENTITIES_MAP["spi"]
        self.entities = entities
        self.validator = GuardrailsPII(entities=entities, on_fail="fix", use_local=True)

    def detect(self, text: str, lang: str):
        _, spans = self.validator.anonymize(text, self.entities)
        return [(s.start, s.end, s.reason) for s in spans]

    def info(self) -> dict:
        return {"versions": versions("guardrails-ai", "guardrails_pii", "gliner", "presidio-analyzer"),
                "model": self.validator.model_name, "entities": self.entities, "language": "en (hard-coded)",
                "changed_from_default": {"entities": "pii + spi"}}
