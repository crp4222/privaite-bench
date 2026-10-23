"""DataFog: its regex engine for text, and its own LiteLLM guardrail
(`datafog.integrations.litellm_guardrail.DataFogGuardrail`) for requests.

The guardrail's default type list is EMAIL, PHONE, CREDIT_CARD, SSN; here it is
widened to every type the regex engine supports, and `fail_policy` is "closed",
the broadest protection it offers. `detect` uses the same engine and types.
"""

from __future__ import annotations

from pbench.systems._common import versions


class System:
    supports_payload = True

    def setup(self) -> None:
        import datafog
        from datafog.integrations.litellm_guardrail import DataFogGuardrail

        self.types = list(datafog.get_supported_entities())
        self.guard = DataFogGuardrail(action="redact", entity_types=self.types,
                                      fail_policy="closed", guardrail_name="datafog-pii",
                                      event_hook="pre_call")

    def detect(self, text: str, lang: str):
        import datafog

        res = datafog.scan(text, engine="regex", entity_types=self.types)
        return [(e.start, e.end, e.type) for e in res.entities]

    async def payload(self, messages: list, lang: str):
        data = await self.guard.async_pre_call_hook(None, None, {"messages": messages}, "completion")
        return data["messages"]

    def info(self) -> dict:
        return {
            "versions": versions("datafog", "litellm"),
            "engine": "regex",
            "entity_types": self.types,
            "changed_from_default": {"entity_types": "all regex types (default: EMAIL, PHONE, CREDIT_CARD, SSN)",
                                     "fail_policy": "closed"},
        }
