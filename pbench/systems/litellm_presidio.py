"""LiteLLM's built-in Presidio guardrail (`guardrail: presidio`), pre_call,
default_on, through the real proxy. Presidio runs as an HTTP service, as LiteLLM
requires. The guardrail's language is set per request to the document language
through LiteLLM's documented per-request `guardrail_config` (its default is "en")."""

from __future__ import annotations

from typing import ClassVar

from pbench.systems._litellm_proxy import LiteLLMProxySystem


class System(LiteLLMProxySystem):
    needs_presidio = True
    guardrail: ClassVar[dict] = {"guardrail_name": "presidio-pii", "litellm_params": {
        "guardrail": "presidio", "mode": "pre_call", "default_on": True}}

    def request_extra(self, lang: str) -> dict:
        return {"metadata": {"guardrail_config": {"language": lang}}}
