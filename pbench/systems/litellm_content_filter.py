"""LiteLLM's built-in content filter (`guardrail: litellm_content_filter`),
pre_call, default_on, through the real proxy, with every prebuilt identifier
pattern enabled with action MASK (PII, payment card, credential, network and the
regional ID/phone/postal sets). The keyword categories for protected classes and
dangerous content are left out: they are content policies, not personal data."""

from __future__ import annotations

import json
from pathlib import Path

from pbench.systems._litellm_proxy import LiteLLMProxySystem

SKIP = ("Protected Class", "Dangerous Content")


def _patterns() -> list[str]:
    import litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter as cf

    data = json.loads((Path(cf.__file__).parent / "patterns.json").read_text())
    return [p["name"] for p in data["patterns"] if not p.get("category", "").startswith(SKIP)]


class System(LiteLLMProxySystem):
    def setup(self) -> None:
        self.guardrail = {"guardrail_name": "content-filter-pii", "litellm_params": {
            "guardrail": "litellm_content_filter", "mode": "pre_call", "default_on": True,
            "patterns": [{"pattern_type": "prebuilt", "pattern_name": n, "action": "MASK"}
                         for n in _patterns()]}}
        super().setup()
