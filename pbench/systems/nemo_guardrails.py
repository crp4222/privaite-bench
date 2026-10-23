"""NVIDIA NeMo Guardrails, sensitive data detection rails, through the real
`LLMRails` pipeline in passthrough mode, in front of the same capturing fake
provider as the LiteLLM rows.

Rails enabled: "mask sensitive data on input" and "on output", with every entity
Presidio supports in English (NeMo builds its analyzer on en_core_web_lg only).
The library ships no sensitive-data flow for tool inputs or tool outputs, so none
is configured: writing one would be our code, not the product's.

On multimodal content its masking action receives the list of parts, spaCy
raises, and NeMo answers the client with its internal-error reply without
calling the model: recorded as failed closed (nothing leaked, nothing served).
"""

from __future__ import annotations

from pbench.systems._capture import Capture
from pbench.systems._common import versions

INTERNAL_ERROR = "I'm sorry, an internal error has occurred."  # NeMo's reply when an action raises


class System:
    supports_payload = True
    supports_text = False

    def setup(self) -> None:
        from nemoguardrails import LLMRails, RailsConfig
        from presidio_analyzer import AnalyzerEngine

        self.capture = Capture()
        self.entities = sorted(AnalyzerEngine().get_supported_entities("en"))
        ents = "[" + ", ".join(self.entities) + "]"
        yaml = f"""
models:
  - type: main
    engine: openai
    model: mock
    parameters:
      base_url: http://127.0.0.1:{self.capture.port}/v1
      api_key: unused
passthrough: true
rails:
  config:
    sensitive_data_detection:
      input:
        entities: {ents}
      output:
        entities: {ents}
  input:
    flows:
      - mask sensitive data on input
  output:
    flows:
      - mask sensitive data on output
"""
        self.rails = LLMRails(RailsConfig.from_content(yaml_content=yaml))

    def detect(self, text: str, lang: str):
        raise NotImplementedError("payload-only system")

    async def payload(self, messages: list, lang: str):
        self.capture.last = None
        reply = await self.rails.generate_async(messages=messages)
        if self.capture.last is None:
            if isinstance(reply, dict) and reply.get("content") == INTERNAL_ERROR:
                return {"failed_closed": INTERNAL_ERROR}
            raise RuntimeError(f"NeMo answered without calling the provider: {str(reply)[:200]}")
        return self.capture.last["messages"]

    def info(self) -> dict:
        return {"versions": versions("nemoguardrails", "presidio-analyzer", "langchain-openai"),
                "rails": ["mask sensitive data on input", "mask sensitive data on output"],
                "entities": self.entities, "passthrough": True}
