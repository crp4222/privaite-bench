"""Every system under test: where it runs and how to describe it.

`env` is the virtualenv under .venvs/ built from envs/<env>.txt. `kind` says
what the system is: a request-level integration (a proxy, gateway, guardrail or
agent middleware that sees whole requests), a text library, or a standalone
model. `text` / `payload` say which suites it is scored on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class System:
    name: str
    title: str
    env: str
    kind: str          # "integration" | "library" | "model"
    text: bool
    payload: bool
    url: str
    note: str = ""


SYSTEMS = [
    System("privaite_onnx", "PrivAiTe 0.7.2 (onnx, default)", "privaite", "integration", True, True,
           "https://github.com/crp4222/PrivAiTe"),
    System("privaite_light", "PrivAiTe 0.7.2 (light)", "privaite", "integration", True, True,
           "https://github.com/crp4222/PrivAiTe"),
    System("privaite_max", "PrivAiTe 0.7.2 (max)", "privaite", "integration", True, True,
           "https://github.com/crp4222/PrivAiTe"),
    System("litellm_presidio", "LiteLLM 1.104 Presidio guardrail", "litellm", "integration", False, True,
           "https://docs.litellm.ai/docs/proxy/guardrails/pii_masking_v2",
           "detection is Presidio's (see the Presidio row)"),
    System("litellm_content_filter", "LiteLLM 1.104 content filter", "litellm", "integration", False, True,
           "https://docs.litellm.ai/docs/proxy/guardrails/litellm_content_filter"),
    System("nemo_guardrails", "NeMo Guardrails 0.24 (sensitive data)", "nemo", "integration", False, True,
           "https://github.com/NVIDIA/NeMo-Guardrails",
           "detection is Presidio's, English only"),
    System("kiji_proxy", "Kiji Privacy Proxy 1.6.4 (Dataiku)", "harness", "integration", True, True,
           "https://github.com/dataiku/kiji-proxy"),
    System("langchain_pii", "LangChain 1.4 PIIMiddleware", "langchain", "integration", True, True,
           "https://docs.langchain.com/oss/python/langchain/middleware"),
    System("datafog_pii", "DataFog 4.9 (+ its LiteLLM guardrail)", "datafog", "integration", True, True,
           "https://github.com/DataFog/datafog-python"),
    System("agentgateway", "agentgateway 1.6.0 (prompt guard)", "harness", "integration", False, True,
           "https://github.com/agentgateway/agentgateway", "regex guards; no detection API"),
    System("occludra", "occludra/AISG gateway (e948e23)", "occludra", "integration", True, True,
           "https://github.com/occludra/gateway"),
    System("presidio", "Microsoft Presidio 2.2.364", "presidio", "library", True, False,
           "https://github.com/microsoft/presidio"),
    System("guardrails_pii", "Guardrails AI GuardrailsPII 1.0.8", "guardrails", "library", True, False,
           "https://github.com/guardrails-ai/guardrails_pii"),
    System("llm_guard", "LLM Guard 0.3.16 Anonymize", "llmguard", "library", True, False,
           "https://github.com/protectai/llm-guard"),
    System("scrubadub_pii", "scrubadub 2.0.1", "scrubadub", "library", True, False,
           "https://github.com/LeapBeyond/scrubadub"),
    System("openmed_pii", "OpenMed 3.0 (privacy-filter-multilingual)", "openmed", "library", True, False,
           "https://huggingface.co/OpenMed/privacy-filter-multilingual"),
    System("privacy_filter", "openai/privacy-filter (model alone)", "privaite", "model", True, False,
           "https://huggingface.co/openai/privacy-filter"),
    System("gliner_pii", "GLiNER multi PII v1", "models", "model", True, False,
           "https://huggingface.co/urchade/gliner_multi_pii-v1"),
    System("nvidia_gliner_pii", "NVIDIA GLiNER-PII", "models", "model", True, False,
           "https://huggingface.co/nvidia/gliner-PII"),
    System("gliner2_pii", "GLiNER2 PII (fastino)", "models", "model", True, False,
           "https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi"),
    System("piiranha", "Piiranha v1", "models", "model", True, False,
           "https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information"),
    System("kiji", "Kiji PII (Dataiku, ONNX)", "privaite", "model", True, False,
           "https://huggingface.co/DataikuNLP/kiji-pii-model-onnx"),
]
BY_NAME = {s.name: s for s in SYSTEMS}

# Which training data each learned component saw, from its model card. A system
# scored on a corpus from the same generator family has a home-field advantage
# there; the report marks those cells instead of hiding them.
CONTAMINATION = {
    "llm_guard": {"openpii"},          # Isotonic DeBERTa fine-tuned on ai4privacy pii-masking-200k
    "piiranha": {"openpii"},           # ai4privacy pii-masking-400k
    "openmed_pii": {"openpii"},        # ai4privacy 200k, 400k, open-pii-masking-500k
    "nvidia_gliner_pii": {"nemotron"},  # trained on nvidia/Nemotron-PII
}

# Tested on purpose and left out, with the reason, so absence is never silent.
NOT_TESTED = [
    ("AWS Comprehend, Google Cloud DLP, Azure AI Language PII, Private AI, Tonic Textual",
     "commercial hosted or licensed services: no free self-hosted build to run"),
    ("Lakera Guard, Pangea, Protecto, Skyflow", "hosted services only"),
    ("Portkey AI gateway (open source)",
     ("its PII guardrails call hosted partners or Portkey's hosted guardrail; the open-source "
      "gateway alone only offers regexes the user writes")),
    ("Bifrost, Kong AI Gateway", "their PII redaction is an enterprise feature"),
    ("Presidio's LLM recognizer (LangExtract)",
     ("it needs a language model of its own to detect; the only one in this setup is the provider "
      "the redaction layer protects data from")),
    ("Guardrails AI DetectPII", ("a thin Presidio wrapper; GuardrailsPII, its Presidio + GLiNER "
                                 "validator, is measured")),
    ("philter-ai-proxy, zeroleak-llm-proxy, redaction-proxy, LLM-Shield-Proxy, llm-redact-proxy",
     "below the inclusion bar (fewer than 25 GitHub stars when the benchmark was built)"),
]
