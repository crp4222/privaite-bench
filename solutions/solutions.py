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


class LiteLLMPresidioSolution(Solution):
    """LiteLLM's built-in Presidio guardrail, reproduced faithfully in-process.

    LiteLLM's guardrail POSTs text to a Presidio Analyzer/Anonymizer HTTP service.
    We reproduce its detection with the presidio libraries directly (same result,
    no sidecar needed). Its request path scrubs message string content AND the
    `text` of multimodal content parts, but it never sends tool-call arguments to
    Presidio, so those leak. We give it the per-document language and the full
    default entity set (its best-case config, not the en-only default), so this is
    a fair representation of its detection, not a strawman. See
    litellm/proxy/guardrails/guardrail_hooks/presidio.py (input hook lines 736-786).
    """

    name = "litellm-presidio"

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
        # LiteLLM sends no entities filter and no score_threshold -> all default
        # recognizers, Presidio default threshold, default 'replace' operator.
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
            elif isinstance(content, list):
                # LiteLLM's guardrail DOES scrub the text of multimodal parts.
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        part["text"] = self._scrub(part["text"], lang)[0]
            # tool_calls[].function.arguments are NOT scrubbed on the input path,
            # so PII inside a tool call leaks -> the honest architectural gap.
        return out


class LLMGuardSolution(Solution):
    """LLM Guard (github.com/protectai/llm-guard) Anonymize input scanner.

    LLM Guard hard-conflicts with this env (it pins transformers 4.51 / needs
    torch), so it is run once in an isolated venv by scripts/build_llm_guard_cache.py
    and its per-document output is read from results/llm_guard_cache.json here.
    It is a flat-string scanner (its own DeBERTa AI4Privacy model + Presidio +
    regex): no awareness of message structure, multimodal parts, or tool calls, so
    the payload model scrubs string content only and everything structured leaks.
    English/Chinese only, so its non-English recall is honestly low (see the
    per-language table); its tool-call/multimodal leak is language-independent.
    """

    name = "llm-guard"
    CACHE = Path(__file__).resolve().parents[1] / "results" / "llm_guard_cache.json"

    def __init__(self) -> None:
        self._by_text: dict[str, dict] = {}
        # Real inference latency measured offline in the isolated venv; the bench
        # reads this cache in ~0ms, so this is the honest number for the latency
        # column (marked "offline").
        self.offline_latency_ms: float | None = None

    async def setup(self) -> None:
        import json

        data = json.loads(self.CACHE.read_text(encoding="utf-8"))
        meta = data.pop("__meta__", {})
        self.offline_latency_ms = meta.get("mean_scan_ms")
        # cache is keyed by document id; index by exact text for lookup here.
        self._by_text = {rec["text"]: rec for rec in data.values()}

    def _rec(self, text: str) -> dict:
        rec = self._by_text.get(text)
        if rec is None:
            # every corpus + clean text must be in the cache; fail loud rather
            # than silently under/over-counting llm-guard.
            raise KeyError("llm-guard cache miss: regenerate results/llm_guard_cache.json")
        return rec

    async def anonymize_text(self, text: str, lang: str) -> tuple[str, set[str]]:
        rec = self._rec(text)
        return rec["sanitized"], set(rec["detected"])

    async def anonymize_payload(self, messages: list[dict], lang: str) -> list[dict]:
        out = copy.deepcopy(messages)
        for msg in out:
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = self._rec(content)["sanitized"]
            # list (multimodal) content and tool_calls untouched: LLM Guard is a
            # flat-string scanner with no structural awareness -> structured leak.
        return out


def all_solutions() -> list[Solution]:
    sols: list[Solution] = [
        PrivAiTeSolution("onnx"),
        PrivAiTeSolution("light-all"),
        PrivAiTeSolution("light"),
        LiteLLMPresidioSolution(),
    ]
    # llm-guard only if its isolated-venv cache has been generated.
    if LLMGuardSolution.CACHE.exists():
        sols.append(LLMGuardSolution())
    return sols
