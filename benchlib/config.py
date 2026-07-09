"""One PIIConfig factory for the six near-identical builders across the runners.

Every knob that varied between callers (languages, entities, faker_locale, preset,
onnx on/off, fuzzy_matching) is an explicit argument with no lossy default, because
faker_locale and fuzzy_matching feed the anonymizer/deanonymizer: a wrong default
would silently change anonymized output. Each call site passes its exact values, so
the produced config is identical to the inline one it replaces.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchlib.paths import ensure_privaite_path

ensure_privaite_path()

from privaite.config.schema import (  # noqa: E402
    AnonymizationConfig,
    DeanonymizationConfig,
    DetectorsConfig,
    PIIConfig,
    PresidioDetectorConfig,
)

# The 9-entity Presidio allowlist (the old shipped example config). Named once here;
# previously copied verbatim into five files.
LIGHT_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE",
    "IP_ADDRESS", "DATE_TIME", "US_SSN", "UK_NHS",
]


def langs_for(lang: str) -> list[str]:
    """The Presidio language list for a doc: the doc language, plus English as a
    fallback for anything but English."""
    return [lang] if lang == "en" else [lang, "en"]


def pii_config(
    *,
    languages: Sequence[str],
    entities: Sequence[str] | None,
    faker_locale: Sequence[str],
    preset: str | None = None,
    presidio_enabled: bool = True,
    onnx_enabled: bool = False,
    fuzzy_matching: bool | None = None,
    score_threshold: float = 0.4,
    method: str = "placeholder",
    deanon_enabled: bool = True,
) -> PIIConfig:
    """Build a PIIConfig the way the runners do: Presidio (optionally pinned to an
    entity allowlist) plus an anonymizer, optionally with a preset or the ONNX
    detector turned on. `entities=None` means Presidio's full recognizer set."""
    presidio = PresidioDetectorConfig(
        enabled=presidio_enabled,
        languages=list(languages),
        score_threshold=score_threshold,
        entities=list(entities) if entities is not None else None,
    )
    deanon_kwargs: dict = {"enabled": deanon_enabled}
    if fuzzy_matching is not None:
        deanon_kwargs["fuzzy_matching"] = fuzzy_matching
    config = PIIConfig(
        enabled=True,
        preset=preset,
        detectors=DetectorsConfig(presidio=presidio),
        anonymization=AnonymizationConfig(method=method, faker_locale=list(faker_locale)),
        deanonymization=DeanonymizationConfig(**deanon_kwargs),
    )
    if onnx_enabled:
        config.detectors.onnx.enabled = True
    return config
