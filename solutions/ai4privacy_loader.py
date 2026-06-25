"""
Loader for the AI4Privacy pii-masking-200k corpus (en, fr, de, it).

The dataset is downloaded on demand from the Hugging Face Hub and cached locally.
We do NOT commit its raw text, because the dataset declares no explicit license;
this benchmark instead commits only its own derived ground-truth labels (keyed by
the dataset row id) and reconstructs the document text through this loader at run
time.

The AI4Privacy mask is fine grained and includes non-sensitive tags (JOBAREA,
JOBTITLE, SEX, ...). `SENSITIVE_LABELS` maps only the labels a privacy proxy is
actually expected to hide to this benchmark's taxonomy; the rest are ignored. The
mapped spans are used as a cross-check against the agent-produced ground truth,
not as the ground truth itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ai4privacy/pii-masking-200k"
FILES = {
    "en": "english_pii_43k.jsonl",
    "fr": "french_pii_62k.jsonl",
    "de": "german_pii_52k.jsonl",
    "it": "italian_pii_50k.jsonl",
}

SENSITIVE_LABELS = {
    "PREFIX": "PERSON",
    "FIRSTNAME": "PERSON",
    "MIDDLENAME": "PERSON",
    "LASTNAME": "PERSON",
    "FULLNAME": "PERSON",
    "USERNAME": "PERSON",
    "EMAIL": "EMAIL_ADDRESS",
    "PHONENUMBER": "PHONE_NUMBER",
    "PHONEIMEI": "PHONE_NUMBER",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "CREDITCARDCVV": "CREDIT_CARD",
    "IBAN": "IBAN_CODE",
    "ACCOUNTNUMBER": "FINANCIAL",
    "ACCOUNTNAME": "FINANCIAL",
    "BIC": "FINANCIAL",
    "MASKEDNUMBER": "FINANCIAL",
    "BITCOINADDRESS": "FINANCIAL",
    "ETHEREUMADDRESS": "FINANCIAL",
    "LITECOINADDRESS": "FINANCIAL",
    "IP": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "MAC": "IP_ADDRESS",
    "SSN": "US_SSN",
    "VEHICLEVIN": "US_SSN",
    "VEHICLEVRM": "US_SSN",
    "DATE": "DATE_TIME",
    "DOB": "DATE_TIME",
    "STREET": "LOCATION",
    "CITY": "LOCATION",
    "COUNTY": "LOCATION",
    "ZIPCODE": "LOCATION",
    "BUILDINGNUMBER": "LOCATION",
    "SECONDARYADDRESS": "LOCATION",
    "NEARBYGPSCOORDINATE": "LOCATION",
    "PASSWORD": "SECRET",
    "PIN": "SECRET",
    "URL": "URL",
}

MIN_LEN = 150
MAX_LEN = 1200
MIN_SENSITIVE_SPANS = 4


def _mapped_gold(mask: list[dict]) -> list[dict]:
    """Map AI4Privacy spans to our taxonomy, keeping only sensitive ones."""
    out = []
    for span in mask:
        mapped = SENSITIVE_LABELS.get(span.get("label", ""))
        value = span.get("value", "")
        if mapped and value:
            out.append({"value": value, "type": mapped})
    return out


def _is_good(text: str, gold: list[dict]) -> bool:
    return MIN_LEN <= len(text) <= MAX_LEN and len(gold) >= MIN_SENSITIVE_SPANS


def load_sample(n_per_lang: int = 30) -> list[dict]:
    """Return a deterministic, spread-out sample of PII-rich documents.

    Each item: {id, lang, text, gold} where gold is the mapped sensitive spans
    (used only as a cross-check against the agent ground truth).
    """
    sample: list[dict] = []
    for lang, filename in FILES.items():
        path = hf_hub_download(REPO_ID, filename, repo_type="dataset")
        candidates: list[dict] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = row.get("source_text", "")
                gold = _mapped_gold(row.get("privacy_mask", []))
                if _is_good(text, gold):
                    candidates.append(
                        {"id": str(row.get("id")), "lang": lang, "text": text, "gold": gold}
                    )
        candidates.sort(key=lambda r: int(r["id"]))
        # Evenly spaced pick across the whole file for template diversity.
        if candidates:
            step = max(1, len(candidates) // n_per_lang)
            picked = candidates[::step][:n_per_lang]
            sample.extend(picked)
    return sample


def load_text_by_id() -> dict[str, dict]:
    """Map id -> {lang, text, gold} for every candidate doc, for run-time lookup."""
    index: dict[str, dict] = {}
    for lang, filename in FILES.items():
        path = hf_hub_download(REPO_ID, filename, repo_type="dataset")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = row.get("source_text", "")
                gold = _mapped_gold(row.get("privacy_mask", []))
                if _is_good(text, gold):
                    index[str(row.get("id"))] = {"lang": lang, "text": text, "gold": gold}
    return index


if __name__ == "__main__":
    docs = load_sample()
    out = Path(__file__).parent / "_ai4privacy_sample.json"
    out.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    by_lang: dict[str, int] = {}
    spans = 0
    for doc in docs:
        by_lang[doc["lang"]] = by_lang.get(doc["lang"], 0) + 1
        spans += len(doc["gold"])
    print(f"sampled {len(docs)} docs ({by_lang}), {spans} mapped sensitive spans")
    print(f"written to {out}")
