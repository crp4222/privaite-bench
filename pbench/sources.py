"""The public corpora: download (pinned revision + sha256), seeded selection, and
the mapping of each dataset's labels onto `pbench.corpus` kinds and types.

Raw files land in ~/.cache/privaite-bench/raw and the selected documents in
~/.cache/privaite-bench/corpora; neither is committed (several licenses require
attribution rather than forbid redistribution, but keeping third-party text out
of git keeps the repo small and the provenance unambiguous). What is committed is
results/corpora.json: per document its id, sha256 and span counts, which pins the
exact selection without carrying any text.

Run `python -m pbench.sources` (harness env) to download and select everything.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from pbench.corpus import Doc, Span, check_spans, write_jsonl

CACHE = Path.home() / ".cache" / "privaite-bench"
RAW = CACHE / "raw"
CORPORA = CACHE / "corpora"
SEED = 20261023
LANGS = ("en", "fr", "de", "es", "it", "pt", "nl")

HF = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"

# (repo, revision, path, sha256, local name). Revisions are the dataset commits the
# committed results were produced from; a mismatching file aborts the build.
FILES = {
    "openpii": ("ai4privacy/pii-masking-openpii-1m", "ecfdc547f4a0955600cfe6ab98ba2a162207fcc0",
                "data/validation.jsonl",
                "f945f662cab477e8aef02ce6d9a5525d8b9d4ff6436809649d8c398156fea010"),
    "nemotron": ("nvidia/Nemotron-PII", "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed",
                 "data/test-00000-of-00001.parquet",
                 "1a4b0512ecb5370f0992d29d0f9c07351e6de13f0d7ea33bb18cecb984780247"),
    "gretel_en": ("gretelai/gretel-pii-masking-en-v1", "e06eb1499ca8d54470f085021cd8e54f9efac7fd",
                  "data/test-00000-of-00001.parquet",
                  "c89a6bffa838edb0f2fc6686bf35da4930944519ac97308b0b6e32298594b5b1"),
    "gretel_fin": ("gretelai/synthetic_pii_finance_multilingual",
                   "7b844d16738527a04264f50214cb426a4cea0897",
                   "data/test-00000-of-00001.parquet",
                   "014e1057978f030fce4f4cad7c93b9e3377fd0ded713d2bf4c7de00e3e3c8c72"),
    "mapa": ("joelniklaus/mapa", "bbb2a0157b760465002fd12a61af81b475cd387a", "test.jsonl",
             "9498f67f080b1745b5b41feb32f66beb30e2bd2ca024306a910a7c59c3134d14"),
    "tab": ("ildpil/text-anonymization-benchmark", "1f22ce098996054689b5b331fb20b3a74a78e1c5",
            "echr_test.zip",
            "22e3c7a1f96a9646ca9806f6d7ae6d3156e7bf3a417842518e25a72a126a828a"),
    "privy": ("beki/privy", "dc137a6a976f6b5bb8768e9bb51ec58df930ccd1", "privy-dataset.zip",
              "ad4ffbc8f7102a92f01f8d8023ddf4f5244fea6b23c3f116641fe6bbd8cbbde4"),
}
# PIILO is distributed through Kaggle only (CC BY 4.0). Fetch it with
# `kaggle datasets download lburleigh/piilo-dataset` into RAW/piilo.
PIILO_SHA256 = "aca94ace98d6ba8a1c3fdbef5ee72a42caa578d937181ef897f1ed9f1d61aa2c"

# Who made each corpus, under which license, and how its gold was produced. The
# README and the report render this table verbatim.
INFO = {
    "tab": {
        "title": "TAB (Text Anonymization Benchmark)", "license": "MIT", "real": True,
        "annotation": "human (law students), quality-checked", "langs": "en",
        "url": "https://huggingface.co/datasets/ildpil/text-anonymization-benchmark",
        "what": "European Court of Human Rights judgments"},
    "mapa": {
        "title": "MAPA EUR-Lex", "license": "CC BY 4.0", "real": True,
        "annotation": "human (MAPA project)", "langs": "en fr de es it pt nl",
        "url": "https://huggingface.co/datasets/joelniklaus/mapa",
        "what": "EU court decisions and legal acts"},
    "piilo": {
        "title": "PIILO student essays", "license": "CC BY 4.0", "real": True,
        "annotation": "human; original PII replaced by surrogates", "langs": "en",
        "url": "https://www.kaggle.com/datasets/lburleigh/piilo-dataset",
        "what": "essays written by MOOC learners"},
    "openpii": {
        "title": "Ai4Privacy OpenPII 1M (validation)", "license": "CC BY 4.0", "real": False,
        "annotation": "generated with the text", "langs": "en fr de es it pt nl",
        "url": "https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m",
        "what": "synthetic messages, forms and letters"},
    "nemotron": {
        "title": "NVIDIA Nemotron-PII (test)", "license": "CC BY 4.0", "real": False,
        "annotation": "generated with the text", "langs": "en",
        "url": "https://huggingface.co/datasets/nvidia/Nemotron-PII",
        "what": "synthetic documents across 50+ industries, structured and free text"},
    "gretel_en": {
        "title": "Gretel PII masking EN v1 (test)", "license": "Apache 2.0", "real": False,
        "annotation": "generated with the text", "langs": "en",
        "url": "https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1",
        "what": "synthetic documents across 30+ domains"},
    "gretel_fin": {
        "title": "Gretel synthetic PII finance multilingual (test)", "license": "Apache 2.0",
        "real": False, "annotation": "generated with the text", "langs": "en fr de es it nl",
        "url": "https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual",
        "what": "synthetic financial documents (contracts, EDI, statements)"},
    "privy": {
        "title": "Privy (test-large)", "license": "MIT", "real": False,
        "annotation": "generated with the payload", "langs": "en",
        "url": "https://huggingface.co/datasets/beki/privy",
        "what": "API payloads: JSON, XML, HTML, SQL"},
}

# Corpora whose labels cover every personal-data mention, so a masked character
# outside any span is a genuine over-masking. PIILO labels student names only
# (cited authors and teachers are left unlabeled on purpose), so its precision
# would punish correct masking; it is scored on recall only.
EXHAUSTIVE = {"tab", "mapa", "openpii", "nemotron", "gretel_en", "gretel_fin", "privy"}

D, Q, N = "direct", "quasi", "neutral"

OPENPII = {
    "GIVENNAME": ("PERSON", D), "SURNAME": ("PERSON", D), "TITLE": ("OTHER", N),
    "EMAIL": ("CONTACT", D), "TELEPHONENUM": ("CONTACT", D),
    "STREET": ("ADDRESS", D), "BUILDINGNUM": ("ADDRESS", D), "ZIPCODE": ("LOCATION", Q),
    "CITY": ("LOCATION", Q), "SOCIALNUM": ("GOV_ID", D), "IDCARDNUM": ("GOV_ID", D),
    "PASSPORTNUM": ("GOV_ID", D), "DRIVERLICENSENUM": ("GOV_ID", D), "TAXNUM": ("GOV_ID", D),
    "CREDITCARDNUMBER": ("FINANCIAL", D), "DATE": ("DATE", Q), "TIME": ("DATE", Q),
    "AGE": ("AGE", Q), "SEX": ("DEMOGRAPHIC", Q), "GENDER": ("DEMOGRAPHIC", Q),
}

NEMOTRON = {
    "first_name": ("PERSON", D), "last_name": ("PERSON", D), "name": ("PERSON", D),
    "user_name": ("CONTACT", D), "email": ("CONTACT", D), "phone_number": ("CONTACT", D),
    "fax_number": ("CONTACT", D), "street_address": ("ADDRESS", D), "address": ("ADDRESS", D),
    "ssn": ("GOV_ID", D), "national_id": ("GOV_ID", D), "tax_id": ("GOV_ID", D),
    "certificate_license_number": ("GOV_ID", D),
    "credit_debit_card": ("FINANCIAL", D), "credit_card_number": ("FINANCIAL", D),
    "account_number": ("FINANCIAL", D), "bank_routing_number": ("FINANCIAL", D),
    "swift_bic": ("FINANCIAL", D), "cvv": ("SECRET", D), "pin": ("SECRET", D),
    "password": ("SECRET", D), "api_key": ("SECRET", D), "http_cookie": ("SECRET", D),
    "ipv4": ("NETWORK_ID", D), "ipv6": ("NETWORK_ID", D), "mac_address": ("NETWORK_ID", D),
    "device_identifier": ("NETWORK_ID", D), "customer_id": ("RECORD_ID", D),
    "employee_id": ("RECORD_ID", D), "medical_record_number": ("RECORD_ID", D),
    "health_plan_beneficiary_number": ("RECORD_ID", D), "unique_id": ("RECORD_ID", D),
    "unique_identifier": ("RECORD_ID", D), "vehicle_identifier": ("RECORD_ID", D),
    "license_plate": ("RECORD_ID", D), "biometric_identifier": ("RECORD_ID", D),
    "date_of_birth": ("DATE_OF_BIRTH", D), "date": ("DATE", Q), "date_time": ("DATE", Q),
    "time": ("DATE", Q), "city": ("LOCATION", Q), "county": ("LOCATION", Q),
    "state": ("LOCATION", Q), "country": ("LOCATION", Q), "postcode": ("LOCATION", Q),
    "coordinate": ("LOCATION", Q), "url": ("CONTACT", Q), "company_name": ("ORGANIZATION", Q),
    "age": ("AGE", Q), "occupation": ("DEMOGRAPHIC", Q), "employment_status": ("DEMOGRAPHIC", Q),
    "education_level": ("DEMOGRAPHIC", Q), "race_ethnicity": ("DEMOGRAPHIC", Q),
    "gender": ("DEMOGRAPHIC", Q), "language": ("DEMOGRAPHIC", Q),
    "religious_belief": ("DEMOGRAPHIC", Q), "political_view": ("DEMOGRAPHIC", Q),
    "blood_type": ("DEMOGRAPHIC", Q), "sexuality": ("DEMOGRAPHIC", Q),
}

GRETEL_FIN = {
    "name": ("PERSON", D), "first_name": ("PERSON", D), "last_name": ("PERSON", D),
    "email": ("CONTACT", D), "phone_number": ("CONTACT", D), "user_name": ("CONTACT", D),
    "street_address": ("ADDRESS", D), "iban": ("FINANCIAL", D), "bban": ("FINANCIAL", D),
    "swift_bic_code": ("FINANCIAL", D), "bank_routing_number": ("FINANCIAL", D),
    "credit_card_number": ("FINANCIAL", D), "credit_card_security_code": ("SECRET", D),
    "account_pin": ("SECRET", D), "password": ("SECRET", D), "api_key": ("SECRET", D),
    "ssn": ("GOV_ID", D), "driver_license_number": ("GOV_ID", D),
    "passport_number": ("GOV_ID", D), "customer_id": ("RECORD_ID", D),
    "employee_id": ("RECORD_ID", D), "ipv4": ("NETWORK_ID", D), "ipv6": ("NETWORK_ID", D),
    "date_of_birth": ("DATE_OF_BIRTH", D), "date": ("DATE", Q), "time": ("DATE", Q),
    "date_time": ("DATE", Q), "local_latlng": ("LOCATION", Q), "company": ("ORGANIZATION", Q),
}

PRIVY = {
    "PERSON": ("PERSON", D), "EMAIL_ADDRESS": ("CONTACT", D), "PHONE_NUMBER": ("CONTACT", D),
    "URL": ("CONTACT", Q), "IBAN_CODE": ("FINANCIAL", D), "US_BANK_NUMBER": ("FINANCIAL", D),
    "FINANCIAL": ("FINANCIAL", D), "CREDIT_CARD": ("FINANCIAL", D), "US_SSN": ("GOV_ID", D),
    "US_PASSPORT": ("GOV_ID", D), "US_DRIVER_LICENSE": ("GOV_ID", D), "US_ITIN": ("GOV_ID", D),
    "US_LICENSE_PLATE": ("RECORD_ID", D), "PASSWORD": ("SECRET", D),
    "IP_ADDRESS": ("NETWORK_ID", D), "MAC_ADDRESS": ("NETWORK_ID", D), "IMEI": ("NETWORK_ID", D),
    "LOCATION": ("LOCATION", Q), "COORDINATE": ("LOCATION", Q), "DATE_TIME": ("DATE", Q),
    "ORGANIZATION": ("ORGANIZATION", Q), "AGE": ("AGE", Q), "NRP": ("DEMOGRAPHIC", Q),
    "TITLE": ("OTHER", N),
}

PIILO = {
    "NAME_STUDENT": ("PERSON", D), "EMAIL": ("CONTACT", D), "USERNAME": ("CONTACT", D),
    "ID_NUM": ("RECORD_ID", D), "PHONE_NUM": ("CONTACT", D), "URL_PERSONAL": ("CONTACT", D),
    "STREET_ADDRESS": ("ADDRESS", D),
}

# MAPA coarse classes. Its ADDRESS class is overwhelmingly countries and cities in
# these legal texts, so it is a quasi identifier, as are dates and organizations.
MAPA = {
    "PERSON": ("PERSON", D), "ADDRESS": ("LOCATION", Q), "DATE": ("DATE", Q),
    "ORGANISATION": ("ORGANIZATION", Q), "AMOUNT": ("OTHER", N), "TITLE": ("OTHER", N),
}

# TAB entity types; the kind comes from the annotator's identifier_type instead.
TAB = {
    "PERSON": "PERSON", "CODE": "RECORD_ID", "LOC": "LOCATION", "ORG": "ORGANIZATION",
    "DATETIME": "DATE", "DEM": "DEMOGRAPHIC", "QUANTITY": "OTHER", "MISC": "OTHER",
}
TAB_KIND = {"DIRECT": D, "QUASI": Q, "NO_MASK": N}


def dev_exposure() -> dict:
    return json.loads((Path(__file__).parent / "dev_exposure.json").read_text())


# Honorifics in the corpus languages. TAB writes "Mr Ruslan Kurkaev" as one
# PERSON span, Gretel and Privy sometimes "Dr. Kaitlin Wong"; other corpora label
# the honorific apart or not at all. Splitting a leading honorific off every
# PERSON span into a neutral span puts all corpora on the same footing: a tool
# that masks the name and keeps "Mr" has protected the person.
HONORIFICS = frozenset("""
    mr mrs ms miss mx dr prof sir dame lord lady
    herr frau fr m mme mlle me
    sig sig.ra sig.na signor signora dott dott.ssa avv
    sr sra srta don doña dña
    dhr mevr mevrouw meneer
""".split())  # noqa: SIM905  (one line per language family reads better)


def split_honorifics(text: str, spans: list[Span]) -> list[Span]:
    out = []
    for sp in spans:
        if sp.type == "PERSON" and sp.kind != "neutral":
            value = text[sp.start:sp.end]
            head, _, rest = value.partition(" ")
            if rest.strip() and head.rstrip(".").lower() in HONORIFICS:
                cut = sp.start + len(value) - len(rest.lstrip())
                out.append(Span(sp.start, sp.start + len(head), sp.label, "OTHER", "neutral"))
                out.append(Span(cut, sp.end, sp.label, sp.type, sp.kind))
                continue
        out.append(sp)
    return sorted(set(out), key=lambda s: (s.start, s.end, s.label, s.type, s.kind))


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(name: str) -> Path:
    repo, rev, path, sha = FILES[name]
    dest = RAW / name / Path(path).name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(HF.format(repo=repo, rev=rev, path=path), tmp)
        tmp.rename(dest)
    if _sha256(dest) != sha:
        raise ValueError(f"{dest}: sha256 mismatch, not the pinned {repo}@{rev[:10]}")
    return dest


def _spans(text: str, raw: list[tuple[int, int, str]], table: dict) -> list[Span]:
    out = []
    for start, end, label in raw:
        if label not in table:
            raise KeyError(f"unmapped label {label!r}")
        etype, kind = table[label]
        out.append(Span(start, end, label, etype, kind))
    return sorted(set(out), key=lambda s: (s.start, s.end, s.label, s.type, s.kind))


def _sample(rng: random.Random, items: list, k: int) -> list:
    return items if len(items) <= k else rng.sample(items, k)


def build_openpii(per_lang: int = 30) -> list[Doc]:
    """Stratified by language over the whole validation split (the file is sorted
    by language, so a prefix would be one language)."""
    pools: dict[str, list[dict]] = defaultdict(list)
    with fetch("openpii").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["language"] in LANGS and 200 <= len(row["source_text"]) <= 4000:
                pools[row["language"]].append(row)
    rng = random.Random(SEED)
    docs = []
    for lang in LANGS:
        docs += [openpii_doc(row) for row in
                 _sample(rng, sorted(pools[lang], key=lambda r: r["uid"]), per_lang)]
    return docs


def openpii_doc(row: dict) -> Doc:
    text = row["source_text"]
    raw = [(m["start"], m["end"], m["label"]) for m in row["privacy_mask"]]
    return Doc(f"openpii-{row['uid']}", "openpii", row["language"], text,
               _spans(text, raw, OPENPII), {"region": row.get("region")})


def _parquet(name: str):
    import pandas as pd

    return pd.read_parquet(fetch(name))


def build_nemotron(per_cell: int = 50) -> list[Doc]:
    """50 documents per (locale us/intl) x (format structured/unstructured)."""
    df = _parquet("nemotron")
    rng = random.Random(SEED)
    docs = []
    for locale in ("us", "intl"):
        for fmt in ("structured", "unstructured"):
            cell = df[(df.locale == locale) & (df.document_format == fmt)
                      & (df.text.str.len() <= 6000)].sort_values("uid")
            docs += [nemotron_doc(row) for _, row in _sample(rng, list(cell.iterrows()), per_cell)]
    return docs


def nemotron_doc(row) -> Doc:
    spans = row["spans"]
    spans = ast.literal_eval(spans) if isinstance(spans, str) else spans
    raw = [(s["start"], s["end"], s["label"]) for s in spans]
    return Doc(f"nemotron-{row['uid']}", "nemotron", "en", row["text"],
               _spans(row["text"], raw, NEMOTRON),
               {"locale": row["locale"], "format": row["document_format"], "domain": row["domain"]})


GRETEL_EN = {
    "name": ("PERSON", D), "first_name": ("PERSON", D), "last_name": ("PERSON", D),
    "user_name": ("CONTACT", D), "email": ("CONTACT", D), "phone_number": ("CONTACT", D),
    "fax_number": ("CONTACT", D), "street_address": ("ADDRESS", D), "address": ("ADDRESS", D),
    "ssn": ("GOV_ID", D), "national_id": ("GOV_ID", D), "tax_id": ("GOV_ID", D),
    "certificate_license_number": ("GOV_ID", D), "credit_card_number": ("FINANCIAL", D),
    "account_number": ("FINANCIAL", D), "bank_routing_number": ("FINANCIAL", D),
    "swift_bic": ("FINANCIAL", D), "cvv": ("SECRET", D), "pin": ("SECRET", D),
    "password": ("SECRET", D), "api_key": ("SECRET", D), "ipv4": ("NETWORK_ID", D),
    "ipv6": ("NETWORK_ID", D), "device_identifier": ("NETWORK_ID", D),
    "customer_id": ("RECORD_ID", D), "employee_id": ("RECORD_ID", D),
    "medical_record_number": ("RECORD_ID", D), "health_plan_beneficiary_number": ("RECORD_ID", D),
    "unique_identifier": ("RECORD_ID", D), "vehicle_identifier": ("RECORD_ID", D),
    "license_plate": ("RECORD_ID", D), "biometric_identifier": ("RECORD_ID", D),
    "date_of_birth": ("DATE_OF_BIRTH", D), "date": ("DATE", Q), "date_time": ("DATE", Q),
    "time": ("DATE", Q), "city": ("LOCATION", Q), "state": ("LOCATION", Q),
    "country": ("LOCATION", Q), "postcode": ("LOCATION", Q), "coordinate": ("LOCATION", Q),
    "url": ("CONTACT", Q), "company_name": ("ORGANIZATION", Q),
}


def _occurrences(text: str, value: str) -> list[tuple[int, int]]:
    out, i = [], text.find(value)
    while i != -1:
        out.append((i, i + len(value)))
        i = text.find(value, i + len(value))
    return out


def build_gretel_en_docs(k: int = 200) -> tuple[list[Doc], dict]:
    """Gretel EN ships entity strings without offsets: every occurrence of each
    labeled string is gold. A string that does not occur is counted and dropped."""
    df = _parquet("gretel_en").sort_values("uid")
    rng = random.Random(SEED)
    docs, missing = [], 0
    for _, row in _sample(rng, list(df.iterrows()), k):
        doc, n = gretel_en_doc(row)
        docs.append(doc)
        missing += n
    return docs, {"labeled_strings_not_in_text": missing}


def gretel_en_doc(row) -> tuple[Doc, int]:
    """Every occurrence of each labeled string is gold; an entity with several
    types takes its first. Returns the document and how many labeled strings do
    not occur in the text (dropped)."""
    raw, missing = [], 0
    for ent in ast.literal_eval(row["entities"]):
        hits = _occurrences(row["text"], ent["entity"])
        missing += not hits
        raw += [(s, e, ent["types"][0]) for s, e in hits]
    return Doc(f"gretel_en-{row['uid']}", "gretel_en", "en", row["text"],
               _spans(row["text"], raw, GRETEL_EN),
               {"domain": row["domain"], "document_type": row["document_type"]}), missing


GRETEL_FIN_LANG = {"English": "en", "France": "fr", "German": "de", "Spanish": "es",
                   "Italian": "it", "Dutch": "nl"}


def build_gretel_fin(per_lang: int = 30) -> list[Doc]:
    """Stratified by language; the 120 rows the earlier out-of-distribution study
    used during PrivAiTe's development are excluded."""
    seen = set(dev_exposure()["gretel_fin_index"])
    df = _parquet("gretel_fin").sort_values("index")
    rng = random.Random(SEED)
    docs = []
    for src in GRETEL_FIN_LANG:
        rows = [r for _, r in df[(df.language == src)
                                 & (df.generated_text.str.len() <= 6000)].iterrows()]
        # 8 of the 5,594 test rows carry spans past the end of their text; they are
        # dropped before sampling rather than clipped, since the intended span is unknown.
        rows = [r for r in rows if gretel_fin_valid(r) and int(r["index"]) not in seen]
        docs += [gretel_fin_doc(row) for row in _sample(rng, rows, per_lang)]
    return docs


def gretel_fin_valid(row) -> bool:
    return all(0 <= s["start"] < s["end"] <= len(row["generated_text"])
               for s in json.loads(row["pii_spans"]))


def gretel_fin_doc(row) -> Doc:
    raw = [(s["start"], s["end"], s["label"]) for s in json.loads(row["pii_spans"])]
    return Doc(f"gretel_fin-{row['index']}", "gretel_fin", GRETEL_FIN_LANG[row["language"]],
               row["generated_text"], _spans(row["generated_text"], raw, GRETEL_FIN),
               {"document_type": row["document_type"]})


def build_privy() -> list[Doc]:
    """300 payloads, 60 per format, one per template, none seen in development
    (see pbench.privy)."""
    from pbench.privy import build

    seen = dev_exposure()
    rows, _ = build(fetch("privy"), frozenset(seen["privy_row_index"]),
                    frozenset(tuple(t) for t in seen["privy_template"]))
    return [Doc(r["id"], "privy", "en", r["text"], _spans(r["text"], r["spans"], PRIVY),
                {"format": r["format"]}) for r in rows]


def tokens_to_text(tokens: list[str], spaces: list[bool]) -> tuple[str, list[tuple[int, int]]]:
    """Join tokens, adding one space after each token flagged in `spaces`, and
    return the text with every token's (start, end)."""
    parts, offsets, pos = [], [], 0
    for tok, sp in zip(tokens, spaces):
        offsets.append((pos, pos + len(tok)))
        parts.append(tok + (" " if sp else ""))
        pos += len(tok) + (1 if sp else 0)
    return "".join(parts), offsets


def bio_spans(offsets: list[tuple[int, int]], tags: list[str]) -> list[tuple[int, int, str]]:
    """(start, end, label) from BIO tags. An I- tag that does not continue a span
    of the same label opens a new one (tolerates the I-without-B the sources have)."""
    out: list[list] = []
    cur = None
    for (s, e), tag in zip(offsets, tags):
        if tag == "O":
            cur = None
            continue
        kind, label = tag[:2], tag[2:]
        if kind == "B-" or cur is None or cur[2] != label:
            cur = [s, e, label]
            out.append(cur)
        else:
            cur[1] = e
    return [tuple(x) for x in out]


def mapa_tags(row: dict) -> list[str]:
    """Coarse tags, except that tokens the fine layer tags TITLE ("sig.", "Herr",
    "Mr") become their own TITLE span: an honorific inside a PERSON span would
    require masking it for the name to count as protected, which the other
    corpora never require."""
    out = []
    for coarse, fine in zip(row["coarse_grained"], row["fine_grained"]):
        if fine.endswith("-TITLE"):
            out.append(fine)
        elif coarse.startswith("I-") and out and out[-1].endswith("-TITLE"):
            out.append("B-" + coarse[2:])  # the name restarts after the title
        else:
            out.append(coarse)
    return out


def mapa_file_docs(fname: str, rows: list[dict], max_chars: int = 4000) -> list[Doc]:
    """One MAPA document (its sentences in order) cut into chunks of at most
    `max_chars` at sentence boundaries, sentences joined by a newline."""
    rows = sorted(rows, key=lambda r: r["sentence_number"])
    docs: list[Doc] = []
    chunk, raw = "", []

    def flush() -> None:
        nonlocal chunk, raw
        if chunk.strip():
            docs.append(Doc(f"mapa-{fname}-{len(docs)}", "mapa", rows[0]["language"], chunk,
                            _spans(chunk, raw, MAPA), {"file": fname}))
        chunk, raw = "", []

    for row in rows:
        toks = row["tokens"]
        sent, offs = tokens_to_text(toks, [True] * (len(toks) - 1) + [False])
        if chunk and len(chunk) + 1 + len(sent) > max_chars:
            flush()
        base = len(chunk) + (1 if chunk else 0)
        chunk = f"{chunk}\n{sent}" if chunk else sent
        raw += [(s + base, e + base, lbl) for s, e, lbl in bio_spans(offs, mapa_tags(row))]
    flush()
    return docs


def build_mapa(max_chars: int = 4000) -> list[Doc]:
    """MAPA ships tokenized sentences with BIO tags and no whitespace, so tokens
    are joined with one space. Each document is cut at sentence boundaries into
    chunks of at most `max_chars`, so a 60-page opinion becomes a run of
    ordinary-length requests. Every chunk of every test document is used."""
    by_file: dict[str, list[dict]] = defaultdict(list)
    with fetch("mapa").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["language"] in LANGS:
                by_file[row["file_name"]].append(row)
    return [d for fname in sorted(by_file) for d in mapa_file_docs(fname, by_file[fname], max_chars)]


def tab_doc(annotations: list[dict]) -> Doc:
    """One TAB judgment with a single annotator's labels: the quality-checked
    annotator when there is one, else the first annotator by id."""
    anns = sorted(annotations, key=lambda r: r["annotator_id"])
    qc = [a for a in anns if a["annotator_id"] in (a.get("quality_checked") or [])]
    ann = (qc or anns)[0]
    spans = sorted({
        Span(m["start_offset"], m["end_offset"], m["entity_type"],
             TAB[m["entity_type"]], TAB_KIND[m["identifier_type"]])
        for m in ann["entity_mentions"]
    }, key=lambda s: (s.start, s.end, s.label, s.type, s.kind))
    return Doc(f"tab-{ann['doc_id']}", "tab", "en", ann["text"], spans,
               {"annotator": ann["annotator_id"], "quality_checked": bool(qc)})


def build_tab() -> list[Doc]:
    """Every test judgment once, one consistent annotation per document instead of
    weighting documents by how many people annotated them."""
    with zipfile.ZipFile(fetch("tab")) as z:
        rows = json.loads(z.read("echr_test.json"))
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)
    return [tab_doc(by_doc[d]) for d in sorted(by_doc)]


def piilo_doc(row: dict) -> Doc:
    """Offsets rebuilt from tokens and trailing-whitespace flags; the rebuilt text
    must equal the dataset's full_text exactly."""
    text, offs = tokens_to_text(row["tokens"], row["trailing_whitespace"])
    if text != row["full_text"]:
        raise ValueError(f"PIILO {row['document']}: token rebuild differs from full_text")
    return Doc(f"piilo-{row['document']}", "piilo", "en", text,
               _spans(text, bio_spans(offs, row["labels"]), PIILO))


def build_piilo(k_pii: int = 200) -> list[Doc]:
    """Essays with at least one labeled span, sampled with the shared seed."""
    path = RAW / "piilo" / "piilo-dataset.zip"
    if _sha256(path) != PIILO_SHA256:
        raise ValueError("PIILO archive differs from the evaluated one")
    with zipfile.ZipFile(path) as z:
        rows = json.load(io.TextIOWrapper(z.open("train.json"), encoding="utf-8"))
    pii_rows = sorted((r for r in rows if any(lab != "O" for lab in r["labels"])),
                      key=lambda r: r["document"])
    return [piilo_doc(r) for r in _sample(random.Random(SEED), pii_rows, k_pii)]


BUILDERS = {
    "tab": build_tab,
    "mapa": build_mapa,
    "piilo": build_piilo,
    "openpii": build_openpii,
    "nemotron": build_nemotron,
    "gretel_en": lambda: build_gretel_en_docs()[0],
    "gretel_fin": build_gretel_fin,
    "privy": build_privy,
}


def manifest_entry(docs: list[Doc]) -> dict:
    kinds = Counter(s.kind for d in docs for s in d.spans)
    return {
        **INFO[docs[0].corpus],
        "exhaustive_labels": docs[0].corpus in EXHAUSTIVE,
        "docs": len(docs),
        "chars": sum(len(d.text) for d in docs),
        "by_lang": dict(sorted(Counter(d.lang for d in docs).items())),
        "spans_by_kind": {k: kinds.get(k, 0) for k in ("direct", "quasi", "neutral")},
        "spans_by_type": dict(sorted(Counter(
            s.type for d in docs for s in d.spans if s.kind != "neutral").items())),
        "selection": [{"id": d.id, "sha256": d.sha256} for d in docs],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpora", nargs="*", default=list(BUILDERS))
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[1] / "results" / "corpora.json"
    manifest = json.loads(out.read_text()) if out.exists() else {}
    for name in args.corpora:
        docs = BUILDERS[name]()
        for d in docs:
            d.spans = split_honorifics(d.text, d.spans)
            check_spans(d)
        write_jsonl(CORPORA / f"{name}.jsonl", docs)
        manifest[name] = manifest_entry(docs)
        m = manifest[name]
        print(f"{name}: {m['docs']} docs, {m['chars']} chars, {m['spans_by_kind']}")
    manifest["_meta"] = {"seed": SEED, "files": {k: {"repo": v[0], "revision": v[1],
                                                       "path": v[2], "sha256": v[3]}
                                                   for k, v in FILES.items()},
                         "piilo_sha256": PIILO_SHA256}
    out.write_text(json.dumps(dict(sorted(manifest.items())), indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
