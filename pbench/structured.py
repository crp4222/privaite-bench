"""The structured suite: the same public documents, placed where agents and
chat clients actually put user data in a request.

No new text is written. Each selected document from the public corpora is
embedded verbatim into six request shapes (the wrapper strings around it carry no
personal data). A gold value leaks through a carrier when it still appears
anywhere in the request the system forwards, tool-call arguments decoded.

Carriers:
  user_text         plain user message (the baseline every tool handles)
  multimodal_text   text part next to an image part
  system_prompt     system message holding the record (CRM context, RAG)
  tool_call_args    assistant tool call whose JSON arguments hold the record
  tool_call_nested  same, record split into a nested object/array structure
  tool_result       role=tool message returning the record to the model
"""

from __future__ import annotations

import json
import random
from collections import Counter

from pbench.corpus import Doc, read_jsonl
from pbench.sources import CORPORA, SEED

CARRIERS = ("user_text", "multimodal_text", "system_prompt", "tool_call_args",
            "tool_call_nested", "tool_result")
IMAGE = {"type": "image_url", "image_url": {"url": "https://example.com/scan.png"}}
MIN_VALUE_CHARS = 4  # shorter values ("12", "Al") match by accident in any request

# (corpus, how many, filter): small, document-shaped records, the kind an agent
# passes to or receives from a tool. Long judgments and essays are left out.
PLAN = [
    ("nemotron", 40, lambda d: d.meta.get("format") == "structured"),
    ("nemotron", 20, lambda d: d.meta.get("format") == "unstructured"),
    ("gretel_en", 40, lambda d: True),
    ("gretel_fin", 30, lambda d: True),
    ("openpii", 42, lambda d: True),
    ("privy", 40, lambda d: d.meta.get("format") == "json"),
]


def gold_values(doc: Doc) -> list[dict]:
    seen, out = set(), []
    for s in doc.spans:
        if s.kind == "neutral":
            continue
        value = doc.text[s.start:s.end].strip()
        if len(value) < MIN_VALUE_CHARS or value in seen:
            continue
        seen.add(value)
        out.append({"value": value, "type": s.type, "kind": s.kind})
    return out


def _call(name: str, arguments: str) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": name, "arguments": arguments}}]}


def _record_args(text: str) -> str:
    """A JSON payload is passed as-is (Privy rows are real API bodies);
    anything else is wrapped as one string field."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=False)
    except ValueError:
        pass
    return json.dumps({"record": text}, ensure_ascii=False)


def _nested_args(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return json.dumps({"customer": {"history": [{"entry": {"text": ln}} for ln in lines],
                                    "source": "crm"}}, ensure_ascii=False)


def carriers(text: str) -> dict[str, list[dict]]:
    return {
        "user_text": [{"role": "user", "content": text}],
        "multimodal_text": [{"role": "user", "content": [{"type": "text", "text": text}, IMAGE]}],
        "system_prompt": [{"role": "system", "content": "Context for this conversation:\n" + text},
                          {"role": "user", "content": "Summarize the context."}],
        "tool_call_args": [{"role": "user", "content": "Save this record."},
                           _call("save_record", _record_args(text)),
                           {"role": "tool", "tool_call_id": "call_1", "content": "saved"}],
        "tool_call_nested": [{"role": "user", "content": "Update the customer file."},
                             _call("update_customer", _nested_args(text)),
                             {"role": "tool", "tool_call_id": "call_1", "content": "updated"}],
        "tool_result": [{"role": "user", "content": "Look up the record."},
                        _call("get_record", json.dumps({"id": "42"})),
                        {"role": "tool", "tool_call_id": "call_1", "content": text}],
    }


def build() -> list[dict]:
    rng = random.Random(SEED)
    items = []
    for corpus, k, keep in PLAN:
        docs = [d for d in read_jsonl(CORPORA / f"{corpus}.jsonl") if keep(d) and gold_values(d)]
        picked = docs if len(docs) <= k else rng.sample(docs, k)
        for doc in sorted(picked, key=lambda d: d.id):
            values = gold_values(doc)
            for carrier, messages in carriers(doc.text).items():
                items.append({"id": f"{doc.id}#{carrier}", "doc": doc.id, "corpus": corpus,
                              "lang": doc.lang, "carrier": carrier, "messages": messages,
                              "values": values})
    return items


def item_sha(item: dict) -> str:
    """Fingerprint of what a structured outcome depends on: the request and the
    gold values its booleans are aligned with."""
    import hashlib

    blob = json.dumps([item["messages"], item["values"]], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def strings(value) -> list[str]:
    """Every string a provider receives, JSON-encoded arguments decoded."""
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, (dict, list)):
            out.extend(strings(decoded))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(strings(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(strings(v))
    return out


def args_valid(messages) -> bool | None:
    """Whether every tool-call `arguments` string still parses as JSON (a
    redaction that breaks it breaks the agent). None when there is no tool call."""
    found = False
    for m in messages if isinstance(messages, list) else []:
        for c in m.get("tool_calls") or []:
            found = True
            try:
                json.loads(c["function"]["arguments"])
            except (ValueError, TypeError, KeyError):
                return False
    return True if found else None


def _shape(message: dict) -> tuple:
    content = message.get("content")
    if isinstance(content, list):
        parts = tuple(p.get("type") for p in content if isinstance(p, dict))
    else:
        parts = "text" if isinstance(content, str) and content else None
    calls = tuple((c.get("id"), (c.get("function") or {}).get("name"))
                  for c in message.get("tool_calls") or [])
    return (message.get("role"), parts, calls, message.get("tool_call_id"))


def not_forwarded(output) -> bool:
    """The system answered the request itself and sent nothing on: a policy
    block ({"blocked": ...}) or its own error reply ({"failed_closed": ...})."""
    return isinstance(output, dict) and ("blocked" in output or "failed_closed" in output)


def structure_kept(original: list[dict], forwarded) -> bool | None:
    """Whether the forwarded request still has the shape of the original: the
    same messages and roles in order, the same tool calls (ids and names), the
    tool results still tied to their call, the same multimodal parts. A system
    that drops a tool call does not leak its arguments, but the provider rejects
    the request or the agent loses the call: that is scored here, not as
    protection. None when nothing was forwarded."""
    if not_forwarded(forwarded):
        return None
    if not isinstance(forwarded, list) or len(forwarded) != len(original):
        return False
    return all(_shape(a) == _shape(b) for a, b in zip(original, forwarded))


def leaks(item: dict, output) -> list[bool]:
    if not_forwarded(output):
        return [False] * len(item["values"])
    haystack = "\n".join(strings(output))
    return [v["value"] in haystack for v in item["values"]]


def main() -> None:
    items = build()
    path = CORPORA / "structured.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    tmp.replace(path)  # a run reading the suite never sees half a file
    print(len(items), "requests;", dict(Counter(i["corpus"] for i in items if i["carrier"] == "user_text")),
          "docs;", sum(len(i["values"]) for i in items if i["carrier"] == "user_text"), "values per carrier")


if __name__ == "__main__":
    main()
