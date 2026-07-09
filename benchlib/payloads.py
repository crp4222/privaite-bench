"""OpenAI-style message shapes the runners embed PII into, and the flattener.

`compare.py` and `bench_structured.py` both build multimodal parts and tool-call
messages and both read the text back out. The shapes are shared here; the tool
function name stays a caller argument (compare uses `save_record`, bench_structured
uses `save_note`/`save_profile`), and the two read-back helpers keep their
first-match semantics (compare probes a single tool call / single multimodal part).
"""

from __future__ import annotations

import json

# The placeholder image URL both structured probes attach; identical in both files.
IMAGE_URL = "https://example.com/scan.png"


def multimodal_content(text: str) -> list[dict]:
    """A user `content` list: one text part plus one image_url part."""
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
    ]


def tool_call_message(name: str, arguments_obj: dict) -> dict:
    """An assistant message carrying one function tool call whose `arguments` is
    `arguments_obj` JSON-encoded (ensure_ascii=False, so non-ASCII PII stays verbatim)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments_obj, ensure_ascii=False),
            },
        }],
    }


def flatten_messages(messages: list[dict]) -> str:
    """Every string a provider would actually receive across shapes: string content,
    the text of multimodal parts, and tool-call argument strings, newline-joined."""
    chunks: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                chunks.append(function["arguments"])
    return "\n".join(chunks)


def first_tool_args(messages: list[dict]) -> str:
    """The `arguments` string of the first tool call, or ""."""
    for msg in messages:
        for call in (msg.get("tool_calls") or []):
            return call.get("function", {}).get("arguments", "")
    return ""


def first_multimodal_text(messages: list[dict]) -> str:
    """The joined text parts of the first list-shaped `content`, or ""."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""
