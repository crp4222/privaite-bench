"""The process every system runs in: one JSON request per stdin line, one JSON
reply per stdout line.

Each system lives in its own virtualenv (their dependencies conflict: LLM Guard
pins transformers 4.x, GLiNER needs another torch, LiteLLM's proxy extras clash
with Presidio's), so the harness never imports a system. It starts
`<env>/bin/python -m pbench.worker <system>` and talks to it over pipes. This
module is stdlib only for that reason.

Requests:
  {"op": "info"}                                   -> versions and configuration
  {"op": "text", "text": str, "lang": str}         -> {"spans": [[s, e, label]], "ms": float}
  {"op": "payload", "messages": list, "lang": str} -> {"messages": list, "ms": float}

`ms` is wall time inside the worker around the system call only. The first
request in each language is preceded by an untimed warm-up call in that language
(several systems build one pipeline per language lazily), so process start, model
load and pipe overhead are never counted.
Libraries that print to stdout cannot corrupt the protocol: fd 1 is moved to
stderr before any system is imported, and replies go to a private copy of it.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
import time
import traceback

WARMUP_TEXT = "Warm-up: contact Jane Doe at jane.doe@example.com or +1 202 555 0143."


def _protocol_stream():
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return out


async def _call(fn, *args):
    result = fn(*args)
    if inspect.isawaitable(result):
        result = await result
    return result


async def serve(name: str) -> None:
    out = _protocol_stream()
    # A bare name is a module of pbench.systems; a dotted path is used as is
    # (the tests drive the protocol with a fake system that way).
    module = importlib.import_module(name if "." in name else f"pbench.systems.{name}")
    system = module.System()
    await _call(system.setup)
    warm: set[str] = set()

    async def warm_up(lang: str) -> None:
        if lang in warm:
            return
        warm.add(lang)
        if getattr(system, "supports_text", True):
            await _call(system.detect, WARMUP_TEXT, lang)
        if getattr(system, "supports_payload", False):
            await _call(system.payload, [{"role": "user", "content": WARMUP_TEXT}], lang)

    await warm_up("en")
    out.write(json.dumps({"ready": True}) + "\n")
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        req = json.loads(line)
        try:
            if req["op"] == "info":
                reply = {"info": system.info()}
            elif req["op"] == "text":
                await warm_up(req["lang"])
                t0 = time.perf_counter()
                spans = await _call(system.detect, req["text"], req["lang"])
                ms = (time.perf_counter() - t0) * 1000
                reply = {"spans": [[int(s), int(e), str(lbl)] for s, e, lbl in spans], "ms": ms}
            elif req["op"] == "payload":
                await warm_up(req["lang"])
                t0 = time.perf_counter()
                messages = await _call(system.payload, req["messages"], req["lang"])
                ms = (time.perf_counter() - t0) * 1000
                reply = {"messages": messages, "ms": ms}
            elif req["op"] == "shutdown":
                break
            else:
                reply = {"error": f"unknown op {req['op']!r}"}
        except Exception as exc:  # noqa: BLE001  (reported per document, never fatal for the run)
            reply = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            traceback.print_exc(file=sys.stderr)
        out.write(json.dumps(reply, ensure_ascii=False) + "\n")
    teardown = getattr(system, "teardown", None)
    if teardown:
        await _call(teardown)


if __name__ == "__main__":
    asyncio.run(serve(sys.argv[1]))
