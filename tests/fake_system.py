"""A system for driving pbench.worker in tests: deterministic, instant, and
noisy on stdout on purpose (the protocol must survive that)."""

from __future__ import annotations

import time

CALLS: list[tuple[str, str, str]] = []


class System:
    supports_payload = True

    def setup(self) -> None:
        print("library banner printed to stdout")

    def detect(self, text: str, lang: str):
        CALLS.append(("detect", text, lang))
        if text == "boom":
            raise RuntimeError("detector exploded")
        if text == "hang":
            time.sleep(30)
        print("noise during detection")
        out, i = [], text.find("Jane")
        while i != -1:
            out.append((i, i + 4, "PERSON"))
            i = text.find("Jane", i + 4)
        return out

    async def payload(self, messages: list, lang: str):
        CALLS.append(("payload", str(len(messages)), lang))
        if any(m["content"] == "fail closed" for m in messages):
            return {"failed_closed": "internal error"}
        return [{**m, "content": m["content"].replace("Jane", "<P>")} for m in messages]

    def info(self) -> dict:
        return {"calls": CALLS}
