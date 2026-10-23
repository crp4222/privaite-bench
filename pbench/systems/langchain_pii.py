"""LangChain v1 `PIIMiddleware`, every built-in type, every surface switched on.

LangChain ships five built-in detectors (email, credit_card, ip, mac_address,
url); other types need a user-supplied regex, so they are not part of the
product's detection and are not added here. One middleware per type, strategy
"redact", with `apply_to_input`, `apply_to_output` and `apply_to_tool_results`
all enabled (the default enables input only): the broadest protection the
middleware offers.

`payload` replays the conversation through the middleware the way an agent loop
runs it: each human or tool message triggers `before_model` on the state at that
point, each AI message triggers `after_model` as it is produced, and the final
state is what the model provider receives. Messages are converted to LangChain
types and back with no other change.
"""

from __future__ import annotations

import json

from pbench.systems._common import versions

TYPES = ("email", "credit_card", "ip", "mac_address", "url")


class System:
    supports_payload = True

    def setup(self) -> None:
        from langchain.agents.middleware import PIIMiddleware

        self.middlewares = [
            PIIMiddleware(t, strategy="redact", apply_to_input=True, apply_to_output=True,
                          apply_to_tool_results=True)
            for t in TYPES
        ]

    def detect(self, text: str, lang: str):
        spans = []
        for mw in self.middlewares:
            spans += [(m["start"], m["end"], m["type"]) for m in mw.detector(text)]
        return spans

    @staticmethod
    def _to_lc(msg: dict):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        role = msg["role"]
        if role == "user":
            return HumanMessage(content=msg["content"])
        if role == "system":
            return SystemMessage(content=msg["content"])
        if role == "tool":
            return ToolMessage(content=msg["content"], tool_call_id=msg["tool_call_id"])
        calls = [{"name": c["function"]["name"], "args": json.loads(c["function"]["arguments"]),
                  "id": c["id"], "type": "tool_call"} for c in msg.get("tool_calls") or []]
        return AIMessage(content=msg.get("content") or "", tool_calls=calls)

    @staticmethod
    def _from_lc(msg) -> dict:
        kind = msg.type
        if kind == "human":
            return {"role": "user", "content": msg.content}
        if kind == "system":
            return {"role": "system", "content": msg.content}
        if kind == "tool":
            return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
        out = {"role": "assistant", "content": msg.content or None}
        if msg.tool_calls:
            out["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                "name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}}
                for c in msg.tool_calls]
        return out

    def payload(self, messages: list, lang: str):
        state = {"messages": []}
        for raw in messages:
            msg = self._to_lc(raw)
            state["messages"] = state["messages"] + [msg]
            for mw in self.middlewares:
                hook = mw.after_model if msg.type == "ai" else mw.before_model
                update = hook(state, None)
                if update:
                    state = {**state, **update}
        return [self._from_lc(m) for m in state["messages"]]

    def info(self) -> dict:
        return {
            "versions": versions("langchain", "langchain-core"),
            "types": list(TYPES),
            "strategy": "redact",
            "changed_from_default": {"apply_to_output": True, "apply_to_tool_results": True},
        }
