"""The real LiteLLM proxy, in front of a capturing fake provider.

LiteLLM decides in its proxy pipeline which parts of a request a guardrail sees
(its unified guardrail layer extracts texts and tool calls before calling a
guardrail; older guardrails get the raw request in a pre-call hook). Calling a
guardrail class directly would bypass that dispatch, so this adapter starts
`litellm --config` exactly as a user would, sends each request to it, and records
the body the proxy forwards to the "provider": a local HTTP server that answers
with a fixed completion. What that server receives is what the provider would.

Text-level detection is not measured for these rows: the Presidio guardrail's
detection is Presidio (see the `presidio` row), and the content filter is scored
on what it lets through in the structured suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import ClassVar

from pbench.systems._capture import MASTER_KEY, REPO, Capture, free_port, wait_http
from pbench.systems._common import versions


class LiteLLMProxySystem:
    supports_payload = True
    supports_text = False
    guardrail: ClassVar[dict] = {}
    needs_presidio = False

    def setup(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="pbench-litellm-")
        self.capture = Capture()
        self.procs: list[subprocess.Popen] = []
        guardrail = dict(self.guardrail)
        if self.needs_presidio:
            pport = free_port()
            env = {**os.environ, "PYTHONPATH": str(REPO)}
            self.procs.append(subprocess.Popen(
                [str(REPO / ".venvs/presidio/bin/python"), "-m", "pbench.presidio_service", str(pport)],
                cwd=REPO, env=env, stdout=sys.stderr, stderr=sys.stderr))
            wait_http(f"http://127.0.0.1:{pport}/health")
            base = f"http://127.0.0.1:{pport}/"
            guardrail["litellm_params"] = {**guardrail["litellm_params"],
                                           "presidio_analyzer_api_base": base,
                                           "presidio_anonymizer_api_base": base}
        config = {
            "model_list": [{"model_name": "mock", "litellm_params": {
                "model": "openai/mock", "api_base": f"http://127.0.0.1:{self.capture.port}/v1",
                "api_key": "unused"}}],
            "guardrails": [guardrail],
            "general_settings": {"master_key": MASTER_KEY},
        }
        cfg = Path(self.tmp.name) / "config.yaml"
        cfg.write_text(json.dumps(config))  # JSON is valid YAML
        self.port = free_port()
        litellm_bin = Path(sys.executable).parent / "litellm"
        env = {**os.environ, "LITELLM_LOG": "ERROR", "LITELLM_TELEMETRY": "False"}
        self.procs.append(subprocess.Popen(
            [str(litellm_bin), "--config", str(cfg), "--port", str(self.port), "--host", "127.0.0.1"],
            env=env, stdout=sys.stderr, stderr=sys.stderr))
        wait_http(f"http://127.0.0.1:{self.port}/health/liveliness")

    def detect(self, text: str, lang: str):
        raise NotImplementedError("payload-only system")

    def request_extra(self, lang: str) -> dict:
        return {}

    def payload(self, messages: list, lang: str):
        self.capture.last = None
        body = json.dumps({"model": "mock", "messages": messages, **self.request_extra(lang)}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {MASTER_KEY}"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                r.read()
        except urllib.error.HTTPError as exc:
            # Every guardrail here masks; none is configured to block, so a refusal
            # is a failure (scored as a full leak), never a protective block.
            raise RuntimeError(f"proxy answered {exc.code}: {exc.read()[:200]!r}") from None
        if self.capture.last is None:
            raise RuntimeError("proxy answered without calling the provider")
        return self.capture.last["messages"]

    def info(self) -> dict:
        return {"versions": versions("litellm"), "guardrail": self.guardrail,
                "path": "litellm --config (real proxy) -> capturing fake provider"}

    def teardown(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
        self.capture.server.shutdown()
        self.tmp.cleanup()
