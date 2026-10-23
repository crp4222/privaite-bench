"""agentgateway 1.6.0 (Linux Foundation), the standalone binary, with its
prompt guard on requests: every built-in sensitive-data pattern (ssn,
creditCard, phoneNumber, email, caSin), action mask. The gateway routes to the
capturing fake provider through `hostOverride`; what the provider receives is
scored. Its guards are regexes with no language parameter and no named-entity
model, and it has no detection API, so it is scored on requests only.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pbench.systems._capture import TOOLS, Capture, free_port, post_chat, wait_http

VERSION = "1.6.0"
BINARY = TOOLS / "agentgateway" / "agentgateway"
BUILTINS = ["ssn", "creditCard", "phoneNumber", "email", "caSin"]


class System:
    supports_payload = True
    supports_text = False

    def setup(self) -> None:
        if not BINARY.exists():
            raise RuntimeError("agentgateway binary missing: run envs/tools.sh")
        self.capture = Capture()
        self.port = free_port()
        ready = free_port()
        config = {
            "config": {"readinessAddr": f"127.0.0.1:{ready}", "adminAddr": f"127.0.0.1:{free_port()}",
                       "statsAddr": f"127.0.0.1:{free_port()}"},
            "binds": [{"port": self.port, "listeners": [{"routes": [{
                "backends": [{"ai": {"name": "openai", "hostOverride": f"127.0.0.1:{self.capture.port}",
                                     "provider": {"openAI": {"model": "mock"}}}}],
                "policies": {"ai": {"promptGuard": {"request": [{"regex": {
                    "action": "mask", "rules": [{"builtin": b} for b in BUILTINS]}}]}}},
            }]}]}],
        }
        self.tmp = tempfile.TemporaryDirectory(prefix="pbench-agw-")
        cfg = Path(self.tmp.name) / "config.yaml"
        cfg.write_text(json.dumps(config))  # JSON is valid YAML
        self.proc = subprocess.Popen([str(BINARY), "-f", str(cfg)], stdout=sys.stderr, stderr=sys.stderr)
        wait_http(f"http://127.0.0.1:{ready}/healthz/ready")

    def detect(self, text: str, lang: str):
        raise NotImplementedError("payload-only system")

    def payload(self, messages: list, lang: str):
        self.capture.last = None
        status, body = post_chat(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                 {"model": "mock", "messages": messages},
                                 {"Authorization": "Bearer unused"})
        if self.capture.last is None:
            # The guard masks; it is not configured to reject, so a refusal is a
            # failure (scored as a full leak), never a protective block.
            raise RuntimeError(f"gateway answered {status} without calling the provider: {body[:200]}")
        return self.capture.last["messages"]

    def info(self) -> dict:
        out = subprocess.run([str(BINARY), "--version"], capture_output=True, text=True, check=False).stdout
        return {"versions": {"agentgateway": json.loads(out).get("version", VERSION)},
                "prompt_guard": {"request": {"regex": {"action": "mask", "builtins": BUILTINS}}},
                "path": "agentgateway binary -> capturing fake provider"}

    def teardown(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=20)
        self.capture.server.shutdown()
        self.tmp.cleanup()
