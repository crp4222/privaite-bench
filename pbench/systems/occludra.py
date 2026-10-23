"""occludra/gateway (the AISG gateway), from its source at a pinned commit, run
as it ships: its Presidio service (custom recognizers and post-processor) and
its FastAPI proxy, two processes on the loopback as in its docker-compose file.
Its shipped gateway.yaml is used unchanged except for the provider key: DLP
action "redact", threshold 0.4, its default entity list. LiteLLM inside the
proxy is pointed at the capturing fake provider with OPENAI_API_BASE. Its
per-key limit of 10 requests a second is raised: a benchmark client sends
faster than that, and a 429 says nothing about PII.

`detect` sends the document to the Presidio service with exactly the payload the
proxy sends for a message (its proxy always sends language "en").
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from pbench.systems._capture import TOOLS, Capture, free_port, post_chat, wait_http
from pbench.systems._common import versions

SRC = TOOLS / "occludra"
COMMIT = "e948e230d78f83852c5662d312ed846c32520c17"
API_KEY = "pbench-local-key"
RATE_LIMIT_RPS = 100_000  # shipped default 10; unrelated to detection


class System:
    supports_payload = True

    def setup(self) -> None:
        import yaml

        head = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        if head != COMMIT:
            raise RuntimeError(f"occludra checkout is {head}, expected {COMMIT}: run envs/tools.sh")
        config = yaml.safe_load((SRC / "config" / "gateway.yaml").read_text())
        self.dlp = config["dlp"]
        config["providers"]["openai"]["api_key"] = "unused"
        config["api_keys"] = [{"key": API_KEY, "name": "pbench"}]
        self.tmp = tempfile.TemporaryDirectory(prefix="pbench-occludra-")
        cfg = Path(self.tmp.name) / "gateway.yaml"
        cfg.write_text(yaml.safe_dump(config))
        self.capture = Capture()
        self.presidio_port, self.port = free_port(), free_port()
        uvicorn = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--log-level", "warning"]
        base = {**os.environ, "PYTHONPATH": ""}
        self.procs = [subprocess.Popen(uvicorn + ["--port", str(self.presidio_port)], cwd=SRC / "presidio",
                                       env={**base, "PORT": str(self.presidio_port)},
                                       stdout=sys.stderr, stderr=sys.stderr)]
        wait_http(f"http://127.0.0.1:{self.presidio_port}/health")
        self.procs.append(subprocess.Popen(
            uvicorn + ["--port", str(self.port)], cwd=SRC / "proxy-api",
            env={**base, "PRESIDIO_URL": f"http://127.0.0.1:{self.presidio_port}", "CONFIG_PATH": str(cfg),
                 "PROVIDERS_CONFIG_PATH": str(SRC / "config" / "providers.json"),
                 "DEFAULT_PROVIDER": "openai", "DEFAULT_MODEL": "mock", "RATE_LIMIT_RPS": str(RATE_LIMIT_RPS),
                 "OPENAI_API_BASE": f"http://127.0.0.1:{self.capture.port}/v1"},
            stdout=sys.stderr, stderr=sys.stderr))
        wait_http(f"http://127.0.0.1:{self.port}/health")

    def detect(self, text: str, lang: str):
        body = {"text": text, "language": "en", "entities": self.dlp.get("entities") or None,
                "score_threshold": self.dlp.get("confidence_threshold", 0.4), "anonymize": False}
        req = urllib.request.Request(f"http://127.0.0.1:{self.presidio_port}/process",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            results = json.loads(r.read())["results"]
        return [(x["start"], x["end"], x["entity_type"]) for x in results]

    def payload(self, messages: list, lang: str):
        self.capture.last = None
        status, body = post_chat(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                 {"model": "mock", "messages": messages},
                                 {"Authorization": f"Bearer {API_KEY}"})
        if self.capture.last is None:
            # Its policy block (PII in block mode, or a prompt-injection hit, which
            # it blocks even in redact mode) is a real outcome: nothing left the
            # gateway. Any other refusal is a failure and is scored as one.
            if status == 400 and "pii_policy_violation" in body:
                return {"blocked": status}
            raise RuntimeError(f"gateway answered {status} without calling the provider: {body[:200]}")
        return self.capture.last["messages"]

    def info(self) -> dict:
        return {"versions": {"occludra": COMMIT, **versions("litellm", "presidio-analyzer", "spacy")},
                "dlp": self.dlp, "path": "occludra proxy-api -> occludra presidio -> capturing fake provider",
                "changed_from_default": {"rate_limit_rps": RATE_LIMIT_RPS, "provider api_key": "unused"}}

    def teardown(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            p.wait(timeout=20)
        self.capture.server.shutdown()
        self.tmp.cleanup()
