"""Dataiku's Kiji Privacy Proxy 1.6.4, the product (not only its model): the Go
backend signed by Dataiku, taken from the official macOS release and started
headless as the desktop app starts it. Its own masking pipeline runs (ONNX model
with CRF transitions, regex detector, its default PII profile).

Isolation: HOME and the database point to a temporary directory (the user's own
Kiji data and settings are never read or written), the transparent MITM proxy
is off (no system proxy change), the UI is not served, telemetry stays off.

`detect` is `POST /api/pii/check`, which returns the spans it masks
("detected_entities ... so evaluators can score detection", in its source).
Those offsets are UTF-8 byte offsets and are converted to character offsets.
`payload` goes through its forward proxy (`/v1/chat/completions`) to the
capturing fake provider (OPENAI_BASE_URL).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from pbench.systems._capture import TOOLS, Capture, free_port, post_chat, wait_http

VERSION = "1.6.4"
RESOURCES = TOOLS / "kiji" / "app" / "Kiji Privacy Proxy.app" / "Contents" / "Resources"


def byte_to_char_offsets(text: str, start: int, end: int) -> tuple[int, int]:
    raw = text.encode("utf-8")
    return len(raw[:start].decode("utf-8", errors="ignore")), len(raw[:end].decode("utf-8", errors="ignore"))


class System:
    supports_payload = True

    def setup(self) -> None:
        binary = RESOURCES / "kiji-proxy"
        if not binary.exists():
            raise RuntimeError("Kiji app missing: run envs/tools.sh")
        self.home = tempfile.TemporaryDirectory(prefix="pbench-kiji-")
        self.capture = Capture()
        self.port = free_port()
        env = {
            **os.environ, "HOME": self.home.name, "DB_PATH": f"{self.home.name}/kiji.db",
            "PROXY_PORT": f":{self.port}", "OPENAI_BASE_URL": f"http://127.0.0.1:{self.capture.port}",
            "OPENAI_API_KEY": "unused", "TRANSPARENT_PROXY_ENABLED": "false", "KIJI_SERVE_UI": "false",
            "KIJI_TELEMETRY_ENABLED": "false", "MODEL_VARIANT": "quantized",
            "ONNXRUNTIME_SHARED_LIBRARY_PATH": str(RESOURCES / "libonnxruntime.1.24.2.dylib"),
            "LOG_REQUESTS": "false", "LOG_RESPONSES": "false", "LOG_PII_CHANGES": "false",
            "LOG_VERBOSE": "false",
        }
        # The binary extracts its embedded model under ./model; run it from the
        # temporary directory so the app bundle is never modified.
        self.proc = subprocess.Popen([str(binary)], cwd=self.home.name, env=env,
                                     stdout=sys.stderr, stderr=sys.stderr)
        wait_http(f"http://127.0.0.1:{self.port}/health")

    def detect(self, text: str, lang: str):
        body = json.dumps({"message": text}).encode()
        for attempt in range(10):
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/pii/check", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    found = json.loads(r.read())["detected_entities"] or []
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429:  # its per-IP rate limiter; wait and retry
                    raise
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError("rate limited 10 times in a row")
        return [(*byte_to_char_offsets(text, e["start"], e["end"]), e["label"]) for e in found]

    def payload(self, messages: list, lang: str):
        self.capture.last = None
        status, body = post_chat(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                 {"model": "mock", "messages": messages},
                                 {"Authorization": "Bearer unused"})
        if self.capture.last is None:
            raise RuntimeError(f"proxy answered {status} without calling the provider: {body[:200]}")
        return self.capture.last["messages"]

    def info(self) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/version", timeout=10) as r:
            version = json.loads(r.read()).get("version")
        manifest = json.loads((RESOURCES / "model" / "quantized" / "model_manifest.json").read_text())
        return {"versions": {"kiji-proxy": version}, "model": "quantized (bundled)",
                "model_sha256": manifest["hashes"]["sha256"],
                "path": "kiji-proxy forward proxy -> capturing fake provider",
                "changed_from_default": {"transparent_proxy": False, "serve_ui": False,
                                         "home": "temporary directory"}}

    def teardown(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=30)
        self.capture.server.shutdown()
        self.home.cleanup()
