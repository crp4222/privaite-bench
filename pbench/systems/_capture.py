"""Shared plumbing for systems that run as a separate proxy or gateway process:
a capturing fake provider (an OpenAI-compatible server that records the body it
receives and answers with a fixed completion), free ports, readiness waits, and
the chat-completions POST. What the fake provider records is what a real
provider would receive. Stdlib only."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MASTER_KEY = "sk-pbench-local"
COMPLETION = {
    "id": "chatcmpl-pbench", "object": "chat.completion", "created": 0, "model": "mock",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "ok"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    raise TimeoutError(f"{url} did not come up")


class Capture:
    """Fake OpenAI-compatible provider that keeps the last request body."""

    def __init__(self) -> None:
        self.last: dict | None = None
        capture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                self._json({"data": [{"id": "mock", "object": "model"}]})

            def do_POST(self) -> None:
                capture.last = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._json(COMPLETION)

            def _json(self, payload) -> None:
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.port = free_port()
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()




def post_chat(url: str, body: dict, headers: dict | None = None, timeout: float = 600):
    """POST a chat request; returns (status, parsed body or text)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")


TOOLS = Path.home() / ".cache" / "privaite-bench" / "tools"
