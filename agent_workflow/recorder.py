#!/usr/bin/env python3
"""
Recording proxy for the agent workflow leak benchmark.

Sits between the client (an agent CLI, or the PrivAiTe gateway) and the real
provider. Every forwarded POST is recorded, then the request is proxied to the
upstream with the client's own headers (auth relayed verbatim) and the
response is streamed back chunk by chunk, so SSE keeps working. This measures
what the provider actually receives without changing behavior.

Each forwarded request produces two JSONL records sharing a `seq`:

  {"seq", "ts", "path", "upstream", "body"}
      written before the upstream call, so a hung or killed upstream still
      leaves the provider-bound body on disk;
  {"seq", "kind": "response", "ts_done", "status", "ttfb_s", "upstream_s",
   "response_bytes", "response_truncated", "response"}
      written after the relay completes, carrying the upstream timing and the
      (capped) response text so provider cache counters
      (`cache_read_input_tokens`, `cached_tokens`) can be read offline.

Neither record changes what is forwarded. The same binary also serves as the
timing tap in front of the PrivAiTe gateway (`run.py` starts a second instance
whose upstream bases point at the gateway), which is what makes exact
per-request gateway latency measurable: tap `ts` is the request before the
scrub, recorder `ts` behind the gateway is the same request after it.

Routing is by path suffix, which covers both arms with one instance:
  .../messages, .../messages/count_tokens  -> Anthropic Messages API
  .../responses                            -> the Codex subscription backend

Stdlib only, no third-party dependencies.

Usage:
    python agent_workflow/recorder.py --port 8401 --log capture.jsonl
"""

from __future__ import annotations

import argparse
import http.client
import itertools
import json
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ANTHROPIC_BASE = "https://api.anthropic.com/v1"
CODEX_BASE = "https://chatgpt.com/backend-api/codex"

# Pre-flight probe marker (see run.py preflight_probe). A request carrying this
# header is harness plumbing, not agent traffic: a tap instance (local http
# upstream) forwards it onward so it traverses the full chain, while a recorder
# instance whose upstream is the real provider (https) absorbs it and answers
# {"probe": "ok"} without contacting the provider. Probe records are written
# with kind "probe"/"probe_response" and no "body" key, so the leak scan and
# the timing analysis never see them.
PROBE_HEADER = "X-Bench-Probe"

# Response text kept per record. Captures are gitignored; the cap only bounds
# memory and disk for pathological streams.
RESPONSE_CAP = 2_000_000

# Hop-by-hop and connection-derived headers: http.client recomputes these for
# the connection it opens, and we force identity encoding so the relayed
# stream is raw bytes.
_SKIP_REQUEST = {
    "host",
    "content-length",
    "accept-encoding",
    "connection",
    "transfer-encoding",
    "content-encoding",
    "keep-alive",
}
_SKIP_RESPONSE = {
    "content-length",
    "transfer-encoding",
    "content-encoding",
    "connection",
    "keep-alive",
    "date",
    "server",
}

_LOG_LOCK = threading.Lock()
_SEQ = itertools.count(1)


def _route(path: str, anthropic_base: str, codex_base: str) -> str | None:
    if path.endswith("/messages/count_tokens"):
        return anthropic_base + "/messages/count_tokens"
    if path.endswith("/messages"):
        return anthropic_base + "/messages"
    if path.endswith("/responses"):
        return codex_base + "/responses"
    return None


class RecorderHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    log_path: str = "capture.jsonl"
    anthropic_base: str = ANTHROPIC_BASE
    codex_base: str = CODEX_BASE

    def log_message(self, fmt: str, *args) -> None:  # quiet the default access log
        pass

    def handle(self) -> None:
        # Codex drops keep-alive connections abruptly; that is not an error
        # worth a traceback in the recorder log.
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log(self, record: dict) -> None:
        with _LOG_LOCK, open(self.log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok", "role": "recorder"})
        else:
            self._json(404, {"error": "recorder only proxies known POST routes"})

    def do_POST(self) -> None:
        upstream = _route(self.path.split("?", 1)[0], self.anthropic_base, self.codex_base)
        if upstream is None:
            self._json(404, {"error": f"no upstream route for {self.path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        is_probe = self.headers.get(PROBE_HEADER) is not None

        seq = next(_SEQ)
        if is_probe and urlsplit(upstream).scheme == "https":
            # Recorder role: the probe proved the chain up to here; never send
            # synthetic harness traffic to the real provider.
            self._log(
                {"seq": seq, "ts": time.time(), "kind": "probe", "path": self.path,
                 "upstream": upstream, "absorbed": True}
            )
            self._json(200, {"probe": "ok"})
            return
        if is_probe:
            # Tap role: log it as a probe (no "body" key, invisible to the
            # scan and the timing pairing) and forward it through the chain.
            self._log(
                {"seq": seq, "ts": time.time(), "kind": "probe", "path": self.path,
                 "upstream": upstream, "absorbed": False}
            )
        else:
            self._log(
                {
                    "seq": seq,
                    "ts": time.time(),
                    "path": self.path,
                    "upstream": upstream,
                    "body": body.decode("utf-8", "replace"),
                }
            )

        resp_kind = "probe_response" if is_probe else "response"
        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _SKIP_REQUEST
        }
        headers["Accept-Encoding"] = "identity"

        parts = urlsplit(upstream)
        query = self.path.split("?", 1)
        target = parts.path + ("?" + query[1] if len(query) == 2 else "")
        if parts.scheme == "https":
            conn = http.client.HTTPSConnection(
                parts.netloc, context=ssl.create_default_context(), timeout=600
            )
        else:
            conn = http.client.HTTPConnection(parts.netloc, timeout=600)
        t0 = time.time()
        try:
            conn.request("POST", target, body=body, headers=headers)
            resp = conn.getresponse()
            ttfb = time.time() - t0
            self.send_response(resp.status)
            for key, value in resp.getheaders():
                if key.lower() not in _SKIP_RESPONSE:
                    self.send_header(key, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            kept: list[bytes] = []
            kept_len = 0
            total = 0
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                total += len(chunk)
                if kept_len < RESPONSE_CAP:
                    take = chunk[: RESPONSE_CAP - kept_len]
                    kept.append(take)
                    kept_len += len(take)
            self.wfile.write(b"0\r\n\r\n")
            done = time.time()
            self._log(
                {
                    "seq": seq,
                    "kind": resp_kind,
                    "ts_done": done,
                    "status": resp.status,
                    "ttfb_s": round(ttfb, 3),
                    "upstream_s": round(done - t0, 3),
                    "response_bytes": total,
                    "response_truncated": total > kept_len,
                    "response": b"".join(kept).decode("utf-8", "replace"),
                }
            )
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            self._log(
                {
                    "seq": seq,
                    "kind": resp_kind,
                    "ts_done": time.time(),
                    "status": None,
                    "error": "connection dropped mid-relay (client closed, or the upstream died)",
                }
            )
        except OSError as exc:
            self.close_connection = True
            self._log(
                {
                    "seq": seq,
                    "kind": resp_kind,
                    "ts_done": time.time(),
                    "status": None,
                    "error": f"upstream relay failed: {exc}",
                }
            )
            try:
                self._json(502, {"error": f"upstream relay failed: {exc}"})
            except OSError:
                pass
        finally:
            conn.close()


def _selftest() -> None:
    import tempfile
    import urllib.request
    from pathlib import Path

    a, c = "https://api.anthropic.com/v1", "https://chatgpt.com/backend-api/codex"
    cases = {
        "/v1/messages": a + "/messages",
        "/v1/messages/count_tokens": a + "/messages/count_tokens",
        "/v1/responses": c + "/responses",
        "/v1/models": None,
        "/health": None,
    }
    for path, expected in cases.items():
        got = _route(path, a, c)
        assert got == expected, f"route {path}: {got!r} != {expected!r}"

    # Loopback relay: dummy upstream + recorder on ephemeral ports, one POST,
    # then check both JSONL records (body before relay, timing + response
    # after) and that the relayed body reaches the client intact.
    upstream_payload = b'data: {"usage":{"cache_read_input_tokens":42}}\n\n'

    class _Upstream(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            pass

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(upstream_payload)))
            self.end_headers()
            self.wfile.write(upstream_payload)

    upstream_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream_srv.serve_forever, daemon=True).start()
    up_port = upstream_srv.server_address[1]

    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "capture.jsonl"

        class _H(RecorderHandler):
            log_path = str(log)
            codex_base = f"http://127.0.0.1:{up_port}/backend"

        rec_srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        threading.Thread(target=rec_srv.serve_forever, daemon=True).start()
        rec_port = rec_srv.server_address[1]

        req = urllib.request.Request(
            f"http://127.0.0.1:{rec_port}/v1/responses",
            data=b'{"input": "loopback"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            relayed = resp.read()
        assert relayed == upstream_payload, "relayed body must be byte-identical"

        records = [json.loads(line) for line in log.read_text().splitlines()]
        reqs = [r for r in records if "body" in r]
        resps = [r for r in records if r.get("kind") == "response"]
        assert len(reqs) == 1 and len(resps) == 1, records
        assert reqs[0]["seq"] == resps[0]["seq"]
        assert reqs[0]["body"] == '{"input": "loopback"}'
        assert resps[0]["status"] == 200
        assert resps[0]["upstream_s"] >= 0 and resps[0]["ttfb_s"] >= 0
        assert resps[0]["response_bytes"] == len(upstream_payload)
        assert resps[0]["response_truncated"] is False
        assert '"cache_read_input_tokens":42' in resps[0]["response"]
        assert resps[0]["ts_done"] >= reqs[0]["ts"]

        # Probe, tap role (http upstream): forwarded through the chain, logged
        # only as probe records, never as scannable request bodies.
        probe_req = urllib.request.Request(
            f"http://127.0.0.1:{rec_port}/v1/responses",
            data=b'{"probe": true}',
            headers={"Content-Type": "application/json", PROBE_HEADER: "1"},
            method="POST",
        )
        with urllib.request.urlopen(probe_req, timeout=10) as resp:
            assert resp.read() == upstream_payload
        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert len([r for r in records if "body" in r]) == 1, "probe must not add a scannable body"
        probes = [r for r in records if r.get("kind") == "probe"]
        assert len(probes) == 1 and probes[0]["absorbed"] is False, probes
        assert len([r for r in records if r.get("kind") == "probe_response"]) == 1

        # Probe, recorder role (https upstream): absorbed, answered locally,
        # the real provider is never contacted.
        absorb_log = Path(td) / "absorb.jsonl"

        class _Absorb(RecorderHandler):
            log_path = str(absorb_log)
            codex_base = "https://127.0.0.1:1/never-reached"

        absorb_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Absorb)
        threading.Thread(target=absorb_srv.serve_forever, daemon=True).start()
        absorb_req = urllib.request.Request(
            f"http://127.0.0.1:{absorb_srv.server_address[1]}/v1/responses",
            data=b'{"probe": true}',
            headers={"Content-Type": "application/json", PROBE_HEADER: "1"},
            method="POST",
        )
        with urllib.request.urlopen(absorb_req, timeout=10) as resp:
            assert json.loads(resp.read()) == {"probe": "ok"}
        absorb_records = [json.loads(line) for line in absorb_log.read_text().splitlines()]
        assert len(absorb_records) == 1 and absorb_records[0]["kind"] == "probe"
        assert absorb_records[0]["absorbed"] is True

        # Fail closed: an unreachable upstream must produce a 502, never a
        # silent bypass to anywhere else.
        dead_log = Path(td) / "dead.jsonl"

        class _Dead(RecorderHandler):
            log_path = str(dead_log)
            codex_base = "http://127.0.0.1:1/dead"

        dead_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Dead)
        threading.Thread(target=dead_srv.serve_forever, daemon=True).start()
        dead_req = urllib.request.Request(
            f"http://127.0.0.1:{dead_srv.server_address[1]}/v1/responses",
            data=b'{"input": "x"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(dead_req, timeout=10)
            raise AssertionError("dead upstream must not yield a success response")
        except urllib.error.HTTPError as exc:
            assert exc.code == 502, exc.code
            assert b"upstream relay failed" in exc.read()
        dead_records = [json.loads(line) for line in dead_log.read_text().splitlines()]
        assert any(
            r.get("kind") == "response" and "upstream relay failed" in (r.get("error") or "")
            for r in dead_records
        ), dead_records

        absorb_srv.shutdown()
        dead_srv.shutdown()
        rec_srv.shutdown()
    upstream_srv.shutdown()
    print(
        f"recorder selftest ok ({len(cases)} routes, loopback relay recorded, "
        "probe forward+absorb, dead upstream fails closed)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="check routing and relay, then exit")
    parser.add_argument("--port", type=int, default=8401)
    parser.add_argument("--log", help="JSONL file for forwarded request bodies")
    parser.add_argument("--anthropic-base", default=ANTHROPIC_BASE)
    parser.add_argument("--codex-base", default=CODEX_BASE)
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.log:
        parser.error("--log is required unless --selftest")
    RecorderHandler.log_path = args.log
    RecorderHandler.anthropic_base = args.anthropic_base.rstrip("/")
    RecorderHandler.codex_base = args.codex_base.rstrip("/")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), RecorderHandler)
    print(f"recorder on 127.0.0.1:{args.port}, logging to {args.log}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
