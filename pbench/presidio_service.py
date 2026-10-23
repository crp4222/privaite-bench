"""A local Presidio service with the analyzer's and anonymizer's REST contract
(POST /analyze, POST /anonymize, GET /health), as the official
mcr.microsoft.com/presidio-* images expose it. LiteLLM's Presidio guardrail is an
HTTP client of that service, so the LiteLLM rows talk to this over the loopback,
exactly as a deployment would. Runs in the presidio env:

    .venvs/presidio/bin/python -m pbench.presidio_service PORT
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pbench.systems._common import SPACY_MODELS, presidio_nlp_engine


def build():
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    analyzer = AnalyzerEngine(nlp_engine=presidio_nlp_engine(), supported_languages=list(SPACY_MODELS))
    anonymizer = AnonymizerEngine()

    def analyze(body: dict) -> list[dict]:
        results = analyzer.analyze(
            text=body["text"], language=body.get("language", "en"),
            entities=body.get("entities"), score_threshold=body.get("score_threshold"),
            return_decision_process=False,
        )
        return [{"entity_type": r.entity_type, "start": r.start, "end": r.end,
                 "score": r.score} for r in results]

    def anonymize(body: dict) -> dict:
        results = [RecognizerResult(r["entity_type"], r["start"], r["end"], r["score"])
                   for r in body.get("analyzer_results", [])]
        operators = {k: OperatorConfig(v.pop("type"), v)
                     for k, v in (body.get("anonymizers") or {}).items()}
        out = anonymizer.anonymize(text=body["text"], analyzer_results=results,
                                   operators=operators or None)
        return {"text": out.text, "items": [
            {"operator": i.operator, "entity_type": i.entity_type, "start": i.start,
             "end": i.end, "text": i.text} for i in out.items]}

    return analyze, anonymize


def main() -> None:
    port = int(sys.argv[1])
    analyze, anonymize = build()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep stderr quiet
            pass

        def _send(self, code: int, payload) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send(200, {"status": "ok"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                if self.path.rstrip("/").endswith("/analyze"):
                    self._send(200, analyze(body))
                elif self.path.rstrip("/").endswith("/anonymize"):
                    self._send(200, anonymize(body))
                else:
                    self._send(404, {"error": "not found"})
            except Exception as exc:  # noqa: BLE001  (an HTTP 400, as the official service answers)
                self._send(400, {"error": type(exc).__name__})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("presidio service ready", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
