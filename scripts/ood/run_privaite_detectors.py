"""Run PrivAiTe's own detectors (openai/privacy-filter + Presidio) on the OOD corpus.

Writes char-span caches results/ood_spans_pf.json and results/ood_spans_presidio.json
(each {doc_id: [[start, end], ...]}). Spans are integer offsets only, no PII text, so
these are safe to commit. Run from the repo root with the PrivAiTe venv:
    <privaite venv>/bin/python -m scripts.ood.run_privaite_detectors
"""

import asyncio
import json

from benchlib.data import load_json
from benchlib.paths import RESULTS, SOLUTIONS, bootstrap

bootstrap()


async def main() -> None:
    corpus = load_json(SOLUTIONS / "_gretel_ood_corpus.json")

    from privaite.config.schema import OnnxDetectorConfig
    from privaite.pii.detector_onnx import OnnxPrivacyFilterDetector

    pf = OnnxPrivacyFilterDetector(OnnxDetectorConfig(enabled=True))
    await pf.initialize()

    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    models = [{"lang_code": lc, "model_name": mn} for lc, mn in
              [("en", "en_core_web_lg"), ("de", "de_core_news_md"), ("it", "it_core_news_md")]]
    nlp = NlpEngineProvider(
        nlp_configuration={"nlp_engine_name": "spacy", "models": models}).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp,
                              supported_languages=[m["lang_code"] for m in models])

    pf_out, pres_out = {}, {}
    for doc in corpus:
        ents = await pf.detect(doc["text"], doc["lang"])
        pf_out[doc["id"]] = [[e.start, e.end] for e in ents]
        results = analyzer.analyze(text=doc["text"], language=doc["lang"])
        pres_out[doc["id"]] = [[r.start, r.end] for r in results]

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ood_spans_pf.json").write_text(json.dumps(pf_out), "utf-8")
    (RESULTS / "ood_spans_presidio.json").write_text(json.dumps(pres_out), "utf-8")
    print("pf spans:", sum(len(v) for v in pf_out.values()),
          "| presidio spans:", sum(len(v) for v in pres_out.values()))
    await pf.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
