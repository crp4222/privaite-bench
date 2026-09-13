"""Benchmark-only adapter for the published Kiji ONNX artifact.

This does not add a product preset. The published ONNX graph is distinct from
the DeBERTa/CRF source model described by another card. Load data files only,
pin the revision, and never execute remote Python or silently truncate a log.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

from benchlib.paths import bootstrap

bootstrap()

from privaite.pii.boundaries import refine_boundary
from privaite.pii.detector_base import PIIDetector
from privaite.pii.detector_onnx import _stitch_windows, decode_bioes_spans
from privaite.pii.entity import PIIEntity

MODEL = "DataikuNLP/kiji-pii-model-onnx"
REVISION = "a0c9dd2ff9023b63378f67bafa50e33925663219"
FILES = (
    "model_quantized.onnx",
    "label_mappings.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)
# The actual pinned artifact. The publisher's model_manifest.json contains a
# different hash; document that mismatch rather than silently trusting it.
MODEL_SHA256 = "b5f3311391a74470ff26c5b700a422832e5dcdb74fedc205ad5a68e70dbf487c"
LABEL_MAP = {
    "AGE": "AGE",
    "BUILDINGNUM": "LOCATION",
    "CITY": "LOCATION",
    "COMPANYNAME": "ORGANIZATION",
    "COUNTRY": "LOCATION",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "DATEOFBIRTH": "DATE_TIME",
    "DRIVERLICENSENUM": "DRIVER_LICENSE",
    "EMAIL": "EMAIL_ADDRESS",
    "FIRSTNAME": "PERSON",
    "IBAN": "IBAN_CODE",
    "IDCARDNUM": "ID_NUMBER",
    "LICENSEPLATENUM": "LICENSE_PLATE",
    "NATIONALID": "NATIONAL_ID",
    "PASSPORTID": "PASSPORT",
    "PASSWORD": "SECRET",
    "PHONENUMBER": "PHONE_NUMBER",
    "SECURITYTOKEN": "SECRET",
    "SSN": "US_SSN",
    "STATE": "LOCATION",
    "STREET": "LOCATION",
    "SURNAME": "PERSON",
    "TAXNUM": "TAX_ID",
    "URL": "URL",
    "USERNAME": "USERNAME",
    "ZIP": "LOCATION",
}


class KijiDetector(PIIDetector):
    def __init__(self, threshold: float = 0.5, offline: bool = True) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold
        self.offline = offline
        self.session = None
        self.tokenizer = None
        self.labels: dict[int, str] = {}
        self.artifacts: dict = {}

    @property
    def name(self) -> str:
        return "kiji-onnx"

    async def initialize(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        paths = {
            name: Path(
                hf_hub_download(
                    MODEL,
                    name,
                    revision=REVISION,
                    token=False,
                    local_files_only=self.offline,
                )
            )
            for name in FILES
        }
        hashes = {}
        for name, path in paths.items():
            with path.open("rb") as file:
                hashes[name] = hashlib.file_digest(file, "sha256").hexdigest()
        if hashes["model_quantized.onnx"] != MODEL_SHA256:
            raise ValueError("The Kiji model hash differs from the evaluated artifact")
        self.labels = {
            int(key): label
            for key, label in json.loads(paths["label_mappings.json"].read_text())[
                "pii"
            ]["id2label"].items()
            if int(key) >= 0
        }
        if set(self.labels) != set(range(53)):
            raise ValueError("Unexpected Kiji label IDs")
        if {
            label.split("-", 1)[1] for label in self.labels.values() if label != "O"
        } != set(LABEL_MAP):
            raise ValueError("Unmapped Kiji labels")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(paths["model_quantized.onnx"]),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        outputs = {o.name: o for o in self.session.get_outputs()}
        if "pii_logits" not in outputs or outputs["pii_logits"].shape[-1] != 53:
            raise ValueError("Unexpected Kiji output contract")
        self.tokenizer = AutoTokenizer.from_pretrained(
            paths["tokenizer_config.json"].parent,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.artifacts = {
            "model": MODEL,
            "revision": REVISION,
            "sha256": hashes,
            "files_bytes": {name: path.stat().st_size for name, path in paths.items()},
            "providers": self.session.get_providers(),
            "inputs": {i.name: i.shape for i in self.session.get_inputs()},
            "outputs": {o.name: o.shape for o in self.session.get_outputs()},
            "tokenizer": type(self.tokenizer).__name__,
            "vocab_size": self.tokenizer.vocab_size,
            "window_tokens": 512,
            "overlap_tokens": 64,
            "threshold": self.threshold,
            "coreference_head_used": False,
        }

    def _predict(self, text: str) -> list[PIIEntity]:
        if self.session is None or self.tokenizer is None:
            raise RuntimeError("Kiji detector not initialized")
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            stride=64,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        input_names = {i.name for i in self.session.get_inputs()}
        windows = []
        for row in range(len(encoding["input_ids"])):
            feed = {
                key: np.asarray([encoding[key][row]], dtype=np.int64)
                for key in input_names
            }
            logits = self.session.run(["pii_logits"], feed)[0][0]
            if (
                logits.shape != (len(encoding["input_ids"][row]), 53)
                or not np.isfinite(logits).all()
            ):
                raise ValueError("Invalid Kiji logits")
            exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probabilities = exp / exp.sum(axis=-1, keepdims=True)
            predicted = probabilities.argmax(axis=-1)
            scores = probabilities.max(axis=-1)
            tokens = [
                (self.labels[int(predicted[i])], float(scores[i]), (int(a), int(b)))
                for i, (a, b) in enumerate(encoding["offset_mapping"][row])
                if a != b
            ]
            windows.append(
                ([t[0] for t in tokens], [t[1] for t in tokens], [t[2] for t in tokens])
            )
        labels, scores, offsets = _stitch_windows(windows, 64)
        # The shared decoder handles BIO as well as BIOES. Decode before label
        # normalization so separate FIRSTNAME/SURNAME spans remain faithful to
        # the model's predictions. Coreference output is intentionally unused.
        spans = decode_bioes_spans(labels, scores, offsets, text)
        return [
            refine_boundary(
                text,
                PIIEntity(
                    LABEL_MAP[s["entity_type"]],
                    s["text"],
                    s["start"],
                    s["end"],
                    s["score"],
                    self.name,
                ),
            )
            for s in spans
            if s["score"] >= self.threshold
        ]

    async def detect(self, text: str, language: str = "en") -> list[PIIEntity]:
        return await asyncio.to_thread(self._predict, text)

    async def shutdown(self) -> None:
        self.session = None
        self.tokenizer = None
