"""PrivAiTe 0.8.0, opt-in `minilm` preset: Presidio and PrivAiTe's recognizers as
under the onnx preset, with PrivAiTe's own model (crp4222/privaite-pii-minilm,
ONNX fp32 on CPU) in place of openai/privacy-filter. Same configuration changes
as the other PrivAiTe rows (see _privaite.py). The model trained on the train
splits of Nemotron-PII, OpenPII, the two Gretel sets and MAPA (see the
contamination table)."""

from pbench.systems._privaite import PrivAiTeSystem


class System(PrivAiTeSystem):
    preset = "minilm"
