"""PrivAiTe 0.8.0, opt-in `ultrafast` preset: PrivAiTe's own model
(crp4222/privaite-pii-minilm) as Core ML on the Apple Neural Engine, and Presidio
on blank spaCy pipelines (its pattern recognizers and PrivAiTe's, no spaCy NER).
The one PrivAiTe row that is not CPU only: the Neural Engine is what the preset
is for. Same other configuration changes as the PrivAiTe rows (see
_privaite.py)."""

from pbench.systems._privaite import PrivAiTeSystem


class System(PrivAiTeSystem):
    preset = "ultrafast"
