from pbench.systems._gliner import GlinerSystem


class System(GlinerSystem):
    """nvidia/gliner-PII: GLiNER trained on Nemotron-PII (so its Nemotron score
    is in-distribution; see the contamination table)."""

    model_id = "nvidia/gliner-PII"
    revision = "bd23e8ef4425fd04e34c5204ab49ffaa706eae79"
