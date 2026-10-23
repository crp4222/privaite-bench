"""scrubadub 2, default detectors (credential, credit_card, email, phone,
twitter, url) with the locale matching the document language. A text library
with no request integration."""

from __future__ import annotations

from pbench.systems._common import versions

LOCALES = {"en": "en_US", "fr": "fr_FR", "de": "de_DE", "es": "es_ES", "it": "it_IT",
           "pt": "pt_PT", "nl": "nl_NL"}


class System:
    supports_payload = False

    def setup(self) -> None:
        self.scrubbers = {}

    def detect(self, text: str, lang: str):
        import scrubadub

        if lang not in self.scrubbers:
            self.scrubbers[lang] = scrubadub.Scrubber(locale=LOCALES[lang])
        return [(f.beg, f.end, f.type) for f in self.scrubbers[lang].iter_filth(text)]

    def info(self) -> dict:
        import scrubadub

        return {"versions": versions("scrubadub"),
                "detectors": sorted(scrubadub.Scrubber()._detectors),
                "changed_from_default": {"locale": "document language"}}
