"""trtts wiring, checked against a stand-in engine (no torch or model download needed)."""

import pytest

from trtts import TurkishTTS, prepare
from trtts.cli import main


class FakeFrontend:
    def spoken(self, text):
        return "NORMALIZER-TR"


class FakeEngine:
    def __init__(self):
        self._frontend = FakeFrontend()
        self.calls = []

    def say(self, text, **kw):
        self.calls.append(("say", text, kw))
        return text

    def stream(self, text, **kw):
        self.calls.append(("stream", text, kw))
        return iter([text])


def test_prepare_is_trnorm():
    assert prepare("Saat 09:30'da 3. kata gelin.") == "Saat dokuz otuzda üçüncü kata gelin."


def test_front_end_replaced():
    eng = FakeEngine()
    TurkishTTS(engine=eng)
    assert eng._frontend.spoken("15.00'te") == "15.00'te"  # identity: trnorm already ran


def test_say_and_stream_get_spoken_form():
    eng = FakeEngine()
    tts = TurkishTTS(engine=eng)
    assert tts.say("%20'si", seed=0) == "yüzde yirmisi"
    assert tts.say(["4'üncü", "SGK'ye"]) == ["dördüncü", "se ge kaya"]
    assert list(tts.stream("10 TL")) == ["on lira"]
    assert eng.calls[0][2]["seed"] == 0
    with pytest.raises(TypeError):
        tts.stream(["a", "b"])


def test_brand_names_reach_the_model_in_turkish_spelling():
    # EMA Lightning issue #8: with spoken() replaced, brands must already be read by trnorm
    assert prepare("akşam iPhone'umdan ChatGPT ile Claude'u karşılaştırdım.") == (
        "akşam ayfonumdan Çet ci pi ti ile Klodu karşılaştırdım."
    )
    # EMA's alphabet step lowercases I to ı: a Latin capital I must not reach it
    assert "I" not in prepare("Instagram'da gördüm, Intel ve LinkedIn")


def test_unsupported_engine():
    with pytest.raises(RuntimeError):
        TurkishTTS(engine=object())


def test_cli_dry_run(capsys):
    assert main(["Ücret 1.250,50 TL.", "--dry-run"]) == 0
    assert capsys.readouterr().out.strip() == "Ücret bin iki yüz elli lira elli kuruş."
