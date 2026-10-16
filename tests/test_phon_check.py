"""tts/voxcpm/phon_check.py (the c-check: every Turkish c heard as /dʒ/) on the IPA token strings of the 2026-10-12
probes, its CTC decoding, and phon_probe.py's helpers (no torch, no transformers, no Modal)."""

import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VOXCPM = REPO / "tts" / "voxcpm"
sys.path[:0] = [str(VOXCPM), str(REPO / "src")]

import phon_check as pc  # noqa: E402

# facebook/wav2vec2-xlsr-53-espeak-cv-ft on f022-1500's "Vay canına, bu manzara..." (probe-vay v1a right, v1c wrong)
RIGHT = "v aɪ dʒ a n ə n a b u m a n z a r a".split()
WRONG = "v aɪ k a n ə n a b u m a n z a r a".split()


def test_import_needs_only_the_standard_library():
    code = (f"import sys; sys.path[:0] = [{str(VOXCPM)!r}]; import phon_check; "
            "bad = {'torch', 'numpy', 'transformers', 'huggingface_hub', 'soundfile', 'librosa'} & set(sys.modules); "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_c_check_on_the_probe_tokens():
    assert pc.c_check("Vay canına, bu manzara!", RIGHT) == (True, "dʒ 1 of 1 c")
    assert pc.c_check("Vay canına, bu manzara!", WRONG) == (False, "dʒ 0 of 1 c")
    # several c's (probe-canina2 k7: "Cam" heard as "Dam", "cadde" right)
    cam = "Cam kırılmış, cadde de kapanmış!"
    assert pc.c_expected(cam) == 2
    assert pc.c_check(cam, "dʒ a m k ɯ ɾ ɯ l m ɯ ʃ dʒ a d d e".split())[0]
    assert pc.c_check(cam, "d a m k ɯ ɾ ɯ l m ɯ ʃ dʒ a d d e".split()) == (False, "dʒ 1 of 2 c")
    # no c: nothing to hear
    assert pc.c_check("Bütün bu yemekleri hazırladın!", WRONG) == (True, "dʒ 0 of 0 c")
    assert pc.c_check("", []) == (True, "dʒ 0 of 0 c")


def test_c_count_is_turkish_and_counts_runs():
    assert pc.c_expected("CADDE") == 1 and pc.c_expected("Canım, cumartesi!") == 2
    assert pc.c_expected("Çiçekçi çok güzel.") == 0  # ç is another letter
    assert pc.c_expected(unicodedata.normalize("NFD", "Çiçek")) == 0  # a decomposed Ç is still ç
    assert pc.c_expected("hacca gitti") == 1  # a geminate: one long phone for CTC
    assert pc.tr_lower("IŞIK İnce") == "ışık ince"
    # /dʒ/ in the model's vocabulary: long, palatalised, X-SAMPA; the ligature too
    assert pc.dj_count(["dʒ", "dʒː", "dʒʲ", "dZ", "ʤ", "d", "ʒ", "tʃ", "k", "ɡ"]) == 5


def test_spoken_part_of_a_training_text():
    assert pc.spoken_part("(Astonished, with rising intonation.)Vay canına!") == (
        "Astonished, with rising intonation.", "Vay canına!")
    assert pc.spoken_part("Vay canına!") == (None, "Vay canına!")
    assert pc.spoken_part("") == (None, "")


def test_ctc_decoding_matches_batch_decode():
    vocab = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>", 9: "a", 25: "v", 37: "aɪ", 60: "dʒ", 11: "k"}
    ids = [0, 25, 25, 0, 37, 60, 60, 0, 60, 9, 9, 1, 2, 3, 11, 0]
    toks = pc.ctc_tokens(ids, vocab)
    assert toks == ["v", "aɪ", "dʒ", "dʒ", "a", "k"]  # repeats merged unless a blank parts them; specials dropped
    assert pc.ipa_text(toks) == "v aɪ dʒ dʒ a k" and pc.ctc_tokens([], vocab) == []
    assert pc.ctc_tokens([99], vocab) == []  # unknown id -> <unk>: dropped


def test_vocab_and_pins(tmp_path):
    (tmp_path / "vocab.json").write_text(json.dumps({"<pad>": 0, "dʒ": 60}, ensure_ascii=False), encoding="utf-8")
    assert pc.load_vocab(tmp_path / "vocab.json") == {0: "<pad>", 60: "dʒ"}
    assert len(pc.IPA_REV) == 40 and pc.IPA_MODEL == "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    assert "vocab.json" in pc.IPA_FILES and "dʒ" in pc.DJ and pc.IPA_SR == 16000


def test_get_model_loads_once_and_moves(monkeypatch):
    made = []

    class Fake:
        def __init__(self, device):
            made.append(device)
            self.device = device

        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr(pc, "PhonModel", Fake)
    monkeypatch.setattr(pc, "_MODELS", {})
    assert not pc.loaded()
    m = pc.get_model("cuda")
    assert pc.get_model("cpu") is m and m.device == "cpu" and pc.get_model() is m and made == ["cuda"]
    assert pc.loaded()


# ---- phon_probe.py ------------------------------------------------------------------------------------------------


def _phon_probe():
    import importlib.util
    from unittest import mock

    spec = importlib.util.spec_from_file_location("phon_probe_test", VOXCPM / "phon_probe.py")
    mod = importlib.util.module_from_spec(spec)
    had, saved = "modal" in sys.modules, sys.modules.get("modal")
    sys.modules["modal"] = mm = mock.MagicMock()
    try:
        spec.loader.exec_module(mod)
    finally:
        if had:
            sys.modules["modal"] = saved
        else:
            del sys.modules["modal"]
    return mod, mm


def test_phon_probe_helpers(tmp_path):
    pp, mm = _phon_probe()
    (tmp_path / "emo").mkdir()
    (tmp_path / "short").mkdir()
    for rel in ("emo/k1-s0.wav", "emo/k7-s0.wav", "short/k8-s0.wav"):
        (tmp_path / rel).write_bytes(b"RIFF")
    assert sorted(pp.collect(tmp_path)) == ["k1-s0", "k7-s0", "k8-s0"]
    (tmp_path / "short" / "k1-s0.wav").write_bytes(b"RIFF")
    with pytest.raises(ValueError, match="two WAVs"):
        pp.collect(tmp_path)
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [
        {"id": "k1", "text": "Vaay canına! Bütün bu yemekler!", "refs": ["Vay canına! Bütün bu yemekler!"]},
        {"id": "k7-s0", "text": "Cam kırılmış, cadde de kapanmış!"}]), encoding="utf-8")
    texts = pp.canonical_texts(rows)
    assert texts == {"k1": "Vay canına! Bütün bu yemekler!", "k7-s0": "Cam kırılmış, cadde de kapanmış!"}
    res = pp.annotate({"k1-s0": {"tokens": WRONG}, "k1-s1": {"tokens": RIGHT}, "k7-s0": {"tokens": ["d", "dʒ"]},
                       "x-s0": {"tokens": RIGHT}}, texts)
    assert (res["k1-s0"]["c_ok"], res["k1-s1"]["c_ok"], res["k7-s0"]["c_expected"]) == (False, True, 2)
    assert res["k7-s0"]["c_got"] == 1 and not res["k7-s0"]["c_ok"] and "c_ok" not in res["x-s0"]
    fn = mm.App.return_value.function.call_args_list
    assert len(fn) == 1 and fn[0].kwargs["timeout"] == pp.PROBE_TIMEOUT_S and "gpu" not in fn[0].kwargs
    assert pp.PROBE_PRICE * (pp.PROBE_TIMEOUT_S + 2) / 3600 < 0.15 and len(pp.WHISPER_REV) == 40
