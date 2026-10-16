"""tts/voxcpm/refprep.py (reference preparation, output level) and tts/voxcpm/onset.py (stray onsets; clip edges,
sibilance and laughter for render_corpus.py): synthetic signals, and round 1's real clips and the f022 pilot when they
are here (./voxcpm-contest/, ./voxcpm-corpus/f022/, not in git: skipped otherwise)."""

import io
import json
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tts" / "voxcpm"), str(REPO / "src")]

import onset  # noqa: E402
import refprep  # noqa: E402

CONTEST = REPO / "voxcpm-contest"
AB_TEXTS = {0: "Yarın sabah geliyorsun.", 1: "Yarın sabah geliyor musun?"}  # r1 A/B trials 0-1 (fb/ab_map.json)


def _speech(n: int, sr: int = 16000, level: float = 0.05, seed: int = 0):
    """Speech-like float32 audio: a 150 Hz pulse train with formants, in 3 syllables, quiet gaps between."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    x = sum(np.sin(2 * np.pi * f * t) * a for f, a in ((150, 1.0), (300, 0.6), (700, 0.5), (1200, 0.3), (2500, 0.1)))
    env = (np.sin(np.pi * 3 * t / (n / sr)) ** 2)
    return (level * x * env + 1e-4 * rng.standard_normal(n)).astype(np.float32)


def _write(p: Path, x, sr: int, subtype: str = "PCM_16") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), x, sr, subtype=subtype)
    return p


def _read(b: bytes):
    x, sr = sf.read(io.BytesIO(b), dtype="float32")
    return x, sr


# ---- refprep ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sr, seconds, level", [(16000, 1.23, 0.05), (48000, 2.0, 0.01), (22050, 0.7, 0.4),
                                                (16000, 3.0, 0.9)])
def test_prep_ref_16k_right_padded_and_levelled(tmp_path, sr, seconds, level):
    p = _write(tmp_path / "r.wav", _speech(int(sr * seconds), sr, level), sr)
    y, ysr = _read(refprep.prep_ref(p))
    assert ysr == 16000 and y.ndim == 1 and len(y) % 2560 == 0 and len(y) >= round(16000 * seconds)
    assert len(y) - round(16000 * seconds) < 2560
    lvl = refprep.active_dbfs(y, 16000)
    assert refprep.peak_dbfs(y) <= -1.0 + 1e-3
    if refprep.peak_dbfs(y) < -1.0 - 1e-3:  # the peak ceiling did not take any level back
        assert abs(lvl - refprep.TARGET_DBFS) <= 1.0
    assert not np.any(y[round(16000 * seconds) + 1:])  # the padding: zeros, on the right


def test_prep_ref_level_within_a_db_of_the_target(tmp_path):
    for level in (0.006, 0.02, 0.2):  # quiet (-47 dBFS active; Dima's: -32) to loud references
        p = _write(tmp_path / f"r{level}.wav", _speech(16000 * 2, 16000, level), 16000)
        y, _ = _read(refprep.prep_ref(p))
        assert abs(refprep.active_dbfs(y, 16000) - refprep.TARGET_DBFS) <= 1.0 and refprep.peak_dbfs(y) <= -1.0


def test_prep_ref_of_whole_patches_only_changes_level(tmp_path):
    x = _speech(2560 * 7, 16000, 0.03)
    p = _write(tmp_path / "r.wav", x, 16000, "FLOAT")
    y, _ = _read(refprep.prep_ref(p))
    assert len(y) == len(x)  # nothing padded, nothing trimmed
    g = float(np.dot(y, x) / np.dot(x, x))
    assert g > 1 and np.allclose(y, g * x, atol=1e-6)  # one gain
    assert abs(refprep.active_dbfs(y, 16000) - refprep.TARGET_DBFS) <= 1.0


def test_normalize_output_level_peak_and_silence():
    x = _speech(48000 * 2, 48000, 0.01)
    y = refprep.normalize_output(x, 48000)
    assert y.dtype == np.float32 and len(y) == len(x)
    assert abs(refprep.active_dbfs(y, 48000) - refprep.TARGET_DBFS) <= 0.01 and refprep.peak_dbfs(y) <= -1.0
    spiky = x.copy()
    spiky[1000] = 0.5  # a peak the target gain would push over -1 dBFS: the whole clip comes down
    z, info = refprep.normalize_level(spiky, 48000)
    assert refprep.peak_dbfs(z) == pytest.approx(-1.0, abs=1e-3) and info["peak_cut_db"] < 0
    assert refprep.normalize_level(np.zeros(4800), 48000)[1]["skipped"] == "silent"
    tiny = np.full(48000, 1e-5, np.float32)  # -100 dBFS: never raised by more than MAX_GAIN_DB
    assert np.array_equal(refprep.normalize_output(tiny, 48000), tiny)


def test_prep_ref_file_writes_once(tmp_path):
    p = _write(tmp_path / "r.wav", _speech(16000, 16000), 16000)
    out = refprep.prep_ref_file(p, tmp_path / "cache")
    assert Path(out).read_bytes() == refprep.prep_ref(p)
    m = Path(out).stat().st_mtime_ns
    assert refprep.prep_ref_file(p, tmp_path / "cache") == out and Path(out).stat().st_mtime_ns == m


# ---- onset --------------------------------------------------------------------------------------------------------


def test_text_class_and_coverage():
    assert [onset.text_class(t) for t in ("Yarın", "(Angry tone.)Yeter!", "Bu", "Ankara", "İyi", "Işık", "Çok",
                                          "tamam", '"Gel"', "...")] == [
        "y", "y", "voiced", "vowel", "vowel", "vowel", "voiceless", "voiceless", "voiced", "other"]
    x = _speech(16000, 16000)
    assert onset.onset_flags(x, 16000, "Toplantıya kırk kişi") == {"covered": False, "cls": "voiceless", "bad": False,
                                                                   "why": []}
    silent = onset.onset_flags(np.zeros(16000, np.float32), 16000, "Yarın")
    assert silent["error"] == "silent" and not silent["covered"] and not silent["bad"]


def test_a_clean_voiced_start_is_not_flagged():
    sr = 16000
    x = np.concatenate([np.zeros(int(0.1 * sr), np.float32), _speech(sr, sr, 0.1)])
    f = onset.onset_flags(x, sr, "Bu kitabı okudum.")
    assert f["covered"] and not f["bad"] and f["why"] == []


def test_a_burst_at_sample_0_is_flagged_and_cut_before_the_voicing():
    sr = 48000
    rng = np.random.default_rng(1)
    burst = (0.05 * rng.standard_normal(int(0.04 * sr))).astype(np.float32)  # broadband, within 30 dB of the voice
    gap = np.zeros(int(0.06 * sr), np.float32)
    voice = _speech(sr, sr, 0.1)
    f = onset.onset_flags(np.concatenate([burst, gap, voice]), sr, "Ankara'ya gidiyor.")
    assert f["bad"] and "burst" in f["why"] and f["hf30"] > 0.5 and 0.09 <= f["cut_s"] <= 0.2


def test_trim_onset_cuts_and_fades_in():
    x = np.ones(16000, np.float32)
    y = onset.trim_onset(x, 16000, 0.1)
    assert len(y) == 16000 - int(0.07 * 16000) and y[0] == 0 and y[159] < 1 and y[160] == 1
    assert len(onset.trim_onset(x, 16000, 0.01)) == 16000  # never before the start


def _ab_clips():
    key = CONTEST / "r1" / "ab_anahtar.json"
    if not key.is_file():
        pytest.skip("no voxcpm-contest/r1/ab (round 1's A/B pages)")
    codes = json.loads(key.read_text(encoding="utf-8"))
    out = [(CONTEST / "r1" / "ab" / "s" / f"{t}_{c}.wav", text) for t, text in AB_TEXTS.items() for c in codes.values()]
    if not all(p.is_file() for p, _ in out):
        pytest.skip("round 1's A/B clips are not all here")
    return out


def test_round1_ab_trials_0_1_are_flagged():
    clips = _ab_clips()  # 7 Nano-vLLM voices x "Yarın sabah geliyorsun / geliyor musun?": strays heard in most
    flags = [onset.onset_flags(*sf.read(str(p), dtype="float32"), text) for p, text in clips]
    assert len(clips) == 14 and sum(f["bad"] for f in flags) >= 11


def test_official_api_smoke_clips_are_not_flagged():
    clips = []
    for d in sorted((CONTEST / "smoke").glob("*/")):
        rows = d / "rows.jsonl"
        if rows.is_file():
            clips += [(d / r["wav"], r["prompt"]) for r in map(json.loads, rows.read_text(encoding="utf-8").splitlines())
                      if (d / r["wav"]).is_file()]
    if len(clips) != 21:
        pytest.skip("no voxcpm-contest/smoke (round 1's official-API smoke run, 21 clips)")
    flags = [onset.onset_flags(*sf.read(str(p), dtype="float32"), text) for p, text in clips]
    assert sum(f["bad"] for f in flags) <= 1


def test_tracks_follow_a_known_pitch_and_formants():
    sr = 16000
    t = np.arange(int(0.6 * sr)) / sr
    f0 = 120.0
    pulses = np.zeros_like(t)
    pulses[(np.arange(0, len(t), sr / f0)).astype(int)] = 1.0
    from scipy.signal import lfilter

    y = pulses
    for fc, bw in ((700, 80), (1200, 90), (2600, 120)):  # /a/-like resonators
        r = math.exp(-math.pi * bw / sr)
        y = lfilter([1 - r], [1, -2 * r * math.cos(2 * math.pi * fc / sr), r * r], y)
    T = onset.tracks((0.3 * y / np.abs(y).max()).astype(np.float32), sr)
    mid = slice(10, 50)
    assert np.median(T["F0"][mid]) == pytest.approx(f0, rel=0.02)
    assert np.median(T["F1"][mid]) == pytest.approx(700, rel=0.1) and np.median(T["F2"][mid]) == pytest.approx(
        1200, rel=0.1)


# ---- clip edges, sibilance, laughter (render_corpus.py's gates) ---------------------------------------------------


def _vowel(seconds: float, sr: int = 16000, f0: float = 170.0, amp: float = 0.2):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 10))).astype(np.float32)


def _sil(seconds: float, sr: int = 16000):
    return np.zeros(int(sr * seconds), np.float32)


def test_edges_cut_a_stray_after_the_last_word_and_flag_one_inside():
    sr = 16000
    x = np.concatenate([_vowel(0.7), _sil(0.28), _vowel(0.07), _sil(0.03)])  # pilot clip 41: "n" 280 ms later
    words = [["Özür", 0.0, 0.32], ["dilerim.", 0.32, 0.7]]
    e = onset.edges(x, sr, words, "Özür dilerim.")
    assert e["tail"] == "stray" and e["tail_ms"] <= 100 and e["speech_end_s"] == pytest.approx(0.7, abs=0.02)
    assert e["cut_s"] == pytest.approx(0.82, abs=0.02) and e["head"] == "" and e["pauses"] == []
    early = onset.edges(x, sr, [["Özür", 0.0, 0.3], ["dilerim.", 0.3, 0.45]], "Özür dilerim.")
    assert early["speech_end_s"] == pytest.approx(0.7, abs=0.02)  # Whisper's end too early: the word's own sound
    inside = onset.edges(x, sr, [["Özür", 0.0, 0.32], ["dilerim.", 0.32, 1.05]], "Özür dilerim.")
    assert inside["tail"] == "inside" and "cut_s" not in inside
    long = np.concatenate([_vowel(0.7), _sil(0.3), _vowel(0.6)])
    assert onset.edges(long, sr, words, "Özür dilerim.")["tail"] == "long"
    clean = np.concatenate([_vowel(0.7), _sil(0.4)])
    c = onset.edges(clean, sr, words, "Özür dilerim.")
    assert c["tail"] == "" and c["cut_s"] == pytest.approx(0.82, abs=0.02)  # trailing silence to 120 ms
    assert onset.edges(clean, sr, [], "x") == {"covered": False}
    head = onset.edges(np.concatenate([_vowel(0.3), _vowel(0.7)]), sr, [["Özür", 0.3, 1.0]], "Özür.")
    assert head["head"] == "sound" and head["head_s"] == pytest.approx(0.3, abs=0.02)
    paused = onset.edges(np.concatenate([_vowel(0.5), _sil(0.3), _vowel(0.5)]), sr,
                         [["Bir", 0.0, 0.5], ["iki.", 0.5, 1.3]], "Bir iki.")
    assert paused["pauses"] == [[pytest.approx(0.5, abs=0.02), pytest.approx(0.8, abs=0.02)]]


def test_trim_tail_fades_out():
    y = onset.trim_tail(np.ones(16000, np.float32), 16000, 0.5)
    assert len(y) == 8000 and y[-1] < 0.01 and y[7999 - 320] == 1.0 and y[7999 - 300] < 1.0


def test_voiced_lead_before_a_voiceless_start():
    sr = 16000
    x = np.concatenate([_vowel(0.08, f0=330, amp=0.3), _sil(0.25), _vowel(0.6)])  # pilot "Kan tahlili", "Hiç ..."
    words = [["Kan", 0.0, 0.5], ["tahlili", 0.5, 0.9]]
    assert onset.edges(x, sr, words, "Kan tahlili için aç gelmem gerekiyor.")["head"] == "vlead"
    assert onset.edges(x, sr, words, "Bu kan tahlili.")["head"] == ""  # a voiced start: onset_flags' business
    plain = np.concatenate([_sil(0.05), _vowel(0.6)])
    assert onset.edges(plain, sr, words, "Kan tahlili.")["head"] == ""


def test_sibilance_finds_a_loud_sustained_sibilant():
    sr = 16000
    rng = np.random.default_rng(0)
    from scipy.signal import butter, sosfilt

    hiss = sosfilt(butter(6, 2500, "highpass", fs=sr, output="sos"), rng.standard_normal(int(0.15 * sr)))
    hiss = (hiss / np.abs(hiss).max()).astype(np.float32)
    v = _vowel(0.3)
    loud = np.concatenate([v, 0.5 * hiss, v])
    soft = np.concatenate([v, 0.03 * hiss, v])
    assert onset.sibilance(loud, sr)["sib_db"] > onset.SIB_MAX_DB >= onset.sibilance(soft, sr)["sib_db"]
    assert onset.sibilance(loud, sr)["sib_ms"] >= 100 and onset.sibilance(v, sr)["sib_db"] is None


def _ha(n: int, sr: int = 16000):
    rng = np.random.default_rng(1)
    out = []
    for k in range(n):  # aspiration, a pitched vowel, a short pause
        out += [0.03 * rng.standard_normal(int(0.06 * sr)).astype(np.float32),
                _vowel(0.1, f0=260 + 50 * math.sin(k), amp=0.25), _sil(0.04)]
    return np.concatenate(out)


def test_laugh_detector_on_a_free_span():
    sr = 16000
    x = np.concatenate([_ha(6), _vowel(0.8)])
    lg = onset.laugh(x, sr, (0.0, 1.2))
    assert lg["laugh"] and lg["bursts"] >= 5 and 3 <= lg["burst_rate"] <= 7 and lg["f0_sd"] >= onset.LAUGH_F0_SD
    assert not onset.laugh(x, sr, (1.2, 2.0))["laugh"]  # one steady vowel
    assert not onset.laugh(np.concatenate([_sil(1.0), _vowel(0.5)]), sr, (0.0, 1.0))["laugh"]  # silence
    assert onset.laugh(x, sr, (0.0, 0.2)) == {"laugh": False, "why": "span under 250 ms", "span_s": 0.2}


PILOT = REPO / "voxcpm-corpus" / "f022"
PILOT_WORDS = {"s_156": [["Özür", 0.0, 0.32], ["dilerim.", 0.32, 0.74]]}  # faster-whisper large-v3-turbo


def _pilot():
    rows = [json.loads(x) for x in (PILOT / "sample.jsonl").read_text(encoding="utf-8").splitlines()] if (
        PILOT / "sample.jsonl").is_file() else []
    if len(rows) != 99 or not all((PILOT / "wav" / f"{r['id']}.wav").is_file() for r in rows):
        pytest.skip("no f022 pilot here (voxcpm-corpus/f022: corpus-fetch of the 2026-10-11 pilot)")
    return rows


def test_pilot_sibilance_tail_and_voiced_lead():
    """The listener's notes on the f022 pilot: clip 15 ("Her şey ...", pressed ş) and an unheard angry clip are the
    only sibilance fails; clip 41 ("Özür dilerim." + "n") the only stray tail; two voiceless starts with a voiced
    burst before them the only "vlead"."""
    rows = _pilot()
    clips = {json.loads(x)["id"]: json.loads(x) for x in (PILOT / "clips.jsonl").read_text(
        encoding="utf-8").splitlines()}
    sib, vlead = set(), set()
    for r in rows:
        x, sr = sf.read(str(PILOT / "wav" / f"{r['id']}.wav"), dtype="float32")
        if (onset.sibilance(x, sr)["sib_db"] or -99) > onset.SIB_MAX_DB:
            sib.add(r["id"])
        if onset.edges(x, sr, [["x", 0.0, len(x) / sr]], clips[r["id"]]["render_spoken"]).get("head") == "vlead":
            vlead.add(r["id"])
    assert sib == {"e_calm_093", "e_angry_101"} and vlead == {"q_duz_075", "s_128"}
    x, sr = sf.read(str(PILOT / "wav" / "s_156.wav"), dtype="float32")
    e = onset.edges(x, sr, PILOT_WORDS["s_156"], "Özür dilerim.")
    assert e["tail"] == "stray" and e["tail_db"] > -10 and e["cut_s"] == pytest.approx(0.86, abs=0.02)
