"""tts/voxcpm/pause_clean.py (breath noise in pauses turned down): synthetic tone bursts with noise in the gaps, and
three of round 4's long s3-1500 renders when they are here (./voxcpm-eval/, not in git: skipped otherwise)."""

import json
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tts" / "voxcpm"), str(REPO / "src")]

import onset  # noqa: E402
import pause_clean as pc  # noqa: E402

LEVEL_DBFS = -20.0  # the tones' level: L of the synthetic clips
S3_1500 = REPO / "voxcpm-eval" / "voxcpm2-f022-s3" / "step_0001500@ts16-cfg2" / "wav" / "heldout"
PAGE_IDS = ("heldout_246f19253a025c4e", "heldout_ffd37c4fa20156e7", "heldout_c0a2042fd4d1c99f")  # r4 trials 17 16 15


def _ramp(n: int, k: int):
    """1 with raised-cosine ramps of k samples at both ends."""
    w = np.ones(n)
    r = 0.5 - 0.5 * np.cos(np.pi * (np.arange(k) + 0.5) / k)
    w[:k], w[n - k:] = r, r[::-1]
    return w


def _amp(db: float) -> float:
    """dB -> amplitude."""
    return 10 ** (db / 20)


def _tone(n: int, sr: int, f0: float = 180.0):
    """A voiced-like harmonic complex at LEVEL_DBFS RMS, 10 ms ramps."""
    t = np.arange(n) / sr
    x = sum(a * np.sin(2 * np.pi * f0 * k * t) for k, a in ((1, 1.0), (2, 0.7), (3, 0.5), (4, 0.4), (5, 0.3), (6, 0.2)))
    return x / np.sqrt(np.mean(x ** 2)) * _amp(LEVEL_DBFS) * _ramp(n, int(0.01 * sr))


def _noise(n: int, sr: int, rel_db: float, seed: int, band=(1000.0, 4000.0), fade_s: float = 0.04):
    """Band-limited noise (a breath: broadband, centroid ~2.4 kHz) at rel_db re LEVEL_DBFS, raised-cosine fades."""
    from scipy.signal import butter, sosfilt

    v = sosfilt(butter(4, band, "bandpass", fs=sr, output="sos"), np.random.default_rng(seed).standard_normal(n + 2000))
    v = v[2000:]
    return v / np.sqrt(np.mean(v ** 2)) * _amp(LEVEL_DBFS + rel_db) * _ramp(n, max(1, int(fade_s * sr)))


def _clip(sr: int, gaps, seed: int = 0):
    """floor 0.2 s, then (tone 0.5 s, gap) per gap, a last tone, floor 0.2 s; gap = list of (start_s, samples) to put in
    a 0.45 s pause. A -70 dB white floor everywhere (no digital silence). (audio, tone spans, gap spans) in samples."""
    s = lambda t: int(round(t * sr))  # noqa: E731
    parts, tones, gspans, pos = [np.zeros(s(0.2))], [], [], s(0.2)
    for g in list(gaps) + [None]:
        parts.append(_tone(s(0.5), sr))
        tones.append((pos, pos + s(0.5)))
        pos += s(0.5)
        if g is None:
            break
        gap = np.zeros(s(0.45))
        for t0, v in g:
            gap[s(t0):s(t0) + len(v)] += v
        parts.append(gap)
        gspans.append((pos, pos + s(0.45)))
        pos += s(0.45)
    parts.append(np.zeros(s(0.2)))
    x = np.concatenate(parts)
    x += np.random.default_rng(seed + 99).standard_normal(len(x)) * _amp(LEVEL_DBFS - 70)
    return x, tones, gspans


def _frame_db(x, sr: int):
    """dB of 25 ms windows on the 10 ms grid (pause_clean's frames) at sr."""
    h, w = sr // 100, int(0.025 * sr)
    n = len(x) // h
    xp = np.concatenate([np.zeros(w), x, np.zeros(w)])
    c = ((np.arange(n) + 0.5) * h).astype(int)
    return 10 * np.log10((xp[c[:, None] + w - w // 2 + np.arange(w)[None, :]] ** 2).mean(axis=1) + 1e-20)


def _breath_gap(sr: int, rel_db: float = -32.0, seed: int = 1):
    return [(0.1, _noise(int(0.25 * sr), sr, rel_db, seed))]


@pytest.mark.parametrize("sr", [16000, 48000])
def test_breaths_in_gaps_are_turned_down_and_tones_untouched(sr):
    x, tones, gaps = _clip(sr, [_breath_gap(sr, -32.0, 1), _breath_gap(sr, -38.0, 2)])
    y, rep = pc.clean(x, sr)
    assert y.dtype == np.float32 and y.shape == x.shape
    assert rep["pauses"] == 2 and rep["breaths"] == 2 and abs(rep["L_dbfs"] - LEVEL_DBFS) < 0.5
    x0 = x - x.mean()
    for a, b in tones:  # the tones and 30 ms around them: the input sample for sample
        a, b = a - int(0.03 * sr), b + int(0.03 * sr)
        assert np.max(np.abs(y[a:b] - x0[a:b])) < 1e-6
    for (a, b), r, rel in zip(gaps, rep["regions"], (-32.0, -38.0)):
        assert a / sr < r["t0"] < r["t1"] < b / sr
        assert abs(r["peak_db"] - rel) < 1.5 and abs(r["cut_db"] - (-pc.TARGET_DB - r["peak_db"])) < 0.05
        before = _frame_db(x0[a:b], sr) - LEVEL_DBFS
        after = _frame_db(y[a:b].astype(np.float64), sr) - LEVEL_DBFS
        assert before.max() > rel - 1.5
        assert after.max() <= -pc.TARGET_DB + 1.0  # the breath now under L - 55 dB (25 ms frames)
        assert np.all(after <= before + 1e-6)  # nothing louder anywhere in the gap


def test_cut_is_capped_and_fades_are_smooth():
    sr = 48000
    x, _, gaps = _clip(sr, [_breath_gap(sr, -26.0, 3)])  # 29 dB above the target: the cut stops at 25 dB
    y, rep = pc.clean(x, sr)
    (r,) = rep["regions"]
    assert r["cut_db"] == -pc.MAX_CUT_DB
    a, b = gaps[0]
    x0 = (x - x.mean())[a:b]
    ok = np.abs(x0) > 1e-7
    g = 20 * np.log10(np.abs(y[a:b][ok].astype(np.float64) / x0[ok]))
    assert g.max() < 1e-4 and abs(g.min() + pc.MAX_CUT_DB) < 0.01  # unity outside, -25 dB inside, nothing else
    t = np.flatnonzero(ok) / sr
    steepest = pc.MAX_CUT_DB * np.pi / 2 / (pc.FADE_S * sr)  # dB a sample, at the middle of a raised-cosine ramp
    assert np.max(np.abs(np.diff(g)) / np.maximum(np.diff(t) * sr, 1)) < steepest * 1.05
    down = t[(g < -0.1) & (g > -pc.MAX_CUT_DB + 0.1)]
    first = down[down < (r["t0"] + r["t1"]) / 2 - a / sr]  # the fade in: one ramp of FADE_S
    assert pc.FADE_S * 0.9 < first.max() - first.min() < pc.FADE_S * 1.1


def test_no_change_without_pauses_or_breaths():
    sr = 16000
    x = _tone(sr * 2, sr) + 0.003  # one long voiced stretch, with DC
    y, rep = pc.clean(x, sr)
    assert rep["pauses"] == 0 and rep["breaths"] == 0
    assert np.max(np.abs(y - (x - x.mean()))) < 1e-6  # only the DC is gone
    x, _, _ = _clip(sr, [[], []])  # pauses with only the floor in them
    y, rep = pc.clean(x, sr)
    assert rep["pauses"] == 2 and rep["breaths"] == 0 and np.max(np.abs(y - (x - x.mean()))) < 1e-6
    y, rep = pc.clean(np.zeros(100), sr)  # under 3 frames: unchanged
    assert rep["pauses"] == 0 and not np.any(y)


def test_word_edges_in_a_pause_are_kept():
    """A tone's own noisy decay into the pause (a final "ş" / "h") and a stop burst with its aspiration are not
    breaths; a breath hump in the same pause is."""
    sr = 48000
    n = int(0.12 * sr)
    tail = _noise(n, sr, -18.0, 4, fade_s=0.001) * 10 ** (np.linspace(0, -36, n) / 20)  # -18 dB falling 36 dB
    burst = np.concatenate([_noise(int(0.003 * sr), sr, -22.0, 5, (500.0, 7000.0), 0.0005),
                            _noise(int(0.05 * sr), sr, -30.0, 6, (1500.0, 5000.0), 0.0005)
                            * np.exp(np.linspace(0, -4, int(0.05 * sr)))])
    gap = [(0.0, tail), (0.17, _noise(int(0.14 * sr), sr, -32.0, 7)), (0.36, burst)]
    x, tones, gaps = _clip(sr, [gap])
    y, rep = pc.clean(x, sr)
    a = gaps[0][0]
    x0 = x - x.mean()
    span = lambda t0, t1: slice(a + int(t0 * sr), a + int(t1 * sr))  # noqa: E731
    assert np.max(np.abs(y[span(0.0, 0.12)] - x0[span(0.0, 0.12)])) < 1e-6  # the decay
    assert np.max(np.abs(y[span(0.35, 0.42)] - x0[span(0.35, 0.42)])) < 1e-6  # the burst and its aspiration
    (r,) = rep["regions"]
    assert 0.17 - 0.03 < r["t0"] - a / sr and r["t1"] - a / sr < 0.31 + 0.03 and r["cut_db"] < -15


@pytest.mark.parametrize("sr", [16000, 48000])
def test_word_tail_with_a_bump_is_kept(sr):
    """A noisy tail glued to the word, no dip, a 5 dB bump (a fricated final "r": f022-1500 "tutar,", EMA
    "paylaşılıyor."; the walk against the last frame alone cut it by 25 dB from 40 ms after the word); a breath after
    a dip in the same pause is still turned down."""
    n = int(0.09 * sr)
    env = np.interp(np.arange(n) / sr, [0, 0.015, 0.045, 0.09], [-30.0, -30.0, -25.0, -28.0])
    tail = _noise(n, sr, 0.0, 11, (1200.0, 2800.0), 0.002) * 10 ** (env / 20)
    tail[-int(0.01 * sr):] *= np.linspace(1, 0, int(0.01 * sr))
    x, _, gaps = _clip(sr, [[(0.0, tail), (0.2, _noise(int(0.14 * sr), sr, -32.0, 12))]])
    y, rep = pc.clean(x, sr)
    a = gaps[0][0]
    x0 = x - x.mean()
    assert np.max(np.abs(y[a:a + n + int(0.02 * sr)] - x0[a:a + n + int(0.02 * sr)])) < 1e-6  # the tail
    (r,) = rep["regions"]
    assert 0.2 - 0.03 < r["t0"] - a / sr and r["t1"] - a / sr < 0.34 + 0.03 and r["cut_db"] < -15


def test_stereo_gets_one_gain_and_is_deterministic():
    sr = 16000
    x, _, _ = _clip(sr, [_breath_gap(sr)])
    st = np.stack([x, 0.5 * x], axis=1)
    y, rep = pc.clean(st, sr)
    y2, rep2 = pc.clean(st, sr)
    assert y.shape == st.shape and rep["breaths"] == 1 and np.array_equal(y, y2) and rep == rep2
    m = np.abs(st[:, 0] - st[:, 0].mean()) > 1e-6
    assert np.allclose(y[m, 1] / (st[m, 1] - st[:, 1].mean()), y[m, 0] / (st[m, 0] - st[:, 0].mean()), atol=1e-4)


def test_cli_cleans_a_folder(tmp_path, capsys):
    sr = 16000
    x, _, _ = _clip(sr, [_breath_gap(sr)])
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    sf.write(str(src / "a.wav"), x, sr, subtype="PCM_16")
    sf.write(str(src / "b.wav"), _tone(sr, sr), sr, subtype="FLOAT")
    assert pc.main([str(src), str(dst)]) == 0
    assert sf.info(str(dst / "a.wav")).subtype == "PCM_16" and sf.info(str(dst / "b.wav")).subtype == "FLOAT"
    reps = {r["file"]: r for r in map(json.loads, (dst / "pause_clean.jsonl").read_text(encoding="utf-8").splitlines())}
    assert reps["a.wav"]["breaths"] == 1 and reps["b.wav"]["breaths"] == 0
    a, _ = sf.read(str(src / "a.wav"), dtype="int16")
    b, _ = sf.read(str(dst / "a.wav"), dtype="int16")
    out = np.flatnonzero(np.abs(a.astype(int) - b.astype(int)) > 1)  # PCM: rounded, so +-1 step outside the gate
    r = reps["a.wav"]["regions"][0]
    assert len(out) and out.min() / sr >= r["t0"] - pc.FADE_S - 0.001 and out.max() / sr <= r["t1"] + pc.FADE_S + 0.001
    with pytest.raises(SystemExit):
        pc.main([str(src), str(src)])


def _page_clips():
    paths = [S3_1500 / f"{i}.wav" for i in PAGE_IDS]
    if not all(p.is_file() for p in paths):
        pytest.skip("no round-4 s3-1500 held-out renders here (voxcpm-eval/voxcpm2-f022-s3: eval-fetch)")
    return paths


def test_real_renders_speech_frames_unchanged_breaths_down():
    """Speech frames (voiced or within 14 dB of L) of the output: within 0.1 dB of the input's (its DC removed: DC is
    no sound); each turned-down breath, measured as the gate measures (80 Hz high-pass, 16 kHz), at most at
    max(L - 55, peak - 25) dB. Full band the pauses also hold sub-80 Hz energy (-30 dB re L in 246f...)."""
    for p in _page_clips():
        x, sr = sf.read(str(p), dtype="float64")
        y, rep = pc.clean(x, sr)
        y2, _ = pc.clean(x, sr)
        assert rep["breaths"] >= 2 and np.array_equal(y, y2), p.name
        x0 = x - x.mean()
        A = pc.analyse(onset._resample(pc._hp(x0, sr, pc.HP_HZ), sr, pc.SR))
        speech = A["voiced"] | (A["rel"] >= -pc.PAUSE_DB)
        before, after = _frame_db(x0, sr)[:len(speech)], _frame_db(y.astype(np.float64), sr)[:len(speech)]
        assert speech.sum() > 500 and np.max(np.abs(after - before)[speech]) < 0.1, p.name  # speech frames: identical
        B = pc.analyse(onset._resample(pc._hp(y.astype(np.float64), sr, pc.HP_HZ), sr, pc.SR))
        rel_after = B["rel"] + B["L_dbfs"] - A["L_dbfs"]
        for r in rep["regions"]:  # the frames whose 25 ms window is all under the full cut
            k0, k1 = math.ceil(r["t0"] * 100 + 0.75), math.floor(r["t1"] * 100 - 1.75) + 1
            assert k1 > k0 and not speech[k0:k1].any()
            assert A["rel"][k0:k1].max() > -pc.BREATH_LO_DB
            assert rel_after[k0:k1].max() <= max(-pc.TARGET_DB, r["peak_db"] - pc.MAX_CUT_DB) + 0.5, (p.name, r)
