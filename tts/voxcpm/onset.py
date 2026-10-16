#!/usr/bin/env python3
"""Stray onsets: a clip that starts with a short vowel, fry or burst before its first word (Nano-vLLM's first 160 ms
generation patch: 85 of 144 Y-initial r1 clips vs 2 of 32 from the official API). Built from the round-1 forensics
(detect2: formant and creak rules; pair_lead / seed_hf30: the start transient and the unvoiced lead).

    onset_flags(wav, sr, text) -> {"covered", "cls", "bad", "why": [...], measures..., "cut_s"}

Covered: texts whose spoken form starts with a vowel, a voiced consonant (b c d g j l m n r v z) or y; a clip of
another text (voiceless start: an unvoiced lead is the consonant itself) is {"covered": False, "bad": False}. The
text may carry VoxCPM's "(instruction)" prefix. Tracks on the first HEAD_S at 16 kHz, 10 ms frames from sample 0:
level (dB re the clip's 95th-percentile frame), F0 (Praat's to_pitch_ac with its defaults: autocorrelation in a 50 ms
Hann window, 60-550 Hz, up to 15 candidates a frame, the cheapest path over octave jumps and voicing changes) and
F1 / F2 (Praat's to_formant_burg: Burg LPC, 10 poles at 11 kHz, 50 ms Gaussian window, pre-emphasis from 50 Hz); on
80 r1 / eval clips against Praat's own: the same voicing on 96 % of the frames, F0 within 1 st on 99.5 % of those both
call voiced.
"on" = the first frame within 40 dB of the loud level. Rules (why):
  stray  y-initial: a vowel-like run (>= 30 ms, F1 >= 500 Hz, F2 1000-2100 Hz, within 40 dB) between "on" and the
         first /j/ frame (F1 < 430, F2 >= 1700, within 35 dB; first 400 ms): the "a / ay / e + Yarın" pattern.
         Voiced consonant: a vowel-like run (>= 30 ms, F1 >= 600) starting <= 40 ms after "on" and followed within
         150 ms by >= 20 ms of constriction (F1 < 450, >= 8 dB below the run)
  creak  a first voiced run of 30-200 ms starting <= 60 ms after "on", its median F0 >= 5 st under the next run's and
         <= -4 st re the clip's median, the next run within 4 st of it, after a >= 20 ms break or a >= 5 st jump
  burst  a transient at sample 0: > BURST_HF of the first 30 ms' energy above 300 Hz while that 30 ms is within
         -BURST_DB of the loud level (a clean start is silence or voicing)
  lead   > LEAD_MS of audible (within 35 dB) unvoiced sound before the first voiced frame, peaking between LEAD_DB
         and LEAD_MAX_DB (louder: speech whose voicing the tracker missed, not a stray)
"bad" = covered and any rule. Calibration (round-1 forensics): A/B trials 0-1 of round 1 (Nano-vLLM, "Yarın sabah
..."): 12 of 14 flagged; the official API's smoke clips: 1 of 21; of the covered clips, Nano-vLLM r1: 33 % (Y-initial
59 %), official API eval runs: 7 %, Trendyol-TTS: 1 %; the stray rule agrees with the Praat prototype (detect2) on 1002
of 1014 clips. cut_s: where the word starts (the /j/ frame, the end of the stray vowel, the next voiced run, the first voiced
frame), only when it is at most MAX_CUT_S (the stray lives in the first patch); trim_onset cuts there - PRE_S with a
FADE_S fade-in.

render_corpus.py's other clip measures, on the same 10 ms frames (level re the clip's 95th-percentile frame; quiet =
under EDGE_SIL_DB), calibrated on the f022 pilot (99 kept clips, a native listener's notes):
  edges(wav, sr, words, text)  Whisper's word times [[word, start, end], ...]. Speech end: from the last word's end
         through sound until a quiet run of TAIL_GAP_S. "stray": sound of >= TAIL_ART_DB after that gap (pilot clip 41, "Özür
         dilerim." + an "n" 280 ms later at -4 dB); short (<= STRAY_MAX_S): cut, longer: "long" (not cut, rejected).
         "inside": the last word itself holds a quiet run of >= TAIL_GAP_S followed by <= STRAY_MAX_S of sound to its
         end (Whisper put the stray into the word: rejected). cut_s = speech end + TAIL_PAD_S when the clip goes on
         past it (trim_tail: TAIL_FADE_S fade-out). head_s: sound before the first word (> HEAD_MAX_S: suspicious;
         Whisper mostly starts the first word at 0); head "vlead": a text with a voiceless start (not covered by the
         onset rules) whose clip starts with a loud voiced burst (>= -10 dB, voiced from its first audible 10 ms, at
         most VLEAD_MAX_S) and then VLEAD_GAP_S without loud voicing (pilot: "Kan tahlili ...", "Hiç fikrim yok."; no
         other of the 40 voiceless starts); on_s: the first sound frame; pauses: quiet runs >= PAUSE_S between the
         first and the last word.
  sibilance(wav, sr)  frication frames (2-8 kHz >= SIB_FRIC_DB over < 1.5 kHz, within 40 dB): per run, the
         SIB_FRAMES-th loudest frame's 2-8 kHz level re the clip's loud frames (a sibilant sustained for 40 ms that
         loud); sib_db = the highest run. Pilot: "her şey" pressed (clip 15) -3.4 dB, then -4.0 (not heard), all
         others <= -5.7 (median -12).
  laugh(wav, sr, span)  in a span with no recognised word: bursts (30 ms energy maxima >= -30 dB with >= LAUGH_DIP_DB
         dips between), their rate, voiced share and F0 spread (st); laugh = LAUGH_BURSTS bursts at 2.5-9 a second,
         voiced share 0.15-0.9 and F0 sd >= LAUGH_F0_SD (a "ha-ha-ha": aspirated, pitched, bursty).
Importing this module needs no numpy (numpy and scipy when used).
"""

from __future__ import annotations

import math
import re

SR, HOP = 16000, 160  # analysis rate, 10 ms frames
HEAD_S = 0.6  # the stray lives in the first 160 ms patch, after at most some silence
FMIN, FMAX, VOICING, SILENCE, OCTAVE_COST = 60.0, 550.0, 0.45, 0.03, 0.01  # Praat's to_pitch_ac defaults
OCTAVE_JUMP, VUV, MAX_CANDS = 0.35, 0.14, 15
MAX_FORMANT, POLES = 5500.0, 10
BURST_HF, BURST_DB = 0.5, 30.0
LEAD_MS, LEAD_DB, LEAD_MAX_DB = 80, -30.0, -12.0
MAX_CUT_S, PRE_S, FADE_S = 0.3, 0.03, 0.01
SILENT_DBFS = -90.0  # a clip whose loud level is under this has nothing to check
EDGE_SIL_DB, TAIL_GAP_S, TAIL_ART_DB, STRAY_MAX_S = -40.0, 0.15, -30.0, 0.35
TAIL_PAD_S, TAIL_FADE_S, HEAD_MAX_S, PAUSE_S = 0.12, 0.02, 0.15, 0.12
VLEAD_MAX_S, VLEAD_GAP_S, VLEAD_DB = 0.12, 0.15, -10.0
SIB_LO, SIB_FRIC_DB, SIB_FRAMES, SIB_MAX_DB = 2000.0, 10.0, 4, -4.5
LAUGH_DIP_DB, LAUGH_BURSTS, LAUGH_F0_SD = 6.0, 3, 1.5
VOWELS, VOICED, VOICELESS = set("aeıioöuüâîû"), set("bcdgjlmnrvz"), set("çfhkpsşt")
COVERED = ("y", "voiced", "vowel")


def spoken(text: str) -> str:
    """The spoken form of a prompt: without VoxCPM's leading "(instruction)"."""
    return re.sub(r"^\s*\([^)]*\)", "", text or "").strip()


def text_class(text: str) -> str:
    """y | voiced | vowel | voiceless | other, by the first letter of the spoken form (Turkish I / İ)."""
    m = re.search(r"[^\W\d_]", spoken(text))
    if not m:
        return "other"
    c = {"I": "ı", "İ": "i"}.get(m.group(), m.group().lower())
    return ("y" if c == "y" else "voiced" if c in VOICED else "vowel" if c in VOWELS else "voiceless"
            if c in VOICELESS else "other")


def _runs(mask, lo: int = 0) -> list[tuple[int, int]]:
    """[start, end) of the True runs of a boolean sequence (offset lo)."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((lo + i, lo + j))
            i = j
        else:
            i += 1
    return out


def _resample(x, sr: int, to: int):
    if sr == to:
        return x
    from scipy.signal import resample_poly

    g = math.gcd(int(sr), int(to))
    return resample_poly(x, to // g, int(sr) // g)


def _pitch(x, n: int):
    """F0 (Hz, 0 = unvoiced) of n 10 ms frames of x (16 kHz) centred at (i + 0.5) x 10 ms: Praat's ac method (module
    doc): per frame the unvoiced candidate and up to MAX_CANDS autocorrelation peaks, then the cheapest path (octave
    jumps and voicing changes cost)."""
    import numpy as np

    win = int(round(3 * SR / FMIN))  # three periods of the floor
    centers = ((np.arange(n) + 0.5) * HOP).astype(int)
    xp = np.concatenate([np.zeros(win), x - x.mean(), np.zeros(win)])
    fr = xp[centers[:, None] + win - win // 2 + np.arange(win)[None, :]]
    fr = fr - fr.mean(axis=1, keepdims=True)
    local = np.minimum(np.abs(fr).max(axis=1) / max(float(np.abs(x - x.mean()).max()), 1e-9), 1.0)
    w = np.hanning(win)
    nfft = 1 << (2 * win - 1).bit_length()
    r = np.fft.irfft(np.abs(np.fft.rfft(fr * w, nfft)) ** 2, nfft)[:, :win]
    rw = np.fft.irfft(np.abs(np.fft.rfft(w, nfft)) ** 2, nfft)[:win]
    r = r / np.maximum(r[:, :1], 1e-20) / np.maximum(rw / rw[0], 1e-3)
    lo, hi = max(2, int(SR / FMAX)), min(int(SR / FMIN) + 2, win // 2)
    seg = r[:, lo - 1:hi + 1]
    a, b, c = seg[:, :-2], seg[:, 1:-1], seg[:, 2:]
    peak = (b > a) & (b >= c) & (b > 0.5 * VOICING)
    den = 2 * b - a - c
    ok = peak & (den > 1e-12)
    dr = 0.5 * (c - a)
    lag = np.arange(lo, hi)[None, :] + np.where(ok, dr / np.where(ok, den, 1.0), 0.0)
    val = b + np.where(ok, 0.5 * dr * dr / np.where(ok, den, 1.0), 0.0)
    val = np.where(val > 1.0, 1.0 / np.maximum(val, 1e-9), val)  # short windows: reflected around 1
    freq = SR / lag
    rank = np.where(peak, val - OCTAVE_COST * np.log2(FMIN / freq), -np.inf)
    top = np.argsort(-rank, axis=1)[:, :MAX_CANDS]
    rows = np.arange(n)[:, None]
    keep = np.isfinite(rank[rows, top])
    F = np.concatenate([np.zeros((n, 1)), np.where(keep, freq[rows, top], 0.0)], axis=1)
    S = np.concatenate([(VOICING + np.maximum(0.0, 2 - local / (SILENCE / (1 + VOICING))))[:, None],
                        np.where(keep, val[rows, top] - OCTAVE_COST * np.log2(FMAX / np.where(keep, F[:, 1:], 1.0)),
                                 -np.inf)], axis=1)
    voiced = F > 0
    D, psi = S[0].copy(), np.zeros(F.shape, dtype=int)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(1, n):
            both = voiced[i - 1][:, None] & voiced[i][None, :]
            jump = np.abs(np.log2(np.where(both, F[i - 1][:, None] / np.where(both, F[i][None, :], 1.0), 1.0)))
            cost = np.where(both, OCTAVE_JUMP * jump, np.where(voiced[i - 1][:, None] != voiced[i][None, :], VUV, 0.0))
            v = D[:, None] - cost + S[i][None, :]
            psi[i] = v.argmax(axis=0)
            D = v.max(axis=0)
    k = int(D.argmax())
    f0 = np.zeros(n)
    for i in range(n - 1, -1, -1):
        f0[i] = F[i, k]
        k = psi[i, k]
    return f0


def _burg(x, order: int):
    """LPC polynomial (1, a1 .. a_order) of x by Burg's method."""
    import numpy as np

    f, b, a = x[1:].astype(np.float64), x[:-1].astype(np.float64), np.array([1.0])
    for _ in range(order):
        den = float(f @ f + b @ b)
        if den <= 1e-20:
            break
        k = -2.0 * float(f @ b) / den
        a = np.concatenate([a, [0.0]])
        a = a + k * a[::-1]
        f, b = (f + k * b)[1:], (b + k * f)[:-1]
    return a


def _formants(x, n: int):
    """(F1, F2) in Hz (0 where none) of n 10 ms frames of x (16 kHz) centred at (i + 0.5) x 10 ms."""
    import numpy as np

    fs = int(2 * MAX_FORMANT)
    y = _resample(x, SR, fs)
    y = np.append(y[0], y[1:] - math.exp(-2 * math.pi * 50.0 / fs) * y[:-1])  # pre-emphasis from 50 Hz
    win = int(round(0.05 * fs))
    t = (np.arange(win) - (win - 1) / 2) / (win + 1)
    w = (np.exp(-48 * t ** 2) - math.exp(-12)) / (1 - math.exp(-12))  # Praat's Gaussian window
    yp = np.concatenate([np.zeros(win), y, np.zeros(win)])
    out = np.zeros((n, 2))
    for i in range(n):
        c = int(round((i + 0.5) * 0.01 * fs))
        fr = yp[c + win - win // 2:c + win - win // 2 + win] * w
        if not np.any(fr):
            continue
        z = np.roots(_burg(fr, POLES))
        z = z[np.imag(z) > 0]
        f = np.sort(np.angle(z) * fs / (2 * math.pi))
        f = f[(f > 50.0) & (f < MAX_FORMANT - 50.0)]
        out[i, :min(2, len(f))] = f[:2]
    return out[:, 0], out[:, 1]


def tracks(wav, sr: int) -> dict:
    """Level, F0, F1 and F2 of the head of a clip (module doc), and the clip's loud level and median F0."""
    import numpy as np

    x = np.asarray(wav, dtype=np.float64)
    x = x.mean(axis=1) if x.ndim == 2 else x.reshape(-1)
    x = _resample(x, int(sr), SR)
    n_all = len(x) // HOP
    if n_all < 3:
        raise ValueError("clip too short")
    e_all = 10 * np.log10((x[:n_all * HOP].reshape(n_all, HOP) ** 2).mean(axis=1) + 1e-12)
    pk = float(np.percentile(e_all, 95))
    n = min(n_all, int(round(HEAD_S * SR / HOP)))
    f0_all = _pitch(x, n_all)
    f1, f2 = _formants(x[:(n + 5) * HOP], n)
    g = f0_all[f0_all > 0]
    head = x[:int(0.03 * SR)]
    spec = np.abs(np.fft.rfft(head * np.hanning(len(head)), 512)) ** 2
    freqs = np.fft.rfftfreq(512, 1 / SR)
    return {"E": e_all[:n] - pk, "F0": f0_all[:n], "F1": f1, "F2": f2, "f0_med": float(np.median(g)) if len(g) else 0.0,
            "hf30": float(spec[freqs >= 300].sum() / (spec.sum() + 1e-20)),
            "db30": float(10 * np.log10(np.mean(head ** 2) + 1e-12) - pk), "loud_db": pk, "dur": len(x) / SR}


def flags_from_tracks(T: dict, cls: str) -> dict:
    """The rules of the module doc on tracks() of a clip whose text is of class cls."""
    import numpy as np

    E, F0, F1, F2 = T["E"], T["F0"], T["F1"], T["F2"]
    n = len(E)
    o = {"covered": cls in COVERED, "cls": cls, "bad": False, "why": [], "hf30": round(T["hf30"], 3),
         "db30": round(T["db30"], 1)}
    if not o["covered"]:
        return o
    aud = np.flatnonzero(E >= -40)
    if not len(aud) or T["loud_db"] < SILENT_DBFS:
        o.update(covered=False, error="silent")
        return o
    on = int(aud[0])
    o["on_s"] = round(on * 0.01, 2)
    cuts = {}
    es = np.array([E[max(0, i - 1):i + 2].max() for i in range(n)])  # 30 ms max: fry pulses alternate loud / quiet
    if cls == "y":
        vowel = (F1 >= 500) & (F2 >= 1000) & (F2 <= 2100) & (es >= -40)
        j = np.flatnonzero((F1 > 0) & (F1 < 430) & (F2 >= 1700) & (E >= -35))
        j = j[j < 40]
        if len(j):
            kj = int(j[0])
            o["j_s"] = round(kj * 0.01, 2)
            best = max(_runs(vowel[on:kj], on), key=lambda ab: ab[1] - ab[0], default=None)
            if best and best[1] - best[0] >= 3:
                a, b = best
                o.update(stray_ms=(b - a) * 10, stray_f1=int(np.median(F1[a:b])), stray_f2=int(np.median(F2[a:b])),
                         stray_db=round(float(E[a:b].max()), 1))
                cuts["stray"] = kj
    elif cls == "voiced":
        v6 = (F1 >= 600) & (F2 >= 1000) & (F2 <= 2100) & (es >= -38)
        for a, b in _runs(v6[:min(n, on + 25)]):
            if b - a >= 3 and a <= on + 4:
                top = float(E[a:b].max())
                seg = (F1[b:b + 15] < 450) & (F1[b:b + 15] > 0) & (E[b:b + 15] <= top - 8)
                if any(y - x >= 2 for x, y in _runs(seg)):
                    o.update(stray_ms=(b - a) * 10, stray_f1=int(np.median(F1[a:b])),
                             stray_f2=int(np.median(F2[a:b])), stray_db=round(top, 1))
                    cuts["stray"] = b
                break
    vr = [ab for ab in _runs((F0 > 0) & (E >= -40)) if ab[1] - ab[0] >= 3]
    med = T["f0_med"]
    if len(vr) >= 2 and med > 0:
        (a1, b1), (a2, _) = vr[0], vr[1]
        if 3 <= b1 - a1 <= 20 and a1 <= on + 6:
            m1 = 12 * math.log2(float(np.median(F0[a1:b1])) / med)
            nx = F0[a2:min(a2 + 30, n)]
            m2 = 12 * math.log2(float(np.median(nx[nx > 0])) / med)
            if (m2 - m1 >= 5 and m1 <= -4 and abs(m2) <= 4
                    and (a2 - b1 >= 2 or abs(12 * math.log2(F0[a2] / F0[b1 - 1])) >= 5)):
                o.update(creak_st=round(m1, 1), creak_ms=(b1 - a1) * 10)
                cuts["creak"] = a2
    voiced = np.flatnonzero(F0 > 0)
    v0 = int(voiced[0]) if len(voiced) else n
    if T["hf30"] > BURST_HF and T["db30"] >= -BURST_DB and v0 < n:
        cuts["burst"] = v0
    loud = np.flatnonzero(E >= -35)
    if len(loud) and v0 - int(loud[0]) > LEAD_MS // 10:
        a = int(loud[0])
        o.update(lead_ms=(v0 - a) * 10, lead_db=round(float(E[a:v0].max()), 1))
        if LEAD_DB <= o["lead_db"] <= LEAD_MAX_DB and v0 < n:
            cuts["lead"] = v0
    o["why"] = sorted(cuts)
    o["bad"] = bool(cuts)
    if cuts and max(cuts.values()) * 0.01 <= MAX_CUT_S:
        o["cut_s"] = round(max(cuts.values()) * 0.01, 2)
    return o


def onset_flags(wav, sr: int, text: str) -> dict:
    """Stray-onset flags of a clip (float samples, mono or (n, channels)) of a text (module doc)."""
    cls = text_class(text)
    if cls not in COVERED:
        return {"covered": False, "cls": cls, "bad": False, "why": []}
    return flags_from_tracks(tracks(wav, sr), cls)


def trim_onset(wav, sr: int, cut_s: float):
    """The clip from cut_s - PRE_S on, with a FADE_S linear fade-in (float32)."""
    import numpy as np

    y = np.asarray(wav, dtype=np.float32).reshape(-1)[max(0, int(round((cut_s - PRE_S) * sr))):].copy()
    k = min(len(y), int(round(FADE_S * sr)))
    y[:k] *= np.linspace(0.0, 1.0, k, endpoint=False, dtype=np.float32)
    return y


# ---- clip edges, sibilance, laughter (render_corpus.py) ----------------------------------------------------------


def _mono16(wav, sr: int):
    import numpy as np

    x = np.asarray(wav, dtype=np.float64)
    x = x.mean(axis=1) if x.ndim == 2 else x.reshape(-1)
    return _resample(x, int(sr), SR)


def frame_db(x):
    """dB of the 10 ms frames of x (16 kHz) re its 95th-percentile frame."""
    import numpy as np

    n = len(x) // HOP
    if n < 3:
        raise ValueError("clip too short")
    e = 10 * np.log10((x[:n * HOP].reshape(n, HOP) ** 2).mean(axis=1) + 1e-12)
    return e - float(np.percentile(e, 95))


def _words(words) -> list[tuple[float, float]]:
    """(start, end) of the words with a letter or digit ("5" for "beş"), in order."""
    return [(float(w[1]), float(w[2])) for w in words or [] if re.search(r"[^\W_]", str(w[0]))]


def voiced_lead(x, e, text: str) -> dict:
    """The "vlead" check (edges) on x (16 kHz) with frame_db e: {} when it does not apply or finds nothing."""
    import numpy as np

    if text_class(text) != "voiceless":
        return {}
    aud = np.flatnonzero(e >= -35)
    if not len(aud):
        return {}
    n = min(len(e), int(round(HEAD_S * SR / HOP)))
    loud = (_pitch(x[:n * HOP], n) > 0) & (e[:n] >= -20)
    runs = _runs(loud)
    if not runs:
        return {}
    a, b = runs[0]
    nxt = runs[1][0] if len(runs) > 1 else n
    if (a - int(aud[0]) <= 1 and (b - a) * 0.01 <= VLEAD_MAX_S and float(e[a:b].max()) >= VLEAD_DB
            and (nxt - b) * 0.01 >= VLEAD_GAP_S):
        return {"head": "vlead", "vlead_ms": (b - a) * 10, "vlead_gap_ms": (nxt - b) * 10}
    return {}


def edges(wav, sr: int, words, text: str = "") -> dict:
    """Speech end, stray tail, cut point, sound before the first word, a voiced burst before a voiceless start,
    pauses (module doc); {"covered": False} without a timed word."""
    import numpy as np

    ws = _words(words)
    if not ws:
        return {"covered": False}
    x = _mono16(wav, sr)
    e = frame_db(x)
    n, dur = len(e), len(x) / SR
    quiet = e < EDGE_SIL_DB
    gap = int(round(TAIL_GAP_S / 0.01))
    fr = lambda t: min(max(int(round(t / 0.01)), 0), n)  # noqa: E731
    t0, t1, last0 = ws[0][0], ws[-1][1], ws[-1][0]
    k = fr(t1)
    end, run, i = k, 0, k
    while i < n:  # through the word's own decay until a quiet run of TAIL_GAP_S
        run = run + 1 if quiet[i] else 0
        if not quiet[i]:
            end = i + 1
        if run >= gap:
            break
        i += 1
    after = e[end + gap:] if end + gap < n else e[:0]
    o = {"covered": True, "t0": round(t0, 3), "t1": round(t1, 3), "speech_end_s": round(end * 0.01, 2), "tail": ""}
    loud = np.flatnonzero(after >= TAIL_ART_DB)
    if len(loud):
        a = end + gap + int(loud[0])
        b = end + gap + int(np.flatnonzero(after > EDGE_SIL_DB)[-1]) + 1
        o.update(tail_db=round(float(after.max()), 1), tail_at_s=round(a * 0.01, 2), tail_ms=(b - a) * 10)
        o["tail"] = "stray" if (b - a) * 0.01 <= STRAY_MAX_S else "long"
    else:  # Whisper may have put the stray into the last word: a quiet run of TAIL_GAP_S inside it, little after
        seg = quiet[fr(last0):end]
        runs = [(a, b) for a, b in _runs(seg, fr(last0)) if b - a >= gap and b < end]
        if runs and (end - runs[-1][1]) * 0.01 <= STRAY_MAX_S and runs[-1][0] > fr(last0) + 5:
            o.update(tail="inside", tail_at_s=round(runs[-1][1] * 0.01, 2), tail_ms=(end - runs[-1][1]) * 10,
                     tail_db=round(float(e[runs[-1][1]:end].max()), 1))
    cut = end * 0.01 + TAIL_PAD_S
    if o["tail"] in ("", "stray") and cut < dur - 0.005:
        o["cut_s"] = round(cut, 3)
    on = np.flatnonzero(~quiet)
    o["on_s"] = round(float(on[0]) * 0.01, 2) if len(on) else None
    o["head_s"] = round(float((~quiet[:fr(t0)]).sum()) * 0.01, 2)
    o["head"] = "sound" if o["head_s"] > HEAD_MAX_S else ""
    o.update(voiced_lead(x, e, text))
    p = int(round(PAUSE_S / 0.01))
    o["pauses"] = [[round(a * 0.01, 2), round(b * 0.01, 2)] for a, b in _runs(quiet[fr(t0):fr(t1)], fr(t0))
                   if b - a >= p]
    return o


def trim_tail(wav, sr: int, cut_s: float):
    """The clip up to cut_s with a TAIL_FADE_S linear fade-out (float32)."""
    import numpy as np

    y = np.asarray(wav, dtype=np.float32).reshape(-1)[:max(1, int(round(cut_s * sr)))].copy()
    k = min(len(y), int(round(TAIL_FADE_S * sr)))
    y[len(y) - k:] *= np.linspace(1.0, 0.0, k, endpoint=False, dtype=np.float32)
    return y


def _spectra(x):
    """Power spectra of 32 ms Hann frames, 10 ms hop, of x (16 kHz); their frequencies."""
    import numpy as np

    n_fft = 512
    if len(x) < n_fft:
        x = np.concatenate([x, np.zeros(n_fft - len(x))])
    n = 1 + (len(x) - n_fft) // HOP
    idx = np.arange(n)[:, None] * HOP + np.arange(n_fft)[None, :]
    return np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1)) ** 2, np.fft.rfftfreq(n_fft, 1 / SR)


def sibilance(wav, sr: int) -> dict:
    """sib_db: how loud the loudest sustained sibilant is re the clip's loud frames (module doc); sib_ms its run."""
    import numpy as np

    P, f = _spectra(_mono16(wav, sr))
    db = lambda m: 10 * np.log10(P[:, m].sum(axis=1) + 1e-12)  # noqa: E731
    tot, lo, hi = db(f >= 0), db(f < 1500), db(f >= SIB_LO)
    rel = hi - float(np.percentile(tot, 95))
    best, ms = None, 0
    for a, b in _runs((hi - lo >= SIB_FRIC_DB) & (rel > -40)):
        if b - a >= SIB_FRAMES:
            v = float(np.sort(rel[a:b])[-SIB_FRAMES])
            if best is None or v > best:
                best, ms = v, (b - a) * 10
    return {"sib_db": None if best is None else round(best, 1), "sib_ms": ms}


def laugh(wav, sr: int, span: tuple[float, float]) -> dict:
    """Laughter measures of a span (s) of a clip (module doc); {"laugh": False, "why": ...} when it is too short."""
    import numpy as np

    x = _mono16(wav, sr)
    e_all = frame_db(x)
    a, b = max(0, int(span[0] / 0.01)), min(len(e_all), int(math.ceil(span[1] / 0.01)))
    if b - a < 25:
        return {"laugh": False, "why": "span under 250 ms", "span_s": round((b - a) * 0.01, 2)}
    e = e_all[a:b]
    es = np.convolve(e, np.ones(3) / 3, mode="same")
    f0 = _pitch(x[a * HOP:b * HOP], b - a)
    peaks = [i for i in range(1, len(es) - 1) if es[i] >= es[i - 1] and es[i] > es[i + 1] and es[i] >= -30]
    bursts, last = [], None
    for i in peaks:  # a new burst once the energy dipped LAUGH_DIP_DB under both maxima since the last one
        if last is None or float(es[last:i + 1].min()) <= min(es[last], es[i]) - LAUGH_DIP_DB:
            bursts.append(i)
            last = i
        elif es[i] > es[last]:
            bursts[-1] = last = i
    v = (f0 > 0) & (e > EDGE_SIL_DB)
    st = 12 * np.log2(f0[v] / 100.0) if v.sum() >= 5 else np.zeros(0)
    span_s = (b - a) * 0.01
    o = {"span_s": round(span_s, 2), "bursts": len(bursts), "burst_rate": round(len(bursts) / span_s, 2),
         "voiced": round(float(v.sum()) / max(int((e > EDGE_SIL_DB).sum()), 1), 2),
         "f0_sd": round(float(np.std(st)), 2) if len(st) else None}
    o["laugh"] = bool(o["bursts"] >= LAUGH_BURSTS and 2.5 <= o["burst_rate"] <= 9 and 0.15 <= o["voiced"] <= 0.9
                      and (o["f0_sd"] or 0.0) >= LAUGH_F0_SD)
    o["score"] = round(min(o["bursts"], 8) / 8 + min(o["f0_sd"] or 0.0, 6.0) / 6 + (1.0 if o["laugh"] else 0.0), 3)
    return o
