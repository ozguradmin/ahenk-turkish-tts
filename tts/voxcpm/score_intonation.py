#!/usr/bin/env python3
"""Question intonation: does the synthesized speech put the pitch peak where a Turkish question needs it?

Expected shapes (`pattern` in benchmarks/tts/sets/tr_soru.jsonl; `peak` = 0-based word of the spoken form):
  peak_fall   yes/no question with the particle mI (mi_soru), and "..., değil mi?" tags (degil_mi): the peak is on
              the syllable right before mI, then the pitch falls (most such questions end low)
  wh_peak     wh-question (ne_soru: ne, kim, nerede, nasıl, ...): the peak is on the wh-word, lower after it
  final_rise  question with neither mI nor a wh-word (yukselen): the pitch rises at the end
  fall        the matched declarative (duz_eslesik; pair_id links it to its questions): ends low, no final rise

Per item, pitch in semitones (st) re the median of the utterance (clause, in a passage):
  F0       librosa.pyin at 16 kHz, 10 ms hop, in two passes: a wide first pass gives the speaker's range, the second
           runs in [0.72 * q35, 2.2 * q65] of it. Threshold prior Beta(1, 4), not pyin's Beta(2, 18): that one leaves
           most frames of VoxCPM's output unvoiced (human speech changes little). Frames at the floor (creak) or in
           silence dropped, octave jumps between neighbouring frames folded back (fold_octaves), 5-frame median, what
           still jumps (> 9 st off a 41-frame running median) removed
  timing   faster-whisper word timestamps (row["words"], written by synth_eval's judge for this set, or --whisper
           here), matched to the spoken form; else the speech time (RMS > -35 dB re max, pauses >= 120 ms left out)
           shared out by syllables (Turkish: one per vowel), phrase breaks (, ; : . ? !) at the pauses that keep the
           speech rate of every stretch closest to the mean (_align)
  peak_st  90th percentile in the target window: peak_fall / fall: the target word's last two syllables and the next
           syllable (mI); wh_peak: the word and the next syllable; final_rise: the last word
  peak_at  peak_fall: where the F0 maximum lies, from one syllable before the target ones through mI's first, in
           syllables re the start of mI: -1..0 the syllable before mI, -2..-1 the one before that, 0..1 mI's first
  lead_st  wh_peak: peak_st - the 90th percentile over the words before the wh-word (None when it comes first)
  end_st   median of the last 150 ms of voiced speech; drop_st = peak_st - end_st; slope_st_s: line fit, last 300 ms
  pass     peak_fall: peak_st >= 2, drop_st >= 3 and the maximum on the target's last two syllables or the first
           third of mI (peak delay; not on mI itself, not earlier); wh_peak: peak_st >= 2, drop_st >= 3, range_st >=
           WH_RANGE_ST (not flat) and lead_st >= WH_LEAD_ST (the wh-word is the peak, not a word before it: f022 pilot
           "Çocuklar yeni okullarına nasıl uyum sağladı?", heard as no question, peak 2.4 st but 3.9 st under
           "Çocuklar"; from 2026-10-12, earlier summaries used peak and drop only);
           final_rise: end_st >= 1.5 and slope >= 5 st/s; fall: end_st <= 0 and slope < 5 st/s. None (not scored)
           when a measure the pattern needs is missing (too few voiced frames). Provisional thresholds, not yet
           calibrated on human recordings.
Summary per set and set/category: pass rate and mean measures of the scored items (unscored: how many were not);
per set also questions_pass_rate, auc_peak (peak_st of mi_soru items vs the declaratives: 0.5 = no difference, 1 =
every question peaks higher) and pair_acc (matched pairs whose question has the higher peak_st).

Other sets (--sets trzor,heldout) take their items from the text: each clause ending in "?" is classified as above
(mI particle, else wh-word, else final_rise; a bare "Ne?" is not scored); each ending in "." is a declarative control
"duz" (target: its second-to-last word). Long passages without word timestamps get the syllable split only: a rough
smoke test.

    python tts/voxcpm/score_intonation.py voxcpm-eval/step_0000600 [DIR ...] [--sets soru] [--dry-run]

adds row["intonation"] (list of items) to DIR/rows.jsonl and summary["intonation"] to DIR/summary.json (each replaced
atomically; WAVs from DIR/wav/<set>/<id>.wav) and prints the summary; --dry-run writes nothing. summary
["intonation_error"] names rows that could not be scored. synth_eval.py runs it on the soru set after synthesis.
Importing this module needs no numpy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from score_utmos import write_atomic  # noqa: E402  (script dir on sys.path; no torch)
from trnorm import normalize_for_eval, tr_lower  # noqa: E402

SET_FILE = REPO / "benchmarks" / "tts" / "sets" / "tr_soru.jsonl"
CATEGORIES = ("mi_soru", "ne_soru", "degil_mi", "yukselen", "duz_eslesik")
PATTERNS = {"mi_soru": "peak_fall", "degil_mi": "peak_fall", "ne_soru": "wh_peak", "yukselen": "final_rise",
            "duz_eslesik": "fall", "duz": "fall"}
QUESTION_PATTERNS = ("peak_fall", "wh_peak", "final_rise")
SR, HOP, FRAME = 16000, 160, 1024
HOP_S = HOP / SR
SIL_DB, PAUSE_FRAMES = -35.0, 12  # silence re the loudest frame; shortest pause a phrase break snaps to
PEAK_MIN_ST, DROP_MIN_ST = 2.0, 3.0
PEAK_LATE = 1 / 3  # peak delay: the latest a peak_fall maximum may lie, as a fraction of mI's first syllable
RISE_END_ST, RISE_SLOPE = 1.5, 5.0
END_S, SLOPE_S = 0.15, 0.30
OCTAVE_JUMP_ST = 9.0  # off the 41-frame running median: dropped
PASS2_HI = 2.2  # second pyin pass up to PASS2_HI * q65 of the first: ~ +13.6 st, room for a large real peak
PYIN_BETA = (1.0, 4.0)
WH_LEAD_ST, WH_RANGE_ST = -2.0, 5.0
MEASURES = ("peak_st", "drop_st", "end_st", "slope_st_s", "peak_at")  # summary means (lead_st: per item)

VOWELS = set("aeıioöuüâîûAEIİOÖUÜÂÎÛ")
TRAIL = "\"'”’)»"
_PERS = r"y?[ıiuü][mz]|s[ıiuü]n([ıiuü]z)?"  # -(y)Im -(y)Iz -sIn -sInIz
MI = re.compile(rf"^(mı|mi|mu|mü)({_PERS}|d[ıiuü]r(lar|ler)?|yd[ıiuü](m|n|k|n[ıiuü]z|lar|ler)?"
                rf"|ym[ıiuü]ş({_PERS}|lar|ler)?)?$")  # miyim misiniz midir(ler) miydin miydiniz miymişsin
WH = re.compile(r"^(ne|neden|niye|niçin|ne(yi|ye|yle|den|de|yin|si|dir|ydi|ymiş|ler\w*)|nasıl\w*|nere\w*"
                r"|kim(i|e|in|inle|den|de|le|dir|di|miş|sin|siniz|iz|ler\w*)?|hangi\w*|kaç(a|ta|tan|ı|ın|ıncı|ar)?)$")


def syllables(word: str) -> int:
    return sum(ch in VOWELS for ch in word)


def core(word: str) -> str:
    """Lowercase letters of a token: "Ankara'ya?" -> "ankaraya"."""
    return re.sub(r"[\W\d_]", "", tr_lower(word))


def terminal(word: str) -> str:
    """Clause-final punctuation of a token ("?" for "mı?"), "" if none."""
    m = re.search(r"[.?!…]+$", word.rstrip(TRAIL))
    return m.group(0) if m else ""


def load_set(path: Path = SET_FILE) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def set_item(row: dict, words: list[str]) -> dict:
    """The one item of a tr_soru row; ValueError when its peak index does not point at peak_word."""
    k = row["peak"]
    if not 0 <= k < len(words) or core(words[k]) != core(row["peak_word"]):
        raise ValueError(f"peak {k} is not {row['peak_word']!r} in {' '.join(words)!r}")
    return {"category": row["category"], "pattern": row["pattern"], "first": 0, "last": len(words) - 1, "target": k,
            "pair_id": row.get("pair_id")}


def classify(words: list[str], idx: list[int], end: str) -> dict | None:
    """Item of one clause (word indices idx, all with a vowel; end: its final punctuation), None unless "?" or "."."""
    cores = [core(words[j]) for j in idx]
    item = {"first": idx[0], "last": idx[-1]}
    if "?" in end:
        mis = [k for k, c in enumerate(cores) if k > 0 and MI.match(c)]
        whs = [k for k, c in enumerate(cores) if WH.match(c)]
        if mis:
            k = mis[-1] - 1
            cat, pat, t = ("degil_mi" if cores[k] == "değil" else "mi_soru"), "peak_fall", idx[k]
        elif whs:
            cat, pat, t = "ne_soru", "wh_peak", idx[whs[0]]
            if len(idx) == 1:  # bare "Ne?", "Neden?": nothing after the wh-word, often said rising
                item["unscored"] = "one-word wh-question"
        else:
            cat, pat, t = "yukselen", "final_rise", idx[-1]
    elif "." in end:
        cat, pat, t = "duz", "fall", idx[-2] if len(idx) > 1 else idx[-1]
    else:
        return None
    return {**item, "category": cat, "pattern": pat, "target": t}


def infer_items(words: list[str]) -> list[dict]:
    """Items of a text from its clauses (split after . ? ! …); see classify."""
    items, start = [], 0
    for i, w in enumerate(words):
        if terminal(w) or i == len(words) - 1:
            idx = [j for j in range(start, i + 1) if syllables(words[j])]
            if idx and (it := classify(words, idx, terminal(w))):
                items.append(it)
            start = i + 1
    return items


NEEDS = {"peak_fall": ("peak_st", "drop_st", "peak_at"), "wh_peak": ("peak_st", "drop_st"),
         "final_rise": ("end_st", "slope_st_s"), "fall": ("end_st", "slope_st_s")}


def judge(pattern: str, m: dict, earliest: float = -2.0) -> bool | None:
    """Pass/fail of one item's measures against its expected pattern; None (not scorable) when one it needs is missing.
    peak_fall: earliest = where the target syllables start, in peak_at's units (-2, or -1 for a one-syllable word)."""
    if pattern not in NEEDS:
        raise ValueError(f"unknown pattern {pattern!r}")
    if not all(m.get(k) is not None and math.isfinite(m[k]) for k in NEEDS[pattern]):
        return None
    if pattern == "peak_fall":
        return m["peak_st"] >= PEAK_MIN_ST and m["drop_st"] >= DROP_MIN_ST and earliest <= m["peak_at"] <= PEAK_LATE
    if pattern == "wh_peak":
        lead, rng = m.get("lead_st"), m.get("range_st")
        return (m["peak_st"] >= PEAK_MIN_ST and m["drop_st"] >= DROP_MIN_ST
                and (lead is None or not math.isfinite(lead) or lead >= WH_LEAD_ST)
                and (rng is None or not math.isfinite(rng) or rng >= WH_RANGE_ST))
    if pattern == "final_rise":
        return m["end_st"] >= RISE_END_ST and m["slope_st_s"] >= RISE_SLOPE
    return m["end_st"] <= 0.0 and m["slope_st_s"] < RISE_SLOPE


# ---- audio ------------------------------------------------------------------------------------------------------


def load_audio(path: str | Path):
    """Mono float32 at 16 kHz."""
    import librosa
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = np.ascontiguousarray(y.mean(axis=1))
    return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y


def fold_octaves(s, idx):
    """pyin's octave jumps in the voiced contour s (st) at frames idx, cut into pieces at jumps > 6 st between frames
    <= 3 apart and at longer gaps: of two pieces an octave-sized jump (9-16 st) apart within 3 frames, the one farther
    from the utterance median moves an octave if that brings it closer. Only a jump is evidence: a piece that is merely
    far from the median (a high peak or final rise after a voiceless consonant) stays; folding those too doubled the
    frames > 6 st off Praat's pitch on Antalia ground truth (5 -> 11 %) and VoxCPM output (15 -> 22 %)."""
    import numpy as np

    ref, out = float(np.median(s)), np.array(s, dtype=float)
    pieces = np.split(np.arange(len(s)), np.flatnonzero((np.abs(np.diff(s)) > 6.0) | (np.diff(idx) > 3)) + 1)
    med = [float(np.median(out[p])) - ref for p in pieces]
    for a in range(len(pieces) - 1):
        b = a + 1
        if idx[pieces[b][0]] - idx[pieces[a][-1]] <= 3 and 9.0 <= abs(out[pieces[b][0]] - out[pieces[a][-1]]) <= 16.0:
            k = a if abs(med[a]) > abs(med[b]) else b
            if abs(med[k] - 12.0 * math.copysign(1, med[k])) < abs(med[k]):
                out[pieces[k]] -= 12.0 * math.copysign(1, med[k])
                med[k] -= 12.0 * math.copysign(1, med[k])
    return out


def f0_track(y):
    """(st, silent): per 10 ms frame, F0 in st re 100 Hz (NaN unvoiced) and the silence mask (quiet and unvoiced)."""
    import librosa
    import numpy as np
    from scipy.ndimage import median_filter

    f0, vf, _ = librosa.pyin(y, fmin=55.0, fmax=700.0, sr=SR, frame_length=FRAME, hop_length=2 * HOP, resolution=0.25,
                             beta_parameters=PYIN_BETA)
    hz = f0[vf & np.isfinite(f0)]
    if len(hz) < 10:
        raise ValueError("no voiced speech found")
    q35, q65 = np.percentile(hz, [35, 65])
    lo, hi = max(40.0, 0.72 * q35), min(1000.0, PASS2_HI * q65)
    f0, vf, _ = librosa.pyin(y, fmin=lo, fmax=hi, sr=SR, frame_length=FRAME, hop_length=HOP, beta_parameters=PYIN_BETA)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, vf, rms = f0[:n], vf[:n], rms[:n]
    db = 20 * np.log10(np.maximum(rms, 1e-10) / max(float(rms.max()), 1e-10))
    v = vf & np.isfinite(f0) & (f0 > lo * 1.03) & (db > SIL_DB)  # floor frames: creak or tracking failures
    st = np.full(n, np.nan)
    st[v] = 12 * np.log2(f0[v] / 100.0)
    idx = np.flatnonzero(v)
    if len(idx) > 5:
        s = fold_octaves(st[idx], idx)
        st[idx], idx = s, idx[np.isfinite(s)]
    if len(idx) > 5:
        st[idx] = median_filter(st[idx], size=5, mode="nearest")
        jump = np.abs(st[idx] - median_filter(st[idx], size=41, mode="nearest")) > OCTAVE_JUMP_ST
        st[idx[jump]] = np.nan
    return st, (db < SIL_DB) & ~np.isfinite(st)


# ---- word timing ------------------------------------------------------------------------------------------------


def _spread(t0: float, t1: float, sizes: list[int]) -> list[tuple[float, float]]:
    """[t0, t1] cut into consecutive spans proportional to sizes (all-zero sizes: equal spans)."""
    sizes = sizes if sum(sizes) else [1] * len(sizes)
    tot, acc, out = sum(sizes), 0, []
    for s in sizes:
        out.append((t0 + (t1 - t0) * acc / tot, t0 + (t1 - t0) * (acc + s) / tot))
        acc += s
    return out


def _pauses(silent, a: int, b: int) -> list[tuple[int, int]]:
    """Silent runs of >= PAUSE_FRAMES frames inside frames [a, b)."""
    out, i = [], a
    while i < b:
        if silent[i]:
            j = i
            while j < b and silent[j]:
                j += 1
            if j - i >= PAUSE_FRAMES:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _align(syl: list[int], pauses: list[tuple[int, int]], a: int, b: int) -> list[int | None]:
    """Pause (index) for each phrase break (after phrase i of len(syl)), or None: the monotone assignment minimising,
    over the stretches between the matched pauses, n * ln(speech frames / (n * rate))^2 (n: the stretch's syllables;
    rate: speech frames per syllable, pauses left out), + 1 per break without a pause, + 2 per pause inside a phrase
    (less if it is shorter than 0.5 s)."""
    import numpy as np

    lens = np.array([e - s for s, e in pauses], dtype=float)
    plen = np.r_[0.0, np.cumsum(lens)]
    pcost = np.r_[0.0, np.cumsum(2.0 * np.minimum(lens * HOP_S / 0.5, 1.0))]
    csyl = np.r_[0, np.cumsum(syl)]
    rate = max(b - a - plen[-1], 1.0) / max(csyl[-1], 1)
    P, M = len(syl), len(pauses)

    def cost(i0: int, j0: int, i1: int, j1: int) -> float:  # phrases i0 + 1 .. i1, from pause j0 (-1: a) to j1 (M: b)
        d = (pauses[j1][0] if j1 < M else b) - (pauses[j0][1] if j0 >= 0 else a) - (plen[j1] - plen[j0 + 1])
        n = max(float(csyl[i1 + 1] - csyl[i0 + 1]), 1.0)
        return n * math.log(max(d, 1.0) / (n * rate)) ** 2 + (i1 - i0 - 1) + pcost[j1] - pcost[j0 + 1]

    best: dict[tuple[int, int], tuple[float, tuple | None]] = {(-1, -1): (0.0, None)}  # (break, pause): cost, previous
    for i in range(P - 1):
        for j in range(M):
            c = [(v + cost(i0, j0, i, j), (i0, j0)) for (i0, j0), (v, _) in best.items() if i0 < i and j0 < j]
            if c:
                best[(i, j)] = min(c)
    node = min((v + cost(i0, j0, P - 1, M), (i0, j0)) for (i0, j0), (v, _) in best.items())[1]
    out: list[int | None] = [None] * (P - 1)
    while node != (-1, -1):
        out[node[0]] = node[1]
        node = best[node][1]
    return out


def syllable_times(words: list[str], silent) -> list[tuple[float, float]]:
    """Word spans (s) without word timestamps: the speech time (pauses >= PAUSE_FRAMES left out) shared out by
    syllables, phrase breaks (after , ; : . ? !) at the pauses _align picks, the other breaks by syllables between."""
    import numpy as np

    on = np.flatnonzero(~silent)
    if not len(on):
        raise ValueError("no speech found")
    a, b = int(on[0]), int(on[-1]) + 1
    pauses = _pauses(silent, a, b)
    phrases, cur = [], []
    for k, w in enumerate(words):
        cur.append(k)
        if re.search(r"[,;:.?!…]", w.rstrip(TRAIL)[-1:]) and k < len(words) - 1:
            phrases.append(cur)
            cur = []
    phrases.append(cur)
    syl = [sum(syllables(words[k]) for k in p) for p in phrases]
    keep = np.ones(b - a, bool)
    for s, e in pauses:
        keep[s - a:e - a] = False
    sp = a + np.flatnonzero(keep)  # speech frames; position x on the speech time = x speech frames before it
    xs = [0.0] + [None if j is None else float(np.searchsorted(sp, pauses[j][0])) for j in _align(syl, pauses, a, b)]
    xs.append(float(len(sp)))
    k = 1
    while k < len(xs) - 1:  # breaks without a pause: by syllables between the placed neighbours
        if xs[k] is None:
            j = next(i for i in range(k, len(xs)) if xs[i] is not None)
            xs[k:j] = [e for _, e in _spread(xs[k - 1], xs[j], syl[k - 1:j])[:-1]]
            k = j
        k += 1

    def frame(x: float, end: bool) -> float:  # of position x: where a span starts, or where one ends (before a pause)
        i = min(max(math.ceil(x) - 1 if end else math.floor(x), 0), len(sp) - 1)
        return float(sp[i]) + x - i

    return [(frame(u, False) * HOP_S, frame(v, True) * HOP_S) for p, x0, x1 in zip(phrases, xs, xs[1:])
            for u, v in _spread(x0, x1, [syllables(words[i]) for i in p])]


def whisper_times(words: list[str], wwords: list) -> list[tuple[float, float]] | None:
    """Word spans from faster-whisper words [[text, start, end], ...]: each is cut into its normalize_for_eval tokens
    (by syllables) and matched to the words (difflib); unmatched words are spread by syllables between their matched
    neighbours. None when fewer than half the words with a vowel match."""
    toks = []
    for w, t0, t1 in wwords:
        parts = normalize_for_eval(str(w)).split()
        toks += [(p, *span) for p, span in zip(parts, _spread(float(t0), float(t1), [syllables(p) for p in parts]))]
    if not toks:
        return None
    keys = [normalize_for_eval(w).replace(" ", "") for w in words]
    times: list = [None] * len(words)
    sm = SequenceMatcher(None, keys, [t[0] for t in toks], autojunk=False)
    for i, j, size in sm.get_matching_blocks():
        for k in range(size):
            if keys[i + k]:
                times[i + k] = toks[j + k][1:]
    voiced = [i for i, w in enumerate(words) if syllables(w)]
    if sum(times[i] is not None for i in voiced) * 2 < len(voiced):
        return None
    i = 0
    while i < len(words):  # fill each run of unmatched words between the neighbours' times
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(words) and times[j] is None:
            j += 1
        t0 = times[i - 1][1] if i > 0 else toks[0][1]
        t1 = times[j][0] if j < len(words) else toks[-1][2]
        times[i:j] = _spread(t0, max(t1, t0), [syllables(w) for w in words[i:j]])
        i = j
    return times


def whisper_words(model, y) -> list[list]:
    """[[word, start, end], ...] of a 16 kHz clip (faster-whisper, language tr, beam 5, word timestamps)."""
    segs, _ = model.transcribe(y, language="tr", beam_size=5, word_timestamps=True)
    return [[w.word.strip(), round(w.start, 3), round(w.end, 3)] for s in segs for w in (s.words or [])]


# ---- measures ---------------------------------------------------------------------------------------------------


def _r(x: float) -> float | None:
    return round(float(x), 2) if x is not None and math.isfinite(x) else None


def measure(st, times: list[tuple[float, float]], words: list[str], item: dict) -> dict:
    """Measures of one item (see the module doc) on the frames of its clause."""
    import numpy as np

    first, last, k = item["first"], item["last"], item["target"]
    t0, t1 = times[first][0], times[last][1]
    fa, fb = max(int(t0 / HOP_S), 0), min(int(math.ceil(t1 / HOP_S)) + 1, len(st))
    seg, tt = st[fa:fb], np.arange(fa, fb) * HOP_S
    v = np.isfinite(seg)
    out = {"category": item["category"], "pattern": item["pattern"], "word": core(words[k])}
    if item.get("pair_id"):
        out["pair_id"] = item["pair_id"]
    if v.sum() < 15:
        return {**out, **dict.fromkeys(MEASURES + ("lead_st", "range_st")), "pass": None,
                "note": "not scored: fewer than 15 voiced frames"}
    sv, tv = seg[v], tt[v]
    rel = sv - np.median(sv)
    sylspan = lambda i: _spread(*times[i], [1] * max(syllables(words[i]), 1))  # noqa: E731
    syl = sylspan(k)[-2:]  # the target's last two syllables (its only one)
    nxt = next((i for i in range(k + 1, last + 1) if syllables(words[i])), None)
    mi = sylspan(nxt)[0] if nxt is not None else (t1, t1)  # the next syllable (mI)
    pat = item["pattern"]
    win = {"peak_fall": (syl[0][0], mi[1]), "fall": (syl[0][0], mi[1]), "wh_peak": (times[k][0], mi[1]),
           "final_rise": (times[k][0], t1)}[pat]
    inw = (tv >= win[0]) & (tv <= win[1])
    peak = float(np.percentile(rel[inw], 90)) if inw.sum() >= 3 else math.nan
    end = float(np.median(rel[tv >= tv[-1] - END_S]))
    tail = tv >= tv[-1] - SLOPE_S
    slope = float(np.polyfit(tv[tail], rel[tail], 1)[0]) if tail.sum() >= 5 else math.nan
    at, lead = math.nan, math.nan
    if pat == "wh_peak":  # the words before the wh-word: their own high point
        pre = (tv >= t0) & (tv < times[k][0])
        lead = peak - float(np.percentile(rel[pre], 90)) if pre.sum() >= 3 else math.nan
    if pat == "peak_fall":  # where the F0 maximum lies, from one syllable before the target ones through mI's first
        prv = next((i for i in range(k - 1, item["first"] - 1, -1) if syllables(words[i])), None)
        guard = sylspan(k)[-3:-2] or (sylspan(prv)[-1:] if prv is not None else [])
        spans = guard + syl + [mi]
        inw = (tv >= spans[0][0]) & (tv <= spans[-1][1])
        if inw.sum() >= 3:
            tpk = float(tv[inw][np.argmax(rel[inw])])
            j = max(i for i, (s, _) in enumerate(spans) if s <= tpk or i == 0)
            s, e = spans[j]
            at = j - len(spans) + 1 + (min(max((tpk - s) / (e - s), 0.0), 1.0) if e > s else 0.0)
    m = {"peak_st": peak, "drop_st": peak - end, "end_st": end, "slope_st_s": slope, "peak_at": at, "lead_st": lead,
         "range_st": float(np.percentile(rel, 95) - np.percentile(rel, 5))}
    out.update({key: _r(x) for key, x in m.items()})
    out["pass"] = None if item.get("unscored") else judge(pat, m, -float(len(syl)))
    if item.get("unscored"):
        out["note"] = f"not scored: {item['unscored']}"
    elif out["pass"] is None:
        out["note"] = f"not scored: too few voiced frames for {', '.join(x for x in NEEDS[pat] if out[x] is None)}"
    return out


def score_row(row: dict, path: str | Path, whisper=None) -> list[dict]:
    """Items of one row (tr_soru annotation for set soru, else from its text), each with its measures."""
    words = (row.get("model_text") or row["text"]).split()
    items = [set_item(row, words)] if row["set"] == "soru" else infer_items(words)
    if not items:
        return []
    y = load_audio(path)
    st, silent = f0_track(y)
    if row.get("words") is None and whisper is not None:
        row["words"] = whisper_words(whisper, y)
    times, how = None, "syllables"
    if row.get("words"):
        times = whisper_times(words, row["words"])
        how = "whisper" if times else how
    times = times or syllable_times(words, silent)
    out = []
    for it in items:
        m = measure(st, times, words, it)
        m["timing"] = how
        if row["set"] != "soru":
            m["clause"] = " ".join(words[it["first"]:it["last"] + 1])
        out.append(m)
    return out


def score_rows(rows: list[dict], wav_of: Callable[[dict], str | Path | None], sets=("soru",), label: str = "",
               whisper=None) -> dict[str, str]:
    """row["intonation"] for every row of `sets` with a WAV (wav_of(row) None: left as it is); one bad clip costs only
    its own row (row["intonation_error"]). Returns {"set/id": error}."""
    todo = [(r, p) for r in rows if r["set"] in sets and (p := wav_of(r)) is not None]
    t0, errors = time.time(), {}
    for i, (r, p) in enumerate(todo):
        try:
            r["intonation"] = score_row(r, p, whisper)
            r.pop("intonation_error", None)
        except Exception as e:
            r.pop("intonation", None)
            r["intonation_error"] = errors[f"{r['set']}/{r['id']}"] = f"{type(e).__name__}: {e}"
            print(f"[inton] {label}: {r['set']}/{r['id']} not scored: {r['intonation_error']}", flush=True)
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"[inton] {label}: {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    return errors


def errors_note(errors: dict[str, str]) -> str:
    k, v = next(iter(errors.items()))
    return f"{len(errors)} rows not scored, left out of the summary (row intonation_error); first {k}: {v}"


# ---- summary ----------------------------------------------------------------------------------------------------


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(a positive scores above a negative), ties half (Mann-Whitney); None without both."""
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def intonation_summary(rows: list[dict]) -> dict[str, dict]:
    """{set and set/category: items, unscored (pass None), passed, pass_rate and mean measures of the scored ones};
    per set also questions_pass_rate, auc_peak (mi_soru vs declaratives), pair_acc over pair_id pairs, and the timing
    source counts."""
    groups = defaultdict(list)
    for r in rows:
        for it in r.get("intonation") or []:
            for key in (r["set"], f"{r['set']}/{it['category']}"):
                groups[key].append(it)
    out = {}
    for key, all_its in sorted(groups.items()):
        its = [it for it in all_its if it["pass"] is not None]  # scored
        g = {"items": len(all_its), "unscored": len(all_its) - len(its), "passed": sum(bool(it["pass"]) for it in its)}
        g["pass_rate"] = round(g["passed"] / len(its), 3) if its else None
        g.update({m: _mean([it.get(m) for it in its]) for m in MEASURES})
        if "/" not in key:
            q = [it for it in its if it["pattern"] in QUESTION_PATTERNS]
            g["questions_pass_rate"] = round(sum(bool(it["pass"]) for it in q) / len(q), 3) if q else None
            peaks = [(it["category"], it["peak_st"]) for it in its if it["peak_st"] is not None]
            a = auc([p for c, p in peaks if c == "mi_soru"], [p for c, p in peaks if c in ("duz_eslesik", "duz")])
            g["auc_peak"] = round(a, 3) if a is not None else None
            pairs = defaultdict(dict)
            for it in its:
                if it.get("pair_id") and it["category"] in ("mi_soru", "duz_eslesik") and it["peak_st"] is not None:
                    pairs[it["pair_id"]][it["category"]] = it["peak_st"]
            won = [p["mi_soru"] > p["duz_eslesik"] for p in pairs.values() if len(p) == 2]
            g["pairs"], g["pair_acc"] = len(won), round(sum(won) / len(won), 3) if won else None
            g["timing"] = dict(sorted(Counter(it.get("timing", "?") for it in all_its).items()))
        out[key] = g
    return out


def print_summary(name: str, summ: dict[str, dict]) -> None:
    f = lambda x, fmt="+.1f": "-" if x is None else format(x, fmt)  # noqa: E731
    print(f"[inton] {name}")
    for k, v in summ.items():
        un = v.get("unscored", 0)
        line = (f"  {'  ' if '/' in k else ''}{k:<22} pass {f(v['pass_rate'], '.3f')} ({v['passed']}/{v['items'] - un}"
                f"{f', {un} unscored' if un else ''})")
        if "/" in k:
            line += (f"  peak {f(v['peak_st'])}  drop {f(v['drop_st'])}  end {f(v['end_st'])} st  "
                     f"slope {f(v['slope_st_s'], '+.0f')} st/s" + (f"  peak_at {f(v['peak_at'], '+.2f')}"
                                                                   if v.get("peak_at") is not None else ""))
        else:
            line += (f"  questions {f(v['questions_pass_rate'], '.3f')}  auc_peak {f(v['auc_peak'], '.3f')}  "
                     f"pairs {f(v['pair_acc'], '.3f')} of {v['pairs']}  timing {v['timing']}")
        print(line)


# ---- CLI --------------------------------------------------------------------------------------------------------


def _rows(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def _wav(d: Path, r: dict) -> Path | None:
    p = d / "wav" / r["set"] / f"{r['id']}.wav"
    return p if p.exists() else None


def load_whisper(name: str):
    """faster-whisper model for word timestamps: None for "none", or for "auto" without faster_whisper installed."""
    if name == "none":
        return None
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        if name == "auto":
            return None
        raise
    return WhisperModel("large-v3-turbo" if name == "auto" else name, device="auto", compute_type="default")


def rescore(d: Path, sets: tuple[str, ...], whisper=None, dry_run: bool = False) -> dict[str, dict]:
    rows = _rows(d)
    if not any(r["set"] in sets and _wav(d, r) for r in rows):
        raise FileNotFoundError(f"{d}: no WAV of the sets {','.join(sets)} under {d / 'wav'}; nothing changed")
    errors = score_rows(rows, lambda r: _wav(d, r), sets, d.name, whisper)
    summ = intonation_summary(rows)
    if dry_run:
        return summ
    write_atomic(d / "rows.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    sp = d / "summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    summary["intonation"] = summ
    summary.pop("intonation_error", None)
    if errors:
        summary["intonation_error"] = errors_note(errors)
    write_atomic(sp, json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="synth_eval output dirs (rows.jsonl, summary.json, wav/)")
    ap.add_argument("--sets", default="soru", help="comma-separated; sets other than soru are scored from their text")
    ap.add_argument("--whisper", default="auto", help="faster-whisper model for word timestamps of rows without "
                    "row['words']: auto = large-v3-turbo if faster_whisper is installed, none = syllable split")
    ap.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    args = ap.parse_args(argv)
    sets = tuple(s for s in args.sets.split(",") if s)
    for d in args.dirs:  # before the (possibly downloading) whisper load
        if not (d / "rows.jsonl").exists():
            ap.error(f"{d / 'rows.jsonl'} not found")
        if not any(r["set"] in sets and _wav(d, r) for r in _rows(d)):
            ap.error(f"{d}: no WAV of the sets {','.join(sets)} under {d / 'wav'}; nothing changed")
    t0 = time.time()
    whisper = load_whisper(args.whisper)
    print(f"[inton] word timing: {'faster-whisper ' + args.whisper if whisper else 'syllable split'}", flush=True)
    results = {d: rescore(d, sets, whisper, args.dry_run) for d in args.dirs}
    for d, summ in results.items():
        print_summary(str(d), summ)
    print(f"[inton] total {time.time() - t0:.1f} s{' (dry run: nothing written)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
