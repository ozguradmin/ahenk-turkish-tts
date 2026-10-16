#!/usr/bin/env python3
"""Breath noise in the pauses of rendered speech, turned down (blind round 4 on the long sentences: "after commas and
punctuation there is always a breath sound"; s3-1500 holds a sustained breath, >= 0.12 s above L - 40 dB, in 29 % of
its punctuation pauses, EMA in 13 %, mostly at sentence ends and ; :).

    clean(wav, sr) -> (float32 audio, report)
    python tts/voxcpm/pause_clean.py IN_DIR OUT_DIR [--glob "*.wav"] [--hp-hz 80] [--out-hp-hz 0]

1. Filter: the output loses its DC (the renders carry DC at -34 dB re RMS; under a gain change it would thump) and,
   with out_hp_hz > 0, gets a high-pass (off by default: at 80 Hz it moves 7 % of the s3-1500 frames within 14 dB of
   L by more than 0.1 dB, p99 1.4 dB, and the sub-80 Hz thumps at some word onsets by 3-19 dB). The analysis copy is
   high-passed at hp_hz (4th-order Butterworth, zero phase): sub-60 Hz energy would otherwise fill the pauses.
2. Analysis at 16 kHz on onset.py's grid (10 ms frames centred at (i + 0.5) x 10 ms): level of a WIN_S window in dB re
   L, the clip's active speech level (mean power of the frames within ACTIVE_DB of the 95th-percentile frame and above
   -60 dBFS); voicing from onset._pitch (Praat's ac method); spectral centroid of the WIN_S Hann frame (0-8 kHz); the
   rise of 2.5 ms blocks (a stop burst jumps >= BURST_RISE_DB within 10 ms, a breath onset about 5-17 dB from the
   renders' floor).
3. Pause: a run of >= MIN_PAUSE_S unvoiced frames under L - PAUSE_DB with sound on both sides (leading and trailing
   silence are not pauses).
4. Kept in the pause (word offsets and onsets): GUARD_S at each edge; the word's decay into it and the next word's
   rise out of it (the level, smoothed over 3 frames, from the pause start / end inwards while it stays under the
   lowest level reached + WALK_TOL_DB, up to that dip: a final "ş", "h", fricated "r" or release fades out, bumps and
   all; a breath is a hump in the silence, WALK_TOL_DB or more over the dip before it); a stop burst (step 2: the
   jump, the level's top within 30 ms of it and BURST_FALL_DB lower within 60 ms of the top, or the pause's end; a
   breath stays up) from 20 ms before it through its decay (the release and aspiration of "Sokak", "eksik,",
   "çözülecek."). The rest are gate zones.
5. Breath: a run of >= MIN_BREATH_S of zone frames (gaps of up to CLOSE_S of quiet frames bridged) louder than
   max(floor + FLOOR_UP_DB, L - BREATH_LO_DB), quieter than L - BREATH_HI_DB (louder: speech) with a centroid in
   CENTROID_HZ (broadband noise, not a sibilant, not voicing); floor = the clip's 3rd-percentile frame.
6. Gate: one gain per breath puts its peak at L - TARGET_DB, the cut at most MAX_CUT_DB, with FADE_S raised-cosine
   ramps (in dB) inside its zone; every other sample is the filtered input, unchanged. Deterministic, numpy + scipy,
   CPU (0.16 s for an 18 s clip at 48 kHz).

Report: sr, dur_s, L_dbfs, floor_db, pauses, breaths, gated_s, regions [{t0, t1 (full cut, s), cut_db, peak_db (re
L), pause [start, end] (s)}]. The breath measure of the analysis (word times from Whisper; a breath = >= 60 ms of such
frames 30 ms inside the pause, sustained = >= 0.12 s peaking above L - 40 dB) needs words; the gate does not.
On the 20 long held-out renders of s3-1500 (round 4) it turns 87 breaths down (median cut 17 dB): sustained breaths in
punctuation pauses 31 of 108 -> 0, no word's speech frame touched (Whisper: tts/voxcpm/README.md).

After synthesis: voice_contest.py render --pause-clean and synth_eval.py --pause-clean run clean() on every clip
before its level, onset check and scores; rows "pause_clean" = row_record(report), summary "pause_clean" =
rows_summary(rows), voice_contest settings "pause_clean" = SETTINGS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
if str(_HERE.parent) not in sys.path:  # also when imported by path (tests)
    sys.path.insert(0, str(_HERE.parent))

import onset  # noqa: E402  (no numpy at import)

SR, HOP, WIN_S = onset.SR, onset.HOP, 0.025
HP_HZ, ACTIVE_DB, ACTIVE_FLOOR_DBFS = 80.0, 30.0, -60.0
PAUSE_DB, MIN_PAUSE_S, GUARD_S = 14.0, 0.10, 0.03
FLOOR_UP_DB, BREATH_LO_DB, BREATH_HI_DB, CENTROID_HZ = 6.0, 50.0, 24.0, (700.0, 5000.0)
MIN_BREATH_S, CLOSE_S, WALK_TOL_DB = 0.04, 0.02, 6.0
BLOCKS, BURST_RISE_DB, BURST_FALL_DB, BURST_FALL_FR = 4, 24.0, 10.0, 7
TARGET_DB, MAX_CUT_DB, FADE_S = 55.0, 25.0, 0.015
# settings "pause_clean" of voice_contest / synth_eval runs: bump v1 when clean() decides otherwise
SETTINGS = f"v1/hp{HP_HZ:g}/L-{TARGET_DB:g}dB/max{MAX_CUT_DB:g}dB"
RECORD_KEYS = ("pauses", "breaths", "gated_s")  # a rendered row's "pause_clean"


def _hp(x, sr: int, hz: float):
    """x (samples x channels) high-passed at hz (4th-order Butterworth, zero phase); unchanged when hz <= 0."""
    if hz <= 0 or len(x) <= 64:
        return x
    from scipy.signal import butter, sosfiltfilt

    return sosfiltfilt(butter(4, hz, "highpass", fs=sr, output="sos"), x, axis=0)


def analyse(x16) -> dict:
    """Frame measures of mono 16 kHz audio (module doc, step 2): rel (dB re L), voiced, cen (Hz), rise (dB: the
    largest jump in the frame of a 2.5 ms block over the quietest of the BLOCKS blocks before it, or over the clip's
    floor when that is louder: digital silence is not a burst's start), L_dbfs, floor_db (re L)."""
    import numpy as np

    x16 = np.asarray(x16, dtype=np.float64)
    n = len(x16) // HOP
    if n < 3:
        raise ValueError("clip too short")
    win = int(round(WIN_S * SR))
    centers = ((np.arange(n) + 0.5) * HOP).astype(int)
    xp = np.concatenate([np.zeros(win), x16, np.zeros(win)])
    fr = xp[centers[:, None] + win - win // 2 + np.arange(win)[None, :]]
    db = 10 * np.log10((fr ** 2).mean(axis=1) + 1e-12)
    top = float(np.percentile(db, 95))
    act = (db > top - ACTIVE_DB) & (db > ACTIVE_FLOOR_DBFS)
    L = float(10 * np.log10(np.mean(10 ** (db[act] / 10)))) if act.any() else float(db.max())
    rel = db - L
    floor = float(np.percentile(rel, 3))
    n_fft = 1 << (win - 1).bit_length()
    P = np.abs(np.fft.rfft(fr * np.hanning(win), n_fft, axis=1)) ** 2
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    cen = (P * f).sum(axis=1) / (P.sum(axis=1) + 1e-20)
    k = HOP // BLOCKS  # 2.5 ms blocks, BLOCKS a frame
    e = 10 * np.log10((x16[:n * HOP].reshape(n * BLOCKS, k) ** 2).mean(axis=1) + 1e-12)
    ep = np.concatenate([np.full(BLOCKS, e[0]), e])
    before = np.maximum(np.min(np.stack([ep[j:j + len(e)] for j in range(BLOCKS)]), axis=0), L + floor)
    rise = (e - before).reshape(n, BLOCKS).max(axis=1)
    return {"rel": rel, "voiced": onset._pitch(x16, n) > 0, "cen": cen, "rise": rise, "L_dbfs": L, "floor_db": floor}


def _smooth(v):
    import numpy as np

    p = np.concatenate([v[:1], v, v[-1:]])
    return (p[:-2] + p[1:-1] + p[2:]) / 3


def _walk(sm, k: int, stop: int, step: int) -> int:
    """From frame k towards stop (exclusive) while the smoothed level stays under the lowest level reached +
    WALK_TOL_DB (a word's noisy decay, bumps and all): the dip (that lowest frame) when a hump stops it, else the last
    frame. Against the last frame alone a word-final fricated "r" (f022-1500 "tutar,", EMA "paylaşılıyor.") lost its
    protection at its first 1 dB bump."""
    lo = k
    while k + step != stop and sm[k + step] < sm[lo] + WALK_TOL_DB:
        k += step
        if sm[k] < sm[lo]:
            lo = k
    return k if k + step == stop else lo


def breaths(A: dict) -> tuple[list[tuple[int, int]], list[dict]]:
    """Pauses and breaths of analyse()'s frames (module doc, steps 3-5): ([(start, end) frames of each pause],
    [{a, b (frames), z0, z1 (its gate zone), pause: (start, end), peak_db}])."""
    import numpy as np

    rel, voiced, cen, rise = A["rel"], A["voiced"], A["cen"], A["rise"]
    n = len(rel)
    sm = _smooth(rel)
    lo = max(A["floor_db"] + FLOOR_UP_DB, -BREATH_LO_DB)
    like = (rel > lo) & (rel < -BREATH_HI_DB) & (cen >= CENTROID_HZ[0]) & (cen <= CENTROID_HZ[1])
    guard, min_pause = int(round(GUARD_S / 0.01)), int(round(MIN_PAUSE_S / 0.01))
    close, min_br = int(round(CLOSE_S / 0.01)), int(round(MIN_BREATH_S / 0.01))
    pauses, out = [], []
    for a, b in onset._runs(~voiced & (rel < -PAUSE_DB)):
        if b - a < min_pause or a == 0 or b == n:  # leading / trailing silence
            continue
        pauses.append((a, b))
        keep = np.zeros(n, dtype=bool)  # protected frames: the guards, the words' decay and rise, bursts
        keep[a:a + guard] = keep[b - guard:b] = True
        keep[a:_walk(sm, a, b, 1)] = True
        keep[_walk(sm, b - 1, a - 1, -1) + 1:b] = True
        for i in range(a, b):  # a stop release and its aspiration: a jump, its top within 30 ms, then a fall
            if rise[i] >= BURST_RISE_DB and rel[i] > lo:
                pk = i + int(np.argmax(rel[i:min(b, i + 4)]))
                if pk + BURST_FALL_FR > b or rel[pk:pk + BURST_FALL_FR].min() <= rel[pk] - BURST_FALL_DB:
                    keep[max(a, i - 2):_walk(sm, pk, b, 1)] = True
        for z0, z1 in onset._runs(~keep[a:b], a):  # gate zones
            m = like[z0:z1].copy()
            for g0, g1 in onset._runs(~m):  # bridge short quiet gaps between breath frames
                if 0 < g0 and g1 < len(m) and g1 - g0 <= close and rel[z0 + g0:z0 + g1].max() < -BREATH_HI_DB:
                    m[g0:g1] = True
            for r0, r1 in onset._runs(m, z0):
                if r1 - r0 >= min_br:
                    out.append({"a": r0, "b": r1, "z0": z0, "z1": z1, "pause": (a, b),
                                "peak_db": float(rel[r0:r1][like[r0:r1]].max())})
    return pauses, out


def clean(wav, sr: int, hp_hz: float = HP_HZ, out_hp_hz: float = 0.0):
    """(float32 audio of wav's shape, report): wav without DC (high-passed at out_hp_hz > 0) with its pause breaths
    turned down (module doc); hp_hz: the analysis high-pass. Multi-channel audio is analysed on its mean and gets one
    gain on every channel."""
    import numpy as np

    x = np.asarray(wav, dtype=np.float64)
    shape = x.shape
    x = x.reshape(len(x), -1)
    x = np.ascontiguousarray(_hp(x - x.mean(axis=0, keepdims=True), int(sr), out_hp_hz))
    rep = {"sr": int(sr), "dur_s": round(len(x) / sr, 3), "pauses": 0, "breaths": 0, "gated_s": 0.0, "regions": []}
    try:
        A = analyse(onset._resample(_hp(x.mean(axis=1), int(sr), hp_hz), int(sr), SR))
    except ValueError:  # under 3 frames
        return x.reshape(shape).astype(np.float32), rep
    pauses, brs = breaths(A)
    rep.update(L_dbfs=round(A["L_dbfs"], 2), floor_db=round(A["floor_db"], 1), pauses=len(pauses))
    s = lambda fr: min(int(round(fr * 0.01 * sr)), len(x))  # noqa: E731  (frame index -> sample)
    fade = max(1, int(round(FADE_S * sr)))
    gdb = np.zeros(len(x))
    for br in brs:
        cut = max(-MAX_CUT_DB, min(0.0, -TARGET_DB - br["peak_db"]))
        c0, c1 = max(s(br["a"]), s(br["z0"]) + fade), min(s(br["b"]), s(br["z1"]) - fade)  # full cut; ramps in zone
        if c1 - c0 < fade or cut >= 0:
            continue
        ramp = cut * (0.5 - 0.5 * np.cos(np.pi * (np.arange(fade) + 0.5) / fade))
        curve = np.full(c1 - c0 + 2 * fade, cut)
        curve[:fade], curve[-fade:] = ramp, ramp[::-1]
        seg = slice(c0 - fade, c1 + fade)
        gdb[seg] = np.minimum(gdb[seg], curve)
        rep["regions"].append({"t0": round(c0 / sr, 3), "t1": round(c1 / sr, 3), "cut_db": round(cut, 1),
                               "peak_db": round(br["peak_db"], 1),
                               "pause": [round(br["pause"][0] * 0.01, 2), round(br["pause"][1] * 0.01, 2)]})
    rep["breaths"] = len(rep["regions"])
    rep["gated_s"] = round(sum(r["t1"] - r["t0"] for r in rep["regions"]), 3)
    y = x * (10 ** (gdb / 20))[:, None]
    return y.reshape(shape).astype(np.float32), rep


def row_record(rep: dict) -> dict:
    """A rendered clip's row "pause_clean" (voice_contest, synth_eval): pauses, breaths, gated_s of clean()'s report."""
    return {k: rep[k] for k in RECORD_KEYS}


def rows_summary(rows: list[dict]) -> dict | None:
    """Summary "pause_clean" of rendered rows: settings, clips (rows with a record), pauses, breaths, gated_s (sums);
    None when no row's clip went through clean()."""
    recs = [r["pause_clean"] for r in rows if isinstance(r.get("pause_clean"), dict)]
    if not recs:
        return None
    return {"settings": SETTINGS, "clips": len(recs), "pauses": sum(x["pauses"] for x in recs),
            "breaths": sum(x["breaths"] for x in recs), "gated_s": round(sum(x["gated_s"] for x in recs), 3)}


def clean_file(src: Path, dst: Path, hp_hz: float = HP_HZ, out_hp_hz: float = 0.0) -> dict:
    """clean() of a WAV written to dst in src's sample format (PCM: rounded to the nearest step and clipped, as
    libsndfile would floor); the report + "file"."""
    import numpy as np
    import soundfile as sf

    info = sf.info(str(src))
    x, sr = sf.read(str(src), dtype="float64", always_2d=info.channels > 1)
    y, rep = clean(x, sr, hp_hz, out_hp_hz)
    bits = {"PCM_U8": 7, "PCM_S8": 7, "PCM_16": 15, "PCM_24": 23, "PCM_32": 31}.get(info.subtype)
    if bits:
        q = 2.0 ** bits
        y = np.clip(np.round(y.astype(np.float64) * q), -q, q - 1) / q
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, sr, subtype=info.subtype)
    return {"file": src.name, **rep}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir", type=Path, help="folder of WAVs")
    ap.add_argument("out_dir", type=Path, help="cleaned WAVs (same names and sample format) + pause_clean.jsonl")
    ap.add_argument("--glob", default="*.wav", help="files of in_dir to clean (default: *.wav)")
    ap.add_argument("--hp-hz", type=float, default=HP_HZ, help="high-pass of the analysis copy (Hz; 0: none)")
    ap.add_argument("--out-hp-hz", type=float, default=0.0, help="high-pass of the output (Hz; default 0: DC only)")
    args = ap.parse_args(argv)
    files = sorted(p for p in args.in_dir.glob(args.glob) if p.is_file())
    if not files:
        ap.error(f"no {args.glob} in {args.in_dir}")
    if args.out_dir.resolve() == args.in_dir.resolve():
        ap.error("out_dir must differ from in_dir")
    t0, reps = time.time(), []
    for p in files:
        r = clean_file(p, args.out_dir / p.name, args.hp_hz, args.out_hp_hz)
        reps.append(r)
        print(f"{p.name}: {r['pauses']} pauses, {r['breaths']} breaths turned down ({r['gated_s']:.2f} s)", flush=True)
    (args.out_dir / "pause_clean.jsonl").write_bytes("".join(json.dumps(r) + "\n" for r in reps).encode("utf-8"))
    nb = sum(r["breaths"] for r in reps)
    print(f"[pause_clean] {len(reps)} files, {sum(r['pauses'] for r in reps)} pauses, {nb} breaths, "
          f"{time.time() - t0:.1f} s -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
