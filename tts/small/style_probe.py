#!/usr/bin/env python3
"""Does the student's style embedding change how it speaks? The same sentences with every style id, measured.

    render  (venv-student): <py> tts/small/style_probe.py render --ckpt data/small/runs/final/student.pt \
                                --decoder data/small/runs/final/decoder.pt --out voxcpm-contest/small-bench/style-probe
    measure (test venv, librosa): <py> tts/small/style_probe.py measure --out voxcpm-contest/small-bench/style-probe

render: SENTENCES x the 7 trained styles (laughter excluded), seed 0, euler 16 cfg 2 -> OUT/<k>_<style>.wav.
measure: per clip duration, speech rate (letters / s), F0 median (Hz) and range (5-95 %, semitones; librosa pyin,
65-500 Hz, 16 kHz, 10 ms hop), RMS dBFS; per style the mean over sentences, and the difference to neutral; next to
it the corpus reference per style (frames per letter of the f022 training clips; index.jsonl). Writes OUT/style.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[2]
SENTENCES = ["Kargonuz yarın öğleden önce adresinize teslim edilecek.",
             "Bu akşam toplantı yine iptal edildi, haberiniz var mı?",
             "Annemle babam hafta sonu bize gelecekler."]
STYLES = ("neutral", "angry", "calm", "cheerful", "empathetic", "sad", "surprised")


def render(args):
    import soundfile as sf

    from eval_student import Student

    st = Student({"ckpt": args.ckpt, "sampler": "euler", "steps": 16, "cfg": 2.0, "sway": 0.0,
                  "decoder": args.decoder}, "cuda")
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    for k, text in enumerate(SENTENCES):
        for sid, name in enumerate(STYLES):
            sf.write(str(out / f"{k}_{name}.wav"), st.say(text, sid), 48000, subtype="PCM_16")
    print(f"{len(SENTENCES) * len(STYLES)} clips in {out}")


def measure(args):
    import librosa
    import numpy as np

    out = REPO / args.out
    rows = []
    for k, text in enumerate(SENTENCES):
        letters = sum(c.isalpha() for c in text)
        for name in STYLES:
            y, _ = librosa.load(str(out / f"{k}_{name}.wav"), sr=16000)
            f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=500, sr=16000, hop_length=160)
            f = f0[voiced & np.isfinite(f0)]
            lo, hi = (np.percentile(f, [5, 95]) if len(f) else (np.nan, np.nan))
            rows.append({"k": k, "style": name, "dur": len(y) / 16000, "rate": letters / (len(y) / 16000),
                         "f0_med": float(np.median(f)) if len(f) else None,
                         "f0_range_st": float(12 * np.log2(hi / lo)) if len(f) else None,
                         "rms_db": float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))})
    idx = [json.loads(x) for x in (REPO / "data/small/f022-latents/index.jsonl").read_text(encoding="utf-8").splitlines()]
    corpus = {}
    for name in STYLES:
        rr = [r for r in idx if r["style"] == name and r["use"]]
        corpus[name] = round(float(np.mean([r["frames"] / max(1, sum(c.isalpha() for c in r["ema_text"]))
                                            for r in rr])), 3)
    per = {}
    for name in STYLES:
        rr = [r for r in rows if r["style"] == name]
        per[name] = {m: round(float(np.mean([r[m] for r in rr])), 2) for m in ("dur", "rate", "f0_med", "f0_range_st",
                                                                           "rms_db")}
        per[name]["corpus_frames_per_letter"] = corpus[name]
    for name in STYLES:
        per[name]["d_f0_st_vs_neutral"] = round(12 * float(np.log2(per[name]["f0_med"] / per["neutral"]["f0_med"])), 2)
        per[name]["d_rate_pct_vs_neutral"] = round(100 * (per[name]["rate"] / per["neutral"]["rate"] - 1), 1)
    (out / "style.json").write_text(json.dumps({"per_style": per, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    for name, v in per.items():
        print(name, v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["render", "measure"])
    ap.add_argument("--ckpt", default="data/small/runs/final/student.pt")
    ap.add_argument("--decoder", default="data/small/runs/final/decoder.pt")
    ap.add_argument("--out", default="voxcpm-contest/small-bench/style-probe")
    args = ap.parse_args(argv)
    render(args) if args.cmd == "render" else measure(args)


if __name__ == "__main__":
    main()
