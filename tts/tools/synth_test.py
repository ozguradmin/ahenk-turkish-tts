#!/usr/bin/env python3
"""Synthesize test sentences with an exported Piper voice on CPU and measure the real-time factor.

Front-end exactly as in production: text -> trnorm.normalize -> piper runtime (PiperVoice:
espeak-ng phonemes -> ONNX model on onnxruntime). One process, --threads intra-op threads,
no thread spinning; a warm-up sentence is synthesized first and model loading is not timed.

    python tts/tools/synth_test.py --model X.onnx --set benchmarks/tts/sets/tr_zor.jsonl --n 8 --out DIR
    python tts/tools/synth_test.py --model X.onnx --heldout-csv data/.../heldout.csv --out DIR

RTF = synthesis wall-clock / audio duration (lower is faster). Every output is checked: 16-bit WAV
at the voice's sample rate, non-empty, finite, not silent. Held-out sentences also get
duration_ratio = synthesized / real recording duration (speaking-rate sanity check, ~1.0 when
trained). Writes DIR/*.wav, DIR/synth.jsonl and DIR/summary.json; exits non-zero if a check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")


def pick_rows(path: Path, n: int) -> list[dict]:
    """First row of every category, then the next ones, until n rows (covers numbers, dates, ...)."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r.get("category", "?"), []).append(r)
    picked: list[dict] = []
    while len(picked) < n and any(by_cat.values()):
        for cat in list(by_cat):
            if by_cat[cat] and len(picked) < n:
                picked.append(by_cat[cat].pop(0))
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None, help="default: <model>.json")
    ap.add_argument("--set", type=Path, default=None, help="jsonl test set with id/category/text (e.g. TR-Zor)")
    ap.add_argument("--n", type=int, default=8, help="sentences from --set")
    ap.add_argument("--heldout-csv", type=Path, default=None, help="also synthesize these held-out texts (file|text)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    from piper import PiperVoice, SynthesisConfig
    from trnorm import normalize

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    voice = PiperVoice.load(args.model, config_path=args.config)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    ort.set_seed(args.seed)
    voice.session = ort.InferenceSession(str(args.model), sess_options=opts, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    rate = voice.config.sample_rate

    items: list[dict] = []
    if args.set:
        for r in pick_rows(args.set, args.n):
            items.append({"id": r["id"], "category": r.get("category", ""), "text": r["text"]})
    if args.heldout_csv:
        with open(args.heldout_csv, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                ref = args.heldout_csv.parent / "wav" / row[0]  # the real recording (prepare_antalia.py layout)
                items.append({"id": "heldout_" + Path(row[0]).stem[-16:], "category": "heldout", "text": row[-1],
                              "already_normalized": True, "reference": str(ref) if ref.exists() else None})
    if not items:
        ap.error("nothing to synthesize: give --set and/or --heldout-csv")

    def synth(text: str) -> np.ndarray:
        chunks = [c.audio_float_array for c in voice.synthesize(text, syn_config=SynthesisConfig())]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)

    synth(normalize("Merhaba, bu bir ısınma cümlesidir."))  # warm-up (not timed)

    rows, ok = [], True
    for it in items:
        model_text = it["text"] if it.get("already_normalized") else normalize(it["text"])
        t = time.perf_counter()
        audio = synth(model_text)
        sec = time.perf_counter() - t
        dur = len(audio) / rate
        wav = args.out / f"{it['id']}.wav"
        sf.write(str(wav), np.clip(audio, -1, 1), rate, subtype="PCM_16")
        info = sf.info(str(wav))
        checks = {
            "non_empty": info.frames > 0,
            "rate_ok": info.samplerate == rate == 22050,
            "finite": bool(np.isfinite(audio).all()),
            "not_silent": bool(audio.size and float(np.max(np.abs(audio))) > 0.01),
        }
        ok &= all(checks.values())
        if it.get("reference"):  # speaking-rate sanity check against the actor's recording of the same text
            ref_s = sf.info(it["reference"]).duration
            it = {**it, "reference_seconds": round(ref_s, 3), "duration_ratio": round(dur / ref_s, 3) if ref_s else None}
        rows.append({**it, "model_text": model_text, "phonemes": " | ".join("".join(p) for p in voice.phonemize(model_text)),
                     "wav": str(wav), "seconds": round(dur, 3), "synth_seconds": round(sec, 3),
                     "rtf": round(sec / dur, 4) if dur else None, "sample_rate": info.samplerate,
                     "peak": round(float(np.max(np.abs(audio))), 3) if audio.size else 0.0,
                     "rms": round(float(np.sqrt(np.mean(audio ** 2))), 4) if audio.size else 0.0, **checks})
        print(f"[synth] {it['id']:28s} {dur:6.2f} s audio  RTF {rows[-1]['rtf']}  {'OK' if all(checks.values()) else checks}",
              flush=True)

    with open(args.out / "synth.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rtfs = [r["rtf"] for r in rows if r["rtf"]]
    summary = {"model": str(args.model), "threads": args.threads, "load_seconds": round(load_s, 2),
               "sentences": len(rows), "audio_seconds": round(sum(r["seconds"] for r in rows), 2),
               "synth_seconds": round(sum(r["synth_seconds"] for r in rows), 2),
               "rtf_overall": round(sum(r["synth_seconds"] for r in rows) / max(1e-9, sum(r["seconds"] for r in rows)), 4),
               "rtf_median": round(statistics.median(rtfs), 4) if rtfs else None,
               "sample_rate": rate, "all_checks_passed": bool(ok)}
    ratios = [r["duration_ratio"] for r in rows if r.get("duration_ratio")]
    if ratios:  # synthesized / real duration on held-out sentences; ~1.0 once the voice is trained
        summary["heldout_duration_ratio_median"] = round(statistics.median(ratios), 3)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
