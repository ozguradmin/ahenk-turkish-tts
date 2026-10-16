#!/usr/bin/env python3
"""ASR round-trip judge: transcribe synthesized audio with faster-whisper large-v3-turbo.

    python benchmarks/tts/judge.py --set dev_numbers10                    # every system/front-end
    python benchmarks/tts/judge.py --set tr_zor --systems ema-lightning --frontends raw trnorm

Judge: faster-whisper "large-v3-turbo" (mobiuslabsgmbh/faster-whisper-large-v3-turbo), int8 on CPU,
cpu_threads=2, language="tr", beam_size=5, otherwise faster-whisper defaults (temperature fallback;
ctranslate2's random seed is reset before every file, so fallback sampling is repeatable too).
Audio is read with soundfile and resampled to 16 kHz with soxr (VHQ) in-process; no band-limiting.

Writes benchmarks/tts/results/<set>/hyps/<system>__<frontend>.jsonl with the RAW hypothesis (no
normalization: score.py normalizes at scoring time). Resumable. Identical audio (same sha1, which
happens whenever two front-ends give a system the same model input) is transcribed once.
"""

from __future__ import annotations

import common  # noqa: F401  (thread caps, HF_HOME)

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from common import (FRONTENDS, REPO, RESULTS_DIR, THREADS, append_jsonl, hyp_path, latest_by_id, loadavg,
                    read_jsonl, set_name, synth_path)

MODEL = "large-v3-turbo"
MODEL_REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
OPTS = dict(language="tr", beam_size=5)
JUDGE = f"faster-whisper:{MODEL}:int8:beam5:tr"


def load_audio_16k(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000 and len(audio):
        try:
            import soxr

            audio = soxr.resample(audio, sr, 16000, quality="VHQ")
        except ImportError:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g)
    return np.ascontiguousarray(audio, dtype=np.float32)


class Judge:
    def __init__(self, threads=THREADS):
        import ctranslate2
        from faster_whisper import WhisperModel

        self._ct2 = ctranslate2
        self.model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=threads, num_workers=1)

    def transcribe(self, audio: np.ndarray) -> dict:
        self._ct2.set_random_seed(0)
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(audio, **OPTS)
        segments = list(segments)
        return {
            "hyp": " ".join(s.text.strip() for s in segments).strip(),
            "asr_seconds": round(time.perf_counter() - t0, 3),
            "avg_logprob": round(float(np.mean([s.avg_logprob for s in segments])), 4) if segments else None,
            "no_speech_prob": round(float(max(s.no_speech_prob for s in segments)), 4) if segments else None,
        }


def write_meta(sname):
    import json
    from datetime import datetime, timezone

    from common import cpu_model, pkg_versions

    try:
        from huggingface_hub import snapshot_download

        revision = Path(snapshot_download(MODEL_REPO, local_files_only=True)).name
    except Exception:
        revision = None
    meta = {"judge": JUDGE, "model": MODEL, "model_repo": MODEL_REPO, "revision": revision,
            "compute_type": "int8", "device": "cpu", "cpu_threads": THREADS, "options": OPTS,
            "resample": "soundfile + soxr VHQ to 16 kHz", "cpu": cpu_model(),
            "versions": {k: v for k, v in pkg_versions().items() if k in ("faster-whisper", "ctranslate2")},
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    from common import write_text_atomic

    write_text_atomic(RESULTS_DIR / sname / "hyps" / "judge.meta.json", json.dumps(meta, indent=1) + "\n")


def pairs_for(sname, systems, frontends):
    out = []
    for p in sorted((RESULTS_DIR / sname / "synth").glob("*__*.jsonl")):
        system, frontend = p.stem.split("__", 1)
        if (not systems or system in systems) and (not frontends or frontend in frontends):
            out.append((system, frontend))
    # raw first, so later front-ends can reuse its transcriptions
    order = {f: i for i, f in enumerate(FRONTENDS)}
    return sorted(out, key=lambda sf: (sf[0], order.get(sf[1], 99)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--frontends", nargs="*")
    args = ap.parse_args(argv)
    sname = set_name(args.set)

    pairs = pairs_for(sname, args.systems, args.frontends)
    if not pairs:
        print(f"[judge {sname}] no synthesis results found under {RESULTS_DIR / sname / 'synth'}")
        return 0

    # what is left to do, and transcriptions we already have, by audio sha1 (per system, from every
    # front-end of that system, not only the ones selected for this run)
    known: dict[tuple[str, str], dict] = {}
    for system, frontend in pairs_for(sname, {s for s, _ in pairs}, None):
        for h in read_jsonl(hyp_path(sname, system, frontend)):
            if h.get("judge") == JUDGE and h.get("audio_sha1"):
                known[(system, h["audio_sha1"])] = h
    work = []
    for system, frontend in pairs:
        recs = latest_by_id(read_jsonl(synth_path(sname, system, frontend)))
        hyps = latest_by_id(read_jsonl(hyp_path(sname, system, frontend)))
        for uid, rec in recs.items():
            h = hyps.get(uid)
            if h is not None and h.get("judge") == JUDGE and h.get("audio_sha1") == rec.get("audio_sha1"):
                continue
            work.append((system, frontend, rec))
    need_model = len({(s, r["audio_sha1"]) for s, _, r in work
                      if r.get("audio_sha1") and r.get("duration") and (s, r["audio_sha1"]) not in known})
    print(f"[judge {sname}] {len(work)} hypotheses to write, {need_model} need ASR", flush=True)
    if not work:
        return 0

    judge = Judge() if need_model else None
    if judge is not None:
        write_meta(sname)
    n_asr = 0
    t_start = time.perf_counter()
    for i, (system, frontend, rec) in enumerate(work, 1):
        out = {"id": rec["id"], "system": system, "frontend": frontend, "audio_sha1": rec.get("audio_sha1"),
               "judge": JUDGE}
        sha = rec.get("audio_sha1")
        if rec.get("error") or not sha or not rec.get("duration"):
            out.update(hyp="", note="no audio (synthesis error or empty output)", reused_from=None)
        elif (system, sha) in known:
            old = known[(system, sha)]
            out.update(hyp=old["hyp"], asr_seconds=old.get("asr_seconds"), avg_logprob=old.get("avg_logprob"),
                       no_speech_prob=old.get("no_speech_prob"), reused_from=old.get("frontend"))
        else:
            audio = load_audio_16k(REPO / rec["audio_path"])
            out.update(judge.transcribe(audio), reused_from=None, loadavg=loadavg())
            known[(system, sha)] = out
            n_asr += 1
        append_jsonl(hyp_path(sname, system, frontend), out)
        if n_asr and (n_asr % 10 == 0) and out.get("reused_from") is None:
            el = time.perf_counter() - t_start
            print(f"[judge {sname}] {i}/{len(work)} ({n_asr} transcribed, {el / n_asr:.1f}s each)", flush=True)
    print(f"[judge {sname}] done: {len(work)} hypotheses, {n_asr} transcribed in "
          f"{time.perf_counter() - t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
