#!/usr/bin/env python3
"""UTMOS: predicted naturalness MOS (1-5, higher is better) of synthesized speech.

UTMOS22 strong learner (Saeki et al. 2022; weights MIT) through torch.hub `tarepan/SpeechMOS:v1.2.0`, the
predictor Piper's val_mos used. Needs only torch + torchaudio; the hub code and weights (~400 MB) are cached in
$TORCH_HOME/hub. The model takes (B, T) float audio at any rate and resamples it to 16 kHz itself.

    python tts/voxcpm/score_utmos.py voxcpm-eval/step_0000600 voxcpm-eval/trendyol [--device cpu|cuda]

rescores existing synth_eval output dirs: adds "utmos" to every row of DIR/rows.jsonl whose WAV is in
DIR/wav/<set>/<id>.wav (not the row's container path; a row whose WAV is not there keeps the score it has, a dir
with none of its WAVs is refused untouched), rewrites rows.jsonl atomically, sets summary["utmos"] (mean per set and
per set/category, the keys of summary["wer"]) in DIR/summary.json and prints the means. summary["utmos_error"] says
which rows have no score, and is dropped once every row has one.
modal_app --action fetch keeps a local dir whose summary has UTMOS when the Volume's copy has none (e.g.
step_0000600, trendyol, scored only here); if a fetch does replace one (local dir renamed), run this again on it.
synth_eval.py imports load_predictor/score_rows/utmos_summary to score right after synthesis.
Importing this module needs no torch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

HUB_REPO, HUB_MODEL = "tarepan/SpeechMOS:v1.2.0", "utmos22_strong"
NET_TIMEOUT_S = 60.0  # per blocking socket call: a stalled GitHub fetch raises instead of idling the GPU


def load_predictor(device: str = "cpu"):
    """A cold $TORCH_HOME fetches the hub code and the weights from GitHub (torch.hub's urlopen has no timeout).
    skip_validation: the tag is pinned, and the check is unauthenticated GitHub API calls (60/h per IP)."""
    import socket

    import torch

    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(NET_TIMEOUT_S)
    try:
        return torch.hub.load(HUB_REPO, HUB_MODEL, trust_repo=True, skip_validation=True).to(device).eval()
    finally:
        socket.setdefaulttimeout(old)


def score_wav(predictor, path: str | Path, device: str = "cpu") -> float:
    """UTMOS of one file: mono float32 [1, T] at its native rate (SpeechMOS forward(wave, sr) -> (B,))."""
    import numpy as np
    import soundfile as sf
    import torch

    wave, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = np.ascontiguousarray(wave.mean(axis=1))
    with torch.inference_mode():
        v = float(predictor(torch.from_numpy(wave).unsqueeze(0).to(device), sr)[0])
    if not math.isfinite(v):
        raise ValueError(f"UTMOS is {v}")
    return v


def score_rows(predictor, rows: list[dict], wav_of: Callable[[dict], str | Path | None], device: str,
               label: str) -> dict[str, str]:
    """row["utmos"] for every row with a WAV (wav_of(row) None: the row is left as it is). One bad clip costs only its
    own row (UTMOS raises on < ~25 ms of audio): its score is dropped, row["utmos_error"] says why, the rest are still
    scored. Returns {"set/id": error}."""
    todo = [(r, p) for r in rows if (p := wav_of(r)) is not None]
    t0, errors = time.time(), {}
    for i, (r, p) in enumerate(todo):
        try:
            r["utmos"] = round(score_wav(predictor, p, device), 4)
            r.pop("utmos_error", None)
        except Exception as e:
            r.pop("utmos", None)
            r["utmos_error"] = errors[f"{r['set']}/{r['id']}"] = f"{type(e).__name__}: {e}"
            print(f"[utmos] {label}: {r['set']}/{r['id']} not scored: {r['utmos_error']}", flush=True)
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"[utmos] {label}: {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    return errors


def errors_note(errors: dict[str, str]) -> str:
    """summary["utmos_error"] for the rows score_rows could not score."""
    k, v = next(iter(errors.items()))
    return f"{len(errors)} rows not scored, left out of the means (row utmos_error); first {k}: {v}"


def utmos_summary(rows: list[dict]) -> dict[str, dict]:
    """{set and set/category: {"mean", "rows"}}; rows without "utmos" are skipped."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("utmos") is None:
            continue
        for key in (r["set"], f"{r['set']}/{r['category']}"):
            groups[key].append(r["utmos"])
    return {k: {"mean": round(sum(v) / len(v), 3), "rows": len(v)} for k, v in sorted(groups.items())}


def write_atomic(path: Path, text: str) -> None:
    """Replace path in one step: a kill mid-write leaves the previous complete file (and a stray .tmp)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")  # LF as written in the container, also on Windows
    os.replace(tmp, path)


def print_summary(name: str, summ: dict[str, dict]) -> None:
    print(f"[utmos] {name}")
    for k, v in summ.items():
        print(f"  {'  ' if '/' in k else ''}{k:<22} {v['mean']:.3f}  ({v['rows']} rows)")


def _rows(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def _wav(d: Path, r: dict) -> Path | None:
    p = d / "wav" / r["set"] / f"{r['id']}.wav"
    return p if p.exists() else None


def rescore(d: Path, predictor, device: str) -> dict[str, dict]:
    rows = _rows(d)
    here = sum(_wav(d, r) is not None for r in rows)
    if not here:  # e.g. a copy of the HF eval repo (json only): never wipe the scores computed on the GPU
        raise FileNotFoundError(f"{d}: none of the {len(rows)} WAVs is under {d / 'wav'}; nothing changed")
    t0 = time.time()
    errors = score_rows(predictor, rows, lambda r: _wav(d, r), device, d.name)
    no_wav = [r for r in rows if _wav(d, r) is None]
    if no_wav:
        print(f"[utmos] {d.name}: {len(no_wav)} WAVs not here, those rows keep their score "
              f"({sum(r.get('utmos') is not None for r in no_wav)} have one)")
    write_atomic(d / "rows.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    sp = d / "summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    summary["utmos"] = utmos_summary(rows)
    summary.pop("utmos_error", None)  # stale: a failed or unfinished UTMOS in the container
    notes = [errors_note(errors)] if errors else []
    if unscored := sum(r.get("utmos") is None for r in no_wav):
        notes.append(f"{unscored} rows not scored: WAV not under {d / 'wav'} at the rescore")
    if notes:
        summary["utmos_error"] = "; ".join(notes)
    write_atomic(sp, json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(f"[utmos] {d.name}: {here - len(errors)}/{len(rows)} rows scored in {time.time() - t0:.1f} s", flush=True)
    return summary["utmos"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="synth_eval output dirs (rows.jsonl, summary.json, wav/)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None, help="default: cuda if available")
    args = ap.parse_args(argv)
    for d in args.dirs:  # before the (possibly downloading) predictor load
        if not (d / "rows.jsonl").exists():
            ap.error(f"{d / 'rows.jsonl'} not found")
        if not any(_wav(d, r) for r in _rows(d)):
            ap.error(f"{d}: none of its WAVs is under {d / 'wav'} (fetch them first); nothing changed")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    predictor = load_predictor(device)
    print(f"[utmos] {HUB_REPO} {HUB_MODEL} on {device}, loaded in {time.time() - t0:.1f} s", flush=True)
    results = {d: rescore(d, predictor, device) for d in args.dirs}
    for d, summ in results.items():
        print_summary(str(d), summ)
    print(f"[utmos] total {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
