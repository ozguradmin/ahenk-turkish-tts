#!/usr/bin/env python3
"""Compute 80-dim log-mel fbank (lhotse Fbank, Kaldi-compatible; the same
front-end sherpa-onnx uses at inference) for raw cut manifests and store them
with lilcom compression (~8 KB per second of audio, ~29 GB per 1000 h).

    # every manifests/**/*_raw.jsonl.gz that has no fbank/ counterpart yet
    python stt/prepare/compute_fbank.py --scan
    # on the laptop the GPU makes this ~10x faster
    python stt/prepare/compute_fbank.py --scan --device cuda --num-workers 4
    # free disk: delete the synthetic FLACs once their features exist
    python stt/prepare/compute_fbank.py --scan --drop-audio-for alania

Output for manifests/<rel>/<name>_raw.jsonl.gz:
    fbank/<rel>/<name>.jsonl.gz  and  fbank/<rel>/<name>_feats.lca
Resumable: finished outputs are skipped; outputs are written atomically.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with python -I
from common import limit_threads, setup_logging, stt_data_dir  # noqa: E402

log = setup_logging("fbank")


def compute_one(raw: Path, out: Path, device: str, num_workers: int, batch_duration: float,
                drop_audio: bool) -> None:
    import torch
    from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, load_manifest_lazy

    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    storage = out.with_name(out.name.replace(".jsonl.gz", "_feats"))
    tmp = out.with_name(out.name.replace(".jsonl.gz", ".tmp.jsonl.gz"))
    cuts = load_manifest_lazy(raw)
    n = sum(1 for _ in cuts)
    if n == 0:
        CutSet.from_cuts([]).to_file(out)
        log.info(f"{raw.name}: empty manifest")
        return
    extractor = Fbank(FbankConfig(num_mel_bins=80, device=device))
    with torch.no_grad():
        cuts.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=storage,
            manifest_path=tmp,
            batch_duration=batch_duration,
            num_workers=num_workers,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )
    tmp.rename(out)
    hours = sum(c.duration for c in load_manifest_lazy(out)) / 3600
    dt = time.time() - t0
    log.info(f"{raw.name}: {n} cuts, {hours:.2f} h in {dt:.0f}s ({hours * 3600 / max(dt, 1e-3):.0f}x real time)")
    if drop_audio:
        removed = 0
        for c in load_manifest_lazy(raw):
            for src in c.recording.sources:
                if src.type == "file":
                    Path(src.source).unlink(missing_ok=True)
                    removed += 1
        log.info(f"{raw.name}: deleted {removed} audio files (features kept)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="default: $STT_DATA")
    p.add_argument("--scan", action="store_true", help="process all pending raw manifests")
    p.add_argument("--in", dest="inp", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-workers", type=int, default=2, help="audio-loading DataLoader workers")
    p.add_argument("--batch-duration", type=float, default=600.0)
    p.add_argument("--drop-audio-for", default="",
                   help="comma list of dataset prefixes (e.g. alania,audiobooks) whose FLACs are deleted "
                        "after feature extraction (train/dev only; test audio is always kept)")
    a = p.parse_args()
    limit_threads()
    data_dir = stt_data_dir(a.data_dir)
    drop = {s for s in a.drop_audio_for.split(",") if s}

    jobs = []
    if a.inp:
        jobs.append((Path(a.inp), Path(a.out)))
    if a.scan:
        man = data_dir / "manifests"
        for raw in sorted(man.rglob("*_raw.jsonl.gz")):
            if ".tmp." in raw.name:
                continue
            rel = raw.relative_to(man)
            out = data_dir / "fbank" / rel.parent / raw.name.replace("_raw.jsonl.gz", ".jsonl.gz")
            if not out.exists():
                jobs.append((raw, out))
    if not jobs:
        log.info("nothing to do")
    for raw, out in jobs:
        name = str(raw.relative_to(data_dir / "manifests")) if a.scan else raw.name
        ds = name.split("/")[0].split("_")[0]
        drop_audio = ds in drop and "_test" not in raw.name
        compute_one(raw, out, a.device, a.num_workers, a.batch_duration, drop_audio)


if __name__ == "__main__":
    main()
