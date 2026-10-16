#!/usr/bin/env python3
"""Alania synthetic Turkish speech (cloud0day3/alania-synthetic-speech-tr,
config ``cc-by`` only; CC-BY-4.0) -> 16 kHz FLAC + lhotse cuts, shard by shard.

The corpus is 655 Parquet shards (~0.8 GB, ~3.9 h each; 540 GB in total) with
48 kHz FLAC inside. This script streams N shards: download (next shard is
prefetched in the background) -> decode, resample to 16 kHz, write FLAC ->
manifests -> delete the Parquet file. Resumable: finished shards are recorded
in manifests/alania/state.json and skipped.

Transcript = trnorm.normalize_for_eval(text_spoken). About 45 % of the rows
also carry ``audio_channel`` (room/mic/noise-degraded copy); with the default
``--channel-policy degraded`` that copy is used instead of the clean one.
Voices whose stable hash falls below ``--dev-voice-percent`` are held out for
the dev set (never used for training).

Examples:
    # smoke test: the smallest shard, 200 rows
    python stt/prepare/alania.py --shards train-00654 --max-rows-per-shard 200
    # laptop: keep adding shards until 250 h of training audio
    python stt/prepare/alania.py --target-hours 250 --num-workers 6
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with python -I
from common import (  # noqa: E402
    SplitStats,
    asr_text,
    chars_per_second_ok,
    decode_bytes,
    hf_token,
    keep_duration,
    make_cut,
    resample,
    setup_logging,
    stt_data_dir,
    write_flac,
)
from hf_parquet import (  # noqa: E402
    Prefetcher,
    ShardState,
    download_shard,
    list_shards,
    order_shards,
    read_row_group,
    stable_percent,
)

REPO = "cloud0day3/alania-synthetic-speech-tr"
REVISION = "277dfc8a2322014099ff72688bcaee12480c4605"  # v1.3, 2026-10-04
COLUMNS = [
    "id", "audio", "audio_channel", "text", "text_spoken", "duration", "voice_id",
    "voice_type", "gender", "render_mode", "channel_preset", "qc_cer", "text_source",
]

AVG_SHARD_HOURS = 3.9  # 2565 h / 655 shards

log = setup_logging("alania")


def _use_degraded(row_id: str, policy: str, has_channel: bool) -> list:
    if not has_channel or policy == "clean":
        return [False]
    if policy == "degraded":
        return [True]
    if policy == "both":
        return [False, True]
    if policy == "random":  # deterministic 50/50
        return [stable_percent(row_id) < 50.0]
    raise ValueError(policy)


def process_row_group(task: dict) -> list:
    """Runs in a worker process. Returns ("train"|"dev", meta) or ("drop", reason)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rows = read_row_group(Path(task["parquet"]), task["rg"], COLUMNS)[: task["limit"]]
    out = []
    audio_dir = Path(task["audio_dir"])
    for row in rows:
        voice = row.get("voice_id") or "unknown"
        split = "dev" if stable_percent(voice) < task["dev_pct"] else "train"
        cer = row.get("qc_cer")
        if cer is not None and cer > task["max_cer"]:
            out.append(("drop", "qc_cer"))
            continue
        text = asr_text(row.get("text_spoken")) or asr_text(row.get("text"))
        if text is None:
            out.append(("drop", "text"))
            continue
        chan = row.get("audio_channel") or {}
        has_channel = bool(chan and chan.get("bytes"))
        for degraded in _use_degraded(row["id"], task["policy"], has_channel):
            blob = (chan if degraded else row["audio"])["bytes"]
            try:
                x, sr = decode_bytes(blob)
            except Exception:
                out.append(("drop", "decode"))
                continue
            x = resample(x, sr)
            seconds = len(x) / 16000
            if not keep_duration(seconds, 0.5, 30.0) or not chars_per_second_ok(text, seconds):
                out.append(("drop", "duration_or_rate"))
                continue
            cut_id = f"alania_{row['id']}" + ("_ch" if degraded else "")
            flac = audio_dir / f"{cut_id}.flac"
            n = write_flac(flac, x)
            out.append(
                (
                    split,
                    {
                        "cut_id": cut_id,
                        "flac": str(flac),
                        "num_samples": n,
                        "text": text,
                        "speaker": f"alania_{voice}",
                        "gender": row.get("gender") if row.get("gender") in ("male", "female") else None,
                        "custom": {
                            "channel": row.get("channel_preset") if degraded else "clean",
                            "render_mode": row.get("render_mode"),
                            "voice_type": row.get("voice_type"),
                            "text_source": row.get("text_source"),
                            "synthetic": True,
                        },
                    },
                )
            )
    return out


def process_shard(name: str, parquet: Path, data_dir: Path, args, pool: ProcessPoolExecutor) -> dict:
    import pyarrow.parquet as pq
    from lhotse import CutSet

    stem = Path(name).stem  # train-00654
    pf = pq.ParquetFile(str(parquet))
    rg_rows = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    budget = args.max_rows_per_shard or sum(rg_rows)
    tasks = []
    for i, n in enumerate(rg_rows):
        if budget <= 0:
            break
        tasks.append(
            {
                "parquet": str(parquet),
                "rg": i,
                "limit": min(n, budget),
                "audio_dir": str(data_dir / "audio" / "alania" / stem),
                "dev_pct": args.dev_voice_percent,
                "max_cer": args.max_cer,
                "policy": args.channel_policy,
            }
        )
        budget -= n

    man_dir = data_dir / "manifests" / "alania"
    man_dir.mkdir(parents=True, exist_ok=True)
    stats = {"train": SplitStats("alania", f"{stem}_train"), "dev": SplitStats("alania", f"{stem}_dev")}
    finals = {s: man_dir / f"{stem}_cuts_{s}_raw.jsonl.gz" for s in ("train", "dev")}
    tmps = {s: p.with_name(p.name + ".tmp.jsonl.gz") for s, p in finals.items()}
    dev_written = 0
    with CutSet.open_writer(tmps["train"]) as wtr, CutSet.open_writer(tmps["dev"]) as wdev:
        for results in pool.map(process_row_group, tasks):
            for split, meta in results:
                if split == "drop":
                    stats["train"].drop(meta)
                    continue
                if split == "dev" and dev_written >= args.dev_max_per_shard:
                    stats["dev"].drop("dev_cap")
                    Path(meta["flac"]).unlink(missing_ok=True)
                    continue
                cut = make_cut(
                    meta["cut_id"], Path(meta["flac"]), meta["num_samples"], meta["text"],
                    origin="alania", speaker=meta["speaker"], gender=meta["gender"],
                    custom=meta["custom"],
                )
                (wdev if split == "dev" else wtr).write(cut)
                stats[split].add(cut.duration)
                dev_written += split == "dev"
    for s in ("train", "dev"):
        tmps[s].rename(finals[s])
        stats[s].save(data_dir, name=f"alania/{stem}_{s}")
    return {
        "train_hours": round(stats["train"].hours, 4),
        "dev_hours": round(stats["dev"].hours, 4),
        "train_utts": stats["train"].utts,
        "dev_utts": stats["dev"].utts,
        "dropped": stats["train"].dropped,
        "partial": bool(args.max_rows_per_shard),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="default: $STT_DATA")
    p.add_argument("--config", default="cc-by", choices=["cc-by", "cc-by-sa"])
    p.add_argument("--allow-sharealike", action="store_true",
                   help="required for --config cc-by-sa (CC BY-SA 4.0 obligations!)")
    sel = p.add_argument_group("which shards")
    sel.add_argument("--shards", nargs="*", default=None, help="explicit shard stems, e.g. train-00654")
    sel.add_argument("--num-shards", type=int, default=None, help="process at most N new shards")
    sel.add_argument("--target-hours", type=float, default=None,
                     help="stop once this many training hours are prepared (counting earlier runs)")
    sel.add_argument("--shard-order", default="shuffled", choices=["shuffled", "sorted"])
    sel.add_argument("--seed", type=int, default=20261006)
    sel.add_argument("--max-rows-per-shard", type=int, default=None, help="smoke tests only")
    p.add_argument("--channel-policy", default="degraded", choices=["degraded", "clean", "both", "random"])
    p.add_argument("--dev-voice-percent", type=float, default=2.0)
    p.add_argument("--dev-max-per-shard", type=int, default=60)
    p.add_argument("--max-cer", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--keep-parquet", action="store_true")
    a = p.parse_args()
    if a.config == "cc-by-sa" and not a.allow_sharealike:
        p.error("cc-by-sa is share-alike; pass --allow-sharealike if you really want it")
    if not (a.shards or a.num_shards or a.target_hours):
        p.error("give --shards, --num-shards or --target-hours")

    data_dir = stt_data_dir(a.data_dir)
    token = hf_token()
    dl_dir = data_dir / "downloads" / "alania"
    state = ShardState(data_dir / "manifests" / "alania" / "state.json")

    if a.shards:
        order = [f"data/{a.config}/{s if s.endswith('.parquet') else s + '.parquet'}" for s in a.shards]
    else:
        order = order_shards(list_shards(REPO, f"data/{a.config}/*.parquet", REVISION, token), a.shard_order, a.seed)
    todo = [s for s in order if not state.is_done(Path(s).stem)]
    log.info(f"{len(order)} shards selected, {len(order) - len(todo)} already done, "
             f"{state.total('train_hours'):.2f} h prepared so far")

    prefetch = Prefetcher(lambda name: download_shard(REPO, name, REVISION, dl_dir, token), todo)
    ctx = mp.get_context("spawn")
    processed = 0
    t0 = time.time()
    try:
        with ProcessPoolExecutor(max_workers=a.num_workers, mp_context=ctx) as pool:
            for i, name in enumerate(todo):
                if a.num_shards is not None and processed >= a.num_shards:
                    break
                if a.target_hours is not None and state.total("train_hours") >= a.target_hours:
                    log.info(f"target of {a.target_hours} h reached")
                    break
                ts = time.time()
                # do not start downloading a shard we will not need
                more = not (a.num_shards is not None and processed + 1 >= a.num_shards) and not (
                    a.target_hours is not None
                    and state.total("train_hours") + AVG_SHARD_HOURS >= a.target_hours
                )
                parquet = prefetch.get(i, prefetch_next=more)
                td = time.time() - ts
                info = process_shard(name, parquet, data_dir, a, pool)
                info["seconds"] = round(time.time() - ts, 1)
                state.mark_done(Path(name).stem, info)
                processed += 1
                if not a.keep_parquet:
                    parquet.unlink(missing_ok=True)
                log.info(
                    f"[{processed}] {Path(name).stem}: {info['train_utts']} train / {info['dev_utts']} dev utts, "
                    f"{info['train_hours']:.2f} h; waited {td:.0f}s for download, total {info['seconds']:.0f}s; "
                    f"cumulative {state.total('train_hours'):.2f} h"
                )
    finally:
        prefetch.close()
    log.info(f"done in {time.time() - t0:.0f}s: {state.total('train_hours'):.2f} h train, "
             f"{state.total('dev_hours'):.2f} h dev over {len(state.data['done'])} shards")


if __name__ == "__main__":
    main()
