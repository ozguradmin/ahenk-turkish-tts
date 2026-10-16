#!/usr/bin/env python3
"""OPTIONAL, flag-gated: serdarcaglar/turkish-tts-audiobooks (gated on HF;
1,311 h of human audiobook narration with Whisper machine transcripts).

NOT part of the default (permissive-only) recipe. Licence "open-weights-with-
attribution": you must (1) cite the dataset and (2) publish the weights of any
model trained on it under a licence allowing commercial use (Apache-2.0 is
fine). The rights of the source recordings are NOT cleared (you accept legal
responsibility; delete on rightsholder request). Read the dataset card, accept
the terms on Hugging Face with the owner's account, then:

    export HF_TOKEN=hf_...            # token of the account that accepted the terms
    python stt/prepare/audiobooks.py --i-accept-the-terms --target-hours 300 --num-workers 6

Uses only the `train` (-> train) and `validation` (-> dev) pools, never
`review`. Audio is already 16 kHz FLAC; transcripts keep digits/punctuation,
so trnorm.normalize_for_eval is applied. Extra filters: --max-asr-cer on the
in-row two-pass ASR disagreement, --min-dnsmos.
"""

from __future__ import annotations

import argparse
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
    safe_id,
    setup_logging,
    stt_data_dir,
    write_flac,
)
from hf_parquet import Prefetcher, ShardState, download_shard, list_shards, order_shards, read_row_group  # noqa: E402

REPO = "serdarcaglar/turkish-tts-audiobooks"
REVISION = "744ea6aacefcc4b9385b8c10cd630fb60ca91e39"  # v1.0.0, 2026-08-23
COLUMNS = ["id", "audio", "text", "duration", "speaker_id", "source_id", "channel",
           "quality_asr_cer", "quality_dnsmos_ovrl"]
AVG_SHARD_HOURS = 5.0  # 1292 h / 258 train shards
log = setup_logging("audiobooks")


def process_row_group(task: dict) -> list:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out, audio_dir = [], Path(task["audio_dir"])
    for r in read_row_group(Path(task["parquet"]), task["rg"], COLUMNS)[: task["limit"]]:
        cer, mos = r.get("quality_asr_cer"), r.get("quality_dnsmos_ovrl")
        if cer is not None and cer > task["max_asr_cer"]:
            out.append(("drop", "asr_cer"))
            continue
        if mos is not None and mos < task["min_dnsmos"]:
            out.append(("drop", "dnsmos"))
            continue
        text = asr_text(r.get("text"))
        if text is None:
            out.append(("drop", "text"))
            continue
        try:
            x, sr = decode_bytes(r["audio"]["bytes"])
        except Exception:
            out.append(("drop", "decode"))
            continue
        x = resample(x, sr)
        seconds = len(x) / 16000
        if not keep_duration(seconds, 1.0, 20.5) or not chars_per_second_ok(text, seconds, lo=4.0, hi=25.0):
            out.append(("drop", "duration_or_rate"))
            continue
        uid = safe_id(r["id"])
        flac = audio_dir / f"{uid}.flac"
        out.append(("ok", {"uid": uid, "flac": str(flac), "n": write_flac(flac, x), "text": text,
                           "speaker": f"ab_{r.get('speaker_id') or r.get('source_id') or 'unk'}",
                           "source_id": r.get("source_id"), "channel": r.get("channel")}))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--i-accept-the-terms", action="store_true", required=True,
                   help="confirms you read and accepted the dataset terms (see docstring)")
    p.add_argument("--data-dir", default=None, help="default: $STT_DATA")
    p.add_argument("--splits", nargs="+", default=["train", "validation"], choices=["train", "validation"])
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--target-hours", type=float, default=None, help="train hours to collect")
    p.add_argument("--shard-order", default="shuffled", choices=["shuffled", "sorted"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows-per-shard", type=int, default=None)
    p.add_argument("--max-asr-cer", type=float, default=0.05)
    p.add_argument("--min-dnsmos", type=float, default=2.8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--keep-parquet", action="store_true")
    a = p.parse_args()
    token = hf_token()
    if not token:
        p.error("gated dataset: set HF_TOKEN (account that accepted the terms)")
    data_dir = stt_data_dir(a.data_dir)
    state = ShardState(data_dir / "manifests" / "audiobooks" / "state.json")
    dl = data_dir / "downloads" / "audiobooks"
    from lhotse import CutSet
    import pyarrow.parquet as pq

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(a.num_workers, mp_context=ctx) as pool:
        for hf_split in a.splits:
            split = "dev" if hf_split == "validation" else "train"
            files = list_shards(REPO, f"data/{hf_split}-*.parquet", REVISION, token)
            if split == "train":
                files = order_shards(files, a.shard_order, a.seed)
            todo = [f for f in files if not state.is_done(Path(f).stem)]
            prefetch = Prefetcher(lambda f: download_shard(REPO, f, REVISION, dl, token), todo)
            done = 0
            try:
                for i, name in enumerate(todo):
                    if split == "train":
                        if a.num_shards is not None and done >= a.num_shards:
                            break
                        if a.target_hours is not None and state.total("train_hours") >= a.target_hours:
                            break
                    more = split == "dev" or not (
                        (a.num_shards is not None and done + 1 >= a.num_shards)
                        or (a.target_hours is not None
                            and state.total("train_hours") + AVG_SHARD_HOURS >= a.target_hours))
                    t0 = time.time()
                    parquet = prefetch.get(i, prefetch_next=more)
                    stem = Path(name).stem
                    meta = pq.ParquetFile(str(parquet)).metadata
                    budget = a.max_rows_per_shard or meta.num_rows
                    tasks = []
                    for g in range(meta.num_row_groups):
                        if budget <= 0:
                            break
                        n = meta.row_group(g).num_rows
                        tasks.append({"parquet": str(parquet), "rg": g, "limit": min(n, budget),
                                      "audio_dir": str(data_dir / "audio" / "audiobooks" / stem),
                                      "max_asr_cer": a.max_asr_cer, "min_dnsmos": a.min_dnsmos})
                        budget -= n
                    out = data_dir / "manifests" / "audiobooks" / f"{stem}_cuts_{split}_raw.jsonl.gz"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_name(out.name + ".tmp.jsonl.gz")
                    st = SplitStats("audiobooks", f"{stem}_{split}")
                    with CutSet.open_writer(tmp) as w:
                        for res in pool.map(process_row_group, tasks):
                            for kind, m in res:
                                if kind == "drop":
                                    st.drop(m)
                                    continue
                                w.write(make_cut(f"ab_{m['uid']}", Path(m["flac"]), m["n"], m["text"],
                                                 origin="audiobooks", speaker=m["speaker"],
                                                 custom={"source_id": m["source_id"], "channel": m["channel"]}))
                                st.add(m["n"] / 16000)
                    tmp.rename(out)
                    st.save(data_dir, name=f"audiobooks/{stem}_{split}")
                    key = "train_hours" if split == "train" else "dev_hours"
                    state.mark_done(stem, {key: round(st.hours, 4), "utts": st.utts, "dropped": st.dropped,
                                           "seconds": round(time.time() - t0, 1)})
                    done += 1
                    if not a.keep_parquet:
                        parquet.unlink(missing_ok=True)
                    log.info(f"{stem}: {st.summary()}")
            finally:
                prefetch.close()
    log.info(f"audiobooks: {state.total('train_hours'):.1f} h train, {state.total('dev_hours'):.1f} h dev")


if __name__ == "__main__":
    main()
