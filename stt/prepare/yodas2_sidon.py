#!/usr/bin/env python3
"""PHASE-2 DATA (flag-gated, not used by default): YODAS2 Turkish with
human-uploaded captions, speech-restored by Sidon
(sarulab-speech/yodas2_sidon, subset tr000, CC-BY-3.0, ungated) ->
16 kHz FLAC segments + lhotse cuts, shard by shard.

Layout (inspected by streaming train-00042.tar.gz): WebDataset .tar.gz shards
(43 x ~2.9 GB = 126 GB for tr000), one sample per YouTube video:
    <key>.flac            whole video, 24 kHz mono (Sidon output)
    <key>.metadata.json   {"video_id", "duration", "utterances": {"utt_id": [...],
                           "text": [...], "start": [...], "end": [...]}}
Segments are cut at the caption timestamps. Captions are uploader-provided
and NOISY: titles, phone numbers, on-screen text and mistimed lines are
common. Mitigations here: duration 1-20 s, 4-25 characters/s, trnorm text
normalisation, optional teacher-CER filter (--teacher-cer TSV from an
MIT/Apache ASR model such as Whisper; never from non-commercial models).
``tr100`` (automatic captions) is refused unless --allow-auto-captions.

    python stt/prepare/yodas2_sidon.py --num-shards 4 --num-workers 6
    python stt/prepare/yodas2_sidon.py --shards train-00042 --max-videos 20      # smoke test
    python stt/prepare/filter_teacher_cer.py ...   # later, once a teacher CER TSV exists

Outputs: audio/yodas/<shard>/*.flac, manifests/yodas/<subset>-<shard>_cuts_{train,dev}_raw.jsonl.gz,
manifests/yodas/state.json (resumable; finished shards are skipped).
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with python -I
from common import (  # noqa: E402
    SplitStats,
    asr_text,
    hf_resolve_url,
    hf_token,
    http_stream,
    make_cut,
    resample,
    safe_id,
    setup_logging,
    stt_data_dir,
    to_mono_float32,
    write_flac,
)
from hf_parquet import Prefetcher, ShardState, download_shard, list_shards, order_shards, stable_percent  # noqa: E402

REPO = "sarulab-speech/yodas2_sidon"
REVISION = "7c905ae21ac78e0f393bfa5f8b1c86efb1d0108f"  # 2025-12-06
log = setup_logging("yodas2")


def load_teacher_cer(path: str | None) -> dict:
    if not path:
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass  # header
    return out


_TEACHER: dict = {}


def _init(teacher_path):
    global _TEACHER
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _TEACHER = load_teacher_cer(teacher_path)


def process_video(task: dict) -> list:
    import soundfile as sf

    meta = json.loads(task["meta"])
    try:
        x, sr = sf.read(io.BytesIO(task["flac"]), dtype="float32", always_2d=False)
    except Exception:
        return [("drop", "decode", None)]
    x = resample(to_mono_float32(x), sr)
    utt = meta.get("utterances") or {}
    vid = meta.get("video_id") or task["key"]
    split = "dev" if stable_percent(vid) < task["dev_pct"] else "train"
    out = []
    for uid, text, st, en in zip(utt.get("utt_id", []), utt.get("text", []),
                                 utt.get("start", []), utt.get("end", [])):
        dur = float(en) - float(st)
        if not (task["min_dur"] <= dur <= task["max_dur"]):
            out.append(("drop", "duration", None))
            continue
        t = asr_text(text)
        if t is None:
            out.append(("drop", "text", None))
            continue
        cps = len(t.replace(" ", "")) / dur
        if not (task["min_cps"] <= cps <= task["max_cps"]):
            out.append(("drop", "chars_per_sec", None))
            continue
        if _TEACHER and _TEACHER.get(uid, 0.0) > task["max_teacher_cer"]:
            out.append(("drop", "teacher_cer", None))
            continue
        a, b = int(float(st) * 16000), int(float(en) * 16000)
        seg = x[a:b]
        if len(seg) < 16000 * task["min_dur"] * 0.9:
            out.append(("drop", "short_audio", None))
            continue
        flac = Path(task["audio_dir"]) / f"{safe_id(uid)}.flac"
        n = write_flac(flac, seg)
        out.append((split, "ok", {"uid": safe_id(uid), "flac": str(flac), "n": n, "text": t, "video": vid}))
    return out


def iter_videos(source, max_videos: int | None):
    """source: local .tar.gz path, or an http(s) URL to stream from."""
    pending: dict = {}
    count = 0
    stream = http_stream(source, hf_token()) if isinstance(source, str) and source.startswith("http") else None
    with (tarfile.open(fileobj=stream, mode="r|gz") if stream else tarfile.open(source, mode="r|gz")) as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            key, _, ext = name.partition(".")
            if ext not in ("flac", "metadata.json"):
                continue
            pending.setdefault(key, {})[ext] = tf.extractfile(m).read()
            if len(pending[key]) == 2:
                d = pending.pop(key)
                yield {"key": key, "flac": d["flac"], "meta": d["metadata.json"].decode("utf-8")}
                count += 1
                if max_videos and count >= max_videos:
                    return


def process_shard(name: str, source, data_dir: Path, a, pool) -> dict:
    from lhotse import CutSet

    stem = f"{a.subset}-{Path(name).name.split('.')[0]}"  # tr000-train-00042
    man = data_dir / "manifests" / "yodas"
    man.mkdir(parents=True, exist_ok=True)
    finals = {s: man / f"{stem}_cuts_{s}_raw.jsonl.gz" for s in ("train", "dev")}
    tmps = {s: p.with_name(p.name + ".tmp.jsonl.gz") for s, p in finals.items()}
    stats = {s: SplitStats("yodas", f"{stem}_{s}") for s in ("train", "dev")}
    common_task = {
        "audio_dir": str(data_dir / "audio" / "yodas" / stem), "dev_pct": a.dev_video_percent,
        "min_dur": a.min_duration, "max_dur": a.max_duration, "min_cps": a.min_chars_per_sec,
        "max_cps": a.max_chars_per_sec, "max_teacher_cer": a.max_teacher_cer,
    }
    tasks = ({**common_task, **v} for v in iter_videos(source, a.max_videos))
    with CutSet.open_writer(tmps["train"]) as wtr, CutSet.open_writer(tmps["dev"]) as wdev:
        # bounded in-flight work: imap-like over the generator in small windows
        window = []
        def drain(futs):
            for f in futs:
                for split, why, m in f.result():
                    if split == "drop":
                        stats["train"].drop(why)
                        continue
                    cut = make_cut(f"yodas_{m['uid']}", Path(m["flac"]), m["n"], m["text"], origin="yodas",
                                   speaker=f"yt_{m['video']}", custom={"video_id": m["video"]})
                    (wdev if split == "dev" else wtr).write(cut)
                    stats[split].add(cut.duration)
        for t in tasks:
            window.append(pool.submit(process_video, t))
            if len(window) >= 4 * a.num_workers:
                drain(window[: a.num_workers])
                window = window[a.num_workers:]
        drain(window)
    for s in ("train", "dev"):
        tmps[s].rename(finals[s])
        stats[s].save(data_dir, name=f"yodas/{stem}_{s}")
    return {"train_hours": round(stats["train"].hours, 4), "dev_hours": round(stats["dev"].hours, 4),
            "train_utts": stats["train"].utts, "dev_utts": stats["dev"].utts,
            "dropped": stats["train"].dropped, "partial": bool(a.max_videos)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="default: $STT_DATA")
    p.add_argument("--subset", default="tr000")
    p.add_argument("--allow-auto-captions", action="store_true", help="needed for tr100 (YouTube ASR captions)")
    p.add_argument("--shards", nargs="*", default=None, help="e.g. train-00042")
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--target-hours", type=float, default=None)
    p.add_argument("--shard-order", default="sorted", choices=["sorted", "shuffled"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-videos", type=int, default=None, help="per shard, smoke tests only")
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=20.0)
    p.add_argument("--min-chars-per-sec", type=float, default=4.0)
    p.add_argument("--max-chars-per-sec", type=float, default=25.0)
    p.add_argument("--teacher-cer", default=None, help="TSV utt_id<TAB>cer (0..1) from a permissive teacher ASR")
    p.add_argument("--max-teacher-cer", type=float, default=0.15)
    p.add_argument("--dev-video-percent", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--keep-tar", action="store_true")
    p.add_argument("--stream", action="store_true",
                   help="read shards over HTTP instead of downloading them (implied by --max-videos)")
    a = p.parse_args()
    if a.subset.startswith("tr1") and not a.allow_auto_captions:
        p.error(f"{a.subset} has automatic (ASR) captions; pass --allow-auto-captions to use it anyway")
    if not (a.shards or a.num_shards or a.target_hours):
        p.error("give --shards, --num-shards or --target-hours")

    data_dir = stt_data_dir(a.data_dir)
    token = hf_token()
    state = ShardState(data_dir / "manifests" / "yodas" / "state.json")
    if a.shards:
        order = [f"{a.subset}/{s if s.endswith('.tar.gz') else s + '.tar.gz'}" for s in a.shards]
    else:
        order = order_shards(list_shards(REPO, f"{a.subset}/*.tar.gz", REVISION, token), a.shard_order, a.seed)
    stem_of = lambda n: f"{a.subset}-{Path(n).name.split('.')[0]}"  # noqa: E731
    todo = [s for s in order if not state.is_done(stem_of(s))]
    dl = data_dir / "downloads" / "yodas"
    prefetch = Prefetcher(lambda n: download_shard(REPO, n, REVISION, dl, token), todo)
    ctx = mp.get_context("spawn")
    done = 0
    try:
        with ProcessPoolExecutor(a.num_workers, mp_context=ctx, initializer=_init,
                                 initargs=(a.teacher_cer,)) as pool:
            for i, name in enumerate(todo):
                if a.num_shards is not None and done >= a.num_shards:
                    break
                if a.target_hours is not None and state.total("train_hours") >= a.target_hours:
                    break
                more = not (a.num_shards is not None and done + 1 >= a.num_shards)
                t0 = time.time()
                if a.stream or a.max_videos:
                    source = hf_resolve_url(REPO, name, REVISION)
                else:
                    source = prefetch.get(i, prefetch_next=more)
                info = process_shard(name, source, data_dir, a, pool)
                info["seconds"] = round(time.time() - t0, 1)
                state.mark_done(stem_of(name), info)
                done += 1
                if isinstance(source, Path) and not a.keep_tar:
                    source.unlink(missing_ok=True)
                log.info(f"{stem_of(name)}: {info}")
    finally:
        prefetch.close()
    log.info(f"total {state.total('train_hours'):.2f} h train / {state.total('dev_hours'):.2f} h dev")


if __name__ == "__main__":
    main()
