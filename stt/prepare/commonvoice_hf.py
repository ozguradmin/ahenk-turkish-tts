#!/usr/bin/env python3
"""Common Voice 17.0 Turkish (CC0-1.0) from the ungated Hugging Face Parquet
mirror ``fixie-ai/common_voice_17_0`` -> 16 kHz FLAC + lhotse cuts.

Files: tr/{train,validation,test,validated,other,invalidated}/chunk-range-*/*.parquet
(train 8 files / 742 MB / 35,147 clips; validation 243 MB / 11,258; test
306 MB / 11,290; validated 80 files / 2.6 GB / 114,056). Audio = 48 kHz MP3
bytes inside the Parquet, transcript = ``sentence``.

Default split mapping (official CV splits, speaker-disjoint):
    train <- tr/train      dev <- tr/validation      test <- tr/test
``--train-from validated`` uses tr/validated instead of tr/train, minus every
clip whose sentence occurs in dev/test (no test-sentence leakage) and, unless
``--allow-test-speakers``, minus every clip of a dev/test speaker. That gives
roughly 2-3x the training hours of tr/train.

This is an alternative to commonvoice.py (local Mozilla Data Collective
release); both write manifests/cv_cuts_<split>_raw.jsonl.gz, use one of them.

    python stt/prepare/commonvoice_hf.py --num-workers 6
    python stt/prepare/commonvoice_hf.py --train-from validated --num-workers 6
    python stt/prepare/commonvoice_hf.py --splits test --max-utts 20      # smoke test
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
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
    norm_gender,
    resample,
    setup_logging,
    stt_data_dir,
    write_flac,
)
from hf_parquet import Prefetcher, download_shard, list_shards, read_row_group  # noqa: E402

REPO = "fixie-ai/common_voice_17_0"
REVISION = "34f78a43893414e7b6e271ba94c1d5e05f18b239"  # 2025-01-17
LANG = "tr"
COLUMNS = ["client_id", "path", "audio", "sentence", "gender"]

log = setup_logging("cv17")

_EXCL_SENT: frozenset = frozenset()
_EXCL_SPK: frozenset = frozenset()


def _init_worker(sents, spks):
    global _EXCL_SENT, _EXCL_SPK
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _EXCL_SENT, _EXCL_SPK = frozenset(sents), frozenset(spks)


def process_row_group(task: dict) -> list:
    rows = read_row_group(Path(task["parquet"]), task["rg"], COLUMNS)[: task.get("limit")]
    out, audio_dir = [], Path(task["audio_dir"])
    for r in rows:
        text = asr_text(r.get("sentence"))
        if text is None:
            out.append(("drop", "text"))
            continue
        if text in _EXCL_SENT:
            out.append(("drop", "sentence_in_dev_test"))
            continue
        if r.get("client_id") in _EXCL_SPK:
            out.append(("drop", "speaker_in_dev_test"))
            continue
        try:
            x, sr = decode_bytes(r["audio"]["bytes"])
        except Exception:
            out.append(("drop", "decode"))
            continue
        x = resample(x, sr)
        seconds = len(x) / 16000
        if not keep_duration(seconds) or not chars_per_second_ok(text, seconds, lo=2.0):
            out.append(("drop", "duration_or_rate"))
            continue
        stem = Path(r.get("path") or r["audio"].get("path") or "x").stem
        flac = audio_dir / f"{stem}.flac"
        n = write_flac(flac, x)
        out.append(("ok", {"stem": stem, "flac": str(flac), "n": n, "text": text,
                           "client_id": r.get("client_id") or "", "gender": norm_gender(r.get("gender"))}))
    return out


def run_split(split: str, hf_split: str, data_dir: Path, a, token, excl_sent=(), excl_spk=()) -> tuple:
    """Returns (set of normalized sentences, set of client_ids) seen in this split."""
    from lhotse import CutSet
    import pyarrow.parquet as pq

    out = data_dir / "manifests" / f"cv_cuts_{split}_raw.jsonl.gz"
    files = list_shards(REPO, f"{LANG}/{hf_split}/*/*.parquet", REVISION, token)
    dl_dir = data_dir / "downloads" / "cv17"
    sents, spks = set(), set()
    if out.exists() and not a.overwrite:
        log.info(f"{out} exists, skipping (sentences/speakers re-read from it)")
        from lhotse import load_manifest_lazy

        for c in load_manifest_lazy(out):
            sents.add(c.supervisions[0].text)
            spks.add(c.supervisions[0].custom.get("client_id", ""))
        return sents, spks
    log.info(f"{split}: {len(files)} parquet files from {LANG}/{hf_split}")
    st = SplitStats("cv", split)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.jsonl.gz")
    prefetch = Prefetcher(lambda f: download_shard(REPO, f, REVISION, dl_dir, token), files)
    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(a.num_workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=(tuple(excl_sent), tuple(excl_spk))) as pool, \
                CutSet.open_writer(tmp) as w:
            for i, f in enumerate(files):
                if a.max_utts and st.utts >= a.max_utts:
                    break
                parquet = prefetch.get(i, prefetch_next=not a.max_utts)
                meta = pq.ParquetFile(str(parquet)).metadata
                budget = int((a.max_utts - st.utts) * 1.3) + 10 if a.max_utts else None
                tasks = []
                for g in range(meta.num_row_groups):
                    n_rows = meta.row_group(g).num_rows
                    tasks.append({"parquet": str(parquet), "rg": g, "limit": min(n_rows, budget) if budget else None,
                                  "audio_dir": str(data_dir / "audio" / "cv" / split)})
                    if budget is not None:
                        budget -= n_rows
                        if budget <= 0:
                            break
                for res in pool.map(process_row_group, tasks):
                    for kind, m in res:
                        if kind == "drop":
                            st.drop(m)
                            continue
                        if a.max_utts and st.utts >= a.max_utts:
                            Path(m["flac"]).unlink(missing_ok=True)
                            continue
                        w.write(make_cut(
                            f"cv_{split}_{m['stem']}", Path(m["flac"]), m["n"], m["text"], origin="cv",
                            speaker=f"cv_{m['client_id'][:16]}", gender=m["gender"],
                            custom={"client_id": m["client_id"]},
                        ))
                        st.add(m["n"] / 16000)
                        sents.add(m["text"])
                        spks.add(m["client_id"])
                if not a.keep_parquet:
                    parquet.unlink(missing_ok=True)
    finally:
        prefetch.close()
    tmp.rename(out)
    st.save(data_dir)
    log.info(f"{split}: {st.summary()}")
    return sents, spks


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="default: $STT_DATA")
    p.add_argument("--train-from", choices=["train", "validated"], default="train")
    p.add_argument("--allow-test-speakers", action="store_true",
                   help="with --train-from validated: only drop dev/test sentences, keep their speakers")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--max-utts", type=int, default=None, help="per split (smoke tests)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--keep-parquet", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    data_dir = stt_data_dir(a.data_dir)
    token = hf_token()

    excl_sent, excl_spk = set(), set()
    for split, hf_split in (("dev", "validation"), ("test", "test")):
        if split in a.splits or (a.train_from == "validated" and "train" in a.splits):
            s, k = run_split(split, hf_split, data_dir, a, token)
            excl_sent |= s
            excl_spk |= k
    if "train" in a.splits:
        if a.train_from == "validated":
            log.info(f"excluding {len(excl_sent)} dev/test sentences"
                     + ("" if a.allow_test_speakers else f" and {len(excl_spk)} dev/test speakers"))
            run_split("train", "validated", data_dir, a, token, excl_sent,
                      () if a.allow_test_speakers else excl_spk)
        else:
            run_split("train", "train", data_dir, a, token)


if __name__ == "__main__":
    main()
