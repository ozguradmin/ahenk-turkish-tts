#!/usr/bin/env python3
"""ISSAI Turkish Speech Corpus (issai/Turkish_Speech_Corpus, MIT, 218.2 h,
186,171 utterances) -> 16 kHz FLAC + lhotse cuts.

The corpus is ONE 21.4 GB tarball (ISSAI_TSC_218.tar.gz). Layout (inspected by
streaming the first 80 MB):

    ISSAI_TSC_218/Test/<id>.wav    16 kHz mono PCM
    ISSAI_TSC_218/Test/<id>.txt    lower-case transcript, numbers spelled out
    ISSAI_TSC_218/Train/...  (and a dev split; names are matched case-insensitively)

The .txt files split apostrophe suffixes off proper nouns ("erdoğan ın");
--rejoin-suffixes (default on) re-attaches the unambiguous ones, see
common.rejoin_split_suffixes. Then trnorm.normalize_for_eval as everywhere.

Modes:
    # laptop: resumable download (21.4 GB) then convert; --delete-archive frees it
    python stt/prepare/issai_tsc.py --delete-archive
    # already downloaded somewhere
    python stt/prepare/issai_tsc.py --archive /mnt/d/ISSAI_TSC_218.tar.gz
    # no disk for the archive: stream over HTTP (not resumable)
    python stt/prepare/issai_tsc.py --stream
    # smoke test: first 40 test utterances only
    python stt/prepare/issai_tsc.py --stream --splits test --max-utts 40
Peak disk: archive 21.4 GB + FLAC ~14 GB (FLAC is ~45 % smaller than the WAVs).
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with python -I
from common import (  # noqa: E402
    SplitStats,
    asr_text,
    chars_per_second_ok,
    decode_bytes,
    hf_resolve_url,
    http_stream,
    keep_duration,
    make_cut,
    rejoin_split_suffixes,
    resample,
    setup_logging,
    stt_data_dir,
    write_flac,
)

REPO = "issai/Turkish_Speech_Corpus"
REVISION = "d79ea01392882322e8f46657cb635236af25b6d4"  # 2025-02-13
FILENAME = "ISSAI_TSC_218.tar.gz"
def split_of(dirname: str):
    """Map the corpus' split directory names (Train/Dev/Test, any case) to ours."""
    d = dirname.lower()
    if "train" in d:
        return "train"
    if "dev" in d or "val" in d:
        return "dev"
    if "test" in d or "eval" in d:
        return "test"
    return None

log = setup_logging("issai")


def _encode(wav_bytes: bytes, flac: Path):
    x, sr = decode_bytes(wav_bytes)
    x = resample(x, sr)
    return write_flac(flac, x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="default: $STT_DATA")
    p.add_argument("--archive", default=None, help="path of an existing ISSAI_TSC_218.tar.gz")
    p.add_argument("--stream", action="store_true", help="read the tarball over HTTP, do not store it")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--max-utts", type=int, default=None, help="per split (smoke tests)")
    p.add_argument("--no-rejoin-suffixes", dest="rejoin", action="store_false")
    p.add_argument("--delete-archive", action="store_true", help="delete the tarball when done")
    p.add_argument("--num-workers", type=int, default=2, help="FLAC encoder threads")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    data_dir = stt_data_dir(a.data_dir)
    finals = {s: data_dir / "manifests" / f"issai_cuts_{s}_raw.jsonl.gz" for s in a.splits}
    if not a.overwrite and all(f.exists() for f in finals.values()):
        log.info("all requested ISSAI manifests exist; use --overwrite to redo")
        return

    stream = None
    archive = Path(a.archive) if a.archive else None
    if archive is None and not a.stream:
        from huggingface_hub import hf_hub_download

        log.info(f"downloading {FILENAME} (21.4 GB, resumable) ...")
        archive = Path(
            hf_hub_download(REPO, FILENAME, repo_type="dataset", revision=REVISION,
                            local_dir=str(data_dir / "downloads" / "issai"))
        )
    if archive is not None:
        tf = tarfile.open(archive, mode="r|gz")
    else:
        stream = http_stream(hf_resolve_url(REPO, FILENAME, REVISION))
        tf = tarfile.open(fileobj=stream, mode="r|gz")

    from lhotse import CutSet

    (data_dir / "manifests").mkdir(parents=True, exist_ok=True)
    tmps = {s: f.with_name(f.name + ".tmp.jsonl.gz") for s, f in finals.items()}
    writers = {s: CutSet.open_writer(t) for s, t in tmps.items()}
    stats = {s: SplitStats("issai", s) for s in a.splits}
    texts: dict = {}  # (split, id) -> text
    audio: dict = {}  # (split, id) -> future(num_samples), flac path
    done_splits: set = set()
    unknown_dirs: set = set()
    pool = ThreadPoolExecutor(max_workers=max(1, a.num_workers))

    def emit(key):
        split, uid = key
        text = texts.pop(key)
        fut, flac = audio.pop(key)
        n = fut.result()
        seconds = n / 16000
        st = stats[split]
        if text is None:
            st.drop("text")
        elif not keep_duration(seconds) or not chars_per_second_ok(text, seconds):
            st.drop("duration_or_rate")
        else:
            writers[split].write(make_cut(f"issai_{split}_{uid}", flac, n, text, origin="issai"))
            st.add(seconds)
            if a.max_utts and st.utts >= a.max_utts:
                done_splits.add(split)
            return
        flac.unlink(missing_ok=True)

    try:
        for m in tf:
            if not m.isfile():
                continue
            parts = Path(m.name).parts
            if len(parts) < 3:
                continue
            split = split_of(parts[-2])
            if split is None and parts[-2] not in unknown_dirs:
                unknown_dirs.add(parts[-2])
                log.warning(f"unknown split directory '{parts[-2]}' in the archive -- skipped")
            if split not in stats or split in done_splits:
                if a.max_utts and len(done_splits) == len(stats):
                    break
                continue
            uid, ext = os.path.splitext(parts[-1])
            key = (split, uid)
            if ext == ".txt":
                raw = tf.extractfile(m).read().decode("utf-8", "replace").strip()
                if a.rejoin:
                    raw = rejoin_split_suffixes(raw)
                texts[key] = asr_text(raw)
            elif ext == ".wav":
                flac = data_dir / "audio" / "issai" / split / f"{uid}.flac"
                audio[key] = (pool.submit(_encode, tf.extractfile(m).read(), flac), flac)
            else:
                continue
            if key in texts and key in audio:
                emit(key)
    finally:
        tf.close()
        if stream is not None:
            stream.close()
        pool.shutdown(wait=True)
        for w in writers.values():
            w.close()

    for key, (fut, flac) in list(audio.items()):  # wav whose .txt never came
        fut.result()
        flac.unlink(missing_ok=True)
        stats[key[0]].drop("unpaired_wav" if not a.max_utts else "unpaired_at_early_stop")
    for split in a.splits:
        tmps[split].rename(finals[split])
        stats[split].save(data_dir)
        log.info(f"{split}: {stats[split].summary()}")
    if archive is not None and a.delete_archive:
        archive.unlink(missing_ok=True)
        log.info(f"deleted {archive}")


if __name__ == "__main__":
    main()
