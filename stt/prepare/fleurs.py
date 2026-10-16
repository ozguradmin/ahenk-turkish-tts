#!/usr/bin/env python3
"""FLEURS Turkish (google/fleurs, config tr_tr; CC-BY-4.0) -> lhotse cuts.

    python stt/prepare/fleurs.py                       # all splits, full download (~2.1 GB)
    python stt/prepare/fleurs.py --max-utts 300 --splits train   # smoke test: streams
                                                       # only the first part of the tarball

Outputs: audio/fleurs/<split>/*.flac and manifests/fleurs_cuts_<split>_raw.jsonl.gz
FLEURS has no speaker ids; gender is kept. Transcripts: trnorm(raw_transcription).
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
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
    resample,
    setup_logging,
    stt_data_dir,
    write_flac,
)

REPO = "google/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"  # 2026-05-15
LANG = "tr_tr"

log = setup_logging("fleurs")


def read_tsv(path: Path) -> dict:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            sid, fname, raw, norm, _chars, _nsamp, gender = parts[:7]
            rows[fname] = {"sentence_id": sid, "raw": raw, "norm": norm, "gender": gender.lower()}
    return rows


def fetch(path_in_repo: str, out_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            REPO, path_in_repo, repo_type="dataset", revision=REVISION, local_dir=out_dir
        )
    )


def process_split(split: str, data_dir: Path, max_utts: int | None, keep_archive: bool, overwrite: bool):
    out = data_dir / "manifests" / f"fleurs_cuts_{split}_raw.jsonl.gz"
    if out.exists() and not overwrite:
        log.info(f"{out} exists, skipping (use --overwrite to redo)")
        return
    dl = data_dir / "downloads" / "fleurs"
    tsv = fetch(f"data/{LANG}/{split}.tsv", dl)
    rows = read_tsv(tsv)
    log.info(f"{split}: {len(rows)} rows in {tsv.name}")

    archive_name = f"data/{LANG}/audio/{split}.tar.gz"
    if max_utts:
        # partial: stream the tarball over HTTP and stop early (no full download)
        stream = http_stream(hf_resolve_url(REPO, archive_name, REVISION))
        tf = tarfile.open(fileobj=stream, mode="r|gz")
        archive_path = None
    else:
        archive_path = fetch(archive_name, dl)  # resumable
        stream = None
        tf = tarfile.open(archive_path, mode="r|gz")

    from lhotse import CutSet

    audio_dir = data_dir / "audio" / "fleurs" / split
    stats = SplitStats("fleurs", split)
    seen = set()
    tmp_out = out.with_name(out.name + ".tmp.jsonl.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with CutSet.open_writer(tmp_out) as writer:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".wav"):
                continue
            fname = Path(member.name).name
            row = rows.get(fname)
            if row is None or fname in seen:
                continue
            seen.add(fname)
            text = asr_text(row["raw"]) or asr_text(row["norm"])
            if text is None:
                stats.drop("text")
                continue
            x, sr = decode_bytes(tf.extractfile(member).read())
            x = resample(x, sr)
            seconds = len(x) / 16000
            if not keep_duration(seconds) or not chars_per_second_ok(text, seconds):
                stats.drop("duration_or_rate")
                continue
            stem = Path(fname).stem
            flac = audio_dir / f"{stem}.flac"
            n = write_flac(flac, x)
            cut = make_cut(
                f"fleurs_{split}_{stem}",
                flac,
                n,
                text,
                origin="fleurs",
                gender=row["gender"] if row["gender"] in ("male", "female") else None,
                custom={"sentence_id": row["sentence_id"]},
            )
            writer.write(cut)
            stats.add(seconds)
            if max_utts and stats.utts >= max_utts:
                break
    tf.close()
    if stream is not None:
        stream.close()
    tmp_out.rename(out)
    stats.save(data_dir)
    log.info(f"{split}: {stats.summary()}")
    if archive_path is not None and not keep_archive:
        archive_path.unlink(missing_ok=True)
        log.info(f"deleted {archive_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="default: $STT_DATA")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--max-utts", type=int, default=None, help="per split; streams instead of downloading")
    p.add_argument("--keep-archive", action="store_true", help="keep the downloaded tar.gz")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    data_dir = stt_data_dir(a.data_dir)
    for split in a.splits:
        process_split(split, data_dir, a.max_utts, a.keep_archive, a.overwrite)


if __name__ == "__main__":
    main()
