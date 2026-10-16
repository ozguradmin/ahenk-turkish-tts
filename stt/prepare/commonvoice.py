#!/usr/bin/env python3
"""Mozilla Common Voice Turkish (CC0-1.0) from a LOCAL extracted release ->
16 kHz FLAC + lhotse cuts.

Common Voice is now distributed through the Mozilla Data Collective (login
required), so this script does not download anything. Point it at the
extracted language directory (standard layout):

    cv-corpus-XX.X-YYYY-MM-DD/tr/
        clips/*.mp3
        validated.tsv  train.tsv  dev.tsv  test.tsv  (other.tsv invalidated.tsv ...)

    python stt/prepare/commonvoice.py --cv-dir ~/cv-corpus-23.0-2025-09-05/tr --num-workers 6

Splits: dev/test = the official dev.tsv/test.tsv. Train (default
``--train-from validated``) = every validated clip whose sentence AND speaker
(client_id) do not occur in dev/test -- about 2x more data than train.tsv while
keeping dev/test speaker- and sentence-disjoint. ``--train-from train`` uses
the official train.tsv only. MP3 is decoded by libsndfile (soundfile wheel).
"""

from __future__ import annotations

import argparse
import csv
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
    keep_duration,
    make_cut,
    norm_gender,
    resample,
    setup_logging,
    stt_data_dir,
    write_flac,
)

log = setup_logging("commonvoice")
csv.field_size_limit(10**7)


def read_tsv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def find_lang_dir(p: Path) -> Path:
    if (p / "clips").is_dir():
        return p
    for cand in [p / "tr", *p.glob("*/tr")]:
        if (cand / "clips").is_dir():
            return cand
    raise SystemExit(f"no clips/ directory under {p}; pass the .../tr directory of the release")


def convert(task: dict) -> list:
    """Worker: mp3 -> 16 kHz FLAC. Returns (row index, num_samples | None, reason)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out = []
    for idx, mp3, flac in task["items"]:
        try:
            with open(mp3, "rb") as f:
                x, sr = decode_bytes(f.read())
            x = resample(x, sr)
            out.append((idx, write_flac(Path(flac), x), None))
        except Exception as e:  # corrupt/empty clip
            out.append((idx, None, type(e).__name__))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cv-dir", required=True)
    p.add_argument("--data-dir", default=None, help="default: $STT_DATA")
    p.add_argument("--train-from", choices=["validated", "train"], default="validated")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--max-utts", type=int, default=None, help="per split (smoke tests)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    cv = find_lang_dir(Path(a.cv_dir).expanduser())
    data_dir = stt_data_dir(a.data_dir)
    dev, test = read_tsv(cv / "dev.tsv"), read_tsv(cv / "test.tsv")
    held_sent = {r["sentence"].strip() for r in dev + test}
    held_spk = {r["client_id"] for r in dev + test}
    if a.train_from == "validated":
        train = [
            r for r in read_tsv(cv / "validated.tsv")
            if r["sentence"].strip() not in held_sent and r["client_id"] not in held_spk
        ]
    else:
        train = read_tsv(cv / "train.tsv")
    rows_by_split = {"train": train, "dev": dev, "test": test}
    log.info(f"{cv}: train={len(train)} ({a.train_from}) dev={len(dev)} test={len(test)}")

    from lhotse import CutSet

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.num_workers, mp_context=ctx) as pool:
        for split in a.splits:
            out = data_dir / "manifests" / f"cv_cuts_{split}_raw.jsonl.gz"
            if out.exists() and not a.overwrite:
                log.info(f"{out} exists, skipping")
                continue
            rows = rows_by_split[split]
            st = SplitStats("cv", split)
            keep, seen = [], set()
            for r in rows:
                text = asr_text(r.get("sentence"))
                if text is None:
                    st.drop("text")
                    continue
                if r["path"] in seen:
                    continue
                seen.add(r["path"])
                keep.append((r, text))
            if a.max_utts:
                keep = keep[: a.max_utts]
            audio_dir = data_dir / "audio" / "cv" / split
            items = [
                (i, str(cv / "clips" / r["path"]), str(audio_dir / (Path(r["path"]).stem + ".flac")))
                for i, (r, _) in enumerate(keep)
            ]
            chunks = [{"items": items[i: i + 64]} for i in range(0, len(items), 64)]
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp.jsonl.gz")
            with CutSet.open_writer(tmp) as w:
                for res in pool.map(convert, chunks):
                    for idx, n, err in res:
                        r, text = keep[idx]
                        flac = Path(items[idx][2])
                        if n is None:
                            st.drop(f"decode:{err}")
                            continue
                        seconds = n / 16000
                        if not keep_duration(seconds) or not chars_per_second_ok(text, seconds, lo=2.0):
                            st.drop("duration_or_rate")
                            flac.unlink(missing_ok=True)
                            continue
                        w.write(make_cut(
                            f"cv_{split}_{flac.stem}", flac, n, text, origin="cv",
                            speaker=f"cv_{r['client_id'][:16]}", gender=norm_gender(r.get("gender")),
                        ))
                        st.add(seconds)
            tmp.rename(out)
            st.save(data_dir)
            log.info(f"{split}: {st.summary()}")


if __name__ == "__main__":
    main()
