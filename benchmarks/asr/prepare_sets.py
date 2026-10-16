#!/usr/bin/env python3
"""Build the Turkish ASR evaluation sets as JSONL manifests + 16 kHz mono WAVs.

Output layout (under <repo>/data/, which is gitignored):

    data/asr_eval/<set>/manifest.jsonl   one JSON object per utterance:
                                         {"id", "audio_path", "ref_text", "duration"}
    data/asr_eval/<set>/wav/<id>.wav     16 kHz, mono, PCM16
    data/asr_eval/<set>/info.json        source, pinned revision, license, size

`audio_path` is relative to the manifest's directory (run_asr.py resolves it).

Sets:
    fleurs_tr_test   google/fleurs tr_tr test split, all 743 utterances (CC-BY-4.0).
                     ref_text = the RAW written transcription (cased, punctuated);
                     normalisation happens at scoring time (trnorm.normalize_for_eval).
    mediaspeech_tr   ymoslem/MediaSpeech "tr" (CC-BY-4.0), a seeded random subset of
                     500 utterances (default seed 1234) out of ~2.5k (~10 h).

Usage:
    python benchmarks/asr/prepare_sets.py                 # both sets
    python benchmarks/asr/prepare_sets.py --sets fleurs_tr_test --force
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import random
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = DATA_DIR / "asr_eval"
DL_DIR = DATA_DIR / "downloads"
SR = 16_000

# Pinned dataset revisions (HF commit SHAs) so the sets are reproducible.
FLEURS_REPO = "google/fleurs"
FLEURS_REV = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
MS_REPO = "ymoslem/MediaSpeech"
MS_REV = "4008a968760f2187b0c5b2b2db965f1283433059"
MS_FILES = ["tr/train-00000-of-00002.parquet", "tr/train-00001-of-00002.parquet"]

log = logging.getLogger("prepare_sets")


def hf_download(repo_id: str, filename: str, revision: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=str(dest),
        )
    )
    log.info("downloaded %s/%s (%.1f MB) in %.0fs", repo_id, filename,
             path.stat().st_size / 1e6, time.time() - t0)
    return path


def to_16k_mono(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR).astype(np.float32)
    return audio


def write_wav(path: Path, audio: np.ndarray) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), SR, subtype="PCM_16")
    return len(audio) / SR


def write_manifest(out_dir: Path, rows: list[dict], info: dict) -> None:
    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    durs = [r["duration"] for r in rows]
    info.update(
        n_utts=len(rows),
        total_hours=round(sum(durs) / 3600, 3),
        mean_duration_s=round(float(np.mean(durs)), 2),
        max_duration_s=round(float(np.max(durs)), 2),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    (out_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    log.info("%s: %d utts, %.2f h -> %s", info["set"], len(rows), info["total_hours"],
             out_dir / "manifest.jsonl")


def prepare_fleurs(keep_downloads: bool) -> None:
    name = "fleurs_tr_test"
    out_dir = EVAL_DIR / name
    dl = DL_DIR / "fleurs_tr"
    tsv = hf_download(FLEURS_REPO, "data/tr_tr/test.tsv", FLEURS_REV, dl)
    # FLEURS tsv columns: id, file_name, raw_transcription, transcription(normalised),
    # chars, num_samples, gender.  We keep the RAW transcription as the reference.
    entries = []
    with open(tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            sent_id, file_name, raw = row[0], row[1], row[2]
            entries.append((sent_id, file_name, raw.strip()))
    log.info("%s: %d rows in test.tsv", name, len(entries))
    by_file = {fn: (sid, raw) for sid, fn, raw in entries}
    assert len(by_file) == len(entries), "duplicate file names in test.tsv"

    tar_path = hf_download(FLEURS_REPO, "data/tr_tr/audio/test.tar.gz", FLEURS_REV, dl)
    durations: dict[str, float] = {}
    with tarfile.open(tar_path, mode="r|gz") as tf:  # streamed; nothing extracted to disk
        for member in tf:
            if not member.isfile():
                continue
            base = Path(member.name).name
            if base not in by_file:
                continue
            sid, _ = by_file[base]
            utt_id = f"{sid}_{Path(base).stem}"
            audio, sr = sf.read(io.BytesIO(tf.extractfile(member).read()), dtype="float32",
                                always_2d=True)
            durations[base] = write_wav(out_dir / "wav" / f"{utt_id}.wav", to_16k_mono(audio, sr))
            if len(durations) % 100 == 0:
                log.info("%s: %d/%d wavs written", name, len(durations), len(entries))
    missing = [fn for _, fn, _ in entries if fn not in durations]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} audio files missing from the tarball, "
                         f"e.g. {missing[:3]}")
    rows = [
        {"id": f"{sid}_{Path(fn).stem}", "audio_path": f"wav/{sid}_{Path(fn).stem}.wav",
         "ref_text": raw, "duration": round(durations[fn], 3)}
        for sid, fn, raw in entries
    ]
    write_manifest(out_dir, rows, {
        "set": name, "source": f"https://huggingface.co/datasets/{FLEURS_REPO}",
        "config": "tr_tr", "split": "test", "revision": FLEURS_REV,
        "license": "CC-BY-4.0", "ref_field": "raw_transcription",
    })
    if not keep_downloads:
        tar_path.unlink()
        log.info("deleted %s", tar_path)


def prepare_mediaspeech(n: int, seed: int, keep_downloads: bool) -> None:
    import pyarrow.parquet as pq

    name = "mediaspeech_tr"
    out_dir = EVAL_DIR / name
    dl = DL_DIR / "mediaspeech_tr"
    paths = [hf_download(MS_REPO, fn, MS_REV, dl) for fn in MS_FILES]

    # Pass 1: transcripts only, to define the global row order and draw the subset.
    sentences: list[str] = []
    file_of_row: list[tuple[int, int]] = []  # global row -> (file index, row within file)
    for fi, p in enumerate(paths):
        col = pq.read_table(p, columns=["sentence"]).column("sentence").to_pylist()
        for ri, s in enumerate(col):
            sentences.append(s or "")
            file_of_row.append((fi, ri))
    candidates = [i for i, s in enumerate(sentences) if s.strip()]
    log.info("%s: %d rows, %d with non-empty transcripts", name, len(sentences), len(candidates))
    chosen = sorted(random.Random(seed).sample(candidates, n))

    # Pass 2: decode audio for the chosen rows only.
    rows = []
    for fi, p in enumerate(paths):
        wanted = [(g, file_of_row[g][1]) for g in chosen if file_of_row[g][0] == fi]
        if not wanted:
            continue
        table = pq.read_table(p, columns=["audio", "sentence"])
        audio_col = table.column("audio")
        for g, ri in wanted:
            cell = audio_col[ri].as_py()
            audio, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32", always_2d=True)
            src_name = Path(cell.get("path") or f"row{g}").stem
            utt_id = f"ms{g:05d}_{src_name}"
            dur = write_wav(out_dir / "wav" / f"{utt_id}.wav", to_16k_mono(audio, sr))
            rows.append({"id": utt_id, "audio_path": f"wav/{utt_id}.wav",
                         "ref_text": sentences[g].strip(), "duration": round(dur, 3)})
            if len(rows) % 100 == 0:
                log.info("%s: %d/%d wavs written", name, len(rows), n)
        del table, audio_col
    write_manifest(out_dir, rows, {
        "set": name, "source": f"https://huggingface.co/datasets/{MS_REPO}", "config": "tr",
        "split": "train (MediaSpeech has a single split; used as test data)",
        "revision": MS_REV, "license": "CC-BY-4.0", "subset": f"random.Random({seed}).sample("
        f"non-empty rows, {n}), sorted by global row index", "seed": seed,
        "n_rows_total": len(sentences),
    })
    if not keep_downloads:
        shutil.rmtree(dl)
        log.info("deleted %s", dl)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["fleurs_tr_test", "mediaspeech_tr"],
                    choices=["fleurs_tr_test", "mediaspeech_tr"])
    ap.add_argument("--mediaspeech-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--keep-downloads", action="store_true",
                    help="keep the tarball / parquet files in data/downloads/")
    ap.add_argument("--force", action="store_true", help="rebuild even if manifest.jsonl exists")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    for name in args.sets:
        out_dir = EVAL_DIR / name
        if (out_dir / "manifest.jsonl").exists() and not args.force:
            log.info("%s: manifest exists, skipping (use --force to rebuild)", name)
            continue
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        if name == "fleurs_tr_test":
            prepare_fleurs(args.keep_downloads)
        else:
            prepare_mediaspeech(args.mediaspeech_n, args.seed, args.keep_downloads)


if __name__ == "__main__":
    main()
