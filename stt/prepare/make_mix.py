#!/usr/bin/env python3
"""Combine per-dataset fbank manifests into the training/dev mixtures.

    python stt/prepare/make_mix.py                       # all sources found under fbank/
    python stt/prepare/make_mix.py --sources fleurs,issai,cv,alania --max-synthetic-ratio 0.45

* train: every <src>_cuts_train.jsonl.gz (or <src>/*_cuts_train.jsonl.gz for
  sharded sources), duration filter [min, max] s, synthetic sources capped to
  --max-synthetic-ratio of the total hours (random, seeded subsample), then a
  global line-level shuffle (the bucketing sampler only shuffles inside a
  buffer, so the input order must already be mixed).
* dev: each source's dev cuts, subsampled to --dev-hours-per-source, combined.
* test sets stay per source (fbank/<src>_cuts_test.jsonl.gz).
Writes fbank/train_mix_cuts.jsonl.gz, fbank/dev_mix_cuts.jsonl.gz, stats/mix.json.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with python -I
from common import setup_logging, stt_data_dir  # noqa: E402

SYNTHETIC = {"alania"}
ALL_SOURCES = ["fleurs", "issai", "cv", "alania", "audiobooks", "yodas"]
log = setup_logging("mix")


def manifests_for(fbank: Path, src: str, split: str) -> list:
    single = fbank / f"{src}_cuts_{split}.jsonl.gz"
    sharded = sorted((fbank / src).glob(f"*_cuts_{split}.jsonl.gz")) if (fbank / src).is_dir() else []
    return ([single] if single.exists() else []) + sharded


def read_lines(paths: list, min_d: float, max_d: float) -> list:
    out = []
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                dur = float(d["duration"])
                if min_d <= dur <= max_d and d.get("features"):
                    out.append((dur, line if line.endswith("\n") else line + "\n"))
    return out


def write_lines(path: Path, lines: list) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.writelines(lines)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="default: $STT_DATA")
    p.add_argument("--sources", default=",".join(ALL_SOURCES))
    p.add_argument("--max-synthetic-ratio", type=float, default=0.45)
    p.add_argument("--dev-hours-per-source", type=float, default=0.5)
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    data_dir = stt_data_dir(a.data_dir)
    fbank = data_dir / "fbank"
    rng = random.Random(a.seed)

    train, dev, stats = {}, [], {"sources": {}}
    for src in [s for s in a.sources.split(",") if s]:
        tr = manifests_for(fbank, src, "train")
        dv = manifests_for(fbank, src, "dev")
        if not tr and not dv:
            log.warning(f"{src}: no fbank manifests, skipped")
            continue
        train[src] = read_lines(tr, a.min_duration, a.max_duration)
        dlines = read_lines(dv, a.min_duration, a.max_duration)
        rng.shuffle(dlines)
        budget, kept, kept_s = a.dev_hours_per_source * 3600, [], 0.0
        for dur, line in dlines:
            if budget <= 0:
                break
            kept.append(line)
            budget -= dur
            kept_s += dur
        dev.extend(kept)
        stats["sources"][src] = {
            "train_hours_available": round(sum(d for d, _ in train[src]) / 3600, 3),
            "train_utts_available": len(train[src]),
            "dev_utts": len(kept),
            "dev_hours": round(kept_s / 3600, 3),
            "synthetic": src in SYNTHETIC,
        }

    real_h = sum(sum(d for d, _ in v) for s, v in train.items() if s not in SYNTHETIC) / 3600
    syn_h = sum(sum(d for d, _ in v) for s, v in train.items() if s in SYNTHETIC) / 3600
    r = a.max_synthetic_ratio
    cap_h = real_h * r / (1 - r) if r < 1 else float("inf")
    if syn_h > cap_h:
        scale = cap_h / syn_h
        log.info(f"synthetic {syn_h:.1f} h > cap {cap_h:.1f} h ({r:.0%} of total): keeping {scale:.1%}")
        for s in SYNTHETIC & train.keys():
            rng.shuffle(train[s])
            budget, kept = sum(d for d, _ in train[s]) * scale, []
            for dur, line in train[s]:
                if budget - dur < 0:
                    break
                kept.append((dur, line))
                budget -= dur
            train[s] = kept
    lines = [line for v in train.values() for _, line in v]
    rng.shuffle(lines)
    for s, v in train.items():
        stats["sources"][s]["train_hours_used"] = round(sum(d for d, _ in v) / 3600, 3)
        stats["sources"][s]["train_utts_used"] = len(v)
    total_h = sum(st["train_hours_used"] for st in stats["sources"].values() if "train_hours_used" in st)
    used_syn = sum(stats["sources"][s]["train_hours_used"] for s in SYNTHETIC & train.keys())
    stats.update({
        "train_hours": round(total_h, 3),
        "train_utts": len(lines),
        "synthetic_ratio": round(used_syn / total_h, 4) if total_h else 0.0,
        "dev_utts": len(dev),
        "duration_filter": [a.min_duration, a.max_duration],
    })
    if not lines:
        raise SystemExit("no training cuts found -- run the prepare + fbank stages first")
    write_lines(fbank / "train_mix_cuts.jsonl.gz", lines)
    write_lines(fbank / "dev_mix_cuts.jsonl.gz", dev)
    (data_dir / "stats").mkdir(parents=True, exist_ok=True)
    (data_dir / "stats" / "mix.json").write_text(json.dumps(stats, indent=2))
    log.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
