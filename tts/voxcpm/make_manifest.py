#!/usr/bin/env python3
"""VoxCPM fine-tuning manifests from a prepared Antalia directory (tts/prepare_antalia.py output).

    python tts/prepare_antalia.py --sample-rate 16000 --no-phonemes --out $TTS_DATA/antalia16k
    python tts/voxcpm/make_manifest.py --data $TTS_DATA/antalia16k

Writes <data>/voxcpm_train.jsonl and <data>/voxcpm_val.jsonl, one {"audio", "text", "duration"}
object per line (absolute paths, text = trnorm spoken form, the same text the inference front-end
produces). The held-out clips of prepare_antalia.py go to val and are never trained on.

--ref-fraction of the training rows also get a "ref_audio": another clip of the same speaker, so
the fine-tuned model keeps VoxCPM2's reference-audio path (the VoxCPM fine-tuning guide recommends
30-50%). References are drawn from training clips of 3-10 s, never the row's own clip.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="prepare_antalia.py output directory")
    ap.add_argument("--ref-fraction", type=float, default=0.3, help="training rows with a ref_audio [0.3]")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    data = args.data.resolve()
    clips = [json.loads(line) for line in open(data / "clips.jsonl", encoding="utf-8")]
    train = [c for c in clips if c["split"] == "train"]
    val = [c for c in clips if c["split"] == "heldout"]
    if not train:
        raise SystemExit(f"no training clips in {data}/clips.jsonl")
    for c in train + val:
        if not (data / "wav" / c["wav"]).is_file():
            raise SystemExit(f"missing audio: {data / 'wav' / c['wav']}")

    rng = random.Random(args.seed)
    refs = [c for c in train if 3.0 <= c["duration"] <= 10.0] or train

    def row(c, ref=None):
        r = {"audio": str(data / "wav" / c["wav"]), "text": c["text"], "duration": round(c["duration"], 3)}
        if ref is not None:
            r["ref_audio"] = str(data / "wav" / ref["wav"])
            r["ref_duration"] = round(ref["duration"], 3)
        return r

    n_ref = round(args.ref_fraction * len(train))
    with_ref = set(rng.sample(range(len(train)), n_ref)) if len(refs) > 1 else set()
    rows = []
    for i, c in enumerate(train):
        ref = None
        if i in with_ref:
            ref = rng.choice([r for r in refs if r["wav"] != c["wav"]])
        rows.append(row(c, ref))
    rng.shuffle(rows)

    for name, items in (("voxcpm_train.jsonl", rows), ("voxcpm_val.jsonl", [row(c) for c in val])):
        with open(data / name, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(c["duration"] for c in train) / 3600
    print(f"[manifest] {len(rows)} train ({hours:.2f} h, {len(with_ref)} with ref_audio), {len(val)} val -> {data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
