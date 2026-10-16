#!/usr/bin/env python3
"""Measure real training speed from icefall's log and estimate time per epoch.

    python stt/tools/eta.py --exp-dir $STT_DATA/exp/small-causal-bpe1000

Reads the newest log/log-train-* file, takes the logged batches (each line has
"over N frames" at 25 frames/s after subsampling) and their timestamps, and
prints the real-time factor (seconds of audio trained per wall-clock second)
plus hours per epoch for the current training mix ($STT_DATA/stats/mix.json).
Validation and checkpoint pauses are included, thermal-guard pauses too.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO \[train\.py:\d+\] Epoch (\d+), batch (\d+), "
                  r"loss\[.*?over ([\d.]+) frames")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--last", type=int, default=50, help="use the last N logged points")
    a = p.parse_args()
    logs = sorted((Path(a.exp_dir) / "log").glob("log-train-*"), key=os.path.getmtime)
    if not logs:
        raise SystemExit("no log-train-* yet")
    pts = []
    for line in logs[-1].read_text(errors="replace").splitlines():
        m = LINE.match(line)
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            pts.append((t, int(m.group(2)), int(m.group(3)), float(m.group(4))))
    pts = pts[-a.last:]
    if len(pts) < 3:
        raise SystemExit("need at least 3 logged batches; wait a bit")
    (t0, _, b0, _), (t1, _, b1, _) = pts[0], pts[-1]
    wall = (t1 - t0).total_seconds()
    batches = b1 - b0
    if wall <= 0 or batches <= 0:
        raise SystemExit("not enough progress inside one epoch yet")
    audio_per_batch = sum(f for *_, f in pts) / len(pts) / 25.0  # seconds
    rtf = batches * audio_per_batch / wall
    print(f"log: {logs[-1].name}")
    print(f"{batches} batches in {wall / 60:.1f} min, ~{audio_per_batch:.0f} s audio/batch "
          f"-> {rtf:.0f}x real time, {wall / batches:.2f} s/batch")
    stats = Path(os.environ.get("STT_DATA", Path(__file__).resolve().parents[2] / "data" / "stt")) / "stats" / "mix.json"
    if stats.exists():
        hours = json.loads(stats.read_text())["train_hours"]
        print(f"training mix {hours:.0f} h -> {hours / rtf:.1f} h per epoch (plus validation/checkpoint overhead)")


if __name__ == "__main__":
    main()
