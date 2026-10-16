#!/usr/bin/env python3
"""Dedicated CPU speed probe: every system reads the same 10 already-spoken sentences.

    python benchmarks/tts/speed.py                                  # all systems, 1 and 2 threads
    python benchmarks/tts/speed.py --systems piper-tr-dfki --threads 2

Each (system, threads) pair runs in its own subprocess (torch's thread pool is process-wide):
load, one warm-up sentence, then the `spoken` forms of sets/dev_numbers10.jsonl one by one.
Reports RTF = wall seconds / audio seconds and CPU-RTF = process CPU seconds / audio seconds,
with the 1-minute load average at start and end (the machine is shared, so RTF is indicative).
Appends to results/speed/speed.jsonl and rewrites results/speed/SPEED.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from common import RESULTS_DIR, cpu_model, load_set, read_jsonl, write_text_atomic

OUT = RESULTS_DIR / "speed"


def probe(system: str, threads: int, set_: str) -> dict:
    os.environ["TTS_BENCH_THREADS"] = str(threads)
    import time

    import common
    from systems import load_system

    texts = [r["spoken"] for r in load_set(set_)]
    la0 = common.loadavg()
    t0 = time.perf_counter()
    tts = load_system(system, threads=threads)
    load_s = time.perf_counter() - t0
    tts.reset()
    tts.synthesize("Merhaba, bu cümle yalnızca modeli ısıtmak için okunuyor.")
    wall = cpu = dur = 0.0
    for t in texts:
        tts.reset()
        c0, w0 = time.process_time(), time.perf_counter()
        audio, sr = tts.synthesize(t)
        wall += time.perf_counter() - w0
        cpu += time.process_time() - c0
        dur += len(audio) / sr
    return {"system": system, "threads": threads, "set": set_, "n": len(texts), "audio_seconds": round(dur, 2),
            "rtf": round(wall / dur, 4), "cpu_rtf": round(cpu / dur, 4), "load_seconds": round(load_s, 1),
            "loadavg_start": la0, "loadavg_end": common.loadavg(), "cpu": cpu_model(),
            "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", "default"),
            "when": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def render(rows: list[dict]) -> str:
    latest = {}
    for r in rows:
        latest[(r["system"], r["threads"], r.get("omp_wait_policy", "PASSIVE"))] = r
    L = ["# CPU speed probe\n",
         "Same 10 spoken-form sentences (`sets/dev_numbers10.jsonl`, field `spoken`) for every system, "
         "after one warm-up sentence; latest run per (system, threads, OpenMP wait policy). RTF = wall "
         "seconds / audio seconds; CPU-RTF = process CPU seconds / audio seconds. The machine is shared "
         "with other jobs: compare load averages before reading much into small differences. "
         "`PASSIVE` (the harness default) avoids the large slowdown seen when the CPU is "
         "oversubscribed (EMA Lightning RTF 1.37 with the default vs 0.26 with PASSIVE at load ~9); at "
         "the light load of these probes the OpenMP `default` (spin, then sleep) was within about 20 %. "
         "pocket-tts runs its decoder in a second Python thread, so its CPU-RTF exceeds its RTF even "
         "at 1 torch thread.\n",
         "| system | threads | OMP wait policy | RTF | CPU-RTF | audio s | load avg (start-end) | when (UTC) |",
         "|---|---:|---|---:|---:|---:|---|---|"]
    for (s, t, pol), r in sorted(latest.items()):
        L.append(f"| {s} | {t} | {pol} | {r['rtf']:.3f} | {r['cpu_rtf']:.3f} | {r['audio_seconds']:.1f} | "
                 f"{r['loadavg_start']}-{r['loadavg_end']} | {r['when'][:16]} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    from systems import SYSTEMS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    ap.add_argument("--threads", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--set", default="dev_numbers10")
    ap.add_argument("--_child", nargs=2, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args._child:
        print(json.dumps(probe(args._child[0], int(args._child[1]), args.set)))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for system in args.systems:
        for threads in args.threads:
            env = {**os.environ, "TTS_BENCH_THREADS": str(threads), "OMP_NUM_THREADS": str(threads),
                   "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads)}
            if os.environ.get("TTS_BENCH_OMP_WAIT_POLICY", "").lower() == "default":
                env.pop("OMP_WAIT_POLICY", None)  # common.py set it in this parent process
            p = subprocess.run([sys.executable, __file__, "--set", args.set, "--_child", system, str(threads)],
                               capture_output=True, text=True, env=env)
            line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("{")), None)
            if p.returncode or line is None:
                print(f"{system} threads={threads}: FAILED\n{p.stderr[-2000:]}", file=sys.stderr)
                continue
            r = json.loads(line)
            with (OUT / "speed.jsonl").open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{system:20s} threads={threads} RTF={r['rtf']:.3f} CPU-RTF={r['cpu_rtf']:.3f} "
                  f"load {r['loadavg_start']}->{r['loadavg_end']}", flush=True)
    write_text_atomic(OUT / "SPEED.md", render(read_jsonl(OUT / "speed.jsonl")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
