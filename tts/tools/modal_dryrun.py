#!/usr/bin/env python3
"""Run the container command lines of tts/modal_app.py on this machine, on CPU, with local paths.

Modal itself cannot be reached from every machine (and costs money), but everything a Modal
container does is a bash line built by modal_app.container_cmd() from prepare_cmds(),
probe_cmds(), train_cmd() and export_cmd(). This script imports those functions and runs the very
same lines locally:

  * the repo is copied with modal_app._IGNORE (what the Modal mount would contain) to <work>/repo,
    and the container's writable copy goes to <work>/work (same `cp -a` + CRLF stripping);
  * /data -> --data, /opt/venv-piper -> .venv-piper, /opt/piper1-gpl -> tts/piper1-gpl;
  * CPU-sized extras are appended (a 40-clip subset, --allow-cpu, small batches, 2 threads); the
    rest of each line is byte-for-byte what the container runs.

    .venv/bin/python tts/tools/modal_dryrun.py --steps prepare,train,probe,export     # needs `modal` importable

Afterwards it prints the probe summary with modal_app._fmt_probe, as `--action probe` would.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_app():
    spec = importlib.util.spec_from_file_location("tts_modal_app", REPO / "tts" / "modal_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ignored(rel: str, patterns: list[str]) -> bool:
    parts = rel.split("/")
    for pat in patterns:
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(parts[0], pat):
            return True
        if pat.startswith("**/") and fnmatch.fnmatch(parts[-1], pat[3:]):
            return True
        if "/" in pat and not pat.startswith("**/") and (rel == pat or rel.startswith(pat + "/")):
            return True
        if pat.startswith("benchmarks/**/") and rel.startswith("benchmarks/") and f"/{pat.split('/')[-1]}/" in f"/{rel}/":
            return True
    return False


def mount_copy(dst: Path, patterns: list[str]) -> int:
    """Copy the repo the way the Modal mount sees it (honouring the app's ignore list)."""
    if dst.exists():
        shutil.rmtree(dst)
    n = 0
    for root, dirs, files in os.walk(REPO):
        rel_root = Path(root).relative_to(REPO).as_posix()
        rel_root = "" if rel_root == "." else rel_root
        dirs[:] = [d for d in dirs if not ignored(f"{rel_root}/{d}".lstrip("/"), patterns)]
        for f in files:
            rel = f"{rel_root}/{f}".lstrip("/")
            if ignored(rel, patterns):
                continue
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(root) / f, dst / rel)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="prepare,train,probe,export")
    ap.add_argument("--data", type=Path, default=REPO / "data" / "tts_train" / "modal_dryrun")
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "tts_modal_dryrun")
    ap.add_argument("--reuse-base", type=Path, default=REPO / "data" / "tts_train" / "base" / "lj-med_1000.ckpt",
                    help="link this already-downloaded base checkpoint instead of downloading it again")
    args = ap.parse_args()

    app = load_app()
    data = args.data.resolve()
    work = args.work.resolve()
    data.mkdir(parents=True, exist_ok=True)
    n = mount_copy(work / "repo", app._IGNORE)
    print(f"[dryrun] mount emulation: {n} files in {work / 'repo'}", flush=True)
    paths = dict(repo=str(work / "repo"), work=str(work / "work"), venv=str(REPO / ".venv-piper"),
                 piper_dir=str(REPO / "tts" / "piper1-gpl"), data=str(data))
    if args.reuse_base.exists() and not (data / "base" / "lj-med_1000.ckpt").exists():
        (data / "base").mkdir(parents=True, exist_ok=True)
        os.symlink(args.reuse_base, data / "base" / "lj-med_1000.ckpt")

    exp = f"{data}/exp/antalia-medium"
    cpu_train = "--allow-cpu --batch-size 2 --num-workers 0 -- --trainer.limit_train_batches 2"
    steps = {
        "prepare": [c.replace("--piper-cache ", "--piper-cache --limit 40 --heldout 4 --workers 2 ", 1)
                    if c.startswith("python tts/prepare_antalia.py") else c for c in app.prepare_cmds()],
        "train": [app.train_cmd(app.PRESET["A100-80GB"], 0.05, 2, exp, cpu_train)],
        "probe": app.probe_cmds(app.PRESET["A100-80GB"], 1.5, f"{data}/exp/probe-cpu", f"{work}/step_bench.jsonl",
                                batches=(1,), extra=cpu_train),
        "export": [app.export_cmd(exp, "", "--n 3", threads=2)],
    }
    rc_all = 0
    for step in args.steps.split(","):
        for cmd in steps[step]:
            line = app.container_cmd(cmd, 2, **paths)
            print(f"\n[dryrun:{step}] $ {cmd}", flush=True)
            t = time.time()
            rc = subprocess.run(["bash", "-c", line]).returncode
            print(f"[dryrun:{step}] exit {rc} after {time.time() - t:.0f} s", flush=True)
            if rc not in (0, 75):
                rc_all = rc
        if step == "probe":
            bench = Path(work / "step_bench.jsonl")
            ep = Path(data) / "exp" / "probe-cpu" / "epochs.jsonl"
            raw = json.dumps({"gpu": "local CPU (dry run)",
                              "step_bench": [json.loads(x) for x in bench.read_text().splitlines() if x.startswith("{")]
                              if bench.exists() else [],
                              "epochs": [json.loads(x) for x in ep.read_text().splitlines() if x.strip()] if ep.exists() else []})
            print(app._fmt_probe(raw, app.GPU_PRICE["A100-80GB"] + app.GPU_SIDE_PRICE), flush=True)
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
