#!/usr/bin/env python3
"""LoRA fine-tuning of VoxCPM2 on the Antalia voice, with a hard time limit and resume.

    python tts/voxcpm/train_lora.py --model $TTS_DATA/models/VoxCPM2 --data $TTS_DATA/antalia16k \
        --exp $TTS_DATA/exp/voxcpm2-antalia --steps 1000 --hours 1.5

Writes <exp>/lora.yaml from the arguments (the official recipe's conf/voxcpm_v2/voxcpm_finetune_lora.yaml
with our paths and batch) and runs the official trainer (VoxCPM scripts/train_voxcpm_finetune.py).
The trainer resumes from <exp>/ckpt/latest on its own and saves a checkpoint when it gets SIGTERM;
after --hours this wrapper sends SIGTERM, so a run never outlives its budget and the next run
continues where it stopped. Checkpoints: <exp>/ckpt/step_XXXXXXX/ (lora_weights.safetensors,
lora_config.json, optimizer and scheduler state).

Two-stage training: --init-lora DIR (a checkpoint folder of another experiment, e.g. stage 1 on Alania) starts a new
experiment from DIR/lora_weights.safetensors: it is copied alone into <exp>/ckpt/latest/, so the trainer begins at step
0 with those weights and a fresh optimizer and LR schedule (it reads optimizer, scheduler and training_state only if
present). Only while <exp> has no checkpoint: after that its own checkpoints win (resume) and --init-lora is ignored.
DIR's LoRA rank must equal --rank.

Resume guard: an experiment that has checkpoints continues only with the corpus its last run used (<exp>/lora.yaml's
train_manifest; --switch-data allows another) and its checkpoint's LoRA rank (the weights would not load otherwise).

Exit code 0 when all --steps are done, 75 when stopped at the time limit (resumable), 2 for a bad --init-lora or a
resume the guard refuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

VOXCPM_SRC = Path(os.environ.get("VOXCPM_SRC", Path(__file__).resolve().parents[1] / "voxcpm-src"))
WEIGHTS = "lora_weights.safetensors"


def lora_settings(rank: int) -> dict:
    return {"enable_lm": True, "enable_dit": True, "enable_proj": False, "r": rank, "alpha": rank, "dropout": 0.0}


def write_config(args, exp: Path) -> Path:
    data = args.data.resolve()
    cfg = {
        "pretrained_path": str(Path(args.model).resolve()),
        "train_manifest": str(data / "voxcpm_train.jsonl"),
        "val_manifest": "",  # held-out scoring is done by synth_eval.py (real synthesis + ASR)
        "sample_rate": args.sample_rate,
        "out_sample_rate": 48000 if args.sample_rate == 16000 else args.sample_rate,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "num_workers": args.num_workers,
        "preprocessing_num_workers": max(1, args.num_workers),
        "num_iters": args.steps,
        "log_interval": 10,
        "valid_interval": 10**9,
        "save_interval": args.save_every,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "warmup_steps": args.warmup,
        "max_steps": args.steps,
        "max_batch_tokens": 8192,
        "max_grad_norm": 1.0,
        "save_path": str(exp / "ckpt"),
        "tensorboard": str(exp / "tb"),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": lora_settings(args.rank),
        "hf_model_id": args.hf_model_id,
        "distribute": True,  # lora_config.json names the hub model, so the checkpoint loads anywhere
    }
    path = exp / "lora.yaml"
    # JSON is valid YAML; argbind reads it with yaml.safe_load, whose YAML 1.1 floats need a dot: json's "5e-05"
    # would load as a string (AdamW then fails), so exponent floats are written "5.0e-05"
    text = json.dumps(cfg, indent=1, ensure_ascii=False)
    text = re.sub(r"(?<![\w.\"])(-?\d+)([eE][-+]?\d+)(?![\w.])", r"\1.0\2", text)
    path.write_text(text + "\n")
    return path


def last_step(exp: Path) -> int:
    steps = [int(m.group(1)) for p in (exp / "ckpt").glob("step_*") if (m := re.match(r"step_(\d+)$", p.name))]
    return max(steps, default=0)


def has_checkpoint(exp: Path) -> bool:
    """A step_* folder, or a ckpt/latest with LoRA weights (the trainer resumes from it). A latest without weights is
    the leftover of an --init-lora copy that was cut short."""
    ck = exp / "ckpt"
    return (any(p.is_dir() for p in ck.glob("step_*"))
            or any((ck / "latest" / n).is_file() for n in (WEIGHTS, "lora_weights.ckpt")))


def _read_json(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def ckpt_rank(latest_config: dict | None, init_record: dict | None) -> int | None:
    """LoRA rank of the checkpoint the trainer resumes from: ckpt/latest/lora_config.json, else (latest is an
    --init-lora copy, weights only) init_lora.json's r; None when neither says."""
    lc = (latest_config or {}).get("lora_config")
    for r in (lc.get("r") if isinstance(lc, dict) else None, (init_record or {}).get("r")):
        try:
            return int(r)
        except (TypeError, ValueError):
            continue
    return None


def resume_conflicts(prev_config: dict | None, rank_of_ckpt: int | None, manifest: str, rank: int,
                     switch_data: bool = False) -> list[str]:
    """Why an experiment WITH checkpoints must not resume with these settings: another corpus than its last run's
    (prev_config = that run's lora.yaml; allowed with switch_data) or another LoRA rank than its checkpoint's.
    Corpora compare by folder name (antalia16k, alania16k): the Volume's mount path may differ between machines."""
    out = []
    prev = (prev_config or {}).get("train_manifest")
    if isinstance(prev, str) and prev and Path(prev).parent.name != Path(manifest).parent.name and not switch_data:
        out.append(f"it was trained on {Path(prev).parent.as_posix()} (lora.yaml), this run's --data is "
                   f"{Path(manifest).parent.as_posix()}: keep that --data, use a new experiment, or add --switch-data")
    if rank_of_ckpt is not None and rank_of_ckpt != rank:
        out.append(f"its checkpoint is a rank {rank_of_ckpt} LoRA, this run's --rank is {rank}: add --rank "
                   f"{rank_of_ckpt} (the weights would not load)")
    return out


def init_lora(exp: Path, src: Path, rank: int) -> dict | None:
    """--init-lora: src/lora_weights.safetensors (only) -> <exp>/ckpt/latest/ when exp has no checkpoint; also writes
    <exp>/init_lora.json (source, its step, sha256). Returns that record, or None when exp has checkpoints (resume
    wins, src is not even read). ValueError when src has no weights / lora_config.json or another LoRA rank."""
    if has_checkpoint(exp):
        return None
    w, conf = src / WEIGHTS, src / "lora_config.json"
    for p in (w, conf):
        if not p.is_file():
            raise ValueError(f"--init-lora: no {p}")
    try:
        lcfg = json.loads(conf.read_text(encoding="utf-8"))["lora_config"]
        r = int(lcfg["r"])
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"--init-lora: no LoRA rank (lora_config.r) in {conf}: {e!r}")
    if r != rank:
        raise ValueError(f"--init-lora: {src} is a rank {r} LoRA, this run has --rank {rank}; pass --rank {r}")
    # strict=False load: weights of a module this run has no LoRA on are dropped, a missing module starts from zero
    if diff := {k: (lcfg.get(k), v) for k, v in lora_settings(rank).items() if k != "dropout" and lcfg.get(k) != v}:
        print(f"[train_lora] warning: --init-lora LoRA settings differ (theirs, ours): {diff}", flush=True)
    latest = exp / "ckpt" / "latest"
    shutil.rmtree(latest, ignore_errors=True)  # a cut-short copy
    latest.mkdir(parents=True)
    part, h = latest / (WEIGHTS + ".part"), hashlib.sha256()
    with open(w, "rb") as fi, open(part, "wb") as fo:
        while b := fi.read(1 << 20):
            h.update(b)
            fo.write(b)
    os.replace(part, latest / WEIGHTS)  # latest counts as a checkpoint only once the weights are complete
    try:
        step = json.loads((src / "training_state.json").read_text(encoding="utf-8")).get("step")
    except (OSError, ValueError, AttributeError):
        step = None
    rec = {"from": str(src), "step": step, "r": r, "sha256": h.hexdigest(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (exp / "init_lora.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


def build_parser() -> argparse.ArgumentParser:
    """The command line; modal_app.py parses --args with it before a GPU starts."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="VoxCPM2 snapshot directory")
    ap.add_argument("--data", type=Path, required=True, help="directory with voxcpm_train.jsonl")
    ap.add_argument("--exp", type=Path, required=True, help="experiment directory")
    ap.add_argument("--steps", type=int, default=1000, help="optimizer steps in total (also the LR schedule) [1000]")
    ap.add_argument("--hours", type=float, default=0.0, help="stop (and save) after this long; 0 = no limit")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rank", type=int, default=64, help="LoRA rank (VoxCPM guide: 32 speaker, 64 language/style; Trendyol-TTS used 64)")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--sample-rate", type=int, default=16000, help="AudioVAE encoder rate (VoxCPM2: 16000)")
    ap.add_argument("--hf-model-id", default="openbmb/VoxCPM2", help="base model named in lora_config.json")
    ap.add_argument("--init-lora", type=Path, default=None,
                    help="checkpoint folder whose lora_weights.safetensors a new experiment starts from (step 0)")
    ap.add_argument("--switch-data", action="store_true",
                    help="let an experiment with checkpoints continue on another --data than its last run's")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    exp = args.exp.resolve()
    exp.mkdir(parents=True, exist_ok=True)
    if has_checkpoint(exp):  # before write_config rewrites lora.yaml, the record of the last run's corpus
        latest = exp / "ckpt" / "latest" / "lora_config.json"
        if bad := resume_conflicts(_read_json(exp / "lora.yaml"),
                                   ckpt_rank(_read_json(latest), _read_json(exp / "init_lora.json")),
                                   str(args.data.resolve() / "voxcpm_train.jsonl"), args.rank, args.switch_data):
            print(f"[train_lora] {exp.name} has checkpoints, not resumed: {'; '.join(bad)}", file=sys.stderr,
                  flush=True)
            return 2
    init = None
    if args.init_lora:
        src = args.init_lora.resolve()
        try:
            init = init_lora(exp, src, args.rank)
        except ValueError as e:
            print(f"[train_lora] {e}", file=sys.stderr, flush=True)
            return 2
        if init:
            print(f"[train_lora] --init-lora: {src / WEIGHTS} (r={init['r']}, its step {init['step']}) -> "
                  f"{exp / 'ckpt' / 'latest'}: step 0 from these weights, fresh optimizer and LR schedule", flush=True)
        else:
            try:
                was = json.loads((exp / "init_lora.json").read_text(encoding="utf-8"))["from"]
            except (OSError, ValueError, KeyError, TypeError):
                was = "no --init-lora"
            print(f"[train_lora] {exp.name} already has checkpoints (started from {was}): resume wins, "
                  f"--init-lora {src} ignored", flush=True)
    cfg = write_config(args, exp)
    start = last_step(exp)
    if start >= args.steps:
        print(f"[train_lora] already at step {start} >= {args.steps}; nothing to do")
        return 0
    cmd = [sys.executable, str(VOXCPM_SRC / "scripts" / "train_voxcpm_finetune.py"), "--config_path", str(cfg)]
    print(f"[train_lora] {'resuming at step ' + str(start) if start else 'starting'}; "
          f"limit {args.hours or 'none'} h; $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    stopped = False
    try:
        while proc.poll() is None:
            if args.hours and time.time() - t0 > args.hours * 3600 and not stopped:
                print(f"[train_lora] time limit ({args.hours} h): SIGTERM -> trainer saves and exits", flush=True)
                proc.send_signal(signal.SIGTERM)
                stopped = True
            time.sleep(5)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)
        stopped = True
    rc = proc.wait()
    step = last_step(exp)
    info = {"step": step, "target_steps": args.steps, "seconds": round(time.time() - t0, 1), "rc": rc,
            "stopped_at_limit": stopped, "data": str(args.data.resolve())}
    if args.init_lora:
        info |= {"init_lora": str(args.init_lora.resolve()), "init_lora_applied": init is not None}
    with open(exp / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({**info, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    print(f"[train_lora] {json.dumps(info)}", flush=True)
    if rc != 0:
        return rc
    return 75 if stopped and step < args.steps else 0


if __name__ == "__main__":
    raise SystemExit(main())
