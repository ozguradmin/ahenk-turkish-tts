#!/usr/bin/env python3
"""Fine-tune EMA Lightning's acoustic model on the f022 latents (P1; design in docs/KUCUK_MODEL_P0.md section 4), on
the local RTX 5060, in chunks: every call resumes from the run's newest checkpoint and stops after --max-minutes
(or at --steps), saving a checkpoint first.

    <venv-student>/python tts/small/train.py --run data/small/runs/ft-a --init ema --steps 20000 --max-minutes 25

Data: data/small/f022-latents (index.jsonl use + train; latents.npy memory-mapped), length-bucketed batches under a
frame budget (student_data.bucket_batches), built one batch ahead on a background thread. Model: student.StudentAcoustic
(EMA's Acoustic + zero-init style embedding), init ema.pt's EMA weights or scratch. Losses: student.losses (flow +
duration, condition dropout, style dropout). AdamW, warm-up + cosine (student_data.lr_at), grad clip, an EMA of the
weights. Every --val-every steps: validation loss on the 61 val clips with fixed noise / t / no dropout, for the
trained weights and their EMA. Checkpoint RUN/ckpt_<step>.pt (model, ema, optimiser, RNG, data position); kept: the
newest two and every --keep-every. RUN/log.jsonl: train (every --log-every) and val records; RUN/args.json.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0common as C  # noqa: E402
import student_data as D  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import student as S  # noqa: E402


class Batches:
    """Endless batches in a fixed order per (seed, epoch); resumable at (epoch, position); one batch prefetched."""

    def __init__(self, rows, lat, budget, max_batch, seed, epoch=0, pos=0):
        self.rows, self.lat, self.budget, self.max_batch, self.seed = rows, lat, budget, max_batch, seed
        self.epoch, self.pos = epoch, pos
        self.q: queue.Queue = queue.Queue(maxsize=2)
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def plan(self, epoch):
        return D.bucket_batches(self.rows, self.budget, self.max_batch, random.Random(self.seed * 100003 + epoch))

    def _run(self):
        epoch, pos = self.epoch, self.pos
        order = self.plan(epoch)
        while not self.stop:
            if pos >= len(order):
                epoch, pos = epoch + 1, 0
                order = self.plan(epoch)
            b = D.collate([self.rows[i] for i in order[pos]], self.lat)
            pos += 1
            self.q.put((b, epoch, pos))

    def next(self):
        b, self.epoch, self.pos = self.q.get()
        return b


def val_loss(model, val_batches, dev, ts=(0.1, 0.3, 0.5, 0.7, 0.9)) -> dict:
    """Mean flow loss over fixed t values and fixed noise (seed 0), and duration loss; no dropout."""
    model.eval()
    fl, dl, n = 0.0, 0.0, 0
    with torch.no_grad():
        for bi, b in enumerate(val_batches):
            bt = S.to_device(b, dev)
            B = bt["ids"].shape[0]
            for k, t in enumerate(ts):
                g = torch.Generator(device=dev).manual_seed(1000 * bi + k)
                lf, ld = S.losses(model, bt, p_cond=0.0, p_style=0.0, t_dist=torch.full((B,), t), gen=g)
                fl += float(lf) * B
                dl += float(ld) * B
                n += B
    model.train()
    return {"flow": round(fl / n, 5), "dur": round(dl / n, 5)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder (checkpoints, log.jsonl, args.json)")
    ap.add_argument("--data", default="data/small/f022-latents")
    ap.add_argument("--init", default="ema", choices=["ema", "scratch"])
    ap.add_argument("--steps", type=int, default=20000, help="total steps of the run (the cosine schedule's length)")
    ap.add_argument("--max-minutes", type=float, default=25.0, help="this call stops (and saves) after this long")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--frame-budget", type=int, default=4000, help="batch size x longest clip, in 25 Hz frames")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--p-cond", type=float, default=0.1)
    ap.add_argument("--p-style", type=float, default=0.1)
    ap.add_argument("--t-dist", default="uniform", choices=["uniform", "logitnormal"])
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--keep-every", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (keep the PC usable)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    run = Path(args.run) if Path(args.run).is_absolute() else C.REPO / args.run
    run.mkdir(parents=True, exist_ok=True)
    data = C.REPO / args.data
    rows = D.read_index(data, "train")
    val_rows = D.read_index(data, "val")
    for r in rows + val_rows:
        D.check_row(r)
    lat = np.load(data / "latents.npy", mmap_mode="r")
    val_rows.sort(key=lambda r: r["frames"])
    val_batches = [D.collate(val_rows[i: i + 16], lat) for i in range(0, len(val_rows), 16)]

    model, meta = S.build(args.init, dev)
    model.train()
    ema = S.build("scratch", dev)[0].eval().requires_grad_(False)
    ema.load_state_dict(model.state_dict())
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.99))
    step, epoch, pos = 0, 0, 0
    last = D.latest_ckpt(run)
    if last is not None:
        ck = torch.load(str(last), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, epoch, pos = ck["step"], ck["epoch"], ck["pos"]
        torch.set_rng_state(ck["rng_cpu"])
        torch.cuda.set_rng_state(ck["rng_cuda"])
        print(f"resumed {last.name}: step {step}, epoch {epoch}, batch {pos}", flush=True)
    else:
        torch.manual_seed(args.seed)
        a = vars(args) | {"train_clips": len(rows), "val_clips": len(val_rows), "gpu": C.gpu_name(),
                          "torch": torch.__version__, "params_acoustic_M": round(sum(p.numel() for p in params) / 1e6, 3)}
        (run / "args.json").write_text(json.dumps(a, indent=1) + "\n", encoding="utf-8")
        if args.init == "ema":
            v = val_loss(model, val_batches, dev)
            D.append_jsonl(run / "log.jsonl", {"kind": "val", "step": 0, "model": v, "ema": v})
            print(f"val at init: {v}", flush=True)
    if step >= args.steps:
        print(f"run already at step {step} >= {args.steps}; nothing to do")
        return 0

    def save():
        ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
              "epoch": epoch, "pos": pos, "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
              "cfg": meta["cfg"], "vocab": meta["vocab"], "styles": list(D.STYLES), "init": args.init,
              "args": vars(args)}
        tmp = run / (D.ckpt_name(step) + ".tmp")
        torch.save(ck, str(tmp))
        os.replace(tmp, run / D.ckpt_name(step))
        D.prune_ckpts(run, keep=2, keep_every=args.keep_every)
        print(f"saved {D.ckpt_name(step)}", flush=True)

    feed = Batches(rows, lat, args.frame_budget, args.max_batch, args.seed, epoch, pos)
    ema_p = list(ema.parameters())
    mod_p = list(model.parameters())
    t_start = time.time()
    acc = {"flow": 0.0, "dur": 0.0, "gn": 0.0, "n": 0, "frames": 0}
    torch.cuda.reset_peak_memory_stats()
    t_log = time.time()
    while step < args.steps:
        b = feed.next()
        epoch, pos = feed.epoch, feed.pos
        bt = S.to_device(b, dev)
        lr = D.lr_at(step, args.lr, args.warmup, args.steps, args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr
        l_flow, l_dur = S.losses(model, bt, args.p_cond, args.p_style, args.t_dist)
        loss = l_flow + l_dur
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(mod_p, args.clip)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_p, mod_p, 1 - args.ema_decay)
        step += 1
        acc["flow"] += l_flow.item()
        acc["dur"] += l_dur.item()
        acc["gn"] += gn.item()
        acc["n"] += 1
        acc["frames"] += int(b["frames"].sum())
        if not np.isfinite(float(loss)):
            raise FloatingPointError(f"loss {float(loss)} at step {step}")
        if step % args.log_every == 0:
            n = acc["n"]
            rec = {"kind": "train", "step": step, "epoch": epoch, "lr": round(lr, 8), "flow": round(acc["flow"] / n, 5),
                   "dur": round(acc["dur"] / n, 5), "grad_norm": round(acc["gn"] / n, 4),
                   "s_per_step": round((time.time() - t_log) / n, 4), "frames_per_step": acc["frames"] // n,
                   "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
            D.append_jsonl(run / "log.jsonl", rec)
            if step % (args.log_every * 10) == 0:
                print(json.dumps(rec), flush=True)
            acc = {"flow": 0.0, "dur": 0.0, "gn": 0.0, "n": 0, "frames": 0}
            t_log = time.time()
        if step % args.val_every == 0 or step == args.steps:
            v = {"kind": "val", "step": step, "model": val_loss(model, val_batches, dev),
                 "ema": val_loss(ema, val_batches, dev)}
            D.append_jsonl(run / "log.jsonl", v)
            print(json.dumps(v), flush=True)
            t_log = time.time()
        out_of_time = (time.time() - t_start) / 60 >= args.max_minutes
        if step % args.ckpt_every == 0 or step == args.steps or out_of_time:
            save()
            t_log = time.time()
        if out_of_time and step < args.steps:
            print(f"time limit: stopped at step {step}; run again to resume", flush=True)
            break
    feed.stop = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
