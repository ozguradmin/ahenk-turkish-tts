#!/usr/bin/env python3
"""Speed/memory probe for fine-tuning EMA Lightning's acoustic model on the f022 latents (no real training).

    <venv-student>/python tts/small/train_probe.py --data data/small/f022-latents --steps 60 --batch 32

Builds real batches from index.jsonl + latents.npy, runs the losses the P0 trainer will use and AdamW steps on
ema.pt's weights, and reports s/step, peak GPU memory and the loss values (they must be finite and the
flow-matching loss at init should be far below the loss of an untrained model). Nothing is saved.

Losses (design in docs/KUCUK_MODEL_P0.md):
  flow    x_t = (1 - t) x0 + t x1 (x0 ~ N(0, I), x1 = target latents), backbone(x_t, cond, t) ~ x1 - x0, masked MSE
          (the parametrisation EMA's 4-step sampler uses: x1_hat = x_t + (1 - t) v)
  dur     per EMA word: (log(1 + sum of predicted letter frames) - log(1 + target frames))^2
  cond    text condition dropped (zeroed) with p = 0.1 for classifier-free guidance; style embedding (zero-init)
          added to the text features
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def batch_tensors(rows, lat, dev):
    """Teacher-forced timeline: fw/fp from the target word frames, cw/wstart from the text, as engine.plan/think."""
    B = len(rows)
    L = max(len(r["ids"]) for r in rows)
    T = max(r["frames"] for r in rows)
    ids = torch.zeros(B, L, dtype=torch.long)
    cw = torch.full((B, L), -1, dtype=torch.long)
    wstart = torch.zeros(B, L, dtype=torch.long)
    fw = torch.full((B, T), -1, dtype=torch.long)
    fp = torch.zeros(B, T)
    x1 = torch.zeros(B, T, 64)
    wf = torch.zeros(B, L, dtype=torch.float)  # target frames per word, indexed by word id
    nw = torch.zeros(B, dtype=torch.long)
    style = torch.tensor([r["style_id"] for r in rows])
    for b, r in enumerate(rows):
        text = r["ema_text"]
        ids[b, : len(r["ids"])] = torch.tensor(r["ids"])
        starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
        bounds = [0] + starts[1:] + [len(text)]
        for w, (a, e) in enumerate(zip(bounds, bounds[1:])):
            cw[b, a:e] = w
            wstart[b, a:e] = a
        n = torch.tensor(r["word_frames"])
        nw[b] = len(n)
        wf[b, : len(n)] = n.float()
        f = torch.repeat_interleave(torch.arange(len(n)), n)
        fw[b, : len(f)] = f
        fp[b, : len(f)] = (torch.arange(len(f)) - (n.cumsum(0) - n)[f]).float() / n[f].float()
        x1[b, : r["frames"]] = torch.from_numpy(lat[r["offset"]: r["offset"] + r["frames"]].astype(np.float32))
    mask, fmask = ids != 0, fw >= 0
    return {k: v.to(dev) for k, v in dict(ids=ids, cw=cw, wstart=wstart, fw=fw, fp=fp, x1=x1, wf=wf, nw=nw,
                                           mask=mask, fmask=fmask, style=style).items()}


def losses(model, style_emb, bt, p_drop=0.1):
    from ema_lightning.model import rope

    h = model.text(bt["ids"], bt["mask"]) + style_emb(bt["style"])[:, None]
    dur = model.chardur(h, bt["mask"])  # frames per letter
    word = bt["cw"].clamp(min=0)
    pred_w = torch.zeros_like(bt["wf"]).scatter_add_(1, word, dur * bt["mask"])
    wmask = torch.arange(bt["wf"].shape[1], device=h.device)[None] < bt["nw"][:, None]
    l_dur = (((torch.log1p(pred_w) - torch.log1p(bt["wf"])) ** 2) * wmask).sum() / wmask.sum()
    # condition, as sound_stage builds it, from the (detached) predicted letter durations
    m = bt["mask"]
    c = dur.detach().clamp(min=1e-4) * m
    done = c.cumsum(-1)
    before = done - c
    total = torch.zeros_like(c).scatter_add_(1, word, c).gather(1, word)
    cp = ((done - before.gather(1, bt["wstart"]) - 0.5 * c) / total.clamp(min=1e-8)).clamp(0.0, 1.0) * m
    cond = model.aligner(h, bt["cw"], cp, bt["fw"], bt["fp"], m, bt["cw"].shape[1])
    keep = (torch.rand(cond.shape[0], 1, 1, device=h.device) >= p_drop).float()
    cond = cond * keep
    B, T, _ = bt["x1"].shape
    x0 = torch.randn_like(bt["x1"])
    t = torch.rand(B, device=h.device)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * bt["x1"]
    cos, sin = rope(torch.arange(T, device=h.device), model.dh)
    v = model.backbone(xt, cond, t, bt["fmask"], cos, sin)
    fm = bt["fmask"][..., None].float()
    l_flow = (((v - (bt["x1"] - x0)) ** 2) * fm).sum() / (fm.sum() * 64)
    return l_flow, l_dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/small/f022-latents")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--scratch", action="store_true", help="random init instead of ema.pt (arm b)")
    args = ap.parse_args()
    from ema_lightning.model import Acoustic

    d = C.REPO / args.data
    rows = [r for r in C.read_jsonl(d / "index.jsonl") if r["use"] and r["split"] == "train"]
    lat = np.load(d / "latents.npy", mmap_mode="r")
    dev = "cuda"
    ck = torch.load(str(C.EMA_DIR / "ema.pt"), map_location="cpu", weights_only=True)
    model = Acoustic(ck["cfg"], ck["vocab"], bool(ck["cfg"].get("shared_ada", True)))
    if not args.scratch:
        model.load_state_dict(ck["ema"])
    model = model.to(dev).train()
    style_emb = nn.Embedding(8, model.d).to(dev)
    nn.init.zeros_(style_emb.weight)
    params = list(model.parameters()) + list(style_emb.parameters())
    print(f"params acoustic {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    rng = random.Random(0)
    # length-bucketed batches (sorted chunks, shuffled), like a real trainer
    rows.sort(key=lambda r: r["frames"])
    batches = [rows[i: i + args.batch] for i in range(0, len(rows), args.batch)]
    torch.cuda.reset_peak_memory_stats()
    times, log = [], []
    for step in range(args.steps):
        bt = batch_tensors(rng.choice(batches), lat, dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16"):
            l_flow, l_dur = losses(model, style_emb, bt)
        loss = l_flow + l_dur
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        log.append((float(l_flow), float(l_dur), int(bt["x1"].shape[1])))
        if step % 10 == 0:
            print(f"step {step} flow {log[-1][0]:.4f} dur {log[-1][1]:.4f} T {log[-1][2]} {times[-1]:.3f} s", flush=True)
    st = times[5:]
    res = {"init": "scratch" if args.scratch else "ema.pt", "batch": args.batch, "amp": args.amp,
           "train_clips": len(rows), "batches_per_epoch": len(batches),
           "s_per_step_mean": round(float(np.mean(st)), 4), "s_per_step_p90": round(float(np.percentile(st, 90)), 4),
           "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 0),
           "flow_first5": round(float(np.mean([x[0] for x in log[:5]])), 4),
           "flow_last10": round(float(np.mean([x[0] for x in log[-10:]])), 4),
           "dur_first5": round(float(np.mean([x[1] for x in log[:5]])), 4),
           "dur_last10": round(float(np.mean([x[1] for x in log[-10:]])), 4)}
    res["hours_per_10k_steps"] = round(res["s_per_step_mean"] * 10000 / 3600, 2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
