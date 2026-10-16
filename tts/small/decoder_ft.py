#!/usr/bin/env python3
"""Optional P1 step: fine-tune EMA Lightning's decoder.pt (HiFi-GAN-style, 3.0M, Apache-2.0) on f022, locally.

decoder.pt reads VoxCPM2 AudioVAE latents but costs ~0.4 UTMOS on f022 (P0: 3.59 original, 3.56 through VoxCPM2's own
decoder, 3.20 through decoder.pt). Targets here are VoxCPM2's own decoder outputs (48 kHz) of the f022 latents, so the
small decoder learns to sound like the big one on this voice.

    prep:  <venv-student>/python tts/small/decoder_ft.py prep            (VoxCPM2 decoder over every clip -> int16)
    train: <venv-student>/python tts/small/decoder_ft.py train --run data/small/runs/dec-a --steps 6000 --max-minutes 25

prep writes data/small/f022-dec48k/audio.npy (int16, clip i = audio[offset*1920 : (offset+frames)*1920], the same
offsets as latents.npy) and meta.json. train: random segments of --seg frames (+ --ctx frames of latent context on each
side inside the clip, cut off after decoding, as EMA's engine decodes windows), losses = multi-resolution log-mel L1
(x --mel-weight) + LSGAN with multi-period (2, 3, 5, 7, 11) and multi-scale (3) discriminators written from the
HiFi-GAN paper (arXiv 2010.05646) + feature matching (x2); the discriminators alone for --d-warmup steps first.
Checkpoints RUN/ckpt_<step>.pt (G, D, optimisers, step), G also as RUN/decoder_<step>.pt in decoder.pt's format
({"G", "cfg"}), loadable by ema_lightning.decoder.load_decoder. RUN/log.jsonl: losses and the val mel L1 (fixed val
clips, whole clips) every --val-every steps.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0common as C  # noqa: E402
import student_data as D  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HOP = 1920
SR = 48000
LAT = C.REPO / "data" / "small" / "f022-latents"
DEC48 = C.REPO / "data" / "small" / "f022-dec48k"


# ---------------------------------------------------------------------------------------------------- targets
def prep(args):
    from audiovae import load_audiovae

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = D.read_index(LAT, None, use_only=False)
    lat = np.load(LAT / "latents.npy", mmap_mode="r")
    DEC48.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(DEC48 / "audio.npy", mode="w+", dtype=np.int16, shape=(lat.shape[0] * HOP,))
    vae = load_audiovae(C.VOXCPM_DIR / "audiovae.pth", "cuda")
    rows.sort(key=lambda r: r["frames"])
    t0, i = time.time(), 0
    while i < len(rows):  # batches of ~1500 frames (60 s), right-padded (the decoder is causal: exact)
        b = [rows[i]]
        while i + len(b) < len(rows) and (len(b) + 1) * rows[i + len(b)]["frames"] <= args.batch_frames:
            b.append(rows[i + len(b)])
        T = max(r["frames"] for r in b)
        z = torch.zeros(len(b), 64, T)
        for k, r in enumerate(b):
            z[k, :, : r["frames"]] = torch.from_numpy(lat[r["offset"]: r["offset"] + r["frames"]].astype(np.float32).T)
        with torch.no_grad():
            y = vae.decode(z.cuda())[:, 0].float().cpu().numpy()
        for k, r in enumerate(b):
            a = np.clip(y[k, : r["frames"] * HOP], -1, 1)
            out[r["offset"] * HOP: (r["offset"] + r["frames"]) * HOP] = np.round(a * 32767).astype(np.int16)
        i += len(b)
        if i % 500 < len(b):
            print(f"{i}/{len(rows)} {time.time() - t0:.0f} s", flush=True)
    out.flush()
    (DEC48 / "meta.json").write_text(json.dumps({"source": "VoxCPM2 AudioVAE decoder (audiovae.pth) of latents.npy",
                                                 "sr": SR, "hop": HOP, "dtype": "int16", "clips": len(rows),
                                                 "seconds": round(time.time() - t0, 1)}, indent=1) + "\n")
    print(f"done: {len(rows)} clips in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------- losses
class MelLoss(nn.Module):
    """Multi-resolution log-mel L1 at 48 kHz."""

    def __init__(self, res=((512, 128, 64), (1024, 256, 128), (2048, 512, 128))):
        super().__init__()
        import torchaudio

        self.mels = nn.ModuleList(torchaudio.transforms.MelSpectrogram(SR, n_fft=n, hop_length=h, n_mels=m, f_min=0,
                                                                       f_max=SR // 2, power=1.0) for n, h, m in res)

    def forward(self, y, t):
        return sum(F.l1_loss(torch.log(m(y).clamp(min=1e-5)), torch.log(m(t).clamp(min=1e-5))) for m in self.mels) / len(self.mels)


class PeriodD(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        wn = nn.utils.parametrizations.weight_norm
        ch = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList(wn(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1), padding=(2, 0))) for i in range(4))
        self.convs.append(wn(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))))
        self.post = wn(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.p:
            x = F.pad(x, (0, self.p - t % self.p), "reflect")
        x = x.view(b, c, -1, self.p)
        fm = []
        for l in self.convs:
            x = F.leaky_relu(l(x), 0.1)
            fm.append(x)
        x = self.post(x)
        fm.append(x)
        return x.flatten(1), fm


class ScaleD(nn.Module):
    def __init__(self, spectral=False):
        super().__init__()
        norm = nn.utils.spectral_norm if spectral else nn.utils.parametrizations.weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)), norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)), norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)), norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2))])
        self.post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fm = []
        for l in self.convs:
            x = F.leaky_relu(l(x), 0.1)
            fm.append(x)
        x = self.post(x)
        fm.append(x)
        return x.flatten(1), fm


class Discriminators(nn.Module):
    def __init__(self):
        super().__init__()
        self.ds = nn.ModuleList([PeriodD(p) for p in (2, 3, 5, 7, 11)] + [ScaleD(True), ScaleD(), ScaleD()])
        self.pool = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x):
        outs = []
        for i, d in enumerate(self.ds):
            k = i - 5  # scale discriminators: raw, /2, /4
            xi = x
            for _ in range(max(0, k)):
                xi = self.pool(xi)
            outs.append(d(xi))
        return outs


def d_loss(real, fake):
    return sum(torch.mean((1 - r[0]) ** 2) + torch.mean(f[0] ** 2) for r, f in zip(real, fake))


def g_adv(fake):
    return sum(torch.mean((1 - f[0]) ** 2) for f in fake)


def fm_loss(real, fake):
    return sum(F.l1_loss(fa, ra.detach()) for r, f in zip(real, fake) for ra, fa in zip(r[1], f[1]))


# ---------------------------------------------------------------------------------------------------- data
def segment_batch(rows, lat, audio, n, seg, ctx, rng: random.Random):
    """n random segments: latents [n, 64, seg + 2 ctx] (zero where the context leaves the clip), target audio
    [n, 1, seg * HOP], and per item the frame where the segment starts inside the latent window."""
    z = np.zeros((n, 64, seg + 2 * ctx), np.float32)
    y = np.zeros((n, 1, seg * HOP), np.float32)
    lens = np.zeros(n, np.int64)
    for k in range(n):
        r = rng.choice(rows)
        s = rng.randint(0, max(0, r["frames"] - seg))
        e = min(r["frames"], s + seg)
        a, b = max(0, s - ctx), min(r["frames"], e + ctx)
        o = r["offset"]
        z[k, :, ctx - (s - a): ctx - (s - a) + (b - a)] = lat[o + a: o + b].astype(np.float32).T
        y[k, 0, : (e - s) * HOP] = audio[(o + s) * HOP: (o + e) * HOP].astype(np.float32) / 32767
        lens[k] = e - s
    return z, y, lens


def save_decoder(G, cfg, path):
    sd = {k: v.detach().cpu() for k, v in G.state_dict().items()}
    tmp = Path(str(path) + ".tmp")
    torch.save({"G": sd, "cfg": cfg}, str(tmp))
    os.replace(tmp, path)


def train(args):
    from ema_lightning.decoder import load_decoder

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    run = C.REPO / args.run if not Path(args.run).is_absolute() else Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    rows = D.read_index(LAT, "train")
    val = sorted(D.read_index(LAT, "val"), key=lambda r: r["id"])[: args.val_clips]
    lat = np.load(LAT / "latents.npy", mmap_mode="r")
    audio = np.load(DEC48 / "audio.npy", mmap_mode="r")
    cfg = torch.load(str(C.EMA_DIR / "decoder.pt"), map_location="cpu", weights_only=True)["cfg"]
    G = load_decoder(str(C.EMA_DIR / "decoder.pt"), dev).train().requires_grad_(True)
    Dn = Discriminators().to(dev)
    mel = MelLoss().to(dev)
    og = torch.optim.AdamW(G.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.0)
    od = torch.optim.AdamW(Dn.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.0)
    step = 0
    last = D.latest_ckpt(run)
    if last is not None:
        ck = torch.load(str(last), map_location="cpu", weights_only=False)
        G.load_state_dict(ck["G"])
        Dn.load_state_dict(ck["D"])
        og.load_state_dict(ck["og"])
        od.load_state_dict(ck["od"])
        step = ck["step"]
        print(f"resumed {last.name} at step {step}", flush=True)
    else:
        (run / "args.json").write_text(json.dumps(vars(args), indent=1) + "\n", encoding="utf-8")
    rng = random.Random(args.seed * 7919 + step)

    def val_mel():
        G.eval()
        tot = 0.0
        with torch.no_grad():
            for r in val:
                z = torch.from_numpy(lat[r["offset"]: r["offset"] + r["frames"]].astype(np.float32).T)[None].to(dev)
                t = torch.from_numpy(audio[r["offset"] * HOP: (r["offset"] + r["frames"]) * HOP].astype(np.float32)
                                     / 32767)[None, None].to(dev)
                tot += float(mel(G(z)[:, None], t))
        G.train()
        return round(tot / len(val), 5)

    if step == 0:
        v = val_mel()
        D.append_jsonl(run / "log.jsonl", {"kind": "val", "step": 0, "mel": v})
        print(f"val mel at init {v}", flush=True)
    t0 = t_log = time.time()
    acc: dict[str, float] = {}
    torch.cuda.reset_peak_memory_stats()
    while step < args.steps:
        z, y, lens = segment_batch(rows, lat, audio, args.batch, args.seg, args.ctx, rng)
        z, y = torch.from_numpy(z).to(dev), torch.from_numpy(y).to(dev)
        g = G(z)[:, None, args.ctx * HOP: (args.ctx + args.seg) * HOP]
        keep = (torch.arange(g.shape[-1], device=dev)[None] < torch.from_numpy(lens * HOP).to(dev)[:, None])[:, None]
        g = g * keep
        # discriminators
        od.zero_grad(set_to_none=True)
        ld = d_loss(Dn(y), Dn(g.detach()))
        ld.backward()
        od.step()
        # generator
        lm = mel(g[:, 0], y[:, 0])
        if step >= args.d_warmup:
            real, fake = Dn(y), Dn(g)
            la, lf = g_adv(fake), fm_loss(real, fake)
            lg = args.mel_weight * lm + la + 2 * lf
            og.zero_grad(set_to_none=True)
            lg.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1000.0)
            og.step()
        else:
            la = lf = torch.zeros(())
        step += 1
        for k, v in (("d", ld), ("mel", lm), ("adv", la), ("fm", lf)):
            acc[k] = acc.get(k, 0.0) + float(v)
        acc["n"] = acc.get("n", 0) + 1
        if step % args.log_every == 0:
            n = acc.pop("n")
            rec = {"kind": "train", "step": step, **{k: round(v / n, 4) for k, v in acc.items()},
                   "s_per_step": round((time.time() - t_log) / n, 3),
                   "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
            D.append_jsonl(run / "log.jsonl", rec)
            print(json.dumps(rec), flush=True)
            acc = {}
            t_log = time.time()
        out_of_time = (time.time() - t0) / 60 >= args.max_minutes
        if step % args.val_every == 0 or step == args.steps or out_of_time:
            v = val_mel()
            D.append_jsonl(run / "log.jsonl", {"kind": "val", "step": step, "mel": v,
                                               "minutes": round((time.time() - t0) / 60, 2)})
            print(f"val mel {step}: {v}", flush=True)
        if step % args.ckpt_every == 0 or step == args.steps or out_of_time:
            tmp = run / (D.ckpt_name(step) + ".tmp")
            torch.save({"G": G.state_dict(), "D": Dn.state_dict(), "og": og.state_dict(), "od": od.state_dict(),
                        "step": step}, str(tmp))
            os.replace(tmp, run / D.ckpt_name(step))
            D.prune_ckpts(run, keep=1)
            save_decoder(G, cfg, run / f"decoder_{step:08d}.pt")
            print(f"saved step {step}", flush=True)
        if out_of_time and step < args.steps:
            print(f"time limit: stopped at step {step}; run again to resume", flush=True)
            break


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--batch-frames", type=int, default=1500)
    t = sub.add_parser("train")
    t.add_argument("--run", required=True)
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--max-minutes", type=float, default=25.0)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--seg", type=int, default=16, help="frames per segment (25 Hz)")
    t.add_argument("--ctx", type=int, default=8, help="latent context frames on each side")
    t.add_argument("--lr-g", type=float, default=5e-5)
    t.add_argument("--lr-d", type=float, default=2e-4)
    t.add_argument("--mel-weight", type=float, default=45.0)
    t.add_argument("--d-warmup", type=int, default=500)
    t.add_argument("--val-every", type=int, default=500)
    t.add_argument("--ckpt-every", type=int, default=1000)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--val-clips", type=int, default=20)
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    prep(args) if args.cmd == "prep" else train(args)


if __name__ == "__main__":
    main()
