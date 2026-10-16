#!/usr/bin/env python3
"""Time one Piper "medium" (VITS) training step and measure its peak memory.

The step is exactly VitsModel's: generator forward (text encoder, posterior encoder, flow,
monotonic alignment, stochastic duration predictor, HiFi-GAN decoder on an 8192-sample slice),
discriminator forward, generator + discriminator backward and AdamW steps. Batches are synthetic
but shaped like this corpus: every clip `--seconds` long with `--ids` phoneme ids (the corpus'
worst case after prepare_antalia.py is 20.24 s / 763 ids, the median 17.6 s / 565 ids).

    python tts/tools/step_bench.py --batch 4 --seconds 20.24 --ids 763          # CPU or CUDA
    python tts/tools/step_bench.py --batch 32 --seconds 20.24 --ids 763 --precision bf16-mixed

Prints one JSON line: seconds per step (mean of --steps after --warmup), peak memory (CUDA:
max_memory_allocated; CPU: growth of the process' max RSS over the loaded model), parameters.
Run one configuration per process: the CPU number is a process-lifetime high-water mark.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time

import torch


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=20.24, help="length of every clip")
    ap.add_argument("--ids", type=int, default=763, help="phoneme ids per clip (incl. padding ids)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--precision", default="32-true", choices=["32-true", "bf16-mixed", "16-mixed"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", default="", help="optional: start from these weights (e.g. the LJSpeech checkpoint)")
    args = ap.parse_args()

    from piper.train.vits.dataset import Batch
    from piper.train.vits.lightning import VitsModel

    torch.manual_seed(0)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = VitsModel(batch_size=args.batch, mos_metric=None)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(sd, strict=False)
    model.to(dev).train()
    h = model.hparams
    opt_g = torch.optim.AdamW(model.model_g.parameters(), lr=h.learning_rate, betas=h.betas, eps=h.eps)
    opt_d = torch.optim.AdamW(model.model_d.parameters(), lr=h.learning_rate_d, betas=h.betas_d, eps=h.eps)

    b, sr, hop, n_fft = args.batch, h.sample_rate, h.hop_length, h.filter_length
    samples = int(args.seconds * sr)
    frames = (samples - n_fft) // hop + 1  # spectrogram_torch(center=False)

    def batch() -> Batch:
        return Batch(
            phoneme_ids=torch.randint(1, 150, (b, args.ids), device=dev),
            phoneme_lengths=torch.full((b,), args.ids, dtype=torch.long, device=dev),
            spectrograms=torch.rand(b, n_fft // 2 + 1, frames, device=dev),
            spectrogram_lengths=torch.full((b,), frames, dtype=torch.long, device=dev),
            audios=torch.randn(b, 1, samples, device=dev) * 0.1,
            audio_lengths=torch.full((b,), samples, dtype=torch.long, device=dev),
            speaker_ids=None,
        )

    amp_dtype = {"bf16-mixed": torch.bfloat16, "16-mixed": torch.float16}.get(args.precision)
    scaler = torch.amp.GradScaler(dev.type) if args.precision == "16-mixed" else None

    def step() -> None:
        x = batch()
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss_g, loss_d, _ = model._compute_loss(x)  # noqa: SLF001  (the module's own loss)
        for opt, loss in ((opt_g, loss_g), (opt_d, loss_d)):
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()

    base_rss = rss_mb()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base_cuda = torch.cuda.memory_allocated() / 2**20
    for _ in range(args.warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    sec = (time.perf_counter() - t0) / max(1, args.steps)
    out = {
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else f"cpu ({torch.get_num_threads()} threads, {platform.processor() or platform.machine()})",
        "precision": args.precision, "batch": b, "seconds_per_clip": args.seconds, "ids_per_clip": args.ids,
        "s_per_step": round(sec, 3), "s_per_clip": round(sec / b, 4),
        "params_g_M": round(sum(p.numel() for p in model.model_g.parameters()) / 1e6, 2),
        "params_d_M": round(sum(p.numel() for p in model.model_d.parameters()) / 1e6, 2),
        "torch": torch.__version__,
    }
    if dev.type == "cuda":
        out["peak_alloc_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        out["peak_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        out["model_and_state_GiB"] = round(base_cuda / 1024, 2)
    else:
        out["rss_before_MB"] = round(base_rss)
        out["peak_rss_growth_MB"] = round(rss_mb() - base_rss)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
