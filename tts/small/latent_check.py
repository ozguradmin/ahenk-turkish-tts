#!/usr/bin/env python3
"""Does EMA Lightning's decoder.pt decode VoxCPM2 AudioVAE latents of f022 clips? (P0 task 3a, local GPU)

    <venv-student>/python tts/small/latent_check.py --corpus <xfer>/f022corpus --corpus2 <xfer>/f022corpus2

Picks a fixed spread of f022 clips (random.Random(0): 3 neutral, 2 questions, 3 emotions, 2 long, 2 cwords),
encodes each at 16 kHz with the cached VoxCPM2 audiovae.pth encoder (mu, unnormalised, x1.0), decodes with
(a) EMA decoder.pt and (b) the VoxCPM2 AudioVAE decoder, plus (c) EMA decoder on time-shuffled latents as a
floor. Whisper (judge settings) on original / (a) / (b) / (c) -> CER/WER against the clip's spoken text;
log-mel L1 and correlation to the original; UTMOS of each. Also checks GPU vs CPU encoder agreement.
Writes OUT/rows.jsonl, OUT/summary.json and the wavs.
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

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from audiovae import load_audiovae  # noqa: E402


def logmel(y, sr=16000):
    import torchaudio

    m = torchaudio.transforms.MelSpectrogram(sr, n_fft=1024, hop_length=256, n_mels=80, f_max=8000)(torch.from_numpy(y))
    return np.log(m.numpy() + 1e-5)


def pick(c1: Path, c2: Path) -> list[dict]:
    rng = random.Random(0)
    clips = [dict(r, _dir=str(c1)) for r in C.read_jsonl(c1 / "clips.jsonl")] + \
            [dict(r, _dir=str(c2)) for r in C.read_jsonl(c2 / "clips.jsonl")]
    want = {"neutral": 3, "questions": 2, "emotions": 3, "long": 2, "cwords": 2}
    out = []
    for cat, n in want.items():
        pool = sorted([r for r in clips if r["category"] == cat], key=lambda r: r["id"])
        out += rng.sample(pool, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpus2", required=True)
    ap.add_argument("--out", default="voxcpm-contest/small-bench/latent-check")
    args = ap.parse_args()
    out = C.REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    from ema_lightning.decoder import load_decoder

    dev = "cuda"
    vae = load_audiovae(C.VOXCPM_DIR / "audiovae.pth", dev)
    ema = load_decoder(str(C.EMA_DIR / "decoder.pt"), dev)
    vae_cpu_enc = load_audiovae(C.VOXCPM_DIR / "audiovae.pth", "cpu", with_decoder=False)
    sys.path.insert(0, str(C.UTMOS_HUB))
    utmos = torch.hub.load(str(C.UTMOS_HUB), "utmos22_strong", source="local", trust_repo=True).to(dev).eval()
    judge = C.Judge(dev)
    rows = []
    for r in pick(Path(args.corpus), Path(args.corpus2)):
        wav = Path(r["_dir"]) / "wav" / Path(r["wav"]).name
        x = C.load_16k(wav)
        ref, _ = C.strip_instruction(r.get("render_spoken") or r["spoken"])
        xt = torch.from_numpy(x)[None, None].to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        z = vae.encode(xt)
        torch.cuda.synchronize()
        t_enc = time.perf_counter() - t0
        z_cpu = vae_cpu_enc.encode(torch.from_numpy(x)[None, None])
        enc_diff = float((z.cpu() - z_cpu).abs().max())
        g = torch.Generator(device="cpu").manual_seed(0)
        zs = z[..., torch.randperm(z.shape[-1], generator=g).to(dev)]
        with torch.no_grad():
            outs = {"orig": (x, 16000), "vae_dec": (vae.decode(z)[0, 0].float().cpu().numpy(), 48000),
                    "ema_dec": (ema(z)[0].float().cpu().numpy(), 48000),
                    "ema_dec_shuffled": (ema(zs)[0].float().cpu().numpy(), 48000)}
        ref_mel = logmel(x)
        row = {"id": r["id"], "category": r["category"], "emotion": r.get("emotion"), "duration": r["duration"],
               "ref": ref, "frames": int(z.shape[-1]), "z_mean": round(float(z.mean()), 4), "z_std": round(float(z.std()), 4),
               "encode_gpu_s": round(t_enc, 4), "enc_gpu_cpu_maxabs": enc_diff}
        for name, (a, sr) in outs.items():
            if name != "orig":
                sf.write(str(out / f"{r['id']}__{name}.wav"), a, sr, subtype="PCM_16")
            else:
                sf.write(str(out / f"{r['id']}__orig16k.wav"), a, sr, subtype="PCM_16")
            a16 = C.to_16k(a, sr)
            j = judge.transcribe(a16)
            sc = C.score_row([ref], j["hyp"])
            n = min(len(a16), len(x))
            m = logmel(a16[:n])
            t = min(m.shape[1], ref_mel.shape[1])
            with torch.no_grad():
                u = float(utmos(torch.from_numpy(a16)[None].to(dev), 16000)[0])
            row[name] = {"hyp": j["hyp"], "word_edits": sc["word_edits"], "ref_words": sc["ref_words"],
                         "char_edits": sc["char_edits"], "ref_chars": sc["ref_chars"], "utmos": round(u, 3),
                         "mel_l1": round(float(np.abs(m[:, :t] - ref_mel[:, :t]).mean()), 3),
                         "mel_corr": round(float(np.corrcoef(m[:, :t].ravel(), ref_mel[:, :t].ravel())[0, 1]), 3)}
        rows.append(row)
        print(r["id"], {k: (row[k]["char_edits"], row[k]["mel_l1"], row[k]["utmos"]) for k in outs}, flush=True)
    C.write_jsonl(out / "rows.jsonl", rows)
    summ = {"n": len(rows), "judge": judge.desc}
    for name in ("orig", "vae_dec", "ema_dec", "ema_dec_shuffled"):
        rs = [r[name] for r in rows]
        we, wn = sum(x["word_edits"] for x in rs), sum(x["ref_words"] for x in rs)
        ce, cn = sum(x["char_edits"] for x in rs), sum(x["ref_chars"] for x in rs)
        summ[name] = {"wer": round(100 * we / wn, 2), "cer": round(100 * ce / cn, 2),
                      "rows_ok": sum(x["word_edits"] == 0 for x in rs),
                      "utmos": round(float(np.mean([x["utmos"] for x in rs])), 3),
                      "mel_l1": round(float(np.mean([x["mel_l1"] for x in rs])), 3),
                      "mel_corr": round(float(np.mean([x["mel_corr"] for x in rs])), 3)}
    summ["enc_gpu_cpu_maxabs"] = max(r["enc_gpu_cpu_maxabs"] for r in rows)
    summ["z_std_mean"] = round(float(np.mean([r["z_std"] for r in rows])), 3)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
