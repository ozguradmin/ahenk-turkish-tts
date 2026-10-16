"""Ahenk Cep without torch: the in-browser pipeline (site/js/engine.js) written in numpy + onnxruntime.

It is the reference the page is checked against and the bridge to PyTorch: same text front-end (site/py/
ahenk_front.py), same frame timeline as ema_lightning.engine.Engine.plan, the same 100-frame decoder windows with 8
frames of context, the same pauses, and torch.randn's CPU noise reproduced bit for bit from the seed (mt19937 and
the 16-value Box-Muller of ATen's normal_fill), so a seed gives the same audio here, in the page and in
tts/small/ahenk_cep.py up to float rounding.

    from web_pipeline import OnnxCep
    tts = OnnxCep("site/models")              # or variant="f16"
    audio, sr = tts.say("Saat 09:30'da.", style="sad", seed=0)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SITE_PY = REPO / "site" / "py"


class MT19937:
    """std::mt19937 as at::mt19937 seeds it (seed & 0xffffffff, Knuth's initializer)."""

    def __init__(self, seed: int):
        s = [0] * 624
        s[0] = seed & 0xFFFFFFFF
        for j in range(1, 624):
            s[j] = (1812433253 * (s[j - 1] ^ (s[j - 1] >> 30)) + j) & 0xFFFFFFFF
        self.state, self.index = s, 624

    def _twist(self):
        s = self.state
        for i in range(624):
            y = (s[i] & 0x80000000) | (s[(i + 1) % 624] & 0x7FFFFFFF)
            v = s[(i + 397) % 624] ^ (y >> 1)
            if y & 1:
                v ^= 0x9908B0DF
            s[i] = v
        self.index = 0

    def next_u32(self) -> int:
        if self.index >= 624:
            self._twist()
        y = self.state[self.index]
        self.index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF


def torch_randn(seed: int, n: int) -> np.ndarray:
    """The first n values of torch.randn(..., generator=torch.Generator().manual_seed(seed)) on CPU (float32).

    ATen's normal_fill: n uniforms (24-bit, (u32 & 0xFFFFFF) / 2**24), then each block of 16 becomes 16 normals by
    Box-Muller pairing value j with j + 8. Only the tail of a tensor whose size is not a multiple of 16 is redrawn,
    so a prefix that is a multiple of 16 is exact whatever the full tensor size. n must be a multiple of 16.
    """
    if n % 16:
        raise ValueError("n must be a multiple of 16")
    gen = MT19937(seed)
    u = np.array([gen.next_u32() & 0xFFFFFF for _ in range(n)], dtype=np.float64) * 2.0 ** -24
    u = u.astype(np.float32).reshape(-1, 16)
    u1, u2 = (np.float32(1) - u[:, :8]), u[:, 8:]
    radius = np.sqrt(np.float32(-2) * np.log(u1, dtype=np.float32), dtype=np.float32)
    theta = (2.0 * math.pi * u2.astype(np.float64)).astype(np.float32)
    out = np.empty_like(u)
    out[:, :8] = radius * np.cos(theta, dtype=np.float32)
    out[:, 8:] = radius * np.sin(theta, dtype=np.float32)
    return out.reshape(-1)


def piece_seed(seed: int, i: int) -> int:
    return (seed * 1_000_003 + i) % 2 ** 63


def plan(dur: np.ndarray, cw: np.ndarray, speed: float, max_word=250, max_frames=3000):
    """Engine.plan for one unpadded piece: per-word frames (round half to even, clamp 1..250), fw, fp."""
    dur = (dur / np.float32(speed)).astype(np.float32)
    n_words = int(cw[-1]) + 1
    counts = np.zeros(n_words, np.float32)
    for i, w in enumerate(cw):  # float32 sums in letter order, like scatter_add_ on one CPU thread
        counts[w] = np.float32(counts[w] + dur[i])
    n = np.clip(np.round(counts), 1, max_word).astype(np.int64)
    frames = min(int(n.sum()), max_frames)
    fw = np.repeat(np.arange(n_words), n)[:frames]
    fp = ((np.arange(frames) - (np.cumsum(n) - n)[fw]).astype(np.float64) / n[fw].astype(np.float64)).astype(
        np.float32)
    return dur, fw, fp


def windows(frames: int, first=100, window=100):
    spans, s = [], 0
    while s < frames:
        e = min(frames, s + (first if s == 0 else window))
        spans.append((s, e))
        s = e
    return spans


class OnnxCep:
    def __init__(self, models: str | Path, variant: str = "fp32", threads: int | None = None, variants=None):
        import onnxruntime as ort

        models = Path(models)
        self.manifest = json.loads((models / "ahenk-cep.json").read_text(encoding="utf-8"))
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        pick = dict.fromkeys(("text", "cond", "step", "decoder"), variant)
        pick.update(variants or {})
        self.sessions = {}
        for name, v in pick.items():
            files = self.manifest["files"][name]
            f = files.get(v, files["fp32"])["file"]
            self.sessions[name] = ort.InferenceSession(str(models / f), so, providers=["CPUExecutionProvider"])
        self.vocab = self.manifest["vocab"]
        if str(SITE_PY) not in sys.path:
            sys.path.insert(0, str(SITE_PY))
        import ahenk_front

        self.front = ahenk_front

    def run(self, name, **feeds):
        return self.sessions[name].run(None, feeds)

    def pieces(self, text: str, speed: float = 1.0):
        return json.loads(self.front.prepare(text, speed, self.vocab))["pieces"]

    def latents(self, p: dict, style: int, speed: float, seed: int, steps: int, cfg: float):
        ids = np.array([p["ids"]], np.int64)
        cw, wstart = np.array([p["cw"]], np.int64), np.array([p["wstart"]], np.int64)
        h, dur = self.run("text", ids=ids, style=np.array([style], np.int64))
        dur, fw, fp = plan(dur[0], cw[0], speed)
        T = fw.size
        (cond,) = self.run("cond", h=h, dur=dur[None], cw=cw, wstart=wstart, fw=fw[None].astype(np.int64),
                           fp=fp[None])
        x = torch_randn(seed, T * self.manifest["latent_dim"]).reshape(1, T, -1)
        for k in range(steps):
            (x,) = self.run("step", x=x, cond=cond, t=np.array([k / steps], np.float32),
                            dt=np.array([1.0 / steps], np.float32), cfg=np.array([cfg], np.float32))
        return x[0], fw

    def decode(self, lat: np.ndarray):
        lim, hop = self.manifest["limits"], self.manifest["hop"]
        frames, out = lat.shape[0], []
        for s, e in windows(frames, lim["window"], lim["window"]):
            a, b = max(0, s - lim["context"]), min(frames, e + lim["context"])
            (audio,) = self.run("decoder", z=np.ascontiguousarray(lat[a:b].T[None]))
            out.append(audio[0, (s - a) * hop:(e - a) * hop])
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def say(self, text: str, style: str = "neutral", steps: int = 16, cfg: float = 2.0, speed: float = 1.0,
            seed: int = 0):
        style_id = self.manifest["styles"].index(style)
        sr, chunks = self.manifest["sample_rate"], []
        for i, p in enumerate(self.pieces(text, speed)):
            lat, _ = self.latents(p, style_id, speed, piece_seed(seed, i), steps, cfg)
            chunks.append(self.decode(lat))
            if p["pause"]:
                chunks.append(np.zeros(round(p["pause"] * sr), np.float32))
        audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
        return audio, sr
