"""Ahenk Cep (~8.6M parameters): inference only, CPU friendly.

A self-contained copy of the inference half of tts/small/student.py (ahenk-turkish-tts, Apache-2.0): EMA Lightning's
acoustic model (Apache-2.0, canberkkkkkk/ema-lightning) with a style embedding, sampled with Euler steps and
classifier-free guidance, inside EMA Lightning's own engine (chunking, timeline, pauses, decoder windows).
trnorm turns written Turkish into its spoken form before the model sees it.

    from ahenk_cep import AhenkCep
    tts = AhenkCep("weights/student.pt", "weights/decoder.pt")
    audio, sr = tts.say("Randevunuz 14.10.2026 saat 09:30'da.")
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from ema_lightning import EMA
from ema_lightning.decoder import load_decoder
from ema_lightning.frontend import Frontend
from ema_lightning.model import Acoustic, rope
from trnorm import normalize

# Style ids, in training order (tts/small/student_data.py). "laughter" exists in the table but was not trained.
STYLES = ("neutral", "angry", "calm", "cheerful", "empathetic", "sad", "surprised", "laughter")
SAMPLE_RATE = 48000


class StudentAcoustic(Acoustic):
    def __init__(self, cfg, vocab, shared=True, n_styles: int = len(STYLES)):
        super().__init__(cfg, vocab, shared)
        self.style_emb = nn.Embedding(n_styles, self.d)
        self.style = 0
        self.steps, self.cfg_scale = 16, 2.0

    def features(self, ids, mask, style):
        return self.text(ids, mask) + self.style_emb(style)[:, None]

    def text_stage(self, ids, mask):
        style = torch.full((ids.shape[0],), int(self.style), dtype=torch.long, device=ids.device)
        h = self.features(ids, mask, style)
        return h, self.chardur(h, mask)

    def condition(self, h, dur, mask, cw, wstart, fw, fp):
        c = dur.clamp(min=1e-4) * mask
        done = c.cumsum(-1)
        before = done - c
        word = cw.clamp(min=0)
        total = torch.zeros_like(c).scatter_add_(1, word, c).gather(1, word)
        cp = ((done - before.gather(1, wstart) - 0.5 * c) / total.clamp(min=1e-8)).clamp(0.0, 1.0) * mask
        return self.aligner(h, cw, cp, fw, fp, mask, cw.shape[1])

    def velocity(self, x, cond, t, fmask, cos, sin, cfg):
        step = torch.full((x.shape[0],), float(t), device=x.device)
        if cfg == 1.0:
            return self.backbone(x, cond, step, fmask, cos, sin)
        v = self.backbone(torch.cat([x, x]), torch.cat([cond, torch.zeros_like(cond)]), torch.cat([step, step]),
                          torch.cat([fmask, fmask]), cos, sin)
        vc, vu = v.chunk(2)
        return vu + cfg * (vc - vu)

    def sound_stage(self, h, dur, mask, cw, wstart, fw, fp, fmask, noise):
        cond = self.condition(h, dur, mask, cw, wstart, fw, fp)
        cos, sin = rope(torch.arange(fw.shape[1], device=h.device), self.dh)
        x = noise[:, 0]
        n = self.steps
        for k in range(n):
            x = x + (1.0 / n) * self.velocity(x, cond, k / n, fmask, cos, sin, self.cfg_scale)
        return x


def load_student(path, device="cpu") -> StudentAcoustic:
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    model = StudentAcoustic(ck["cfg"], ck["vocab"], bool(ck["cfg"].get("shared_ada", True)))
    model.load_state_dict(ck["ema"])
    return model.to(device).float().eval().requires_grad_(False)


class AhenkCep:
    def __init__(self, student_path, decoder_path, device="cpu"):
        self.model = load_student(student_path, device)
        decoder = load_decoder(str(decoder_path), device)
        fe = Frontend.__new__(Frontend)  # EMA's alphabet step only; trnorm already made the spoken form
        fe.vocab = frozenset(self.model.vocab)
        fe.spoken = lambda t: t
        self.engine = EMA._from_parts(self.model, decoder, fe, device)

    @torch.inference_mode()
    def say(self, text: str, style: str = "neutral", steps: int = 16, cfg: float = 2.0, speed: float = 1.0,
            seed: int = 0) -> tuple[np.ndarray, int]:
        self.model.style = STYLES.index(style)
        self.model.steps, self.model.cfg_scale = int(steps), float(cfg)
        speech = self.engine.say(normalize(text), speed=speed, seed=seed)
        return speech.audio, speech.sample_rate


def trnorm_ema(device="cpu"):
    """Upstream EMA Lightning with trnorm as its front-end (src/trtts in ahenk-turkish-tts)."""
    from trtts import TurkishTTS

    return TurkishTTS(device=device)
