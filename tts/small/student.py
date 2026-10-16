"""The student: EMA Lightning's acoustic model (Apache-2.0) with a style embedding, trained by flow matching on the f022
latents (train.py), and a sampler that plugs into EMA's own engine (chunking, timeline, decoder windows unchanged).

- StudentAcoustic(Acoustic): style_emb (8 x d, zero-init: at init it is exactly ema.pt) added to the text features
  before the duration predictor and the aligner; `style` (int) picks the style at inference
- losses(): OT-CFM velocity MSE on real frames (x_t = (1 - t) x0 + t x1, target x1 - x0) + per-word L2 on
  log(1 + frames); text condition (aligner output) zeroed with p_cond, style set to neutral with p_style
- sound_stage(): `sampler` "euler" / "midpoint" with `steps` and classifier-free guidance `cfg` (uncond = zeroed
  condition, both halves in one batch), optional sway schedule (F5-TTS: s < 0 puts more steps near t = 0);
  "ema4" = EMA's own 4-step re-noising schedule (what ema.pt was distilled for)
- load_student(): a train.py checkpoint (its EMA weights by default) or ema.pt itself ("ema.pt")
- make_tts(): an ema_lightning.EMA built from these parts, with the trnorm front-end (as bench.py's "ema")
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0common as C  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from ema_lightning.model import Acoustic, rope  # noqa: E402

from student_data import N_STYLES  # noqa: E402


class StudentAcoustic(Acoustic):
    def __init__(self, cfg, vocab, shared=True, n_styles: int = N_STYLES):
        super().__init__(cfg, vocab, shared)
        self.style_emb = nn.Embedding(n_styles, self.d)
        nn.init.zeros_(self.style_emb.weight)
        self.style = 0
        self.sampler, self.steps, self.cfg_scale, self.sway = "euler", 16, 1.0, 0.0

    def set_sampling(self, sampler="euler", steps=16, cfg=1.0, sway=0.0):
        self.sampler, self.steps, self.cfg_scale, self.sway = sampler, int(steps), float(cfg), float(sway)
        return self

    def features(self, ids, mask, style):
        return self.text(ids, mask) + self.style_emb(style)[:, None]

    def text_stage(self, ids, mask):
        style = torch.full((ids.shape[0],), int(self.style), dtype=torch.long, device=ids.device)
        h = self.features(ids, mask, style)
        return h, self.chardur(h, mask)

    def condition(self, h, dur, mask, cw, wstart, fw, fp):
        """The aligner's frame condition from the letter durations, exactly as Acoustic.sound_stage builds it."""
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
        if self.sampler == "ema4":
            return super().sound_stage(h, dur, mask, cw, wstart, fw, fp, fmask, noise)
        cond = self.condition(h, dur, mask, cw, wstart, fw, fp)
        cos, sin = rope(torch.arange(fw.shape[1], device=h.device), self.dh)
        ts = time_grid(self.steps, self.sway)
        x = noise[:, 0]
        for k in range(self.steps):
            t, dt = ts[k], ts[k + 1] - ts[k]
            v = self.velocity(x, cond, t, fmask, cos, sin, self.cfg_scale)
            if self.sampler == "midpoint":
                v = self.velocity(x + 0.5 * dt * v, cond, t + 0.5 * dt, fmask, cos, sin, self.cfg_scale)
            x = x + dt * v
        return x


def time_grid(steps: int, sway: float = 0.0) -> list[float]:
    """steps + 1 times from 0 to 1; sway s (F5-TTS): t = u + s (cos(pi u / 2) - 1 + u), s < 0 = denser near 0."""
    us = [k / steps for k in range(steps + 1)]
    return [u + sway * (math.cos(math.pi * u / 2) - 1 + u) for u in us]


def losses(model: StudentAcoustic, bt: dict, p_cond=0.1, p_style=0.1, t_dist="uniform", gen=None):
    """(flow loss, duration loss) of one batch (torch tensors from student_data.collate on the device).

    gen: a torch.Generator for x0 / t / dropout (validation: fixed); None = the global RNG."""
    dev = bt["ids"].device
    B = bt["ids"].shape[0]
    mask = bt["ids"] != 0
    style = bt["style"]
    if p_style > 0:
        style = torch.where(torch.rand(B, device=dev, generator=gen) < p_style, torch.zeros_like(style), style)
    h = model.features(bt["ids"], mask, style)
    dur = model.chardur(h, mask)
    word = bt["cw"].clamp(min=0)
    pred_w = torch.zeros_like(bt["wf"]).scatter_add_(1, word, dur * mask)
    wmask = torch.arange(bt["wf"].shape[1], device=dev)[None] < bt["nw"][:, None]
    l_dur = (((torch.log1p(pred_w) - torch.log1p(bt["wf"])) ** 2) * wmask).sum() / wmask.sum()
    cond = model.condition(h, dur.detach(), mask, bt["cw"], bt["wstart"], bt["fw"], bt["fp"])
    if p_cond > 0:
        keep = (torch.rand(B, 1, 1, device=dev, generator=gen) >= p_cond).float()
        cond = cond * keep
    x1 = bt["x1"]
    T = x1.shape[1]
    x0 = torch.randn(x1.shape, device=dev, generator=gen)
    if t_dist == "logitnormal":
        t = torch.sigmoid(torch.randn(B, device=dev, generator=gen))
    elif isinstance(t_dist, torch.Tensor):
        t = t_dist.to(dev)
    else:
        t = torch.rand(B, device=dev, generator=gen)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    fmask = bt["fw"] >= 0
    cos, sin = rope(torch.arange(T, device=dev), model.dh)
    v = model.backbone(xt, cond, t, fmask, cos, sin)
    fm = fmask[..., None].float()
    l_flow = (((v - (x1 - x0)) ** 2) * fm).sum() / (fm.sum() * x1.shape[-1])
    return l_flow, l_dur


def to_device(batch: dict, dev) -> dict:
    return {k: torch.from_numpy(v).to(dev, non_blocking=True) for k, v in batch.items()}


def build(init: str = "ema", device="cpu") -> tuple[StudentAcoustic, dict]:
    """A StudentAcoustic on `device`: init "ema" = ema.pt's EMA weights (style_emb zero), "scratch" = random."""
    ck = torch.load(str(C.EMA_DIR / "ema.pt"), map_location="cpu", weights_only=True)
    model = StudentAcoustic(ck["cfg"], ck["vocab"], bool(ck["cfg"].get("shared_ada", True)))
    if init == "ema":
        missing, unexpected = model.load_state_dict(ck["ema"], strict=False)
        if unexpected or set(missing) != {"style_emb.weight"}:
            raise RuntimeError(f"ema.pt does not fit: missing {missing}, unexpected {unexpected}")
    elif init != "scratch":
        raise ValueError(init)
    return model.to(device), {"cfg": ck["cfg"], "vocab": ck["vocab"]}


def load_student(path, device="cuda", weights="ema") -> StudentAcoustic:
    """A trained student for inference: `path` a train.py checkpoint (weights "ema" or "model"), or "ema.pt"."""
    if str(path) == "ema.pt":
        model, _ = build("ema", "cpu")
    else:
        ck = torch.load(str(path), map_location="cpu", weights_only=True)
        model = StudentAcoustic(ck["cfg"], ck["vocab"], bool(ck["cfg"].get("shared_ada", True)))
        model.load_state_dict(ck[weights])
    return model.to(device).float().eval().requires_grad_(False)


def make_tts(model: StudentAcoustic, device="cuda", decoder=None):
    """ema_lightning.EMA around the student: EMA's chunker, timeline, pauses and decoder windows; trnorm decides
    what is said (the text given to say() must already be trnorm's spoken form, as bench.py's "ema" system)."""
    from ema_lightning import EMA
    from ema_lightning.decoder import load_decoder
    from ema_lightning.frontend import Frontend

    dec = decoder if decoder is not None else load_decoder(str(C.EMA_DIR / "decoder.pt"), device)
    fe = Frontend.__new__(Frontend)
    fe.vocab = frozenset(model.vocab)
    fe.spoken = lambda t: t
    return EMA._from_parts(model, dec, fe, device)
