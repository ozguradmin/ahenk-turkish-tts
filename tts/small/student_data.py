"""Data side of the small-model trainer (docs/KUCUK_MODEL_P1.md): no torch, numpy only, so the unit tests run in the
plain test venv.

- index rows: data/small/f022-latents/index.jsonl (encode_corpus.py); latents.npy float16 [sum frames, 64]
- word_bounds: each letter's word and that word's first letter, exactly as ema_lightning.engine.Engine.piece
  (a word runs from its first letter to the next word's first letter, so the space after it is its own)
- timeline: frame -> word (fw) and position inside the word (fp) from frames per word, exactly as Engine.plan
- collate: one padded batch as numpy arrays (ids, cw, wstart, fw, fp, x1, wf, nw, style)
- bucket_batches: length-sorted batches under a frame budget (B x longest clip), shuffled per epoch
- lr_at: linear warm-up, then cosine down to min_ratio x lr
- checkpoint files: ckpt_<step 8 digits>.pt, latest_ckpt / prune_ckpts (the newest `keep` plus every `keep_every`)
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

LATENT_DIM = 64
STYLES = ("neutral", "angry", "calm", "cheerful", "empathetic", "sad", "surprised", "laughter")
N_STYLES = len(STYLES)
CKPT_RE = re.compile(r"^ckpt_(\d{8})\.pt$")


def read_index(d, split: str | None = "train", use_only: bool = True, max_frames: int | None = None) -> list[dict]:
    """Rows of index.jsonl: the default training set (use) of one split (train / val / None for both)."""
    rows = [json.loads(x) for x in (Path(d) / "index.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    for r in rows:
        if use_only and not r.get("use"):
            continue
        if split is not None and r.get("split") != split:
            continue
        if max_frames is not None and r["frames"] > max_frames:
            continue
        out.append(r)
    return out


def check_row(r: dict) -> None:
    """ValueError if a row cannot be trained on as it is (the trainer checks every row once at start)."""
    if len(r["ids"]) != len(r["ema_text"]):
        raise ValueError(f"{r['id']}: {len(r['ids'])} ids for {len(r['ema_text'])} letters")
    cw = word_bounds(r["ema_text"])[0]
    nw = cw[-1] + 1 if cw else 0
    if len(r["word_frames"]) != nw:
        raise ValueError(f"{r['id']}: {len(r['word_frames'])} word_frames for {nw} words")
    if sum(r["word_frames"]) != r["frames"] or min(r["word_frames"]) < 1:
        raise ValueError(f"{r['id']}: word_frames sum {sum(r['word_frames'])} != frames {r['frames']} or a word < 1")
    if not 0 <= r["style_id"] < N_STYLES:
        raise ValueError(f"{r['id']}: style_id {r['style_id']}")


def word_bounds(text: str) -> tuple[list[int], list[int]]:
    """(cw, wstart) per letter, as Engine.piece: word index of each letter and the index of that word's first letter."""
    starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
    bounds = [0] + starts[1:] + [len(text)]
    cw, wstart = [], []
    for w, (a, b) in enumerate(zip(bounds, bounds[1:])):
        cw += [w] * (b - a)
        wstart += [a] * (b - a)
    return cw, wstart


def timeline(word_frames):
    """(fw, fp) per frame from frames per word, as Engine.plan: fp = frame index inside its word / word length."""
    import numpy as np

    n = np.asarray(word_frames, dtype=np.int64)
    fw = np.repeat(np.arange(len(n)), n)
    start = np.cumsum(n) - n
    fp = ((np.arange(len(fw)) - start[fw]).astype(np.float64) / n[fw].astype(np.float64)).astype(np.float32)
    return fw, fp


def collate(rows: list[dict], lat) -> dict:
    """One padded batch (numpy). Padding: ids 0, cw -1, wstart 0, fw -1, fp 0, x1 0, wf 0.

    ids/cw/wstart [B, L]; fw/fp [B, T]; x1 [B, T, 64] float32; wf [B, L] target frames per word (word index, not
    letter); nw [B] words; style [B]; frames [B]; letters [B]."""
    import numpy as np

    B = len(rows)
    L = max(len(r["ids"]) for r in rows)
    T = max(r["frames"] for r in rows)
    out = {"ids": np.zeros((B, L), np.int64), "cw": np.full((B, L), -1, np.int64), "wstart": np.zeros((B, L), np.int64),
           "fw": np.full((B, T), -1, np.int64), "fp": np.zeros((B, T), np.float32),
           "x1": np.zeros((B, T, LATENT_DIM), np.float32), "wf": np.zeros((B, L), np.float32),
           "nw": np.zeros(B, np.int64), "style": np.array([r["style_id"] for r in rows], np.int64),
           "frames": np.array([r["frames"] for r in rows], np.int64),
           "letters": np.array([len(r["ids"]) for r in rows], np.int64)}
    for b, r in enumerate(rows):
        n = len(r["ids"])
        out["ids"][b, :n] = r["ids"]
        cw, ws = word_bounds(r["ema_text"])
        out["cw"][b, :n] = cw
        out["wstart"][b, :n] = ws
        wfr = r["word_frames"]
        out["nw"][b] = len(wfr)
        out["wf"][b, : len(wfr)] = wfr
        fw, fp = timeline(wfr)
        out["fw"][b, : len(fw)] = fw
        out["fp"][b, : len(fp)] = fp
        out["x1"][b, : r["frames"]] = lat[r["offset"]: r["offset"] + r["frames"]].astype(np.float32)
    return out


def bucket_batches(rows: list[dict], frame_budget: int, max_batch: int, rng: random.Random) -> list[list[int]]:
    """Indices into rows, in batches of similar length: sorted by frames (ties broken at random, so batches differ
    between epochs), cut greedily so that len(batch) * longest <= frame_budget and len(batch) <= max_batch (a clip
    longer than the budget goes alone), then the batch order shuffled."""
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["frames"], rng.random()))
    batches, cur, longest = [], [], 0
    for i in order:
        f = rows[i]["frames"]
        if cur and ((len(cur) + 1) * max(longest, f) > frame_budget or len(cur) >= max_batch):
            batches.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, f)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def lr_at(step: int, lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """Learning rate at `step` (0-based): linear warm-up over `warmup` steps, cosine to min_ratio * lr at `total`."""
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if total <= warmup:
        return lr
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def ckpt_name(step: int) -> str:
    return f"ckpt_{step:08d}.pt"


def list_ckpts(run_dir) -> list[tuple[int, Path]]:
    d = Path(run_dir)
    if not d.exists():
        return []
    out = [(int(m.group(1)), p) for p in d.iterdir() if (m := CKPT_RE.match(p.name))]
    return sorted(out)


def latest_ckpt(run_dir) -> Path | None:
    ck = list_ckpts(run_dir)
    return ck[-1][1] if ck else None


def prune_ckpts(run_dir, keep: int = 2, keep_every: int = 0) -> list[Path]:
    """Delete checkpoint files except the newest `keep` and those whose step is a multiple of keep_every (0: none).
    Returns the deleted paths."""
    ck = list_ckpts(run_dir)
    gone = []
    for step, p in ck[:-keep] if keep > 0 else ck:
        if keep_every and step % keep_every == 0:
            continue
        p.unlink()
        gone.append(p)
    return gone


def append_jsonl(p, rec: dict) -> None:
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_student(spec: str) -> dict:
    """eval_student.py system spec 'student:NAME=CKPT[@sampler,steps,cfg,sway,decoder]' -> dict (defaults euler, 16,
    1.0, 0.0, None = EMA's decoder.pt)."""
    if not spec.startswith("student:") or "=" not in spec:
        raise ValueError(f"not a student spec: {spec!r}")
    name, rest = spec.split(":", 1)[1].split("=", 1)
    ckpt, _, samp = rest.partition("@")
    parts = (samp.split(",") if samp else []) + [""] * 5
    if not name or not ckpt or parts[0] not in ("", "euler", "midpoint", "ema4"):
        raise ValueError(f"bad student spec: {spec!r}")
    return {"name": name, "ckpt": ckpt, "sampler": parts[0] or "euler", "steps": int(parts[1] or 16),
            "cfg": float(parts[2] or 1.0), "sway": float(parts[3] or 0.0), "decoder": parts[4] or None}
