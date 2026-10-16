#!/usr/bin/env python3
"""f022 corpus -> VoxCPM2 AudioVAE latents + aligned EMA-alphabet text, training-ready (P0 task 3b, local GPU).

    <venv-student>/python tts/small/encode_corpus.py --corpus <xfer>/f022corpus <xfer>/f022corpus2 \
        --out data/small/f022-latents

Inputs per corpus folder: clips.jsonl (render metadata, Whisper word timings), voxcpm_train.jsonl (the training
manifest; its /data/<corpus>/wav/... paths are mapped to <folder>/wav/...), wav/*.wav (16 kHz), and for
f022corpus c_audit.jsonl: rows with ok=false (6 instructed c-errors) are skipped, as corpus-merge does.

Per clip:
  text      what the teacher actually read: render_spoken (e.g. "Vaay") else spoken (trnorm spoken form), with any
            "(instruction)" prefix stripped; ema_text = EMA alphabet of it (lowercase, 49 symbols, â->a)
  style     emotion label -> style id (neutral 0, angry 1, calm 2, cheerful 3, empathetic 4, sad 5, surprised 6,
            laughter 7); the instruction text itself is dropped (kept in the index for reference only)
  latents   mu of the VoxCPM2 AudioVAE encoder, [T, 64] float16, T = ceil(samples16k / 640), 25 Hz, raw scale
  word_frames  frames per EMA word (EMA's word = letters up to the next word start, incl. trailing space and
            punctuation), summing to T: Whisper word timings (stored in clips.jsonl; local Whisper word timestamps
            for clips without them) mapped letter by letter onto ema_text (difflib over letter streams after
            trnorm.normalize_for_eval per Whisper word); a word starts at its first letter's time, so silence
            between words goes to the preceding word and leading silence to the first word
Outputs in OUT: latents.npy ([sum T, 64] float16), index.jsonl (one row per clip: offset, frames, ids, word_frames,
flags, split), stats.json (counts, hours, timing, GPU memory, format description).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from audiovae import FPS, HOP, load_audiovae  # noqa: E402

STYLES = ["neutral", "angry", "calm", "cheerful", "empathetic", "sad", "surprised", "laughter"]
MAX_FRAMES = 3000  # EMA engine limit
MAX_WORD_FRAMES = 250


def letters(s: str) -> str:
    return "".join(ch for ch in s if ch.isalpha())


def ema_words(text: str) -> list[str]:
    """EMA engine.piece word split: a word runs from a non-space letter after a space to the next such letter."""
    starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
    bounds = [0] + starts[1:] + [len(text)]
    return [text[a:b] for a, b in zip(bounds, bounds[1:])]


def word_frames(ema_text: str, words: list, T: int) -> tuple[list[int] | None, dict]:
    """Frames per EMA word from Whisper word timings; (None, info) when the two texts do not match well.

    Whisper words -> trnorm.normalize_for_eval per word -> tokens (a word's span split over its tokens by letter
    count). EMA words and those tokens are matched as whole words (difflib); a matched EMA word starts at its
    token's start. Unmatched EMA words (mostly numbers, whose per-word reading differs from the in-context
    one: "38548" vs "üç sekiz beş dört sekiz") get start times interpolated by letter position between the
    neighbouring matched words. Accepted when >= 60 % of the Whisper words and >= 50 % of the EMA words
    match (Whisper words with digits not counted) and no run of unmatched EMA words is longer than 12."""
    from trnorm import normalize_for_eval

    ew = ema_words(ema_text)
    e_tok = [letters(w) for w in ew]
    info = {"n_words": len(ew)}
    merged = []  # Whisper splits "Antalya 'ya", "E -Devlete": glue apostrophe/hyphen pieces back on
    for w, s, e in words:
        if merged and w[:1] in "'’-" and len(w) > 1:
            pw, ps, _ = merged[-1]
            merged[-1] = [pw + w, ps, e]
        else:
            merged.append([w, s, e])
    n_e = sum(1 for t in e_tok if t)
    if not n_e:
        return None, {**info, "why": "no letters"}

    def match(join_words):
        """join_words: a digit-free Whisper word is one token ("e-devlet'e"); else its normalised words are."""
        h_tok, h_start, h_end, h_word = [], [], [], []
        for w, s, e in merged:
            digits = any(ch.isdigit() for ch in w)
            toks = [letters(C.ema_alphabet(t)) for t in normalize_for_eval(w).split()]
            toks = [t for t in toks if t]
            if join_words and not digits and toks:
                toks = ["".join(toks)]
            h_word.extend([not digits] * len(toks))
            n = sum(len(t) for t in toks)
            e = max(e, s + 1e-3)
            pos = 0
            for t in toks:
                h_tok.append(t)
                h_start.append(s + (e - s) * pos / n)
                pos += len(t)
                h_end.append(s + (e - s) * pos / n)
        if not h_tok:
            return None
        sm = difflib.SequenceMatcher(None, e_tok, h_tok, autojunk=False)
        anchor, h_hit = {}, 0
        for a, b, n in sm.get_matching_blocks():
            for i in range(n):
                if e_tok[a + i]:
                    anchor[a + i] = (h_start[b + i], h_end[b + i])
                    h_hit += h_word[b + i]
        run = longest = 0
        for k, t in enumerate(e_tok):
            if t:
                run = 0 if k in anchor else run + 1
                longest = max(longest, run)
        h_ratio, e_ratio = h_hit / max(1, sum(h_word)), len(anchor) / n_e  # h: Whisper words without digits
        inf = dict(h_match=round(h_ratio, 4), e_match=round(e_ratio, 4), longest_gap=longest,
                   align_mode="exact" if len(anchor) == n_e == len(h_tok) else "anchors", joined=join_words)
        ok = h_ratio >= 0.6 and e_ratio >= 0.5 and longest <= 12
        return ok, e_ratio, anchor, h_start, h_end, inf

    tries = [m for m in (match(False), match(True)) if m is not None]
    good = sorted([m for m in tries if m[0]], key=lambda m: -m[1])
    if not good:
        return None, {**info, **(tries[0][5] if tries else {}), "why": "texts differ"}
    _, _, anchor, h_start, h_end, inf = good[0]
    info.update(inf)
    L = np.cumsum([0] + [len(t) for t in e_tok])  # letter offset of each EMA word, and the total
    xs, ys = [], []
    if 0 not in anchor:
        xs.append(0.0), ys.append(h_start[0])
    for k in sorted(anchor):
        xs.append(float(L[k])), ys.append(anchor[k][0])
    last = max(anchor)
    xs.append(float(L[last + 1])), ys.append(anchor[last][1])
    if xs[-1] < L[-1]:
        xs.append(float(L[-1])), ys.append(max(h_end[-1], ys[-1]))
    ys = np.maximum.accumulate(np.array(ys))
    starts = [anchor[k][0] if k in anchor else float(np.interp(L[k], xs, ys)) for k in range(len(ew))]
    dur = T / FPS
    starts[0] = 0.0  # leading silence belongs to the first word
    b = np.maximum.accumulate(np.clip(np.array(starts + [dur]), 0, dur))
    fr = np.diff(np.round(b * FPS).astype(int))
    fr[-1] += T - int(fr.sum())
    for k in range(len(fr)):  # every word gets at least one frame, taken from the longest word
        while fr[k] < 1:
            j = int(np.argmax(fr))
            fr[j] -= 1
            fr[k] += 1
    assert int(fr.sum()) == T and (fr >= 1).all()
    return [int(v) for v in fr], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--out", default="data/small/f022-latents")
    ap.add_argument("--batch-seconds", type=float, default=160.0, help="audio seconds per encoder batch")
    ap.add_argument("--val-share", type=float, default=0.02)
    args = ap.parse_args()
    out = C.REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    vocab = C.EMA_VOCAB
    stoi = {ch: i for i, ch in enumerate(vocab)}

    # ------------------------------------------------------------------ manifest
    clips, skipped = [], Counter()
    for folder in map(Path, args.corpus):
        name = folder.name
        meta = {r["id"]: r for r in C.read_jsonl(folder / "clips.jsonl")}
        bad = set()
        if (folder / "c_audit.jsonl").exists():
            bad = {r["id"] for r in C.read_jsonl(folder / "c_audit.jsonl") if not r.get("ok")}
        for m in C.read_jsonl(folder / "voxcpm_train.jsonl"):
            rel = m["audio"].split(f"/{name}/", 1)[1]  # /data/<name>/wav/x.wav -> wav/x.wav
            wav = folder / rel
            cid = Path(rel).stem
            r = meta.get(cid)
            if r is None:
                skipped["no clip metadata"] += 1
                continue
            if cid in bad:
                skipped["c_audit ok=false"] += 1
                continue
            if not wav.exists():
                skipped["wav missing"] += 1
                continue
            text, instr = C.strip_instruction(r.get("render_spoken") or r["spoken"])
            _, train_instr = C.strip_instruction(m["text"])
            style = "laughter" if r["category"] == "laughter" else (r.get("emotion") or "neutral")
            et = C.ema_alphabet(text, vocab)
            clips.append({"id": cid, "corpus": name, "category": r["category"], "style": style,
                          "style_id": STYLES.index(style), "instruction": train_instr or r.get("instruction"),
                          "text": text, "ema_text": et, "wav": str(wav), "duration": m["duration"],
                          "words": r.get("words"), "utmos": r.get("utmos"), "sim": r.get("sim")})
    print(f"clips {len(clips)}, skipped {dict(skipped)}", flush=True)

    # ------------------------------------------------------------------ Whisper word timings where missing
    need = [c for c in clips if not c["words"]]
    t_whisper = 0.0
    if need:
        judge = C.Judge("cuda")
        t0 = time.time()
        for c in need:
            j = judge.transcribe(C.load_16k(c["wav"]), words=True)
            c["words"], c["words_src"], c["local_hyp"] = j["words"], "local-whisper", j["hyp"]
            sc = C.score_row([c["text"]], j["hyp"])
            c["local_cer_edits"] = sc["char_edits"]
        t_whisper = time.time() - t0
        print(f"whisper word timings for {len(need)} clips: {t_whisper:.0f} s", flush=True)
        del judge
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ encode (GPU, fp32, length-sorted batches)
    dev = "cuda"
    # TF32 convolutions moved latents by up to 0.04 (mean 0.001) vs CPU; full fp32 matches CPU to 2e-5
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    vae = load_audiovae(C.VOXCPM_DIR / "audiovae.pth", dev, with_decoder=False)
    order = sorted(range(len(clips)), key=lambda i: clips[i]["duration"])
    lat = [None] * len(clips)
    t_load = t_enc = 0.0
    t_all = time.time()
    i = 0
    while i < len(order):
        batch, longest = [], 0.0
        while i < len(order) and (not batch or (len(batch) + 1) * max(longest, clips[order[i]]["duration"]) <= args.batch_seconds):
            batch.append(order[i])
            longest = max(longest, clips[order[i]]["duration"])
            i += 1
        t0 = time.time()
        xs = [C.load_16k(clips[k]["wav"]) for k in batch]
        t_load += time.time() - t0
        n = max(len(x) for x in xs)
        X = torch.zeros(len(xs), 1, n)
        for b, x in enumerate(xs):
            X[b, 0, : len(x)] = torch.from_numpy(x)
        torch.cuda.synchronize()
        t0 = time.time()
        Z = vae.encode(X.to(dev))
        torch.cuda.synchronize()
        t_enc += time.time() - t0
        Z = Z.transpose(1, 2).float().cpu()
        for b, (k, x) in enumerate(zip(batch, xs)):
            T = -(-len(x) // HOP)
            lat[k] = Z[b, :T].numpy().astype(np.float16)
            clips[k]["samples16k"] = len(x)
    t_all = time.time() - t_all
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"encoded {len(clips)} clips: total {t_all:.0f} s (GPU encode {t_enc:.0f} s, load+resample {t_load:.0f} s), "
          f"peak {peak:.0f} MiB", flush=True)

    # ------------------------------------------------------------------ align, flag, write
    rows, offset, why = [], 0, Counter()
    for c, z in zip(clips, lat):
        T = int(z.shape[0])
        wf, info = word_frames(c["ema_text"], c["words"], T)
        if wf is None:
            why[info.get("why")] += 1
        ids = [stoi.get(ch, 1) for ch in c["ema_text"]]
        flags = []
        if c["style"] == "laughter":
            flags.append("laughter")
        if wf is None:
            flags.append("unaligned")
        elif max(wf) > MAX_WORD_FRAMES:
            flags.append("word_over_250_frames")
        if T > MAX_FRAMES:
            flags.append("over_3000_frames")
        if c["duration"] > 10.0:
            flags.append("over_10s")
        if c["ema_text"][-1:] not in ".!?":
            flags.append("no_final_punct")
        if 1 in ids:
            flags.append("unk_symbol")
        if c.get("local_cer_edits"):
            flags.append("local_whisper_cer>0")
        h = int(hashlib.sha1(c["id"].encode()).hexdigest()[:8], 16) / 2**32
        split = "val" if h < args.val_share else "train"
        hard = {"laughter", "unaligned", "over_3000_frames", "unk_symbol", "word_over_250_frames"}
        rows.append({"id": c["id"], "corpus": c["corpus"], "category": c["category"], "style": c["style"],
                     "style_id": c["style_id"], "text": c["text"], "ema_text": c["ema_text"], "ids": ids,
                     "offset": offset, "frames": T, "duration": c["duration"], "word_frames": wf,
                     "align": info, "words_src": c.get("words_src", "clips.jsonl"), "words": c["words"], "flags": flags,
                     "split": split, "use": not (set(flags) & hard), "instruction": c["instruction"],
                     "utmos": c["utmos"], "sim": c["sim"], "wav": c["wav"]})
        offset += T
    allz = np.concatenate(lat, 0)
    np.save(out / "latents.npy", allz)
    C.write_jsonl(out / "index.jsonl", rows)

    use = [r for r in rows if r["use"]]
    by = lambda key, rr: dict(Counter(r[key] for r in rr))  # noqa: E731
    hours = lambda rr: round(sum(r["frames"] for r in rr) / FPS / 3600, 3)  # noqa: E731
    z32 = allz.astype(np.float32)
    stats = {
        "format": {"latents.npy": "float16 [sum frames, 64]; clip i = latents[offset : offset + frames]",
                   "index.jsonl": "one clip per line: id, corpus, category, style, style_id, text, ema_text, ids "
                                  "(EMA vocab indices), offset, frames, duration, word_frames (per EMA word, sums to "
                                  "frames), align, words_src, words (Whisper [word, start, end]), flags, split (train/val, sha1 of id), use (default training set), "
                                  "instruction (reference only), utmos, sim, wav",
                   "latent": "VoxCPM2 AudioVAE v2 encoder mu at 16 kHz, hop 640 (25 Hz), raw scale (x1.0)",
                   "vocab": vocab, "styles": STYLES},
        "clips": len(rows), "hours_all": hours(rows), "skipped": dict(skipped),
        "use": len(use), "hours_use": hours(use),
        "use_train": sum(r["split"] == "train" for r in use), "use_val": sum(r["split"] == "val" for r in use),
        "hours_use_train": hours([r for r in use if r["split"] == "train"]),
        "by_category_all": by("category", rows), "by_style_all": by("style", rows), "by_style_use": by("style", use),
        "hours_by_style_use": {s: hours([r for r in use if r["style"] == s]) for s in STYLES},
        "flags": dict(Counter(f for r in rows for f in r["flags"])), "unaligned_why": dict(why),
        "align_e_match_mean": round(float(np.mean([r["align"]["e_match"] for r in rows if "e_match" in r["align"]])), 4),
        "align_modes": dict(Counter(r["align"].get("align_mode", "none") for r in rows)),
        "words_from_local_whisper": sum(r["words_src"] == "local-whisper" for r in rows),
        "frames_total": int(allz.shape[0]), "latents_mb": round(allz.nbytes / 2**20, 1),
        "frames_per_clip": {"mean": round(float(np.mean([r["frames"] for r in rows])), 1),
                            "max": int(max(r["frames"] for r in rows))},
        "letters_per_clip_mean": round(float(np.mean([len(r["ids"]) for r in rows])), 1),
        "frames_per_letter": round(sum(r["frames"] for r in use) / sum(len(r["ids"]) for r in use), 3),
        "latent_stats": {"mean": round(float(z32.mean()), 4), "std": round(float(z32.std()), 4),
                         "dim_std_min": round(float(z32.std(0).min()), 4), "dim_std_max": round(float(z32.std(0).max()), 4)},
        "timing": {"whisper_word_timings_s": round(t_whisper, 1), "encode_total_s": round(t_all, 1),
                   "encode_gpu_s": round(t_enc, 1), "load_resample_s": round(t_load, 1),
                   "audio_x_realtime_gpu": round(sum(r["duration"] for r in rows) / max(t_enc, 1e-9), 1)},
        "gpu": C.gpu_name(), "gpu_peak_mib": round(peak, 0), "batch_seconds": args.batch_seconds,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "format"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
