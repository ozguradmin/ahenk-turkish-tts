#!/usr/bin/env python3
"""Prepare the Antalia voice corpus for Piper training (see tts/README.md, section 2).

Corpus: cloud0day3/antalia-voice-corpus (CC-BY-4.0), one consenting professional female voice
actor, 1,073 clips, 5.008 h, 24 kHz mono 16-bit WAV + metadata.jsonl.

Steps
 1. metadata.jsonl at a pinned dataset revision.
 2. Training text = trnorm.normalize(transcript) (whitespace collapsed first; the scripts contain
    paragraph breaks), so the voice learns exactly the spoken forms the inference front-end
    produces. It is compared with the corpus' own `normalized_transcript`, and both are checked
    against the corpus' independent ASR transcript (`asr_text`, ElevenLabs Scribe v2) to see which
    one matches the audio (trnorm.normalize_for_eval on every side).
 3. Filters, each counted in the report: duration outside [--min-seconds, --max-seconds]; trnorm
    text that disagrees with the ASR transcript (CER > --max-asr-cer); empty text.
 4. Held-out: --heldout clips, stratified over the 11 recording categories, unique texts, clean ASR
    agreement. Listed in heldout.txt; piper_fit.py validates on exactly these and never trains on them.
 5. Audio: download (sha256-checked), resample 24 kHz -> --sample-rate (soxr VHQ), 16-bit PCM WAV.
 6. Outputs in --out: wav/, metadata.csv (Piper format: csv with "|" delimiter, `file|text`, all kept
    clips = train + held-out), heldout.txt, heldout.csv, clips.jsonl (per-clip details),
    prep_report.json and PREP_REPORT.md.
 7. --piper-cache: also run piper's own preprocessing (espeak-ng phonemes, silence trim,
    spectrograms) into <out>/piper_cache/<sha1(metadata.csv)[:12]>-<voice>/, the directory train.sh
    uses, so a paid GPU job does not spend its first minutes on it.

Examples
    python tts/prepare_antalia.py                                # full corpus -> $TTS_DATA/antalia
    python tts/prepare_antalia.py --report-only                  # metadata analysis only (no audio)
    python tts/prepare_antalia.py --out data/tts_train/antalia_smoke --limit 40 --heldout 4
"""

from __future__ import annotations

import argparse
import collections
import csv
import difflib
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ID = "cloud0day3/antalia-voice-corpus"
REVISION = "9e67eeb2616f4e7ed6b607155e1ab3d1fabd912e"  # 2026-09-14, the only published revision
SOURCE_RATE = 24000
ATTRIBUTION = (
    "Antalia (Patientdesk.ai), Antalia Turkish single-speaker scripted speech, CC BY 4.0, "
    "https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus"
)
DURATION_BINS = [0, 3, 5, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 20.5, 21, 25]

TTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TTS_DIR.parent


# --------------------------------------------------------------------------- text helpers
def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_ASR_REPEAT = re.compile(r"(?<!\S)(\S*[\d.\-:/%]\S*)(?:\s+\1)+(?!\S)")


def dedup_asr(text: str) -> str:
    """Scribe v2 writes some tokens two to four times in a row ("389,50 389,50 TL",
    "www.x.com www.x.com", "e-posta e-posta"); collapse immediate repeats of tokens that hold a
    digit or one of . - : / %. Plain repeated words ("evet evet") are left alone."""
    return _ASR_REPEAT.sub(r"\1", text)


def levenshtein(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    return levenshtein(hyp, ref) / max(1, len(ref))


def diff_spans(a: str, b: str) -> list[tuple[str, str]]:
    """Word spans that differ between a and b (difflib opcodes, equal parts dropped)."""
    wa, wb = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    return [(" ".join(wa[i1:i2]), " ".join(wb[j1:j2])) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


# --------------------------------------------------------------------------- stats helpers
def quantiles(values: list[float]) -> dict:
    v = sorted(values)
    if not v:
        return {}

    def q(p: float) -> float:
        return round(v[min(len(v) - 1, int(p * len(v)))], 3)

    return {"n": len(v), "min": round(v[0], 3), "p10": q(0.10), "p25": q(0.25), "median": round(statistics.median(v), 3),
            "p75": q(0.75), "p90": q(0.90), "p95": q(0.95), "p99": q(0.99), "max": round(v[-1], 3),
            "mean": round(statistics.mean(v), 4)}


def histogram(durations: list[float]) -> list[dict]:
    out = []
    for lo, hi in zip(DURATION_BINS, DURATION_BINS[1:]):
        sel = [d for d in durations if lo <= d < hi]
        out.append({"bin": f"{lo}-{hi}", "clips": len(sel), "hours": round(sum(sel) / 3600, 3)})
    return out


# --------------------------------------------------------------------------- selection
def pick_stratified(clips: list[dict], n: int, seed: int, distinct_prompts: bool = True) -> list[dict]:
    """Round-robin over categories (sorted), random order inside each category (seeded); with
    distinct_prompts, a category prefers clips from parent recordings it has not used yet."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = collections.defaultdict(list)
    for c in sorted(clips, key=lambda c: c["clip_id"]):
        by_cat[c["category"]].append(c)
    for v in by_cat.values():
        rng.shuffle(v)
    picked: list[dict] = []
    used_prompts: dict[str, set] = collections.defaultdict(set)
    cats = sorted(by_cat)
    while len(picked) < n and any(by_cat.values()):
        for cat in cats:
            if len(picked) >= n or not by_cat[cat]:
                continue
            pool = by_cat[cat]
            idx = 0
            if distinct_prompts:
                fresh = [i for i, c in enumerate(pool) if c["prompt_id"] not in used_prompts[cat]]
                idx = fresh[0] if fresh else 0
            c = pool.pop(idx)
            used_prompts[cat].add(c["prompt_id"])
            picked.append(c)
    return picked


# --------------------------------------------------------------------------- audio
def resample_one(src: Path, dst: Path, sha256: str, rate: int) -> tuple[float, float]:
    """sha256-check src, resample to `rate`, write 16-bit PCM atomically. Returns (seconds, peak)."""
    import numpy as np
    import soundfile as sf
    import soxr

    if dst.exists():
        info = sf.info(str(dst))
        return info.frames / info.samplerate, float("nan")
    data = src.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if sha256 and got != sha256:
        raise RuntimeError(f"sha256 mismatch for {src.name}: {got} != {sha256}")
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # the corpus is mono; this is a no-op guard
    if sr != rate:
        audio = soxr.resample(audio, sr, rate, quality="VHQ")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    audio = np.clip(audio, -1.0, 1.0)
    tmp = dst.with_name(f".{dst.name}.tmp{os.getpid()}.wav")
    sf.write(str(tmp), audio, rate, subtype="PCM_16")
    tmp.replace(dst)
    return len(audio) / rate, peak


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data_root = Path(os.environ.get("TTS_DATA", REPO_ROOT / "data" / "tts_train"))
    ap.add_argument("--out", type=Path, default=data_root / "antalia", help="output directory [%(default)s]")
    ap.add_argument("--revision", default=REVISION, help="dataset revision (commit) [%(default)s]")
    ap.add_argument("--sample-rate", type=int, default=22050, help="Piper 'medium' rate [%(default)s]")
    ap.add_argument("--min-seconds", type=float, default=1.0, help="drop shorter clips [%(default)s]")
    ap.add_argument("--max-seconds", type=float, default=20.5,
                    help="drop longer clips [%(default)s]; the corpus segmenter capped clips at 20 s of speech "
                         "+ 2 x 0.12 s padding, so nothing exceeds 20.24 s (see PREP_REPORT.md)")
    ap.add_argument("--max-asr-cer", type=float, default=0.08,
                    help="drop clips whose trnorm text disagrees with the corpus ASR transcript by more than "
                         "this CER; 0.08 is just above the worst clip without digits/acronyms (0.072). "
                         "<= 0 disables [%(default)s]")
    ap.add_argument("--heldout", type=int, default=20, help="held-out validation clips [%(default)s]")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only N clips in total (held-out included), stratified; for smoke tests")
    ap.add_argument("--workers", type=int, default=2, help="download/resample threads [%(default)s]")
    ap.add_argument("--report-only", action="store_true", help="analyse metadata only; no audio download")
    ap.add_argument("--no-phonemes", action="store_true", help="skip espeak-ng phoneme statistics")
    ap.add_argument("--piper-cache", action="store_true",
                    help="also run piper's preprocessing (phonemes, trimmed audio, spectrograms) for train.sh")
    ap.add_argument("--espeak-voice", default="tr")
    args = ap.parse_args()

    t_start = time.time()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(data_root / "hf_home"))
    from huggingface_hub import hf_hub_download, snapshot_download
    from trnorm import normalize, normalize_for_eval

    meta_path = hf_hub_download(REPO_ID, "metadata.jsonl", repo_type="dataset", revision=args.revision)
    rows = [json.loads(line) for line in open(meta_path, encoding="utf-8") if line.strip()]
    print(f"[prepare] {len(rows)} clips in metadata.jsonl @ {args.revision[:10]}", flush=True)

    # ---- text: trnorm, comparison with the corpus normalizer, agreement with the ASR transcript
    raw_counts = collections.Counter(collapse_ws(r["transcript"]) for r in rows)
    clips: list[dict] = []
    for r in rows:
        raw = collapse_ws(r["transcript"])
        theirs = collapse_ws(r["normalized_transcript"])
        try:
            text = normalize(raw)
            err = ""
        except Exception as e:  # noqa: BLE001  (report and drop, never crash on one row)
            text, err = "", f"{type(e).__name__}: {e}"
        asr = dedup_asr(collapse_ws(r.get("asr_text") or ""))
        e_ours, e_theirs, e_asr = normalize_for_eval(text), normalize_for_eval(theirs), normalize_for_eval(asr)
        clips.append({
            "clip_id": r["clip_id"], "file_name": r["file_name"], "sha256": r.get("sha256", ""),
            "wav": f"{r['clip_id']}.wav", "duration": float(r["duration_seconds"]),
            "category": r["campaign_category"], "prompt_id": r["prompt_id"], "segment_index": r["segment_index"],
            "raw": raw, "text": text, "theirs": theirs, "asr": asr, "normalize_error": err,
            "cer_asr": round(cer(e_ours, e_asr), 4) if e_asr else None,
            "cer_asr_theirs": round(cer(e_theirs, e_asr), 4) if e_asr else None,
            "eval_ours": e_ours, "eval_theirs": e_theirs, "eval_asr": e_asr,
            "unique_text": raw_counts[raw] == 1,
            "has_special": bool(re.search(r"\d|\b[A-ZÇĞİÖŞÜ]{2,}\b|[%/₺$€]", raw)),
        })

    # ---- filters
    for c in clips:
        reason = ""
        if c["normalize_error"] or not c["text"].strip():
            reason = "empty_or_normalize_error"
        elif c["duration"] < args.min_seconds:
            reason = f"shorter_than_{args.min_seconds:g}s"
        elif c["duration"] > args.max_seconds:
            reason = f"longer_than_{args.max_seconds:g}s"
        elif args.max_asr_cer > 0 and c["cer_asr"] is not None and c["cer_asr"] > args.max_asr_cer:
            reason = f"asr_cer_above_{args.max_asr_cer:g}"
        c["drop_reason"] = reason
    kept = [c for c in clips if not c["drop_reason"]]

    # ---- held-out (validation) and optional smoke-test subset
    eligible = [c for c in kept if c["unique_text"] and (c["cer_asr"] or 0) <= 0.03 and c["duration"] >= 3.0]
    heldout = pick_stratified(eligible, args.heldout, args.seed)
    held_ids = {c["clip_id"] for c in heldout}
    train = [c for c in kept if c["clip_id"] not in held_ids]
    if args.limit:
        n_train = max(0, args.limit - len(heldout))
        train = pick_stratified(train, n_train, args.seed + 1, distinct_prompts=False)
    train_ids = {c["clip_id"] for c in train}
    for c in clips:
        c["split"] = ("heldout" if c["clip_id"] in held_ids else "train" if c["clip_id"] in train_ids
                      else "dropped" if c["drop_reason"] else "unused")
    selected = sorted(heldout + train, key=lambda c: c["clip_id"])

    # ---- espeak-ng phoneme statistics (same phonemizer/espeak build as training and inference)
    phon = None
    if not args.no_phonemes:
        try:
            from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP, phonemes_to_ids
            from piper.phonemize_espeak import EspeakPhonemizer

            ph = EspeakPhonemizer()
            inventory: collections.Counter = collections.Counter()
            missing: collections.Counter = collections.Counter()
            lengths, ids_per_s = [], []
            for c in kept:
                sentences = ph.phonemize(args.espeak_voice, c["text"])
                n_ids = 0
                for s in sentences:
                    inventory.update(s)
                    missing.update(p for p in s if p not in DEFAULT_PHONEME_ID_MAP)
                    n_ids += len(phonemes_to_ids(s))
                c["phoneme_ids"] = n_ids
                lengths.append(n_ids)
                ids_per_s.append(n_ids / c["duration"])
            phon = {"espeak_voice": args.espeak_voice, "phoneme_ids_per_clip": quantiles(lengths),
                    "phoneme_ids_per_second": quantiles(ids_per_s), "inventory_size": len(inventory),
                    "inventory": dict(inventory.most_common()), "missing_from_id_map": dict(missing)}
        except ImportError as e:
            print(f"[prepare] phoneme statistics skipped ({e})", flush=True)

    # ---- report
    report = build_report(args, rows, clips, kept, heldout, train, phon)
    (out / "prep_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "PREP_REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    with open(out / "clips.jsonl", "w", encoding="utf-8") as f:
        for c in clips:
            f.write(json.dumps({k: v for k, v in c.items() if not k.startswith("eval_")}, ensure_ascii=False) + "\n")
    print(f"[prepare] report: {out / 'PREP_REPORT.md'}", flush=True)
    print(f"[prepare] kept {len(kept)}/{len(clips)} clips ({sum(c['duration'] for c in kept) / 3600:.3f} h); "
          f"selected train {len(train)} + held-out {len(heldout)}", flush=True)
    if args.report_only:
        return 0

    # ---- audio: download (only what is needed), check, resample
    wav_dir = out / "wav"
    wav_dir.mkdir(exist_ok=True)
    t0 = time.time()
    snap = Path(snapshot_download(REPO_ID, repo_type="dataset", revision=args.revision,
                                  allow_patterns=[c["file_name"] for c in selected], max_workers=max(1, args.workers * 2)))
    print(f"[prepare] downloaded/verified {len(selected)} source files in {time.time() - t0:.1f} s", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(lambda c: resample_one(snap / c["file_name"], wav_dir / c["wav"], c["sha256"],
                                                       args.sample_rate), selected))
    peaks = [p for _, p in results if not math.isnan(p)]
    for c, (secs, _) in zip(selected, results):
        if abs(secs - c["duration"]) > 0.05:
            print(f"[prepare] WARNING {c['wav']}: {secs:.3f} s after resampling vs {c['duration']:.3f} s in metadata")
    print(f"[prepare] resampled {len(selected)} clips to {args.sample_rate} Hz in {time.time() - t0:.1f} s"
          + (f" (max peak {max(peaks):.3f})" if peaks else ""), flush=True)

    # ---- Piper files
    with open(out / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", lineterminator="\n")  # quotes fields with " so csv.reader round-trips
        for c in selected:
            w.writerow([c["wav"], c["text"]])
    (out / "heldout.txt").write_text("".join(f"{c['wav']}\n" for c in sorted(heldout, key=lambda c: c["clip_id"])),
                                     encoding="utf-8")
    with open(out / "heldout.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", lineterminator="\n")
        for c in sorted(heldout, key=lambda c: c["clip_id"]):
            w.writerow([c["wav"], c["text"]])
    (out / "ATTRIBUTION.txt").write_text(ATTRIBUTION + "\n", encoding="utf-8")
    sha = hashlib.sha1((out / "metadata.csv").read_bytes()).hexdigest()[:12]
    print(f"[prepare] wrote {out / 'metadata.csv'} ({len(selected)} rows, sha1 {sha}), heldout.txt ({len(heldout)})",
          flush=True)

    if args.piper_cache:
        build_piper_cache(out, sha, args)
    print(f"[prepare] done in {time.time() - t_start:.1f} s", flush=True)
    return 0


def build_piper_cache(out: Path, sha: str, args) -> None:
    """piper's VitsDataModule.prepare_data(): espeak-ng phonemes, VAD silence trim, spectrograms."""
    import logging

    from piper.train.vits.dataset import VitsDataModule

    logging.basicConfig(level=logging.INFO)
    cache_dir = out / "piper_cache" / f"{sha}-{args.espeak_voice}"
    t0 = time.time()
    dm = VitsDataModule(csv_path=out / "metadata.csv", cache_dir=cache_dir, espeak_voice=args.espeak_voice,
                        config_path=cache_dir / "config.json", voice_name="tr_TR-antalia-medium",
                        sample_rate=args.sample_rate, audio_dir=out / "wav")
    dm.prepare_data()
    print(f"[prepare] piper cache {cache_dir} built in {time.time() - t0:.1f} s", flush=True)


# --------------------------------------------------------------------------- reporting
def build_report(args, rows, clips, kept, heldout, train, phon) -> dict:
    hours = lambda cs: round(sum(c["duration"] for c in cs) / 3600, 3)  # noqa: E731
    durs = [c["duration"] for c in clips]
    by_cat = collections.defaultdict(list)
    for c in clips:
        by_cat[c["category"]].append(c["duration"])
    drops = collections.Counter(c["drop_reason"] for c in clips if c["drop_reason"])

    # trnorm vs corpus normalizer
    identical = [c for c in clips if c["text"] == c["theirs"]]
    same_words = [c for c in clips if c["text"] != c["theirs"] and c["eval_ours"] == c["eval_theirs"]]
    word_diff = [c for c in clips if c["eval_ours"] != c["eval_theirs"]]
    with_asr = [c for c in word_diff if c["cer_asr"] is not None]
    spans = collections.Counter()
    for c in word_diff:
        for theirs, ours in diff_spans(c["eval_theirs"], c["eval_ours"]):
            spans[(theirs, ours)] += 1
    asr_all = [c["cer_asr"] for c in clips if c["cer_asr"] is not None]
    asr_theirs = [c["cer_asr_theirs"] for c in clips if c["cer_asr_theirs"] is not None]
    clean = [c["cer_asr"] for c in clips if c["cer_asr"] is not None and not c["has_special"]]
    special = [c["cer_asr"] for c in clips if c["cer_asr"] is not None and c["has_special"]]
    vs_asr_spans = collections.Counter()
    for c in clips:
        if c["cer_asr"] is not None and c["cer_asr"] > 0.03:
            for ours, asr in diff_spans(c["eval_ours"], c["eval_asr"]):
                vs_asr_spans[(ours, asr)] += 1
    chars = collections.Counter(ch for c in kept for ch in c["text"] if not ch.isalpha() and ch != " ")
    tl_clips = [c for c in clips if re.search(r"\bTL\b", c["raw"])]

    rep = {
        "corpus": {"repo": REPO_ID, "revision": args.revision, "clips": len(rows), "hours": hours(clips),
                   "attribution": ATTRIBUTION, "source_sample_rate": SOURCE_RATE},
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "durations": {"all": quantiles(durs), "histogram": histogram(durs),
                      "above": {f">{t}s": {"clips": sum(d > t for d in durs), "hours": round(sum(d for d in durs if d > t) / 3600, 3)}
                                for t in (10, 15, 18, 19, 20, 20.5)},
                      "by_category": {k: {"clips": len(v), "median": round(statistics.median(v), 2), "max": max(v)}
                                      for k, v in sorted(by_cat.items())}},
        "filters": {reason: {"clips": n, "hours": hours([c for c in clips if c["drop_reason"] == reason])}
                    for reason, n in drops.most_common()},
        "kept": {"clips": len(kept), "hours": hours(kept)},
        "selected": {"train": {"clips": len(train), "hours": hours(train)},
                     "heldout": {"clips": len(heldout), "hours": hours(heldout),
                                 "by_category": dict(collections.Counter(c["category"] for c in heldout)),
                                 "ids": sorted(c["clip_id"] for c in heldout)}},
        "text_vs_corpus_normalizer": {
            "identical": len(identical), "same_words_punctuation_or_case_only": len(same_words),
            "different_words": len(word_diff),
            "different_words_by_category": dict(collections.Counter(c["category"] for c in word_diff).most_common()),
            "closer_to_asr": {"trnorm": sum(c["cer_asr"] < c["cer_asr_theirs"] for c in with_asr),
                              "corpus_normalizer": sum(c["cer_asr_theirs"] < c["cer_asr"] for c in with_asr),
                              "tie": sum(c["cer_asr"] == c["cer_asr_theirs"] for c in with_asr)},
            "top_spans_corpus_vs_trnorm": [{"corpus": a, "trnorm": b, "count": n} for (a, b), n in spans.most_common(40)],
        },
        "asr_agreement": {
            "cer_trnorm_vs_asr": quantiles(asr_all), "cer_corpus_normalizer_vs_asr": quantiles(asr_theirs),
            "cer_trnorm_vs_asr_clean_text": quantiles(clean), "cer_trnorm_vs_asr_digits_or_acronyms": quantiles(special),
            "clips_above": {f">{t}": sum(x > t for x in asr_all) for t in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20)},
            "top_spans_trnorm_vs_asr": [{"trnorm": a, "asr": b, "count": n} for (a, b), n in vs_asr_spans.most_common(30)],
            "TL_clips": {"clips": len(tl_clips), "TL_tokens": sum(len(re.findall(r"\bTL\b", c["raw"])) for c in tl_clips),
                         "asr_writes_turk_lirasi": sum(len(re.findall(r"Türk lira", c["asr"])) for c in tl_clips)},
        },
        "text_characters_non_letter": dict(chars.most_common()),
        "phonemes": phon,
    }
    worst = sorted([c for c in clips if c["cer_asr"] is not None], key=lambda c: -c["cer_asr"])[:12]
    rep["asr_agreement"]["worst"] = [{"clip_id": c["clip_id"], "category": c["category"], "cer": c["cer_asr"],
                                      "trnorm": c["text"][:240], "asr": c["asr"][:240]} for c in worst]
    examples = sorted(word_diff, key=lambda c: (c["category"], c["clip_id"]))
    rng = random.Random(args.seed)
    rep["text_vs_corpus_normalizer"]["examples"] = [
        {"clip_id": c["clip_id"], "category": c["category"], "raw": c["raw"][:240], "corpus": c["theirs"][:240],
         "trnorm": c["text"][:240], "asr": c["asr"][:240], "cer_trnorm": c["cer_asr"], "cer_corpus": c["cer_asr_theirs"]}
        for c in rng.sample(examples, min(8, len(examples)))]
    return rep


def render_markdown(r: dict) -> str:
    L = []
    a = L.append
    c = r["corpus"]
    a("# Antalia corpus preparation report (generated by `tts/prepare_antalia.py`)\n")
    a(f"Corpus `{c['repo']}` @ `{c['revision'][:10]}`: **{c['clips']} clips, {c['hours']} h**, {c['source_sample_rate']} Hz.  ")
    a(f"Attribution: {c['attribution']}\n")
    s = r["settings"]
    a(f"Settings: min {s['min_seconds']} s, max {s['max_seconds']} s, max ASR CER {s['max_asr_cer']}, "
      f"held-out {s['heldout']}, limit {s['limit'] or 'none'}, seed {s['seed']}, rate {s['sample_rate']} Hz.\n")
    d = r["durations"]
    a("## Clip durations\n")
    q = d["all"]
    a(f"min {q['min']} · p10 {q['p10']} · p25 {q['p25']} · **median {q['median']}** · p75 {q['p75']} · p90 {q['p90']} · "
      f"p99 {q['p99']} · max {q['max']} s (mean {q['mean']})\n")
    a("| seconds | clips | hours |\n|---|---:|---:|")
    for b in d["histogram"]:
        if b["clips"]:
            a(f"| {b['bin']} | {b['clips']} | {b['hours']} |")
    a("")
    a("| longer than | clips | hours |\n|---|---:|---:|")
    for k, v in d["above"].items():
        a(f"| {k} | {v['clips']} | {v['hours']} |")
    a("\n| category | clips | median s | max s |\n|---|---:|---:|---:|")
    for k, v in d["by_category"].items():
        a(f"| {k} | {v['clips']} | {v['median']} | {v['max']} |")
    a("\n## Filters\n")
    a("| reason | clips | hours |\n|---|---:|---:|")
    for k, v in r["filters"].items():
        a(f"| {k} | {v['clips']} | {v['hours']} |")
    a(f"\nKept **{r['kept']['clips']} clips, {r['kept']['hours']} h**. Selected: train {r['selected']['train']['clips']} "
      f"({r['selected']['train']['hours']} h), held-out {r['selected']['heldout']['clips']} "
      f"({r['selected']['heldout']['hours']} h) {r['selected']['heldout']['by_category']}.\n")
    t = r["text_vs_corpus_normalizer"]
    a("## Training text: `trnorm.normalize(transcript)` vs the corpus' `normalized_transcript`\n")
    a(f"- identical: {t['identical']}; same words, only punctuation/quotes/case differ: "
      f"{t['same_words_punctuation_or_case_only']}; **different words: {t['different_words']}**")
    a(f"- different words by category: {t['different_words_by_category']}")
    ca = t["closer_to_asr"]
    a(f"- which one matches the audio better (CER against the independent ASR transcript, clips with different words): "
      f"**trnorm {ca['trnorm']}**, corpus normalizer {ca['corpus_normalizer']}, tie {ca['tie']}\n")
    a("Most frequent differing word spans (corpus normalizer → trnorm), after `normalize_for_eval`:\n")
    a("| corpus | trnorm | n |\n|---|---|---:|")
    for x in t["top_spans_corpus_vs_trnorm"][:25]:
        a(f"| {x['corpus'] or '∅'} | {x['trnorm'] or '∅'} | {x['count']} |")
    a("\nExamples:\n")
    for x in t["examples"]:
        a(f"- `{x['clip_id'][-16:]}` ({x['category']}, CER vs ASR trnorm {x['cer_trnorm']} / corpus {x['cer_corpus']})  \n"
          f"  raw: {x['raw']}  \n  corpus: {x['corpus']}  \n  trnorm: {x['trnorm']}  \n  ASR: {x['asr']}")
    g = r["asr_agreement"]
    a("\n## Text/audio agreement (trnorm text vs the corpus ASR transcript)\n")
    for k in ("cer_trnorm_vs_asr", "cer_corpus_normalizer_vs_asr", "cer_trnorm_vs_asr_clean_text",
              "cer_trnorm_vs_asr_digits_or_acronyms"):
        v = g[k]
        if v:
            a(f"- {k}: n {v['n']}, mean {v['mean']}, median {v['median']}, p90 {v['p90']}, p95 {v['p95']}, "
              f"p99 {v['p99']}, max {v['max']}")
    a(f"- clips above CER thresholds: {g['clips_above']}")
    a(f"- `TL`: {g['TL_clips']['clips']} clips / {g['TL_clips']['TL_tokens']} tokens; the ASR wrote \"Türk lirası\" "
      f"{g['TL_clips']['asr_writes_turk_lirasi']} times (trnorm reads TL as \"lira\")\n")
    a("Most frequent trnorm-vs-ASR differences (clips with CER > 0.03):\n")
    a("| trnorm | ASR | n |\n|---|---|---:|")
    for x in g["top_spans_trnorm_vs_asr"][:20]:
        a(f"| {x['trnorm'] or '∅'} | {x['asr'] or '∅'} | {x['count']} |")
    a("\nWorst clips:\n")
    for x in g["worst"][:8]:
        a(f"- {x['cer']} `{x['clip_id'][-16:]}` ({x['category']})  \n  trnorm: {x['trnorm']}  \n  ASR: {x['asr']}")
    a(f"\nNon-letter characters in the kept training text: {r['text_characters_non_letter']}\n")
    p = r["phonemes"]
    if p:
        a("## espeak-ng phonemes (`tr`)\n")
        a(f"- phoneme ids per clip (incl. BOS/EOS/padding): {p['phoneme_ids_per_clip']}")
        a(f"- phoneme ids per second: median {p['phoneme_ids_per_second']['median']}, max {p['phoneme_ids_per_second']['max']}")
        a(f"- inventory: {p['inventory_size']} symbols; missing from Piper's id map: {p['missing_from_id_map'] or 'none'}")
        a(f"- {' '.join(f'{k}:{v}' for k, v in p['inventory'].items())}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
