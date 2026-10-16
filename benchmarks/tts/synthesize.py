#!/usr/bin/env python3
"""Synthesize a test set with one TTS system through one text front-end.

    python benchmarks/tts/synthesize.py --set dev_numbers10 --system ema-lightning --frontend trnorm
    python benchmarks/tts/synthesize.py --set tr_zor --system qwen3-tts-tiny-tr --frontend raw --sample 60

Writes   data/tts/out/<set>/<system>/<frontend>/<id>.wav          (16-bit PCM, system's native rate)
and      benchmarks/tts/results/<set>/synth/<system>__<frontend>.jsonl, one record per row:
         text (as written), frontend_text (what the front-end produced = input to the system),
         model_text (exact text the model consumes after the system's own input preparation),
         duration, synth_seconds (wall), synth_cpu_seconds, rtf, audio_sha1, ...

Resumable: a row is skipped when its record exists, its wav is intact and its frontend_text and
system config are unchanged (so rows are redone automatically when trnorm changes).
Rows whose model_text equals that of another front-end's finished record for the same system are
not re-synthesized: the (deterministic, same-seed) audio is copied and marked `reused_from`.
RTF is computed on actually synthesized rows only. Timing excludes model loading and a warm-up.
"""

from __future__ import annotations

import common  # noqa: F401  (thread caps, HF_HOME)

import argparse
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

from common import (FRONTENDS, REPO, SEED, THREADS, append_jsonl, cpu_model, file_sha1, git_head,
                    latest_by_id, load_set, loadavg, pkg_versions, read_jsonl, set_name, synth_path,
                    wav_path, write_text_atomic, write_wav)
from frontends import frontend_version, get_frontend
from systems import SYSTEMS, NotApplicable, load_system, native_status

WARMUP = "Merhaba, bu cümle yalnızca modeli ısıtmak için okunuyor."


def select_rows(rows, args):
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    if args.sample:
        # seeded, stratified by category, original order kept
        by_cat = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r)
        rng = random.Random(args.sample_seed)
        quota = {c: max(1, round(args.sample * len(v) / len(rows))) for c, v in by_cat.items()}
        keep = set()
        for c, v in by_cat.items():
            keep.update(r["id"] for r in rng.sample(v, min(quota[c], len(v))))
        rows = [r for r in rows if r["id"] in keep]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="name under benchmarks/tts/sets/ or a .jsonl path")
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--frontend", required=True, choices=FRONTENDS)
    ap.add_argument("--limit", type=int, help="first N rows only")
    ap.add_argument("--sample", type=int, help="seeded, category-stratified subset of ~N rows")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--ids", help="comma-separated row ids")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--threads", type=int, help=f"CPU threads (default: the system's own default, capped at {THREADS})")
    ap.add_argument("--no-reuse", action="store_true", help="always synthesize, never copy identical inputs")
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo every selected row")
    args = ap.parse_args(argv)

    sname = set_name(args.set)
    rows = select_rows(load_set(args.set), args)
    tag = f"[{sname} {args.system} {args.frontend}]"

    mode, note = native_status(args.system)
    if args.frontend == "native" and mode != "separable":
        what = "raw == native" if mode == "always_on" else "no bundled normalizer"
        print(f"{tag} not applicable: {what} ({note}); nothing to do")
        return 0

    out_path = synth_path(sname, args.system, args.frontend)
    done = latest_by_id(read_jsonl(out_path))
    fe = get_frontend(args.frontend)

    # front-end texts first (cheap), to find the rows that need the model at all
    todo = []
    for r in rows:
        fe_err = None
        try:
            fe_text = fe(r["text"])
        except Exception as e:  # a crashing front-end is a front-end failure: fall back to the raw text
            fe_text, fe_err = r["text"], f"{type(e).__name__}: {e}"
            print(f"{tag} {r['id']}: front-end error {fe_err}", file=sys.stderr, flush=True)
        rec = done.get(r["id"])
        wav = wav_path(sname, args.system, args.frontend, r["id"])
        fresh = (rec is not None and not args.force and rec.get("frontend_text") == fe_text
                 and rec.get("seed") == args.seed
                 and (rec.get("error") is None or not args.retry_errors)
                 and (rec.get("error") is not None or (wav.exists() and file_sha1(wav) == rec.get("audio_sha1"))))
        if not fresh:
            todo.append((r, fe_text, fe_err))
    print(f"{tag} {len(rows)} rows selected, {len(rows) - len(todo)} already done, {len(todo)} to do",
          flush=True)
    if not todo:
        return 0

    t0 = time.perf_counter()
    try:
        tts = load_system(args.system, native=args.frontend == "native", seed=args.seed, threads=args.threads)
    except NotApplicable as e:
        print(f"{tag} not applicable: {e}")
        return 0
    load_seconds = time.perf_counter() - t0
    cfg, cfg_hash = tts.config(), tts.config_hash()

    # earlier records of the same system (other front-ends) whose audio can be reused
    pool = {}
    if not args.no_reuse:
        for f in FRONTENDS:
            if f == args.frontend:
                continue
            for rec in read_jsonl(synth_path(sname, args.system, f)):
                if rec.get("error") is None and rec.get("config_hash_base") == base_hash(cfg):
                    pool[(rec["id"], rec["model_text"])] = (f, rec)

    meta = {
        "set": sname, "system": args.system, "frontend": args.frontend, "config": cfg,
        "config_hash": cfg_hash, "frontend_version": frontend_version(args.frontend),
        "native_mode": mode, "native_note": note, "threads": tts.threads, "cpu": cpu_model(),
        "git_head": git_head(), "versions": pkg_versions(), "load_seconds": round(load_seconds, 2),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out_path.with_suffix(".meta.json"), json.dumps(meta, ensure_ascii=False, indent=1) + "\n")

    fe_ver = meta["frontend_version"]
    tts.reset()
    t0 = time.perf_counter()
    tts.synthesize(WARMUP)
    print(f"{tag} loaded in {load_seconds:.1f}s, warm-up {time.perf_counter() - t0:.1f}s", flush=True)

    n_synth = n_reuse = n_err = 0
    tot_audio = tot_synth = 0.0

    def progress(i):
        if i % 10 == 0 or i == len(todo):
            rtf = tot_synth / tot_audio if tot_audio else float("nan")
            print(f"{tag} {i}/{len(todo)} synthesized={n_synth} reused={n_reuse} errors={n_err} "
                  f"RTF={rtf:.3f}", flush=True)

    for i, (r, fe_text, fe_err) in enumerate(todo, 1):
        wav = wav_path(sname, args.system, args.frontend, r["id"])
        rec = {"id": r["id"], "category": r["category"], "system": args.system, "frontend": args.frontend,
               "text": r["text"], "frontend_text": fe_text, "frontend_error": fe_err, "seed": args.seed,
               "frontend_version": fe_ver, "config_hash": cfg_hash, "config_hash_base": base_hash(cfg)}
        try:
            model_text = tts.model_text(fe_text)
            rec["model_text"] = model_text
            src = pool.get((r["id"], model_text))
            if src is not None:
                f, old = src
                old_wav = REPO / old["audio_path"]
                if old_wav.exists() and file_sha1(old_wav) == old["audio_sha1"]:
                    wav.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(old_wav, wav)
                    for k in ("extra", "sample_rate", "duration", "synth_seconds", "synth_cpu_seconds",
                              "rtf", "loadavg", "audio_sha1"):
                        rec[k] = old.get(k)
                    rec.update(audio_path=str(wav.relative_to(REPO)), reused_from=f, error=None)
                    append_jsonl(out_path, rec)
                    n_reuse += 1
                    progress(i)
                    continue
            rec["extra"] = tts.extra(fe_text)
            tts.reset()
            la = loadavg()
            c0, w0 = time.process_time(), time.perf_counter()
            audio, sr = tts.synthesize(fe_text)
            wall, cpu = time.perf_counter() - w0, time.process_time() - c0
            write_wav(wav, audio, sr)
            dur = len(audio) / sr
            rec.update(audio_path=str(wav.relative_to(REPO)), audio_sha1=file_sha1(wav), sample_rate=sr,
                       duration=round(dur, 3), synth_seconds=round(wall, 3), synth_cpu_seconds=round(cpu, 3),
                       rtf=round(wall / dur, 4) if dur else None, loadavg=la, reused_from=None, error=None)
            n_synth += 1
            tot_audio += dur
            tot_synth += wall
        except Exception as e:  # keep going; the row scores as an error (empty hypothesis)
            rec.update(error=f"{type(e).__name__}: {e}", duration=0.0, audio_sha1=None, audio_path=None)
            n_err += 1
            print(f"{tag} {r['id']}: ERROR {rec['error']}", file=sys.stderr, flush=True)
        append_jsonl(out_path, rec)
        progress(i)
    meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_text_atomic(out_path.with_suffix(".meta.json"), json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    return 0


def base_hash(cfg: dict) -> str:
    """Config hash without the `native` flag: audio is shareable across front-ends iff this matches."""
    from hashlib import sha1

    c = {k: v for k, v in cfg.items() if k != "native"}
    return sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


if __name__ == "__main__":
    sys.exit(main())
