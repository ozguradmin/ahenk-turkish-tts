#!/usr/bin/env python3
"""Synthesize TR-Zor and the held-out Antalia sentences with VoxCPM2 (+ a LoRA), then score them.

    python tts/voxcpm/synth_eval.py --model $TTS_DATA/models/VoxCPM2 --lora $EXP/step_0000500 \
        --val $TTS_DATA/antalia16k/voxcpm_val.jsonl --out $EXP/eval/step_0000500
    python tts/voxcpm/synth_eval.py --model ... --ref some_antalia_clip.wav --out .../eval/base   # base + cloning

Text front-end: trnorm.normalize, then VoxCPM's own normalizer is switched off (normalize=False),
the same spoken forms the LoRA was trained on. Every utterance uses seed 0.

Judge: faster-whisper large-v3-turbo (float16 on GPU, int8 on CPU), language tr, beam 5, the judge of
benchmarks/tts. WER is corpus-level after trnorm.normalize_for_eval on both sides; a TR-Zor row is
scored against the closest of its `spoken` form and `alts`, as benchmarks/tts/score.py does.

Naturalness: UTMOS (score_utmos.py, UTMOS22 strong via torch.hub tarepan/SpeechMOS:v1.2.0) on every WAV
once VoxCPM is freed, also with --no-judge; --no-utmos skips it. rows.jsonl and summary.json are written before
it (summary["utmos_error"] says it is pending) and replaced once it is done, so a kill or a hang there (first hub
download on a cold $TORCH_HOME) costs only the UTMOS score: score_utmos.py DIR adds it later from the WAVs.

Question intonation: --sets ...,soru adds benchmarks/tts/sets/tr_soru.jsonl (35 rows: questions and their matched
declaratives; refs = its spoken form, so WER covers it too). The judge keeps word timestamps for those rows
(row["words"]), and score_intonation.py measures where each one's pitch peak falls (summary["intonation"]) right after
the first write; --no-intonation skips it, a failure only sets summary["intonation_error"] (score_intonation.py DIR).

--pause-clean (default off): pause_clean.clean (breath noise in pauses turned down, speech frames untouched,
deterministic, CPU) on every clip right after synthesis, before it is written, judged and scored (RTF counts synthesis
only); rows "pause_clean" {pauses, breaths, gated_s}, summary "pause_clean" {settings, clips, pauses, breaths,
gated_s} (None without it).

--limit N for cheap sweeps: --sample stratified (default) takes TR-Zor rows round-robin over its 13
categories (the file is ordered by category: its first 30 rows are 15 cardinals, 12 ordinals, 3 dates) and
the first N held-out rows; --sample head takes the first N rows of each set (the old behaviour).

Writes <out>/wav/<set>/<id>.wav (48 kHz), <out>/rows.jsonl and <out>/summary.json (each replaced atomically),
and prints the summary (WER and UTMOS per set and per TR-Zor category, RTF).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from score_utmos import errors_note, utmos_summary, write_atomic  # noqa: E402  (script dir on sys.path; no torch)
from trnorm import normalize, normalize_for_eval  # noqa: E402


def stratified(rows: list[dict], n: int) -> list[dict]:
    """n rows round-robin over categories (in order of first appearance, each category in file order): the
    first row of every category, then every second row, ...; returned in file order. Deterministic, and
    covers every category once n >= the number of categories."""
    order, seen, keyed = {}, defaultdict(int), []
    for i, r in enumerate(rows):
        c = order.setdefault(r["category"], len(order))
        keyed.append((seen[c], c, i))
        seen[c] += 1
    return [rows[i] for i in sorted(i for _, _, i in sorted(keyed)[:n])]


def load_sets(val_manifest: Path | None, sets: list[str], limit: int, sample: str = "stratified") -> list[dict]:
    rows = []
    if "trzor" in sets:
        for line in open(REPO / "benchmarks" / "tts" / "sets" / "tr_zor.jsonl", encoding="utf-8"):
            r = json.loads(line)
            rows.append({"set": "trzor", "id": r["id"], "category": r["category"], "text": r["text"],
                         "model_text": normalize(r["text"]), "refs": [r["spoken"]] + r.get("alts", [])})
    if "heldout" in sets and val_manifest:
        for line in open(val_manifest, encoding="utf-8"):
            r = json.loads(line)
            rid = "heldout_" + Path(r["audio"]).stem.rsplit("-", 1)[-1]
            rows.append({"set": "heldout", "id": rid, "category": "heldout", "text": r["text"],
                         "model_text": r["text"], "refs": [r["text"]]})
    if "soru" in sets:  # question intonation (score_intonation.py): pattern, peak = word index in the spoken form
        for line in open(REPO / "benchmarks" / "tts" / "sets" / "tr_soru.jsonl", encoding="utf-8"):
            r = json.loads(line)
            rows.append({"set": "soru", "id": r["id"], "category": r["category"], "text": r["text"],
                         "model_text": normalize(r["text"]), "refs": [r["spoken"]], "pair_id": r["pair_id"],
                         "pattern": r["pattern"], "peak": r["peak"], "peak_word": r["peak_word"]})
    if limit:
        by = defaultdict(list)
        for r in rows:
            by[r["set"]].append(r)
        # heldout rows share one category, so stratified = its first N there
        rows = [r for s in by.values() for r in (stratified(s, limit) if sample == "stratified" else s[:limit])]
    return rows


def wer_parts(ref: str, hyp: str):
    import jiwer

    ref, hyp = normalize_for_eval(ref), normalize_for_eval(hyp)
    if not ref:
        return 0, 0
    o = jiwer.process_words(ref, hyp or "<empty>")
    return o.substitutions + o.deletions + o.insertions, o.substitutions + o.deletions + o.hits


def write_results(out: Path, rows: list[dict], summary: dict) -> None:
    write_atomic(out / "rows.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_atomic(out / "summary.json", json.dumps(summary, indent=1, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="VoxCPM2 snapshot directory (or hub id)")
    ap.add_argument("--lora", default="", help="LoRA checkpoint directory (step_XXXXXXX); empty = base model")
    ap.add_argument("--ref", default="", help="reference WAV for voice cloning (reference_wav_path)")
    ap.add_argument("--val", type=Path, help="voxcpm_val.jsonl (held-out Antalia sentences)")
    ap.add_argument("--sets", default="trzor,heldout", help="comma-separated: trzor, heldout, soru")
    ap.add_argument("--limit", type=int, default=0, help="N rows of each set (0 = all), picked by --sample")
    ap.add_argument("--sample", choices=["stratified", "head"], default="stratified",
                    help="with --limit: TR-Zor round-robin over categories (stratified) or the first N rows (head)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--utmos", action=argparse.BooleanOptionalAction, default=True, help="UTMOS naturalness score")
    ap.add_argument("--intonation", action=argparse.BooleanOptionalAction, default=True,
                    help="question intonation of the soru set (score_intonation.py)")
    ap.add_argument("--pause-clean", action="store_true",
                    help="pause_clean.clean on every clip after synthesis, before it is written and scored")
    args = ap.parse_args(argv)

    import numpy as np
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    rows = load_sets(args.val, args.sets.split(","), args.limit, args.sample)
    pc = None
    if args.pause_clean:
        import pause_clean as pc  # numpy + scipy when used
    cuda = torch.cuda.is_available()
    t0 = time.time()
    tts = VoxCPM.from_pretrained(args.model, load_denoiser=False, optimize=False,
                                 lora_weights_path=args.lora or None)
    sr = tts.tts_model.sample_rate
    print(f"[eval] model loaded in {time.time() - t0:.1f} s; {len(rows)} rows; lora={args.lora or 'none'} "
          f"ref={args.ref or 'none'}", flush=True)

    (args.out / "wav").mkdir(parents=True, exist_ok=True)
    synth_s = audio_s = 0.0
    for i, r in enumerate(rows):
        t = time.time()
        wav = tts.generate(text=r["model_text"], reference_wav_path=args.ref or None, cfg_value=args.cfg,
                           inference_timesteps=args.timesteps, normalize=False, seed=args.seed)
        if cuda:
            torch.cuda.synchronize()
        dt = time.time() - t
        wav = np.asarray(wav, dtype=np.float32)
        if pc is not None:
            wav, rep = pc.clean(wav, sr)
            r["pause_clean"] = pc.row_record(rep)
        path = args.out / "wav" / r["set"] / f"{r['id']}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, wav, sr)
        r.update(wav=str(path), duration=round(len(wav) / sr, 3), synth_seconds=round(dt, 3))
        synth_s += dt
        audio_s += len(wav) / sr
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            print(f"[eval] synthesized {i + 1}/{len(rows)}, RTF {synth_s / max(audio_s, 1e-9):.3f}", flush=True)
    del tts
    if cuda:
        torch.cuda.empty_cache()

    if not args.no_judge:
        from faster_whisper import WhisperModel

        judge = WhisperModel("large-v3-turbo", device="cuda" if cuda else "cpu",
                             compute_type="float16" if cuda else "int8")
        for r in rows:
            words = r["set"] == "soru"  # word timestamps for score_intonation
            segs, _ = judge.transcribe(r["wav"], language="tr", beam_size=5, word_timestamps=words)
            segs = list(segs)
            r["hyp"] = " ".join(s.text.strip() for s in segs).strip()
            if words:
                r["words"] = [[w.word.strip(), round(w.start, 3), round(w.end, 3)]
                              for s in segs for w in (getattr(s, "words", None) or [])]
            best = min(r["refs"], key=lambda ref: wer_parts(ref, r["hyp"])[0] / max(wer_parts(ref, r["hyp"])[1], 1))
            r["errors"], r["ref_words"] = wer_parts(best, r["hyp"])
        del judge

    summary = {"model": args.model, "lora": args.lora, "ref": args.ref, "cfg": args.cfg, "timesteps": args.timesteps,
               "seed": args.seed, "limit": args.limit, "sample": args.sample, "device": "cuda" if cuda else "cpu",
               "rtf": round(synth_s / max(audio_s, 1e-9), 4), "pause_clean": pc.rows_summary(rows) if pc else None,
               "wer": {}}
    if not args.no_judge:
        groups = defaultdict(lambda: [0, 0, 0, 0])
        for r in rows:
            for key in (r["set"], f"{r['set']}/{r['category']}"):
                g = groups[key]
                g[0] += r["errors"]
                g[1] += r["ref_words"]
                g[2] += 1
                g[3] += r["errors"] == 0
        summary["wer"] = {k: {"wer": round(100 * e / max(n, 1), 2), "rows": c, "rows_correct": ok}
                          for k, (e, n, c, ok) in sorted(groups.items())}
    intonation = args.intonation and any(r["set"] == "soru" for r in rows)
    if intonation:  # replaced below, like the UTMOS note
        summary["intonation_error"] = "not scored: the run ended before it (score_intonation.py DIR adds it)"
    if args.utmos:  # replaced below; stays if this process is killed or hangs (cold hub download) before
        summary["utmos_error"] = "not scored: the run ended before UTMOS finished (score_utmos.py DIR adds it)"
    write_results(args.out, rows, summary)  # synthesis + WER are on disk from here on

    if intonation:  # CPU, seconds; before UTMOS so a UTMOS hang cannot cost it
        t, inton_error = time.time(), ""
        try:
            from score_intonation import errors_note as inton_note, intonation_summary, score_rows as score_inton

            errors = score_inton(rows, lambda r: r["wav"], ("soru",), "eval")  # one bad clip costs one row
            summary["intonation"] = intonation_summary(rows)
            inton_error = inton_note(errors) if errors else ""
        except Exception as e:  # e.g. librosa missing: keep the rest, score_intonation.py DIR adds it later
            inton_error = f"{type(e).__name__}: {e}"
        del summary["intonation_error"]  # the pending note
        if inton_error:
            summary["intonation_error"] = inton_error
            print(f"[eval] intonation incomplete: {inton_error}", flush=True)
        print(f"[eval] intonation {sum('intonation' in r for r in rows)}/{sum(r['set'] == 'soru' for r in rows)} "
              f"soru rows in {time.time() - t:.1f} s", flush=True)
        write_results(args.out, rows, summary)

    if args.utmos:
        from score_utmos import load_predictor, score_rows

        t, dev, utmos_error = time.time(), "cuda" if cuda else "cpu", ""
        try:
            predictor = load_predictor(dev)
        except Exception as e:  # hub fetch failed or stalled (socket timeout): keep the WER, rescore later
            predictor, utmos_error = None, f"{type(e).__name__}: {e}"
        if predictor is not None:
            errors = score_rows(predictor, rows, lambda r: r["wav"], dev, "eval")  # one bad clip costs one row
            del predictor
            utmos_error = errors_note(errors) if errors else ""
        del summary["utmos_error"]  # the pending note
        summary["utmos"] = utmos_summary(rows)
        if utmos_error:
            summary["utmos_error"] = utmos_error
            print(f"[eval] UTMOS incomplete, rows without a score are left out of its means: {utmos_error}", flush=True)
        print(f"[eval] UTMOS {sum('utmos' in r for r in rows)}/{len(rows)} rows in {time.time() - t:.1f} s", flush=True)
        write_results(args.out, rows, summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
