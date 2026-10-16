#!/usr/bin/env python3
"""Score raw ASR hypotheses: corpus WER/CER after trnorm.normalize_for_eval, plus RTF.

    python benchmarks/asr/score.py                         # every results/<set>/<system>.jsonl
    python benchmarks/asr/score.py --sets fleurs_tr_test --systems whisper-small duxx-zipformer
    python benchmarks/asr/score.py --common                # same utterances for every system

Both reference and hypothesis go through trnorm.normalize_for_eval (Turkish lower-casing,
numbers -> words, punctuation stripped) at scoring time, so re-run this script whenever
trnorm changes; decoding never needs to be repeated.

WER/CER are corpus-level (total edits / total reference words or characters, jiwer).
RTF = sum(decode_seconds) / sum(audio duration), wall-clock on CPU with 2 threads.
Writes results/summary.json and results/LEADERBOARD.md (and prints the table).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import jiwer

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
EVAL_DIR = HERE.parents[1] / "data" / "asr_eval"

# Per-system metadata for the leaderboard (params = total model parameters).
MODEL_META: dict[str, dict] = {
    "whisper-tiny": dict(params="39M", license="MIT", streaming=False,
                         model="openai/whisper-tiny (Systran/faster-whisper-tiny, int8)"),
    "whisper-base": dict(params="74M", license="MIT", streaming=False,
                         model="openai/whisper-base (Systran/faster-whisper-base, int8)"),
    "whisper-small": dict(params="244M", license="MIT", streaming=False,
                          model="openai/whisper-small (Systran/faster-whisper-small, int8)"),
    "whisper-medium": dict(params="769M", license="MIT", streaming=False,
                           model="openai/whisper-medium (Systran/faster-whisper-medium, int8)"),
    "whisper-large-v3-turbo": dict(params="809M", license="MIT", streaming=False,
                                   model="openai/whisper-large-v3-turbo "
                                         "(dropbox-dash/faster-whisper-large-v3-turbo, int8)"),
    "buzzasr-turkish": dict(params="1.55B", license="MIT", streaming=False,
                            model="BuzzASR/turkish (whisper-large-v3 fine-tune; local CT2 int8)"),
    "duxx-zipformer": dict(params="23.6M", license="CC-BY-NC-4.0", streaming=True,
                           model="duxx/turkish-stt-zipformer (streaming zipformer RNN-T, "
                                 "chunk 32, int8, sherpa-onnx)"),
    "mihuai-turkish-stt": dict(params="66M", license="mihu Community License*", streaming=True,
                               model="mihuai/turkish-stt (streaming transducer, int8 enc/joiner, "
                                     "sherpa-onnx)"),
    "vosk-tr": dict(params="~3M AM + LM graph", license="Apache-2.0", streaming=True,
                    model="vosk-model-small-tr-0.3 (Kaldi TDNN, alphacephei)"),
}
SET_META: dict[str, dict] = {
    "fleurs_tr_test": dict(license="CC-BY-4.0", desc="google/fleurs tr_tr test, read Wikipedia "
                                                     "sentences, raw (cased/punctuated) refs"),
    "mediaspeech_tr": dict(license="CC-BY-4.0", desc="ymoslem/MediaSpeech tr, seeded 500-utt "
                                                     "subset of YouTube media speech"),
}


def trnorm_fingerprint() -> dict:
    import trnorm

    src = Path(trnorm.__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(src.glob("*.py")):
        h.update(p.name.encode() + b"\0" + p.read_bytes())
    return {"version": getattr(trnorm, "__version__", "?"), "src_sha256": h.hexdigest()[:12]}


def manifest_size(set_name: str) -> int | None:
    p = EVAL_DIR / set_name / "manifest.jsonl"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_results(path: Path) -> list[dict]:
    rows, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # truncated last line of a running/killed job
                continue
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    return rows


def score_rows(rows: list[dict], normalize) -> dict:
    refs = [normalize(r["ref"]) for r in rows]
    hyps = [normalize(r["hyp"]) for r in rows]
    w = jiwer.process_words(refs, hyps)
    c = jiwer.process_characters(refs, hyps)
    # A decoder that was suspended (SIGSTOP, laptop sleep) mid-utterance records the pause as
    # decode time. Leave such outliers (> 50x real time and > 10 min) out of the RTF.
    timed = [r for r in rows if not (r["decode_seconds"] > 600 and r["decode_seconds"] > 50 * r["duration"])]
    dur = sum(r["duration"] for r in timed)
    dec = sum(r["decode_seconds"] for r in timed)
    cpu_rows = [r for r in rows if "cpu_seconds" in r]  # older rows lack it
    cpu_dur = sum(r["duration"] for r in cpu_rows)
    return {
        "n": len(rows),
        "wer": w.wer, "cer": c.cer,
        "word_sub": w.substitutions, "word_del": w.deletions, "word_ins": w.insertions,
        "ref_words": w.substitutions + w.deletions + w.hits,
        "char_edits": c.substitutions + c.deletions + c.insertions,
        "ref_chars": c.substitutions + c.deletions + c.hits,
        "empty_hyps": sum(1 for h in hyps if not h),
        "audio_seconds": round(dur, 2), "decode_seconds": round(dec, 2),
        "rtf": dec / dur if dur else None,
        "rtf_outliers_skipped": len(rows) - len(timed),
        # process CPU seconds (all decoder threads) per audio second; robust to other load
        "cpu_rtf": sum(r["cpu_seconds"] for r in cpu_rows) / cpu_dur if cpu_dur else None,
        "cpu_rtf_n": len(cpu_rows),
    }


def leaderboard_md(entries: list[dict], fp: dict, cpu: str) -> str:
    lines = [
        "# Turkish ASR leaderboard",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `benchmarks/asr/score.py`. "
        f"Normalisation: `trnorm.normalize_for_eval` (trnorm {fp['version']}, "
        f"src sha256 {fp['src_sha256']}) applied to both reference and hypothesis. "
        "Corpus-level WER/CER (jiwer). RTF = decode wall-clock / audio duration on "
        f"{cpu}, 2 CPU threads per decoder (vosk is single-threaded), 1-2 decoders running "
        "at once on a 4-core machine shared with other jobs, so treat RTF as indicative "
        "(summary.json also has `cpu_rtf` = process CPU seconds / audio second).",
        "",
        "| system | params | license | set | WER % | CER % | RTF (CPU, 2 threads) | n |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    partial = False
    for e in entries:
        n = str(e["n"])
        if not e["complete"]:
            partial = True
            n = f"{e['n']}/{e['n_total'] or '?'} (partial)"
        rtf = f"{e['rtf']:.3f}" if e["rtf"] is not None else "-"
        lines.append(f"| {e['system']} | {e['params']} | {e['license']} | {e['set']} | "
                     f"{100 * e['wer']:.2f} | {100 * e['cer']:.2f} | {rtf} | {n} |")
    lines.append("")
    if partial:
        lines += ["Partial rows are scored on the utterances decoded so far (a prefix of the "
                  "manifest) and are not directly comparable with complete rows.", ""]
    lines += ["## Systems", "", "| system | model | params | license | streaming |",
              "|---|---|---|---|---|"]
    for name in sorted({e["system"] for e in entries}):
        m = MODEL_META.get(name, {})
        lines.append(f"| {name} | {m.get('model', '?')} | {m.get('params', '?')} | "
                     f"{m.get('license', '?')} | {'yes' if m.get('streaming') else 'no'} |")
    lines += ["", "\\* mihu Community License: free below USD 500k annual revenue, requires "
              "\"Powered by mihu\" attribution, forbids using the model or its outputs to train "
              "competing models. Evaluated here for research/benchmarking only; its outputs must "
              "never be used as training data.", "",
              "## Sets", "", "| set | description | license | n |", "|---|---|---|---:|"]
    for name in sorted({e["set"] for e in entries}):
        s = SET_META.get(name, {})
        lines.append(f"| {name} | {s.get('desc', '?')} | {s.get('license', '?')} | "
                     f"{manifest_size(name) or '?'} |")
    return "\n".join(lines) + "\n"


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", help="limit to these sets (prints only, writes no files)")
    ap.add_argument("--systems", nargs="*", help="limit to these systems (prints only, writes no files)")
    ap.add_argument("--common", action="store_true",
                    help="score every system of a set on the ids decoded by ALL selected systems "
                         "(makes partial runs comparable; prints only, writes no files)")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    from trnorm import normalize_for_eval

    fp = trnorm_fingerprint()
    loaded: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(args.results_dir.glob("*/*.jsonl")):
        set_name, system = path.parent.name, path.stem
        if (args.sets and set_name not in args.sets) or (args.systems and system not in args.systems):
            continue
        rows = read_results(path)
        if rows:
            loaded[(set_name, system)] = rows
    if args.common:
        for set_name in {s for s, _ in loaded}:
            keys = [k for k in loaded if k[0] == set_name]
            common = set.intersection(*({r["id"] for r in loaded[k]} for k in keys))
            for k in keys:
                loaded[k] = [r for r in loaded[k] if r["id"] in common]

    entries = []
    for (set_name, system), rows in loaded.items():
        if not rows:
            continue
        s = score_rows(rows, normalize_for_eval)
        n_total = manifest_size(set_name)
        meta = MODEL_META.get(system, {})
        entries.append({
            "set": set_name, "system": system, "n_total": n_total,
            "complete": n_total is not None and s["n"] >= n_total,
            "params": meta.get("params", "?"), "license": meta.get("license", "?"),
            "streaming": meta.get("streaming"), **s,
        })
    entries.sort(key=lambda e: (e["set"], e["wer"]))

    cpu = cpu_name()
    md = leaderboard_md(entries, fp, cpu)
    if args.common:
        md = md.replace("# Turkish ASR leaderboard", "# Turkish ASR leaderboard (common-subset "
                        "scoring: each set restricted to ids decoded by every listed system)")
    if not (args.sets or args.systems or args.common):
        summary = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "trnorm": fp, "cpu": cpu,
                   "metric_notes": "corpus WER/CER after trnorm.normalize_for_eval on ref and "
                                   "hyp; rtf = sum(decode_seconds)/sum(duration)",
                   "results": entries}
        (args.results_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        (args.results_dir / "LEADERBOARD.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
