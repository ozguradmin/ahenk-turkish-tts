#!/usr/bin/env python3
"""Score ASR round-trip hypotheses: WER/CER per system x front-end, per category, plus speed.

    python benchmarks/tts/score.py --set dev_numbers10 freya_sample40 tr_zor
    python benchmarks/tts/score.py            # every set with results

trnorm is imported HERE, at scoring time: reference (`spoken`, and every entry of the optional
`alts` list) and the raw ASR hypothesis both go through trnorm.normalize_for_eval, so re-run this
whenever trnorm changes; nothing needs to be re-synthesized or re-transcribed.

Metrics (corpus level = total edits / total reference length):
  WER, CER       best reference per row: the one among spoken + alts with the fewest word edits
                 (ties: fewer character edits, then `spoken`); CER counts spaces, like jiwer.cer
  WER_strict     against `spoken` only
  row_ok         share of rows with zero word errors
  RTF            synthesis wall-clock / audio duration, rows actually synthesized (not reused),
                 2 CPU threads, shared machine; CPU-RTF = process CPU time / audio duration

Writes results/<set>/summary.json, results/<set>/RESULTS.md and results/<set>/rows.jsonl.
"""

from __future__ import annotations

import common  # noqa: F401

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

from rapidfuzz.distance import Levenshtein

from common import (FRONTENDS, RESULTS_DIR, cpu_model, latest_by_id, load_set, pkg_versions, read_jsonl,
                    set_name, trnorm_fingerprint, write_text_atomic)
from frontends import FRONTEND_DESCRIPTIONS
from systems import SYSTEMS

SYSTEM_ORDER = list(SYSTEMS)


def edits(ref: str, hyp: str) -> tuple[int, int, int, int]:
    rw, hw = ref.split(), hyp.split()
    return Levenshtein.distance(rw, hw), len(rw), Levenshtein.distance(ref, hyp), len(ref)


def best_ref(refs: list[str], hyp: str):
    """(index, word_edits, ref_words, char_edits, ref_chars) of the closest reference."""
    best = None
    for i, r in enumerate(refs):
        we, wn, ce, cn = edits(r, hyp)
        key = (we, ce, i)
        if best is None or key < best[0]:
            best = (key, (i, we, wn, ce, cn))
    return best[1]


class Acc:
    def __init__(self):
        self.we = self.wn = self.ce = self.cn = self.swe = self.swn = 0
        self.n = self.ok = 0

    def add(self, we, wn, ce, cn, swe, swn):
        self.we += we
        self.wn += wn
        self.ce += ce
        self.cn += cn
        self.swe += swe
        self.swn += swn
        self.n += 1
        self.ok += we == 0

    def out(self):
        pct = lambda a, b: round(100 * a / b, 2) if b else None  # noqa: E731
        return {"n": self.n, "wer": pct(self.we, self.wn), "cer": pct(self.ce, self.cn),
                "wer_strict": pct(self.swe, self.swn), "row_ok": pct(self.ok, self.n),
                "word_edits": self.we, "ref_words": self.wn}


def score_set(sname: str) -> dict:
    from trnorm import normalize_for_eval

    rows = {r["id"]: r for r in load_set(sname)}
    has_alts = any(r.get("alts") for r in rows.values())
    refs = {i: [normalize_for_eval(r["spoken"])] + [normalize_for_eval(a) for a in r.get("alts") or []]
            for i, r in rows.items()}
    categories = sorted({r["category"] for r in rows.values()}, key=lambda c: list(
        dict.fromkeys(r["category"] for r in rows.values())).index(c))

    synth_dir = RESULTS_DIR / sname / "synth"
    pairs = []
    for p in sorted(synth_dir.glob("*__*.jsonl")):
        system, frontend = p.stem.split("__", 1)
        pairs.append((system, frontend))
    order_s = {s: i for i, s in enumerate(SYSTEM_ORDER)}
    order_f = {f: i for i, f in enumerate(FRONTENDS)}
    pairs.sort(key=lambda sf: (order_s.get(sf[0], 99), order_f.get(sf[1], 99)))
    if not pairs:
        return None

    results: dict = defaultdict(dict)
    speed: dict = defaultdict(lambda: {"synth": 0.0, "cpu": 0.0, "dur_timed": 0.0, "n_timed": 0,
                                       "dur_all": 0.0, "n_audio": 0})
    detail_rows = []
    for system, frontend in pairs:
        recs = latest_by_id(read_jsonl(synth_dir / f"{system}__{frontend}.jsonl"))
        hyps = latest_by_id(read_jsonl(RESULTS_DIR / sname / "hyps" / f"{system}__{frontend}.jsonl"))
        overall, per_cat = Acc(), defaultdict(Acc)
        n_missing = n_err = n_empty = n_fe_err = 0
        for uid, rec in recs.items():
            if uid not in rows:
                continue
            s = speed[system]
            if rec.get("duration"):
                s["dur_all"] += rec["duration"]
                s["n_audio"] += 1
                if rec.get("reused_from") is None and rec.get("synth_seconds") is not None:
                    s["synth"] += rec["synth_seconds"]
                    s["cpu"] += rec.get("synth_cpu_seconds") or 0.0
                    s["dur_timed"] += rec["duration"]
                    s["n_timed"] += 1
            h = hyps.get(uid)
            if h is None or h.get("audio_sha1") != rec.get("audio_sha1"):
                n_missing += 1
                continue
            n_err += bool(rec.get("error"))
            n_empty += (not rec.get("error")) and not rec.get("duration")
            n_fe_err += bool(rec.get("frontend_error"))
            hyp_n = normalize_for_eval(h.get("hyp") or "")
            i, we, wn, ce, cn = best_ref(refs[uid], hyp_n)
            swe, swn, _, _ = edits(refs[uid][0], hyp_n)
            cat = rows[uid]["category"]
            overall.add(we, wn, ce, cn, swe, swn)
            per_cat[cat].add(we, wn, ce, cn, swe, swn)
            detail_rows.append({"id": uid, "category": cat, "system": system, "frontend": frontend,
                                "text": rows[uid]["text"], "frontend_text": rec.get("frontend_text"),
                                "model_text": rec.get("model_text"), "hyp": h.get("hyp"), "hyp_norm": hyp_n,
                                "ref_norm": refs[uid][i], "ref_index": i, "word_edits": we, "ref_words": wn,
                                "char_edits": ce, "ref_chars": cn, "swe": swe, "swn": swn})
        res = overall.out()
        res.update(n_missing=n_missing, n_synth_errors=n_err, n_empty_audio=n_empty,
                   n_frontend_errors=n_fe_err, complete=(overall.n == len(rows)),
                   per_category={c: per_cat[c].out() for c in categories if c in per_cat})
        results[system][frontend] = res

    subsets = {}
    for system, fr in results.items():
        ids = {d["id"] for d in detail_rows if d["system"] == system}
        if fr and not any(r["complete"] for r in fr.values()) and 0 < len(ids) < len(rows):
            table: dict = defaultdict(dict)
            for d in detail_rows:
                if d["id"] in ids:
                    table[d["system"]].setdefault(d["frontend"], Acc()).add(
                        d["word_edits"], d["ref_words"], d["char_edits"], d["ref_chars"], d["swe"], d["swn"])
            subsets[system] = {"n_ids": len(ids),
                               "results": {sy: {f: {**a.out(), "complete": a.n == len(ids)} for f, a in fr2.items()}
                                           for sy, fr2 in table.items()}}

    threads = defaultdict(set)
    for mp in synth_dir.glob("*__*.meta.json"):
        try:
            threads[mp.name.split("__", 1)[0]].add(json.loads(mp.read_text()).get("threads"))
        except (OSError, ValueError):
            pass
    speed_out = {}
    for system, s in speed.items():
        speed_out[system] = {
            "threads": "/".join(str(t) for t in sorted(t for t in threads[system] if t is not None)) or None,
            "rtf": round(s["synth"] / s["dur_timed"], 4) if s["dur_timed"] else None,
            "cpu_rtf": round(s["cpu"] / s["dur_timed"], 4) if s["dur_timed"] else None,
            "n_timed": s["n_timed"],
            "mean_audio_seconds": round(s["dur_all"] / s["n_audio"], 2) if s["n_audio"] else None,
        }

    judge_meta = {}
    jm = RESULTS_DIR / sname / "hyps" / "judge.meta.json"
    if jm.exists():
        judge_meta = json.loads(jm.read_text())
    summary = {
        "set": sname, "n_rows": len(rows), "has_alts": has_alts, "categories": categories,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_normalizer": {"trnorm": trnorm_fingerprint(), "function": "trnorm.normalize_for_eval"},
        "judge": judge_meta, "cpu": cpu_model(), "versions": pkg_versions(),
        "results": results, "subsets": subsets, "speed": speed_out,
        "systems": {n: {"model": c.model_id, "params": c.params, "license": c.license,
                        "native_mode": c.native_mode, "native_note": c.native_note,
                        "sample_rate": c.sample_rate} for n, c in SYSTEMS.items()},
        "frontends": FRONTEND_DESCRIPTIONS,
    }
    out_dir = RESULTS_DIR / sname
    write_text_atomic(out_dir / "rows.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in detail_rows))
    write_text_atomic(out_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    write_text_atomic(out_dir / "RESULTS.md", render_md(summary))
    return summary


def _cell(results, system, frontend, key="wer", key2="cer"):
    mode = SYSTEMS[system].native_mode if system in SYSTEMS else "separable"
    r = results.get(system, {}).get(frontend)
    if r is None:
        if frontend == "native" and mode == "always_on":
            return "= raw"
        if frontend == "native" and mode == "none":
            return "n/a"
        return "–"
    if r[key] is None:
        return "–"
    s = f"{r[key]:.1f}"
    if key2 and r.get(key2) is not None:
        s += f" ({r[key2]:.1f})"
    if not r["complete"]:
        s += f" [n={r['n']}]"
    return s


def render_md(sm: dict) -> str:
    res, n = sm["results"], sm["n_rows"]
    systems = [s for s in SYSTEM_ORDER if s in res] + [s for s in res if s not in SYSTEM_ORDER]
    fes = [f for f in FRONTENDS if any(f in res[s] for s in systems) or f == "native"]
    tn = sm["eval_normalizer"]["trnorm"]
    j = sm.get("judge") or {}
    L = []
    L.append(f"# TTS round-trip benchmark: `{sm['set']}`\n")
    L.append(f"Generated {sm['generated'][:16].replace('T', ' ')} UTC by `benchmarks/tts/score.py`. "
             f"{n} rows, {len(sm['categories'])} categories. "
             f"Judge: faster-whisper `{j.get('model', 'large-v3-turbo')}` int8 on CPU, beam 5, language tr. "
             f"Both the reference and the hypothesis go through `trnorm.normalize_for_eval` "
             f"(trnorm {tn['version']}, src sha256 {tn['src_sha256']}). "
             "WER/CER are corpus-level; " + (
                 "each row is scored against the closest of `spoken` and its `alts`. "
                 if sm["has_alts"] else "reference = `spoken`. ")
             + "Lower is better.\n")
    L.append("## WER % (CER %) by system and front-end\n")
    L.append("| system | " + " | ".join(fes) + " |")
    L.append("|---|" + "---:|" * len(fes))
    for s in systems:
        L.append(f"| {s} | " + " | ".join(_cell(res, s, f) for f in fes) + " |")
    L.append("")
    L.append("`= raw`: the system always applies its own normalization (piper: espeak-ng reads digits), "
             "so `raw` already is `native`. `n/a`: the system has no bundled normalizer. "
             "`[n=…]`: scored on a subset only.\n")
    if sm["has_alts"]:
        L.append("### Strict WER % (reference = `spoken` only, alts ignored)\n")
        L.append("| system | " + " | ".join(fes) + " |")
        L.append("|---|" + "---:|" * len(fes))
        for s in systems:
            L.append(f"| {s} | " + " | ".join(_cell(res, s, f, "wer_strict", None) for f in fes) + " |")
        L.append("")
    L.append("### Rows read without any word error (%)\n")
    L.append("| system | " + " | ".join(fes) + " |")
    L.append("|---|" + "---:|" * len(fes))
    for s in systems:
        L.append(f"| {s} | " + " | ".join(_cell(res, s, f, "row_ok", None) for f in fes) + " |")
    L.append("")

    for sub_sys, sub in sm.get("subsets", {}).items():
        sres = sub["results"]
        L.append(f"### Same {sub['n_ids']} rows as `{sub_sys}` (it ran on a subset): WER % (CER %)\n")
        L.append("| system | " + " | ".join(fes) + " |")
        L.append("|---|" + "---:|" * len(fes))
        for s in systems:
            if s in sres:
                L.append(f"| {s} | " + " | ".join(_cell(sres, s, f) for f in fes) + " |")
        L.append("")

    cats = sm["categories"]
    L.append("## WER % per category\n")
    L.append("| system | front-end | all | " + " | ".join(cats) + " |")
    L.append("|---|---|---:|" + "---:|" * len(cats))
    for s in systems:
        for f in FRONTENDS:
            r = res[s].get(f)
            if r is None:
                continue
            pc = r["per_category"]
            cells = [f"{pc[c]['wer']:.1f}" if c in pc and pc[c]["wer"] is not None else "–" for c in cats]
            L.append(f"| {s} | {f} | {r['wer']:.1f} | " + " | ".join(cells) + " |"
                     if r["wer"] is not None else f"| {s} | {f} | – | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Speed (CPU, <= 2 threads)\n")
    L.append("| system | params | out rate | threads | mean audio s | RTF (wall) | CPU-RTF | n timed |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in systems:
        sp, meta = sm["speed"].get(s, {}), sm["systems"].get(s, {})
        f = lambda v, d=3: "–" if v is None else f"{v:.{d}f}"  # noqa: E731
        L.append(f"| {s} | {meta.get('params', '?')} | {meta.get('sample_rate', '?')} | {sp.get('threads') or '?'} | "
                 f"{f(sp.get('mean_audio_seconds'), 2)} | {f(sp.get('rtf'))} | {f(sp.get('cpu_rtf'))} | "
                 f"{sp.get('n_timed', 0)} |")
    L.append("")
    L.append(f"RTF = synthesis wall-clock / audio duration over rows actually synthesized (rows whose model "
             f"input equals another front-end's are copied, not re-timed), after a warm-up, on "
             f"{sm['cpu']} with torch/onnxruntime capped at 2 threads (pocket-tts keeps its own default of "
             "1), `OMP_WAIT_POLICY=PASSIVE`, under `nice` on a machine shared with other jobs, so treat RTF "
             "as indicative; CPU-RTF (process CPU seconds / audio seconds, summed over threads) is less "
             "sensitive to contention. `results/speed/SPEED.md` has a side-by-side probe.\n")

    L.append("## Front-ends\n")
    for f in FRONTENDS:
        L.append(f"- `{f}`: {sm['frontends'][f]}")
    L.append("")
    L.append("## Systems\n")
    L.append("| system | model | params | licence | bundled normalizer |")
    L.append("|---|---|---|---|---|")
    for s in systems:
        m = sm["systems"].get(s, {})
        L.append(f"| {s} | {m.get('model')} | {m.get('params')} | {m.get('license')} | "
                 f"{m.get('native_mode')}: {m.get('native_note')} |")
    L.append("")
    probs = []
    for s in systems:
        for f, r in res[s].items():
            bits = [f"{k[2:].replace('_', ' ')}={r[k]}" for k in
                    ("n_missing", "n_synth_errors", "n_empty_audio", "n_frontend_errors") if r.get(k)]
            if bits:
                probs.append(f"- {s} / {f}: " + ", ".join(bits))
    if probs:
        L.append("## Incomplete or failed rows\n")
        L.extend(probs)
        L.append("")
    L.append("Per-row details (front-end text, model text, hypothesis, normalized reference/hypothesis, "
             "edits): `rows.jsonl` next to this file.\n")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="*", help="set names (default: every set under results/)")
    args = ap.parse_args(argv)
    names = [set_name(s) for s in args.set] if args.set else sorted(
        p.name for p in RESULTS_DIR.iterdir() if (p / "synth").is_dir())
    for name in names:
        sm = score_set(name)
        if sm is None:
            print(f"=== {name}: no synthesis results yet, nothing scored")
            continue
        print(f"\n=== {name}: {RESULTS_DIR / name / 'RESULTS.md'}")
        for s, fr in sm["results"].items():
            for f, r in fr.items():
                print(f"  {s:20s} {f:24s} n={r['n']:4d} WER={r['wer']} CER={r['cer']} strict={r['wer_strict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
