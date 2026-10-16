#!/usr/bin/env python3
"""P0 benchmark: EMA Lightning vs Antalia-2 Mini on a fixed subset, locally (RTX 5060 + CPU), $0.

    set HF_HUB_OFFLINE=1
    <venv-student>/python tts/small/bench.py --out voxcpm-contest/small-bench

Subset (written to OUT/subset.jsonl): 40 TR-Zor rows stratified by category (largest-remainder allocation of
40/155 per category, rows drawn with random.Random(0) inside each category) + the 20 held-out sentences of the
VoxCPM2 evals (voxcpm-eval/voxcpm2-f022-s3tr/*/rows.jsonl, set=heldout; refs as there).

Systems (seed 0 everywhere, default synthesis settings of each package):
  ema             EMA Lightning 1.1.2 (ema.pt + decoder.pt), trnorm front-end (= trtts; normalizer-tr bypassed)
  antalia         Antalia-2 Mini 1.0.1, its own built-in normaliser on the raw text (native)
  antalia_trnorm  Antalia-2 Mini with trnorm's spoken form; its normaliser bypassed (lowercase + symbol filter only)

Renders: OUT/<system>/<set>/<id>.wav (GPU renders, 48 kHz as each model outputs). Judge: faster-whisper
large-v3-turbo fp16 on the GPU, tr, beam 5; WER/CER after trnorm.normalize_for_eval, best reference per row.
UTMOS22 strong (cached torch hub code + weights, no download). RTF = synthesis wall time / audio seconds, per
sentence call, after warm-up; GPU (plain path, no CUDA graphs) and CPU at --cpu-threads.
Writes OUT/rows.jsonl, OUT/summary.json, OUT/SUMMARY.md.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0common as C  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

SYSTEMS = ("ema", "antalia", "antalia_trnorm")
HELDOUT_SRC = C.REPO / "voxcpm-eval" / "voxcpm2-f022-s3tr" / "step_0000500@ts16-cfg2" / "rows.jsonl"


def build_subset(n_trzor: int = 40, seed: int = 0) -> list[dict]:
    rows = C.read_jsonl(C.REPO / "benchmarks" / "tts" / "sets" / "tr_zor.jsonl")
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    quota = {c: n_trzor * len(v) / len(rows) for c, v in by.items()}
    alloc = {c: int(q) for c, q in quota.items()}
    for c in sorted(quota, key=lambda c: (-(quota[c] - alloc[c]), c))[: n_trzor - sum(alloc.values())]:
        alloc[c] += 1
    rng = random.Random(seed)
    out = []
    for c, v in by.items():
        for r in sorted(rng.sample(v, alloc[c]), key=lambda r: r["id"]):
            out.append({"set": "trzor", "id": r["id"], "category": c, "text": r["text"],
                        "refs": [r["spoken"], *(r.get("alts") or [])]})
    for r in C.read_jsonl(HELDOUT_SRC):
        if r.get("set") == "heldout":
            out.append({"set": "heldout", "id": r["id"], "category": "heldout", "text": r["text"], "refs": r["refs"]})
    return out


class System:
    def __init__(self, name: str, device: str):
        self.name, self.device = name, device
        from trnorm import normalize

        self.trnorm = normalize
        if name == "ema":
            from ema_lightning import EMA

            self.tts = EMA(device=device)
            self.tts._frontend.spoken = lambda t: t  # trtts: trnorm decides what is said
            self.sr = 48000
        else:
            import antalia_mini.api as aapi
            from antalia_mini import Antalia
            from antalia_mini.text import clean_model_text, turkish_lower

            if name == "antalia_trnorm":
                aapi.model_text = lambda raw: clean_model_text(turkish_lower(unicodedata.normalize("NFC", raw)))
            self.tts = Antalia(str(C.ANTALIA_DIR), device=device, threads=None)
            self.sr = self.tts.sample_rate

    def model_input(self, text: str) -> str:
        return text if self.name == "antalia" else self.trnorm(text)

    def model_text(self, text: str) -> str:
        """What the network reads (after each package's own last text step)."""
        t = self.model_input(text)
        if self.name == "ema":
            return self.tts._frontend(t)
        import antalia_mini.api as aapi

        return aapi.model_text(t)

    def say(self, text: str) -> np.ndarray:
        sp = self.tts.say(self.model_input(text), seed=0)
        a = sp.audio
        return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a, dtype=np.float32)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def render(name: str, device: str, subset: list[dict], out: Path | None, threads: int | None = None) -> dict:
    if threads:
        torch.set_num_threads(threads)
    s = System(name, device)
    for w in ("Merhaba.", "Bugün hava çok güzel, yarın da yağmur yağacakmış.", "Saat on dörtte görüşelim."):
        s.say(w)
    sync()
    res = {}
    for r in subset:
        t0 = time.perf_counter()
        a = s.say(r["text"])
        sync()
        dt = time.perf_counter() - t0
        dur = len(a) / s.sr
        rec = {"synth_seconds": round(dt, 4), "duration": round(dur, 3)}
        if out is not None:
            p = out / name / r["set"] / f"{r['id']}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), a, s.sr, subtype="PCM_16")
            rec.update(wav=str(p.relative_to(C.REPO)).replace("\\", "/"), model_text=s.model_text(r["text"]),
                       sr=s.sr)
        res[(r["set"], r["id"])] = rec
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    del s
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return {"rows": res, "peak_mib": peak}


def load_utmos(device):
    sys.path.insert(0, str(C.UTMOS_HUB))
    m = torch.hub.load(str(C.UTMOS_HUB), "utmos22_strong", source="local", trust_repo=True)
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="voxcpm-contest/small-bench")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--cpu-threads", nargs="*", type=int, default=[2, 8])
    ap.add_argument("--cpu-systems", nargs="*", default=["ema", "antalia"])
    ap.add_argument("--no-utmos", action="store_true")
    args = ap.parse_args()
    out = (C.REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    subset = build_subset()
    C.write_jsonl(out / "subset.jsonl", subset)
    print(f"subset: {len(subset)} rows", flush=True)

    gpu, cpu = {}, defaultdict(dict)
    for name in args.systems:
        t0 = time.time()
        gpu[name] = render(name, "cuda", subset, out)
        print(f"[gpu] {name}: {time.time() - t0:.0f} s, peak {gpu[name]['peak_mib']:.0f} MiB", flush=True)
    for name in args.cpu_systems:
        for th in args.cpu_threads:
            t0 = time.time()
            cpu[name][th] = render(name, "cpu", subset, None, threads=th)
            print(f"[cpu] {name} {th} threads: {time.time() - t0:.0f} s", flush=True)
    torch.set_num_threads(os.cpu_count() or 4)

    judge = C.Judge("cuda")
    utmos = None if args.no_utmos else load_utmos("cuda")
    rows = []
    for name in args.systems:
        for r in subset:
            rec = gpu[name]["rows"][(r["set"], r["id"])]
            wav = C.REPO / rec["wav"]
            a16 = C.load_16k(wav)
            j = judge.transcribe(a16)
            sc = C.score_row(r["refs"], j["hyp"])
            row = {"system": name, **{k: r[k] for k in ("set", "id", "category", "text")}, **rec, "hyp": j["hyp"], **sc}
            row["rtf_gpu"] = round(rec["synth_seconds"] / rec["duration"], 4) if rec["duration"] else None
            for th, c in cpu.get(name, {}).items():
                cr = c["rows"][(r["set"], r["id"])]
                row[f"rtf_cpu{th}"] = round(cr["synth_seconds"] / cr["duration"], 4) if cr["duration"] else None
            if utmos is not None and rec["duration"] > 0.1:
                with torch.no_grad():
                    row["utmos"] = round(float(utmos(torch.from_numpy(a16)[None].to("cuda"), 16000)[0]), 4)
            rows.append(row)
        print(f"[judge] {name} done", flush=True)
    C.write_jsonl(out / "rows.jsonl", rows)

    summary = {"judge": judge.desc, "gpu": C.gpu_name(), "cpu": C.cpu_name(), "cpu_count": os.cpu_count(),
               "torch": torch.__version__, "n_subset": len(subset), "systems": {}}
    for name in args.systems:
        rs = [x for x in rows if x["system"] == name]
        s = {}
        for st in ("trzor", "heldout"):
            sub = [x for x in rs if x["set"] == st]
            s[st] = C.corpus(sub)
            ut = [x["utmos"] for x in sub if x.get("utmos") is not None]
            s[st]["utmos"] = round(float(np.mean(ut)), 3) if ut else None
        s["all"] = C.corpus(rs)
        tot = lambda key, rr: sum(x[key] for x in rr)  # noqa: E731
        s["rtf_gpu"] = round(sum(gpu[name]["rows"][k]["synth_seconds"] for k in gpu[name]["rows"])
                             / sum(gpu[name]["rows"][k]["duration"] for k in gpu[name]["rows"]), 4)
        s["gpu_peak_mib"] = round(gpu[name]["peak_mib"], 0)
        for th, c in cpu.get(name, {}).items():
            s[f"rtf_cpu{th}"] = round(sum(v["synth_seconds"] for v in c["rows"].values())
                                      / sum(v["duration"] for v in c["rows"].values()), 4)
        s["audio_seconds"] = round(tot("duration", rs), 1)
        cats = defaultdict(list)
        for x in rs:
            if x["set"] == "trzor":
                cats[x["category"]].append(x)
        s["trzor_by_category"] = {c: C.corpus(v) for c, v in cats.items()}
        summary["systems"][name] = s
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "SUMMARY.md").write_text(render_md(summary, rows), encoding="utf-8")
    print((out / "SUMMARY.md").read_text(encoding="utf-8"))


def render_md(s, rows) -> str:
    names = list(s["systems"])
    th = sorted({k[7:] for n in names for k in s["systems"][n] if k.startswith("rtf_cpu")}, key=int)
    L = [f"# Small-model P0 benchmark (local, {s['gpu']}, $0)", "",
         f"Subset: {s['n_subset']} rows = 40 TR-Zor (stratified) + 20 held-out. Judge: {s['judge']}; "
         "WER/CER after trnorm.normalize_for_eval, best reference per row. Seed 0. RTF = wall time / audio.", "",
         "| System | TR-Zor-40 WER | CER | ok | Held-out-20 WER | CER | ok | UTMOS trzor / heldout | GPU RTF | "
         + " | ".join(f"CPU RTF {t}t" for t in th) + " |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(th)]
    for n in names:
        x = s["systems"][n]
        a, b = x["trzor"], x["heldout"]
        L.append(f"| {n} | {a['wer']} | {a['cer']} | {a['row_ok']}/{a['n']} | {b['wer']} | {b['cer']} | {b['row_ok']}/{b['n']} | "
                 f"{a['utmos']} / {b['utmos']} | {x['rtf_gpu']} | " + " | ".join(str(x.get(f'rtf_cpu{t}')) for t in th) + " |")
    cats = list(next(iter(s["systems"].values()))["trzor_by_category"])
    L += ["", "TR-Zor-40 WER by category (n rows):", "", "| Category | n | " + " | ".join(names) + " |",
          "|---|---:|" + "---:|" * len(names)]
    for c in cats:
        L.append(f"| {c} | {s['systems'][names[0]]['trzor_by_category'][c]['n']} | "
                 + " | ".join(str(s['systems'][n]['trzor_by_category'][c]['wer']) for n in names) + " |")
    L += ["", "Rows with word errors:", ""]
    for r in rows:
        if r["word_edits"]:
            L.append(f"- {r['system']} {r['set']}/{r['id']}: ref `{r['ref_norm']}` | hyp `{r['hyp_norm']}`")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
