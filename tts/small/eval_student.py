#!/usr/bin/env python3
"""P1 evaluation of the student against EMA Lightning, Antalia-2 Mini and the teacher, exactly as the P0 benchmark
(tts/small/bench.py): the same 60 rows (OUT/subset.jsonl), seed 0, faster-whisper large-v3-turbo fp16 on the GPU (tr,
beam 5), WER/CER after trnorm.normalize_for_eval with the best reference per row, UTMOS22 strong at 16 kHz, RTF = wall
time / audio seconds per sentence after a warm-up (GPU plain path; CPU at --cpu-threads). Added for P1: ECAPA speaker
similarity (speechbrain/spkrec-ecapa-voxceleb, cached) of every clip to the f022 reference clip.

    <venv-student>/python tts/small/eval_student.py --systems student:ft-a=data/small/runs/ft-a/ckpt_00020000.pt \
        ema antalia teacher teacher_emadec --soru --styles --cpu-threads 2 8

Systems (wavs at OUT/<system>/<set>/<id>.wav):
  student:NAME=CKPT[@sampler,steps,cfg,sway,decoder]  rendered here (student.make_tts: EMA's engine and decoder.pt; trnorm
                                              spoken form in); CKPT "ema.pt" = EMA's own weights through this path
  ema, antalia                                the P0 renders (bench.py, same seed and settings) when present, else
                                              rendered with bench.System
  teacher                                     the big model's renders of the same sentences (VoxCPM2 + f022 LoRA,
                                              s3tr step 500, ts16 cfg2: voxcpm-eval/voxcpm2-f022-s3tr), 48 kHz
  teacher_emadec[:NAME=DECODER]               those teacher renders encoded with the VoxCPM2 AudioVAE (16 kHz, as the
                                              corpus) and decoded with EMA's decoder.pt: the ceiling of decoder.pt
--soru: the 35 TR-Soru rows, rendered (or the teacher's) under OUT/inton/<system>/wav/soru/, judged with word
timestamps; rows.jsonl there is ready for tts/voxcpm/score_intonation.py (run it with the test venv: librosa).
--styles: one sentence per style id (students only), OUT/<system>/styles/<style>.wav, judged.
Writes OUT/p1/<system>.rows.jsonl and OUT/p1/summary.json (merged with earlier systems), OUT/p1/SUMMARY.md.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0common as C  # noqa: E402
from student_data import parse_student  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

TEACHER = C.REPO / "voxcpm-eval" / "voxcpm2-f022-s3tr" / "step_0000500@ts16-cfg2"
F022_REF = Path(os.environ.get("F022_REF", C.REPO / "data" / "xfer" / "f022corpus" / "ref" /
                                "designed-f022-s0.wav"))
ECAPA = C.HF_HUB / "models--speechbrain--spkrec-ecapa-voxceleb" / "snapshots" / "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
SORU = C.REPO / "benchmarks" / "tts" / "sets" / "tr_soru.jsonl"
STYLE_TEXTS = {  # one sentence per style (style ids of student_data.STYLES; laughter is not trained)
    "neutral": "Kargonuz yarın öğleden önce adresinize teslim edilecek.",
    "angry": "Yeter artık! Üç haftadır aynı arıza için arıyorum ve hâlâ kimse gelmedi!",
    "calm": "Acele etmenize gerek yok. Derin bir nefes alın, her şeyi birlikte halledeceğiz.",
    "cheerful": "Harika haber! Sınavı geçtim, akşam hep birlikte kutlayalım!",
    "empathetic": "Yaşadığınız sorun için gerçekten çok üzgünüm, sizi çok iyi anlıyorum.",
    "sad": "Ne kadar uğraştıysam da yetişemedim, dedemle son kez vedalaşamadım.",
    "surprised": "Vay canına! Bütün bu yemekleri tek başına hazırladığına inanamıyorum!",
}


def repo_path(p) -> Path:
    return Path(p) if Path(p).is_absolute() else C.REPO / p


class Student:
    def __init__(self, spec: dict, device: str):
        import student as S
        from trnorm import normalize

        self.spec, self.trnorm, self.sr = spec, normalize, 48000
        ck = spec["ckpt"]
        path = ck if ck == "ema.pt" else (C.REPO / ck if not Path(ck).is_absolute() else Path(ck))
        self.model = S.load_student(path, device).set_sampling(spec["sampler"], spec["steps"], spec["cfg"], spec["sway"])
        dec = None
        if spec.get("decoder"):
            from ema_lightning.decoder import load_decoder

            dec = load_decoder(str(repo_path(spec["decoder"])), device)
        self.tts = S.make_tts(self.model, device, dec)

    def say(self, text: str, style: int = 0) -> np.ndarray:
        self.model.style = style
        return self.tts.say(self.trnorm(text), seed=0).audio


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def render(make, items, out_of, device, threads=None, warm=True) -> dict:
    """{key: {synth_seconds, duration}} for items [(key, text, style)]; wavs written where out_of(key) is not None."""
    if threads:
        torch.set_num_threads(threads)
    s = make(device)
    if warm:
        for w in ("Merhaba.", "Bugün hava çok güzel, yarın da yağmur yağacakmış.", "Saat on dörtte görüşelim."):
            s.say(w)
    sync()
    res = {}
    for key, text, style in items:
        t0 = time.perf_counter()
        a = s.say(text, style) if style is not None else s.say(text)
        sync()
        dt = time.perf_counter() - t0
        rec = {"synth_seconds": round(dt, 4), "duration": round(len(a) / s.sr, 3)}
        p = out_of(key)
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), a, s.sr, subtype="PCM_16")
        res[key] = rec
    del s
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


class Ecapa:
    def __init__(self, device):
        import torchaudio

        if not hasattr(torchaudio, "list_audio_backends"):  # torchaudio >= 2.9 dropped it; speechbrain 1.0 calls it
            torchaudio.list_audio_backends = lambda: ["soundfile"]
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        d = str(ECAPA)
        self.m = EncoderClassifier.from_hparams(source=d, savedir=d, overrides={"pretrained_path": d},
                                                run_opts={"device": device}, local_strategy=LocalStrategy.NO_LINK)
        self.ref = self.embed(C.load_16k(F022_REF))

    def embed(self, a16):
        with torch.inference_mode():
            e = self.m.encode_batch(torch.from_numpy(a16).unsqueeze(0)).reshape(-1).float()
        return torch.nn.functional.normalize(e, dim=0).cpu()

    def sim(self, a16) -> float:
        return round(float(self.embed(a16) @ self.ref), 4)


class Scorers:
    def __init__(self, device="cuda"):
        import bench

        self.judge = C.Judge(device)
        self.utmos = bench.load_utmos(device)
        self.ecapa = Ecapa(device)
        self.device = device

    def score(self, wav: Path, refs: list[str] | None, words=False) -> dict:
        a16 = C.load_16k(wav)
        j = self.judge.transcribe(a16, words=words)
        out = {"hyp": j["hyp"]}
        if words:
            out["words"] = j["words"]
        if refs:
            out.update(C.score_row(refs, j["hyp"]))
        if len(a16) > 1600:
            with torch.no_grad():
                out["utmos"] = round(float(self.utmos(torch.from_numpy(a16)[None].to(self.device), 16000)[0]), 4)
            out["sim"] = self.ecapa.sim(a16)
        return out


def teacher_emadec(src: Path, dst: Path, vae, dec, device):
    """teacher wav -> 16 kHz -> VoxCPM2 AudioVAE encoder (mu) -> EMA decoder.pt -> 48 kHz."""
    x = C.load_16k(src)
    with torch.no_grad():
        z = vae.encode(torch.from_numpy(x)[None, None].to(device))
        a = dec(z)[0].float().cpu().numpy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), a, 48000, subtype="PCM_16")


def summarize(rows: list[dict]) -> dict:
    s = {}
    for st in ("trzor", "heldout"):
        sub = [x for x in rows if x["set"] == st]
        if not sub:
            continue
        s[st] = C.corpus(sub)
        for k in ("utmos", "sim"):
            v = [x[k] for x in sub if x.get(k) is not None]
            s[st][k] = round(float(np.mean(v)), 3) if v else None
    main = [x for x in rows if x["set"] in ("trzor", "heldout")]
    if main:
        s["all"] = C.corpus(main)
        for k in ("utmos", "sim"):
            v = [x[k] for x in main if x.get(k) is not None]
            s["all"][k] = round(float(np.mean(v)), 3) if v else None
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="voxcpm-contest/small-bench")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--cpu-threads", nargs="*", type=int, default=[])
    ap.add_argument("--soru", action="store_true")
    ap.add_argument("--styles", action="store_true")
    ap.add_argument("--rerender", action="store_true", help="render ema/antalia again instead of using P0's wavs")
    ap.add_argument("--quick", action="store_true", help="bench rows only, no RTF on CPU, summary only to stdout+file")
    ap.add_argument("--rtf-only", action="store_true", help="only time synthesis of the 60 rows (GPU and --cpu-threads), "
                    "systems in turn and --repeats rounds, nothing written but OUT/p1/rtf.json")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    out = C.REPO / args.out
    p1 = out / "p1"
    p1.mkdir(parents=True, exist_ok=True)
    subset = C.read_jsonl(out / "subset.jsonl")
    soru = C.read_jsonl(SORU) if args.soru else []
    dev = "cuda"
    torch.set_num_threads(4)

    if args.rtf_only:
        return rtf_only(args, subset, out)

    # ---- render
    timing: dict[str, dict] = {}
    for spec in args.systems:
        if spec.startswith("student:"):
            st = parse_student(spec)
            name = st["name"]
            make = lambda d, st=st: Student(st, d)  # noqa: E731
        elif spec in ("ema", "antalia", "antalia_trnorm"):
            name = spec
            import bench

            def make(d, name=name):
                return bench.System(name, d)
        elif spec in ("teacher", "teacher_emadec") or spec.startswith("teacher_emadec:"):
            name = spec.split(":", 1)[1].split("=", 1)[0] if ":" in spec else spec
            make = None
        else:
            raise SystemExit(f"unknown system {spec}")
        items = [((r["set"], r["id"]), r["text"], None) for r in subset]
        wav_of = lambda k, name=name: out / name / k[0] / f"{k[1]}.wav"  # noqa: E731
        tm = {"gpu": {}, "cpu": {}}
        if make is not None and (spec.startswith("student:") or args.rerender
                                 or not all(wav_of(k).exists() for k, _, _ in items)):
            t0 = time.time()
            tm["gpu"] = render(make, items, wav_of, dev)
            print(f"[gpu] {name}: {time.time() - t0:.0f} s", flush=True)
            if not args.quick:
                for th in args.cpu_threads:
                    t0 = time.time()
                    tm["cpu"][th] = render(make, items, lambda k: None, "cpu", threads=th)
                    print(f"[cpu] {name} {th} threads: {time.time() - t0:.0f} s", flush=True)
                torch.set_num_threads(4)
        if make is not None and soru and not args.quick:
            render(make, [(("soru", r["id"]), r["text"], None) for r in soru],
                   lambda k, name=name: out / "inton" / name / "wav" / "soru" / f"{k[1]}.wav", dev)
        if make is not None and args.styles and spec.startswith("student:") and not args.quick:
            from student_data import STYLES

            render(make, [(("styles", s), STYLE_TEXTS[s], STYLES.index(s)) for s in STYLE_TEXTS],
                   lambda k, name=name: out / name / "styles" / f"{k[1]}.wav", dev, warm=False)
        timing[name] = (spec, tm)

    # ---- teacher wavs
    need_t = [n for n, (spec, _) in timing.items() if spec.startswith("teacher")]
    if need_t:
        for r in subset:
            src = TEACHER / "wav" / r["set"] / f"{r['id']}.wav"
            if "teacher" in need_t:
                dst = out / "teacher" / r["set"] / f"{r['id']}.wav"
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src.read_bytes())
        if soru and "teacher" in need_t:
            for r in soru:
                dst = out / "inton" / "teacher" / "wav" / "soru" / f"{r['id']}.wav"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes((TEACHER / "wav" / "soru" / f"{r['id']}.wav").read_bytes())
        for name in [n for n in need_t if timing[n][0].startswith("teacher_emadec")]:
            from audiovae import load_audiovae
            from ema_lightning.decoder import load_decoder

            spec = timing[name][0]
            dpath = repo_path(spec.split("=", 1)[1]) if "=" in spec else C.EMA_DIR / "decoder.pt"
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            vae = load_audiovae(C.VOXCPM_DIR / "audiovae.pth", dev, with_decoder=False)
            dec = load_decoder(str(dpath), dev)
            for r in subset:
                teacher_emadec(TEACHER / "wav" / r["set"] / f"{r['id']}.wav",
                               out / name / r["set"] / f"{r['id']}.wav", vae, dec, dev)
            if soru:
                for r in soru:
                    teacher_emadec(TEACHER / "wav" / "soru" / f"{r['id']}.wav",
                                   out / "inton" / name / "wav" / "soru" / f"{r['id']}.wav", vae, dec, dev)
            del vae, dec
            torch.cuda.empty_cache()

    # ---- score
    sc = Scorers(dev)
    p0rows = {(x["system"], x["set"], x["id"]): x for x in C.read_jsonl(out / "rows.jsonl")} \
        if (out / "rows.jsonl").exists() else {}
    sp = p1 / "summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {"judge": sc.judge.desc, "systems": {}}
    for name, (spec, tm) in timing.items():
        rows = []
        for r in subset:
            k = (r["set"], r["id"])
            wav = out / name / k[0] / f"{k[1]}.wav"
            row = {"system": name, **{x: r[x] for x in ("set", "id", "category", "text")},
                   "wav": str(wav.relative_to(C.REPO)).replace("\\", "/")}
            row.update(sc.score(wav, r["refs"]))
            row["duration"] = round(sf.info(str(wav)).duration, 3)
            g = tm["gpu"].get(k) or (p0rows.get((name, *k)) or {})
            if g.get("synth_seconds") is not None and g.get("duration"):
                row["rtf_gpu"] = round(g["synth_seconds"] / g["duration"], 4)
                row["synth_seconds"] = g["synth_seconds"]
            for th, c in tm["cpu"].items():
                row[f"rtf_cpu{th}"] = round(c[k]["synth_seconds"] / c[k]["duration"], 4)
            if not tm["cpu"] and (name, *k) in p0rows:
                for kk, v in p0rows[(name, *k)].items():
                    if kk.startswith("rtf_cpu"):
                        row[kk] = v
            rows.append(row)
        C.write_jsonl(p1 / f"{name}.rows.jsonl", rows)
        s = summarize(rows)
        s["spec"] = spec
        rr = [x for x in rows if x.get("rtf_gpu")]
        if rr:
            s["rtf_gpu"] = round(sum(x["synth_seconds"] for x in rr) / sum(x["synth_seconds"] / x["rtf_gpu"] for x in rr), 4)
        for th in sorted({kk for x in rows for kk in x if kk.startswith("rtf_cpu")}):
            if tm["cpu"]:
                c = tm["cpu"][int(th[7:])]
                s[th] = round(sum(v["synth_seconds"] for v in c.values()) / sum(v["duration"] for v in c.values()), 4)
        if not tm["cpu"]:
            p0s = json.loads((out / "summary.json").read_text(encoding="utf-8"))["systems"].get(name, {}) \
                if (out / "summary.json").exists() else {}
            for kk, v in p0s.items():
                if kk.startswith("rtf_"):
                    s[kk] = v
        s["audio_seconds"] = round(sum(x["duration"] for x in rows), 1)
        # soru: judge with word timestamps -> rows for score_intonation.py
        idir = out / "inton" / name
        if soru and (idir / "wav" / "soru").exists():
            irows = []
            for r in soru:
                wav = idir / "wav" / "soru" / f"{r['id']}.wav"
                if not wav.exists():
                    continue
                x = sc.score(wav, [r["spoken"]], words=True)
                irows.append({"set": "soru", "id": r["id"], "category": r["category"], "text": r["text"],
                              "model_text": r["spoken"], "pattern": r["pattern"], "peak": r["peak"],
                              "peak_word": r["peak_word"], "pair_id": r.get("pair_id"), **x})
            C.write_jsonl(idir / "rows.jsonl", irows)
            s["soru"] = C.corpus(irows)
            s["soru"]["utmos"] = round(float(np.mean([x["utmos"] for x in irows])), 3)
            s["soru"]["sim"] = round(float(np.mean([x["sim"] for x in irows])), 3)
        sdir = out / name / "styles"
        if args.styles and sdir.exists():
            s["styles"] = {}
            for st_name, text in STYLE_TEXTS.items():
                wav = sdir / f"{st_name}.wav"
                if wav.exists():
                    x = sc.score(wav, [text])
                    s["styles"][st_name] = {"hyp": x["hyp"], "word_edits": x["word_edits"], "utmos": x.get("utmos"),
                                            "sim": x.get("sim"), "duration": round(sf.info(str(wav)).duration, 2)}
        summary["systems"][name] = s
        print(f"[score] {name}: {json.dumps({k: v for k, v in s.items() if k != 'styles'}, ensure_ascii=False)}",
              flush=True)
    summary["f022_ref"] = str(F022_REF.name)
    sp.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (p1 / "SUMMARY.md").write_text(render_md(summary), encoding="utf-8")
    print((p1 / "SUMMARY.md").read_text(encoding="utf-8"))


def maker(spec: str):
    if spec.startswith("student:"):
        st = parse_student(spec)
        return st["name"], (lambda d, st=st: Student(st, d))
    import bench

    return spec, (lambda d, spec=spec: bench.System(spec, d))


def rtf_only(args, subset, out):
    """Synthesis RTF (sum of wall times / sum of audio seconds over the 60 rows), each system on the GPU and at each
    --cpu-threads, the systems interleaved and the whole round repeated (laptop clocks drift): median of rounds."""
    items = [((r["set"], r["id"]), r["text"], None) for r in subset]
    res: dict[str, dict[str, list]] = {}
    for rep in range(args.repeats):
        for spec in args.systems:
            name, make = maker(spec)
            for dev, th in [("cuda", None)] + [("cpu", t) for t in args.cpu_threads]:
                t = render(make, items, lambda k: None, dev, threads=th)
                rtf = sum(v["synth_seconds"] for v in t.values()) / sum(v["duration"] for v in t.values())
                key = "gpu" if dev == "cuda" else f"cpu{th}"
                res.setdefault(name, {}).setdefault(key, []).append(round(rtf, 4))
                print(f"[rtf] round {rep} {name} {key}: {rtf:.4f}", flush=True)
            torch.set_num_threads(4)
    summ = {n: {k: {"median": float(np.median(v)), "rounds": v} for k, v in d.items()} for n, d in res.items()}
    (out / "p1" / "rtf.json").write_text(json.dumps({"cpu": C.cpu_name(), "gpu": C.gpu_name(), "rtf": summ}, indent=1)
                                         + "\n", encoding="utf-8")
    for n, d in summ.items():
        print(n, {k: v["median"] for k, v in d.items()})


def render_md(s: dict) -> str:
    L = ["# Small-model P1 benchmark (local, $0)", "",
         f"Same 60 rows as P0 (40 TR-Zor + 20 held-out), judge {s.get('judge')}, WER/CER after normalize_for_eval, "
         f"UTMOS22 strong, ECAPA cosine to the f022 reference ({s.get('f022_ref')}). Seed 0.", "",
         "| System | TR-Zor-40 WER | CER | ok | Held-out-20 WER | ok | All-60 WER | UTMOS trzor / heldout | ECAPA sim | "
         "GPU RTF | CPU RTF 2t | CPU RTF 8t | TR-Soru-35 WER |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, x in s["systems"].items():
        a, b, al = x.get("trzor", {}), x.get("heldout", {}), x.get("all", {})
        L.append(f"| {n} | {a.get('wer')} | {a.get('cer')} | {a.get('row_ok')}/{a.get('n')} | {b.get('wer')} | "
                 f"{b.get('row_ok')}/{b.get('n')} | {al.get('wer')} | {a.get('utmos')} / {b.get('utmos')} | "
                 f"{al.get('sim')} | {x.get('rtf_gpu')} | {x.get('rtf_cpu2')} | {x.get('rtf_cpu8')} | "
                 f"{(x.get('soru') or {}).get('wer')} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
