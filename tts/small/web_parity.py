#!/usr/bin/env python3
"""Parity of the in-browser demo (site/) with the Python pipeline. Local, CPU + optional local GPU for Whisper; $0.

    python tts/small/web_parity.py --deps <dir with node_modules/{pyodide,onnxruntime-web}> --out <dir>
        [--weights hf-publish/ahenk-cep-tr] [--models site/models] [--variants fp32,f16,int8] [--audio 24]
        [--whisper cuda|cpu|none]

1. Text: every row of benchmarks/tts/sets/*.jsonl plus edge cases goes through the page's front-end in Node
   (Pyodide 0.27.7 running site/py) and through CPython (src/trnorm + site/py/ahenk_front.py, and, when
   ema_lightning is installed, EMA's own Frontend / chunk / Engine.piece around trnorm as tts/small/ahenk_cep.py
   uses them). Spoken form and pieces must match exactly.
2. Noise: site/js/randn.js against torch.randn for a few seeds.
3. Audio: the page's engine (site/js/engine.js on onnxruntime-web, in Node) against tts/small/web_pipeline.py
   (onnxruntime, same graphs) and against tts/small/ahenk_cep.py (PyTorch), same text / style / seed: length,
   max abs difference, SNR.
4. Whisper (faster-whisper large-v3-turbo, tr, beam 5, the P1 judge): WER of each variant's audio and PyTorch's on
   the audio rows, after trnorm.normalize_for_eval.

Writes <out>/report.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SITE = REPO / "site"
sys.path[:0] = [str(HERE), str(REPO / "src"), str(SITE / "py")]

SETS = ("tr_zor", "tr_soru", "contest", "freya_sample40", "dev_numbers10")
EDGE = [
    "", "   ", "...", "?!", "123", "%50", "Merhaba", "Merhaba dünya", "İSTANBUL'DA KIŞ", "Iğdır ılık.",
    "Çok güzel 😀 bir gün!", "naïve café résumé", "Saat 09:30'da​ buluşalım.", "A4 kâğıt, 2x3 m², 5 km/sa hız.",
    "Hello, how are you today?", "Tel: 0532 123 45 67, e-posta: destek@ozgurguler.tech",
    "Bu cümle uzun, " * 30 + "bitti.",
    "Birinci cümle. İkinci cümle! Üçüncü cümle mi? " * 8,
    "1.250,75 TL; %20'si peşin, kalan 3 taksitte. 14.10.2026 tarihinde, saat 15.00'te 3. katta görüşelim.",
]
AUDIO_EXTRA = [
    ("ex:randevu", "Randevunuz 14.10.2026 saat 09:30'da, 3. katta.", "neutral", 0),
    ("ex:garanti", "Bu ürünün garantisi var mı?", "neutral", 0),
    ("ex:sgk", "SGK'ye 15.00'te başvurun; ücret 1.250,75 TL, %20'si peşin.", "cheerful", 3),
    ("ex:uzgun", "Ne kadar uğraştıysam da yetişemedim, onunla son kez vedalaşamadım.", "sad", 0),
    ("ex:sakin", "Derin bir nefes alın, her şeyi adım adım birlikte halledeceğiz.", "calm", 1),
    ("ex:uzun", "Karşılaştırma isteyenlere eski ve yeni paketi yan yana anlatırız; farkı müşteri kendisi görür. "
                "Akşam olduğunda kısa bir değerlendirme yaparız: hangi sorular arttı, hangi açıklama iyi çalıştı, "
                "nerede zorlandık?", "empathetic", 2),
]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def text_rows() -> list[dict]:
    rows = []
    for s in SETS:
        for r in read_jsonl(REPO / "benchmarks" / "tts" / "sets" / f"{s}.jsonl"):
            refs = r.get("refs") or ([r["spoken"], *(r.get("alts") or [])] if r.get("spoken") else [])
            rows.append({"id": f"{s}:{r['id']}", "text": r["text"], "refs": refs})
    rows += [{"id": f"edge:{i:02d}", "text": t, "refs": []} for i, t in enumerate(EDGE)]
    return rows


def audio_rows(rows: list[dict], n: int) -> list[dict]:
    """The demo's examples, then TR-Zor rows spread over its categories, then TR-Soru, n in all."""
    out = [{"id": i, "text": t, "style": st, "seed": sd, "refs": [__import__("trnorm").normalize(t)]}
           for i, t, st, sd in AUDIO_EXTRA]
    zor = [r for r in rows if r["id"].startswith("tr_zor:")]
    soru = [r for r in rows if r["id"].startswith("tr_soru:")]
    pool = zor[::7] + soru[::9]
    styles = ("neutral", "cheerful", "sad", "angry", "surprised", "calm", "empathetic")
    for k, r in enumerate(pool[: max(0, n - len(out))]):
        out.append({**r, "style": styles[k % len(styles)], "seed": k})
    return out[:n]


def cpython_front(rows, vocab):
    import ahenk_front

    return {r["id"]: json.loads(ahenk_front.prepare(r["text"], 1.0, vocab)) for r in rows}


def ema_front(rows, vocab):
    """EMA Lightning's own front half around trnorm, as tts/small/ahenk_cep.py drives it (None without the package)."""
    try:
        from ema_lightning.chunker import chunk
        from ema_lightning.frontend import Frontend
    except ImportError:
        return None
    from trnorm import normalize

    fe = Frontend.__new__(Frontend)
    fe.vocab = frozenset(vocab)
    fe.spoken = lambda t: t
    out = {}
    for r in rows:
        spoken = normalize(r["text"])
        out[r["id"]] = [[p, pause] for p, pause in chunk(fe(spoken), 1.0)]
    return out


def file_id(row_id: str) -> str:  # as tools/web/parity_node.mjs names the files
    return re.sub(r"[^\w.-]", "_", row_id)


def snr(a, b) -> float:
    n = min(len(a), len(b))
    err = float(((a[:n] - b[:n]) ** 2).sum())
    return round(10 * np.log10(float((a[:n] ** 2).sum()) / max(err, 1e-20)), 1)


def compare_audio(x, y) -> dict:
    n = min(len(x), len(y))
    return {"len": [len(x), len(y)], "same_len": len(x) == len(y),
            "max_abs": round(float(np.abs(x[:n] - y[:n]).max()) if n else 0.0, 7), "snr_db": snr(x, y)}


def mos_scores(systems: dict, ref_torch: dict | None, sr: int, device: str) -> dict:
    """Mean UTMOS22 strong (16 kHz) per system; mean ECAPA cosine of each row to torch's render of the same row."""
    import p0common as C
    import torch
    import torchaudio

    sys.path.insert(0, str(C.UTMOS_HUB))
    utmos = torch.hub.load(str(C.UTMOS_HUB), "utmos22_strong", source="local", trust_repo=True).to(device).eval()
    if not hasattr(torchaudio, "list_audio_backends"):  # torchaudio >= 2.9 dropped it; speechbrain 1.0 calls it
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    d = str(C.HF_HUB / "models--speechbrain--spkrec-ecapa-voxceleb" / "snapshots" /
            "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286")
    ecapa = EncoderClassifier.from_hparams(source=d, savedir=d, overrides={"pretrained_path": d},
                                           run_opts={"device": device}, local_strategy=LocalStrategy.NO_LINK)

    def embed(x16):
        with torch.inference_mode():
            e = ecapa.encode_batch(torch.from_numpy(x16).unsqueeze(0)).reshape(-1).float()
        return torch.nn.functional.normalize(e, dim=0).cpu()

    ref = {i: embed(C.to_16k(x, sr)) for i, x in (ref_torch or {}).items()}
    out = {}
    for name, aud in systems.items():
        us, sims = [], []
        for i, x in aud.items():
            x16 = C.to_16k(x, sr)
            with torch.inference_mode():
                us.append(float(utmos(torch.from_numpy(x16)[None].to(device), 16000)[0]))
            if i in ref:
                sims.append(float(embed(x16) @ ref[i]))
        out[name] = {"utmos": round(float(np.mean(us)), 3), "n": len(us)}
        if sims:
            out[name]["ecapa_to_torch"] = round(float(np.mean(sims)), 3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deps", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=REPO / "hf-publish" / "ahenk-cep-tr")
    ap.add_argument("--models", type=Path, default=SITE / "models")
    ap.add_argument("--variants", default="fp32,f16,int8")
    ap.add_argument("--audio", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--whisper", default="cuda", choices=("cuda", "cpu", "none"))
    ap.add_argument("--mos", action="store_true", help="also UTMOS22 strong per system and ECAPA cosine to torch's render")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.models / "ahenk-cep.json").read_text(encoding="utf-8"))
    vocab = manifest["vocab"]
    rows = text_rows()
    arows = audio_rows(rows, a.audio)
    node_rows = arows + [r for r in rows if r["id"] not in {x["id"] for x in arows}]
    (a.out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in node_rows),
                                      encoding="utf-8")
    report = {"rows": len(node_rows), "audio_rows": len(arows), "variants": {}}

    # ---- node runs (the page's code) ----
    for v in a.variants.split(","):
        d = a.out / v
        cmd = ["node", str(REPO / "tools" / "web" / "parity_node.mjs"), "--deps", str(a.deps), "--texts",
               str(a.out / "rows.jsonl"), "--out", str(d), "--models", str(a.models), "--variant", v,
               "--audio", str(len(arows)), "--threads", str(a.threads)]
        print("[web_parity]", " ".join(cmd[:2]), "--variant", v)
        subprocess.run(cmd, check=True)

    # ---- 1. text ----
    first = a.out / a.variants.split(",")[0]
    node = {r["id"]: r for r in read_jsonl(first / "front.jsonl")}
    cpy = cpython_front(node_rows, vocab)
    ema = ema_front(node_rows, vocab)
    bad_spoken = [i for i in cpy if cpy[i]["spoken"] != node[i]["spoken"]]
    bad_pieces = [i for i in cpy if cpy[i]["pieces"] != node[i]["pieces"]]
    bad_ema = None if ema is None else [i for i in cpy if ema[i] != [[p["text"], p["pause"]] for p in cpy[i]["pieces"]]]
    report["text"] = {"rows": len(cpy), "spoken_mismatch": bad_spoken, "pieces_mismatch": bad_pieces,
                      "ema_frontend_mismatch": bad_ema}
    print(f"[web_parity] text: {len(cpy)} rows, spoken mismatches {len(bad_spoken)}, piece mismatches "
          f"{len(bad_pieces)}, vs EMA's own front {'n/a' if bad_ema is None else len(bad_ema)}")

    # ---- 2. noise ----
    try:
        import torch

        rj = json.loads((first / "randn.json").read_text())
        diffs = {s: float(np.abs(np.array(v, np.float32) - torch.randn(
            (4, 37, 64), generator=torch.Generator().manual_seed(int(s)))[0].reshape(-1).numpy()).max())
            for s, v in rj.items()}
        report["randn_max_abs_vs_torch"] = diffs
        print(f"[web_parity] randn vs torch: max {max(diffs.values()):.2e}")
    except ImportError:
        torch = None

    # ---- 3. audio ----
    from web_pipeline import OnnxCep

    sr = manifest["sample_rate"]
    node_audio = {v: {r["id"]: np.fromfile(a.out / v / "audio" / f"{file_id(r['id'])}.f32", dtype=np.float32)
                      for r in arows} for v in a.variants.split(",")}
    py_onnx = OnnxCep(a.models, "fp32", threads=a.threads)
    ref_onnx = {r["id"]: py_onnx.say(r["text"], r["style"], seed=r["seed"])[0] for r in arows}
    ref_torch = None
    if torch is not None:
        from ahenk_cep import AhenkCep

        torch.set_num_threads(a.threads)
        pt = AhenkCep(a.weights / "student.pt", a.weights / "decoder.pt")
        pt.engine._batch_size = 1
        ref_torch = {}
        t0 = time.perf_counter()
        for r in arows:
            ref_torch[r["id"]] = pt.say(r["text"], r["style"], seed=r["seed"])[0]
        report["torch_cpu_seconds"] = round(time.perf_counter() - t0, 2)
    for v, aud in node_audio.items():
        rep = {"vs_python_onnx_fp32": {i: compare_audio(aud[i], ref_onnx[i]) for i in aud}}
        if ref_torch:
            rep["vs_torch"] = {i: compare_audio(aud[i], ref_torch[i]) for i in aud}
        rep["timing"] = json.loads((a.out / v / "timing.json").read_text())
        report["variants"][v] = rep
        same = sum(x["same_len"] for x in rep["vs_python_onnx_fp32"].values())
        snrs = [x["snr_db"] for x in rep.get("vs_torch", rep["vs_python_onnx_fp32"]).values() if x["same_len"]]
        audio_s = sum(t["seconds"] for t in rep["timing"]["rows"])
        wall = sum(t["ms"] for t in rep["timing"]["rows"]) / 1000
        print(f"[web_parity] {v}: same length as python {same}/{len(aud)}; SNR vs torch (same-length rows) "
              f"min {min(snrs) if snrs else 'n/a'} median {np.median(snrs) if snrs else 'n/a'} dB; "
              f"node RTF {wall / audio_s:.3f} ({a.threads} threads)")

    # ---- the page's WAV file ----
    import soundfile as sf

    wav, wsr = sf.read(str(first / "first.wav"), dtype="int16")
    expect = (np.clip(node_audio[a.variants.split(",")[0]][arows[0]["id"]], -1, 1) * 32767).round().astype(np.int16)
    report["wav"] = {"sample_rate": wsr, "samples": len(wav), "max_lsb_diff": int(np.abs(wav.astype(int) - expect).max())}
    print(f"[web_parity] page WAV: {wsr} Hz, {len(wav)} samples, max diff {report['wav']['max_lsb_diff']} LSB vs write_wav")

    # ---- 4. whisper ----
    if a.whisper != "none":
        import p0common as C

        judge = C.Judge(a.whisper)
        systems = {f"web-{v}": aud for v, aud in node_audio.items()}
        if ref_torch:
            systems["torch"] = ref_torch
        report["whisper"] = {"judge": judge.desc}
        for name, aud in systems.items():
            scored = []
            for r in arows:
                j = judge.transcribe(C.to_16k(aud[r["id"]], sr))
                scored.append({"id": r["id"], "hyp": j["hyp"], **C.score_row(r["refs"], j["hyp"])})
            report["whisper"][name] = {**C.corpus(scored), "rows": scored}
            c = report["whisper"][name]
            print(f"[web_parity] whisper {name}: WER {c['wer']} CER {c['cer']} ok {c['row_ok']}/{c['n']}")
    if a.mos:
        report["mos"] = mos_scores(systems if a.whisper != "none" else {f"web-{v}": x for v, x in node_audio.items()},
                                   ref_torch, sr, "cuda" if a.whisper == "cuda" else "cpu")
        for name, m in report["mos"].items():
            print(f"[web_parity] {name}: UTMOS {m['utmos']}, ECAPA to torch {m.get('ecapa_to_torch')}")
    (a.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[web_parity] wrote {a.out / 'report.json'}")


if __name__ == "__main__":
    main()
