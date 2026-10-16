#!/usr/bin/env python3
"""Decode audio with an exported streaming Zipformer transducer in sherpa-onnx
(exactly what runs on Android/iOS/WASM/RPi) and score WER/CER.

Audio is fed in --chunk-ms pieces, like a live microphone. References and
hypotheses are both passed through trnorm.normalize_for_eval before scoring.

    python stt/tools/decode_sherpa.py --model-dir EXPORT_DIR --cuts $STT_DATA/manifests/fleurs_cuts_test_raw.jsonl.gz
    python stt/tools/decode_sherpa.py --model-dir EXPORT_DIR --int8 --wav-list my_tests.tsv   # "path<TAB>text"

Writes <out-dir>/<name>.{tsv,json}: per-utterance results and the summary
(WER, CER, RTF). With --itn-examples N, prints N hypotheses after
trnorm.inverse_normalize (spoken -> written form, e.g. "on beşte" -> "15'te");
ITN is display-only, scoring always uses normalize_for_eval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_items(a):
    items = []
    if a.cuts:
        from lhotse import load_manifest_lazy

        for c in load_manifest_lazy(a.cuts):
            items.append((c.id, c.supervisions[0].text or "", c))
            if a.max_utts and len(items) >= a.max_utts:
                break
    else:
        for line in Path(a.wav_list).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            path, _, text = line.partition("\t")
            items.append((Path(path).stem, text, path))
            if a.max_utts and len(items) >= a.max_utts:
                break
    return items


def load_audio(obj) -> np.ndarray:
    if isinstance(obj, str):
        import soundfile as sf

        x, sr = sf.read(obj, dtype="float32", always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if sr != 16000:
            import soxr

            x = soxr.resample(x, sr, 16000)
        return np.ascontiguousarray(x, dtype=np.float32)
    x = obj.load_audio()  # lhotse cut, (channels, samples)
    if obj.sampling_rate != 16000:
        x = obj.resample(16000).load_audio()
    return np.ascontiguousarray(x[0], dtype=np.float32)


def find_model(model_dir: Path, part: str, int8: bool) -> str:
    names = [f"{part}.int8.onnx"] if int8 else [f"{part}.onnx"]
    for n in names:
        if (model_dir / n).exists():
            return str(model_dir / n)
    pat = f"{part}-*.int8.onnx" if int8 else f"{part}-*.onnx"
    cands = [p for p in sorted(model_dir.glob(pat)) if int8 or ".int8." not in p.name and ".fp16." not in p.name]
    if not cands:
        sys.exit(f"no {pat} in {model_dir}")
    return str(cands[0])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tokens", default=None, help="default: <model-dir>/tokens.txt")
    p.add_argument("--int8", action="store_true")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cuts", help="lhotse cut manifest with recordings (e.g. *_raw.jsonl.gz)")
    g.add_argument("--wav-list", help="TSV: audio path <TAB> reference text")
    p.add_argument("--max-utts", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--decoding-method", default="greedy_search", choices=["greedy_search", "modified_beam_search"])
    p.add_argument("--chunk-ms", type=int, default=320)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--itn-examples", type=int, default=0)
    p.add_argument("--show", type=int, default=5, help="print this many ref/hyp pairs")
    a = p.parse_args()

    import kaldialign
    import sherpa_onnx
    import trnorm

    md = Path(a.model_dir)
    enc, dec, joi = (find_model(md, n, a.int8) for n in ("encoder", "decoder", "joiner"))
    tokens = a.tokens or str(md / "tokens.txt")
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens, encoder=enc, decoder=dec, joiner=joi,
        num_threads=a.num_threads, sample_rate=16000, feature_dim=80,
        decoding_method=a.decoding_method, provider="cpu", enable_endpoint_detection=False,
    )
    items = load_items(a)
    name = a.name or (Path(a.cuts or a.wav_list).name.split(".")[0] + ("-int8" if a.int8 else "-fp32"))
    out_dir = Path(a.out_dir or md / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk = int(16000 * a.chunk_ms / 1000)
    tail = np.zeros(int(0.66 * 16000), dtype=np.float32)
    w_err = w_ref = c_err = c_ref = 0
    audio_s = decode_s = 0.0
    rows = []
    for uid, ref, src in items:
        x = load_audio(src)
        audio_s += len(x) / 16000
        t0 = time.perf_counter()
        s = rec.create_stream()
        for i in range(0, len(x), chunk):
            s.accept_waveform(16000, x[i: i + chunk])
            while rec.is_ready(s):
                rec.decode_stream(s)
        s.accept_waveform(16000, tail)
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        hyp = rec.get_result(s)
        hyp = hyp if isinstance(hyp, str) else hyp.text
        decode_s += time.perf_counter() - t0
        r, h = trnorm.normalize_for_eval(ref), trnorm.normalize_for_eval(hyp)
        we = kaldialign.edit_distance(r.split(), h.split())
        ce = kaldialign.edit_distance(list(r.replace(" ", "")), list(h.replace(" ", "")))
        w_err += we["total"]
        w_ref += len(r.split())
        c_err += ce["total"]
        c_ref += len(r.replace(" ", ""))
        rows.append((uid, r, h, we["total"], len(r.split())))

    summary = {
        "name": name, "utterances": len(rows), "audio_hours": round(audio_s / 3600, 4),
        "WER": round(100 * w_err / max(w_ref, 1), 2), "CER": round(100 * c_err / max(c_ref, 1), 2),
        "RTF": round(decode_s / max(audio_s, 1e-9), 4), "num_threads": a.num_threads,
        "decoding_method": a.decoding_method, "chunk_ms": a.chunk_ms,
        "models": [os.path.basename(m) for m in (enc, dec, joi)],
    }
    with open(out_dir / f"{name}.tsv", "w", encoding="utf-8") as f:
        f.write("id\tref\thyp\tword_errors\tref_words\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")
    (out_dir / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for uid, r, h, *_ in rows[: a.show]:
        print(f"{uid}\n  REF: {r}\n  HYP: {h}")
    if a.itn_examples and hasattr(trnorm, "inverse_normalize"):
        print("ITN (display only):")
        for uid, _r, h, *_ in rows[: a.itn_examples]:
            print(f"  {uid}: {trnorm.inverse_normalize(h) if h else '(empty)'}")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
