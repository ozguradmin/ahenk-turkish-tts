#!/usr/bin/env python3
"""Pre-render the demo page's example sentences with Ahenk Cep, so each example plays at once without the model.

    python tools/web/render_samples.py [--models site/models] [--variant f16] [--seed 0] [--steps 16]
                                       [--release data/release-samples]

Uses the page's own pipeline (tts/small/web_pipeline.py: the same ONNX graphs, text front-end and torch.randn noise
as site/js), so "use this text" with the same seed gives the same voice in the browser. Writes, under site/samples/:

  cep_<id>.ogg   Opus in Ogg, 48 kHz mono (Chrome, Firefox, Edge, Safari 18.4+)
  cep_<id>.mp3   MP3, 48 kHz mono (fallback for older Safari)
  samples.json   texts, labels, durations and waveform peaks for every clip, plus the big model's clips

The big model's emotional samples are served from the Hugging Face model card (samples/ in ozgurguler/ahenk-tts-tr).
For local testing while that repo is private, --release copies them to site/samples/big/ (gitignored); the page uses
that copy only on localhost. Needs ffmpeg on PATH (libopus, libmp3lame) and the venv with onnxruntime.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "site"
sys.path[:0] = [str(REPO / "tts" / "small"), str(REPO / "src")]

# id, text, Turkish label, English label
EXAMPLES = [
    ("tarih", "Randevunuz 14.10.2026 saat 09:30'da, 3. katta.", "Tarih ve saat", "Date and time"),
    ("soru", "Bu ürünün garantisi var mı?", "Soru", "Question"),
    ("yuzde", "Bu ay faturanız %12 artışla 1.480 TL oldu.", "Yüzde", "Percent"),
    ("telefon", "Bize 0212 555 34 87 numarasından ulaşabilirsiniz.", "Telefon numarası", "Phone number"),
    ("para", "Siparişiniz 3 iş günü içinde, 249,90 TL karşılığında kapınızda olacak.", "Sayı ve para",
     "Numbers and money"),
    ("uzun", "Karşılaştırma isteyenlere eski ve yeni paketi yan yana anlatırız; farkı müşteri kendisi görür. "
     "Akşam olduğunda kısa bir değerlendirme yaparız: hangi sorular arttı, hangi açıklama iyi çalıştı, "
     "nerede zorlandık?", "Uzun metin", "A longer text"),
]

# the big model's samples on its model card: file, Turkish label, English label, text
BIG = [
    ("big_11_laughing.wav", "Gülerek", "Laughing", "Hahaha, bu fıkrayı daha önce hiç duymamıştım!"),
    ("big_07_cheerful.wav", "Neşeli", "Cheerful", "Harika haber! Sınavı geçtim, akşam hep birlikte kutlayalım!"),
    ("big_09_calm.wav", "Sakin", "Calm", "Derin bir nefes alın, her şeyi adım adım birlikte halledeceğiz."),
]

PEAKS = 160


def peaks(audio: np.ndarray, n: int = PEAKS) -> list[float]:
    """Peak absolute value in n equal slices, scaled so the loudest slice is 1 (two decimals, for drawing)."""
    if audio.size == 0:
        return [0.0] * n
    edges = np.linspace(0, audio.size, n + 1).astype(int)
    p = np.array([np.abs(audio[a:b]).max() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    p = p / max(p.max(), 1e-9)
    return [round(float(x), 2) for x in p]


def encode(wav: Path, out: Path, codec: list[str]):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", "48000", *codec,
                    "-map_metadata", "-1", "-metadata", "comment=AI-generated speech (Ahenk Cep, designed voice f022)",
                    str(out)], check=True)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=str(SITE / "models"))
    ap.add_argument("--variant", default="f16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--release", default=str(REPO / "data" / "release-samples"))
    ap.add_argument("--out", default=str(SITE / "samples"))
    a = ap.parse_args(argv)
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not on PATH")

    import soundfile as sf
    from web_pipeline import OnnxCep

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tts = OnnxCep(a.models, a.variant)
    examples = []
    with tempfile.TemporaryDirectory() as tmp:
        for eid, text, tr, en in EXAMPLES:
            audio, sr = tts.say(text, "neutral", steps=a.steps, seed=a.seed)
            wav = Path(tmp) / f"{eid}.wav"
            sf.write(str(wav), np.clip(audio, -1, 1), sr, subtype="PCM_16")
            encode(wav, out / f"cep_{eid}.ogg", ["-c:a", "libopus", "-b:a", "40k", "-application", "voip"])
            encode(wav, out / f"cep_{eid}.mp3", ["-c:a", "libmp3lame", "-b:a", "64k"])
            examples.append({"id": eid, "text": text, "spoken": tts.front.normalize(text), "tr": tr, "en": en,
                             "ogg": f"cep_{eid}.ogg", "mp3": f"cep_{eid}.mp3",
                             "seconds": round(audio.size / sr, 2), "peaks": peaks(audio)})
            print(f"[render_samples] {eid}: {audio.size / sr:.2f} s")

    big = []
    release = Path(a.release)
    for file, tr, en, text in BIG:
        entry = {"file": file, "tr": tr, "en": en, "text": text}
        src = release / file
        if src.is_file():
            audio, sr = read_wav(src)
            entry.update(seconds=round(audio.size / sr, 2), peaks=peaks(audio))
            (out / "big").mkdir(exist_ok=True)
            shutil.copyfile(src, out / "big" / file)
        big.append(entry)

    meta = {
        "note": "Pre-rendered with Ahenk Cep in the page's pipeline (tools/web/render_samples.py). All audio is "
                "synthetic: designed voice f022, no real person.",
        "model": f"Ahenk Cep ONNX {a.variant}, style neutral, {a.steps} steps, CFG 2, speed 1, seed {a.seed}",
        "seed": a.seed, "steps": a.steps, "examples": examples, "big": big,
    }
    (out / "samples.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in out.glob("cep_*"))
    print(f"[render_samples] {len(examples)} clips, {size / 1e3:.0f} kB; big model clips: "
          f"{sum('peaks' in b for b in big)}/{len(big)} copied")


if __name__ == "__main__":
    main()
