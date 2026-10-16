"""Ahenk Cep demo Space: an 8.6M-parameter Turkish TTS that runs on the Space's CPU (CPU Basic, 2 vCPU).

Voices:
  - "Ahenk Cep (f022)": the on-device student, designed synthetic voice f022 (no real person)
  - "EMA Lightning + trnorm": upstream EMA Lightning (Apache-2.0) with trnorm as its text front-end
The 2B big model (Ahenk, VoxCPM2 + LoRA) needs a GPU and is not served here; its samples are on its model card.

Weights, first match wins:
  1. AHENK_CEP_WEIGHTS: a local dir with student.pt and decoder.pt (local testing)
  2. ./weights/ inside the Space
  3. the model repo MODEL_REPO (default ozgurguler/ahenk-cep-tr; a private repo needs an HF_TOKEN secret)
"""
from __future__ import annotations

import os

if os.environ.get("SPACES_ZERO_GPU"):  # optional: on ZeroGPU hardware the runtime wants one @spaces.GPU function
    import spaces  # noqa: F401  (import before torch on ZeroGPU)

import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from trnorm import normalize

from ahenk_cep import STYLES, AhenkCep, trnorm_ema

MODEL_REPO = os.environ.get("MODEL_REPO", "ozgurguler/ahenk-cep-tr")
MAX_CHARS = 600  # keeps one request under ~30 s of audio on 2 vCPUs

torch.set_num_threads(max(1, os.cpu_count() or 1))


def weight(name: str) -> str:
    if local := os.environ.get("AHENK_CEP_WEIGHTS"):
        path = Path(local) / name
        if not path.is_file():
            raise FileNotFoundError(f"AHENK_CEP_WEIGHTS={local} has no {name}")
        return str(path)
    bundled = Path(__file__).parent / "weights" / name
    if bundled.is_file():
        return str(bundled)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(MODEL_REPO, name)


if os.environ.get("SPACES_ZERO_GPU"):
    @spaces.GPU(duration=10)
    def _gpu_probe() -> bool:  # never called: the model runs on CPU and spends nobody's GPU quota
        return torch.cuda.is_available()


SMALL = AhenkCep(weight("student.pt"), weight("decoder.pt"))
EMA_TR = None  # loaded on first use (downloads ~34 MB from canberkkkkkk/ema-lightning)

VOICES = ["Ahenk Cep (f022)", "EMA Lightning + trnorm"]
STYLE_LABELS = {
    "Nötr": "neutral", "Kızgın": "angry", "Sakin": "calm", "Neşeli": "cheerful",
    "Empatik": "empathetic", "Üzgün": "sad", "Şaşkın": "surprised",
}
assert set(STYLE_LABELS.values()) <= set(STYLES)
EXAMPLES = [
    ["Randevunuz 14.10.2026 saat 09:30'da, 3. katta.", VOICES[0], "Nötr"],
    ["Bu ürünün garantisi var mı?", VOICES[0], "Nötr"],
    ["SGK'ye 15.00'te başvurun; ücret 1.250,75 TL, %20'si peşin.", VOICES[0], "Nötr"],
    ["Ne kadar uğraştıysam da yetişemedim, onunla son kez vedalaşamadım.", VOICES[0], "Üzgün"],
    ["Derin bir nefes alın, her şeyi adım adım birlikte halledeceğiz.", VOICES[0], "Sakin"],
    ["Kargonuz 7 Ağustos'ta teslim edilecek.", VOICES[1], "Nötr"],
]

NOTICE = (
    "**Bu ses yapaydır.** f022, VoxCPM2 ile bir metin tarifinden tasarlanmış bir sestir; gerçek bir kişinin sesi ya da "
    "kaydı değildir. Ürettiğiniz sesi paylaşırken yapay olduğunu belirtin; kimseyi taklit etmek, kandırmak ya da "
    "dolandırmak için kullanmayın.  \n"
    "*This voice is synthetic: a designed voice, not a real person. Label generated audio as AI-generated when you "
    "share it, and do not use it to impersonate or deceive anyone.*"
)


def synth(text: str, voice: str, style_label: str, steps: int, speed: float, seed: int):
    global EMA_TR
    text = (text or "").strip()
    if not text:
        raise gr.Error("Metin boş / empty text")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"En çok {MAX_CHARS} karakter / at most {MAX_CHARS} characters")
    t0 = time.perf_counter()
    if voice == VOICES[0]:
        audio, sr = SMALL.say(text, STYLE_LABELS[style_label], steps=int(steps), speed=float(speed), seed=int(seed))
    else:
        if EMA_TR is None:
            EMA_TR = trnorm_ema()
        sp = EMA_TR.say(text, speed=float(speed), seed=int(seed))
        audio, sr = sp.audio, sp.sample_rate
    dt = time.perf_counter() - t0
    dur = len(audio) / sr
    info = f"{dur:.1f} sn ses, {dt:.1f} sn'de üretildi (RTF {dt / max(dur, 1e-6):.2f}, {os.cpu_count()} CPU)"
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return (sr, pcm), normalize(text), info


with gr.Blocks(title="Ahenk Cep") as demo:
    gr.Markdown(
        "# Ahenk Cep: küçük Türkçe metinden sese modeli\n"
        "8,6 milyon parametre, 34 MB; bu demo işlemcide çalışır. Metin önce "
        "[trnorm](https://github.com/ozguradmin/ahenk-turkish-tts) ile okunuş formuna çevrilir (sayılar, tarihler, "
        "saat, para, kısaltmalar). Büyük model Ahenk (2B, GPU gerekir) için örnekler: "
        "[model kartı](https://huggingface.co/ozgurguler/ahenk-tts-tr)."
    )
    gr.Markdown(NOTICE)
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Metin", lines=4, value=EXAMPLES[0][0], max_length=MAX_CHARS)
            voice = gr.Radio(VOICES, value=VOICES[0], label="Ses")
            style = gr.Dropdown(list(STYLE_LABELS), value="Nötr", label="Üslup (yalnız f022; etkisi hafif)")
            with gr.Accordion("Ayarlar", open=False):
                steps = gr.Slider(4, 16, value=16, step=4, label="Adım (az = hızlı, biraz daha az doğal)")
                speed = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Hız")
                seed = gr.Number(value=0, precision=0, label="Tohum (aynı tohum, aynı ses)")
            btn = gr.Button("Seslendir", variant="primary")
        with gr.Column():
            out = gr.Audio(label="Ses (48 kHz)", type="numpy")
            spoken = gr.Textbox(label="Modelin okuduğu metin (trnorm)", lines=3)
            info = gr.Markdown()
    btn.click(synth, [text, voice, style, steps, speed, seed], [out, spoken, info], concurrency_limit=1)
    gr.Examples(EXAMPLES, [text, voice, style])
    gr.Markdown(
        "Sınırlamalar: tek tasarlanmış kadın ses; üslup etkisi hafif, gülme yok; sonu yükselen sorular "
        "(\"Yarın geliyorsun?\") yükselmez; uzun metinde doğallık büyük modelin gerisinde. Ölçümler ve kod: "
        "[github.com/ozguradmin/ahenk-turkish-tts](https://github.com/ozguradmin/ahenk-turkish-tts)."
    )

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
