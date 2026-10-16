"""Ahenk demo Space (draft, ZeroGPU): VoxCPM2 2B + the Ahenk LoRA, trnorm front-end, instruction_pick for emotions,
pause_clean after synthesis.

The model is built once at startup, outside any GPU function, and moved to the GPU inside the @spaces.GPU function.
One request is capped at MAX_CHARS characters so that it fits the GPU_SECONDS budget: RTF was 2.28 on an A100 at 16
steps, so 200 characters (about 13 s of audio) take about 30 s.

Environment:
  MODEL_REPO       LoRA repo [ozgurguler/ahenk-tts-tr] (a private repo needs an HF_TOKEN secret)
  AHENK_LORA_DIR   a local dir with lora_weights.safetensors, lora_config.json and the helper .py files (testing)
  BASE_MODEL       [openbmb/VoxCPM2]

Not tested on ZeroGPU yet (only wired up locally with a stand-in VoxCPM). Check on the first run:
  - the build: VoxCPM at the pinned commit installs on Python 3.12 with torch 2.11, and its install size;
  - cold start: downloading VoxCPM2 (~5 GB) and building it on CPU; whether VoxCPM keeps device strings that
    _on() does not update (if so, build it with .to("cuda") at module level, the usual ZeroGPU pattern);
  - GPU_SECONDS against real timings for a 200-character request, and the visitors' daily quota it spends.
"""
from __future__ import annotations

try:
    import spaces  # ZeroGPU runtime; import before torch
except ImportError:  # local runs: a no-op decorator
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(*args, **kwargs):  # noqa: N802
            if args and callable(args[0]):
                return args[0]
            return lambda fn: fn

import json
import os
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download
from trnorm import normalize
from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

MODEL_REPO = os.environ.get("MODEL_REPO", "ozgurguler/ahenk-tts-tr")
BASE_MODEL = os.environ.get("BASE_MODEL", "openbmb/VoxCPM2")
MAX_CHARS = 200
GPU_SECONDS = 60  # per request, from the visitor's daily ZeroGPU quota

LORA_DIR = os.environ.get("AHENK_LORA_DIR") or snapshot_download(
    MODEL_REPO, allow_patterns=["lora_*", "*.py", "instructions_tr.json"])
sys.path.insert(0, LORA_DIR)
from instruction_pick import EMOTIONS, compose, pick  # noqa: E402  (stdlib, from the model repo)
from pause_clean import clean  # noqa: E402  (numpy + scipy, from the model repo)

_cfg = json.loads(Path(LORA_DIR, "lora_config.json").read_text(encoding="utf-8"))["lora_config"]
TTS = VoxCPM.from_pretrained(BASE_MODEL, load_denoiser=False, optimize=False, lora_config=LoRAConfig(**_cfg),
                             lora_weights_path=LORA_DIR)
SR = TTS.tts_model.sample_rate  # 48 kHz


def _on(device: str) -> None:
    """Move the VoxCPM model to device once (ZeroGPU attaches the GPU only inside @spaces.GPU calls)."""
    model = TTS.tts_model
    if next(model.parameters()).device.type != device:
        model.to(device)
    if hasattr(model, "device"):  # VoxCPM keeps a device attribute for the tensors it creates
        try:
            model.device = device
        except AttributeError:  # a read-only property: it follows the parameters
            pass


EMOTION_LABELS = {"Yok (düz okuma)": "", "Kızgın": "angry", "Neşeli": "cheerful", "Üzgün": "sad", "Sakin": "calm",
                  "Şaşkın": "surprised", "Empatik": "empathetic", "Gülerek": "laughing"}
assert set(EMOTION_LABELS.values()) - {""} <= set(EMOTIONS)
EXAMPLES = [
    ["Randevunuz 14.10.2026 saat 09:30'da; ücret 1.250 TL.", "Yok (düz okuma)"],
    ["Cihazın garanti süresi 2 yıl mı?", "Yok (düz okuma)"],
    ["Yeter artık! Üç gündür aynı sorun için arıyorum!", "Kızgın"],
    ["Vay canına! Bütün bu emeği tek başına mı verdin?", "Şaşkın"],
    ["Derin bir nefes alın, her şeyi adım adım birlikte halledeceğiz.", "Sakin"],
    ["Hahaha, bu fıkrayı daha önce hiç duymamıştım!", "Gülerek"],
]
NOTICE = (
    "**Bu ses yapaydır.** f022, VoxCPM2 ile bir metin tarifinden tasarlanmış bir sestir; gerçek bir kişinin sesi ya da "
    "kaydı değildir. Ürettiğiniz sesi paylaşırken yapay olduğunu belirtin; kimseyi taklit etmek, kandırmak ya da "
    "dolandırmak için kullanmayın.  \n"
    "*This voice is synthetic: a designed voice, not a real person. Label generated audio as AI-generated when you "
    "share it, and do not use it to impersonate or deceive anyone.*"
)


@spaces.GPU(duration=GPU_SECONDS)
def _generate(prompt: str, timesteps: int, seed: int) -> np.ndarray:
    _on("cuda" if torch.cuda.is_available() else "cpu")
    wav = TTS.generate(text=prompt, cfg_value=2.0, inference_timesteps=int(timesteps), normalize=False,
                       seed=int(seed))
    return np.asarray(wav, dtype=np.float32)


def synth(text: str, emotion_label: str, prefer: str, timesteps: int, seed: int):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Metin boş / empty text")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"En çok {MAX_CHARS} karakter / at most {MAX_CHARS} characters")
    instruction, model_text, why = pick(EMOTION_LABELS[emotion_label], normalize(text), prefer)
    prompt = compose(instruction, model_text)
    t0 = time.perf_counter()
    wav = _generate(prompt, timesteps, seed)
    dt = time.perf_counter() - t0
    wav, report = clean(wav, SR)
    dur = len(wav) / SR
    info = (f"{dur:.1f} sn ses, {dt:.1f} sn'de üretildi (RTF {dt / max(dur, 1e-6):.2f}); "
            f"duraklarda kısılan nefes: {report['breaths']}. Talimat seçimi: {why}")
    pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
    return (SR, pcm), prompt, info


with gr.Blocks(title="Ahenk") as demo:
    gr.Markdown(
        "# Ahenk: Türkçe metinden sese (VoxCPM2 + LoRA)\n"
        "Tek tasarlanmış kadın sesi, 6 duygu ve gülme. Metin önce "
        "[trnorm](https://github.com/ozguradmin/ahenk-turkish-tts) ile okunuş formuna çevrilir. "
        f"Bir istekte en çok {MAX_CHARS} karakter. Model kartı ve ölçümler: "
        "[ozgurguler/ahenk-tts-tr](https://huggingface.co/ozgurguler/ahenk-tts-tr). İşlemcide çalışan küçük model: "
        "[Ahenk Cep](https://huggingface.co/spaces/ozgurguler/ahenk-cep)."
    )
    gr.Markdown(NOTICE)
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Metin", lines=3, value=EXAMPLES[0][0], max_length=MAX_CHARS)
            emotion = gr.Dropdown(list(EMOTION_LABELS), value="Yok (düz okuma)", label="Duygu")
            with gr.Accordion("Ayarlar", open=False):
                prefer = gr.Radio(["auto", "tr", "en"], value="auto",
                                  label="Talimat dili (auto: ilk iki kelimeden biri c ile başlıyorsa Türkçe)")
                timesteps = gr.Slider(10, 16, value=16, step=2, label="Adım (az = hızlı)")
                seed = gr.Number(value=0, precision=0, label="Tohum")
            btn = gr.Button("Seslendir", variant="primary")
        with gr.Column():
            out = gr.Audio(label="Ses (48 kHz)", type="numpy")
            prompt_box = gr.Textbox(label="Modelin okuduğu metin (talimat + trnorm)", lines=3)
            info = gr.Markdown()
    btn.click(synth, [text, emotion, prefer, timesteps, seed], [out, prompt_box, info], concurrency_limit=1)
    gr.Examples(EXAMPLES, [text, emotion])
    gr.Markdown(
        "Bilinen sorunlar: İngilizce talimattan sonra baştaki \"c\" yanlış okunabilir (Talimat dili: tr); sonu "
        "yükselen sorular yükselmez; bazı duygulu kliplerde ses \"telefon gibi\" parlak çıkar, başka tohum deneyin."
    )

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
