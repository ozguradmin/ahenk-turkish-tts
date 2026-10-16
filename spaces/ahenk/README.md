---
title: Ahenk
emoji: 🗣️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Türkçe TTS, VoxCPM2 + LoRA, 6 duygu ve gülme
models:
  - ozgurguler/ahenk-tts-tr
  - openbmb/VoxCPM2
suggested_hardware: zero-a10g
---

# Ahenk demo (ZeroGPU)

> **Bu ses yapaydır.** f022, VoxCPM2 ile bir metin tarifinden tasarlanmış bir sestir; gerçek bir kişinin sesi ya da
> kaydı değildir. Ürettiğiniz sesi paylaşırken yapay olduğunu belirtin. Kimseyi taklit etmek, kandırmak ya da
> dolandırmak için kullanmayın.
>
> **This voice is synthetic.** It is a designed voice, not a real person. Label what you generate as AI-generated,
> and do not use it to impersonate or deceive anyone.

[Ahenk](https://huggingface.co/ozgurguler/ahenk-tts-tr): VoxCPM2 (2B) için Türkçe LoRA. Tek tasarlanmış kadın sesi,
soru tonlaması, 6 duygu ve gülme. Metin önce trnorm ile okunuş formuna çevrilir; duygu talimatının dilini
`instruction_pick` seçer; sentezden sonra `pause_clean` duraklardaki nefesi kısar. Bu iki dosya model reposundan
indirilir.

Her istek en çok 200 karakterdir ve ziyaretçinin günlük ZeroGPU kotasından en çok 60 saniye harcar. İşlemcide çalışan
küçük model için: [Ahenk Cep](https://huggingface.co/spaces/ozgurguler/ahenk-cep).

Ölçümler, sınırlamalar ve lisans (ağırlıklar CC-BY-4.0): [model kartı](https://huggingface.co/ozgurguler/ahenk-tts-tr).
