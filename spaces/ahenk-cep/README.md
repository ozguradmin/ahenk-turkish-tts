---
title: Ahenk Cep
emoji: 🗣️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Küçük Türkçe TTS (8.6M), işlemcide çalışır
models:
  - ozgurguler/ahenk-cep-tr
  - canberkkkkkk/ema-lightning
suggested_hardware: cpu-basic
---

# Ahenk Cep demo

> **Bu ses yapaydır.** f022, VoxCPM2 ile bir metin tarifinden tasarlanmış bir sestir; gerçek bir kişinin sesi ya da
> kaydı değildir. Ürettiğiniz sesi paylaşırken yapay olduğunu belirtin. Kimseyi taklit etmek, kandırmak ya da
> dolandırmak için kullanmayın.
>
> **This voice is synthetic.** It is a designed voice, not a real person. Label what you generate as AI-generated,
> and do not use it to impersonate or deceive anyone.

Türkçe metni sese çeviren 8,6 milyon parametrelik (34 MB) bir model. Büyük model
[Ahenk](https://huggingface.co/ozgurguler/ahenk-tts-tr) (VoxCPM2 2B + LoRA) sesinden öğrenen bir öğrenci; EMA
Lightning'in mimarisi ve motoru üzerine kurulu. Metin önce [trnorm](https://github.com/ozguradmin/ahenk-turkish-tts)
ile okunuş formuna çevrilir: `14.10.2026` → *on dört ekim iki bin yirmi altı*, `1.250,75 TL` → *bin iki yüz elli
lira yetmiş beş kuruş*.

| Ses | Model | Lisans |
|---|---|---|
| Ahenk Cep (f022) | [ozgurguler/ahenk-cep-tr](https://huggingface.co/ozgurguler/ahenk-cep-tr) | ağırlıklar CC-BY-4.0, kod Apache-2.0 |
| EMA Lightning + trnorm | [canberkkkkkk/ema-lightning](https://huggingface.co/canberkkkkkk/ema-lightning) | Apache-2.0 |

**Hız:** bir dizüstü işlemcide 2 iş parçacığıyla 16 adımda RTF 0,20, 8 adımda 0,16. Space işlemcisi daha yavaş
olabilir; uygulama her sonuçta ölçülen RTF'yi ve işlemci sayısını gösterir. Aynı tohum işlemcide GPU'dakinden farklı
bir ses verir, model kartındaki örnekler GPU'dan.

**Ölçümler** (TR-Zor, 155 zor cümle; Whisper large-v3-turbo + trnorm; UTMOS22):

| Model | Boyut | WER | UTMOS |
|---|---|---:|---:|
| Ahenk Cep | 8.6M | 2.32 | 3.44 |
| EMA Lightning + trnorm | 8.6M | 2.43 | 3.36 |
| Antalia-2 Mini | 7.6M | 3.30 | 3.23 |
| Ahenk (büyük, burada yok) | 2B + LoRA | 2.54 | 3.69 |

Bunlar otomatik ölçümlerdir. Kör dinleme tek dinleyiciyle yapıldı (10 cümle: Ahenk 8, Ahenk Cep 2, diğerleri 0).

**Sınırlamalar:** tek tasarlanmış kadın ses, yalnız Türkçe. Üslup etkisi hafif, gülme yok. Sonu yükselen sorular
("Yarın geliyorsun?") yükselmez. Uzun metinde doğallık büyük modelin gerisinde.

Daha uzun bir metin ya da PDF, Word, EPUB belge mi? [Ahenk Okur](https://ozguradmin.github.io/ahenk-cep-pdfreader/) aynı modelle tarayıcıda tamamını
seslendirir, sonucu MP3 ya da WAV olarak indirebilirsiniz.

Kod ve ölçüm ayrıntıları: [github.com/ozguradmin/ahenk-turkish-tts](https://github.com/ozguradmin/ahenk-turkish-tts).
