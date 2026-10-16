---
language:
  - tr
license: cc-by-4.0
library_name: piper
pipeline_tag: text-to-speech
tags:
  - piper
  - vits
  - turkish
  - synthetic-voice
datasets:
  - cloud0day3/antalia-voice-corpus
---

# {{name}} — Türkçe Piper sesi (Antalia)

> **Bu bir sentetik (yapay) sestir.** Gerçek bir seslendirme sanatçısının izinli kayıtlarıyla
> eğitildi. Dinleyicinin yanılabileceği her yerde sesin yapay olduğunu belirtin; bu sesi
> sanatçıyı ya da herhangi bir gerçek kişiyi taklit etmek için **asla** kullanmayın.
>
> **This is a synthetic voice.** It was trained on consented recordings of a real voice actor.
> Disclose that it is synthetic wherever a listener could be misled, and **never** use it to
> impersonate the actor or any real person.

| | |
|---|---|
| Model | Piper VITS "medium" (~15.7 M generator parameters at inference), 22,050 Hz, mono |
| Dil / Language | Türkçe (`tr_TR`), espeak-ng `tr` fonemleri |
| Eğitim / Training | LJSpeech "medium" kontrol noktasından ince ayar (fine-tuned from LJSpeech medium): {{epochs_trained}} epoch, {{batches_trained}} batch (`{{checkpoint}}`) |
| Dışa aktarma / Exported | {{exported}}, ONNX {{onnx_mb}} MB |
| Lisans / Licence | **CC-BY-4.0** (ağırlıklar / weights) |
| Ön uç / Front-end | metni **mutlaka** `trnorm.normalize` ile verin / always feed text through `trnorm.normalize` |

## Kullanım / Usage

```bash
pip install piper-tts trnorm   # trnorm: https://github.com/ozguradmin/ahenk-turkish-tts
python -c "from trnorm import normalize; print(normalize('Randevunuz 14.10.2026 saat 09:30\'da.'))" \
  | piper -m {{name}}.onnx -f cikti.wav
```

Home Assistant: copy `{{name}}.onnx` and `{{name}}.onnx.json` into the Piper add-on's
`/share/piper/` (or the `--data-dir` of `wyoming-piper`), restart, and pick the voice. Home Assistant
does not run trnorm; numbers, dates and abbreviations are then read by espeak-ng's own rules.

## Eğitim verisi ve atıf / Training data and attribution

- **Antalia (Patientdesk.ai)** — *Antalia Turkish single-speaker scripted speech*, CC BY 4.0,
  https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus — one professional female voice
  actor who recorded under a commercial agreement and signed an addendum permitting public
  redistribution. Changes made: resampled 24 kHz → 22.05 kHz; transcripts converted to spoken form
  with trnorm; clips whose text disagreed with the corpus' ASR transcript removed; 20 clips held out.
- **Base checkpoint:** `lj-med_1000` (Piper "medium", trained from scratch for 1000 epochs by Bryce
  Beattie) from `rhasspy/piper-checkpoints` (MIT), trained on **LJ Speech** (Keith Ito & Linda
  Johnson, public domain).
- Training code: [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) (GPL-3.0) at a pinned commit;
  recipe and front-end: [ahenk-turkish-tts](https://github.com/ozguradmin/ahenk-turkish-tts) (Apache-2.0).

Bu ağırlıkları kullanırken yukarıdaki Antalia atfını koruyun (CC-BY-4.0 şartı).
Keep the Antalia attribution above when you use or redistribute these weights (CC-BY-4.0).

## Etik ve kabul edilebilir kullanım / Ethics and acceptable use

Veri setinin sahipleri, konuşmacının izninin şu anlayışla verildiğini belirtiyor ve buna uyulmasını
rica ediyor; biz de aynı ricayı bu model için tekrarlıyoruz:

- Bu modeli **konuşmacıyı veya herhangi bir gerçek kişiyi taklit etmek**, birine söylemediği bir
  şeyi söyletmek için kullanmayın.
- Üretilen sesi **gerçek bir insan kaydı gibi sunmayın**; dinleyicinin yanılabileceği yerlerde
  (telefon, IVR, duyuru, video) sentetik olduğunu açıkça belirtin.
- **Dolandırıcılık, sosyal mühendislik, siyasi otomatik arama** için kullanmayın.

The dataset owners state that the speaker consented on this understanding and ask users to honour
it; we ask the same for this model: no impersonation of the speaker or any real person, no passing
synthetic speech off as a genuine recording (disclose it where a listener could be misled), no
fraud, scams, social engineering or political robocalls. (Under CC-BY-4.0 these are requests, not
licence conditions; the Antalia 1 model of the dataset owners is Open RAIL-M, which makes them binding.)

## Sınırlamalar / Limitations

- VITS + espeak-ng: anlaşılır ve çok hızlı, ama büyük modeller (EMA Lightning, pocket-tts-turkish)
  kadar doğal değil; vurgu ve tonlama espeak-ng'nin Türkçe kurallarına bağlı.
- Yalnızca okuma konuşması, tek konuşmacı, çağrı merkezi / asistan üslubu ağırlıklı.
- Metin `trnorm` ile normalize edilmeden verilirse sayıları espeak-ng okur (farklı okunuşlar).
- Bilinen okunuş farkları: seslendirmen `TL`'yi "Türk lirası", kodlardaki tireyi "tire" diye okudu;
  trnorm "lira" yazıyor ve tireyi atlıyor (bkz. ahenk-turkish-tts `tts/DATA_REPORT.md`).
- espeak-ng (GPL-3.0) fonemleştirme için gerekir; piper-tts de GPL-3.0'dır. Ağırlıklar CC-BY-4.0;
  piper'ı ayrı bir servis olarak (Wyoming/HTTP) çalıştırmak ticari kullanımda en sorunsuz yoldur.

## İletişim / Contact

Sorular, izin ve kötüye kullanım bildirimleri / questions, permissions and abuse reports: destek@ozgurguler.tech
