---
language:
  - tr
license: cc-by-4.0
base_model: openbmb/VoxCPM2
base_model_relation: adapter
pipeline_tag: text-to-speech
tags:
  - voxcpm
  - lora
  - turkish
  - synthetic-voice
datasets:
  - cloud0day3/antalia-voice-corpus
---

# {{name}} — Türkçe ses (VoxCPM2 + Antalia LoRA)

> **Bu bir sentetik (yapay) sestir.** Gerçek bir seslendirme sanatçısının izinli kayıtlarıyla
> eğitildi. Dinleyicinin yanılabileceği her yerde sesin yapay olduğunu belirtin; bu sesi
> sanatçıyı ya da herhangi bir gerçek kişiyi taklit etmek için **asla** kullanmayın.
>
> **This is a synthetic voice.** It was trained on consented recordings of a real voice actor.
> Disclose that it is synthetic wherever a listener could be misled, and **never** use it to
> impersonate the actor or any real person.

**{{name}}**, [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (OpenBMB, 2B parametre, 48 kHz) için bir LoRA
adaptörüdür. Tek bir profesyonel kadın seslendirmenin izinli stüdyo kayıtlarıyla
([Antalia](https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus), 4.8 saat) ince ayarlandı. Metin ön ucu
[trnorm](https://github.com/ozguradmin/ahenk-turkish-tts): sayı, tarih, saat, para, kısaltma ve telefon numarası gibi
yazıları okunuşa çevirir.

**{{name}}** is a LoRA adapter for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (OpenBMB, 2B parameters,
48 kHz), fine-tuned on 4.8 hours of consented studio recordings of one professional female voice actor (Antalia).
Text front-end: [trnorm](https://github.com/ozguradmin/ahenk-turkish-tts), which turns numbers, dates, times, money,
abbreviations and phone numbers into their spoken Turkish form.

| | |
|---|---|
| Temel model / Base model | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), Apache-2.0. Bu repo yalnızca LoRA ağırlıklarını içerir / this repo holds only the LoRA weights |
| Adaptör / Adapter | LoRA r={{lora_r}}, alpha={{lora_alpha}}; {{lora_targets}}; {{lora_params}} parametre / parameters, {{lora_mb}} MB |
| Kontrol noktası / Checkpoint | `{{exp}}/{{checkpoint}}` (adım / step {{step}}); `lora_weights.safetensors` sha256 `{{lora_sha256}}` |
| Dil / Language | Türkçe, tek kadın ses / Turkish, one female voice |
| Çıkış / Output | 48 kHz, mono |
| Lisans / Licence | LoRA ağırlıkları **CC-BY-4.0** (Antalia verisinden türedi, atıf zorunlu); temel model VoxCPM2 **Apache-2.0** / LoRA weights CC-BY-4.0 (derived from Antalia, attribution required); base model Apache-2.0 |
| Ön uç / Front-end | metni `trnorm.normalize` ile verin, VoxCPM'in kendi normalizer'ı kapalı (`normalize=False`) / feed text through `trnorm.normalize` and keep VoxCPM's own normalizer off |

## Kullanım / Usage

```bash
# VoxCPM: eğitimde ve ölçümde kullanılan commit / the commit used for training and evaluation
pip install "voxcpm @ git+https://github.com/OpenBMB/VoxCPM@{{voxcpm_commit}}"
# trnorm: Türkçe yazı -> okunuş / Turkish written -> spoken form
pip install "git+https://github.com/ozguradmin/ahenk-turkish-tts"
```

```python
import json
from pathlib import Path

import soundfile as sf
from huggingface_hub import snapshot_download
from trnorm import normalize
from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

# lora_weights.safetensors + lora_config.json (r, alpha, hedef katmanlar / target layers)
lora_dir = snapshot_download("{{repo}}", allow_patterns=["lora_*"])
lora_cfg = json.loads(Path(lora_dir, "lora_config.json").read_text(encoding="utf-8"))["lora_config"]
tts = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False,
                             lora_config=LoRAConfig(**lora_cfg), lora_weights_path=lora_dir)

# trnorm yazıyı okunuşa çevirir; LoRA bu okunuş formuyla eğitildi. VoxCPM'in normalizer'ı kapalı kalmalı.
# trnorm gives the spoken form the LoRA was trained on; VoxCPM's own normalizer must stay off (normalize=False).
text = normalize("Randevunuz 14.10.2026 saat 09:30'da; ücret 1.250 TL.")
wav = tts.generate(text=text, cfg_value={{cfg}}, inference_timesteps={{timesteps}}, normalize=False)
sf.write("cikti.wav", wav, tts.tts_model.sample_rate)  # 48 kHz
```

- `cfg_value={{cfg}}`, `inference_timesteps={{timesteps}}`: aşağıdaki ölçümlerin ayarları / the settings of the
  evaluation below. Ölçümler `optimize=False` ile alındı; `optimize=True` torch.compile kullanır, etkisini
  ölçmedik / measured with `optimize=False`; `optimize=True` uses torch.compile, which we did not measure.
- PyPI'deki `voxcpm` 2.0.3 de bu kodla çalışır: `lora_config` açıkça veriliyor (2.0.3 `lora_config.json`'u
  kendisi okumaz, varsayılan r=8 ile yüklemeye çalışır). 2.0.3'te `generate()` `seed` almaz; yukarıdaki commit
  `seed=0` gibi sabit bir tohum alır / PyPI `voxcpm` 2.0.3 also runs this code because `lora_config` is passed
  explicitly (2.0.3 does not read `lora_config.json` itself and would assume r=8); its `generate()` has no `seed`,
  the pinned commit does.

## Ölçümler / Evaluation

Yargıç / judge: **faster-whisper large-v3-turbo** (beam 5, `language="tr"`) sesi geri okur; referans ve hipotez
`trnorm.normalize_for_eval` ile normalize edilir. WER %, düşük daha iyi / lower is better.
**TR-Zor**: sayı, tarih, saat, para, yüzde, kısaltma, telefon numarası gibi zor yazılar içeren 155 cümle
([benchmarks/tts](https://github.com/ozguradmin/ahenk-turkish-tts/tree/main/benchmarks/tts)); metin modele trnorm'dan
geçerek verilir / the text is fed through trnorm. **Ayrılmış / Held-out**: Antalia'nın eğitimde hiç kullanılmamış
20 cümlesi / 20 Antalia sentences never used in training. **Hatasız / error-free**: yargıcın tek kelimeyi bile
yanlış okumadığı cümleler / sentences without a single word error. **UTMOS**: UTMOS22 strong
([SpeechMOS](https://github.com/tarepan/SpeechMOS)), 1–5, yüksek daha iyi / higher is better.

> UTMOS doğallığı **tahmin eder** (bir modelin tahmini, insan puanı değil), WER ise **anlaşılırlığı** ölçer. /
> UTMOS **predicts** naturalness (a model's estimate, not human ratings); WER measures **intelligibility**.

| Model | Çıkarım / Inference (timesteps / cfg) | TR-Zor WER | TR-Zor hatasız / error-free | Ayrılmış / Held-out WER | Ayrılmış hatasız / error-free | UTMOS TR-Zor | UTMOS ayrılmış / held-out | RTF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VoxCPM2, ince ayarsız: bir Antalia klibini klonlayarak / not fine-tuned, cloning an Antalia clip | — | 5.96 | — | 3.80 | — | — | — | — |
{{results_rows}}

RTF: sentez süresi / ses süresi, {{eval_gpu}} üzerinde, cümle cümle sentez; 1'in altı gerçek zamandan hızlı. Aynı
ayarla farklı konteynerlerde ölçülen RTF belirgin değişebilir / synthesis time over audio duration on {{eval_gpu}},
one sentence at a time, below 1 is faster than real time; the same settings can measure quite differently from one
container to the next. "—": ölçülmedi / not measured.
{{eval_note}}

### TR-Zor kategorileri / categories

WER % (hatasız / toplam cümle) / WER % (error-free / total sentences).

{{category_table}}

## Örnekler / Samples

{{samples}}

## Eğitim / Training

- Temel model / base model: `openbmb/VoxCPM2`; resmi ince ayar betiği / official fine-tuning script
  [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) `{{voxcpm_commit}}`.
- Veri / data: Antalia, 1.026 eğitim + 20 ayrılmış klip, 4.8 saat, 16 kHz; metinler trnorm'un okunuş formunda /
  1,026 training + 20 held-out clips, 4.8 h, 16 kHz; transcripts in trnorm's spoken form.
- LoRA r={{lora_r}}, alpha={{lora_alpha}}, hedef / targets: {{lora_targets}}.
- lr 1e-4, batch 16, 1.000 adım / steps; her 100 adımda bir kontrol noktası, WER'e göre en iyisi 600. adım /
  a checkpoint every 100 steps, the best by WER at step 600.
- Donanım / hardware: A100 80 GB, 1.94 s/adım / s/step.
- Tarif / recipe: [ahenk-turkish-tts/tts/voxcpm](https://github.com/ozguradmin/ahenk-turkish-tts/tree/main/tts/voxcpm) (Apache-2.0).

## Eğitim verisi ve atıf / Training data and attribution

- {{attribution}}
  — tek bir profesyonel kadın seslendirmen; ticari bir sözleşmeyle kayıt yaptı ve herkese açık yeniden dağıtıma izin
  veren bir ek protokol imzaladı / one professional female voice actor who recorded under a commercial agreement and
  signed an addendum permitting public redistribution. Yapılan değişiklikler / changes made: 24 kHz → 16 kHz
  yeniden örnekleme / resampled; metinler trnorm ile okunuş formuna çevrildi / transcripts converted to spoken form
  with trnorm; metni korpusun ASR dökümüyle uyuşmayan klipler çıkarıldı / clips whose text disagreed with the corpus'
  ASR transcript removed; 20 klip ayrıldı / 20 clips held out.
- Temel model / base model: [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (OpenBMB), Apache-2.0.
- Ön uç ve tarif / front-end and recipe: [ahenk-turkish-tts](https://github.com/ozguradmin/ahenk-turkish-tts) (Apache-2.0).

Bu ağırlıkları kullanırken veya dağıtırken yukarıdaki Antalia atfını koruyun (CC-BY-4.0 şartı).
Keep the Antalia attribution above when you use or redistribute these weights (CC-BY-4.0).

## Etik ve kabul edilebilir kullanım / Ethics and acceptable use

Veri setinin sahipleri, konuşmacının izninin şu anlayışla verildiğini belirtiyor ve buna uyulmasını rica ediyor;
biz de aynı ricayı bu model için tekrarlıyoruz:

- Bu modeli **konuşmacıyı veya herhangi bir gerçek kişiyi taklit etmek**, birine söylemediği bir şeyi söyletmek için
  kullanmayın. VoxCPM2 ses klonlayabilir (`reference_wav_path`); başkasının sesini izni olmadan klonlamayın.
- Üretilen sesi **gerçek bir insan kaydı gibi sunmayın**; dinleyicinin yanılabileceği yerlerde (telefon, IVR, duyuru,
  video) sentetik olduğunu açıkça belirtin.
- **Dolandırıcılık, sosyal mühendislik, siyasi otomatik arama** için kullanmayın.

The dataset owners state that the speaker consented on this understanding and ask users to honour it; we ask the
same for this model: no impersonation of the speaker or any real person (VoxCPM2 can clone voices from a reference
clip; do not clone anyone without their consent), no passing synthetic speech off as a genuine recording (disclose
it where a listener could be misled), no fraud, scams, social engineering or political robocalls. (Under
CC-BY-4.0 these are requests, not licence conditions.)

## Sınırlamalar / Limitations

- Tek kadın ses, yalnızca Türkçe; çağrı merkezi / asistan üslubunda okuma konuşması ağırlıklı / a single female voice,
  Turkish only, mostly read speech in a call-centre / assistant style.
- Yalnızca 4.8 saat veriyle eğitildi; zayıf kategoriler için yukarıdaki tabloya bakın / trained on only 4.8 hours;
  see the category table for the weak spots.
- WER bir ASR yargıcıyla ölçüldü (faster-whisper large-v3-turbo); yargıcın kendi hataları sonuca karışır ve sistematik
  bir insan dinleme testi (MOS) yapılmadı / WER is measured by an ASR judge whose own errors are part of the score;
  no systematic human listening test (MOS) was run.
- GPU gerekir: yukarıdaki ayarlarla RTF {{rtf}} ({{eval_gpu}}); 1'in üstü gerçek zamandan yavaş demektir, daha
  az adım (`inference_timesteps`) daha hızlıdır (tablodaki RTF'lere bakın). CPU'da ölçülmedi; 2B parametreli bir
  modelden CPU'da gerçek zamanlı hız beklemeyin / needs a GPU: RTF {{rtf}} on {{eval_gpu}} with the settings above
  (above 1 is slower than real time; fewer `inference_timesteps` are faster, see the RTF column); not measured on
  CPU, do not expect real time there from a 2B model.
- Metin `trnorm` ile normalize edilmeden verilirse sayılar, tarihler ve kısaltmalar yanlış okunabilir / without
  `trnorm`, numbers, dates and abbreviations may be misread.
- Bilinen okunuş farkları: seslendirmen `TL`'yi "Türk lirası", kodlardaki tireyi "tire" diye okudu; trnorm "lira"
  yazıyor ve tireyi atlıyor / known reading differences: the actor read `TL` as "Türk lirası" and the hyphen in codes
  as "tire"; trnorm writes "lira" and drops the hyphen.

## İletişim / Contact

Sorular, izin ve kötüye kullanım bildirimleri / questions, permissions and abuse reports: destek@ozgurguler.tech
