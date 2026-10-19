---
language:
  - tr
license: cc-by-4.0
pipeline_tag: text-to-speech
base_model: canberkkkkkk/ema-lightning
base_model_relation: finetune
tags:
  - ema-lightning
  - flow-matching
  - on-device
  - cpu
  - synthetic-voice
  - designed-voice
  - distillation
datasets:
  - cloud0day3/alania-synthetic-speech-tr
---

# Ahenk Cep: küçük Türkçe metinden sese modeli (CPU)

Ahenk Cep, büyük [Ahenk](https://huggingface.co/ozgurguler/ahenk-tts-tr) modelinin sesinden öğrenen küçük bir öğrenci
modeldir: 8.6M parametre, 34 MB (fp32), EMA Lightning mimarisi, tek tasarlanmış kadın sesi (f022), 48 kHz. İşlemcide
2 iş parçacığıyla gerçek zamanın yaklaşık 0,2'si sürede konuşur. Ağırlıklar CC-BY-4.0.

**Tarayıcıda deneyin / try it in your browser:** [ozguradmin.github.io/ahenk-turkish-tts](https://ozguradmin.github.io/ahenk-turkish-tts/) (model sizin
cihazınızda çalışır, metin sunucuya gitmez / runs on your device).

**Belgeleri dinleyin / listen to whole documents:** [Ahenk Okur](https://ozguradmin.github.io/ahenk-cep-pdfreader/) PDF, Word, EPUB, metin ve fotoğrafı
(OCR ile) aynı modelle tarayıcıda seslendirir; uzun belgeleri parça parça üretir, MP3 ya da WAV indirilir / reads PDF,
Word, EPUB, text and photos (OCR) aloud in the browser with this model and downloads the result as MP3 or WAV.

> **Bu ses yapaydır.** f022, VoxCPM2 ile bir metin tarifinden tasarlanmış bir sestir; gerçek bir kişinin sesi ya da
> kaydı değildir. Ürettiğiniz sesi paylaşırken yapay olduğunu belirtin (AB Yapay Zekâ Yasası md. 50).
>
> **This voice is synthetic.** It is a designed voice, not a real person. Label what you generate as AI-generated.

[Demo Space](https://huggingface.co/spaces/ozgurguler/ahenk-cep) ·
[Büyük model: Ahenk](https://huggingface.co/ozgurguler/ahenk-tts-tr) ·
[Kod](https://github.com/ozguradmin/ahenk-turkish-tts/tree/main/tts/small) · [English](#english)

## Örnekler

Değerlendirme turundaki klipler (GPU'da üretildi; 16 Euler adımı, cfg 2.0, tohum 0, trnorm ön ucu). Yeni sentez
yapılmadı.

| Ne gösteriyor | Metin | Ses | Seçim |
|---|---|---|---|
| Soru | Bu ürünün garantisi var mı? | <audio controls src="https://huggingface.co/{{repo}}/resolve/main/samples/small_01_question_garantisi.wav"></audio> | Kör dinlemede dört sistem arasından dinleyici seçti. UTMOS yalnız 2.48: ölçü ile kulak burada ayrışıyor. |
| Soru | Saat kaçta buluşalım? | <audio controls src="https://huggingface.co/{{repo}}/resolve/main/samples/small_02_question_saat.wav"></audio> | Kör dinlemede dinleyici seçti. Hatasız, UTMOS 3.69. |
| Uzun cümle, eğitimde yok | Karşılaştırma isteyenlere eski ve yeni paketi yan yana anlatırız; farkı müşteri kendisi görür. Akşam olduğunda kısa bir değerlendirme yaparız: hangi sorular arttı, hangi açıklama iyi çalıştı, nerede zorlandık? | <audio controls src="https://huggingface.co/{{repo}}/resolve/main/samples/small_03_long_heldout583922.wav"></audio> | Dinleyici seçmedi (iki uzun cümlede de büyük modeli seçti). 28 kelime hatasız. Büyük modelin aynı cümlesi: [dinle](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_04_long_heldout583922.wav). |

## Hızlı başlangıç

```bash
pip install torch ema-lightning==1.1.2 "git+https://github.com/ozguradmin/ahenk-turkish-tts"
```

```python
import sys
import soundfile as sf
from huggingface_hub import snapshot_download

d = snapshot_download("{{repo}}", allow_patterns=["*.pt", "ahenk_cep.py"])
sys.path.insert(0, d)
from ahenk_cep import AhenkCep  # noqa: E402

tts = AhenkCep(f"{d}/student.pt", f"{d}/decoder.pt")  # CPU; device="cuda" de olur
audio, sr = tts.say("Randevunuz 14.10.2026 saat 09:30'da; ücret 1.250 TL.")
sf.write("randevu.wav", audio, sr)
audio, sr = tts.say("Ne kadar uğraştıysam da yetişemedim.", style="sad")
```

`say(text, style="neutral", steps=16, cfg=2.0, speed=1.0, seed=0)`. Üsluplar: `neutral`, `angry`, `calm`,
`cheerful`, `empathetic`, `sad`, `surprised`. Metin önce trnorm ile okunuş formuna çevrilir; siz çevirmeyin. Tarayıcıda ya da Node'da aynı çıktıyı veren
JavaScript sürümü: [trnorm-js](https://github.com/ozguradmin/trnorm-js) (`npm install @ozgurguler/trnorm`).
Aynı tohum CPU ve GPU'da farklı ses verir (gürültü üreteci cihaza bağlı); yukarıdaki örnekler GPU'dan.

## Ölçümler

Aynı düzenek, büyük modelle aynı: faster-whisper large-v3-turbo yargıç, `trnorm.normalize_for_eval`, UTMOS22 strong.
TR-Zor: 155 zor cümle. Ayrılmış: Antalia'nın 20 uzun cümlesi, eğitimde yok. Benzerlik: ECAPA kosinüsü, f022
referans klibine (60 cümle). Hepsi bir RTX 5060 dizüstüde ölçüldü.

| Model | Boyut | TR-Zor WER (hatasız) | Ayrılmış WER (hatasız) | UTMOS TR-Zor / ayrılmış | f022'ye benzerlik |
|---|---|---:|---:|---:|---:|
| **Ahenk Cep** | 8.6M | **2.32** (144/155) | 0.95 (17/20) | 3.44 / **3.18** | 0.62 |
| [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning) + trnorm | 8.6M | 2.43 (145/155) | 1.49 (16/20) | 3.36 / 2.95 | 0.22 |
| [Antalia-2 Mini](https://huggingface.co/cloud0day3/antalia-mini), kendi ön ucu | 7.6M | 3.30 (140/155) | 1.22 (16/20) | 3.23 / 2.94 | 0.24 |
| [Ahenk](https://huggingface.co/ozgurguler/ahenk-tts-tr) (öğretmen, GPU) | 2B + LoRA | 2.54 (141/155) | **0.82** (16/20) | **3.69** / 3.13 | 0.62 |

| Hız (RTF, düşük daha hızlı) | GPU | CPU, 2 iş parçacığı | CPU, 8 iş parçacığı |
|---|---:|---:|---:|
| Ahenk Cep, 16 adım | 0.025 | 0.20 | 0.14 |
| Ahenk Cep, 8 adım | 0.016 | 0.16 | 0.12 |
| EMA Lightning (4 adım) | 0.014 | 0.13 | 0.10 |
| Antalia-2 Mini | 0.020 | 0.08 | 0.05 |

- Kısa cümlede doğallık EMA ile başa baş, büyük modelin gerisinde (3.44'e karşı 3.69). Uzun cümlede EMA'dan iyi.
- Hızda EMA ve Antalia'nın gerisindeyiz: aynı işlemcide 0.20'ye karşı 0.13 ve 0.08.
- Soru tonlaması (TR-Soru, 35 satır, `score_intonation.py` geçme oranı): Ahenk Cep 0.60, öğretmen 0.56, Antalia 0.51,
  EMA 0.43. Sonu yükselen sorular hiçbir sistemde yükselmiyor (0/4).
- 8 adım: UTMOS 3.30 / 2.99, WER aynı aralıkta. 4 adım doğallıktan ~0.4 kaybettiriyor.
- Otomatik ölçümlerdir. 60 cümlelik alt kümede sistemler arası WER farkları birkaç kelimedir; beraberlik sayın.

**Kör dinleme (tek dinleyici, 10 cümle, dört sistem rastgele sırada, ses düzeyleri eşit):** öğretmen 8, Ahenk Cep 2,
EMA Lightning 0, Antalia-2 Mini 0. Ahenk Cep'in kazandığı iki cümle soruydu. Tek kişinin kulağıdır; MOS çalışması
yapılmadı.

## Nasıl yapıldı

- EMA Lightning'in `ema.pt` ağırlıklarından başlayıp büyük modelin f022 sesiyle okuduğu 4,5 saatlik derlemin
  latentleri üzerinde 20.000 adım ince ayar (4.164 klip). Sıfırdan eğitim aynı veriyle WER %13,4 verdi; kullanılamaz.
- Ağırlık karışımı (WiSE-FT): %75 ince ayarlı + %25 özgün `ema.pt`. Uzun ince ayar sesi f022'ye yaklaştırıp doğallığı
  düşürüyordu; karışım ikisini dengeler.
- Ses çözücüsünün (decoder) f022 sesine ince ayarı: 25 dakika; f022'ye benzerlik 0,48'den 0,62'ye.
- 8 üsluplu bir stil gömmesi eklendi (gülme eğitilmedi).
- Maliyet $0: hepsi tek bir dizüstü GPU'da (RTX 5060, 8 GB).

## Sınırlamalar

- **Üslup hafif:** "üzgün" 1,9 yarım ton pes ve %8 yavaş, "sakin" %8 yavaş; eğitim verisindeki fark daha büyük
  (~%20). Gülme yok.
- **Sonu yükselen sorular yükselmez** ("Yarın geliyorsun?"). "mı" ekleyin.
- **Telefon için henüz yavaş:** 2 iş parçacığında RTF ~0,2. Az adımlı damıtma (planlı) yapılmadı; şimdilik masaüstü
  ve sunucu CPU'su için.
- **Öğretmene benzerlik 0,68** (aynı cümlede, ECAPA); koyduğumuz 0,75 hedefini tutmadı.
- Uzun metinde doğallık büyük modelin gerisinde. Tek ses, yalnız Türkçe.
- **Köken:** `ema.pt` EMA Lightning'in açıklanmamış ~1000 saatlik verisiyle eğitildi (Apache-2.0). Ahenk Cep o
  ağırlıklardan türedi.

## Dosyalar

| Dosya | İçerik | Boyut | sha256 |
|---|---|---:|---|
| `student.pt` | akustik model (5.6M + stil gömmesi), fp32 | {{student_mb}} MB | `{{student_sha256}}` |
| `decoder.pt` | ses çözücüsü (3M), fp32, EMA Lightning `decoder.pt` biçiminde | {{decoder_mb}} MB | `{{decoder_sha256}}` |
| `ahenk_cep.py` | çıkarım kodu (Apache-2.0), [kaynak](https://github.com/ozguradmin/ahenk-turkish-tts/blob/main/tts/small/ahenk_cep.py) | {{code_kb}} KB | `{{code_sha256}}` |

`.pt` dosyaları `torch.load(..., weights_only=True)` ile yüklenir.

## Lisans

| | Lisans |
|---|---|
| EMA Lightning (taban ağırlıklar ve motor) | Apache-2.0 |
| Öğretmen sesi f022 / Ahenk (Alania verilerinden türedi) | CC BY 4.0 |
| **Ahenk Cep ağırlıkları** | **CC BY 4.0** |
| `ahenk_cep.py` | Apache-2.0 |

Atıf: Alania Turkish Synthetic Speech by PatientDesk AI
(https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr) ve Alania Turkish Domain Text by PatientDesk AI
(https://huggingface.co/datasets/cloud0day3/alania-domain-text-tr), licensed under CC BY 4.0, öğretmen Ahenk üzerinden.
EMA Lightning (https://huggingface.co/canberkkkkkk/ema-lightning), Apache-2.0.

Ağırlıkları kullanırken ya da dağıtırken şöyle atıf yapın: "Ahenk Cep, Ahenk
(https://github.com/ozguradmin/ahenk-turkish-tts), CC BY 4.0", ve yukarıdaki atıfları koruyun.

## Sorumlu kullanım

- Üretilen sesi gerçek bir insan kaydı gibi sunmayın; yapay olduğunu söyleyin (AB Yapay Zekâ Yasası md. 50). Filigran
  yok; isterseniz [AudioSeal](https://github.com/facebookresearch/audioseal) (MIT) ekleyin.
- Kimseyi taklit etmeyin.
- Üslupları gerçek bir insanın duygusu gibi sunup dinleyiciyi baskı altına almayın.
- Dolandırıcılık, sosyal mühendislik ve otomatik arama için kullanmayın.

## Sürümler

| Sürüm | Ne |
|---|---|
| v0.1 (Ekim 2026) | İlk deneme sürümü: WiSE α 0,25 öğrenci + 2.500. adım decoder |

## Atıf

```bibtex
@misc{guler2026ahenkcep,
  title  = {Ahenk Cep: a small on-device Turkish TTS student of Ahenk},
  author = "{Ahenk}",
  year   = {2026},
  url    = {https://huggingface.co/{{repo}}}
}
@misc{ema_lightning,
  title  = {EMA Lightning},
  author = "{canberkkkkkk}",
  url    = {https://huggingface.co/canberkkkkkk/ema-lightning}
}
@inproceedings{wortsman2022wiseft,
  title     = {Robust fine-tuning of zero-shot models},
  author    = {Wortsman, Mitchell and others},
  booktitle = {CVPR},
  year      = {2022}
}
```

İletişim: destek@ozgurguler.tech

---

## English

Ahenk Cep is a small student of the [Ahenk](https://huggingface.co/ozgurguler/ahenk-tts-tr) model: 8.6M parameters,
34 MB (fp32), EMA Lightning architecture, one designed female voice (f022, no real person), 48 kHz. On 2 CPU threads
it runs at about 0.2 of real time. Weights CC BY 4.0.

**Quick start.** The code above: `AhenkCep(student, decoder).say(text, style="neutral")` returns float32 audio and the
sample rate. trnorm turns the written text into its spoken form inside `say()`; [trnorm-js](https://github.com/ozguradmin/trnorm-js) gives the same output in
JavaScript (`npm install @ozgurguler/trnorm`). The same seed gives different audio
on CPU and GPU; the samples above were made on a GPU.

**Results** (same harness as the big model, measured on one RTX 5060 laptop). TR-Zor WER 2.32 % vs EMA Lightning 2.43
and Antalia-2 Mini 3.30; similarity to the f022 voice 0.62, the same as the teacher. Predicted naturalness is level
with EMA on short sentences and below the teacher (3.44 vs 3.69). It is slower than EMA and Antalia on CPU (RTF 0.20
vs 0.13 and 0.08 at 2 threads). Question intonation pass rate 0.60, the best of the four. Blind listening, one
listener, 10 sentences: teacher 8, Ahenk Cep 2 (both questions), EMA 0, Antalia 0. Automatic metrics and one
listener; no MOS study.

**How it was made.** Fine-tuned from EMA Lightning's `ema.pt` on 4.5 h of f022 latents (training from scratch gave
13.4 % WER), mixed 75/25 with the original weights (WiSE-FT), plus a 25-minute decoder fine-tune (similarity 0.48 →
0.62). Cost $0, one laptop GPU.

**Limitations.** Styles are weak (sad: −1.9 semitones, 8 % slower) and there is no laughter. Questions that should rise
at the end do not. Not distilled for phones yet (RTF ~0.2 on 2 threads). Similarity to the teacher is 0.68, below our
0.75 target. Long-text naturalness is below the big model. `ema.pt` was trained by its authors on ~1000 h of
undisclosed data (Apache-2.0); Ahenk Cep derives from it.

**Responsible use.** Label the output as synthetic (EU AI Act Art. 50); no watermark is built in. No impersonation,
no deceptive emotion, no scams or robocalls.

**Licence.** Weights CC BY 4.0 (EMA Lightning Apache-2.0 + the CC BY Alania data behind the teacher; keep the
attribution above). Credit it as "Ahenk Cep, Ahenk (https://github.com/ozguradmin/ahenk-turkish-tts), CC BY
4.0". Code Apache-2.0. Contact: destek@ozgurguler.tech
