---
language:
  - tr
license: cc-by-4.0
base_model: openbmb/VoxCPM2
base_model_relation: adapter
pipeline_tag: text-to-speech
library_name: voxcpm
tags:
  - voxcpm
  - lora
  - turkish
  - synthetic-voice
  - designed-voice
  - emotional-tts
datasets:
  - cloud0day3/alania-synthetic-speech-tr
  - cloud0day3/alania-domain-text-tr
---

# {{name}}: Türkçe metinden sese (VoxCPM2 + LoRA)

{{name}}, [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) için Türkçe bir LoRA adaptörüdür: 2B taban model +
{{lora_params}} parametrelik LoRA, 48 kHz, tek bir tasarlanmış kadın sesi (f022), 6 duygu ve gülme, ağırlıklar
CC-BY-4.0. GPU gerektirir.

> **Bu ses yapaydır.** f022, VoxCPM2 ile İngilizce bir metin tarifinden tasarlandı; gerçek bir kişinin sesi ya da
> kaydı değildir. Eğitim seslerinin tamamı yapay zekâ ile üretildi. Ürettiğiniz sesi paylaşırken yapay olduğunu
> belirtin (AB Yapay Zekâ Yasası md. 50).
>
> **This voice is synthetic.** It was designed from a text description; it is no real person. Label what you
> generate as AI-generated.

[Küçük model: Ahenk Cep](https://huggingface.co/ozgurguler/ahenk-cep-tr) ·
[Kod ve ölçüm düzeneği](https://github.com/ozguradmin/ahenk-turkish-tts) · [English](#english)


Küçük model Ahenk Cep'i tarayıcıda deneyin / try the small model Ahenk Cep in your browser: [ahenk-tts.pages.dev](https://ahenk-tts.pages.dev).

## Örnekler

Hepsi yayınlanan kontrol noktasından (`{{exp}}/{{checkpoint}}`), {{timesteps}} adım, cfg {{cfg}}, tohum 0 ya da 1,
trnorm ön ucu, sentez sonrası `pause_clean`. Yeni sentez yapılmadı; değerlendirme ve dinleme turlarındaki klipler.

{{extra_samples_table}}

"Dinleyici" satırları kör dinlemede tek bir anadil dinleyicinin seçtiği kliplerdir (aşağıda). "Ölçüm" satırları
otomatik ölçüyle seçildi; dinleyici bunları ayrıca seçmedi. Soru çiftini art arda dinleyin: aynı cümle, bir kez
noktayla, bir kez "mı?" ile.

## Hızlı başlangıç

```bash
pip install "voxcpm @ git+https://github.com/OpenBMB/VoxCPM@{{voxcpm_commit}}" soundfile scipy
pip install "git+https://github.com/ozguradmin/ahenk-turkish-tts"   # trnorm
```

```python
import json, sys
from pathlib import Path
import soundfile as sf
from huggingface_hub import snapshot_download
from trnorm import normalize
from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

repo_dir = snapshot_download("{{repo}}", allow_patterns=["lora_*", "*.py", "instructions_tr.json"])
sys.path.insert(0, repo_dir)
from instruction_pick import compose, pick  # noqa: E402  (stdlib)
from pause_clean import clean  # noqa: E402  (numpy + scipy)

lora_cfg = json.loads(Path(repo_dir, "lora_config.json").read_text(encoding="utf-8"))["lora_config"]
tts = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False,
                             lora_config=LoRAConfig(**lora_cfg), lora_weights_path=repo_dir)
SR = tts.tts_model.sample_rate  # 48 kHz


def say(text: str, emotion: str = "", path: str = "cikti.wav", prefer: str = "auto") -> None:
    instruction, model_text, _ = pick(emotion, normalize(text), prefer)
    wav = tts.generate(text=compose(instruction, model_text), cfg_value={{cfg}}, inference_timesteps={{timesteps}}, normalize=False)
    sf.write(path, clean(wav, SR)[0], SR)


say("Randevunuz 14.10.2026 saat 09:30'da; ücret 1.250 TL.", path="duz.wav")
say("Yeter artık! Bu sorun hâlâ çözülmedi!", "angry", "kizgin.wav")
```

Referans klip vermeyin: ses LoRA'nın içinde. PyPI'deki `voxcpm` 2.0.3 de çalışır (`lora_config` açıkça veriliyor),
ama onun `generate()`'i `seed` almaz.

## Duygu ve talimat

`say()`'in ikinci argümanı bir duygu adıdır (Türkçe adı da olur) ya da kendi talimatınız. `pick()` her duygu için
eğitimdeki tek bir ifadeyi kullanır ve dili seçer: metnin ilk iki kelimesinden biri "c" ile başlıyorsa Türkçe talimat,
yoksa İngilizce. Nedeni: İngilizce talimat duyguyu daha güçlü verir ama baştaki "c"yi İngilizce okutur
("Vay canına" → "vey kenına"); Türkçe talimat "c"yi doğru okutur, duyguyu daha hafif verir. `prefer="tr"` ya da
`"en"` dili zorlar.

| Duygu | `pick()` adı | İngilizce talimat | Türkçe talimat |
|---|---|---|---|
| kızgın | `angry` | Angry tone, voice getting louder. | Kızgın bir ses tonuyla, sesi giderek yükselerek. |
| neşeli | `cheerful` | Cheerful and warm, smiling while speaking. | Neşeli ve sıcak, konuşurken gülümseyerek. |
| üzgün | `sad` | Sad and quiet, with a heavy heart. | Üzgün ve sessiz, içi burkularak. |
| sakin | `calm` | Calm, slow and soothing. | Sakin, yavaş ve yatıştırıcı. |
| şaşkın | `surprised` | Astonished, with rising intonation. | Şaşkın, yükselen bir tonlamayla. |
| empatik | `empathetic` | Warm and compassionate, as if comforting someone. | Sıcak ve şefkatli, birini teselli eder gibi. |
| gülerek | `laughing` | Laughing heartily while speaking. | Konuşurken kahkahalarla gülerek. |

Gülme için "haha" metinde kalmalı; her tohumda gerçek gülme çıkmayabilir. Kural, iki kör dinleme turundaki 20
kararın 18'iyle uyuşuyor (tek dinleyici). Eğitimdeki bütün ifadeler:

<details><summary>Eğitimde kullanılan talimat ifadeleri / all training wordings</summary>

{{instructions_tr_table}}

</details>

## Önerilen ayarlar

- `inference_timesteps={{timesteps}}`, `cfg_value={{cfg}}`: aşağıdaki ölçümlerin ayarı. 10 adım daha hızlıdır, ölçülmüş
  tam bir karşılaştırma yok.
- `normalize=False` ve önce `trnorm.normalize`: model trnorm'un okunuş formuyla eğitildi. VoxCPM'in kendi normalizer'ı
  sayıları ve ekleri bozar.
- `pause_clean.clean`: duraklardaki nefes sesini kısar (en çok 25 dB), konuşmaya dokunmaz. Kör A/B'de (4 uzun cümle)
  temizlenmiş sürümde nefes 4/4 daha az bulundu, doğallık 3'ünde aynı, 1'inde daha iyi.
- Uzun metni cümlelere bölüp tek tek sentezleyin.

## Ölçümler

TR-Zor: sayı, tarih, saat, para, kısaltma ve telefon içeren 155 zor cümle. Ayrılmış: Antalia veri setinin 20 uzun
cümlesi (12-19 sn), eğitimde yok. Yargıç faster-whisper large-v3-turbo (beam 5); referans ve çıktı
`trnorm.normalize_for_eval` ile normalize edildi. WER %, düşük iyi; "hatasız" tek kelime hatası olmayan cümleler.
UTMOS22 strong doğallığı **tahmin eder** (1-5). RTF: sentez süresi / ses süresi, {{eval_gpu}}, cümle cümle.

| Model | Adım / cfg | TR-Zor WER | TR-Zor hatasız | Ayrılmış WER | Ayrılmış hatasız | UTMOS TR-Zor | UTMOS ayrılmış | RTF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{{results_rows}}
| [Ahenk Cep](https://huggingface.co/ozgurguler/ahenk-cep-tr)† (8.6M) | 16 / 2.0 | 2.32 | 144/155 | 0.95 | 17/20 | 3.44 | 3.18 | - |
| [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning)† (8.6M) + trnorm | 4 / - | 2.43 | 145/155 | 1.49 | 16/20 | 3.36 | 2.95 | - |
| [Antalia-2 Mini](https://huggingface.co/cloud0day3/antalia-mini)† (7.6M), kendi ön ucu | - | 3.30 | 140/155 | 1.22 | 16/20 | 3.23 | 2.94 | - |

† Küçük modeller ayrı bir oturumda (RTX 5060 dizüstü, aynı yargıç ve normalizasyon) ölçüldü. Aynı oturumda {{name}}'in
klipleri yeniden puanlandı: 2.54 / 0.82 WER, UTMOS 3.69 / 3.13, yani tablodaki satırla ±0.01 içinde. Trendyol-TTS
kendi sesiyle konuşur; UTMOS farkının bir kısmı ses ve kayıttan gelebilir.

- **Önde olduğumuz yer:** zor metni doğru okuma (TR-Zor 2.54'e karşı Trendyol 3.98).
- **Geride olduğumuz yerler:** tahmini doğallık, özellikle uzun cümlelerde (UTMOS 3.14'e karşı 3.73); ayrılmış
  WER (0.82'ye karşı 0.68); hız (RTF {{rtf}}'e karşı 1.37, aynı GPU türü).
- Bunlar otomatik ölçümlerdir. MOS çalışması yapılmadı.

{{pause_clean_note}}

<details><summary>TR-Soru (soru tonlaması) ve TR-Zor kategorileri</summary>

TR-Soru: 35 satır (mı soruları, "değil mi", soru sözcüklü ve sonu yükselen sorular, eşli düz cümleler);
`score_intonation.py` perde tepesinin yerini ve yüksekliğini ölçer. Eşler: sorunun tepesi eşli düz cümleninkinden
yüksek olan çiftler. AUC: mı sorularının tepesi düz cümlelerden ne kadar ayrışıyor (0.5 fark yok, 1 tam ayrım).

{{intonation_table}}

TR-Zor kategorileri, WER % (hatasız / toplam):

{{category_table}}

</details>

## İnsan dinlemesi

Hepsi kör, hepsi **tek** anadil dinleyiciyle (proje sahibi), her cümlede en iyi bulunan klip seçildi. Bunlar işarettir,
istatistiksel sonuç değildir.

- **Büyük ve küçük modeller, 10 cümle** (düz, sayılı, soru, uzun; dört sistem rastgele sırada, ses düzeyi eşitlenmiş):
  {{name}} 8, Ahenk Cep 2, EMA Lightning 0, Antalia-2 Mini 0. {{name}}'in klipleri yayınlanan kontrol noktasındandı.
- **4. tur, 20 cümle**, yayınlanan kontrol noktası eğitilmeden önce: ince ayarın başladığı aşama 2 / 1500. adım 8,
  önceki sürüm 6, aşama 2 / 2000. adım 4, EMA Lightning 2. EMA'nın kazandığı iki cümle bizim "c"yi bozduğumuz
  cümlelerdi.
- **Talimat dili, iki tur, 20 karar:** saf duyguda (kızgın, neşeli, üzgün) İngilizce talimat 6/6; ilk iki kelimeden
  biri "c" ile başlayan cümlelerde Türkçe talimat 10/10. `pick()` bu sonuçtan çıktı.
- Yayınlanan modelin İngilizce talimatlı duyguları ayrıca kulakla puanlanmadı; örneklerdeki duygu klipleri iki tohum
  arasından seçildi.

## Bilinen sorunlar

- **İngilizce talimattan sonra "c":** baştaki "c" /k/, /s/ ya da /z/ gibi okunabilir ("Cam" → "Kam", "Cidden" →
  "Zidden"). IPA probunda c doğru: İngilizce talimatla 1/14, Türkçe talimatla 13/14. `pick()` ilk iki kelimeye bakar;
  ileride yanlış çıkan bir "c" için `prefer="tr"`.
- **Sonu yükselen sorular yükselmez:** "Yarın geliyorsun?" gibi mı'sız sorular düz cümle gibi biter (0/4). mı soruları
  ve soru sözcüklü sorular doğru. Eğitim verisindeki bir eksik; geçici çözüm: "mı" ekleyin.
- **"Telefon gibi" tını:** bazı duygulu kliplerde ses parlak ve cızırtılı (1-5 kHz şişmesi). Probda İngilizce
  talimatla 7/20, Türkçe talimatla 4/20 klipte. Başka tohum deneyin.
- **Türkçe talimatla duygu hafif:** probda kızgın perdesi 312 Hz (İngilizce 325), üzgün 190 Hz (223). Türkçe talimatla
  "Cam" iki klipte de "Cem" gibi çıktı.
- **Nefes:** duraklarda nefes sesi kalır, çoğu taban modelden. `pause_clean` kısar.
- **Harf harf okunan kısaltmalar ve telefon numaraları** en zayıf kategoriler (WER 8.05 ve 7.56; kategori tablosu).
- **GPU gerekir:** {{eval_gpu}} üzerinde RTF {{rtf}}, yani gerçek zamandan yavaş. CPU'da ölçülmedi. CPU için:
  [Ahenk Cep](https://huggingface.co/ozgurguler/ahenk-cep-tr).

## Model ayrıntıları

| | |
|---|---|
| Taban model | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), 2B, Apache-2.0 |
| Adaptör | LoRA r={{lora_r}}, alpha={{lora_alpha}}; {{lora_targets}}; {{lora_params}} parametre, {{lora_mb}} MB |
| Kontrol noktası | `{{exp}}/{{checkpoint}}` (adım {{step}}): aşama 2'nin 1500. adımından başlayan, her talimatlı satırı bir kez İngilizce bir kez Türkçe talimatla gören ince ayar. İngilizce talimatı unutmadan Türkçe talimatı öğrendiği için seçildi; düz okumada 1500. adımla aynı ya da biraz iyi (TR-Zor 2.54 / 2.87). |
| `lora_weights.safetensors` sha256 | `{{lora_sha256}}` |
| VoxCPM sürümü | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) `{{voxcpm_commit}}` (eğitim ve ölçüm) |
| Eğitim | Aşama 1: Alania'dan 25 sa, 23.706 klip, düz okuma; 2.000 adım. Aşama 2: f022 sesiyle 4.274 klip, 4,58 sa (düz, sayılı, uzun, soru, duygu, gülme); 1500. adım dinlemeyle seçildi. İnce ayar: 4.916 satır, lr 5e-5, 500 adım. Hepsi A100 80 GB. |
| Ses tasarımı | Referanssız VoxCPM2 tarifi: *"Female speaker, in their fifties, a counsellor; low-pitched and velvety; patient and polite delivery at a measured pace; recorded close to the microphone in a quiet studio."* ~150 tarif × 2 tohum arasından UTMOS, Whisper CER ve iki kör dinleme turuyla seçildi. Tarif kimseyi anmaz. |

Paketteki yardımcı dosyalar ([ahenk-turkish-tts/tts/voxcpm](https://github.com/ozguradmin/ahenk-turkish-tts/tree/main/tts/voxcpm)'ten olduğu gibi, Apache-2.0):

{{extra_files}}

## Eğitim verisi ve lisans

| | Lisans |
|---|---|
| Taban model VoxCPM2 (OpenBMB) | Apache-2.0 |
| Alania Turkish Synthetic Speech by PatientDesk AI (aşama 1; tamamı sentetik konuşma) | CC BY 4.0 |
| Alania Turkish Domain Text by PatientDesk AI (aşama 2 metinleri) | CC BY 4.0 |
| Soru, duygu, ünlem ve gülme metinleri, Türkçe talimatlar (bu proje) | Apache-2.0 |
| **Bu LoRA'nın ağırlıkları** | **CC BY 4.0** |
| `instruction_pick.py`, `pause_clean.py`, `onset.py`, `instructions_tr.json` | Apache-2.0 |

Ağırlıkları kullanırken ya da dağıtırken CC BY 4.0 atıf ister. Şöyle yazın:
"Ahenk (https://github.com/ozguradmin/ahenk-turkish-tts), CC BY 4.0" ve aşağıdaki Alania atıflarını koruyun:

- Alania Turkish Synthetic Speech by PatientDesk AI (https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr),
  licensed under CC BY 4.0. Includes Turkish web text from FineWeb-2
  (https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), used under ODC-By 1.0 and subject to Common Crawl's terms
  of use. Değişiklik: filtrelenmiş alt küme, 48 → 16 kHz.
- Alania Turkish Domain Text by PatientDesk AI (https://huggingface.co/datasets/cloud0day3/alania-domain-text-tr),
  licensed under CC BY 4.0. Değişiklik: filtrelendi, trnorm ile okunuş formuna çevrildi, VoxCPM2 f022 sesiyle
  seslendirildi.
- Voice designed from a text description with VoxCPM2 (OpenBMB, https://huggingface.co/openbmb/VoxCPM2, Apache-2.0);
  no real person's voice.

Bu sürümün eğitiminde Antalia sesi ya da metni yok; yalnız ayrılmış ölçüm cümlelerinin metni Antalia veri setinden
(Patientdesk.ai, CC BY 4.0).

## Sorumlu kullanım

- Üretilen sesi gerçek bir insan kaydı gibi sunmayın; telefon, IVR, duyuru ve videoda yapay olduğunu söyleyin
  (AB Yapay Zekâ Yasası md. 50). Bu modelde filigran yok; makinece okunur bir işaret için
  [AudioSeal](https://github.com/facebookresearch/audioseal) (MIT) eklenebilir.
- Kimseyi taklit etmeyin. VoxCPM2 referans klipten ses klonlayabilir; kimsenin sesini izni olmadan klonlamayın.
- Üretilen öfke, üzüntü, şefkat ya da gülmeyi gerçek bir insanın tepkisi gibi kullanıp dinleyiciyi baskı altına
  almayın (sahte acil durum, sahte yakın).
- Dolandırıcılık, sosyal mühendislik ve siyasi otomatik arama için kullanmayın.

CC BY 4.0 altında bunlar lisans şartı değil, ricadır; AB Yapay Zekâ Yasası yükümlülükleri ayrıca geçerlidir.

## Sürümler

| Sürüm | Kontrol noktası | Ne değişti |
|---|---|---|
| v1 (Ekim 2026) | `{{exp}}/{{checkpoint}}` | İlk açık sürüm: Türkçe ve İngilizce talimat, `instruction_pick`, `pause_clean` |

## Atıf

```bibtex
@misc{guler2026ahenk,
  title  = {Ahenk: a Turkish LoRA for VoxCPM2 with a designed synthetic voice},
  author = "{Ahenk}",
  year   = {2026},
  url    = {https://huggingface.co/{{repo}}}
}
@misc{voxcpm2,
  title  = {VoxCPM2},
  author = "{OpenBMB}",
  year   = {2026},
  url    = {https://huggingface.co/openbmb/VoxCPM2}
}
@misc{alania_tr,
  title  = {Alania Turkish Synthetic Speech and Alania Turkish Domain Text},
  author = "{PatientDesk AI}",
  url    = {https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr}
}
```

## İletişim

Sorular, izin ve kötüye kullanım bildirimleri: destek@ozgurguler.tech

---

## English

{{name}} is a Turkish LoRA adapter for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (2B base + {{lora_params}}
LoRA, 48 kHz) that speaks in one designed female voice ("f022", no real person), with six emotions and laughter.
Weights CC BY 4.0. It needs a GPU.

**Samples.** The table above: plain reading, a price read through trnorm, a statement/question pair, a long held-out
sentence and the emotions. "Listener" rows were picked blind by one native listener; "metric" rows were picked by
automatic scores only.

**Quick start.** The code above: `say(text)` for plain text, `say(text, "angry")` for an emotion. Feed the text through
`trnorm.normalize`, keep `normalize=False`, pass no reference clip, and run `pause_clean.clean` on the output.

**Emotion and instruction.** `pick()` uses one training wording per emotion and chooses its language: Turkish when one
of the first two words starts with "c", else English. English instructions give stronger emotion but often read a
leading "c" the English way ("Cam" → "Kam"); Turkish ones get the "c" right with milder emotion. `prefer="tr"` or
`"en"` forces it. The rule agrees with 18 of 20 blind judgements (one listener).

**Evaluation.** Same harness for every row (Whisper large-v3-turbo judge, trnorm evaluation normaliser, UTMOS22).
{{name}} reads hard text best (TR-Zor WER 2.54 vs Trendyol-TTS 3.98). It loses on predicted naturalness, most on long
sentences (UTMOS 3.14 vs 3.73), on held-out WER (0.82 vs 0.68) and on speed (RTF {{rtf}} vs 1.37). Rows marked † were
measured in a separate local session; re-scoring {{name}} there matched within 0.01. Automatic metrics only, no MOS
study.

**Listening.** One listener, blind: in a 10-sentence round against the small models {{name}} was picked 8 times, Ahenk
Cep 2, EMA Lightning 0, Antalia-2 Mini 0. In an earlier 20-sentence round (before this checkpoint) our checkpoints
won 18, EMA Lightning 2.

**Known issues.** A leading "c" after an English instruction (use `prefer="tr"`). Questions without "mı" that should
rise at the end do not (0/4). A bright "phone-like" timbre on some emotional clips (7/20 with English instructions).
Milder emotion with Turkish instructions. Breath in pauses (use `pause_clean`). Spelled-out acronyms and phone numbers
are the weakest categories. Slower than real time on {{eval_gpu}} (RTF {{rtf}}); not measured on CPU.

**Responsible use.** Label the output as synthetic (EU AI Act Art. 50); there is no watermark, AudioSeal is an
option. No impersonation and no voice cloning without consent. Do not present generated emotion as a real person's
reaction. No fraud, scams or robocalls.

**Licence.** LoRA weights CC BY 4.0 (derived from the Alania datasets; keep the attribution above); base model
Apache-2.0; helper files Apache-2.0. Credit it as "Ahenk (https://github.com/ozguradmin/ahenk-turkish-tts),
CC BY 4.0". Contact: destek@ozgurguler.tech
