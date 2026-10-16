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
  - designed-voice
  - emotional-tts
datasets:
  - cloud0day3/alania-synthetic-speech-tr
  - cloud0day3/alania-domain-text-tr
---

# {{name}} — Türkçe ses (VoxCPM2 + f022 LoRA, tasarlanmış ses)

> **Bu bir sentetik (yapay) sestir.** Ses, VoxCPM2 ile İngilizce bir metin tarifinden tasarlandı; gerçek bir
> kişinin sesi ya da kaydı değildir ve eğitim seslerinin tamamı yapay zekâ ile üretildi. Dinleyicinin
> yanılabileceği her yerde sesin yapay olduğunu belirtin (AB Yapay Zekâ Yasası md. 50); bu modeli herhangi bir
> gerçek kişiyi taklit etmek için **asla** kullanmayın.
>
> **This is a synthetic voice.** It was designed with VoxCPM2 from an English text description; it is not the
> voice or a recording of any real person, and all of its training audio is AI-generated. Disclose that it is
> synthetic wherever a listener could be misled (EU AI Act Article 50), and **never** use it to impersonate any
> real person.

**{{name}}**, [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (OpenBMB, 2B parametre, 48 kHz) için bir LoRA
adaptörüdür. Bu sürüm tek bir **tasarlanmış** kadın sesiyle ("f022": ellili yaşlarda, danışman; kalın, kadifemsi,
sabırlı) konuşur; soruları soru gibi okumak, İngilizce bir talimatla altı duygu (kızgın, neşeli, üzgün, şaşkın,
sakin, empatik) vermek ve gülmek için eğitildi. İki aşamada eğitildi: önce
[Alania](https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr) sentetik Türkçe konuşmasının 25
saatlik filtreli bir alt kümesiyle Türkçe telaffuz güçlendirildi; sonra aynı LoRA, taban VoxCPM2'nin f022 sesiyle
okuduğu ve her klibi tek tek elenmiş 3.5 saatlik bir derlemle sürdürüldü. Önceki sürüm bir seslendirmenin (Antalia)
sesiyle konuşuyordu; **bu sürümde Antalia sesi yok**. Metin ön ucu [trnorm](https://github.com/ozguradmin/ahenk-turkish-tts):
sayı, tarih, saat, para, kısaltma ve telefon numarası gibi yazıları okunuşa çevirir.

**{{name}}** is a LoRA adapter for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (OpenBMB, 2B parameters,
48 kHz). This version speaks in one **designed** female voice ("f022": in her fifties, a counsellor; low-pitched,
velvety, patient) and was trained to ask questions like questions, to take six emotions (angry, cheerful, sad,
surprised, calm, empathetic) from an English instruction, and to laugh. Two training stages: a filtered 25-hour
subset of Alania synthetic Turkish speech for pronunciation, then the same LoRA on a 3.5-hour corpus that base
VoxCPM2 read in the f022 voice, every clip rejection-sampled. The previous version spoke in a voice actor's voice
(Antalia); **the Antalia voice is not in this version**. Text front-end: [trnorm](https://github.com/ozguradmin/ahenk-turkish-tts),
which turns numbers, dates, times, money, abbreviations and phone numbers into their spoken Turkish form.

| | |
|---|---|
| Temel model / Base model | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), Apache-2.0. Bu repo yalnızca LoRA ağırlıklarını içerir / this repo holds only the LoRA weights |
| Adaptör / Adapter | LoRA r={{lora_r}}, alpha={{lora_alpha}}; {{lora_targets}}; {{lora_params}} parametre / parameters, {{lora_mb}} MB |
| Kontrol noktası / Checkpoint | `{{exp}}/{{checkpoint}}` (adım / step {{step}}); `lora_weights.safetensors` sha256 `{{lora_sha256}}` |
| Ses / Voice | f022: VoxCPM2 ses tasarımı, gerçek bir kişi değil; referans klip gerekmez, ses LoRA'nın içinde / VoxCPM2 voice design, no real person; no reference clip needed, the voice is in the LoRA |
| Dil / Language | Türkçe, tek kadın ses / Turkish, one female voice |
| Kontrol / Control | düz metin; `(İngilizce talimat)` + metin ile 6 duygu ve gülme / plain text; 6 emotions and laughter with `(English instruction)` + text |
| Çıkış / Output | 48 kHz, mono |
| Lisans / Licence | LoRA ağırlıkları **CC-BY-4.0** (Alania verilerinden türedi, atıf zorunlu); temel model VoxCPM2 **Apache-2.0** / LoRA weights CC-BY-4.0 (derived from the Alania datasets, attribution required); base model Apache-2.0 |
| Ön uç / Front-end | metni `trnorm.normalize` ile verin, VoxCPM'in kendi normalizer'ı kapalı (`normalize=False`) / feed text through `trnorm.normalize` and keep VoxCPM's own normalizer off |

## Bu sürümde / In this version

- **Soru tonlaması / question intonation:** TR-Soru setinde (10 adım) soru-düz cümle eşlerinin 8/8'inde soru daha
  yüksek perde tepesiyle okundu, AUC 0.96; önceki sürümde ~0.73 ölçülmüştü / on the TR-Soru set (10 steps) the
  question had the higher pitch peak in 8 of 8 question-statement pairs, AUC 0.96; the previous version measured
  ~0.73. Ayrıntı: aşağıdaki tonlama tablosu / details: the intonation table below.
- **Duygu ve gülme / emotion and laughter:** 6 duygu ve gülme, metnin başına İngilizce talimatla
  (bkz. Kullanım) / six emotions and laughter through an English instruction in front of the text (see Usage).
- **Kör dinleme / blind listening:** aşağıda / below. Kısaca: duygulu satırlarda bu sürüm, düz okumada (sayı,
  telefon, akronim) küçük EMA Lightning daha çok beğenildi / in short: this version won on emotional lines, the
  small EMA Lightning on plain reading (numbers, phone numbers, acronyms).

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


def say(text: str, instruction: str = "", path: str = "cikti.wav") -> None:
    # trnorm yalnız Türkçe metne; İngilizce talimat parantez içinde, boşluksuz önüne gelir.
    # trnorm only on the Turkish text; the English instruction goes in front, in parentheses, no space.
    prompt = (f"({instruction})" if instruction else "") + normalize(text)
    wav = tts.generate(text=prompt, cfg_value={{cfg}}, inference_timesteps={{timesteps}}, normalize=False)
    sf.write(path, wav, tts.tts_model.sample_rate)  # 48 kHz


# 1. Düz metin / plain text
say("Randevunuz 14.10.2026 saat 09:30'da; ücret 1.250 TL.", path="duz.wav")
# 2. Soru: soru işareti yeter / question: the question mark is enough
say("Kargonuz yarın sabah mı geliyor?", path="soru.wav")
# 3. Duygu / emotion: "(Angry tone, voice getting louder.)" + trnorm(metin / text)
say("Bu üçüncü aramam ve sorun hâlâ çözülmedi!", "Angry tone, voice getting louder.", "kizgin.wav")
# 4. Gülme / laughter: "haha" metinde kalır / keep the "haha" in the text
say("Hahaha, bunu hiç beklemiyordum!", "Laughing heartily while speaking.", "gulme.wav")
```

Önerilen talimatlar (eğitimde kullanılan ifadelerden) / recommended instructions (wordings used in training):

| Duygu / Emotion | Talimat / Instruction |
|---|---|
| kızgın / angry | `Angry tone, voice getting louder.` |
| neşeli / cheerful | `Cheerful and warm, smiling while speaking.` |
| üzgün / sad | `Sad and quiet, with a heavy heart.` |
| şaşkın / surprised | `Astonished, with rising intonation.` |
| sakin / calm | `Calm, slow and soothing.` |
| empatik / empathetic | `Warm and compassionate, as if comforting someone.` |
| gülme / laughter | `Laughing heartily while speaking.` — metinde "haha" / "hahaha" kalsın / keep "haha" in the text |

- Eğitimde her duygu için 8 talimat ifadesi vardı ([`build_textbank.py`](https://github.com/ozguradmin/ahenk-turkish-tts/blob/main/tts/voxcpm/build_textbank.py)
  `INSTRUCTIONS`); hafif bir cümleye "Slightly ...", güçlüsüne "Very ..." türü ifade uyar. Başka ifadeler de çoğu
  zaman çalışır, en güvenlisi bunlar / training used 8 wordings per emotion; "Slightly ..." suits a mild sentence,
  "Very ..." a strong one. Other wordings often work; these are the safest.
- Talimat trnorm'dan **geçmez** (İngilizce); yalnız Türkçe metin normalize edilir / the instruction does **not**
  go through trnorm (it is English); only the Turkish text is normalized.
- Ünlemler: talimatsız okumada "Vay" doğru okunur. Duygu talimatıyla (ör. şaşkın) "Vay" çoğu zaman "vey" çıkar;
  modele "Vaay" yazın (8/8 doğru). Şaşkın tonda "canına" bazen "kanına" gibi sertleşir (bilinen sorun). "Vây", "Vaaay"
  yazımları okunuşu bozar / interjections: plain "Vay" is read right without an instruction; with an emotion
  instruction (e.g. surprised) "Vay" often comes out as "vey", so send "Vaay" (8/8 right). In a surprised tone
  "canına" can harden to "kanına" (known issue). Spellings like "Vây" or "Vaaay" hurt.
- Ayar / settings: önerilen **16 adım, cfg 2.0**; hız gerekiyorsa 10 adım (RTF tablosuna bakın). Yukarıdaki kod
  aşağıdaki ölçümün ayarını kullanır (`cfg_value={{cfg}}`, `inference_timesteps={{timesteps}}`) / recommended
  **16 timesteps, cfg 2.0**; 10 for speed (see the RTF column). The code above uses the settings of the evaluation
  below. Ölçümler `optimize=False` ile alındı; `optimize=True` torch.compile kullanır, etkisini ölçmedik /
  measured with `optimize=False`; `optimize=True` uses torch.compile, which we did not measure.
- Referans klip (`reference_wav_path`) vermeyin: ses LoRA'nın içinde, ölçümler ve dinlemeler referanssız yapıldı /
  pass no reference clip: the voice is in the LoRA; evaluation and listening ran without one.
- PyPI'deki `voxcpm` 2.0.3 de bu kodla çalışır: `lora_config` açıkça veriliyor (2.0.3 `lora_config.json`'u
  kendisi okumaz, varsayılan r=8 ile yüklemeye çalışır). 2.0.3'te `generate()` `seed` almaz; yukarıdaki commit
  `seed=0` gibi sabit bir tohum alır / PyPI `voxcpm` 2.0.3 also runs this code because `lora_config` is passed
  explicitly (2.0.3 does not read `lora_config.json` itself and would assume r=8); its `generate()` has no `seed`,
  the pinned commit does.

## İnsan dinlemesi / Human listening

Kör dinleme, **tek** anadil dinleyici, 3. tur: 14 cümle (zor yazılar, sorular, duygulu satırlar), 4 sistem, her
cümlede en iyi bulunan seçildi / blind, **one** native listener, round 3: 14 sentences (hard written forms,
questions, emotional lines), 4 systems, the best one picked per sentence.

| Sistem / System | Seçildi / Picked | Nerede / Where |
|---|---:|---|
| **{{name}}**, bu sürüm / this version (`{{checkpoint}}`) | 5/14 | 6 duygulu satırın 4'ü (kızgın, neşeli, üzgün, empatik) ve bir "değil mi" sorusu / 4 of the 6 emotional lines (angry, cheerful, sad, empathetic) and one "değil mi" question |
| [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning) (8.6M, Apache-2.0) + trnorm | 8/14 | düz cümle, "mı" soruları, şaşkın ("Vay canına") ve sakin satırlar, para, telefon numarası, akronim / plain statement, "mı" questions, the surprised ("Vay canına") and calm lines, money, phone number, acronym |
| Aynı eğitimin 1000. adımı / the same training at step 1000 | 1/14 | bir "neden" sorusu / one "neden" question |
| Önceki {{name}} sürümü (Antalia sesi) / the previous version (Antalia voice) | 0/14 | |

- Gülme / laughter: dinleyici bu modelin 12 gülme klibinin 7'sini gerçek gülme olarak değerlendirdi (örneklerle
  yönlendirilmiş Gemini 3.8 Flash de aynı 7'yi seçti) / the listener judged 7 of 12 laughter clips of this model as real
  laughter (Gemini 3.8 Flash, prompted with labelled examples, picked the same 7).
- Tek dinleyici ve 14 cümle küçük bir örnektir; istatistiksel bir sonuç değil, bir işarettir / one listener and 14
  sentences are a small sample: an indication, not a statistical result.

## Ölçümler / Evaluation

Yargıç / judge: **faster-whisper large-v3-turbo** (beam 5, `language="tr"`) sesi geri okur; referans ve hipotez
`trnorm.normalize_for_eval` ile normalize edilir. WER %, düşük daha iyi / lower is better.
**TR-Zor**: sayı, tarih, saat, para, yüzde, kısaltma, telefon numarası gibi zor yazılar içeren 155 cümle
([benchmarks/tts](https://github.com/ozguradmin/ahenk-turkish-tts/tree/main/benchmarks/tts)); metin modele trnorm'dan
geçerek verilir / the text is fed through trnorm. **Ayrılmış / Held-out**: Antalia veri setinin ayrılmış 20
cümlesinin metni, uzun cümleler (~12-19 sn); bu modelin eğitiminde yok, burada f022 sesiyle okunur / the text of the
20 held-out sentences of the Antalia dataset, long ones (~12-19 s), not in this model's training, read here in the
f022 voice. **Hatasız / error-free**: yargıcın tek kelimeyi bile yanlış okumadığı cümleler / sentences without a
single word error. **UTMOS**: UTMOS22 strong ([SpeechMOS](https://github.com/tarepan/SpeechMOS)), 1–5, yüksek daha
iyi / higher is better.

> UTMOS doğallığı **tahmin eder** (bir modelin tahmini, insan puanı değil), WER ise **anlaşılırlığı** ölçer. /
> UTMOS **predicts** naturalness (a model's estimate, not human ratings); WER measures **intelligibility**.

| Model | Çıkarım / Inference (timesteps / cfg) | TR-Zor WER | TR-Zor hatasız / error-free | Ayrılmış / Held-out WER | Ayrılmış hatasız / error-free | UTMOS TR-Zor | UTMOS ayrılmış / held-out | RTF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{{results_rows}}

RTF: sentez süresi / ses süresi, {{eval_gpu}} üzerinde, cümle cümle sentez; 1'in altı gerçek zamandan hızlı. Aynı
ayarla farklı konteynerlerde ölçülen RTF belirgin değişebilir / synthesis time over audio duration on {{eval_gpu}},
one sentence at a time, below 1 is faster than real time; the same settings can measure quite differently from one
container to the next. "—": ölçülmedi / not measured.
{{eval_note}}

### Soru tonlaması / Question intonation (TR-Soru)

**TR-Soru** ([`tr_soru.jsonl`](https://github.com/ozguradmin/ahenk-turkish-tts/blob/main/benchmarks/tts/sets/tr_soru.jsonl)):
35 satır; evet-hayır soruları (mI), "değil mi", soru sözcüklü ve mI'siz yükselen sorular, eşli düz cümleleriyle /
35 rows: yes-no questions (mI), "değil mi", wh-questions and rising questions without mI, with matched statements.
`score_intonation.py` perde tepesinin yerini ve yüksekliğini ölçer / measures where the pitch peak falls and how
high. **Eşler / pairs**: sorunun tepesi eşli düz cümleninkinden yüksek olan çiftler / pairs whose question peaks
higher than its statement. **AUC**: mI sorularının tepesi düz cümlelerden ne kadar ayrışıyor (0.5 fark yok, 1 tam
ayrım) / how well the peaks of mI questions separate from the statements (0.5 none, 1 complete). **Kalıba uyan
sorular / questions matching the pattern**: Türkçe kalıba (evet-hayır sorusunda tepe mI'den önceki hecede, sonra
düşüş; soru sözcüğünde tepe; mI'siz soruda sonda yükseliş) uyan soruların payı; katı bir ölçü / the share of
questions with the Turkish shape (yes-no: peak on the syllable before mI, then a fall; wh: peak on the question
word; no mI: final rise); a strict measure.

{{intonation_table}}

### TR-Zor kategorileri / categories

WER % (hatasız / toplam cümle) / WER % (error-free / total sentences).

{{category_table}}

## Örnekler / Samples

{{samples}}

## Eğitim / Training

- Temel model / base model: `openbmb/VoxCPM2`; resmi ince ayar betiği / official fine-tuning script
  [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) `{{voxcpm_commit}}`.
- **Aşama 1 / stage 1** (önceki sürümle aynı LoRA / the same LoRA as the previous version's stage 1): Alania
  (`cc-by`, commit `277dfc8a23`), 25 saat, 23.706 klip, 1.177 kadın sesi; yalnızca düz okumalar, `qc_cer == 0`,
  `qc_utmos >= 3.7`, FineWeb-2 metni en çok %25; TR-Zor ve Antalia'nın ayrılmış cümleleri hariç / 25 h, 23,706
  clips, 1,177 female voices; plain reads only, zero CER, UTMOS >= 3.7, at most 25 % FineWeb-2 text; TR-Zor and
  Antalia's held-out sentences excluded. LoRA r={{lora_r}}, lr 1e-4, batch 16, 2.000 adım / steps, A100 80 GB.
- **Ses / voice f022:** VoxCPM2 ses tasarımı, referanssız, şu İngilizce tariften / VoxCPM2 voice design, no
  reference, from this English description: *"Female speaker, in their fifties, a counsellor; low-pitched and
  velvety; patient and polite delivery at a measured pace; recorded close to the microphone in a quiet studio."*
  Tarif hiçbir gerçek kişiyi anmaz ve taklit istemez. ~150 tarif × 2 tohumluk bir taramadan, 4 klonun ortalama
  UTMOS'u, Whisper CER ve iki kör dinleme turuyla seçildi / the description names no real person and asks for no
  imitation. Picked from a screen of ~150 descriptions × 2 seeds by the mean UTMOS of 4 clones, Whisper CER and two
  blind listening rounds.
- **Aşama 2 verisi / stage-2 data:** taban VoxCPM2 f022'nin referans klibini klonlayarak okudu (10 adım, cfg 2.0) /
  base VoxCPM2 read it cloning f022's reference clip (10 steps, cfg 2.0). 3.612 klip, 3.53 saat / 3,612 clips,
  3.53 h: düz cümle / plain sentences 2.55 sa/h (Alania Domain Text), soru / questions 0.29, duygu / emotions 0.54,
  ünlem / interjections 0.10, kısa cevap / short replies 0.05; ve / plus **48 gülme klibi / laughter clips**
  (~135 sn/s, `(Laughing heartily while speaking.)`), 156 adaydan bir dinleyicinin "gerçekten gülüyor" dedikleri /
  the ones a listener approved as real laughter out of 156 candidates. Duygulu satırlar o duygunun kendi f022
  referans klibini klonladı (f022 sesiyle, o talimatla üretilip prozodisine göre seçilmiş) / emotional lines cloned
  a per-emotion f022 reference (rendered with that instruction, picked by its prosody).
- **Ret örneklemesi / rejection sampling** (11.418 puanlanan deneme / scored tries → 3.612 klip / clips): her klip
  şunların hepsini geçti, yoksa yeni tohumla en çok 4 deneme / every clip passed all of these, else up to 4 tries
  with new seeds: Whisper large-v3-turbo CER 0 ve / and p(tr) ≥ 0.95; başta fazladan ses yok, sondaki kısa fazladan
  ses kesildi / no stray sound at the start, a short stray sound at the end cut off; ilk kelime uzaması, aşırı "ş/s", konuşma hızı sınırları / first-word stretch, sibilance and speaking-rate
  limits; f022'ye ECAPA benzerliği ≥ 0.6 (kısa kliplerde daha düşük) / ECAPA similarity to f022 ≥ 0.6 (lower for
  short clips); UTMOS kategorisinin alt %20'sinde değil / UTMOS not in the bottom 20 % of its category; soru tonlaması
  kalıba uyar ve eşli düz cümle soru gibi yükselmez / question intonation matches its pattern and the matched
  statement does not rise; duygu prozodisi nötr kliplere göre ters değil, üzgün/sakin daha yavaş ve alçak / emotion
  prosody not opposite to neutral clips, sad/calm slower and lower.
- **Aşama 2 eğitimi / stage-2 training:** aşama 1'in LoRA'sından / from the stage-1 LoRA, lr 1e-4, batch 16,
  1.500 adım / steps, A100 80 GB, ~24 dk/min. Metinler trnorm'un okunuş formunda; duygu ve gülme satırlarında
  `(talimat)` + okunuş / transcripts in trnorm's spoken form; emotion and laughter rows as `(instruction)` + spoken
  form.
- LoRA hedefleri / targets: {{lora_targets}}.
- Tarif / recipe: [ahenk-turkish-tts/tts/voxcpm](https://github.com/ozguradmin/ahenk-turkish-tts/tree/main/tts/voxcpm) (Apache-2.0);
  derlem / corpus: `render_corpus.py`, metin bankası / text bank: `tts/voxcpm/textbank`.

## Eğitim verisi ve atıf / Training data and attribution

- Alania Turkish Synthetic Speech by PatientDesk AI (https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr),
  licensed under CC BY 4.0 — aşama 1 verisi; tamamı yapay zekâ ile üretilmiş (VoxCPM2) konuşma, gerçek bir kişinin
  sesi yok. Duygu ve gülme talimatlarına dair bilgimiz de bu veri setinden geldi / stage-1 data; entirely
  AI-generated speech (VoxCPM2), no real person's voice; also the source of our know-how on emotion and laughter
  instructions. Değişiklik / changes: filtrelenmiş alt küme / filtered subset, 48 → 16 kHz.
  - Includes Turkish web text from FineWeb-2 (https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), used under
    ODC-By 1.0 and subject to Common Crawl's terms of use.
- Alania Turkish Domain Text by PatientDesk AI (https://huggingface.co/datasets/cloud0day3/alania-domain-text-tr),
  licensed under CC BY 4.0 — aşama 2'nin düz cümleleri / the plain sentences of stage 2 (commit `20fd903fe6`).
  Değişiklik / changes: kategori, uzunluk (25-180 karakter), soru ve dil kimliği filtreleri; trnorm ile okunuş
  formuna çevrildi; VoxCPM2 f022 sesiyle seslendirdi / category, length (25-180 characters), question and
  language-ID filters; converted to spoken form with trnorm; voiced by VoxCPM2 in the f022 voice.
- Ses / voice: Voice designed from a text description with VoxCPM2 (OpenBMB, https://huggingface.co/openbmb/VoxCPM2,
  Apache-2.0); no real person's voice. / Ses, VoxCPM2 ile metin tarifinden tasarlandı; gerçek bir kişinin sesi
  değildir.
- Soru, duygu, ünlem, kısa cevap ve gülme metinleri: kendi metin bankamız, bu proje için yazıldı / questions,
  emotional lines, interjections, short replies and laughter lines: our own text bank, written for this project
  ([ahenk-turkish-tts/tts/voxcpm/textbank](https://github.com/ozguradmin/ahenk-turkish-tts/tree/main/tts/voxcpm/textbank),
  Apache-2.0).
- Temel model / base model: [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (OpenBMB), Apache-2.0.
- Ön uç ve tarif / front-end and recipe: [ahenk-turkish-tts](https://github.com/ozguradmin/ahenk-turkish-tts) (Apache-2.0).
- Antalia: önceki sürüm Antalia seslendirmeninin izinli kayıtlarıyla eğitilmişti; **bu sürümün eğitiminde Antalia
  sesi ya da metni yok**. Ölçümdeki "ayrılmış" cümlelerin metni Antalia veri setinden (Patientdesk.ai, CC BY 4.0,
  https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus) / the previous version was trained on the
  consented recordings of the Antalia voice actor; **no Antalia audio or text is in this version's training**. The
  text of the "held-out" evaluation sentences comes from the Antalia dataset.

Bu ağırlıkları kullanırken veya dağıtırken yukarıdaki Alania atıflarını koruyun (CC-BY-4.0 şartı).
Keep the Alania attributions above when you use or redistribute these weights (CC-BY-4.0).

## Etik ve kabul edilebilir kullanım / Ethics and acceptable use

- **Sentetik ses / synthetic voice:** f022 tasarlanmış bir sestir, kimsenin sesi değildir; gerçek bir kişiye
  benzerliği tesadüftür ve hiç kimse olarak sunulmamalıdır. Üretilen sesi **gerçek bir insan kaydı gibi sunmayın**;
  dinleyicinin yanılabileceği yerlerde (telefon, IVR, duyuru, video) yapay olduğunu açıkça belirtin ve çıktıyı
  yapay zekâ üretimi olarak etiketleyin (AB Yapay Zekâ Yasası md. 50: makinece okunur işaret, ör. AudioSeal, MIT) /
  f022 is a designed voice, nobody's voice; any resemblance to a real person is accidental and it must not be
  presented as anyone. Do **not** pass its speech off as a genuine recording: disclose it where a listener could be
  misled and label the output as AI-generated (EU AI Act Article 50: a machine-readable mark, e.g. AudioSeal, MIT).
- **Taklit yok / no impersonation:** bu modeli herhangi bir gerçek kişiyi taklit etmek, birine söylemediği bir şeyi
  söyletmek için kullanmayın. VoxCPM2 ses klonlayabilir (`reference_wav_path`); başkasının sesini izni olmadan
  klonlamayın / do not use it to impersonate any real person or make someone appear to say what they did not.
  VoxCPM2 can clone voices from a reference clip; do not clone anyone without their consent.
- **Duygu ve gülme aldatmak için değil / emotion and laughter are not for deception:** üretilen öfke, üzüntü,
  şefkat ya da gülmeyi gerçek bir insanın tepkisi gibi sunmayın; dinleyiciyi duygusal baskıyla yönlendirmek ya da
  kandırmak için (sahte acil durum, sahte yakın, sahte samimiyet) kullanmayın / do not present generated anger,
  sadness, warmth or laughter as a real person's reaction, and do not use them to pressure or deceive listeners
  (fake emergencies, fake relatives, fake rapport).
- **Dolandırıcılık, sosyal mühendislik, siyasi otomatik arama** için kullanmayın / no fraud, scams, social
  engineering or political robocalls.

(CC-BY-4.0 altında bunlar lisans şartı değil, ricadır / under CC-BY-4.0 these are requests, not licence
conditions; the EU AI Act duties apply on their own.)

## Sınırlamalar / Limitations

- Tek tasarlanmış kadın ses, yalnızca Türkçe / a single designed female voice, Turkish only.
- Eğitim seslerinin tamamı sentetik (VoxCPM2): doğallık taban modelin ve tasarlanan sesin üstüne çıkamaz / all
  training audio is synthetic (VoxCPM2): naturalness cannot exceed the base model's and the designed voice's.
- Düz okumada (sayı, telefon numarası, akronim) dinleyici küçük EMA Lightning'i daha çok seçti; harf harf okunan
  kısaltmalar zayıf, kategori tablosuna bakın / on plain reading (numbers, phone numbers, acronyms) the listener
  preferred the small EMA Lightning; spelled-out acronyms are weak, see the category table.
- Aşama 2 klipleri kısa (ortalama 3.5 sn, en uzun 11.7 sn): uzun cümlelerde doğallık düşebilir, ayrılmış setin
  (12-19 sn) UTMOS'una bakın. Uzun metni cümlelere bölüp tek tek sentezleyin / stage-2 clips are short (3.5 s on
  average, 11.7 s at most): naturalness may drop on long sentences, see the held-out UTMOS (12-19 s sentences).
  Split long text into sentences and synthesize them one by one.
- Duygular: üzgün ve sakin en zayıfları (derlemde de bu iki duygunun kliplerinin çoğu hafif kaldı); gülme her
  denemede gerçek gülme olmayabilir (dinleyici 12 klipten 7'sini gerçek buldu), birkaç tohum deneyin / emotions: sad
  and calm are the weakest (most of their corpus clips were mild too); laughter is not real laughter every time
  (7 of 12 clips for the listener), try a few seeds.
- WER bir ASR yargıcıyla ölçüldü (faster-whisper large-v3-turbo); yargıcın kendi hataları sonuca karışır. İnsan
  dinlemesi tek dinleyiciyle yapıldı, sistematik bir MOS testi yapılmadı / WER is measured by an ASR judge whose own
  errors are part of the score; the human listening had one listener, no systematic MOS test was run.
- GPU gerekir: yukarıdaki ayarlarla RTF {{rtf}} ({{eval_gpu}}); 1'in üstü gerçek zamandan yavaş demektir, daha
  az adım (`inference_timesteps`) daha hızlıdır (tablodaki RTF'lere bakın). CPU'da ölçülmedi; 2B parametreli bir
  modelden CPU'da gerçek zamanlı hız beklemeyin / needs a GPU: RTF {{rtf}} on {{eval_gpu}} with the settings above
  (above 1 is slower than real time; fewer `inference_timesteps` are faster, see the RTF column); not measured on
  CPU, do not expect real time there from a 2B model.
- Metin `trnorm` ile normalize edilmeden verilirse sayılar, tarihler ve kısaltmalar yanlış okunabilir / without
  `trnorm`, numbers, dates and abbreviations may be misread.

## İletişim / Contact

Sorular, izin ve kötüye kullanım bildirimleri / questions, permissions and abuse reports: destek@ozgurguler.tech
