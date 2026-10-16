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

# {{name}} — Türkçe ses (VoxCPM2 + f022 LoRA, tasarlanmış ses, Türkçe ve İngilizce talimatlar)

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
adaptörüdür. Tek bir **tasarlanmış** kadın sesiyle ("f022": ellili yaşlarda, danışman; kalın, kadifemsi, sabırlı)
konuşur; soruları soru gibi okumak, metnin başındaki bir talimatla altı duygu (kızgın, neşeli, üzgün, şaşkın, sakin,
empatik) vermek ve gülmek için eğitildi. Talimat **İngilizce ya da Türkçe** olabilir; model ikisini de bilir ve her
biri başka bir konuda güçlüdür: İngilizce talimat saf duyguyu (kızgın, neşeli, üzgün) daha güçlü verir ama cümle
başındaki "c"yi çoğu zaman İngilizce okutur ("Vay canına" → "vey kenına", "Cam" → "Kam"); Türkçe talimat "c"yi doğru
okutur, duyguyu daha hafif verir. Paketteki `instruction_pick.py` her satır için seçer: ilk iki kelimeden biri "c" ile
başlıyorsa Türkçe, yoksa İngilizce talimat. Önceki f022 sürümüne göre yenilikler: uzun cümleler, sayılı cümleler
ve kelime başında "c" olan duygulu cümlelerle ikinci bir derlem; Türkçe talimatlar; `instruction_pick.py`; sentezden
sonra duraklardaki nefes sesini kısan `pause_clean.py`. Metin ön ucu
[trnorm](https://github.com/ozguradmin/ahenk-turkish-tts): sayı, tarih, saat, para, kısaltma ve telefon numarası gibi
yazıları okunuşa çevirir.

**{{name}}** is a LoRA adapter for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (OpenBMB, 2B parameters,
48 kHz). It speaks in one **designed** female voice ("f022": in her fifties, a counsellor; low-pitched, velvety,
patient) and was trained to ask questions like questions, to take six emotions (angry, cheerful, sad, surprised,
calm, empathetic) from an instruction in front of the text, and to laugh. The instruction can be **English or
Turkish**; the model knows both and each is good at something else: an English instruction gives pure emotions
(angry, cheerful, sad) more strongly but often makes the model read a "c" near the start the English way ("Vay
canına" → "vey kenına", "Cam" → "Kam"); a Turkish one gets the "c" right and gives milder emotion.
`instruction_pick.py`, in this repo, picks per line: a Turkish instruction when one of the first two words starts
with a "c", else an English one. New since the previous f022 version: a second corpus of long sentences, sentences
with numbers and emotional sentences with a word-initial "c"; Turkish instructions; `instruction_pick.py`; and
`pause_clean.py`, which turns breath noise in pauses down after synthesis. Text front-end:
[trnorm](https://github.com/ozguradmin/ahenk-turkish-tts), which turns numbers, dates, times, money, abbreviations and
phone numbers into their spoken Turkish form.

| | |
|---|---|
| Temel model / Base model | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), Apache-2.0. Bu repo LoRA ağırlıklarını ve küçük yardımcı dosyaları içerir / this repo holds the LoRA weights and small helper files |
| Adaptör / Adapter | LoRA r={{lora_r}}, alpha={{lora_alpha}}; {{lora_targets}}; {{lora_params}} parametre / parameters, {{lora_mb}} MB |
| Kontrol noktası / Checkpoint | `{{exp}}/{{checkpoint}}` (adım / step {{step}}): aşama 2'nin 1500. adımından başlayan iki dilli talimat ince ayarı / the two-language instruction fine-tune started from step 1500 of stage 2; `lora_weights.safetensors` sha256 `{{lora_sha256}}` |
| Ses / Voice | f022: VoxCPM2 ses tasarımı, gerçek bir kişi değil; referans klip gerekmez, ses LoRA'nın içinde / VoxCPM2 voice design, no real person; no reference clip needed, the voice is in the LoRA |
| Dil / Language | Türkçe, tek kadın ses; talimatlar İngilizce ya da Türkçe / Turkish, one female voice; instructions in English or Turkish |
| Kontrol / Control | düz metin; `(talimat)` + metin ile 6 duygu ve gülme; talimatın dilini `instruction_pick.py` seçer (aşağıda) / plain text; 6 emotions and laughter with `(instruction)` + text; `instruction_pick.py` picks the instruction's language (below) |
| Sentez sonrası / After synthesis | `pause_clean.py` (bu repoda): duraklardaki nefes sesini kısar, konuşmanın kendisine dokunmaz (yalnız DC atılır) / (in this repo): turns breath noise in pauses down, leaves the speech itself as it is (only the DC is removed) |
| Çıkış / Output | 48 kHz, mono |
| Lisans / Licence | LoRA ağırlıkları **CC-BY-4.0** (Alania verilerinden türedi, atıf zorunlu); temel model VoxCPM2 **Apache-2.0**; `instruction_pick.py`, `pause_clean.py`, `onset.py`, `instructions_tr.json` **Apache-2.0** ([ahenk-turkish-tts](https://github.com/ozguradmin/ahenk-turkish-tts)) / LoRA weights CC-BY-4.0 (derived from the Alania datasets, attribution required); base model Apache-2.0; the helper files Apache-2.0 |
| Ön uç / Front-end | metni `trnorm.normalize` ile verin, VoxCPM'in kendi normalizer'ı kapalı (`normalize=False`) / feed text through `trnorm.normalize` and keep VoxCPM's own normalizer off |

### Dosyalar / Files

`lora_weights.safetensors` + `lora_config.json` (LoRA), `samples/*.wav` (örnekler / samples) ve / and
[ahenk-turkish-tts](https://github.com/ozguradmin/ahenk-turkish-tts/tree/main/tts/voxcpm)'ten olduğu gibi kopyalanan yardımcılar /
helpers copied as they are from ahenk-turkish-tts:

{{extra_files}}

## Bu sürümde / In this version

- **İkinci derlem / second corpus:** 1,015 sa, 620 klip: günlük ve sayılı düz cümleler 0,51 sa, uzun cümleler 0,47 sa
  (medyan ~14 sn), kelime başında "c" olan duygulu cümleler 47 klip; ilk derlemle (3,53 sa, c'si yanlış okunmuş 6
  klip çıkarıldı) birlikte 4.274 klip, 4,58 sa / 1.015 h, 620 clips: everyday and number sentences 0.51 h, long
  sentences 0.47 h (median ~14 s), emotional sentences with a word-initial "c" 47 clips; with the first corpus (3.53 h,
  6 clips whose "c" was misread dropped) 4,274 clips, 4.58 h.
- **İki dilde talimat / instructions in two languages:** aşama 2'nin 1500. adımından 500 adımlık bir ince ayar; her
  talimatlı satır bir kez İngilizce, bir kez Türkçe talimatla (aynı klip; Türkçeler `instructions_tr.json`'dan).
  Yayınlanan model bunun 500. adımı. İngilizce talimatı unutmadı: probda perde (F0 ortancası) kızgın 325, neşeli 341,
  üzgün 223 Hz (1500. adım: 244, 331, 219 Hz); Türkçe talimatla "c" doğru 13/14, İngilizceyle 1/14 / a 500-step
  fine-tune from step 1500 of stage 2: every instructed row once with its English and once with its Turkish
  instruction (the same clip; the Turkish side from `instructions_tr.json`). The released model is its step 500. It
  kept the English instructions: probe pitch (median F0) angry 325, cheerful 341, sad 223 Hz (step 1500: 244, 331,
  219 Hz); "c" right 13 of 14 with Turkish instructions, 1 of 14 with English ones.
- **Talimat seçimi / choosing the instruction:** `instruction_pick.py`, iki kör dinleme turunun sonucuna göre: saf
  duyguda İngilizce talimat 6/6, ilk iki kelimeden biri "c" ile başlayan cümlelerde Türkçe talimat 10/10 seçildi
  (aşağıda) / from two blind listening rounds: English instructions were picked 6 of 6 times on pure emotions,
  Turkish ones 10 of 10 times on sentences with a word starting with "c" among the first two (below).
- **Kör dinleme, 4. tur / blind listening, round 4** (tek dinleyici, 20 cümle, çoğu düz okuma / one listener, 20
  sentences, mostly plain reading): ince ayarın başladığı 1500. adım 8, önceki sürüm 6, aynı eğitimin 2000. adımı 4,
  EMA Lightning 2 (ayrıntı aşağıda) / step 1500, where the fine-tune started, 8, the previous version 6, step 2000 4,
  EMA Lightning 2 (details below). Yayınlanan 500. adım bu turda yoktu; düz okumada 1500. adımla aynı ya da biraz daha iyi: TR-Zor 2,54 / 2,87 (ikisi de 141/155), ayrılmış 0,82 / 1,22 (16/20 / 15/20), UTMOS kısa 3,70 / 3,71, uzun 3,14 / 3,06 (500. adımın ölçümünde `pause_clean` açık, 1500. adımınkinde kapalı) / step 500 was not in this round; its plain reading is the same as step 1500's or slightly better: TR-Zor 2.54 / 2.87 (141/155 both), held-out 0.82 / 1.22 (16/20 / 15/20), UTMOS short 3.70 / 3.71, long 3.14 / 3.06 (step 500 evaluated with `pause_clean`, step 1500 without).
- **Nefes / breath:** `pause_clean.py` duraklardaki nefes sesini sentezden sonra kısar; kör A/B'de (4 uzun cümle)
  temizlenmiş sürümde nefes 4/4 daha az bulundu, doğallık 3'ünde aynı, 1'inde daha iyi / turns breath noise in pauses
  down after synthesis; in a blind A/B (4 long sentences) the cleaned version had less breath 4 of 4 times,
  naturalness the same 3 times, better once.

## Kullanım / Usage

```bash
# VoxCPM: eğitimde ve ölçümde kullanılan commit / the commit used for training and evaluation
pip install "voxcpm @ git+https://github.com/OpenBMB/VoxCPM@{{voxcpm_commit}}"
# trnorm: Türkçe yazı -> okunuş / Turkish written -> spoken form
pip install "git+https://github.com/ozguradmin/ahenk-turkish-tts"
# pause_clean.py (bu repoda / in this repo): numpy + scipy; instruction_pick.py: yalnız standart kütüphane / stdlib only
pip install numpy scipy
```

```python
import json
import sys
from pathlib import Path

import soundfile as sf
from huggingface_hub import snapshot_download
from trnorm import normalize
from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

# LoRA (lora_weights.safetensors + lora_config.json), instruction_pick.py, pause_clean.py + onset.py,
# instructions_tr.json
repo_dir = snapshot_download("{{repo}}", allow_patterns=["lora_*", "*.py", "instructions_tr.json"])
sys.path.insert(0, repo_dir)
from instruction_pick import compose, pick  # noqa: E402  (stdlib)
from pause_clean import clean  # noqa: E402  (numpy + scipy)

lora_cfg = json.loads(Path(repo_dir, "lora_config.json").read_text(encoding="utf-8"))["lora_config"]
tts = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False,
                             lora_config=LoRAConfig(**lora_cfg), lora_weights_path=repo_dir)
SR = tts.tts_model.sample_rate  # 48 kHz


def say(text: str, emotion: str = "", path: str = "cikti.wav", prefer: str = "auto") -> None:
    # trnorm yalnız metne; pick talimatı seçer (ilk iki kelimeden biri "c" ile başlıyorsa Türkçe, yoksa İngilizce),
    # "Vay" -> "Vaay" / trnorm only on the text; pick chooses the instruction (Turkish when one of the first two words
    # starts with "c", else English).
    instruction, model_text, why = pick(emotion, normalize(text), prefer)
    prompt = compose(instruction, model_text)  # "(talimat)metin" / "(instruction)text"
    wav = tts.generate(text=prompt, cfg_value={{cfg}}, inference_timesteps={{timesteps}}, normalize=False)
    wav, report = clean(wav, SR)  # duraklardaki nefes kısılır / breath noise in pauses turned down (report["breaths"])
    sf.write(path, wav, SR)
    print(f"{path}: {prompt}  # {why}")


# 1. Düz metin, talimatsız / plain text, no instruction
say("Randevunuz 14.10.2026 saat 09:30'da; ücret 1.250 TL.", path="duz.wav")
# 2. Soru: soru işareti yeter / question: the question mark is enough
say("Kargonuz yarın sabah mı geliyor?", path="soru.wav")
# 3. İlk iki kelimeden biri "c" ile başlıyor -> Türkçe talimat / one of the first two starts with "c" -> Turkish:
#    (Şaşkın, yükselen bir tonlamayla.)Vaay canına! Bu manzarayı hiç beklemiyordum!
say("Vay canına! Bu manzarayı hiç beklemiyordum!", "surprised", "saskin.wav")
# 4. Öyle bir "c" yok -> İngilizce talimat, duygu daha güçlü / no such "c" -> English instruction, stronger emotion:
#    (Angry tone, voice getting louder.)Yeter artık! Bu sorun hâlâ çözülmedi!
say("Yeter artık! Bu sorun hâlâ çözülmedi!", "angry", "kizgin.wav")
# 5. "Acele"nin "c"si kelimenin içinde -> İngilizce: "Asele" diyebilir ama sakin (dinleyici bunu seçti); "c" daha
#    önemliyse dili siz seçin / the "c" of "Acele" is inside the word -> English: it may say "Asele" but calm (the
#    listener's pick); if the "c" matters more, pick the language yourself
say("Acele etmenize gerek yok, her şeyi birlikte halledeceğiz.", "calm", "sakin.wav")
say("Acele etmenize gerek yok, her şeyi birlikte halledeceğiz.", "calm", "sakin_tr.wav", prefer="tr")
# 6. Gülme: "haha" metinde kalır / laughter: keep the "haha" in the text
say("Hahaha, bunu hiç beklemiyordum!", "laughing", "gulme.wav")
# 7. Kendi talimatınız, olduğu gibi (aşağıdaki tablodan) / your own instruction, as given (from the table below)
say("Bugün çok yoruldum, eve erken döneceğim.", "Biraz üzgün ve yorgun, enerjisiz.", "yorgun.wav")
```

### Talimat seçimi / Choosing the instruction

`pick(duygu, metin, prefer="auto")` → `(talimat, modele giden metin, neden)`; `compose(talimat, metin)` →
`"(talimat)metin"`. Kural: metnin ilk iki kelimesinden biri "c" ile başlıyorsa Türkçe talimat, yoksa İngilizce.
"ç" sayılmaz; kelimenin içindeki "c" de sayılmaz ("Acele", "gece", "üçüncü" İngilizce alır; "Acele"de dinleyici
İngilizceyi seçti); baştaki tırnak, tire ve üç nokta kelime sayılmaz; büyük harfler Türkçe küçültülür ("CİDDEN" de
sayılır). `prefer="en"` ya da `"tr"` dili zorlar. Duygular: `angry`, `cheerful`, `sad`, `calm`, `surprised`,
`empathetic`, `laughing` (ya da `kızgın`, `neşeli`, `üzgün`, `sakin`, `şaşkın`, `empatik`, `gülerek`; büyük harfle
de olur); her biri için tek bir eğitim ifadesi kullanılır: aşağıdaki tablonun "Önerilen" sütunu ve İngilizce aslı.
Duygu adı yerine kendi talimatınızı da verebilirsiniz (iki ya da daha çok kelime ya da ".", "!" veya "…" ile biten
tek kelime; noktalı bir duygu adı, ör. "angry.", hata verir): seçim yapılmaz, olduğu gibi kullanılır. Talimatlı
metinde okunacak bir kelime olmalı. `pick`'e trnorm'un çıktısını verin, model onu okur ("3 cami" → "üç cami").
Komut satırından / from the command line: `python instruction_pick.py surprised "Cam kırılmış!"`.

`pick(emotion, text, prefer="auto")` → `(instruction, text for the model, why)`; `compose(instruction, text)` →
`"(instruction)text"`. Rule: a Turkish instruction when one of the text's first two words starts with a "c", else an
English one. "ç" does not count, nor does a "c" inside a word ("Acele", "gece", "üçüncü" get English; on "Acele" the
listener picked English); leading quotes, dashes and ellipses are not words; capitals are lowercased the Turkish way
("CİDDEN" counts). `prefer="en"` or `"tr"` forces the language. Emotions: the names above (or their Turkish names,
in any case); each uses one training wording: the table's "Recommended" column and its English original. Instead of
an emotion name you can pass your own instruction (two or more words, or one word ending in ".", "!" or "…"; an
emotion name with punctuation such as "angry." is an error): it is used as given, no picking. Instructed text must
hold a word to read. Feed `pick` trnorm's output, which is what the model reads.

Neden bu kural: iki kör dinleme turu (tek dinleyici, her turda 10 duygulu cümle, her cümlede üç sürüm: İngilizce
talimatlı bir sürüm ve Türkçe talimatlı iki sürüm; 20 karar) / why this rule: two blind listening rounds (one listener,
10 emotional sentences per round, three versions per sentence: one with an English and two with a Turkish instruction;
20 judgements):

| Cümleler / Sentences | Dinleyicinin seçtiği / Listener's pick | `pick` (auto) |
|---|---|---|
| Kızgın, neşeli, üzgün; "c" yok (3 cümle) / angry, cheerful, sad; no "c" (3 sentences) | **İngilizce 6/6** / **English 6 of 6**; Türkçede duygu zayıf ya da yapmacık / weak or forced in Turkish | İngilizce / English |
| Şaşkın, ilk iki kelimeden biri "c" ile başlıyor: canına, Cam, Cuma, Cesaret, Cidden (5) / surprised, one of the first two words starts with "c" (5) | **Türkçe 10/10** / **Turkish 10 of 10**; İngilizce: "vey genına", "Kam", "Sesaretine", "Zidden" / English | Türkçe / Turkish |
| Sakin: "Acele etmenize gerek yok…", "c" kelimenin içinde (1) / calm, the "c" inside the word (1) | **İngilizce 2/2** / **English 2 of 2**: "Asele" ama sakin; Türkçe "Aacele", sakin değil / "Asele" but calm; Turkish "Aacele", not calm | İngilizce / English (`prefer="tr"`: "c" doğru / "c" right) |
| Empatik, "c" yok (1) / empathetic, no "c" (1) | Türkçe 2/2, ama 2. turda "üçü de aynı gibi" / Turkish 2 of 2, but in round 2 "all three alike" | İngilizce / English |

Toplam İngilizce 8/20, Türkçe 12/20; kural 20 kararın 18'iyle uyuşur (uymayan: empatik 2). Dikkat: dinlemede
İngilizce talimatlı sürüm aşama 2'nin 1500. adımıydı; Türkçe talimatlı sürümler yayınlanan 500. adım (7 seçim), aynı
ince ayarın 300. adımı (2) ve yalnız Türkçe talimatla sürdürülmüş, yayınlanmayan bir ince ayar (3). Dinlenen
talimatların yarısı `pick`'in ifadeleri değildi: canına, Acele, neşeli, üzgün ve empatik cümlelerinde eğitimde
olmayan ifadeler vardı (ör. "Speaking softly through tears, sad and subdued."); `pick`'in ifadeleri yalnız kızgın
cümlede ve Cam, Cuma, Cesaret, Cidden cümlelerinde kullanıldı. Yani saf duyguda İngilizcenin 6/6'sının yalnız 2'si
(kızgın) `pick`'in ifadesiyle dinlendi. Yayınlanan modelin İngilizce talimatı kulakla dinlenmedi; probda perdesi
1500. adımınkinden yüksek ya da ona yakın, "c"yi daha da kötü okuyor (1/14, 1500. adım 3/14) / in all English 8 of
20, Turkish 12 of 20; the rule agrees with 18 of the 20 judgements (not with empathetic 2). Note: in these rounds
the English-instruction version was step 1500 of stage 2; the Turkish-instruction versions were the released step
500 (7 picks), step 300 of the same fine-tune (2) and an unreleased fine-tune continued on Turkish instructions only
(3).
Half of the judged instructions were not `pick`'s wordings: the canına, Acele, cheerful, sad and empathetic
sentences had wordings that were never trained (e.g. "Speaking softly through tears, sad and subdued."); `pick`'s
own were used only on the angry and the Cam, Cuma, Cesaret, Cidden sentences. So of English's 6 of 6 on pure
emotions only 2 (angry) were judged with `pick`'s wording. The released model's English instructions were not judged
by ear; in the probe their pitch is higher than or close to step 1500's and their "c" even worse (1 of 14, step 1500
3 of 14).

Eğitimdeki ifadeler (`instructions_tr.json`'daki karşılıklarıyla); `pick` her duygunun "Önerilen" ifadesini kullanır,
kendi talimatınız için en güvenlisi bunlar / the training wordings (their Turkish side in `instructions_tr.json`);
`pick` uses each emotion's "Recommended" one, and they are the safest for your own instruction:

{{instructions_tr_table}}

- Talimat parantez içinde, metnin hemen önünde, boşluksuz (`compose`); trnorm'dan **geçmez**, yalnız metin
  normalize edilir. Ölçümlerde model Türkçe talimatın sözcüklerini okumadı (Whisper metninde talimat sözcüğü yok) /
  the instruction goes in parentheses right in front of the text, no space (`compose`), and does **not** go through
  trnorm; only the text is normalized. In our measurements the model never read the Turkish instruction's words aloud
  (no instruction word in Whisper's text).
- Kendi ifadenizde tablodakiler en güvenlisi; başka ifadeler de çoğu zaman çalışır. Hafif bir cümleye "Biraz ...",
  güçlüsüne "Çok ..." türü ifade uyar / for your own instruction the table's wordings are the safest; other wordings
  often work too. "Biraz ..." (slightly) suits a mild sentence, "Çok ..." (very) a strong one.
- Gülme: `laughing` ve metinde "haha" / "hahaha"; her denemede gerçek gülme olmayabilir, birkaç tohum deneyin.
  Türkçe gülme talimatı eğitimde vardı ama hiç ölçülmedi; ilk iki kelimeden biri "c" ile başlayan bir gülme satırı
  ("Hahaha, cidden mi?") Türkçe talimat alır; gülme zayıf çıkarsa `prefer="en"` / laughter: `laughing` and "haha" /
  "hahaha" in the text; not every try laughs for real, try a few seeds. The Turkish laughing instruction was trained
  but never measured; a laughter line with one of its first two words starting with "c" ("Hahaha, cidden mi?") gets
  it, so if the laugh comes out weak, use `prefer="en"`.
- Ünlemler: duygu talimatıyla "Vay" çoğu zaman "vey" çıkıyordu; `pick` talimatlı metinde modele "Vaay" yazar (önceki
  sürümde İngilizce talimatla 8/8 doğru). Yayınlanan modelin probunda "Vay" yazımının 4 klibinin (2 İngilizce, 2
  Türkçe talimat) hiçbirinde "vey" yok: IPA'da 3 "vay", 1 "vaa", Whisper 4/4 "Vay"; "Vaay" bu modelde ölçülmedi.
  "Vây", "Vaaay" yazımları okunuşu bozar; talimatsız metne dokunulmaz / interjections: with an emotion instruction
  "Vay" often came out as "vey"; `pick` sends "Vaay" in instructed text (8 of 8 right on the previous version with
  English instructions). In the released model's probe none of the 4 clips of the "Vay" spelling (2 English, 2 Turkish
  instructions) said "vey": IPA 3 "vay", 1 "vaa", Whisper "Vay" 4 of 4; "Vaay" was not measured on this model.
  Spellings like "Vây" or "Vaaay" hurt; plain text is left as it is.
- `pause_clean.clean(wav, sr)` → (ses / audio, rapor / report): yalnız DC'yi atar ve iki yanında konuşma olan
  duraklardaki nefes tepelerini kısar (tepe L-55 dB'e, en çok 25 dB); kelimenin sönüşü, patlamalı ünsüzün patlaması ve
  baştaki / sondaki sessizlik korunur; belirlenimci, CPU, 48 kHz'lik 18 sn'lik klipte ~0,16 sn. Yalnız sese
  bakar; başka bir modelin çıktısına da uygulanabilir / removes only the DC and turns the breath humps in pauses
  between words down (peak to L-55 dB, at most 25 dB); word decays, stop bursts and leading / trailing silence are
  kept; deterministic, CPU, ~0.16 s for an 18 s clip at 48 kHz. It works on the audio alone, any model's output.
- Ayar / settings: önerilen **16 adım, cfg 2.0**; hız gerekiyorsa 10 adım. Yukarıdaki kod aşağıdaki ölçümün ayarını
  kullanır (`cfg_value={{cfg}}`, `inference_timesteps={{timesteps}}`) / recommended **16 timesteps, cfg 2.0**; 10 for
  speed. The code above uses the settings of the evaluation below. Ölçümler `optimize=False` ile alındı / measured with
  `optimize=False`.
- Referans klip (`reference_wav_path`) vermeyin: ses LoRA'nın içinde / pass no reference clip: the voice is in the LoRA.
- PyPI'deki `voxcpm` 2.0.3 de bu kodla çalışır: `lora_config` açıkça veriliyor (2.0.3 `lora_config.json`'u kendisi
  okumaz). 2.0.3'te `generate()` `seed` almaz; yukarıdaki commit alır / PyPI `voxcpm` 2.0.3 also runs this code
  because `lora_config` is passed explicitly (2.0.3 does not read `lora_config.json` itself); its `generate()` has no
  `seed`, the pinned commit does.

## Bilinen sorunlar / Known issues

- **İngilizce talimat ve "c" / English instructions and "c":** İngilizce bir talimattan sonra cümle başındaki "c" çoğu
  zaman İngilizce, /k/, /s/, /z/ ya da /tʃ/ gibi okunur: "Vay canına" → "vey kenına", "Acele" → "Asele", "Cam" →
  "Kam", "Cesaret" → "Sesaret", "Cidden" → "Zidden"; taban VoxCPM2 de aynısını yapar. Prob (7 cümle × 2 tohum, IPA ile
  c kontrolü): yayınlanan modelde c doğru İngilizce talimatla 1/14, Türkçe talimatla 13/14 (kaçan: bir kez "Acele" →
  "Açele"). `pick` ilk iki kelimenin başına bakar; cümlenin ilerisindeki c çoğu zaman doğru (yayınlanan modelde
  İngilizce talimatla 8'de 7: cadde 2/2, yapacağız 2/2, halledeceğiz 2/2, birincisi 1/2); başka bir "c" yanlış
  çıkarsa `prefer="tr"` / after an English instruction a "c" near the start is often read the English way, like /k/,
  /s/, /z/ or /tʃ/; base VoxCPM2 does the same. Probe (7 sentences × 2 seeds, IPA check of the c): on the released
  model c right 1 of 14 with English, 13 of 14 with Turkish instructions (the miss: "Acele" → "Açele" once). `pick`
  looks at how the first two words begin; a "c" further on is mostly right (7 of 8 with English instructions on the
  released model: cadde 2/2, yapacağız 2/2, halledeceğiz 2/2, birincisi 1/2); if another one comes out wrong, use
  `prefer="tr"`.
- **Türkçe talimatla duygu daha hafif / milder emotion with Turkish instructions:** dinleyici kızgın, neşeli ve üzgün
  cümlelerde 6/6 İngilizce talimatlı sürümü seçti (Türkçede duygu zayıf ya da yapmacık); probda yayınlanan modelde
  (cümle başına 2 klip) perde Türkçe talimatla kızgın 312, neşeli 284, üzgün 190 Hz; İngilizceyle 325, 341, 223 Hz.
  Bu yüzden `pick`, ilk iki kelimeden biri "c" ile başlamıyorsa İngilizceyi seçer / the listener picked the
  English-instruction version 6 of 6 times on angry, cheerful and sad lines (weak or forced in Turkish); in the probe
  (2 clips per sentence) the released model's pitch with Turkish instructions was angry 312, cheerful 284, sad 190 Hz,
  with English ones 325, 341, 223 Hz. That is why `pick` takes English unless one of the first two words starts with
  a "c".
- **"Acele" (sakin / calm):** İngilizce talimatla "Asele"; Türkçe talimatla "Aacele" ya da bir kez "Açele" ve daha
  az sakin; dinleyici iki turda da İngilizceyi seçti. "c" kelimenin içinde olduğu için `pick` İngilizceyi seçer; "c"
  önemliyse `prefer="tr"` / with an English instruction "Asele"; with a Turkish one "Aacele" or once "Açele", and
  less calm; the listener picked English both times. The "c" is inside the word, so `pick` takes English; if the "c"
  matters, use `prefer="tr"`.
- **Türkçe talimat ve "c"den sonraki ünlü / Turkish instructions and the vowel after "c":** probda Whisper yayınlanan
  modelin iki Türkçe talimatlı "Cam" klibinde de "Cem" yazdı (IPA'da biri /ɛ/, öbürü /aː/); 2. turda dinleyici de
  "Cem" duydu. İnce ayarın 300. adımında da 2/2 "Cem"; 1500. adım Türkçe talimatla 2/2 "Cam" / in the probe Whisper
  wrote "Cem" for both of the released model's Turkish-instruction "Cam" clips (IPA: one /ɛ/, the other /aː/); in
  round 2 the listener heard "Cem" too. Step 300 of the fine-tune: "Cem" 2 of 2 as well; step 1500 with a Turkish
  instruction: "Cam" 2 of 2.
- **İngilizce talimat ve ses rengi / English instructions and timbre:** bazı İngilizce talimatlı duygulu klipler
  "telefon gibi" parlak ve cızırtılı (1-5 kHz şişmesi, bas kaybı, yüksek perde); 4. turda dinleyicinin böyle dediği 4
  İngilizce talimatlı cümledeki 12 klibimizin hepsinde bu iz var, diğer 44 klibimizin hiçbirinde yok. Probda iz
  yayınlanan modelde İngilizce talimatla 7/20, Türkçe talimatla 4/20 klipte (1500. adım: 8/20 ve 0/20) / some
  emotional clips with an English instruction sound bright and harsh, "like a phone" (a 1-5 kHz bulge, less bass,
  higher pitch); all 12 of our clips in the 4 English-instruction sentences the listener called so in round 4 carry
  this trace, none of our other 44. In the probe the trace was in 7 of 20 clips with English and 4 of 20 with Turkish
  instructions on the released model (step 1500: 8 of 20 and 0 of 20).
- **Nefes / breath:** noktalamadan sonraki duraklarda nefes sesi kalır, çoğu taban modelden gelir (aşama 2'nin 1500.
  adımı, 20 uzun cümle: noktalama duraklarının %29'unda uzun nefes); `pause_clean.py` bunları kısar (31/108 → 0) /
  breath noise stays in the pauses after punctuation, mostly from the base model (step 1500 of stage 2, 20 long
  sentences: a sustained breath in 29 % of the punctuation pauses); `pause_clean.py` turns them down (31 of 108 → 0).

## İnsan dinlemesi / Human listening

Kör dinleme, **tek** anadil dinleyici, 4. tur: 20 cümle (3. turun 14 cümlesi: zor yazılar, sorular, duygulu
satırlar; 4 uzun ayrılmış cümle; 2 "c" cümlesi), her cümlede en iyi bulunan seçildi. Yayınlanan 500. adım bu turdan
sonra eğitildi; turda ince ayarın başladığı 1500. adım var / blind, **one** native listener, round 4: 20 sentences
(round 3's 14: hard written forms, questions, emotional lines; 4 long held-out sentences; 2 "c" sentences), the best
one picked per sentence. The released step 500 was trained after this round; the round has step 1500, where its
fine-tune started.

| Sistem / System | Seçildi / Picked |
|---|---:|
| Aşama 2, 1500. adım / stage 2, step 1500 (`voxcpm2-f022-s3/step_0001500`) | 8/20 |
| Önceki sürüm / the previous version (`voxcpm2-f022-s2/step_0001500`) | 6/20 |
| Aşama 2, 2000. adım / stage 2, step 2000 | 4/20 |
| [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning) (8.6M, Apache-2.0) + trnorm | 2/20 |

- 3. turun aynı 14 cümlesinde EMA Lightning 8'den 2'ye indi; kazandığı iki cümle bizim "c"yi bozduğumuz cümlelerdi
  (İngilizce talimat: "vey kenına", "Asele"). 4 uzun cümlenin 3'ünde 1500. adım seçildi / on round 3's same 14
  sentences EMA Lightning fell from 8 to 2, both on sentences where we broke the "c" (English instruction); step
  1500 was picked on 3 of the 4 long sentences.

Talimat dinlemesi, iki tur (tek dinleyici, her turda 10 duygulu cümle, her cümlede en iyi; cümle cümle sonuç
"Talimat seçimi"nde) / instruction listening, two rounds (one listener, 10 emotional sentences per round, best per
sentence; per sentence under "Choosing the instruction"):

| Sistem + talimat / System + instruction | Seçildi / Picked |
|---|---:|
| Aşama 2, 1500. adım + İngilizce / stage 2, step 1500 + English (2 tur / rounds) | 8/20 |
| **Yayınlanan / released:** iki dilli ince ayar, 500. adım + Türkçe / two-language fine-tune, step 500 + Turkish (2 tur / rounds) | 7/20 |
| Aynı ince ayarın 300. adımı + Türkçe / its step 300 + Turkish (1. tur / round 1) | 2/10 |
| 500. adımdan yalnız Türkçe talimatla sürdürülen 800 adım + Türkçe (yayınlanmadı) / 800 more steps from step 500 on Turkish instructions only + Turkish (not released; 2. tur / round 2) | 3/10 |

- Yalnız Türkçe sürdürülen ince ayarın 3 seçimi "c"li cümlelerdeydi, ama dinleyici "çok şaşkın ama tiz, telefon gibi"
  dedi; probda sert / parlak iz 6/20, "c" 12/14 / the Turkish-only continuation's 3 picks were sentences with a "c",
  but the listener called it "very surprised but shrill, like a phone"; in the probe the harsh / bright trace was in 6
  of 20 clips, "c" right 12 of 14.
- Nefes A/B (4 uzun cümle, aynı klibin özgün ve `pause_clean` sürümü): nefes 4/4 temizlenmişte daha az, doğallık 3
  aynı / 1 daha iyi / breath A/B (4 long sentences, the same clip raw and through `pause_clean`): less breath in the
  cleaned one 4 of 4, naturalness the same 3, better 1.
- Yayınlanan modelin düz okuması kulakla dinlenmedi; aşağıdaki ölçüme bakın / the released model's plain reading was
  not judged by ear; see the evaluation below.
- Tek dinleyici ve 10-20 cümle küçük bir örnektir; istatistiksel bir sonuç değil, bir işarettir / one listener and
  10-20 sentences are a small sample: an indication, not a statistical result.

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
iyi / higher is better. Ölçüm talimatsızdır / the evaluation uses no instruction.

> UTMOS doğallığı **tahmin eder** (bir modelin tahmini, insan puanı değil), WER ise **anlaşılırlığı** ölçer. /
> UTMOS **predicts** naturalness (a model's estimate, not human ratings); WER measures **intelligibility**.

| Model | Çıkarım / Inference (timesteps / cfg) | TR-Zor WER | TR-Zor hatasız / error-free | Ayrılmış / Held-out WER | Ayrılmış hatasız / error-free | UTMOS TR-Zor | UTMOS ayrılmış / held-out | RTF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{{results_rows}}

RTF: sentez süresi / ses süresi, {{eval_gpu}} üzerinde, cümle cümle sentez (`pause_clean` hariç); 1'in altı gerçek
zamandan hızlı. Aynı ayarla farklı konteynerlerde ölçülen RTF belirgin değişebilir / synthesis time over audio
duration on {{eval_gpu}}, one sentence at a time (`pause_clean` not included), below 1 is faster than real time;
the same settings can measure quite differently from one container to the next. "—": ölçülmedi / not measured.
{{pause_clean_note}}
{{eval_note}}

### Soru tonlaması / Question intonation (TR-Soru)

**TR-Soru** ([`tr_soru.jsonl`](https://github.com/ozguradmin/ahenk-turkish-tts/blob/main/benchmarks/tts/sets/tr_soru.jsonl)):
35 satır; evet-hayır soruları (mI), "değil mi", soru sözcüklü ve mI'siz yükselen sorular, eşli düz cümleleriyle /
35 rows: yes-no questions (mI), "değil mi", wh-questions and rising questions without mI, with matched statements.
`score_intonation.py` perde tepesinin yerini ve yüksekliğini ölçer / measures where the pitch peak falls and how
high. **Eşler / pairs**: sorunun tepesi eşli düz cümleninkinden yüksek olan çiftler / pairs whose question peaks
higher than its statement. **AUC**: mI sorularının tepesi düz cümlelerden ne kadar ayrışıyor (0,5 fark yok, 1 tam
ayrım) / how well the peaks of mI questions separate from the statements (0.5 none, 1 complete). **Kalıba uyan
sorular / questions matching the pattern**: Türkçe kalıba uyan soruların payı; katı bir ölçü / the share of
questions with the Turkish shape; a strict measure.

{{intonation_table}}

### TR-Zor kategorileri / categories

WER % (hatasız / toplam cümle) / WER % (error-free / total sentences).

{{category_table}}

## Örnekler / Samples

{{samples}}

### Duygu ve gülme / emotion and laughter

Yayınlanan kontrol noktasından; talimatı `instruction_pick.pick` seçti (ilk iki kelimeden biri "c" ile başlıyorsa Türkçe, yoksa İngilizce), çıktı `pause_clean`'den geçti; her cümlenin iki tohumundan iyisini dinleyici seçti / from the released checkpoint; the instruction was chosen by `instruction_pick.pick` (Turkish when one of the first two words starts with "c", else English), the output went through `pause_clean`; of two seeds per sentence the listener picked the better one.

{{extra_samples}}

## Eğitim / Training

- Temel model / base model: `openbmb/VoxCPM2`; resmi ince ayar betiği / official fine-tuning script
  [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) `{{voxcpm_commit}}`.
- **Aşama 1 / stage 1** (önceki sürümlerle aynı LoRA / the same LoRA as the previous versions' stage 1): Alania
  (`cc-by`, commit `277dfc8a23`), 25 saat, 23.706 klip, 1.177 kadın sesi; yalnızca düz okumalar, `qc_cer == 0`,
  `qc_utmos >= 3.7`, FineWeb-2 metni en çok %25; TR-Zor ve Antalia'nın ayrılmış cümleleri hariç / 25 h, 23,706
  clips, 1,177 female voices; plain reads only, zero CER, UTMOS >= 3.7, at most 25 % FineWeb-2 text; TR-Zor and
  Antalia's held-out sentences excluded. LoRA r={{lora_r}}, lr 1e-4, batch 16, 2.000 adım / 2,000 steps, A100 80 GB.
- **Ses / voice f022:** VoxCPM2 ses tasarımı, referanssız, şu İngilizce tariften / VoxCPM2 voice design, no
  reference, from this English description: *"Female speaker, in their fifties, a counsellor; low-pitched and
  velvety; patient and polite delivery at a measured pace; recorded close to the microphone in a quiet studio."*
  Tarif hiçbir gerçek kişiyi anmaz ve taklit istemez; ~150 tarif × 2 tohumluk bir taramadan, UTMOS, Whisper CER ve
  iki kör dinleme turuyla seçildi / the description names no real person and asks for no imitation; picked from a
  screen of ~150 descriptions × 2 seeds by UTMOS, Whisper CER and two blind listening rounds.
- **Aşama 2 verisi / stage-2 data:** taban VoxCPM2 f022'nin referans klibini klonlayarak okudu / base VoxCPM2 read it
  cloning f022's reference clip. İlk derlem / first corpus: 3,53 sa / 3.53 h (düz cümle / plain sentences 2.55, soru /
  questions 0.29, duygu / emotions 0.54, ünlem / interjections 0.10, kısa cevap / short replies 0.05) + 48 gülme klibi
  / laughter clips; c'si yanlış okunmuş 6 klip IPA denetimiyle çıkarıldı / 6 clips whose "c" was misread dropped by an
  IPA audit. İkinci derlem / second corpus: 1,015 sa / 1.015 h, 620 klip / clips (günlük ve sayılı düz cümleler /
  everyday and number sentences 0.51, uzun cümleler / long sentences 0.47, kelime başı "c"li duygulu cümleler /
  emotional sentences with a word-initial "c" 47 klip / clips; bunların 21'i talimatsız üretildi, duyguyu o duygunun
  f022 referans klibi taşır, eğitim metninde talimat yine var / 21 of them rendered without the instruction, the
  emotion carried by that emotion's f022 reference clip, the training text keeps the instruction). Toplam 4.274 klip,
  4,58 sa / in all 4,274 clips, 4.58 h. Duygulu satırlar o duygunun kendi f022 referans klibini klonladı / emotional
  lines cloned a per-emotion f022 reference.
- **Ret örneklemesi / rejection sampling:** her klip şunların hepsini geçti, yoksa yeni tohumla yeniden / every clip
  passed all of these, else a new seed: Whisper large-v3-turbo CER 0 ve / and p(tr) ≥ 0.95; başta fazladan ses
  yok, sondaki kısa fazladan ses kesildi / no stray sound at the start, a short stray sound at the end cut off; ilk
  kelime uzaması, aşırı "ş/s", konuşma hızı sınırları / first-word stretch, sibilance and speaking-rate limits;
  f022'ye ECAPA benzerliği / ECAPA similarity to f022 ≥ 0.6; UTMOS kategorisinin alt %20'sinde değil / not in the
  bottom 20 % of its category; soru tonlaması kalıba uyar / question intonation matches its pattern; duygu prozodisi
  doğru yönde / emotion prosody in the right direction; uzun kliplerde duraklama sınırları / pause limits on long
  clips; talimatlı ve "c"li kliplerde IPA ile c kontrolü / an IPA check of the "c" on instructed clips with a "c".
- **Aşama 2 eğitimi / stage-2 training** (`voxcpm2-f022-s3`): aşama 1'in LoRA'sından / from the stage-1 LoRA, lr
  1e-4 (kosinüs, 2.000 adım / cosine over 2,000 steps), batch 8 × gradyan birikimi / gradient accumulation 2, A100
  80 GB; 1500. adım dinlemeyle seçildi ve iki dilli ince ayarın başlangıcı oldu / step 1500 picked by listening, the
  start of the two-language fine-tune. Metinler trnorm'un okunuş formunda; duygu ve gülme satırlarında `(talimat)` +
  okunuş / transcripts in trnorm's spoken form; emotion and laughter rows as `(instruction)` + spoken form.
- **İki dilli talimat ince ayarı / two-language instruction fine-tune** (`voxcpm2-f022-s3tr`, **yayınlanan /
  released**): aşama 2'nin 1500. adımından / from step 1500 of stage 2, her talimatlı satır bir kez İngilizce, bir kez
  Türkçe talimatla (aynı klip; 4.916 satır, 642 + 642 talimatlı) / each instructed row once with its English and once
  with its Turkish instruction (the same clip; 4,916 rows, 642 + 642 instructed), yeni AdamW / a fresh AdamW, lr 5e-5
  (ısınma / warmup 50, kosinüsle 500. adımda 0 / cosine to 0 at step 500), batch 8 × 2, 500 adım / steps; 500. adım
  yayınlandı / step 500 released.
- Denenip yayınlanmayan / tried, not released: 500. adımdan yalnız Türkçe talimatlı satırlarla 800 adım daha
  (`voxcpm2-f022-s3tr2`): Türkçe talimatla duygu daha güçlü (probda kızgın perdesi 352 Hz) ama ses daha tiz / sert
  (probda iz 6/20) ve "c" 12/14; dinlemede 3/10 / 800 more steps from step 500 on the Turkish-instruction rows only:
  stronger emotion with Turkish instructions (probe angry pitch 352 Hz) but a shriller / harsher voice (trace in 6 of
  20 probe clips) and "c" 12 of 14; 3 of 10 in listening.
- LoRA hedefleri / targets: {{lora_targets}}.
- Tarif / recipe: [ahenk-turkish-tts/tts/voxcpm](https://github.com/ozguradmin/ahenk-turkish-tts/tree/main/tts/voxcpm) (Apache-2.0);
  derlem / corpus: `render_corpus.py`, metin bankası / text bank: `tts/voxcpm/textbank`, Türkçe talimat tablosu /
  Turkish instruction table: `tts/voxcpm/instructions_tr.json`.

## Eğitim verisi ve atıf / Training data and attribution

- Alania Turkish Synthetic Speech by PatientDesk AI (https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr),
  licensed under CC BY 4.0 — aşama 1 verisi; tamamı yapay zekâ ile üretilmiş (VoxCPM2) konuşma, gerçek bir kişinin
  sesi yok. Duygu ve gülme talimatlarına dair bilgimiz de bu veri setinden geldi / stage-1 data; entirely
  AI-generated speech (VoxCPM2), no real person's voice; also the source of our know-how on emotion and laughter
  instructions. Değişiklik / changes: filtrelenmiş alt küme / filtered subset, 48 → 16 kHz.
  - Includes Turkish web text from FineWeb-2 (https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), used under
    ODC-By 1.0 and subject to Common Crawl's terms of use.
- Alania Turkish Domain Text by PatientDesk AI (https://huggingface.co/datasets/cloud0day3/alania-domain-text-tr),
  licensed under CC BY 4.0 — aşama 2'nin düz, sayılı ve uzun cümleleri / the plain, number and long sentences of
  stage 2 (commit `20fd903fe6`). Değişiklik / changes: kategori, uzunluk, soru ve dil kimliği filtreleri; uzun
  cümleler aynı üretim isteminin 1-3 ardışık satırı; trnorm ile okunuş formuna çevrildi; VoxCPM2 f022 sesiyle
  seslendirdi / category, length, question and language-ID filters; long items join 1-3 consecutive lines of one
  generation prompt; converted to spoken form with trnorm; voiced by VoxCPM2 in the f022 voice.
- Ses / voice: Voice designed from a text description with VoxCPM2 (OpenBMB, https://huggingface.co/openbmb/VoxCPM2,
  Apache-2.0); no real person's voice. / Ses, VoxCPM2 ile metin tarifinden tasarlandı; gerçek bir kişinin sesi
  değildir.
- Soru, duygu, kelime başı "c", ünlem, kısa cevap ve gülme metinleri ile Türkçe talimatlar: kendi metin bankamız,
  bu proje için yazıldı / questions, emotional lines, word-initial "c" lines, interjections, short replies and
  laughter lines, and the Turkish instructions: our own, written for this project
  ([ahenk-turkish-tts/tts/voxcpm](https://github.com/ozguradmin/ahenk-turkish-tts/tree/main/tts/voxcpm), Apache-2.0).
- Temel model / base model: [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (OpenBMB), Apache-2.0.
- Ön uç, `pause_clean.py` ve tarif / front-end, `pause_clean.py` and recipe:
  [ahenk-turkish-tts](https://github.com/ozguradmin/ahenk-turkish-tts) (Apache-2.0).
- Antalia: **bu sürümün eğitiminde Antalia sesi ya da metni yok**. Ölçümdeki "ayrılmış" cümlelerin metni Antalia veri
  setinden (Patientdesk.ai, CC BY 4.0, https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus) / **no
  Antalia audio or text is in this version's training**. The text of the "held-out" evaluation sentences comes from
  the Antalia dataset.

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
  `pause_clean.py` nefesleri kısar ama sesi daha "insan gibi" göstermek için değil, dinlemeyi rahatlatmak için;
  etiket yine gerekir / `pause_clean.py` turns breaths down for easier listening, not to pass the voice off as human;
  the label is still needed.
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
- Harf harf okunan kısaltmalar ve telefon numaraları en zayıf kategoriler arasında; kategori tablosuna bakın /
  spelled-out acronyms and phone numbers are among the weakest categories, see the category table.
- Uzun metni cümlelere bölüp tek tek sentezleyin; ayrılmış setin (12-19 sn) UTMOS'una bakın / split long text into
  sentences and synthesize them one by one; see the held-out UTMOS (12-19 s sentences).
- Duygular: talimatın dili duygunun gücünü ve "c"nin okunuşunu etkiler; `instruction_pick.py` 20 dinleme kararının
  18'iyle uyuşan basit bir kuraldır, her cümle için doğruyu bilmez (Bilinen sorunlar); gülme her denemede gerçek
  gülme olmayabilir, birkaç tohum deneyin / emotions: the instruction's language affects how strong they come out and
  how a "c" is read; `instruction_pick.py` is a simple rule that agrees with 18 of the 20 listening judgements and
  does not know the best choice for every sentence (Known issues); laughter is not real laughter every time, try a
  few seeds.
- `pause_clean.py` yalnız iki yanında konuşma olan durakları ve nefes benzeri gürültüyü (0,7-5 kHz) hedefler;
  kelimeye yapışık nefesli bitişleri bilerek korur / only targets pauses between words and breath-like noise
  (0.7-5 kHz); breathy word endings are kept on purpose.
- WER bir ASR yargıcıyla ölçüldü (faster-whisper large-v3-turbo); yargıcın kendi hataları sonuca karışır. İnsan
  dinlemesi tek dinleyiciyle yapıldı, sistematik bir MOS testi yapılmadı / WER is measured by an ASR judge whose own
  errors are part of the score; the human listening had one listener, no systematic MOS test was run.
- GPU gerekir: yukarıdaki ayarlarla RTF {{rtf}} ({{eval_gpu}}); 1'in üstü gerçek zamandan yavaş demektir, aynı
  ayarla konteynerden konteynere belirgin değişebilir; daha az adım (`inference_timesteps`) daha hızlıdır. CPU'da
  ölçülmedi / needs a GPU: RTF {{rtf}} on {{eval_gpu}} with the settings above (above 1 is slower than real time;
  it can measure quite differently from one container to the next); fewer `inference_timesteps` are faster. Not
  measured on CPU.
- Metin `trnorm` ile normalize edilmeden verilirse sayılar, tarihler ve kısaltmalar yanlış okunabilir / without
  `trnorm`, numbers, dates and abbreviations may be misread.

## İletişim / Contact

Sorular, izin ve kötüye kullanım bildirimleri / questions, permissions and abuse reports: destek@ozgurguler.tech
