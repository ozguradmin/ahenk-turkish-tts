# Küçük model planı ve Fish Audio Drama 3 kararı (2026-10-14)

Bu belge yalnızca araştırmadır. Hiçbir ücretli servis çalıştırılmadı, model ağırlığı indirilmedi ve repoda değişiklik yapılmadı.
Önce kısa Türkçe özet var, ardından İngilizce teknik ek ve kaynaklar.

---

## A. Fish Audio'nun yeni modeli (Drama 3): kullanmalı mıyız?

**Kısa cevap: hayır. Taban model ya da öğretmen olarak kullanmıyoruz. İstersek yalnızca karşılaştırma için dinleriz.**

- **Ne:** "Fish Audio Drama 3", 13 Ekim 2026'da açık ağırlıklı olarak çıktı. 35 milyar parametreli; her kelimede yaklaşık 3 milyarı çalışıyor. Türkçede gerçekten çok iyi okuyor. Fish'in kendi testinde Türkçe hata oranı %0,52. Bu sayı bizim TR-Zor ölçümümüzle karşılaştırılamaz, çünkü cümleler ve ölçüm yöntemi farklı.
- **Neden kullanmıyoruz:**
  1. **Lisans açık değil.** Ticari kullanım yalnızca yıllık ciro 500 bin doların altındayken ücretsiz. Başkalarına metin okutan bir hizmet ya da API açmak için Fish'ten ayrı izin gerekiyor. Ayrıca her üründe "Built with Fish Audio" yazmak zorunlu. Bizim Apache-2.0 açıklığımızla uyuşmuyor.
  2. **Öğretmen olamaz.** Lisans, Drama 3'ün çıktısıyla başka bir ses modeli "eğitmeyi, damıtmayı, iyileştirmeyi" açıkça yasaklıyor. Kural çiğnenirse lisans kendiliğinden düşüyor. Fish'in API kullanım şartlarında da "rakip model geliştirmek için çıktı kullanma" yasağı var. Google Cloud projemizin kısıtlanmasına yol açan kural da bu türdendi.
  3. **Donanım ve bütçe yetmez.** En az 48 GB FP8 destekli GPU istiyor (H100/H200 sınıfı). Bizim A100'ümüzde çalışacağı belgelenmemiş. İnce ayar kodu da yayınlanmadı. 35 milyar parametrelik bir eğitim aylık 6-60 dolarlık bütçemizin çok üstünde.
- **Ne yapabiliriz:** Kasım'dan sonra istersek birkaç cümleyi Drama 3 ile okutup kör dinlemede rakip olarak koyarız. Kurallar:
  - Çıkan sesler yalnızca puanlama ve dinleme için kullanılır.
  - Eğitim verisine, ödül modeline ya da hakem ayarına asla girmez.
  - Yalnızca kendi tasarım sesimiz f022 kullanılır; gerçek bir kişinin sesi kullanılmaz.
  - Önceliği düşük.
- **Önceki Fish modeli (S2-Pro):** Lisansı tamamen ticari olmayan kullanımla sınırlı, çıktıyla eğitim de yasak. Bilgisayardaki önbellekte yalnızca adı duruyor, ağırlıkları yok.

## B. Küçük model (telefonda ve tarayıcıda çalışan sürüm) planı

### Fikir
Büyük modelimiz f022 sesiyle konuşma üretir, küçük model bu konuşmadan öğrenir. Hedef yaklaşık 10 milyon parametre ve 20-40 MB dosya: internetsiz, telefonun işlemcisinde gerçek zamandan hızlı çalışan, aynı sesli bir model. Doğallıkta büyük modeli geçemez ama düz okumada ona çok yaklaşabilir. Bu tür küçük modeller kelime atlama ya da ekleme hatasını da daha az yapar.

### Önemli yeni bulgu
EMA Lightning, sesi büyük modelimizin (VoxCPM2) kullandığı ses kodlamasıyla üretiyor. Bunu bilgisayarda bedavaya denedik: bizim f022 sesimizi EMA'nın ses çözücüsü anlaşılır biçimde geri çalabiliyor. Bu üç şey kazandırıyor:
- Yeni bir ses çözücü yapmamız gerekmiyor.
- Öğretmenin ürettiği kodlar doğrudan öğrencinin eğitim hedefi olabiliyor.
- EMA'nın telefon ve tarayıcı sürümleri (ONNX/Rust, iPhone) bizim modelimize de uyabilir.

### Üç aday (en iyiden başlayarak)
1. **EMA mimarisiyle kendi öğrencimiz (önerilen).** Yaklaşık 8,6 milyon parametre, Apache-2.0.
   - Türkçeyi zaten biliyor ve büyük modelimizle aynı ses kodlamasını kullanıyor.
   - Telefonda çalıştığı kanıtlandı.
   - Yazarın söz verdiği eğitim kodu henüz yayınlanmadı. Eğitim kodunu biz yazacağız; model kodu açık.
   - Tek soru işareti: EMA'nın eğitildiği 1000 saatlik sesin nereden geldiği belirtilmemiş. Bunu aşağıda "kararlar"da soruyorum.
2. **Matcha-TTS (MIT lisanslı, eğitim kodu hazır, telefon/tarayıcı araçları destekliyor).** EMA yolu takılırsa yedek plan. Eğitim parçalarını (hizalama, süre tahmini) 1. adayda da kullanacağız.
3. **Pocket TTS (Kyutai, 100 milyon parametre).** Duygu ve ifade için ileride ikinci, daha büyük bir seçenek. Türkçe ince ayarları var, eğitim kodu açık. Ama yaklaşık 2,5 kat daha yavaş ve en az 100 saat veri istiyor. Şimdi değil.

Yeni rakip **Antalia-2 Mini** (7,6 milyon parametre, Apache-2.0, VoxCPM2 öğretmeniyle eğitilmiş) tam bizim planımızı yapmış ve iyi sonuç almış. Eğitim kodu yok. Onu yalnızca kıyas için ölçeceğiz.

### İlk ucuz ve belirleyici deney (Ekim, ~$0)
Elimizde zaten temizlenmiş **4,6 saat f022 verisi** var: büyük modeli eğitirken ürettiğimiz derlem. Kelime zamanları da içinde kayıtlı. Deney şu:
1. Bu sesi ses koduna çeviririz. Kendi bilgisayarında, bedava.
2. EMA'nın küçük modelini bu veriyle f022 sesine alıştırırız. Ücretsiz Kaggle GPU'sunda, haftada 30 saat kota var.
3. Sonucu kendi bilgisayarında TR-Zor ile ölçeriz ve kulakla dinleriz.

**Karar kuralı:** Küçük model f022 gibi konuşuyorsa ve TR-Zor hata oranı %3 civarı ya da daha iyiyse, Kasım'da veriyi büyütürüz. Ses benzemiyorsa ya da hata oranı %6'nın üstündeyse dururuz ve yöntemi değiştiririz. Böylece Piper'daki gibi 13 doları boşa harcama riski olmaz.

### Veri planı (Kasım)
- **Ne kadar:** Aynı f022 sesiyle 10-20 saat ek veri:
  - yaklaşık %60 günlük cümle ve sayılı cümle
  - yaklaşık %30 soru
  - 6 duygunun her biri için 400-1000 klip
  - gülme başta yok
- **Nasıl temizlenir:** Her klip hatasız okunmuş olmalı. Başta ve sonda fazladan ses, "c → k" hatası, "telefon gibi" cızırtılı tını ve aşırı nefes olmamalı.
- **Ne kadar tutar:** Ölçülen gerçek maliyet, kullanılabilir saat başına yaklaşık $2. Bunun çoğu üretim değil, kontrol. Kontrolleri hızlandırarak bunu $1'ın altına indirmeyi hedefliyoruz. İlk $2'lık çağrı gerçek fiyatı ölçecek.
- **Kesin yasaklar:** Fish, Gemini, ElevenLabs ya da başka bir şirketin çıktısı veriye girmez. Test cümleleri de eğitime sızmaz; mevcut koruma bunu engelliyor.

### Maliyet tablosu (Modal; A100 ~$3,26/sa, L4 ~$0,80/sa GPU)
| Adım | Nerede | Tahmini maliyet |
|---|---|---|
| Antalia-Mini ve EMA'yı yan yana ölçmek (~30 MB indirme, onayınla) | Kendi bilgisayarında | $0 |
| Eğitim kodunu yazmak ve test etmek | Kod | $0 |
| 4,6 saatlik veriyi ses koduna çevirmek | Kendi bilgisayarında | $0 |
| Deneme eğitimi | Kaggle (ücretsiz) | $0 (Modal'da olsaydı ~$1-3) |
| TR-Zor ölçümü ve kör dinleme | Kendi bilgisayarında | $0 |
| (İsteğe bağlı) Hızlı üreticiye LoRA'yı yükleme denemesi | Modal A100, 10 dk | ~$0,6 |
| 10-20 saat yeni veri | Modal A100 | ~$15-40 (ölçüme göre) |
| Asıl eğitim | Kaggle ya da Modal | $0-7 |
| (Gerekirse) Ses çözücüyü f022'ye alıştırmak | Modal A100, 2-4 sa | ~$7-13 |

### Zaman çizelgesi
- **Ekim (kalan ~$6):**
  - Yalnızca bedava işler: kıyas ölçümü, eğitim kodu, deneme eğitimi, ölçüm ve kör dinleme.
  - En fazla isteğe bağlı $0,6'lık bir deneme.
  - En az $5 yedekte kalır.
- **1-15 Kasım (~$60 yenilenir):**
  - Deneme başarılıysa veriyi parça parça üretiriz. Her çağrı en çok $2-4, toplam en çok ~$30.
  - Ardından asıl eğitim ve 5. kör dinleme: küçük model, EMA ve büyük model yan yana.
- **15-30 Kasım:**
  - Telefon ve tarayıcı paketi.
  - "Yapay ses" işareti (filigran).
  - Model kartı.
  - Gerekirse ses çözücü ince ayarı (~$10).
- **Aralık ve sonrası:** Duygu etiketleri. İsterseniz daha ifadeli ikinci model (Pocket TTS).

### Telefon ve tarayıcı paketi
- **Dosya boyutu:** ~17 MB (yarım hassasiyet). Daha da sıkıştırılırsa ~9 MB.
- **Nerede çalışır:** Tarayıcıda (internetsiz demo sayfası), Android'de, iPhone'da ve bilgisayarda.
- **Ek iş 1:** Metin düzelticimiz trnorm şu an yalnızca Python'da. Telefonda ve tarayıcıda çalışması için JavaScript'e ya da Rust'a çevrilmesi gerekiyor. Bu, en büyük ek iş.
- **Ek iş 2:** AB kuralı gereği çıkan ses makinece "yapay" diye işaretlenmeli. Bunun için küçük bir filigran adımı eklenecek.

### Senden beklenen kararlar
1. **Antalia-Mini kıyası:** ~30 MB'lık açık lisanslı ağırlığı kendi bilgisayarına indirip ücretsiz ölçelim mi? **Önerim: evet.**
2. **EMA ağırlıklarından başlamak:** Bu ağırlıkların eğitim verisinin kaynağı belirsiz. Deneme için kullanmamızda sorun görmüyorum. Yayından önce iki yol var:
   - (a) Yazara veri kaynağını sorarsın; bu bir GitHub mesajı olur, sen gönderirsin.
   - (b) Kaynağı belli Alania verisiyle sıfırdan ön eğitim yaparız (~$5-15 ek).
   
   **Önerim:** Deneme EMA'dan başlasın; yayın kararı denemeden sonra verilsin.
3. **Kaggle:** Eğitim için ücretsiz Kaggle GPU kotasını kullanalım mı? **Önerim: evet.**
4. **Kasım bütçesi:** ~$30 veri, ~$10 eğitim ve ölçüm, ~$20 yedek. Uygun mu?
5. **Drama 3 kıyası:** Kasım'da ~$2-4'lık bir dinleme kıyası yapalım mı? **Önerim: şimdilik hayır.** Önce küçük model.

---

## Technical appendix (English)

### A1. Fish Audio Drama 3: facts, verified 2026-10-14
- **Release.** Released 2026-10-13 as open weights and API. 35B MoE with ~3B active parameters, initialised from Qwen3.5-35B-A3B. S2 codec (44.1 kHz, 10 codebooks). FP8 experts. [blog](https://fish.audio/blog/fish-audio-drama-3/), [HF card](https://huggingface.co/fishaudio/drama-3), [paper arXiv 2610.18010](https://arxiv.org/html/2610.18010)
- **Turkish.** It *is* evaluated. Paper Table 6 (MiniMax multilingual set) reports Turkish WER:

  | System | Turkish WER % |
  |---|---|
  | Drama 3 | 0.52 |
  | Fish S2 | 0.87 |
  | ElevenLabs v4 | 0.87 |
  | FireRedTTS3-Base | 0.92 |
  | Inworld TTS-2 | 1.82 |
  | Soniox TTS RT v2 | 1.91 |
  | OmniVoice | 2.17 |

  No Turkish speaker similarity is reported. The paper does not list the 83 languages; it says "about 80 languages and dialects" in pre-training, and Table 6 covers 24. These numbers are not comparable with our TR-Zor 2.54 % (different text, ASR and normalisation). [paper](https://arxiv.org/html/2610.18010)
- **Hardware.** Needs an NVIDIA GPU with ≥48 GB in FP8, with no CPU, AMD or Apple support. [install](https://speech.fish.audio/install/) A100 has no FP8 tensor cores. With bf16 experts the weights are ~70 GB (estimate), so it is unlikely to fit on our A100-80GB.
- **Fine-tuning.** No fine-tuning code or docs found. [fish-speech](https://github.com/fishaudio/fish-speech)
- **Licence: Fish Audio Community License v1.0, last updated 2026-10-13.** Read in full from [LICENSE](https://huggingface.co/fishaudio/drama-3/raw/main/LICENSE).
  - **§II.** Research and non-commercial use is free. "Evaluation and testing" is named as a Non-Commercial Purpose.
  - **§III(a).** Commercial use only while Annual Revenue (the group's total, from all sources) is below USD 500k.
  - **§III(c).** No Hosted Service at any revenue. Generating speech inside your own product is excluded from that definition, as is a free, unmonetised research demo.
  - **§IV(a).** Every product must show "Built with Fish Audio".
  - **§IV(b).** Bans using the materials, derivatives, "or any output or results" to "create, train, distill, or improve any foundational generative AI model or any speech, audio, or voice generation model". Derivatives of Drama 3 itself are excluded, and training detection/anti-spoof models is allowed.
  - **Correction to one earlier brief.** §V's definition *excludes* "any other model trained or distilled using such output" from Derivative Works. The ban comes from §IV(b), not from the derivative definition.
- **API terms.** Item (j) forbids using output "to develop models that compete with Fish.Audio". [ToS](https://fish.audio/terms)
- **S2-Pro.** Research licence: non-commercial only, plus an output-training ban. [LICENSE](https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md) The local HF cache holds only `refs/main`, not weights.
- **Verdict.**
  - Base model: no.
  - Teacher or data source: no, banned outright.
  - Judge or calibration data: no.
  - Evaluation-only competitor: allowed. This is a commercial project, so even evaluation may count as "internal operations" (a Commercial Purpose). That is still covered by §III(a) while revenue is below $500k.
  - **Rules if it is used for evaluation:** run it on H100-class hardware (Modal H100 list $3.95/h GPU), keep its wavs in a quarantined folder outside every training, reward or calibration path, and clone only f022.

### A2. Student candidates, ranked

| Rank | Candidate | Params | Licence | Training code | Why / risk |
|---|---|---|---|---|---|
| 1 | **EMA-architecture flow-matching student** on VoxCPM2 AudioVAE latents; init from `ema.pt`, or pretrain on Alania | 5.56M acoustic + 3.0M decoder | Apache-2.0 ([card](https://huggingface.co/canberkkkkkk/ema-lightning), [repo](https://github.com/canberk7/ema-lightning)) | None. Promised 2026-10-07 ("In couple of days"), not delivered as of 2026-10-14 ([issue #2](https://github.com/canberk7/ema-lightning/issues/2)). We write it. | Turkish prior; latent space identical to our teacher (verified locally); decoder reusable; ports exist ([ONNX/Rust](https://github.com/sewox/turkish-neural-tts), [Core ML](https://huggingface.co/elphi-si/elphi)). Risk: undisclosed provenance of the ~1000 h corpus behind `ema.pt`. |
| 2 | **Matcha-TTS** (OT-CFM + MAS durations) | 18.2M | MIT ([repo](https://github.com/shivammehta25/Matcha-TTS), [paper](https://arxiv.org/html/2309.03199v2)) | Complete, with ONNX export; sherpa-onnx supports it ([docs](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html)) | Fallback. Its MAS, duration loss and CFM loop are the parts we borrow for #1. Used as-is, it needs a 48 kHz vocoder or a latent output head, and drop espeak (GPL) for trnorm graphemes. |
| 3 | **Pocket TTS** fine-tune (expressive tier) | ~100M; 6-layer distilled student | Code MIT ([GitHub](https://github.com/kyutai-labs/pocket-tts)); weights CC-BY-4.0 ([HF](https://huggingface.co/kyutai/pocket-tts)); Turkish base [alperiox](https://huggingface.co/alperiox/pocket-tts-turkish) MIT, trained on Alania CC-BY-4.0, 262.5 h | `finetune.yaml`, `finetune_language.yaml`, `depth_distill.yaml`; needs ≥100 h and word alignments ([training README](https://raw.githubusercontent.com/kyutai-labs/pocket-tts/main/training/README.md)) | Emotion and expressiveness later. About 2.5× slower on CPU than EMA (our measurement: RTF 0.46 vs 0.18). |

**Reference ceiling (benchmark only).**
- [Antalia-2 Mini](https://huggingface.co/cloud0day3/antalia-mini): 7.62M params, Apache-2.0. Trained on ~1,578 h of single-voice VoxCPM2 synthetic speech, filtered with Whisper, UTMOS and similarity checks.
- Freya-TR-Eval: WER 1.02 %, UTMOS 3.18.
- CPU RTF 0.052 on 8 EPYC cores; runs in ONNX Runtime Web.
- No training code. Its text sources include Turkish Wikipedia (CC BY-SA 4.0), so its share-alike status is the author's matter; we only benchmark it.

### A3. First decisive experiment: "P0 f022-mini" ($0)

**Inputs (already exist)**
- `/data/f022mix` on the Modal Volume: 4,274 clips, 4.58 h. The 6 c-failures are already dropped.
- The manifest stores Whisper word timings (`words`) per clip (`render_corpus.py` CLIP keys).
- For P0, use neutral and question clips only, and drop laughter and English-instruction rows. That should leave about 3-3.5 h (to be confirmed).

**Steps**
1. **Get the data.** `modal volume get` the wav and manifest. This is a transfer with no GPU, so it should be free; check the panel.
2. **Encode latents on local CPU.** Resample to 16 kHz and run the VoxCPM2 AudioVAE encoder (`audiovae.pth`, cached) to get 64×T latents at 25 Hz, raw and unnormalised. A scale sweep showed ×1.0 is best. Measure throughput on 50 clips first.
3. **Prepare text and durations.**
   - Text: trnorm spoken form, lowercased, filtered to EMA's 49-symbol alphabet. Map â/î/û to a/i/u for now, as EMA's vocabulary does.
   - Clips must be ≤10 s and end in `.`, `!` or `?`.
   - Per-word frame counts come from Whisper word timings. Assign inter-word silence to the preceding word, which matches the engine's even spread within words. MAS (Glow-TTS, MIT) is the fallback.
4. **Write the trainer** (new code, with unit tests) around `model.py` (Apache-2.0):
   - Losses: OT-CFM velocity loss on latents, plus an L2 log-duration loss per word.
   - Text-condition dropout p = 0.1, so that CFG works.
   - Initialise from the `ema.pt` student weights. The DMD2 student weights work as a flow-matching init only approximately, so sample with 8-16 Euler or midpoint steps rather than relying on the baked 4-step schedule.
   - Optimiser: AdamW at lr 1e-4, 20-50k steps, batch ~32. Keep an EMA of the weights.
5. **Run two arms on Kaggle T4** (free, 30 GPU-h/week; superseded, see the corrections below):
   - (a) init from `ema.pt`
   - (b) from scratch, as a control that sizes how much data a scratch model needs.
6. **Evaluate on local CPU.**
   - Measures: TR-Zor (155), held-out (20), TR-Soru (35), UTMOS, ECAPA similarity to the f022 reference, CPU RTF.
   - Decode with EMA `decoder.pt`, and also with the VoxCPM2 AudioVAE decoder as the upper bound.
   - Then a short blind page with the student, EMA and s3tr-500.

**Go/stop gate**
- **Go (scale data):** arm (a) reaches TR-Zor ≤ ~3 % (EMA is 1.9 % with trnorm), ECAPA similarity to f022 ≥ 0.70, UTMOS ≥ 3.3, and listening says "this is f022".
- **Stop or change method:** WER > 6 %, or the voice does not move toward f022.
- **Fix the decoder first:** latents look right but audio is muffled. The VoxCPM2 decoder sounds fine while `decoder.pt` does not.

### A4. Data plan (from Nov 1)

**Teacher**
- Base VoxCPM2 cloning the f022-s0 reference, through Nano-vLLM: the existing `render_corpus.py` path that produced f022corpus. This is the same voice identity the LoRA learned, and the CER-0 gate removes the base model's extra reading errors.
- Optional ~$0.6 probe: merge s3tr-500 LoRA into the base weights and load the merged checkpoint in Nano-vLLM, which cannot load LoRAs. If it works, the teacher becomes the release model.

**Save latents directly**
- Patch the generator to save the AudioVAE latents it decodes. They are exactly the student targets: 64 × 25 Hz in fp16 is ≈3.2 KB/s, about 230 MB for 20 h. This avoids an encode/decode round trip.
- Keep 16 kHz wavs for gating.

**Mix (hours kept)**
- 10-20 h total: ~60 % everyday and numeric sentences, ~30 % questions across all textbank kinds, ~10 % long lines (≤10 s chunks).
- Plus 400-1000 clips per emotion, rendered with `instruction_pick`. The student sees only a style ID (neutral plus 6 emotions), never instruction text.
- Laughter is deferred.
- Text comes from our textbank plus Alania `text_spoken`, CC-BY config only. FineWeb-2 is ODC-By. Skip the cc-by-sa config.
- Greedy syllable/bigram coverage selection ([CMU ARCTIC method](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.pdf)).
- The existing leakage guard keeps TR-Zor, TR-Soru, held-out and probe sentences out.

**Gates** (existing, plus student-specific)
- Existing: onset; 7-25 chars/s; Whisper CER 0 and p(tr) ≥ 0.95; ECAPA ≥ 0.70; UTMOS lower-tail cut; c-check on instructed or c-word rows; intonation peak for questions; pause gate; `pause_clean` applied to kept clips.
- Student-specific: the harsh-timbre detector, which rejects "telephone" renders in English-instruction emotions.
- Re-render up to 4 seeds instead of dropping. Re-rendering mattered in [arXiv 2609.03502](https://arxiv.org/html/2609.03502), where scaling without re-rendering made the student worse.

**Cost reality**
- Measured kept-hour cost: f022corpus 3.53 h for ~$6.7 (≈$1.9/h); f022corpus2 1.015 h for $2.71 (≈$2.7/h, with long lines and c-words).
- Raw Nano-vLLM generation is only ≈$0.07-0.09 per audio hour (37-50 s of audio per second on A100).
- Scoring dominates, at ≈0.7 s per clip for Whisper + UTMOS + ECAPA (README).
- Levers:
  - Longer clips (5-10 s) mean fewer scorer calls per hour.
  - Batched Whisper.
  - UTMOS/ECAPA only on Whisper-passing attempts.
  - Optionally, gate on Kaggle T4 for free.
- Plan at $2/h until the first `--max-usd 2` call measures the rate. Target is ≤$1/h.

**Collapse hygiene**
- One generation only: never train on student outputs.
- Vary seeds.
- Arm (a) keeps a real-data-free but large prior (`ema.pt`). If provenance forces scratch, pretrain on Alania instead.

**Clean-provenance alternative to `ema.pt`**
- Pretrain on a 300-500 h Alania CC-BY-4.0 subset: 2,647 designed voices, all VoxCPM2 synthetic, no real speakers ([dataset](https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr)). Its latent space is the same AudioVAE.
- Encoding cost: ~$1-3 on an A100 or L4 batch (estimate), or Kaggle.
- Training: scratch pretraining 5-15 A100-h ($15-45), or ~1-2 Kaggle weeks. Then fine-tune on f022.

### A5. Training plan
- **Stage 1** (only if `ema.pt` is rejected): pretrain on Alania, single "speaker" or with a speaker embedding.
- **Stage 2:** fine-tune on f022 (P0 data plus Nov data), with a style-ID embedding added to the text-encoder output (GST-like, [arXiv 1803.09017](https://arxiv.org/abs/1803.09017)). Emotion IDs get 0.1 dropout to neutral.
- **Stage 3** (optional): DMD2 re-distillation to 4 steps, implemented from the [paper](https://arxiv.org/abs/2405.14867) only. The official DMD2 code is CC BY-NC-SA ([repo](https://github.com/tianweiy/DMD2)).
- **Stage 4** (optional): fine-tune `decoder.pt` on f022 (latent, 48 kHz) pairs with new MPD/MSD discriminators ([HiFi-GAN, MIT](https://github.com/jik876/hifi-gan)).
- **Ship rule:**
  - TR-Zor ≤ 1.9 % (EMA with trnorm).
  - Ties or beats EMA in blind listening on plain reading.
  - Student vs teacher similarity ≥ 0.75.
  - CPU RTF ≤ 0.2 on 2 threads.
  - Otherwise do not ship.

### A6. Cost per step (Modal list prices from `tts/voxcpm/modal_app.py`)
Prices: A100-80GB $2.50/h GPU (≈$3.26/h with CPU/RAM per README), H100 $3.95, L4 $0.80, T4 $0.59.

| Step | Compute | Est. cost |
|---|---|---|
| Benchmark antalia-mini vs EMA (TR-Zor, UTMOS, CPU RTF) | local CPU | $0 |
| Latent export of f022mix (4.6 h) | local CPU | $0 |
| P0 training, 2 arms × 20-50k steps | Kaggle T4 | $0 (Modal L4 alternative ~$1-3) |
| P0 eval + blind page | local CPU | $0 |
| Merged-LoRA Nano-vLLM probe (optional) | A100 ~10 min | ~$0.6 |
| Nov data, 10-20 h kept | A100 Nano-vLLM + scorers | $15-40 ($1-2/h) |
| Stage-2 training | Kaggle T4, or A100 1-2 h | $0-7 |
| Decoder fine-tune (optional) | A100 2-4 h | $7-13 |
| Alania-pretrain path (only if needed) | A100 5-15 h | $15-45 |
| Drama 3 eval-only comparison (optional) | H100 ~30-45 min incl. 41.5 GB load | ~$2-4 |

### A7. Timeline
- **Oct 14-31** (≈$6 left; keep ≥$5 reserve):
  - antalia-mini benchmark (after download approval).
  - Trainer code with unit tests.
  - P0 on local CPU and Kaggle.
  - Eval and blind page.
  - Optionally the $0.6 merged-LoRA probe.
- **Nov 1-15** (~$60): if P0 passes, render data in $2-4 calls (≤$30 total, `--max-usd` each). Then stage-2 training, full eval, and listening round 5 (student vs EMA vs s3tr-500 vs antalia-mini).
- **Nov 15-30:**
  - ONNX export.
  - trnorm port (TypeScript first, for the browser demo).
  - AudioSeal marking.
  - Model card with the attribution chain: VoxCPM2 Apache-2.0; Alania CC-BY + PatientDesk; FineWeb-2 ODC-By; EMA Apache-2.0 if `ema.pt` is used.
- **Dec+:** emotion-ID quality, optional Pocket TTS expressive tier (needs ≥100 h ≈ $100-200 of data at current gate cost, so a later month).

### A8. On-device packaging
- **ONNX graphs:**
  1. text encoder + duration predictor
  2. DiT velocity step, with CFG batched as 2× or baked
  3. decoder
- **Size:** ~8.6M params ≈ 34 MB fp32, ~17 MB fp16, ~9 MB int8 (decoder int8 needs a listening check).
- **Runtimes:** onnxruntime (Android/iOS/desktop), onnxruntime-web (WASM/WebGPU) for an offline browser page, and Core ML following the [Elphi port](https://huggingface.co/elphi-si/elphi).
- **Possible reuse:** the existing EMA ports may load our weights unchanged if the architecture and alphabet stay identical. This is a reason to keep EMA's 49-symbol alphabet in v1.
- **Front-end:** trnorm is Python. On device it needs a port, either TypeScript for web or Rust with bindings for mobile. Its 155 unit tests become the port's conformance suite.
- **Synthetic marking:**
  - EU AI Act Art. 50(2) requires machine-readable marking ([text](https://artificialintelligenceact.eu/article/50/)).
  - Run AudioSeal (MIT) after the decoder, on server and device. Measure its CPU cost.
  - Do not rely on any teacher watermark surviving distillation ([arXiv 2503.19176](https://arxiv.org/pdf/2503.19176)).
- **Cloning:** none. A fixed-voice student has no reference input, so it cannot clone anyone.

### A9. Licence and terms checklist for the student
- **Allowed inputs:**
  - Teacher: VoxCPM2, Apache-2.0, no output-use restriction; labelling and no impersonation requested ([card](https://huggingface.co/openbmb/VoxCPM2)).
  - Our LoRA: ours.
  - f022: a designed voice.
  - Alania: CC-BY config only.
  - `ema.pt`: Apache-2.0, with unknown data provenance (user decision).
  - Matcha: MIT.
  - Pocket TTS: code MIT, weights CC-BY-4.0.
- **Forbidden inputs:**
  - Fish (Drama 3 §IV(b), S2 licence, API ToS (j)).
  - Gemini or Vertex (Google competitive-use terms).
  - ElevenLabs output.
  - DMD2 official code (NC).
  - MMS aligner weights (NC).
  - espeak-ng in shipped code (GPL).
  - Alania cc-by-sa config.
- **Internal-only until licences are checked:** emotion2vec (licence unclear).

### A10. Decisions requested from the user
1. Approve a ~30 MB download of `cloud0day3/antalia-mini` (Apache-2.0) for a free CPU benchmark.
2. Accept `ema.pt` as the init for the internal P0, and decide the release path after P0: ask the EMA author about provenance (the user sends that message), or take the Alania-pretrain path.
3. Use the free Kaggle T4 quota for P0 and stage-2 training.
4. November split: ≈$30 data, ≈$10 training and eval, ≈$20 reserve.
5. Drama 3 eval-only comparison: recommended *not now*.
6. Still open from before: public release of HF and GitHub. The student card should point to the public trnorm.

### Local sources used
- Project log (internal): Nano-vLLM cannot load the LoRA; corpus costs; listening rounds.
- `docs/SES_PLANI.md` "Sonra: küçük model".
- `tts/voxcpm/README.md`: gates, ≈0.7 s/clip scoring, A100 ≈$3.26/h.
- `tts/voxcpm/modal_app.py`: GPU_PRICE and SCORE_GPU_PRICE.
- `benchmarks/tts/systems.py`: EMA, Pocket TTS, Piper, MMS and Qwen3-tiny registered; antalia-mini not yet.

**Sibling notes in this folder:** `fish_drama3_notes.md`, `ema_lightning_notes.md`, `small_model_brief.md`, `scripts/latent_check.py`, `out/*.wav`.


---

## Eleştiri ve düzeltmeler / Critique and corrections (2026-10-14)

> Bu düzeltmeler yukarıdaki planın önüne geçer: **Kaggle kullanılmaz** (kullanım şartları ticari olmayan kişisel kullanım; ayrıca bir Google hizmeti). Deneme eğitimleri **yerel RTX 5060 (8 GB)** üzerinde, $0.

**Corrections to the small-model plan and the Fish Drama 3 decision (2026-10-14)**

I checked the three riskiest claims myself: Kaggle use, provenance and licences of the Alania, EMA and Antalia data and weights, and the Pocket TTS licence. Everything below is a correction; the rest of the plan can stand.

**1. Kaggle is not allowed for this work (wrong in the Turkish summary, A3, A5, A6, A7, decision 3).**
- Kaggle's Terms of Use say: "You will only use the Services for your own internal, personal, non-commercial use", "and not on behalf of or for the benefit of any third party" (terms effective 2025-06-22).
  - Sources: [kaggle.com/terms](https://www.kaggle.com/terms); the wording is quoted second-hand in [agentdeals PR #2360](https://github.com/robhunter/agentdeals/pull/2360) and [agentdeals Kaggle page](https://agentdeals.dev/vendor/kaggle).
  - The live page only renders with JavaScript, so I could not read the original text. Kaggle's own forum thread on this has no official answer ([discussion 272074](https://www.kaggle.com/discussions/general/272074)).
  - [Kaggle's notebook docs](https://www.kaggle.com/docs/notebooks) say GPU use is subject to these terms and that misuse can get the account suspended.
- This is a commercial project, and training a model to ship is commercial use.
- Kaggle is also a Google service, subject to the same competitive-use concerns as Gemini or Vertex.
- **Fix:** remove Kaggle everywhere and recommend against it in decision 3. The same applies to the earlier idea of training STT on Kaggle.

**2. There is a local GPU, and the plan does not use it.**
- `nvidia-smi` shows an NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB.
- An 8.6M-parameter student with batch about 32, plus AudioVAE latent encoding, fits easily on 8 GB.
- **Fix:** run P0 runs (a) and (b), stage-2 training, latent encoding and the optional Alania pretraining on this local GPU, all at $0.
  - Replace "Kaggle T4 (free)" and "local CPU" for encoding with "local RTX 5060".
  - Drop "1-2 weeks of Kaggle quota" in A4.
  - Keep Modal L4 (about $1-3) only as the fallback.

**3. Alania facts (checked on the [dataset card](https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr)).**
- Size: 3,411 h, 2,037,461 clips and 2,752 designed voices (plus one-off voices). The plan says "2,647 designed voices"; that should read 2,752.
- No real voices: "Every clip is AI-generated: no real person's voice is in this dataset." The `reference` render mode is the designed voice reading plainly. The plan's "no real speakers" claim is confirmed.
- Generator: `openbmb/VoxCPM2@32279eff+nanovllm-voxcpm@c702631e`. This confirms the same latent space.
- Configs and licences:
  - `cc-by` (1,660,381 clips) and `voices` are CC BY 4.0.
  - `cc-by-sa` (377,080 clips) is CC BY-SA 4.0 because of Turkish Wikipedia text.
  - FineWeb-2 text is ODC-By 1.0 and also subject to Common Crawl's terms.
- **Missing from the A8 model-card attribution chain:** the required credit line "Alania Turkish Synthetic Speech by PatientDesk AI". The card also asks that audio from models trained on Alania be labelled AI-generated where the law requires.

**4. EMA Lightning (checked on the [model card](https://huggingface.co/canberkkkkkk/ema-lightning)).**
- Licence: Apache-2.0 for weights and code, "Commercial use included". Confirmed.
- Sizes: 5.6M (`ema.pt`) plus 3.0M (`decoder.pt`). Confirmed.
- Data: the only description is "An internal corpus of about 1000 hours of single-speaker Turkish speech". There is no source, so the provenance risk stands.
- The card does not mention VoxCPM2 or AudioVAE. The claim that EMA uses the same latents rests only on our local `latent_check.py` test. Say "verified locally, not documented" rather than implying the author states it.
- The card adds usage guidance:
  - disclose AI-generated speech;
  - no impersonation;
  - no speech a real person did not say;
  - a human checks banking, health, legal and emergency messages.

  Our model card should carry these forward.

**5. Antalia-2 Mini (checked on the [model card](https://huggingface.co/cloud0day3/antalia-mini)).**
- Licence: Apache-2.0 for code, text normaliser and weights. It depends on `cmudict` (BSD).
- Size: 7.62M = 3.69M acoustic + 3.93M vocoder.
- Data: about 1,578 h of one synthetic **male** voice from VoxCPM2.
- Text sources: Turkish Wikipedia, FineWeb-2, Qwen3-written customer-service sentences, and templates.
- Freya-TR-Eval: WER 1.02 %, CER 0.21 %, UTMOS 3.18. On the same harness, **EMA scores WER 1.00 %**, so Antalia is not ahead of EMA on WER. An independent re-run gives 1.09 ± 0.12.
- **Unverified in the plan:** "CPU RTF 0.052 on 8 EPYC cores; runs in ONNX Runtime Web". The card I read gives only GPU RTFs (0.026 on L40S, 0.029 on L4). Cite the exact source or mark these as unverified.
- Its text includes Qwen3-generated sentences. Before we ever reuse any Antalia-derived data, check the Qwen licence and terms; benchmarking only is fine.

**6. Pocket TTS (checked on the [model card](https://huggingface.co/kyutai/pocket-tts)).**
- The weights are **gated**: you must share contact information and accept conditions. That counts as accepting terms, so it needs the user's explicit approval.
- Prohibited uses include "voice impersonation or cloning without explicit and lawful consent".
- Official languages are English, French, German, Portuguese, Italian and Spanish only. Turkish comes only from the alperiox community fine-tune.
- The CC-BY-4.0 attribution duty to Kyutai survives alperiox's MIT label, so include it in the attribution chain.
- The plan's "code MIT" claim is not stated on the HF card; cite the GitHub LICENSE file.

**7. Internal inconsistencies.**
- **Alania pretraining cost:** the Turkish decision 2(b) says "~$5-15 ek", but A4 and A6 say $15-45 of A100 time plus $1-3 for encoding. With the local GPU (item 2) it is about $0 in compute. You then need to download and store a 300-500 h subset (latents alone are about 5.8 GB fp16 for 500 h, and the raw audio is far larger). Fix the figure and add the disk and bandwidth requirement.
- **November data cost:** 10-20 h at the measured ~$2/h is $20-40. The plan caps data at $30 (Turkish timeline and decision 4), which buys only about 15 h at $2/h. Either say "10-15 h at $2/h, 20 h only if the ≤ $1/h target is reached", or raise the cap.
- **Phone speed versus sampling steps:**
  - `ema.pt` is a DMD2 few-step generator, and that is what EMA's measured 0.18 CPU RTF is based on.
  - Our flow-matching student samples with 8-16 Euler or midpoint steps. With CFG that is 16-64 DiT passes, so the acoustic model will be several times slower.
  - The ship rule (CPU RTF ≤ 0.2 on 2 threads) therefore probably needs Stage 3 (DMD2 re-distillation), so Stage 3 is not optional for phone and browser.
  - The 0.18 figure did not record how many threads it used; re-measure at 2 threads.

**8. EU AI Act Art. 50(2) labelling.**
- Art. 50 has applied since 2 August 2026 ([Art. 113](https://artificialintelligenceact.eu/article/113/)).
- It applies to the **current** Ahenk release (VoxCPM2 + LoRA) as well, not only to the future small model. The plan adds AudioSeal only in Nov 15-30, for the student.
- Add a release-model marking item now. Whether the EU's "Digital Omnibus" delay proposal changes this is not verified.
- AudioSeal's code is MIT. The licence of the **weights** is not cited; check the LICENSE in the [AudioSeal repo](https://github.com/facebookresearch/audioseal) before relying on it.

**9. Minor.**
- "Telefonda çalıştığı kanıtlandı" (proven to run on phones) is cited only to community ports ([sewox/turkish-neural-tts](https://github.com/sewox/turkish-neural-tts), [elphi-si/elphi](https://huggingface.co/elphi-si/elphi)). I did not check their licences. Check them before we copy any code or ship weights through them.
- `modal volume get` is not a GPU call. Whether Modal bills data egress is unverified; keep "check the panel".
- I did not re-check the Fish Drama 3 licence. I found nothing that changes the "no" verdict; it only gets more conservative given the Kaggle finding.
