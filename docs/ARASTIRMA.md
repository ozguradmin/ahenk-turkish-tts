# Türkçe STT / TTS: Araştırma ve Karar (Ekim 2026)

> Soru: "Türkçe için daha iyi bir STT ve TTS modeli çıkarabilir miyiz?"
> Bu belge o soruya dürüst cevabı ve gerekçesini veriyor. Rakamların çoğu
> model kartlarında **beyan edilmiş** değerlerdir; kendi ölçtüklerimiz
> ayrıca işaretlenmiştir (bkz. `benchmarks/`).

## 1. Kısa cevap

| Alan | "Daha iyisini yapabilir miyiz?" | Neden |
|---|---|---|
| **TTS: genel ses kalitesi** | **Hayır, 8 GB laptopla gerçekçi değil.** | Açık modeller (EMA Lightning, Trendyol-TTS, VoxCPM2, pocket-tts-turkish, qwen3-tts-tiny-tr) temiz metinde ASR ölçüm tabanına (~%1 WER) inmiş durumda. Bunları geçmek için onlarca–yüzlerce GPU saati ve lisanslı insan sesi verisi gerekiyor. |
| **TTS: gerçek hayattaki metin** | **Evet, şimdi ve GPU'suz.** | Sayı, tarih, saat, para, yüzde, sıra sayısı, kısaltma, telefon, plaka... Modeller bunları okuyamıyor. Mevcut açık normalizer `normalizer-tr` ya rakamları çözmeden bırakıyor (varsayılan modda cümlelerin ~%29'u) ya da sembolleri harfiyen okutuyor (`21:00'e` → "yirmi bir iki nokta sıfır sıfır kesme e"). `trnorm` rakam bırakmıyor, ekleri okunuşa göre uyumluyor ve bağımsız bir sette Alania okunuşlarıyla %89 eşleşiyor (normalizer-tr: %44–54; bkz. §5). Bu, *her* Türkçe TTS'in çıktısını doğrudan iyileştirir. |
| **STT: büyük model doğruluğu** | **Hayır, gerek de yok.** | Whisper large-v3 ve ince ayarları FLEURS-TR'de ~%7–8 WER'de. 1.5B parametrelik bir modeli geçmek bütçemizin çok dışında ve getirisi az. |
| **STT: küçük, açık lisanslı, streaming** | **Evet, gerçek bir boşluk var. Eğitim GPU istiyor.** | 100M parametrenin altında, ticari kullanıma açık, CPU/telefonda gerçek zamanlı çalışan iyi bir Türkçe model **yok**. En iyi küçük modeller lisans kısıtlı (`mihuai/turkish-stt`: gelir sınırı + "çıktılarıyla rakip model eğitme" yasağı; `duxx/turkish-stt-zipformer`: CC-BY-NC). Eğitim altyapısını hazırladık (`stt/`). Gerçek eğitim laptopta birkaç gece veya ~$30–60 kiralık GPU. |

## 2. Senin verdiğin kaynaklar

### 2.1 EMA Lightning (TTS): `canberkkkkkk/ema-lightning`
- 8.6M parametre (5.6M akustik + 3.0M vokoder), harf tabanlı flow-matching, 4 adımlık DMD2 damıtma, 48 kHz.
- Freya-TR-Eval'de beyan: **WER %0.92, CER %0.18**, UTMOS ~3.30. CPU'da ~6× gerçek zamandan hızlı.
- Apache-2.0. Ama: **tek ses**, klonlama yok, **eğitim kodu yok** (pip paketi sadece çıkarım), eğitim verisinin kaynağı açıklanmamış (~1000 saat tek konuşmacı).
- Metin önişleme için `normalizer-tr` (Apache-2.0, Rust) kullanıyor. Biz bu paketi test ettik: kesin ama dar kapsamlı (`3.`, `26.09.2027'de`, `21:00'e`, telefon numaraları, `4-5`, `ÖTV/THY/ABD`, `25°C`, `1/2` vb. çözülmeden kalıyor).

### 2.2 Qwen3-TTS Tiny TR: `erkamk/qwen3-tts-tiny-tr`
- Qwen3-TTS-0.6B'den budanmış (~370M paket), ses klonlama var. Alania `cc-by` (~1.863 saat) üzerinde eğitilmiş + RL.
- Beyan: Freya-TR-Eval WER %1.28. Sınırlamalar: rakamları yazıyla vermek gerekiyor, İ/I → i/ı küçültme gerekiyor, ~12 saniyeden uzun çıktı kesiliyor, tamamen sentetik veriyle (VoxCPM2) eğitilmiş.
- **Bizim normalizer'ımız tam olarak bu modelin istediği girdiyi üretiyor** (rakamsız, Türkçe küçük harf eval modu).

### 2.3 Whistle (Cactus): `Cactus-Compute/whistle`
- **STT** modeli (TTS değil). 55M parametre (çoğu gömme tablosu), Whisper benzeri encoder-decoder, 2–4 bit kuantize `.cact` dosyası 16.9 MB. CPU'da çok hızlı (M4 Pro'da ilk token 11 ms).
- Diller: en, de, fr, es, it, nl, pl. **Türkçe yok.** Apache-2.0.
- **Türkçe eklemek pratik değil:** ağırlıklar açık ama Whistle'ın eğitim/ince ayar kodu yayınlanmamış; encoder ve cross-attention'ı checkpoint'tan yeniden yazmak, 8k'lık tokenizer'ı Türkçe için genişletmek (ş, ğ, ı, İ), kuantizasyon-farkında eğitimi tekrar yapmak gerekiyor. Ayrıca girdisi 250–3500 Hz'e kısılmış; bu s/ş/z/ç ayrımına zarar verebilir.
- **Bizim yaklaşımımız:** Whistle'ın ruhuna uygun (küçük, hızlı, cihaz üstü) bir Türkçe modeli, eğitim kodu tamamen açık olan k2/icefall Zipformer ile sıfırdan eğitmek ve sherpa-onnx ile Android/iOS/WASM/Raspberry Pi'de çalıştırmak.

### 2.4 Alania Synthetic Speech TR: `cloud0day3/alania-synthetic-speech-tr`
- **3.411 saat** (tweet'teki 3.451 yerine kartta 3.411), 2.04M kayıt, 2.752 tasarlanmış ses, 48 kHz.
- `cc-by` config (2.565 saat, 540 GB, CC BY 4.0) + `cc-by-sa` config (846 saat, Vikipedi metni, CC BY-SA 4.0).
- Kolonlar: `text` (yazılı), **`text_spoken` (okunuş formu)**, `voice_description`, `audio_channel` (kayıtların ~%45'inde oda/mikrofon/gürültü simülasyonlu kopya), kalite skorları (`qc_cer`, `qc_utmos`).
- **Tamamı tek motordan (VoxCPM2) sentetik.** Kalitesi VoxCPM2'yi geçemez. Whisper ile filtrelendiği için "kolay" seslere yanlı. Eval için kullanılmamalı.
- Bizim için değeri: (1) `text` → `text_spoken` çiftleri normalizer'ı ölçmek için hazır bir doğruluk kaynağı, (2) STT eğitiminde kelime/sayı/alan çeşitliliği ve gürültülü kopyalarla dayanıklılık (gerçek sesle karıştırarak, saatlerin ≤%40–50'si).

### 2.5 Alania Domain Text TR: `cloud0day3/alania-domain-text-tr`
- 1.44M cümle, CC BY 4.0, Qwen3-30B ile üretilmiş (randevu, kargo, bankacılık, adres, sayı/kod...). `text_raw` → `text_model` (okunuş) çiftleri.
- Normalizer karşılaştırmamızı ve STT için BPE/dil modeli verisini buradan aldık.

## 3. Türkçe TTS'te durum (Ekim 2026)

Beyan edilen WER değerleri farklı test düzenlerinden geliyor; sadece aynı düzen içinde kıyaslanabilir.

| Model | Boyut | Lisans | TR anlaşılırlık (beyan) | Not |
|---|---|---|---|---|
| [Trendyol/Trendyol-TTS](https://huggingface.co/Trendyol/Trendyol-TTS) | 2B | MIT etiketi (VoxCPM2 tabanlı) | WER ~%1 | GPU gerekir, "araştırma ürünü" |
| [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | 2B | Apache-2.0 | WER ~%1–2 | Klonlama + ses tasarımı, 48 kHz |
| [canberkkkkkk/ema-lightning](https://huggingface.co/canberkkkkkk/ema-lightning) | 8.6M | Apache-2.0 | WER %0.92 | Tek ses, CPU'da çok hızlı |
| [wite-tech/pocket-tts-turkish](https://huggingface.co/wite-tech/pocket-tts-turkish) | 110M | CC-BY-4.0 | WER ~%1.9 | CPU'da ~5× gerçek zaman, klonlama |
| [erkamk/qwen3-tts-tiny-tr](https://huggingface.co/erkamk/qwen3-tts-tiny-tr) | ~370M | Apache-2.0 | WER %1.28 | Klonlama, sentetik veri |
| [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) (çok dilli) | 0.5B | MIT | WER ~%2 | Filigran ekliyor |
| [freyavoice/Freya-TTS](https://huggingface.co/freyavoice/Freya-TTS) | 183M | Apache-2.0 | WER %8–12 | **Eğitim kodu açık** |
| Piper `tr_TR-dfki-medium` | 16M | veri CC-BY-NC-SA | WER %3–4 | eSpeak fonemleri, robotik |
| [facebook/mms-tts-tur](https://huggingface.co/facebook/mms-tts-tur) | 36M | CC-BY-NC-4.0 | WER %6–7 | İncil okuma tarzı |
| [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) | 470M | ticari değil | WER %4–11 | Coqui kapandı |

**Değerlendirme:** Freya-TR-Eval (495 cümle) fiilen standart, ama **bilerek sayı, marka ve kısaltma içermiyor**. Yani gerçek hayattaki en büyük hata kaynağı hiç ölçülmüyor. Bunu kapatmak için `benchmarks/tts/sets/tr_zor.jsonl` (155 cümle, 13 kategori) hazırladık.

## 4. Türkçe STT'de durum (Ekim 2026)

| Model | Boyut | Lisans | FLEURS-TR WER (beyan) | Streaming / CPU |
|---|---|---|---|---|
| openai/whisper-large-v3 | 1.55B | MIT | ~%7 | Hayır, CPU'da yavaş |
| [BuzzASR/turkish](https://huggingface.co/BuzzASR/turkish) (large-v3 ince ayar) | 1.55B | MIT | ~%8, CER %2 | Hayır |
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | 600M | OpenMDW-1.1 | %11–12 | **Evet**, ama 600M |
| [Qwen/Qwen3-ASR-1.7B / 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 0.6–1.7B | Apache-2.0 | TR destekli, sayı yok | Evet |
| [GoktugD/DUSUNEN-Dinle-244M-v1](https://huggingface.co/GoktugD/DUSUNEN-Dinle-244M-v1) (whisper-small) | 244M | Apache-2.0 | ~%14.4 | Hayır |
| [mihuai/turkish-stt](https://huggingface.co/mihuai/turkish-stt) | **66M** | **Kısıtlı** (gelir sınırı, rakip eğitim yasağı) | ~%13.9 | **Evet**, sherpa-onnx |
| [duxx/turkish-stt-zipformer](https://huggingface.co/duxx/turkish-stt-zipformer) | **23.6M** | **CC-BY-NC-4.0** | ~%20 (val) | **Evet** |
| Vosk TR | ~50M | Apache-2.0 | ~%34 | Evet |

Türkçe desteği **olmayanlar:** Parakeet v3, Canary, Moonshine, Whistle, Voxtral Transcribe, Cohere Transcribe, Kyutai STT, Distil-Whisper.

**Kendi ölçümlerimiz** (aynı `trnorm` normalizasyonu, aynı test setleri, CPU; ayrıntı: [benchmarks/asr/results/LEADERBOARD.md](../benchmarks/asr/results/LEADERBOARD.md)):

| Model | Boyut | Lisans | Streaming | FLEURS-TR test WER / CER | MediaSpeech-TR WER / CER |
|---|---|---|---|---|---|
| *whisper-large-v3-turbo (referans; FLEURS'un ilk 300 cümlesi)* | *809M* | *MIT* | *hayır* | *5.2 / 1.3* | — |
| mihuai/turkish-stt | 66M | kısıtlı (gelir sınırı, rakip eğitim yasağı) | evet | 9.3 / 2.5 | 14.0 / 5.6 |
| whisper-small | 244M | MIT | hayır | 11.9 / 2.8 | 19.2 / 8.6 |
| duxx/turkish-stt-zipformer | 23.6M | CC-BY-NC-4.0 | evet | 13.1 / 3.3 | 18.7 / 7.0 |
| whisper-base | 74M | MIT | hayır | 23.8 / 6.1 | 30.4 / 11.9 |
| Vosk TR (small 0.3) | küçük | Apache-2.0 | evet | 31.5 / 9.8 | 30.3 / 14.9 |
| whisper-tiny | 39M | MIT | hayır | 38.1 / 10.7 | 43.5 / 16.9 |

Whisper large-v3-turbo yalnızca referans olarak FLEURS'un ilk 300 cümlesinde ölçüldü (CPU'da yavaş; aynı 300 cümlede mihuai %9.3, whisper-small %12.7, duxx %13.4). Not: duxx ve mihuai'nin model kartlarındaki rakamlar farklı normalizasyonlarla hesaplanmış; burada hepsi aynı kuralla puanlandı (mihuai kendi kartındaki %13.9'dan daha iyi çıktı).

**Boşluk (ölçümle doğrulandı):** <100M parametre + **ticari kullanıma açık lisans** + streaming + FLEURS-TR'de ≤%14 WER olan bir model yok. İzinli lisanslı küçük streaming modellerin en iyisi Vosk: %31.5.

**Veri gerçeği:** `duxx/turkish-stt-zipformer` (23.6M) Common Voice 17 + **WorldSpeech (~1.000 saat, ticari değil)** + FLEURS ile eğitilip FLEURS'ta ~%20'de kalıyor. Yani küçük bir modeli %14'ün altına indirmek için yüzlerce değil, **bin saati aşan gerçek konuşma** gerekiyor. İzinli lisanslı gerçek Türkçe konuşma havuzu:

| Kaynak | Saat | Lisans | Not |
|---|---|---|---|
| Common Voice 17 TR (`fixie-ai/common_voice_17_0` HF aynası) | ~120 | CC0 | Okuma, cümleler tekrar ediyor |
| ISSAI Turkish Speech Corpus | 218 | MIT | TV/talk show |
| FLEURS tr (train+dev) | ~11 | CC-BY-4.0 | Okuma |
| **YODAS2 tr000** (`sarulab-speech/yodas2_sidon`, insan yapımı altyazılı YouTube) | ~1.500 saat video → **~500 kullanılabilir** (ölçüldü) | CC-BY-3.0 | 126 GB indirme; altyazı gürültüsü var, filtrelenmeli |
| Alania Synthetic TR (`cc-by`) | 2.565 | CC-BY-4.0 | Sentetik (VoxCPM2); en fazla %40–50 oranında karıştırılmalı |
| turkish-tts-audiobooks | 1.311 | "open-weights-with-attribution" | Sesli kitaplardan türetilmiş; kaynak sesin telif durumu belirsiz → varsayılan olarak **kullanılmıyor** |

## 5. Bu gece yaptıklarımız (GPU'suz)

### 5.1 `trnorm`: Türkçe metin normalizer'ı
Yazılı Türkçeyi okunuş formuna çeviriyor ve kesme işaretli ekleri okunuşa göre yeniden uyumlandırıyor (`15.00'de` → "on beşte", `SGK'ye` → "se ge kaya", `4'üncü` → "dördüncü"). 124 birim testi.

**Bağımsız ölçüm:** Alania Domain Text'in geliştirme sırasında hiç bakmadığımız parçasından (`part-00003`) sayı/kısaltma içeren 10.000 cümle; referans = Alania'nın kendi okunuş formu (`text_model`):

| Normalizer | Birebir eşleşme (boşluksuz) | Karakter hatası | Çıktıda rakam kalan cümle |
|---|---|---|---|
| **trnorm** | **%89.3** | **%1.01** | **%0.0** |
| normalizer-tr 0.4.0, `ambiguity_policy="fallback"` (EMA Lightning'in kullandığı mod) | %54.4 | %7.50 | %0.0 |
| normalizer-tr 0.4.0, varsayılan (`preserve`) | %44.3 | %9.34 | %28.7 |

Notlar:
- "Eşleşme" kısmen bir üslup ölçüsüdür: Alania'nın okuma tercihlerine (ör. 6 haneli kodları rakam rakam okumak, `QR` → "kü re") ne kadar benzediğimizi de ölçer. Bazı yerlerde bilerek Alania'dan ayrıldık (`QR` → "kü ar", `LC Waikiki` → "el si").
- `normalizer-tr` varsayılan modda belirsiz ifadeleri olduğu gibi bırakıyor (her 3-4 cümleden birinde TTS'e rakam gidiyor). `fallback` modunda rakam bırakmıyor ama sembolleri harfiyen okuyor: `21:00'e` → "yirmi bir iki nokta sıfır sıfır kesme e", `4-5` → "dört tire beş", `25°C` → "iki beş derece işareti ce", `3. kat` → "üç. kat".
- ITN (okunuş → rakam) için: aynı parçadan 10.000 okunuş cümlesinde gidiş-dönüş güvenli oran %96.4; farkların çoğu rakam rakam okunmuş kodların yazıdan geri okunuş biçiminin belirsizliği.

### 5.2 Benchmark'lar
- `benchmarks/asr/`: FLEURS-TR test ve MediaSpeech-TR üzerinde CPU'da çalışabilen modellerin WER/CER/RTF ölçümü, sabit normalizasyonla.
- `benchmarks/tts/`: TR-Zor seti + Freya örnekleri üzerinde küçük TTS modellerinin, farklı ön işlemcilerle (ham / modelin kendi / normalizer-tr / trnorm) ASR geri-dönüş (round-trip) ölçümü.

**TTS sonuçları (TR-Zor, 155 cümle; hakem: Whisper large-v3-turbo; WER %, düşük daha iyi):**

| Model | Ham metin | Kendi normalizer'ı | normalizer-tr (preserve) | normalizer-tr (fallback) | **trnorm** |
|---|---:|---:|---:|---:|---:|
| EMA Lightning | 49.7 | 14.0 | 28.3 | 14.0 | **1.9** |
| pocket-tts-turkish | 46.5 | 10.2 | 27.7 | 16.1 | **3.2** |
| qwen3-tts-tiny-tr (63 cümle) | 57.8 | — | 36.3 | 15.6 | **1.9** |
| Piper tr_TR-dfki | 13.3 | = ham | 12.1 | 15.6 | **3.5** |
| MMS-TTS-tur | 55.1 | — | 36.2 | 24.6 | **9.9** |

Hatasız okunan cümle oranı trnorm ile: EMA %94.8, qwen3-tiny %93.7, pocket %88.4, Piper %87.1, MMS %62.6.

**Dürüstlük notu ve bağımsız kontroller:** TR-Zor setinin okunuşlarını ve trnorm'u aynı kişi (bu oturum) yazdı, okuma tercihleri örtüşüyor. Bu, WER'i trnorm lehine şişirebilir (her cümlede alternatif geçerli okumalar `alts` kabul ediliyor, ama bu yanlılığı tamamen gidermez). Bu yüzden iki referanssız kontrol daha yaptık ([number_check.py](../benchmarks/tts/number_check.py)):
1. **Sayı değeri doğruluğu:** yazılı cümledeki sayılar, sesin ASR çözümünde (değer olarak) geri geliyor mu? trnorm her modelde en yüksek ya da eşit: EMA %93.6 (kendi ön ucu %90.7), pocket %92.1 (%89.3), Piper %92.9 (%92.1), MMS %85.0 (fallback ile eşit). EMA + trnorm'da kaçan 9 sayının tamamı elle kontrol edildi: hepsi Whisper'ın yazım biçimi ("52 bin", "100-000-001-46"); gerçekte 140/140 doğru. Bu ölçüt hoşgörülü: "3. kat"ı "üç kat" okumak (sıra sayısı yerine asıl sayı) da doğru sayılıyor.
2. **Sembolü sesli okuma:** normalizer-tr fallback modunda (EMA'nın kullandığı) 155 cümlenin 17–18'inde model sembolleri okuyor: `7'si` → "yedi kesme si", `THY'nin` → "te he ye kesme ne i ne", `12:00'ye` → "on iki iki nokta sıfır sıfır kesme ye". trnorm ile hiçbir cümlede yok.

3. **Temiz metinde zarar yok:** Freya-TR-Eval'den 40 rakamsız cümlede tüm ön işlemciler her modelde birebir aynı WER'i veriyor (EMA %0.6, pocket %1.9, qwen3-tiny %2.2, Piper %3.2, MMS %8.0).

Ek gözlemler: pocket-tts-turkish'in kendi normalizer'ı cümle başındaki kısaltmaları bozuyor (`ABD'den` → ", abdden"; akronim kategorisinde WER %36.4, ham metinde %11.4). trnorm'da kalan hatalar çoğunlukla Whisper'ın kısaltmaları tek kelime yazmasından ("Tehaye'nin") ve `MR`'ın "me re"/"emar" okunuş farkından.

CPU hızı (2 iş parçacığı, RTF): Piper 0.07, EMA Lightning 0.18, MMS 0.33, pocket-tts 0.46, qwen3-tts-tiny 3.3 ([SPEED.md](../benchmarks/tts/results/speed/SPEED.md)).

Sonuç: trnorm'un gerçek katkısı, sayı değerlerinde küçük (+1–3 puan), **okuma biçiminde büyük**. Sembol okumayı, yanlış ek ve sıra sayısı okumasını ortadan kaldırıyor. Kendi ön ucu zayıf modellerde (MMS, qwen3-tiny) ise sayıların okunup okunmaması düzeyinde fark yaratıyor.

### 5.3 STT eğitim altyapısı (`stt/`)
Zipformer transducer (k2/icefall), streaming, sherpa-onnx'e export. Yalnızca izinli lisanslı veri (FLEURS CC-BY, ISSAI TSC MIT, Common Voice CC0, Alania CC-BY). Laptop için: sıcaklık koruması (83 °C'de duraklat, 72 °C'de devam), N saat sonra temiz durma, kaldığı yerden devam.

## 6. Plan

| Faz | İş | Donanım | Süre |
|---|---|---|---|
| 0 ✅ | Normalizer + benchmark + eğitim altyapısı | CPU (bu gece) | — |
| 1 | STT pilot: small Zipformer, ~100 saat veri (FLEURS + CV train + Alania), GPU yolunu ve gerçek hızı doğrulama | Laptop GPU | ~3 saat (eğitim ~2 saat) |
| 2 | STT asıl eğitim: small (24,5M) veya medium (67M), ~600 saat (ISSAI + CV validated + FLEURS + Alania) | Laptop: small ~34–48 GPU saati (4 saatlik ~9–12 oturum) / kiralık RTX 4090: bir gece, ~$5–14 | `stt/README.md` §6 |
| 3 | Yayın: Apache-2.0 model + sherpa-onnx paketleri (Android/iOS/WASM), HF model kartı, benchmark tablosu | — | — |
| 4 ✅ | TTS: trnorm'u EMA Lightning ile paketleyen ön uç → [`trtts`](../src/trtts) | CPU | — |
| 4b ⏸️ | TTS: Antalia (CC-BY) ile Piper ince ayarı. Modal'da 426 epoch, ~$13. TR-Zor'da %24.6 WER, kendi verisinden ayrılmış cümlelerde %10.7; dinleyince robotik. VITS medium + espeak-ng'nin tavanı dfki düzeyi (%3.5, yine robotik), EMA Lightning (Apache-2.0) ise %1.9 ve doğal. Durduruldu, kalan bütçe STT'ye ([tts/README.md](../tts/README.md#sonuç-426-epoch-durduruldu)) | Modal A100 | — |

**Başarı ölçütü (STT), ölçümlere göre:**
- **Asgari:** FLEURS-TR'de whisper-base'i (%23.8) ve Vosk'u (%31.5) geçen, 24.5M parametrelik, streaming, Apache-2.0 model. Bu bile "en iyi izinli lisanslı küçük streaming Türkçe model" olur.
- **Hedef:** FLEURS-TR'de ≤%14, yani ticari olmayan duxx (%13.1) ile aynı seviye, ama ticari kullanıma açık.
- Tutturamazsak dürüstçe yayınlarız: "şu kadar veri ve hesapla şu kadar oldu".

**Gerçekçi beklenti:** Faz 1 verisiyle (~350 saat gerçek + sentetik) küçük bir model muhtemelen FLEURS'ta %18–25 civarında kalır; bu, asgari ölçütü karşılar. %14 hedefi için Faz 2'de YODAS2 tr000 gerekir. Ölçüldü: altyazılar videoların yalnız ~%38'ini kapsıyor ve filtrelerden ~%34'ü geçiyor, yani ~1.500 saat videodan **~500 saat** kullanılabilir konuşma çıkıyor (öğretmen-CER filtresiyle ~350–450). ~1.000–1.100 saatlik toplam veriyle medium model kiralık tek bir RTX 4090'da ~16–20 saat (~$6–14); laptopta ise haftayı bulur.

## 7. Lisans politikası
- Varsayılan tarifte **yalnızca** ticari kullanıma açık veri: CC0, CC-BY, MIT, Apache.
- Kullanmadıklarımız: WorldSpeech (CC-BY-NC), mihuai ve duxx çıktıları (biri lisansla yasaklıyor, diğeri ticari değil; veri filtrelemek için bile kullanmıyoruz), XTTS/MMS/F5 tabanları (ticari değil), kiraat (lisanssız), turkish-tts-audiobooks (kaynak sesin telif durumu belirsiz; yalnızca opsiyonel bayrakla), ünlü sesi içeren derlemeler.
- Alania kullanıldığında atıf zorunlu: "PatientDesk AI — Alania Synthetic Speech TR (CC BY 4.0)". `cc-by-sa` config'i kullanılırsa türev veri de CC BY-SA olur; model ağırlıkları için tartışmalı olduğundan varsayılan olarak sadece `cc-by` config'ini kullanıyoruz.

## 8. Kaynaklar
- Alania: https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr, https://huggingface.co/datasets/cloud0day3/alania-domain-text-tr
- EMA Lightning: https://huggingface.co/canberkkkkkk/ema-lightning, EMA-TTS raporu: https://huggingface.co/canberkkkkkk/ema-tts
- Qwen3-TTS Tiny TR: https://huggingface.co/erkamk/qwen3-tts-tiny-tr
- Whistle: https://cactuscompute.com/blog/whistle, https://huggingface.co/Cactus-Compute/whistle
- normalizer-tr: https://pypi.org/project/normalizer-tr/
- Freya-TR-Eval: https://huggingface.co/datasets/freyavoice/freya-tr-eval
- pocket-tts-turkish: https://huggingface.co/wite-tech/pocket-tts-turkish
- FLEURS: https://huggingface.co/datasets/google/fleurs, MediaSpeech: https://huggingface.co/datasets/ymoslem/MediaSpeech
- ISSAI TSC: https://huggingface.co/datasets/issai/Turkish_Speech_Corpus
- Common Voice: https://datacollective.mozillafoundation.org
- k2/icefall: https://github.com/k2-fsa/icefall, sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
- Webrazzi özeti (Türkçe açık modeller): https://webrazzi.com/2026/02/18/turkce-icin-one-cikan-acik-kaynak-modeller/
