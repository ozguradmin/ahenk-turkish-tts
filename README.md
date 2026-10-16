# Ahenk — Türkçe metinden sese (Turkish text-to-speech)

**Türkçe** · [English](#english)

Türkçe metinden-sese (TTS) ve sesten-metne (STT) için açık kaynak araçlar, ölçümler ve eğitim tarifleri. Tek kişilik,
küçük bütçeli bir proje: ne yaptığımızı ve neyi **yapamadığımızı** ölçümleriyle birlikte yazıyoruz. Neden bu işe
giriştiğimiz ve hedeflediğimiz boşluklar: [docs/ARASTIRMA.md](docs/ARASTIRMA.md).

| Bileşen | Ne işe yarar | Durum |
|---|---|---|
| [`trnorm`](src/trnorm) | Yazılı Türkçeyi okunuş formuna çevirir (sayı, tarih, saat, para, yüzde, sıra sayısı, kısaltma, telefon, plaka, marka adları...). Ekleri okunuşa göre uyumlar. TTS ön ucu ve ASR değerlendirmesi için. Bağımlılığı yok. | ✅ Hazır, testli |
| **Ahenk** ([Hugging Face](https://huggingface.co/ozgurguler/ahenk-tts-tr), tarif: [`tts/voxcpm`](tts/voxcpm)) | VoxCPM2 (Apache-2.0) için Türkçe LoRA: tek bir **tasarlanmış** (gerçek kişiye ait olmayan) kadın sesi, soru tonlaması, 6 duygu ve gülme, İngilizce ya da Türkçe talimat. Ön uç trnorm. GPU gerekir. | ✅ Hazır, Hugging Face yayını hazırlanıyor (CC BY 4.0) |
| [`trtts`](src/trtts) | CPU'da çalışan hazır Türkçe TTS: EMA Lightning (Apache-2.0, 8.6M) + trnorm ön ucu. TR-Zor'da %14.0 → **%1.9** WER. | ✅ Hazır |
| [`benchmarks/tts`](benchmarks/tts) | **TR-Zor** (155 elle doğrulanmış zor cümle, 13 kategori) ve **TR-Soru** (35 satır soru tonlaması) + TTS'i ASR geri-okumasıyla ölçen düzenek | ✅ |
| [`benchmarks/asr`](benchmarks/asr) | FLEURS-TR ve MediaSpeech-TR üzerinde, sabit normalizasyonla Türkçe ASR karşılaştırması (CPU) | ✅ |
| [`stt/`](stt) | Küçük, streaming, Apache-2.0 Türkçe STT için eğitim tarifi (k2/icefall Zipformer → sherpa-onnx), yalnızca ticari kullanıma açık veriyle | 🛠️ Altyapı hazır, eğitim yapılmadı |
| **Ahenk Cep** ([`tts/small`](tts/small)) | Cihaz üstü küçük model (~9M; EMA Lightning mimarisinde öğrenci, Ahenk'in f022 sesinden). Planlanan HF adı `ozgurguler/ahenk-cep-tr` | 🧪 Deneme, yayınlanmadı |
| [`tts/`](tts) | Piper (VITS) sesi denemesi | ⏸️ Durduruldu: TR-Zor'da %24.6 WER, robotik ([neden](tts/README.md#sonuç-426-epoch-durduruldu)) |

## Ahenk (VoxCPM2 + LoRA)

> **Sentetik ses.** Ahenk'in sesi VoxCPM2 ile bir İngilizce metin tarifinden tasarlandı; gerçek bir kişinin sesi
> ya da kaydı değildir ve eğitim seslerinin tamamı yapay zekâ ile üretildi. Dinleyicinin yanılabileceği her yerde sesin
> yapay olduğunu belirtin (AB Yapay Zekâ Yasası md. 50). Hiç kimseyi taklit etmek için kullanmayın.

- **Model, kullanım kodu ve örnek sesler:** <https://huggingface.co/ozgurguler/ahenk-tts-tr>
- **Eğitim:** aşama 1, Alania sentetik konuşma verisinden 25 saatlik filtreli alt küme (Türkçeyi güçlendirmek); aşama 2,
  taban VoxCPM2'nin tasarlanmış "f022" sesiyle okuduğu 4.6 saatlik derlem (düz, sayılı, uzun, soru, duygu ve gülme
  cümleleri); son olarak İngilizce + Türkçe talimat ince ayarı. Tarif ve komutlar: [`tts/voxcpm`](tts/voxcpm).
- **Ölçüm** (faster-whisper large-v3-turbo ile geri okuma, WER %, düşük iyi; UTMOS doğallık *tahmini*, 1-5):

| Model | TR-Zor WER (hatasız) | Ayrılmış 20 uzun cümle WER (hatasız) | UTMOS TR-Zor / ayrılmış |
|---|---:|---:|---:|
| **Ahenk** (yayınlanan, 16 adım, cfg 2.0) | **2.54** (141/155) | **0.82** (16/20) | 3.70 / 3.14 |
| Trendyol-TTS (VoxCPM2 + LoRA, ~20 sa kapalı veri; aynı düzenek) | 3.98 (134/155) | 0.68 | 3.82 / 3.73 |

- **Dürüst notlar:** doğallık tahmini Trendyol-TTS'in gerisinde, özellikle uzun cümlelerde (12-19 sn). İngilizce duygu
  talimatı cümle başındaki "c"yi İngilizce okutabiliyor ("Vay canına" → "vey kenına"); paketteki `instruction_pick.py`
  bunun için Türkçe talimata geçiyor. Bazı İngilizce talimatlı duygulu kliplerde ses "telefon gibi" parlak ve cızırtılı.
  İnsan dinlemesi tek dinleyiciyle yapıldı, sistematik bir MOS testi yapılmadı. Tüm sınırlamalar model kartında.

## Hızlı başlangıç: Türkçe TTS (trtts, CPU)

```bash
pip install "trnorm[tts] @ git+https://github.com/ozguradmin/ahenk-turkish-tts"
trtts "Randevunuz 14.10.2026 saat 09:30'da; ücret 1.250,50 TL." -o randevu.wav
```

```python
from trtts import TurkishTTS

tts = TurkishTTS()        # EMA Lightning'i bir kez indirir (~34 MB); GPU varsa onu kullanır
tts.say("SGK'ye 15.00'te, %20'si ödendi.", path="ornek.wav")
for parca in tts.stream("Uzun bir metin..."):   # 48 kHz float32 parçalar, üretildikçe
    ...
```

Ses EMA Lightning'in; trtts yalnızca ne söyleneceğine trnorm ile karar verir. EMA Lightning'in kendi ön ucu `%20'si`'ni "yüzde yirmi kesme si", `0212 444 55 66`'yı rakam rakam okuyor; trtts ile "yüzde yirmisi", "sıfır iki yüz on iki, dört yüz kırk dört, elli beş, altmış altı".

## Hızlı başlangıç: trnorm

```bash
pip install git+https://github.com/ozguradmin/ahenk-turkish-tts
trnorm "Saat 09:30'da 3. kata gelin, ücret 1.250,50 TL."
# Saat dokuz otuzda üçüncü kata gelin, ücret bin iki yüz elli lira elli kuruş.
```

```python
from trnorm import normalize, normalize_for_eval

normalize("SGK'ye 15.00'te, %20'si ödendi; plaka 34 ABC 123.")
# 'se ge kaya on beşte, yüzde yirmisi ödendi; plaka otuz dört a be ce yüz yirmi üç.'

normalize_for_eval("İstanbul'da 1,5 saat bekledik!")   # WER/CER için
# 'istanbulda bir virgül beş saat bekledik'
```

Neleri çözüyor (seçme örnekler; kurallar ve gerekçeler: [docs/TRNORM.md](docs/TRNORM.md)):

| Girdi | Çıktı |
|---|---|
| `26.09.2027'de sabah 10.00'a` | yirmi altı Eylül iki bin yirmi yedide sabah ona |
| `15.00'de` (yanlış ek) | on beşte (ek okunuşa göre düzeltilir) |
| `4'üncü`, `XIX. yüzyıl`, `3. kat` | dördüncü, on dokuzuncu yüzyıl, üçüncü kat |
| `₺1.250,75`, `%3,5`, `2,5 milyon TL` | bin iki yüz elli lira yetmiş beş kuruş, yüzde üç buçuk, iki buçuk milyon lira |
| `0555 555 55 55'i` | sıfır beş yüz elli beş, beş yüz elli beş, elli beş, elli beşi |
| `TBMM'de`, `ABD'ye`, `NATO'ya`, `SMS` | te be me mede, a be deye, nato'ya, esemes |
| `iPhone'umdan`, `ChatGPT'ye`, `Google'da` | ayfonumdan, Çet ci pi tiye, Gugılda (marka okunuşu, ek ona uyar) |
| `120 km/sa`, `250 mg'lık`, `38°C'ye` | saatte yüz yirmi kilometre, iki yüz elli miligramlık, otuz sekiz dereceye |
| `Sipariş no 584213`, `kodunuz 4821` | beş sekiz dört iki bir üç, dört sekiz iki bir |

### Ölçüm: mevcut normalizer ile karşılaştırma
Alania Domain Text'in geliştirmede kullanılmayan bir parçasından sayı/kısaltma içeren 10.000 cümle, referans Alania'nın okunuş formu ([ayrıntı](docs/ARASTIRMA.md#51-trnorm-türkçe-metin-normalizerı)):

| | Eşleşme | Karakter hatası | Çıktıda rakam kalan cümle |
|---|---|---|---|
| **trnorm** | **%89.3** | **%1.01** | **%0** |
| normalizer-tr 0.4.0, `ambiguity_policy="fallback"` (EMA Lightning'in kullandığı) | %54.4 | %7.50 | %0 — ama sembolleri harfiyen okuyor (`21:00'e` → "yirmi bir iki nokta sıfır sıfır kesme e") |
| normalizer-tr 0.4.0, varsayılan | %44.3 | %9.34 | %28.7 |

## Benchmark sonuçları

### ASR (sesten metne): FLEURS-TR test ve MediaSpeech-TR, aynı normalizasyon, CPU
| Model | Boyut | Lisans | Streaming | FLEURS WER | MediaSpeech WER |
|---|---|---|---|---:|---:|
| *whisper-large-v3-turbo (referans, 300 cümle)* | *809M* | *MIT* | *✘* | *5.2* | — |
| mihuai/turkish-stt | 66M | kısıtlı | ✔ | 9.3 | 14.0 |
| whisper-small | 244M | MIT | ✘ | 11.9 | 19.2 |
| duxx/turkish-stt-zipformer | 23.6M | CC-BY-NC | ✔ | 13.1 | 18.7 |
| whisper-base | 74M | MIT | ✘ | 23.8 | 30.4 |
| Vosk TR | küçük | Apache-2.0 | ✔ | 31.5 | 30.3 |
| whisper-tiny | 39M | MIT | ✘ | 38.1 | 43.5 |

Ticari kullanıma açık küçük streaming modellerin en iyisi %31.5'te kalıyor; `stt/` tarifi bu boşluğu hedefliyor. Tam tablo, RTF ve yöntem: [benchmarks/asr/results/LEADERBOARD.md](benchmarks/asr/results/LEADERBOARD.md).

### TTS ön ucu: TR-Zor, 155 zor cümle, Whisper ile geri-okuma
Aynı model, farklı metin ön işlemcisi. WER, % (düşük daha iyi):

| Model | Ham metin | Modelin kendi normalizer'ı | normalizer-tr (fallback) | **trnorm** |
|---|---:|---:|---:|---:|
| EMA Lightning (8.6M) | 49.7 | 14.0 | 14.0 | **1.9** |
| pocket-tts-turkish (110M) | 46.5 | 10.2 | 16.1 | **3.2** |
| qwen3-tts-tiny-tr (63 cümlelik alt küme) | 57.8 | — | 15.6 | **1.9** |
| Piper tr_TR-dfki | 13.3 | (= ham) | 15.6 | **3.5** |
| MMS-TTS-tur | 55.1 | — | 24.6 | **9.9** |

TR-Zor'un okunuşlarını ve trnorm'u aynı ekip yazdı; bu yüzden referanstan bağımsız kontroller de yaptık ([number_check.py](benchmarks/tts/number_check.py)):
- **Sayı değeri doğruluğu** (yazılı metindeki sayılar sesten geri okununca aynı mı): trnorm her modelde en yüksek ya da eşit (EMA %93.6, pocket %92.1, Piper %92.9, MMS %85.0). EMA + trnorm'da kaçan 9 sayının hepsi Whisper'ın yazım biçiminden ("52 bin" gibi); elle kontrolde 140 sayının 140'ı doğru okunmuş.
- **Sembolü sesli okuma:** normalizer-tr (fallback) ile 155 cümlenin 17–18'inde model "kesme", "iki nokta", "eğik çizgi" diyor; trnorm ile **hiçbirinde**.
- **Temiz metinde zarar yok:** rakam ve kısaltma içermeyen Freya-TR-Eval örneklerinde (40 cümle) bütün ön işlemciler her modelde aynı sonucu veriyor.

CPU hızı (2 iş parçacığı, RTF; düşük daha hızlı): Piper 0.07, EMA Lightning 0.18, MMS 0.33, pocket-tts 0.46, qwen3-tts-tiny 3.3 ([SPEED.md](benchmarks/tts/results/speed/SPEED.md)). Ayrıntılı tablolar: [benchmarks/tts/results/tr_zor/RESULTS.md](benchmarks/tts/results/tr_zor/RESULTS.md).

## Sınırlamalar
- Ahenk büyük bir GPU ister (A100'de 16 adımda gerçek zamandan yavaş, RTF ~2.3) ve tek ses, yalnızca Türkçe konuşur.
- Ölçümlerin yargıcı bir ASR modelidir (Whisper); onun hataları da sonuca karışır. Dinleme testleri tek dinleyicilidir.
- trnorm sözlükteki marka ve kısaltmaları bilir; sözlükte olmayan yabancı adlar yazıldığı gibi kalır ([docs/TRNORM.md](docs/TRNORM.md)).
- STT tarifi henüz eğitilmedi; ASR tablosundaki modellerin hiçbiri bizim değil.
- `docs/` içindeki planlama belgeleri tarih sırasıyla yazıldı ve eski sonuçlar içerebilir.

## Etik
- Bu depodaki modellerin ve tariflerin ürettiği ses **yapaydır**: dinleyicinin yanılabileceği yerde belirtin, mümkünse makinece okunur bir işaret ekleyin (AB Yapay Zekâ Yasası md. 50).
- Kimseyi taklit etmeyin; VoxCPM2 ses klonlayabilir, kimsenin sesini izni olmadan klonlamayın.
- Dolandırıcılık, sosyal mühendislik ve siyasi otomatik arama için kullanmayın.
- Eğitimde yalnızca ticari kullanıma açık veri kullanıyoruz (CC0, CC BY, MIT, Apache-2.0); ayrıntı: [docs/ARASTIRMA.md §7](docs/ARASTIRMA.md#7-lisans-politikası).

## Katkı
Hata bildirimi ve öneriler için issue açın; özellikle trnorm'un yanlış okuduğu cümleler çok işe yarar (girdi, beklenen okunuş, aldığınız çıktı). Kod değişikliklerinde `python -m pytest -q` geçmeli; trnorm kurallarını değiştirirken [docs/TRNORM.md](docs/TRNORM.md) ile `tests/test_normalizer.py`'yi birlikte güncelleyin. Yalnızca lisansı ticari kullanıma açık veri ve model önerin. Katkılar Apache-2.0 ile lisanslanır.

## Lisans ve atıf
- Kod, belgeler, test setleri (TR-Zor, TR-Soru) ve metin bankası: [Apache-2.0](LICENSE).
- Depodaki üçüncü taraf metinler (FLEURS, MediaSpeech ve Freya-TR-Eval referans cümleleri) kendi lisanslarıyla (CC BY 4.0) gelir: [NOTICE](NOTICE).
- Ahenk ağırlıkları **CC BY 4.0**'dır, çünkü şu verilerden türedi: *Alania Turkish Synthetic Speech by PatientDesk AI* ve *Alania Turkish Domain Text by PatientDesk AI* (CC BY 4.0). Temel model [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (OpenBMB) Apache-2.0'dır. Tam atıf model kartında.
- Kullanılan diğer kaynaklar: FLEURS (Google, CC BY 4.0), MediaSpeech (CC BY 4.0), Freya-TR-Eval (Freya Voice, CC BY 4.0), Antalia (Patientdesk.ai, CC BY 4.0; yalnızca eski denemelerde ve değerlendirmede), EMA Lightning (Apache-2.0), k2 / icefall / lhotse / sherpa-onnx (Apache-2.0), faster-whisper ve Whisper (MIT). Karşılaştırmadaki diğer modellerin lisansı kendi sayfalarında.
- Kullanırsanız [CITATION.cff](CITATION.cff) ile atıf yapabilirsiniz.

## İletişim
Sorular, izin ve kötüye kullanım bildirimleri için: destek@ozgurguler.tech

---

## English

Open tools, benchmarks and training recipes for Turkish speech, built by a single developer on a small budget. We
report what works and what does not, with measurements.

- **`trnorm`**: a dependency-free Turkish text normalizer (written → spoken form) for TTS front-ends and ASR evaluation.
  It reads numbers, dates, times, money, percentages, ordinals (Arabic and Roman), units, phone numbers, IDs, plates,
  abbreviations, acronyms and common brand names, and **re-agrees apostrophe suffixes with the spoken form**
  (`15.00'de` → *on beşte*, `SGK'ye` → *se ge kaya*, `iPhone'umdan` → *ayfonumdan*). On 10,000 held-out sentences it
  matches Alania's reference readings 89.3% of the time (vs 54.4% / 44.3% for `normalizer-tr`).
  `pip install git+https://github.com/ozguradmin/ahenk-turkish-tts`, then `trnorm "metin"`.
- **Ahenk** ([Hugging Face](https://huggingface.co/ozgurguler/ahenk-tts-tr), recipe in
  [`tts/voxcpm`](tts/voxcpm)): a Turkish LoRA adapter for VoxCPM2 (Apache-2.0) speaking in one **designed** female voice
  that belongs to no real person, with question intonation, six emotions and laughter driven by English or Turkish
  instructions. TR-Zor WER 2.54% (141/155 error-free), held-out long sentences 0.82%; predicted naturalness (UTMOS)
  3.70 / 3.14, below Trendyol-TTS (3.82 / 3.73) on the same harness. Needs a GPU. Weights CC BY 4.0.
  **Its speech is synthetic: disclose it wherever a listener could be misled (EU AI Act Art. 50) and never use it to
  impersonate anyone.**
- **`trtts`**: ready-to-use CPU Turkish TTS: EMA Lightning (Apache-2.0) with trnorm as its front-end; TR-Zor WER
  14.0% → 1.9%. `pip install "trnorm[tts] @ git+https://github.com/ozguradmin/ahenk-turkish-tts"`, then
  `trtts "metin" -o out.wav`.
- **TR-Zor / TR-Soru**: 155 hand-verified hard sentences and 35 question-intonation rows for Turkish TTS, with a harness
  that scores systems by ASR round-trip.
- **Turkish ASR benchmark**: FLEURS-TR and MediaSpeech-TR with one fixed normalizer, CPU-measured.
- **`stt/`**: a recipe for a small streaming Apache-2.0 Turkish ASR model (k2/icefall Zipformer → sherpa-onnx), trained
  only on permissively licensed data. Not trained yet.
- **Ahenk Cep** ([`tts/small`](tts/small)): a ~9M-parameter on-device student in the same voice (EMA Lightning
  architecture). Experimental, not released yet; planned Hugging Face id `ozgurguler/ahenk-cep-tr`.
- **`tts/`**: a Piper voice experiment, stopped (24.6% WER, robotic).

**Limitations:** the WER judge is an ASR model (Whisper) whose errors count too; listening tests had a single
listener; Ahenk is slower than real time at its recommended settings. The planning notes in `docs/` were
written chronologically and may contain superseded results.

**Contributing:** issues are welcome, especially sentences trnorm reads wrongly (input, expected reading, actual
output). `python -m pytest -q` must pass; keep `docs/TRNORM.md` and `tests/test_normalizer.py` in step with rule
changes; only propose commercially usable data and models. Contributions are licensed under Apache-2.0.

**Licence:** code, docs, test sets and text bank Apache-2.0 ([LICENSE](LICENSE)); third-party reference texts in
`benchmarks/` keep their CC BY 4.0 licences ([NOTICE](NOTICE)). Ahenk weights are CC BY 4.0, derived from
*Alania Turkish Synthetic Speech by PatientDesk AI* and *Alania Turkish Domain Text by PatientDesk AI* (CC BY 4.0), on
the Apache-2.0 VoxCPM2 base model (OpenBMB). Please cite via [CITATION.cff](CITATION.cff).

**Contact:** questions, permissions and abuse reports: destek@ozgurguler.tech
