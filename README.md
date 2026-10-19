# Ahenk: Türkçe metinden sese

Açık Türkçe konuşma araçları: Ahenk (büyük model, GPU), Ahenk Cep (8.6M parametrelik küçük model, CPU), trnorm
(Türkçe metin normalizasyonu) ve TR-Zor / TR-Soru ölçüm setleri.

> **Sentetik ses.** Ahenk ve Ahenk Cep tek bir tasarlanmış sesle konuşur (f022); bu ses gerçek bir kişinin sesi ya da
> kaydı değildir. Ürettiğiniz sesi paylaşırken yapay olduğunu belirtin (AB Yapay Zekâ Yasası md. 50).

[Tarayıcıda deneyin](https://ozguradmin.github.io/ahenk-turkish-tts/) · [Ahenk model kartı](https://huggingface.co/ozgurguler/ahenk-tts-tr) ·
[Ahenk Cep model kartı](https://huggingface.co/ozgurguler/ahenk-cep-tr) · [English](#english)

## Dinleyin

GitHub sayfada ses çalmaz: bağlantılar WAV dosyasını Hugging Face'te açar. Model kartlarında aynı örnekler oynatıcıyla
var. Ahenk Cep'i kurulum gerektirmeden tarayıcıda deneyebilirsiniz: [ozguradmin.github.io/ahenk-turkish-tts](https://ozguradmin.github.io/ahenk-turkish-tts/).

| Ne gösteriyor | Metin (yazıldığı gibi) | Model | Ses |
|---|---|---|---|
| Gülerek | Hahaha, bu fıkrayı daha önce hiç duymamıştım! | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_11_laughing.wav) |
| Neşeli | Harika haber! Sınavı geçtim, akşam hep birlikte kutlayalım! | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_07_cheerful.wav) |
| Sakin | Derin bir nefes alın, her şeyi adım adım birlikte halledeceğiz. | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_09_calm.wav) |
| Düz okuma | Eskişehir'den İzmir'e otobüsle gittik. | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_01_plain_ozel04.wav) |
| Para (trnorm: "doksan dokuz lira doksan dokuz kuruşa") | İndirimle 99,99 TL'ye düştü. | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_02_numbers_para09.wav) |
| Soru çifti, önce düz cümle | Cihazın garanti süresi 2 yıl. | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_03a_statement_garanti.wav) |
| Soru çifti, aynı cümle "mı?" ile | Cihazın garanti süresi 2 yıl mı? | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_03b_question_garanti.wav) |
| Uzun cümle, eğitimde yok (12 sn) | Karşılaştırma isteyenlere eski ve yeni paketi yan yana anlatırız; farkı müşteri kendisi görür. Akşam olduğunda … | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_04_long_heldout583922.wav) |
| Kızgın | Yeter artık! Üç gündür aynı sorun için arıyorum ve hâlâ kimse ilgilenmedi! | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_06_angry.wav) |
| Aynı uzun cümle | Karşılaştırma isteyenlere eski ve yeni paketi yan yana anlatırız; … | Ahenk Cep | [WAV](https://huggingface.co/ozgurguler/ahenk-cep-tr/resolve/main/samples/small_03_long_heldout583922.wav) |

Seçim: düz okuma, para, gülme ve duygular kör dinlemede dinleyicinin seçtiği klipler. Soru çifti ve
uzun cümle otomatik ölçüyle seçildi (hatasız ve en yüksek UTMOS). Hepsi var olan değerlendirme klipleri, yeniden
üretilmedi.

## Neler var

| Bileşen | Ne yapar | Boyut, donanım | Lisans |
|---|---|---|---|
| [Ahenk](https://huggingface.co/ozgurguler/ahenk-tts-tr) | VoxCPM2 için Türkçe LoRA: soru tonlaması, 6 duygu ve gülme, Türkçe ya da İngilizce talimat. Tarif: [`tts/voxcpm`](tts/voxcpm) | 2B + 36M LoRA, GPU | ağırlıklar CC BY 4.0 |
| [Ahenk Cep](https://huggingface.co/ozgurguler/ahenk-cep-tr) | Ahenk'in sesinden öğrenen küçük öğrenci model (EMA Lightning mimarisi). Kod: [`tts/small`](tts/small) | 8.6M, 34 MB, CPU'da RTF ~0,2 | ağırlıklar CC BY 4.0 |
| [`trnorm`](src/trnorm) | Yazılı Türkçeyi okunuş formuna çevirir: sayı, tarih, saat, para, yüzde, kısaltma, telefon; ekleri okunuşa uyar. Bağımlılığı yok. | saf Python | Apache-2.0 |
| [`trnorm` (JavaScript)](js) | trnorm'un tarayıcıda ve Node'da çalışan sürümü; Python'la aynı çıktı (`npm install trnorm`) | saf JavaScript | Apache-2.0 |
| [`trtts`](src/trtts) | EMA Lightning + trnorm: hazır, CPU'da çalışan Türkçe TTS | 8.6M, CPU | Apache-2.0 |
| [`benchmarks/tts`](benchmarks/tts) | TR-Zor (155 zor cümle, 13 kategori), TR-Soru (35 satır soru tonlaması) ve ASR ile geri okuyan ölçüm düzeneği | | Apache-2.0 |
| [`benchmarks/asr`](benchmarks/asr) | FLEURS-TR ve MediaSpeech-TR üzerinde Türkçe ASR karşılaştırması ([tablo](benchmarks/asr/results/LEADERBOARD.md)) | CPU | Apache-2.0 |

## Hızlı başlangıç

Ahenk Cep, işlemcide:

```bash
pip install torch ema-lightning==1.1.2 "git+https://github.com/ozguradmin/ahenk-turkish-tts"
```

```python
import sys
import soundfile as sf
from huggingface_hub import snapshot_download

d = snapshot_download("ozgurguler/ahenk-cep-tr", allow_patterns=["*.pt", "ahenk_cep.py"])
sys.path.insert(0, d)
from ahenk_cep import AhenkCep

tts = AhenkCep(f"{d}/student.pt", f"{d}/decoder.pt")
audio, sr = tts.say("Randevunuz 14.10.2026 saat 09:30'da; ücret 1.250 TL.")
sf.write("randevu.wav", audio, sr)
```

Ahenk (GPU, VoxCPM2) için kurulum ve duygu kullanımı [model kartında](https://huggingface.co/ozgurguler/ahenk-tts-tr).
Yalnız metin normalizasyonu:

```bash
trnorm "Saat 09:30'da 3. kata gelin, ücret 1.250,50 TL."
# Saat dokuz otuzda üçüncü kata gelin, ücret bin iki yüz elli lira elli kuruş.
```

trnorm'un kuralları ve örnekleri: [docs/TRNORM.md](docs/TRNORM.md).

## Ölçümler

TR-Zor: 155 zor cümle. Ayrılmış: Antalia veri setinin 20 uzun cümlesi (12-19 sn), hiçbir modelimizin eğitiminde yok.
Yargıç faster-whisper large-v3-turbo; referans ve çıktı `trnorm.normalize_for_eval` ile normalize edildi. WER %,
düşük iyi; parantezde tek kelime hatası olmayan cümle sayısı. UTMOS22 doğallığı tahmin eder (1-5, yüksek iyi).

| Model | Boyut | TR-Zor WER | Ayrılmış WER | UTMOS TR-Zor / ayrılmış |
|---|---|---:|---:|---:|
| **Ahenk** | 2B + LoRA | 2.54 (141/155) | 0.82 (16/20) | 3.70 / 3.14 |
| [Trendyol-TTS](https://huggingface.co/Trendyol/Trendyol-TTS) | 2B + LoRA | 3.98 (134/155) | **0.68** (16/20) | **3.82 / 3.73** |
| **Ahenk Cep** | 8.6M | **2.32** (144/155) | 0.95 (17/20) | 3.44 / 3.18 |
| [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning) + trnorm | 8.6M | 2.43 (145/155) | 1.49 (16/20) | 3.36 / 2.95 |
| [Antalia-2 Mini](https://huggingface.co/cloud0day3/antalia-mini) | 7.6M | 3.30 (140/155) | 1.22 (16/20) | 3.23 / 2.94 |

- Ahenk ve Trendyol-TTS bir A100'de, küçük modeller bir dizüstü GPU'da aynı yargıçla ölçüldü. Ahenk'in klipleri
  ikinci düzenekte yeniden puanlanınca sonuç 0,01 içinde aynı çıktı.
- **Kaybettiğimiz yerler:** Trendyol-TTS doğallık tahmininde önde, en çok uzun cümlede (3,73'e karşı 3,14); ayrılmış
  cümlelerde WER'i daha düşük (0,68'e karşı 0,82) ve daha hızlı (A100'de RTF 1,37'ye karşı 2,28). Trendyol kendi
  sesiyle konuşur; farkın bir kısmı sesten gelebilir. İşlemcide Ahenk Cep, EMA ve Antalia'dan yavaş (2 iş
  parçacığında RTF 0,20'ye karşı 0,13 ve 0,08).
- trnorm'un etkisi: EMA Lightning'in TR-Zor hatası kendi ön ucuyla %14,0, trnorm ile %1,9 (bu satır CPU'da int8
  yargıçla ölçüldü, yukarıdaki tabloyla doğrudan karşılaştırmayın; [ayrıntı](benchmarks/tts/results/tr_zor/RESULTS.md)).
- Bunlar otomatik ölçümlerdir. MOS çalışması yapılmadı.

**Kör dinleme.** Hepsi tek bir anadil dinleyiciyle (proje sahibi), her cümlede en iyi klip seçilerek yapıldı; bir
işarettir, istatistiksel sonuç değildir.
- 10 cümle, dört sistem rastgele sırada, ses düzeyi eşit: Ahenk 8, Ahenk Cep 2 (ikisi de soru), EMA Lightning 0,
  Antalia-2 Mini 0.
- Duygu talimatının dili, 20 karar: saf duyguda İngilizce talimat 6/6, ilk iki kelimesinde "c" ile başlayan bir kelime
  olan cümlelerde Türkçe talimat 10/10. Ahenk'teki `instruction_pick` bu sonuçla çalışır.

## Sınırlamalar

- **Sonu yükselen sorular yükselmez:** "Yarın geliyorsun?" gibi mı'sız sorular düz cümle gibi biter, iki modelde de.
  "mı" ile kurulan sorular doğru.
- **İngilizce talimattan sonra "c":** Ahenk'te baştaki "c" /k/, /s/ ya da /z/ gibi okunabilir ("Cam" → "Kam").
  `instruction_pick` Türkçe talimata geçer; gerekirse `prefer="tr"`.
- **"Telefon gibi" tını:** Ahenk'in bazı duygulu kliplerinde ses parlak ve cızırtılı (probda İngilizce talimatla
  20 klipte 7). Başka tohum deneyin.
- **Duraklarda nefes:** Ahenk'te kalır; paketteki `pause_clean` kısar.
- **Ahenk Cep'te üslup hafif:** "üzgün" yalnız 1,9 yarım ton pes ve %8 yavaş; gülme yok. Telefon hızı için henüz
  damıtılmadı.
- **Ahenk GPU ister:** A100'de 16 adımda RTF 2,28, gerçek zamandan yavaş.
- **Tek ses, yalnız Türkçe.** trnorm sözlüğünde olmayan yabancı adları yazıldığı gibi bırakır.

## Sorumlu kullanım

- Üretilen sesi gerçek bir insan kaydı gibi sunmayın; yapay olduğunu söyleyin ve mümkünse makinece okunur bir işaret
  ekleyin (AB Yapay Zekâ Yasası md. 50). Modellerde filigran yok; [AudioSeal](https://github.com/facebookresearch/audioseal)
  eklenebilir.
- Kimseyi taklit etmeyin. VoxCPM2 ses klonlayabilir; kimsenin sesini izni olmadan klonlamayın.
- Üretilen öfke, üzüntü ya da şefkati gerçek bir insanın tepkisi gibi sunup dinleyiciyi baskı altına almayın.
- Dolandırıcılık, sosyal mühendislik ve otomatik arama için kullanmayın.

## Lisans ve atıf

- Kod, belgeler, TR-Zor ve TR-Soru setleri: [Apache-2.0](LICENSE). Depodaki üçüncü taraf referans metinleri kendi
  lisanslarıyla gelir: [NOTICE](NOTICE).
- Ahenk ve Ahenk Cep ağırlıkları **CC BY 4.0**. Kullanırken ya da dağıtırken atıf zorunludur. Şöyle yazın:
  "Ahenk (https://github.com/ozguradmin/ahenk-turkish-tts), CC BY 4.0". Model kartlarındaki veri atıflarını da koruyun:
  *Alania Turkish Synthetic Speech by PatientDesk AI* ve *Alania Turkish Domain Text by PatientDesk AI* (CC BY 4.0).
- Taban modeller: VoxCPM2 (OpenBMB) ve EMA Lightning, ikisi de Apache-2.0.
- Akademik atıf: [CITATION.cff](CITATION.cff).

## Yol haritası

- Ahenk Cep'i telefonda gerçek zamanlı çalışacak hale getirmek (az adımlı damıtma)
- Küçük model için daha büyük öğretmen verisi, daha güçlü duygular ve soru tonlaması
- Tarayıcıda çalışan demo sayfası
- Ticari kullanıma açık, küçük bir Türkçe konuşma tanıma (STT) modeli

## Depo yapısı

| Klasör | İçerik |
|---|---|
| `src/trnorm`, `src/trtts` | metin normalizasyonu, EMA Lightning + trnorm |
| `js` | trnorm'un JavaScript sürümü (npm paketi `trnorm`) |
| `tts/voxcpm` | Ahenk eğitim ve ölçüm tarifi, `instruction_pick`, `pause_clean`, model kartı |
| `tts/small` | Ahenk Cep eğitimi, çıkarım kodu (`ahenk_cep.py`), model kartı |
| `benchmarks` | TR-Zor, TR-Soru, ASR karşılaştırması ve sonuçlar |
| `spaces` | Hugging Face demo uygulamaları |
| `docs` | çalışma notları |

Katkı: issue açın; en çok trnorm'un yanlış okuduğu cümleler işe yarar (girdi, beklenen okunuş, aldığınız çıktı).
`python -m pytest -q` geçmeli.

İletişim: destek@ozgurguler.tech

---

## English

Open Turkish speech tools: Ahenk (large model, GPU), Ahenk Cep (8.6M-parameter small model, CPU), trnorm (Turkish
text normalisation) and the TR-Zor / TR-Soru test sets.

> **Synthetic voice.** Ahenk and Ahenk Cep speak in one designed voice (f022) that is not a real person's voice or
> recording. Label what you generate as AI-generated (EU AI Act Art. 50).

### Listen

GitHub does not play audio in the page; the links in the table above open the WAV files on Hugging Face, and the
model cards have the same samples with players. You can try Ahenk Cep in your browser, no install: [ozguradmin.github.io/ahenk-turkish-tts](https://ozguradmin.github.io/ahenk-turkish-tts/). The plain, price, emotion and
small-model question samples were picked by a listener in blind tests; the question pair and the long sentence were
picked by automatic scores.

### What is here

- **[Ahenk](https://huggingface.co/ozgurguler/ahenk-tts-tr):** a Turkish LoRA for VoxCPM2 (2B + 36M LoRA, GPU) with
  question intonation, six emotions and laughter. Weights CC BY 4.0.
- **[Ahenk Cep](https://huggingface.co/ozgurguler/ahenk-cep-tr):** a small student of Ahenk's voice (EMA Lightning
  architecture, 8.6M parameters, 34 MB, RTF ~0.2 on 2 CPU threads). Weights CC BY 4.0.
- **trnorm:** dependency-free Turkish written → spoken form (numbers, dates, times, money, abbreviations, phone
  numbers, suffix agreement); also in JavaScript, with the same output ([`js`](js), `npm install trnorm`).
  **trtts:** EMA Lightning with trnorm as its front-end.
- **Benchmarks:** TR-Zor (155 hard sentences), TR-Soru (35 question-intonation rows), a Turkish ASR comparison.

### Quick start

See the Ahenk Cep code above (`pip install torch ema-lightning==1.1.2 "git+https://github.com/ozguradmin/ahenk-turkish-tts"`).
For Ahenk on a GPU, see its [model card](https://huggingface.co/ozgurguler/ahenk-tts-tr). `trnorm "text"` prints the
spoken form.

### Results

Table above; same judge (faster-whisper large-v3-turbo, trnorm evaluation normaliser, UTMOS22) for every row. Ahenk
reads hard text better than Trendyol-TTS (TR-Zor WER 2.54 vs 3.98) and Ahenk Cep is the best of the small models
(2.32 vs EMA Lightning 2.43, Antalia-2 Mini 3.30). Where we lose: Trendyol-TTS has higher predicted naturalness,
most on long sentences (UTMOS 3.73 vs 3.14), lower held-out WER (0.68 vs 0.82) and is faster (RTF 1.37 vs 2.28 on
an A100); on CPU Ahenk Cep is slower than EMA and Antalia (0.20 vs 0.13 and 0.08). Automatic metrics only, no MOS
study. Blind listening, one listener (the author), 10 sentences: Ahenk 8, Ahenk Cep 2, EMA Lightning 0, Antalia-2
Mini 0.

### Limitations

Questions that should rise at the end ("Yarın geliyorsun?") do not. After an English instruction Ahenk may read a
leading "c" the English way (use `prefer="tr"`). Some emotional clips sound bright and "phone-like". Breath in
pauses (use `pause_clean`). Ahenk Cep's styles are weak and it has no laughter. Ahenk needs a GPU (RTF 2.28 on an
A100). One voice, Turkish only.

### Responsible use

Label synthetic speech as such (EU AI Act Art. 50); there is no built-in watermark. No impersonation, no voice
cloning without consent, no deceptive emotion, no scams or robocalls.

### Licence and credit

Code, docs and test sets Apache-2.0 ([LICENSE](LICENSE), third-party texts in [NOTICE](NOTICE)). Ahenk and Ahenk Cep
weights are CC BY 4.0: credit them as "Ahenk (https://github.com/ozguradmin/ahenk-turkish-tts), CC BY 4.0" and keep
the Alania dataset attributions from the model cards. Base models VoxCPM2 and EMA Lightning are Apache-2.0.

### Roadmap

Real-time Ahenk Cep on phones (few-step distillation); more teacher data for stronger emotions and questions; a
browser demo; a small, commercially usable Turkish speech-recognition model.

Contact: destek@ozgurguler.tech
