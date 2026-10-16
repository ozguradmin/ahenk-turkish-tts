# Türkçe Piper sesi `tr_TR-antalia-medium`: ticari kullanıma açık, insan sesli (eğitim tarifi)

**Türkçe** · [English summary](#english-summary)

Ticari kullanıma açık, gerçek bir insan sesinden türetilmiş ve Raspberry Pi / Home Assistant'ta
çalışan bir Türkçe **Piper** sesi eğitmek için gereken her şey: sabitlenmiş ortam, veri hazırlığı,
eğitim (dizüstü GPU ya da Modal), ONNX'e dışa aktarma ve test.

- Uçtan uca CPU duman testi (veri → LJSpeech'ten ince ayar → devam ettirme → ONNX → Türkçe cümle):
  [SMOKE_TEST.md](SMOKE_TEST.md). Verinin ayrıntılı raporu: [DATA_REPORT.md](DATA_REPORT.md).
## Sonuç: 426 epoch, durduruldu

Modal'da A100 ile 426 epoch (13.632 adım, ~$13) eğitildi, ONNX'e aktarıldı ve ölçüldü
(`benchmarks/tts/results/tr_zor/`, Whisper large-v3-turbo ile geri-dönüş):

| | TR-Zor WER, trnorm ile | Kendi verisinden ayrılmış 20 cümle | Dinleyince |
|---|---:|---:|---|
| **bu ses, epoch 426** | **%24.6** (ham metin %41.2) | %10.7 | zor anlaşılıyor, robotik |
| Piper tr_TR-dfki | %3.5 | — | anlaşılır, robotik |
| EMA Lightning | %1.9 | — | doğal |

Hatalar sayılarda değil, sıradan kelimelerde ("mağaza" → "masa", "Tamam" → "çınar"). Model az
eğitilmiş. Bütçenin kalanıyla (~$14, toplam ~900 epoch) anlaşılırlık artardı. Ama VITS medium ve
espeak-ng'nin tavanı dfki düzeyi: anlaşılır, ama robotik. Aynı lisans sınıfında (ticari kullanıma
açık) EMA Lightning (Apache-2.0) bundan çok daha iyi. Bu yüzden eğitimi durdurduk; kalan GPU
bütçesi STT'ye gidiyor. Kullanıma hazır Türkçe TTS için: [`trtts`](../src/trtts) (EMA Lightning +
trnorm).

Tarif çalışır durumda ve kaldığı yerden devam edebilir. Piper/Home Assistant için ticari
kullanıma açık bir Türkçe ses isteyen biri, `--action train` ile aynı deneyi sürdürebilir.

> **Etik:** Bu ses, gerçek ve izin vermiş bir seslendirme sanatçısının kayıtlarından türetilir.
> Ürettiği konuşmanın **sentetik olduğunu belirtin**; sanatçıyı ya da herhangi bir gerçek kişiyi
> **taklit etmek için kullanmayın** (§2.3).

## 1. Ne ve neden

- **Bugün tek Türkçe Piper sesi `tr_TR-dfki-medium`:** verisi (dfki-ot-data) **CC-BY-NC-SA**, yani
  ticari kullanım yok; sesi de robotik.
- **Bu tarif:** **CC-BY-4.0** veri ([Antalia](https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus),
  tek profesyonel kadın seslendirmen, stüdyo kaydı, 5 saat) + **kamu malı** LJ Speech ile sıfırdan eğitilmiş
  Piper "medium" ağırlıkları (`rhasspy/piper-checkpoints`, MIT). Sonuç ağırlıklar **CC-BY-4.0**: atıfla
  ticari kullanım serbest. (dfki kontrol noktası bilerek kullanılmıyor: verisi NC-SA.)
- **Piper ekosistemi:** tek bir ~61 MB ONNX + JSON. Raspberry Pi, Home Assistant (Piper / Wyoming
  eklentisi), `piper` komut satırı, Python ve C++ (libpiper). Bu makinede 2 CPU iş parçacığıyla
  RTF ~0.07 (gerçek zamanın ~14 katı hızlı).

### Beklenen kalite (dürüst)

| | **bu ses** (hedef) | Piper tr_TR-dfki | EMA Lightning | pocket-tts-turkish |
|---|---|---|---|---|
| Mimari | VITS medium + espeak-ng | VITS medium + espeak-ng | 8.6 M, 48 kHz | ~110 M, 24 kHz |
| Lisans | **CC-BY-4.0** (atıf) | **CC-BY-NC-SA** (ticari değil) | Apache-2.0 | CC-BY-4.0 |
| TR-Zor WER, trnorm ile | %24.6 (epoch 426) | %3.5 | %1.9 | %3.2 |
| Temiz metin (Freya) WER | eğitimden sonra | %3.2 | %0.6 | %1.9 |
| CPU RTF (2 iş parçacığı) | ~0.08 (epoch 426) | 0.068 | 0.18 | 0.46 |
| Çalıştığı yer | Piper, Home Assistant, RPi | aynı | Python paketi | Python paketi |

Rakamlar bu deponun ölçümleri ([benchmarks/tts](../benchmarks/tts)). Beklentimiz:

- **Doğallık:** VITS + espeak-ng, EMA Lightning ve pocket-tts-turkish'ten **daha az doğal** olacak.
  Vurgu ve tonlama espeak-ng'nin kural tabanlı Türkçe fonemlerinden geliyor; model küçük, ses 22 kHz.
  Profesyonel stüdyo kaydı sayesinde dfki'den daha insan gibi bir ses umuyoruz, ama bu **ölçülmedi**.
- **Anlaşılırlık:** aynı mimarideki dfki gibi, trnorm ile %3–4 WER civarı bekliyoruz.
- **Üstünlükleri** kalite değil: **lisans** (ticari kullanım), **hız** (RPi'de bile gerçek zamandan
  hızlı) ve **ekosistem** (Home Assistant'ta seçip kullanmak). Piper ekosistemine ihtiyacınız yoksa,
  ticari kullanıma açık en iyi Türkçe ses bugün büyük olasılıkla EMA Lightning (Apache-2.0).

## 2. Veri, lisans, atıf, etik

### 2.1 Kaynaklar

| Kaynak | Lisans | Rol |
|---|---|---|
| [cloud0day3/antalia-voice-corpus](https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus) @ `9e67eeb261` | **CC-BY-4.0** | eğitim verisi: 1.073 kayıt, 5.008 sa, 24 kHz |
| `rhasspy/piper-checkpoints` `en/en_US/ljspeech/medium/lj-med_1000.ckpt` @ `37d1bea250` (sha256 `dcf2449b…`) | depo MIT; veri LJ Speech **kamu malı** | başlangıç ağırlıkları (üreteç + ayırt ediciler) |
| [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) @ `efffbfb226` | GPL-3.0 | eğitim kodu (setup_env.sh klonlar, depoya konmaz) |
| espeak-ng @ `724808c` (piper derlemesinin içinde) | GPL-3.0 | Türkçe fonemleştirme (`tr`) |
| `trnorm` (bu depo) | Apache-2.0 | metin ön ucu: eğitimde ve kullanımda aynı |

**Lisans özeti:** Eğitilen ağırlıklar CC-BY-4.0 (veri atfıyla). Çalıştırmak için gereken piper-tts ve
espeak-ng GPL-3.0: ticari kullanıma engel değil, ama bunları **gömen bir uygulamayı dağıtıyorsanız**
GPL yükümlülükleri o yazılıma uygulanır. Piper'ı ayrı bir servis olarak (Wyoming/HTTP) çalıştırmak en
sorunsuz yoldur.

### 2.2 Atıf satırları (model kartına aynen)

- *"Antalia (Patientdesk.ai) — Antalia Turkish single-speaker scripted speech, CC BY 4.0,
  https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus. Changes: resampled to 22.05 kHz,
  transcripts converted to spoken form with trnorm, 27 clips removed, 20 held out."*
- *"Base checkpoint: Piper lj-med_1000 (Bryce Beattie, rhasspy/piper-checkpoints, MIT), trained on
  LJ Speech (Keith Ito & Linda Johnson, public domain)."*

`export_and_test.sh` bu satırları ve etik notu içeren `MODEL_CARD.md`'yi her dışa aktarımla birlikte
yazar ([şablon](MODEL_CARD.template.md)).

### 2.3 Etik not: konuşmacının rızası

Veri kartına göre konuşmacı, kayıtları ticari bir sözleşmeyle yapan ve **kamuya yeniden dağıtıma izin
veren bir ek sözleşme imzalayan** profesyonel bir Türk seslendirme sanatçısı; adı kendi isteğiyle
gizli (`vd-speaker-e4440f86f3f6808056e3`). Rıza şu anlayışla verilmiş ve veri sahipleri buna uyulmasını
istiyor; aynı ricayı model kartında tekrarlıyoruz:

- Sesi ya da bununla eğitilmiş herhangi bir şeyi **sanatçıyı veya herhangi bir gerçek kişiyi taklit
  etmek**, birine söylemediği bir şeyi söyletmek için kullanmayın.
- Sentetik konuşmayı **gerçek insan kaydı gibi sunmayın**; dinleyicinin yanılabileceği her yerde
  (telefon, IVR, duyuru, video) **sentetik olduğunu belirtin**.
- **Dolandırıcılık, sosyal mühendislik, siyasi otomatik arama** yok.

CC-BY-4.0 altında bunlar lisans şartı değil, ricadır (veri sahiplerinin kendi Antalia 1 modeli bunları
Open RAIL-M ile bağlayıcı yapıyor). Yayınlarken bağlayıcı kısıtlar isteyip istemediğiniz sizin
kararınız; şablon şimdilik CC-BY-4.0 + rica diyor. Senaryolardaki numara, kod ve adlar veri kartına
göre **kurgusal** (`corpus-summary.json` 30 bölütü "PII regex" eşleşmesi olarak listeliyor; bunlar
bilerek yazılmış uydurma numaralar).

### 2.4 Veri hazırlama: sonuçlar ([DATA_REPORT.md](DATA_REPORT.md))

**Süre dağılımı:** Kayıtlar uzun: medyan **17.6 sn**, p10 12.7, p90 19.8, en uzun **20.24 sn**
(ortalama 16.8). 20 sn'yi aşan 59 kayıt var (0.33 sa), hepsi 20.0–20.24 sn; 20.5 sn'yi aşan yok.

**Uzunluk kararı: filtre yok, bölme yok.** Veri setinin bölütleyicisi kayıtları zaten 20 sn konuşma +
2 × 0.12 sn dolguyla sınırlamış; 20.0–20.24 sn'lik 59 kayıt aykırı değer değil, dağılımın tepesi.
Onları atmak verinin %6.6'sını götürür, en uzun örneği yalnız %1 kısaltır, yani belleğe katkısı yok.
Cümle sonundan bölmek için güvenilir bir yöntem yok (veride kelime zaman damgası yok; ASR ile hizalama
ayrı bir hata kaynağı olurdu). Varsayılan `--max-seconds 20.5`. Uzun kayıtların bedeli bellekte:
toplu iş boyları §3.3'te bu yüzden ölçülerek seçildi. Eski Piper'ın `max_phoneme_ids=400` sınırı bu
verinin ~%90'ını atardı (kayıt başına medyan 565, en çok 763 fonem kimliği); piper1-gpl'de böyle bir
sınır yok.

**Eğitim metni = `trnorm.normalize(transcript)`** (önce satır sonları/boşluklar tek boşluğa). Böylece
model, kullanımda ön ucun üreteceği okunuşları öğrenir. Veri setinin kendi `normalized_transcript`'iyle
karşılaştırma (1.073 kayıt):

| | kayıt |
|---|---:|
| aynı | 696 |
| yalnız noktalama/tırnak/büyük harf farkı | 136 |
| **kelime farkı** | **241** (çoğu `readback_confirmation` 119, `numeric_normalization` 65) |

Hangisi **sesle** daha uyumlu? Veri setindeki bağımsız ASR çözümüne (`asr_text`, ElevenLabs Scribe v2)
karşı CER (her iki taraf `normalize_for_eval`'dan geçerek): kelime farkı olan 241 kaydın **201'inde
trnorm**, 27'sinde veri setinin normalizer'ı daha yakın (13 eşit). Ortalama CER trnorm %0.9, veri
setininki %1.5. Tipik farklar: veri seti `SMS` → "se me se" (trnorm "esemes"), `389,50 TL` → "üç yüz
seksen dokuz virgül beş sıfır Türk lirası" (trnorm "… lira elli kuruş"), ekleri uyumlamıyor ("on sekiz
virgül dört'ünü"; trnorm "dördünü"), kodlardaki sayıları sıra sayısı okuyor ("doksan bin üç yüz on
beşinci").

**Metin–ses uyumu filtresi:** trnorm metni ASR çözümüyle CER > **0.08** farklıysa kayıt atılır. Eşik
veriden: rakam/kısaltma içermeyen 742 kayıtta ASR gürültüsü ortalama %0.28, **en kötü %7.2**; 0.08 bunun
hemen üstü. Atılan **27 kayıt (0.126 sa)**, hepsi sayı/kod yoğun; çoğu bölüt sınırının sayının ortasına
denk gelmesi (metin "45 arasında…", ses "15:20–16:45 arasında…") ya da kodların farklı okunması. Kalan:
**1.046 kayıt, 4.882 sa**; doğrulama için **20 kayıt** ayrıldı (11 kategoriye dağıtılmış, metni veride
tek olan, ASR uyumu ≤ 0.03) → eğitim **1.026 kayıt, 4.785 sa**.

**Bilinen okunuş farkları (sanatçı ↔ trnorm), trnorm'a geri bildirim:**
- `TL`: sanatçı en az 25 yerde **"Türk lirası"** diyor (ASR, 82 `TL`'nin 25'inde bunu açıkça yazıyor;
  kalanında "TL" yazdığı için okunuş belirsiz). trnorm rakamdan sonra "lira", yazıyla yazılmış sayıdan
  sonra ise **"te le"** yazıyor (kendi içinde de tutarsız).
- Kodlarda tire ve gruplama: `SPT-4600-M` → ASR "SPT **tire** 46 **çift sıfır** tire M" (sanatçı
  tireyi okuyor, rakamları ikişerli grupluyor); trnorm "se pe te dört bin altı yüz me" (tireyi atıyor,
  sayıyı tek bir sayı okuyor). Benzer: `KDZ-5800-R` → "KDZ tire 58 00 tire R".
- `www` → espeak-ng "dabılyu dabılyu dabılyu" okuyor (sanatçı "ve ve ve"); `.com` → sanatçı "kom".
- `Mbps` açılmıyor (sanatçı "megabit"); `7/24` → trnorm "yirmi dörtte yedi" (doğrusu "yedi yirmi dört").

Bunlar az sayıda kayıtta küçük metin–ses uyumsuzluğu bırakıyor (eşik altında kalanlar eğitimde).
Daha sıkı bir küme için `--max-asr-cer 0.05` (70 kayıt atılır).

## 3. Eğitim

### 3.1 Ortam

```bash
./tts/setup_env.sh            # CUDA 12.8 (RTX 20xx..50xx, L4, L40S, A100, H100)
./tts/setup_env.sh --cpu      # yalnız CPU (duman testi, dışa aktarma, sentez)
source tts/path.sh            # her yeni terminalde
```

Sabitler: Python 3.12, torch/torchaudio **2.8.0** (2.9'dan itibaren `torch.onnx.export` varsayılan olarak
dynamo dışa aktarıcısını kullanıyor; piper'ın `export_onnx.py`'si eski yola göre yazılmış), Lightning 2.6.6, jsonargparse 4.52.0, onnx 1.23.2, onnxruntime
1.30.0, piper1-gpl `efffbfb226` (2026-09-28: v1.8.0 + öğrenme hızı zamanlayıcısının hiç adım atmaması
hatasının ve "devam ederken warmstart yapma" hatasının düzeltmeleri), espeak-ng `724808c`.
Tüm bağımlılıklar [constraints-tts.txt](constraints-tts.txt) ve [locks/](locks) altında donduruldu.
Betik espeak-ng'yi kaynaktan derler (git, C derleyicisi ve GitHub erişimi gerekir; ~1 dk). Kurulum
`.venv-piper`'a, piper1-gpl `tts/piper1-gpl`'e (git-ignore'lu).

### 3.2 Veri

```bash
python tts/prepare_antalia.py                 # ~3.5 dk: 1.046 kayıt indir + sha256 + 22.05 kHz, rapor
python tts/prepare_antalia.py --piper-cache   # + piper'ın fonem/spektrogram önbelleği (~2.5 dk, 4.2 GB)
```

Çıktı `data/tts_train/antalia/` (git-ignore'lu): `wav/`, `metadata.csv` (Piper biçimi `dosya|metin`,
`csv` modülüyle; tırnaklı metinler doğru okunur), `heldout.txt`, `heldout.csv`, `clips.jsonl` (kayıt
başına her şey), `PREP_REPORT.md`. ~0.75 GB ses + isteğe bağlı 4.2 GB önbellek.

### 3.3 Eğitim

```bash
./tts/train.sh --preset a100 --max-epochs 1000              # bulut GPU
./tts/train.sh --preset laptop8gb --hours 4 --max-epochs 1000   # dizüstü: 4 saatte temiz durur (çıkış 75)
./tts/train.sh --preset laptop8gb --hours 4 --max-epochs 1000   # ertesi gün: aynı komut, kaldığı yerden
./tts/train.sh --stop                                       # çalışan eğitime "kaydet ve dur"
tensorboard --logdir data/tts_train/exp/antalia-medium     # kayıplar, doğrulama sesleri
```

- **LJSpeech'ten başlangıç:** ilk koşu `--model.warmstart_ckpt lj-med_1000.ckpt` ile tüm ağırlıkları
  (üreteç **ve** ayırt ediciler, 784 tensör) kopyalar; optimizer, epoch ve öğrenme hızı sıfırdan başlar.
  piper1-gpl belgelerinin önerdiği `--ckpt_path lj-med_1000.ckpt` güncel sürümlerle **çalışmıyor**:
  Lightning 2.6 kontrol noktasının hiperparametrelerini `torch.load(weights_only=True)` ile okuyor ve
  bu eski (Lightning 1.9.5) dosyadaki `PosixPath`'leri açamıyor; ayrıca eski `sample_bytes` parametresi
  reddediliyor ([aynı tespit](https://github.com/LewisSmallwood/piper-tts-training)). warmstart bu
  iki sorunu yamasız aşar; ayrıca `--max-epochs` ve öğrenme hızı azalması bu koşunun epoch'larına göre
  hesaplanır (LJ kontrol noktası epoch 999'da; `--ckpt_path` ile sayaç oradan devam ederdi).
- **Devam etme:** `<exp>/checkpoints/last.ckpt` varsa aynı komut oradan sürer (ağırlıklar, optimizer,
  öğrenme hızı, epoch). `last.ckpt` her `--ckpt-every` epoch'ta (varsayılan 5), sürenin sonunda, `--stop`
  ile ve kesinti/SIGTERM'de (bulutta öncelik kaybı) **atomik** yazılır (geçici dosya + yeniden adlandırma).
  Son durum `RESUME.json`'da; epoch süreleri `epochs.jsonl`'da.
- **Öğrenme hızı:** piper1-gpl, `--max-epochs`'tan bir azalma hesaplar (bitişte başlangıcın %5'i). Bu
  yüzden `--max-epochs`'u baştan seçin ve devam ederken aynı tutun.
- **Doğrulama:** ayrılan 20 kayıtla (piper'ın her başlangıçta değişen rastgele %10'u yerine sabit; bkz.
  [piper_fit.py](piper_fit.py)) her `--val-every` (5) epoch'ta `val_mel`; 4 cümle sentezlenip TensorBoard'a
  yazılır ve UTMOS ile `val_mos` (doğallık tahmini, 1–5) hesaplanır.
- **Kontrol noktaları:** `last.ckpt` (tam durum, 0.85 GB); `val_mel`'e göre en iyi 3 ve `val_mos`'a göre
  en iyi 3 (yalnız ağırlık, 0.28 GB); her 100 epoch'ta bir `snapshot-epoch=N.ckpt` (dinleme testi için;
  N, Lightning'in 0'dan başlayan sayacı: `snapshot-epoch=499` = 500 epoch).
  piper geliştiricileri de seçimi dinleyerek yapmayı öneriyor (`val_mel` erken doyuyor).
- **Kaç epoch?** Yayımlanmış piper1-gpl ince ayarı: ~970 kayıt, 1000 epoch (batch 32). Antalia'da bir
  epoch 33 adım (batch 32), 1000 epoch ≈ 33 bin adım. Dil değişimi (İngilizce → Türkçe) olduğu için
  800–1000 öneriyoruz; 300–500'den sonra snapshot'ları dinleyip karar verin.

**Toplu iş boyu (VRAM).** Bu verinin kayıtları uzun olduğu için bellek, kayıt sayısı × uzunlukla
büyüyor. `tools/step_bench.py` ile ölçüldü (fp32, gerçek eğitim adımı, en kötü durum 20.24 sn / 763
fonem kimliği): sabit ~1.7 GB (gradyanlar, Adam durumu) + **kayıt başına ~0.7 GB** (6.6 sn'lik
LJSpeech uzunluğunda ~0.3 GB). Ölçüm CPU'da yapıldı (bu makinede GPU yok); GPU'da ±%30 fark
olabilir, `--action probe` gerçek kartta ölçer.

| `--preset` | kart | batch | tahmini tepe bellek (en kötü toplu iş) |
|---|---|---:|---|
| `a100` / `h100` | 80 GB | 32 | ~24 GB (+ ayırıcı payı ~28 GB) |
| `l40s` | 48 GB | 32 | ~28 GB |
| `l4` | 24 GB | 16 | ~13 GB (32 sığmaz) |
| `laptop8gb` | 8 GB (RTX 3060/4060 Laptop) | 4 | ~4.5 GB + CUDA ~0.5 GB |
| `cpu` | — | 4 | duman testi |

`--batch-size N` ön ayarı ezer; RTX 30xx+ kartlarda `--precision bf16-mixed` belleği yaklaşık yarıya
indirir (ölçmedik; varsayılan, yayımlanmış ince ayarlar gibi fp32). Dizüstünde `train.sh`, STT
tarifindeki sıcaklık korumasını (`stt/thermal_guard.py`: 83 °C'de duraklat, 72 °C'de sürdür) kendiliğinden
başlatır. Ama dizüstünde 1000 epoch ~100+ saat sürer (yayımlanmış RTX 2070 / batch 4: ~970 kayıtta
242 sn/epoch; bu verinin kayıtları kayıt başına ~2 kat pahalı); asıl eğitim için Modal öneriliyor.

### 3.4 Dışa aktarma ve test

```bash
./tts/export_and_test.sh                                      # last.ckpt
./tts/export_and_test.sh --ckpt data/tts_train/exp/antalia-medium/checkpoints/snapshot-epoch=499.ckpt
```

Çıktı `<exp>/export/<kontrol noktası>/`: `tr_TR-antalia-medium.onnx` (~61 MB), `.onnx.json` (Home
Assistant'ın beklediği dil/veri/kalite alanlarıyla), `MODEL_CARD.md`, `samples/` (TR-Zor'dan 8 cümle,
farklı kategorilerden birer (`--n`), ve 20 ayrılmış cümle; trnorm → piper çalışma zamanı, CPU, 2 iş parçacığı;
`summary.json`'da RTF ve ayrılmış cümlelerde sentez/gerçek süre oranı) ve `piper` komut satırıyla bir
deneme. Ses ayarları `.onnx.json`'daki `inference` alanında (yeniden dışa aktarmadan değişir):
`--noise-scale` (0.667; ~0.4–0.5 daha tutarlı), `--length-scale` (>1 daha yavaş).

**Home Assistant:** `.onnx` ve `.onnx.json`'ı Piper eklentisinin `/share/piper/` klasörüne (ya da
`wyoming-piper --data-dir`) kopyalayın, eklentiyi yeniden başlatın, sesi seçin. Home Assistant trnorm
çalıştırmaz: sayıları ve kısaltmaları espeak-ng'nin kendi kuralları okur. En iyi sonuç için metni
önceden `trnorm.normalize`'dan geçirin.

### 3.5 Bulut: Modal (önerilen)

Aynı betikler [Modal](https://modal.com)'da, `setup_env.sh` ile kurulmuş sabit bir imajda (CUDA 12.8)
çalışır ([modal_app.py](modal_app.py)). Veri, önbellek, temel ağırlık ve deneyler kalıcı Modal Volume
`turkish-tts-data`'da durur; her adım durdurulup devam ettirilebilir.

```bash
pip install modal && modal setup
modal run tts/modal_app.py --action prepare                 # CPU 4 çekirdek, ~7 dk, ~$0,03
modal run tts/modal_app.py --action probe                   # ~12 dk GPU (~$0,6): gerçek hız + bellek + maliyet tablosu
modal run --detach tts/modal_app.py --action train --max-usd 30 --max-epochs 800
modal run tts/modal_app.py --action status                  # ilerleme (ucuz, CPU)
modal run tts/modal_app.py --action export                  # CPU: ONNX + örnekler + RTF
modal volume get turkish-tts-data exp/antalia-medium/export/last ./antalia-export
```

- **GPU:** `TTS_GPU=A100-80GB|L40S|L4|H100` (varsayılan **A100-80GB**, §4).
- **Bütçe kilidi:** `--max-usd`, GPU + CPU/RAM saat ücretinden bir süre sınırına çevrilir
  (`train.sh --hours`, epoch sınırında temiz durur). Modal saniye başına ücretlendirir. GPU geri
  alınırsa (preemption) iş kendiliğinden yeniden başlar, `last.ckpt`'den sürer; Volume'deki defter
  (`modal_ledger/`) önceki denemelerin süresini düşer. Ayrıca Modal'da çalışma alanı harcama sınırı koyun.
- `--max-epochs`'u bütçeye göre seçin: `probe` epoch başına süreyi ve maliyeti yazar. Bütçe biterse
  aynı `train` komutu (aynı `--max-epochs`) sonraki ay kaldığı yerden devam eder.
- Belirli bir kontrol noktasını dışa aktarmak: `--action export --ckpt snapshot-epoch=499.ckpt`.
- İsteğe bağlı: daha yüksek Hugging Face limiti için `modal secret create huggingface HF_TOKEN=...` ve
  `TTS_HF_SECRET=huggingface`.

Modal bu ortamdan erişilebilir değildi; konteynerin çalıştırdığı komut satırları
[tools/modal_dryrun.py](tools/modal_dryrun.py) ile bu makinede CPU'da birebir çalıştırıldı
([SMOKE_TEST.md](SMOKE_TEST.md) §6).

### 3.6 Windows'tan Modal ile çalıştırma (PowerShell)

Yerel tarafta yalnızca Python ve `modal` istemcisi gerekir; bash/WSL gerekmez. Her şey Modal'ın Linux
konteynerlerinde çalışır.

```powershell
# bir kerelik: Python 3.10+ (python.org ya da: winget install Python.Python.3.12) ve git
git clone https://github.com/ozguradmin/ahenk-turkish-tts
cd ahenk-turkish-tts
git checkout claude/turkish-stt-tts-models-j62h9n
py -m venv .venv-modal
.\.venv-modal\Scripts\Activate.ps1      # hata verirse: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install modal
modal setup                             # tarayıcıda Modal girişi

# veri (CPU, ~7 dk) ve hız ölçümü (~12 dk GPU)
modal run tts/modal_app.py --action prepare
modal run tts/modal_app.py --action probe

# başka bir GPU'yu denemek için (bu PowerShell oturumunda geçerli):
$env:TTS_GPU = "L4"
modal run tts/modal_app.py --action probe
Remove-Item Env:TTS_GPU                 # varsayılana (A100-80GB) dön

# eğitim: arka planda, bütçe kilitli; bilgisayarı kapatabilirsiniz
modal run --detach tts/modal_app.py --action train --max-usd 30 --max-epochs 800
modal run tts/modal_app.py --action status

# dışa aktarma ve indirme
modal run tts/modal_app.py --action export
modal volume get turkish-tts-data exp/antalia-medium/export/last .\antalia-export
```

Windows'ta git satır sonlarını CRLF'ye çevirse bile sorun olmaz: depo kökündeki `.gitattributes`
betikleri LF tutar ve `modal_app.py` konteynerde betikleri yine de LF'ye çevirir.

### 3.7 TR-Zor ile ölçüm

Dışa aktarılan ses, mevcut benchmark düzeneğine `piper-antalia` adıyla eklenir; yalnız ortam değişkeni
verildiğinde kayıtlı olur (diğer sistemlerin davranışı ve varsayılan listeler değişmez):

```bash
export PIPER_ANTALIA_ONNX=$PWD/antalia-export/tr_TR-antalia-medium.onnx
SYSTEMS="piper-antalia" FRONTENDS="raw trnorm" benchmarks/tts/run_all.sh tr_zor freya_sample40
```

## 4. Süre ve maliyet tahmini

**Ölçülenler (bu makine, 2 CPU iş parçacığı, fp32):** gerçek eğitimde batch 4 → **11–16 sn/adım**;
`step_bench.py` ile kayıt başına adım maliyeti ≈ 0.51 sn + 0.127 sn × (kayıt süresi): 20.24 sn'lik
kayıt 3.08 sn, LJSpeech uzunluğunda (6.57 sn) 1.34 sn. CPU'da çalışan monotonik hizalama (MAS) 32'lik
toplu işte ~0.13–0.2 sn (LJSpeech uzunluğunda ~0.02 sn).

**CPU'dan GPU'ya geçiş varsayımı:** CPU hızını GPU'ya bir çarpanla taşımıyoruz. CPU'yu yalnız
"Antalia'nın bir epoch'u, LJSpeech'in bir epoch'unun kaçı?" sorusu için kullanıyoruz: 1.026 kayıt ×
2.64 sn'ye karşı 13.100 × 1.34 sn → **0.154**. GPU süresini yayımlanmış ölçümlere bağlıyoruz:

1. **RTX 3090, LJSpeech, batch 32: 2.000 epoch ≈ 10 gün → 432 sn/epoch**
   ([rhasspy/piper #430](https://github.com/rhasspy/piper/discussions/430)) → Antalia epoch'u
   0.154 × 432 ≈ **66 sn** + MAS ~5 sn.
2. **H100, piper1-gpl ince ayarı, ~970 kayıt, batch 32: ~16 sn/epoch**
   ([LewisSmallwood/piper-tts-training](https://github.com/LewisSmallwood/piper-tts-training)); kayıtları
   LJSpeech uzunluğundaysa Antalia'da ~38 sn/epoch. Aynı kaynak: "küçük VITS büyük GPU'yu dolduramıyor;
   batch 64 ve karma hassasiyet hızlandırmadı" → H100'e para vermeye gerek yok.
3. GPU oranları (PyTorch convnet eğitimi, TF32): A100 SXM4 = 1.92 × RTX A6000
   ([Lambda](https://lambda.ai/blog/nvidia-rtx-a6000-benchmarks)) ve A6000 = 0.92 × RTX 3090
   ([Lambda](https://lambda.ai/blog/nvidia-rtx-a6000-vs-rtx-3090-benchmarks)) → **A100 ≈ 1.75 × 3090**;
   RTX 4090 = 1.3–1.9 × 3090 ([Lambda](https://lambda.ai/blog/nvidia-rtx-4090-vs-rtx-3090-deep-learning-benchmark)),
   L40S aynı yonga, daha düşük saat → ≈ 1.4 × 3090; **L4** için yayımlanmış ölçüm bulamadık:
   özelliklerinden (~30 TFLOPS FP32, 300 GB/sn) ≈ 0.5 × 3090, ve 24 GB'a ancak batch 16 sığıyor.

| GPU (Modal, $/sa GPU + 0,32 CPU/RAM) | tahmini sn/epoch | $/epoch | 500 epoch | 800 epoch | 1000 epoch |
|---|---:|---:|---|---|---|
| **A100-80GB** ($2,50) | ~45 | ~0,035 | 6,3 sa · ~$18 | 10 sa · ~$28 | 12,5 sa · ~$35 |
| L40S ($1,95) | ~55 | ~0,035 | 7,6 sa · ~$17 | 12 sa · ~$28 | 15 sa · ~$35 |
| L4 ($0,80), batch 16 | ~140 | ~0,043 | 19 sa · ~$22 | 31 sa · ~$35 | 39 sa · ~$43 |
| H100 ($3,95) | ~40 | ~0,047 | 5,6 sa · ~$24 | 9 sa · ~$38 | 11 sa · ~$47 |

Epoch süresine doğrulama (5 epoch'ta bir, 20 kayıt + 4 sentez + UTMOS) ve kontrol noktası yazımı için
~2 sn eklendi. Belirsizlik **±%50**: bu modelin GPU oranları ölçülmedi, 3090 ölçümü eski piper
kodundan, uzunluk ölçeklemesi CPU'dan. **Varsayılan GPU A100-80GB:** tahmini epoch maliyeti L4'ten
düşük (L4 bellek yüzünden batch 16'ya iniyor ve bellek bant genişliği A100'ün 1/7'si), duvar saati ~3 kat
kısa (öncelik kaybına daha az açık), batch 32 rahat sığıyor. L40S benzer maliyetli bir alternatif.
**Asıl eğitimden önce `--action probe` çalıştırın** (A100 ~$0,6; L4 ~$0,25): gerçek sn/epoch, tepe bellek
ve 300/500/1000 epoch maliyetini yazar; L4 ya da L40S epoch başına daha ucuz çıkarsa `TTS_GPU` ile geçin.
Veri hazırlama (CPU, ~7 dk) ~$0,03, dışa aktarma ~$0,01. Aylık $30 kredi ~800 epoch'a (A100) yetiyor;
1000 epoch iki ayda, aynı komutla devam ederek biter.

## 5. Riskler ve bilinen sorunlar

- **Gerçek GPU'da hiç çalışmadı:** CUDA yolu, gerçek bellek ve hız doğrulanmadı (bu makinede GPU yok);
  önce `probe`.
- **Kalite ölçülmedi:** duman testi modelleri 4–21 toplu iş eğitildi; ayrılmış cümleleri gerçek kayıttan
  ~2 kat hızlı okuyorlar (sentez/gerçek süre oranı 0.48–0.61: LJSpeech'in süre tahmincisi henüz Türkçeye
  uymadı), `val_mos` 1.24 → 1.30. Gerçek kalite ancak eğitimden sonra TR-Zor ve dinleme testiyle belli olur.
- **espeak-ng Türkçe fonemleri** kural tabanlı ve yaklaşık; vurgu/tonlama doğallığı sınırlı. Eğitim ve
  kullanımdaki espeak-ng sürümü aynı olmalı (piper-tts ≥ 1.8.0 aynı `724808c`'yi gömüyor).
- **Metin–ses uyumsuzlukları:** eşik altında kalan ~40 kayıtta küçük farklar (TL/"Türk lirası", "tire",
  "www") — §2.4. Sıkı küme: `--max-asr-cer 0.05`.
- **Tek konuşmacı, okuma konuşması, çağrı merkezi üslubu:** sohbet, duygu ve uzun anlatı sınırlı.
- **Lightning/piper sürüm hassasiyeti:** sarmalayıcı (`piper_fit.py`) Lightning 2.6.6'nın iç
  davranışına dayanıyor (ModelCheckpoint sırası, eksik metrik istisnası); sürümler sabit.
- **Lisans:** ağırlıklar CC-BY-4.0, ama çalışma zamanı (piper-tts, espeak-ng) GPL-3.0 (§2.1).
- **Etik:** CC-BY-4.0 taklit yasağını bağlayıcı yapmıyor; yayınlarken model kartındaki notu koruyun.

## 6. Dosyalar

| Dosya | Ne yapar |
|---|---|
| [setup_env.sh](setup_env.sh), [path.sh](path.sh) | sabit ortam (CPU/CUDA), piper1-gpl klonu ve derlemesi |
| [requirements-tts.txt](requirements-tts.txt), [constraints-tts.txt](constraints-tts.txt), [locks/](locks) | sürümler |
| [prepare_antalia.py](prepare_antalia.py) | indirme, sha256, 22.05 kHz, trnorm, filtreler, ayrılmış küme, rapor, piper önbelleği |
| [piper_fit.py](piper_fit.py) | `python -m piper.train fit` + sabit doğrulama kümesi, atomik `last.ckpt`, süre sınırı, epoch günlüğü |
| [train.sh](train.sh) | eğitim: warmstart/devam, ön ayarlar, süre sınırı, `--stop`, sıcaklık koruması |
| [export_and_test.sh](export_and_test.sh) | ONNX + `.onnx.json` + model kartı + TR-Zor sentezi + RTF + `piper` CLI |
| [modal_app.py](modal_app.py) | Modal: prepare / probe / train / status / export, bütçe kilidi |
| [tools/step_bench.py](tools/step_bench.py) | eğitim adımı süresi ve tepe bellek (CPU/GPU) |
| [tools/synth_test.py](tools/synth_test.py) | sentez testi, RTF, süre oranı |
| [tools/modal_dryrun.py](tools/modal_dryrun.py) | Modal konteyner komutlarını yerelde CPU'da çalıştırır |
| [MODEL_CARD.template.md](MODEL_CARD.template.md) | dışa aktarımla yazılan model kartı (lisans, atıf, etik) |
| [DATA_REPORT.md](DATA_REPORT.md), [SMOKE_TEST.md](SMOKE_TEST.md) | veri raporu, duman testi |

---

## English summary

A recipe for a **commercially usable Turkish Piper voice** (`tr_TR-antalia-medium`, VITS "medium",
22,050 Hz). The only Turkish Piper voice today (`tr_TR-dfki`) is non-commercial (CC-BY-NC-SA data) and
robotic. This one fine-tunes the **public-domain LJSpeech** Piper medium checkpoint
(`rhasspy/piper-checkpoints`, MIT) on the **Antalia voice corpus** (CC-BY-4.0, one consenting
professional female voice actor, 5.008 h), so the weights can be released under CC-BY-4.0 and run
anywhere Piper runs (Raspberry Pi, Home Assistant via Wyoming Piper, ~61 MB ONNX, RTF 0.068 on 2 CPU
threads). **No voice has been trained yet**; the pipeline was smoke-tested end to end on CPU
([SMOKE_TEST.md](SMOKE_TEST.md)).

- **Expected quality, honestly:** Piper/VITS + espeak-ng will sound less natural than EMA Lightning or
  pocket-tts-turkish (rule-based espeak-ng prosody, small model, 22 kHz); intelligibility should be
  close to the dfki voice (TR-Zor WER ≈ 3–4 % with trnorm). Its advantages are licence, speed and the
  Piper/Home Assistant ecosystem. Disclose the voice as synthetic and never use it to impersonate the
  actor or anyone else (the corpus owners' consent terms, repeated in the model card).
- **Data** ([DATA_REPORT.md](DATA_REPORT.md)): clips are long (median 17.6 s, max 20.24 s; none above
  20.5 s), so no duration filtering or splitting. Training text is `trnorm.normalize(transcript)`; against
  the corpus' independent ASR transcript it matches the audio better than the corpus' own
  `normalized_transcript` in 201 of 241 clips where they differ. 27 clips whose text disagrees with the
  ASR (CER > 0.08, just above the clean-text ASR noise ceiling of 0.072) are dropped; 20 clips are held
  out: 1,026 training clips, 4.785 h.
- **Training:** piper1-gpl at a pinned commit (`efffbfb226`), torch 2.8.0, Lightning 2.6.6. Starts from
  the LJSpeech weights via `--model.warmstart_ckpt` (the documented `--ckpt_path` route fails on this
  old checkpoint with Lightning 2.6 / torch ≥ 2.6); resumable atomic `last.ckpt`, time limit, fixed
  held-out validation (`val_mel`, UTMOS `val_mos`), batch 32 on A100/L40S, 16 on L4, 4 on 8 GB laptops
  (activation memory measured: ~0.7 GB per 20 s clip, fp32).
- **Cost (Modal, estimated ±50 %):** ~45 s/epoch on an A100-80GB → 800 epochs ≈ 10 h ≈ $28,
  1000 epochs ≈ $35; L4 ≈ 140 s/epoch, more expensive per epoch. Grounded on published Piper timings
  (RTX 3090 LJSpeech 432 s/epoch; H100 piper1-gpl fine-tune 16 s/epoch) scaled to this corpus by a
  CPU-measured per-clip cost model. Run `--action probe` first for real numbers. Windows users run
  everything from PowerShell (§3.6).
