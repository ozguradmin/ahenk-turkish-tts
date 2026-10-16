# Türkçe akış (streaming) ASR — izinli lisanslı veriyle, Apache-2.0

Bu klasör, **ticari kullanıma açık (Apache-2.0)**, küçük ve **akışlı** (streaming) bir
Türkçe konuşma tanıma modelini sıfırdan eğitmek için gereken her şeyi içerir:
veri indirme/hazırlama, token modeli, dizüstü bilgisayara uygun eğitim, sıcaklık koruması,
sherpa-onnx'e dışa aktarma ve test.

- **Model:** icefall Zipformer2 transducer (pruned RNN-T), `--causal 1` (akışlı).
  İki boy: `small` **24,5 M** parametre (telefon/RPi/WASM için), `medium` **67,1 M**.
- **Çalışma ortamı:** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — CPU, Android, iOS,
  WASM, Raspberry Pi. 320 ms'lik parçalarla (chunk 16) gerçek zamanlı çözer; fp32 ve int8.
- **Neden?** Bugünkü en iyi küçük Türkçe akışlı modeller ticari kullanıma uygun değil:
  `mihuai/turkish-stt` (66 M) gelir sınırlı ve rakip model eğitimini yasaklayan bir lisansla,
  `duxx/turkish-stt-zipformer` (23,6 M) CC-BY-NC ile yayımlanmış. Bu tarif **yalnızca izinli
  lisanslı veri** kullanır; bu modellerin çıktıları (etiket, damıtma, filtreleme) **hiçbir aşamada
  kullanılmaz**.
- **Araçlar (sabitlenmiş sürümler):** Python 3.11, torch/torchaudio 2.8.0 (CUDA 12.8 ya da CPU),
  k2 1.24.4.dev20260625, lhotse 1.33.0, icefall `3f848bb6d0ac` (2026-07-16; klonlanır, depoya
  konmaz), sentencepiece 0.2.1, onnx 1.19.1, onnxruntime 1.23.2, sherpa-onnx 1.13.8.
  Tüm geçişli bağımlılıklar `constraints-stt.txt` ve `locks/` altında donduruldu.

> CPU üzerinde uçtan uca duman testi (veri → eğitim → ONNX → sherpa-onnx) bu makinede
> çalıştırıldı: komutlar ve süreler [SMOKE_TEST.md](SMOKE_TEST.md) dosyasında.

**Hızlı başlangıç (dizüstü, ayrıntılar aşağıda):**

```bash
./stt/setup_env.sh && source stt/path.sh                                   # §2
./stt/prepare.sh --cv-train-from validated --num-workers 6 --fbank-device cuda   # §3, ~3–5 sa
./stt/train.sh --preset small --hours 6       # §4 — bitene kadar aynı komutu tekrarlayın
./stt/export_and_test.sh --preset small       # §5 — ONNX (fp32+int8) + WER/CER
```

## 1. Veri ve lisanslar

| Kaynak | Lisans | Saat (yaklaşık) | Kullanım |
|---|---|---|---|
| [google/fleurs](https://huggingface.co/datasets/google/fleurs) `tr_tr` | CC-BY-4.0 | ~12 (eğitim ~8,4) | eğitim + dev + **test** |
| [issai/Turkish_Speech_Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus) (TSC) | MIT | 218 | eğitim + dev + **test** |
| Common Voice 17 `tr` ([fixie-ai/common_voice_17_0](https://huggingface.co/datasets/fixie-ai/common_voice_17_0) Parquet aynası) | CC0-1.0 | train ~45, validated ~120–140 | eğitim + dev + **test** |
| — ya da yerel Mozilla Data Collective sürümü (`--cv-dir`) | CC0-1.0 | sürüme göre | aynı |
| [cloud0day3/alania-synthetic-speech-tr](https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr) yalnız `cc-by` | CC-BY-4.0 | 2.565 mevcut; **toplamın ≤ %45'i** | eğitim (sentetik) + dev |
| [cloud0day3/alania-domain-text-tr](https://huggingface.co/datasets/cloud0day3/alania-domain-text-tr) | CC-BY-4.0 | metin | yalnız token modeli |
| *İsteğe bağlı:* [serdarcaglar/turkish-tts-audiobooks](https://huggingface.co/datasets/serdarcaglar/turkish-tts-audiobooks) | open-weights-with-attribution (kapılı) | 1.292 | `--audiobooks H` |
| *Faz 2:* [sarulab-speech/yodas2_sidon](https://huggingface.co/datasets/sarulab-speech/yodas2_sidon) `tr000` | CC-BY-3.0 | ~1.500 video → **~500 kullanılabilir** | `--yodas-shards N` |

Varsayılan tarif (`--cv-train-from validated` ile): gerçek konuşma ~340 sa + sentetik ~280 sa
≈ **600 sa**. Sentetik pay `make_mix.py` tarafından her durumda `--synthetic-ratio` (0,45) ile sınırlanır.

**Ayrıntılar ve tuzaklar**

- **Transkriptler:** hepsi `trnorm.normalize_for_eval` ile küçük harfli, noktalamasız, sayıları
  okunuşuyla yazılmış hale getirilir (Alania'da önce `text_spoken`). Türkçe alfabe (+q, w, x)
  dışında harf kalan cümleler atılır; Türkçe olmayan aksanlar (é, ñ…) sadeleştirilir.
- **ISSAI TSC** transkriptleri özel isim eklerini ayırmış ("erdoğan ın"); kesin ek olan parçalar
  (`ın, nin, ları, …`) geri birleştirilir, belirsizlere ("de", "da", "ya", "tür", "un"…) dokunulmaz.
  Tek 21,4 GB'lık tar.gz; indirme kaldığı yerden devam eder, dönüştürmeden sonra silinebilir.
- **Common Voice:** varsayılan resmi bölünme (`tr/train`, `tr/validation`, `tr/test`).
  `--cv-train-from validated`: test/dev'deki **cümlelerin ve konuşmacıların** hiçbirini içermeyen
  tüm doğrulanmış kayıtlar (~3 kat veri, sızıntı yok). Mozilla Data Collective'den indirilen
  yerel sürüm için `prepare/commonvoice.py --cv-dir .../tr` (aynı çıktı adları; ikisinden biri).
- **Alania (sentetik, VoxCPM2):** yalnız `cc-by` yapılandırması (`cc-by-sa` paylaşım-aynı
  yükümlülüğü yüzünden kapalı). Satırların ~%45'inde oda/mikrofon/gürültü bozulmalı
  `audio_channel` var; varsayılan olarak bu sürüm kullanılır (artırma etkisi). Shard'lar akış
  halinde indirilir (her biri ~0,8 GB/~3,9 sa), 16 kHz'e indirilir, işlenince **silinir**.
  Ses kimliklerinin ~%2'si dev için ayrılır, eğitimde hiç görülmez.
- **Atıf (CC-BY):** model kartına ekleyin — *"Alania Turkish Synthetic Speech / Alania Turkish
  Domain Text by PatientDesk AI, CC BY 4.0"*, *"FLEURS (Conneau et al., 2022), CC BY 4.0"*,
  *"ISSAI Turkish Speech Corpus (Mussakhojayeva et al., 2023), MIT"*, *"Common Voice (Mozilla), CC0"*.
  YODAS2 kullanılırsa *"YODAS2 (Li et al., 2023) / Sidon (Nakata et al., 2025), CC BY 3.0"*.
- **Audiobooks (kapalı varsayılan):** kapılı; şartlar: atıf + bu veriyle eğitilen her modelin
  ağırlıklarını ticari kullanıma izin veren lisansla yayımlamak (Apache-2.0 uygundur). Kaynak
  kayıtların telif durumu **temizlenmemiş**; hukuki sorumluluk kullanıcıda. Yalnız `train` /
  `validation` havuzları, asla `review`. Kullanmak için sahibinin hesabıyla şartları kabul edip
  `export HF_TOKEN=...` ve `./stt/prepare.sh --audiobooks 600`.

### Faz 2 verisi: YODAS2-Sidon `tr000` (CC-BY-3.0)

YouTube videoları + **yükleyicinin elle yazdığı altyazılar**, Sidon ile konuşma onarımından
geçirilmiş (24 kHz). 43 shard × ~2,9 GB = **126 GB indirme**, ~1.500 sa **video** sesi. Her örnek
bir videonun tamamı + altyazı zaman damgaları; `prepare/yodas2_sidon.py` bunları cümlelere böler.

**Ölçülen verim** (`tr000/train-00007`, 150 video, 19,9 sa video): altyazılar videonun yalnız
**%38'ini** kapsıyor (7,6 sa), süre/karakter-hızı filtrelerinden **%34'ü** geçiyor (6,7 sa).
Yani tr000'den beklenen kullanılabilir konuşma ~1.500 sa değil **~500 sa**; öğretmen-CER
filtresiyle muhtemelen ~350–450 sa.

- **Altyazı gürültüsü:** başlıklar, telefon numaraları, ekrandaki yazılar, kayık zamanlamalar
  sık görülür (ör. 2 sn'lik bölüme yazılmış video başlığı). Filtreler: 1–20 sn, 4–25 karakter/sn,
  trnorm normalizasyonu. Daha sağlamı **öğretmen-CER filtresi**: izinli lisanslı bir modelle
  (ör. Whisper large-v3/turbo, MIT) çözüp `utt_id<TAB>cer` TSV'si üretin, sonra
  `prepare/filter_teacher_cer.py --max-cer 0.15` (ya da hazırlık sırasında `--teacher-cer`).
  mihuai/duxx çıktıları **kullanılmaz**.
- `tr100` (otomatik/ASR altyazıları) varsayılan olarak **reddedilir** (`--allow-auto-captions`).
- Disk/süre (shard başına): ~2,9 GB indirme (100 Mbit/sn'de ~4 dk), ~32–36 sa video → filtre
  sonrası ~11–12 sa konuşma; 16 kHz FLAC ~0,6–0,8 GB, öznitelik ~0,4 GB; 6 çekirdekte ~5–8 dk
  (videonun tamamı çözülüp yeniden örneklenir). Tamamı: 126 GB indirme (geçici, shard shard
  silinir), ~30 GB FLAC + ~16 GB öznitelik, ~4–6 sa (çoğu indirme).

## 2. Dizüstü kurulumu (8 GB NVIDIA GPU)

### 2a. Windows + WSL2 (önerilen yol Windows'ta)

1. **NVIDIA sürücüsü (Windows tarafında):** güncel Game Ready/Studio sürücüsü, **R570 veya
   üstü** (CUDA 12.8 ve RTX 50xx için şart). WSL içine *Linux* NVIDIA sürücüsü **kurmayın** —
   WSL2 GPU desteğini Windows sürücüsü sağlar.
2. Yönetici PowerShell: `wsl --install -d Ubuntu-24.04`, yeniden başlatın, Ubuntu'yu açıp
   kullanıcı oluşturun. Sonra `wsl --update`.
3. `%UserProfile%\.wslconfig` (RAM'in ~%75'i; özellik çıkarma ve veri yükleyiciler için):
   ```ini
   [wsl2]
   memory=12GB
   processors=8
   swap=16GB
   ```
   `wsl --shutdown` ile uygulayın. WSL diski varsayılan olarak C:'dedir ve büyür; **en az 100 GB
   boş yer** gerekir (gerekirse dağıtımı başka sürücüye taşıyın: `wsl --export` / `wsl --import`).
4. WSL içinde: `nvidia-smi` GPU'yu göstermeli. Ardından:
   ```bash
   sudo apt update && sudo apt install -y git curl tmux
   cd ~ && git clone <bu depo> ahenk-turkish-tts && cd ahenk-turkish-tts   # /mnt/c altına DEĞİL (çok yavaş)
   ```
5. Windows güç ayarları: prize takılıyken **uyku: asla**. Dizüstünü sert bir zemine/standa
   koyun; üretici aracında (Lenovo Vantage, Armoury Crate…) "dengeli/sessiz" mod sıcaklığı düşürür.

### 2b. Doğal Ubuntu 22.04/24.04

`sudo ubuntu-drivers install` (ya da `sudo apt install nvidia-driver-570`), yeniden başlatın,
`nvidia-smi` ile doğrulayın; `git curl tmux` kurun ve depoyu klonlayın.

### 2c. Ortam (her iki yol)

```bash
./stt/setup_env.sh              # CUDA 12.8: RTX 20xx/30xx/40xx/50xx (sm_75 … sm_120)
# ./stt/setup_env.sh --cuda 12.6  # GTX 10xx (Pascal) ya da R560 sürücü
source stt/path.sh              # her yeni terminalde
```

`uv` yoksa kurar (0.11.32), `.venv-stt`'yi Python 3.11 ile oluşturur, torch 2.8.0+cu128,
k2 `1.24.4.dev20260625+cuda12.8.torch2.8.0` (k2-fsa.github.io/k2/cuda.html), kalan paketleri
kilitli sürümlerle kurar, icefall'ı sabit commit'te `stt/icefall`'a klonlar (git-ignore'lu) ve
GPU'da küçük bir fp16 matmul testi yapar. ~3,5 GB indirme, 5–15 dk. Tekrar çalıştırmak güvenlidir.

### 2d. Bulut GPU: Modal (laptop yerine)

Aynı scriptler [Modal](https://modal.com) üzerinde, sabit sürümlü bir imajda çalışır
(`stt/modal_app.py`). Veri, öznitelikler ve checkpoint'ler kalıcı bir Modal Volume'de
(`turkish-stt-data`) durur; her adım durdurulup devam ettirilebilir.

```bash
pip install modal && modal setup        # ya da MODAL_TOKEN_ID / MODAL_TOKEN_SECRET ortam değişkenleri
modal run stt/modal_app.py --action prepare                       # CPU 8 çekirdek: ~3–5 sa, ~$3–5
modal run stt/modal_app.py --action probe                         # 10 dk GPU hız ölçümü (~$0,5)
modal run --detach stt/modal_app.py --action train --max-usd 20   # bütçe kilitli eğitim
modal run stt/modal_app.py --action export                        # ONNX fp32/int8 + WER/CER
modal volume get turkish-stt-data exp/<koşu>/<export-klasörü> ./exports/
```

- GPU seçimi: `STT_GPU=H100|A100-80GB|L40S|L4` (varsayılan A100-80GB, 8 GB için ayarlı
  `--max-duration` yerine büyük kartlarda 1000 sn).
- **Bütçe kilidi:** `--max-usd`, GPU + CPU/RAM saat ücretinden bir süre sınırına çevrilir
  (`train.sh --hours`). Modal saniye başına ücretlendirdiği için koşu bu tutarı aşamaz. Modal
  bir GPU'yu geri alırsa (preemption) iş kendiliğinden yeniden başlar, checkpoint'ten devam eder
  ve Volume'deki defter (`modal_ledger/`) önceki denemelerin süresini düşer.
- Fiyatlar (Modal liste fiyatı, 2026): H100 $3,95/sa, A100 80GB $2,50, L40S $1,95, L4 $0,80;
  CPU $0,047/çekirdek-sa, RAM $0,008/GiB-sa; Volume ayda 1 TiB'a kadar ücretsiz. Starter planda
  ayda **$30** ücretsiz kredi var.
- **Tahmin (small, ~600 sa × 20 epoch):** A100 80GB'de ~5–7 sa ≈ $15–20, veri hazırlama ~$3–5 →
  toplam ~$20–25: aylık $30 krediye sığar. Önce `probe` ile gerçek hızı ölçün; epoch sayısını
  bütçeye göre ayarlayın. Medium bir sonraki aya kalır.
- İsteğe bağlı: daha yüksek Hugging Face indirme limiti için
  `modal secret create huggingface HF_TOKEN=...` ve `STT_HF_SECRET=huggingface`.

## 3. Veri hazırlama

```bash
tmux new -s stt                  # bağlantı/terminal kapansa da süreç yaşasın
./stt/prepare.sh --cv-train-from validated --num-workers 6 --fbank-device cuda
```

| Aşama | Ne yapar | Süre (100 Mbit/sn, 8 çekirdek) | Disk |
|---|---|---|---|
| 1 FLEURS | 2,1 GB indir, FLAC'a çevir | 5–10 dk | ~0,7 GB |
| 2 ISSAI TSC | 21,4 GB indir (devam ettirilebilir), akış halinde FLAC'a çevir, arşivi sil | ~30 dk indirme + 20–40 dk | tepe ~37 GB, sonra ~13 GB |
| 3 Common Voice 17 | 0,7–2,6 GB Parquet → FLAC (MP3 çözme) | 10–25 dk | ~6–8 GB |
| 6 Alania | otomatik hedef = gerçek saat × 0,45/0,55 (≤ 400 sa); ~70 shard × 0,8 GB | 1–2 sa (çoğu indirme) | shard başına ~0,3 GB FLAC |
| 7 fbank | 80 boyutlu log-mel, lilcom | GPU 15–30 dk / CPU 1–2 sa | ~32 GB / 1000 sa |
| 8 karışım | süre filtresi 1–20 sn, sentetik sınırı, global karıştırma | 2–5 dk | — |
| 9 token modeli | unigram, 1000 token, + 300 bin alan cümlesi | 3–5 dk | — |

Toplam ~3–5 saat (çoğu indirme). Her aşama tekrar başlatılabilir; biten çıktılar atlanır
(`--stage N --stop-stage M`). Alania sesleri öznitelik çıkarıldıktan sonra silinir
(`--keep-synthetic-audio` ile tutulur). Tüm büyük dosyalar `data/stt/` altında (git-ignore'lu;
`STT_DATA=/başka/disk` ile taşınabilir). ~600 sa'lik tarif için **≥ 80 GB boş alan** önerilir.

**Token modeli: neden 1000?** 5 boyut denendi (unigram; FLEURS + Alania alan metni, ayrılmış
cümlelerde ölçüldü):

| vocab | token/kelime | token/sn (medyan / p99) |
|---|---|---|
| 500 | 2,49 | 5,3 / 8,8 |
| **1000** | **1,98** | **4,5 / 7,6** |
| 1500 | 1,74 | 4,0 / 7,0 |
| 2000 | 1,61 | 3,8 / 6,6 |
| 3000 | 1,46 | 3,4 / 5,9 |

Hepsi kodlayıcının 25 Hz çıkış hızının çok altında (pruned RNN-T'nin T ≥ U şartı sorun değil).
1000'de kelime başına ~2 token, Türkçe'nin kök + ek-öbeği yapısına denk düşüyor: ekler kendi
token'ları olarak paylaşılıyor, en küçük gerçek-konuşma alt kümesinde bile her token binlerce kez
görülüyor. 2000'e çıkmak token/kelimeyi yalnız %19 azaltıp karar verici/birleştirici çıkış
katmanlarını iki katına çıkarıyor (mobilde bedava değil). 2000 denemek isteyen:
`prepare.sh --stage 9 --vocab 2000` ve `train.sh --vocab 2000`.

## 4. Eğitim

```bash
./stt/train.sh --preset small --hours 6     # 6 saat sonra temiz durur (checkpoint + çıkış kodu 75)
./stt/train.sh --preset small --hours 6     # ertesi gün: kaldığı yerden (epoch ortasından) devam
./stt/train.sh --stop                       # çalışan eğitime "kaydet ve dur" de (~10 batch içinde)
tensorboard --logdir data/stt/exp           # kayıp eğrileri
python stt/tools/eta.py --exp-dir data/stt/exp/small-causal-bpe1000   # ölçülen hız ve epoch süresi
```

- **Devam etme:** komutu aynen tekrar çalıştırın. `train_tr.py`, `--exp-dir` içindeki en yeni
  `epoch-E.pt` / `checkpoint-N.pt`'yi bulur (`--start-epoch` / `--start-batch` gerekmez); çökme
  ya da elektrik kesintisinde de son checkpoint'ten devam eder (en fazla `--save-every-n 1000`
  batch ≈ 20 dk kayıp). Son oturumun nasıl bittiği `RESUME.json`'da.
- **Önemli düzeltme:** icefall (3f848bb) + lhotse 1.33'te epoch ortasından devam edince,
  başlangıçtaki OOM taraması örnekleyiciyi sona kadar tüketip epoch'u **baştan** başlatıyor
  (duman testinde doğrulandı). Epoch'tan kısa oturumlarda epoch hiç bitmezdi. `train_tr.py` bu
  durumda taramayı atlıyor; doğru noktadan devam ettiği duman testinde batch sırasıyla doğrulandı.
- **Ctrl-C:** ilk basış = nazik durma (checkpoint), ikinci = hemen öldür.
- **Akış ayarları:** `--causal 1 --chunk-size 16,32,64,-1 --left-context-frames 64,128,256,-1`
  (eğitimde rastgele; dışa aktarmada biri seçilir). fp16 (AMP) açık; RTX 30xx+ için `-- --use-fp16 0 --use-bf16 1`
  de çalışır.
- **Sıcaklık koruması:** `train.sh`, `thermal_guard.py`'yi otomatik başlatır: 10 sn'de bir
  `nvidia-smi` okur, **83 °C'de** eğitimin tüm süreç grubunu `SIGSTOP` ile dondurur, **72 °C'de**
  `SIGCONT` ile sürdürür (WSL2'de de çalışır; `/usr/lib/wsl/lib/nvidia-smi`). Seçenekler:
  `--guard-args "--pause-at 80 --resume-at 70 --duty-cycle 0.85"` (zamanın %15'i zorunlu mola —
  güç sınırı koyamayan dizüstüler için kaba bir güç tavanı), `--power-limit 80` (`nvidia-smi -pl`;
  çoğu dizüstü GeForce'ta ve WSL2'de desteklenmez, hata kaydedilip yok sayılır). Kayıt:
  `<exp>/log/thermal_guard.log`.
- **Disk:** checkpoint başına small ~0,4 GB, medium ~1,1 GB; `--keep-last-k 3` + en yeni 12
  epoch (`--keep-epochs`) + best-*.pt → small ~7 GB, medium ~18 GB.

### VRAM: `--max-duration` (batch başına saniye ses)

icefall'ın yayımlanmış eğitim kayıtlarından (V100, fp16; başlangıç OOM taramasının ölçtüğü tepe
bellek): medium 1000 sn'de 16,0 GB, small 1500 sn'de 17,8 GB → tepe ≈ 1,3 GB + 15,7 MB/sn (medium),
0,4 GB + 12,6 MB/sn (small), 1000 token'lık birleştirici dahil. 8 GB kartta ~6,5–7 GB kullanılabilir:

| Ön ayar | Parametre | `--max-duration` (8 GB) | Tahmini tepe | 12 GB | 16 GB |
|---|---|---|---|---|---|
| small | 24,5 M | **400** | ~5,5 GB | 650 | 900 |
| medium | 67,1 M | **250** | ~5,6 GB | 450 | 650 |

Eğitim başında icefall en kötü batch'leri dener; OOM verirse 50 düşürün. Masaüstü ekranını
NVIDIA GPU değil tümleşik GPU sürüyorsa (Optimus) daha fazla VRAM kalır.

## 5. Dışa aktarma ve test

```bash
./stt/export_and_test.sh --preset small                       # en yeni epoch, avg=min(6, E-1)
./stt/export_and_test.sh --preset small --epoch 20 --avg 8 --chunk 32 --left 256   # 640 ms
```

1. icefall'ın ortalaması (`--use-averaged-model`, epoch E−avg … E) → `pretrained.pt`
2. `export-onnx-streaming.py` → `encoder/decoder/joiner.onnx` + **int8** (`*.int8.onnx`)
3. Test kümeleri (`fleurs`, `cv`, `issai`; `--max-utts` ile kısaltılabilir) sherpa-onnx'te
   320 ms'lik parçalarla, mikrofon gibi beslenerek çözülür; **WER/CER**, iki taraf da
   `trnorm.normalize_for_eval` ile normalize edilerek hesaplanır, RTF ölçülür. Birkaç hipotez
   ayrıca `trnorm.inverse_normalize` (ITN: "on beşte" → "15'te") ile **yalnız gösterim** için basılır.

Çıktı: `data/stt/exp/<ad>/export-epoch-E-avg-N-chunk-16-left-128/` (ONNX'ler, `tokens.txt`,
`bpe.model`, `MODEL_CARD.md`, `results/*.json|tsv`). Bu klasör doğrudan sherpa-onnx'in
"online transducer" yapılandırmasına verilir (Python: `sherpa_onnx.OnlineRecognizer.from_transducer`;
Android/iOS/WASM/RPi örnekleri için https://k2-fsa.github.io/sherpa/onnx/). Kendi kayıtlarınızla:
`python stt/tools/decode_sherpa.py --model-dir <export> --int8 --wav-list liste.tsv` (`yol<TAB>metin`).

## 6. Süre ve maliyet tahmini (dürüst hesap)

**Ölçüm noktası:** icefall'ın Hugging Face'teki eğitim kayıtları. Akışlı medium (66 M), 4×V100
32 GB, fp16, batch 1000 sn: epoch başına 2.940 sa ses ~70 dakikada → **GPU başına ~630× gerçek
zaman**. Small (23 M), 2×V100, batch 1500 sn: 2.880 sa / 92 dk → **~940×**.

**Dizüstü varsayımı:** RTX 4060 Laptop (256 GB/sn bellek bant genişliği) bu bellek-yoğun iş
yükünde bir V100'ün (900 GB/sn) ~%30–40'ı; RTX 3070 Laptop (448 GB/sn) ~%40–50. 8 GB'a sığan
küçük batch'ler (250–400 sn) bir ~%10–20 daha kaybettirir. Sonuç: **small ≈ 250–350×**,
**medium ≈ 150–220×** gerçek zaman. (İlk 15 dakikadan sonra `tools/eta.py` gerçek değeri verir.)

| Veri | Epoch | small: sa/epoch | small: toplam | medium: sa/epoch | medium: toplam |
|---|---|---|---|---|---|
| 600 sa | 20 | 1,7–2,4 | **34–48 sa** | 2,7–4,0 | **55–80 sa** |
| 1000 sa | 20 | 2,9–4,0 | **57–80 sa** | 4,5–6,7 | **90–135 sa** |
| 1500 sa | 15 | 4,3–6,0 | **64–90 sa** | 6,8–10 | **100–150 sa** |

Sıcaklık molaları için +%10–25 ekleyin. LibriSpeech tarifleri ~86.000 sa ses işler
(30 epoch × 2.880 sa); biz 12.000–22.500 sa işleyebiliyoruz — bu yüzden epoch sayısı
mütevazı ve medium modelin dizüstünde tam potansiyeline ulaşması beklenmemeli.

**Kiralık GPU** (aynı ölçüm noktasından; fiyatlar 2025–26 piyasa aralıkları, kiralamadan önce
kontrol edin; medium, 1000 sa × 20 epoch = 20.000 sa ses):

| GPU | V100'e göre | medium hızı | Süre | Saatlik | Maliyet |
|---|---|---|---|---|---|
| RTX 4090 24 GB | ~1,6–2× | ~1.000–1.250× | ~16–20 sa | $0,3–0,7 | **~$6–14** |
| L40S 48 GB | ~1,5–1,8× | ~950–1.100× | ~18–21 sa | $0,8–1,2 | ~$15–25 |
| A100 80 GB | ~2–2,5× | ~1.250–1.600× | ~13–16 sa | $1,2–2,0 | ~$16–32 |
| H100 80 GB | ~3–4× | ~1.900–2.500× | ~8–11 sa | $2–3 | ~$16–33 |

Small modelde süre ve maliyet ~%35 daha az. Kiralık makinede veri hazırlığına ayrıca 3–5 sa
(CPU + indirme) ve ~100 GB disk ekleyin. Kısacası: dizüstü small için ~2–4 gün, medium için
~3–6 gün sürer; aynı iş kiralık tek bir 4090'da bir gecede ve ~$10–20'ye biter. Tarif ikisinde de
aynı komutlarla çalışır (büyük kartta yalnız `--max-duration`'ı büyütün).

## 7. Sorun giderme

- **`CUDA out of memory`** → `--max-duration`'ı 50 azaltın. **`no kernel image`** → Pascal kart:
  `setup_env.sh --cuda 12.6`. **`nvidia-smi` WSL'de yok** → Windows sürücüsünü güncelleyin, `wsl --update`.
- **Çok yavaş veri hazırlığı/eğitim** → depo `/mnt/c` altında olmasın; `.wslconfig` bellek ayarı.
- **Sürekli `PAUSE` (thermal_guard.log)** → havalandırma, üretici "sessiz" modu,
  `--guard-args "--duty-cycle 0.8"`; gerekirse `--max-duration`'ı düşürün (daha az güç).
- **`grad scale is too small`** (fp16 kararsızlığı) → son checkpoint'ten devam edin; tekrar
  ederse `-- --use-fp16 0 --use-bf16 1` (RTX 30xx ve sonrası).

## 8. Dosyalar

```
setup_env.sh  path.sh  presets.sh  prepare.sh  train.sh  thermal_guard.py  export_and_test.sh
requirements-stt.txt  constraints-stt.txt  locks/linux-py311-{cpu,cu128}.txt
prepare/  common.py hf_parquet.py fleurs.py issai_tsc.py commonvoice.py commonvoice_hf.py
          alania.py audiobooks.py yodas2_sidon.py filter_teacher_cer.py
          compute_fbank.py make_mix.py make_bpe.py
recipe/   train_tr.py (icefall train.py sarmalayıcısı)  tr_datamodule.py
tools/    decode_sherpa.py  icefall_run.py  eta.py
SMOKE_TEST.md
```

---

## English summary

Recipe for an **Apache-2.0, streaming (causal) Zipformer transducer** for Turkish trained only
on permissively licensed data (FLEURS CC-BY-4.0, ISSAI TSC MIT, Common Voice 17 CC0 via the
`fixie-ai/common_voice_17_0` Parquet mirror or a local Mozilla Data Collective release, and the
CC-BY-4.0 Alania synthetic speech capped at ≤45 % of hours using its degraded channel). Optional:
the gated audiobooks corpus (open-weights-with-attribution, rights not cleared) and, as phase-2
data, YODAS2-Sidon `tr000` (CC-BY-3.0, noisy uploader captions; char-rate filter + teacher-CER hook;
`tr100` auto-captions refused; measured on 150 videos: captions cover only ~38 % of the video time,
~34 % survives the filters, so expect ~500 h usable from ~1,500 h of video). Outputs of
non-commercial models (mihuai, duxx) are never used.

Laptop flow (8 GB NVIDIA GPU, native Ubuntu or Windows+WSL2 with a ≥R570 Windows driver):
`./stt/setup_env.sh` → `./stt/prepare.sh --cv-train-from validated --fbank-device cuda`
(~3–5 h, ≥80 GB free) → `./stt/train.sh --preset small --hours 6` repeated until done (auto-resume,
graceful stop, thermal guard pauses the process group at 83 °C and resumes at 72 °C) →
`./stt/export_and_test.sh --preset small` (averaged model → streaming ONNX fp32/int8 → sherpa-onnx
decoding of the FLEURS/CV/ISSAI test sets, WER/CER on `trnorm.normalize_for_eval`).
Presets: small 24.5 M params (max-duration 400 on 8 GB), medium 67.1 M (250); BPE: unigram 1000
(≈2 tokens/word ≈ Turkish stem + suffix group). Estimated time on an RTX 4060-class laptop:
~35–90 h (small) / ~55–150 h (medium) for 600–1500 h of data; a rented RTX 4090 does medium in
~16–20 h for ~$6–14. A fix for icefall+lhotse mid-epoch resume (the OOM scan rewound the sampler)
is included in `recipe/train_tr.py`. The full CPU smoke test is documented in `SMOKE_TEST.md`.
