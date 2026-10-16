# VoxCPM2 + Antalia: kendi Türkçe sesimiz (LoRA ince ayar tarifi)

> **Güncel durum (2026-10-14):** yayınlanan model **Ahenk**'tir
> ([ozgurguler/ahenk-tts-tr](https://huggingface.co/ozgurguler/ahenk-tts-tr), `voxcpm2-f022-s3tr/step_0000500`):
> tasarlanmış "f022" sesi, Alania aşama 1 + f022 derlemi + iki dilli talimat ince ayarı. Antalia sesi bu sürümde
> **yok**. Bu belge, ilk Antalia denemesinden başlayan kronolojik bir tarif günlüğüdür; aşağıdaki "Antalia" bölümleri
> eski denemeleri anlatır. Modal komutlarındaki profil adları (`$env:MODAL_PROFILE`) kendi hesabınızınkiyle değiştirilmelidir.

**Ne:** Açık kaynak **VoxCPM2** (OpenBMB, 2B parametre, 30 dil, 48 kHz, **Apache-2.0**) modelini,
izinli bir profesyonel seslendirme sanatçısının 5 saatlik stüdyo kaydıyla (**Antalia**, CC-BY-4.0)
LoRA yöntemiyle ince ayarlıyoruz. Metin ön ucu **trnorm**. Ortaya kendi adımızla yayınlanan, kendi sesi
olan bir Türkçe TTS modeli çıkıyor. Model kartında temel model ve veri açıkça belirtiliyor.

Aynı yolu [Trendyol-TTS](https://huggingface.co/Trendyol/Trendyol-TTS) izledi: VoxCPM2 + ~20 saat Türkçe ses
+ LoRA (2.000 adım). Açık Türkçe modeller içinde doğallıkta en iyisi o (UTMOS 3.83).

| | |
|---|---|
| Temel model | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), Apache-2.0, ticari kullanım serbest |
| Eğitim kodu | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) resmi ince ayar betiği, sabit commit `f0c787f` |
| Veri | [Antalia voice corpus](https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus), CC-BY-4.0, tek kadın seslendirmen, 1.026 eğitim + 20 ayrılmış klip (4.8 sa) |
| Metin | `trnorm.normalize` (eğitimde de çıkarımda da aynı okunuş formu); VoxCPM'in kendi normalizer'ı kapalı |
| Yöntem | LoRA r=64 (LM + DiT; Trendyol-TTS ile aynı; aşağıdaki sonuçlar r=32 ile eğitildi), lr 1e-4, batch 16, 1.000 adım (~16 epoch), her 100 adımda checkpoint. A100 80 GB: 1,94 s/adım, 1.000 adım ~$1,6 |
| Ses örneği | 16 kHz girdi (AudioVAE kodlayıcısı), 48 kHz çıktı |

## Modal ile adım adım

Depo kökünden (PowerShell'de de aynı komutlar):

```powershell
git pull
python -m modal run tts/voxcpm/modal_app.py --action prepare          # CPU ~15 dk, ~$0.1: 16 kHz veri + model (~5 GB)
python -m modal run tts/voxcpm/modal_app.py --action probe            # A100 ~14 dk, ~$0.7: gerçek s/adım -> maliyet tablosu
python -m modal run --detach tts/voxcpm/modal_app.py --action train --max-usd 8 --steps 1000
python -m modal run tts/voxcpm/modal_app.py --action status
python -m modal run tts/voxcpm/modal_app.py --action eval --ckpt base,step_0000300,step_0000600,latest --max-usd 3.5
python -m modal run tts/voxcpm/modal_app.py --action fetch --fetch step_0000600   # GPU açmaz: wav + skor indir
```

- `train`, `--max-usd`'yi kesin bir süre sınırına çevirir. Süre dolunca eğitim checkpoint kaydedip durur.
  Aynı komut tekrar çalıştırılınca kaldığı yerden devam eder.
- `eval`, her checkpoint için TR-Zor'un 155 cümlesini ve eğitimde hiç kullanılmamış 20 Antalia
  cümlesini sentezler. Ardından Whisper large-v3-turbo ile geri okuyup WER, UTMOS ile doğallık puanı yazar.
  `base`, eğitilmemiş VoxCPM2'nin bir Antalia klibini taklit etmesi (klonlama); `trendyol`, Trendyol-TTS
  (16 adım); karşılaştırma için.
- `eval`'in `--max-usd`'si (varsayılan 2) tüm çağrının kesin üst sınırı. Komut önce her koşunun tahmini
  süresini ve "at most $X" yazar. Varsayılan $2 hepsine yetmiyorsa hiç başlamadan durur ve gereken
  `--max-usd`'yi söyler. Sığmayan koşu başlatılmaz, para harcamaz; süre dolduğunda hâlâ süren koşu öldürülür,
  eski `eval/<ckpt>` olduğu gibi kalır. Tam ölçüm A100'de ~17 dk (~$0.85), Trendyol ~23 dk.
- Biten koşuların wav'ları ve skorları `voxcpm-eval/<ad>/`'a iner (dinlemek için), skor dosyaları gizli HF
  reposuna (`ozgurguler/voxcpm2-antalia-eval`) gider. Sonradan indirmek için `--action fetch` yeter.
  Yerelde UTMOS eklenmiş bir klasörün üstüne, UTMOS'u olmayan Volume kopyası yazılmaz.

### Çıkarım ayarı taraması (`--grid`)

```powershell
python -m modal run tts/voxcpm/modal_app.py --action eval --ckpt step_0000600 --limit 30 --grid "10:1.5,10:2.5,16:1.5,16:2.0,16:2.5" --max-usd 3.9
```

- `--grid "adım:cfg,..."`: her nokta ayrı bir koşu, `eval/<ckpt>@ts<adım>-cfg<cfg>` klasörüne yazılır; tam
  ölçüm (`eval/<ckpt>`) bozulmaz. `--limit` ve `--args` yalnızca `--grid` ile kabul edilir.
- `--limit 30`: TR-Zor'dan 13 kategorinin hepsine dağılmış 30 cümle (dosya kategori sırasında, eskiden ilk 30
  satır yalnızca sayı/sıra/tarih idi) + ayrılmış 20 cümlenin hepsi. `--args "--sample head"` eski davranış.
- `10:2.0` (varsayılan) zaten tam ölçüldü: aynı 30 + 20 satırın skorları `voxcpm-eval/step_0000600/rows.jsonl`'da
  (satır başına `errors`, `ref_words`, `utmos`). Bu yüzden ızgarada yok; 5 koşu, A100'de ~67 dk, en fazla $3.9.
- 30 cümle gürültülü: 1-2 cümlelik fark anlamlı değil. En iyi 1-2 ayarı tam ölç (`--grid 16:2.0`, limitsiz,
  ~$1.3).

### LoRA'yı indirme, başka Modal hesabına taşıma

```powershell
python -m modal run tts/voxcpm/modal_app.py --action fetch-ckpt --fetch step_0000600   # -> voxcpm-lora/step_0000600/
python -m modal profile activate <hedef-profil>                                        # ör. ikinci hesabınız
python -m modal run tts/voxcpm/modal_app.py --action prepare                           # o hesapta bir kez, CPU ~$0.1
python -m modal run tts/voxcpm/modal_app.py --action push-ckpt --fetch step_0000600    # -> o hesabın Volume'u
```

- `fetch-ckpt` GPU açmaz; yalnızca `lora_weights.safetensors`, `lora_config.json`, `training_state.json` iner
  (optimizer yok). `push-ckpt` aktif profilin Volume'una yükler, adını yazar, var olan dosyanın üstüne yazmaz.
- Yeni hesapta ilk `eval`, Whisper (~1.6 GB) ve UTMOS (~400 MB) ağırlıklarını indirir (Volume'da kalır).

### Doğallık (UTMOS), yerelde CPU'da

```powershell
python tts/voxcpm/score_utmos.py voxcpm-eval/step_0000600 voxcpm-eval/trendyol --device cpu
```

Var olan wav'ları yeniden puanlar, para harcamaz: `rows.jsonl`'a satır başına `utmos`, `summary.json`'a set ve
kategori ortalaması. UTMOS22 strong (`tarepan/SpeechMOS:v1.2.0`, torch.hub; torch + torchaudio + soundfile
yeter). İlk çalıştırma ~400 MB ağırlık indirir; sonra 175 klip ~30 s. `eval` bunu koşunun sonunda kendisi yapar
(`--args "--no-utmos"` kapatır).

### Hugging Face'e yayın

```powershell
python tts/voxcpm/publish_hf.py --name "<Ad>" --repo ozgurguler/<ad> --lora voxcpm-lora/step_0000600 --evals voxcpm-eval/step_0000600 voxcpm-eval/trendyol
python tts/voxcpm/publish_hf.py --repo ozgurguler/<ad> --upload             # gizli repo
python tts/voxcpm/publish_hf.py --repo ozgurguler/<ad> --upload --public    # yalnızca kullanıcı onayıyla
```

- İlk komut hiçbir şey yüklemez: `hf-publish/<ad>/` içine model kartını (TR + EN, `MODEL_CARD.template.md`;
  sayılar `summary.json`'lardan, r/alpha LoRA dosyasından), iki LoRA dosyasını ve 6 örnek wav'ı koyar.
  Kartı oku, gerekirse düzenle.
- `--upload` klasörü olduğu gibi yükler. Kart ağırlıklarla ya da örneklerle uyuşmuyorsa, `{{` kaldıysa
  reddeder. `--public` olmadan repo gizli açılır.
- **s3 sürümü (`--template tts/voxcpm/MODEL_CARD.f022s3.template.md`, 2026-10-13):** yayınlanacak model
  `voxcpm2-f022-s3tr/step_0000500` (iki dilde talimat). Paket LoRA'nın yanına `instruction_pick.py`, `pause_clean.py`
  + içe aktardığı `onset.py` ve `instructions_tr.json` koyar (tts/voxcpm'den olduğu gibi; kart her birini sha256'sıyla
  listeler). Kartın kullanım kodu: trnorm ön ucu, `instruction_pick.pick` (talimatın dili), `compose`, `generate`'ten
  sonra `pause_clean.clean`; Türkçe eğitim ifadeleri tablosu duygu başına `build_textbank.INSTRUCTIONS` ve
  `instructions_tr.json`'dan üretilir. Kartta iki talimat dinlemesinin karar tablosu var; düz okuma ölçümünün cümlesi
  ve örnekler için `TODO(final)` işaretleri kaldı: işaret kaldıkça `--upload` reddeder (kartı elle doldurun). Kart,
  ilk ölçüm klasörünün `summary.json`'ında `pause_clean` varsa (synth_eval `--pause-clean`) bunu yazar.
  `check_staging` paket dosyalarını da denetler (listede olup olmayan, değişmiş, içe aktardığı modül eksik `.py`,
  bozuk tablo, `instruction_pick.EMOTIONS`'ın Türkçesi tablodakinden farklı); başka bir şablonla yeniden hazırlama
  paket dosyalarını siler, `--upload` uzaktakileri de aynı commit'te siler.
- **`instruction_pick.py`** (yalnız standart kütüphane): `pick(duygu, metin, prefer="auto")` → (talimat, metin,
  neden). İlk iki kelimeden biri "c" ile başlıyorsa ("ç" değil; kelime içindeki c sayılmaz: "Acele" İngilizce alır;
  NFC + Türkçe küçük harf) Türkçe talimat, yoksa İngilizce; duygu başına tek eğitim ifadesi (`EMOTIONS`), Türkçe adlar
  da (`ALIASES`, büyük harfle de); kendi talimat cümlesi olduğu gibi kullanılır; talimatlı metinde "Vay" -> "Vaay",
  okunacak kelimesi olmayan talimatlı metin hata verir. İki dinleme turu: İngilizce saf duyguda 6/6, Türkçe "c" ile
  başlayan cümlelerde 10/10, "Acele"de İngilizce 2/2: kural 20 kararın 18'iyle uyuşur (uymayan: empatik 2); "c"
  önemliyse `prefer="tr"`. `pick`'e trnorm çıktısı verilir. `python tts/voxcpm/instruction_pick.py surprised "Cam
  kırılmış!"`.

```powershell
# s3tr-500'ün ölçümünü indir (GPU açmaz): ölçümün durduğu hesapta; depo kökünden (yoksa --out <depo>/voxcpm-eval),
# --hf-repo= skorları HF'ye yüklemez (boş değer; PowerShell 5.1 "" argümanını düşürür, = ile yazın)
$env:MODAL_PROFILE="<profiliniz>"
python -m modal run tts/voxcpm/modal_app.py --action fetch --exp-name voxcpm2-f022-s3tr --fetch step_0000500 --grid 16:2.0 --hf-repo= --out voxcpm-eval
# kuru hazırlama (hiçbir şey yüklenmez)
python tts/voxcpm/publish_hf.py --name "Ahenk" --repo ozgurguler/ahenk-tts-tr --exp-name voxcpm2-f022-s3tr --lora voxcpm-lora/voxcpm2-f022-s3tr/step_0000500 --evals "voxcpm-eval/voxcpm2-f022-s3tr/step_0000500@ts16-cfg2" "voxcpm-eval/voxcpm2-f022-s3/step_0001500@ts16-cfg2=Aşama 2, 1500. adım / stage 2, step 1500" "voxcpm-eval/voxcpm2-f022-s2/step_0001500@ts16-cfg2=Önceki sürüm / previous version" --samples 6 --template tts/voxcpm/MODEL_CARD.f022s3.template.md --out hf-publish/ahenk-tts-s3tr
```

### İki aşamalı eğitim (Alania + Antalia)

Aşama 1 Türkçe telaffuzu geniş bir veriyle güçlendirir, aşama 2 aynı LoRA'yı Antalia ile sürdürür, ses Antalia kalır.

- **Veri:** [Alania Turkish Synthetic Speech](https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr)
  (`cc-by`, CC BY 4.0, PatientDesk AI; VoxCPM2'nin kalite filtreli kendi çıktısı), sabit commit `277dfc8a23`.
  `prepare_alania.py` yalnızca düz okumaları (`reference`), `qc_cer == 0`, `qc_utmos >= 3.7`, 1.5-20 sn klipleri,
  ses başına en fazla 40 klibi, FineWeb metnini en fazla %25 alır; metin `text_spoken`. Kullandığımız ayar:
  `--hours 25 --gender female`.
- **Sızıntı koruması:** TR-Zor cümleleri ve Antalia'nın ayrılmış 20 cümlesi (tamamı ya da ≥4 kelimelik parçası)
  eğitim verisine giremez.
- **Atıf:** "Alania Turkish Synthetic Speech by PatientDesk AI (https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr), licensed under CC BY 4.0" (`/data/alania16k/ATTRIBUTION.txt`).

Komutlar (`$env:MODAL_PROFILE="<profiliniz>"`, depo kökünde; her biri kendi `--max-usd` sınırıyla durur):

```powershell
python -m modal run tts/voxcpm/modal_app.py --action prepare                                # CPU, ~$0.1 (bir kez)
python -m modal run tts/voxcpm/modal_app.py --action prepare-alania --max-usd 1 --args "--hours 25 --gender female"
python -m modal run tts/voxcpm/modal_app.py --action probe --args "--data /data/alania16k"  # A100 ~14 dk, ~$0.7
python -m modal run --detach tts/voxcpm/modal_app.py --action train --exp-name voxcpm2-alania-s1 --steps N --max-usd X --args "--data /data/alania16k"
python -m modal run --detach tts/voxcpm/modal_app.py --action train --exp-name voxcpm2-alania-s2 --steps M --max-usd Y --args "--init-lora /data/exp/voxcpm2-alania-s1/ckpt/step_XXXXXXX"
python -m modal run tts/voxcpm/modal_app.py --action eval --exp-name voxcpm2-alania-s2 --ckpt step_... --grid 16:2.0 --max-usd 2
```

N, M ve `--max-usd` probe'un ölçtüğü s/adım'dan hesaplanır. `prepare-alania` sınırda durursa aynı komut kaldığı
yerden devam eder; eğitim yarım kalmış veriyle başlamayı reddeder.

**Aşama 1 için gerçek insan sesi (alternatif):** `prepare_human.py` aynı düzende (`/data/human16k`) Common Voice
17.0 Türkçe (CC0, `fixie-ai/common_voice_17_0`) kliplerinden bir derlem kurar; isteğe bağlı ISSAI TSC (MIT,
`--sources cv,issai`). Metin `trnorm.normalize` ile okunuş formuna çevrilir, sessizlik kırpılır, ses -20 dBFS'e
getirilir, her aday UTMOS ile GPU'da puanlanır (`--min-utmos`, varsayılan 3.3), konuşmacı başına en fazla 60 klip.
Sızıntı koruması Alania ile aynı. Rapor (`PREP_REPORT.md`) aday ve seçilen UTMOS dağılımını ve her eşikte kaç saat
kalacağını verir; puanlar `scores.jsonl`'da saklanır, `--restart` ile başka eşik yeniden puanlamaz. İlk yerel ölçüm
(1.200 CV satırı): aday medyanı 2.52, 3.3 eşiğinde CV'nin tamamından ancak ~2 saat çıkıyor; 25 saat için eşik ~2.5
ya da ISSAI gerekir.

```powershell
python -m modal run tts/voxcpm/modal_app.py --action prepare-human --max-usd 2 --args "--min-utmos 2.5"   # L4, ~$1.30/sa
python -m modal run --detach tts/voxcpm/modal_app.py --action train --exp-name voxcpm2-human-s1 --steps N --max-usd X --args "--data /data/human16k"
```

## Ses yarışması

Sıradaki modelin sesini seçmek için (`docs/SES_PLANI.md` adım 1). Bütün adaylar aynı VoxCPM2 ile aynı 70 cümleyi
okur, puanlanır, sonra kör dinlenir. Cümleler `benchmarks/tts/sets/contest.jsonl`'da: 40 TR-Zor (13 kategorinin
hepsi), 15 soru (eşli düz cümlelerle), 15 duygulu satır. Duygulu satırlarda 6 duygu var, model
`(İngilizce talimat)` + trnorm(cümle) alır. Kod: `voice_contest.py`, Modal tarafı `modal_app.py --action contest-*`.

- **Tasarlanmış sesler:** `voice_descriptions.jsonl`'daki 150 tarifin her biri 2 tohumla, referanssız ses
  tasarımıyla bir tarama cümlesi okur. Bu klipler Whisper CER, Türkçe dil tahmini ve UTMOS ile puanlanır.
  - En iyi 30 aday 4 cümle daha klonlar (yalnız `reference_wav_path`).
  - Sıralama 4 klonun **ortalama** UTMOS'una göre yapılır, çünkü tek klibe bakmak yanıltır.
  - Şartlar: 4 klondan en az 3'ünde CER 0, ECAPA benzerliği en az 0.4 (geçici eşik), p(tr) en az 0.9.
  - İlk 5 ses (her cinsiyetten en az 1) `screen/entries.json`'a yazılır.
- **Başka dilden izinli sesler:** Masha ve Dima (Dialogs-ru, OpenRAIL), KazakhTTS2 F1/M1 (CC BY 4.0), Thorsten
  (CC0). Masha ile Thorsten'in duygulu klibi ayrı giriştir.
  - Yalnız referanstan klonlanır; devam modu (`prompt_text`) aksanı taşıdığı için reddedilir.
  - Lisans, rıza ve atıf `CONTEST_SOURCES.md`'de ve her sesin `sources.json`'ında. Kaydı eksik ses yüklenmez ve
    yarışmaya girmez.
- **Kontroller:** `galaktik-s2` yayındaki model (s2 step 600, referanssız). `base-antalia`, taban VoxCPM2'nin bir
  Antalia klibini klonlaması. İsteğe bağlı ikinci kol, aynı girişlerin aşama-1 LoRA'sıyla okunuşu (`+s1`).
- **Puanlar** (klip ve giriş başına `rows.jsonl`, `summary.json`; hepsi birden `leaderboard.md`):
  - Whisper large-v3-turbo ile WER ve CER (synth_eval ile aynı yargıç).
  - Türkçe olasılığı p(tr), Whisper'ın kendi dil tahmininden (dil zorlanmaz).
  - UTMOS22.
  - Referansa ve girişin kendi kliplerine ses benzerliği: ECAPA, SpeechBrain `spkrec-ecapa-voxceleb` (Apache-2.0).
  - Soru satırlarında tonlama başarısı ve AUC (`score_intonation.py`).
  - Duygulu satırlarda F0 medyanı ve aralığı, hız ve enerjinin düz satırlara göre farkı.
- **Hız:** varsayılan arka uç Nano-vLLM-VoxCPM'dir (`--backend nanovllm`, `nanovllm_base.py`).
  - `nanovllm_app.py` probe'u A100'de ölçtü: 64 istek aynı anda, saniyede 53 sn ses (10 adım, referanssız), 49
    (referanslı), 16 adımda 38 / 36. Kalite resmi API ile aynı (4 cümlede WER 0, UTMOS gürültü içinde).
  - İşler (LoRA, adım sayısı) gruplarına ayrılır. Her grup bir sunucu açar (~60 sn), işlerini bitirir ve kapanır.
    Sonraki grup ve puanlayıcılar (Whisper, UTMOS, ECAPA) boş GPU bulur.
  - İstek resmi API'nin çağrısıyla aynıdır: tasarımda `(tarif)` + metin, klonda yalnız referans
    (`ref_audio_latents`, devam modu yok), duygulu satırda `(talimat)` + metin. Uzunluk sınırı ve hatalı klibi
    yeniden üretme de aynıdır: 6 × metin token'ı, tohum + 1, en çok 3 deneme.
  - Tohum: Nano-vLLM her isteği kendi tohumuyla üretir, toplu işteki öbür isteklerden bağımsız. Taramadaki 2 tohum
    iki ayrı istektir. Aynı tohum resmi API'dekinden başka bir ses verir; ayarlar bu yüzden `backend`'i de kaydeder.
  - LoRA: sabitlenmiş Nano-vLLM kaynağı resmi eğiticinin LoRA'sını (`lora_weights.safetensors` +
    `lora_config.json`; s2: r 64, LM + DiT q/k/v/o) yükler. Yükleyemediği bir LoRA (başka katmanlar, config yok ya
    da sunucu reddeder) yalnız o girişler için resmi API'ye gider. Bu karar `entry.json`'da `fallback` olarak kalır.
  - `--backend voxcpm` eski yoldur: birkaç VoxCPM2 süreci aynı kuyruktan (`modal_app.py ... --backend voxcpm
    --args "--workers 6"`). Smoke ölçümü: 6 süreçten 4'ü yüklendi, toplam yalnız ~1.0 sn ses / sn.
  - Ölçülen hız `/data/contest/throughput.jsonl`'a yazılır (Nano-vLLM: sunucu başına bir satır), sonraki planlar
    onu kullanır.
  - Tarama ve yarışma 10 adımla, final 16 adımla yapılır.

Komutlar `$env:MODAL_PROFILE="<profiliniz>"` ile, depo kökünden çalışır:

- Her GPU komutu önce planı ve "at most $X"i yazar. `--max-usd` kesin süre sınırıdır.
- Yarıda kalan iş aynı komutla devam eder.
- `--args "... --dry-run"` GPU açmadan yalnız planı yazar.

```powershell
python tts/voxcpm/fetch_refs.py                                              # yerel, ücretsiz (yapıldı): ./voxcpm-refs/<ses>/
python -m modal run tts/voxcpm/modal_app.py --action contest-refs-push       # ücretsiz: -> /data/contest/refs/<ses>/
python tts/voxcpm/nanovllm_app.py                                            # Nano-vLLM hız testinin planı, yerel
python -m modal run tts/voxcpm/nanovllm_app.py --action probe --max-usd 0.8  # ~16 dk, en fazla $0.80
python -m modal run tts/voxcpm/modal_app.py --action contest-render --run smoke2 --max-usd 0.7 --args "--limit 1"
python -m modal run tts/voxcpm/modal_app.py --action contest-screen --run r1 --max-usd 1.2
python -m modal run tts/voxcpm/modal_app.py --action contest-render --run r1 --max-usd 1.5
python -m modal run tts/voxcpm/modal_app.py --action contest-render --run r1 --max-usd 0.8 --spec s1.json --args "--only a+s1,b+s1,c+s1,d+s1"
python -m modal run tts/voxcpm/modal_app.py --action contest-render --run r1-final --max-usd 0.9 --args "--timesteps 16 --screen r1 --only a,b,c,d"
python -m modal run tts/voxcpm/modal_app.py --action contest-fetch --run r1  # GPU açmaz: ./voxcpm-contest/r1/
```

Sırasıyla (A100-80GB, ~$3.26/sa CPU/RAM dahil):

1. **smoke2:** 9 giriş × 3 cümle, Nano-vLLM ile ilk gerçek deneme (~9 dk, ~$0.5). İki sunucu açılır: s2 LoRA'lı
   kontrol ve referanslı girişler. Eski `smoke` resmi API ile yapıldı; onun girişleri `--backend voxcpm` ister.
2. **Tarama:** 300 aday + 120 klon (~17 dk, ~$0.9).
3. **r1:** 14 giriş × 70 cümle (~22 dk, ~$1.2).
4. **+s1 kolu:** `leaderboard.md`'deki en iyi 4 aday (kontroller hariç), aşama-1 LoRA'sıyla (~11 dk, ~$0.6).
5. **Final:** en iyi 4 giriş, 16 adımla (~12 dk, ~$0.6; biri `+s1` ise ~$0.7).

Süreler probe'un ölçtüğü hızla ve sunucu başına ~70 sn açılış/kapanışla hesaplandı. Artık süreyi çoğunlukla puanlama
belirler (klip başına ~0.7 sn: Whisper, UTMOS, ECAPA). Hepsi ~$4 eder; resmi API ile (`--backend voxcpm`) ~$12.
`--max-usd` sınırları tahminin ~1.25 katıdır. `s1.json`:

```json
{"refs": "auto", "arms": [{"lora": "voxcpm2-alania-s1/ckpt/step_0002000", "suffix": "+s1", "entries": ["a", "b", "c", "d"]}]}
```

Final girişlerinden biri `+s1` ise final komutuna `--spec s1.json` eklenir. Kör dinleme için wav'lar
`voxcpm-contest/r1-final/<giriş>/wav/` altında.

### Tur 2 düzeltmeleri: baştaki fazladan ses ve ses düzeyi

r1'de Nano-vLLM kliplerinin çoğu kelimeden önce ~80 ms'lik bir "a / e", gırtlak sesi ya da patlamayla başladı
(Y ile başlayan 144 klibin 85'i; resmi API'de 32'de 2). Ses düzeyi de referansı izledi: sessiz referans, sessiz klip
(Dima -32 dBFS, Masha -26). Düzeltmeler:

- **Metin başına tohum:** her klip `--seed` + crc32(`<set>/<id>`) tohumuyla istenir. Bir cümle her seste aynı tohumu,
  her cümle başka bir tohumu alır (ayar: `seed_mode: per_text`). r1'de her klip tohum 0 aldı. Nano-vLLM her adımın
  gürültüsünü (tohum, adım)'dan türettiği için bütün klipler aynı ilk-adım gürültüsünü paylaştı (tasarım adayları:
  tohum 0'da 12'nin 6'sı, tohum 1'de 18'in hiçbiri). Satırın kendi `seed`'i varsa o kullanılır. Taramada her tarif
  kendi tohumunu alır.
- **Referans hazırlığı (`refprep.py`):** her referans iki motora da 16 kHz mono, **sağdan** sıfırla 2560 örneğin
  katına (160 ms'lik yama) tamamlanmış, etkin konuşma -20 dBFS, tepe en çok -1 dBFS olarak gider. Nano-vLLM eksik
  yamayı soldan, resmi API ve eğitim sağdan dolduruyordu. Her klip de yazılmadan önce -20 dBFS'e getirilir
  (`loudness`). `--no-refprep` ya da spec girişinde `"ref_prep": false` ham klibi verir.
- **Başlangıç kontrolü (`onset.py`):** ünlü, ötümlü ünsüz ya da y ile başlayan metinlerde kelimeden önceki fazladan
  ünlüyü, gırtlak sesini, baştaki patlamayı ve ötümsüz girişi arar (Praat'ın perde ve formant izlerinin numpy
  karşılığı). r1 A/B 0-1. denemelerin 14 klibinin 12'sini, resmi API smoke kliplerinin 21'inin 1'ini işaretler.
  - Nano-vLLM'de işaretli klip tohum + 1, + 2, + 3 ile yeniden üretilir. Hâlâ işaretliyse kelimenin başlangıcından
    30 ms önceden kesilir (10 ms yumuşak giriş) ve loglanır. Resmi API'nin klipleri yalnız işaretlenir.
  - Satırlarda `onset_bad` (ilk denemede işaret; kapsam dışı metinde boş), `onset_why`, `onset_retries`,
    `onset_trimmed`. `summary.json`'da `onset`, `leaderboard.md`'de "onset stray" sütunu. `score` komutu eski
    kliplere de (r1) bakar.
- Ayarlar değiştiği için r1 dizinlerine bu kodla klip eklenmez (reddedilir): tur 2 yeni bir `--run` ister.

Doğrulama (`verify_onset.json`: masha, dima, designed-f037-s1, hazır ve ham referansla; Y ile başlayan 6 cümle):

```powershell
python -m modal run tts/voxcpm/modal_app.py --action contest-render --run onset-v1 --max-usd 0.5 --spec tts/voxcpm/verify_onset.json --args "--ids sira_01,soru_01,duz_eslesik_01,mi_soru_01,emo_angry_01,emo_empathetic_01 --seeds 0,1,2,3 --no-score"
python -m modal run tts/voxcpm/modal_app.py --action contest-render --run onset-v1-api --backend voxcpm --max-usd 0.5 --spec tts/voxcpm/verify_onset.json --args "--only masha,dima,designed-f037-s1 --ids sira_01,soru_01,duz_eslesik_01,mi_soru_01,emo_angry_01,emo_empathetic_01 --seeds 0,1 --workers 2 --no-score"
```

İlki Nano-vLLM'de 6 giriş × 6 cümle × 4 tohum = 144 klip (tahmin ~$0.33), ikincisi resmi API'de 3 giriş × 6 cümle ×
2 tohum = 36 klip (tahmin ~$0.39). Puanlama yok; başlangıç işaretleri sentezden gelir. Tohum 0 r1'in koşuludur.
Oran tohumla değişiyorsa düzeltmeler yeter. Nano-vLLM her tohumda kötü, resmi API temizse finaller resmi API ile
yapılır.

### Seçilen sesle eğitim verisi (`render_corpus.py`)

`docs/SES_PLANI.md` adım 3: designed-f022-s0 sesiyle ~5 sa temiz derlem, `/data/f022corpus/`, `train_lora.py --data`
düzeninde (`wav/<id>.wav` 16 kHz, `voxcpm_train.jsonl`, `clips.jsonl`, `prep_report.json` + `PREP_REPORT.md`,
`ATTRIBUTION.txt`). Nano-vLLM LoRA'mızı yükleyemediği için taban VoxCPM2 yalnız referanstan klonlar
(`/data/contest/r1/screen/cand/wav/f022-s0.wav`, refprep ile hazırlanmış).

- **Metin karışımı (tutulan saate göre, `--mix`):** %40 düz cümle `cloud0day3/alania-domain-text-tr`'den (CC BY 4.0,
  sabit commit; sayı, saat, para, tarih, telefon, adres, randevu, banka, kargo kategorileri; `text_raw` trnorm'dan
  geçer, 25-180 karakter), %25 soru, %20 duygu, %10 kısa, %5 ünlem `textbank/textbank.jsonl`'dan. Biten kategorinin
  açığı düz cümleye gider. Sızıntı koruması her zaman: `benchmarks/tts/sets/*.jsonl` (refs dahil) ve Antalia'nın
  ayrılmış cümleleri (Modal'da `/data/antalia16k/voxcpm_val.jsonl`, yerelde `voxcpm-eval/antalia-gt/rows.jsonl`).
  Tekrar anahtarı son işareti de tutar: "Pardon!" / "Pardon?" gibi bilinçli soru-düz çiftleri ikisi de kalır.
  Duygularda yalnız `instruction_split: train` satırları üretilir (heldout talimatlar değerlendirme için).
- **Eğitim metni:** trnorm okunuşu; duyguda `(talimat)` + okunuş. Ünlemler `render_variants` yazımlarıyla denenir
  ("vaay"), Whisper kanonik kelimeyi ("vay") duyunca tutulur, eğitim metni kanonik kalır.
- **Kapılar (hepsi geçmeli; yoksa tohum + 1000 ile yeniden, en çok `--tries` 4, sonra atılır):** baştaki fazladan ses
  yok (onset.py), 7-25 karakter/sn, Whisper CER 0 ve p(tr) ≥ 0.95, ECAPA benzerliği ≥ `--min-sim` (0.70; 4 sn'den kısa
  kliplerde 1 sn'de 0.12 daha az, çünkü ECAPA kısa klibi düşük puanlar), UTMOS kategorinin ilk denemelerinin alt %20'si
  değil (duygular her duygu için ayrı), soru türleri tonlama puanlayıcısından geçer, eşli düz cümle soru gibi
  yükselmez, duygu prozodisi nötr kliplere göre açıkça ters değil.
- **Pilot sonrası kapılar (2026-10-12):** deneme derleminin 99 klibi ve bir anadil dinleyicisinin 44 klip üzerindeki
  notlarıyla ayarlandı. Whisper artık her denemede kelime zamanlarını da verir.
  - *Kuyruk:* konuşma sonu, son kelimenin bitişinden kendi sesi boyunca 150 ms sessizliğe kadar. Ondan sonra gelen kısa
    ses (≤ 0,35 sn) kesilir: konuşma sonu + 120 ms, 20 ms sönümle. Klip 41'deki "Özür dilerim." + "n" böyle kesildi;
    99 klipte başka kesilen yok. Ses son kelimenin içindeyse ya da uzunsa deneme atılır.
  - *Baş:* sessiz harfle (k, h, ş, ...) başlayan metinde klip önce kısa ve yüksek bir sesli patlamayla başlıyorsa atılır
    ("Kan tahlili ...", "Hiç fikrim yok."; onset.py bunlara bakmıyordu). İlk kelimeden önce 150 ms'den fazla ses de
    atılır.
  - *İlk kelime uzaması:* ilk kelimenin hece başına süresi, diğer kelimelerin ortancasının 2,2 katını aşarsa atılır
    (ünlemler hariç). **Klip 7 ("Buu iki elbiseden") bununla yakalanamıyor:** Whisper'a göre 1,19 kat. Sesli sınır
    düzeltmeleri de (F2 geçişi, enerji çukuru, harf başına süre) onu diğer kliplerden ayıramadı. Yakaladığı iki klip,
    başında sesli patlama olan iki klip (3,5 ve 2,5 kat).
  - *Aşırı "ş/s":* 2-8 kHz sürtünmeli sesin 40 ms sürdüğü en yüksek düzeyi, klibin yüksek çerçevelerine göre
    -4,5 dB'yi aşarsa atılır. Klip 15'teki "her şey" -3,4 dB; dinlenip sorun bulunmayan kliplerin hepsi ≤ -5,7 dB
    (ortanca -12).
  - *Hız:* konuşma süresine (kelime zamanları, duraklar hariç) göre hece/sn üst sınırı: düz cümle ve kısa cevap 8;
    soru 8,5; duygu 7,8; ünlem 7,5 (`--speed-scale` ile ölçeklenir). Klip 9 8,9 hece/sn; işaretlenmeyen en hızlı soru
    8,3.
  - *Ne/nasıl soruları:* soru sözcüğündeki tepe, kendinden önceki kelimelerin tepesinin 2 yarım tondan fazla altında
    kalırsa ya da perde düz kalırsa (aralık < 5 yarım ton) geçmez. Klip 5'te "Çocuklar", "nasıl"dan 3,9 yarım ton
    yüksek. Örnekteki 9 ne_soru klibinden önce 9'u geçiyordu, şimdi 8'i geçiyor. mI kuralları aynı.
  - *Üzgün/sakin:* nötr kliplere göre hem daha yavaş hem daha alçak perdeli olmalı ve normalizasyondan önceki
    düzeyde 1 dB'den fazla yüksek olmamalı (ses düzeyi artık sentez sırasında, normalizasyondan önce kaydediliyor).
    Önceden tutulan 21 üzgün/sakin klibin 7'si bu kuralla elenirdi. Klip 17 (üzgün gelmiyor) bu ölçülerle zaten
    "üzgün" görünüyor: onu düzeltecek olan duygu referansı.
  - **99 klipte yanlış alarm:** yeni kapılardan biri ya da birkaçı 7 klibi eliyor. Bunların 3'ü dinleyicinin
    işaretlediği klip (9, 15, 5); 41 kesilip tutuluyor. Dinlenip sorunsuz bulunan 38 klipten hiçbiri elenmiyor; beğenilen
    13, 19, 33, 34 geçiyor. Dinlenmemiş 4 klip eleniyor: ikisinin başında sesli patlama var, biri sert "s" (kızgın),
    biri nötrden yavaş olmayan bir "sakin". Tahminler bu kapıların geçme oranını `NEW_GATE_PASS` ile hesaba katar.
- **Duygu referansları (ilk çağrıda bir kez):** her duygu için kısa bir nötr cümle ("Yarın sabah erkenden yola
  çıkacağız, ...") f022 referansı ve o duygunun talimatıyla 6 tohumla, ayrıca 4 kez talimatsız üretilir. Whisper'ın
  doğru duyduğu, ECAPA benzerliği ≥ 0,6 olan adaylar arasından duyguya en uygun prozodiye sahip olan seçilir.
  Talimatsız üretimlere göre üzgün için daha alçak, yavaş ve normalizasyondan önce daha kısık; kızgın/neşeli/şaşkın
  için daha geniş perde aralığı ve daha yüksek ses aranır. Seçilen aday `refs/<duygu>.wav` olur ve o duygunun
  satırları f022 yerine bunu klonlar. Uygun aday çıkmayan duygu f022 referansıyla devam eder (`PREP_REPORT.md`'de yazar).
- **Gülme denemesi (eğitime girmez):** `textbank/laughter.jsonl`'daki 52 özgün satır ("Hahaha, çok komiksin!"),
  3 talimat ("Laughing heartily while speaking." ...) × 2 tohumla üretilir; bütçe `--laughter-usd` (varsayılan 0,3 $,
  tahmin ~0,2 $). Whisper "haha"yı düşürdüğü için CER gülme çıkarılarak ölçülür. Gülme, Whisper kelimelerinin
  boş bıraktığı aralıkta patlama/perde ölçüsüyle aranır. AST AudioSet modeli yalnız HF önbelleğinde zaten varsa
  kullanılır (yerelde pilot kliplerinde gülme olasılığı en çok 0,03; konuşmada yanlış alarm vermiyor). Her satırın en
  iyi 3 denemesi `/data/f022corpus/laughter_review/`'a gider. `corpus-fetch` bunları
  `voxcpm-corpus/f022/laughter.html` sayfasıyla getirir: her denemede "gerçekten gülüyor" kutusu, satır başına yorum
  ve sonucu kopyalayan düğme.
- **Onaylanan gülmeler eğitime girer (`--add-reviewed`):** dinleyicinin "gerçekten gülüyor" dediği denemeler
  `textbank/laughter_approved.json`'da (`{"l_001-1": true, ...}`, `best`: en iyi bulunanlar, `note`; 156 adaydan 48'i,
  ~135 sn; en iyi talimat "Laughing heartily while speaking."). `corpus-add-reviewed` bunları GPU'suz, birkaç saniyelik
  bir CPU çağrısıyla derleme ekler: `wav/laugh_<id>.wav` (16 kHz PCM16), `reviewed.jsonl` (kategori `laughter`,
  kaynak `human-approved`), `clips.jsonl` ve `voxcpm_train.jsonl` sonuna birer satır (metin `(talimat)` + trnorm
  okunuşu, gülme yazıldığı gibi: "(Laughing heartily while speaking.)Hahaha, çok komiksin!"; `--ref-fraction`
  kadarına ref_audio), raporda `laughter` kategorisinin saati. Tekrar çalıştırmak yeni bir şey eklemez. Yalnız iki
  `corpus-render` çağrısının arasında çalışır: `status.json` çağrının sürdüğünü söylüyorsa ya da `state.json` rapordan
  yeniyse reddeder (yarıda öldürülen çağrıda önce aynı `corpus-render` komutu). Sonraki `corpus-render` çağrıları
  `reviewed.jsonl`'ı korur: WAV'ları silinmez, manifest ve rapor onlarla yazılır.
- **Dalgalar:** bir dalga gereken metinlerin 2 denemesini birden üretir (tek Nano-vLLM sunucusu, 64 istek), sunucu
  kapanır, puanlayıcılar sırayla dener, karar verir; kalanlar sonraki dalgada. `state.json` + `attempts/` ile kaldığı
  yerden sürer; `--max-usd` sınırında durur, aynı komut devam eder. `--backend voxcpm` resmi API ile aynı işi yapar.
- **Maliyet tahmini:** komut önce kalan işi hesaplar (Nano-vLLM 49 sn ses/sn, puanlama r1'deki ölçümle klip başına
  0.25 + 0.08 × süre sn; sonraki çağrılar `throughput.jsonl`'daki gerçek ölçümü kullanır). Derlem kendi geçme oranını
  ölçene kadar (kategori başına 30 karar) `/data/f022corpus`'ta ölçülen deneme başına geçme oranı kullanılır
  (`--min-sim 0.6`: düz 0.41, soru 0.27, duygu 0.23, kısa 0.20, ünlem 0.20; `--min-sim` değişirse ECAPA payıyla
  ölçeklenir).

```powershell
python tts/voxcpm/render_corpus.py --out ./scratch-corpus --dry-run                        # yerel plan, ücretsiz
python -m modal run tts/voxcpm/modal_app.py --action corpus-render --max-usd 4 --args "--dry-run"
python -m modal run tts/voxcpm/modal_app.py --action corpus-render --max-usd 4             # tekrarla: kaldığı yerden
python -m modal run tts/voxcpm/modal_app.py --action corpus-fetch                          # rapor + 100 rastgele klip
python -m modal run tts/voxcpm/modal_app.py --action corpus-add-reviewed --args "--dry-run" # onaylı gülmeler: plan
python -m modal run tts/voxcpm/modal_app.py --action corpus-add-reviewed                   # ekle (CPU, ~0,0001 $)
python -m modal run --detach tts/voxcpm/modal_app.py --action train --exp-name voxcpm2-f022 --steps N --args "--data /data/f022corpus"
```

`corpus-fetch` raporu ve dinlemek için 100 klibi (`sample.jsonl` ile) `./voxcpm-corpus/f022/`'ye getirir. Duygu
referanslarını (`refs/`) ve gülme denemesini (`laughter_review/`, `laughter.html`) de getirir. Eğitim, derlem yarıda
kesilmişse ya da karışımı tamamlanmamışsa başlamaz. Kuru çalıştırma, ilk çağrının ön işini (duygu referansları +
gülme denemesi, ~460 sn ≈ 0,40 $) toplam tahmine katar.

### İkinci derlem: uzun cümleler ve sayılar (`--out /data/f022corpus2`), sonra birleştirme

f022 modeli düz okumada EMA Lightning'e kaybediyor ve uzun cümlelerde zayıf (12-19 sn: UTMOS 3.12, kısalarda 3.71;
eğitim klipleri ortalama 3,5 sn, en uzunu 11,7 sn). Yeni ~1,5 sa veri ayrı bir klasöre üretilir, `/data/f022corpus`'a
(state'ine, onaylı gülmelerine) dokunulmaz; sonra iki derlem tek manifestte birleşir ve aşama-1 LoRA'sından yeniden
eğitilir.

- **Uzun öğeler (`long`, yalnız `--mix`'te payı varsa):** Alania Domain Text'te her satır kendi `doc_id`'sine sahip
  (`qwen3:s3-p<istem>-<k>`, `unit_idx` 0); aynı üretim isteminin satırları aynı kategori ve alt konuda, `k` sırasıyla.
  Aynı istemin 1-3 **ardışık** satırı bir öğe olur: okunuşu 120-300 karakter (`--long-chars`); her dizide okunuşu
  `--long-target` aralığının (190-280 karakter, ~12,5-17 sn) ortasına en yakın pencere seçilir, çekilişte bu aralıktaki
  pencereler önce alınır (2180 gerçek satırda: medyan 14,5 sn, hepsi ≥ 12 sn; "sığan en az satır" kuralı medyan
  9,9 sn veriyordu). Tek satır ancak kendisi o uzunluktaysa öğe olur (uzun bir cümle: çoğu `long_explanations`
  satırı). Her satırın cümle sonu işareti korunur (yoksa ya da satır virgül/iki noktayla bitiyorsa "." konur), böylece
  model cümle arası duraklamayı öğrenir. Araya giren bir soru, düşük dil skoru, okunmamış bir `k` ya da başka
  derlemde kullanılmış bir satır diziyi böler: birleşen satırlar hep komşu.
  Kategoriler `--long-weights` (açıklamalar %25, destek/randevu/banka %15'er, kargo/sayılar/adresler %10'ar);
  sızıntı koruması hem bütün metne hem her satıra; havuzda `--long-lines` (1500) öğe, düz cümlelerden sonra çekilir
  ve onlarla satır paylaşmaz.
- **Uzun klip kapıları:** CER 0 (gevşetme yok), metin başına `--long-tries` 6 deneme, baş/kuyruk/ş-s/hız kapıları aynı,
  ayrıca **duraklama kapısı**: iki kelime arasındaki sessizlik Whisper kelime zamanlarındaki boşluk ya da klibin o
  boşluğa en yakın sessiz bölümü (hangisi uzunsa). Satırların birleştiği her yerde 0,12-0,9 sn (`--pause-sent`),
  başka her yerde en çok 0,6 sn (`--pause-mid-max`), hiçbir yerde 1,2 sn'den uzun değil (`--pause-max`). Cümle
  sınırı satır birleşimleridir (öğenin `breaks` alanı), trnorm'un satır içinde bıraktığı "." değil (baş harf "ka.",
  sıra sayısı "beş."): bunlar isimde duraklama istemez. Kelimeler eşleşmezse deneme atılır.
- **Sayılı düz cümleler (`--number-share`, varsayılan 0,6):** yazılı biçim kontrolleri (telefon, saat, para, tarih,
  yüzde, kısaltma, diğer rakamlar). Çekilişte her kategoride önce sayılı satırlar, türlere yayılarak (telefon, saat,
  para, tarih, yüzde de en az rakam kadar sık) alınır; toplamda payları 0,6'nın altındaysa başka kategorilerden
  tamamlanır. Her dalga yeni düz metinleri, tutulan saniyelerde sayılı payı ≥ 0,6 kalacak sırayla alır (sınıfların
  ölçülen geçme oranıyla). İkisi de havuz yettiği kadar. Rapor: havuzdaki ve tutulandaki pay, türler.
- **İlk derlemden ödünç (`--refs-from /data/f022corpus`):** f022 referansı (`ref/`, ham klibin aynı olduğu
  doğrulanır) ve duygu referansları (`refs/` + state kayıtları) kopyalanır. **`--avoid-pool /data/f022corpus`:** ilk
  derlemin havuzundaki hiçbir metin ve hiçbir domain satırı yeniden kullanılmaz. İlk derlem yalnız okunur.
- **Birleştirme (`corpus-merge`, CPU, saniyeler):** `/data/f022mix/voxcpm_train.jsonl` = iki manifestin birleşimi
  (satırlar aynen: mutlak WAV yolları, ref_audio; aynı ses ya da önceki derlemde olan metin bir kez), `prep_report.json`
  (`merged`, `complete`, kaynak manifestlerin bayt/sha256'sı), `README.md`, `ATTRIBUTION.txt`. Kaynaklardan birinde
  `corpus-render` sürüyorsa ya da karışımı tamamlanmadıysa reddeder (`--allow-incomplete` ile alır). Eğitim ön
  kontrolü, birleştirmeden sonra değişmiş bir kaynak manifesti görürse eğitimi başlatmaz: yeniden birleştir.
- **Ölçmeden harcamama:** uzun klibin geçme oranı tahmin (temkinli 0,10: CER 0'ı ~3 kat karakterde tutturmak ~0,2 ×
  diğer kapılar 0,7 × duraklama ~0,7). Kategori 30 karar görene kadar aynı anda en çok 40 uzun metin denenir; ilk
  dalga oranı ölçer (ilk $2'lık çağrıda uzunlara ~$0,14), sonraki dalgalar ölçülenle planlar. Ölçülen oran
  `--min-pass`'ın (0,05) altındaysa kategori yeni metin almaz (günlükte ve raporda "stopped"; hedefi düz cümlelere
  devredilmez, derlem tamamlanmaz: kapı başarısızlıklarına bak, `--min-pass 0` ile yine de sürer).
- **Maliyet:** sentez süresi ×1,3, puanlama klip süresiyle orantılı (ölçülen hız: 37,5 sn ses/GPU sn, puanlamada
  model × 1,008). Kuru çalıştırma bu çağrının ve **kalan işin tamamının** tutarını yazar: kaç `--max-usd` çağrısı
  gerektiği, her çağrının açılış + bitiş payı (~137 sn) ve sonraki her çağrının sunucu + puanlayıcı yüklemesi
  (~212 sn). Tahmin (uzun geçme oranı 0,10 / 0,15 / 0,20; $2'lık çağrılar, $3'lıklarla ~$0,3 daha az):
  1,5 sa ~$6,4 / $5,0 / $4,4; 1,0 sa ~$4,6 / $3,9 / $3,2; 0,75 sa ~$3,9 / $3,1 / $2,8. `cwords` %6 ile 1,0 sa
  ~$4,7 / $4,0 / $3,3 (c-check ~63 sn; `cwords` dalgada 4 deneme üretir, 8 denemesi dalga eklemez; duygu referansları
  `--refs-from`'dan ödünç, prelude yok); `--max-usd 3` çağrılarla ~$4,4 / $3,7 / $3,3. Üstüne prob ~$0,7, eğitim
  ~$1,5-3. (Puanlama satırları `throughput.jsonl`'a artık faktörsüz tahmini yazar, `"raw": true`; plan bunları
  tercih eder, eskileri kendi çağrısının faktörünü içeriyordu.)

```powershell
$env:MODAL_PROFILE="<profiliniz>"
python -m modal run tts/voxcpm/modal_app.py --action corpus-audit-c --args "--dry-run"           # ilk derlemin c denetimi: plan
python -m modal run tts/voxcpm/modal_app.py --action corpus-audit-c                              # L4, ~3 dk, ~$0,06 (en çok $0,26)
python -m modal run tts/voxcpm/modal_app.py --action corpus-render --out /data/f022corpus2 --max-usd 2 --args "--hours 1.0 --mix long=0.47,neutral=0.47,cwords=0.06 --min-sim 0.6 --laughter-usd 0 --refs-from /data/f022corpus --avoid-pool /data/f022corpus --dry-run"
python -m modal run tts/voxcpm/modal_app.py --action corpus-render --out /data/f022corpus2 --max-usd 2 --args "--hours 1.0 --mix long=0.47,neutral=0.47,cwords=0.06 --min-sim 0.6 --laughter-usd 0 --refs-from /data/f022corpus --avoid-pool /data/f022corpus"
python -m modal run tts/voxcpm/modal_app.py --action corpus-fetch --out /data/f022corpus2        # ./voxcpm-corpus/f022corpus2/: uzunları dinle, PREP_REPORT.md'de duraklama + geçme oranı + c-check
# kuru çalıştırma yeniden (ölçülen oranla kalan iş), sonra aynı komut --max-usd 3 ile tamamlanana kadar
python -m modal run tts/voxcpm/modal_app.py --action corpus-merge --args "--dry-run"            # kaynak başına c_audit.jsonl'ın elediği klip sayısı
python -m modal run tts/voxcpm/modal_app.py --action corpus-merge                               # -> /data/f022mix
python -m modal run tts/voxcpm/modal_app.py --action probe --args "--data /data/f022mix --batch-size 8 --grad-accum 2"
python -m modal run --detach tts/voxcpm/modal_app.py --action train --exp-name voxcpm2-f022-s3 --steps 1500 --max-usd 3 --args "--data /data/f022mix --init-lora /data/exp/voxcpm2-alania-s1/ckpt/step_0002000 --batch-size 8 --grad-accum 2"
```

### c harfi (/dʒ/): kapı, `cwords`, denetim, prob

f022-1500 İngilizce duygu talimatıyla ("(Astonished, with rising intonation.)", "(Pleasantly surprised.)") kelime
başındaki "c"yi /k/ ya da /d/ okuyor: "Vay canına" 6 yazımda 18'de 0 doğru ("kanına"), "Cam kırılmış" -> "Kam"/"Dam";
talimatsız 8'de 8 doğru, "Very happy and excited." ile "Canım"/"cumartesi" doğru. Whisper large-v3 bunu çoğu zaman
gizliyor (dil modeli /k/ klibine "canına" yazıyor), yani CER 0 kapısı yetmiyor.
`facebook/wav2vec2-xlsr-53-espeak-cv-ft` (Apache-2.0, IPA sesbirimi CTC, dil modeli yok; sabit revizyon) kalibre
dinleyicimizle 8'de 8 uyuştu: doğru c -> "dʒ", yanlış -> "k"/"d"/"ɡ" (`probe-vay`: v1a/v1b doğru, v1c/v1d yanlış;
`probe-canina2`: k5/k8/k9 doğru, k1/k4/k7 yanlış).

- **`phon_check.py`:** model bir kez yüklenir (GPU varsa GPU'da, fp32, HF önbelleği Volume'da `/data/hf_home`),
  `ipa(wav, sr)` -> sesbirim listesi; `c_check(okunuş, sesbirimler)`: beklenen = okunuştaki "c" dizileri (Türkçe küçük
  harf, "ç" sayılmaz, "cc" bir), duyulan = "dʒ" (uzun/damaksıl biçimleri, "ʤ" de); duyulan ≥ beklenen ise geçer.
  Hangi c'nin yanlış olduğunu söylemez, yalnız birinin yanlış olduğunu.
- **Kapı (`render_corpus.py`):** talimatı olan ve okunuşunda "c" bulunan her deneme (duygu satırları, `cwords`; gülme
  yok) CER'den sonra bu kontrolden geçer; geçmeyen yeniden üretilir (kapı adı `c_dj`, raporda kategori başına sayılır,
  `PREP_REPORT.md`'de "c-check" satırı). Düz satırlara uygulanmaz (onlar doğruydu, deneme harcar). Model yalnız böyle
  bir satır varsa yüklenir, dalgalar arasında CPU'da bekler. `--no-c-check` kapatır.
- **`cwords` kategorisi:** `textbank/cwords.jsonl`'ın 90 satırı (kelime başı c; şaşkın 46, neşeli 20, kızgın 8, üzgün 6,
  empatik 6, sakin 4) duygu satırı gibi: duygunun referans klibi, prozodi kapısı, eğitim metninde `(talimat)`; metin
  başına `--cwords-tries` 8 deneme (taban model de bu hatayı yapıyor), dalga başına 4'ü üretilir (kısa klipler; 2'şer
  olsa 8 deneme uzunların 6'sından bir dalga, ~212 sn, uzun sürerdi). Yalnız `--mix`'te payı varsa havuza girer
  (`--mix long=0.47,neutral=0.47,cwords=0.06`: 1 sa'te ~216 sn, ~85 klip). Havuz paydan küçükse plan bunu yazar,
  kalanı `--fill`'e (düz cümle) gider. Geçme oranı ölçülene kadar tahmin 0,35 (ilk dalga 40 metinle ölçer; duygular
  aynı kapılarla 0,225 ölçüldü, gerçek oran daha düşük çıkabilir: 0,15 ise ~$0,1, 0,08 ise ~$0,2 fazla).
- **Talimatsız üretim (`--plain-render cwords`, 2026-10-13):** `/data/f022corpus2`'nin 1. çağrısında `cwords`
  denemelerinin yalnız 0,044'ü geçti (590 denemede 26 klip; `--min-pass` 0,05 durdurdu). CER başarısızlıklarının
  384/396'sında ilk kelime yanlış: taban VoxCPM2, İngilizce "(talimat)"tan hemen sonra gelen "C"yi İngilizce okuyor
  ("Cuma" -> "Kuma"/"Çuma", "Cesaret" -> "Sesaret", "Cebimdeki" -> "Zebimdeki"); talimatsız aynı ses doğru (prob 8/8).
  Bu bayrakla verilen kategorilerin (`emotions`, `cwords`; `--mix`'te payı olmalı) satırları modele talimatsız
  verilir, yine duygunun referans klibini (`refs/<duygu>.wav`, duyguyu o taşır) klonlar; eğitim metni aynı kalır
  ("(talimat)" + okunuş: LoRA talimattan sonra da c'nin /dʒ/ kaldığını öğrenir), c-check yine çalışır (satırın
  talimatı var), denemeler ve klipler `"render": "plain"` yazar. Duygu kontrolü daha sıkı: prozodi kararı "expected"
  olmalı **ve** ölçüler birlikte duygunun yönüne gitmeli (`plain_score > 0`: `emo_ref_score`, referansın seçildiği
  ölçü); kuralı olmayan empatik için yalnız daha pes **ve** kısık (`plain_score`; referansı -3,9 st, -1,6 dB). Hız
  sayılmaz: 1. çağrıda her duygudaki `cwords` denemesi nötr kliplerden ~1 hece/sn yavaştı (kısa cümleler), "yavaş +
  kısık" kuralı 42 şaşkın denemenin 37'sini geçirirdi, "pes + kısık" 2'sini (empatiklerin 12'de 5'ini). "weak",
  "opposite" ve ölçülemeyen ("unchecked") düşer, kapı adı `emotion_plain`. (Talimatlı satırlarda "weak"
  kızgın/neşeli/şaşkın ve her empatik geçiyordu:
  talimat duyguyu istiyordu; talimatsız klipte duyguyu yalnız klonlanan ses taşır, gösterilmesi gerekir.) Duygunun
  referansı ya da 20 nötr kliplik taban çizgisi yoksa satırlar bekler.
  Mod kategori başına `state.json`'da (`"modes"`). Geçişte: tutulan metinler kalır; diğerleri (denemesi bitmiş
  olanlar dahil) sıradaki denemeden başlayan **yeni modda bir deneme bütçesi** alır (`try0`: yeni tohumlar, hiçbiri
  tekrarlanmaz); bir metin her modda en çok bir bütçe dener (`mode_tries`: daha önce olduğu moda dönünce oradaki
  denemeleri sayılır, bitirmiş olan düşmüş kalır), eski modun karar verilmemiş denemeleri silinir; eski modun
  sayaçları kendi adı altında kalır
  (`stats.<kategori>.modes`), geçme oranı, `--min-pass` durdurması ve 40 metinlik ölçüm yeni modda yeniden başlar
  (ölçülene kadar tahmin `PLAIN_PASS` 0,15: 1. çağrıdaki denemelerden: nötrlerde Whisper 0,585 doğru, `cwords`'te
  CER + c'den geçenlerin 0,26'sı gerisini geçti, ECAPA 100'de 59'unu eledi); UTMOS talimatsız denemeleri kendi ilk
  denemeleriyle kalibre eder (`cwords:plain`). `--mix`'te payı olan talimatsız bir kategori varken sonraki her
  çağrı bayrağı yeniden vermeli: bayraksız çağrı (yerelde `modal_app` da GPU başlatmadan) reddedilir, `--plain-render
  none` talimata geri döner (yalnız talimatlı bütçesi kalan metinler dener). `--mix`'te payı olmayan kategori modunu
  korur, bayrak istemez, bir şey üretmez.
  `PREP_REPORT.md`: "Render modes" tablosu (mod başına deneme/tutulan/geçme/ilk deneme/başarısız kapılar), duygu
  kararları ve c-check `cwords (plain)` olarak ayrı.
- **Sızıntı koruması:** `probe_canina*.jsonl` ve `probe_tr_instr.jsonl` cümleleri (test setimiz, yazım varyantları ve
  kanonik `refs`) hiçbir havuza giremez.
- **Eski derlemin denetimi (`corpus-audit-c`):** `/data/f022corpus` manifestindeki talimatlı ve c'li klipler (171
  duygu + 3 onaylı gülme, ~664 sn ses) modelden geçer -> `<derlem>/c_audit.jsonl` (`id, wav, category, text, expected,
  got, ok, ipa`). Manifest, `clips.jsonl`, `state.json`, rapor değişmez. `corpus-merge` `"ok": false` klipleri dışarıda
  bırakır (sayı ve kimlikler günlükte, `PREP_REPORT.md`'de ve `prep_report.json` `sources[].c_audit`'te). Ne kapıdan
  geçmiş ne denetlenmiş talimatlı c'li klibi olan kaynağı reddeder (denetimden önceki ilk derlem; `--args
  "--allow-unaudited"` olduğu gibi alır). Birleşmeden sonra bir kaynağın `c_audit.jsonl`'ı değişirse (sha256 ile:
  yeniden denetim boyutu koruyabilir), silinirse ya da yeni çıkarsa eğitim ön kontrolü eğitimi başlatmaz: yeniden
  birleştir. Model kliplerin en az 10'unu ve yarısını okuyamazsa sorun modelde ya da kurulumdadır: denetim hiçbir şey
  yazmaz (eskisi kalır), çıkış kodu 1. Günlük her başarısız klibin okunuşunu ve IPA'sını yazar: birleştirmeden önce
  oku (kalibrasyonda yalnız a/o/u önünde c vardı; ön ünlülü ya da cümle içi c'ler "dʑ"/"ʒ" ile düşüyorsa kontrol
  yanlıştır, `phon_check.DJ`'ye ekle).
- **Prob (`phon_probe.py`, CPU, Modal):** bir WAV klasörü -> her klip için IPA + Whisper large-v3 (sabit revizyon)
  metni, JSON. `--rows` (yarışma `rows.jsonl`'ı ya da `probe_*.jsonl`) verilirse her klibe `c_check` sonucu eklenir.
  8 çekirdek + 12 GiB, 15 dk zaman aşımı (en çok ~$0,12), HF önbelleği Volume'da. Problar kendileri yarışma
  araçlarıyla üretilir (`probe_canina*.jsonl`, `probe_vay_*.jsonl`, `probe_spec.json`).

```powershell
python -m modal run tts/voxcpm/phon_probe.py --src voxcpm-contest/probe-canina2/f022-1500/wav --out voxcpm-contest/probe-canina2/phon.json --rows voxcpm-contest/probe-canina2/f022-1500/rows.jsonl
python -m modal run tts/voxcpm/modal_app.py --action corpus-audit-c --out /data/f022corpus2 --args "--dry-run"  # ikinci derlem: kapı zaten var, yalnız istenirse
```

`cwords`'ü talimatsız üretmek (ikinci derlem; önce kuru çalıştırma: geçiş satırı, 64 metin, tahmin; 2. çağrının
ölçümüyle plan ~$1,0, ama plan dalga başına 212 sn ve 4 dalga sayar, 2. çağrı $0,90-1,00 planlayıp $0,46 tuttu:
beklenen ~$0,45-0,60). Talimatsız oran 40 metinlik ölçümden sonra 0,05'in altında çıkarsa `cwords` yine durur ve hiç
denenmemiş metinler kalır: derlem "incomplete" kalır, aynı komut bir şey eklemez (`PREP_REPORT.md` bunu yazar).
O zaman `corpus-merge`'ü `--allow-incomplete` ile çalıştırın (tutulan klipler girer); `--plain-render none`
çalıştırmayın (talimatlı oran da 0,044'te durur, yeni bir şey üretmez):

```powershell
$env:MODAL_PROFILE="<profiliniz>"
python -m modal run tts/voxcpm/modal_app.py --action corpus-render --out /data/f022corpus2 --max-usd 2 --args "--hours 1.0 --mix long=0.47,neutral=0.47,cwords=0.06 --min-sim 0.6 --laughter-usd 0 --refs-from /data/f022corpus --avoid-pool /data/f022corpus --plain-render cwords --dry-run"
python -m modal run tts/voxcpm/modal_app.py --action corpus-render --out /data/f022corpus2 --max-usd 2 --args "--hours 1.0 --mix long=0.47,neutral=0.47,cwords=0.06 --min-sim 0.6 --laughter-usd 0 --refs-from /data/f022corpus --avoid-pool /data/f022corpus --plain-render cwords"
python -m modal run tts/voxcpm/modal_app.py --action corpus-fetch --out /data/f022corpus2   # PREP_REPORT.md "Render modes", c-check "cwords (plain)"
```

### Türkçe talimatlar (`corpus-merge --instructions`, 2026-10-13)

Dördüncü dinleme turunda (s3-1500 8, f022-1500 6, s3-2000 4, EMA 2) kalan c hatası: İngilizce talimattan sonra
cümle başı c İngilizce okunuyor ("Vay canına" -> "vey kenına", sakin talimatla "Acele" -> "Asele", "Cam" -> "Çam");
taban VoxCPM2 de aynısını yapıyor. Türkçe talimat ("(Şaşkın ve heyecanlı, yükselen tonla.)") c'yi 2/2 doğru okuttu,
ama f022-1500 onu anlamadı (perde nötr gibi). Çözüm: LoRA'ya Türkçe talimatı öğretmek. Ses aynı, klip aynı, yalnız
talimat Türkçe.

- **Tablo `tts/voxcpm/instructions_tr.json`** (İngilizce ifade -> Türkçe ifade, parantezsiz, her Türkçe ifade tek):
  eğitim metinlerindeki her ifade (textbank duyguları: 48 eğitim + 12 ayrılmış ifade, `cwords`, `EMO_REF_INSTRUCTIONS`,
  onaylı gülmelerin 3 ifadesi) ve `benchmarks/tts/sets/contest.jsonl`'ın 14 değerlendirme ifadesi (dinleme turları,
  prob), toplam 77. Testler kapsamı (eksik ya da fazla ifade), biçimi ve tekliği denetler. Gözden geçirildi
  (2026-10-13: 26 ifade düzeltildi, ör. "Usulca mutlu" -> "Hafif bir mutlulukla", "nazik duraklamalarla" -> "ara ara
  usulca duraklayarak", "sönük" -> "durgun", "parlak" yalnız ses için); yine de eğitimden önce bir kez okuyun. Tablo
  değişirse `probe_tr_instr.jsonl`'ın `_tr` talimatları `table[İngilizce]` olarak yeniden yazılır (test bunu ister) ve
  birleşim yeniden yapılır (aşağıda).
- **`corpus-merge --args "--instructions both|tr|en"`** (`render_corpus.py --merge --instructions`): `en` (varsayılan)
  bugünkü gibi; `both` her talimatlı satırı iki kez yazar (İngilizce talimatla, aynı ses + aynı süre + aynı ref_audio
  ile Türkçe talimatla), LoRA Türkçeyi öğrenir, İngilizceyi unutmaz; `tr` yalnız Türkçe kopyayı yazar; talimatsız
  satırlar bir kez. Tabloda olmayan bir İngilizce ifade birleştirmeyi durdurur (sessizce geçmez; kuru çalıştırma ve
  yerel ön kontrol de, Modal çağrısından önce). Rapor dil başına satır sayar (`prep_report.json` `instructions`: mod,
  tablonun sha256'sı, `rows` plain / en / tr; `selected` manifest satırı sayar, `audio` her klibi bir kez). Bir
  birleşim klasörü modunu korur: başka `--instructions` başka `--out` ister (`both` -> `/data/f022mix_both`, `tr` ->
  `/data/f022mix_tr`; ret mesajı adı önerir), böylece bir klasörde eğitilmiş deney başka metinle sürmez. Eğitim ön
  kontrolü: kaynak manifestleri değiştiyse **ya da tablo birleşimden sonra değiştiyse** (sha256) reddeder; yeniden
  birleştirme komutunu `--instructions`'la yazar (aynı `--out` kabul eder; yerel tablo eskiyse önce `git pull`).
  Satırları "rows" ve sesi ayrıca (`audio: N clips, H h`) basar.
- **Beklenen boyut (f022mix'in durduğu hesapta):** /data/f022mix 4274 satır, bunların ~642'si talimatlı (yerel manifestlerde 600:
  553 + 47 cwords; Modal kopyasında +48 gülme, -6 c hatası) -> `both` ~4916 satır (ses yine 4274 klip), satırların
  ~%13'ü Türkçe, ~%26'sı talimatlı (önce ~%15). Talimatlı (duygulu) klipler iki kez görülür: dinlerken duygulu
  kliplerdeki "telefon gibi / cızırtılı" sesin artıp artmadığına bakın; artarsa aynı eğitimi `/data/f022mix_tr` ile
  (aşağıdaki `tr` komutu, yeni `--exp-name`).
- **Eğitim (`--init-lora` ve kısa koşu):** `train_lora.py --init-lora` yalnız `lora_weights.safetensors`'ı yeni deneyin
  `ckpt/latest`'ine kopyalar; resmi eğitici optimizer/scheduler/training_state bulamaz, adım 0'dan **yeni** bir AdamW ve
  yeni bir LR çizelgesiyle başlar: `--warmup` (50) adımda 0'dan `--lr`'ye, sonra kosinüsle `--steps`'te 0'a (çizelgenin
  uzunluğu `--steps`). s3, step 1500'ü kaydettiğinde LR ~1,5e-5'teydi (1e-4 tepe, 2000 adımlık kosinüs); s3-2000
  (aynı veride 500 adım daha, LR ~1,5e-5 -> 0) dinlemede s3-1500'den geri kaldı (4'e 8; 20 cümlede kanıt zayıf ama
  uyarı). Öneri: `--lr 5e-5` (tepe, s3'ün yarısı; Türkçe eşleme yeni bir şey, 1,5e-5 500 adımda az kalabilir),
  `--warmup 50` (yeni AdamW'nin ilk adımlarını yumuşatır), `--steps 500` (her 100 adımda checkpoint), batch 8 x
  grad-accum 2 (s3 gibi): ~1,6 epok, her Türkçe satır ~1,6 kez. Probe 1,0 sn/adım ölçtü: ~8-9 dk eğitim + açılış,
  ~$0,5 (`--max-usd 1` tavan). Step 300 gerçek bir aday (LR o noktada ~2e-5'e inmiştir): prob 300 ve 500'ü
  s3-1500'e karşı ölçer. 500, `_en` satırlarında ya da sesin rengi/doğallığında s3-1500'den kayarsa: yeni bir
  `--exp-name` ile `--lr 3e-5` (aynı komut). Türkçe hâlâ zayıfsa aynı deneyi `--steps` büyütüp sürdürmeyin (çizelge
  500'de 0'a indi, yeni uzunlukla yeniden yükselir): yeni bir `--exp-name` ile `--init-lora
  .../voxcpm2-f022-s3tr/ckpt/step_0000500`.
- **Prob `tts/voxcpm/probe_tr_instr.jsonl`** (20 satır, `<ad>_en` / `<ad>_tr`: aynı metin, aynı duygu, İngilizce ve
  tablonun Türkçe talimatı): c cümleleri canına (4. turun şaşkın cümlesi, değerlendirme ifadesi), acele (4. turun sakin
  cümlesi), cam (`probe_canina2` k7), cuma / cesaret / cidden (`probe_dj`; bu üçünde eğitim ifadesi "Astonished, with
  rising intonation."), duygu başına bir cümle: kızgın (eğitim ifadesi), neşeli, üzgün, empatik (4. turun cümleleri,
  değerlendirme ifadeleri; şaşkın = canına, sakin = acele). Eğitim ifadeleri öğrenilen eşlemeyi, değerlendirme ifadeleri
  genellemeyi ölçer. Cümlelerin hiçbiri eğitim verisinde yok; dosya sızıntı korumasında (`PROBE_GUARD`). `--seeds 0,1`
  her çifti aynı tohumlarla üretir (eşli karşılaştırma). Spec: `tts/voxcpm/probe_tr_spec.json` (s3-1500, s3tr-300,
  s3tr-500).

```powershell
$env:MODAL_PROFILE="<profiliniz>"
git pull
python -m modal run tts/voxcpm/modal_app.py --action corpus-merge --out /data/f022mix_both --args "--instructions both --dry-run"   # dil başına satır, eksik ifade varsa durur
python -m modal run tts/voxcpm/modal_app.py --action corpus-merge --out /data/f022mix_both --args "--instructions both"             # CPU, saniyeler, ~$0,001
python -m modal run --detach tts/voxcpm/modal_app.py --action train --exp-name voxcpm2-f022-s3tr --steps 500 --max-usd 1 --args "--data /data/f022mix_both --init-lora /data/exp/voxcpm2-f022-s3/ckpt/step_0001500 --batch-size 8 --grad-accum 2 --lr 5e-5 --warmup 50"
python -m modal run tts/voxcpm/modal_app.py --action contest-render --run probe-trinstr --backend voxcpm --max-usd 0.8 --spec tts/voxcpm/probe_tr_spec.json --args "--texts tts/voxcpm/probe_tr_instr.jsonl --timesteps 16 --seeds 0,1 --no-score"
python -m modal run tts/voxcpm/modal_app.py --action contest-fetch --run probe-trinstr
# phon_probe giriş başına bir çağrı (aynı adlı WAV'lar tek çağrıda olmaz), her biri en çok ~$0,12
python -m modal run tts/voxcpm/phon_probe.py --src voxcpm-contest/probe-trinstr/f022s3-1500/wav --out voxcpm-contest/probe-trinstr/phon-f022s3-1500.json --rows voxcpm-contest/probe-trinstr/f022s3-1500/rows.jsonl
python -m modal run tts/voxcpm/phon_probe.py --src voxcpm-contest/probe-trinstr/s3tr-300/wav --out voxcpm-contest/probe-trinstr/phon-s3tr-300.json --rows voxcpm-contest/probe-trinstr/s3tr-300/rows.jsonl
python -m modal run tts/voxcpm/phon_probe.py --src voxcpm-contest/probe-trinstr/s3tr-500/wav --out voxcpm-contest/probe-trinstr/phon-s3tr-500.json --rows voxcpm-contest/probe-trinstr/s3tr-500/rows.jsonl
```

Yalnız Türkçe (`both` duygulu sesi bozarsa): kendi klasörü ve kendi deneyi; proba `probe_tr_spec.json`'a
`/data/exp/voxcpm2-f022-s3tronly/ckpt/step_0000500` girişi eklenerek.

```powershell
python -m modal run tts/voxcpm/modal_app.py --action corpus-merge --out /data/f022mix_tr --args "--instructions tr"
python -m modal run --detach tts/voxcpm/modal_app.py --action train --exp-name voxcpm2-f022-s3tronly --steps 500 --max-usd 1 --args "--data /data/f022mix_tr --init-lora /data/exp/voxcpm2-f022-s3/ckpt/step_0001500 --batch-size 8 --grad-accum 2 --lr 5e-5 --warmup 50"
```

Karar ölçüsü (s3tr-300 / s3tr-500, s3-1500'e karşı, `phon-*.json`): `_tr` satırlarında c doğru (`c_ok`) **ve** "Vay"
"vay" okunuyor (IPA'da `a`, "vey" değil) **ve** Whisper metninde (`whisper`) talimatın hiçbir sözcüğü yok (Türkçe
talimat cümleyle aynı dilde: model onu okumamalı) **ve** duygu İngilizcedeki kadar belirgin (kulak + prozodi); `_en`
satırları s3-1500'den kötü değil. Contest-render önce kendi tahminini yazar (120 klip; probe-dj'nin 40 klibi ~2 dk
sürdü).

### Duraklardaki nefes sesi (`pause_clean.py`, 2026-10-13)

4. turda uzun cümleler için: "virgül ve noktalamalardan sonra hep bir nefes sesi var". Ölçüm (faster-whisper medium
kelime zamanları, 80 Hz yüksek geçiren; nefes = durağın kenarlarından 30 ms içeride en az 60 ms, L-50 ile L-24 dB
arası, ağırlık merkezi 0,7-5 kHz gürültü; uzun nefes = en az 0,12 sn ve tepesi L-40 dB'in üstünde; L konuşma düzeyi):
s3-1500'ün 20 uzun ayrılmış cümlesinde noktalama duraklarının %29'unda uzun nefes var (f022-1500 %24, EMA %13,
eğitimdeki uzun klipler %40); en çok cümle sonunda ve ; : sonrasında. Bizim nefeslerimiz kelimeye yapışık değil: kelime
söner, kısa bir sessizlik, 0,1-0,26 sn'lik düz bir gürültü tepesi, yine sessizlik, sonra yeni kelime. EMA'nınkiler
kelimenin kısa, sönen kuyrukları.

`tts/voxcpm/pause_clean.py` bu tepeleri kısar: CPU, numpy + scipy, ASR'siz, belirlenimci; 48 kHz'lik 18 sn'lik klip
0,16 sn.
- Çıkışta yalnız DC atılır (renderlarda RMS'ye göre -34 dB). `--out-hp-hz 80` yüksek geçireni çıkışa da uygular, ama
  konuşmayı değiştirir: s3-1500'de L'nin 14 dB yakınındaki çerçevelerin %7'si 0,1 dB'den fazla oynar (p99 1,4 dB),
  bazı kelime başlarındaki 80 Hz altı darbeler 3-19 dB iner. Bu yüzden varsayılan kapalı. Analiz kopyası 80 Hz'te
  yüksek geçirilir, yoksa 60 Hz altı uğultu durakları doldurur.
- Durak: en az 100 ms, ötümsüz, L'nin 14 dB altında, iki yanında ses olan aralık. Baştaki ve sondaki sessizlik durak
  sayılmaz.
- Durakta dokunulmayan kısımlar:
  - kenarlarda 30 ms;
  - kelimenin durağa sönüşü ve sonraki kelimeye yükselişi (son "ş", "h", sürtünmeli "r", patlama kuyruğu): düzey o
    ana kadarki en düşük düzeyin 6 dB üstüne çıkmadıkça kelimenin sayılır, küçük tümsekler dahil. Nefes, sessizlikte
    dibinden en az 6 dB yükselen bir tepedir. (İlk sürüm yalnız bir önceki çerçeveye bakıyordu; 1 dB'lik ilk tümsekte
    korumayı bırakıp f022-1500'ün "tutar," ve EMA'nın "paylaşılıyor." sonundaki "r"yi 25 dB kısıyordu.)
  - patlamalı ünsüzün patlaması ve aspirasyonu: 2,5 ms'lik bloklarda en az 24 dB sıçrama, tepe 30 ms içinde, 60 ms
    içinde en az 10 dB düşüş ("Sokak", "eksik,", "çözülecek.").
- Kalan bölgede nefes: L-50 ile L-24 dB arasında (tabanın en az 6 dB üstünde), ağırlık merkezi 0,7-5 kHz, en az 40 ms
  süren çerçeveler. Her nefes tek bir kazançla kısılır: tepesi L-55 dB'e iner, kesme en çok 25 dB. 15 ms'lik
  yükseltilmiş kosinüs geçişleri bölgenin içinde kalır. Diğer her örnek girişle aynıdır.

```powershell
python tts/voxcpm/pause_clean.py "voxcpm-eval/voxcpm2-f022-s3/step_0001500@ts16-cfg2/wav/heldout" voxcpm-eval/clean/s3-1500-heldout
```

Çıktıda aynı adlar ve aynı örnek biçimi bulunur, ayrıca `pause_clean.jsonl` (klip başına durak, nefes ve kısılan
bölgeler, kesme dB'i). Python'dan: `pause_clean.clean(wav, sr) -> (ses, rapor)`.

Sonuç (yerel; aynı ölçüm, özgün kliplerin aynı Whisper kelime zamanlarıyla):

| | noktalama durağı | nefes > L-40 dB | uzun nefes (≥ 0,12 sn) | ≥ 1 uzun nefesli klip | kısılan nefes |
|---|---:|---:|---:|---:|---:|
| s3-1500, 20 uzun ayrılmış | 108 | 45 → 0 | 31 (%29) → 0 | 15/20 → 0 | 87 |
| f022-1500 | 103 | 36 → 0 | 25 (%24) → 0 | 13/20 → 0 | 76 |
| EMA | 97 | 34 → 21 | 13 (%13) → 9 | 8/20 → 7 | 21 |
| eğitim: f022corpus2 uzun (16 kHz) | 140 | 65 → 5 | 56 (%40) → 0 | 36/43 → 0 | 137 |

- EMA'da kalanlar kelimeye yapışık kuyruklar: ünlünün formantlarını sürdüren ötümsüz, nefesli bitiş ("mı?",
  "miyim?", "unutma.") ya da sürtünmeli son "r" ("paylaşılıyor."); dipten 6 dB'den az yükselirler, bilerek korunur.
  Ölçüm bunları da nefes sayar, çünkü kelimeye yapışık olup olmadığına bakmaz.
- s3-1500'de kesmenin ortancası 17 dB, 10 nefes 25 dB sınırında.
- **Kelime kaybı yok:**
  - Konuşma çerçeveleri (ötümlü ya da L-14 dB'den yüksek) DC'si atılmış girişle örnek örnek aynı kalır.
  - Kısılan örneklerin en yakın konuşma çerçevesine uzaklığı en az 30 ms (kenar payı); bölge başına en yakın örneğin
    ortancası 45 ms.
  - 87 bölgenin hiçbiri bir konuşma çerçevesi içermez. 60'ı Whisper'ın kelime zamanlarının taştığı durak payında,
    27'si kelimesiz duraklarda.
- **Whisper medium** (CPU, aynı ayarlar, 20 klip):
  - Yalnız DC atılmış hali özgünle 20/20 aynı. Temizlenmiş hal 17/20 aynı. Kuyruk düzeltmesi 3 klibin sesini
    değiştirdi; yazılar ilk temizlenmiş halle 20/20 aynı kaldı.
  - Kalan 3 klipte değişen sözcüklerin sesi bit bit aynı. Fark Whisper'ın 30 sn'lik bağlamdaki yazımı: "412 lira 73"
    yerine "412,73", "Blok Tutar" yerine "Block2tar", "tabi" yerine "tabii".
  - Aynı sözcüklerin çevresindeki 3 sn'lik pencereler iki sürümde aynı yazıyı verir. Bu pencerelerin üçünde kısılmış
    bir nefes de var.
  - WER (referansa göre) %2,04 → %2,85. Artışın tamamı bu yazım farkları.

Dinleme: `voxcpm-contest/r4/nefes/index.html`. 4. turun 4 uzun cümlesi (s3-1500), özgün ve temizlenmiş sürüm rastgele
sırada. Sorular "hangisinde nefes sesi daha az / hangisi daha doğal" (1, 2 ya da fark yok), yorum alanı var ve sonunda
kopyalanacak bir metin çıkar. Anahtar `voxcpm-contest/r4/nefes_anahtar.json`: her cümlenin kendi iki kodu ve
rastgele dosya adları var (bir klibi tanımak ötekileri ele vermez), temizlenmiş sürüm iki cümlede önde, ikisinde
arkada. Kapıdaki düzeltme bu 4 cümlenin seslerini değiştirmedi (bit bit aynı). İki sürüm de özgünün düzey kazancını
alır (DC'si atılmış olarak), yani aradaki tek fark kapıdır. Sonuç (kullanıcı, 4 cümle): temizlenmişte nefes 4/4 daha
az, doğallık 3 aynı / 1 daha iyi -> sentezden sonra kullanılır.

**Çıkarımda (`--pause-clean`, varsayılan kapalı):** `synth_eval.py --pause-clean` ve `voice_contest.py render
--pause-clean` her klibi sentezden hemen sonra `pause_clean.clean`'den geçirir: voice_contest'te düzey
(`refprep.normalize_output`), başlangıç kontrolü ve puanlardan önce (Nano-vLLM'de her deneme, ayrı bir iş parçacığında),
synth_eval'de yazmadan, Whisper'dan ve UTMOS'tan önce (RTF yalnız sentezi sayar). Satırlarda `"pause_clean": {pauses,
breaths, gated_s}`, özette `"pause_clean": {settings, clips, pauses, breaths, gated_s}` (kapalıyken `null`).
voice_contest'te ayar `"pause_clean": "v1/hp80/L-55dB/max25dB"` (`pause_clean.SETTINGS`) yalnız açıkken yazılır: eski
girişlerin anahtarı değişmez, açık ve kapalı klipler aynı giriş klasörüne girmez (başka `--run`; modal_app bunu GPU
başlamadan söyler). Belirlenimci: aynı klip aynı sonucu verir.

```powershell
python -m modal run tts/voxcpm/modal_app.py --action contest-render --run <yeni-run> --spec <spec> --args "--timesteps 16 --pause-clean"
python -m modal run tts/voxcpm/modal_app.py --action eval --ckpt step_0001500 --exp-name voxcpm2-f022-s3 --grid 16:2.0 --args "--sets trzor,heldout,soru --pause-clean"
```

Dikkat: `eval` çıkış klasörünün adı bayrağı içermez (`eval/<ckpt>@ts16-cfg2`); `--pause-clean`'li ölçüm aynı adlı
temizlenmemiş ölçümün yerine geçer (eskisi `<ad>.replaced-<UTC zamanı>` olarak kalır). `--args` yalnız verilen
bayrakları ekler ve synth_eval'in `--sets` varsayılanı `trzor,heldout`'tur: TR-Soru satırları olan bir ölçümün
yerine geçecekse (s3-1500'ün ölçümü gibi; kartın tonlama tablosu ondan gelir) `--sets trzor,heldout,soru` da verin,
yoksa yeni ölçümde tonlama olmaz. Özetteki `pause_clean` hangisi olduğunu söyler; `publish_hf.py` kartına da bunu
yazar. Sıradaki: eğitim verisindeki uzun klipleri temizleyip birleştirmek (yukarıdaki satır: %40 → 0) ve yeniden
eğitmek.

## Sonuçlar (2026-10-09, 16 adım ve UTMOS 2026-10-10)

Aynı düzenek, aynı yargıç: faster-whisper large-v3-turbo (A100, float16, beam 5), trnorm ile normalize
edilmiş referans ve hipotez. TR-Zor: 155 zor cümle (metin trnorm'dan geçerek modele verilir). "Ayrılmış":
Antalia'nın eğitimde hiç kullanılmamış 20 cümlesi. WER %, düşük daha iyi.

| Model | Veri | TR-Zor WER | Hatasız TR-Zor satırı | Ayrılmış WER | UTMOS TR-Zor / ayrılmış | RTF (A100) |
|---|---|---:|---:|---:|---:|---:|
| VoxCPM2, ince ayarsız (Antalia klibini klonlayarak) | — | 5.96 | — | 3.80 | — | — |
| **Bizim: VoxCPM2 + Antalia LoRA, adım 600** | 4.8 sa, açık (CC-BY) | **4.20** | 131/155 | **1.90** | 3.38 / 3.29 | 0.87 |
| **Bizim, adım 600, 16 adım çıkarım** (yayınlanan ayar) | 4.8 sa, açık | **3.87** | 133/155 | 1.49 | 3.45 / 3.45 | 2.29 |
| **f022 tasarlanmış ses (Alania aşama 1 → f022 3.5 sa), adım 1500, 16 adım** (2026-10-12, yayında) | 25 sa + 3.5 sa sentetik, açık | **3.09** | **140/155** | **0.82** | 3.71 / 3.12 | 1.26 |
| **İki aşamalı (Alania 25 sa → Antalia), adım 600, 16 adım** (2026-10-10) | 25 sa sentetik + 4.8 sa, açık | **3.42** | **137/155** | **1.36** | 3.57 / 3.40 | 1.66 |
| İki aşamalı, adım 1000, 16 adım | | 4.20 | 136/155 | 2.58 | 3.61 / 3.48 | 1.46 |
| Gerçek Antalia kaydı (ayrılmış 20 cümle, 16 kHz) | | | | | — / 3.44 | |
| Bizim, adım 200 / 400 / 1000 | | 5.86 / 4.64 / 4.31 | | 2.17 / 2.45 / 1.90 | — | |
| Trendyol-TTS (VoxCPM2 + LoRA, 16 adım çıkarım) | ~20 sa, gizli | 3.98 | 134/155 | 0.68 | **3.82 / 3.73** | 1.37 |

- İnce ayar, eğitimsiz VoxCPM2'ye göre TR-Zor'da %5.96 → %4.20, ayrılmış cümlelerde %3.80 → %1.90.
  En iyi kayıt 600. adım; sonrası iyileştirmiyor.
- Trendyol-TTS ile TR-Zor'da başa baş (155 cümlenin 131'i ve 134'ü hatasız). Ayrılmış 20 cümlede fark 1
  cümle (15 ve 16 hatasız). Veri 4 kat az ve tamamen açık.
- Kategori bazında (TR-Zor WER %, bizim / Trendyol): kısaltma-harf okuma (akronim) 10.3 / 13.8, tek
  kelimelik kısa cevaplar 16.0 / 20.0, numara 7.6 / 6.7, saat 4.8 / 1.6, para 3.8 / 1.3, tarih 0.9 / 0,
  sıra sayısı, soru: ikisi de 0. Kısa cevaplar kategorisinde toplam ~25 kelime var; 15 cümlenin 2'sindeki (bizim) ve 3'ündeki (Trendyol) hata yüzdeyi büyütüyor.
- Doğallık (UTMOS22 strong, 1-5, yüksek iyi; bir modelin tahmini, insan puanı değil): 2026-10-10'da mevcut
  wav'lar üzerinde yerelde CPU'da ölçüldü (`score_utmos.py`). Trendyol için bulduğumuz 3.82, Trendyol'un
  yayımladığı 3.83 ile uyuşuyor. Bizimki ~0.44 düşük ve 13 kategorinin hepsinde geride (fark kısa cevapta
  0.25, saatte 0.63). Ama Trendyol bu ölçümde kendi sesiyle konuşuyor: farkın bir kısmı modelden değil ses ve
  kayıt farkından gelebilir. Antalia'nın gerçek kayıtları 3.44 alıyor: bu sesin tavanı bu. 16 adım çıkarımla
  bizim model ayrılmış cümlelerde 3.45'e çıktı, yani doğallıkta sesin tavanında; Trendyol farkı sesten geliyor.
- 16 adım çıkarım (Trendyol'un ayarı) TR-Zor'u %4.20 → %3.87'ye indirdi (Trendyol 3.98), ama A100'de RTF 2.29
  (gerçek zamandan yavaş; Trendyol aynı ayarla başka bir konteynerde 1.37 ölçüldü, konteynerler arası fark büyük).
  Hız gerekiyorsa 10 adım. cfg 2.5 bütçe yüzünden ölçülmedi. Bu tarama ~$1.44.
- Maliyet: deneme 10 dk + eğitim 36 dk + değerlendirme, A100 80 GB, toplam ~$6.

## Yerelde doğrulama (CPU, GPU'suz)

Bu makinede (4 çekirdek, 15 GB RAM) uçtan uca denendi: veri → manifest → resmi eğitim betiği → süre
sınırında SIGTERM ile kaydedip durma (adım 5) → kaldığı yerden devam (adım 5 → 8) → LoRA'yı yükleyip
sentez → Whisper ile WER. 2B'lik VoxCPM2'yi CPU'da float32 eğitmeye 15 GB RAM yetmediği için eğitim
döngüsü aynı betik ve aynı sınıflarla küçük kardeş **VoxCPM1.5** (750M) üzerinde koşturuldu. VoxCPM2'nin
veri hattı (16 kHz, ref_audio) ileri geçişe kadar denendi. GPU'da eğitim bf16 autocast ile çalışıyor.

## Kaggle'da denendi: sığmıyor

Kaggle'ın ücretsiz kotasındaki 2× T4 (16 GB) ile ölçüldü (2026-10-08). Kurulum, veri hazırlığı ve model
indirme ~5 dakikada tamamlandı. Ama eğitim **1'lik pakette bile** ilk adımda bellek yetmedi (14.4 GiB):
resmi eğitim betiği 2B'lik temel modeli eğitimde float32 tutuyor. Bu yüzden VoxCPM2 eğitimi Modal'da
(A100 80 GB) yapılıyor. Kaggle'ın kotası STT eğitimine ayrıldı. Kaggle'ın ücretsiz kotasında L4 sunulmuyor:
`--accelerator NvidiaL4` ile açılan notebook CPU'da çalıştı.

## Lisans ve etik

- **Ahenk (yayınlanan, f022):** ağırlıklar CC BY 4.0 (Alania verilerinden türedi: "Alania Turkish Synthetic
  Speech by PatientDesk AI" ve "Alania Turkish Domain Text by PatientDesk AI", CC BY 4.0; yalnız `cc-by` config'i);
  temel model VoxCPM2 Apache-2.0. Ses tasarlanmıştır, gerçek bir kişinin sesi değildir. Tam atıf model kartında
  (`MODEL_CARD.f022s3.template.md`).
- **Eski Antalia denemeleri:** VoxCPM2'nin Apache-2.0 lisansı + Antalia verisinin CC BY 4.0 atıf şartı ("Antalia
  (Patientdesk.ai)"). Antalia sesi gerçek ve izin vermiş bir sanatçıdan türetilir; sanatçıyı ya da herhangi bir gerçek
  kişiyi **taklit etmek için kullanılmamalı**.
- Her durumda üretilen sesin **yapay olduğu belirtilmeli** (AB Yapay Zekâ Yasası md. 50).
