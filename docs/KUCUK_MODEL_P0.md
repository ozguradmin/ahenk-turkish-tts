# Küçük model P0: yerel ölçüm ve veri hazırlığı (2026-10-14)

Her şey kendi bilgisayarında yapıldı: NVIDIA RTX 5060 Laptop GPU (8 GB) ve AMD Ryzen AI 7 350 işlemci. **Harcama $0.** Modal, Google, Gemini, Kaggle ya da başka ücretli bir hizmet kullanılmadı. Repoda var olan hiçbir dosya değişmedi; yeni kod `tts/small/` altında. Commit atılmadı.

## Kısa özet (Türkçe)

- **Kurulum çalışıyor.**
  - Ayrı bir Python 3.12 ortamı kuruldu, PyTorch 2.11 + CUDA 12.8 ile. Ekran kartı tanınıyor.
  - Bir sorun çıktı: Windows Uygulama Denetimi, `av` paketinin bir DLL'ini engelledi. Bu paket Whisper'ın ses dosyası açma parçası. Güvenlik ayarına dokunulmadı; o parça hiç yüklenmeden sesler doğrudan veriliyor.
- **EMA ile Antalia-2 Mini aynı seviyede.** 60 cümlede ölçtük (40 TR-Zor + 20 ayrılmış):
  - Hata oranı: EMA %1,68, Antalia %1,57. 60 cümlede bu fark anlamsız.
  - Doğallık (UTMOS): EMA biraz önde, 3,35'e karşı 3,22.
  - Hız: Antalia işlemcide ~1,5 kat hızlı (2 çekirdekte RTF 0,069; EMA 0,104). İkisi de gerçek zamanın çok altında.
- **Büyük modelin ses kodu ile EMA'nın ses çözücüsü aynı dili konuşuyor; bu yeniden doğrulandı.**
  - Yöntem: 12 f022 klibi VoxCPM2 ile koda çevrildi, EMA'nın çözücüsüyle sese geri döndürüldü.
  - Sonuç: Whisper 12 klibin 12'sini hatasız anlıyor.
  - Bedeli doğallıkta: UTMOS 3,59'dan 3,20'ye düşüyor. VoxCPM2'nin kendi çözücüsüyle 3,56.
  - Yani küçük modelin doğallık tavanını EMA'nın çözücüsü belirliyor. Plandaki "UTMOS ≥ 3,3" geçme şartı bu çözücüyle zor. Ya şart bu tavana göre konmalı, ya da çözücü f022'ye alıştırılmalı. Bu iş artık yerelde, bedava yapılabilir.
- **Eğitim verisi hazır:** `data/small/f022-latents/`.
  - 4.274 klip, 4,58 saat. Talimatı hatalı 6 klip atıldı.
  - Varsayılan eğitim kümesinde 4.225 klip / 4,55 saat var; 48 gülme klibi ayrı tutuldu.
  - Metin trnorm okunuş biçiminde ve EMA'nın alfabesinde. Duygu talimatı metinden silindi; duygu bir "stil numarası" olarak saklandı.
  - Kelime süreleri 4.274 klibin 4.273'ünde hizalandı.
  - Kodlama ekran kartında 7,5 dakika sürdü; en fazla 2,7 GB bellek kullandı.
- **Sıradaki adım: eğitim kodu.** Kısa bir hız denemesi yaptık (80 adım, sonuç kaydedilmedi).
  - Bir adım ~0,06 sn sürüyor ve 2,6-3,2 GB bellek kullanıyor.
  - EMA ağırlıklarından başlayan 20 bin adımlık bir deneme yaklaşık 20-30 dakika sürer.
  - Sıfırdan başlayan kontrol deneyi (80 bin adım) yaklaşık 1,5-2 saat sürer.
  - Ölçümlerle birlikte iki deneme toplam ~3-4 saat ekran kartı ister, $0.

---

## 1. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB, compute capability 12.0 (sm_120), driver 616.64 |
| CPU | AMD Ryzen AI 7 350, 8 cores / 16 threads |
| venv | `<venv-student>`, a separate virtual environment outside the repo (Python 3.12.10; system Python untouched; 4.8 GB) |
| torch | 2.11.0+cu128 (download.pytorch.org/whl/cu128), arch list includes sm_120; torchaudio 2.11.0+cu128 |
| GPU check | `torch.cuda.is_available()` True; 4096² matmul 11.0 TFLOPS fp32, 34.8 TFLOPS fp16 |
| Other (PyPI) | faster-whisper 1.2.1, ctranslate2 4.8.3 (CUDA via torch's bundled cuBLAS/cuDNN DLLs), ema-lightning 1.1.2 (`--no-deps`), normalizer-tr 0.7.0, numpy 2.5.4, soundfile, soxr, scipy, rapidfuzz, huggingface_hub 2.2.0, safetensors, cmudict |
| Downloads (huggingface.co) | `cloud0day3/antalia-mini@8a79528` (config, fp32 safetensors, 1.0.1 wheel, README/LICENSE/NOTICE; 31 MB); `mobiuslabsgmbh/faster-whisper-large-v3-turbo@0a363e9` (MIT, 1.6 GB, the judge used by `benchmarks/tts/judge.py` and `synth_eval.py`) |
| Cached, reused | EMA Lightning `7a6ba1a`, VoxCPM2 `bffb3df` (`audiovae.pth`), UTMOS22 strong (torch hub cache, loaded with `source="local"`, no network) |
| Full freeze | `venv-student-freeze.txt` (local, not in the repo) |

**Problems found:**
- **Windows App Control blocked PyAV's DLL.** `ImportError: DLL load failed while importing opaque` ("Uygulama Denetimi ilkesi bu dosyayı engelledi").
  - faster-whisper imports `av` only to decode files. We always pass 16 kHz numpy arrays, so `p0common.no_pyav()` registers a placeholder module, and the blocked DLL is never loaded.
  - No security setting was touched.
  - `regex` was not needed, because we don't use transformers.
- **The Antalia wheel's code was read before installing it.**
  - It loads the release weights with safetensors only. Its `torch.load(weights_only=False)` is reached only by its training-checkpoint helpers, which we don't call.
  - Its only network code is the weight download, and that is bypassed by passing the local snapshot directory.
  - It was installed with `--no-deps`.
- **EMA's first GPU call took 57 s.** This is a one-time batch-size calibration, cached in `~/.cache/ema_lightning/batch_size_v2.json`. The timings exclude it by warming up.

## 2. Benchmark: EMA Lightning vs Antalia-2 Mini

**Script:** `tts/small/bench.py`. **Outputs:** `voxcpm-contest/small-bench/`, containing `subset.jsonl`, `rows.jsonl`, `summary.json`, `SUMMARY.md`, and `<system>/<set>/<id>.wav` at 48 kHz.

**Subset (60 rows)**
- 40 TR-Zor rows, stratified by category. Each category gets a largest-remainder share of 40/155, and rows are drawn inside each category with `random.Random(0)`.
- The 20 held-out sentences, taken from `voxcpm-eval/voxcpm2-f022-s3tr/step_0000500@ts16-cfg2/rows.jsonl`, with the references used there.

**Systems** (seed 0, package defaults)
- `ema`: EMA Lightning with the trnorm front-end. This is exactly trtts, with normalizer-tr bypassed.
- `antalia`: Antalia-2 Mini 1.0.1 using its own built-in normaliser on the raw text.
- `antalia_trnorm`: Antalia with trnorm's spoken form; its own normaliser is replaced by lowercasing plus its symbol filter.

**Judge and scoring**
- faster-whisper large-v3-turbo, float16 on the GPU, language `tr`, beam 5. These are our eval settings; the CPU judge in `benchmarks/tts` uses int8.
- WER/CER are computed after `trnorm.normalize_for_eval`, taking the best reference per row, the same as `benchmarks/tts/score.py`.
- UTMOS22 strong is computed at 16 kHz.
- RTF is synthesis wall time divided by audio length, per sentence call, after a warm-up. The GPU path is plain (no CUDA graphs). The CPU path uses `torch.set_num_threads(N)`.

| System | TR-Zor-40 WER / CER | rows ok | Held-out-20 WER / CER | rows ok | All 60 WER | UTMOS TR-Zor / held-out | GPU RTF | CPU RTF 2 thr | CPU RTF 8 thr | GPU peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ema** (trnorm) | 2.34 / 0.46 | 37/40 | 1.49 / 0.42 | 16/20 | 1.68 | **3.35** / 2.95 | 0.018 | 0.104 | 0.092 | 155 MiB |
| **antalia** (native) | 2.76 / 0.68 | 37/40 | **1.22 / 0.24** | 16/20 | **1.57** | 3.22 / 2.94 | 0.022 | **0.069** | **0.052** | 146 MiB |
| antalia + trnorm | 3.24 / 0.98 | 36/40 | 1.22 / 0.24 | 16/20 | 1.68 | 3.21 / 2.94 | 0.018 | — | — | 146 MiB |

**Reading the table**
- **Accuracy is a tie.** The whole 60-row difference is one word: 16 vs 15 word edits.
- **Errors are shared.** Both systems fail `kisa_08` ("hoşgeldiniz", a Whisper spelling) and the long held-out "uyar mı" → "uyarma".
- **EMA-specific errors**
  - "emar" → "em ere" (akronim)
  - "siparış" (a vowel)
  - a Whisper hallucination, "altyazı m k", on trailing silence in `heldout_246f…`. This is a judge artefact, not the TTS.
- **Antalia-specific errors**
  - Its native normaliser reads "584213" grouped, as "beş yüz seksen dört…". That is valid as an alternative, but Whisper heard "ila iki bin üç".
  - "güzün" → "yüzün".
- **trnorm in front of Antalia does not help.** Its built-in normaliser is already good, and Antalia was trained on its own normaliser's output style.
- **EMA is consistent with the full 155-row result** (1.9 % with trnorm, `benchmarks/tts/results/tr_zor/RESULTS.md`). 40 rows is ±1 sentence noise.
- **Naturalness:** EMA is slightly higher on short sentences (3.35 vs 3.22) and equal on the long held-out lines (2.95 vs 2.94). Both are well below our big model (s3tr-500: 3.70 / 3.14).
- **Speed:** Antalia is ~1.5× faster on CPU and speaks ~13 % slower (default speed 0.95), with 419 s of audio vs 369 s for the same text.
  - EMA's CPU RTF at 2 threads is **0.104 on this laptop**. The plan quoted 0.18 with an unknown thread count.
  - Antalia at 8 threads (0.052) happens to equal the plan's unverified "0.052 on 8 EPYC cores".
- **Conclusion:** Antalia-2 Mini is a peer, not a ceiling to chase. A student that matches EMA in accuracy and keeps the f022 voice is a valid target. On-device speed favours keeping the DiT small and few-step (Stage 3).

## 3. Latent pipeline feasibility

### 3a. Check (re-verified on f022, GPU)

**Script:** `tts/small/latent_check.py`. **Outputs:** `voxcpm-contest/small-bench/latent-check/` (`rows.jsonl`, `summary.json`, wavs).

**Method**
- 12 f022 clips: 3 neutral, 2 questions, 3 emotions, 2 long, 2 cwords.
- Encoded at 16 kHz with the cached VoxCPM2 `audiovae.pth` encoder (mu, raw scale).
- Decoded three ways: EMA `decoder.pt`, the VoxCPM2 AudioVAE decoder, and EMA's decoder on time-shuffled latents as a floor.

| Path | WER | CER | rows ok | UTMOS | log-mel L1 | log-mel corr |
|---|---:|---:|---:|---:|---:|---:|
| original 16 kHz wav | 0.00 | 0.00 | 12/12 | 3.59 | 0 | 1.000 |
| VoxCPM2 AudioVAE decoder | 0.00 | 0.00 | 12/12 | 3.56 | 0.269 | 0.995 |
| **EMA decoder.pt** | **0.00** | **0.00** | **12/12** | **3.20** | 0.856 | 0.959 |
| EMA decoder, shuffled latents | 100 | 90.6 | 0/12 | 1.25 | 3.787 | 0.167 |

- **The latent space is confirmed.** EMA's decoder reads VoxCPM2 AudioVAE latents without scaling, and the result is fully intelligible. This was verified locally; EMA's model card does not document it.
- **The decoder caps naturalness.** It costs about 0.39 UTMOS on f022 (3.59 → 3.20), while the VoxCPM2 decoder costs 0.03.
  - So the go-gate "UTMOS ≥ 3.3 with `decoder.pt`" is very likely unreachable, even with perfect latents.
  - **Recommendation:** gate the student on UTMOS relative to `decoder.pt(ground-truth latents)` = 3.20, for example ≥ 3.1. Then decide on the Stage-4 decoder fine-tune separately. It can now be done locally: 3M parameters, f022 latent/48 kHz pairs, also $0.
- **GPU vs CPU encoder.** With the default TF32 convolutions, the GPU latents differ from the CPU ones by up to 0.043 (mean 0.001; latent std 1.47). With TF32 off, the gap is 2e-5. The corpus encoder therefore turns TF32 off.
- **Right-padding is exact** (2e-5), because the encoder is causal. That makes padded batches safe.

### 3b. f022 corpus → training-ready latents

**Script:** `tts/small/encode_corpus.py`. **Output:** `data/small/f022-latents/` (gitignored `data/`, 56 MB).

**Inputs**
- `<xfer>/f022corpus` and `<xfer>/f022corpus2` (the two f022 corpora copied from the Modal Volume to a local folder).
- The manifests' `/data/<corpus>/wav/x.wav` paths are mapped to `<folder>/wav/x.wav`.
- Rows with `ok=false` in `f022corpus/c_audit.jsonl` are skipped: 6 clips, the same ones `corpus-merge` dropped. The result is exactly the `/f022mix` set: 4,274 clips, 4.58 h.

**Format**
- `latents.npy`: float16, `[412,489 frames, 64]`, 25 Hz. Clip *i* is `latents[offset : offset+frames]`.
- `index.jsonl`: one row per clip, with these fields:
  - `id`, `corpus`, `category`
  - `style` / `style_id`: neutral 0, angry 1, calm 2, cheerful 3, empathetic 4, sad 5, surprised 6, laughter 7
  - `text`: what the teacher read, i.e. `render_spoken` (e.g. "Vaay") or the trnorm `spoken` form, with the "(instruction)" prefix stripped
  - `ema_text`: EMA's 49-symbol alphabet, lowercase, â→a
  - `ids`, `offset`, `frames`, `duration`
  - `word_frames`: per EMA word, summing to `frames`
  - `align`, `words_src`, `words` (Whisper)
  - `flags`, `split` (train/val by sha1 of id, 2 %), `use` (default training set)
  - `instruction`: reference only; the student never sees it
  - `utmos`, `sim`, `wav`
- `stats.json` holds the numbers below plus a format description.

**Durations**
- Source: Whisper word timings. These come from `clips.jsonl`, or from local GPU Whisper word timestamps for the 312 clips without them (84 s).
- Matching: timings are matched to EMA words as whole words. Each Whisper word is run through `trnorm.normalize_for_eval`, and apostrophe/hyphen pieces split by Whisper are glued back together.
- Unmatched EMA words get start times interpolated by letter position. These are mostly numbers, whose per-word reading differs from the in-context one.
- Silence between words goes to the preceding word, and every word gets ≥ 1 frame.
- Alignment results:
  - 3,814 clips (89 %) match exactly.
  - 460 match through anchors. Inside numbers the split is approximate, for example "on iki bin" from "12 .000".
  - 1 clip fails (`q_duz_123`).
  - The mean share of EMA words matched is 98.6 %.
  - Every non-laughter clip with stored timings aligned.

| | Clips | Hours |
|---|---:|---:|
| All | 4,274 | 4.583 |
| **Default training set (`use`)** | **4,225** | **4.545** |
| of which train / val | 4,164 / 61 | 4.481 / 0.064 |
| Excluded: laughter (style 7, kept in the index) | 48 | |
| Excluded: unaligned | 1 | |

| Style (use) | neutral | surprised | empathetic | cheerful | angry | calm | sad |
|---|---:|---:|---:|---:|---:|---:|---:|
| clips | 3,631 | 132 | 127 | 106 | 94 | 86 | 49 |
| hours | 3.98 | 0.10 | 0.13 | 0.09 | 0.09 | 0.10 | 0.05 |

**Categories (all clips):** neutral 2,702, questions 501, emotions 547, interjections 164, short 145, long 120, cwords 47, laughter 48.

**Soft flags** (rows kept, for the trainer to decide)
- 132 clips are over 10 s (max 18.1 s / 454 frames, under EMA's 3,000-frame cap).
- 8 clips have no final `.`/`!`/`?`.
- 49 clips: local Whisper CER > 0. 43 of these are laughter (already excluded). 5 are interjection elongations ("Öff" heard as "Öf"), and 1 is the unaligned question.

**Statistics**
- Text: 62 letters per clip on average; 1.55 frames per letter.
- Latents: mean −0.027, std 1.47; per-dimension std ranges from 0.99 to 1.93.

**Cost**
- GPU encode: 444 s for 4.58 h of audio (≈37× real time) in fp32 with TF32 off, batches of 60 s of audio.
- Peak 2.7 GB. A 160 s batch ran out of memory: the encoder needs ~55 MB per audio second.
- With TF32 on it would be faster but less exact. bf16 was not tried.
- Load and resample: 1 s, because the wavs are already 16 kHz.

## 4. Next step: trainer for EMA's DiT on f022 latents (8 GB GPU)

**Throughput probe**
- Script: `tts/small/train_probe.py`, 80 AdamW steps at batch 32, nothing saved.
- It runs on real batches from the index, with the losses below. Speed:

| Init | Precision | s/step (mean) | Peak GPU | flow loss (first 5 steps → last 10) | duration loss |
|---|---|---:|---:|---:|---:|
| `ema.pt` | fp32 | 0.055 | 2.6 GB | 1.97 → 1.22 | 0.061 → 0.043 |
| `ema.pt` | bf16 autocast | ~0.06 (one warm-up outlier makes the mean 0.146) | 3.2 GB | 1.92 → 1.24 | 0.061 → 0.043 |
| scratch | bf16 | 0.065 | 3.2 GB | 3.45 → 2.69 | 0.95 → 0.05 |

- **`ema.pt` is a strong init for this data.**
  - Its duration predictor already fits the f022 word durations: log-error² is 0.06 vs 0.95 from scratch.
  - Its DiT starts far below scratch on the flow loss (1.9 vs 3.45; the noise-only level is ≈ 3.2).
  - This also confirms that the text/timeline/latent format plugs into EMA's model unchanged.
- **fp32 is as fast as bf16 at this size.** Use fp32 with TF32.

**Trainer design** (`tts/small/train.py`, to be written next with unit tests)
1. **Data**
   - `index.jsonl` (`use` and `train`) plus memory-mapped `latents.npy`.
   - Length-bucketed batches with a frame budget of about 32 × 100 frames.
   - A DataLoader with workers prefetching on CPU. The probe built batches outside the timed step.
2. **Model**
   - `ema_lightning.model.Acoustic` (Apache-2.0), initialised from `ema.pt["ema"]`. Keep the 49-symbol vocabulary, so the existing ONNX/Core ML ports can stay compatible.
   - Add a style embedding (8 × 224, zero-initialised) to the text features.
   - Style dropout to neutral, p = 0.1.
3. **Timeline (teacher forcing)**
   - `fw`/`fp` come from the target `word_frames`.
   - `cp` comes from the detached predicted letter durations, exactly as `sound_stage` does at inference. This is already implemented in `train_probe.batch_tensors`/`losses`.
4. **Losses**
   - OT-CFM velocity MSE on masked frames: x_t = (1−t)x0 + t·x1, target x1 − x0. This is the parametrisation EMA's sampler uses (x1 = x_t + (1−t)v).
   - Plus L2 on log(1 + word frames).
   - Text-condition dropout p = 0.1 (aligner output zeroed) for CFG.
   - Optionally sample t from a logit-normal.
5. **Optimisation**
   - AdamW, lr 1e-4 (5e-5 if the init drifts), warm-up 500, cosine schedule.
   - Clip gradients at 1.0. Keep an EMA of the weights at 0.999.
   - Checkpoint every 2k steps.
6. **Evaluation per checkpoint (~2 min on GPU)**
   - Euler/midpoint sampler, 8–16 steps, CFG 1–2.
   - Decode with `decoder.pt`, and also with the VoxCPM2 decoder as the upper bound.
   - Metrics: `bench.py` subset WER/CER, UTMOS, and val flow loss.
   - Then the full TR-Zor 155 + held-out 20 + TR-Soru 35, plus ECAPA similarity to the f022 reference. `speechbrain/spkrec-ecapa-voxceleb` is cached.
7. **Unit tests**
   - Batch builder: `word_frames` sums, masks, cw/wstart identical to `engine.piece`.
   - The sampler with EMA's own 4 times reproduces `sound_stage`.
   - A one-batch overfit test.

**Expected GPU time on the RTX 5060**

| Run | Steps | Training time | Notes |
|---|---:|---:|---|
| Arm (a), init `ema.pt` | 10–20k | ~10–20 min | One epoch is 131 batches (≈ 8 s), so 20k steps ≈ 150 epochs of 4.5 h. Watch val loss and checkpoint WER for overfitting. |
| Arm (b), scratch control | 50–100k | ~50–100 min | Sizes how much data a scratch model needs; expected to be clearly worse on 4.5 h. |
| Evaluation | ~6 checkpoints × 2 arms | ~30 min | Subset per checkpoint, then the full sets for the best checkpoint. |
| Optional Stage 4: `decoder.pt` fine-tune on f022 | 20–50k GAN steps | ~2–4 h (estimate; HiFi-GAN-style discriminators, MIT) | Lifts the 3.20 UTMOS ceiling; needs 48 kHz targets. f022 wavs are 16 kHz, so use VoxCPM2-decoded 48 kHz audio of the same latents as the target. |

- **Total for P0 (a) + (b) + evaluation:** ≈ 2–3 GPU-hours at $0.
- Each run fits in 30-minute chunks with resume, so no long unattended job is needed.
- **Go/stop gates (corrected):**
  - TR-Zor ≤ ~3 %.
  - ECAPA ≥ 0.70 to f022.
  - UTMOS within ~0.1 of the 3.20 `decoder.pt` ceiling.
  - Listening says it is f022.
- Few-step re-distillation (Stage 3) is still needed for phone-speed RTF.

## 5. Files

**Code (new, uncommitted)**

| File | Purpose |
|---|---|
| `tts/small/p0common.py` | Shared helpers: judge, scoring, EMA alphabet, paths, PyAV placeholder |
| `tts/small/audiovae.py` | VoxCPM2 AudioVAE v2 encoder/decoder, Apache-2.0, copied |
| `tts/small/bench.py` | The EMA vs Antalia benchmark |
| `tts/small/latent_check.py` | The latent pipeline check |
| `tts/small/encode_corpus.py` | Corpus encoding into training-ready latents |
| `tts/small/train_probe.py` | The throughput probe |

**Outputs**
- `voxcpm-contest/small-bench/`: benchmark renders, `SUMMARY.md`, `summary.json`, `rows.jsonl`, `subset.jsonl`; `latent-check/` (gitignored).
- `data/small/f022-latents/`: `latents.npy`, `index.jsonl`, `stats.json` (gitignored).
- Logs: `bench.log`, `encode.log` (local, not in the repo).

**How to run**

```powershell
$py = "<venv-student>/Scripts/python.exe"
$env:HF_HUB_OFFLINE = "1"
& $py tts/small/bench.py --out voxcpm-contest/small-bench
& $py tts/small/latent_check.py --corpus <xfer>/f022corpus --corpus2 <xfer>/f022corpus2
& $py tts/small/encode_corpus.py --corpus <xfer>/f022corpus <xfer>/f022corpus2 --out data/small/f022-latents --batch-seconds 60
& $py tts/small/train_probe.py --steps 80 --batch 32 --amp fp32
```
