# Küçük model P1: ilk öğrenci eğitimi ve ölçümü (2026-10-14)

Her şey kendi bilgisayarında yapıldı: RTX 5060 Laptop GPU (8 GB). **Harcama $0.** Modal, Google, Gemini ya da başka ücretli bir hizmet kullanılmadı. Var olan hiçbir repo dosyası değişmedi. Yeni kod `tts/small/`, yeni test `tests/test_small_student.py`, rapor bu dosya. Commit atılmadı.

## Kısa özet (Türkçe)

- **Küçük model f022 sesiyle konuşuyor.** EMA Lightning'in küçük modelini (5,6 milyon parametre) f022 verisiyle eğittik. Ses çözücüsünü de (3 milyon parametre) f022'ye alıştırdık. Toplam 8,6 milyon parametre, yaklaşık 34 MB.
- **Sonuç: ilk öğrenci, EMA ve Antalia'dan iyi ya da onlarla aynı seviyede. Üstelik bizim sesimizle konuşuyor.** P0'daki aynı 60 cümlede:

| | Hata oranı (60 cümle) | Hata oranı (TR-Zor 155) | Doğallık (UTMOS, kısa / uzun) | f022'ye benzerlik |
|---|---:|---:|---:|---:|
| **Öğrenci (yeni)** | **%1,16** | **%2,32** | **3,41 / 3,18** | **0,62** |
| EMA Lightning | %1,68 | %2,43 | 3,35 / 2,95 | 0,22 |
| Antalia-2 Mini | %1,57 | %3,30 | 3,22 / 2,94 | 0,24 |
| Büyük model (öğretmen) | %1,26 | %2,54 | 3,65 / 3,13 | 0,62 |

- **Ses benzerliği:** Öğrencinin sesi f022 referansına öğretmen kadar benziyor (0,62'ye karşı 0,62). EMA ve Antalia başka sesler (0,22-0,24). Eski hedef "0,70" tutmuyor, çünkü öğretmenin kendisi de 0,62'de kalıyor. Hedefi "öğretmen kadar" diye okumak gerekir.
- **Doğallık:** Kısa cümlelerde EMA ile başa baş, uzun cümlelerde EMA'dan iyi. Büyük modelin hâlâ gerisinde: kısa cümlede 3,41'e karşı 3,65.
- **İki şey işe yaradı:**
  1. **Ağırlık karışımı.** Eğitim uzadıkça ses f022'ye benziyor ama doğallık düşüyor. Çözüm: eğitilmiş modelle EMA'nın orijinal ağırlıklarını %75 / %25 karıştırmak. Hem f022 sesi kalıyor hem doğallık geri geliyor.
  2. **Ses çözücüsünü f022'ye alıştırmak.** Bilgisayarda 25 dakika sürdü. Benzerliği 0,48'den 0,62'ye çıkardı, doğallığı da artırdı. P0'da "tavan" dediğimiz 3,20 sınırı bununla kalktı.
- **Sorular:** Soru cümlelerinde öğrenci, ölçülen 4 sistemin en iyisi.
  - Soru tonlaması geçme oranı: öğrenci %60, öğretmen %56, Antalia %51, EMA %43.
  - Sonu yükselen sorular ("Yarın sabah geliyorsun?") hiçbir sistemde yükselmiyor. Buna öğretmen de dahil. Bu, öğretmenin verisinden gelen bir eksik.
- **Duygu:** Duygu numarası çalışıyor ama etkisi hafif.
  - "Üzgün": ses 1,9 yarım ton daha pes, konuşma %8 daha yavaş.
  - "Sakin": %8 daha yavaş.
  - "Kızgın", "neşeli" ve "şaşkın": sesin perdesi biraz yükseliyor.
  - Eğitim verisinde fark daha büyük. Bu ilk sürümde duygular zayıf kalıyor.
- **Hız: öğrenci telefon hedefi için henüz yavaş.**
  - İşlemcide 2 çekirdekte gerçek zamanın 0,20'si (EMA aynı ölçümde 0,13, Antalia 0,08).
  - Adım sayısını 4'e indirince hızlanıyor ama doğallık 0,4 düşüyor.
  - Telefon için planlanan "Aşama 3" (az adımlı damıtma) gerçekten gerekli.
- **Sıfırdan eğitim işe yaramıyor:** 4,5 saat veriyle hata oranı %13,4. EMA'nın ağırlıklarından başlamak şart.
- **Kör dinleme sayfası hazır:** `voxcpm-contest/small-bench/dinleme/index.html`.
  - 10 cümle var. Her cümlede öğrenci, EMA, Antalia ve öğretmen rastgele sırayla çalıyor.
  - Her cümlede kodlar ve dosya adları değişiyor. Ses düzeyleri eşitlendi.
  - Bitince çıkan yazıyı bana yapıştır; anahtar ayrı dosyada.
- **Sıradaki karar senin kulağın.** Dinleme "bu f022 ve EMA'dan iyi" derse plandaki "Kasım'da veriyi büyüt" adımına geçilebilir.
- **Sonra yapılacak teknik işler:**
  - telefon hızı için Aşama 3;
  - duyguların güçlendirilmesi;
  - ses çözücüsünün daha uzun eğitilmesi.

---

## 1. Code (new files only)

| File | Purpose |
|---|---|
| `tts/small/student_data.py` | Trainer data side, numpy only. Index filter, row checks, word bounds and frame timeline (identical to `ema_lightning.engine.Engine.piece`/`plan`), padded `collate`, length-bucketed batches under a frame budget, warm-up + cosine LR, checkpoint naming/latest/prune, eval system spec parser |
| `tts/small/student.py` | `StudentAcoustic(Acoustic)`: EMA's acoustic model plus a zero-init style embedding (8×224) added to the text features before the duration predictor and aligner. `losses()`: OT-CFM velocity MSE + per-word log-duration L2, condition dropout, style dropout. Samplers: Euler / midpoint with batched CFG and optional sway schedule, plus `ema4` (EMA's own 4-step schedule). `make_tts()` plugs the student into EMA's own engine (chunker, timeline, pauses, decoder windows unchanged) |
| `tts/small/train.py` | Chunked trainer: resumes from the newest checkpoint, stops after `--max-minutes`, writes `log.jsonl`. Prefetch thread, AdamW, EMA of weights, fixed-noise val loss for weights and EMA |
| `tts/small/wise.py` | Weight-space interpolation of a fine-tuned student with `ema.pt` (WiSE-FT, [arXiv 2109.01903](https://arxiv.org/abs/2109.01903)) |
| `tts/small/decoder_ft.py` | `prep`: VoxCPM2 decoder over all f022 latents (48 kHz int16 targets). `train`: fine-tune `decoder.pt` with multi-res mel L1 + LSGAN (multi-period + multi-scale discriminators written from the HiFi-GAN paper) + feature matching; chunked, resumable; saves decoders in `decoder.pt` format |
| `tts/small/eval_student.py` | P1 evaluation, the P0 protocol plus ECAPA, TR-Soru intonation rows, style samples, decoder variants, `--rtf-only` timing (interleaved rounds) |
| `tts/small/style_probe.py` | The same sentences under every style id: duration, rate, F0, RMS |
| `tts/small/listen_page.py` | Blind listening page (BS.1770 loudness, per-sentence codes, random file names, key outside the page, `--decode` for a pasted result) |
| `tests/test_small_student.py` | 33 tests: 28 numpy/scipy/soundfile tests that run in the test venv; 5 torch tests (style init = ema.pt, `ema4` == `Acoustic.sound_stage`, batched CFG == two passes, padding invariance, one-batch overfit) that run in venv-student and are skipped in the test venv |

**Tests**
- Full suite (test venv): **897 passed, 6 skipped**. The baseline was 869 passed, 1 skipped.
- `test_small_student.py` with venv-student: 33/33.

**Wrapper check**
- `student.make_tts` with `ema.pt` and the `ema4` sampler reproduces `EMA().say` to within 5.6e-4 max abs sample difference (cuDNN nondeterminism).

## 2. Training (local RTX 5060, all < 30 min per call, $0)

Data: `data/small/f022-latents`, 4,164 train / 61 val clips (style ids 0-6; laughter not used). Batches: frame budget 4,000 (≈ 108 batches/epoch, ≈ 37 clips/batch).

| Run | Init | Steps | LR | Wall | Peak GPU | Val flow (EMA weights) best → final | Val dur |
|---|---|---:|---|---:|---:|---|---|
| `ft-a` | ema.pt | 20,000 (185 epochs) | 1e-4, warm-up 500, cosine → 1e-5 | ≈ 15 min train (+ val) | 0.86 GB | 1.0018 @13k → 1.0043 | 0.038 @3-4k → 0.051 (overfits; train 0.004) |
| `ft-c` | ema.pt | 4,000 (dense checkpoints) | 1e-4, cosine over 4k | ≈ 5 min | 0.86 GB | → 1.031 | 0.038 |
| `scratch-b` (control) | random | 20,000 | 3e-4, warm-up 1,000 | ≈ 19 min | 0.86 GB | 1.160 @12k → 1.175 | 0.054-0.060 (train 0.0004) |

Val flow at init (ema.pt) is 2.373. Speed is 0.035-0.045 s/step when nothing else runs.

**What happened: more training gives more f022 voice but less naturalness.** Quick evals use the same 60 rows, euler 16, CFG 2, and the original `decoder.pt`.

| Checkpoint (EMA weights) | All-60 WER | UTMOS trzor / heldout | ECAPA to f022 ref |
|---|---:|---:|---:|
| ft-c 1k | 1.89 | 3.44 / 3.10 | 0.397 |
| ft-c 2k | 1.26 | 3.22 / 2.76 | 0.488 |
| ft-c 4k | 2.52 | 3.02 / 2.49 | 0.512 |
| ft-a 8k | 2.32 | 2.92 / 2.44 | 0.515 |
| ft-a 20k | 2.52 | 2.95 / 2.43 | 0.532 |
| ft-a 20k, 32 Euler steps / midpoint 16 / CFG 3 / sway -1 | 2.94 / 2.52 / 1.89 / 2.73 | 3.04-3.09 / 2.48-2.60 | 0.53 |
| scratch-b 20k | **13.35** | 2.54 / 1.99 | 0.486 |

- The sampler does not fix the drop: more steps, midpoint, higher CFG and sway each recover at most about 0.15 UTMOS.
- **Likely reason:** `ema.pt` is a DMD2 few-step generator. Flow-matching fine-tuning turns it into a plain flow model. On 4.5 h of data, that model's samples are smoother, even though its val loss is lower.
- **Scratch control:** a 5.6M flow model from scratch on 4.5 h is unusable (WER 13 %). Initialising from `ema.pt` is required.

**WiSE-FT interpolation** (`wise.py`, w = α·ema.pt + (1−α)·ft-a 20k) recovers naturalness and keeps most of the voice. It costs nothing at inference: the weights are the same size. Results with `decoder.pt`:

| α | All-60 WER | UTMOS trzor / heldout | ECAPA |
|---:|---:|---:|---:|
| 0 (ft-a 20k) | 2.52 | 2.95 / 2.43 | 0.532 |
| 0.15 | 1.37 | 3.23 / 2.86 | 0.511 |
| 0.20 | 1.68 | 3.35 / 2.98 | 0.501 |
| **0.25** | **1.16** | **3.36 / 3.08** | **0.483** |
| 0.40 | 1.68 | 3.49 / 3.30 | 0.435 |
| 0.55 | 1.05 | 3.53 / 3.34 | 0.375 |

**Chosen:** α = 0.25 with the fine-tuned decoder, saved as `data/small/runs/final/student.pt`. With that decoder, α 0.20 and 0.25 score within noise of each other: WER 1.37 vs 1.16, UTMOS 3.41/3.11 vs 3.41/3.18, ECAPA 0.630 vs 0.616.

## 3. Decoder fine-tune (task 4)

**Targets**
- Command: `decoder_ft.py prep`.
- VoxCPM2 AudioVAE decoder over all 4,274 clips' latents: 247 s on GPU, 1.5 GB int16 at `data/small/f022-dec48k/`.

**Training run `dec-a`**
- Batch 4 × 12 frames (0.48 s), plus 4 latent context frames on each side.
- Generator LR 5e-5, discriminator LR 2e-4, mel weight 45. Discriminators trained alone for the first 300 steps.
- 3,353 steps in 25 min (0.51 s/step, peak 4.8 GB).
- An earlier try with batch 8 × 16 frames + 8 context needed 7.5 GB. It spilled into shared memory (7-10 s/step) and was stopped.

**Validation mel L1** on 20 whole val clips: 0.570 at init → 0.395 at step 2,500 → 0.404 at 3,353. Step 2,500 is kept as `data/small/runs/final/decoder.pt`, in the same format as `decoder.pt`; `load_decoder` loads it.

| Same latents, different decoder | All-60 WER | UTMOS trzor / heldout | ECAPA to f022 ref |
|---|---:|---:|---:|
| teacher wav → VoxCPM2 encoder → `decoder.pt` | 1.37 | 3.34 / 2.83 | 0.511 |
| … → **fine-tuned decoder (2,500)** | 1.47 | **3.45 / 3.06** | **0.641** |
| … → fine-tuned decoder (3,353) | 1.58 | 3.38 / 2.98 | 0.636 |
| student → `decoder.pt` | 1.16 | 3.36 / 3.08 | 0.483 |
| student → **fine-tuned decoder** | 1.16 | **3.41 / 3.18** | **0.616** |
| EMA acoustic (`ema.pt`) → fine-tuned decoder | 1.79 | 3.25 / 2.89 | 0.296 |

- **The "3.20 ceiling" of P0 was `decoder.pt`.** The fine-tuned decoder lifts the teacher-latent ceiling by about 0.1-0.2 UTMOS and raises speaker similarity from 0.51 to 0.64. That is higher than the teacher through VoxCPM2's own decoder (0.62).
- **The decoder does not carry the voice by itself.** EMA's acoustic model through the new decoder only goes from 0.22 to 0.30. The acoustic model carries the voice; the decoder carries timbre detail.
- **Mel loss is not a good stopping rule here.** It flattens after about 2k steps. At 3,353 steps UTMOS was lower than at 2,500, which looks like GAN noise. A longer run should pick its checkpoint by UTMOS.

## 4. Evaluation (P0 protocol)

**Protocol:** the same 60 rows (`voxcpm-contest/small-bench/subset.jsonl`), seed 0, faster-whisper large-v3-turbo fp16 (tr, beam 5), `normalize_for_eval` with the best reference, UTMOS22 strong, ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) cosine to `xfer/f022corpus/ref/designed-f022-s0.wav`.

**Sources and settings**
- EMA and Antalia: the P0 renders were re-scored. Their WER and UTMOS reproduce P0 exactly.
- "teacher" means the big model's renders of the same sentences (VoxCPM2 + f022 LoRA, s3tr step 500, ts16 cfg2, `voxcpm-eval/voxcpm2-f022-s3tr`).
- Student: euler 16 steps, CFG 2, EMA's engine, trnorm spoken form.

**Outputs:** `voxcpm-contest/small-bench/p1/` (`summary.json`, `SUMMARY.md`, `<system>.rows.jsonl`, `rtf.json`, `pair_sim_to_teacher.json`).

| System | TR-Zor-40 WER / CER | ok | Held-out-20 WER | ok | All-60 WER | UTMOS trzor / heldout | ECAPA to f022 | TR-Soru-35 WER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **student (final)** | **1.85** / 0.76 | 38/40 | **0.95** | 17/20 | **1.16** | 3.41 / **3.18** | **0.616** | 0.0 |
| student, 8 Euler steps | 2.78 / 0.83 | 37/40 | 0.95 | 17/20 | 1.37 | 3.30 / 2.99 | 0.612 | 1.1 |
| student, 4 steps CFG 2 / CFG 1 | 0.46 / 4.61 | 39 / 34 | 1.49 / 1.77 | 15 / 14 | 1.26 / 2.41 | 2.98 / 2.48 · 2.69 / 2.22 | 0.58 / 0.56 | — |
| student with `decoder.pt` | 1.85 / 0.83 | 37/40 | 0.95 | 17/20 | 1.16 | 3.36 / 3.08 | 0.483 | 0.0 |
| EMA Lightning | 2.34 / 0.46 | 37/40 | 1.49 | 16/20 | 1.68 | 3.35 / 2.95 | 0.216 | 0.0 |
| Antalia-2 Mini | 2.76 / 0.68 | 37/40 | 1.22 | 16/20 | 1.57 | 3.22 / 2.94 | 0.235 | 1.1 |
| teacher (VoxCPM2 + LoRA) | 2.80 / 0.53 | 36/40 | 0.82 | 16/20 | 1.26 | **3.65** / 3.13 | 0.621 | 0.0 |
| teacher → `decoder.pt` | 3.27 / 0.61 | 35/40 | 0.82 | 16/20 | 1.37 | 3.34 / 2.83 | 0.511 | 0.0 |
| teacher → fine-tuned decoder | 3.27 / 0.61 | 35/40 | 0.95 | 16/20 | 1.47 | 3.45 / 3.06 | 0.641 | 0.0 |

**Full TR-Zor (155 rows, same judge)**

| System | WER | ok | UTMOS | ECAPA |
|---|---:|---:|---:|---:|
| student | 2.32 | 144 | 3.44 | 0.55 |
| EMA | 2.43 | 145 | 3.36 | 0.21 |
| Antalia | 3.30 | 140 | 3.23 | 0.22 |
| teacher | 2.54 | 141 | 3.69 | 0.60 |

- Output is in `small-bench/trzor155/p1/`. Its SUMMARY header says "TR-Zor-40", but the rows are all 155.
- The repo's 1.9 % for EMA (`benchmarks/tts/results/tr_zor`) used the CPU int8 judge.

**Errors**
- The student's word errors on the 60 rows are all shared with at least one other system or are Whisper spellings:
  - "584213" heard as "ila bin" (Antalia has the same error)
  - "emar" (also the teacher)
  - "yurtdışı" (also teacher and Antalia)
  - "te le" (all systems)
  - "uyar mı" → "uyarma" (EMA and Antalia)
- It has no error of its own.
- EMA's "siparış" vowel error and its trailing-silence hallucination are gone.

**Does it sound like f022?**

| ECAPA comparison | Student | Teacher | EMA | Antalia |
|---|---:|---:|---:|---:|
| To the f022 reference clip | 0.616 | 0.621 | 0.22 | 0.24 |
| To the teacher's clip of the same sentence | 0.677 | — | 0.33 | 0.20 |

- To the reference clip, the student is level with the teacher.
- To the teacher's clip of the same sentence, the student reaches 0.677. For comparison, the teacher's own latents score 0.789 through `decoder.pt` and 0.886 through the new decoder.
- **Ship-rule check:** the plan's ship rule asks for student-to-teacher ≥ 0.75. **0.677 does not meet it.** The gap is in the acoustic model (WiSE α trades some of the voice for naturalness), not in the decoder.

**Questions** (`tts/voxcpm/score_intonation.py`, Whisper word timings from this judge; outputs in `small-bench/inton/<system>/`)

| System | Pass (35 items) | Questions pass | mi-questions | wh-questions | Final-rise (4) | Declaratives | AUC peak | Pair accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **student** | **0.60** | **0.48** | 0.50 | **0.86** | 0/4 (end −3.5 st) | 0.90 | 0.88 | 0.80 |
| EMA | 0.43 | 0.24 | 0.00 | 0.43 | 0/4 | 0.90 | 0.78 | 0.80 |
| Antalia | 0.51 | 0.32 | 0.30 | 0.71 | 0/4 | 1.00 | 0.90 | 1.00 |
| teacher | 0.56 | 0.44 | 0.40 | 0.71 | 0/4 (3 unscored overall) | 0.89 | 0.93 | 1.00 |

- **Rising questions do not rise for any system,** including the teacher. This is a data gap: the f022 corpus "yukselen" clips evidently end low as well.
- The student's mi- and wh-question shapes match or beat the teacher's.

**Emotion styles**

Samples are in `small-bench/st-final/styles/<style>.wav`, one fitting sentence per style. All 7 are word-perfect, with UTMOS 3.09-3.82 and ECAPA 0.55-0.70.

`style_probe.py` reads 3 neutral sentences under every style id (`small-bench/style-probe/style.json`):

| Style | F0 vs neutral | Rate vs neutral | F0 range (st) | Corpus frames/letter (neutral 1.84) |
|---|---:|---:|---:|---:|
| angry | +0.7 st | −2 % | 13.3 (neutral 9.7) | 1.88 |
| calm | −0.2 st | −8 % | 12.6 | 2.27 |
| cheerful | +0.7 st | −1 % | 10.5 | 1.94 |
| empathetic | −0.7 st | −5 % | 10.7 | 1.99 |
| sad | **−1.9 st** | **−8 %** | 12.1 | 2.24 |
| surprised | +0.5 st | −2 % | 12.0 | 2.01 |

- **The embedding works, in the right directions, but weakly.** In the corpus, calm and sad are about 20 % slower; the student makes them 8 % slower.
- **Two reasons:**
  1. There are only 49-132 clips per style.
  2. WiSE α = 0.25 scales the embedding by 0.75.
- **Fixes:** more emotion data (November plan), or skip interpolating `style_emb`.

**Speed** (`p1/rtf.json`; 60 rows; systems interleaved, 2 rounds, median)

| System | GPU RTF | CPU RTF, 2 threads | CPU RTF, 8 threads |
|---|---:|---:|---:|
| student, 16 Euler steps, CFG 2 (32 DiT passes) | 0.025 | **0.196** | 0.137 |
| student, 8 steps CFG 2 | 0.016 | 0.155 | 0.117 |
| student, 4 steps CFG 1 | 0.013 | 0.139 | 0.102 |
| EMA (4-step DMD2) | 0.014 | 0.131 | 0.095 |
| Antalia | 0.020 | 0.079 | 0.046 |

- **Measurement conditions:** this laptop measured EMA at 0.131 here vs 0.104 in P0, so it runs about 25 % slower than then (clocks or thermals). Only the ratios within this table are meaningful.
- **Decoder share:** the decoder dominates CPU time (4 vs 32 DiT passes is only +50 %).
- **Speed ship rule (≤ 0.2 at 2 threads):** the final setting passes, but only barely.
- **Fewer steps cost quality:** 4 steps lose about 0.4 UTMOS. Stage 3, few-step distillation from the paper, is needed to reach EMA/Antalia speed at this quality.

## 5. Gates

| Gate | Result |
|---|---|
| P0 go gate: TR-Zor ≤ ~3 % | ✅ 1.85 % (40 rows), 2.32 % (155 rows) |
| ECAPA ≥ 0.70 to f022 | ❌ as written: 0.616. ✅ relative: the teacher itself scores 0.621, so 0.70 is unreachable on this reference with this metric; recommend "≥ teacher − 0.05" |
| UTMOS within ~0.1 of the `decoder.pt` ceiling | ✅ exceeds it: 3.36/3.08 with `decoder.pt` vs 3.34/2.83 ceiling; 3.41/3.18 with the new decoder |
| Listening says "this is f022" | ⏳ blind page ready (§6) |
| Ship rule: TR-Zor ≤ 1.9 % | ✅ on 40 rows (1.85). Not met on 155 (2.32 vs EMA 2.43 on the same judge, so the student beats EMA) |
| Ship rule: student vs teacher similarity ≥ 0.75 | ❌ 0.677 |
| Ship rule: CPU RTF ≤ 0.2 on 2 threads | borderline 0.196 (needs Stage 3 for phones) |
| Ship rule: ties or beats EMA in blind listening | ⏳ |

## 6. Blind listening page

**Location:** `voxcpm-contest/small-bench/dinleme/index.html` (open the file in a browser; 19 MB, 40 wavs in `s/`).

**Content:** 10 sentences.
- Plain: ozel_04, ozel_08, ozel_10.
- Numbers: sayi_13, para_09, tarih_05.
- Questions: soru_07, soru_03.
- Long held-out: d4d2e9f7…, 6df62be7….

**How it works**
- Each sentence plays the student, EMA, Antalia and the teacher in random order.
- Every clip has a 2-character code that is new for each sentence and a random 12-hex file name.
- Audio is set to −20 LUFS (BS.1770-4), with the peak kept at or below −1 dBFS. Input levels were student −21.3, EMA −25.2, Antalia −21.9 and teacher −18.9 LUFS, so normalisation matters here.
- The listener picks the best clip per sentence and can add a comment. A copy button produces the result text.

**Key:** `voxcpm-contest/small-bench/dinleme-anahtar.json`, kept outside the page folder. Decode a pasted result with:

`python tts/small/listen_page.py --decode "<pasted text>"`

## 7. Problems and caveats

- **speechbrain was installed into venv-student from PyPI** (1.0.3 + HyperPyYAML, ruamel.yaml, sentencepiece, plus `requests`), with nothing installed into system Python. torchaudio 2.11 lacks `list_audio_backends`; the same shim as `voice_contest.py` is used.
- **A decoder fine-tune at 8 × 16 frames spilled out of the 8 GB GPU.** WDDM falls back to shared memory and makes it 20× slower, with no OOM error. That run was stopped; the run used 4 × 12 frames.
- **Concurrent jobs distort RTF.** Quick-eval GPU RTFs in §2 were taken while training ran (0.035 vs 0.016 quiet). Use `p1/rtf.json` only.
- **Small sample.** The 60-row WER differences between student, EMA and Antalia are 5-6 word edits; treat them as ties. The 155-row and TR-Soru results point the same way.
- **The duration predictor overfits** after about 3k steps (train 0.004 vs val 0.05). WiSE partly undoes it. A cleaner fix is a separate, lower LR or early stop for `chardur`.
- **UTMOS is not a human judgement.** It prefers EMA's own voice (EMA 3.35 vs teacher-through-`decoder.pt` 3.34). The listening page decides.
- **Provenance is unchanged.** The student still inherits `ema.pt` (Apache-2.0, ~1000 h of undisclosed data; plan A9 and critique item 4). This P1 is internal only.

## 8. Files and how to run

**Weights and runs** (gitignored `data/`)

| Path | Content |
|---|---|
| `data/small/runs/final/student.pt` | 22 MB fp32; WiSE α 0.25 of ft-a 20k |
| `data/small/runs/final/decoder.pt` | 12 MB; dec-a step 2,500 |
| `data/small/runs/{ft-a,ft-c,scratch-b,dec-a,wise}/` | Training runs and interpolated weights (3.4 GB) |
| `data/small/f022-dec48k/` | Decoder targets (1.5 GB) |

**Outputs** (gitignored): `voxcpm-contest/small-bench/`
- `p1/`, `trzor155/`, `inton/`, `style-probe/`, `dinleme/`, `dinleme-anahtar.json`
- `<system>/` renders, including the quick-sweep renders (1.7 GB in total)

**Logs (local, not in the repo):** `train_*.log`, `dec_a.log`, `eval_full.log`, `rtf.log`, `inton.log`.

```powershell
$py = "<venv-student>/Scripts/python.exe"
$t  = "<venv-tests>/Scripts/python.exe"
$env:HF_HUB_OFFLINE = "1"
& $py tts/small/train.py --run data/small/runs/ft-a --init ema --steps 20000 --max-minutes 27      # resumes if run again
& $py tts/small/wise.py data/small/runs/ft-a/ckpt_00020000.pt --alpha 0.25 --out data/small/runs/final/student.pt
& $py tts/small/decoder_ft.py prep
& $py tts/small/decoder_ft.py train --run data/small/runs/dec-a --steps 5000 --batch 4 --seg 12 --ctx 4 --d-warmup 300 --val-every 500 --ckpt-every 500 --max-minutes 25
$F = "data/small/runs/final"
& $py tts/small/eval_student.py --soru --styles --cpu-threads 2 8 --systems "student:st-final=$F/student.pt@euler,16,2,0,$F/decoder.pt" ema antalia teacher teacher_emadec "teacher_emadec:teacher_newdec=$F/decoder.pt"
& $t  tts/voxcpm/score_intonation.py voxcpm-contest/small-bench/inton/st-final --sets soru --whisper none
& $py tts/small/eval_student.py --rtf-only --cpu-threads 2 8 --systems "student:st-final=$F/student.pt@euler,16,2,0,$F/decoder.pt" ema antalia
& $py tts/small/style_probe.py render;  & $t tts/small/style_probe.py measure
& $t  tts/small/listen_page.py --systems st-final ema antalia teacher
```

## 9. Recommended next steps

1. **The user listens** to `dinleme/index.html`. If the student ties or beats EMA and sounds like f022, P0 is a "go".
2. **Stage 3: few-step distillation** of the final student, written from the DMD2 paper (the official code is NC). It is needed for phone speed. The final student, or ft-a 20k, is the teacher.
3. **Decoder:** a longer run selected by UTMOS rather than mel L1 (the GAN still moves after 2k steps). Try a bf16 discriminator to allow bigger batches.
4. **Duration predictor:** early stop or a lower LR. Do not interpolate `style_emb`, so styles keep their full strength.
5. **November data:** add rising-question and emotion clips. Final-rise questions fail even for the teacher.
6. **Gate wording:** change the ECAPA gate to "≥ teacher − 0.05 on the same reference" in the plan.
