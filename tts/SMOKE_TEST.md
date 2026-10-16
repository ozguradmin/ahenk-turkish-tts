# CPU smoke test (2026-10-07)

Goal: prove the whole chain works on this machine without a GPU: environment → corpus download and
preparation → fine-tuning from the LJSpeech Piper checkpoint → checkpoint / resume / time limit /
stop / preemption → ONNX export → Turkish synthesis with the `piper` runtime at 22,050 Hz, plus the
Modal container commands and the benchmark integration. The model is trained for a handful of
steps only; its audio is not meant to be good.

**Machine:** 4 vCPU (shared), 15 GB RAM, **no GPU**. All jobs ran with `OMP_NUM_THREADS=2`, one heavy
process at a time. Environment `.venv-piper` from `./tts/setup_env.sh --cpu`: Python 3.12.3,
torch/torchaudio 2.8.0+cpu, Lightning 2.6.6, jsonargparse 4.52.0, piper1-gpl `efffbfb226` (piper-tts
1.8.0, espeak-ng `724808c`), onnx 1.23.2, onnxruntime 1.30.0, librosa 0.11.0, numpy 2.5.3.
Commands run from the repo root after `source tts/path.sh`; data in `data/tts_train/` (git-ignored).

## 1. Environment

| Command | Wall | Result |
|---|---|---|
| `./tts/setup_env.sh --cpu --venv <scratch> --piper-dir <scratch>` (from scratch, warm uv cache) | 44 s | clones piper1-gpl, builds espeak-ng + bridge (CMake) and `monotonic_align` (Cython), self-test phonemizes `trnorm` output with espeak-ng `tr` |
| `./tts/setup_env.sh --cpu` (existing venv) | 17 s | idempotent; reuses torch and the checkout |
| `uv pip compile` of the CUDA 12.8 variant | 1 s | resolves: torch 2.8.0+cu128, triton 3.4.0, cuDNN 9.10.2.21 → `locks/linux-py312-cu128.txt` |

Found while setting up: `pip install -e piper1-gpl[train]` only links `src/`; the espeak-ng bridge
needs the dev build (`setup.py build_ext --inplace`), which `setup_env.sh` runs.

## 2. Data

```bash
python tts/prepare_antalia.py --report-only --out data/tts_train/antalia_report   # metadata only
python tts/prepare_antalia.py --out data/tts_train/antalia_smoke --limit 40 --heldout 4
python tts/prepare_antalia.py                                                      # full corpus
```

| Step | Wall | Output |
|---|---|---|
| metadata analysis, full corpus (trnorm, ASR agreement, espeak-ng phonemes for 1,046 texts) | 27 s | [DATA_REPORT.md](DATA_REPORT.md) |
| smoke subset: 40 clips (36 train + 4 held-out, stratified), download 9.3 s, resample 0.4 s | 35 s | 10.7 min of audio, 22,050 Hz mono PCM_16 |
| **full corpus:** 1,046 clips downloaded + sha256-checked (163 s), resampled (10.4 s, 2 threads) | **199 s** | 742 MB of WAV, max peak 0.906 (no clipping), every duration within 0.05 s of the metadata |
| piper's own preprocessing (`--piper-cache`), 40 clips | 5.1 s | 0.13 s and 4.1 MiB per clip → full corpus ≈ 2.3 min, 4.2 GiB |

## 3. Base checkpoint

`lj-med_1000.ckpt` (rhasspy/piper-checkpoints @ `37d1bea250`, 846 MB, sha256 `dcf2449b…`): download
9 s via `train.sh --fetch-base-only`. Epoch 999, global step 1,114,000, written by Lightning 1.9.5,
generator 23.66 M + discriminator 46.75 M parameters, LR 1.76e-4 (old piper stepped its schedulers).

- `torch.load(weights_only=True)` fails on it (`PosixPath` in the hyper-parameters). Lightning 2.6's
  `LightningCLI` does exactly that for `--ckpt_path`, so the upstream-documented fine-tuning route
  stops before training; the recipe uses `--model.warmstart_ckpt` instead ("Copied 784 parameters …
  0 skipped": generator and discriminators).

## 4. Training on CPU (`./tts/train.sh --allow-cpu`, batch 4, 2 threads)

Base command (each session differs only as listed):

```bash
./tts/train.sh --allow-cpu --exp-dir data/tts_train/exp/smoke --data-dir data/tts_train/antalia_smoke \
  --max-epochs 2 --val-every 1 --ckpt-every 1 --snapshot-every 1 -- --trainer.limit_train_batches 3
```

| # | Change | Tests | Wall | Result |
|---|---|---|---|---|
| 1 | — | warm start from LJSpeech, fixed split (36 train / 4 validation), `val_mel` + UTMOS `val_mos`, all checkpoint kinds | 161 s | exit 0; `val_mel` 0.551 → 0.529, `val_mos` 1.24 → 1.26 |
| 2 | `--max-epochs 3` | resume from `last.ckpt` (epoch 2, step 12); upstream skips the warm start; LR decay re-derived | 78 s | exit 0, epoch 3 |
| 3 | `--max-epochs 10 --hours 0.02` | time limit (72 s) | 72 s | **exit 75** after epoch 4, `RESUME.json` reason `time_limit` |
| 4 | `--max-epochs 10`, then `./tts/train.sh --stop` | STOP file | 71 s | **exit 75**, reason `stop_file` |
| 5 | `--max-epochs 10`, `kill -TERM <python pid>` mid-run | preemption | — | `last.ckpt` saved (`exception:SIGTERMException`, step 36) |
| 6 | `--max-epochs 7` | resume from the SIGTERM checkpoint | 76 s | exit 0, epoch 7 (step 42) |
| 7 | new `exp/smoke_final`: fresh run, then `--max-epochs 3` | regression with the final code | 130 s + 81 s | exit 0 + exit 0; metrics identical to session 1 (deterministic, seed 0); 3 files per metric |

Measured speed (`epochs.jsonl`, real clips): **11–16 s per training batch of 4** (2 threads, shared machine);
validation (4 clips + 4 syntheses + UTMOS) 11–15 s. `last.ckpt` (846 MB) written in 3–8 s.

Bugs found and fixed by this test:
- Lightning 2.6 raises `MisconfigurationException` when a monitored metric is missing after
  validation. piper's default `val_mos` checkpoint therefore stops the run whenever UTMOS cannot be
  downloaded (here: torch.hub's GitHub API call returns 403 through the proxy). `piper_fit.py` makes
  that checkpoint optional; confirmed in the Modal dry run (§6), where UTMOS was unavailable.
- Lightning's `save_last` only writes when a top-k file is saved in the same step and writes in place;
  `piper_fit.py` writes `last.ckpt` itself, atomically (temp file + rename).
- The first version wrote `last.ckpt` *before* the top-k callbacks updated their state, so after a
  resume the newest `best-*.ckpt` was forgotten and never deleted (7 "top-3" files after 6 sessions).
  Fixed by ordering; re-tested over two sessions: exactly 3 files per metric, bookkeeping matches disk.
- Lightning exits with code 0 after SIGTERM; `train.sh` now reports interrupted runs as 75.
- `--trainer.limit_train_batches 1` is parsed as the float 1.0 (= 100 %); use ≥ 2 for "N batches".

## 5. Training-step benchmark (`tools/step_bench.py`, CPU, 2 threads, fp32)

Synthetic batches with this corpus' shapes; one configuration per process.

| batch | clip s | phoneme ids | s/step | peak RSS growth |
|---:|---:|---:|---:|---:|
| 1 | 20.24 | 763 | 4.56 | 2,274 MB |
| 2 | 20.24 | 763 | 7.31 | 2,872 MB |
| 4 | 20.24 | 763 | 13.79 | 4,365 MB |
| 2 | 17.56 | 565 | 6.59 | 2,687 MB |
| 2 | 6.57 (LJSpeech mean) | 200 | 3.52 | 2,098 MB |
| 4 | 6.57 | 200 | 6.19 | 2,693 MB |

→ per extra clip: 3.08 s and ~0.7 GB (20.24 s clips), 1.34 s and ~0.3 GB (6.57 s); fixed part ~1.7 GB.
Per-clip cost ≈ 0.51 s + 0.127 s per audio second, so an Antalia epoch (1,026 clips, mean 16.8 s)
costs 0.154 × an LJSpeech epoch (13,100 clips). Monotonic alignment (Cython, CPU) for a batch of 32:
0.12–0.2 s with this corpus' lengths, 0.02 s with LJSpeech's. These numbers feed the GPU estimate in
README §4.

## 6. Modal: offline checks and container dry run

- `python -c 'import …tts/modal_app.py'`: imports, 5 functions (prepare, train, probe, status, export).
- `modal run tts/modal_app.py --help` (modal 1.6.1, offline): lists `--action --max-usd --hours
  --max-epochs --exp-name --ckpt --minutes --args`.
- `tools/modal_dryrun.py` runs the **same bash lines** the containers run (`container_cmd()` +
  `prepare_cmds()` / `train_cmd()` / `probe_cmds()` / `export_cmd()`), with the repo copied using the
  app's mount ignore list, the CRLF-stripping work copy, `/data` → `data/tts_train/modal_dryrun`, and
  CPU-sized extras (40 clips, `--allow-cpu`, batch 2):

| Step | Wall | Result |
|---|---|---|
| prepare: `prepare_antalia.py --workers 4 --piper-cache` (+ `--limit 40 --heldout 4`) | 35 s | data + piper cache |
| prepare: `train.sh --fetch-base-only` | 1 s | checkpoint present, sha256 OK |
| prepare: UTMOS prefetch via torch.hub | 2 s | 403 here (GitHub API blocked); `|| true`, as designed |
| train: `train.sh --preset a100 --hours 0.050 --max-epochs 2 … --no-guard --ckpt-every 10 --val-every 5 --snapshot-every 100 --num-workers 4` | 47 s | exit 0 |
| probe: `step_bench.py` + `train.sh --hours 0.025 …` | 23 s + 95 s | time-limit stop, **exit 75**; `val_mos` unavailable, no crash; probe summary printed by `_fmt_probe` |
| export: `export_and_test.sh --exp-dir … --threads 2 --n 3` | 22 s | ONNX + samples + `piper` CLI |

Total 226 s, all steps OK. Not covered here (needs Modal): image build with CUDA 12.8, GPU speed and
memory, Volume commits, preemption retries.

## 7. Export and synthesis

```bash
./tts/export_and_test.sh --exp-dir data/tts_train/exp/smoke --data-dir data/tts_train/antalia_smoke --n 8 --threads 2
```

Smoke model (7 epochs × 3 batches = 21 batches): export 14 s, `tr_TR-antalia-medium.onnx` **60.6 MB**
(15.65 M parameters, opset 15, inputs `input`, `input_lengths`, `scales`), `.onnx.json` with
`audio.sample_rate 22050`, `espeak.voice tr`, `language tr_TR`, `dataset antalia`; `MODEL_CARD.md`.

| | Result |
|---|---|
| 8 TR-Zor sentences (one per category: sayı, sıra, tarih, saat, para, yüzde, birim, numara) via `trnorm.normalize` + `PiperVoice` | all valid: 16-bit mono WAV, **22,050 Hz**, non-empty, finite, not silent |
| 4 held-out sentences | valid; synthesized/real duration 0.50–0.58 (an almost untrained model speaks too fast) |
| speed, 2 onnxruntime threads | **RTF 0.068** overall (0.060–0.087 per sentence), model load 2.8 s |
| `echo "<trnorm text>" \| piper -m tr_TR-antalia-medium.onnx -f piper_cli.wav` | 2.68 s, 22,050 Hz, PCM_16 |

**Final regression with the final code** (§4 row 7, then `./tts/export_and_test.sh --exp-dir
data/tts_train/exp/smoke_final …`, 22 s): `val_mel` 0.5506 → 0.5294 → 0.5414 and `val_mos` 1.2374 →
1.2632 → 1.2830, identical to sessions 1–2 of the first run, i.e. training is reproducible under the
CLI's seed 0. Export info `epochs_trained 3, batches_trained 9`; 12/12 syntheses valid at 22,050 Hz,
**RTF 0.072** overall; held-out duration ratio 0.48–0.57; `piper` CLI 2.84 s at 22,050 Hz.

## 8. Benchmark integration (`benchmarks/tts/systems.py`)

In the benchmark venv (`.venv-tts`, **PyPI** `piper-tts==1.8.0`, i.e. the runtime users install):

- without `PIPER_ANTALIA_ONNX`, `SYSTEMS` is unchanged (5 systems); with it, `piper-antalia` appears;
- `load_system("piper-antalia")` loads the smoke ONNX, output is deterministic (seeded, session
  re-created per utterance like `piper-tr-dfki`), 22,050 Hz; `config()` includes the ONNX sha256;
- `synthesize.py --set <2-row set> --system piper-antalia --frontend trnorm`: 2/2 rows, RTF 0.080
  (outputs deleted afterwards).

## 9. Not covered here

CUDA training (fp32 / bf16), real GPU memory and speed (`modal run tts/modal_app.py --action probe`
measures both), a full-length training run and its quality (TR-Zor WER, UTMOS, listening), UTMOS
download through torch.hub (blocked by this proxy; works on Modal and laptops), Home Assistant /
Wyoming playback (the `.onnx.json` mirrors the official `tr_TR-dfki-medium.onnx.json` fields).
Large files were deleted after the test (corpus audio, caches, checkpoints).
