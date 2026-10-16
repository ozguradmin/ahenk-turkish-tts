# CPU smoke test (2026-10-07)

Goal: prove the whole chain — data prep → fbank → BPE → icefall training (causal Zipformer
transducer) → checkpoint/resume → averaged streaming ONNX export (fp32 + int8) → sherpa-onnx
decoding with WER/CER — works end to end. The model is deliberately tiny and undertrained;
the WER is garbage by design.

**Machine:** 4 vCPU shared with two other agents' jobs (load average 5.5–7.7 during the test,
so wall times are inflated ~2x), 15 GB RAM, **no GPU**. All my jobs ran with
`OMP_NUM_THREADS=2` / `--num-threads 2`. Environment: `.venv-stt` from `./stt/setup_env.sh --cpu`
(Python 3.11.17, torch 2.8.0+cpu, k2 1.24.4.dev20260625+cpu.torch2.8.0, lhotse 1.33.0,
icefall 3f848bb6d0ac, sentencepiece 0.2.1, onnx 1.19.1, onnxruntime 1.23.2, sherpa-onnx 1.13.8).

All commands from the repo root after `source stt/path.sh`; data in `data/stt/` (git-ignored).

## 1. Environment

| Command | Wall time | Result |
|---|---|---|
| `./stt/setup_env.sh --cpu` | 23 s | first try failed its self-test: lhotse 1.33.0 imports `urllib3` without declaring it → pinned `urllib3==2.5.0` |
| `./stt/setup_env.sh --cpu` (re-run) | 7.7 s | OK, self-test prints versions + `trnorm` sample |
| `./stt/setup_env.sh --cpu --venv <scratch> --skip-icefall` | 16 s | clean install (warm uv cache) OK |
| `uv pip compile` of the CUDA 12.8 laptop env | 11 s | resolves: torch 2.8.0+cu128, triton 3.4.0, cuDNN 9.10.2.21, k2 …+cuda12.8.torch2.8.0 → `locks/linux-py311-cu128.txt` |

## 2. Data (tiny subsets, ~1.75 h of audio)

```bash
python -I -u stt/prepare/fleurs.py --splits train --max-utts 300   # streams the tarball, no full download
python -I -u stt/prepare/fleurs.py --splits dev   --max-utts 60
python -I -u stt/prepare/fleurs.py --splits test  --max-utts 24
python -I -u stt/prepare/issai_tsc.py --stream --splits test --max-utts 40
python -I -u stt/prepare/commonvoice_hf.py --splits test --max-utts 20
python -I -u stt/prepare/alania.py --shards train-00654 --max-rows-per-shard 230 --dev-voice-percent 10
python -I -u stt/prepare/compute_fbank.py --scan --num-workers 2
python -I -u stt/prepare/make_mix.py --dev-hours-per-source 0.1
python -I -u stt/prepare/make_bpe.py --vocab-size 1000 --extra-text alania-domain --extra-max-lines 100000
```

| Step | Wall | Output |
|---|---|---|
| FLEURS train / dev / test | 10.2 s / 4.3 s / 3.8 s | 300 / 60 / 24 utts (0.995 / 0.208 / 0.082 h); 2 dropped (non-Turkish letters) |
| ISSAI TSC test (HTTP stream of the 21.4 GB tarball, first ~80 MB) | 6.5 s | 40 utts, 0.05 h |
| Common Voice 17 test (fixie-ai Parquet mirror) | 6.9 s | 20 utts, 0.027 h (2 dropped by the char-rate filter) |
| Alania `cc-by/train-00654` (smallest shard, 201 MB, deleted afterwards) | 13.8 s (7 s download) | 207 train + 23 dev utts (dev = held-out voices), 0.31 h + 0.04 h; 87 of 207 degraded-channel copies |
| fbank, 80-dim lilcom | 23 s | 1.75 h at 180–350x real time on 2 threads; 32 MB per hour of audio |
| mix | 1 s | 492 train cuts = 1.21 h (synthetic share 24.6 %), 51 dev cuts |
| BPE (unigram 1000, + 100k alania-domain sentences) | 68 s | `lang_bpe_1000/{bpe.model,tokens.txt}` |
| **all of the above via `./stt/prepare.sh --smoke` into a fresh `STT_DATA`** | **1 min 51 s** | identical cut set |

BPE size study (`make_bpe.py --analyze 500,1000,1500,2000,3000`, 6 min 50 s): tokens/word
2.49 / 1.98 / 1.74 / 1.61 / 1.46; median tokens/s 5.3 / 4.5 / 4.0 / 3.8 / 3.4 — far below the
25 Hz encoder frame rate for every size. Chosen: 1000 (see README §3).
SentencePiece with 2 threads is not reproducible; `make_bpe.py` now defaults to `--threads 1`
(two runs give identical pieces and scores). The smoke model below used the earlier 2-thread BPE.

## 3. Training (`--preset tiny`, 1.19 M parameters, causal, chunk 16/32/64/-1)

71 batches per epoch at `--max-duration 60`. Every run is the same command; `train_tr.py`
resumes from the newest checkpoint on its own.

```bash
./stt/train.sh --preset tiny --allow-cpu --num-workers 0 --save-every-n 10 --hours 0.07 \
  --exp-dir data/stt/exp/smoke-tiny -- --num-threads 2 --valid-interval 30 --log-interval 5
```

| Session | What happened | Wall |
|---|---|---|
| 1 (`--hours 0.07`) | OOM scan, 50 batches (~4 s/batch), **session limit hit → `checkpoint-50.pt`, exit 75**; loss 8.94 → 5.45 | 4 min 20 s |
| 2 (same command) | auto-resume `--start-batch 50`; `./stt/train.sh --stop` → checkpoint-60.pt, exit 75. **Bug found:** first batch had 1175 frames = the first batch of the epoch, i.e. the epoch restarted | 1 min 29 s |
| fix | icefall's OOM scan (`find_pessimistic_batches`) exhausts the just-restored lhotse sampler; `train_tr.py` now skips the scan on mid-epoch resume | — |
| 3 (same command) | auto-resume `--start-batch 60`; first batch 1309 frames = position 11 of an offline replay of the sampler (position 10 was consumed by the stop hook); trained exactly the remaining 60 batches → `epoch-1.pt`, exit 0; valid loss 3.94 → 2.25 | 4 min 26 s |
| 4 (`--num-epochs 4`) | auto-resume `--start-epoch 2`; epochs 2–4; valid loss 1.441 / 1.309 / 1.256 | 15 min 21 s |

`python stt/tools/eta.py --exp-dir data/stt/exp/smoke-tiny` → "40 batches in 2.9 min, ~53 s audio/batch
→ 12x real time, 4.35 s/batch" (2 contended CPU threads).

Thermal guard (no GPU here, so with simulated temperatures and a dummy process group):
`python stt/thermal_guard.py --pid <pid> --simulate 60,85,84,80,71,70 --interval 0.3 --max-polls 6`
→ process state went S → **T (stopped) at 85 °C** → **S (resumed) at 71 °C**; SIGCONT on exit.

## 4. Export + sherpa-onnx test

```bash
./stt/export_and_test.sh --preset tiny --exp-dir data/stt/exp/smoke-tiny --num-threads 2
```

58.5 s in total. Export (8 s): epoch 4, avg 3 → icefall averaged model over epochs (1, 4],
`export.py` → `pretrained.pt`, `export-onnx-streaming.py` with chunk 16 (320 ms) / left context 128
→ `encoder.onnx` 4.4 MB (int8 2.4 MB), `decoder.onnx` 0.28 MB (0.07), `joiner.onnx` 0.26 MB (0.07),
`tokens.txt`, `bpe.model`, `MODEL_CARD.md` in
`data/stt/exp/smoke-tiny/export-epoch-4-avg-3-chunk-16-left-128/`.

Decoding with `sherpa_onnx.OnlineRecognizer.from_transducer`, greedy search, audio fed in 320 ms
chunks, WER/CER on `trnorm.normalize_for_eval` (ITN examples via `trnorm.inverse_normalize` printed
for display):

| Test set | Utts | fp32 WER / CER / RTF | int8 WER / CER / RTF |
|---|---|---|---|
| FLEURS test | 24 | 100 % / 100 % / 0.026 | 100 % / 100 % / 0.028 |
| Common Voice 17 test | 20 | 100 % / 100 % / 0.033 | 100 % / 100 % / 0.033 |
| ISSAI TSC test | 40 | 100 % / 100 % / 0.030 | 100 % / 100 % / 0.033 |

As expected for a 1.2 M-parameter model trained for 4 epochs on 1.2 h, greedy search emits only
blanks (empty hypotheses). With `--decoding-method modified_beam_search` on 8 training
utterances it starts to emit tokens (CER 99.7 %), i.e. the joiner/decoder/token path is live.

A second export with a different latency setting also works (16.8 s incl. decoding):
`./stt/export_and_test.sh --preset tiny --exp-dir data/stt/exp/smoke-tiny --chunk 32 --left 256 --test-sets fleurs --max-utts 6`
→ `export-epoch-4-avg-3-chunk-32-left-256/`, 640 ms chunks, RTF 0.021 (fp32) / 0.027 (int8).

## 5. Other scripts exercised

- `prepare/commonvoice.py` (local Mozilla layout): fake 30-clip corpus with 48 kHz MP3s made from
  FLEURS audio, both `--train-from validated` (dev/test speaker+sentence exclusion) and `train`.
- `prepare/yodas2_sidon.py --shards train-00042 --max-videos 12` (HTTP stream): 287 utterances,
  0.25 h; dropped 4 (duration), 3 (chars/s), 3 (text). `prepare/filter_teacher_cer.py` on a
  synthetic CER table: kept 191, dropped 96.
- `prepare/audiobooks.py`: row processing on a fake Parquet with the documented schema (CER/DNSMOS/
  char-rate filters, trnorm of "2024'te, %50"). The gated download itself was not run (no token).

## 6. Not covered here (needs the laptop)

CUDA training/fp16 and real VRAM use, the thermal guard against a real `nvidia-smi`, full-size
downloads (ISSAI 21.4 GB, ~70 Alania shards), and real throughput — check it with `tools/eta.py`
after the first ~15 minutes of training.
