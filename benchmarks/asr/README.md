# Turkish ASR benchmark

CPU benchmark of open Turkish speech-to-text systems on two public test sets.
Current numbers: [`results/LEADERBOARD.md`](results/LEADERBOARD.md) (machine-readable:
`results/summary.json`).

Decoding and scoring are separate steps. `run_asr.py` stores **raw** hypotheses; `score.py`
normalises reference and hypothesis with `trnorm.normalize_for_eval` at scoring time, so when
`trnorm` changes you only re-run `score.py` (seconds), never the decoders (hours).

Status (2026-10-07): whisper-tiny/base/small, duxx-zipformer, mihuai-turkish-stt and vosk-tr
are complete on both sets. whisper-large-v3-turbo is limited to the first 300 utterances of
each set (`--limit 300`, CPU budget); compare it with `score.py --common`, or finish it by
re-running without `--limit` (resumable). whisper-medium and buzzasr-turkish are wired up
(BuzzASR already converted, see below) but have not been run.

## Reproduce

```bash
source .venv/bin/activate
uv pip install vosk                      # only extra dependency (faster-whisper, sherpa-onnx, jiwer... are in the venv)

# 1. eval sets -> data/asr_eval/<set>/{manifest.jsonl,wav/,info.json}   (~2 min, ~0.5 GB)
python benchmarks/asr/prepare_sets.py

# 2. decode one system (resumable; --limit N decodes only the first N utterances)
OMP_NUM_THREADS=2 python benchmarks/asr/run_asr.py --set fleurs_tr_test --system whisper-small

#    ... or the whole queue (fastest first), one decoder at a time by default (PARALLEL=2 for two),
#    logs in benchmarks/asr/logs/<set>__<system>.log
nohup benchmarks/asr/run_queue.sh > benchmarks/asr/logs/queue.log 2>&1 &
JOBS=$'fleurs_tr_test whisper-large-v3-turbo --limit 300\nmediaspeech_tr buzzasr-turkish' benchmarks/asr/run_queue.sh

# 3. score everything -> results/summary.json + results/LEADERBOARD.md
python benchmarks/asr/score.py
python benchmarks/asr/score.py --common     # partial runs: score all systems on the same utterances (print only)
```

Models download on first use (HF cache for Hugging Face repos, `data/models/` for Vosk and
converted models). Every HF repo is pinned to a commit in `run_asr.py`.

## Files

| file | purpose |
|---|---|
| `prepare_sets.py` | builds the eval sets (16 kHz mono PCM16 WAV + JSONL manifest `{id, audio_path, ref_text, duration}`; `audio_path` is relative to the manifest) |
| `run_asr.py` | decodes a set with one system -> `results/<set>/<system>.jsonl` `{id, ref, hyp, duration, decode_seconds, cpu_seconds}` + `<system>.meta.json` (config, versions, CPU) |
| `run_queue.sh` | runs a list of `<set> <system> [args]` jobs, `PARALLEL` (default 1) at a time |
| `score.py` | corpus WER/CER after `trnorm.normalize_for_eval`, RTF; model metadata (params, license, streaming) lives in `MODEL_META` |

## Eval sets

| set | source (pinned revision) | utts | hours | license | notes |
|---|---|---:|---:|---|---|
| `fleurs_tr_test` | [google/fleurs](https://huggingface.co/datasets/google/fleurs) `tr_tr` test @ `70bb2e8` | 743 | 2.61 | CC-BY-4.0 | read Wikipedia sentences; ref = `raw_transcription` (cased, punctuated, digits), not FLEURS's own normalised column |
| `mediaspeech_tr` | [ymoslem/MediaSpeech](https://huggingface.co/datasets/ymoslem/MediaSpeech) `tr` @ `4008a96` | 500 | 1.99 | CC-BY-4.0 | YouTube media speech (news, talk shows); `random.Random(1234).sample` of the 2513 rows, sorted by row index; refs are already lower-case and unpunctuated, and segments are cut mid-word at boundaries, so WER has a floor well above 0 |

Both sets are used as test data only. Several systems were trained on FLEURS **train**
(duxx-zipformer, BuzzASR); their authors state FLEURS test was held out.

## Normalisation and metrics

- `trnorm.normalize_for_eval(text)` on both ref and hyp: Turkish-aware lower-casing
  (I->ı, İ->i), numbers -> Turkish words, punctuation stripped, whitespace collapsed (see
  `src/trnorm` for the exact rules, which are still evolving). `score.py` records the trnorm
  version and a hash of its source in `summary.json` and `LEADERBOARD.md`, so every number can
  be traced to the normaliser that produced it.
- WER / CER: corpus-level (total edits / total reference words or characters) via `jiwer`.
- RTF: `sum(decode_seconds) / sum(duration)`, wall-clock of the decode call only (audio in
  memory, model loaded, one warm-up decode excluded), 2 CPU threads per decoder
  (`cpu_threads=2` / `num_threads=2`, `OMP_NUM_THREADS=2`; Vosk is single-threaded), on a
  4-core VM (Intel Xeon @ 2.1 GHz) shared with other jobs; whisper-tiny and part of whisper-base
  ran two decoders at once, everything else one at a time. It is indicative only;
  `summary.json` also has `cpu_rtf` (process CPU seconds per audio second; recorded for
  utterances decoded after it was added, i.e. not for whisper-tiny), which is less sensitive
  to other load.

## Systems

| system | model (pinned revision) | params | license | streaming | decoding |
|---|---|---:|---|---|---|
| `whisper-tiny/base/small/medium` | `Systran/faster-whisper-*` (OpenAI Whisper) | 39M/74M/244M/769M | MIT | no | faster-whisper, int8, `language="tr"`, `beam_size=5`, `condition_on_previous_text=False`, no VAD, defaults otherwise (timestamps on, temperature fallback) |
| `whisper-large-v3-turbo` | `dropbox-dash/faster-whisper-large-v3-turbo` (faster-whisper's `large-v3-turbo` alias redirects there) | 809M | MIT | no | as above |
| `buzzasr-turkish` | [BuzzASR/turkish](https://huggingface.co/BuzzASR/turkish) @ `928b475`, whisper-large-v3 fine-tune (FLEURS train + Common Voice 25), converted to CTranslate2 | 1.55B | MIT | no | as above plus `without_timestamps=True` (its generation config has `return_timestamps=False`). Its card suggests greedy + `no_repeat_ngram_size=3` + `repetition_penalty=1.2`; we use the common Whisper settings instead |
| `duxx-zipformer` | [duxx/turkish-stt-zipformer](https://huggingface.co/duxx/turkish-stt-zipformer) @ `cebec19` | 23.6M | CC-BY-NC-4.0 | yes (chunk 32, left 128) | sherpa-onnx `OnlineRecognizer.from_transducer`, int8 enc/dec/joiner, modified beam search, 12 active paths, 1 s tail padding (as in the repo's `transcribe.py`) |
| `mihuai-turkish-stt` | [mihuai/turkish-stt](https://huggingface.co/mihuai/turkish-stt) @ `4f868b7` | 66M | mihu Community License (custom) | yes | sherpa-onnx `OnlineRecognizer.from_transducer`, `encoder.int8` + `decoder` (fp32) + `joiner.int8`, modified beam search (4 paths), 0.5 s tail padding (as in the repo's `eval/run_eval.py`) |
| `vosk-tr` | [vosk-model-small-tr-0.3](https://alphacephei.com/vosk/models) | ~3M AM + LM graph | Apache-2.0 | yes | `KaldiRecognizer`, 0.25 s chunks, segment results joined + `FinalResult()` |

Observed output conventions (scored as is):
- duxx-zipformer writes the suffix of an apostrophised proper noun as a separate word
  (`japonya nın`, `fransa yı` for `Japonya'nın`, `Fransa'yı`), which costs two word errors
  each against the apostrophe-joining normalisation.
- mihuai-turkish-stt and Whisper emit cased, punctuated text with digits; duxx-zipformer and
  vosk-tr emit lower-case words only, so any case-dependent rule in the normaliser (e.g.
  acronym expansion) affects the two groups differently.

License notes:
- **mihuai/turkish-stt** (mihu Community License): free only below USD 500k annual revenue,
  requires "Powered by mihu" attribution, and forbids using the model **or its outputs** to
  train competing models. It is evaluated here for research/benchmarking only. Never use its
  hypotheses (`results/*/mihuai-turkish-stt.jsonl`) as training data or pseudo-labels.
- **duxx/turkish-stt-zipformer** is CC-BY-NC-4.0 (non-commercial).

### BuzzASR conversion

Done once, in a throwaway venv so torch does not end up in the project venv:

```bash
uv venv /tmp/ct2conv --python 3.12
uv pip install --python /tmp/ct2conv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/ct2conv/bin/python "transformers>=4.57,<5" ctranslate2==4.8.2 accelerate
/tmp/ct2conv/bin/python -c "
from ctranslate2.converters import TransformersConverter as C
C('BuzzASR/turkish', revision='928b47596e31776dab9daf1e606ecf76234b9ecd',
  copy_files=['tokenizer.json', 'preprocessor_config.json'],
  load_as_float16=True, low_cpu_mem_usage=True).convert('data/models/buzzasr-turkish-ct2', quantization='int8')"
```

`preprocessor_config.json` must be copied: it sets the 128 mel bins large-v3 needs. int8
storage gives the same weights as fp16 storage quantised at load time (`compute_type="int8"`).
