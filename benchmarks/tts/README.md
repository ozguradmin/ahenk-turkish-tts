# Turkish TTS benchmark: real-world text and text-normalization front-ends

Open Turkish TTS models are close to the ASR floor on clean conversational text (Freya-TR-Eval), but
real text has numbers, dates, times, money, percentages, ordinals, abbreviations and acronyms, phone
numbers, units and very short utterances. This harness measures

1. how small, CPU-friendly open Turkish TTS systems cope with such text, and
2. how much a text-normalization front-end in front of them helps (`raw` vs the system's own
   normalizer vs `normalizer-tr` vs our `trnorm`).

Everything runs on CPU, at most 2 threads per process, one heavy process at a time.

## Method: ASR round trip

```
written text --front-end--> text for the system --TTS--> audio --faster-whisper--> hypothesis
                                                                                     |
reference: hand-checked spoken form (`spoken`, plus accepted `alts`) ----------------+--> WER / CER
```

- **Judge**: faster-whisper `large-v3-turbo` (`mobiuslabsgmbh/faster-whisper-large-v3-turbo`), int8,
  CPU, `cpu_threads=2`, `language="tr"`, `beam_size=5`, other options at their defaults
  (ctranslate2's seed is reset before every file, so even temperature fallback is repeatable).
  Audio is read with soundfile and resampled to 16 kHz with soxr in-process, with no 8 kHz
  band-limiting. faster-whisper 1.2.1 cannot open files with PyAV 19 (`metadata_errors` error), so
  it always gets numpy arrays.
- **Scoring**: `trnorm.normalize_for_eval` is applied to both the reference and the raw ASR hypothesis
  **at scoring time**, so `score.py` can be re-run whenever trnorm improves (it is being developed in
  parallel) without re-synthesizing or re-transcribing. Whisper usually writes spoken numbers as
  digits ("on beş" -> "15"); the eval normalizer turns them back into words, so scores depend on it.
  WER/CER are corpus-level (total edits / total reference words or characters; CER counts spaces).
  If a row has `alts` (alternative valid readings, e.g. "iki buçuk" vs "on dört otuz"), it is
  scored against whichever of `spoken` + `alts` gives the fewest word edits; `WER_strict` uses
  `spoken` only. `row_ok` = share of rows read with zero word errors.
- **Speed**: real-time factor RTF = synthesis wall-clock / audio duration, measured per utterance
  after a warm-up and excluding model loading; CPU-RTF = process CPU seconds / audio seconds.
  The machine is shared with other jobs and our steps run under `nice -n 10`, so treat the per-run
  RTF as indicative; `speed.py` gives a side-by-side probe on identical sentences.
- **Threads**: `common.py` caps OpenMP/MKL at 2 threads (`TTS_BENCH_THREADS`) and sets
  `OMP_WAIT_POLICY=PASSIVE`; piper's onnxruntime session gets `intra_op_num_threads=2` and no
  spinning. With the default spin-waiting, on this oversubscribed 4-core box (load ~9) EMA
  Lightning ran at RTF 1.37 instead of 0.26, and the spinning threads also took CPU from other jobs.
- **Determinism**: every utterance uses seed 0, set right before synthesis, so a row's audio does not
  depend on run order or on resuming.

## Systems

| system | model | params | out rate | licence | bundled normalizer |
|---|---|---|---:|---|---|
| `ema-lightning` | [canberkkkkkk/ema-lightning](https://huggingface.co/canberkkkkkk/ema-lightning) (`pip install ema-lightning`) | 8.6M | 48 kHz | Apache-2.0 (code and weights) | yes, separable: `EMA.say()` always runs normalizer-tr (policy `fallback`) |
| `pocket-tts-turkish` | [wite-tech/pocket-tts-turkish](https://huggingface.co/wite-tech/pocket-tts-turkish) (`pip install pocket-tts-turkish`), voice `female_1` | ~110M | 24 kHz | weights CC-BY-4.0; package Apache-2.0; pocket-tts library MIT | yes, separable: its own `normalizer.py` |
| `piper-tr-dfki` | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) `tr_TR-dfki-medium` (`pip install piper-tts`) | ~15M | 22.05 kHz | **voice: CC-BY-NC-SA-4.0** (dfki-ot-data); piper-tts GPL-3.0-or-later (bundles espeak-ng, GPL-3.0) | always on: espeak-ng reads digits, so `raw` == `native` |
| `mms-tts-tur` | [facebook/mms-tts-tur](https://huggingface.co/facebook/mms-tts-tur) (transformers `VitsModel`) | 36M | 16 kHz | **CC-BY-NC-4.0** | none |
| `qwen3-tts-tiny-tr` | [erkamk/qwen3-tts-tiny-tr](https://huggingface.co/erkamk/qwen3-tts-tiny-tr) (`pip install qwen-tts`) | 199M talker + 171M codec | 24 kHz | Apache-2.0 (weights and qwen-tts) | none |

The non-commercial licences (piper dfki voice, MMS) allow research benchmarking; the generated audio
stays under `data/tts/` (gitignored) and is not redistributed.

**Opt-in: `piper-antalia`**, our own Piper voice from [`tts/`](../../tts) (LJSpeech medium fine-tuned on
the CC-BY-4.0 Antalia corpus; weights CC-BY-4.0). It is registered only when `PIPER_ANTALIA_ONNX` points
to an exported `.onnx` (its `.onnx.json` next to it, or `PIPER_ANTALIA_CONFIG`), so the default system
list and all existing results are unchanged. Set the variable for `synthesize.py`, `judge.py` and
`score.py`; the ONNX file's sha256 is part of the system config, so a newly exported voice is
synthesized again instead of reusing cached rows:

```bash
export PIPER_ANTALIA_ONNX=$PWD/antalia-export/tr_TR-antalia-medium.onnx
SYSTEMS="piper-antalia" FRONTENDS="raw trnorm" benchmarks/tts/run_all.sh tr_zor freya_sample40
```

How each one is wrapped (`systems.py`), and what each does to the text in every mode
(this is input encoding the model needs, not normalization):

- **ema-lightning**: `EMA(device="cpu").say(text, seed=0)`. EMA's `Frontend` = `alphabet(spoken(text))`,
  where `spoken()` is normalizer-tr (`ambiguity_policy="fallback"`) and `alphabet()` does Turkish
  lowercasing and turns every character outside the model's vocabulary (letters and `!"%&'(),-./:;?`)
  into a space, so digits disappear. For every front-end except `native` the wrapper replaces
  `spoken` with the identity. Gotchas: the first `say()` in a fresh environment spends ~20 s
  measuring a batch size (cached in `~/.cache/ema_lightning/`); the `normalizer_tr_fallback`
  front-end gives exactly the same model input as `native`.
- **pocket-tts-turkish**: `TurkishTTS.from_pretrained(device="cpu").generate(text, voice="female_1",
  seed=0)`. `native` = the package's `TextFrontend` (normalizer.py, then the alphabet fit, which also
  spells leftover digits one by one, then a sentence split and lowercasing of a capitalized first
  word). For other front-ends the wrapper uses the same frontend without normalizer.py and without
  the digit spelling: w/x/q are transliterated, characters the SentencePiece model lacks (all digits,
  `%`, ...) become spaces. Gotcha: pocket-tts sets torch to 1 thread; with 2 threads it ran ~3x
  slower on this shared machine (RTF 1.8 vs 0.62 under load ~6.5), so it runs at its own default of
  1 thread.
- **piper-tr-dfki**: `PiperVoice.load(...).synthesize(text)` with the voice's default noise/length
  scales; sentence chunks are concatenated. espeak-ng (bundled in piper-tts 1.8) phonemizes and reads
  digits itself (`01.11.2026` -> "sıfır bir nokta on bir nokta iki bin yirmi altı"), so there is no
  separate `native` run. Gotcha: piper's ONNX graph has unseeded `RandomNormalLike` ops that are
  seeded when the session is created, so the wrapper calls `onnxruntime.set_seed(0)` and re-creates
  the session before every utterance (outside the timer) to make output deterministic.
- **mms-tts-tur**: transformers `VitsModel` + `VitsTokenizer`, `torch.manual_seed(0)` per utterance.
  The tokenizer lowercases with Python's `str.lower()` (which maps `I` to `i`, wrong for Turkish), so
  the wrapper first maps `I` to `ı` and `İ` to `i`. It then drops out-of-vocabulary characters; the
  vocabulary has the digits 0-4 and 6 but not 5, 7, 8, 9, so `raw` digits are partly kept and partly
  dropped.
- **qwen3-tts-tiny-tr**: `Qwen3TTSModel.from_pretrained(..., device_map="cpu", dtype=float32)`, voice
  from an x-vector of the first 10 s of the repo's own `samples/paragraph-01.wav` (the model needs a
  reference recording; none ships with it), `language="Auto"`, model-card sampling (temperature 0.7,
  subtalker temperature 0.6, repetition penalty 1.0, `max_new_tokens = 25 + 2 * len(text)`),
  `torch.manual_seed(0)`. As the card requires, `I` becomes `ı` and `İ` becomes `i`. It runs at
  RTF ~2 to 4 on 2 CPU threads, so on `tr_zor` it is run on a seeded, category-stratified subset
  (`QWEN_SAMPLE_tr_zor=65`, which gives 63 rows).

## Front-ends (`frontends.py`)

| front-end | text given to the system | system's bundled normalizer |
|---|---|---|
| `raw` | as written | bypassed (piper: cannot be) |
| `native` | as written | on (only EMA and pocket; `= raw` for piper, `n/a` for MMS and Qwen) |
| `normalizer_tr` | `Normalizer().normalize(t).normalized_text` (normalizer-tr 0.4.0, default `ambiguity_policy="preserve"`) | bypassed |
| `normalizer_tr_fallback` | `Normalizer().normalize(t, ambiguity_policy="fallback").normalized_text` | bypassed |
| `trnorm` | `trnorm.normalize(t)` | bypassed |

Note that normalizer-tr's default `preserve` policy leaves anything it finds ambiguous **as written**
(dates like `01.11.2026`, times like `07:45'te`, `3.` ordinals, `1990'lı`), and those digits then
reach the model. `fallback` is what EMA Lightning uses in production (it reads everything, sometimes
literally: `1990'lı` -> "bin dokuz yüz doksan kesme lı"), so it is included as a fifth front-end for a
fair comparison. If a front-end raises on a row, the raw text is used and `frontend_error` is recorded.

## Test sets (`sets/*.jsonl`: `id`, `category`, `text`, `spoken`, optional `alts`)

| set | rows | content | licence |
|---|---:|---|---|
| `freya_sample40` | 40 | seeded sample of [Freya-TR-Eval](https://huggingface.co/datasets/freyavoice/freya-tr-eval) (33 everyday + 7 short-native), no digits; `spoken = text`. Rebuild with `make_dev_sets.py` | CC-BY-4.0 |
| `dev_numbers10` | 10 | hand-written number-heavy sentences, one per category (cardinal, date, time, money, percent, ordinal, phone, abbreviation, unit, short) with careful spoken forms; abbreviations are expanded (`TL` -> "Türk lirası", `Cad.` -> "Caddesi") | ours |
| `tr_zor` | 155 | the hard set (13 categories, `alts` for alternative readings) | ours |

## Running

```bash
cd <repo>   # repository root
uv venv -p 3.12 .venv-tts
uv pip install -p .venv-tts/bin/python --torch-backend cpu -e . "torch==2.11.0" "torchaudio==2.11.0" \
    ema-lightning pocket-tts-turkish piper-tts normalizer-tr faster-whisper "transformers==4.57.3" \
    qwen-tts soundfile soxr numpy jiwer
```

Models download on first use into `data/tts/hf_home` (`HF_HOME` is set by `common.py`).

Everything for a set, resumable, strictly one process at a time (<= 2 threads, `nice -n 10`):
for each system it synthesizes every front-end, transcribes that system's audio, and re-scores, so
results fill in system by system.

```bash
benchmarks/tts/run_all.sh dev_numbers10 freya_sample40
QWEN_SAMPLE_tr_zor=65 nohup benchmarks/tts/run_all.sh tr_zor > benchmarks/tts/logs/run_tr_zor.log 2>&1 &
# subsets of systems / front-ends:
SYSTEMS="ema-lightning piper-tr-dfki" FRONTENDS="raw trnorm" benchmarks/tts/run_all.sh tr_zor
```

Single steps:

```bash
PY=.venv-tts/bin/python
$PY benchmarks/tts/synthesize.py --set tr_zor --system ema-lightning --frontend trnorm   # [--limit N|--sample N|--ids a,b]
$PY benchmarks/tts/judge.py --set tr_zor                    # [--systems ...] [--frontends ...]
$PY benchmarks/tts/score.py --set tr_zor dev_numbers10      # re-run after every trnorm change
$PY benchmarks/tts/speed.py                                 # RTF probe, 1 and 2 threads per system
```

After trnorm changes, `score.py` alone re-scores everything (the eval normalizer changed). To also
refresh the `trnorm` front-end audio, re-run `run_all.sh` (or `synthesize.py ... --frontend trnorm`
and `judge.py`): only rows whose trnorm output changed are synthesized and transcribed again.

`synthesize.py` skips rows whose record, wav and front-end text are unchanged. When a front-end gives
a system exactly the same model input as another front-end already did (e.g. all front-ends on most
Freya sentences), the deterministic audio is copied instead of re-synthesized (`reused_from`), and
`judge.py` transcribes each distinct audio file only once. RTF is computed only on rows that were
actually synthesized.

## Outputs

```
data/tts/out/<set>/<system>/<frontend>/<id>.wav            audio (16-bit PCM, system's own rate)
benchmarks/tts/results/<set>/synth/<system>__<frontend>.jsonl   per row: text, frontend_text,
        model_text (exact model input after the system's own preparation; pocket: sentences joined
        by " | "), extra (piper: espeak phonemes), duration, synth_seconds, synth_cpu_seconds, rtf,
        loadavg, audio_sha1, reused_from, frontend_error, error
benchmarks/tts/results/<set>/synth/<system>__<frontend>.meta.json   config, versions, threads, CPU
benchmarks/tts/results/<set>/hyps/<system>__<frontend>.jsonl    raw ASR hypothesis per row
benchmarks/tts/results/<set>/hyps/judge.meta.json               judge model/revision/options
benchmarks/tts/results/<set>/summary.json                       all metrics (overall, per category, speed)
benchmarks/tts/results/<set>/RESULTS.md                         tables
benchmarks/tts/results/<set>/rows.jsonl                         per-row normalized ref/hyp and edits
        (these three are rewritten atomically, via a temp file and rename, on every score.py run)
benchmarks/tts/results/speed/SPEED.md, speed.jsonl              speed probe
benchmarks/tts/logs/                                            run logs (gitignored)
```

## Caveats

- WER measures what Whisper understood, so part of every number is the recogniser's error. On clean
  Freya text this floor is roughly 1 to 3 % WER for good systems. Number rows also depend on how
  `normalize_for_eval` maps Whisper's digits back to words.
- A front-end can be "right" in a way the reference does not list (e.g. "on yedi otuz" vs "beş buçuk"
  for 17:30); `alts` cover the common cases in `tr_zor`. `dev_numbers10` has no alts, so prefer
  `tr_zor` for conclusions.
- One voice and one seed per system: differences of one or two words on a small set are noise.
- `tr_zor` is not a blind test for `trnorm`: the set was added together with trnorm refinements
  (commit dac82ac), and the same package also provides the scoring normalizer
  (`normalize_for_eval`), so its reading conventions match the references by construction. Confirm
  `trnorm` gains on held-out text before quoting them as general.
- Audio is resampled to 16 kHz without the 8 kHz band-limiting of the Freya-TR-Eval recipe, the judge
  is large-v3-turbo (not large-v3) and the eval normalizer is trnorm, so absolute WERs are not
  directly comparable with the numbers on the model cards.
- On clean text (`freya_sample40`, no digits) every front-end gives the same model input, so all
  columns are identical per system: the front-ends do no harm there.
