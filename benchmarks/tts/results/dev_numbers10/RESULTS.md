# TTS round-trip benchmark: `dev_numbers10`

Generated 2026-10-07 06:25 UTC by `benchmarks/tts/score.py`. 10 rows, 10 categories. Judge: faster-whisper `large-v3-turbo` int8 on CPU, beam 5, language tr. Both the reference and the hypothesis go through `trnorm.normalize_for_eval` (trnorm 0.1.0, src sha256 0dfc83a182f9). WER/CER are corpus-level; reference = `spoken`. Lower is better.

## WER % (CER %) by system and front-end

| system | raw | native | normalizer_tr | normalizer_tr_fallback | trnorm |
|---|---:|---:|---:|---:|---:|
| ema-lightning | 55.8 (43.0) | 6.2 (3.8) | 24.8 (18.3) | 6.2 (3.8) | 1.8 (1.2) |
| pocket-tts-turkish | 55.8 (43.1) | 3.5 (2.2) | 27.4 (20.4) | 9.7 (5.9) | 1.8 (1.2) |
| piper-tr-dfki | 6.2 (4.6) | = raw | 4.4 (2.8) | 7.1 (4.1) | 1.8 (1.2) |
| mms-tts-tur | 62.8 (39.9) | n/a | 31.9 (18.6) | 11.5 (6.5) | 9.7 (5.6) |
| qwen3-tts-tiny-tr | 58.4 (36.9) | n/a | 31.0 (19.1) | 11.5 (6.9) | 1.8 (1.2) |

`= raw`: the system always applies its own normalization (piper: espeak-ng reads digits), so `raw` already is `native`. `n/a`: the system has no bundled normalizer. `[n=…]`: scored on a subset only.

### Rows read without any word error (%)

| system | raw | native | normalizer_tr | normalizer_tr_fallback | trnorm |
|---|---:|---:|---:|---:|---:|
| ema-lightning | 0.0 | 60.0 | 40.0 | 60.0 | 90.0 |
| pocket-tts-turkish | 0.0 | 80.0 | 40.0 | 60.0 | 90.0 |
| piper-tr-dfki | 60.0 | = raw | 70.0 | 60.0 | 90.0 |
| mms-tts-tur | 0.0 | n/a | 30.0 | 50.0 | 70.0 |
| qwen3-tts-tiny-tr | 0.0 | n/a | 40.0 | 60.0 | 90.0 |

## WER % per category

| system | front-end | all | cardinal | date | time | money | percent | ordinal | phone | abbreviation | unit | short |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ema-lightning | raw | 55.8 | 64.3 | 60.0 | 54.5 | 63.6 | 55.6 | 36.4 | 72.2 | 18.2 | 61.5 | 60.0 |
| ema-lightning | native | 6.2 | 14.3 | 0.0 | 0.0 | 18.2 | 0.0 | 18.2 | 0.0 | 9.1 | 0.0 | 0.0 |
| ema-lightning | normalizer_tr | 24.8 | 64.3 | 60.0 | 54.5 | 18.2 | 0.0 | 36.4 | 0.0 | 9.1 | 0.0 | 0.0 |
| ema-lightning | normalizer_tr_fallback | 6.2 | 14.3 | 0.0 | 0.0 | 18.2 | 0.0 | 18.2 | 0.0 | 9.1 | 0.0 | 0.0 |
| ema-lightning | trnorm | 1.8 | 0.0 | 0.0 | 0.0 | 18.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| pocket-tts-turkish | raw | 55.8 | 64.3 | 60.0 | 54.5 | 63.6 | 55.6 | 27.3 | 72.2 | 27.3 | 61.5 | 60.0 |
| pocket-tts-turkish | native | 3.5 | 0.0 | 0.0 | 0.0 | 18.2 | 0.0 | 18.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| pocket-tts-turkish | normalizer_tr | 27.4 | 64.3 | 60.0 | 54.5 | 0.0 | 0.0 | 27.3 | 33.3 | 9.1 | 0.0 | 0.0 |
| pocket-tts-turkish | normalizer_tr_fallback | 9.7 | 14.3 | 0.0 | 0.0 | 0.0 | 0.0 | 18.2 | 33.3 | 9.1 | 0.0 | 0.0 |
| pocket-tts-turkish | trnorm | 1.8 | 0.0 | 0.0 | 0.0 | 18.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| piper-tr-dfki | raw | 6.2 | 0.0 | 0.0 | 0.0 | 18.2 | 0.0 | 18.2 | 0.0 | 18.2 | 7.7 | 0.0 |
| piper-tr-dfki | normalizer_tr | 4.4 | 0.0 | 0.0 | 0.0 | 18.2 | 0.0 | 18.2 | 0.0 | 9.1 | 0.0 | 0.0 |
| piper-tr-dfki | normalizer_tr_fallback | 7.1 | 14.3 | 0.0 | 0.0 | 18.2 | 0.0 | 27.3 | 0.0 | 9.1 | 0.0 | 0.0 |
| piper-tr-dfki | trnorm | 1.8 | 0.0 | 0.0 | 0.0 | 18.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| mms-tts-tur | raw | 62.8 | 64.3 | 70.0 | 72.7 | 72.7 | 55.6 | 27.3 | 77.8 | 27.3 | 69.2 | 100.0 |
| mms-tts-tur | normalizer_tr | 31.9 | 64.3 | 70.0 | 72.7 | 9.1 | 0.0 | 27.3 | 38.9 | 9.1 | 0.0 | 0.0 |
| mms-tts-tur | normalizer_tr_fallback | 11.5 | 14.3 | 0.0 | 0.0 | 9.1 | 0.0 | 18.2 | 38.9 | 9.1 | 0.0 | 0.0 |
| mms-tts-tur | trnorm | 9.7 | 0.0 | 0.0 | 0.0 | 18.2 | 0.0 | 0.0 | 38.9 | 18.2 | 0.0 | 0.0 |
| qwen3-tts-tiny-tr | raw | 58.4 | 71.4 | 60.0 | 63.6 | 63.6 | 55.6 | 27.3 | 72.2 | 18.2 | 69.2 | 80.0 |
| qwen3-tts-tiny-tr | normalizer_tr | 31.0 | 71.4 | 60.0 | 63.6 | 0.0 | 0.0 | 27.3 | 44.4 | 9.1 | 0.0 | 0.0 |
| qwen3-tts-tiny-tr | normalizer_tr_fallback | 11.5 | 14.3 | 0.0 | 0.0 | 0.0 | 0.0 | 18.2 | 44.4 | 9.1 | 0.0 | 0.0 |
| qwen3-tts-tiny-tr | trnorm | 1.8 | 0.0 | 0.0 | 0.0 | 18.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Speed (CPU, <= 2 threads)

| system | params | out rate | threads | mean audio s | RTF (wall) | CPU-RTF | n timed |
|---|---|---:|---:|---:|---:|---:|---:|
| ema-lightning | 8.6M (5.6M DiT + 3M decoder) | 48000 | 2 | 3.65 | 0.166 | 0.185 | 25 |
| pocket-tts-turkish | ~110M (incl. codec) | 24000 | 1 | 4.38 | 0.512 | 0.706 | 28 |
| piper-tr-dfki | ~15M (VITS medium) | 22050 | 2 | 4.43 | 0.066 | 0.110 | 25 |
| mms-tts-tur | 36M (VITS) | 16000 | 2 | 5.11 | 0.320 | 0.527 | 24 |
| qwen3-tts-tiny-tr | 199M talker + 171M speech tokenizer | 24000 | 2 | 6.09 | 2.753 | 3.780 | 25 |

RTF = synthesis wall-clock / audio duration over rows actually synthesized (rows whose model input equals another front-end's are copied, not re-timed), after a warm-up, on Intel(R) Xeon(R) Processor @ 2.10GHz with torch/onnxruntime capped at 2 threads (pocket-tts keeps its own default of 1), `OMP_WAIT_POLICY=PASSIVE`, under `nice` on a machine shared with other jobs, so treat RTF as indicative; CPU-RTF (process CPU seconds / audio seconds, summed over threads) is less sensitive to contention. `results/speed/SPEED.md` has a side-by-side probe.

## Front-ends

- `raw`: text as written, system's bundled normalizer bypassed
- `native`: text as written, system's bundled normalizer on
- `normalizer_tr`: normalizer-tr, default policy (preserve)
- `normalizer_tr_fallback`: normalizer-tr, ambiguity_policy='fallback'
- `trnorm`: trnorm.normalize

## Systems

| system | model | params | licence | bundled normalizer |
|---|---|---|---|---|
| ema-lightning | canberkkkkkk/ema-lightning | 8.6M (5.6M DiT + 3M decoder) | Apache-2.0 (code and weights) | separable: EMA always runs normalizer-tr (policy 'fallback') in EMA.say(); raw replaces Frontend.spoken with the identity |
| pocket-tts-turkish | wite-tech/pocket-tts-turkish | ~110M (incl. codec) | CC-BY-4.0 (weights); pocket-tts-turkish package Apache-2.0 | separable: package TextFrontend = normalizer.py + alphabet fit (+ stray digits spelled one by one) |
| piper-tr-dfki | rhasspy/piper-voices tr/tr_TR/dfki/medium | ~15M (VITS medium) | voice data CC-BY-NC-SA-4.0 (dfki-ot-data); piper-tts code GPL-3.0-or-later (bundles espeak-ng, GPL-3.0) | always_on: espeak-ng's Turkish rules read digits, '%', thousands/decimal separators internally; cannot be bypassed |
| mms-tts-tur | facebook/mms-tts-tur | 36M (VITS) | CC-BY-NC-4.0 | none: no normalizer; VitsTokenizer lowercases and drops out-of-vocabulary characters |
| qwen3-tts-tiny-tr | erkamk/qwen3-tts-tiny-tr | 199M talker + 171M speech tokenizer | Apache-2.0 (weights, inherited from Qwen3-TTS); qwen-tts code Apache-2.0 | none: no normalizer; model card: write numbers as words and I/İ as ı/i |

Per-row details (front-end text, model text, hypothesis, normalized reference/hypothesis, edits): `rows.jsonl` next to this file.
