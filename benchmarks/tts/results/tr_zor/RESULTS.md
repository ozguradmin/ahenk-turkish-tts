# TTS round-trip benchmark: `tr_zor`

Generated 2026-10-07 19:02 UTC by `benchmarks/tts/score.py`. 155 rows, 13 categories. Judge: faster-whisper `large-v3-turbo` int8 on CPU, beam 5, language tr. Both the reference and the hypothesis go through `trnorm.normalize_for_eval` (trnorm 0.1.0, src sha256 0dfc83a182f9). WER/CER are corpus-level; each row is scored against the closest of `spoken` and its `alts`. Lower is better.

## WER % (CER %) by system and front-end

| system | raw | native | normalizer_tr | normalizer_tr_fallback | trnorm |
|---|---:|---:|---:|---:|---:|
| ema-lightning | 49.7 (35.2) | 14.0 (8.6) | 28.3 (20.1) | 14.0 (8.6) | 1.9 (0.7) |
| pocket-tts-turkish | 46.5 (34.2) | 10.2 (6.3) | 27.7 (20.2) | 16.1 (9.9) | 3.2 (1.1) |
| piper-tr-dfki | 13.3 (6.8) | = raw | 12.1 (6.0) | 15.6 (8.6) | 3.5 (0.7) |
| mms-tts-tur | 55.1 (34.3) | n/a | 36.2 (21.0) | 24.6 (12.8) | 9.9 (2.8) |
| qwen3-tts-tiny-tr | 57.8 (32.5) [n=63] | n/a | 36.3 (20.0) [n=63] | 15.6 (8.7) [n=63] | 1.9 (0.5) [n=63] |
| piper-antalia | 41.2 (21.6) | = raw | – | 38.8 (21.0) | 24.6 (10.2) |

`= raw`: the system always applies its own normalization (piper: espeak-ng reads digits), so `raw` already is `native`. `n/a`: the system has no bundled normalizer. `[n=…]`: scored on a subset only.

### Strict WER % (reference = `spoken` only, alts ignored)

| system | raw | native | normalizer_tr | normalizer_tr_fallback | trnorm |
|---|---:|---:|---:|---:|---:|
| ema-lightning | 50.3 | 14.8 | 29.6 | 14.8 | 2.4 |
| pocket-tts-turkish | 47.2 | 11.0 | 29.0 | 16.9 | 3.6 |
| piper-tr-dfki | 14.6 | = raw | 13.6 | 16.4 | 4.1 |
| mms-tts-tur | 55.6 | n/a | 37.3 | 26.3 | 10.3 |
| qwen3-tts-tiny-tr | 58.3 [n=63] | n/a | 36.9 [n=63] | 15.8 [n=63] | 1.9 [n=63] |
| piper-antalia | 42.7 | = raw | – | 40.0 | 24.8 |

### Rows read without any word error (%)

| system | raw | native | normalizer_tr | normalizer_tr_fallback | trnorm |
|---|---:|---:|---:|---:|---:|
| ema-lightning | 22.6 | 69.0 | 55.5 | 69.0 | 94.8 |
| pocket-tts-turkish | 23.9 | 72.9 | 52.3 | 62.6 | 88.4 |
| piper-tr-dfki | 61.3 | = raw | 67.1 | 60.6 | 87.1 |
| mms-tts-tur | 14.2 | n/a | 33.5 | 43.2 | 62.6 |
| qwen3-tts-tiny-tr | 23.8 [n=63] | n/a | 52.4 [n=63] | 68.2 [n=63] | 93.7 [n=63] |
| piper-antalia | 18.1 | = raw | – | 25.8 | 34.2 |

### Same 63 rows as `qwen3-tts-tiny-tr` (it ran on a subset): WER % (CER %)

| system | raw | native | normalizer_tr | normalizer_tr_fallback | trnorm |
|---|---:|---:|---:|---:|---:|
| ema-lightning | 50.0 (35.9) | 11.4 (7.1) | 31.4 (21.9) | 11.4 (7.1) | 0.8 (0.1) |
| pocket-tts-turkish | 46.7 (35.4) | 9.0 (5.7) | 28.4 (21.2) | 15.1 (9.1) | 2.6 (0.5) |
| piper-tr-dfki | 9.8 (4.7) | = raw | 9.3 (4.2) | 13.8 (7.6) | 4.5 (0.9) |
| mms-tts-tur | 55.0 (35.3) | n/a | 36.7 (22.1) | 21.2 (11.6) | 7.1 (2.2) |
| qwen3-tts-tiny-tr | 57.8 (32.5) | n/a | 36.3 (20.0) | 15.6 (8.7) | 1.9 (0.5) |
| piper-antalia | 33.2 (17.3) | = raw | – | 34.5 (17.6) | 18.7 (7.2) |

## WER % per category

| system | front-end | all | sayi | sira | tarih | saat | para | yuzde | birim | numara | kisaltma | akronim | kisa | soru | ozel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ema-lightning | raw | 49.7 | 45.4 | 36.1 | 62.6 | 54.1 | 66.2 | 46.1 | 59.7 | 76.9 | 45.7 | 47.1 | 4.0 | 0.0 | 0.0 |
| ema-lightning | native | 14.0 | 11.2 | 19.7 | 9.2 | 30.6 | 9.9 | 0.0 | 11.0 | 14.3 | 25.7 | 27.6 | 4.0 | 0.0 | 0.0 |
| ema-lightning | normalizer_tr | 28.3 | 17.4 | 18.0 | 48.6 | 46.8 | 16.1 | 3.9 | 9.6 | 61.5 | 37.1 | 29.9 | 4.0 | 0.0 | 0.0 |
| ema-lightning | normalizer_tr_fallback | 14.0 | 11.2 | 19.7 | 9.2 | 30.6 | 9.9 | 0.0 | 11.0 | 14.3 | 25.7 | 27.6 | 4.0 | 0.0 | 0.0 |
| ema-lightning | trnorm | 1.9 | 0.0 | 3.3 | 0.0 | 0.0 | 1.2 | 0.0 | 2.7 | 4.2 | 0.0 | 6.9 | 4.0 | 0.0 | 0.0 |
| pocket-tts-turkish | raw | 46.5 | 46.5 | 34.4 | 62.6 | 62.3 | 65.0 | 48.1 | 56.2 | 74.4 | 41.4 | 11.4 | 12.0 | 3.0 | 6.8 |
| pocket-tts-turkish | native | 10.2 | 0.0 | 9.8 | 8.3 | 11.1 | 12.3 | 0.0 | 4.1 | 5.9 | 15.7 | 36.4 | 12.0 | 3.0 | 9.1 |
| pocket-tts-turkish | normalizer_tr | 27.7 | 16.3 | 18.0 | 48.6 | 53.2 | 24.7 | 3.9 | 10.8 | 58.1 | 34.3 | 11.4 | 12.0 | 3.0 | 6.8 |
| pocket-tts-turkish | normalizer_tr_fallback | 16.1 | 11.2 | 14.8 | 8.3 | 38.7 | 18.5 | 0.0 | 11.0 | 13.4 | 25.7 | 34.5 | 12.0 | 3.0 | 6.8 |
| pocket-tts-turkish | trnorm | 3.2 | 0.0 | 0.0 | 0.0 | 1.6 | 0.0 | 0.0 | 0.0 | 8.4 | 0.0 | 12.6 | 12.0 | 3.0 | 6.8 |
| piper-tr-dfki | raw | 13.3 | 2.2 | 18.0 | 0.0 | 8.1 | 3.8 | 11.5 | 25.0 | 16.5 | 44.3 | 23.6 | 4.0 | 3.0 | 4.5 |
| piper-tr-dfki | normalizer_tr | 12.1 | 1.1 | 16.4 | 0.0 | 8.1 | 14.8 | 3.9 | 4.1 | 16.5 | 42.9 | 25.8 | 4.0 | 3.0 | 4.5 |
| piper-tr-dfki | normalizer_tr_fallback | 15.6 | 12.4 | 18.0 | 5.5 | 33.9 | 17.3 | 3.9 | 8.2 | 11.0 | 31.4 | 35.6 | 4.0 | 3.0 | 4.5 |
| piper-tr-dfki | trnorm | 3.5 | 1.1 | 0.0 | 0.0 | 9.5 | 0.0 | 3.9 | 1.4 | 0.0 | 5.7 | 16.1 | 4.0 | 3.0 | 4.5 |
| mms-tts-tur | raw | 55.1 | 48.8 | 34.4 | 66.4 | 60.7 | 65.0 | 51.9 | 62.5 | 82.9 | 52.9 | 54.0 | 16.0 | 9.1 | 22.7 |
| mms-tts-tur | normalizer_tr | 36.2 | 25.6 | 19.7 | 50.5 | 56.5 | 25.9 | 11.5 | 12.3 | 68.4 | 44.3 | 43.7 | 16.0 | 9.1 | 22.7 |
| mms-tts-tur | normalizer_tr_fallback | 24.6 | 22.5 | 27.9 | 14.8 | 35.5 | 23.5 | 7.7 | 6.8 | 28.9 | 41.4 | 44.8 | 16.0 | 9.1 | 22.7 |
| mms-tts-tur | trnorm | 9.9 | 9.0 | 6.6 | 4.6 | 6.3 | 6.2 | 7.7 | 0.0 | 11.8 | 11.4 | 24.1 | 16.0 | 9.1 | 22.7 |
| qwen3-tts-tiny-tr | raw | 57.8 | 57.9 | 51.6 | 79.2 | 96.0 | 76.5 | 68.4 | 53.1 | 78.8 | 40.0 | 33.3 | 0.0 | 0.0 | 0.0 |
| qwen3-tts-tiny-tr | normalizer_tr | 36.3 | 23.7 | 16.1 | 54.2 | 96.0 | 44.1 | 0.0 | 21.9 | 65.4 | 28.0 | 30.3 | 0.0 | 0.0 | 0.0 |
| qwen3-tts-tiny-tr | normalizer_tr_fallback | 15.6 | 0.0 | 6.5 | 6.2 | 44.0 | 20.6 | 0.0 | 22.6 | 22.2 | 28.0 | 30.3 | 0.0 | 0.0 | 0.0 |
| qwen3-tts-tiny-tr | trnorm | 1.9 | 0.0 | 0.0 | 2.1 | 3.9 | 0.0 | 0.0 | 0.0 | 3.7 | 0.0 | 9.1 | 0.0 | 0.0 | 0.0 |
| piper-antalia | raw | 41.2 | 21.4 | 49.2 | 25.9 | 46.0 | 22.5 | 32.7 | 43.8 | 45.5 | 74.3 | 48.9 | 40.0 | 33.3 | 68.2 |
| piper-antalia | normalizer_tr_fallback | 38.8 | 23.6 | 26.2 | 17.6 | 55.6 | 40.7 | 26.9 | 29.7 | 33.6 | 57.1 | 70.0 | 40.0 | 33.3 | 65.9 |
| piper-antalia | trnorm | 24.6 | 18.0 | 21.3 | 12.0 | 19.1 | 22.5 | 15.4 | 13.7 | 20.2 | 27.1 | 44.3 | 40.0 | 33.3 | 68.2 |

## Speed (CPU, <= 2 threads)

| system | params | out rate | threads | mean audio s | RTF (wall) | CPU-RTF | n timed |
|---|---|---:|---:|---:|---:|---:|---:|
| ema-lightning | 8.6M (5.6M DiT + 3M decoder) | 48000 | 2 | 2.02 | 0.204 | 0.185 | 337 |
| pocket-tts-turkish | ~110M (incl. codec) | 24000 | 1 | 2.60 | 0.527 | 0.720 | 389 |
| piper-tr-dfki | ~15M (VITS medium) | 22050 | 2 | 2.51 | 0.065 | 0.103 | 340 |
| mms-tts-tur | 36M (VITS) | 16000 | 2 | 3.03 | 0.335 | 0.502 | 336 |
| qwen3-tts-tiny-tr | 199M talker + 171M speech tokenizer | 24000 | 2 | 3.01 | 2.426 | 3.298 | 135 |
| piper-antalia | 15.7M (VITS medium, fine-tuned from LJSpeech medium) | 22050 | 1/2 | 2.39 | 0.097 | 0.125 | 432 |

RTF = synthesis wall-clock / audio duration over rows actually synthesized (rows whose model input equals another front-end's are copied, not re-timed), after a warm-up, on Intel(R) Xeon(R) Processor @ 2.80GHz with torch/onnxruntime capped at 2 threads (pocket-tts keeps its own default of 1), `OMP_WAIT_POLICY=PASSIVE`, under `nice` on a machine shared with other jobs, so treat RTF as indicative; CPU-RTF (process CPU seconds / audio seconds, summed over threads) is less sensitive to contention. `results/speed/SPEED.md` has a side-by-side probe.

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
| piper-antalia | local export of tts/ (tr_TR-antalia-medium) | 15.7M (VITS medium, fine-tuned from LJSpeech medium) | weights CC-BY-4.0 (Antalia voice corpus CC-BY-4.0, base LJSpeech public domain); piper-tts GPL-3.0-or-later (bundles espeak-ng, GPL-3.0) | always_on: espeak-ng's Turkish rules read digits, '%', thousands/decimal separators internally; cannot be bypassed |

Per-row details (front-end text, model text, hypothesis, normalized reference/hypothesis, edits): `rows.jsonl` next to this file.
