# TTS round-trip benchmark: `freya_sample40`

Generated 2026-10-07 06:25 UTC by `benchmarks/tts/score.py`. 40 rows, 2 categories. Judge: faster-whisper `large-v3-turbo` int8 on CPU, beam 5, language tr. Both the reference and the hypothesis go through `trnorm.normalize_for_eval` (trnorm 0.1.0, src sha256 0dfc83a182f9). WER/CER are corpus-level; reference = `spoken`. Lower is better.

## WER % (CER %) by system and front-end

| system | raw | native | normalizer_tr | normalizer_tr_fallback | trnorm |
|---|---:|---:|---:|---:|---:|
| ema-lightning | 0.6 (0.1) | 0.6 (0.1) | 0.6 (0.1) | 0.6 (0.1) | 0.6 (0.1) |
| pocket-tts-turkish | 1.9 (0.6) | 1.9 (0.6) | 1.9 (0.6) | 1.9 (0.6) | 1.9 (0.6) |
| piper-tr-dfki | 3.2 (0.5) | = raw | 3.2 (0.5) | 3.2 (0.5) | 3.2 (0.5) |
| mms-tts-tur | 8.0 (1.6) | n/a | 8.0 (1.6) | 8.0 (1.6) | 8.0 (1.6) |
| qwen3-tts-tiny-tr | 2.2 (0.4) | n/a | 2.2 (0.4) | 2.2 (0.4) | 2.2 (0.4) |

`= raw`: the system always applies its own normalization (piper: espeak-ng reads digits), so `raw` already is `native`. `n/a`: the system has no bundled normalizer. `[n=…]`: scored on a subset only.

### Rows read without any word error (%)

| system | raw | native | normalizer_tr | normalizer_tr_fallback | trnorm |
|---|---:|---:|---:|---:|---:|
| ema-lightning | 95.0 | 95.0 | 95.0 | 95.0 | 95.0 |
| pocket-tts-turkish | 90.0 | 90.0 | 90.0 | 90.0 | 90.0 |
| piper-tr-dfki | 82.5 | = raw | 82.5 | 82.5 | 82.5 |
| mms-tts-tur | 57.5 | n/a | 57.5 | 57.5 | 57.5 |
| qwen3-tts-tiny-tr | 92.5 | n/a | 92.5 | 92.5 | 92.5 |

## WER % per category

| system | front-end | all | freya-everyday | freya-short-native |
|---|---|---:|---:|---:|
| ema-lightning | raw | 0.6 | 0.4 | 3.2 |
| ema-lightning | native | 0.6 | 0.4 | 3.2 |
| ema-lightning | normalizer_tr | 0.6 | 0.4 | 3.2 |
| ema-lightning | normalizer_tr_fallback | 0.6 | 0.4 | 3.2 |
| ema-lightning | trnorm | 0.6 | 0.4 | 3.2 |
| pocket-tts-turkish | raw | 1.9 | 1.1 | 9.7 |
| pocket-tts-turkish | native | 1.9 | 1.1 | 9.7 |
| pocket-tts-turkish | normalizer_tr | 1.9 | 1.1 | 9.7 |
| pocket-tts-turkish | normalizer_tr_fallback | 1.9 | 1.1 | 9.7 |
| pocket-tts-turkish | trnorm | 1.9 | 1.1 | 9.7 |
| piper-tr-dfki | raw | 3.2 | 2.9 | 6.5 |
| piper-tr-dfki | normalizer_tr | 3.2 | 2.9 | 6.5 |
| piper-tr-dfki | normalizer_tr_fallback | 3.2 | 2.9 | 6.5 |
| piper-tr-dfki | trnorm | 3.2 | 2.9 | 6.5 |
| mms-tts-tur | raw | 8.0 | 7.5 | 12.9 |
| mms-tts-tur | normalizer_tr | 8.0 | 7.5 | 12.9 |
| mms-tts-tur | normalizer_tr_fallback | 8.0 | 7.5 | 12.9 |
| mms-tts-tur | trnorm | 8.0 | 7.5 | 12.9 |
| qwen3-tts-tiny-tr | raw | 2.2 | 2.1 | 3.2 |
| qwen3-tts-tiny-tr | normalizer_tr | 2.2 | 2.1 | 3.2 |
| qwen3-tts-tiny-tr | normalizer_tr_fallback | 2.2 | 2.1 | 3.2 |
| qwen3-tts-tiny-tr | trnorm | 2.2 | 2.1 | 3.2 |

## Speed (CPU, <= 2 threads)

| system | params | out rate | threads | mean audio s | RTF (wall) | CPU-RTF | n timed |
|---|---|---:|---:|---:|---:|---:|---:|
| ema-lightning | 8.6M (5.6M DiT + 3M decoder) | 48000 | 2 | 2.95 | 0.157 | 0.176 | 40 |
| pocket-tts-turkish | ~110M (incl. codec) | 24000 | 1 | 3.39 | 0.573 | 0.810 | 41 |
| piper-tr-dfki | ~15M (VITS medium) | 22050 | 2 | 3.45 | 0.074 | 0.116 | 40 |
| mms-tts-tur | 36M (VITS) | 16000 | 2 | 4.24 | 0.370 | 0.572 | 40 |
| qwen3-tts-tiny-tr | 199M talker + 171M speech tokenizer | 24000 | 2 | 3.73 | 3.275 | 4.386 | 40 |

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
