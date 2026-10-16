# Turkish ASR leaderboard

Generated 2026-10-07 06:07 by `benchmarks/asr/score.py`. Normalisation: `trnorm.normalize_for_eval` (trnorm 0.1.0, src sha256 775f47e163fb) applied to both reference and hypothesis. Corpus-level WER/CER (jiwer). RTF = decode wall-clock / audio duration on Intel(R) Xeon(R) Processor @ 2.10GHz, 2 CPU threads per decoder (vosk is single-threaded), 1-2 decoders running at once on a 4-core machine shared with other jobs, so treat RTF as indicative (summary.json also has `cpu_rtf` = process CPU seconds / audio second).

| system | params | license | set | WER % | CER % | RTF (CPU, 2 threads) | n |
|---|---|---|---|---:|---:|---:|---:|
| whisper-large-v3-turbo | 809M | MIT | fleurs_tr_test | 5.22 | 1.33 | 0.720 | 300/743 (partial) |
| mihuai-turkish-stt | 66M | mihu Community License* | fleurs_tr_test | 9.32 | 2.49 | 0.142 | 743 |
| whisper-small | 244M | MIT | fleurs_tr_test | 11.89 | 2.79 | 0.212 | 743 |
| duxx-zipformer | 23.6M | CC-BY-NC-4.0 | fleurs_tr_test | 13.05 | 3.29 | 0.073 | 743 |
| whisper-base | 74M | MIT | fleurs_tr_test | 23.83 | 6.12 | 0.158 | 743 |
| vosk-tr | ~3M AM + LM graph | Apache-2.0 | fleurs_tr_test | 31.54 | 9.78 | 0.111 | 743 |
| whisper-tiny | 39M | MIT | fleurs_tr_test | 38.05 | 10.66 | 0.057 | 743 |
| mihuai-turkish-stt | 66M | mihu Community License* | mediaspeech_tr | 13.96 | 5.57 | 0.094 | 500 |
| duxx-zipformer | 23.6M | CC-BY-NC-4.0 | mediaspeech_tr | 18.72 | 6.96 | 0.098 | 500 |
| whisper-small | 244M | MIT | mediaspeech_tr | 19.15 | 8.55 | 0.230 | 500 |
| vosk-tr | ~3M AM + LM graph | Apache-2.0 | mediaspeech_tr | 30.31 | 14.87 | 0.139 | 500 |
| whisper-base | 74M | MIT | mediaspeech_tr | 30.39 | 11.86 | 0.196 | 500 |
| whisper-tiny | 39M | MIT | mediaspeech_tr | 43.48 | 16.92 | 0.062 | 500 |

Partial rows are scored on the utterances decoded so far (a prefix of the manifest) and are not directly comparable with complete rows.

## Systems

| system | model | params | license | streaming |
|---|---|---|---|---|
| duxx-zipformer | duxx/turkish-stt-zipformer (streaming zipformer RNN-T, chunk 32, int8, sherpa-onnx) | 23.6M | CC-BY-NC-4.0 | yes |
| mihuai-turkish-stt | mihuai/turkish-stt (streaming transducer, int8 enc/joiner, sherpa-onnx) | 66M | mihu Community License* | yes |
| vosk-tr | vosk-model-small-tr-0.3 (Kaldi TDNN, alphacephei) | ~3M AM + LM graph | Apache-2.0 | yes |
| whisper-base | openai/whisper-base (Systran/faster-whisper-base, int8) | 74M | MIT | no |
| whisper-large-v3-turbo | openai/whisper-large-v3-turbo (dropbox-dash/faster-whisper-large-v3-turbo, int8) | 809M | MIT | no |
| whisper-small | openai/whisper-small (Systran/faster-whisper-small, int8) | 244M | MIT | no |
| whisper-tiny | openai/whisper-tiny (Systran/faster-whisper-tiny, int8) | 39M | MIT | no |

\* mihu Community License: free below USD 500k annual revenue, requires "Powered by mihu" attribution, forbids using the model or its outputs to train competing models. Evaluated here for research/benchmarking only; its outputs must never be used as training data.

## Sets

| set | description | license | n |
|---|---|---|---:|
| fleurs_tr_test | google/fleurs tr_tr test, read Wikipedia sentences, raw (cased/punctuated) refs | CC-BY-4.0 | 743 |
| mediaspeech_tr | ymoslem/MediaSpeech tr, seeded 500-utt subset of YouTube media speech | CC-BY-4.0 | 500 |
