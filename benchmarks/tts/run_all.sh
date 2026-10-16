#!/usr/bin/env bash
# Full TTS benchmark pipeline for one or more test sets, strictly ONE heavy process at a time
# (each <= 2 CPU threads, run under `nice`): for every system, synthesize all front-ends, then
# transcribe that system's audio with the ASR judge, then re-score, so results appear system by
# system. Resumable: re-run the same command after an interruption, or after trnorm changes
# (only rows whose front-end text changed are synthesized and transcribed again).
#
#   benchmarks/tts/run_all.sh dev_numbers10 freya_sample40
#   QWEN_SAMPLE_tr_zor=65 nohup benchmarks/tts/run_all.sh tr_zor > benchmarks/tts/logs/run_tr_zor.log 2>&1 &
#
# Environment knobs:
#   SYSTEMS, FRONTENDS          space-separated lists (defaults: everything)
#   QWEN_SAMPLE_<set>=N         run qwen3-tts-tiny-tr on a seeded, category-stratified ~N-row subset
#   NICE                        niceness for every step (default 10: other jobs on the box come first)
#   PY                          python interpreter (default .venv-tts/bin/python)
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-.venv-tts/bin/python}
SYSTEMS=${SYSTEMS:-"ema-lightning piper-tr-dfki mms-tts-tur pocket-tts-turkish qwen3-tts-tiny-tr"}
FRONTENDS=${FRONTENDS:-"raw native normalizer_tr normalizer_tr_fallback trnorm"}
NICE=${NICE:-10}
RUN=(nice -n "$NICE" "$PY")
SETS=("$@")
[ ${#SETS[@]} -gt 0 ] || { echo "usage: $0 SET [SET...]"; exit 2; }
ts() { date -u +%FT%TZ; }
NOISE="Fetching [0-9]+ files|pad_token_id|SoX could not|sox: not found|path variables|^\s*$|Please rename|If you do \(or"

echo "$(ts) start: sets=${SETS[*]} systems=$SYSTEMS frontends=$FRONTENDS"
for set in "${SETS[@]}"; do
  for sys in $SYSTEMS; do
    extra=()
    if [ "$sys" = qwen3-tts-tiny-tr ]; then
      var="QWEN_SAMPLE_${set}"
      [ -n "${!var:-}" ] && extra=(--sample "${!var}")
    fi
    for fe in $FRONTENDS; do
      echo "$(ts) synth $set $sys $fe ${extra[*]}"
      "${RUN[@]}" benchmarks/tts/synthesize.py --set "$set" --system "$sys" --frontend "$fe" "${extra[@]}" 2>&1 \
        | grep --line-buffered -v -E "$NOISE"
      [ "${PIPESTATUS[0]}" -eq 0 ] || echo "$(ts) synth FAILED: $set $sys $fe"
    done
    echo "$(ts) judge $set $sys"
    # shellcheck disable=SC2086
    "${RUN[@]}" benchmarks/tts/judge.py --set "$set" --systems "$sys" --frontends $FRONTENDS \
      || echo "$(ts) judge FAILED: $set $sys"
    "${RUN[@]}" benchmarks/tts/score.py --set "$set" > /dev/null || echo "$(ts) score FAILED: $set"
  done
  "${RUN[@]}" benchmarks/tts/score.py --set "$set"
done
echo "$(ts) done: sets=${SETS[*]}"
