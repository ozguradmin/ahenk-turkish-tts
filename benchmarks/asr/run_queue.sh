#!/usr/bin/env bash
# Run benchmark jobs in order with at most $PARALLEL decoders at a time (default 1, which
# leaves room for other jobs on a shared 4-core machine; each decoder uses 2 threads).
#
#   nohup benchmarks/asr/run_queue.sh > benchmarks/asr/logs/queue.log 2>&1 &
#   JOBS=$'fleurs_tr_test whisper-medium\nmediaspeech_tr whisper-medium' benchmarks/asr/run_queue.sh
#   JOBS='fleurs_tr_test whisper-large-v3-turbo --limit 300' PARALLEL=1 benchmarks/asr/run_queue.sh
#
# Each job line is "<set> <system> [extra run_asr.py args]"; its log goes to
# benchmarks/asr/logs/<set>__<system>.log. run_asr.py is resumable, so re-running the
# queue after an interruption only decodes what is missing.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HERE
PARALLEL="${PARALLEL:-1}"
mkdir -p "$HERE/logs"

# Full benchmark, fastest systems first.
DEFAULT_JOBS="
fleurs_tr_test duxx-zipformer
mediaspeech_tr duxx-zipformer
fleurs_tr_test mihuai-turkish-stt
mediaspeech_tr mihuai-turkish-stt
fleurs_tr_test vosk-tr
mediaspeech_tr vosk-tr
fleurs_tr_test whisper-tiny
mediaspeech_tr whisper-tiny
fleurs_tr_test whisper-base
mediaspeech_tr whisper-base
fleurs_tr_test whisper-small
mediaspeech_tr whisper-small
fleurs_tr_test whisper-large-v3-turbo
mediaspeech_tr whisper-large-v3-turbo
"

echo "$(date '+%F %T') queue start (parallel=$PARALLEL, OMP_NUM_THREADS=$OMP_NUM_THREADS)"
printf '%s\n' "${JOBS:-$DEFAULT_JOBS}" | grep -v '^[[:space:]]*$' | xargs -P "$PARALLEL" -L 1 bash -c '
  set="$1"; sys="$2"; shift 2
  log="$HERE/logs/${set}__${sys}.log"
  echo "$(date "+%F %T") start  $set $sys $*"
  if python "$HERE/run_asr.py" --set "$set" --system "$sys" "$@" >> "$log" 2>&1; then
    echo "$(date "+%F %T") done   $set $sys"
  else
    echo "$(date "+%F %T") FAILED $set $sys (see $log)"
  fi
' _
echo "$(date '+%F %T') queue finished"
