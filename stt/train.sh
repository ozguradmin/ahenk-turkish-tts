#!/usr/bin/env bash
# =============================================================================
# train.sh — train the streaming (causal) Zipformer transducer, laptop-friendly.
#
#   ./stt/train.sh --preset small --hours 4      # 4-hour sitting, then stop cleanly
#   ./stt/train.sh --preset small --hours 4      # ...next day: resumes automatically
#   ./stt/train.sh --preset medium --max-duration 200 --num-epochs 25
#   ./stt/train.sh --stop                        # ask a running training to save & stop
#
# Options:
#   --preset small|medium|tiny  model size (see presets.sh)             [small]
#   --hours N             wall-clock limit of this sitting; at the limit a
#                         checkpoint is written and the run exits (code 75)
#   --exp-dir DIR         default $STT_DATA/exp/<preset>-causal-bpe<V>
#   --vocab V             uses $STT_DATA/lang_bpe_<V>/bpe.model         [1000]
#   --num-epochs E        total epochs (counting earlier sittings)       [preset]
#   --max-duration S      seconds of audio per batch (VRAM knob)         [preset]
#   --num-workers N       DataLoader workers                             [3]
#   --save-every-n N      batches between checkpoints                    [1000]
#   --keep-epochs K       keep only the newest K epoch-*.pt (0 = all)    [12]
#   --no-guard            do not start thermal_guard.py
#   --guard-args "..."    extra args for thermal_guard.py, e.g. "--pause-at 80 --duty-cycle 0.85"
#   --allow-cpu           permit training without CUDA (only sensible for --preset tiny)
#   --stop                touch <exp-dir>/STOP: a running training saves and exits
#   -- ARGS               anything after -- goes verbatim to icefall's train.py
#                         (e.g. -- --base-lr 0.04 --lr-epochs 4)
#
# Resume logic (no flags needed): train_tr.py picks the newest epoch-E.pt or
# checkpoint-N.pt in --exp-dir and continues from there (mid-epoch too).
# =============================================================================
set -euo pipefail
STT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$STT_DIR/path.sh"
# shellcheck disable=SC1091
source "$STT_DIR/presets.sh"

PRESET=small; HOURS=0; EXP_DIR=""; VOCAB=1000; EPOCHS=""; MAXDUR=""; NUM_WORKERS=3
SAVE_N=1000; KEEP_EPOCHS=12; GUARD=1; GUARD_ARGS=""; ALLOW_CPU=0; STOP=0; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --preset) PRESET="$2"; shift ;;
    --hours) HOURS="$2"; shift ;;
    --exp-dir) EXP_DIR="$2"; shift ;;
    --vocab) VOCAB="$2"; shift ;;
    --num-epochs) EPOCHS="$2"; shift ;;
    --max-duration) MAXDUR="$2"; shift ;;
    --num-workers) NUM_WORKERS="$2"; shift ;;
    --save-every-n) SAVE_N="$2"; shift ;;
    --keep-epochs) KEEP_EPOCHS="$2"; shift ;;
    --no-guard) GUARD=0 ;;
    --guard-args) GUARD_ARGS="$2"; shift ;;
    --allow-cpu) ALLOW_CPU=1 ;;
    --stop) STOP=1 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
  esac
  shift
done

preset_args "$PRESET"
EPOCHS="${EPOCHS:-$PRESET_EPOCHS}"
MAXDUR="${MAXDUR:-$PRESET_MAX_DURATION}"
EXP_DIR="${EXP_DIR:-$STT_DATA/exp/${PRESET}-causal-bpe${VOCAB}}"
LANG_DIR="$STT_DATA/lang_bpe_${VOCAB}"

if [ "$STOP" = 1 ]; then
  touch "$EXP_DIR/STOP"
  echo "requested stop: the training saves a checkpoint within ~10 batches and exits"
  exit 0
fi

[ -f "$LANG_DIR/bpe.model" ] || { echo "missing $LANG_DIR/bpe.model (run prepare.sh)" >&2; exit 1; }
[ -f "$STT_DATA/fbank/train_mix_cuts.jsonl.gz" ] || { echo "missing $STT_DATA/fbank/train_mix_cuts.jsonl.gz (run prepare.sh)" >&2; exit 1; }
mkdir -p "$EXP_DIR/log"
rm -f "$EXP_DIR/STOP"
# pin the token model to this experiment (export uses these copies)
for f in bpe.model tokens.txt; do
  if [ ! -f "$EXP_DIR/$f" ]; then
    cp "$LANG_DIR/$f" "$EXP_DIR/$f"
  elif ! cmp -s "$LANG_DIR/$f" "$EXP_DIR/$f"; then
    echo "[train.sh] WARNING: $LANG_DIR/$f differs from the copy in $EXP_DIR (token model changed" \
         "after training started?). Training continues with $EXP_DIR/$f." >&2
  fi
done

HAS_CUDA="$(python -c 'import torch; print(int(torch.cuda.is_available()))')"
if [ "$HAS_CUDA" = 1 ]; then
  FP16=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
else
  FP16=0
  if [ "$ALLOW_CPU" != 1 ]; then
    echo "CUDA is not available. Training ${PRESET} on CPU would take months." >&2
    echo "Check 'nvidia-smi' and setup_env.sh (CUDA build); --allow-cpu overrides." >&2
    exit 1
  fi
  GUARD=0
fi

# icefall asserts save_every_n >= average_period (200 by default)
AVG_PERIOD=200
[ "$SAVE_N" -lt "$AVG_PERIOD" ] && AVG_PERIOD="$SAVE_N"

ARGS=(
  --world-size 1
  --num-epochs "$EPOCHS"
  --use-fp16 "$FP16"
  --exp-dir "$EXP_DIR"
  --bpe-model "$EXP_DIR/bpe.model"
  --manifest-dir "$STT_DATA/fbank"
  --max-duration "$MAXDUR"
  --num-workers "$NUM_WORKERS"
  --save-every-n "$SAVE_N"
  --average-period "$AVG_PERIOD"
  --keep-last-k 3
  --enable-musan 0
  "${MODEL_ARGS[@]}"
  "${STREAM_ARGS[@]}"
  --session-hours "$HOURS"
)

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$EXP_DIR/log/session-$STAMP.out"
{
  echo "# $(date -Iseconds) preset=$PRESET hours=$HOURS cuda=$HAS_CUDA"
  echo "# python -u $STT_DIR/recipe/train_tr.py ${ARGS[*]} ${EXTRA[*]:-}"
} > "$OUT"
echo "[train.sh] exp dir: $EXP_DIR"
echo "[train.sh] log:     $OUT"

# Run training in its own process group (job control) so that the thermal
# guard can freeze it together with its DataLoader workers.
set -m
python -u "$STT_DIR/recipe/train_tr.py" "${ARGS[@]}" "${EXTRA[@]}" >> "$OUT" 2>&1 &
TRAIN_PID=$!
set +m

GUARD_PID=""
if [ "$GUARD" = 1 ]; then
  # shellcheck disable=SC2086
  python "$STT_DIR/thermal_guard.py" --pid "$TRAIN_PID" --log "$EXP_DIR/log/thermal_guard.log" \
    $GUARD_ARGS > /dev/null 2>&1 &
  GUARD_PID=$!
  echo "[train.sh] thermal guard pid $GUARD_PID (log: $EXP_DIR/log/thermal_guard.log)"
fi

INTERRUPTS=0
on_interrupt() {
  INTERRUPTS=$((INTERRUPTS + 1))
  kill -CONT -- "-$TRAIN_PID" 2>/dev/null || true
  if [ "$INTERRUPTS" -eq 1 ]; then
    echo; echo "[train.sh] stopping gracefully (checkpoint within ~10 batches). Ctrl-C again = hard kill."
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
  else
    kill -KILL -- "-$TRAIN_PID" 2>/dev/null || true
  fi
}
trap on_interrupt INT TERM

tail -n +1 -f --pid="$TRAIN_PID" "$OUT" &
TAIL_PID=$!
set +e
# `wait` returns early (>128) when a trapped signal arrives; loop until the child is gone,
# then read its real exit status (bash remembers it).
while kill -0 "$TRAIN_PID" 2>/dev/null; do wait "$TRAIN_PID"; done
wait "$TRAIN_PID"; RC=$?
set -e
sleep 1; kill "$TAIL_PID" 2>/dev/null || true
[ -n "$GUARD_PID" ] && kill "$GUARD_PID" 2>/dev/null || true

# keep only the newest K epoch checkpoints (export --avg needs epoch-(E-avg) .. epoch-E)
if [ "$KEEP_EPOCHS" -gt 0 ]; then
  mapfile -t EPOCH_FILES < <(ls "$EXP_DIR"/epoch-*.pt 2>/dev/null | sed -E 's/.*epoch-([0-9]+)\.pt/\1 &/' | sort -n | awk '{print $2}')
  N=${#EPOCH_FILES[@]}
  if [ "$N" -gt "$KEEP_EPOCHS" ]; then
    for f in "${EPOCH_FILES[@]:0:$((N - KEEP_EPOCHS))}"; do echo "[train.sh] pruning $f"; rm -f "$f"; done
  fi
fi

case "$RC" in
  0)  echo "[train.sh] training finished. Next: ./stt/export_and_test.sh --preset $PRESET --exp-dir $EXP_DIR" ;;
  75) echo "[train.sh] sitting over; run the same command again to resume. ($(cat "$EXP_DIR/RESUME.json" 2>/dev/null | tr -d '\n'))" ;;
  *)  echo "[train.sh] training failed with exit code $RC — see $OUT" >&2 ;;
esac
exit "$RC"
