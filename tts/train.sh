#!/usr/bin/env bash
# =============================================================================
# train.sh — fine-tune the public-domain LJSpeech "medium" Piper checkpoint on the
# Antalia Turkish corpus (piper1-gpl, pinned commit; see tts/README.md section 3).
#
#   ./tts/train.sh --preset a100 --max-epochs 1000                # cloud GPU
#   ./tts/train.sh --preset laptop8gb --hours 4                  # 4-hour sitting, then clean stop (exit 75)
#   ./tts/train.sh --preset laptop8gb --hours 4                  # ...next day: resumes from last.ckpt
#   ./tts/train.sh --stop                                        # ask a running training to save & exit
#
# Options:
#   --preset P          batch size / workers for the card: a100 | l40s | l4 | h100 | laptop8gb | cpu [auto]
#   --batch-size N      override the preset's batch size (clips per step)
#   --max-epochs E      total epochs, counting earlier sittings; also sets the LR anneal   [1000]
#   --hours H           wall-clock limit of this sitting (0 = none); stops at an epoch boundary
#   --exp-dir DIR       experiment directory                    [$TTS_DATA/exp/antalia-medium]
#   --data-dir DIR      output of prepare_antalia.py            [$TTS_DATA/antalia]
#   --base-ckpt PATH    LJSpeech medium checkpoint (fetched + sha256-checked if missing)
#                                                               [$TTS_DATA/base/lj-med_1000.ckpt]
#   --val-every N       validate (val_mel/val_mos on the held-out clips) every N epochs   [5]
#   --ckpt-every N      write the resume checkpoint every N epochs                         [5]
#   --snapshot-every N  weights-only snapshot every N epochs, 0 = none                     [100]
#   --precision P       Lightning precision: 32-true | bf16-mixed | 16-mixed               [32-true]
#   --num-workers N     DataLoader workers                      [preset]
#   --allow-cpu         permit training without CUDA (smoke tests only)
#   --no-guard          do not start stt/thermal_guard.py (laptop GPU temperature guard)
#   --stop              touch <exp-dir>/STOP: a running training saves after the current batch and exits
#   --fetch-base-only   download + sha256-check the base checkpoint, then exit
#   -- ARGS             passed verbatim to piper_fit.py / LightningCLI, e.g.
#                       -- --trainer.limit_train_batches 3 --model.learning_rate 1e-4
#
# Resume: if <exp-dir>/checkpoints/last.ckpt exists the run continues from it (weights, optimizers,
# LR schedule, epoch); otherwise it starts from the LJSpeech weights (--model.warmstart_ckpt).
# Exit codes: 0 = --max-epochs reached, 75 = stopped by --hours or --stop (run again to resume).
# =============================================================================
set -euo pipefail
TTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$TTS_DIR/path.sh"

BASE_REPO="rhasspy/piper-checkpoints"
BASE_REV="37d1bea250e21ca0d617818d2bef1db724edb80c"      # 2026-09-17
BASE_FILE="en/en_US/ljspeech/medium/lj-med_1000.ckpt"
BASE_SHA256="dcf2449bdbdaad09256a08dfac211c59f6b36ce8d3f244fd844a9eb1d7384c7c"

PRESET=""; BATCH=""; MAX_EPOCHS=1000; HOURS=0; EXP_DIR=""; DATA_DIR=""; BASE_CKPT=""
VAL_EVERY=5; CKPT_EVERY=5; SNAP_EVERY=100; PRECISION="32-true"; WORKERS=""; ALLOW_CPU=0; GUARD=1; STOP=0
FETCH_ONLY=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --preset) PRESET="$2"; shift ;;
    --batch-size) BATCH="$2"; shift ;;
    --max-epochs) MAX_EPOCHS="$2"; shift ;;
    --hours) HOURS="$2"; shift ;;
    --exp-dir) EXP_DIR="$2"; shift ;;
    --data-dir) DATA_DIR="$2"; shift ;;
    --base-ckpt) BASE_CKPT="$2"; shift ;;
    --val-every) VAL_EVERY="$2"; shift ;;
    --ckpt-every) CKPT_EVERY="$2"; shift ;;
    --snapshot-every) SNAP_EVERY="$2"; shift ;;
    --precision) PRECISION="$2"; shift ;;
    --num-workers) WORKERS="$2"; shift ;;
    --allow-cpu) ALLOW_CPU=1 ;;
    --no-guard) GUARD=0 ;;
    --stop) STOP=1 ;;
    --fetch-base-only) FETCH_ONLY=1 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '3,/^# ====/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
  esac
  shift
done

EXP_DIR="${EXP_DIR:-$TTS_DATA/exp/antalia-medium}"
DATA_DIR="${DATA_DIR:-$TTS_DATA/antalia}"
BASE_CKPT="${BASE_CKPT:-$TTS_DATA/base/lj-med_1000.ckpt}"

# base checkpoint (public-domain LJSpeech data; MIT-licensed repo), pinned revision + sha256
fetch_base() {
  if [ ! -f "$BASE_CKPT" ]; then
    echo "[train.sh] fetching $BASE_REPO/$BASE_FILE @ ${BASE_REV:0:10}"
    mkdir -p "$(dirname "$BASE_CKPT")"
    python - "$BASE_CKPT" "$BASE_REPO" "$BASE_REV" "$BASE_FILE" <<'PYEOF'
import shutil, sys
from huggingface_hub import hf_hub_download
dst, repo, rev, name = sys.argv[1:]
src = hf_hub_download(repo, name, repo_type="dataset", revision=rev)
shutil.copyfile(src, dst + ".part")
shutil.move(dst + ".part", dst)
PYEOF
  fi
  if [ ! -f "$BASE_CKPT.sha256ok" ]; then
    echo "$BASE_SHA256  $BASE_CKPT" | sha256sum -c --quiet - \
      || { echo "sha256 mismatch for $BASE_CKPT" >&2; exit 1; }
    touch "$BASE_CKPT.sha256ok"
  fi
}

if [ "$FETCH_ONLY" = 1 ]; then fetch_base; echo "[train.sh] base checkpoint ready: $BASE_CKPT"; exit 0; fi
if [ "$STOP" = 1 ]; then
  mkdir -p "$EXP_DIR" && touch "$EXP_DIR/STOP"
  echo "requested stop: the training saves $EXP_DIR/checkpoints/last.ckpt after the current batch and exits"
  exit 0
fi

for f in metadata.csv heldout.txt; do
  [ -f "$DATA_DIR/$f" ] || { echo "missing $DATA_DIR/$f (run tts/prepare_antalia.py)" >&2; exit 1; }
done

HAS_CUDA="$(python -c 'import torch; print(int(torch.cuda.is_available()))')"
if [ "$HAS_CUDA" = 0 ]; then
  if [ "$ALLOW_CPU" != 1 ]; then
    echo "CUDA is not available; training on CPU would take weeks. Check nvidia-smi and setup_env.sh" >&2
    echo "(CUDA build). --allow-cpu overrides (smoke tests)." >&2
    exit 1
  fi
  PRESET="${PRESET:-cpu}"; GUARD=0
fi
if [ -z "$PRESET" ]; then  # pick by GPU memory
  MEM_GB="$(python -c 'import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 2**30))')"
  if [ "$MEM_GB" -ge 70 ]; then PRESET=a100; elif [ "$MEM_GB" -ge 40 ]; then PRESET=l40s;
  elif [ "$MEM_GB" -ge 20 ]; then PRESET=l4; else PRESET=laptop8gb; fi
  echo "[train.sh] ${MEM_GB} GiB GPU -> --preset $PRESET"
fi
# Batch sizes: see tts/README.md section 3 (activation memory measured per clip of this corpus,
# whose clips are long: median 17.6 s, max 20.2 s).
case "$PRESET" in
  a100|h100) P_BATCH=32; P_WORKERS=4 ;;
  l40s)      P_BATCH=32; P_WORKERS=4 ;;
  l4)        P_BATCH=16; P_WORKERS=4 ;;
  laptop8gb) P_BATCH=4;  P_WORKERS=2 ;;
  cpu)       P_BATCH=4;  P_WORKERS=0 ;;
  *) echo "unknown --preset $PRESET" >&2; exit 1 ;;
esac
BATCH="${BATCH:-$P_BATCH}"
WORKERS="${WORKERS:-$P_WORKERS}"

# piper caches phonemes/spectrograms by CSV row number + text: one cache per metadata.csv version
CSV_SHA="$(sha1sum "$DATA_DIR/metadata.csv" | cut -c1-12)"
CACHE_DIR="$DATA_DIR/piper_cache/${CSV_SHA}-tr"
mkdir -p "$EXP_DIR/log"
rm -f "$EXP_DIR/STOP"

LAST="$EXP_DIR/checkpoints/last.ckpt"
if [ -f "$LAST" ]; then
  START=(--ckpt_path "$LAST")
  echo "[train.sh] resuming from $LAST ($(tr -d '\n' < "$EXP_DIR/RESUME.json" 2>/dev/null || true))"
else
  fetch_base  # only a new run needs it; a resume never warm-starts
  START=(--model.warmstart_ckpt "$BASE_CKPT")
  echo "[train.sh] new run: weights (generator + discriminators) from $BASE_CKPT"
fi

ARGS=(
  fit
  --trainer.default_root_dir "$EXP_DIR"
  --trainer.max_epochs "$MAX_EPOCHS"
  --trainer.check_val_every_n_epoch "$VAL_EVERY"
  --trainer.precision "$PRECISION"
  --trainer.accelerator "$([ "$HAS_CUDA" = 1 ] && echo gpu || echo cpu)"
  --trainer.devices 1
  --trainer.log_every_n_steps 10
  --data.voice_name tr_TR-antalia-medium
  --data.espeak_voice tr
  --data.csv_path "$DATA_DIR/metadata.csv"
  --data.audio_dir "$DATA_DIR/wav"
  --data.heldout_list "$DATA_DIR/heldout.txt"
  --data.num_test_examples 4
  --data.cache_dir "$CACHE_DIR"
  --data.config_path "$EXP_DIR/config.json"
  --data.batch_size "$BATCH"
  --data.num_workers "$WORKERS"
  --model.sample_rate 22050
  --ckpt_every "$CKPT_EVERY"
  --snapshot_every "$SNAP_EVERY"
  --time_limit_hours "$HOURS"
  "${START[@]}"
)

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$EXP_DIR/log/session-$STAMP.out"
{
  echo "# $(date -Iseconds) preset=$PRESET batch=$BATCH hours=$HOURS max_epochs=$MAX_EPOCHS cuda=$HAS_CUDA"
  echo "# python -u $TTS_DIR/piper_fit.py ${ARGS[*]} ${EXTRA[*]:-}"
} > "$OUT"
echo "[train.sh] exp dir: $EXP_DIR"
echo "[train.sh] log:     $OUT"

# own process group, so the thermal guard can freeze training together with its DataLoader workers
set -m
python -u "$TTS_DIR/piper_fit.py" "${ARGS[@]}" "${EXTRA[@]}" >> "$OUT" 2>&1 &
TRAIN_PID=$!
set +m

GUARD_PID=""
if [ "$GUARD" = 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  python "$REPO_ROOT/stt/thermal_guard.py" --pid "$TRAIN_PID" --log "$EXP_DIR/log/thermal_guard.log" > /dev/null 2>&1 &
  GUARD_PID=$!
  echo "[train.sh] thermal guard pid $GUARD_PID (pauses at 83 C, resumes at 72 C)"
fi

INTERRUPTS=0
on_interrupt() {
  INTERRUPTS=$((INTERRUPTS + 1))
  kill -CONT -- "-$TRAIN_PID" 2>/dev/null || true
  if [ "$INTERRUPTS" -eq 1 ]; then
    echo; echo "[train.sh] stopping gracefully (saves last.ckpt). Ctrl-C again = hard kill."
    touch "$EXP_DIR/STOP"
  else
    kill -KILL -- "-$TRAIN_PID" 2>/dev/null || true
  fi
}
trap on_interrupt INT TERM

tail -n +1 -f --pid="$TRAIN_PID" "$OUT" &
TAIL_PID=$!
set +e
while kill -0 "$TRAIN_PID" 2>/dev/null; do wait "$TRAIN_PID"; done
wait "$TRAIN_PID"; RC=$?
set -e
sleep 1; kill "$TAIL_PID" 2>/dev/null || true
[ -n "$GUARD_PID" ] && kill "$GUARD_PID" 2>/dev/null || true

REASON="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["reason"])' "$EXP_DIR/RESUME.json" 2>/dev/null || echo "?")"
# Lightning exits 0 after SIGTERM/Ctrl-C (piper_fit.py saved last.ckpt first): report those as resumable
case "$REASON" in time_limit|stop_file|exception:*) [ "$RC" = 0 ] && RC=75 ;; esac
case "$RC" in
  0)  echo "[train.sh] finished ($REASON). Next: ./tts/export_and_test.sh --exp-dir $EXP_DIR" ;;
  75) echo "[train.sh] sitting over ($REASON); run the same command again to resume." ;;
  *)  echo "[train.sh] training failed with exit code $RC — see $OUT" >&2 ;;
esac
exit "$RC"
