# shellcheck shell=bash
# Source this file from bash:   source tts/path.sh
# Sets up PATH and data locations for the Turkish Piper recipe and keeps CPU usage polite.
TTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TTS_DIR/.." && pwd)"
# shellcheck disable=SC1091
[ -f "$TTS_DIR/.env.local" ] && . "$TTS_DIR/.env.local"

VENV="${VENV:-$REPO_ROOT/.venv-piper}"
PIPER_DIR="${PIPER_DIR:-$TTS_DIR/piper1-gpl}"
# All big files (corpus, Piper cache, base checkpoint, experiments) live here (git-ignored).
TTS_DATA="${TTS_DATA:-$REPO_ROOT/data/tts_train}"
export TTS_DIR REPO_ROOT VENV PIPER_DIR TTS_DATA

if [ ! -x "$VENV/bin/python" ]; then
  echo "[path.sh] venv $VENV not found — run tts/setup_env.sh first" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
# Do not let BLAS/OpenMP grab every core (other jobs share the machine).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-$OMP_NUM_THREADS}"
# model/dataset caches stay inside the data dir unless the user set their own
export HF_HOME="${HF_HOME:-$TTS_DATA/hf_home}"
export TORCH_HOME="${TORCH_HOME:-$TTS_DATA/torch_home}"   # UTMOS (val_mos) weights via torch.hub
