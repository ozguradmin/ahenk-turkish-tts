# shellcheck shell=bash
# Source this file from bash:   source stt/path.sh
# Sets up PATH/PYTHONPATH for the STT recipe and keeps CPU usage polite.
STT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$STT_DIR/.." && pwd)"
# shellcheck disable=SC1091
[ -f "$STT_DIR/.env.local" ] && . "$STT_DIR/.env.local"

VENV="${VENV:-$REPO_ROOT/.venv-stt}"
ICEFALL_DIR="${ICEFALL_DIR:-$STT_DIR/icefall}"
# All big files (downloads, audio, features, experiments) live here.
STT_DATA="${STT_DATA:-$REPO_ROOT/data/stt}"
export STT_DIR REPO_ROOT VENV ICEFALL_DIR STT_DATA

if [ ! -x "$VENV/bin/python" ]; then
  echo "[path.sh] venv $VENV not found — run stt/setup_env.sh first" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONPATH="$STT_DIR/recipe:$ICEFALL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
# lhotse: read audio with soundfile, never try ffmpeg/torchaudio first
export LHOTSE_AUDIO_BACKEND="${LHOTSE_AUDIO_BACKEND:-LibsndfileBackend}"
# Do not let BLAS/OpenMP grab every core (other jobs share the machine).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-$OMP_NUM_THREADS}"
# huggingface cache stays inside the data dir unless the user set one
export HF_HOME="${HF_HOME:-$STT_DATA/hf_home}"
