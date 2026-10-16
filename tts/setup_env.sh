#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — pinned environment for the Turkish Piper voice recipe (tts/):
# piper1-gpl training code at a fixed commit + torch + Lightning + trnorm.
#
#   ./tts/setup_env.sh                # CUDA 12.8 build (RTX 20xx..50xx, L4, L40S, A100, H100)
#   ./tts/setup_env.sh --cuda 12.6    # older driver (< R570) or Pascal (GTX 10xx)
#   ./tts/setup_env.sh --cpu          # CPU only (smoke test, export, synthesis)
#
# Options:
#   --venv PATH        venv location               (default: <repo>/.venv-piper)
#   --piper-dir PATH   piper1-gpl checkout         (default: <repo>/tts/piper1-gpl, git-ignored)
#   --python X.Y       Python version for uv       (default: 3.12)
#   --no-verify        skip the self-test at the end
#
# Needs: Linux x86_64, git, a C compiler (build-essential) and network access to
# PyPI, download.pytorch.org and github.com (piper builds espeak-ng from source).
# Re-running is safe: finished steps are detected and skipped.
# =============================================================================
set -euo pipefail

TTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TTS_DIR/.." && pwd)"

# ----------------------------- PIN ------------------------------------------
UV_VERSION="0.11.32"            # only used if uv is not installed yet
PY_VER="3.12"
TORCH_VER="2.8.0"               # torch + torchaudio; 2.9 switched torch.onnx.export to the
                                # dynamo exporter, which piper's export_onnx.py was not written for
PIPER_REPO="https://github.com/OHF-Voice/piper1-gpl.git"
PIPER_COMMIT="efffbfb226bfb511ebbcf55d0cecd8b35a89743d"   # main, 2026-09-28: v1.8.0 + LR-schedule
                                # stepping fix + "never warmstart when resuming" fix (PR #315)
# ----------------------------------------------------------------------------

MODE="cuda"
CUDA_VER="12.8"
VENV="${VENV:-$REPO_ROOT/.venv-piper}"
PIPER_DIR="${PIPER_DIR:-$TTS_DIR/piper1-gpl}"
VERIFY=1

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) MODE="cpu" ;;
    --cuda) MODE="cuda"; CUDA_VER="$2"; shift ;;
    --venv) VENV="$2"; shift ;;
    --piper-dir) PIPER_DIR="$2"; shift ;;
    --python) PY_VER="$2"; shift ;;
    --no-verify) VERIFY=0 ;;
    -h|--help) sed -n '3,/^# ====/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

log() { printf '\033[1;34m[setup_env]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup_env] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) ;;
  *) die "only Linux x86_64 (native Ubuntu, WSL2 or a container) is supported; got $(uname -s)-$(uname -m)" ;;
esac

if [ "$MODE" = "cpu" ]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  TORCH_LOCAL="+cpu"
else
  case "$CUDA_VER" in
    12.6|12.8|12.9) ;;
    *) die "--cuda must be 12.6, 12.8 or 12.9 for torch ${TORCH_VER}" ;;
  esac
  CU="cu${CUDA_VER/./}"
  TORCH_INDEX="https://download.pytorch.org/whl/${CU}"
  TORCH_LOCAL="+${CU}"
fi

command -v git >/dev/null || die "git is missing (sudo apt install -y git)"
command -v cc >/dev/null || command -v gcc >/dev/null || die "no C compiler (sudo apt install -y build-essential)"

# --------------------------------------------------------------------- uv
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    log "installing uv ${UV_VERSION} into ~/.local/bin"
    command -v curl >/dev/null || die "curl is missing (sudo apt install -y curl)"
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_NO_MODIFY_PATH=1 sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
log "uv: $(uv --version)"

# ------------------------------------------------------------------- venv
if [ -x "$VENV/bin/python" ]; then
  log "reusing venv $VENV ($("$VENV/bin/python" --version))"
  "$VENV/bin/python" -c "import sys; v=sys.version_info; exit(0 if f'{v.major}.{v.minor}'=='${PY_VER}' else 1)" \
    || die "existing venv is not Python ${PY_VER}; remove $VENV or pass --python"
else
  log "creating venv $VENV (Python ${PY_VER}; uv downloads it if the system has none)"
  uv venv -p "$PY_VER" "$VENV"
fi
PY="$VENV/bin/python"
UVPIP=(uv pip install --python "$PY")
TORCH_PIN="$(mktemp)"
trap 'rm -f "$TORCH_PIN"' EXIT
printf 'torch==%s\ntorchaudio==%s\n' "${TORCH_VER}${TORCH_LOCAL}" "${TORCH_VER}${TORCH_LOCAL}" > "$TORCH_PIN"
CONSTRAINTS=(-c "$TORCH_PIN" -c "$TTS_DIR/constraints-tts.txt")

# ------------------------------------------------------ torch + torchaudio
installed_torch="$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
if [ "$installed_torch" != "${TORCH_VER}${TORCH_LOCAL}" ]; then
  log "installing torch/torchaudio ${TORCH_VER}${TORCH_LOCAL} from ${TORCH_INDEX}"
  # torch only exists on the PyTorch index; its pure-Python deps may come from PyPI so that the
  # frozen versions in constraints-tts.txt are available.
  "${UVPIP[@]}" --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match --reinstall-package torch --reinstall-package torchaudio \
    -c "$TTS_DIR/constraints-tts.txt" "torch==${TORCH_VER}${TORCH_LOCAL}" "torchaudio==${TORCH_VER}${TORCH_LOCAL}"
else
  log "torch ${installed_torch} already installed"
fi

# ------------------------------------------------------- piper1-gpl checkout
if [ ! -d "$PIPER_DIR/.git" ]; then
  log "cloning piper1-gpl @ ${PIPER_COMMIT:0:10} into $PIPER_DIR"
  mkdir -p "$PIPER_DIR"
  git -C "$PIPER_DIR" init -q
  git -C "$PIPER_DIR" remote add origin "$PIPER_REPO"
fi
if [ "$(git -C "$PIPER_DIR" rev-parse HEAD 2>/dev/null || true)" != "$PIPER_COMMIT" ]; then
  log "checking out piper1-gpl ${PIPER_COMMIT}"
  git -C "$PIPER_DIR" fetch -q --depth 1 origin "$PIPER_COMMIT"
  git -C "$PIPER_DIR" -c advice.detachedHead=false checkout -q "$PIPER_COMMIT"
  rm -f "$PIPER_DIR/.tts-built"
else
  log "piper1-gpl already at ${PIPER_COMMIT:0:10}"
fi

# ----------------------------------------------- python packages (pinned)
log "installing requirements-tts.txt + piper-tts[train] (editable) + trnorm (editable)"
"${UVPIP[@]}" "${CONSTRAINTS[@]}" -r "$TTS_DIR/requirements-tts.txt"
# The editable install only links src/ (the native parts are built below); --no-build-isolation
# reuses the pinned scikit-build/cmake/ninja instead of fetching the newest ones.
"${UVPIP[@]}" "${CONSTRAINTS[@]}" --no-build-isolation -e "${PIPER_DIR}[train]"
"${UVPIP[@]}" "${CONSTRAINTS[@]}" -e "$REPO_ROOT"

# --------------------------------- native parts: espeak-ng bridge + monotonic_align
# (docs/TRAINING.md of piper1-gpl: "python3 setup.py build_ext --inplace" + build_monotonic_align.sh)
STAMP="$PIPER_DIR/.tts-built"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "${PIPER_COMMIT}-py${PY_VER}" ] \
   || [ ! -f "$PIPER_DIR/src/piper/espeakbridge.so" ] \
   || ! ls "$PIPER_DIR"/src/piper/train/vits/monotonic_align/monotonic_align/core*.so >/dev/null 2>&1; then
  log "building espeak-ng + espeakbridge (CMake, clones espeak-ng from GitHub; ~1-3 min)"
  # The nested espeak-ng CMake build looks up `ninja`/`cmake` on PATH; the pinned ones live in the
  # venv (no system ninja on minimal images such as Modal's debian_slim).
  export PATH="$VENV/bin:$PATH"
  ( cd "$PIPER_DIR" && CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}" \
      "$PY" setup.py build_ext --inplace > "$PIPER_DIR/.tts-build.log" 2>&1 ) \
    || { tail -30 "$PIPER_DIR/.tts-build.log" >&2; die "espeak-ng bridge build failed (log: $PIPER_DIR/.tts-build.log)"; }
  log "building monotonic_align (Cython)"
  ( cd "$PIPER_DIR/src/piper/train/vits/monotonic_align" && mkdir -p monotonic_align && rm -f core.c \
      && "$VENV/bin/cythonize" -i core.pyx >> "$PIPER_DIR/.tts-build.log" 2>&1 \
      && mv core*.so monotonic_align/ ) \
    || { tail -30 "$PIPER_DIR/.tts-build.log" >&2; die "monotonic_align build failed (log: $PIPER_DIR/.tts-build.log)"; }
  echo "${PIPER_COMMIT}-py${PY_VER}" > "$STAMP"
else
  log "native parts already built for ${PIPER_COMMIT:0:10}"
fi

# remember non-default locations for path.sh
cat > "$TTS_DIR/.env.local" <<EOF
# written by setup_env.sh on $(date -Iseconds); sourced by path.sh
VENV="$VENV"
PIPER_DIR="$PIPER_DIR"
TTS_ENV_MODE="$MODE"
EOF

# ---------------------------------------------------------------- verify
if [ "$VERIFY" = 1 ]; then
  log "self-test"
  OMP_NUM_THREADS=1 "$PY" - <<'PYEOF'
import importlib.metadata as md, sys
import torch, lightning, onnxruntime, trnorm
from piper.phonemize_espeak import EspeakPhonemizer
from piper.train.vits.monotonic_align import maximum_path  # noqa: F401  (Cython extension)
from piper.train.vits.lightning import VitsModel  # noqa: F401
print("python      ", sys.version.split()[0])
for p in ["torch", "torchaudio", "lightning", "jsonargparse", "piper-tts", "onnx", "onnxruntime", "librosa", "numpy"]:
    print(f"{p:12s}", md.version(p))
text = trnorm.normalize("Saat 09:30'da 3. kata gelin.")
print("trnorm      ", text)
print("espeak tr   ", "".join(EspeakPhonemizer().phonemize("tr", text)[0]))
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU          {p.name}, {p.total_memory/2**30:.1f} GiB, sm_{p.major}{p.minor}")
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize(); print("cuda matmul  ok", float((x @ x).abs().mean()))
    archs = torch.cuda.get_arch_list()
    if f"sm_{p.major}{p.minor}" not in archs:
        print("WARNING: this torch build has no kernels for your GPU:", archs)
else:
    print("GPU          none (CPU mode)")
PYEOF
fi

log "done. Activate with:  source $TTS_DIR/path.sh"
