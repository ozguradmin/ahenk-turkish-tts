#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — create the pinned Python environment for the Turkish
# streaming-ASR recipe (icefall zipformer + k2 + lhotse + sherpa-onnx).
#
#   ./stt/setup_env.sh                # laptop: CUDA 12.8 build (RTX 20xx..50xx)
#   ./stt/setup_env.sh --cuda 12.6    # older GPU (GTX 10xx / Pascal) or driver < 570
#   ./stt/setup_env.sh --cpu          # CPU only (smoke test, export, decoding)
#
# Options:
#   --venv PATH         venv location            (default: <repo>/.venv-stt)
#   --icefall-dir PATH  icefall checkout         (default: <repo>/stt/icefall, git-ignored)
#   --python X.Y        Python version for uv    (default: 3.11)
#   --skip-icefall      do not clone/update icefall
#   --no-verify         skip the import/GPU self-test at the end
#
# Everything is pinned (see the PIN section, requirements-stt.txt and
# constraints-stt.txt). Re-running the script is safe: it reuses the venv and
# only installs what is missing.
# =============================================================================
set -euo pipefail

STT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$STT_DIR/.." && pwd)"

# ----------------------------- PIN ------------------------------------------
UV_VERSION="0.11.32"            # only used if uv is not installed yet
PY_VER="3.11"
TORCH_VER="2.8.0"               # torch + torchaudio
K2_VER="1.24.4.dev20260625"     # from https://k2-fsa.github.io/k2/{cpu,cuda}.html
ICEFALL_REPO="https://github.com/k2-fsa/icefall.git"
ICEFALL_COMMIT="3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1"   # master, 2026-07-16
K2_BASE="https://huggingface.co/csukuangfj2/k2/resolve/main"
# ----------------------------------------------------------------------------

MODE="cuda"
CUDA_VER="12.8"
VENV="${VENV:-$REPO_ROOT/.venv-stt}"
ICEFALL_DIR="${ICEFALL_DIR:-$STT_DIR/icefall}"
SKIP_ICEFALL=0
VERIFY=1

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) MODE="cpu" ;;
    --cuda) MODE="cuda"; CUDA_VER="$2"; shift ;;
    --venv) VENV="$2"; shift ;;
    --icefall-dir) ICEFALL_DIR="$2"; shift ;;
    --python) PY_VER="$2"; shift ;;
    --skip-icefall) SKIP_ICEFALL=1 ;;
    --no-verify) VERIFY=0 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

log() { printf '\033[1;34m[setup_env]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup_env] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) ;;
  *) die "only Linux x86_64 (native Ubuntu or WSL2) is supported by this script; got $(uname -s)-$(uname -m)" ;;
esac

PYTAG="cp${PY_VER/./}"
if [ "$MODE" = "cpu" ]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  TORCH_LOCAL="+cpu"
  K2_WHEEL="k2-${K2_VER}+cpu.torch${TORCH_VER}-${PYTAG}-${PYTAG}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
  K2_URL="${K2_BASE}/cpu/${K2_VER}/linux-x64/${K2_WHEEL}"
else
  case "$CUDA_VER" in
    12.6|12.8|12.9) ;;
    *) die "--cuda must be 12.6, 12.8 or 12.9 for torch ${TORCH_VER}" ;;
  esac
  CU="cu${CUDA_VER/./}"
  TORCH_INDEX="https://download.pytorch.org/whl/${CU}"
  TORCH_LOCAL="+${CU}"
  K2_WHEEL="k2-${K2_VER}+cuda${CUDA_VER}.torch${TORCH_VER}-${PYTAG}-${PYTAG}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
  K2_URL="${K2_BASE}/ubuntu-cuda/${K2_VER}/${K2_WHEEL}"
fi

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
command -v git >/dev/null || die "git is missing (sudo apt install -y git)"

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

# ------------------------------------------------- torch + torchaudio + k2
TORCH_PIN="$(mktemp)"
trap 'rm -f "$TORCH_PIN"' EXIT
printf 'torch==%s\ntorchaudio==%s\n' "${TORCH_VER}${TORCH_LOCAL}" "${TORCH_VER}${TORCH_LOCAL}" > "$TORCH_PIN"
FROZEN=(); [ -f "$STT_DIR/constraints-stt.txt" ] && FROZEN=(-c "$STT_DIR/constraints-stt.txt")

installed_torch="$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
if [ "$installed_torch" != "${TORCH_VER}${TORCH_LOCAL}" ]; then
  log "installing torch==${TORCH_VER}${TORCH_LOCAL} torchaudio from ${TORCH_INDEX}"
  # torch itself only exists on the PyTorch index; its pure-Python deps may come
  # from PyPI so that the frozen versions in constraints-stt.txt are available.
  "${UVPIP[@]}" --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match --reinstall-package torch --reinstall-package torchaudio \
    "${FROZEN[@]}" "torch==${TORCH_VER}${TORCH_LOCAL}" "torchaudio==${TORCH_VER}${TORCH_LOCAL}"
else
  log "torch ${installed_torch} already installed"
fi

installed_k2="$("$PY" -c 'import importlib.metadata as m; print(m.version("k2"))' 2>/dev/null || true)"
want_k2="${K2_VER}+$( [ "$MODE" = cpu ] && echo cpu || echo "cuda${CUDA_VER}" ).torch${TORCH_VER}"
if [ "$installed_k2" != "$want_k2" ]; then
  log "installing k2 ${want_k2}"
  log "  ${K2_URL}"
  "${UVPIP[@]}" --reinstall-package k2 -c "$TORCH_PIN" "k2 @ ${K2_URL}"
else
  log "k2 ${installed_k2} already installed"
fi

# ------------------------------------------------------------ the rest
log "installing pinned requirements (requirements-stt.txt + constraints-stt.txt)"
CONSTRAINTS=(-c "$TORCH_PIN")
[ -f "$STT_DIR/constraints-stt.txt" ] && CONSTRAINTS+=(-c "$STT_DIR/constraints-stt.txt")
"${UVPIP[@]}" -r "$STT_DIR/requirements-stt.txt" "${CONSTRAINTS[@]}"

log "installing trnorm (editable) from $REPO_ROOT"
"${UVPIP[@]}" -e "$REPO_ROOT" "${CONSTRAINTS[@]}"

# --------------------------------------------------------------- icefall
if [ "$SKIP_ICEFALL" = 0 ]; then
  if [ ! -d "$ICEFALL_DIR/.git" ]; then
    log "cloning icefall @ ${ICEFALL_COMMIT:0:10} into $ICEFALL_DIR"
    mkdir -p "$ICEFALL_DIR"
    git -C "$ICEFALL_DIR" init -q
    git -C "$ICEFALL_DIR" remote add origin "$ICEFALL_REPO"
  fi
  if [ "$(git -C "$ICEFALL_DIR" rev-parse HEAD 2>/dev/null || true)" != "$ICEFALL_COMMIT" ]; then
    log "checking out icefall ${ICEFALL_COMMIT}"
    git -C "$ICEFALL_DIR" fetch -q --depth 1 origin "$ICEFALL_COMMIT"
    git -C "$ICEFALL_DIR" -c advice.detachedHead=false checkout -q "$ICEFALL_COMMIT"
  else
    log "icefall already at ${ICEFALL_COMMIT:0:10}"
  fi
fi

# remember non-default locations for path.sh
cat > "$STT_DIR/.env.local" <<EOF
# written by setup_env.sh on $(date -Iseconds); sourced by path.sh
VENV="$VENV"
ICEFALL_DIR="$ICEFALL_DIR"
STT_ENV_MODE="$MODE"
EOF

# ---------------------------------------------------------------- verify
if [ "$VERIFY" = 1 ]; then
  log "self-test"
  PYTHONPATH="$ICEFALL_DIR:${PYTHONPATH:-}" OMP_NUM_THREADS=1 "$PY" - <<'PYEOF'
import importlib.metadata as md, torch, k2, lhotse, sentencepiece, sherpa_onnx, onnx, onnxruntime, trnorm
import icefall  # noqa: F401  (needs PYTHONPATH=<icefall>)
print("python      ", __import__("sys").version.split()[0])
for p in ["torch", "torchaudio", "k2", "lhotse", "sentencepiece", "sherpa-onnx", "onnx", "onnxruntime", "numpy"]:
    print(f"{p:12s}", md.version(p))
print("k2 cuda     ", k2.with_cuda)
print("trnorm      ", trnorm.normalize_for_eval("Saat 14:30'da 3 kişi GELDİ."))
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU          {p.name}, {p.total_memory/2**30:.1f} GiB, sm_{p.major}{p.minor}")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize(); print("cuda matmul  ok", float((x @ x).float().abs().mean()))
    archs = torch.cuda.get_arch_list()
    if f"sm_{p.major}{p.minor}" not in archs:
        print("WARNING: this torch build has no kernels for your GPU:", archs)
else:
    print("GPU          none (CPU mode)")
PYEOF
fi

log "done. Activate with:  source $STT_DIR/path.sh"
