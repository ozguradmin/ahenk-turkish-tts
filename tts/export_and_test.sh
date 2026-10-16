#!/usr/bin/env bash
# =============================================================================
# export_and_test.sh — turn a training checkpoint into a Piper voice and test it on CPU:
#   1. ONNX export (piper.train.export_onnx at the pinned commit)
#   2. <name>.onnx.json (the training config + language/dataset/quality metadata) and MODEL_CARD.md
#   3. TR-Zor sentences -> trnorm.normalize -> piper runtime (onnxruntime, CPU) -> WAV + RTF
#   4. the `piper` command-line runtime on one sentence (what Home Assistant / Wyoming runs)
#
#   ./tts/export_and_test.sh                                      # <exp>/checkpoints/last.ckpt
#   ./tts/export_and_test.sh --ckpt <exp>/checkpoints/snapshot-epoch=499.ckpt
#
# Options:
#   --exp-dir DIR      experiment directory              [$TTS_DATA/exp/antalia-medium]
#   --ckpt PATH        checkpoint to export              [<exp-dir>/checkpoints/last.ckpt]
#   --out DIR          output directory                  [<exp-dir>/export/<checkpoint name>]
#   --name NAME        voice file name                   [tr_TR-antalia-medium]
#   --data-dir DIR     also synthesize <DIR>/heldout.csv [$TTS_DATA/antalia, if present]
#   --n N              TR-Zor sentences (one per category first)          [8]
#   --threads N        onnxruntime intra-op threads                      [2]
#   --noise-scale X --length-scale X --noise-w X   inference defaults in the .onnx.json
#                                                   [0.667 / 1.0 / 0.8, Piper's defaults]
# =============================================================================
set -euo pipefail
TTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$TTS_DIR/path.sh"

EXP_DIR=""; CKPT=""; OUT=""; NAME="tr_TR-antalia-medium"; DATA_DIR=""; N=8; THREADS=2
NOISE=0.667; LENGTH=1.0; NOISE_W=0.8
while [ $# -gt 0 ]; do
  case "$1" in
    --exp-dir) EXP_DIR="$2"; shift ;;
    --ckpt) CKPT="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --name) NAME="$2"; shift ;;
    --data-dir) DATA_DIR="$2"; shift ;;
    --n) N="$2"; shift ;;
    --threads) THREADS="$2"; shift ;;
    --noise-scale) NOISE="$2"; shift ;;
    --length-scale) LENGTH="$2"; shift ;;
    --noise-w) NOISE_W="$2"; shift ;;
    -h|--help) sed -n '3,/^# ====/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
  esac
  shift
done
EXP_DIR="${EXP_DIR:-$TTS_DATA/exp/antalia-medium}"
CKPT="${CKPT:-$EXP_DIR/checkpoints/last.ckpt}"
DATA_DIR="${DATA_DIR:-$TTS_DATA/antalia}"
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
[ -f "$EXP_DIR/config.json" ] || { echo "missing $EXP_DIR/config.json (written by train.sh at the start of training)" >&2; exit 1; }
CK_NAME="$(basename "$CKPT" .ckpt)"
OUT="${OUT:-$EXP_DIR/export/$CK_NAME}"
mkdir -p "$OUT"
export OMP_NUM_THREADS="$THREADS"

echo "[export] $CKPT -> $OUT/$NAME.onnx"
T0=$(date +%s)
# work on a private copy: selection checkpoints can be replaced while a training is running
TMP_CKPT="$OUT/.export-src.ckpt"
cp "$CKPT" "$TMP_CKPT"
python -m piper.train.export_onnx --checkpoint "$TMP_CKPT" --output-file "$OUT/$NAME.onnx"

python - "$TMP_CKPT" "$EXP_DIR/config.json" "$OUT" "$NAME" "$NOISE" "$LENGTH" "$NOISE_W" "$TTS_DIR" "$CKPT" <<'PYEOF'
import json, sys, time
from pathlib import Path
import onnx, torch
tmp_ckpt, cfg_in, out, name, noise, length, noise_w, tts_dir, ckpt = sys.argv[1:]
out = Path(out)
model = onnx.load(str(out / f"{name}.onnx"))
onnx.checker.check_model(model)
cfg = json.loads(Path(cfg_in).read_text(encoding="utf-8"))
cfg["audio"] = {**cfg.get("audio", {}), "quality": "medium"}
cfg["inference"] = {**cfg.get("inference", {}), "noise_scale": float(noise), "length_scale": float(length), "noise_w": float(noise_w)}
cfg["language"] = {"code": "tr_TR", "family": "tr", "region": "TR", "name_native": "Türkçe",
                   "name_english": "Turkish", "country_english": "Turkey"}
cfg["dataset"] = "antalia"
assert cfg["audio"]["sample_rate"] == 22050 and cfg["espeak"]["voice"] == "tr", cfg
(out / f"{name}.onnx.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
ck = torch.load(tmp_ckpt, map_location="cpu", weights_only=False)
ep, gs = ck.get("epoch"), ck.get("global_step") or 0
# Lightning stores the 0-based index of the epoch being trained; end-of-epoch checkpoints = ep + 1 epochs.
# global_step counts generator + discriminator optimizer steps, i.e. 2 per batch.
info = {"epochs_trained": ep + 1 if isinstance(ep, int) else None, "batches_trained": gs // 2,
        "checkpoint": Path(ckpt).name, "exported": time.strftime("%Y-%m-%d"),
        "onnx_mb": round((out / f"{name}.onnx").stat().st_size / 2**20, 1)}
card = (Path(tts_dir) / "MODEL_CARD.template.md").read_text(encoding="utf-8")
for k, v in {**info, "name": name}.items():
    card = card.replace("{{" + k + "}}", str(v))
(out / "MODEL_CARD.md").write_text(card, encoding="utf-8")
(out / "export_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
print("[export]", json.dumps(info))
PYEOF
rm -f "$TMP_CKPT"
echo "[export] done in $(( $(date +%s) - T0 )) s: $(ls "$OUT")"

echo "[test] TR-Zor (${N} sentences) through trnorm + piper runtime, ${THREADS} CPU threads"
HELD=()
[ -f "$DATA_DIR/heldout.csv" ] && HELD=(--heldout-csv "$DATA_DIR/heldout.csv")
python "$TTS_DIR/tools/synth_test.py" --model "$OUT/$NAME.onnx" --set "$REPO_ROOT/benchmarks/tts/sets/tr_zor.jsonl" \
  --n "$N" --threads "$THREADS" --out "$OUT/samples" "${HELD[@]}"

echo "[test] piper command-line runtime"
CLI_TEXT="$(python -c 'from trnorm import normalize; print(normalize("Saat 14:30'"'"'da 3. kattaki toplantıya 25 kişi katıldı."))')"
echo "$CLI_TEXT" | piper -m "$OUT/$NAME.onnx" -f "$OUT/samples/piper_cli.wav"
python - "$OUT/samples/piper_cli.wav" <<'PYEOF'
import sys, soundfile as sf
i = sf.info(sys.argv[1])
assert i.samplerate == 22050 and i.frames > 0 and i.channels == 1, i
print(f"[test] piper CLI: {sys.argv[1]} {i.duration:.2f} s, {i.samplerate} Hz, {i.subtype}")
PYEOF
echo "[test] all done: $OUT"
