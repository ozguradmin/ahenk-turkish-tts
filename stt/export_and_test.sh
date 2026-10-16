#!/usr/bin/env bash
# =============================================================================
# export_and_test.sh — average checkpoints, export a streaming ONNX model for
# sherpa-onnx (fp32 + int8), then decode test sets with sherpa-onnx and report
# WER/CER (both sides normalised with trnorm.normalize_for_eval).
#
#   ./stt/export_and_test.sh --preset small
#   ./stt/export_and_test.sh --preset small --epoch 20 --avg 6 --chunk 32 --left 256
#   ./stt/export_and_test.sh --preset medium --test-sets "fleurs cv issai" --max-utts 0
#
# Options:
#   --preset small|medium|tiny   must match training                        [small]
#   --exp-dir DIR      default $STT_DATA/exp/<preset>-causal-bpe<V>
#   --vocab V          token model $STT_DATA/lang_bpe_<V>                    [1000]
#   --epoch E          last epoch to use                           [newest epoch-*.pt]
#   --avg N            epochs to average (with icefall's running average)   [min(6, E-1)]
#   --chunk C          decode chunk in 50 Hz frames: 16 = 320 ms, 32 = 640 ms [16]
#   --left L           left context frames (64/128/256)                      [128]
#   --test-sets "..."  any of: fleurs cv issai (manifests/<x>_cuts_test_raw)  [fleurs cv issai]
#   --max-utts N       per test set, 0 = all                                  [0]
#   --num-threads N    sherpa-onnx CPU threads                                [2]
#   --decoding-method  greedy_search | modified_beam_search                  [greedy_search]
#   --skip-export      only decode an existing export dir
# Output: <exp-dir>/export-<tag>/ (encoder/decoder/joiner{,.int8}.onnx, tokens.txt,
#         bpe.model, pretrained.pt, MODEL_CARD.md, results/*.json|tsv)
# =============================================================================
set -euo pipefail
STT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$STT_DIR/path.sh"
# shellcheck disable=SC1091
source "$STT_DIR/presets.sh"

PRESET=small; EXP_DIR=""; VOCAB=1000; EPOCH=""; AVG=""; CHUNK=16; LEFT=128
TEST_SETS="fleurs cv issai"; MAX_UTTS=0; THREADS=2; METHOD=greedy_search; SKIP_EXPORT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --preset) PRESET="$2"; shift ;;
    --exp-dir) EXP_DIR="$2"; shift ;;
    --vocab) VOCAB="$2"; shift ;;
    --epoch) EPOCH="$2"; shift ;;
    --avg) AVG="$2"; shift ;;
    --chunk) CHUNK="$2"; shift ;;
    --left) LEFT="$2"; shift ;;
    --test-sets) TEST_SETS="$2"; shift ;;
    --max-utts) MAX_UTTS="$2"; shift ;;
    --num-threads) THREADS="$2"; shift ;;
    --decoding-method) METHOD="$2"; shift ;;
    --skip-export) SKIP_EXPORT=1 ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
  shift
done

preset_args "$PRESET"
EXP_DIR="${EXP_DIR:-$STT_DATA/exp/${PRESET}-causal-bpe${VOCAB}}"
LANG_DIR="$STT_DATA/lang_bpe_${VOCAB}"
[ -d "$EXP_DIR" ] || { echo "no exp dir $EXP_DIR" >&2; exit 1; }
# train.sh keeps a copy of the token model in the exp dir; prefer it
[ -f "$EXP_DIR/tokens.txt" ] && [ -f "$EXP_DIR/bpe.model" ] && LANG_DIR="$EXP_DIR"

if [ -z "$EPOCH" ]; then
  EPOCH="$(ls "$EXP_DIR"/epoch-*.pt 2>/dev/null | sed -E 's/.*epoch-([0-9]+)\.pt/\1/' | sort -n | tail -1)"
  [ -n "$EPOCH" ] || { echo "no epoch-*.pt in $EXP_DIR" >&2; exit 1; }
fi
if [ -z "$AVG" ]; then
  AVG=$(( EPOCH - 1 < 6 ? EPOCH - 1 : 6 ))
  [ "$AVG" -lt 1 ] && AVG=1
fi
# icefall's averaged model needs epoch-(E-avg).pt; with a single epoch use the raw weights
if [ $(( EPOCH - AVG )) -ge 1 ] && [ -f "$EXP_DIR/epoch-$(( EPOCH - AVG )).pt" ]; then
  USE_AVG=1
else
  USE_AVG=0
fi

TAG="epoch-${EPOCH}-avg-${AVG}-chunk-${CHUNK}-left-${LEFT}"
OUT="$EXP_DIR/export-$TAG"
mkdir -p "$OUT/results"
EXPORT_ARGS=(--exp-dir "$EXP_DIR" --tokens "$LANG_DIR/tokens.txt" --epoch "$EPOCH" --avg "$AVG"
             --use-averaged-model "$USE_AVG" "${MODEL_ARGS[@]}"
             --causal 1 --chunk-size "$CHUNK" --left-context-frames "$LEFT")
echo "[export] $EXP_DIR  epoch=$EPOCH avg=$AVG use-averaged-model=$USE_AVG chunk=$CHUNK left=$LEFT"

if [ "$SKIP_EXPORT" = 0 ]; then
  T0=$(date +%s)
  LOG="$OUT/export.log"; : > "$LOG"
  # 1) averaged PyTorch weights (pretrained.pt) — for archiving / fine-tuning later
  python -I -u "$STT_DIR/tools/icefall_run.py" egs/librispeech/ASR/zipformer/export.py \
    "${EXPORT_ARGS[@]}" --jit 0 >> "$LOG" 2>&1 || { tail -20 "$LOG"; echo "export.py failed" >&2; exit 1; }
  mv -f "$EXP_DIR/pretrained.pt" "$OUT/pretrained.pt"
  # 2) streaming ONNX: encoder/decoder/joiner, fp32 + int8 (dynamic quantisation)
  python -I -u "$STT_DIR/tools/icefall_run.py" egs/librispeech/ASR/zipformer/export-onnx-streaming.py \
    "${EXPORT_ARGS[@]}" --enable-int8-quantization 1 >> "$LOG" 2>&1 \
    || { tail -20 "$LOG"; echo "export-onnx-streaming.py failed" >&2; exit 1; }
  grep -E "averag|Exported|total parameters" "$LOG" | sed 's/^.*\] //' || true
  SUFFIX="epoch-${EPOCH}-avg-${AVG}-chunk-${CHUNK}-left-${LEFT}"
  for part in encoder decoder joiner; do
    mv -f "$EXP_DIR/${part}-${SUFFIX}.onnx" "$OUT/${part}.onnx"
    mv -f "$EXP_DIR/${part}-${SUFFIX}.int8.onnx" "$OUT/${part}.int8.onnx"
  done
  cp -f "$LANG_DIR/tokens.txt" "$LANG_DIR/bpe.model" "$OUT/"
  cat > "$OUT/MODEL_CARD.md" <<EOF
# Turkish streaming Zipformer transducer ($PRESET) — sherpa-onnx

- Architecture: icefall Zipformer2 transducer, causal, chunk ${CHUNK} frames ($(( CHUNK * 20 )) ms), left context ${LEFT} frames.
- Files: encoder/decoder/joiner .onnx (fp32) and .int8.onnx (dynamic int8), tokens.txt, bpe.model (unigram, vocab ${VOCAB}).
- Exported from $EXP_DIR, epoch ${EPOCH}, avg ${AVG} (averaged model: ${USE_AVG}).
- Licence of the weights: Apache-2.0. Training data: see stt/README.md (FLEURS CC-BY-4.0, ISSAI TSC MIT,
  Common Voice CC0, Alania synthetic speech CC-BY-4.0 — attribution required:
  "Alania Turkish Synthetic Speech by PatientDesk AI, CC BY 4.0"; FLEURS: Conneau et al. 2022, CC BY 4.0).
- Text output: lower-case spoken form, no punctuation (trnorm.normalize_for_eval convention).
EOF
  echo "[export] done in $(( $(date +%s) - T0 ))s -> $OUT"
  ls -la "$OUT"
fi

# 3) decode test sets with sherpa-onnx (fp32 and int8)
for ts in $TEST_SETS; do
  CUTS="$STT_DATA/manifests/${ts}_cuts_test_raw.jsonl.gz"
  if [ ! -f "$CUTS" ]; then echo "[test] skip $ts (no $CUTS)"; continue; fi
  for prec in fp32 int8; do
    FLAG=(); [ "$prec" = int8 ] && FLAG=(--int8)
    echo "[test] $ts $prec"
    python -I -u "$STT_DIR/tools/decode_sherpa.py" --model-dir "$OUT" "${FLAG[@]}" --cuts "$CUTS" \
      --max-utts "$MAX_UTTS" --num-threads "$THREADS" --decoding-method "$METHOD" \
      --chunk-ms $(( CHUNK * 20 )) --name "${ts}-${prec}" --out-dir "$OUT/results" \
      --show 3 --itn-examples 3
  done
done
echo "[test] summaries ($OUT/results):"
python -I - "$OUT/results" <<'EOF'
import json, sys
from pathlib import Path
for f in sorted(Path(sys.argv[1]).glob("*.json")):
    d = json.loads(f.read_text())
    print("  %-16s WER %6.2f%%  CER %6.2f%%  RTF %.3f  (%d utts, %.3f h)"
          % (d["name"], d["WER"], d["CER"], d["RTF"], d["utterances"], d["audio_hours"]))
EOF
