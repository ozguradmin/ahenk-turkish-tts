#!/usr/bin/env bash
# =============================================================================
# prepare.sh — download + convert all permissive data, compute fbank, build the
# training mix and the token model. Every stage is resumable/idempotent.
#
#   ./stt/prepare.sh                         # default permissive recipe (~500-650 h)
#   ./stt/prepare.sh --stage 6 --stop-stage 9  # only alania -> fbank -> mix -> bpe
#   ./stt/prepare.sh --smoke                 # tiny CPU smoke-test data (~1.3 h)
#
# Stages
#   1 FLEURS tr_tr (CC-BY-4.0)                  ~2.1 GB download, ~10 h
#   2 ISSAI TSC (MIT)                           21.4 GB download, 218 h
#   3 Common Voice 17 tr (CC0) HF mirror        ~1.3 GB (train) / ~3.4 GB (--cv-train-from validated)
#     (or --cv-dir PATH for a local Mozilla Data Collective release)
#   4 [opt] audiobooks (--audiobooks HOURS; gated, see prepare/audiobooks.py)
#   5 [opt] YODAS2-Sidon tr000 (--yodas-shards N; phase-2 data, noisy captions)
#   6 Alania synthetic (CC-BY-4.0): auto = real_hours * r/(1-r), capped by --alania-max-hours
#   7 fbank for everything (lilcom, ~32 GB per 1000 h)
#   8 training/dev mix (synthetic share <= --synthetic-ratio)
#   9 token model (unigram, --vocab, + alania-domain-text sentences)
#
# Options: --stage N --stop-stage M --num-workers N (CPU procs, default 4)
#   --fbank-device cpu|cuda  --cv-dir PATH  --cv-train-from train|validated
#   --alania-hours H|auto  --alania-max-hours H (400)  --synthetic-ratio R (0.45)
#   --audiobooks H  --yodas-shards N  --vocab V (1000)  --keep-synthetic-audio
# =============================================================================
set -euo pipefail
STT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$STT_DIR/path.sh"

STAGE=1; STOP=9; NW=4; FBANK_DEV=cpu; CV_DIR=""; CV_FROM=train; ALANIA=auto; ALANIA_MAX=400
RATIO=0.45; AUDIOBOOKS=0; YODAS=0; VOCAB=1000; SMOKE=0; KEEP_SYN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift ;;
    --stop-stage) STOP="$2"; shift ;;
    --num-workers) NW="$2"; shift ;;
    --fbank-device) FBANK_DEV="$2"; shift ;;
    --cv-dir) CV_DIR="$2"; shift ;;
    --cv-train-from) CV_FROM="$2"; shift ;;
    --alania-hours) ALANIA="$2"; shift ;;
    --alania-max-hours) ALANIA_MAX="$2"; shift ;;
    --synthetic-ratio) RATIO="$2"; shift ;;
    --audiobooks) AUDIOBOOKS="$2"; shift ;;
    --yodas-shards) YODAS="$2"; shift ;;
    --vocab) VOCAB="$2"; shift ;;
    --keep-synthetic-audio) KEEP_SYN=1 ;;
    --smoke) SMOKE=1 ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
  shift
done

PY=(python -I -u)
P="$STT_DIR/prepare"
mkdir -p "$STT_DATA/logs"
run() { [ "$STAGE" -le "$1" ] && [ "$STOP" -ge "$1" ]; }
say() { printf '\n\033[1;32m[prepare] stage %s: %s\033[0m  (%s)\n' "$1" "$2" "$(date '+%F %T')"; }

if [ "$SMOKE" = 1 ]; then
  # ~1.3 h of audio, minutes of CPU: proves every script end to end
  say 1 "FLEURS (streamed subset)"
  "${PY[@]}" "$P/fleurs.py" --splits train --max-utts 300
  "${PY[@]}" "$P/fleurs.py" --splits dev --max-utts 60
  "${PY[@]}" "$P/fleurs.py" --splits test --max-utts 24
  say 2 "ISSAI TSC (streamed test subset)"
  "${PY[@]}" "$P/issai_tsc.py" --stream --splits test --max-utts 40
  say 3 "Common Voice 17 (HF mirror, test subset)"
  "${PY[@]}" "$P/commonvoice_hf.py" --splits test --max-utts 20 --num-workers 2
  say 6 "Alania (smallest shard, 230 rows)"
  "${PY[@]}" "$P/alania.py" --shards train-00654 --max-rows-per-shard 230 --dev-voice-percent 10 --num-workers 2
  say 7 "fbank"
  "${PY[@]}" "$P/compute_fbank.py" --scan --num-workers 2
  say 8 "mix"
  "${PY[@]}" "$P/make_mix.py" --dev-hours-per-source 0.1
  say 9 "BPE"
  "${PY[@]}" "$P/make_bpe.py" --vocab-size "$VOCAB" --extra-text alania-domain --extra-max-lines 100000
  exit 0
fi

if run 1; then say 1 "FLEURS"; "${PY[@]}" "$P/fleurs.py"; fi
if run 2; then say 2 "ISSAI TSC (21.4 GB)"; "${PY[@]}" "$P/issai_tsc.py" --delete-archive --num-workers "$NW"; fi
if run 3; then
  if [ -n "$CV_DIR" ]; then
    say 3 "Common Voice (local release $CV_DIR)"
    "${PY[@]}" "$P/commonvoice.py" --cv-dir "$CV_DIR" --train-from validated --num-workers "$NW"
  else
    say 3 "Common Voice 17 (HF mirror, train from $CV_FROM)"
    "${PY[@]}" "$P/commonvoice_hf.py" --train-from "$CV_FROM" --num-workers "$NW"
  fi
fi
if run 4 && [ "$AUDIOBOOKS" != 0 ]; then
  say 4 "audiobooks ($AUDIOBOOKS h, gated)"
  "${PY[@]}" "$P/audiobooks.py" --i-accept-the-terms --target-hours "$AUDIOBOOKS" --num-workers "$NW"
fi
if run 5 && [ "$YODAS" != 0 ]; then
  say 5 "YODAS2-Sidon tr000 ($YODAS shards, phase-2 data)"
  "${PY[@]}" "$P/yodas2_sidon.py" --num-shards "$YODAS" --num-workers "$NW"
fi
if run 6; then
  if [ "$ALANIA" = auto ]; then
    ALANIA="$(python -I - "$STT_DATA" "$RATIO" "$ALANIA_MAX" <<'EOF'
import json, sys
from pathlib import Path
data, r, cap = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
real = 0.0
for f in (data / "stats").rglob("*.json"):
    d = json.loads(f.read_text())
    if d.get("dataset") in ("fleurs", "issai", "cv", "audiobooks", "yodas") and str(d.get("split", "")).endswith("train"):
        real += d.get("hours", 0.0)
print(round(min(real * r / (1 - r), cap), 1))
EOF
)"
  fi
  say 6 "Alania synthetic: target $ALANIA h (synthetic ratio <= $RATIO)"
  "${PY[@]}" "$P/alania.py" --target-hours "$ALANIA" --num-workers "$NW"
fi
if run 7; then
  say 7 "fbank ($FBANK_DEV)"
  DROP=(); [ "$KEEP_SYN" = 0 ] && DROP=(--drop-audio-for alania)
  "${PY[@]}" "$P/compute_fbank.py" --scan --device "$FBANK_DEV" --num-workers "$NW" "${DROP[@]}"
fi
if run 8; then say 8 "mix"; "${PY[@]}" "$P/make_mix.py" --max-synthetic-ratio "$RATIO"; fi
if run 9; then
  say 9 "token model (vocab $VOCAB)"
  "${PY[@]}" "$P/make_bpe.py" --vocab-size "$VOCAB" --extra-text alania-domain --extra-max-lines 300000
fi
echo; echo "[prepare] done. Hours: $(python -c "import json;d=json.load(open('$STT_DATA/stats/mix.json'));print(d['train_hours'], 'h train, synthetic ratio', d['synthetic_ratio'])" 2>/dev/null || echo 'see stats/')"
