#!/usr/bin/env python3
"""Run an icefall recipe script with the right sys.path, without relying on
PYTHONPATH (so it also works under ``python -I``).

    python -I stt/tools/icefall_run.py egs/librispeech/ASR/zipformer/export-onnx-streaming.py --help

The script path is relative to $ICEFALL_DIR (default: stt/icefall).
"""

import os
import runpy
import sys
from pathlib import Path

ICEFALL = Path(os.environ.get("ICEFALL_DIR", Path(__file__).resolve().parents[1] / "icefall")).resolve()

if len(sys.argv) < 2:
    sys.exit(__doc__)
script = Path(sys.argv[1])
if not script.is_absolute():
    script = ICEFALL / script
sys.argv = [str(script)] + sys.argv[2:]
sys.path[:0] = [str(script.parent), str(ICEFALL)]
runpy.run_path(str(script), run_name="__main__")
