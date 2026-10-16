#!/usr/bin/env python3
"""Filter cut manifests by a teacher-ASR character error rate (CER).

Hook for noisy-caption data (YODAS2, audiobooks review pool, ...): decode the
audio with a *permissively licensed* teacher (e.g. openai/whisper-large-v3 or
-turbo, MIT; NOT models with non-commercial or competitor-restricting licences
such as mihuai/turkish-stt or duxx/turkish-stt-zipformer), write a TSV

    <supervision or cut id> <TAB> <cer in 0..1>

where CER is computed after trnorm.normalize_for_eval on both sides, then:

    python stt/prepare/filter_teacher_cer.py --cer teacher_cer.tsv --max-cer 0.15 \
        --in  $STT_DATA/manifests/yodas/tr000-train-00000_cuts_train_raw.jsonl.gz \
        --out $STT_DATA/manifests/yodas/tr000-train-00000_cuts_train_raw.jsonl.gz   # in place is fine

Cuts without a CER entry are kept unless --drop-missing. Run before
compute_fbank.py (or delete the matching fbank/ file and recompute).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with python -I
from common import setup_logging  # noqa: E402

log = setup_logging("teacher_cer")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cer", required=True)
    p.add_argument("--max-cer", type=float, default=0.15)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--drop-missing", action="store_true")
    a = p.parse_args()
    cer = {}
    for line in Path(a.cer).read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition("\t")
        try:
            cer[k] = float(v.split("\t")[0])
        except ValueError:
            continue
    kept = dropped = missing = 0
    lines = []
    with gzip.open(a.inp, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            keys = [d["id"]] + [s["id"] for s in d.get("supervisions", [])]
            v = next((cer[k] for k in keys if k in cer), None)
            if v is None:
                missing += 1
                if a.drop_missing:
                    dropped += 1
                    continue
            elif v > a.max_cer:
                dropped += 1
                continue
            kept += 1
            lines.append(line)
    tmp = Path(a.out + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.writelines(lines)
    tmp.replace(a.out)
    log.info(f"kept {kept}, dropped {dropped} (CER > {a.max_cer}{' or missing' if a.drop_missing else ''}), "
             f"{missing} without a CER entry")


if __name__ == "__main__":
    main()
