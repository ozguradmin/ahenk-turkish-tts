#!/usr/bin/env python3
"""Build the clean development set sets/freya_sample40.jsonl from Freya-TR-Eval.

    python benchmarks/tts/make_dev_sets.py

Freya-TR-Eval (freyavoice/freya-tr-eval, CC-BY-4.0): 495 everyday conversational sentences with no
digits. We keep a seeded sample of 40 rows, stratified by `register` (33 everyday + 7 short-native,
proportional to 406/89), in original id order. spoken = text (nothing to normalize);
category = "freya-" + register. The hand-made sets/dev_numbers10.jsonl is not generated here.
"""

from __future__ import annotations

import json
import random
import urllib.request

from common import DATA_DIR, SETS_DIR

URL = "https://huggingface.co/datasets/freyavoice/freya-tr-eval/resolve/main/freya_tr_eval.jsonl"
N = 40
SEED = 0


def main():
    src = DATA_DIR / "datasets" / "freya_tr_eval.jsonl"
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, src)
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 495, len(rows)
    rng = random.Random(SEED)
    by_reg: dict[str, list[dict]] = {}
    for r in rows:
        by_reg.setdefault(r["register"], []).append(r)
    keep = set()
    for reg in sorted(by_reg):
        k = round(N * len(by_reg[reg]) / len(rows))
        keep.update(r["id"] for r in rng.sample(by_reg[reg], k))
    out = []
    for r in rows:
        if r["id"] in keep:
            out.append({"id": f"freya_{r['id']}", "category": f"freya-{r['register']}", "text": r["text"],
                        "spoken": r["text"], "source": "freyavoice/freya-tr-eval (CC-BY-4.0)",
                        "source_id": r["id"], "type": r.get("type"), "topic": r.get("topic")})
    assert len(out) == N, len(out)
    dst = SETS_DIR / "freya_sample40.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"wrote {dst} ({len(out)} rows)")


if __name__ == "__main__":
    main()
