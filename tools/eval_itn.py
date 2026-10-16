"""Round-trip check for inverse normalization on Alania spoken forms.

A spoken sentence x is safe when normalize_for_eval(inverse_normalize(x)) == normalize_for_eval(x),
i.e. ITN only changed the writing, never the content. Also reports how many
sentences got at least one number converted to digits.

Usage: python tools/eval_itn.py data/alania_text/data/part-00003.parquet [--n 10000]
"""

import argparse
import random
import re

import pyarrow.parquet as pq

from trnorm import inverse_normalize, normalize_for_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()
    rows = pq.read_table(args.parquet, columns=["text_raw", "text_model"]).to_pylist()
    rows = [r for r in rows if re.search(r"\d", r["text_raw"])]
    random.Random(1).shuffle(rows)
    rows = rows[: args.n]
    safe = converted = 0
    bad = []
    for r in rows:
        spoken = r["text_model"]
        out = inverse_normalize(spoken)
        if re.search(r"\d", out):
            converted += 1
        if normalize_for_eval(out) == normalize_for_eval(spoken):
            safe += 1
        else:
            bad.append((spoken, out))
    n = len(rows)
    print(f"n={n} round_trip_safe={safe/n:.4f} got_digits={converted/n:.4f}")
    for s, o in bad[: args.show]:
        print(f"  spoken: {s}\n  itn:    {o}\n  back:   {normalize_for_eval(o)}")


if __name__ == "__main__":
    main()
