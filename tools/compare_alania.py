"""Compare trnorm (and optionally normalizer-tr) against Alania's spoken forms.

Usage: python tools/compare_alania.py data/alania_text/data/part-00000.parquet [--n 5000] [--show 40]
Only rows whose written text contains a digit, %, or an ALL-CAPS token are used.
"""

import argparse
import random
import re
import sys

import pyarrow.parquet as pq

from trnorm import normalize, normalize_for_eval


def cer(a: str, b: str) -> float:
    import jiwer

    if not a:
        return 0.0 if not b else 1.0
    return jiwer.cer(a, b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--show", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-normalizer-tr", action="store_true")
    args = ap.parse_args()

    t = pq.read_table(args.parquet, columns=["text_raw", "text_model", "category"]).to_pylist()
    rows = [r for r in t if re.search(r"\d|%|\b[A-ZÇĞİÖŞÜ]{2,}\b", r["text_raw"])]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    systems = {"trnorm": normalize}
    if args.with_normalizer_tr:
        from normalizer_tr import Normalizer

        nt = Normalizer()
        systems["normalizer_tr"] = lambda s: nt.normalize(s).normalized_text
        # the policy EMA Lightning uses: ambiguous notation is read with a fallback instead of kept
        systems["normalizer_tr_fallback"] = lambda s: nt.normalize(s, ambiguity_policy="fallback").normalized_text

    nospace = lambda s: s.replace(" ", "")
    for name, fn in systems.items():
        exact = exact_ns = 0
        cers = []
        digits_left = 0
        bad = []
        for r in rows:
            ref = normalize_for_eval(r["text_model"])
            out = fn(r["text_raw"])
            if re.search(r"\d", out):
                digits_left += 1
            hyp = normalize_for_eval(out) if name == "trnorm" else " ".join(
                re.sub(r"[^\w\s]", " ", __import__("trnorm").tr_lower(out)).split()
            )
            if hyp == ref:
                exact += 1
            if nospace(hyp) == nospace(ref):
                exact_ns += 1
            else:
                bad.append((r["category"], r["text_raw"], out, r["text_model"]))
            cers.append(cer(nospace(ref), nospace(hyp)))
        n = len(rows)
        print(f"== {name}: n={n} exact={exact/n:.3f} exact_nospace={exact_ns/n:.3f} "
              f"mean_cer_nospace={sum(cers)/n:.4f} outputs_with_digits={digits_left/n:.3f}")
        for cat, raw, out, ref in bad[: args.show]:
            print(f"  [{cat}] {raw}\n     ours: {out}\n   alania: {ref}")


if __name__ == "__main__":
    sys.exit(main())
