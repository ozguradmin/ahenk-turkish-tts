"""Command line: normalize text from arguments, a file, or stdin.

    trnorm "Saat 09:30'da 3. kata gelin."
    echo "%15'lik indirim" | trnorm
    trnorm --eval < refs.txt
"""

from __future__ import annotations

import argparse
import sys

from . import normalize, normalize_for_eval


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trnorm", description="Turkish written -> spoken text normalization")
    ap.add_argument("text", nargs="*", help="text to normalize (default: read lines from stdin)")
    ap.add_argument("--eval", action="store_true", help="output the WER/CER evaluation form (lowercase, no punctuation)")
    args = ap.parse_args(argv)
    fn = normalize_for_eval if args.eval else normalize
    if args.text:
        print(fn(" ".join(args.text)))
        return 0
    for line in sys.stdin:
        print(fn(line.rstrip("\n")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
