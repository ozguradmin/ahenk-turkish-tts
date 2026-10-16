"""trtts command line: Turkish text -> WAV with EMA Lightning + trnorm.

    trtts "Saat 09:30'da 3. kata gelin." -o ses.wav
    echo "Ücret 1.250,50 TL." | trtts -o ucret.wav
    trtts -f metin.txt -o kitap.wav --speed 1.1
    trtts "SGK'ye 15.00'te" --dry-run           # only print what would be spoken
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="trtts", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="text to speak (default: read stdin)")
    ap.add_argument("-f", "--file", type=Path, help="read the text from a UTF-8 file")
    ap.add_argument("-o", "--output", type=Path, default=Path("trtts.wav"), help="WAV file to write [trtts.wav]")
    ap.add_argument("--speed", type=float, default=1.0, help="0.25-4 [1.0]")
    ap.add_argument("--seed", type=int, default=0, help="same seed, same audio [0]")
    ap.add_argument("--sample-rate", type=int, default=48000, choices=(48000, 24000, 16000, 8000))
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--dry-run", action="store_true", help="print the spoken form and exit (no model needed)")
    args = ap.parse_args(argv)

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()
    if not text.strip():
        ap.error("no text given")

    from trtts import TurkishTTS, prepare

    if args.dry_run:
        print(prepare(text))
        return 0
    speech = TurkishTTS(device=args.device).say(
        text, speed=args.speed, seed=args.seed, sample_rate=args.sample_rate, path=str(args.output)
    )
    print(f"{args.output}  {speech.duration:.2f} s, {speech.sample_rate} Hz", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
