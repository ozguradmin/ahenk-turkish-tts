"""Reference-free checks for TTS round-trip results (complements WER against hand-written readings).

TR-Zor's spoken references and trnorm were written by the same author, so WER against them
can favour trnorm's reading conventions. These two checks need no spoken reference at all:

1. number recall: every number in the *written* sentence (dates split into day/month/year,
   times into hour/minute, phone numbers into groups, decimals into both sides) must come
   back, by value, in the ASR transcript of the audio. Spoken numbers in the transcript are
   parsed back to values (all sizes, ordinals included), runs of single digits may also be
   joined ("beş sekiz dört" == 584), and month names count as month numbers.
   It is lenient: "3. kat" read as "üç kat" (cardinal) still counts as a hit.
2. literal symbol reading: transcripts that contain words like "kesme", "tire",
   "iki nokta", "eğik çizgi", "işareti" which do not occur in the written sentence, i.e. the
   TTS read punctuation symbols aloud.

Usage: python benchmarks/tts/number_check.py [--set tr_zor]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from trnorm import tr_lower
from trnorm.itn import _canonical_value, _split_number_word
from trnorm.lexicon import MONTHS

HERE = Path(__file__).resolve().parent
NUMBER_CATEGORIES = {"sayi", "sira", "tarih", "saat", "para", "yuzde", "birim", "numara"}
MONTH_NUM = {tr_lower(m): i for i, m in enumerate(MONTHS) if m}
SYMBOL_WORDS = ("kesme", "tire", "iki nokta", "eğik çizgi", "işareti", "yüzde işareti", "derece işareti", "artı işareti")


def written_numbers(text: str) -> list[str]:
    out = []
    for m in re.finditer(r"\d+(?:[.,:/-]\d+)*", text):
        tok = m.group()
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", tok):
            tok = tok.replace(".", "")
        out.extend(p.lstrip("0") or "0" for p in re.split(r"[.,:/-]", tok) if p)
    return out


def heard_numbers(hyp: str) -> tuple[collections.Counter, list[list[str]]]:
    """Values heard in an ASR transcript, plus runs of single digits (for joined reading)."""
    values: list[str] = []
    for m in re.finditer(r"\d+(?:[.,:/-]\d+)*", hyp):
        tok = m.group()
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", tok):
            tok = tok.replace(".", "")
        values.extend(p.lstrip("0") or "0" for p in re.split(r"[.,:/-]", tok) if p)
    words = re.findall(r"[a-zçğıöşüâîû]+", tr_lower(hyp))
    i = 0
    digit_runs: list[list[str]] = []
    run: list[str] = []
    while i < len(words):
        w = words[i]
        if w in MONTH_NUM:
            values.append(str(MONTH_NUM[w]))
            run = []
            i += 1
            continue
        parsed = _split_number_word(w)
        if parsed is None:
            if len(run) >= 2:
                digit_runs.append(run)
            run = []
            i += 1
            continue
        # longest canonical number starting here
        bases = []
        j = i
        while j < len(words):
            p = _split_number_word(words[j])
            if p is None:
                break
            bases.append(p[0])
            j += 1
            if p[1] or p[2]:
                break
        end = next((e for e in range(len(bases), 0, -1) if _canonical_value(bases[:e]) is not None), 1)
        v = _canonical_value(bases[:end])
        if v is not None:
            values.append(str(v))
            if v < 10 and end == 1:
                run.append(str(v))
            else:
                if len(run) >= 2:
                    digit_runs.append(run)
                run = []
        i += end
    if len(run) >= 2:
        digit_runs.append(run)
    return collections.Counter(values), digit_runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tr_zor")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(HERE / "results" / args.set / "rows.jsonl", encoding="utf-8")]
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])  # hits, total, rows with symbol words, rows
    for r in rows:
        if r.get("hyp") is None:
            continue
        key = (r["system"], r["frontend"])
        hyp_low = tr_lower(r["hyp"])
        text_low = tr_lower(r["text"])
        a = agg[key]
        a[3] += 1
        if any(w in hyp_low and w not in text_low for w in SYMBOL_WORDS):
            a[2] += 1
        if r["category"] not in NUMBER_CATEGORIES:
            continue
        ref = written_numbers(r["text"])
        heard, runs = heard_numbers(r["hyp"])
        joined = {"".join(run[k:e]).lstrip("0") or "0" for run in runs for k in range(len(run)) for e in range(k + 2, len(run) + 1)}
        for n in ref:
            if heard[n] > 0:
                heard[n] -= 1
                a[0] += 1
            elif n in joined:
                a[0] += 1
        a[1] += len(ref)
    print(f"{'system':18} {'front-end':24} {'number recall':>14} {'symbols read aloud':>20}")
    for (sysname, fe), (hit, tot, sym, n) in sorted(agg.items()):
        rec = f"{100 * hit / tot:5.1f}% ({hit}/{tot})" if tot else "-"
        print(f"{sysname:18} {fe:24} {rec:>14} {f'{sym}/{n} rows':>20}")


if __name__ == "__main__":
    main()
