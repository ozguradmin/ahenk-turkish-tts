#!/usr/bin/env python3
"""Instruction language per line for the f022-s3tr LoRA (step 500 knows English and Turkish instructions): Turkish
when one of the first two words starts with "c", else English. Stdlib only; shipped beside the LoRA.

    pick(emotion, text, prefer="auto") -> Pick(instruction, text, why)   # a 3-tuple
    compose(instruction, text) -> "(instruction)text"                      # what generate() gets
    python instruction_pick.py surprised "Vay canına! Bunu sen mi yaptın?" [--prefer auto|en|tr]

Why (2026-10-13, two blind listening rounds, one listener, 10 emotional sentences each, 20 judgements):
- An English instruction makes the model read a "c" near the start the English way: "Vay canına" -> "vey kenına",
  "Acele" -> "Asele", "Cam" -> "Kam", "Cesaret" -> "Sesaret", "Cidden" -> "Zidden" (IPA probe on step 500: c right
  1/14 with English, 13/14 with Turkish instructions). Turkish instructions won all 10 judgements of the 5 sentences
  with a word starting with c in the first two.
- Otherwise the English instruction gives the stronger emotion: it won all 6 angry / cheerful / sad judgements (the
  Turkish one came out weaker or forced) and both on calm "Acele etmenize gerek yok." (it says "Asele" but sounds
  calm; the Turkish one said "Aacele" and was not calm).
- Half of the judged instructions were not EMOTIONS' wordings (canına, Acele, cheerful, sad, empathetic: wordings
  never trained); angry and the Cam / Cuma / Cesaret / Cidden lines used EMOTIONS' own.
Rule (prefer="auto"): Turkish when one of the first C_WORDS words of the text starts with "c" ("ç" is another letter;
a c inside a word does not count, so "Acele" gets English, the listener's choice; Turkish lowercasing after NFC, so
"CİDDEN" counts and "İşte" stays one word), else English. It agrees with 18 of the 20 judgements (not with
empathetic: Turkish picked twice, in round 2 "all three alike"). Leading quotes, dashes and other punctuation are not
words. prefer="en" / "tr" forces the language (prefer="tr" when the c of "Acele" matters more than the calm).
Laughing with a Turkish instruction was trained but never measured.
Feed it the trnorm.normalize output, which is what the model reads ("3 cami" -> "üç cami"; trnorm lowercases an
all-caps "VAY").

emotion: one of EMOTIONS (or a Turkish name, ALIASES; any case), whose single training wording per language is used
(the first "mid" wording of build_textbank.INSTRUCTIONS and its instructions_tr.json side; laughing: the first
render_corpus.LAUGH_INSTRUCTIONS wording, keep the "haha" in the text); or a custom instruction (two or more words, or
one word ending in ".", "!" or "…"; not an emotion name with punctuation), used as given (no picking; outer
parentheses dropped); or "" for plain text, returned as it is. Instructed text must hold a word to read.
Respelling (instructed text only): with an instruction "Vay" tended to come out "vey"; "Vaay" was read right 8/8
(previous version, English instructions), so every standalone "vay" gets a second "a" (case kept).
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from typing import NamedTuple

EMOTIONS: dict[str, dict[str, str]] = {  # one training wording per language (instructions_tr.json pairs)
    "angry": {"en": "Angry tone, voice getting louder.", "tr": "Kızgın bir ses tonuyla, sesi giderek yükselerek."},
    "cheerful": {"en": "Cheerful and warm, smiling while speaking.", "tr": "Neşeli ve sıcak, konuşurken gülümseyerek."},
    "sad": {"en": "Sad and quiet, with a heavy heart.", "tr": "Üzgün ve sessiz, içi burkularak."},
    "calm": {"en": "Calm, slow and soothing.", "tr": "Sakin, yavaş ve yatıştırıcı."},
    "surprised": {"en": "Astonished, with rising intonation.", "tr": "Şaşkın, yükselen bir tonlamayla."},
    "empathetic": {"en": "Warm and compassionate, as if comforting someone.",
                   "tr": "Sıcak ve şefkatli, birini teselli eder gibi."},
    "laughing": {"en": "Laughing heartily while speaking.", "tr": "Konuşurken kahkahalarla gülerek."},
}
ALIASES = {"kızgın": "angry", "öfkeli": "angry", "neşeli": "cheerful", "mutlu": "cheerful", "üzgün": "sad",
           "sakin": "calm", "şaşkın": "surprised", "şaşırmış": "surprised", "empatik": "empathetic",
           "şefkatli": "empathetic", "gülerek": "laughing", "gülme": "laughing", "laughter": "laughing"}
PREFER = ("auto", "en", "tr")
C_WORDS = 2  # the words after an English instruction whose initial "c" comes out English
ENDS = ".!…"  # a one-word custom instruction ends in one of these

_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")  # letters / digits, an apostrophe suffix kept ("Can'ın")
_VAY = re.compile(r"(?<![^\W_])(v)(a)(y)(?![^\W_])", re.IGNORECASE)
_RISK = "an English instruction tends to misread it (/k/, /s/, /z/ or /tʃ/)"


class Pick(NamedTuple):
    instruction: str  # "" = none
    text: str  # what follows the instruction ("Vay" respelled when instructed)
    why: str


def tr_lower(text: str) -> str:
    """NFC ("C" + U+0327 is "Ç"), then lowercase with Turkish I/ı and İ/i (as trnorm.tr_lower; str.lower makes "İ"
    an "i" + combining dot)."""
    return unicodedata.normalize("NFC", text).translate(_LOWER).lower().replace("i̇", "i")


def first_words(text: str, n: int = C_WORDS) -> list[str]:
    """The first n words of text, Turkish lowercase (punctuation and quotes are not words)."""
    return _WORD.findall(tr_lower(text))[:n]


def c_words(text: str, n: int = C_WORDS) -> list[str]:
    """The words among the first n of text (Turkish lowercase) that start with "c"."""
    return [w for w in first_words(text, n) if w.startswith("c")]


def respell(text: str) -> tuple[str, int]:
    """("Vay" -> "Vaay" in every standalone "vay", case kept; how many)."""
    return _VAY.subn(lambda m: m[1] + m[2] + m[2] + m[3], text)


def compose(instruction: str, text: str) -> str:
    """What generate() gets: "(instruction)" + text, or the text alone (voice_contest.compose)."""
    return f"({instruction}){text}" if instruction else text


def emotion_key(spec: str) -> str | None:
    """The EMOTIONS key of an emotion name or Turkish alias in any case ("KIZGIN", "SURPRISED"), else None."""
    for k in (tr_lower(spec.strip()), spec.strip().lower()):
        k = ALIASES.get(k, k)
        if k in EMOTIONS:
            return k
    return None


def _custom(spec: str) -> str:
    s = spec[1:-1].strip() if spec.startswith("(") and spec.endswith(")") else spec
    if "(" in s or ")" in s:
        raise ValueError(f"instruction {spec!r}: parentheses inside would end it early")
    name = emotion_key(s.rstrip(ENDS))
    if name:
        raise ValueError(f"{spec!r} names an emotion: pass {name!r} (picks the language) or a whole instruction")
    if not s or (" " not in s and s[-1] not in ENDS):
        raise ValueError(f"unknown emotion {spec!r}: one of {', '.join(EMOTIONS)} (also: {', '.join(ALIASES)}), "
                         f"or a custom instruction of two or more words or one word ending in '.', '!' or '…'")
    return s


def pick(emotion: str, text: str, prefer: str = "auto") -> Pick:
    """(instruction, text for the model, why) for one line; compose(*pick(...)[:2]) is the model's input."""
    if prefer not in PREFER:
        raise ValueError(f"prefer must be one of {PREFER}, not {prefer!r}")
    spec = (emotion or "").strip()
    if not spec:
        bracket = "; it starts with '(': the model takes that for an instruction" if text.lstrip()[:1] == "(" else ""
        return Pick("", text, f"no instruction: plain text, as it is{bracket}")
    if text.lstrip().startswith("("):
        raise ValueError(f"text starts with '(': the model would take it for an instruction: {text[:40]!r}")
    if not _WORD.search(text):
        raise ValueError(f"nothing to read after the instruction: {text!r}")
    model_text, n_vay = respell(text.lstrip())  # "(instruction)text", no space between: as trained
    vay = f"; 'Vay' -> 'Vaay' x{n_vay}" if n_vay else ""
    cw = c_words(text)
    risk = f"'{cw[0]}' starts with c, {_RISK}" if cw else ""
    key = emotion_key(spec)
    if key is None:
        note = f" ({risk}; a Turkish one does not)" if risk else ""
        ignored = f"; prefer={prefer!r} ignored" if prefer != "auto" else ""
        return Pick(_custom(spec), model_text, f"custom instruction, as given{note}{ignored}{vay}")
    if prefer != "auto":
        lang = prefer
        why = f"{lang}: prefer={prefer!r}" + (f" ({risk})" if risk and lang == "en" else "")
    elif cw:
        lang, why = "tr", f"tr: {risk}"
    else:
        lang = "en"
        why = "en: no word starting with c in the first two (English won angry/cheerful/sad by ear)"
        inner = [w for w in first_words(text) if "c" in w]
        if inner:
            why += f"; '{inner[0]}' has a c inside: English read 'Acele' as 'Asele' (prefer='tr' if the c matters)"
    return Pick(EMOTIONS[key][lang], model_text, why + vay)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("emotion", help=f"{', '.join(EMOTIONS)}, a Turkish name, a custom instruction or '' (plain)")
    ap.add_argument("text", help="the line, best after trnorm.normalize")
    ap.add_argument("--prefer", choices=PREFER, default="auto", help="instruction language [auto]")
    args = ap.parse_args(argv)
    try:
        p = pick(args.emotion, args.text, args.prefer)
    except ValueError as e:
        ap.error(str(e))
    print(compose(p.instruction, p.text))
    print(f"# {p.why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
