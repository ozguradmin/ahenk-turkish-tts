"""Turkish number reading and suffix (vowel/consonant) harmony.

number_to_words(1923)   -> "bin dokuz yüz yirmi üç"
ordinal_to_words(4)     -> "dördüncü"
attach_suffix("dört", "e")      -> "dörde"
attach_suffix("on beş", "de")   -> "on beşte"   (re-agrees d/t)
attach_suffix("altı", "a")      -> "altıya"     (inserts buffer y)
harmonize_clitic("lira", "mi")  -> "mı"         (separate question particle / dA)
"""

from __future__ import annotations

import re

ONES = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
SCALES = [
    (10**18, "kentilyon"),
    (10**15, "katrilyon"),
    (10**12, "trilyon"),
    (10**9, "milyar"),
    (10**6, "milyon"),
    (10**3, "bin"),
]
MAX_NUMBER = 10**21 - 1

VOWELS = set("aeıioöuüâîû")
BACK_VOWELS = set("aıouâû")
ROUNDED_VOWELS = set("oöuüû")
VOICELESS = set("fstkçşhp")

# Loanword stems whose suffixes take front harmony despite a back last vowel.
_FRONT_EXCEPTIONS = {"saat", "kontrol", "gol", "rol", "alkol", "petrol", "protokol", "hal", "kalp"}


def _below_thousand(n: int) -> list[str]:
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        if hundreds > 1:
            words.append(ONES[hundreds])
        words.append("yüz")
    tens, ones = divmod(rest, 10)
    if tens:
        words.append(TENS[tens])
    if ones:
        words.append(ONES[ones])
    return words


def number_to_words(n: int) -> str:
    """Cardinal reading of a non-negative (or negative) integer."""
    if n < 0:
        return "eksi " + number_to_words(-n)
    if n == 0:
        return "sıfır"
    if n > MAX_NUMBER:
        return digits_to_words(str(n))
    words: list[str] = []
    for value, name in SCALES:
        count, n = divmod(n, value)
        if count:
            if not (value == 1000 and count == 1):  # "bin", never "bir bin"
                words.extend(_below_thousand(count))
            words.append(name)
    if n:
        words.extend(_below_thousand(n))
    return " ".join(words)


def digits_to_words(digits: str) -> str:
    """Read a digit string one digit at a time: "054" -> "sıfır beş dört"."""
    return " ".join(ONES[int(d)] for d in digits if d.isdecimal())


def _last_vowel(word: str) -> str | None:
    for ch in reversed(word):
        if ch in VOWELS:
            return ch
    return None


def _harmony_state(stem: str) -> tuple[bool, bool]:
    """(is_back, is_rounded) for the stem's last vowel."""
    last_word = stem.split()[-1] if stem.split() else stem
    if last_word in _FRONT_EXCEPTIONS or last_word.endswith("saat"):  # also "kilovatsaat" (kWh)
        v = _last_vowel(last_word) or "e"
        return False, v in ROUNDED_VOWELS
    v = _last_vowel(stem)
    if v is None:  # e.g. bare consonant: default to front unrounded
        return False, False
    return v in BACK_VOWELS, v in ROUNDED_VOWELS


def _vowel4(back: bool, rounded: bool) -> str:
    if back:
        return "u" if rounded else "ı"
    return "ü" if rounded else "i"


def _vowel2(back: bool) -> str:
    return "a" if back else "e"


def harmonize_suffix(stem: str, suffix: str) -> str:
    """Re-agree a suffix with the *spoken* stem.

    Writers harmonize suffixes with the reading they have in mind
    ("5'te", "15.00'te"), but often get it wrong ("15.00'de", "SGK'ye").
    We rebuild vowels and the d/t, c/ç alternations from the stem. Invariant
    morphemes (-ki after a locative, -yor, -ken) are kept as written.
    """
    if not suffix:
        return suffix
    s = suffix.lower()
    stem_l = stem.lower()
    stem_last = stem_l[-1] if stem_l else ""
    stem_ends_vowel = stem_last in VOWELS

    # Buffer consonants: drop after consonant-final stems, add after vowel-final.
    if not stem_ends_vowel and len(s) >= 2 and s[0] in "ysnm":
        if s[0] == "y" and s[1] in VOWELS:  # "5'ye" -> "beşe", "5'yle" -> "beşle"
            s = s[1:]
        elif s[0] == "s" and s[1] in "ıiuü":  # possessive: "5'si" -> "beşi" (not conditional -sA)
            s = s[1:]
        elif s[0] == "n" and s[1] in "ıiuü" and s[2:3] == "n":  # genitive: "5'nin" -> "beşin"
            s = s[1:]
        elif (s[0] in "mn" and s[1] not in VOWELS and s[:2] != "nc") or re.match(r"[mn][ıiuü]z", s):
            # possessive -(I)m, -(I)mIz, -(I)nIz after "kuruş": "TL'mda", "TL'nız" -> "kuruşumda", "kuruşunuz"
            s = "ı" + s
    elif not stem_ends_vowel and s in ("m", "n"):  # possessive -m/-n: "25,50 TL'm" -> "... kuruşum"
        s = "ı" + s
    elif stem_ends_vowel and s[0] in VOWELS:
        # genitive -(n)In takes n ("6'ın" -> "altının"), everything else takes y
        if s[0] in "ıiuü" and s[1:2] == "n" and s[2:3] not in VOWELS:
            s = "n" + s
        else:
            s = "y" + s

    # -ki after a locative never harmonizes and sets the harmony of what follows:
    # "2023'teki", "1990'lardakiler", "2026'dakidir" (written -kı is re-spelled -ki)
    fixed_i = {
        i + 1 for i in range(2, len(s) - 1) if s[i] == "k" and s[i + 1] in "ıi" and s[i - 2] in "dt" and s[i - 1] in "ae"
    }

    back, rounded = _harmony_state(stem_l)
    prev = stem_last
    out = []
    for i, ch in enumerate(s):
        if i in fixed_i:
            ch = "i"
            back, rounded = False, False
        elif ch in "aeâ":
            ch = _vowel2(back)
            back, rounded = ch in BACK_VOWELS, False
        elif ch in "ıiuü":
            ch = _vowel4(back, rounded)
            rounded = ch in ROUNDED_VOWELS
        elif ch in "oö":
            back, rounded = ch == "o", True
        elif ch in "dt" and i == 0:
            ch = "t" if prev in VOICELESS else "d"
        elif ch in "cç" and i == 0:
            ch = "ç" if prev in VOICELESS else "c"
        out.append(ch)
        prev = ch
    return "".join(out)


def harmonize_clitic(stem: str, clitic: str) -> str:
    """Re-agree a separately written clitic (question mI and its inflections, dA) with the spoken stem.

    Only vowels change; consonants and buffer y are kept: ("lira", "mi") -> "mı",
    ("altı", "de") -> "da", ("dokuz", "misiniz") -> "musunuz". Both args lowercase.
    """
    back, rounded = _harmony_state(stem)
    out = []
    for ch in clitic:
        if ch in "ae":
            ch = _vowel2(back)
            back, rounded = ch in BACK_VOWELS, False
        elif ch in "ıiuü":
            ch = _vowel4(back, rounded)
            rounded = ch in ROUNDED_VOWELS
        out.append(ch)
    return "".join(out)


def attach_suffix(stem: str, suffix: str) -> str:
    """Attach a (re-harmonized) suffix to a spoken stem, handling dört->dörd."""
    if not suffix:
        return stem
    suf = harmonize_suffix(stem, suffix)
    if suf and suf[0] in VOWELS and stem.endswith("dört"):
        stem = stem[:-4] + "dörd"
    return stem + suf


def ordinal_to_words(n: int) -> str:
    """Ordinal reading: 1 -> birinci, 4 -> dördüncü, 20 -> yirminci."""
    cardinal = number_to_words(n)
    stem = cardinal
    back, rounded = _harmony_state(stem)
    v = _vowel4(back, rounded)
    if stem[-1] in VOWELS:
        suf = "nc" + v
    else:
        suf = v + "nc" + v
    if suf[0] in VOWELS and stem.endswith("dört"):
        stem = stem[:-4] + "dörd"
    return stem + suf


def read_decimal_fraction(digits: str) -> str:
    """Read the part after the decimal comma: "5" -> beş, "05" -> sıfır beş, "25" -> yirmi beş."""
    if not digits:
        return ""
    if len(digits) > 3:
        return digits_to_words(digits)
    zeros = len(digits) - len(digits.lstrip("0"))
    rest = digits[zeros:]
    parts = ["sıfır"] * zeros
    if rest:
        parts.append(number_to_words(int(rest)))
    return " ".join(parts)


def read_digit_group(group: str) -> str:
    """Read a digit group keeping leading zeros: "05" -> "sıfır beş", "693" -> "altı yüz doksan üç"."""
    zeros = len(group) - len(group.lstrip("0"))
    rest = group[zeros:]
    parts = ["sıfır"] * zeros
    if rest:
        parts.append(number_to_words(int(rest)))
    return " ".join(parts)


def read_long_digits(digits: str) -> str:
    """Read an identifier-like digit string in speakable groups (3-3-2-2 style)."""
    n = len(digits)
    if n <= 4:
        return read_digit_group(digits)
    sizes = _GROUP_PATTERNS.get(n)
    if sizes is None:
        sizes = [3] * (n // 3)
        if n % 3 == 1:
            sizes[-1] = 2
            sizes.append(2)
        elif n % 3 == 2:
            sizes.append(2)
    groups = []
    i = 0
    for size in sizes:
        groups.append(digits[i : i + size])
        i += size
    return ", ".join(read_digit_group(g) for g in groups)


_GROUP_PATTERNS = {
    5: [3, 2],
    6: [3, 3],
    7: [3, 2, 2],
    8: [3, 3, 2],
    9: [3, 3, 3],
    10: [3, 3, 2, 2],
    11: [3, 3, 3, 2],
}
