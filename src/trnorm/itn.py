"""Inverse text normalization: spoken Turkish -> written form with digits.

For ASR output, which comes out as words:

    inverse_normalize("saat on dört otuzda yüzde on beşlik indirim")
    -> "saat 14.30'da %15'lik indirim"

A run of number words is split into the longest *canonical* numbers
(a word sequence is a number iff number_to_words(value) reproduces it),
so "sıfır beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi" becomes
"0555 555 55 55". Small numbers (<= 10) standing alone stay as words, as
Turkish style guides recommend ("bir gün", "üç kişi"), unless the context
asks for digits ("saat üçte", "yüzde beş", "beş lira").
"""

from __future__ import annotations

import re

from . import lexicon as L
from .casing import tr_lower
from .numbers import ONES, TENS, VOWELS, number_to_words, ordinal_to_words

_SCALE_WORDS = ["yüz", "bin", "milyon", "milyar", "trilyon", "katrilyon", "kentilyon"]
NUMBER_WORDS = set(ONES) | {t for t in TENS if t} | set(_SCALE_WORDS)
_NUMBER_WORDS_SORTED = sorted(NUMBER_WORDS, key=len, reverse=True)  # "altmış" before "altı"
_SINGLE_VALUE = {w: i for i, w in enumerate(ONES)}
_SINGLE_VALUE.update({w: i * 10 for i, w in enumerate(TENS) if w})
_SINGLE_VALUE.update({w: 10 ** (3 * i) if i else 100 for i, w in enumerate(_SCALE_WORDS)})

_ORDINAL_RE = re.compile(r"^(?:ı|i|u|ü)?nc(?:ı|i|u|ü)")

# plural? + possessive? + case? + ki?  (enough for suffixes that follow numbers)
_V4 = "[ıiuü]"
_V2 = "[ae]"
_SUFFIX_GRAMMAR = re.compile(
    rf"^(?:l{_V2}r)?"
    rf"(?:y?{_V4}|s{_V4}|{_V4}m{_V4}z|{_V4}n{_V4}z|{_V4}m|{_V4}n|l{_V4}k|l{_V4}|ş?{_V2}r)?"
    rf"(?:n?[dt]{_V2}n?|n?[dt]{_V2}ki|y?{_V2}|n?{_V2}|y?{_V4}|n?{_V4}|n{_V4}n|{_V4}n|y?l{_V2}|[cç]{_V2})?"
    rf"$"
)

_MONTHS_LOWER = {tr_lower(m) for m in L.MONTHS if m}
_UNIT_WORDS = {
    "lira", "liraya", "lirayı", "liralık", "kuruş", "dolar", "avro", "euro", "sterlin", "sent",
    "kilometre", "metre", "santimetre", "milimetre", "kilogram", "kilo", "gram", "miligram",
    "litre", "mililitre", "derece", "metrekare", "metreküp", "saniye", "dakika", "gigabayt",
    "megabayt", "kilobayt", "terabayt", "volt", "vat", "kilovat", "hertz", "virgül", "buçuk",
}
_DIGIT_CONTEXT_BEFORE = {
    "saat", "yüzde", "binde", "numara", "numarası", "numaralı", "no", "kod", "kodu", "kodunuz",
    "daire", "sayfa", "madde", "plaka", "hat", "dahili", "virgül", "eksi", "artı",
}


def _split_number_word(token: str) -> tuple[str, str, bool] | None:
    """token -> (base number word, suffix, is_ordinal) or None.

    "beşte" -> ("beş", "te", False); "dördüncüsü" -> ("dört", "sü", True)
    """
    for w in _NUMBER_WORDS_SORTED:
        if token.startswith(w):
            rest = token[len(w) :]
        elif w == "dört" and token.startswith("dörd") and token[4:5] in VOWELS:
            rest = token[4:]
        else:
            continue
        if not rest:
            return w, "", False
        m = _ORDINAL_RE.match(rest)
        if m:
            tail = rest[m.end() :]
            if not tail or _SUFFIX_GRAMMAR.match(tail):
                return w, tail, True
            continue
        if rest in L.BARE_NUMBER_SUFFIXES or _SUFFIX_GRAMMAR.match(rest):
            return w, rest, False
    return None


def _canonical_value(words: list[str]) -> int | None:
    """Value of a word list if it is a canonical Turkish cardinal, else None."""
    if not words or any(w not in NUMBER_WORDS for w in words):
        return None
    total = current = 0
    for w in words:
        if w in ONES:
            current += ONES.index(w)
        elif w in TENS:
            current += TENS.index(w) * 10
        elif w == "yüz":
            current = (current or 1) * 100
        else:
            total += (current or 1) * 10 ** (3 * _SCALE_WORDS.index(w))
            current = 0
    value = total + current
    return value if number_to_words(value) == " ".join(words) else None


def _format_int(v: int) -> str:
    return f"{v:,}".replace(",", ".") if v >= 10000 else str(v)


def _parse_token(raw: str):
    core = raw.rstrip(",.;:!?")
    if not core:
        return None
    low = tr_lower(core)
    parsed = _split_number_word(low)
    if parsed is None:
        return None
    # capitalized tokens are usually names ("Kırklareli"); accept only exact number words
    if core[0].isupper() and parsed[1] and parsed[1] not in L.BARE_NUMBER_SUFFIXES:
        return None
    return parsed[0], parsed[1], parsed[2], raw, core != raw


def inverse_normalize(text: str) -> str:
    """Convert spoken-form Turkish to written form with digits."""
    tokens = text.split()
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        run = []
        j = i
        while j < n:
            p = _parse_token(tokens[j])
            if p is None:
                break
            run.append(p)
            j += 1
            if p[1] or p[2] or p[4]:  # suffix, ordinal or punctuation closes the number
                break
        if not run:
            out.append(tokens[i])
            i += 1
            continue
        out.extend(_render_run(run, out, tokens[j] if j < n else ""))
        i = j
    return _post_patterns(" ".join(out))


def _render_run(run, prev_out: list[str], next_token: str) -> list[str]:
    bases = [r[0] for r in run]
    groups: list[tuple[int, int]] = []
    k = 0
    while k < len(bases):
        end = next((e for e in range(len(bases), k, -1) if _canonical_value(bases[k:e]) is not None), k + 1)
        groups.append((k, end))
        k = end
    values = [_canonical_value(bases[a:b]) for a, b in groups]

    trail_m = re.search(r"[,.;:!?]+$", run[-1][3])
    trail = trail_m.group() if trail_m else ""
    last_suffix, last_ord = run[-1][1], run[-1][2]
    suf = f"'{last_suffix}" if last_suffix else ""
    prev_word = tr_lower(prev_out[-1]).strip(",.;:!?") if prev_out else ""
    next_word = tr_lower(next_token).strip(",.;:!?")

    if all(v is not None for v in values):
        # identifiers read digit by digit: three or more single digits
        if len(groups) >= 3 and all(v < 10 and b - a == 1 for v, (a, b) in zip(values, groups)):
            return ["".join(str(v) for v in values) + suf + trail]
        # phone numbers: sıfır + area code + groups
        if len(groups) >= 4 and values[0] == 0 and 100 <= values[1] <= 999:
            parts = ["0" + str(values[1])] + [f"{v:02d}" if v < 10 else str(v) for v in values[2:]]
            return [" ".join(parts) + suf + trail]

    # "on bir yirmide" -> "11.20'de" (hour + minutes with a suffix)
    if len(groups) == 2 and all(v is not None for v in values) and last_suffix and not last_ord:
        h, mi = values
        if h <= 24 and 10 <= mi <= 59 and groups[1][1] - groups[1][0] <= 2:
            return [f"{h}.{mi:02d}{suf}{trail}"]

    rendered: list[str] = []
    for idx, ((a, b), v) in enumerate(zip(groups, values)):
        last = idx == len(groups) - 1
        originals = [r[3] for r in run[a:b]]
        if v is None:
            rendered.extend(originals)
            continue
        ordinal = last_ord and last
        single = b - a == 1
        small = single and (v <= 10 or bases[a] in ("yüz", "bin"))
        digit_ctx = (
            prev_word in _DIGIT_CONTEXT_BEFORE
            or (last and (next_word in _UNIT_WORDS or next_word in _MONTHS_LOWER))
            or len(groups) > 1
        )
        if small and not digit_ctx:
            rendered.extend(originals)
            continue
        if ordinal:
            if last_suffix:
                ord_tail = ordinal_to_words(_SINGLE_VALUE[bases[b - 1]])[len(bases[b - 1]) :]
                s = f"{v}'{ord_tail}{last_suffix}"
            else:
                s = f"{v}."
        else:
            s = _format_int(v) + (suf if last else "")
        rendered.append(s + (trail if last else ""))
    return rendered


def _post_patterns(text: str) -> str:
    text = re.sub(r"\b(\d+) virgül (\d+)\b", r"\1,\2", text)
    text = re.sub(r"\b(\d+) buçuk\b", r"\1,5", text)
    text = re.sub(r"\byüzde (\d+(?:,\d+)?)", r"%\1", text)
    text = re.sub(r"\bbinde (\d+(?:,\d+)?)", r"‰\1", text)
    text = re.sub(r"\beksi (\d)", r"-\1", text)
    # times: "saat 14 30'da" -> "saat 14.30'da"
    text = re.sub(r"\b([Ss]aat) ([01]?\d|2[0-4]) ([0-5]\d)\b", r"\1 \2.\3", text)
    return text
