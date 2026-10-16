"""Written Turkish -> spoken Turkish.

The normalizer is a sequence of regex passes, from the most specific
patterns (URLs, phone numbers, dates, times, money) to the most generic
(bare numbers, acronyms, symbols). Every numeric pass understands an
optional apostrophe suffix and re-agrees it with the spoken form:

    "15.00'de"  -> "on beşte"      "SGK'ye"   -> "se ge kaya"
    "4'üncü"    -> "dördüncü"      "%15'lik"  -> "yüzde on beşlik"

A separately written question particle (mI, miydi, misiniz...) or clitic dA
right after an expanded token is re-agreed the same way: "5 TL mi" -> "beş lira mı".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from . import lexicon as L
from .casing import is_all_caps, tr_capitalize, tr_lower, tr_upper
from .numbers import (
    VOWELS,
    attach_suffix,
    digits_to_words,
    harmonize_clitic,
    harmonize_suffix,
    number_to_words,
    ordinal_to_words,
    read_decimal_fraction,
    read_digit_group,
    read_long_digits,
)

LOW = "a-zçğıöşüâîû"
UP = "A-ZÇĞİÖŞÜÂÎÛ"
LET = LOW + UP

# optional apostrophe suffix: "'te", "'DA" (uppercase suffixes in all-caps text)
SUF = rf"(?:'([{LET}]+))?"
# integer with optional dot thousands, optional comma decimal
INT_TH = r"\d{1,3}(?:\.\d{3})+"
NUM = rf"(?:{INT_TH}|\d+)(?:,\d+)?"
# number in unit/currency context: also accepts dot decimals ("1.5 kg")
NUMU = rf"(?:{INT_TH}(?:,\d+)?|\d+[.,]\d+|\d+)"
NB = r"(?<![\w.,])"  # number must not continue a word or another number
NE = r"(?![\w]|[.,]\d)"  # ...and must end here

_ORDINAL_SUFFIX_RE = re.compile(r"^(?:[ıiuü]?nc[ıiuü])(.*)$")


# --------------------------------------------------------------------------- helpers


def _suffix(stem: str, suffix: str | None) -> str:
    if not suffix:
        return stem
    return attach_suffix(stem, tr_lower(suffix))


def _attach_number_suffix(n: int, words: str, suffix: str | None) -> str:
    """Attach a suffix to a spoken integer; ordinal suffixes become ordinals."""
    if not suffix:
        return words
    s = tr_lower(suffix)
    m = _ORDINAL_SUFFIX_RE.match(s)
    if m:  # 4'üncü, 2'nci, 1'inci, 3'ncü (misspelt) -> ordinal (+ rest: "si", "de")
        return _suffix(ordinal_to_words(n), m.group(1)) if m.group(1) else ordinal_to_words(n)
    return attach_suffix(words, s)


def read_number(s: str, dot_word: str = "nokta") -> str:
    """Read "1.500", "3,5", "1.250,75", "1.5" (dot decimal), "-5"."""
    s = s.strip()
    neg = s[:1] in "-−"
    if neg:
        s = s[1:]
    if "," in s:
        ip, fp = s.split(",", 1)
        ipv = int(ip.replace(".", "") or "0")
        if fp == "5" and ipv > 0:  # 1,5 -> bir buçuk
            out = f"{number_to_words(ipv)} buçuk"
        else:
            out = f"{number_to_words(ipv)} virgül {read_decimal_fraction(fp)}"
    elif "." in s:
        if re.fullmatch(INT_TH, s):
            out = number_to_words(int(s.replace(".", "")))
        else:
            ip, fp = s.split(".", 1)
            if dot_word == "virgül" and fp == "5" and int(ip or "0") > 0:
                out = f"{number_to_words(int(ip))} buçuk"
            else:
                out = f"{number_to_words(int(ip or '0'))} {dot_word} {read_decimal_fraction(fp.replace('.', ''))}"
    else:
        out = number_to_words(int(s)) if len(s) <= 21 else digits_to_words(s)
    return ("eksi " + out) if neg else out


def _int_value(s: str) -> int | None:
    s = s.replace(".", "")
    return int(s) if s.isdecimal() else None


_BETWEEN_RE = re.compile(r"[ \t]+(?:arası|arasında|arasındaki|aralığı|aralığında)\b")


def _ila(text: str, end: int) -> str:
    """Range word: "ila" ("4 ila 5 dakika"), but "ile" before "arası(nda)" ("13 ile 17 arasında")."""
    return "ile" if _BETWEEN_RE.match(text, end) else "ila"


def _read_range(a: str, b: str, dot_word: str = "nokta", word: str = "ila") -> str:
    return f"{read_number(a, dot_word)} {word} {read_number(b, dot_word)}"


def _letters(token: str) -> str:
    letters = [ch for ch in token if ch.isalpha()]
    # a Latin I inside a spelled acronym is the letter i, not ı: IBM, HDMI, IETT typed without Turkish keys
    return " ".join(
        "i" if ch == "I" and len(letters) > 1 else L.LETTER_NAMES.get(ch, tr_lower(ch)) for ch in letters
    )


def _pronounceable(word: str) -> bool:
    w = tr_lower(word)
    flags = [c in VOWELS for c in w if c.isalpha()]
    n = len(flags)
    if n <= 2 or not any(flags):
        return False
    run = maxrun = 0
    for v in flags:
        run = 0 if v else run + 1
        maxrun = max(maxrun, run)
    if n >= 5:
        return maxrun <= 3 and sum(flags) / n >= 0.3
    starts_cc = not flags[0] and not flags[1]
    ends_cc = not flags[-1] and not flags[-2]
    return maxrun <= 2 and not starts_cc and not ends_cc


# --------------------------------------------------------------------------- clitics (mI, dA)

# Private-use char put after an expansion that is followed by a clitic; _pass_clitics consumes it.
_MARK = "\ue000"
_I4, _A2 = "[ıiuü]", "[ae]"
_PERS = rf"(?:m|n(?:{_I4}z)?|k|l{_A2}r)"  # -m, -n, -nIz, -k, -lAr after -DI / -sA
# question particle mI and its copula forms (miyim, misiniz, midir, miydi, miymiş, miyse...), clitic dA
_CLITIC_RE = re.compile(
    rf"d{_A2}|m{_I4}(?:y{_I4}[mz]|s{_I4}n(?:{_I4}z)?|d{_I4}r(?:l{_A2}r)?|yd{_I4}{_PERS}?"
    rf"|ym{_I4}ş(?:{_I4}[mz]|s{_I4}n(?:{_I4}z)?|l{_A2}r)?|ys{_A2}{_PERS}?)?"
)
_WORD_AFTER_RE = re.compile(rf"([ \t]+)([{LET}]+)(?![\w'])")
_QUESTION_NEXT_RE = re.compile(r"\s*[\"'’”»)\]]*\s*\?")
_BARE_COUNT_RE = re.compile(r"[\d.,\s-]+")  # source token of a plain cardinal / range / list: "2", "1.000", "2-3"


def _clitic_at(text: str, pos: int, count: bool = False) -> re.Match | None:
    """Match " <clitic>" at pos: all-lowercase or all-caps mI / dA forms only.

    count: the word before is a bare cardinal, after which "müdür" is the noun ("2 müdür?").
    """
    m = _WORD_AFTER_RE.match(text, pos)
    if not m:
        return None
    word = m.group(2)
    low = tr_lower(word)
    if word not in (low, tr_upper(low)) or not _CLITIC_RE.fullmatch(low):
        return None
    # "müdür(ler)" is also a noun ("2 müdür, 3 memur", "İK müdür yardımcısı"): mI+dIr only right before "?"
    if low.startswith("müdür") and (count or not _QUESTION_NEXT_RE.match(text, m.end())):
        return None
    return m


def _last_word(s: str) -> str:
    toks = s.split()
    return tr_lower(toks[-1]) if toks else ""


def _sub(rx: re.Pattern, repl: Callable[[re.Match], str], text: str) -> str:
    """rx.sub(repl, text), marking expansions followed by a clitic so _pass_clitics re-agrees it.

    Only a changed last word counts as an expansion: "NATO" -> "nato" keeps the writer's "NATO mu".
    """
    out, pos = [], 0
    for m in rx.finditer(text):
        new = repl(m)
        out += [text[pos : m.start()], new]
        count = bool(_BARE_COUNT_RE.fullmatch(m.group(0)))
        if _last_word(new) and _last_word(new) != _last_word(m.group(0)) and _clitic_at(text, m.end(), count):
            out.append(_MARK)
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def _pass_clitics(text: str) -> str:
    """Re-agree mI / dA after a marked expansion with its last spoken vowel: "beş lira mi" -> "beş lira mı".

    The only chain, dA + mI, is re-agreed left to right: "beş lira de mi" -> "beş lira da mı".
    """
    head, *parts = text.split(_MARK)
    out = head
    for part in parts:
        pos, prev = 0, ""
        while (m := _clitic_at(part, pos)) and (not prev or (prev in ("da", "de") and tr_lower(m.group(2))[0] == "m")):
            word = m.group(2)
            prev = harmonize_clitic(tr_lower(out[-48:] + m.group(1)), tr_lower(word))
            out += m.group(1) + (tr_upper(prev) if word != tr_lower(word) else prev)
            pos = m.end()
        out += part[pos:]
    return out


def _context_before(text: str, pos: int, width: int = 32) -> str:
    return tr_lower(text[max(0, pos - width) : pos])


def _has_word(context: str, words, last: int = 3) -> bool:
    """True if one of the last `last` words before a number starts with one of `words`.

    Digits break the window: in "kodunu gir, 21 Kasım 2025", the year is not a code.
    """
    tail = re.split(r"\d", context)[-1]
    toks = re.findall(rf"[{LOW}]+", tail)
    if toks and toks[-1] in _MONTHS_LOWER:
        return False
    return any(t.startswith(w) for t in toks[-last:] for w in words)


# nominal suffixes after a cue word: possessive / plural / -lI / -sIz, then a case ending
# ("numara-nız", "kod-u-nu", "numara-sı-nda", "hesab-ı-na", "kod-lu", "kişi-ye", "yıl-ı-nda", "lira-lık")
_NOMINAL_SUFFIX_RE = re.compile(
    r"(?:[ıiuü]|s[ıiuü]|[ıiuü]?m(?:[ıiuü]z)?|[ıiuü]?n(?:[ıiuü]z)?|l[ae]r(?:[ıi](?:m[ıi]z|n[ıi]z|[mn])?)?"
    r"|l[ıiuü]k?|s[ıiuü]z)?"
    r"(?:n?[ıiuü]n|y?[ae]|n[ae]|n?[dt][ae]n?|y?l[ae]|y?[ıiuü]|n[ıiuü])?"
)


def _cue_word(tok: str, cues) -> bool:
    """tok is one of `cues`, as a whole word or with nominal suffixes ("numaranız", "kişiye"); never a word
    that merely starts with a cue ("normalde" is not "no", "kartal" not "kart", "takipçi" not "takip")."""
    return any(
        tok == c or (c not in L.ID_EXACT and tok.startswith(c) and _NOMINAL_SUFFIX_RE.fullmatch(tok, len(c)))
        for c in cues
    )


def _has_cue(context: str, cues, last: int) -> bool:
    """Like _has_word, with whole-word cue matching (_cue_word)."""
    toks = re.findall(rf"[{LOW}]+", re.split(r"\d", context)[-1])
    if toks and toks[-1] in _MONTHS_LOWER:
        return False
    return any(_cue_word(t, cues) for t in toks[-last:])


def _quantity(context: str, after: str) -> bool:
    """A count, not an identifier: "Müşteri sayısı 25000", "Telefonla 15000 kişiye", "Kart limiti 25000 lira"."""
    toks = re.findall(rf"[{LOW}]+", re.split(r"\d", context)[-1])
    if toks and toks[-1] in L.QUANTITY_BEFORE:
        return True
    nxt = re.match(rf"[ \t]*([{LOW}]+)", after)
    return bool(nxt) and _cue_word(nxt.group(1), L.QUANTITY_AFTER)


_MONTHS_LOWER = {tr_lower(m) for m in L.MONTHS if m}
# spelled-out unit words after a dot decimal make it a quantity: "10.40 metre" -> on virgül kırk metre
_UNIT_WORDS_RE = "|".join(
    sorted({*L.UNITS.values(), "kilo", "lira", "dolar", "avro", "kuruş", "derece", "saat", "dakika", "saniye", "gün", "yıl", "ay", "hafta"}, key=len, reverse=True)
)


# --------------------------------------------------------------------------- passes


# emoji, pictographs, flags, skin tones, keycap marks: not read aloud (a raw emoji reaches the TTS model otherwise)
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U0001FC00-\U0001FFFD\u2600-\u27BF\u2B00-\u2BFF\u2300-\u23FF\uFE0E\uFE0F\u20E3]+"
)


def _preclean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace(_MARK, "")
    text = _EMOJI_RE.sub(" ", text)
    text = re.sub(r"[​‌‍﻿]", "", text)
    text = text.replace(" ", " ").replace(" ", " ")
    # apostrophes between word characters -> '
    text = re.sub(r"(?<=\w)[’‘`´ʼ′](?=\w)", "'", text)
    text = re.sub(r"[“”„«»]", '"', text)
    text = re.sub(r"[’‘]", "'", text)
    text = text.replace("…", "...")
    # en/em dash between numbers -> hyphen (ranges); elsewhere a pause
    text = re.sub(r"(?<=\d)\s?[–—]\s?(?=\d)", "-", text)
    text = re.sub(r"\s[–—]\s", ", ", text)
    text = text.replace("−", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _name_forms(key: str) -> set[str]:
    """Written forms of a FOREIGN_NAMES key: as written, ALL CAPS, and unless exact-case also lower/Capitalized.

    An exact-case key gets no Turkish ALL CAPS form either: "GEMİNİ" is the Turkish word "gemini" in capitals.
    """
    if key in L.FOREIGN_NAMES_EXACT_CASE:
        return {key, key.upper()}
    low = key.lower()
    # ALL CAPS (Latin and Turkish: "INSTAGRAM", "İNSTAGRAM"), youtube, Youtube, İnstagram
    return {key, key.upper(), tr_upper(key), low, low[:1].upper() + low[1:], tr_upper(low[:1]) + low[1:]}


def _build_names() -> tuple[dict[str, str], re.Pattern]:
    table: dict[str, str] = {}
    for key, spoken in L.FOREIGN_NAMES.items():
        for form in _name_forms(key):
            table.setdefault(form, spoken)
    for key, spoken in L.FOREIGN_NAMES.items():  # an exact key beats another key's generated form
        table[key] = spoken
    alt = "|".join(re.escape(k) for k in sorted(table, key=len, reverse=True))
    # not inside a word, URL, e-mail address or handle ("google.com", "@google", "/google"); an optional
    # apostrophe suffix ("Google'da"), then a word end
    rx = re.compile(rf"(?<![\w@./:\\])(?<!\w')({alt})(?:'([{LET}]+))?(?![\w@+]|'[{LET}]|[./:]\w)")
    return table, rx


_NAMES, _NAMES_RE = _build_names()


def _pass_names(text: str) -> str:
    """Brands and loanwords to their Turkish pronunciation: "iPhone'umdan" -> "ayfonumdan", "Google'da" -> "Gugılda".

    A Capitalized name stays capitalized (Turkish rules: "Instagram" -> "İnstagram", never a dotless ı);
    ALL-CAPS and lowercase ones come out lowercase, like spelled acronyms.
    """

    def repl(m: re.Match) -> str:
        written = m.group(1)
        spoken = _suffix(_NAMES[written], m.group(2))
        return tr_capitalize(spoken) if written[0].isupper() and not is_all_caps(written) else spoken

    return _sub(_NAMES_RE, repl, text)


_URL_RE = re.compile(r"\b(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|gov|edu|io|ai|tr|co|info|biz|me|app|dev)(?:\.[a-z]{2})?(?:/[^\s]*)?", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")

_TLD = {"com": "kom", "tr": "te re", "net": "net", "org": "org", "gov": "gov", "edu": "edu", "io": "ay o", "ai": "ey ay"}


def _read_web_token(tok: str) -> str:
    parts = []
    for piece in re.split(r"([./@_-])", tok):
        if not piece:
            continue
        if piece == ".":
            parts.append("nokta")
        elif piece == "/":
            parts.append("eğik çizgi")
        elif piece == "@":
            parts.append("et")
        elif piece == "_":
            parts.append("alt çizgi")
        elif piece == "-":
            parts.append("tire")
        elif piece.lower() == "www":
            parts.append("ve ve ve")
        elif piece.lower() in _TLD:
            parts.append(_TLD[piece.lower()])
        elif piece.isdecimal():  # not "²" ("¹".isdigit() is True)
            parts.append(read_digit_group(piece) if len(piece) <= 4 else digits_to_words(piece))
        else:
            parts.append(tr_lower(piece))
    return " ".join(parts)


def _pass_web(text: str) -> str:
    text = _sub(_EMAIL_RE, lambda m: _read_web_token(m.group()), text)

    def url(m: re.Match) -> str:
        tok = re.sub(r"^https?://", "", m.group(), flags=re.I).rstrip("/")
        return _read_web_token(tok)

    return _sub(_URL_RE, url, text)


def _build_abbr_re() -> re.Pattern:
    keys = sorted(L.ABBREVIATIONS, key=len, reverse=True)
    alt = "|".join(re.escape(k) for k in keys)
    return re.compile(rf"(?<![\w.])({alt})(?=[\s,;:!?)\"']|$)")


_ABBR_RE = _build_abbr_re()
_PAGE_RE = re.compile(r"(?<![\w.])([sc])\.\s?(?=\d)")


def _pass_abbreviations(text: str) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        out = L.ABBREVIATIONS[key]
        if key in L.SENTENCE_FINAL_ABBR:
            rest = text[m.end() :]
            if not rest.strip() or re.match(rf"\s+[{UP}]", rest):
                out += "."
        return out

    text = _sub(_ABBR_RE, repl, text)
    text = re.sub(r"(?<![\w.])(No|Nu|NO)\s?[:.]\s?(?=\d)", "numara ", text)
    text = re.sub(r"(?<![\w.])(Kat|Daire|Blok|D|Tel|GSM)\s?:\s?(?=\d)", lambda m: {"D": "daire", "Tel": "telefon", "GSM": "ge se me"}.get(m.group(1), m.group(1)) + " ", text)
    return _PAGE_RE.sub(lambda m: "sayfa " if m.group(1) == "s" else "cilt ", text)


_PRODUCT_RE = re.compile(r"(?<=\d)[ \t]?[x×*][ \t]?(?=[\d₺$€£])")


def _pass_products(text: str) -> str:
    """x, ×, * between numbers is "çarpı" ("12 x 833,25 TL", "4*5"); runs before the numbers become words."""

    def repl(m: re.Match) -> str:
        s = m.start()
        if m.group().strip() == "x" and text[s - 1] == "0" and (s < 2 or not text[s - 2].isalnum()):
            return m.group()  # hexadecimal 0x1F
        return " çarpı "

    return _PRODUCT_RE.sub(repl, text)


_PHONE_RE = re.compile(
    rf"(?<![\w+])(\+90[\s.-]?|0[\s.-]?)?\(?([2-58]\d{{2}})\)?[\s.-]?(\d{{3}})[\s.-]?(\d{{2}})[\s.-]?(\d{{2}}){SUF}(?![\w])"
)
_PHONE_444_RE = re.compile(r"(?<![\w])(444)[\s.-]?(\d)[\s.-]?(\d{3})(?!\d)")


def _pass_phones(text: str) -> str:
    def repl(m: re.Match) -> str:
        prefix, area, a, b, c, suf = m.groups()
        raw = m.group(0)
        # without a 0/+90 prefix, require separators so plain 10-digit ids stay ids
        if not prefix and not re.search(r"[\s.()-]", raw):
            return raw
        if prefix and prefix.startswith("+"):
            parts = ["artı doksan", number_to_words(int(area))]
        elif prefix:
            parts = ["sıfır " + number_to_words(int(area))]
        else:
            parts = [number_to_words(int(area))]
        parts += [read_digit_group(a), read_digit_group(b), _suffix(read_digit_group(c), suf)]
        return ", ".join(parts)

    text = _sub(_PHONE_RE, repl, text)
    return _sub(
        _PHONE_444_RE,
        lambda m: f"dört yüz kırk dört, {read_digit_group(m.group(2))}, {read_digit_group(m.group(3))}",
        text,
    )


_IBAN_RE = re.compile(r"\bTR\s?(\d{2})((?:\s?\d{4}){5})\s?(\d{2})\b")


def _pass_iban(text: str) -> str:
    def repl(m: re.Match) -> str:
        # check digits and the last pair as numbers, the four-digit groups digit by digit (easy to write down)
        groups = [read_digit_group(m.group(1))]
        groups += [digits_to_words(g) for g in re.findall(r"\d{4}", m.group(2))]
        return "te re " + ", ".join(groups + [read_digit_group(m.group(3))])

    return _sub(_IBAN_RE, repl, text)


_PLATE_RE = re.compile(rf"(?<![\w])(0[1-9]|[1-7]\d|8[01])\s([{UP}]{{1,3}})\s(\d{{2,5}}){SUF}(?![\w])")


def _pass_plates(text: str) -> str:
    def repl(m: re.Match) -> str:
        city, letters, num, suf = m.groups()
        if len(num) == 4:
            tail = f"{read_digit_group(num[:2])} {read_digit_group(num[2:])}"
        else:
            tail = read_digit_group(num)
        return _suffix(f"{read_digit_group(city)} {_letters(letters)} {tail}", suf)

    return _sub(_PLATE_RE, repl, text)


def _date_words(d: int, mo: int, y: str | None) -> str:
    out = f"{number_to_words(d)} {L.MONTHS[mo]}"
    if y:
        out += f" {number_to_words(int(y))}"
    return out


_DATE_RE = re.compile(rf"{NB}(\d{{1,2}})([./-])(\d{{1,2}})\2(\d{{4}}|\d{{2}}){SUF}(?![\w]|[.,]\d)")
_MONTH_YEAR_RE = re.compile(rf"{NB}(0?[1-9]|1[0-2])[./]((?:19|20)\d{{2}}){SUF}(?![\w]|[.,]\d)")
_ISO_DATE_RE = re.compile(rf"{NB}(\d{{4}})-(\d{{2}})-(\d{{2}}){SUF}(?![\w]|[.,]\d)")
# year first with slashes or dots: 2026/12/31, 2026/1/5, 2026.12.31
_YMD_RE = re.compile(rf"{NB}((?:1[89]|20)\d{{2}})(?:/(\d{{1,2}})/(\d{{1,2}})|\.(\d{{2}})\.(\d{{2}})){SUF}(?![\w]|[.,/]\d)")


def _pass_dates(text: str) -> str:
    def iso(m: re.Match) -> str:
        y, mo, d, suf = m.groups()
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return _suffix(_date_words(int(d), int(mo), y), suf)
        return m.group(0)

    def ymd(m: re.Match) -> str:
        y, mo1, d1, mo2, d2, suf = m.groups()
        mo, d = int(mo1 or mo2), int(d1 or d2)
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return _suffix(_date_words(d, mo, y), suf)
        return m.group(0)

    text = _sub(_YMD_RE, ymd, text)

    def dmy(m: re.Match) -> str:
        d, _, mo, y, suf = m.groups()
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return _suffix(_date_words(int(d), int(mo), y), suf)
        return m.group(0)

    text = _sub(_DATE_RE, dmy, _sub(_ISO_DATE_RE, iso, text))

    def month_year(m: re.Match) -> str:
        return _suffix(f"{L.MONTHS[int(m.group(1))]} {number_to_words(int(m.group(2)))}", m.group(3))

    return _sub(_MONTH_YEAR_RE, month_year, text)


def _time_words(h: str, mi: str, s: str | None = None) -> str:
    out = number_to_words(int(h))
    if s is not None:
        return f"{out} {read_digit_group(mi)} {read_digit_group(s)}"
    if mi != "00":
        out += " " + read_digit_group(mi)
    return out


_TIME_COLON_RE = re.compile(rf"{NB}([01]?\d|2[0-4]):([0-5]\d)(?::([0-5]\d))?{SUF}(?![\w]|[.,]\d)")
_TIME_DOT_RE = re.compile(rf"{NB}([01]?\d|2[0-4])\.([0-5]\d){SUF}(?![\w]|[.,]\d)")
_TIME_RANGE_RE = re.compile(
    rf"{NB}([01]?\d|2[0-4])[:.]([0-5]\d)\s?-\s?([01]?\d|2[0-4])[:.]([0-5]\d){SUF}(?![\w]|[.,]\d)"
)


def _pass_times(text: str) -> str:
    text = _sub(
        _TIME_RANGE_RE,
        lambda m: _suffix(
            f"{_time_words(m.group(1), m.group(2))} {_ila(text, m.end())} {_time_words(m.group(3), m.group(4))}", m.group(5)
        ),
        text,
    )
    text = _sub(_TIME_COLON_RE, lambda m: _suffix(_time_words(m.group(1), m.group(2), m.group(3)), m.group(4)), text)

    def dot(m: re.Match) -> str:
        h, mi, suf = m.groups()
        if re.search(r"[%‰₺$€£]\s?$", text[max(0, m.start() - 2) : m.start()]):
            return m.group(0)
        before = _context_before(text, m.start())
        after = tr_lower(text[m.end() : m.end() + 16])
        timey = bool(suf) or _has_word(before, L.TIME_CONTEXT) or mi in ("00", "30")
        timey = timey or bool(re.match(rf"\s+(?:{'|'.join(L.TIME_CONTEXT)})", after))
        if not timey:
            return m.group(0)
        return _suffix(_time_words(h, mi), suf)

    return _sub(_TIME_DOT_RE, dot, text)


_CUR_SYMBOLS = r"₺|\$|€|£|¥|₽"
_CUR_CODES = r"TL|TRY|USD|EUR|GBP|JPY|RUB|CHF"
_CUR_PREFIX_RE = re.compile(rf"(?<![\w])({_CUR_SYMBOLS})\s?({NUMU}){SUF}(?![\w]|[.,]\d)")
_CUR_POSTFIX_RE = re.compile(
    rf"{NB}({NUMU})(?:\s?-\s?({NUMU}))?\s?({_CUR_SYMBOLS}|(?:{_CUR_CODES})(?![\w])){SUF}(?![\w]|[.,]\d)"
)
_SCALE_CUR_RE = re.compile(rf"\b(bin|milyon|milyar|trilyon)\s({_CUR_SYMBOLS}|{_CUR_CODES})(?![\w]){SUF}")


def _money_words(amount: str, cur_word: str) -> str:
    sub = L.CURRENCY_SUBUNIT.get(cur_word)
    m = re.fullmatch(rf"({INT_TH}|\d+)[.,](\d{{2}})", amount)
    if sub and m and not re.fullmatch(INT_TH, amount):
        major = number_to_words(int(m.group(1).replace(".", "")))
        minor = int(m.group(2))
        return f"{major} {cur_word}" + (f" {number_to_words(minor)} {sub}" if minor else "")
    return f"{read_number(amount, dot_word='virgül')} {cur_word}"


def _pass_money(text: str) -> str:
    def prefix(m: re.Match) -> str:
        sym, amount, suf = m.groups()
        return _suffix(_money_words(amount, L.CURRENCY_WORDS[sym]), suf)

    def postfix(m: re.Match) -> str:
        a, b, cur, suf = m.groups()
        word = L.CURRENCY_WORDS[cur]
        if b:
            return _suffix(f"{read_number(a, 'virgül')} {_ila(text, m.end())} {_money_words(b, word)}", suf)
        return _suffix(_money_words(a, word), suf)

    text = _sub(_CUR_PREFIX_RE, prefix, text)
    text = _sub(_CUR_POSTFIX_RE, postfix, text)
    return _sub(_SCALE_CUR_RE, lambda m: _suffix(f"{m.group(1)} {L.CURRENCY_WORDS[m.group(2)]}", m.group(3)), text)


_PCT_PREFIX_RE = re.compile(rf"(?<![\w])([%‰])\s?({NUMU})(?:\s?-\s?%?\s?({NUMU}))?{SUF}(?![\w]|[.,]\d)")
_PCT_POSTFIX_RE = re.compile(rf"{NB}({NUMU})\s?%{SUF}(?![\w])")


def _pass_percent(text: str) -> str:
    def pre(m: re.Match) -> str:
        sym, a, b, suf = m.groups()
        word = "yüzde" if sym == "%" else "binde"
        body = read_number(a, "virgül") if not b else _read_range(a, b, "virgül", _ila(text, m.end()))
        return _suffix(f"{word} {body}", suf)

    text = _sub(_PCT_PREFIX_RE, pre, text)
    return _sub(_PCT_POSTFIX_RE, lambda m: _suffix(f"yüzde {read_number(m.group(1), 'virgül')}", m.group(2)), text)


_DEG_RE = re.compile(rf"(?<![\w.,])(-?{NUMU})\s?°\s?([CF])?(?![{LET}]){SUF}")


def _pass_degrees(text: str) -> str:
    def repl(m: re.Match) -> str:
        num, scale, suf = m.groups()
        out = f"{read_number(num, 'virgül')} derece"
        if scale == "F":
            out += " fahrenhayt"
        return _suffix(out, suf)

    return _sub(_DEG_RE, repl, text)


def _build_unit_re() -> re.Pattern:
    keys = sorted(L.UNITS, key=len, reverse=True)
    alt = "|".join(re.escape(k) for k in keys)
    return re.compile(rf"{NB}(-?{NUMU})(?:\s?-\s?({NUMU}))?\s?({alt})(?![\w²³/]){SUF}")


_UNIT_RE = _build_unit_re()
_SPEED_RE = re.compile(
    rf"{NB}({NUMU})(?:\s?-\s?({NUMU}))?\s?({'|'.join(re.escape(k) for k in sorted(L.SPEED_UNITS, key=len, reverse=True))})(?![\w]){SUF}"
)


def _pass_units(text: str) -> str:
    def speed(m: re.Match) -> str:
        a, b, unit, suf = m.groups()
        per, word = L.SPEED_UNITS[unit]
        num = read_number(a, "virgül") if not b else _read_range(a, b, "virgül", _ila(text, m.end()))
        return _suffix(f"{per} {num} {word}", suf)

    def unit(m: re.Match) -> str:
        a, b, u, suf = m.groups()
        num = read_number(a, "virgül") if not b else _read_range(a, b, "virgül", _ila(text, m.end()))
        return _suffix(f"{num} {L.UNITS[u]}", suf)

    text = _sub(_SPEED_RE, speed, text)
    return _sub(_UNIT_RE, unit, text)


# tokens mixing letters and digits: TR42648163, A4, 5G, MP3, H2O, COVID-19, F-16
# ...with a decimal version at the end: GPT-3.5
_ALNUM_RE = re.compile(
    rf"(?<![\w'])(?=[{LET}\d-]*\d)(?=[{LET}\d-]*[{UP}])([{LET}\d]+(?:-[{LET}\d]+)*(?:(?<=\d)[.,]\d+)?){SUF}(?![\w]|[.,]\d)"
)


def _read_alnum(tok: str) -> str:
    out = []
    for piece in re.findall(rf"[{LET}]+|\d+(?:[.,]\d+)?", tok):
        if not piece.isdigit() and piece[0].isdigit():  # GPT-3.5
            out.append(read_number(piece))
        elif piece.isdigit():
            out.append(digits_to_words(piece) if len(piece) >= 5 else read_digit_group(piece))
        elif piece in L.ACRONYMS:
            out.append(L.ACRONYMS[piece])
        elif is_all_caps(piece) or len(piece) == 1:
            out.append(tr_lower(piece) if _pronounceable(piece) else _letters(piece))
        else:
            out.append(piece)
    return " ".join(out)


def _pass_alnum(text: str) -> str:
    def repl(m: re.Match) -> str:
        tok, suf = m.groups()
        # plain words with a trailing number like "Ankara2023" are rare; require caps letters
        return _suffix(_read_alnum(tok), suf)

    return _sub(_ALNUM_RE, repl, text)


# either end may be negative: "-10 - 5 derece", "-5 - -2 gün" (a sign only after a space or the start)
_RANGE_RE = re.compile(rf"(?:(?<![\w.,-])(-)|{NB})({NUM})\s?-\s?(-?)({NUM}){SUF}(?![\w]|[.,]\d)")


def _pass_ranges(text: str) -> str:
    def repl(m: re.Match) -> str:
        sa, a, sb, b, suf = m.groups()
        before = _context_before(text, m.start(), 40)
        after = tr_lower(text[m.end() : m.end() + 30])
        av, bv = _int_value(a.split(",")[0]), _int_value(b.split(",")[0])
        if sa and av is not None:
            av = -av
        if sb and bv is not None:
            bv = -bv
        first = read_number((sa or "") + a)

        def last(v: int | None) -> str:
            if v is not None and suf and not sb:
                return _attach_number_suffix(v, read_number(b), suf)
            return _suffix(read_number(sb + b), suf)

        descending = av is not None and bv is not None and bv <= av  # ranges go up; "3-1", "6-6", "2023-24" don't
        if descending or _has_word(before, L.SCORE_CONTEXT) or re.match(rf"\s*(?:{'|'.join(L.SCORE_CONTEXT)})", after):
            return f"{first} {last(bv)}"
        return f"{first} {_ila(text, m.end())} {last(_int_value(b))}"

    return _sub(_RANGE_RE, repl, text)


def _address_number(text: str, start: int, end: int) -> bool:
    """N/M is a building / apartment number: after "No", "numara", "kapı", "daire"... or before "nolu"."""
    words = re.findall(rf"[{LET}]+", _context_before(text, start, 24))
    if words and words[-1] in L.ADDRESS_NUMBER_CONTEXT:
        return True
    return re.match(r"\s*(?:nolu|no'lu|numaralı)\b", tr_lower(text[end : end + 16])) is not None


_FRACTION_RE = re.compile(rf"(?<![\w./,])(\d{{1,3}})/(\d{{1,3}}){SUF}(?![\w/]|[.,]\d)")
_RATIO_RE = re.compile(rf"{NB}(\d+,\d+)/(\d{{1,3}}){SUF}(?![\w/]|[.,]\d)")


def _pass_fractions(text: str) -> str:
    def repl(m: re.Match) -> str:
        a, b, suf = int(m.group(1)), int(m.group(2)), m.group(3)
        if (a, b) == (24, 7):
            return _suffix("yirmi dört yedi", suf)
        if _address_number(text, m.start(), m.end()):  # "No: 12/4", "Daire 3/1", "10/2 nolu": never a date
            return _suffix(f"{number_to_words(a)} bölü {number_to_words(b)}", suf)
        if b == 0:
            return m.group(0)
        if a >= b and 1 <= a <= 31 and 1 <= b <= 12:
            return _suffix(_date_words(a, b, None), suf)
        if a < b:
            return _suffix(f"{attach_suffix(number_to_words(b), 'de')} {number_to_words(a)}", suf)
        return _suffix(f"{number_to_words(a)} bölü {number_to_words(b)}", suf)

    def ratio(m: re.Match) -> str:
        return _suffix(f"{read_number(m.group(1))} bölü {number_to_words(int(m.group(2)))}", m.group(3))

    text = _sub(_RATIO_RE, ratio, text)
    return _sub(_FRACTION_RE, repl, text)


_ROMAN_RE = re.compile(
    rf"(?<![\w.])(M{{0,3}}(?:CM|CD|D?C{{0,3}})(?:XC|XL|L?X{{0,3}})(?:IX|IV|V?I{{0,3}}))\.(?=\s+([{LET}][{LET}']*))"
)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(s: str) -> int:
    total = 0
    for i, ch in enumerate(s):
        v = _ROMAN_VALUES[ch]
        if i + 1 < len(s) and _ROMAN_VALUES[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


_ROMAN_SUFFIX_RE = re.compile(rf"(?<![\w.])([IVXLC]{{2,}}|[IVX])'([{LOW}]+)(?![\w])")


def _ordinal_noun(word: str) -> bool:
    """A capitalized word after which "N." / "IV." is an ordinal: "Lig", "Murat", "Yüzyılda" (also "LİG")."""
    if is_all_caps(word):
        word = tr_capitalize(tr_lower(word))
    return word.split("'")[0] in L.ORDINAL_CAP_NOUNS or word.startswith(L.ORDINAL_CAP_PREFIXES)


_WORD_BEFORE_RE = re.compile(rf"([{LET}]+)[ \t]+$")
# a title ("Dr.", "doktor", "Sayın") or another initial ("J. I. Smith") right before a lone I. / X.
_TITLE_BEFORE_RE = re.compile(
    rf"(?:(?<![\w])(?i:{'|'.join(L.INITIAL_TITLES)})\.?|(?<![\w.])[{UP}]\.)[ \t]+$"
)


def _pass_roman(text: str) -> str:
    def suffixed(m: re.Match) -> str:
        tok, suf = m.groups()
        if not re.fullmatch(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})", tok):
            return m.group(0)
        if len(tok) == 1:
            # the written suffix tells which reading the writer had in mind: "V'te" (beşte) and "Ek I'de"
            # (birde) are numerals, "X'te" (ikste) and "V'ye" (veye) letters; when it fits both or neither,
            # a regnal name or "Madde" before it makes it the numeral ("Louis X'un", "Madde V'e")
            s, n = tr_lower(suf), _roman_to_int(tok)
            as_number = harmonize_suffix(number_to_words(n), s) == s
            as_letter = harmonize_suffix(L.LETTER_NAMES[tok], s) == s
            if as_letter and not as_number:
                return m.group(0)
            if as_letter == as_number:
                prev = _WORD_BEFORE_RE.search(text, max(0, m.start() - 24), m.start())
                if not (prev and _ordinal_noun(prev.group(1))):
                    return m.group(0)
        n = _roman_to_int(tok)
        return _attach_number_suffix(n, number_to_words(n), suf)

    text = _sub(_ROMAN_SUFFIX_RE, suffixed, text)

    def repl(m: re.Match) -> str:
        tok, nxt = m.group(1), m.group(2)
        if not tok or (len(tok) == 1 and tok not in "IVX"):
            return m.group(0)
        # before a capitalized word that is not a known noun, a lone V. is an initial ("V. Öztürk", but
        # "V. Murat"); I. and X. are ordinals ("I. Balkan Savaşı", "I. TBMM") unless a title or another
        # initial comes before them ("Dr. I. Yılmaz")
        if len(tok) == 1 and not (nxt[0].islower() or _ordinal_noun(nxt)):
            if tok == "V" or _TITLE_BEFORE_RE.search(text, max(0, m.start() - 24), m.start()):
                return m.group(0)
        return ordinal_to_words(_roman_to_int(tok))

    return _sub(_ROMAN_RE, repl, text)


_ORDINAL_DOT_RE = re.compile(rf"{NB}(\d{{1,4}})\.(?=\s+([{LET}][{LET}']*))")


def _pass_ordinals(text: str) -> str:
    def repl(m: re.Match) -> str:
        n, nxt = int(m.group(1)), m.group(2)
        if nxt[0].islower() or _ordinal_noun(nxt):
            return ordinal_to_words(n)
        return m.group(0)

    return _sub(_ORDINAL_DOT_RE, repl, text)


_NUMBER_RE = re.compile(
    rf"(?<![\w.,])(-)?((?:{INT_TH}|\d+)(?:,\d+|\.\d+)?)(?:'([{LET}]+)|([{LOW}]+))?(?![\w])"
)
_LIST_RE = re.compile(r"(?<![\w.,])\d+(?:,\d+){2,}(?![\w.,])")


# dotted versions and addresses: 1.2.3, 10.15.7, 192.168.1.1 (thousands groups like 1.250.000 are numbers)
_VERSION_RE = re.compile(r"(?<![\w.,])\d+(?:\.\d+){2,}(?![\w]|[.,]\d)")
# a version wildcard: 3.x, 2.7.x
_VERSION_X_RE = re.compile(r"(?<![\w.,])(\d+(?:\.\d+)*)\.[xX](?![\w])")
# a word, a hyphen, a number: "klod-3.5", "covid-19" (capitalized ones are read by _pass_alnum)
_WORD_HYPHEN_NUM_RE = re.compile(rf"(?<=[{LET}]{{2}})-(?=\d)")


def _dotted(s: str) -> str:
    return " nokta ".join(read_digit_group(p) if len(p) <= 3 else digits_to_words(p) for p in s.split("."))


_NUMBER_WORDS = frozenset(
    "sıfır bir iki üç dört beş altı yedi sekiz dokuz on yirmi otuz kırk elli altmış yetmiş seksen doksan "
    "yüz bin milyon milyar trilyon katrilyon".split()
)


def _postcode(text: str, m: re.Match, num: str) -> bool:
    """A bare 5-digit Turkish postcode inside an address, before the place name: "..., 34728 Kadıköy".

    The address is named by a whole address word (ADDRESS_CONTEXT: "Mah.", "sokak", "No"; not "katılan",
    "mahallede"), and the number comes right after a comma, slash, colon, address word or door number
    ("No:5 D:3 34728 Kadıköy"), never after an ordinary word ("Mahallede 15000 Suriyeli yaşıyor").
    In ALL CAPS only a door number counts: "İSTİKLAL CADDESİ 50000 KİŞİYLE DOLDU" is a count.
    """
    nxt = re.match(rf"[ \t]+([{UP}][{LET}]*)", text[m.end() :])
    if len(num) != 5 or not ("01" <= num[:2] <= "81") or not nxt:
        return False
    words = re.findall(rf"[{LOW}]+(?:'[{LOW}]+)?", _context_before(text, m.start(), 100))
    if not any(w in L.ADDRESS_CONTEXT for w in words):
        return False
    head = text[: m.start()].rstrip(" \t")
    prev = re.search(rf"([{LET}]+)\.?$", head)
    prev = tr_lower(prev.group(1)) if prev else ""
    door = head[-1:].isdecimal() or prev in _NUMBER_WORDS
    if is_all_caps(nxt.group(1)):
        return door
    return door or not head or head[-1] in ",/:;\n" or prev in L.ADDRESS_CONTEXT


def _pass_numbers(text: str) -> str:
    text = _sub(_LIST_RE, lambda m: ", ".join(number_to_words(int(x)) for x in m.group().split(",")), text)
    text = _sub(_VERSION_X_RE, lambda m: _dotted(m.group(1)) + " nokta iks", text)
    text = _sub(_VERSION_RE, lambda m: m.group() if re.fullmatch(INT_TH, m.group()) else _dotted(m.group()), text)
    text = _WORD_HYPHEN_NUM_RE.sub(" ", text)

    def spoken(m: re.Match, sign: str | None, num: str, suf: str | None) -> str:
        neg = bool(sign) and (m.start() == 0 or not text[m.start() - 1].isalnum())
        prefix = "eksi " if neg else ("-" if sign else "")
        after = tr_lower(text[m.end() : m.end() + 24])
        if num.isdigit():
            ctx = _context_before(text, m.start())
            count = _quantity(ctx, after)
            idlike = len(num) >= 5 and (
                num.startswith("0")
                or len(num) >= 10
                or (not count and _has_cue(ctx, L.ID_CONTEXT, last=3))
                or re.match(r"\s*(?:numara|no\b|nolu|kodu|kodlu)", after) is not None
                or _postcode(text, m, num)
            )
            codelike = len(num) >= 4 and not count and _has_cue(ctx, L.CODE_CONTEXT, last=2)
            if idlike or codelike:
                spoken = digits_to_words(num) if len(num) <= 8 else read_long_digits(num)
                return prefix + _suffix(spoken, suf)
            if num.startswith("0") and len(num) > 1:
                return prefix + _suffix(read_digit_group(num), suf)
            n = int(num)
            return prefix + _attach_number_suffix(n, number_to_words(n), suf)
        ctx_before = _context_before(text, m.start(), 24)
        dot_word = (
            "virgül"
            if re.match(r"\s+(?:bin|milyon|milyar|trilyon)\b", after)
            or re.match(rf"\s+(?:{_UNIT_WORDS_RE})", after)
            or _has_word(ctx_before, ("yüzde", "oran", "faiz", "puan", "not", "ortalama"))
            else "nokta"
        )
        words = read_number(num, dot_word)
        val = _int_value(num) if "," not in num and re.fullmatch(INT_TH, num) else None
        if val is not None:
            return prefix + _attach_number_suffix(val, words, suf)
        return prefix + _suffix(words, suf)

    def repl(m: re.Match) -> str:
        sign, num, suf, bare = m.groups()
        if bare is not None and tr_lower(bare) not in L.BARE_NUMBER_SUFFIXES:
            # not a suffix ("4o", "2x", "1975yılında"): the number, then the letters as a word of their own
            return spoken(m, sign, num, None) + " " + ("iks" if bare == "x" else bare)
        return spoken(m, sign, num, suf if bare is None else bare)

    return _sub(_NUMBER_RE, repl, text)


def _pass_leftover_digits(text: str) -> str:
    """Safety net for "no digit is left": a digit run no pass recognized is read plainly ("on iki.31" never reaches TTS).

    A period after a number word is a decimal point ("on iki.31" -> on iki nokta otuz bir); after any other word
    it ends an abbreviation ("Tablo.3'te" -> Tablo üçte, "md.5" -> md beş). A suffix stays on its number.
    """
    text = re.sub(
        rf"([{LET}]+)\.(?=\d)", lambda m: m.group(1) + (" nokta " if tr_lower(m.group(1)) in _NUMBER_WORDS else " "), text
    )

    def read(m: re.Match) -> str:
        digits, suf = m.groups()
        if len(digits) > 21:
            return f" {_suffix(digits_to_words(digits), suf)} "
        words = read_digit_group(digits)
        if digits.startswith("0"):  # "05" keeps its zero: no ordinal
            return f" {_suffix(words, suf)} "
        return f" {_attach_number_suffix(int(digits), words, suf)} "

    return re.sub(rf"(\d+)(?:'([{LET}]+))?", read, text)


_POWER_RE = re.compile(r"(?:(?<![\w])([A-Za-z]{1,2}|\d+(?:[.,]\d+)?)|(\)))([²³])(?![\w])")


def _pass_powers(text: str) -> str:
    """x² -> iks kare, a³ -> a küp, 10² -> 10 kare, (a+b)² -> (a+b) kare, m² (no number before it) -> metrekare."""

    def repl(m: re.Match) -> str:
        base, paren, sup = m.groups()
        base = base or paren
        if base + sup in L.UNITS:
            return L.UNITS[base + sup]
        power = "kare" if sup == "²" else "küp"
        if base.isalpha():
            if len(base) > 1:
                return m.group()
            base = L.LETTER_NAMES.get(tr_upper(base), base)
        return f"{base} {power}"

    return _sub(_POWER_RE, repl, text)


_ACRONYM_RE = re.compile(rf"(?<![\w'])([{UP}][{UP}]+|PhD)(?:'([{LET}]+))?(?![\w])")
_SINGLE_LETTER_RE = re.compile(rf"(?<![\w'.])([A-NPR-ZÇĞİÖŞÜ])(?=\s+[{LOW}])")
# a capital letter with a suffix: "X'te" -> ikste, "A'dan Z'ye" -> adan zeye (not the pronoun "O'nun"); the
# suffix is lowercase or ALL CAPS ("X'TE"), so a name with an elided article stays: "D'Artagnan", "L'Occitane"
_SINGLE_LETTER_SUFFIX_RE = re.compile(rf"(?<![\w'.])([A-NPR-ZÇĞİÖŞÜ])'([{LOW}]+|[{UP}]+)(?![\w'])")
# an initial before a name: "V. Öztürk" -> "ve. Öztürk", "J. K. Rowling"; the period stays (it may end a sentence)
_INITIAL_RE = re.compile(rf"(?<![\w'.])([{UP}])\.(?=\s+[{UP}])")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# lowercase words the number passes wrote ("on dört otuzda", "elli bin", "yüzde", "lira"): in "SAAT 14:30'DA
# BAŞLAYACAK DÖRT ETKİNLİK" they do not make the headline a mixed-case sentence
_EXPANSION_STEMS = tuple(
    sorted(_NUMBER_WORDS | {"dörd", "nokta", "virgül", "buçuk", "yüzde", "binde", "eksi", "artı", "çarpı", "bölü",
                            "ila", "lira", "kuruş", "dolar", "avro", "sent", "sterlin", "derece"})
)


def _lower_caps_headlines(text: str) -> str:
    """Lowercase ALL-CAPS sentences, keeping known acronyms for the acronym pass."""
    out = []
    for sent in _SENTENCE_SPLIT_RE.split(text):
        words = [w for w in re.findall(rf"[{LET}]+", sent) if not (w.islower() and w.startswith(_EXPANSION_STEMS))]
        letters = [c for w in words for c in w]
        caps_words = [w for w in words if is_all_caps(w)]
        if len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.8 and len(caps_words) >= 3:
            sent = re.sub(
                rf"[{LET}]+",
                lambda m: m.group() if m.group() in L.ACRONYMS or not _pronounceable_or_word(m.group()) else tr_lower(m.group()),
                sent,
            )
        out.append(sent)
    return " ".join(out)


# consonant clusters a Turkish word (or a loanword) can start or end with: "KRİZ", "SPOR", "TREN", "TÜRK",
# "DÖRT", "GENÇ", "HALK", "FİLM", "ÜST"
_ONSETS = {"bl", "br", "dr", "fl", "fr", "gl", "gr", "kl", "kr", "pl", "pr", "sf", "sk", "sl", "sm", "sn", "sp", "st",
           "şp", "şt", "tr", "skr", "spr", "str"}
_CODAS = {"rk", "rt", "rç", "rs", "rş", "rz", "rp", "rf", "rm", "rn", "rd", "rb", "rg", "nk", "nç", "nt", "ns", "nd",
          "ng", "nz", "lk", "lt", "lç", "lp", "lm", "lf", "ls", "şk", "şt", "st", "sk", "kt", "ks", "ft", "pt", "ps", "mp"}
_CONSONANT_RUN_RE = re.compile(r"[^aeıioöuüâîû]+")


def _word_shaped(w: str) -> bool:
    """Spelled like a Turkish word or loanword: a vowel, a known cluster at each end, at most four consonants in a
    row inside. "KONTROL", "TÜRK", "KRİZ" are; "AVM" (vm), "İETT" (tt), "DHA" (dh), "AİHM" (hm), "TBMM" are not."""
    low = tr_lower(w)
    if not any(c in VOWELS for c in low):
        return False
    lead = re.match(r"[^aeıioöuüâîû]*", low).group()
    trail = re.search(r"[^aeıioöuüâîû]*$", low).group()
    if (len(lead) > 1 and lead not in _ONSETS) or (len(trail) > 1 and trail not in _CODAS):
        return False
    return all(len(r) <= 4 for r in _CONSONANT_RUN_RE.findall(low.strip("bcçdfgğhjklmnprsştvyzqwx")))


def _pronounceable_or_word(w: str) -> bool:
    """In an ALL-CAPS headline: read as a word ("DÖRT KİŞİ", "KRİZ", "KONTROL") unless it is an acronym."""
    if w in L.CAPS_WORDS or _pronounceable(w) or len(w) == 1:
        return True
    return w not in L.EVAL_ACRONYMS and _word_shaped(w)


def _pass_acronyms(text: str) -> str:
    def repl(m: re.Match) -> str:
        tok, suf = m.groups()
        if tok in L.ACRONYMS:
            spoken = L.ACRONYMS[tok]
        elif tok in L.CAPS_WORDS:
            return tr_lower(tok) + (f"'{tr_lower(suf)}" if suf else "")
        elif _pronounceable(tok):
            return tr_lower(tok) + (f"'{tr_lower(suf)}" if suf else "")
        else:
            spoken = _letters(tok)
        if not suf:
            return spoken
        if " " not in spoken:  # read as a word: keep the apostrophe
            return f"{spoken}'{tr_lower(suf)}"
        return _suffix(spoken, suf)

    text = _sub(_ACRONYM_RE, repl, text)
    # lone capital letters used as labels: "C vitamini", "B planı", "A blok" (not the pronoun "O")
    return _sub(_SINGLE_LETTER_RE, lambda m: L.LETTER_NAMES.get(m.group(1), m.group(1)), text)


def _pass_letter_names(text: str) -> str:
    """A capital letter with a suffix or as an initial is read by its name: "X'te" -> ikste, "V. Öztürk" -> ve. Öztürk."""
    text = _sub(_SINGLE_LETTER_SUFFIX_RE, lambda m: _suffix(L.LETTER_NAMES[m.group(1)], m.group(2)), text)
    return _INITIAL_RE.sub(lambda m: L.LETTER_NAMES.get(m.group(1), tr_lower(m.group(1))) + ".", text)


def _pass_symbols(text: str) -> str:
    text = re.sub(r"\s&\s|(?<=\w)&(?=\w)", " ve ", text)
    text = re.sub(r"(?<=\d)\s?[x×]\s?(?=\d)", " çarpı ", text)
    text = re.sub(r"\s?=\s?", " eşittir ", text)
    text = re.sub(r"(?<=\w)\s\+\s(?=\w)|(?<=\d)\+(?=\d)", " artı ", text)
    text = re.sub(r"±", " artı eksi ", text)
    text = re.sub(r"[~≈]\s?(?=\w)", "yaklaşık ", text)
    text = re.sub(r"(?<!\w)#(?=\w)", "", text)
    text = re.sub(r"(?<!\w)@(?=\w)", "", text)
    text = re.sub(r"§\s?", "paragraf ", text)
    text = re.sub(r"№\s?", "numara ", text)
    text = re.sub(r"[*_|•→←▪►✓✔]", " ", text)
    text = re.sub(rf"\b([{UP}][{LOW}]+)/(?=[{UP}][{LOW}])", r"\1, ", text)  # Kadıköy/İstanbul -> Kadıköy, İstanbul
    text = re.sub(r"(?<=[\w])/(?=[\w])", " ", text)
    text = re.sub(r"([!?])\1+", r"\1", text)
    return text


def _cleanup(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


_PASSES = (
    _pass_names,
    _pass_web,
    _pass_abbreviations,
    _pass_products,
    _pass_iban,
    _pass_phones,
    _pass_plates,
    _pass_dates,
    _pass_times,
    _pass_money,
    _pass_percent,
    _pass_degrees,
    _pass_units,
    _pass_powers,
    _pass_alnum,
    _pass_ranges,
    _pass_fractions,
    _pass_roman,
    _pass_ordinals,
    _pass_numbers,
    _pass_leftover_digits,
    _lower_caps_headlines,
    _pass_acronyms,
    _pass_letter_names,
    _pass_symbols,
    _pass_clitics,
    _cleanup,
)


def normalize(text: str) -> str:
    """Convert written Turkish into its spoken form (for TTS input).

    Case, punctuation and apostrophes on proper nouns are kept, because
    TTS models use them for prosody.
    """
    return _run(text, _PASSES)


# The eval form keeps foreign names and single letters as written: ASR output and references spell
# "iPhone", "John F. Kennedy", "X'te", and lowercase ASR output ("x'te", "f") must still match them.
# (normalize_for_eval also builds STT training transcripts, which should keep the written names.)
_EVAL_PASSES = tuple(p for p in _PASSES if p not in (_pass_names, _pass_letter_names))


def _run(text: str, passes) -> str:
    text = _preclean(text)
    for p in passes:
        text = p(text)
    return text


_CIRCUMFLEX = str.maketrans({"â": "a", "î": "i", "û": "u", "Â": "a", "Î": "i", "Û": "u"})


_LOWER_ACRONYM_RE = re.compile(rf"(?<![\w'])([{LOW}]{{2,6}})(?:'([{LOW}]+))?(?![\w])")


def _uppercase_known_acronyms(text: str) -> str:
    """Lowercase ASR output and references write "abd", "tbmm"; treat them like "ABD", "TBMM"."""

    def repl(m: re.Match) -> str:
        tok, suf = m.groups()
        up = tr_upper(tok)
        if up not in L.EVAL_ACRONYMS:
            return m.group(0)
        return up + (f"'{suf}" if suf else "")

    return _LOWER_ACRONYM_RE.sub(repl, text)


def normalize_for_eval(text: str) -> str:
    """Canonical form for WER/CER: spoken form, Turkish lowercase, no punctuation.

    Apply the same function to references and hypotheses.
    """
    text = tr_lower(_run(_uppercase_known_acronyms(text), _EVAL_PASSES)).translate(_CIRCUMFLEX)
    # "bir buçuk" and "bir virgül beş" are the same quantity
    text = re.sub(r"\bbuçuk\b", "virgül beş", text)
    text = re.sub(r"(?<=\w)'(?=\w)", "", text)
    text = re.sub(r"[^\w\s]|_", " ", text)
    return " ".join(text.split())
