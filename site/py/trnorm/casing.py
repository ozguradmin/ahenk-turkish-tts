"""Turkish-aware casing helpers.

Python's str.lower()/upper() are locale-agnostic: "I".lower() == "i" and
"İ".lower() == "i̇" (i + combining dot). Turkish needs I<->ı and İ<->i.
"""

from __future__ import annotations

_LOWER_MAP = str.maketrans({"I": "ı", "İ": "i"})
_UPPER_MAP = str.maketrans({"i": "İ", "ı": "I"})


def tr_lower(text: str) -> str:
    """Lowercase with Turkish dotted/dotless i rules."""
    return text.translate(_LOWER_MAP).lower().replace("i̇", "i")


def tr_upper(text: str) -> str:
    """Uppercase with Turkish dotted/dotless i rules."""
    return text.translate(_UPPER_MAP).upper()


def tr_capitalize(word: str) -> str:
    """Uppercase the first letter only (Turkish rules), keep the rest."""
    if not word:
        return word
    return tr_upper(word[0]) + word[1:]


def is_all_caps(word: str) -> bool:
    """True when the word has at least two letters and every letter is uppercase."""
    letters = [c for c in word if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)
