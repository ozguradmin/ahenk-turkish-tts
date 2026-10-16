"""trnorm — Turkish text normalization for speech.

Public API:
    normalize(text)          written Turkish -> spoken form (for TTS front-ends)
    normalize_for_eval(text) canonical form for ASR/TTS scoring (WER/CER)
    inverse_normalize(text)  spoken form -> digits (for ASR output display)
    tr_lower / tr_upper      Turkish-aware casing
    number_to_words(n)       cardinal number reading
    ordinal_to_words(n)      ordinal number reading
"""

from .casing import tr_lower, tr_upper
from .numbers import number_to_words, ordinal_to_words
from .itn import inverse_normalize
from .normalizer import normalize, normalize_for_eval

__all__ = [
    "normalize",
    "normalize_for_eval",
    "inverse_normalize",
    "tr_lower",
    "tr_upper",
    "number_to_words",
    "ordinal_to_words",
]

__version__ = "0.1.0"
