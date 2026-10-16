"""Text front-ends: written Turkish in, the text handed to a TTS system out.

    raw                     text exactly as written
    native                  text as written, with the system's OWN bundled normalizer switched on
                            (EMA Lightning: normalizer-tr, policy "fallback"; pocket-tts-turkish: its
                            normalizer.py). Not applicable to systems without one (MMS, Qwen3 tiny) and
                            identical to raw for piper (espeak-ng always reads numbers itself).
    normalizer_tr           PyPI normalizer-tr, default call: Normalizer().normalize(t).normalized_text
                            (ambiguity_policy="preserve": ambiguous notation such as dates, times,
                            "3." ordinals or "1990'lı" is LEFT AS WRITTEN)
    normalizer_tr_fallback  normalizer-tr with ambiguity_policy="fallback" (what EMA Lightning uses:
                            nothing is left unread)
    trnorm                  this repo's package: trnorm.normalize(t)

For every front-end except `native`, the system's bundled normalizer is bypassed (where it can be),
so the front-end alone decides how numbers, dates, symbols... are read. Each system still applies its
own mandatory input encoding (casing, alphabet filter, sentence split); see systems.py.
"""

from __future__ import annotations

from collections.abc import Callable

FRONTEND_DESCRIPTIONS = {
    "raw": "text as written, system's bundled normalizer bypassed",
    "native": "text as written, system's bundled normalizer on",
    "normalizer_tr": "normalizer-tr, default policy (preserve)",
    "normalizer_tr_fallback": "normalizer-tr, ambiguity_policy='fallback'",
    "trnorm": "trnorm.normalize",
}


def _identity(text: str) -> str:
    return text


def _normalizer_tr(policy: str | None) -> Callable[[str], str]:
    from normalizer_tr import Normalizer

    norm = Normalizer()

    def run(text: str) -> str:
        try:
            if policy is None:
                return norm.normalize(text).normalized_text
            return norm.normalize(text, ambiguity_policy=policy).normalized_text
        except Exception as e:  # invalid input / resource limit: keep the text, say so
            raise FrontendError(f"normalizer-tr failed: {e}") from e

    return run


def _trnorm() -> Callable[[str], str]:
    from trnorm import normalize

    return normalize


class FrontendError(RuntimeError):
    pass


def get_frontend(name: str) -> Callable[[str], str]:
    """The text function of a front-end. `native` is the identity here: the system does the work."""
    if name in ("raw", "native"):
        return _identity
    if name == "normalizer_tr":
        return _normalizer_tr(None)
    if name == "normalizer_tr_fallback":
        return _normalizer_tr("fallback")
    if name == "trnorm":
        return _trnorm()
    raise ValueError(f"unknown front-end {name!r}; choose from {sorted(FRONTEND_DESCRIPTIONS)}")


def frontend_version(name: str) -> dict:
    """Which code produced the front-end text (recorded with every synthesis run)."""
    from importlib import metadata

    if name in ("normalizer_tr", "normalizer_tr_fallback"):
        return {"normalizer-tr": metadata.version("normalizer-tr")}
    if name == "trnorm":
        from common import trnorm_fingerprint

        return {"trnorm": trnorm_fingerprint()}
    return {}
