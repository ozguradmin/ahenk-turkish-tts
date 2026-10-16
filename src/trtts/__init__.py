"""trtts: Turkish text to speech that reads real-world text correctly.

EMA Lightning (Apache-2.0, 8.6M parameters, CPU or GPU) speaks; trnorm decides what it says.
EMA Lightning reads clean text very well, but its bundled front-end reads many numbers, times,
prices and abbreviations symbol by symbol ("21:00'e" -> "yirmi bir iki nokta sıfır sıfır kesme e").
trtts swaps that front-end for trnorm, the configuration measured on TR-Zor (155 hard sentences):
14.0% -> 1.9% WER (benchmarks/tts/results/tr_zor/RESULTS.md).

    from trtts import TurkishTTS

    tts = TurkishTTS()                                   # downloads EMA Lightning once (~34 MB)
    speech = tts.say("Randevunuz 14.10.2026 saat 09:30'da.", path="randevu.wav")
    for chunk in tts.stream("Uzun bir metin..."):        # float32 chunks at speech.sample_rate
        ...

Needs the optional dependency: pip install "trnorm[tts]".
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Union

from trnorm import normalize

__all__ = ["TurkishTTS", "prepare"]

SAMPLE_RATE = 48000


def prepare(text: str) -> str:
    """The exact text trtts hands to the model: trnorm's spoken form."""
    return normalize(text)


class TurkishTTS:
    """EMA Lightning with trnorm as its text front-end.

    device: "auto" (GPU if available), "cpu" or "cuda". engine: an already built
    ema_lightning.EMA to reuse (its front-end is switched to trnorm in place).
    """

    def __init__(self, device: str = "auto", engine=None):
        if engine is None:
            try:
                from ema_lightning import EMA
            except ImportError as e:  # pragma: no cover - depends on the environment
                raise ImportError('trtts needs EMA Lightning: pip install "trnorm[tts]"') from e
            engine = EMA(device=device)
        frontend = getattr(engine, "_frontend", None)
        if frontend is None or not hasattr(frontend, "spoken"):
            raise RuntimeError(
                "this ema-lightning version has no replaceable front-end; trtts supports ema-lightning 1.x"
            )
        # EMA's Frontend = alphabet(spoken(text)), where spoken() runs normalizer-tr. trnorm has already
        # produced the spoken form, so spoken() becomes the identity and only the alphabet fit remains.
        # That includes brand names (normalizer-tr's pronunciation catalog): trnorm reads them itself,
        # in Turkish spelling that the alphabet step keeps ("iPhone'umdan" -> "ayfonumdan",
        # "Instagram" -> "İnstagram" -> "instagram", never "ınstagram").
        frontend.spoken = lambda text: text
        self.engine = engine

    def lightning(self) -> "TurkishTTS":
        """EMA Lightning's CUDA-graph fast path (NVIDIA GPUs only; a no-op warning elsewhere)."""
        self.engine.lightning()
        return self

    def say(
        self,
        text: Union[str, List[str]],
        speed: float = 1.0,
        seed: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE,
        path: Optional[str] = None,
    ):
        """Speech for one text (or a list of Speech for a list), as ema_lightning.EMA.say()."""
        spoken = prepare(text) if isinstance(text, str) else [prepare(t) for t in text]
        return self.engine.say(spoken, speed=speed, seed=seed, sample_rate=sample_rate, path=path)

    def stream(
        self, text: str, speed: float = 1.0, seed: Optional[int] = None, sample_rate: int = SAMPLE_RATE
    ) -> Iterator:
        """float32 audio chunks as they are made, as ema_lightning.EMA.stream()."""
        if not isinstance(text, str):
            raise TypeError("stream() takes one text; for many, call stream() once per text")
        return self.engine.stream(prepare(text), speed=speed, seed=seed, sample_rate=sample_rate)
