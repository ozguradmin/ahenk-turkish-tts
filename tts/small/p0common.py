"""Shared helpers for the small on-device model P0 work (docs/KUCUK_MODEL_P0.md).

Local only: RTX 5060 Laptop GPU (8 GB) or CPU, cached weights, no paid service. Run with the
venv-student interpreter (Python 3.12, torch cu128); src/ is put on sys.path here, nothing is installed.

- judge: faster-whisper large-v3-turbo, language tr, beam 5 (the judge of benchmarks/tts/judge.py and
  tts/voxcpm/synth_eval.py), float16 on CUDA here (int8 on CPU with --judge-cpu)
- scoring: trnorm.normalize_for_eval on reference(s) and hypothesis, best reference per row, corpus WER/CER
  (benchmarks/tts/score.py semantics)
- EMA text alphabet: ema_lightning.frontend.Frontend.alphabet (Apache-2.0), copied so no normalizer-tr runs
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import types
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

HF_HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
EMA_DIR = HF_HUB / "models--canberkkkkkk--ema-lightning" / "snapshots" / "7a6ba1ad216bb2f1da9863f80ac8770a6a807632"
VOXCPM_DIR = HF_HUB / "models--openbmb--VoxCPM2" / "snapshots" / "bffb3df5a29440629464e5e839f4d214c8714c3d"
ANTALIA_DIR = HF_HUB / "models--cloud0day3--antalia-mini" / "snapshots" / "8a79528770af2ed5cb6d3b9ac1248d370429f272"
UTMOS_HUB = Path.home() / ".cache" / "torch" / "hub" / "tarepan_SpeechMOS_v1.2.0"
UTMOS_CKPT = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "utmos22_strong_step7459_v1.pt"


def cuda_dlls() -> None:
    """ctranslate2 (faster-whisper) finds cuBLAS/cuDNN in torch's own lib folder (no system CUDA install)."""
    import torch

    if sys.platform == "win32":
        os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))


def no_pyav() -> None:
    """Windows App Control blocks PyAV's DLL (av/opaque) on this PC. faster-whisper imports av only to decode
    files; we always pass 16 kHz float arrays, so a placeholder module is enough and av's DLL is never loaded."""
    try:
        import av  # noqa: F401
    except ImportError:
        sys.modules["av"] = types.ModuleType("av")


def read_jsonl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(p, rows) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_16k(path) -> "np.ndarray":
    import numpy as np
    import soundfile as sf
    import soxr

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000 and len(x):
        x = soxr.resample(x, sr, 16000, quality="VHQ")
    return np.ascontiguousarray(x, dtype=np.float32)


def to_16k(x, sr):
    import numpy as np
    import soxr

    return np.ascontiguousarray(soxr.resample(x, sr, 16000, quality="VHQ") if sr != 16000 else x, dtype=np.float32)


class Judge:
    """faster-whisper large-v3-turbo, language tr, beam 5; ctranslate2 seed reset before every file."""

    def __init__(self, device: str = "cuda"):
        if device == "cuda":
            cuda_dlls()
        no_pyav()
        import ctranslate2
        from faster_whisper import WhisperModel

        self._ct2 = ctranslate2
        self.device = device
        self.desc = f"faster-whisper:large-v3-turbo:{'float16' if device == 'cuda' else 'int8'}:beam5:tr"
        self.model = WhisperModel("large-v3-turbo", device=device,
                                  compute_type="float16" if device == "cuda" else "int8",
                                  cpu_threads=4, num_workers=1)

    def transcribe(self, audio16k, words: bool = False) -> dict:
        self._ct2.set_random_seed(0)
        t0 = time.perf_counter()
        segs, _ = self.model.transcribe(audio16k, language="tr", beam_size=5, word_timestamps=words)
        segs = list(segs)
        out = {"hyp": " ".join(s.text.strip() for s in segs).strip(), "asr_seconds": round(time.perf_counter() - t0, 3)}
        if words:
            out["words"] = [[w.word.strip(), round(w.start, 3), round(w.end, 3)] for s in segs for w in (s.words or [])]
        return out


def edits(ref: str, hyp: str):
    from rapidfuzz.distance import Levenshtein

    rw, hw = ref.split(), hyp.split()
    return Levenshtein.distance(rw, hw), len(rw), Levenshtein.distance(ref, hyp), len(ref)


def score_row(refs: list[str], hyp: str) -> dict:
    """Best reference (fewest word edits, then char edits, then first) after trnorm.normalize_for_eval."""
    from trnorm import normalize_for_eval

    h = normalize_for_eval(hyp or "")
    best = None
    for i, r in enumerate(refs):
        rn = normalize_for_eval(r)
        we, wn, ce, cn = edits(rn, h)
        key = (we, ce, i)
        if best is None or key < best[0]:
            best = (key, {"ref_index": i, "ref_norm": rn, "hyp_norm": h, "word_edits": we, "ref_words": wn,
                          "char_edits": ce, "ref_chars": cn})
    return best[1]


def corpus(rows: list[dict]) -> dict:
    we = sum(r["word_edits"] for r in rows)
    wn = sum(r["ref_words"] for r in rows)
    ce = sum(r["char_edits"] for r in rows)
    cn = sum(r["ref_chars"] for r in rows)
    return {"n": len(rows), "wer": round(100 * we / wn, 2) if wn else None, "cer": round(100 * ce / cn, 2) if cn else None,
            "row_ok": sum(r["word_edits"] == 0 for r in rows), "word_edits": we, "ref_words": wn}


# ------------------------------------------------------------------------- EMA alphabet (Apache-2.0, ema-lightning)
EMA_VOCAB = ["<pad>", "<unk>", " ", "!", '"', "%", "&", "'", "(", ")", ",", "-", ".", "/", ":", ";", "?",
             *"abcdefghijklmnopqrstuvwxyz", *"çöüğış"]
TURKISH = frozenset("çğıöşüÇĞİÖŞÜ")
TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'", "“": '"', "”": '"', "„": '"',
                            "«": '"', "»": '"', "–": "-", "—": "-", "−": "-", "…": "..."})


def ema_alphabet(text: str, vocab=None) -> str:
    """EMA's alphabet step: Turkish lowercase, diacritics off except Turkish letters (â -> a), unknown -> space."""
    vocab = frozenset(vocab or EMA_VOCAB)
    text = text.translate(TYPOGRAPHY).replace("İ", "i").replace("I", "ı").lower()
    out = []
    for ch in text:
        if ch not in TURKISH:
            ch = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        out.append(ch if ch and all(c in vocab for c in ch) else " ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


INSTR = re.compile(r"^\s*\(([^()]*)\)\s*")


def strip_instruction(text: str) -> tuple[str, str | None]:
    """'(Angry tone, ...)Bu üçüncü kez' -> ('Bu üçüncü kez', 'Angry tone, ...')."""
    m = INSTR.match(text or "")
    return (text[m.end():], m.group(1)) if m else (text, None)


def gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"


def cpu_name() -> str:
    import platform

    return platform.processor() or platform.machine()
