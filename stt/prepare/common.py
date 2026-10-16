"""Shared helpers for the Turkish ASR data preparation scripts.

Conventions (all paths under $STT_DATA, default <repo>/data/stt):

    downloads/<ds>/...                      raw downloads (deletable)
    audio/<ds>/<split>/<utt>.flac           16 kHz mono 16-bit FLAC
    manifests/<ds>_cuts_<split>_raw.jsonl.gz   lhotse cuts without features
    manifests/<ds>/<shard>_cuts_<split>_raw.jsonl.gz   (sharded datasets)
    fbank/...                               same names without "_raw", with 80-dim fbank
    stats/<ds>_<split>.json                 hours / counts / dropped

Every transcript goes through ``asr_text``: trnorm.normalize_for_eval (lower-case
spoken form, digits read out, no punctuation) followed by folding of
non-Turkish Latin diacritics. Utterances that still contain characters outside
the Turkish alphabet (+ q, w, x) are dropped.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

# Be polite on shared machines: one BLAS/OpenMP pool of OMP_NUM_THREADS threads.
os.environ.setdefault("OMP_NUM_THREADS", "2")

TR_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
EXTRA_LATIN = "qwx"
ALLOWED_CHARS = frozenset(TR_ALPHABET + EXTRA_LATIN + " ")
_TR_SPECIAL = frozenset("çğıöşü")

SAMPLE_RATE = 16000


def stt_data_dir(arg: Optional[str] = None) -> Path:
    """$STT_DATA (or <repo>/data/stt)."""
    if arg:
        return Path(arg)
    env = os.environ.get("STT_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "stt"


def setup_logging(name: str = "prepare") -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    return logging.getLogger(name)


def limit_threads(n: Optional[int] = None) -> None:
    n = n or int(os.environ.get("OMP_NUM_THREADS", "2"))
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:  # torch is optional for the pure-audio steps
        pass


# --------------------------------------------------------------------- text
try:
    from trnorm import normalize_for_eval as _normalize_for_eval
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "trnorm is not installed in this venv: uv pip install -e <repo root>"
    ) from e


def _fold_char(c: str) -> str:
    """Strip diacritics from non-Turkish letters (é->e, ñ->n, ä->a, ß->ss)."""
    if c in ALLOWED_CHARS or c in _TR_SPECIAL:
        return c
    if c == "ß":
        return "ss"
    decomposed = unicodedata.normalize("NFKD", c)
    base = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return base.lower()


def asr_text(raw: Optional[str]) -> Optional[str]:
    """Training/eval transcript, or None if unusable."""
    if raw is None:
        return None
    text = _normalize_for_eval(raw)
    if not text:
        return None
    text = "".join(_fold_char(c) for c in text)
    text = " ".join(text.split())
    if not text or any(c not in ALLOWED_CHARS for c in text):
        return None
    return text


# ISSAI TSC transcripts split apostrophe suffixes off proper nouns
# ("erdoğan ın", "türkiye nin"). Join only tokens that are (practically) never
# standalone Turkish words. Real words such as "de", "da", "ya", "ne", "ki",
# "un" (flour), "ün" (fame), "tan" (dawn), "ten" (skin), "ta", "tür" (type),
# "tur" (tour), "dur" (stop), "tır" (truck) and single letters are left alone.
_SUFFIX_TOKENS = frozenset(
    """
    ın in nın nin nun nün yı yi yu yü nı ni nu nü
    yla yle dan den ndan nden nda nde te
    lar ler ları leri ların lerin lara lere larda lerde lardan lerden
    dır dir dür tir
    """.split()
)


def rejoin_split_suffixes(text: str) -> str:
    words = text.split()
    out: list[str] = []
    for w in words:
        if out and w in _SUFFIX_TOKENS:
            out[-1] = out[-1] + w
        else:
            out.append(w)
    return " ".join(out)


# -------------------------------------------------------------------- audio
def to_mono_float32(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    if sr_in == sr_out:
        return x
    import soxr

    return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)


def decode_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode wav/flac/ogg/mp3 bytes with libsndfile."""
    import io

    import soundfile as sf

    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return to_mono_float32(x), sr


def write_flac(path: Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> int:
    """Write 16-bit FLAC (peak-protected). Returns number of samples."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak * 0.999
    tmp = path.with_name(path.name + ".tmp")
    sf.write(str(tmp), x, sr, format="FLAC", subtype="PCM_16")
    os.replace(tmp, path)
    return int(x.shape[0])


# ----------------------------------------------------------------- lhotse
def make_cut(
    cut_id: str,
    audio_path: Path,
    num_samples: int,
    text: str,
    origin: str,
    speaker: Optional[str] = None,
    gender: Optional[str] = None,
    custom: Optional[Dict] = None,
    sampling_rate: int = SAMPLE_RATE,
):
    from lhotse import MonoCut, Recording, SupervisionSegment
    from lhotse.audio import AudioSource

    duration = num_samples / sampling_rate
    rec = Recording(
        id=cut_id,
        sources=[AudioSource(type="file", channels=[0], source=str(audio_path))],
        sampling_rate=sampling_rate,
        num_samples=num_samples,
        duration=duration,
    )
    sup = SupervisionSegment(
        id=cut_id,
        recording_id=cut_id,
        start=0.0,
        duration=duration,
        channel=0,
        text=text,
        language="tr",
        speaker=speaker,
        gender=gender,
        custom={"origin": origin, **(custom or {})},
    )
    return MonoCut(
        id=cut_id,
        start=0.0,
        duration=duration,
        channel=0,
        supervisions=[sup],
        recording=rec,
        custom={"origin": origin},
    )


@dataclass
class SplitStats:
    dataset: str
    split: str
    utts: int = 0
    seconds: float = 0.0
    dropped: Dict[str, int] = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    def add(self, seconds: float) -> None:
        self.utts += 1
        self.seconds += seconds

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    @property
    def hours(self) -> float:
        return self.seconds / 3600

    def summary(self) -> Dict:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "utterances": self.utts,
            "hours": round(self.hours, 3),
            "dropped": self.dropped,
            "elapsed_sec": round(time.time() - self.started, 1),
        }

    def save(self, data_dir: Path, name: Optional[str] = None) -> None:
        out = data_dir / "stats" / f"{name or f'{self.dataset}_{self.split}'}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2))


def norm_gender(g: Optional[str]) -> Optional[str]:
    """Common Voice style gender labels -> male/female/None."""
    g = (g or "").lower()
    if g.startswith("male"):
        return "male"
    if g.startswith("female"):
        return "female"
    return None


def keep_duration(seconds: float, min_s: float = 0.5, max_s: float = 30.0) -> bool:
    return min_s <= seconds <= max_s


def chars_per_second_ok(text: str, seconds: float, lo: float = 3.0, hi: float = 30.0) -> bool:
    """Reject obvious transcript/audio mismatches (empty or runaway audio)."""
    n = len(text.replace(" ", ""))
    return seconds > 0 and lo <= n / seconds <= hi


def iter_jsonl_gz(path: Path) -> Iterable[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def hf_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def http_stream(url: str, token: Optional[str] = None, timeout: int = 120):
    """File-like streaming HTTP GET (honours HTTPS_PROXY)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "turkish-stt-prep/1.0"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=timeout)


def hf_resolve_url(repo_id: str, filename: str, revision: str, repo_type: str = "dataset") -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}/resolve/{revision}/{filename}"


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_id(s: str) -> str:
    return _SAFE.sub("_", s)
