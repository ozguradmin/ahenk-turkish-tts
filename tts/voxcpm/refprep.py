#!/usr/bin/env python3
"""Reference clips and rendered clips at one level, for voice_contest.py (both engines).

prep_ref(path) -> WAV bytes (32-bit float): the reference as VoxCPM2's audio encoder takes it, 16 kHz mono (librosa.load
at 16 kHz, as both engines load it), padded with zeros on the RIGHT to a whole number of 2560-sample patches (4
latent frames of 640 samples: 160 ms), so neither engine pads it again (Nano-vLLM pads on the left, the official API
and training on the right), and its active speech at TARGET_DBFS (normalize_level). A clip already a whole number of
patches only changes level. prep_ref_file writes it once to a cache dir (the official API takes a path).

normalize_output(wav, sr): a rendered clip at the same level, before it is written (VoxCPM2's output level follows the
reference's: r = 0.93 over the r1 entries).

Level (prepare_human.speech_bounds' measure at any rate): mean power of the 20 ms frames (10 ms hop) within ACTIVE_DB
of the loudest one and above FLOOR_DB (a simple energy VAD); one gain brings it to TARGET_DBFS, then the whole clip is
scaled down when its sample peak is above PEAK_DBFS (no limiter: the clip then ends a little under the target). A
clip with no frame above FLOOR_DB, or that would need more than MAX_GAIN_DB, keeps its level. Importing this module
needs no numpy.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
from pathlib import Path

SR = 16000  # VoxCPM2's audio encoder (AudioVAE v2 sample_rate)
PATCH = 2560  # samples per generation patch at SR: patch_size 4 x encoder chunk 640 (rates 2 x 5 x 8 x 8)
TARGET_DBFS, PEAK_DBFS, MAX_GAIN_DB = -20.0, -1.0, 30.0
ACTIVE_DB, FLOOR_DB = 30.0, -60.0
LOUDNESS = f"{TARGET_DBFS:g}dBFS"  # voice_contest settings "loudness"
REF_PREP = f"{SR // 1000}k/rpad{PATCH}/{LOUDNESS}"  # voice_contest settings "ref_prep" (False: the raw clip)


def active_dbfs(x, sr: int) -> float | None:
    """Active speech level of mono float audio in dBFS (module doc); None when no frame is above FLOOR_DB."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    hop = max(1, int(round(sr / 100)))
    n = len(x) // hop
    if n < 2:
        return None
    p = (x[:n * hop].reshape(n, hop) ** 2).mean(axis=1)
    fp = (p[:-1] + p[1:]) / 2  # 20 ms frames, 10 ms hop
    db = 10 * np.log10(fp + 1e-12)
    top = float(db.max())
    if top <= FLOOR_DB:
        return None
    return float(10 * np.log10(fp[db > max(top - ACTIVE_DB, FLOOR_DB)].mean()))


def peak_dbfs(x) -> float:
    import numpy as np

    p = float(np.abs(np.asarray(x)).max()) if np.asarray(x).size else 0.0
    return 20 * math.log10(p) if p > 0 else -math.inf


def normalize_level(x, sr: int, target: float = TARGET_DBFS, peak: float = PEAK_DBFS):
    """(float32 audio, info): active speech to `target`, then the whole clip down to `peak` if its peak is above it;
    info: level_in, gain_db (total), peak_cut_db (<= 0), skipped ("silent" / "gain>MAX_GAIN_DB": unchanged)."""
    import numpy as np

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    level = active_dbfs(x, sr)
    if level is None:
        return x, {"level_in": None, "gain_db": 0.0, "peak_cut_db": 0.0, "skipped": "silent"}
    gain = target - level
    for _ in range(3):  # FLOOR_DB can bind on a quiet clip and not once it is raised: measure again at the new gain
        now = active_dbfs(x * np.float32(10 ** (gain / 20)), sr)
        if now is None or abs(target - now) < 0.01:
            break
        gain += target - now
    if gain > MAX_GAIN_DB:
        return x, {"level_in": round(level, 2), "gain_db": 0.0, "peak_cut_db": 0.0, "skipped": "gain>MAX_GAIN_DB"}
    cut = min(0.0, peak - (peak_dbfs(x) + gain))
    y = (x * np.float32(10 ** ((gain + cut) / 20))).astype(np.float32)
    return y, {"level_in": round(level, 2), "gain_db": round(gain + cut, 2), "peak_cut_db": round(cut, 2),
               "skipped": ""}


def normalize_output(wav, sr: int):
    """A rendered clip (float samples) at TARGET_DBFS active, peak <= PEAK_DBFS (float32)."""
    return normalize_level(wav, sr)[0]


def pad_right(x, n: int = PATCH):
    """x with zeros appended up to a whole number of n samples (unchanged when it is one already)."""
    import numpy as np

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return np.concatenate([x, np.zeros((-len(x)) % n, dtype=np.float32)]) if len(x) % n else x


def wav_bytes(x, sr: int) -> bytes:
    """Mono 32-bit float WAV of x."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def prep_audio(x, sr: int = SR):
    """Mono float audio at SR -> the prepared reference samples (module doc): level first, then the zero padding (so
    the padding does not count in the level)."""
    return pad_right(normalize_level(x, sr)[0])


def prep_ref(path: str | Path) -> bytes:
    """A reference clip prepared for VoxCPM2's encoder (module doc), as float WAV bytes at 16 kHz."""
    import librosa

    x, _ = librosa.load(str(path), sr=SR, mono=True)
    return wav_bytes(prep_audio(x), SR)


def prep_ref_file(path: str | Path, cache_dir: str | Path) -> str:
    """prep_ref of a clip written once to cache_dir (named by the clip's path, size and mtime); its path."""
    p = Path(path)
    st = p.stat()
    tag = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}|{REF_PREP}".encode()).hexdigest()[:16]
    out = Path(cache_dir) / f"{p.stem[:40]}-{tag}.wav"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.name}.{os.getpid()}.part")
        tmp.write_bytes(prep_ref(p))
        os.replace(tmp, out)
    return str(out)
