"""Shared paths and helpers for the Turkish TTS benchmark (benchmarks/tts).

Import this module before torch / numpy / onnxruntime / ctranslate2: it caps every
math library at THREADS (default 2, override with TTS_BENCH_THREADS) because the
machine is shared, and points the Hugging Face cache at data/tts/hf_home.
"""

from __future__ import annotations

import os

THREADS = int(os.environ.get("TTS_BENCH_THREADS", "2"))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, str(THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Idle OpenMP threads sleep instead of spinning. On this shared, oversubscribed 4-core machine the
# default (spinning) made EMA Lightning ~5x slower (RTF 1.37 vs 0.26 at load ~9) and burns CPU that
# other jobs need; on an idle machine spinning is faster. Applies to torch (libgomp) and ctranslate2.
# TTS_BENCH_OMP_WAIT_POLICY=default leaves the OpenMP runtime's own default in place.
OMP_POLICY = os.environ.get("TTS_BENCH_OMP_WAIT_POLICY", "PASSIVE")
if OMP_POLICY.lower() != "default":
    os.environ.setdefault("OMP_WAIT_POLICY", OMP_POLICY)

import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SETS_DIR = HERE / "sets"
RESULTS_DIR = HERE / "results"
LOGS_DIR = HERE / "logs"
DATA_DIR = REPO / "data" / "tts"
OUT_DIR = DATA_DIR / "out"

os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_home"))

SEED = 0
FRONTENDS = ("raw", "native", "normalizer_tr", "normalizer_tr_fallback", "trnorm")


def set_path(name: str) -> Path:
    """`--set` accepts a name under sets/ (with or without .jsonl) or a path."""
    p = Path(name)
    if p.suffix == ".jsonl" and p.exists():
        return p
    p = SETS_DIR / (name if name.endswith(".jsonl") else f"{name}.jsonl")
    if not p.exists():
        raise SystemExit(f"test set not found: {p}")
    return p


def set_name(name: str) -> str:
    return Path(name).name.removesuffix(".jsonl")


def load_set(name: str) -> list[dict]:
    rows = read_jsonl(set_path(name))
    seen = set()
    for r in rows:
        for key in ("id", "category", "text", "spoken"):
            if key not in r:
                raise SystemExit(f"{name}: row {r.get('id')} has no '{key}' field")
        if r["id"] in seen:
            raise SystemExit(f"{name}: duplicate id {r['id']}")
        seen.add(r["id"])
    return rows


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # a line cut short by a killed run
                    pass
    return out


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def latest_by_id(rows: list[dict]) -> dict[str, dict]:
    """Records are appended; the last one for an id wins."""
    return {r["id"]: r for r in rows}


def synth_path(set_: str, system: str, frontend: str) -> Path:
    return RESULTS_DIR / set_ / "synth" / f"{system}__{frontend}.jsonl"


def hyp_path(set_: str, system: str, frontend: str) -> Path:
    return RESULTS_DIR / set_ / "hyps" / f"{system}__{frontend}.jsonl"


def wav_path(set_: str, system: str, frontend: str, utt_id: str) -> Path:
    return OUT_DIR / set_ / system / frontend / f"{utt_id}.wav"


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    """Write via a temp file + rename, so a killed process never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_wav(path: Path, audio, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), audio, int(sample_rate), subtype="PCM_16")
    tmp.replace(path)


def trnorm_fingerprint() -> dict:
    """trnorm version + a hash of its source, so results say which normalizer made them."""
    import trnorm

    src = Path(trnorm.__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(src.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return {"version": getattr(trnorm, "__version__", "?"), "src_sha256": h.hexdigest()[:12]}


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "?"


def pkg_versions(names=("torch", "numpy", "transformers", "ema-lightning", "pocket-tts",
                        "pocket-tts-turkish", "piper-tts", "onnxruntime", "qwen-tts",
                        "normalizer-tr", "trnorm", "faster-whisper", "ctranslate2")) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def loadavg() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor() or "?"
