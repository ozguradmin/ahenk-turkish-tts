#!/usr/bin/env python3
"""Stage-1 corpus of REAL human Turkish speech for the two-stage VoxCPM2 LoRA, 16 kHz: a drop-in alternative to
prepare_alania.py (same layout, so train_lora.py --data takes either).

Sources (both allow commercial use; nothing else may be added):
  cv     Common Voice 17.0 Turkish (CC0-1.0) from the ungated HF mirror fixie-ai/common_voice_17_0 (the one
         stt/prepare/commonvoice_hf.py uses), pinned revision, tr/<--cv-split>: validated (default; 80 parquet files,
         2.6 GB, 114,056 clips) or train (8 files, 35,147). Audio: 48 kHz MP3 bytes inside the parquet; metadata:
         client_id, sentence (cased, punctuated), up_votes / down_votes, gender (often empty). Rows are sorted by
         client_id inside a file; files are walked in prepare_alania's spread order.
  issai  ISSAI Turkish Speech Corpus (issai/Turkish_Speech_Corpus, MIT): one 21.4 GB tarball, streamed over HTTP or
         read from --issai-archive; Train/ only. Lower-case, unpunctuated transcripts with numbers spelled out; split
         suffixes ("erdoğan ın") rejoined with stt/prepare/common.rejoin_split_suffixes. No speaker ids: no per-speaker
         cap, never a ref_audio. Optional (--sources cv,issai), walked after all CV files, 1,000 pairs per unit.

    python tts/voxcpm/prepare_human.py --out $TTS_DATA/human16k --hours 25 \
        --exclude-manifest $TTS_DATA/antalia16k/voxcpm_val.jsonl

Per unit (a CV parquet file, or 1,000 ISSAI pairs) in a fixed order, each drop counted with its reason:
 1. Text: trnorm.normalize -> the spoken form the model is trained on (CV keeps case and punctuation). Dropped:
    text_error, text_empty, text_digits (digits left), text_charset (letters outside the Turkish alphabet + q w x â î û,
    or other symbols), contamination (prepare_alania's guard: TR-Zor + every --exclude-manifest text, whole or a
    sentence of >= 4 words inside the row).
 2. Metadata: votes (CV: up_votes > down_votes, when the columns exist), gender (--gender; unknown fails unless any).
 3. Before any audio work, against the clips selected so far: duplicate_id, duplicate_text (normalized text selected
    before, or 3 readings of it already candidates in this unit), speaker_cap (--max-per-speaker selected, or 3x the
    speaker's free slots already candidates in this unit).
 4. Audio: decode -> mono 16 kHz -> clipped (> 0.1 % of samples at full scale) -> leading/trailing silence trimmed
    (energy, 20 ms frames, 35 dB under the loudest one, 150 ms kept) -> too_short / too_long (--min-dur / --max-dur,
    after the trim) -> speech_rate (letters/s outside 5-25: text and audio disagree) -> too_quiet (> 30 dB of gain
    needed) -> active speech normalized to -20 dBFS, peaks <= -1 dBFS -> PCM16.
 5. UTMOS22 strong (score_utmos.load_predictor) of every candidate: on cuda when there is one (length-bucketed
    batches, <= 1 % zero padding), else on the CPU one clip at a time. Every score goes to scores.jsonl, which is also
    a cache: a resumed or --restart run (e.g. another --min-utmos) scores nothing twice.
 6. Selection inside the unit, best UTMOS first: utmos (< --min-utmos), duplicate_text, speaker_cap, target_reached.
    Selection is sequential in unit order and units are committed in that order: a resumed run selects what an
    uninterrupted one would (bar float noise between devices).
 7. Resume, --restart, --out ownership (marker .prepare_human) and outputs as prepare_alania.py: wav/<id>.wav,
    clips.jsonl, voxcpm_train.jsonl (audio, text = spoken form, duration; --ref-fraction of the rows get a ref_audio:
    another 3-10 s clip of the same speaker), ATTRIBUTION.txt, prep_report.json, PREP_REPORT.md. The report has the
    UTMOS percentiles of all candidates and of the kept clips, and what each threshold would keep of the units walked
    (and of all of CV, extrapolated), to pick --min-utmos without rescoring.

Exit code 0 when done, 1 when nothing was selected, 2 on bad arguments, a foreign --out or a state written with other
settings.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import io
import json
import os
import secrets
import sys
import tarfile
import time
import wave
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_alania as pa  # noqa: E402  (stdlib + trnorm; puts <repo>/src on sys.path)
import score_utmos as su  # noqa: E402  (no torch at import)
from prepare_alania import (  # noqa: E402
    Guard,
    assign_refs,
    build_guard,
    collapse_ws,
    eval_key,
    manifest_rows,
    protected_texts,
    quantiles,
    read_jsonl,
    resume_point,
    spread_order,
    wav_name,
    write_atomic,
)
from trnorm import normalize  # noqa: E402

REPO = pa.REPO
CV_REPO = "fixie-ai/common_voice_17_0"
CV_REVISION = "34f78a43893414e7b6e271ba94c1d5e05f18b239"  # 2025-01-17, still the head on 2026-10-10
CV_ROWS = {"validated": 114056, "train": 35147}  # clips per split (commonvoice_hf.py), for the extrapolation
CV_COLUMNS = ["client_id", "path", "sentence", "up_votes", "down_votes", "gender", "age", "accent"]  # + audio
ISSAI_REPO, ISSAI_FILE = "issai/Turkish_Speech_Corpus", "ISSAI_TSC_218.tar.gz"
ISSAI_REVISION = "d79ea01392882322e8f46657cb635236af25b6d4"  # 2025-02-13 (issai_tsc.py)
ISSAI_SPLIT = "train"  # archive directory Train/ (Dev/ and Test/ are left alone)
ISSAI_CHUNK, ISSAI_MAX_CHUNKS = 1000, 400  # pairs per unit; ~186k utterances in all
SOURCES = ("cv", "issai")
ATTRIBUTION = {
    "cv": ("Common Voice Turkish (Mozilla Common Voice 17.0, https://commonvoice.mozilla.org/tr), via the Hugging Face "
           "mirror https://huggingface.co/datasets/fixie-ai/common_voice_17_0; released under CC0 1.0 (public domain "
           "dedication). Thanks to the Common Voice contributors; do not attempt to identify the speakers."),
    "issai": ("ISSAI Turkish Speech Corpus (https://huggingface.co/datasets/issai/Turkish_Speech_Corpus), MIT License, "
              "Institute of Smart Systems and Artificial Intelligence (ISSAI), Nazarbayev University. Citation: "
              "S. Mussakhojayeva, K. Dauletbek, R. Yeshpanov, H. A. Varol, \"Multilingual Speech Recognition for "
              "Turkic Languages\", Information 14(2):74, 2023, https://www.mdpi.com/2078-2489/14/2/74"),
}
LICENSE = {"cv": "CC0-1.0", "issai": "MIT"}
RATE = pa.RATE
MARKER = ".prepare_human"
OWN_FILES = (*pa.OWN_FILES, "scores.jsonl")
STATE_VERSION = 1
PRESCORE = 3  # candidates per free speaker slot / per text inside one unit (a unit's speaker runs are contiguous)
THRESHOLDS = (2.5, 2.75, 3.0, 3.25, 3.3, 3.5, 3.75, 4.0)  # report: what each --min-utmos would keep
PREFETCH = 3  # units loaded ahead

# audio (part of the score-cache key: change one and every clip is scored again)
HOP = RATE // 100  # 10 ms blocks; a frame is two of them
TRIM_DB, TRIM_PAD_S = 35.0, 0.15  # trim: frames this far under the loudest one are silence; kept around the speech
ACTIVE_DB, FLOOR_DB = 30.0, -60.0  # active speech for the level: within 30 dB of the loudest frame and above -60 dBFS
TARGET_DBFS, PEAK_DBFS, MAX_GAIN_DB = -20.0, -1.0, 30.0
CLIP_LEVEL, CLIP_FRACTION = 0.999, 0.001
LETTERS_PER_S = (5.0, 25.0)
PAD_TOL, BATCH_MAX = 0.01, 32  # UTMOS batches (cuda): <= 1 % zero padding (measured: |change| <= 0.007), <= 32 clips
UTMOS_ID = f"{su.HUB_REPO}/{su.HUB_MODEL}"

TR_LETTERS = frozenset("abcçdefgğhıijklmnoöprsştuüvyzqwxâîû" "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWXÂÎÛ")
TEXT_MARKS = frozenset(" .,!?;:'\"-()…’‘“”«»–—")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- pure helpers (no numpy / pyarrow)
def spoken_text(raw: str | None) -> tuple[str, str]:
    """(trnorm spoken form, "" or the reason it is unusable)."""
    try:
        s = collapse_ws(normalize(raw or ""))
    except Exception:  # noqa: BLE001  (one odd row must not stop a paid run)
        return "", "text_error"
    if not any(c.isalpha() for c in s):
        return s, "text_empty"
    if any(c.isdigit() for c in s):
        return s, "text_digits"
    if any(c not in TR_LETTERS and c not in TEXT_MARKS for c in s):
        return s, "text_charset"
    return s, ""


def norm_gender(g: str | None) -> str | None:
    """Common Voice labels (male_masculine, female_feminine, ...) -> male / female / None."""
    g = (g or "").lower()
    return "male" if g.startswith("male") else "female" if g.startswith("female") else None


def unit_kind(unit: str) -> str:
    return "issai" if unit.startswith("issai/") else "cv"


def unit_label(unit: str) -> str:
    """Short name for logs: <chunk range>/<file stem> of a CV file (stems repeat across chunk ranges)."""
    if unit_kind(unit) == "issai":
        return unit
    p = PurePosixPath(unit)
    return f"{p.parent.name.removeprefix('chunk-range-')}/{p.stem}"


def issai_unit(k: int) -> str:
    return f"issai/{k:05d}"


def issai_index(unit: str) -> int:
    return int(unit.split("/", 1)[1])


def parse_sources(s: str) -> tuple[str, ...]:
    """"cv,issai" -> ("cv", "issai") in SOURCES order; ValueError on an unknown one or none."""
    got = {x.strip().lower() for x in s.split(",") if x.strip()}
    if not got or got - set(SOURCES):
        raise ValueError(f"--sources {s!r}: choose from {','.join(SOURCES)}")
    return tuple(x for x in SOURCES if x in got)


def foreign_files(out: Path) -> list[str]:
    """prepare_alania.foreign_files for this script: [] when `out` holds MARKER or a state.json of kind "human", or
    none of OWN_FILES and no wav/*.wav (an Alania or Antalia corpus is foreign)."""
    if (out / MARKER).is_file():
        return []
    try:
        if json.loads((out / "state.json").read_text(encoding="utf-8")).get("kind") == "human":
            return []
    except (OSError, ValueError, AttributeError):
        pass
    return [n for n in OWN_FILES if (out / n).exists()] + ["wav/*.wav"] * any((out / "wav").glob("*.wav"))


def attribution_lines(clips: list[dict]) -> list[str]:
    used = {c.get("source") for c in clips}
    return [ATTRIBUTION[s] for s in SOURCES if s in used]


def percentiles(values: list[float]) -> dict:
    v = sorted(values)
    if not v:
        return {}

    def q(p: float) -> float:
        return round(v[min(len(v) - 1, int(p * len(v)))], 3)

    return {"n": len(v), "mean": round(sum(v) / len(v), 3), "min": round(v[0], 3), "p5": q(0.05), "p10": q(0.10),
            "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p90": q(0.90), "p95": q(0.95), "max": round(v[-1], 3)}


def plan_batches(lengths: list[int], tol: float = PAD_TOL, max_samples: int = 120 * RATE,
                 max_n: int = BATCH_MAX) -> list[list[int]]:
    """Indices of `lengths` in batches, longest first: every clip of a batch is >= (1 - tol) x its longest, so zero
    padding is <= tol of any clip; a batch holds <= max_n clips and <= max_samples padded samples (one clip always)."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    out: list[list[int]] = []
    for i in order:
        b = out[-1] if out else None
        if (b and lengths[i] >= (1 - tol) * lengths[b[0]] and len(b) < max_n
                and (len(b) + 1) * lengths[b[0]] <= max_samples):
            b.append(i)
        else:
            out.append([i])
    return out


@dataclass(frozen=True)
class Filters:
    min_utmos: float = 3.3
    gender: str = "any"
    min_dur: float = 2.0
    max_dur: float = 15.0
    max_per_speaker: int = 60  # <= 0: no cap


@dataclass
class UnitData:
    rows: list[dict]  # id, source, speaker, gender, text, audio (bytes); CV also up, down, age, accent
    nbytes: int = 0
    eof: bool = False  # ISSAI: the archive ended in (or before) this unit


@dataclass
class Take:
    picked: list[dict]
    truncated: bool
    drops: collections.Counter = field(default_factory=collections.Counter)
    drop_s: collections.Counter = field(default_factory=collections.Counter)


class Selector:
    """Selection state in unit order: selected ids and texts, clips per speaker, seconds, the hours target."""

    def __init__(self, f: Filters, guard: Guard, target_s: float):
        self.f, self.guard, self.target_s = f, guard, target_s
        self.seen_text: set[str] = set()
        self.seen_id: set[str] = set()
        self.per_speaker: collections.Counter = collections.Counter()
        self.total_s = 0.0

    def done(self) -> bool:
        return self.total_s >= self.target_s

    def _add(self, clip_id: str, speaker: str | None, key: str, seconds: float) -> None:
        self.seen_id.add(clip_id)
        if key:
            self.seen_text.add(key)
        if speaker:
            self.per_speaker[speaker] += 1
        self.total_s += seconds

    def restore(self, clips: Iterable[dict]) -> None:
        for c in clips:
            self._add(c["id"], c.get("speaker"), eval_key(c["text_spoken"]), c["duration"])

    def _cap(self, speaker: str | None, extra: int = 0) -> bool:
        cap = self.f.max_per_speaker
        return bool(speaker) and cap > 0 and self.per_speaker[speaker] + extra >= cap

    def prefilter(self, rows: list[dict]) -> tuple[list[dict], collections.Counter]:
        """Steps 1-3 of the docstring. Candidates get "text_spoken", "key" and "letters"."""
        drops, cands = collections.Counter(), []
        n_key, n_spk, ids = collections.Counter(), collections.Counter(), set()
        for r in rows:
            spoken, why = spoken_text(r.get("text"))
            key = eval_key(spoken) if not why else ""
            spk = r.get("speaker")
            if why:
                pass
            elif self.guard.hit(eval_key(r.get("text") or ""), key):
                why = "contamination"
            elif r.get("up") is not None and r.get("down") is not None and not r["up"] > r["down"]:
                why = "votes"
            elif self.f.gender != "any" and r.get("gender") != self.f.gender:
                why = "gender"
            elif r["id"] in self.seen_id or r["id"] in ids:
                why = "duplicate_id"
            elif key in self.seen_text or n_key[key] >= PRESCORE:
                why = "duplicate_text"
            elif self._cap(spk) or (spk and self.f.max_per_speaker > 0
                                    and n_spk[spk] >= PRESCORE * (self.f.max_per_speaker - self.per_speaker[spk])):
                why = "speaker_cap"
            if why:
                drops[why] += 1
                continue
            ids.add(r["id"])
            n_key[key] += 1
            if spk:
                n_spk[spk] += 1
            cands.append({**r, "text_spoken": spoken, "key": key, "letters": sum(c.isalpha() for c in spoken)})
        return cands, drops

    def take(self, scored: list[dict]) -> Take:
        """Step 6: scored rows have "utmos" and "duration" (s, after the trim)."""
        t = Take([], False)

        def drop(why: str, r: dict) -> None:
            t.drops[why] += 1
            t.drop_s[why] += r["duration"]

        ok = []
        for r in scored:
            if r["utmos"] < self.f.min_utmos:
                drop("utmos", r)
            else:
                ok.append(r)
        ok.sort(key=lambda r: (-r["utmos"], r["id"]))
        for i, r in enumerate(ok):
            if self.done():
                t.truncated = True
                for rest in ok[i:]:
                    drop("target_reached", rest)
                break
            if r["key"] in self.seen_text:
                drop("duplicate_text", r)
            elif self._cap(r.get("speaker")):
                drop("speaker_cap", r)
            else:
                t.picked.append(r)
                self._add(r["id"], r.get("speaker"), r["key"], r["duration"])
        return t


def clip_record(r: dict, unit: str) -> dict:
    c = {"id": r["id"], "source": r["source"], "speaker": r.get("speaker"), "voice_id": r.get("speaker") or r["id"],
         "gender": r.get("gender"), "text": collapse_ws(r.get("text")), "text_spoken": r["text_spoken"],
         "duration": round(r["duration"], 3), "utmos": round(r["utmos"], 4), "license": LICENSE[r["source"]],
         "unit": unit, "wav": wav_name(r["id"])}
    c.update({k: r[k] for k in ("age", "accent") if r.get(k)})
    return c  # voice_id: assign_refs' speaker key; an ISSAI clip is its own "voice", so it never gets a ref


def score_line(r: dict, unit: str, ak: str, run: str) -> dict:
    return {"id": r["id"], "unit": unit, "src": r["source"], "spk": r.get("speaker"), "key": r["key"],
            "dur": round(r["duration"], 3), "utmos": round(r["utmos"], 4), "ak": ak, "run": run}


def simulate(units: list[list[dict]], threshold: float, cap: int) -> dict:
    """What step 6 keeps of these scored candidates (score lines, per unit in walk order) at another --min-utmos, with
    no hours target. Approximate: the candidates themselves came from the real run's pre-score state."""
    seen, per, secs, n, spk = set(), collections.Counter(), collections.Counter(), 0, set()
    for lines in units:
        for x in sorted(lines, key=lambda x: (-x["utmos"], x["id"])):
            if x["utmos"] < threshold or x["key"] in seen or (x["spk"] and 0 < cap <= per[x["spk"]]):
                continue
            seen.add(x["key"])
            if x["spk"]:
                per[x["spk"]] += 1
                spk.add(x["spk"])
            secs[x["src"]] += x["dur"]
            n += 1
    return {"clips": n, "hours": round(sum(secs.values()) / 3600, 3), "speakers": len(spk),
            **{f"{s}_hours": round(secs[s] / 3600, 3) for s in SOURCES}}


def audio_key() -> str:
    """Digest of everything a clip's processed audio and score depend on (the score cache's validity)."""
    parts = [RATE, HOP, TRIM_DB, TRIM_PAD_S, ACTIVE_DB, FLOOR_DB, TARGET_DBFS, PEAK_DBFS, MAX_GAIN_DB, CLIP_LEVEL,
             CLIP_FRACTION, UTMOS_ID]
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:10]


def settings_key(f: Filters, args: argparse.Namespace, guard: Guard) -> str:
    """Everything that changes which clips a unit yields; not --hours / --max-files / --max-issai-chunks / --workers /
    --seed / --device / --batch-seconds / --issai-archive."""
    return json.dumps({"filters": asdict(f), "sources": list(args.sources), "cv_split": args.cv_split,
                       "cv_revision": args.cv_revision, "row_groups": args.row_groups, "issai": [ISSAI_REVISION,
                       ISSAI_SPLIT, ISSAI_CHUNK], "letters_per_s": LETTERS_PER_S, "audio": audio_key(),
                       "guard": guard.digest()}, sort_keys=True)


def unit_order(state: dict, args: argparse.Namespace) -> list[str]:
    order = []
    if "cv" in args.sources:
        files = state["files"]
        order = [files[i] for i in spread_order(len(files))]
        if args.max_files > 0:
            order = order[:args.max_files]
    if "issai" in args.sources:
        n = state.get("issai_end", ISSAI_MAX_CHUNKS)
        order += [issai_unit(k) for k in range(min(n, args.max_issai_chunks) if args.max_issai_chunks > 0 else n)]
    return order


def load_cache(path: Path, ak: str) -> dict[str, float]:
    return {x["id"]: x["utmos"] for x in read_jsonl(path) if x.get("ak") == ak and isinstance(x.get("utmos"), float)}


def retry(fn: Callable, *a, what: str = "", tries: int = 4):
    for i in range(tries):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001  (network: retried, then raised)
            if i == tries - 1:
                raise
            wait = 10 * 3 ** i
            log(f"[human] {what}: {type(e).__name__}: {e}; retry in {wait} s")
            time.sleep(wait)


def write_pcm(path: Path, pcm: bytes, rate: int = RATE) -> None:
    """PCM16 mono bytes -> WAV, atomically."""
    tmp = path.with_name(path.name + ".tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- audio (numpy) and UTMOS (torch)
def speech_bounds(x) -> tuple[int, int, float | None]:
    """(start, end) of x with leading/trailing silence trimmed (TRIM_PAD_S kept) and the active speech level in dBFS
    (None: nothing above FLOOR_DB)."""
    import numpy as np

    n = len(x) // HOP
    if n < 2:
        return 0, len(x), None
    p = (x[:n * HOP].astype(np.float64).reshape(n, HOP) ** 2).mean(axis=1)
    fp = (p[:-1] + p[1:]) / 2  # 20 ms frames, 10 ms hop
    db = 10 * np.log10(fp + 1e-12)
    top = float(db.max())
    if top <= FLOOR_DB:
        return 0, len(x), None
    idx = np.flatnonzero(db > max(top - TRIM_DB, FLOOR_DB))
    pad = int(TRIM_PAD_S * RATE)
    a, b = max(0, int(idx[0]) * HOP - pad), min(len(x), (int(idx[-1]) + 2) * HOP + pad)
    active = fp[db > max(top - ACTIVE_DB, FLOOR_DB)]
    return a, b, float(10 * np.log10(active.mean()))


def level_normalize(y, level_db: float):
    """Active speech to TARGET_DBFS, then peaks down to PEAK_DBFS; (audio, gain dB) or (None, gain) when the gain
    would exceed MAX_GAIN_DB."""
    import numpy as np

    gain = TARGET_DBFS - level_db
    if gain > MAX_GAIN_DB:
        return None, gain
    y = y * np.float32(10 ** (gain / 20))
    peak, lim = float(np.abs(y).max()) if y.size else 0.0, 10 ** (PEAK_DBFS / 20)
    if peak > lim:
        y = y * np.float32(lim / peak)
    return y.astype(np.float32), gain


def process_audio(data: bytes, letters: int, f: Filters) -> tuple[bytes | None, float, str]:
    """Step 4: encoded audio -> (PCM16 bytes at RATE, seconds after the trim, "") or (None, seconds, reason)."""
    import numpy as np

    try:
        x = pa.decode_16k(data)
    except Exception:  # noqa: BLE001  (a broken file costs its own row)
        return None, 0.0, "decode_error"
    if x.size and float(np.mean(np.abs(x) >= CLIP_LEVEL)) > CLIP_FRACTION:
        return None, x.size / RATE, "clipped"
    a, b, level = speech_bounds(x)
    secs = (b - a) / RATE
    if level is None:
        return None, secs, "too_quiet"
    if secs < f.min_dur:
        return None, secs, "too_short"
    if secs > f.max_dur:
        return None, secs, "too_long"
    if not LETTERS_PER_S[0] <= letters / secs <= LETTERS_PER_S[1]:
        return None, secs, "speech_rate"
    y, _ = level_normalize(x[a:b], level)
    if y is None:
        return None, secs, "too_quiet"
    return np.round(np.clip(y, -1.0, 1.0) * 32767).astype("<i2").tobytes(), secs, ""


class Utmos:
    """UTMOS22 strong on PCM16 clips at RATE; the model is loaded at the first call. cuda: plan_batches batches,
    cpu: one clip at a time (exactly score_utmos.score_wav). A failing batch is retried clip by clip; a failing clip
    scores None."""

    def __init__(self, device: str = "auto", batch_seconds: float = 120.0):
        self.device, self.batch_seconds, self.model = device, batch_seconds, None
        self.seconds = 0.0
        self.clips = 0

    def _load(self) -> None:
        import torch

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        self.model = su.load_predictor(self.device)
        log(f"[human] UTMOS {UTMOS_ID} on {self.device}, loaded in {time.time() - t0:.1f} s")

    def _forward(self, waves: list) -> list[float]:
        import numpy as np
        import torch

        n = max(len(w) for w in waves)
        batch = np.zeros((len(waves), n), dtype=np.float32)
        for i, w in enumerate(waves):
            batch[i, :len(w)] = w
        with torch.inference_mode():
            v = self.model(torch.from_numpy(batch).to(self.device), RATE).float().cpu().tolist()
        if not all(map(lambda s: s == s and abs(s) != float("inf"), v)):
            raise ValueError(f"UTMOS not finite: {v}")
        return v

    def __call__(self, rows: list[dict]) -> list[float | None]:
        import numpy as np

        if self.model is None:
            self._load()
        t0 = time.time()
        waves = [np.frombuffer(r["pcm"], dtype="<i2").astype(np.float32) / 32768 for r in rows]
        out: list[float | None] = [None] * len(rows)
        groups = (plan_batches([len(w) for w in waves], max_samples=int(self.batch_seconds * RATE))
                  if self.device == "cuda" else [[i] for i in range(len(rows))])
        for g in groups:
            try:
                for i, s in zip(g, self._forward([waves[i] for i in g])):
                    out[i] = s
                continue
            except Exception as e:  # noqa: BLE001  (OOM, a too short clip: one by one below)
                if len(g) == 1:
                    log(f"[human] {rows[g[0]]['id']}: not scored: {type(e).__name__}: {e}")
                    continue
                if self.device == "cuda":
                    import torch

                    torch.cuda.empty_cache()
            for i in g:
                try:
                    out[i] = self._forward([waves[i]])[0]
                except Exception as e:  # noqa: BLE001
                    log(f"[human] {rows[i]['id']}: not scored: {type(e).__name__}: {e}")
        self.seconds += time.time() - t0
        self.clips += len(rows)
        return out


# --------------------------------------------------------------------------- sources
class CvSource:
    """Common Voice parquet files on the Hub, read with HfFileSystem (pyarrow imported here)."""

    def __init__(self, revision: str = CV_REVISION, split: str = "validated", row_groups: int = 0):
        from huggingface_hub import HfFileSystem

        self.revision, self.split, self.row_groups, self.fs = revision, split, row_groups, HfFileSystem()

    def list_files(self) -> list[str]:
        from huggingface_hub import HfApi

        tree = HfApi().list_repo_tree(CV_REPO, path_in_repo=f"tr/{self.split}", repo_type="dataset",
                                      revision=self.revision, recursive=True)
        return sorted(f.path for f in tree if f.path.endswith(".parquet"))

    def load(self, name: str) -> UnitData:
        import pyarrow.parquet as pq

        with self.fs.open(f"datasets/{CV_REPO}@{self.revision}/{name}", cache_type="none") as f:
            pf = pq.ParquetFile(f, pre_buffer=True)
            md, names = pf.metadata, set(pf.schema_arrow.names)
            n = md.num_row_groups if self.row_groups <= 0 else min(self.row_groups, md.num_row_groups)
            cols = [c for c in CV_COLUMNS if c in names] + ["audio"]
            table = pf.read_row_groups(list(range(n)), columns=cols)
        rows = [cv_row(r) for r in table.to_pylist()]
        return UnitData(rows, pa._chunk_bytes(md, range(n), cols) + md.serialized_size)


def cv_row(r: dict) -> dict:
    audio = r.get("audio") or {}
    stem = PurePosixPath((r.get("path") or audio.get("path") or "x").replace("\\", "/")).stem
    cid = r.get("client_id") or ""
    return {"id": stem, "source": "cv", "speaker": f"cv_{cid[:16]}" if cid else None,
            "gender": norm_gender(r.get("gender")), "text": r.get("sentence"), "up": r.get("up_votes"),
            "down": r.get("down_votes"), "age": r.get("age") or None, "accent": r.get("accent") or None,
            "audio": audio.get("bytes")}


def stt_common():
    """stt/prepare/common.py (rejoin_split_suffixes, http_stream, hf_resolve_url) without its OMP_NUM_THREADS=2 side
    effect on this process."""
    if mod := sys.modules.get("stt_prepare_common"):
        return mod
    old = os.environ.get("OMP_NUM_THREADS")
    spec = importlib.util.spec_from_file_location("stt_prepare_common", REPO / "stt" / "prepare" / "common.py")
    mod = sys.modules["stt_prepare_common"] = importlib.util.module_from_spec(spec)  # dataclasses look it up
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules["stt_prepare_common"]
        raise
    finally:
        if old is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old
    return mod


class _Counting(io.RawIOBase):
    def __init__(self, f):
        self.f, self.n = f, 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self.f.read(len(b))
        b[:len(data)] = data
        self.n += len(data)
        return len(data)

    def close(self) -> None:
        self.f.close()
        super().close()


class IssaiSource:
    """The ISSAI tarball read as a stream (HTTP, or --issai-archive); a unit is ISSAI_CHUNK consecutive .wav/.txt pairs
    of Train/ in the order they complete in the archive. A .wav and its .txt can lie far apart (the archive is in
    directory order), so a .wav waiting for its .txt is spooled to a temp dir (spool_dir; default the system temp),
    not kept in memory. Units must be loaded in order by one thread; an earlier unit reopens the stream."""

    def __init__(self, archive: Path | None = None, chunk: int = ISSAI_CHUNK, opener: Callable | None = None,
                 spool_dir: Path | None = None):
        self.archive, self.chunk, self.opener, self.spool_dir = archive, chunk, opener, spool_dir
        self._tf = self._raw = self._it = self._spool = None
        self.pos, self.eof = 0, False
        self._rejoin = stt_common().rejoin_split_suffixes

    def _open(self) -> None:
        import tempfile

        self.close()
        if self.opener:
            raw = self.opener()
        elif self.archive:
            raw = open(self.archive, "rb")
        else:
            c = stt_common()
            raw = c.http_stream(c.hf_resolve_url(ISSAI_REPO, ISSAI_FILE, ISSAI_REVISION))
        self._raw = _Counting(raw)
        self._tf = tarfile.open(fileobj=io.BufferedReader(self._raw, 1 << 20), mode="r|gz")
        self._spool = Path(tempfile.mkdtemp(prefix="issai_spool_", dir=self.spool_dir))
        self._it, self.pos, self.eof, self._txt, self._wav = iter(self._tf), 0, False, {}, {}

    def close(self) -> None:
        import shutil

        if self._tf is not None:
            self._tf.close()
            self._raw.close()
            self._tf = self._raw = self._it = None
        if self._spool is not None:
            shutil.rmtree(self._spool, ignore_errors=True)
            self._spool = None

    def _next_pair(self) -> dict | None:
        for m in self._it:
            parts = PurePosixPath(m.name).parts
            if not m.isfile() or len(parts) < 2 or not parts[-2].lower().startswith(ISSAI_SPLIT):
                continue
            uid, ext = os.path.splitext(parts[-1])
            key = (parts[-2].lower(), uid)
            if ext == ".txt":
                self._txt[key] = self._tf.extractfile(m).read().decode("utf-8", "replace").strip()
                if key not in self._wav:
                    continue
                spooled = self._spool / self._wav.pop(key)
                audio = spooled.read_bytes()
                spooled.unlink()
            elif ext == ".wav":
                audio = self._tf.extractfile(m).read()
                if key not in self._txt:
                    self._wav[key] = name = wav_name(f"{key[0]}_{uid}")
                    (self._spool / name).write_bytes(audio)
                    continue
            else:
                continue
            return {"id": f"issai_{key[0]}_{uid}", "source": "issai", "speaker": None, "gender": None,
                    "text": self._rejoin(self._txt.pop(key)), "audio": audio}
        return None

    def load(self, unit: str) -> UnitData:
        start = issai_index(unit) * self.chunk
        reopen = self._tf is None or self.pos > start
        if reopen:
            self._open()
        n0, rows = 0 if reopen else self._raw.n, []
        while len(rows) < self.chunk and not self.eof:
            pair = self._next_pair()
            if pair is None:
                self.eof = True
                break
            if self.pos >= start:
                rows.append(pair)
            self.pos += 1
        return UnitData(rows, self._raw.n - n0, self.eof)


# --------------------------------------------------------------------------- report
def build_report(clips: list[dict], done: list[dict], refs: dict[str, str], settings: dict, guard: Guard,
                 score_units: list[list[dict]], f: Filters, extra: dict | None = None) -> dict:
    hours = lambda cs: round(sum(c["duration"] for c in cs) / 3600, 3)  # noqa: E731

    def by(key: str) -> dict:
        g = collections.defaultdict(list)
        for c in clips:
            g[c.get(key) or "?"].append(c)
        return {k: {"clips": len(v), "hours": hours(v)} for k, v in sorted(g.items(), key=lambda kv: -len(kv[1]))}

    drops, drop_s = collections.Counter(), collections.Counter()
    for e in done:
        drops.update(e.get("drops", {}))
        drop_s.update(e.get("drop_s", {}))
    per_spk = collections.Counter(c["speaker"] for c in clips if c.get("speaker"))
    cand = [x for u in score_units for x in u]
    cv_rows = sum(e.get("rows", 0) for e in done if unit_kind(e["file"]) == "cv")
    cv_total = CV_ROWS.get(settings.get("cv_split"), 0)
    table = []
    for t in sorted({*THRESHOLDS, f.min_utmos}):
        s = simulate(score_units, t, f.max_per_speaker)
        if cv_rows and cv_total:
            s["cv_hours_all_files"] = round(s["cv_hours"] * cv_total / cv_rows, 1)
        table.append({"min_utmos": t, **s})
    used = [s for s in SOURCES if s in settings.get("sources", ())]
    return {
        "dataset": {"sources": {s: {"repo": CV_REPO if s == "cv" else ISSAI_REPO, "license": LICENSE[s],
                                    "revision": settings.get("cv_revision") if s == "cv" else ISSAI_REVISION}
                                for s in used},
                    "revision": settings.get("cv_revision"), "attribution": attribution_lines(clips)},
        "settings": settings,
        "target_hours": settings.get("hours"), "reached": sum(c["duration"] for c in clips) >= settings["hours"] * 3600,
        "selected": {"clips": len(clips), "hours": hours(clips), "speakers": len(per_spk),
                     "clips_without_speaker": sum(not c.get("speaker") for c in clips),
                     "clips_per_speaker": quantiles(list(per_spk.values())),
                     "duration_s": quantiles([c["duration"] for c in clips]),
                     "by_source": by("source"), "by_gender": by("gender"),
                     "examples": [c["text_spoken"][:160] for c in clips[:3] + clips[-3:]] if clips else []},
        "utmos": {"model": UTMOS_ID, "min_utmos": f.min_utmos, "candidates": percentiles([x["utmos"] for x in cand]),
                  "kept": percentiles([c["utmos"] for c in clips]),
                  "candidate_hours": round(sum(x["dur"] for x in cand) / 3600, 3),
                  "thresholds": table,
                  "note": "thresholds: what step 6 keeps of the scored candidates of the units walked (no hours "
                          "target; candidates came from this run's pre-score speaker/text state); cv_hours_all_files: "
                          f"cv_hours x {cv_total} / {cv_rows} CV rows walked"},
        "seen": {"units": len(done), "rows": sum(e.get("rows", 0) for e in done), "cv_rows": cv_rows,
                 "candidates": sum(e.get("candidates", 0) for e in done),
                 "bytes_read": sum(e.get("bytes", 0) for e in done), "unit_list": [e["file"] for e in done]},
        "drops": {k: {"rows": n, "hours": round(drop_s.get(k, 0.0) / 3600, 3)} for k, n in drops.most_common()},
        "contamination_keys": {"exact": len(guard.exact), "contained": len(guard.contain), "digest": guard.digest()},
        "manifest": {"rows": len(clips), "with_ref_audio": len(refs)},
        **(extra or {}),
    }


def render_markdown(r: dict) -> str:
    L = []
    a = L.append
    s, sel, u = r["settings"], r["selected"], r["utmos"]
    a("# Human-speech stage-1 corpus report (generated by `tts/voxcpm/prepare_human.py`)\n")
    for name, d in r["dataset"]["sources"].items():
        a(f"- {name}: `{d['repo']}` @ `{d['revision']}` ({d['license']})")
    a("")
    for x in r["dataset"]["attribution"]:
        a(f"Attribution: {x}\n")
    a(f"Settings: sources {','.join(s['sources'])}, CV split {s['cv_split']}, UTMOS >= {s['min_utmos']}, gender "
      f"{s['gender']}, {s['min_dur']}-{s['max_dur']} s after trimming, <= {s['max_per_speaker']} clips/speaker, ref "
      f"fraction {s['ref_fraction']}, row groups/file {s['row_groups'] or 'all'}, max CV files "
      f"{s['max_files'] or 'all'}, seed {s['seed']}; protected: TR-Zor + "
      f"{', '.join(s['exclude_manifest']) or 'no --exclude-manifest'}.\n")
    status = "reached" if r["reached"] else "**NOT reached** (units used up)"
    a(f"**{sel['hours']} h, {sel['clips']} clips, {sel['speakers']} speakers** (target {r['target_hours']} h: "
      f"{status}); {r['manifest']['with_ref_audio']} manifest rows with ref_audio; "
      f"{sel['clips_without_speaker']} clips without a speaker id.\n")
    sn = r["seen"]
    a(f"Walked {sn['units']} units, {sn['rows']} rows, {sn['candidates']} scored candidates "
      f"({u['candidate_hours']} h), {sn['bytes_read'] / 1e9:.2f} GB read.\n")
    a(f"Durations (s): {sel['duration_s']}  \nclips per speaker: {sel['clips_per_speaker']}\n")
    a(f"## UTMOS ({u['model']})\n\n| | n | mean | p5 | p10 | p25 | median | p75 | p90 | p95 |\n"
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("candidates", "kept"):
        q = u[name]
        if q:
            a(f"| {name} | {q['n']} | " + " | ".join(str(q[k]) for k in ("mean", "p5", "p10", "p25", "median", "p75",
                                                                         "p90", "p95")) + " |")
    a(f"\nWhat each --min-utmos keeps of the units walked ({u['note']}):\n")
    a("| min_utmos | clips | hours | speakers | CV h, all files (extrapolated) |\n|---:|---:|---:|---:|---:|")
    for t in u["thresholds"]:
        a(f"| {t['min_utmos']}{' (used)' if t['min_utmos'] == u['min_utmos'] else ''} | {t['clips']} | {t['hours']} | "
          f"{t['speakers']} | {t.get('cv_hours_all_files', '-')} |")
    for title, key in (("Source", "by_source"), ("Gender", "by_gender")):
        a(f"\n## {title}\n\n| {title.lower()} | clips | hours |\n|---|---:|---:|")
        for k, v in sel[key].items():
            a(f"| {k} | {v['clips']} | {v['hours']} |")
    a("\n## Drops (first failing filter per row; hours known only after decoding)\n\n| reason | rows | hours |\n"
      "|---|---:|---:|")
    for k, v in r["drops"].items():
        a(f"| {k} | {v['rows']} | {v['hours']} |")
    ck = r["contamination_keys"]
    a(f"\nContamination keys: {ck['exact']} sentences, {ck['contained']} also searched inside rows "
      f"(digest {ck['digest']}).\n")
    a("## Examples (spoken form)\n")
    for t in sel["examples"]:
        a(f"- {t}")
    a(f"\nUnits: {' '.join(map(unit_label, sn['unit_list']))}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- the walk
def run_unit(unit: str, data: UnitData, sel: Selector, process: Callable, scorer: Callable, cache: dict[str, float],
             pool: ThreadPoolExecutor) -> tuple[Take, list[dict], dict]:
    """Steps 1-6 for one loaded unit: (take, scored candidates, counts)."""
    cands, pre = sel.prefilter(data.rows)
    drops, drop_s, ok = collections.Counter(pre), collections.Counter(), []
    for r, (pcm, secs, why) in zip(cands, pool.map(process, cands)):
        if why:
            drops[why] += 1
            drop_s[why] += secs
        else:
            ok.append({**r, "pcm": pcm, "duration": secs, "audio": None})
    need = [r for r in ok if r["id"] not in cache]
    for r, s in zip(need, scorer(need) if need else []):
        r["utmos"] = s
    for r in ok:
        if r["id"] in cache:
            r["utmos"] = cache[r["id"]]
    scored = []
    for r in ok:
        if r.get("utmos") is None:
            drops["utmos_error"] += 1
            drop_s["utmos_error"] += r["duration"]
        else:
            scored.append(r)
    take = sel.take(scored)
    take.drops.update(drops)
    take.drop_s.update(drop_s)
    return take, scored, {"rows": len(data.rows), "candidates": len(ok), "scored_now": len(need)}


def walk(sources: dict, todo: list[str], sel: Selector, state: dict, out: Path, process: Callable, scorer: Callable,
         cache: dict[str, float], workers: int, ak: str) -> None:
    """Units loaded ahead in parallel (ISSAI: one thread, in order); everything else in unit order."""
    wav_dir, t0 = out / "wav", time.time()
    loaders = {k: ThreadPoolExecutor(1 if k == "issai" else min(4, workers)) for k in sources}
    pool = ThreadPoolExecutor(workers)
    pending, loads = collections.deque(todo), collections.deque()

    def prefetch() -> None:
        while pending and len(loads) < PREFETCH:
            u = pending.popleft()
            src = sources[unit_kind(u)]
            loads.append((u, loaders[unit_kind(u)].submit(retry, src.load, u, what=u)))

    try:
        prefetch()
        while loads and not sel.done():
            unit, fut = loads.popleft()
            data = fut.result()
            if unit_kind(unit) == "issai":
                if issai_index(unit) >= state.get("issai_end", ISSAI_MAX_CHUNKS + 1):
                    continue  # past the end of the archive (loaded before the end was seen)
                if data.eof:
                    state["issai_end"] = issai_index(unit) + (1 if data.rows else 0)
                    pending = collections.deque(u for u in pending if unit_kind(u) != "issai")
                    if not data.rows:
                        write_atomic(out / "state.json", json.dumps(state, ensure_ascii=False, indent=1))
                        continue
            prefetch()
            take, scored, counts = run_unit(unit, data, sel, process, scorer, cache, pool)
            for r in take.picked:
                if not (wav_dir / wav_name(r["id"])).exists():
                    write_pcm(wav_dir / wav_name(r["id"]), r["pcm"])
            run = state["run"]
            with open(out / "scores.jsonl", "a", encoding="utf-8", newline="\n") as f:
                f.writelines(json.dumps(score_line(r, unit, ak, run), ensure_ascii=False) + "\n" for r in scored)
            cache.update((r["id"], r["utmos"]) for r in scored)
            clips = [clip_record(r, unit) for r in take.picked]
            with open(out / "clips.jsonl", "a", encoding="utf-8", newline="\n") as f:
                f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in clips)
            sec = sum(c["duration"] for c in clips)
            state["done"].append({"file": unit, "ids": [c["id"] for c in clips], "truncated": take.truncated,
                                  **counts, "seconds_candidates": round(sum(r["duration"] for r in scored), 3),
                                  "seconds_selected": round(sec, 3), "bytes": data.nbytes, "drops": dict(take.drops),
                                  "drop_s": {k: round(v, 3) for k, v in take.drop_s.items()}})
            write_atomic(out / "state.json", json.dumps(state, ensure_ascii=False, indent=1))
            gb = sum(e.get("bytes", 0) for e in state["done"]) / 1e9
            log(f"[human] {len(state['done'])} units, {unit_label(unit)}: "
                f"{counts['rows']} rows, {counts['candidates']} candidates ({counts['scored_now']} scored now), +"
                f"{len(clips)} clips ({sec / 3600:.2f} h); total {sel.total_s / 3600:.2f} / {sel.target_s / 3600:g} h, "
                f"{gb:.2f} GB read, {time.time() - t0:.0f} s")
    finally:
        for _, fut in loads:
            fut.cancel()
        for ex in loaders.values():
            ex.shutdown(wait=True, cancel_futures=True)
        pool.shutdown(wait=True, cancel_futures=True)


def refetch_missing(sources: dict, clips: list[dict], wav_dir: Path, process: Callable) -> int:
    """Audio again for committed clips whose WAV is gone (the selection stays as it is); returns how many."""
    absent = collections.defaultdict(dict)
    for c in clips:
        if not (wav_dir / c["wav"]).is_file():
            absent[c["unit"]][c["id"]] = c
    n = sum(map(len, absent.values()))
    if n:
        log(f"[human] {n} committed clips have no WAV ({len(absent)} units): reading their audio again")
    for unit, cs in absent.items():
        data = retry(sources[unit_kind(unit)].load, unit, what=f"{unit} (missing WAVs)")
        for r in data.rows:
            if r["id"] in cs:
                pcm, _, why = process({**r, "letters": sum(ch.isalpha() for ch in cs[r["id"]]["text_spoken"])})
                if pcm is None:
                    raise RuntimeError(f"{unit}: {r['id']} no longer passes the audio checks ({why})")
                write_pcm(wav_dir / cs[r["id"]]["wav"], pcm)
    return n


# --------------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output directory (e.g. $TTS_DATA/human16k)")
    ap.add_argument("--hours", type=float, default=25.0, help="selected audio to stop at [%(default)s]")
    ap.add_argument("--sources", default="cv", help="cv, issai or cv,issai (ISSAI after CV) [%(default)s]")
    ap.add_argument("--min-utmos", type=float, default=3.3,
                    help="minimum UTMOS22 strong (real Antalia studio clips: 3.44); see the report's threshold table "
                         "[%(default)s]")
    ap.add_argument("--gender", choices=["female", "male", "any"], default="any",
                    help="CV metadata, often empty: unknown fails unless any [%(default)s]")
    ap.add_argument("--min-dur", type=float, default=2.0, help="after silence trimming [%(default)s] s")
    ap.add_argument("--max-dur", type=float, default=15.0, help="after silence trimming [%(default)s] s")
    ap.add_argument("--max-per-speaker", type=int, default=60, help="clips per speaker; 0 = no cap [%(default)s]")
    ap.add_argument("--ref-fraction", type=float, default=0.3, help="manifest rows with a ref_audio [%(default)s]")
    ap.add_argument("--exclude-manifest", type=Path, nargs="+", action="extend", default=[],
                    help="jsonl files whose texts must never be trained on (Antalia's voxcpm_val.jsonl); TR-Zor always")
    ap.add_argument("--cv-split", choices=sorted(CV_ROWS), default="validated", help="[%(default)s]")
    ap.add_argument("--cv-revision", default=CV_REVISION, help="mirror commit [%(default)s]")
    ap.add_argument("--issai-archive", type=Path, default=None, help="a local ISSAI_TSC_218.tar.gz (default: stream)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="UTMOS device [%(default)s]")
    ap.add_argument("--batch-seconds", type=float, default=120.0,
                    help="audio per UTMOS batch on cuda (padded) [%(default)s]")
    ap.add_argument("--workers", type=int, default=8, help="parallel decoders [%(default)s]")
    ap.add_argument("--seed", type=int, default=1234, help="ref_audio choice and manifest order [%(default)s]")
    ap.add_argument("--max-files", type=int, default=0, help="walk at most N CV files of the spread order; 0 = all")
    ap.add_argument("--max-issai-chunks", type=int, default=0, help="at most N ISSAI units; 0 = all")
    ap.add_argument("--row-groups", type=int, default=0, help="first N row groups of each CV file only; 0 = all")
    ap.add_argument("--restart", action="store_true",
                    help="discard state.json and clips.jsonl (WAVs and the scores.jsonl cache are reused)")
    args = ap.parse_args(argv)
    try:
        args.sources = parse_sources(args.sources)
    except ValueError as e:
        ap.error(str(e))
    for name, ok in (("--hours", args.hours > 0), ("--ref-fraction", 0 <= args.ref_fraction <= 1),
                     ("--workers", args.workers >= 1), ("--batch-seconds", args.batch_seconds > 0),
                     ("--min-dur/--max-dur", 0 < args.min_dur <= args.max_dur)):
        if not ok:
            ap.error(f"bad {name}")
    return args


def make_sources(args: argparse.Namespace) -> dict:
    out = {}
    if "cv" in args.sources:
        out["cv"] = CvSource(args.cv_revision, args.cv_split, args.row_groups)
    if "issai" in args.sources:
        out["issai"] = IssaiSource(args.issai_archive)
    return out


def main(argv=None, sources: dict | None = None, scorer: Callable | None = None,
         process: Callable | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    out = args.out.resolve()
    missing = [str(p) for p in args.exclude_manifest if not p.is_file()]
    if missing:
        log(f"[human] --exclude-manifest not found: {missing} (the contamination guard needs them)")
        return 2
    if inside := [str(p) for p in args.exclude_manifest if out in p.resolve().parents]:
        log(f"[human] --exclude-manifest {inside} lies inside --out {out}: that is another corpus; use another --out")
        return 2
    if foreign := foreign_files(out):
        log(f"[human] --out {out} holds {', '.join(foreign)} but no {MARKER} or state.json of this script: another "
            "corpus would be overwritten and its WAVs deleted; use an empty or new directory")
        return 2
    if args.issai_archive and not args.issai_archive.is_file():
        log(f"[human] --issai-archive {args.issai_archive} not found")
        return 2
    (out / "wav").mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text("real human speech corpus: written by tts/voxcpm/prepare_human.py\n", encoding="utf-8")
    for p in [*(out / "wav").glob("*.wav.tmp"), *(out / f"{n}.part" for n in OWN_FILES)]:  # a killed run's half files
        p.unlink(missing_ok=True)
    guard = build_guard(protected_texts(pa.TRZOR) + [t for p in args.exclude_manifest for t in protected_texts(p)])
    filters = Filters(args.min_utmos, args.gender, args.min_dur, args.max_dur, args.max_per_speaker)
    key, ak = settings_key(filters, args, guard), audio_key()
    state_path, clips_path = out / "state.json", out / "clips.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() and not args.restart else None
    if state and state.get("key") != key:
        log(f"[human] {state_path} was written with other settings:\n  {state.get('key')}\nnow:\n  {key}\n"
            "use the same settings, another --out, or --restart (keeps the WAVs and the UTMOS scores)")
        return 2
    sources = sources if sources is not None else make_sources(args)
    if not state:
        state = {"version": STATE_VERSION, "kind": "human", "key": key, "run": secrets.token_hex(6),
                 "files": sources["cv"].list_files() if "cv" in sources else [], "done": []}
        clips_path.unlink(missing_ok=True)
    target_s = args.hours * 3600
    done, clips = resume_point(state, read_jsonl(clips_path), target_s)
    state["done"] = done
    write_atomic(clips_path, "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in clips))
    write_atomic(state_path, json.dumps(state, ensure_ascii=False, indent=1))
    finished = {e["file"] for e in done}
    todo = [u for u in unit_order(state, args) if u not in finished]
    sel = Selector(filters, guard, target_s)
    sel.restore(clips)
    cache = load_cache(out / "scores.jsonl", ak)
    scorer = scorer or Utmos(args.device, args.batch_seconds)
    process = process or (lambda r: process_audio(r["audio"], r["letters"], filters))
    log(f"[human] sources {','.join(args.sources)} (CV {args.cv_split}@{args.cv_revision[:10]}: "
        f"{len(state['files'])} files); resuming with {len(done)} units done ({sel.total_s / 3600:.2f} h); target "
        f"{args.hours:g} h; {len(todo)} units left; {len(cache)} cached UTMOS scores; {len(guard.exact)} protected "
        "sentences")
    try:
        refetched = refetch_missing(sources, clips, out / "wav", process)
        if not sel.done() and todo:
            walk(sources, todo, sel, state, out, process, scorer, cache, args.workers, ak)
    finally:
        for s in sources.values():
            if hasattr(s, "close"):
                s.close()
    return finalize(args, out, state, guard, filters, t0, refetched, scorer)


def finalize(args, out: Path, state: dict, guard: Guard, f: Filters, t0: float, refetched: int = 0,
             scorer=None) -> int:
    done = state["done"]
    _, clips = resume_point(state, read_jsonl(out / "clips.jsonl"), 0.0)
    wav_dir = out / "wav"
    absent = [c["wav"] for c in clips if not (wav_dir / c["wav"]).is_file()]
    if absent:
        raise RuntimeError(f"{len(absent)} selected clips have no WAV, e.g. {absent[:3]}; rerun: the next call reads "
                           "the audio of committed clips without a WAV again")
    keep = {c["wav"] for c in clips}
    orphans = [p for p in wav_dir.glob("*.wav") if p.name not in keep] if (out / MARKER).is_file() else []
    for p in orphans:
        p.unlink()
    refs = assign_refs(clips, args.ref_fraction, args.seed)
    rows = manifest_rows(clips, refs, wav_dir, args.seed)
    write_atomic(out / "voxcpm_train.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_atomic(out / "ATTRIBUTION.txt", "".join(x + "\n" for x in attribution_lines(clips)))
    settings = {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if k == "exclude_manifest"
                    else list(v) if k == "sources" else v) for k, v in vars(args).items()}
    units = {e["file"]: i for i, e in enumerate(done)}
    by_unit: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for x in read_jsonl(out / "scores.jsonl"):
        if x.get("run") == state["run"] and x.get("unit") in units:
            by_unit[x["unit"]][x["id"]] = x  # a unit redone (more --hours): its last lines
    score_units = [list(by_unit[u].values()) for u in sorted(by_unit, key=units.get)]
    extra = {"orphan_wavs_removed": len(orphans), "missing_wavs_refetched": refetched,
             "elapsed_s": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if isinstance(scorer, Utmos) and scorer.clips:
        extra["utmos_this_call"] = {"device": scorer.device, "clips": scorer.clips, "seconds": round(scorer.seconds, 1)}
    rep = build_report(clips, done, refs, settings, guard, score_units, f, extra)
    write_atomic(out / "prep_report.json", json.dumps(rep, ensure_ascii=False, indent=1))
    write_atomic(out / "PREP_REPORT.md", render_markdown(rep))
    sel, u = rep["selected"], rep["utmos"]
    log(f"[human] {sel['hours']} h, {sel['clips']} clips, {sel['speakers']} speakers, {len(refs)} with ref_audio "
        f"(target {args.hours:g} h {'reached' if rep['reached'] else 'NOT reached: units used up'}); UTMOS candidates "
        f"median {u['candidates'].get('median')}, kept median {u['kept'].get('median')}; {len(orphans)} orphan WAVs "
        f"removed -> {out}")
    return 0 if clips else 1


if __name__ == "__main__":
    raise SystemExit(main())
