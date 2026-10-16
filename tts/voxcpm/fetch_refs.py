#!/usr/bin/env python3
"""Reference clips of the cross-lingual voices in the voice contest (docs/SES_PLANI.md, step 1).

One clean NEUTRAL 6-12 s clip per voice (+ one expressive clip where the corpus has one), for VoxCPM2's
reference-only cloning: generate(reference_wav_path=...). Never the prompt_wav_path / prompt_text continuation mode:
it carries the source language's accent into Turkish (OpenBMB/VoxCPM issue #321). The transcript is kept for
provenance only.

  entry      voice (corpus label)                         corpus, licence                              clips
  masha      Masha (M), F, professional theatre actor      langswap/dialogs-ru-emotional-conversations  neutral, happy
  dima       Dima (D), M, professional theatre actor       (Russian; OpenRAIL, written consent)          neutral
  kazakh_f1  KazakhTTS F1, F, 44, professional narrator    ISSAI KazakhTTS2 (Kazakh; CC BY 4.0), read    neutral (news)
  kazakh_m1  KazakhTTS M1, M, 46, professional narrator    from the parquet copy ai4kazakh/ISSAI_...    neutral (news)
  thorsten   Thorsten Müller, M, voice donor               Thorsten-Voice/TV-44kHz-Full (German; CC0)   neutral, amused

Per clip: up to --candidates rows are picked from the corpus metadata (speaker, style, duration or text length, no
digits or laughter in the text), downloaded and measured: duration, peak and clipped share, noise floor (10th
percentile of the 20 ms frame level re its 95th: noise or the other actor's voice in the pauses raise it, and so does
pause-free read speech), tail level (last 50 ms re the 95th: a cut-off ending) and UTMOS (score_utmos.py, UTMOS22
strong; --utmos auto uses it when torch loads and the weights are already in the torch hub cache, --utmos on downloads
them). Kept: the highest UTMOS among the candidates without gross faults (duration, clipping, floor > -20 dB,
tail > -12 dB; without UTMOS: the lowest floor). The file is written as the corpus has it: no resampling, trimming or
gain.

Writes voxcpm-refs/<entry>/neutral.wav (+ expressive.wav) and voxcpm-refs/<entry>/sources.json: dataset, revision,
file, speaker, text, licence and its URL, consent wording and its source, attribution, obligations, sha256, the
measures of every candidate; voxcpm-refs/sources.json indexes all entries. An entry is written whole or not at all:
its clips are staged and replace the old ones only when every clip is in hand, so a failed (re-)fetch leaves the
folder as it was. An expressive clip is left out ("missing") only when the metadata has no candidate for it; candidates
that could not be downloaded fail the entry (exit 1), and the next run tries again. An entry whose files match its
sources.json is kept as it is, only its registry fields (licence, consent, attribution, ...) are brought up to date
(--force fetches it again). Nothing is uploaded anywhere (modal_app's contest-refs-push does that).

    python tts/voxcpm/fetch_refs.py                                  # all entries -> voxcpm-refs/
    python tts/voxcpm/fetch_refs.py --only masha,dima --candidates 4
    python tts/voxcpm/fetch_refs.py --list                           # registry, licences, consent; no network

Downloads, all public (a gated dataset is retried with the local huggingface_hub login, else the error names the page
whose terms to accept), ~105 MB for all entries at 8 candidates: Dialogs clips are single WAVs (~0.9 MB per 10 s; its
5 MB metadata.csv and file listing are cached in --cache); KazakhTTS2 reads, by HTTP range requests, only the parquet
pages up to the last candidate (~2 MB dictionary page + ~0.25 MB per row, ~7 MB per voice); Thorsten's parquet keeps
each row group's audio in one ~25-37 MB dictionary page, so all candidates of a clip come from one row group (~66 MB
for both clips). Not counted: the UTMOS22 weights, 411 MB from GitHub into a cold torch hub cache, fetched only with
--utmos on (auto then ranks by the noise floor). Needs numpy + soundfile; pyarrow for kazakh_* and thorsten; torch +
torchaudio for UTMOS (optional).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # score_utmos (optional UTMOS)
OUT = REPO / "voxcpm-refs"
CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "turkish-tts" / "fetch_refs"  # metadata
HF = "https://huggingface.co"
TIMEOUT_S, RETRIES = 60, 3
FRAME_S, HOP_S, TAIL_S = 0.020, 0.010, 0.050
# gross faults only (UTMOS does the ranking): pause-free, tightly trimmed read speech (KazakhTTS) has a floor of -21 to
# -33 dB and a tail of -19 to -36 dB, Dialogs clips with pauses -36 to -63 / -35 to -100 dB
FLOOR_MAX_DB = -20.0  # 10th-percentile frame re the 95th: above it, noise or no dynamics at all
TAIL_MAX_DB = -12.0  # last 50 ms re the 95th: above it the clip probably ends mid-word
CLIP_MAX = 1e-4  # share of samples at full scale
LAUGH = re.compile(r"(?i)\b(ха|хи|хе)\b|ха-ха|haha|hehe|\[|\*")  # laughter syllables, markup
UTMOS_CKPT = "utmos22_strong_step7459_v1.pt"  # what SpeechMOS v1.2.0's hubconf caches (411 MB, GitHub release)

SOURCES = {
    "dialogs": {
        "dataset": "langswap/dialogs-ru-emotional-conversations",
        "revision": "09b2b30d2afe5fe21d1dc191c4415ea76c16ef97",
        "license": "OpenRAIL (BigScience / CreativeML OpenRAIL-M template applied to a dataset); commercial use "
                   "allowed",
        "license_file": "LICENSE.md",
        "consent": "The voice actors \"gave written informed consent for open release and lawful use, including "
                   "commercial\".",
        "consent_source": "dataset card (README.md) and LICENSE.md footer",
        "attribution": "Dialogs: A Studio-Quality Expressive Conversational Russian Speech Corpus for Dialog "
                       "Assistants, I. Shigabeev and I. Latyshev (Langswap), Interspeech 2026, "
                       "https://huggingface.co/datasets/langswap/dialogs-ru-emotional-conversations, OpenRAIL",
        "obligations": "OpenRAIL flow-down: models trained on or derived from the data and their outputs are "
                       "Derivatives; distributing one requires shipping the licence with its use restrictions "
                       "(Section 2 / Attachment A) and binding recipients to them (Section 3). Restriction 10 "
                       "forbids providing medical advice and medical results interpretation.",
    },
    "kazakhtts2": {
        "dataset": "ai4kazakh/ISSAI_KazakhTTS2",
        "revision": "bcf754ffb4074e48e6dae6aa7e0b83630fd5410c",
        "official": "https://huggingface.co/datasets/issai/KazakhTTS (ISSAI_KazakhTTS2.tar.gz, 36 GB) and "
                    "https://github.com/IS2AI/Kazakh_TTS",
        "mirror_note": "ai4kazakh/ISSAI_KazakhTTS2 is a third-party parquet copy (no card text), read because a "
                       "range request can reach one clip in it; the file name is the corpus's own",
        "license": "CC BY 4.0 (KazakhTTS paper; LICENSE of github.com/IS2AI/Kazakh_TTS); free for academic and "
                   "commercial use",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "consent": "Each speaker participated voluntarily and was \"informed of the data collection and use "
                   "protocols through a consent form\" (approved by the Nazarbayev University ethics committee).",
        "consent_source": "KazakhTTS (Interspeech 2021, https://arxiv.org/abs/2104.08459) section 3, which also "
                          "states approval by the Institutional Research Ethics Committee of Nazarbayev University; "
                          "KazakhTTS2 (LREC 2022, https://arxiv.org/abs/2201.05771) section 3.3 says the same of its "
                          "speakers",
        "attribution": "Mussakhojayeva, S., Janaliyeva, A., Mirzakhmetov, A., Khassanov, Y., Varol, H.A. (2021) "
                       "KazakhTTS: An Open-Source Kazakh Text-to-Speech Synthesis Dataset. Proc. Interspeech 2021, "
                       "2786-2790, doi: 10.21437/Interspeech.2021-2124; Mussakhojayeva, S., Khassanov, Y., Varol, "
                       "H.A. (2022) KazakhTTS2: Extending the Open-Source Kazakh TTS Corpus With More Data, Speakers, "
                       "and Topics. Proc. LREC 2022, 5404-5411. The Institute of Smart Systems and Artificial "
                       "Intelligence (issai.nu.edu.kz), Nazarbayev University, Kazakhstan. "
                       "https://github.com/IS2AI/Kazakh_TTS (dataset: "
                       "https://huggingface.co/datasets/issai/KazakhTTS). "
                       "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "obligations": "CC BY 4.0 section 3(a): attribution (above, with the link to the material), keep any "
                       "copyright and disclaimer notice supplied with it, licence link, indicate changes. The consent "
                       "form itself is not published: it is not known whether it names voice cloning in other "
                       "languages.",
    },
    "thorsten": {
        "dataset": "Thorsten-Voice/TV-44kHz-Full",
        "revision": "2b61b98fa8f99abd1ce1587b4bf413d6ebc217d5",
        "license": "CC0 1.0 (public domain dedication by the speaker himself)",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "consent": "Thorsten Müller, \"just an open source speech technology enthusiast who donates his voice\", "
                   "released his own recordings under CC0 so that every project can have a free German TTS voice.",
        "consent_source": "https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full (Disclaimer) and "
                          "https://github.com/thorstenMueller/Thorsten-Voice (README, CC0-1.0)",
        "attribution": "Thorsten-Voice dataset by Thorsten Müller (https://www.Thorsten-Voice.de), TV-44kHz-Full, "
                       "doi: 10.57967/hf/3290, CC0 1.0",
        "obligations": "none (CC0); attribution is courtesy. CC0 1.0 also waives the affirmer's publicity and "
                       "performer rights to the extent he has them.",
    },
}

# clip spec: style (corpus label), duration window (s); dialogs: speaker id; parquet: dir, shards, filters
ENTRIES = {
    "masha": {"source": "dialogs", "voice": "Masha", "gender": "female", "language": "ru",
              "speaker": "Dialogs speaker_id M (Masha): professional theatre actor, 9.9 h in the corpus",
              "clips": {"neutral": {"speaker": "M", "style": "neutral", "dur": (6.0, 12.0)},
                        "expressive": {"speaker": "M", "style": "happy", "dur": (6.0, 12.0)}}},
    "dima": {"source": "dialogs", "voice": "Dima", "gender": "male", "language": "ru",
             "speaker": "Dialogs speaker_id D (Dima): professional theatre actor, 6.2 h in the corpus",
             "clips": {"neutral": {"speaker": "D", "style": "neutral", "dur": (6.0, 12.0)}}},
    "kazakh_f1": {"source": "kazakhtts2", "voice": "KazakhTTS F1", "gender": "female", "language": "kk",
                  "speaker": "KazakhTTS(2) speaker F1: female, 44, 14 years of narration experience, news reading",
                  "clips": {"neutral": {"dir": "f1", "shards": ["train-00000-of-00012.parquet"],
                                        "prefix": "inform_", "chars": (90, 165), "dur": (6.0, 12.0)}}},
    "kazakh_m1": {"source": "kazakhtts2", "voice": "KazakhTTS M1", "gender": "male", "language": "kk",
                  "speaker": "KazakhTTS(2) speaker M1: male, 46, 12 years of narration experience, news reading",
                  "clips": {"neutral": {"dir": "m1", "shards": ["train-00006-of-00025.parquet"],  # reads slower
                                        "prefix": "inform_", "chars": (75, 130), "dur": (6.0, 12.0)}}},
    "thorsten": {"source": "thorsten", "voice": "Thorsten", "gender": "male", "language": "de",
                 "speaker": "Thorsten Müller, native German speaker, not a professional speaker (own words)",
                 "clips": {"neutral": {"dir": "TV-2022.10-Neutral", "style": "neutral", "dur": (6.0, 12.0),
                                       "mic": "good_rodePodcaster", "shards": ["train-00007-of-00008.parquet"]},
                           "expressive": {"dir": "TV-2021.06-Emotional", "style": "amused | amüsiert",
                                          "dur": (4.5, 12.0), "mic": "good_rodePodcaster",
                                          "shards": ["train-00000-of-00002.parquet"]}}},
}
PARQUET_META = {"kazakhtts2": ["text", "audio.path"],  # read for every row group of the spec's shards
                "thorsten": ["style", "text", "durationSeconds", "microphone"]}


class GatedError(RuntimeError):
    pass


class NotDownloaded(RuntimeError):
    """The metadata had candidates for a clip but none could be downloaded (network): the entry is not written."""


# ---- HTTP -------------------------------------------------------------------------------------------------------

_fetched = [0]  # bytes downloaded by this process


class _NoAuthAcrossHosts(urllib.request.HTTPRedirectHandler):
    """Follow redirects but never send the HF token to another host (the CDN URL is signed)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).hostname != urllib.parse.urlsplit(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_NoAuthAcrossHosts)


def _token() -> str | None:
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return os.environ.get("HF_TOKEN") or None


def _get(url: str, rng: tuple[int, int] | None = None, dataset: str = "") -> tuple[bytes, str, Message]:
    """GET url (bytes rng inclusive): (body, final URL, headers). Retries network errors; a 401/403 from the Hub is
    retried once with the local login, then raised as GatedError naming the page to accept."""
    headers = {"User-Agent": "turkish-tts fetch_refs"}
    if rng:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    auth = False
    for attempt in range(RETRIES + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with _OPENER.open(req, timeout=TIMEOUT_S) as r:
                body = r.read()
                _fetched[0] += len(body)
                return body, r.geturl(), r.headers  # case-insensitive get()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) and urllib.parse.urlsplit(url).hostname == "huggingface.co":
                tok = None if auth else _token()
                if tok:
                    headers["Authorization"], auth = f"Bearer {tok}", True
                    continue
                raise GatedError(f"{url}: HTTP {e.code}. Open {HF}/datasets/{dataset} logged in, accept its terms, "
                                 "then log in here: python -c \"from huggingface_hub import login; login()\"") from e
            if e.code < 500 or attempt == RETRIES:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == RETRIES:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def resolve_url(dataset: str, revision: str, path: str) -> str:
    return f"{HF}/datasets/{dataset}/resolve/{revision}/{urllib.parse.quote(path)}"


class RangeFile(io.RawIOBase):
    """Read-only seekable view of a remote file through HTTP range requests (what pyarrow needs to read one page)."""

    def __init__(self, url: str, dataset: str = ""):
        body, self.url, h = _get(url, (0, 0), dataset)  # resolves the redirect once; the signed URL serves ranges
        m = re.search(r"/(\d+)$", h.get("Content-Range", ""))
        if not m:
            raise RuntimeError(f"{url}: no range support (Content-Range {h.get('Content-Range')!r})")
        self.size, self.pos, self.dataset = int(m.group(1)), 0, dataset

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = _get(self.url, (self.pos, self.pos + n - 1), self.dataset)[0]
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


# ---- candidates -------------------------------------------------------------------------------------------------


def text_ok(text: str) -> bool:
    """Plain speech: no digits (read out in another language's way), laughter or markup."""
    return bool(text.strip()) and not re.search(r"\d", text) and not LAUGH.search(text)


def dialogs_pick(rows: list[dict], spec: dict, k: int, seed: int, files: set[str] | None = None) -> list[dict]:
    """k metadata.csv rows of the speaker and style in the duration window whose WAV is in the repo (files; 1,800 of
    its 11,796 rows have none), 7-11 s first, in seeded random order."""
    lo, hi = spec["dur"]
    pool = [r for r in rows if r["speaker_id"] == spec["speaker"] and r["emotion"] == spec["style"]
            and lo <= float(r["duration"]) <= hi and len(r["text"]) >= 40 and text_ok(r["text"])
            and (files is None or r["audio_path"] in files)]
    rnd = random.Random(f"{seed}/{spec['speaker']}/{spec['style']}")
    rnd.shuffle(pool)
    pool.sort(key=lambda r: not 7.0 <= float(r["duration"]) <= 11.0)  # stable: random within each tier
    return pool[:k]


def cached(path: Path, url: str, dataset: str) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(url, dataset=dataset)[0])
    return path.read_text(encoding="utf-8")


def dialogs_candidates(spec: dict, k: int, seed: int, cache: Path) -> list[dict]:
    src, tag = SOURCES["dialogs"], SOURCES["dialogs"]["revision"][:12]
    meta = cached(cache / f"dialogs-{tag}-metadata.csv", resolve_url(src["dataset"], src["revision"], "metadata.csv"),
                  src["dataset"])
    listing = cached(cache / f"dialogs-{tag}-files.json",
                     f"{HF}/api/datasets/{src['dataset']}/revision/{src['revision']}", src["dataset"])
    files = {s["rfilename"] for s in json.loads(listing)["siblings"]}
    rows = list(csv.DictReader(io.StringIO(meta), delimiter="|"))
    out = []
    for r in dialogs_pick(rows, spec, k, seed, files):
        url = resolve_url(src["dataset"], src["revision"], r["audio_path"])
        out.append({"file": r["audio_path"], "url": url, "text": r["text"], "style": r["emotion"],
                    "meta_duration_s": round(float(r["duration"]), 3),
                    "fetch": lambda url=url: _get(url, dataset=src["dataset"])[0]})
    return out


def parquet_eligible(source: str, spec: dict, meta: list[dict]) -> list[int]:
    """Row indices (within one row group) whose metadata fit the clip spec."""
    out = []
    for i, r in enumerate(meta):
        if source == "kazakhtts2":
            path = (r.get("audio") or {}).get("path") or ""
            ok = path.startswith(spec["prefix"]) and spec["chars"][0] <= len(r["text"]) <= spec["chars"][1]
        else:
            ok = (r["style"] == spec["style"] and (not spec.get("mic") or r["microphone"] == spec["mic"])
                  and spec["dur"][0] <= float(r["durationSeconds"]) <= spec["dur"][1])
        if ok and text_ok(r["text"]):
            out.append(i)
    return out


def best_row_group(found: dict[tuple[str, int], list[int]], k: int) -> tuple[tuple[str, int], list[int]] | None:
    """The row group with the most eligible rows (up to k), ties to the one whose k-th is earliest (fewest pages to
    read); returns its key and the chosen row indices."""
    best = None
    for key, idx in found.items():
        take = idx[:k]
        if take and (best is None or (len(take), -take[-1]) > (len(best[1]), -best[1][-1])):
            best = (key, take)
    return best


def parquet_candidates(source: str, spec: dict, k: int) -> list[dict]:
    import pyarrow.parquet as pq

    src = SOURCES[source]
    found, files, metas = {}, {}, {}
    for shard in spec["shards"]:
        path = f"{spec['dir']}/{shard}"
        url = resolve_url(src["dataset"], src["revision"], path)
        rf = RangeFile(url, src["dataset"])
        pf = pq.ParquetFile(io.BufferedReader(rf, buffer_size=1 << 16), pre_buffer=False, buffer_size=2 << 20)
        files[path] = (pf, url)
        for g in range(pf.metadata.num_row_groups):
            meta = pf.read_row_group(g, columns=PARQUET_META[source]).to_pylist()
            metas[(path, g)] = meta
            if idx := parquet_eligible(source, spec, meta):
                found[(path, g)] = idx
    pick = best_row_group(found, k)
    if pick is None:
        return []
    (path, g), take = pick
    pf, url = files[path]
    want, got, row = set(take), {}, 0
    for batch in pf.iter_batches(batch_size=4, row_groups=[g], columns=["audio"], use_threads=False):
        for a in batch.column("audio").to_pylist():
            if row in want:
                got[row] = a
            row += 1
        if row > take[-1]:
            break
    out = []
    for i in take:
        m, a = metas[(path, g)][i], got[i]
        name = a.get("path") or f"row{i}.wav"
        c = {"file": f"{path}#row_group={g}&row={i}&path={name}", "url": url, "corpus_file": name,
             "text": m["text"], "style": m.get("style", "news"), "fetch": lambda b=a["bytes"]: b}
        if "durationSeconds" in m:
            c["meta_duration_s"] = round(float(m["durationSeconds"]), 3)
        out.append(c)
    return out


# ---- measures ---------------------------------------------------------------------------------------------------


def measure(data: bytes) -> dict:
    """Format and level measures of an audio file's bytes (any channel count; levels on the channel mean)."""
    import numpy as np
    import soundfile as sf

    info = sf.info(io.BytesIO(data))
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    clipped = float((np.abs(wav) >= 0.999).mean()) if wav.size else 0.0
    x = wav.mean(axis=1)
    m = {"duration_s": round(len(x) / sr, 3), "sample_rate": int(sr), "channels": int(wav.shape[1]),
         "format": info.format, "subtype": info.subtype, "peak": round(peak, 4), "clipped": round(clipped, 6)}
    frame, hop = int(FRAME_S * sr), int(HOP_S * sr)
    if len(x) < frame + hop:
        return {**m, "floor_db": None, "tail_db": None}
    win = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    db = 10 * np.log10(np.mean(win.astype(np.float64) ** 2, axis=1) + 1e-12)
    ref = float(np.percentile(db, 95))
    tail = x[-int(TAIL_S * sr):].astype(np.float64)
    m["floor_db"] = round(float(np.percentile(db, 10)) - ref, 1)
    m["tail_db"] = round(10 * float(np.log10(np.mean(tail ** 2) + 1e-12)) - ref, 1)
    m["level_dbfs"] = round(ref, 1)
    return m


def issues(m: dict, dur: tuple[float, float]) -> list[str]:
    out = []
    if not dur[0] <= m["duration_s"] <= dur[1]:
        out.append(f"duration {m['duration_s']} s outside {dur[0]:g}-{dur[1]:g}")
    if m["clipped"] > CLIP_MAX:
        out.append(f"clipped {m['clipped']:.2%}")
    if m["floor_db"] is None or m["floor_db"] > FLOOR_MAX_DB:
        out.append(f"noise floor {m['floor_db']} dB > {FLOOR_MAX_DB:g}")
    if m["tail_db"] is not None and m["tail_db"] > TAIL_MAX_DB:
        out.append(f"tail {m['tail_db']} dB > {TAIL_MAX_DB:g}: may end mid-word")
    return out


def rank_key(c: dict) -> tuple:
    """Higher is better: no issues, then UTMOS (when scored), then the lower noise floor."""
    floor = c["floor_db"] if c.get("floor_db") is not None else 0.0
    utmos = c.get("utmos")
    return (not c["issues"], utmos if utmos is not None else float("-inf"), -floor)


def utmos_weights() -> Path:
    """Where torch.hub keeps the UTMOS22 weights (imports torch)."""
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints" / UTMOS_CKPT


class Utmos:
    """score_utmos's predictor, loaded on first use; off, or unavailable under auto: score() returns None. auto never
    downloads the 411 MB weights (only the hub code, < 100 kB, if that is not cached); on does."""

    def __init__(self, mode: str):
        self.mode, self.pred, self.note = mode, None, "off" if mode == "off" else ""

    def score(self, path: Path) -> float | None:
        if self.mode == "off" or (self.note and self.pred is None):
            return None
        if self.pred is None:
            try:
                if self.mode == "auto" and not (w := utmos_weights()).is_file():
                    raise FileNotFoundError(f"{w} not cached; --utmos on downloads it, 411 MB")
                from score_utmos import load_predictor

                self.pred, self.note = load_predictor("cpu"), "UTMOS22 strong (tarepan/SpeechMOS:v1.2.0), cpu"
            except Exception as e:
                if self.mode == "on":
                    raise
                self.note = f"unavailable ({type(e).__name__}: {e}); ranked by noise floor"
                print(f"[refs] UTMOS {self.note}", flush=True)
                return None
        from score_utmos import score_wav

        return round(score_wav(self.pred, path, "cpu"), 4)


# ---- entries ----------------------------------------------------------------------------------------------------


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def license_url(src: dict) -> str:
    if src.get("license_url"):
        return src["license_url"]
    return f"{HF}/datasets/{src['dataset']}/blob/{src['revision']}/{src['license_file']}"


def entry_record(name: str) -> dict:
    e, src = ENTRIES[name], SOURCES[ENTRIES[name]["source"]]
    rec = {"entry": name, "voice": e["voice"], "gender": e["gender"], "language": e["language"],
           "speaker": e["speaker"], "dataset": src["dataset"], "dataset_url": f"{HF}/datasets/{src['dataset']}",
           "revision": src["revision"], "license": src["license"], "license_url": license_url(src),
           "consent": src["consent"], "consent_source": src["consent_source"], "attribution": src["attribution"],
           "obligations": src["obligations"],
           "use": "reference-only cloning (VoxCPM2 reference_wav_path); never prompt_wav/prompt_text continuation"}
    for k in ("official", "mirror_note"):
        if k in src:
            rec[k] = src[k]
    return rec


def is_current(d: Path, name: str) -> bool:
    """d/sources.json lists every clip of the entry, each file present with its recorded sha256."""
    try:
        rec = json.loads((d / "sources.json").read_text(encoding="utf-8"))
        clips = rec["clips"]
        return set(ENTRIES[name]["clips"]) <= set(clips) | set(rec.get("missing", [])) and all(
            sha256((d / c["path"]).read_bytes()) == c["sha256"] for c in clips.values())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def drop_stale(d: Path, kind: str, keep: Path | None = None) -> None:
    """Remove an older <kind>.<ext> clip (another extension, or a clip no longer found): the contest runner takes
    every audio file of the folder as an entry."""
    for old in d.glob(f"{kind}.*"):
        if old != keep and old.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"):
            old.unlink()


def fetch_clip(name: str, kind: str, spec: dict, args, utmos: Utmos, tmp: Path) -> dict | None:
    """The best candidate of one clip, staged in tmp (its "_staged" path; the entry's folder is not touched). None: the
    metadata has no candidate. NotDownloaded: it has some, none could be downloaded."""
    source = ENTRIES[name]["source"]
    t0, b0 = time.time(), _fetched[0]
    if source == "dialogs":
        cands = dialogs_candidates(spec, args.candidates, args.seed, args.cache)
    else:
        cands = parquet_candidates(source, spec, args.candidates)
    if not cands:
        print(f"[refs] {name}/{kind}: no candidate in the metadata", flush=True)
        return None
    got = []
    tmp.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(cands):
        try:
            data = c.pop("fetch")()
        except Exception as e:  # e.g. a 404 behind the listing: one candidate less
            print(f"[refs] {name}/{kind} {i + 1}/{len(cands)} not downloaded ({type(e).__name__}: {e}): {c['file']}")
            continue
        got.append(c)
        c["_path"] = p = tmp / f"{kind}-{i}{Path(c.get('corpus_file') or c['file']).suffix or '.wav'}"
        p.write_bytes(data)
        c.update(sha256=sha256(data), bytes=len(data))
        try:
            c.update(measure(data))
            c["issues"] = issues(c, spec["dur"])
        except Exception as e:  # not decodable: listed, never kept while another candidate exists
            c.update(duration_s=0.0, floor_db=None, tail_db=None, peak=None, issues=[f"unreadable: {e}"])
        c["utmos"] = utmos.score(p) if c["floor_db"] is not None else None
        print(f"[refs] {name}/{kind} {i + 1}/{len(cands)} {c['duration_s']:5.2f} s  utmos {c['utmos']}  floor "
              f"{c['floor_db']} dB  tail {c['tail_db']} dB  {'; '.join(c['issues']) or 'ok'}  {c['file']}", flush=True)
    if not got:
        raise NotDownloaded(f"{name}/{kind}: none of the {len(cands)} candidates could be downloaded (network?); "
                            "the entry is left as it was, run again")
    best = max(got, key=rank_key)
    staged = best.pop("_path")
    for c in got:
        if "_path" in c:
            c.pop("_path").unlink(missing_ok=True)
    keep = ("file", "corpus_file", "url", "style", "text", "meta_duration_s", "duration_s",
            "sample_rate", "channels", "format", "subtype", "level_dbfs", "peak", "clipped", "floor_db", "tail_db",
            "utmos", "issues", "sha256", "bytes")
    clip = {"path": f"{kind}{staged.suffix}", "_staged": staged, **{k: best[k] for k in keep if k in best}}
    clip["candidates"] = [{k: c[k] for k in ("file", "duration_s", "utmos", "floor_db", "tail_db", "peak", "issues")
                           if k in c} for c in got]
    clip["selection"] = ("highest UTMOS among the candidates without issues" if best.get("utmos") is not None
                         else "lowest noise floor among the candidates without issues")
    if best["issues"]:
        clip["selection"] += f"; NONE was clean, kept the best anyway: {'; '.join(best['issues'])}"
    print(f"[refs] {name}/{kind}: kept {best['file']} ({best['duration_s']} s, utmos {best['utmos']}); "
          f"{(_fetched[0] - b0) / 1e6:.1f} MB in {time.time() - t0:.0f} s", flush=True)
    return clip


def fetch_entry(name: str, args, utmos: Utmos) -> dict:
    d = args.out / name
    if not args.force and is_current(d, name):
        rec, fresh = json.loads((d / "sources.json").read_text(encoding="utf-8")), entry_record(name)
        if changed := [k for k, v in fresh.items() if rec.get(k) != v]:  # registry edited since: no download needed
            rec.update(fresh)
            write_json(d / "sources.json", rec)
        print(f"[refs] {name}: up to date ({d / 'sources.json'}"
              f"{'; updated ' + ', '.join(changed) if changed else ''}), --force fetches it again", flush=True)
        return rec
    rec = entry_record(name)
    rec["clips"], rec["missing"] = {}, []
    tmp = d / ".candidates"
    try:
        for kind, spec in ENTRIES[name]["clips"].items():
            clip = fetch_clip(name, kind, spec, args, utmos, tmp)  # NotDownloaded / errors: the folder is untouched
            if clip is None and kind == "neutral":
                raise RuntimeError(f"{name}: no neutral reference clip in the metadata")
            if clip is None:
                rec["missing"].append(kind)  # the corpus has no such clip: the expressive one is optional
            else:
                rec["clips"][kind] = clip
        # every clip is in hand: the old record goes first, so the folder never pairs a new clip with an old record
        (d / "sources.json").unlink(missing_ok=True)
        for kind, clip in rec["clips"].items():
            os.replace(clip.pop("_staged"), d / clip["path"])
            drop_stale(d, kind, d / clip["path"])
        for kind in rec["missing"]:
            drop_stale(d, kind)
    finally:
        if tmp.is_dir():  # empty after a good run; the staged clips of a failed one
            for p in tmp.iterdir():
                p.unlink()
            tmp.rmdir()
    rec["fetched_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec["utmos"] = utmos.note or "not needed"
    rec["tool"] = "tts/voxcpm/fetch_refs.py"
    write_json(d / "sources.json", rec)
    return rec


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def print_registry() -> None:
    for name, e in ENTRIES.items():
        rec = entry_record(name)
        print(f"{name}: {e['voice']} ({e['gender']}, {e['language']}), clips {', '.join(e['clips'])}\n"
              f"  dataset  {rec['dataset_url']} @ {rec['revision'][:12]}\n  licence  {rec['license']}: "
              f"{rec['license_url']}\n  consent  {rec['consent']} ({rec['consent_source']})")


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Turkish / Cyrillic / German text on a Windows console (also --help)
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT, help="output root (default: voxcpm-refs/ in the repo)")
    ap.add_argument("--only", default="", help=f"comma-separated entries (default all: {','.join(ENTRIES)})")
    ap.add_argument("--cache", type=Path, default=CACHE,
                    help="Dialogs metadata.csv and file listing are kept here (default: %(default)s)")
    ap.add_argument("--candidates", type=int, default=8, help="candidates measured per clip (default 8)")
    ap.add_argument("--utmos", choices=["auto", "on", "off"], default="auto",
                    help="rank candidates by UTMOS: auto = if torch loads and the weights are in the torch hub cache; "
                         "on = also download them (411 MB, GitHub)")
    ap.add_argument("--seed", type=int, default=0, help="Dialogs candidate order")
    ap.add_argument("--force", action="store_true", help="fetch entries again that are already up to date")
    ap.add_argument("--list", action="store_true", help="print the registry (sources, licences, consent) and exit")
    args = ap.parse_args(argv)
    if args.list:
        print_registry()
        return 0
    names = [n for n in args.only.split(",") if n] or list(ENTRIES)
    if bad := [n for n in names if n not in ENTRIES]:
        ap.error(f"unknown entries {bad}; known: {', '.join(ENTRIES)}")
    if args.candidates < 1:
        ap.error("--candidates must be >= 1")
    args.out.mkdir(parents=True, exist_ok=True)
    utmos, t0, failed = Utmos(args.utmos), time.time(), {}
    index_path = args.out / "sources.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = {}
    index.setdefault("entries", {})
    for name in names:
        try:
            index["entries"][name] = fetch_entry(name, args, utmos)
        except GatedError as e:
            failed[name] = str(e)
        except Exception as e:  # one voice failing does not cost the others
            failed[name] = f"{type(e).__name__}: {e}"
        if name in failed:
            print(f"[refs] {name} FAILED: {failed[name]}", flush=True)
        write_json(index_path, index)
    print(f"[refs] {len(names) - len(failed)}/{len(names)} entries in {args.out}; downloaded {_fetched[0] / 1e6:.1f} "
          f"MB in {time.time() - t0:.0f} s")
    for name in names:
        for kind, c in index["entries"].get(name, {}).get("clips", {}).items():
            print(f"  {name:<10} {kind:<10} {c['duration_s']:5.2f} s {c.get('sample_rate')} Hz  "
                  f"utmos {c.get('utmos')}  {c['file']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
