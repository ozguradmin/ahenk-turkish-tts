#!/usr/bin/env python3
"""Stage-1 corpus for the two-stage VoxCPM2 LoRA: a filtered subset of Alania synthetic Turkish speech, 16 kHz.

Corpus: cloud0day3/alania-synthetic-speech-tr, config cc-by (CC BY 4.0, PatientDesk AI): VoxCPM2's own output,
QC-scored by its authors. 655 parquet files data/cc-by/train-*.parquet (~830 MB, 2,513 rows, ~3.9 h each), audio FLAC
48 kHz mono 16-bit; `text_spoken` is the text the model read (kept as it is, artefacts like "lira nin" included).

    python tts/voxcpm/prepare_alania.py --out $TTS_DATA/alania16k --hours 25 \
        --exclude-manifest $TTS_DATA/antalia16k/voxcpm_val.jsonl

An expressive corpus (plain rows plus style-instruction rows, no style above 10 % of the hours), in its own --out:

    python tts/voxcpm/prepare_alania.py --out $TTS_DATA/alania_expr16k --hours 25 \
        --modes reference,reference+instruction --max-style-share 0.1 \
        --exclude-manifest $TTS_DATA/antalia16k/voxcpm_val.jsonl

A corpus keeps its --modes: another --modes into the same --out is refused, --restart included (it would replace the
corpus, and train_lora.py tells corpora apart by folder name only, so an experiment trained on it would resume on other
data). modal_app.py's prepare-alania always writes /data/alania16k (it refuses --out in --args), so on Modal the
expressive corpus needs its own Volume folder in modal_app.py first.

Steps
 1. Files of the pinned --revision in a spread order (golden-ratio stride over the sorted list, so the first files
    come from all over the dataset), walked until --hours are selected (or --max-files are used up).
 2. Per file the metadata columns only (HfFileSystem range reads; --row-groups N: only the first N row groups), then
    the filters, in this order, each drop counted with its reason: render_mode not in --modes, style not in --styles
    (reference+instruction rows), qc_cer > --max-cer, qc_utmos < --min-utmos (--min-utmos-instruction for
    reference+instruction rows), --gender, duration outside [--min-dur, --max-dur], empty text_spoken, contamination,
    duplicate id or text (normalized text or text_spoken already selected), --max-per-voice clips per voice_id,
    non-domain (fineweb2) hours above --fineweb-share of the running total, and a style already holding
    --max-style-share of it. Inside a file domain rows come first, best qc_utmos first. Selection is sequential in
    file order and files are committed in that order, so a resumed run selects exactly what an uninterrupted one would.
 3. Contamination guard: a row is dropped when the trnorm.normalize_for_eval form of its text or text_spoken equals a
    protected sentence, or contains one of >= 4 words. Protected: TR-Zor (benchmarks/tts/sets/tr_zor.jsonl: text,
    spoken, alts) and every --exclude-manifest row (Antalia's held-out voxcpm_val.jsonl), whole and per sentence; a
    protected text in control form "(instruction)sentence" also without its prefix. Rows are checked by their
    sentence (text, text_spoken), never by the instruction prefix their manifest text may get.
 4. Audio: only the `audio` column of row groups with a selected row is read, never `audio_channel` (a degraded
    copy). FLAC -> mono -> 16 kHz (soxr VHQ; torchaudio or scipy without it) -> PCM16 <out>/wav/<id>.wav.
 5. Resume: <out>/state.json lists the finished files with their selected ids; clips.jsonl gets each finished file's
    clips. A killed run continues with the next file and reuses WAVs already written; a committed clip whose WAV went
    missing gets its audio read again. A larger --hours continues the walk; other selection settings must match the
    state (--restart starts over, keeping the WAVs; --modes stay those of the state).
 6. Outputs: clips.jsonl (with each row's license, text_doc_id, style and instruction), voxcpm_train.jsonl (audio,
    text = text_spoken, "(instruction)" before it on instruction rows by default, duration; --ref-fraction of the
    rows get a ref_audio: another clip of the same voice_id of 3-10 s, never itself and never an instruction clip),
    ATTRIBUTION.txt (the Alania line; a FineWeb-2 ODC-By line when fineweb2 text is selected), prep_report.json and
    PREP_REPORT.md (hours per mode and per style too). WAVs in wav/ that are not in the selection (left by --restart)
    are removed.
 7. --out must be empty or this script's own (marker file .prepare_alania, or its state.json): another corpus there,
    e.g. antalia16k for alania16k, is refused even with --restart, as is an --exclude-manifest inside --out.

Instruction rows (render_mode reference+instruction, ~17 % of the rows): a designed voice following an English style
instruction (`instruction`, ~2 per style) of one of 18 `style`s (KNOWN_STYLES); their qc_utmos is ~0.2 lower than plain
rows', hence --min-utmos-instruction (default --min-utmos - 0.2). --instruction-format prefix (default) writes their
manifest text in VoxCPM2's control form "(instruction)text_spoken" (the trainer tokenizes text verbatim, so this trains
instruction following); drop writes text_spoken alone (styled audio, no control). Their ref_audio is a plain clip of
the voice, as Alania rendered them from the voice's plain reference. --max-style-share is checked before a row is
added (a style's first row always passes; each style overshoots by one clip at most), so a corpus of instruction rows
only can start and grow (with --max-style-share x styles > 1). voice-design rows keep text_spoken: their `instruction`
is the voice description, recorded in clips.jsonl only.
State: --instruction-format only shapes the manifest (rebuilt by every run); --styles, --min-utmos-instruction and
--max-style-share are part of the settings only when reference+instruction is in --modes, so the state of a run
without it (e.g. /data/alania16k, --modes reference) stays valid.

Exit code 0 when done, 1 when nothing was selected, 2 on bad arguments, a foreign --out or a state written with other
settings.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import wave
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from trnorm import normalize_for_eval  # noqa: E402  (pure Python)

REPO_ID = "cloud0day3/alania-synthetic-speech-tr"
CONFIG_DIR = "data/cc-by"
REVISION = "277dfc8a2322014099ff72688bcaee12480c4605"  # 2026-10-04, current on 2026-10-10
ATTRIBUTION = ("Alania Turkish Synthetic Speech by PatientDesk AI "
               "(https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr), licensed under CC BY 4.0")
FINEWEB_NOTE = ("Includes Turkish web text from FineWeb-2 (https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), "
                "used under ODC-By 1.0 and subject to Common Crawl's terms of use; source ids: text_doc_id in clips.jsonl")
TRZOR = REPO / "benchmarks" / "tts" / "sets" / "tr_zor.jsonl"
RATE = 16000  # VoxCPM2's AudioVAE encoder rate
META_COLUMNS = ["id", "text", "text_spoken", "duration", "voice_id", "gender", "render_mode", "instruction", "style",
                "text_source", "text_category", "qc_cer", "qc_utmos", "license", "text_doc_id"]
AUDIO_COLUMNS = ["id", "audio"]  # never audio_channel
MARKER = ".prepare_alania"  # in --out: this script's directory (only then are its files rewritten and WAVs removed)
OWN_FILES = ("state.json", "clips.jsonl", "voxcpm_train.jsonl", "ATTRIBUTION.txt", "prep_report.json", "PREP_REPORT.md")
CONTAIN_MIN_WORDS = 4  # protected sentences this long are also caught inside longer rows
REF_MIN_S, REF_MAX_S = 3.0, 10.0
STATE_VERSION = 1
INSTRUCTION_MODE = "reference+instruction"  # the rows with an `instruction` and a `style`
KNOWN_STYLES = ("annoyed", "apologetic", "authoritative", "businesslike", "calm", "casual", "cheerful", "empathetic",
                "fast", "neutral", "news", "sad", "sales", "slow", "storytelling", "surprised", "teacher", "warm")
INSTRUCTION_FIELDS = ("styles", "min_utmos_instruction", "max_style_share")  # Filters fields of INSTRUCTION_MODE rows
INSTRUCTION_UTMOS_GAP = 0.2  # their qc_utmos below plain rows' (cc-by sample, qc_cer 0: median 3.47 vs 3.65)
CONTROL = re.compile(r"^\(([^()]*)\)(?=\S)")  # VoxCPM2's "(instruction)text", as manifest_text writes it


# --------------------------------------------------------------------------- pure helpers (no pyarrow / numpy)
def collapse_ws(text: str) -> str:
    return " ".join((text or "").split())


def eval_key(text: str) -> str:
    """trnorm.normalize_for_eval, or a plain lowercase/no-punctuation form if trnorm fails on a row."""
    try:
        return normalize_for_eval(text or "")
    except Exception:  # noqa: BLE001  (one odd row must not stop a paid run)
        return " ".join(re.sub(r"[^\w\s]|_", " ", (text or "").lower()).split())


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?…])\s+", collapse_ws(text)) if s]


def spread_order(n: int) -> list[int]:
    """Permutation of range(n) by a stride near n / golden ratio, coprime with n: every prefix is spread over the
    whole list, so a few files already sample the dataset whatever order it was written in."""
    if n <= 2:
        return list(range(n))
    target = n / ((1 + 5 ** 0.5) / 2)
    stride = min((s for s in range(1, n) if math.gcd(s, n) == 1), key=lambda s: (abs(s - target), s))
    return [i * stride % n for i in range(n)]


def wav_name(clip_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", clip_id) + ".wav"


def is_web(row: dict) -> bool:
    """Counts toward --fineweb-share: every row whose text is not one of the dataset's domain scripts."""
    return (row.get("text_source") or row.get("source")) != "domain"


def style_of(row: dict) -> str | None:
    """The style of an instruction row (metadata: render_mode, clip record: mode); None for every other row."""
    return row.get("style") if row.get("render_mode", row.get("mode")) == INSTRUCTION_MODE else None


def strip_control(text: str) -> str:
    """`text` without a leading VoxCPM2 control prefix "(instruction)" (directly followed by the sentence)."""
    return CONTROL.sub("", text, count=1)


def manifest_text(clip: dict, instruction_format: str = "prefix") -> str:
    """text_spoken; for an instruction clip with instruction_format "prefix" VoxCPM2's control form
    "(instruction)text_spoken", parentheses inside the instruction removed (as VoxCPM2's own app does)."""
    instr = clip.get("instruction") if clip.get("mode") == INSTRUCTION_MODE else ""
    instr = collapse_ws(re.sub(r"[()（）]", " ", instr or ""))
    return f"({instr}){clip['text_spoken']}" if instr and instruction_format == "prefix" else clip["text_spoken"]


def attribution_lines(clips: list[dict]) -> list[str]:
    """ATTRIBUTION.txt: the Alania line, then the FineWeb-2 terms when any selected text is not a domain script."""
    web = any(is_web(c) or "ODC" in (c.get("license") or "") for c in clips)
    return [ATTRIBUTION] + [FINEWEB_NOTE] * web


def instruction_utmos(min_utmos: float) -> float:
    """Default --min-utmos-instruction: --min-utmos less INSTRUCTION_UTMOS_GAP, rounded (3.7 -> 3.5, not 3.5...04)."""
    return round(min_utmos - INSTRUCTION_UTMOS_GAP, 4)


def state_modes(path: Path) -> list[str] | None:
    """--modes of the state.json at `path` (from its settings key); None without a readable one."""
    try:
        modes = json.loads(json.loads(path.read_text(encoding="utf-8"))["key"])["filters"]["modes"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return modes if isinstance(modes, list) else None


def foreign_files(out: Path) -> list[str]:
    """Files in `out` that this script would rewrite or delete but did not write: [] when `out` holds the marker or a
    state.json of this dataset (a run from before the marker), or none of OWN_FILES and no wav/*.wav."""
    if (out / MARKER).is_file():
        return []
    try:
        if json.loads((out / "state.json").read_text(encoding="utf-8")).get("repo") == REPO_ID:
            return []
    except (OSError, ValueError, AttributeError):
        pass
    return [n for n in OWN_FILES if (out / n).exists()] + ["wav/*.wav"] * any((out / "wav").glob("*.wav"))


@dataclass(frozen=True)
class Filters:
    modes: tuple[str, ...] = ("reference",)
    max_cer: float = 0.0
    min_utmos: float = 3.7
    gender: str = "any"
    min_dur: float = 1.5
    max_dur: float = 20.0
    max_per_voice: int = 40  # <= 0: no cap
    fineweb_share: float = 0.25  # >= 1: no cap
    # INSTRUCTION_FIELDS: INSTRUCTION_MODE rows only (in settings_key only with that mode)
    styles: tuple[str, ...] = ()  # () = every style
    min_utmos_instruction: float | None = None  # None: instruction_utmos(min_utmos)
    max_style_share: float = 1.0  # >= 1: no cap

    def __post_init__(self) -> None:
        if self.min_utmos_instruction is None:  # the actual threshold, so settings_key holds it
            object.__setattr__(self, "min_utmos_instruction", instruction_utmos(self.min_utmos))


def cheap_reason(row: dict, f: Filters) -> str:
    """First per-row filter the row fails ("" = passes); NaN and None fail every numeric test."""
    if row.get("render_mode") not in f.modes:
        return "render_mode"
    instruction = row.get("render_mode") == INSTRUCTION_MODE
    if instruction and f.styles and row.get("style") not in f.styles:
        return "style"
    cer, utmos, dur = row.get("qc_cer"), row.get("qc_utmos"), row.get("duration")
    if cer is None or not cer <= f.max_cer:
        return "qc_cer"
    if utmos is None or not utmos >= (f.min_utmos_instruction if instruction else f.min_utmos):
        return "qc_utmos"
    if f.gender != "any" and row.get("gender") != f.gender:
        return "gender"
    if dur is None or not dur >= f.min_dur:
        return "too_short"
    if not dur <= f.max_dur:
        return "too_long"
    if not collapse_ws(row.get("text_spoken") or ""):
        return "empty_text"
    return ""


@dataclass
class Guard:
    """Contamination keys: `exact` normalized sentences, `contain` the long ones, padded for word-boundary search."""
    exact: set[str] = field(default_factory=set)
    contain: list[str] = field(default_factory=list)

    def hit(self, *keys: str) -> bool:
        for k in keys:
            padded = f" {k} "
            if k and (k in self.exact or any(c in padded for c in self.contain)):
                return True
        return False

    def digest(self) -> str:
        return hashlib.sha1("\n".join(sorted(self.exact)).encode()).hexdigest()[:12]


def build_guard(texts: Iterable[str]) -> Guard:
    """Each protected text whole (exact match at any length) and its sentences of >= CONTAIN_MIN_WORDS words; keys of
    >= CONTAIN_MIN_WORDS words are also searched inside rows. Shorter sentences ("Evet.") match only a whole row."""
    exact, contain = set(), set()
    for t in texts:
        whole = eval_key(t)
        if whole:
            exact.add(whole)
            if len(whole.split()) >= CONTAIN_MIN_WORDS:
                contain.add(whole)
        for s in split_sentences(t):
            k = eval_key(s)
            if len(k.split()) >= CONTAIN_MIN_WORDS:
                exact.add(k)
                contain.add(k)
    return Guard(exact, sorted(f" {k} " for k in contain))


def protected_texts(path: Path) -> list[str]:
    """Texts of a jsonl manifest / benchmark set: text, spoken, text_spoken, alts of every row; a text in control form
    "(instruction)sentence" (an instruction manifest) also without the prefix, so the sentence itself is protected."""
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        ts = [r[k] for k in ("text", "spoken", "text_spoken") if isinstance(r.get(k), str)]
        ts += [a for a in r.get("alts") or [] if isinstance(a, str)]
        out += ts + [s for t in ts if (s := strip_control(t)) != t]
    return out


@dataclass
class Take:
    picked: list[dict]
    truncated: bool  # the target was reached inside this file: rows left unconsidered
    drops: collections.Counter
    drop_s: collections.Counter
    rows: int
    seconds: float


class Selector:
    """Global, order-dependent selection state: text dedup, per-voice counts, fineweb and style shares, the target."""

    def __init__(self, f: Filters, guard: Guard, target_s: float):
        self.f, self.guard, self.target_s = f, guard, target_s
        self.seen_text: set[str] = set()
        self.seen_id: set[str] = set()
        self.per_voice: collections.Counter = collections.Counter()
        self.style_s: collections.Counter = collections.Counter()
        self.total_s = self.web_s = 0.0

    def done(self) -> bool:
        return self.total_s >= self.target_s

    def style_full(self, style: str | None) -> bool:
        """`style` already holds --max-style-share of the running total. Checked before a row is added (a style's
        first row always passes, a style overshoots by one row at most), so styles of a corpus of instruction rows
        only can grow together; never for a row without a style."""
        held = self.style_s[style] if style is not None else 0.0
        return self.f.max_style_share < 1 and held > 0 and held >= self.f.max_style_share * self.total_s

    def _add(self, clip_id: str, voice: str, keys: tuple[str, ...], seconds: float, web: bool,
             style: str | None = None) -> None:
        self.seen_text.update(k for k in keys if k)
        self.seen_id.add(clip_id)
        self.per_voice[voice] += 1
        self.total_s += seconds
        self.web_s += seconds if web else 0.0
        if style is not None:
            self.style_s[style] += seconds

    def restore(self, clips: Iterable[dict]) -> None:
        """State of a resumed run from the committed clips (src_duration: the metadata seconds selection counted)."""
        for c in clips:
            keys = (eval_key(c["text_spoken"]), eval_key(c.get("text") or ""))
            self._add(c["id"], c["voice_id"], keys, c.get("src_duration", c["duration"]), is_web(c), style_of(c))

    def take(self, rows: list[dict]) -> Take:
        drops, drop_s = collections.Counter(), collections.Counter()

        def drop(why: str, r: dict) -> None:
            drops[why] += 1
            drop_s[why] += r.get("duration") or 0.0

        cands = []
        for r in rows:
            why = cheap_reason(r, self.f)
            if why:
                drop(why, r)
            else:
                cands.append(r)
        cands.sort(key=lambda r: (is_web(r), -r["qc_utmos"], r["id"]))
        picked, truncated = [], False
        for i, r in enumerate(cands):
            if self.done():
                truncated = True
                for rest in cands[i:]:
                    drop("target_reached", rest)
                break
            d, web, style = r["duration"], is_web(r), style_of(r)
            keys = (eval_key(r["text_spoken"]), eval_key(r.get("text") or ""))  # the sentence, never an instruction
            if self.guard.hit(*keys):
                why = "contamination"
            elif r["id"] in self.seen_id:
                why = "duplicate_id"
            elif any(k and k in self.seen_text for k in keys):
                why = "duplicate_text"
            elif 0 < self.f.max_per_voice <= self.per_voice[r["voice_id"]]:
                why = "voice_cap"
            elif web and self.f.fineweb_share < 1 and self.web_s + d > self.f.fineweb_share * (self.total_s + d) + 1e-9:
                why = "fineweb_share"
            elif self.style_full(style):
                why = "style_share"
            else:
                picked.append(r)
                self._add(r["id"], r["voice_id"], keys, d, web, style)
                continue
            drop(why, r)
        return Take(picked, truncated, drops, drop_s, len(rows), sum(r.get("duration") or 0.0 for r in rows))


def clip_record(r: dict, seconds: float, file: str) -> dict:
    return {"id": r["id"], "voice_id": r["voice_id"], "gender": r.get("gender"), "text": collapse_ws(r.get("text")),
            "text_spoken": collapse_ws(r["text_spoken"]), "category": r.get("text_category"),
            "source": r.get("text_source"), "mode": r.get("render_mode"), "style": r.get("style"),
            "instruction": collapse_ws(r.get("instruction")) or None, "duration": round(seconds, 3),
            "src_duration": r["duration"], "qc_cer": r.get("qc_cer"), "qc_utmos": r.get("qc_utmos"),
            "license": r.get("license"), "text_doc_id": r.get("text_doc_id"), "file": file,
            "row_group": r.get("row_group"), "wav": wav_name(r["id"])}


def plain_ref(clip: dict) -> bool:
    """A clip fit to be a ref_audio: not a reference+instruction one (styled audio; Alania rendered those, like every
    row of a voice, from the voice's plain reference)."""
    return clip.get("mode") != INSTRUCTION_MODE


def assign_refs(clips: list[dict], fraction: float, seed: int,
                ref_ok: Callable[[dict], bool] = lambda c: True) -> dict[str, str]:
    """{clip id: ref clip id} for round(fraction * len(clips)) clips (fewer if not enough have a candidate): a ref is
    another clip of the same voice_id of REF_MIN_S-REF_MAX_S s passing ref_ok, never the clip itself. Deterministic for
    a seed; when every clip passes ref_ok the result is the one without it."""
    rng = random.Random(seed)
    clips = sorted(clips, key=lambda c: c["id"])
    refs: dict[str, list[str]] = collections.defaultdict(list)
    for c in clips:
        if REF_MIN_S <= c["duration"] <= REF_MAX_S and ref_ok(c):
            refs[c["voice_id"]].append(c["id"])
    eligible = [c for c in clips if any(x != c["id"] for x in refs[c["voice_id"]])]
    n = min(len(eligible), round(fraction * len(clips)))
    return {c["id"]: rng.choice([x for x in refs[c["voice_id"]] if x != c["id"]])
            for c in sorted(rng.sample(eligible, n), key=lambda c: c["id"])}


def manifest_rows(clips: list[dict], refs: dict[str, str], wav_dir: Path, seed: int,
                  instruction_format: str = "prefix") -> list[dict]:
    """voxcpm_train.jsonl rows (absolute paths, text = manifest_text: text_spoken, "(instruction)" before it on
    instruction clips with the prefix format), shuffled."""
    by_id = {c["id"]: c for c in clips}
    rows = []
    for c in clips:
        r = {"audio": str(wav_dir / c["wav"]), "text": manifest_text(c, instruction_format),
             "duration": float(c["duration"])}
        if c["id"] in refs:
            ref = by_id[refs[c["id"]]]
            r["ref_audio"] = str(wav_dir / ref["wav"])
            r["ref_duration"] = float(ref["duration"])
        rows.append(r)
    random.Random(seed + 1).shuffle(rows)
    return rows


def settings_key(f: Filters, revision: str, row_groups: int, guard: Guard) -> str:
    """Everything that changes which rows a file yields; not --hours / --max-files / --workers / --seed /
    --ref-fraction / --instruction-format. INSTRUCTION_FIELDS only with INSTRUCTION_MODE in f.modes (without it they
    select nothing), so the key of such a run is the one written before they existed and its state stays valid; with
    it the key is new, so a state of an older run with that mode (clips without the instruction) is refused."""
    filters = {k: v for k, v in asdict(f).items() if k not in INSTRUCTION_FIELDS or INSTRUCTION_MODE in f.modes}
    return json.dumps({"filters": filters, "revision": revision, "row_groups": row_groups, "guard": guard.digest()},
                      sort_keys=True)


def resume_point(state: dict, clips: list[dict], target_s: float) -> tuple[list[dict], list[dict]]:
    """(done entries, clips) to continue from: clips of finished files only (a line written before the kill of its
    state update is dropped), in file order; a last file cut by the old --hours is redone when --hours grew."""
    done = list(state.get("done", []))
    by_id = {c["id"]: c for c in clips}

    def collect(entries):
        return [by_id[i] for e in entries for i in e["ids"] if i in by_id]

    kept = collect(done)
    if done and done[-1].get("truncated") and sum(c.get("src_duration", c["duration"]) for c in kept) < target_s:
        done = done[:-1]
        kept = collect(done)
    missing = sum(len(e["ids"]) for e in done) - len(kept)
    if missing:
        raise RuntimeError(f"clips.jsonl lacks {missing} clips that state.json lists; rerun with --restart")
    return done, kept


# --------------------------------------------------------------------------- report
def quantiles(values: list[float]) -> dict:
    v = sorted(values)
    if not v:
        return {}

    def q(p: float) -> float:
        return round(v[min(len(v) - 1, int(p * len(v)))], 3)

    return {"n": len(v), "min": round(v[0], 3), "p10": q(0.10), "median": round(statistics.median(v), 3),
            "p90": q(0.90), "max": round(v[-1], 3), "mean": round(statistics.mean(v), 3)}


def build_report(clips: list[dict], done: list[dict], refs: dict[str, str], settings: dict, guard: Guard,
                 n_files: int, reached: bool, extra: dict | None = None) -> dict:
    hours = lambda cs: round(sum(c["duration"] for c in cs) / 3600, 3)  # noqa: E731
    fmt = settings.get("instruction_format", "prefix")

    def by(key: str, missing: str = "?") -> dict:
        g = collections.defaultdict(list)
        for c in clips:
            g[c.get(key) or missing].append(c)
        return {k: {"clips": len(v), "hours": hours(v)} for k, v in sorted(g.items(), key=lambda kv: -len(kv[1]))}

    drops, drop_s = collections.Counter(), collections.Counter()
    for e in done:
        drops.update(e.get("drops", {}))
        drop_s.update(e.get("drop_s", {}))
    per_voice = collections.Counter(c["voice_id"] for c in clips)
    examples = collections.defaultdict(list)
    for c in clips:
        if len(examples[c.get("category") or "?"]) < 2:
            examples[c.get("category") or "?"].append(c["text_spoken"][:160])
    return {
        "dataset": {"repo": REPO_ID, "config": "cc-by", "revision": settings.get("revision"),
                    "files_in_config": n_files, "attribution": ATTRIBUTION,
                    "attribution_extra": attribution_lines(clips)[1:]},
        "settings": settings,
        "target_hours": settings.get("hours"), "reached": reached,
        "selected": {"clips": len(clips), "hours": hours(clips), "voices": len(per_voice),
                     "clips_per_voice": quantiles(list(per_voice.values())),
                     "duration_s": quantiles([c["duration"] for c in clips]),
                     "qc_utmos": quantiles([c["qc_utmos"] for c in clips if c.get("qc_utmos") is not None]),
                     "by_source": by("source"), "by_gender": by("gender"), "by_category": by("category"),
                     "by_license": by("license"), "by_mode": by("mode"), "by_style": by("style", "none"),
                     "instructions": len({c["instruction"] for c in clips if style_of(c) and c.get("instruction")}),
                     "text_spoken_with_digits": sum(bool(re.search(r"\d", c["text_spoken"])) for c in clips),
                     "duration_mismatch_over_50ms": sum(abs(c["duration"] - c.get("src_duration", c["duration"])) > 0.05
                                                        for c in clips),
                     "examples": dict(examples)},
        "seen": {"files": len(done), "rows": sum(e.get("rows", 0) for e in done),
                 "hours": round(sum(e.get("seconds", 0.0) for e in done) / 3600, 3),
                 "bytes_read": sum(e.get("bytes", 0) for e in done), "file_list": [e["file"] for e in done]},
        "drops": {k: {"rows": n, "hours": round(drop_s[k] / 3600, 3)} for k, n in drops.most_common()},
        "contamination_keys": {"exact": len(guard.exact), "contained": len(guard.contain), "digest": guard.digest()},
        "manifest": {"rows": len(clips), "with_ref_audio": len(refs), "instruction_format": fmt,
                     "with_instruction": sum(manifest_text(c, fmt) != c["text_spoken"] for c in clips)},
        **(extra or {}),
    }


def render_markdown(r: dict) -> str:
    L = []
    a = L.append
    d, s, sel = r["dataset"], r["settings"], r["selected"]
    a("# Alania stage-1 corpus report (generated by `tts/voxcpm/prepare_alania.py`)\n")
    a(f"Dataset `{d['repo']}` config `{d['config']}` @ `{d['revision']}` ({d['files_in_config']} files).  ")
    a(f"Attribution: {d['attribution']}\n")
    for x in d.get("attribution_extra", []):
        a(f"Text: {x}\n")
    a(f"Settings: modes {','.join(s['modes'])}, qc_cer <= {s['max_cer']}, qc_utmos >= {s['min_utmos']}, gender "
      f"{s['gender']}, {s['min_dur']}-{s['max_dur']} s, <= {s['max_per_voice']} clips/voice, fineweb share <= "
      f"{s['fineweb_share']}, ref fraction {s['ref_fraction']}, row groups/file {s['row_groups'] or 'all'}, max files "
      f"{s['max_files'] or 'all'}, seed {s['seed']}; protected: TR-Zor + "
      f"{', '.join(s['exclude_manifest']) or 'no --exclude-manifest'}.\n")
    if INSTRUCTION_MODE in s["modes"]:
        a(f"{INSTRUCTION_MODE} rows: styles {','.join(s.get('styles') or []) or 'all'}, qc_utmos >= "
          f"{s.get('min_utmos_instruction')}, style share <= {s.get('max_style_share')} (+ one clip), instruction "
          f"format {s.get('instruction_format')}.\n")
    status = "reached" if r["reached"] else "**NOT reached** (files used up)"
    m = r["manifest"]
    a(f"**{sel['hours']} h, {sel['clips']} clips, {sel['voices']} voices** (target {r['target_hours']} h: {status}); "
      f"{m['with_ref_audio']} manifest rows with ref_audio, {m.get('with_instruction', 0)} with an instruction prefix"
      f"{' (' + str(sel['instructions']) + ' distinct instructions)' if sel.get('instructions') else ''}.\n")
    sn = r["seen"]
    a(f"Read {sn['files']} files, {sn['rows']} rows ({sn['hours']} h of audio), "
      f"{sn['bytes_read'] / 1e9:.2f} GB from the Hub.\n")
    a(f"Durations (s): {sel['duration_s']}  \nqc_utmos: {sel['qc_utmos']}  \n"
      f"clips per voice: {sel['clips_per_voice']}  ")
    a(f"text_spoken with digits: {sel['text_spoken_with_digits']}; |WAV - metadata duration| > 50 ms: "
      f"{sel['duration_mismatch_over_50ms']}\n")
    for title, key in (("Source", "by_source"), ("Gender", "by_gender"), ("Category", "by_category"),
                       ("License", "by_license"), ("Mode", "by_mode"), ("Style", "by_style")):
        a(f"## {title}\n\n| {title.lower()} | clips | hours |\n|---|---:|---:|")
        for k, v in sel.get(key, {}).items():
            a(f"| {k} | {v['clips']} | {v['hours']} |")
        a("")
    a("## Drops (first failing filter per row)\n\n| reason | rows | hours |\n|---|---:|---:|")
    for k, v in r["drops"].items():
        a(f"| {k} | {v['rows']} | {v['hours']} |")
    ck = r["contamination_keys"]
    a(f"\nContamination keys: {ck['exact']} sentences, {ck['contained']} also searched inside rows "
      f"(digest {ck['digest']}).\n")
    a("## Examples (text_spoken)\n")
    for cat, ex in sel["examples"].items():
        for t in ex:
            a(f"- {cat}: {t}")
    a(f"\nFiles: {' '.join(Path(f).name for f in sn['file_list'])}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- I/O: the Hub, audio, state
class HubSource:
    """The dataset on the Hub, read with HfFileSystem range requests (pyarrow, numpy, soundfile/soxr imported here)."""

    def __init__(self, revision: str):
        from huggingface_hub import HfFileSystem

        self.revision, self.fs = revision, HfFileSystem()

    def _path(self, name: str) -> str:
        return f"datasets/{REPO_ID}@{self.revision}/{name}"

    def list_files(self) -> list[str]:
        from huggingface_hub import HfApi

        tree = HfApi().list_repo_tree(REPO_ID, path_in_repo=CONFIG_DIR, repo_type="dataset", revision=self.revision)
        return sorted(f.path for f in tree if f.path.endswith(".parquet"))

    def read_meta(self, name: str, row_groups: int) -> tuple[list[dict], object, int]:
        """(metadata rows with "row_group", the parquet footer, bytes read) of the first row_groups (0 = all)."""
        import pyarrow.parquet as pq

        with self.fs.open(self._path(name), cache_type="none") as f:
            pf = pq.ParquetFile(f, pre_buffer=True)  # pre_buffer coalesces the small column chunks of a row group
            md = pf.metadata
            n = md.num_row_groups if row_groups <= 0 else min(row_groups, md.num_row_groups)
            table = pf.read_row_groups(list(range(n)), columns=META_COLUMNS)
        rows, rg_of = table.to_pylist(), []
        for i in range(n):
            rg_of += [i] * md.row_group(i).num_rows
        for r, g in zip(rows, rg_of):
            r["row_group"] = g
        return rows, md, _chunk_bytes(md, range(n), META_COLUMNS) + md.serialized_size

    def fetch_audio(self, name: str, md, picked: list[dict], wav_dir: Path) -> tuple[dict[str, float], int]:
        """Writes the WAVs of `picked` (rows with "id", "row_group") that do not exist yet; md: the footer read_meta
        returned (None: read again); ({id: seconds}, bytes read)."""
        import pyarrow.parquet as pq

        need = collections.defaultdict(list)
        for r in picked:
            if not (wav_dir / wav_name(r["id"])).exists():
                need[r["row_group"]].append(r["id"])
        nbytes = 0
        if need:
            with self.fs.open(self._path(name), cache_type="none") as f:
                pf = pq.ParquetFile(f, metadata=md, pre_buffer=True)
                md = pf.metadata
                for g in sorted(need):
                    t = pf.read_row_group(g, columns=AUDIO_COLUMNS)
                    nbytes += _chunk_bytes(md, [g], AUDIO_COLUMNS)
                    index = {x: i for i, x in enumerate(t.column("id").to_pylist())}
                    for clip_id in need[g]:
                        if clip_id not in index:
                            raise RuntimeError(f"{name} row group {g}: id {clip_id} not found")
                        data = t.column("audio")[index[clip_id]]["bytes"].as_py()
                        write_wav(wav_dir / wav_name(clip_id), decode_16k(data), RATE)
        return {r["id"]: wav_seconds(wav_dir / wav_name(r["id"])) for r in picked}, nbytes


def _chunk_bytes(md, groups: Iterable[int], columns: list[str]) -> int:
    total = 0
    for g in groups:
        rg = md.row_group(g)
        for i in range(rg.num_columns):
            c = rg.column(i)
            if c.path_in_schema.split(".")[0] in columns:
                total += c.total_compressed_size
    return total


def decode_16k(data: bytes, rate: int = RATE):
    """Encoded audio bytes (FLAC) -> mono float32 at `rate`."""
    import io

    import numpy as np

    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except ImportError:
        import torchaudio

        wav, sr = torchaudio.load(io.BytesIO(data))
        audio = wav.numpy().T
    audio = audio.mean(axis=1).astype(np.float32)
    if sr != rate:
        audio = resample(audio, sr, rate)
    return np.clip(audio, -1.0, 1.0)


def resample(audio, sr: int, rate: int):
    import numpy as np

    try:
        import soxr

        return soxr.resample(audio, sr, rate, quality="VHQ").astype(np.float32)
    except ImportError:
        pass
    try:
        import torch
        import torchaudio.functional as AF

        return AF.resample(torch.from_numpy(audio), sr, rate).numpy()
    except ImportError:
        from scipy.signal import resample_poly

        g = math.gcd(sr, rate)
        return resample_poly(audio, rate // g, sr // g).astype(np.float32)


def write_wav(path: Path, audio, rate: int) -> None:
    """PCM16 mono, atomically (a killed run never leaves a truncated WAV under the final name)."""
    import numpy as np

    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    tmp = path.with_name(path.name + ".tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a jsonl file; a line cut by a kill is skipped."""
    out = []
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def retry(fn: Callable, *a, what: str = "", tries: int = 4):
    for i in range(tries):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001  (network: retried, then raised)
            if i == tries - 1:
                raise
            wait = 10 * 3 ** i
            print(f"[alania] {what}: {type(e).__name__}: {e}; retry in {wait} s", flush=True)
            time.sleep(wait)


# --------------------------------------------------------------------------- the walk
def walk(source, todo: list[str], sel: Selector, state: dict, out: Path, row_groups: int, workers: int,
         log: Callable[[str], None] = print) -> None:
    """Metadata prefetched in parallel, selection in file order, audio in parallel, commits in file order."""
    wav_dir, t0 = out / "wav", time.time()
    pending, metas, jobs = collections.deque(todo), collections.deque(), collections.deque()
    meta_pool, audio_pool = ThreadPoolExecutor(min(4, workers)), ThreadPoolExecutor(workers)
    committed = [sum(e.get("seconds_selected", 0.0) for e in state["done"])]

    def prefetch() -> None:
        while pending and len(metas) < workers + 2:
            name = pending.popleft()
            metas.append((name, meta_pool.submit(retry, source.read_meta, name, row_groups, what=f"{name} metadata")))

    def commit_one() -> None:
        name, take, meta_bytes, fut = jobs.popleft()
        secs, audio_bytes = fut.result()
        clips = [clip_record(r, secs[r["id"]], name) for r in take.picked]
        with open(out / "clips.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in clips)
        seconds_selected = sum(c["duration"] for c in clips)
        state["done"].append({"file": name, "ids": [c["id"] for c in clips], "truncated": take.truncated,
                              "rows": take.rows, "seconds": round(take.seconds, 3),
                              "seconds_selected": round(seconds_selected, 3), "bytes": meta_bytes + audio_bytes,
                              "drops": dict(take.drops), "drop_s": {k: round(v, 3) for k, v in take.drop_s.items()}})
        write_atomic(out / "state.json", json.dumps(state, ensure_ascii=False, indent=1))
        committed[0] += seconds_selected
        gb = sum(e.get("bytes", 0) for e in state["done"]) / 1e9
        log(f"[alania] {len(state['done'])} files, {Path(name).name}: +{len(clips)} clips "
            f"({seconds_selected / 3600:.2f} h); total {committed[0] / 3600:.2f} / {sel.target_s / 3600:g} h, "
            f"{gb:.2f} GB read, {time.time() - t0:.0f} s")

    try:
        prefetch()
        while metas and not sel.done():
            name, fut = metas.popleft()
            rows, md, meta_bytes = fut.result()
            take = sel.take(rows)
            jobs.append((name, take, meta_bytes, audio_pool.submit(
                retry, source.fetch_audio, name, md, take.picked, wav_dir, what=f"{name} audio")))
            if not sel.done():
                prefetch()
            while jobs and (jobs[0][-1].done() or len(jobs) > 2 * workers):
                commit_one()
        while jobs:
            commit_one()
    finally:
        for _, fut in metas:
            fut.cancel()
        meta_pool.shutdown(wait=False, cancel_futures=True)
        audio_pool.shutdown(wait=True, cancel_futures=True)


def refetch_missing(source, clips: list[dict], wav_dir: Path, workers: int, log: Callable[[str], None] = print) -> int:
    """Audio again for committed clips whose WAV is gone (deleted, or a Volume snapshot without it); the selection
    stays as it is. Returns how many WAVs were missing."""
    absent = collections.defaultdict(list)
    for c in clips:
        if not (wav_dir / c["wav"]).is_file():
            absent[c["file"]].append(c)
    n = sum(map(len, absent.values()))
    if n:
        log(f"[alania] {n} committed clips have no WAV ({len(absent)} files): reading their audio again")
        with ThreadPoolExecutor(workers) as pool:
            futs = [pool.submit(retry, source.fetch_audio, name, None, cs, wav_dir, what=f"{name} audio (missing WAVs)")
                    for name, cs in absent.items()]
            for fut in futs:
                fut.result()
    return n


# --------------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output directory (e.g. $TTS_DATA/alania16k)")
    ap.add_argument("--hours", type=float, default=25.0, help="selected audio to stop at [%(default)s]")
    ap.add_argument("--min-utmos", type=float, default=3.7, help="minimum qc_utmos (dataset median 3.62) [%(default)s]")
    ap.add_argument("--max-cer", type=float, default=0.0, help="maximum qc_cer [%(default)s]")
    ap.add_argument("--gender", choices=["female", "male", "any"], default="any", help="[%(default)s]")
    ap.add_argument("--modes", nargs="+", default=["reference"],
                    help="render_mode values to keep, space or comma separated: reference, reference+instruction, "
                         "voice-design [%(default)s]")
    ap.add_argument("--styles", nargs="+", default=[],
                    help=f"reference+instruction rows of these styles only, space or comma separated: "
                         f"{', '.join(KNOWN_STYLES)} [all]")
    ap.add_argument("--min-utmos-instruction", type=float, default=None,
                    help=f"minimum qc_utmos of reference+instruction rows (instructions cost "
                         f"~{INSTRUCTION_UTMOS_GAP:g}) [--min-utmos - {INSTRUCTION_UTMOS_GAP:g}]")
    ap.add_argument("--max-style-share", type=float, default=1.0,
                    help="hours of one style at most this share of the total (checked before each row: one clip "
                         "of overshoot at most); 1 = no cap [%(default)s]")
    ap.add_argument("--instruction-format", choices=["prefix", "drop"], default="prefix",
                    help="manifest text of reference+instruction rows: prefix = VoxCPM2's \"(instruction)text\", "
                         "drop = text only [%(default)s]")
    ap.add_argument("--min-dur", type=float, default=1.5, help="[%(default)s] s")
    ap.add_argument("--max-dur", type=float, default=20.0, help="[%(default)s] s")
    ap.add_argument("--max-per-voice", type=int, default=40, help="clips per voice_id; 0 = no cap [%(default)s]")
    ap.add_argument("--fineweb-share", type=float, default=0.25,
                    help="non-domain (fineweb2) hours at most this share of the total; 1 = no cap [%(default)s]")
    ap.add_argument("--ref-fraction", type=float, default=0.3, help="manifest rows with a ref_audio [%(default)s]")
    ap.add_argument("--exclude-manifest", type=Path, nargs="+", action="extend", default=[],
                    help="jsonl files whose texts must never be trained on (Antalia's voxcpm_val.jsonl); TR-Zor always")
    ap.add_argument("--revision", default=REVISION, help="dataset commit [%(default)s]")
    ap.add_argument("--workers", type=int, default=8, help="parallel audio downloads/decoders [%(default)s]")
    ap.add_argument("--seed", type=int, default=1234, help="ref_audio choice and manifest order [%(default)s]")
    ap.add_argument("--max-files", type=int, default=0, help="walk at most N files of the spread order; 0 = all")
    ap.add_argument("--row-groups", type=int, default=0, help="first N row groups of each file only; 0 = all")
    ap.add_argument("--restart", action="store_true", help="discard state.json and clips.jsonl (WAVs are reused)")
    args = ap.parse_args(argv)
    args.modes = sorted({m.strip() for x in args.modes for m in x.split(",") if m.strip()})
    args.styles = sorted({s.strip() for x in args.styles for s in x.split(",") if s.strip()})
    if args.min_utmos_instruction is None:  # resolved here: settings_key and the report hold the actual threshold
        args.min_utmos_instruction = instruction_utmos(args.min_utmos)
    for name, ok in (("--hours", args.hours > 0), ("--fineweb-share", 0 <= args.fineweb_share <= 1),
                     ("--ref-fraction", 0 <= args.ref_fraction <= 1), ("--workers", args.workers >= 1),
                     ("--min-dur/--max-dur", 0 <= args.min_dur <= args.max_dur), ("--modes", bool(args.modes)),
                     ("--max-style-share", 0 < args.max_style_share <= 1)):
        if not ok:
            ap.error(f"bad {name}")
    if unknown := sorted(set(args.styles) - set(KNOWN_STYLES)):
        ap.error(f"--styles: unknown {', '.join(unknown)} (known: {', '.join(KNOWN_STYLES)})")
    if (args.styles or args.max_style_share < 1) and INSTRUCTION_MODE not in args.modes:
        ap.error(f"--styles / --max-style-share act on {INSTRUCTION_MODE} rows only: add it to --modes")
    n_styles = len(args.styles or KNOWN_STYLES)
    if args.modes == [INSTRUCTION_MODE] and args.max_style_share < 1 and args.max_style_share * n_styles <= 1:
        ap.error(f"--max-style-share {args.max_style_share:g} x {n_styles} styles <= 1: with {INSTRUCTION_MODE} rows "
                 "only every style would sit at its cap and the selection stop")
    return args


def main(argv=None, source=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    out = args.out.resolve()
    missing = [str(p) for p in args.exclude_manifest if not p.is_file()]
    if missing:
        print(f"[alania] --exclude-manifest not found: {missing} (the contamination guard needs them)", flush=True)
        return 2
    inside = [str(p) for p in args.exclude_manifest if out in p.resolve().parents]
    if inside:
        print(f"[alania] --exclude-manifest {inside} lies inside --out {out}: that is another corpus; use another "
              "--out", flush=True)
        return 2
    if foreign := foreign_files(out):
        print(f"[alania] --out {out} holds {', '.join(foreign)} but no {MARKER} or state.json of this script: another "
              "corpus would be overwritten and its WAVs deleted; use an empty or new directory", flush=True)
        return 2
    (out / "wav").mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text(f"{REPO_ID}: written by tts/voxcpm/prepare_alania.py\n", encoding="utf-8")
    for p in [*(out / "wav").glob("*.wav.tmp"), *(out / f"{n}.part" for n in OWN_FILES)]:  # a killed run's half files
        p.unlink(missing_ok=True)
    texts = protected_texts(TRZOR) + [t for p in args.exclude_manifest for t in protected_texts(p)]
    guard = build_guard(texts)
    filters = Filters(modes=tuple(args.modes), max_cer=args.max_cer, min_utmos=args.min_utmos, gender=args.gender,
                      min_dur=args.min_dur, max_dur=args.max_dur, max_per_voice=args.max_per_voice,
                      fineweb_share=args.fineweb_share, styles=tuple(args.styles),
                      min_utmos_instruction=args.min_utmos_instruction, max_style_share=args.max_style_share)
    key = settings_key(filters, args.revision, args.row_groups, guard)
    state_path, clips_path = out / "state.json", out / "clips.jsonl"
    if (was := state_modes(state_path)) is not None and was != list(filters.modes):  # checked before --restart acts
        print(f"[alania] {out} holds a corpus of --modes {','.join(was)}: one of --modes {','.join(filters.modes)} "
              "goes into its own --out (--restart would replace this one, and an experiment trained on it would "
              "resume on other data: train_lora.py compares corpora by folder name)", flush=True)
        return 2
    state =json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() and not args.restart else None
    if state and state.get("key") != key:
        print(f"[alania] {state_path} was written with other settings:\n  {state.get('key')}\nnow:\n  {key}\n"
              "use the same settings, another --out, or --restart (keeps the WAVs)", flush=True)
        return 2
    source = source or HubSource(args.revision)
    if not state:
        state = {"version": STATE_VERSION, "repo": REPO_ID, "key": key, "files": source.list_files(), "done": []}
        clips_path.unlink(missing_ok=True)
    target_s = args.hours * 3600
    done, clips = resume_point(state, read_jsonl(clips_path), target_s)
    state["done"] = done
    write_atomic(clips_path, "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in clips))
    write_atomic(state_path, json.dumps(state, ensure_ascii=False, indent=1))
    files = state["files"]
    order = [files[i] for i in spread_order(len(files))]
    if args.max_files > 0:
        order = order[:args.max_files]
    finished = {e["file"] for e in done}
    todo = [f for f in order if f not in finished]
    sel = Selector(filters, guard, target_s)
    sel.restore(clips)
    print(f"[alania] {REPO_ID}@{args.revision[:10]}: {len(files)} files; resuming with {len(done)} done "
          f"({sel.total_s / 3600:.2f} h); target {args.hours:g} h; {len(todo)} files left to walk; "
          f"{len(guard.exact)} protected sentences", flush=True)
    refetched = refetch_missing(source, clips, out / "wav", args.workers)
    if not sel.done() and todo:
        walk(source, todo, sel, state, out, args.row_groups, args.workers)
    return finalize(args, out, state, guard, t0, refetched)


def finalize(args, out: Path, state: dict, guard: Guard, t0: float, refetched: int = 0) -> int:
    done = state["done"]
    _, clips = resume_point(state, read_jsonl(out / "clips.jsonl"), 0.0)
    wav_dir = out / "wav"
    absent = [c["wav"] for c in clips if not (wav_dir / c["wav"]).is_file()]
    if absent:  # not expected: walk writes a file's WAVs before its commit, main refetched older ones
        raise RuntimeError(f"{len(absent)} selected clips have no WAV, e.g. {absent[:3]}; rerun: the next call reads "
                           "the audio of committed clips without a WAV again")
    keep = {c["wav"] for c in clips}
    orphans = [p for p in wav_dir.glob("*.wav") if p.name not in keep] if (out / MARKER).is_file() else []
    for p in orphans:
        p.unlink()
    refs = assign_refs(clips, args.ref_fraction, args.seed, plain_ref)
    rows = manifest_rows(clips, refs, wav_dir, args.seed, args.instruction_format)
    write_atomic(out / "voxcpm_train.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_atomic(out / "ATTRIBUTION.txt", "".join(x + "\n" for x in attribution_lines(clips)))
    settings = {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if k == "exclude_manifest" else v)
                for k, v in vars(args).items()}
    total_s = sum(c.get("src_duration", c["duration"]) for c in clips)
    reached = total_s >= args.hours * 3600
    rep = build_report(clips, done, refs, settings, guard, len(state["files"]), reached,
                       {"orphan_wavs_removed": len(orphans), "missing_wavs_refetched": refetched,
                        "elapsed_s": round(time.time() - t0, 1),
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    write_atomic(out / "prep_report.json", json.dumps(rep, ensure_ascii=False, indent=1))
    write_atomic(out / "PREP_REPORT.md", render_markdown(rep))
    sel = rep["selected"]
    print(f"[alania] {sel['hours']} h, {sel['clips']} clips, {sel['voices']} voices, {len(refs)} with ref_audio, "
          f"{rep['manifest']['with_instruction']} with an instruction prefix "
          f"(target {args.hours:g} h {'reached' if reached else 'NOT reached: files used up'}); "
          f"{len(orphans)} orphan WAVs removed -> {out}", flush=True)
    return 0 if clips else 1


if __name__ == "__main__":
    raise SystemExit(main())
