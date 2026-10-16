#!/usr/bin/env python3
"""Voice contest (docs/SES_PLANI.md step 1): candidate voices for our next model, all rendered by VoxCPM2 on one text
set, scored here, then blind-listened.

    python tts/voxcpm/voice_contest.py screen --out /data/contest/r1/screen [--seeds 0,1 --top 30 --keep 5]
    python tts/voxcpm/voice_contest.py render --out /data/contest/r1 [--spec entries.json] [--timesteps 10]
    python tts/voxcpm/voice_contest.py render --out /data/contest/r1-final --screen r1 --timesteps 16 --only a,b
    python tts/voxcpm/voice_contest.py render --out /data/contest/onset-v1 --spec tts/voxcpm/verify_onset.json \
        --ids sira_01,soru_01,duz_eslesik_01,mi_soru_01,emo_angry_01,emo_empathetic_01 --seeds 0,1,2,3 --no-score
    python tts/voxcpm/voice_contest.py score voxcpm-contest/r1 [--rescore] [--device cpu]
    (on Modal: modal_app.py --action contest-screen / contest-render / contest-fetch / contest-refs-push)

screen  designed voices. Each description of --descriptions (jsonl: id, gender, text = English voice description) x
        each --seeds speaks SCREEN_TEXT by voice design: "(<description>)" + trnorm(text), no reference ->
        cand/wav/<id>-s<seed>.wav (requested with seed text_seed(--seed + SEED_STRIDE x seed, "cand", <id>): no two
        candidates share one), scored with Whisper (CER, p(tr)) and UTMOS. The --top best by UTMOS (CER <=
        --max-cand-cer, p(tr) >= --min-p-tr, a sane length, --per-desc per description) each clone CLONE_TEXTS by
        reference-only cloning (reference_wav_path = the candidate clip) -> clone/wav/<cand>/c<k>.wav, scored with
        Whisper, UTMOS and speaker similarity to the candidate. Voices rank by the MEAN clone UTMOS (picking on one
        clip misleads: Alania voices whose first clip scored >= 4.1 averaged 3.88), eligible with CER 0 on >=
        --need-cer0 clones, mean similarity >= --min-sim and mean p(tr) >= --min-p-tr; the --keep best (at least
        --min-per-gender of each gender where there are any) -> entries.json (a render spec), ranking.json, ranking.md.
        The clone test starts only once every candidate is made and scored (a clip whose synthesis failed in
        SYNTH_TRIES calls is left out), so its top is the whole pool's; entries.json is written only when every clone
        is made and scored too (else ranking.* say "provisional" and an earlier entries.json is removed). A call cut
        short ends with status phase "candidates incomplete" / "clone test incomplete": the same command continues.
render  entries x the contest text set (--texts) -> <out>/<entry>/wav/<set>/<id>.wav, rows.jsonl (one row per clip:
        text, prompt, duration, hyp, errors, char_errors, lang, p_tr, utmos, sim, self_sim, prosody, intonation,
        onset_bad / onset_why / onset_retries / onset_trimmed), summary.json, entry.json (spec + settings), ref.<ext>
        (a copy of the reference, with its sources.json); then <out>/leaderboard.md and contest_summary.json over every
        entry of the run. One entry dir holds clips of one reference, LoRA and settings (timesteps, cfg, seed,
        seed_mode, loudness, ref_prep, pause_clean): other settings are refused, so finals at 16 timesteps go to a new
        --run, with --screen <run> for that run's designed winners (round 1's dirs, made before seed_mode / loudness /
        ref_prep, are refused too). --ids a,b renders only those texts; --seeds 0,1,2,3 each of them once per seed, as
        <id>-s<seed> with that seed as it is (a seed sweep; e.g. tts/voxcpm/verify_onset.json: the stray-onset check).
        --pause-clean: every clip through pause_clean.py after synthesis (below).
score   rescores existing dirs (a render run, one entry, or a screen dir) in place: the metrics a clip lacks, or all of
        them with --rescore; a screen dir is ranked again.

Entry spec (JSON, render --spec; default DEFAULT_SPEC): {"entries": [...], "refs": "auto" | [folder, ...] | false,
"screen": run | [run, ...], "arms": [{"lora": dir, "suffix": "+s1", "entries": [name, ...]}]}. An entry: name (folder
name), kind designed | ref | lora, ref (designed / ref: the voice to clone: a clip, a folder whose clip has "neutral" in
its name or is its only clip, or manifest:<file>#<row> = that row's "audio"; relative = under --refs-root), lora (a
checkpoint folder, lora_weights_path; relative = under /data/exp; any kind may have one: kind lora needs one, its ref is
optional), sim_ref (the clip speaker similarity compares with when there is no ref), notes, description, gender, source,
ref_prep (true / false: the reference prepared or raw, over the call's --no-refprep). prompt_wav / prompt_text are
refused: continuation ("ultimate") cloning carries the speaker's accent (VoxCPM issue #321).
"refs" adds one ref entry per folder of --refs-root whose sources.json (licence and consent record; uploaded by
contest-refs-push from voxcpm-refs/<name>/) names a license, license_url, consent, consent_source, attribution and a
reference-only "use" (check_sources; the record goes along as the entry's "source", into its summary and the
leaderboard): <name> for its neutral clip, <name>-<stem> for any other. "screen" adds the winners of those screen
runs (default: this run's, when it has any; render --screen replaces it). "arms" adds a copy of each named entry with
that LoRA, named <name><suffix> (the stage-1 LoRA as a second arm). Explicit entries win over discovered ones. A spec
without "entries" has DEFAULT_SPEC's (the two controls), so {"refs": "auto", "arms": [...]} keeps them.

Texts (--texts, benchmarks/tts/sets/contest.jsonl): id (unique), set (trzor | soru | emo | ...), text (written form),
category or emotion, refs (spoken forms for WER; else spoken (+ alts); else trnorm(text)), instruction (emo: English,
e.g. "Angry tone, voice getting louder."), soru rows: pattern, peak, peak_word (+ pair_id), as tr_soru.jsonl, seed
(optional integer: that row's seed in every voice). The model gets compose(instruction, trnorm.normalize(text)) =
"(instruction)" + spoken form with VoxCPM's normalizer off (normalize=False), exactly as synth_eval.py.

Seeds (settings "seed_mode": "per_text"): a clip is requested with its row's seed, else text_seed = --seed +
crc32("<set>/<id>"): the same for a sentence in every voice, another for each sentence. Round 1 gave every clip seed 0,
and Nano-vLLM derives each step's noise from (seed, step), so every clip shared one first-step noise (designed
candidates: 6 of 12 with a stray onset at seed 0, none of 18 at seed 1).

References and level (refprep.py; settings "ref_prep", "loudness"): every reference goes to both engines as
refprep.prep_ref makes it, 16 kHz mono, padded with zeros on the RIGHT to whole 160 ms patches (Nano-vLLM pads on the
left, the official API and training on the right) and its active speech at -20 dBFS (--no-refprep or an entry's
"ref_prep": false: the clip as it is); every clip is brought to -20 dBFS active, peak <= -1 dBFS
(refprep.normalize_output) before it is written: the output level follows the reference's (r = 0.93 in round 1).
render --pause-clean (default off): every clip goes through pause_clean.clean (breath noise in pauses turned down,
speech frames untouched, deterministic, CPU) right after synthesis, before that level, its onset check and every score
(clean_pauses; Nano-vLLM: each try); settings "pause_clean" (pause_clean.SETTINGS, only when on: entries made without
it keep their key, and the two never share an entry dir), rows "pause_clean" {pauses, breaths, gated_s}, summary.json
"pause_clean" (pause_clean.rows_summary).

Onsets (onset.py): a clip of a text starting with a vowel, a voiced consonant or y is checked for a stray vowel, fry,
burst or unvoiced lead before its first word. On Nano-vLLM a flagged clip is made again with seed + k (k = 1 ..
NANO_TRIES); still flagged, it is cut where the word starts (30 ms before it, 10 ms fade-in) and logged. The official
API's clips are checked, not made again. Rows: onset_bad (its first try flagged; None: text not covered),
onset_why, onset_retries, onset_trimmed; summary.json "onset" and the leaderboard's "onset stray" give the rate;
the CPU pass checks clips made without it (e.g. round 1's, on score).

Synthesis, --backend nanovllm (the default; NanoPool): Nano-vLLM-VoxCPM (nanovllm_base.py, as nanovllm_app.py's probe
measured it: ~50 audio s per GPU s at 10 timesteps with 64 requests in flight, the official API's quality). The jobs are
grouped by (LoRA, timesteps); each group gets one server, which takes all its requests (--concurrency in flight) and is
stopped, its GPU memory freed, before the next group and before any scorer loads. A request is the official API's
call: the same prompt (voice design "(description)" + text without a reference; reference-only cloning, the clip as
ref_audio_latents, never prompt continuation; "(instruction)" + text on emotional rows), cfg and seed (Nano-vLLM seeds
every request on its own, so two seeds are two independent requests; a seed gives another voice than under voxcpm), the
official length cap and bad-case retry, then the onset retry (above). A LoRA Nano-vLLM cannot load
(nanovllm_base.nano_lora_problem, or its server refusing it) goes to the voxcpm backend for those entries only
(entry.json "fallback"; later calls keep it).
--backend voxcpm (SynthPool): --workers processes on the one GPU, each loading VoxCPM2 once (official API:
VoxCPM.from_pretrained, generate(text, reference_wav_path, cfg_value, inference_timesteps, normalize=False, seed)); jobs
come from one queue, whole entries in turn, the longest first. When any job of the call needs a LoRA every worker loads
that LoRA's structure once: base jobs run with set_lora_enabled(False) (scale 0: the base model's output), another LoRA
of the same shape is swapped in with load_lora. At most --load-parallel workers load at a time (RAM): the parent hands
out load tokens and takes back the one of a worker that dies while loading. The pool is given up (workers killed,
scoring goes on) when no worker is ready after READY_S or no clip finishes for STALL_S. Both backends write the same
files: every finished clip atomically, at the same path; a later call skips it (resume), and a clip whose prompt changed
is made again. The TTS engines are gone before the final scoring (a voxcpm screen keeps its workers between its two
rounds while the GPU has room). Throughput (audio s per wall s; voxcpm: workers, nanovllm: per server) goes to
--throughput-log, which the cost estimates of modal_app use.

Scores: Whisper large-v3-turbo (faster-whisper, the judge of synth_eval: language tr, beam 5) -> hyp; WER and CER
(Levenshtein on trnorm.normalize_for_eval forms, the closest ref, as synth_eval) and, from Whisper's own language-ID
(detect_language, not forced), lang + p_tr = P(Turkish); UTMOS22 strong (score_utmos.py); speaker similarity = cosine
of SpeechBrain ECAPA-TDNN embeddings (speechbrain/spkrec-ecapa-voxceleb, Apache-2.0) to the entry's reference
(sim), and to the centroid of the entry's own clips (self_sim: stability); question intonation (score_intonation.py)
on the soru rows; prosody (F0 median and range in st re 100 Hz, syllables per second of speech, RMS dB) on every clip,
reported for each emotion as the difference to the entry's neutral (trzor) median, with the share of the rough
expected directions (EXPECTED) it meets.

--deadline (epoch seconds, set by modal_app): no job starts that could not finish with time left to score (nanovllm:
no server starts for clips it could not make by then, nano_fit), at the deadline synthesis is cut off (workers and
servers killed), and scoring stops there (no scorer is loaded, and the CPU pass not started, without time to use it);
the summaries are written in any case; what is left is done by the same command later. Importing this module needs
no torch, numpy, voxcpm or Nano-vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zlib
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2]
for _p in (str(REPO / "src"), str(_HERE.parent)):  # also when imported by path (modal_app, tests)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nanovllm_base import (MAX_MODEL_LEN, nano_lora_kwargs, nano_lora_problem, nano_server,  # noqa: E402  (no torch)
                           nano_tokens)
from pause_clean import SETTINGS as PAUSE_CLEAN, rows_summary as pause_summary  # noqa: E402  (no numpy)
from refprep import LOUDNESS, REF_PREP  # noqa: E402  (no numpy)
from score_utmos import utmos_summary, write_atomic  # noqa: E402  (no torch)
from trnorm import normalize, normalize_for_eval  # noqa: E402

DATA = os.environ.get("TTS_DATA", "/data")
# manifests written with Path.resolve() in a container hold the Volume's mount path, not /data
_MODAL_VOLUME = re.compile(r"^/__modal/volumes/[^/]+/")
CONTEST = f"{DATA}/contest"
REFS = f"{CONTEST}/refs"
EXP = f"{DATA}/exp"
MODEL_DIR = f"{DATA}/models/VoxCPM2"
S1_LORA = f"{EXP}/voxcpm2-alania-s1/ckpt/step_0002000"
S2_LORA = f"{EXP}/voxcpm2-alania-s2/ckpt/step_0000600"
ANTALIA_REF = f"manifest:{DATA}/antalia16k/voxcpm_train.jsonl#0"  # the clip evaluate()'s "base" clones
THROUGHPUT_LOG = f"{CONTEST}/throughput.jsonl"
TEXTS = Path("benchmarks/tts/sets/contest.jsonl")  # relative = under the repo
DESCRIPTIONS = Path("tts/voxcpm/voice_descriptions.jsonl")
WHISPER = "large-v3-turbo"
ECAPA_REPO, ECAPA_REV = "speechbrain/spkrec-ecapa-voxceleb", "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
ECAPA_DIR = f"{DATA}/models/spkrec-ecapa-voxceleb"
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")
KINDS = ("designed", "ref", "lora")
ENTRY_KEYS = ("name", "kind", "ref", "lora", "sim_ref", "notes", "description", "gender", "source", "ref_prep")
CONTINUATION_KEYS = {"prompt_wav", "prompt_wav_path", "prompt_text"}
SPEC_KEYS = {"entries", "refs", "screen", "arms"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")  # entry folder names
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")  # text and description ids (file names)
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")  # contest run names (modal_app.contest_out)
RESERVED = {"screen", "refs", "entries.json", "status.json", "leaderboard.md", "contest_summary.json"}  # run dir
WRAPPER_OPTS = ("--out", "--model", "--deadline", "--spec", "--throughput-log", "--backend", "--refs-root")  # modal_app
MAX_WORKERS, MAX_LOAD_PARALLEL = 8, 4
BACKENDS = ("nanovllm", "voxcpm", "fake")
P_TR_OK = 0.95  # docs/SES_PLANI.md: Turkish language-ID >= 0.95
# a reference folder's sources.json (fetch_refs.py) must name these (non-empty), "use" saying reference-only cloning
SOURCE_KEYS = ("license", "license_url", "consent", "consent_source", "attribution", "use")
REF_ONLY_RE = re.compile(r"reference[- _]only", re.I)
SYNTH_TRIES = 2  # calls a clip may fail to synthesize before a screen goes on without it
SEED_MODE = "per_text"  # settings "seed_mode": text_seed (round 1 gave every clip --seed: one first-step noise)
SEED_STRIDE = 1000  # between a screen description's candidate seeds: their retries (seed + k) never meet

SCREEN_TEXT = ("Merhaba, bugün size şehrimizin en sevilen parklarından birini anlatmak istiyorum; ağaçların arasında "
               "yürürken kuşları dinleyebilir, gölün kenarında biraz dinlenebilirsiniz.")
CLONE_TEXTS = (
    "Yarın sabah dokuzda başlayacak toplantıya bütün ekip katılacak, gündemde yeni projenin bütçesi var.",
    "Kargonuz bugün öğleden sonra dağıtıma çıktı; teslimat sırasında kimliğinizi yanınızda bulundurun.",
    "Geçen hafta sonu ailecek Kapadokya'ya gittik ve balonları izlemek için sabahın erken saatlerinde kalktık.",
    "Bu ilacı günde iki kez, yemeklerden sonra bir bardak suyla almanız ve tedaviyi yarıda bırakmamanız önemli.",
)

DEFAULT_SPEC = {
    "entries": [
        {"name": "galaktik-s2", "kind": "lora", "lora": S2_LORA, "sim_ref": ANTALIA_REF,
         "notes": "control: the then-released model (Alania 25 h -> Antalia LoRA, step 600), no reference"},
        {"name": "base-antalia", "kind": "ref", "ref": ANTALIA_REF,
         "notes": "control: base VoxCPM2 cloning an Antalia training clip (evaluate()'s base)"},
    ],
    "refs": "auto",
}

EMOTION_ALIASES = {"anger": "angry", "mad": "angry", "happy": "cheerful", "excited": "cheerful", "joyful": "cheerful",
                   "joy": "cheerful", "sadness": "sad", "surprise": "surprised", "empathy": "empathetic",
                   "compassionate": "empathetic", "neutral": "neutral", "calm": "calm"}
# rough expected direction of each prosody measure re the entry's neutral reading (Banse & Scherer 1996 tendencies)
EXPECTED = {"angry": {"energy_db": 1, "f0_median_st": 1}, "cheerful": {"f0_median_st": 1, "f0_range_st": 1},
            "sad": {"f0_median_st": -1, "rate_syl_s": -1, "energy_db": -1}, "surprised": {"f0_range_st": 1,
            "f0_median_st": 1}, "calm": {"rate_syl_s": -1, "f0_range_st": -1}, "empathetic": {"rate_syl_s": -1,
            "energy_db": -1}}
PROSODY = ("f0_median_st", "f0_range_st", "rate_syl_s", "energy_db")
MIN_VOICED = 20  # voiced 10 ms frames a clip needs for its F0 measures

# cost model (A100-80GB, official API, one sentence per call): VoxCPM2 speaks ~16 chars/s of the spoken form (TR-Zor);
# one process runs at RTF 0.87 at 10 timesteps and 1.4-2.3 at 16 (two containers); what several processes on one GPU
# gain is a guess until throughput.jsonl has a measurement
CHARS_PER_S = 15.0
RTF_10, RTF_16 = 0.9, 2.0
WORKERS_UP_S = 90.0  # imports + the first worker's VoxCPM2 load
READY_S = 2 * WORKERS_UP_S + 120  # no worker ready by then: the pool is given up (a load that hangs)
STALL_S = 600.0  # jobs out and no clip finished for this long (ready workers): the pool is given up
LOAD_S = {"whisper": 60.0, "utmos": 30.0, "ecapa": 30.0}  # scorer loads from the Volume
SCORE_LOAD_S = sum(LOAD_S.values())
SCORE_CLIP_S = 0.7  # Whisper (transcribe + language-ID) + UTMOS + ECAPA per clip; prosody runs beside them on CPU
CPU_MIN_S = 60.0  # the prosody / intonation pool starts only with this much time left
SUMMARY_S = 30.0
# host RAM (GiB, estimates until throughput.jsonl has host_mem_peak_mib): a loaded worker, one VoxCPM2 load at its peak
# (fp32 build + weights), this process and the OS
WORKER_RAM_GB, LOAD_RAM_GB, BASE_RAM_GB = 3.0, 13.0, 2.0
# Nano-vLLM (--backend nanovllm), measured by nanovllm_app.py's probe-20261011-110126 (A100-80GB, 64 requests in flight,
# 128 contest-like requests per scenario): audio s per GPU s, timesteps -> (no reference, reference); its server was
# ready in 40-56 s with ~65 GiB of the card in use (gpu_memory_utilization 0.85: weights + KV cache)
NANO_SPEED = {10: (53.0, 49.0), 16: (38.0, 36.0)}
NANO_REQ_RTF = {10: 1.05, 16: 1.4}  # a request's latency per audio s at 64 in flight: how long a last batch takes
NANO_DUR = 1.1  # its clips ran 7 % (reference) to 20 % (none) longer than the official API's: estimates of its audio
NANO_START_S, NANO_STOP_S = 60.0, 10.0  # per server: start (+ tokenizer, LoRA, reference latents); stop (GPU free)
NANO_READY_S = 300.0  # a server not ready by then is given up
NANO_MIN_JOBS = 2  # x concurrency: a measured server (throughput.jsonl) counts for estimates from this many requests
NANO_CONCURRENCY = 64  # requests in flight per server (its max_num_seqs): the probe's best
# the official API's length cap and bad-case retry (VoxCPM2.generate: a clip of >= 6 x text tokens steps is made again
# with seed + 1, 3 tries in all; each try capped at 6 x tokens + 10 steps, at most 4096); Nano-vLLM has none of its own
NANO_TRIES, NANO_BAD_RATIO, NANO_MAX_STEPS = 3, 6.0, 4096


# ---- small helpers ------------------------------------------------------------------------------------------------


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def _abs(p: str) -> bool:
    return p.startswith("/") or Path(p).is_absolute()


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_rows(p: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    except OSError:
        return []


def write_rows(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(p, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(p, json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")  # Paths -> str


def has_wav(p: Path) -> bool:
    try:
        return p.stat().st_size > 44  # more than a WAV header
    except OSError:
        return False


def wav_duration(p: Path) -> float | None:
    import wave

    try:
        with wave.open(str(p), "rb") as w:
            return round(w.getnframes() / w.getframerate(), 3)
    except (OSError, EOFError, wave.Error):
        return None


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def median(xs) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def parse_ints(s: str) -> list[int]:
    try:
        out = list(dict.fromkeys(int(x) for x in s.split(",") if x.strip()))
    except ValueError:
        raise ValueError(f"want comma-separated integers, got {s!r}") from None
    if not out:
        raise ValueError("no values")
    return out


def strip_instruction(s: str | None) -> str | None:
    """An instruction / voice description without the parentheses compose() adds; ValueError when it has its own."""
    if s is None or not str(s).strip():
        return None
    s = str(s).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if "(" in s or ")" in s:
        raise ValueError(f"instruction {s!r}: parentheses inside would end VoxCPM's instruction early")
    return s or None


def compose(instruction: str | None, model_text: str) -> str:
    """What generate() gets: "(instruction)" + the spoken form (VoxCPM2 voice design / style control), or the form."""
    return f"({instruction}){model_text}" if instruction else model_text


# ---- error rates (pure Python: the same counts as synth_eval's jiwer, no jiwer needed) ----------------------------


def edit_distance(a, b) -> int:
    """Levenshtein distance of two sequences (substitutions + deletions + insertions)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer_cer(refs: list[str], hyp: str) -> tuple[int, int, int, int]:
    """(word errors, ref words, char errors, ref chars) against the ref closest in WER, after normalize_for_eval on
    both sides (synth_eval.wer_parts: jiwer's S + D + I is this word-level Levenshtein distance)."""
    h = normalize_for_eval(hyp or "")
    best = None
    for ref in refs:
        r = normalize_for_eval(ref)
        e, n = edit_distance(r.split(), h.split()), len(r.split())
        if best is None or e / max(n, 1) < best[0] / max(best[1], 1):
            best = (e, n, r)
    if best is None or not best[2]:
        return 0, 0, 0, 0
    return best[0], best[1], edit_distance(best[2], h), len(best[2])


# ---- entry specs --------------------------------------------------------------------------------------------------


def parse_entries(spec) -> list[dict]:
    """Validated entries of a spec ({"entries": [...]} or the list itself); ValueError names the first problem."""
    items = spec.get("entries", []) if isinstance(spec, dict) else spec
    if not isinstance(items, list):
        raise ValueError('an entry spec is a list of entries or {"entries": [...]}')
    out, seen = [], set()
    for i, e in enumerate(items):
        where = f"entry {i + 1}" + (f" ({e.get('name')})" if isinstance(e, dict) and e.get("name") else "")
        if not isinstance(e, dict):
            raise ValueError(f"{where}: not an object")
        if bad := CONTINUATION_KEYS & set(e):
            raise ValueError(f"{where}: {', '.join(sorted(bad))}: continuation cloning (prompt_wav_path + prompt_text) "
                             "carries the speaker's accent; the contest clones with reference_wav_path only (ref)")
        if bad := set(e) - set(ENTRY_KEYS):
            raise ValueError(f"{where}: unknown keys {sorted(bad)} (known: {', '.join(ENTRY_KEYS)})")
        name, kind = e.get("name"), e.get("kind")
        if not isinstance(name, str) or not NAME_RE.match(name) or name in RESERVED:
            raise ValueError(f"{where}: name must match {NAME_RE.pattern} and not be one of {sorted(RESERVED)}")
        if name in seen:
            raise ValueError(f"{where}: duplicate name")
        if kind not in KINDS:
            raise ValueError(f"{where}: kind must be one of {', '.join(KINDS)}")
        for k in ("ref", "lora", "sim_ref", "notes", "description", "gender"):
            if e.get(k) is not None and not (isinstance(e[k], str) and e[k].strip()):
                raise ValueError(f"{where}: {k} must be a non-empty string")
        if e.get("ref_prep") is not None and not isinstance(e["ref_prep"], bool):
            raise ValueError(f"{where}: ref_prep must be true or false")
        if kind in ("designed", "ref") and not e.get("ref"):
            raise ValueError(f"{where}: kind {kind} clones a reference: give ref (a clip, a folder or "
                             "manifest:<file>#<row>)")
        if kind == "lora" and not e.get("lora"):
            raise ValueError(f"{where}: kind lora needs lora (a checkpoint folder)")
        seen.add(name)
        out.append({k: e[k] for k in ENTRY_KEYS if e.get(k) is not None})
    return out


def check_sources(src) -> list[str]:
    """What a reference folder's parsed sources.json lacks ([]: usable): a non-empty string for each SOURCE_KEYS, and a
    "use" that says reference-only cloning."""
    if not isinstance(src, dict):
        return ["readable JSON object"]
    bad = [k for k in SOURCE_KEYS if not (isinstance(src.get(k), str) and src[k].strip())]
    if "use" not in bad and not REF_ONLY_RE.search(src["use"]):
        bad.append("use: reference-only cloning")
    return bad


def source_record(src: dict) -> dict:
    """The licence / consent / attribution fields of a checked sources.json an entry carries (entry.json, summary)."""
    return {k: src[k] for k in (*SOURCE_KEYS, "dataset_url", "voice") if isinstance(src.get(k), str)}


def discover_refs(listing: dict[str, list[str]], root: str = REFS, warn: Callable[[str], None] | None = None,
                  read_sources: Callable[[str], object] | None = None) -> list[dict]:
    """One ref entry per folder of contest-refs-push (listing: folder -> file names): the clip with "neutral" in its
    name (or the only clip) as <folder>, any other clip as <folder>-<stem>. A folder without a clip or a complete
    sources.json (read_sources(folder) -> its parsed sources.json; check_sources: licence, consent, attribution,
    reference-only use) is left out; the entries carry its record as "source"."""
    out = []
    for d in sorted(listing):
        files = listing[d]
        clips = sorted(f for f in files if f.lower().endswith(AUDIO_EXT))
        why = ("no sources.json (licence + consent record)" if "sources.json" not in files else "no audio clip"
               if not clips else "not a usable entry name" if not NAME_RE.match(d) or d in RESERVED else "")
        src = None
        if not why:
            src = read_sources(d) if read_sources else None
            if bad := check_sources(src):
                why = f"sources.json lacks {', '.join(bad)}"
        if why:
            if warn:
                warn(f"refs/{d}: left out: {why}")
            continue
        neutral = [c for c in clips if "neutral" in c.lower()]
        main = neutral[0] if neutral else clips[0] if len(clips) == 1 else None
        gender = src.get("gender") if isinstance(src.get("gender"), str) and src["gender"].strip() else None
        for c in sorted(clips, key=lambda c: c != main):  # the neutral entry first
            stem = re.sub(r"[^A-Za-z0-9._+-]", "-", c.rsplit(".", 1)[0])
            out.append({"name": d if c == main else f"{d}-{stem}"[:64], "kind": "ref", "ref": f"{root}/{d}/{c}",
                        "notes": f"voxcpm-refs/{d}/{c}", "source": source_record(src),
                        **({"gender": gender} if gender else {})})
    return out


def with_screen(spec, screen: str):
    """spec with "screen" = the comma-separated runs of render --screen (spec unchanged when screen is empty)."""
    runs = [r.strip() for r in screen.split(",") if r.strip()]
    if not runs:
        return spec
    if bad := [r for r in runs if not RUN_RE.match(r) or r == "refs"]:
        raise ValueError(f"--screen {', '.join(bad)}: not a contest run name")
    return {**({"entries": spec} if isinstance(spec, list) else spec), "screen": runs}


def resolve_spec(spec, run: str, list_refs: Callable[[], dict[str, list[str]]],
                 read_screen: Callable[[str], object], refs_root: str = REFS,
                 warn: Callable[[str], None] | None = None,
                 read_sources: Callable[[str], object] | None = None) -> list[dict]:
    """The explicit entries of a spec (see the module doc): its entries, the discovered refs ("refs"; read_sources:
    discover_refs), the winners of the screen runs ("screen"; read_screen(run) -> that run's screen/entries.json or
    None), the LoRA arms ("arms")."""
    spec = {"entries": spec} if isinstance(spec, list) else spec
    if not isinstance(spec, dict):
        raise ValueError("an entry spec is a JSON object or list")
    if bad := set(spec) - SPEC_KEYS:
        raise ValueError(f"unknown spec keys {sorted(bad)} (known: {', '.join(sorted(SPEC_KEYS))})")
    entries = parse_entries(spec.get("entries", DEFAULT_SPEC["entries"]))  # omitted: the controls
    names = {e["name"] for e in entries}
    refs = spec.get("refs", False)
    if refs:
        if refs != "auto" and not (isinstance(refs, list) and all(isinstance(x, str) for x in refs)):
            raise ValueError('"refs" is "auto", a list of folder names or false')
        listing = list_refs()
        if refs != "auto":
            if missing := [d for d in refs if d not in listing]:
                raise ValueError(f"refs: no folder {', '.join(missing)} under {refs_root} (contest-refs-push)")
            listing = {d: listing[d] for d in refs}
        entries += [e for e in discover_refs(listing, refs_root, warn, read_sources) if e["name"] not in names]
        names = {e["name"] for e in entries}
    screens = spec.get("screen", [run] if run else [])
    for r in [screens] if isinstance(screens, str) else screens:
        data = read_screen(r)
        if data is None:
            if "screen" in spec:
                raise ValueError(f"screen {r}: no {r}/screen/entries.json (run contest-screen --run {r} first)")
            continue
        for e in parse_entries(data):
            if e["name"] in names:
                raise ValueError(f"screen {r}: entry {e['name']} is also named in the spec")
            entries.append(e)
            names.add(e["name"])
    for a in spec.get("arms", []):
        if not (isinstance(a, dict) and isinstance(a.get("lora"), str) and isinstance(a.get("entries"), list)):
            raise ValueError('an arm is {"lora": dir, "entries": [name, ...], "suffix": "+s1"}')
        by = {e["name"]: e for e in entries}
        for n in a["entries"]:
            if n not in by:
                raise ValueError(f"arm {a['lora']}: no entry {n}")
            if by[n].get("lora"):
                raise ValueError(f"arm {a['lora']}: entry {n} already has a LoRA")
            entries.append({**by[n], "name": f"{n}{a.get('suffix', '+lora')}", "lora": a["lora"],
                            "notes": f"{by[n].get('notes', n)} | second arm: LoRA {a['lora']}"})
    return parse_entries(entries)


class LocalFS:
    """resolve_ref's view of the files (modal_app has one of the Volume)."""

    def kind(self, p: str) -> str | None:
        q = Path(p)
        return "dir" if q.is_dir() else "file" if q.is_file() else None

    def listdir(self, p: str) -> list[str]:
        q = Path(p)
        return sorted(x.name for x in q.iterdir()) if q.is_dir() else []

    def read_text(self, p: str) -> str | None:
        try:
            return Path(p).read_text(encoding="utf-8")
        except OSError:
            return None


def resolve_ref(ref: str, fs, refs_root: str = REFS) -> str:
    """The clip of an entry's ref / sim_ref: a clip as it is; a folder -> its clip named *neutral*, or its only clip;
    manifest:<file>#<row> -> that row's "audio"; relative = under refs_root. ValueError when there is none."""
    if ref.startswith("manifest:"):
        f, _, i = ref[len("manifest:"):].rpartition("#")
        if not f or not i.isdigit():
            raise ValueError(f"{ref}: want manifest:<file>#<row>")
        text = fs.read_text(f)
        if text is None:
            raise ValueError(f"{ref}: no {f}")
        lines = [x for x in text.splitlines() if x.strip()]
        try:
            audio = json.loads(lines[int(i)])["audio"]
        except (IndexError, ValueError, KeyError, TypeError):
            raise ValueError(f"{ref}: no row {i} with an audio path in {f}") from None
        return resolve_ref(_MODAL_VOLUME.sub(f"{DATA}/", str(audio)), fs, refs_root)
    path = (ref if _abs(ref) else f"{refs_root}/{ref}").rstrip("/")
    kind = fs.kind(path)
    if kind == "file":
        if not path.lower().endswith(AUDIO_EXT):
            raise ValueError(f"{ref}: not an audio file ({', '.join(AUDIO_EXT)})")
        return path
    if kind == "dir":
        clips = sorted(n for n in fs.listdir(path) if n.lower().endswith(AUDIO_EXT))
        neutral = [c for c in clips if "neutral" in c.lower()]
        pick = neutral[0] if len(neutral) == 1 else clips[0] if len(clips) == 1 else None
        if pick is None:
            raise ValueError(f"{ref}: {len(clips)} clips in {path} and not exactly one named *neutral*: name the clip")
        return f"{path}/{pick}"
    raise ValueError(f"{ref}: {path} not found")


def resolve_lora(lora: str, fs) -> str:
    """A LoRA checkpoint folder (relative = under /data/exp) with lora_weights.safetensors and lora_config.json."""
    path = (lora if _abs(lora) else f"{EXP}/{lora}").rstrip("/")
    if fs.kind(path) != "dir":
        raise ValueError(f"lora {lora}: {path} not found")
    if missing := {"lora_weights.safetensors", "lora_config.json"} - set(fs.listdir(path)):
        raise ValueError(f"lora {lora}: no {', '.join(sorted(missing))} in {path}")
    return path


def resolve_entry(e: dict, fs, refs_root: str = REFS) -> dict:
    """e with its ref / sim_ref as clips and its lora as a folder (ValueError when one is not there)."""
    out = dict(e)
    for k in ("ref", "sim_ref"):
        if out.get(k):
            out[k] = resolve_ref(out[k], fs, refs_root)
    if out.get("lora"):
        out["lora"] = resolve_lora(out["lora"], fs)
    return out


def lora_config(path: str | None, read_text: Callable[[str], str | None] | None = None) -> dict | None:
    """The "lora_config" a LoRA checkpoint's layers are built from; {"auto": True} without lora_config.json (VoxCPM
    then builds its default); None for no LoRA. read_text(path) -> text or None: another view of the files (modal_app:
    the Volume)."""
    if not path:
        return None
    try:
        text = read_text(f"{path}/lora_config.json") if read_text else Path(path, "lora_config.json").read_text(
            encoding="utf-8")
        cfg = json.loads(text)["lora_config"]
        return cfg if isinstance(cfg, dict) else {"auto": True}
    except (OSError, ValueError, KeyError, TypeError):
        return {"auto": True}


def entry_backend(backend: str, e: dict, lora_cfg: dict | None, old: dict | None = None) -> tuple[str, str | None]:
    """The engine that makes entry e's clips under --backend `backend`, and why it is not that one (else None):
    nanovllm hands an entry whose LoRA it cannot load (nano_lora_problem of its lora_config, lora_cfg) to the official
    API (voxcpm), and keeps an entry an earlier call handed over there (old: its entry.json, "fallback" with the same
    LoRA), so the clips of one entry always come from one engine."""
    if backend != "nanovllm" or not e.get("lora"):
        return backend, None
    if (old and old.get("fallback") and old.get("lora") == e["lora"]
            and (old.get("settings") or {}).get("backend") == "voxcpm"):
        return "voxcpm", str(old["fallback"])
    why = nano_lora_problem(lora_cfg)
    return ("voxcpm", f"Nano-vLLM cannot load LoRA {e['lora']}: {why}") if why else ("nanovllm", None)


def entry_key(e: dict, settings: dict) -> str:
    """What makes an entry's clips: its reference, LoRA and the synthesis settings."""
    blob = json.dumps({"ref": e.get("ref"), "lora": e.get("lora"), **settings}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def render_settings(args) -> dict:
    """The synthesis settings of a render (entry.json "settings" through entry_settings; modal_app checks them before a
    call): seed_mode (text_seed), loudness (refprep.normalize_output on every clip), ref_prep (refprep.prep_ref on
    every reference, or "raw": --no-refprep), pause_clean (pause_clean.SETTINGS with --pause-clean; absent without
    it, so the entries made before it keep their key)."""
    return {"timesteps": args.timesteps, "cfg": args.cfg, "seed": args.seed, "model": args.model,
            "backend": args.backend, "seed_mode": SEED_MODE, "loudness": LOUDNESS,
            "ref_prep": "raw" if getattr(args, "no_refprep", False) else REF_PREP,
            **({"pause_clean": PAUSE_CLEAN} if getattr(args, "pause_clean", False) else {})}


def entry_settings(settings: dict, e: dict, backend: str) -> dict:
    """An entry's own settings: the call's, its engine (entry_backend) and its "ref_prep" when the spec sets one."""
    out = {**settings, "backend": backend}
    if e.get("ref_prep") is not None:
        out["ref_prep"] = REF_PREP if e["ref_prep"] else "raw"
    return out


def screen_settings(args) -> dict:
    """The synthesis settings of a screen (<screen>/settings.json; check_settings, and modal_app before a call)."""
    return {"timesteps": args.timesteps, "cfg": args.cfg, "model": args.model, "backend": args.backend,
            "seed_mode": SEED_MODE, "loudness": LOUDNESS,
            "ref_prep": "raw" if getattr(args, "no_refprep", False) else REF_PREP}


def text_seed(base: int, set_: str, id_: str) -> int:
    """The default seed of a text: --seed + crc32("<set>/<id>"), the same in every voice, another for each sentence
    (Nano-vLLM derives each step's noise from (seed, step): one seed for every clip gave every clip one first-step
    noise, and round 1 its stray onsets)."""
    return int(base) + (zlib.crc32(f"{set_}/{id_}".encode()) & 0x7FFFFFFF)


def job_seed(r: dict, base: int) -> int:
    """The seed a row's clip is requested with: its own "seed" (texts file, render --seeds, screen rows), else
    text_seed."""
    return int(r["seed"]) if r.get("seed") is not None else text_seed(base, r["set"], r["id"])


def entry_conflict(old: dict | None, e: dict, settings: dict) -> dict:
    """{field: "was -> now"} where an entry dir's entry.json (old) was made otherwise than entry e with these settings;
    {} when it was not (or there is no entry.json). Clips of one entry dir all come from one reference, LoRA and
    settings."""
    if not old or old.get("key") == entry_key(e, settings):
        return {}
    was = {"ref": old.get("ref"), "lora": old.get("lora"), **(old.get("settings") or {})}
    diff = {k: f"{was.get(k)} -> {v}" for k, v in {"ref": e.get("ref"), "lora": e.get("lora"), **settings}.items()
            if was.get(k) != v}
    return diff or {"key": f"{old.get('key')} -> {entry_key(e, settings)}"}


def pool_ram_gb(workers: int, load_parallel: int) -> float:
    """Estimated host RAM peak of a synthesis pool while load_parallel workers load VoxCPM2 at once."""
    lp = min(max(load_parallel, 1), workers)
    return (workers - lp) * WORKER_RAM_GB + lp * LOAD_RAM_GB + BASE_RAM_GB


# ---- texts --------------------------------------------------------------------------------------------------------


def load_texts(path: Path) -> list[dict]:
    """The contest text set (module doc): one row per line, trnorm.normalize applied here once (model_text)."""
    if not path.is_file():
        raise ValueError(f"no {path}: the contest text set (benchmarks/tts/sets/contest.jsonl)")
    rows, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            rid, set_, text = r["id"], r["set"], r["text"]
        except (ValueError, KeyError, TypeError):
            raise ValueError(f"{path}:{n}: want a JSON object with id, set, text") from None
        if not (isinstance(rid, str) and ID_RE.match(rid)) or rid in seen:
            raise ValueError(f"{path}:{n}: id {rid!r} is not unique or not {ID_RE.pattern}")
        if not (isinstance(set_, str) and ID_RE.match(set_)) or not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{n}: bad set or empty text")
        seen.add(rid)
        model_text = normalize(text)
        refs = r.get("refs")
        refs = [refs] if isinstance(refs, str) else refs if isinstance(refs, list) and refs else (
            [r["spoken"], *r.get("alts", [])] if r.get("spoken") else [model_text])
        try:
            instruction = strip_instruction(r.get("instruction"))
        except ValueError as e:
            raise ValueError(f"{path}:{n}: {e}") from None
        emo = r.get("emotion") or (r.get("category") if set_ == "emo" else None)
        emo = EMOTION_ALIASES.get(str(emo).lower(), str(emo).lower()) if emo else None
        row = {"set": set_, "id": rid, "category": r.get("category") or emo or set_, "text": text,
               "model_text": model_text, "refs": [str(x) for x in refs]}
        if instruction:
            row["instruction"] = instruction
        if emo:
            row["emotion"] = emo
        if set_ == "soru":
            if any(r.get(k) is None for k in ("pattern", "peak", "peak_word")):
                raise ValueError(f"{path}:{n}: a soru row needs pattern, peak and peak_word (score_intonation)")
            row.update({k: r[k] for k in ("pattern", "peak", "peak_word", "pair_id") if r.get(k) is not None})
        if r.get("seed") is not None:
            if isinstance(r["seed"], bool) or not isinstance(r["seed"], int) or r["seed"] < 0:
                raise ValueError(f"{path}:{n}: seed must be an integer >= 0")
            row["seed"] = r["seed"]
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def select_texts(rows: list[dict], ids: str = "", seeds: str = "") -> list[dict]:
    """render --ids (comma-separated text ids: those rows, in file order) and --seeds (each row once per seed, id
    <id>-s<seed>, requested with that seed as it is, not text_seed: e.g. 0 = round 1's shared seed)."""
    if want := [i.strip() for i in ids.split(",") if i.strip()]:
        if missing := set(want) - {r["id"] for r in rows}:
            raise ValueError(f"--ids: no text {', '.join(sorted(missing))}")
        rows = [r for r in rows if r["id"] in want]
    if seeds.strip():
        ks = parse_ints(seeds)
        if any(k < 0 for k in ks):
            raise ValueError("--seeds: want integers >= 0")
        rows = [{**r, "id": f"{r['id']}-s{k}", "seed": k} for r in rows for k in ks]
        if bad := [r["id"] for r in rows if not ID_RE.match(r["id"])]:
            raise ValueError(f"--seeds: ids {', '.join(bad[:3])} are not {ID_RE.pattern}")
    return rows


def render_texts(args) -> list[dict]:
    """The texts of a render: --texts, --limit per set, then --ids and --seeds (select_texts)."""
    return select_texts(limit_rows(load_texts(repo_path(args.texts)), args.limit), args.ids, args.seeds)


def limit_rows(rows: list[dict], n: int) -> list[dict]:
    """n rows of each set, spread over its categories (synth_eval.stratified)."""
    if not n:
        return rows
    from synth_eval import stratified

    by = defaultdict(list)
    for r in rows:
        by[r["set"]].append(r)
    keep = {id(x) for s in by.values() for x in stratified(s, n)}
    return [r for r in rows if id(r) in keep]


def load_descriptions(path: Path) -> list[dict]:
    """Voice-design descriptions: id, gender (optional), text (English; "description" also accepted)."""
    if not path.is_file():
        raise ValueError(f"no {path}: the voice descriptions (tts/voxcpm/voice_descriptions.jsonl)")
    out, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            did, text = r["id"], r.get("text") or r["description"]
            text = strip_instruction(text)
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{path}:{n}: want id and text ({e})") from None
        if not (isinstance(did, str) and ID_RE.match(did)) or did in seen or not text:
            raise ValueError(f"{path}:{n}: id {did!r} is not unique or not {ID_RE.pattern}, or the text is empty")
        seen.add(did)
        out.append({"id": did, "gender": str(r.get("gender") or "").lower(), "text": text})
    if not out:
        raise ValueError(f"{path}: no descriptions")
    return out


def merge_rows(new: list[dict], old: list[dict], base: Path) -> list[dict]:
    """This call's rows with an earlier call's measurements where the clip is the same (set/id, prompt and ref); the
    WAV of an earlier clip with another prompt or ref is deleted, so it is made again."""
    prev = {(r.get("set"), r.get("id")): r for r in old}
    out = []
    for r in new:
        o = prev.get((r["set"], r["id"]))
        if o is not None and o.get("prompt") == r["prompt"] and o.get("ref") == r.get("ref"):
            out.append({**o, **r})
        else:
            if o is not None:
                (base / r["wav"]).unlink(missing_ok=True)
            out.append(dict(r))
    return out


# ---- cost model ---------------------------------------------------------------------------------------------------


def est_audio_s(model_text: str) -> float:
    """Rough seconds of speech of a spoken form."""
    return 0.3 + len(model_text) / CHARS_PER_S


def rtf(ts: int) -> float:
    """One process's real-time factor at `ts` timesteps (linear through RTF_10 and RTF_16)."""
    return max(0.2, RTF_10 + (ts - 10) * (RTF_16 - RTF_10) / 6)


def speedup(workers: int) -> float:
    """Assumed throughput of `workers` processes on one GPU re one; a measurement in throughput.jsonl wins."""
    return min(float(workers), 1 + 0.4 * (workers - 1), 2.5)


def wall_per_audio(ts: int, workers: int, measured: list[dict] | None = None) -> float:
    """Wall seconds per second of synthesized audio on the voxcpm backend: the last measurement with these timesteps
    and workers, else with these timesteps (rescaled by speedup), else rtf / speedup."""
    same = [m for m in measured or [] if m.get("backend", "voxcpm") == "voxcpm" and m.get("timesteps") == ts
            and m.get("audio_s", 0) >= 30 and m.get("wall_s", 0) > 0 and m.get("workers")]
    if exact := [m for m in same if m["workers"] == workers]:
        return exact[-1]["wall_s"] / exact[-1]["audio_s"]
    if same:
        m = same[-1]
        return m["wall_s"] / m["audio_s"] * speedup(m["workers"]) / speedup(workers)
    return rtf(ts) / speedup(workers)


def _lin(points: dict[int, float], ts: int) -> float:
    """The line through a {timesteps: value} table's two points, at ts."""
    (a, ya), (b, yb) = sorted(points.items())[:2]
    return ya + (ts - a) * (yb - ya) / (b - a)


def nano_wall_per_audio(ts: int, ref: bool, measured: list[dict] | None = None) -> float:
    """GPU seconds per second of Nano-vLLM audio with one server busy at `ts` timesteps, requests with (ref) or without
    a reference: the last server measured so (throughput.jsonl, backend nanovllm, at least NANO_MIN_JOBS x its
    concurrency requests, so its last batch weighs little), else the probe's NANO_SPEED (seconds per audio s, linear
    in the timesteps through its two points)."""
    same = [m for m in measured or [] if m.get("backend") == "nanovllm" and m.get("timesteps") == ts
            and bool(m.get("ref")) == ref and (m.get("jobs") or 0) >= NANO_MIN_JOBS * (m.get("concurrency") or 1)
            and (m.get("audio_s") or 0) > 0 and (m.get("wall_s") or 0) > 0]
    if same:
        return same[-1]["wall_s"] / same[-1]["audio_s"]
    return max(0.005, _lin({t: 1.0 / v[int(ref)] for t, v in NANO_SPEED.items()}, ts))


def nano_speed_note(ts: int, measured: list[dict] | None = None) -> str:
    """The Nano-vLLM speeds the estimates use, for the plan's printout."""
    src = "measured" if any(m.get("backend") == "nanovllm" and m.get("timesteps") == ts for m in measured or []) \
        else "probe"
    return (f"Nano-vLLM ({src}): {1 / nano_wall_per_audio(ts, False, measured):.0f} audio s per GPU s without a "
            f"reference, {1 / nano_wall_per_audio(ts, True, measured):.0f} with one, + {NANO_START_S + NANO_STOP_S:.0f}"
            " s per server")


def nano_server_s(audio: list[float], ts: int, refs: list[bool], measured: list[dict] | None = None,
                  start: bool = True) -> float:
    """Estimated seconds of one Nano-vLLM server making clips of these lengths (est_audio_s; refs: with a reference
    each): start and stop (unless start=False: a server already counted), its share of the busy server, and the last
    batch (the longest clip at NANO_REQ_RTF). The same sum decides what a group starts with (nano_fit)."""
    if not audio:
        return 0.0
    per = {r: nano_wall_per_audio(ts, r, measured) for r in (False, True)}
    synth = sum(a * NANO_DUR * per[bool(r)] for a, r in zip(audio, refs))
    return synth + ((NANO_START_S + NANO_STOP_S + max(audio) * NANO_DUR * _lin(NANO_REQ_RTF, ts)) if start else 0.0)


def nano_groups(jobs: list[dict]) -> list[tuple[tuple[str | None, int], list[dict]]]:
    """Jobs by (LoRA or None, timesteps), one Nano-vLLM server each: groups in order of first appearance, jobs in
    their order (order_jobs: whole entries in turn)."""
    out: dict[tuple[str | None, int], list[dict]] = {}
    for j in jobs:
        out.setdefault((j.get("lora") or None, int(j["ts"])), []).append(j)
    return list(out.items())


def nano_fit(jobs: list[dict], ts: int, budget_s: float, measured: list[dict] | None = None) -> list[dict]:
    """The jobs a group's server starts with: the longest prefix whose estimate (nano_server_s) fits in budget_s; []
    when not even the first does: the group is not started."""
    for n in range(len(jobs), 0, -1):  # the estimate grows with every job: the first that fits from the end
        if nano_server_s([j["est_s"] for j in jobs[:n]], ts, [bool(j.get("ref")) for j in jobs[:n]],
                         measured) <= budget_s:
            return jobs[:n]
    return []


def plan_render(entries: list[dict], rows: list[dict], done: dict[str, set[str]], ts: int, workers: int,
                budget_s: float, measured: list[dict] | None = None, backend: str = "voxcpm",
                engines: dict[str, str] | None = None, score: bool = True) -> list[dict]:
    """Per entry, in order: clips, clips still to make (done: entry -> {"<set>/<id>"} already on disk), estimated
    seconds (synthesis + scoring; score=False, render --no-score: synthesis only) and whether it fits in what budget_s
    leaves after the fixed part (scorer loads, summaries). The entries are made as synthesize() makes them: first those
    on Nano-vLLM (engines: entry -> its engine, default backend; entry_backend), one server per LoRA in order of first
    appearance, whose start, stop and last batch (nano_server_s) the first of its entries with clips to make pays; then
    those on the voxcpm pool (its worker start once). An entry that does not fit is left out of the call and costs
    nothing."""
    eng = {e["name"]: (engines or {}).get(e["name"], backend) for e in entries}
    loras = list(dict.fromkeys(e.get("lora") for e in entries if eng[e["name"]] == "nanovllm"))
    order = sorted(range(len(entries)), key=lambda i: (eng[entries[i]["name"]] != "nanovllm", loras.index(
        entries[i].get("lora")) if eng[entries[i]["name"]] == "nanovllm" else 0, i))
    per_audio = wall_per_audio(ts, workers, measured)
    left, up, started, out = budget_s - (SCORE_LOAD_S if score else 0.0) - SUMMARY_S, False, set(), {}
    for i in order:
        e = entries[i]
        have = done.get(e["name"], set())
        todo = [r for r in rows if f"{r['set']}/{r['id']}" not in have]
        audio = [est_audio_s(r["model_text"]) for r in todo]
        nano = eng[e["name"]] == "nanovllm"
        if nano:
            need = nano_server_s(audio, ts, [bool(e.get("ref"))] * len(audio), measured,
                                 start=e.get("lora") not in started)
        else:
            need = sum(audio) * per_audio + (WORKERS_UP_S if todo and not up else 0.0)
        need += len(todo) * SCORE_CLIP_S if score else 0.0
        fits = need <= left
        if fits:
            left -= need
            if nano and todo:
                started.add(e.get("lora"))
            up = up or (not nano and bool(todo))
        out[i] = {"name": e["name"], "clips": len(rows), "todo": len(todo), "est_s": round(need, 1), "fits": fits,
                  "engine": eng[e["name"]]}
    return [out[i] for i in range(len(entries))]


def clone_top(top: int, n_desc: int, per_desc: int, n_seeds: int) -> int:
    """How many candidates the clone test can get: --top, at most --per-desc of each description's seeds."""
    return max(0, min(top, n_desc * min(per_desc, n_seeds)))


def plan_screen(n_todo: int, top: int, clone_texts: list[str], ts: int, workers: int, screen_text: str = SCREEN_TEXT,
                measured: list[dict] | None = None, clones_done: int = 0, backend: str = "voxcpm") -> float:
    """Estimated seconds of a screen call: n_todo candidates still to make, top x clone_texts clone tests (clones_done
    already made), scoring and scorer loads; voxcpm: the workers' start once; nanovllm: a server per round
    (nano_server_s), and Whisper / UTMOS loaded again for the clones (released while the clone server runs)."""
    n_clones = max(top * len(clone_texts) - clones_done, 0)
    cand_a = est_audio_s(normalize(screen_text))
    clone_a = [est_audio_s(normalize(t)) for t in clone_texts]
    fixed, score = SCORE_LOAD_S + SUMMARY_S, (n_todo + n_clones) * SCORE_CLIP_S
    if backend == "nanovllm":
        clones = [clone_a[k % len(clone_a)] for k in range(n_clones)] if clone_a else []
        reload = LOAD_S["whisper"] + LOAD_S["utmos"] if n_todo and n_clones else 0.0
        return (fixed + score + reload + nano_server_s([cand_a] * n_todo, ts, [False] * n_todo, measured)
                + nano_server_s(clones, ts, [True] * n_clones, measured))
    per_audio = wall_per_audio(ts, workers, measured)
    synth = (n_todo * cand_a + n_clones * sum(clone_a) / max(len(clone_a), 1)) * per_audio
    return WORKERS_UP_S + fixed + synth + score


# ---- job queue and workers ----------------------------------------------------------------------------------------


def order_jobs(jobs: list[dict]) -> list[dict]:
    """Queue order: groups (an entry, a screen round) in order of first appearance, so a call cut short leaves whole
    entries; inside a group the longest first (est_s), so the last to finish are short ones; then by key."""
    first: dict[str, int] = {}
    for j in jobs:
        first.setdefault(j["group"], len(first))
    return sorted(jobs, key=lambda j: (first[j["group"]], -j["est_s"], j["key"]))


class FakeSynth:
    """Test backend (--backend fake): silence as long as est_s, written with the stdlib; "FAIL" in a text raises. Its
    model is a test hook: "die-on-load:<n>:<dir>" kills the first n workers while they load (exit code 137, as the OOM
    killer would; one marker file each in dir), "hang-on-load" never finishes loading."""

    sr = 16000

    def __init__(self, model: str = ""):
        if model == "hang-on-load":
            time.sleep(3600)
        if model.startswith("die-on-load:"):
            n, d = model.split(":", 2)[1:]
            for i in range(int(n)):
                try:
                    os.close(os.open(os.path.join(d, f"die{i}"), os.O_CREAT | os.O_EXCL))
                except FileExistsError:
                    continue
                time.sleep(0.5)
                os._exit(137)

    def run(self, job: dict) -> dict:
        import wave

        if "FAIL" in job["text"]:
            raise RuntimeError("fake failure")
        time.sleep(float(job.get("sleep", 0.0)))
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        n = int(job["est_s"] * self.sr)
        res = {"duration": n / self.sr}
        if job.get("pause_clean"):  # the real gate on the silent clip (numpy, scipy): the flag's path, not its effect
            import numpy as np

            res["pause_clean"] = clean_pauses(np.zeros(n, dtype=np.float32), self.sr)[1]
        tmp = out.with_name(out.name + ".part")
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(b"\x00\x00" * n)
        os.replace(tmp, out)
        return res

    def info(self) -> dict:
        return {"backend": "fake"}


class VoxSynth:
    """VoxCPM2 in one worker process, through the official API (core.VoxCPM at the pinned commit)."""

    def __init__(self, model: str, lora: str | None, optimize: bool, gate=None):
        import torch

        torch.set_num_threads(2)
        self.model, self.optimize, self.gate, self.tts = model, optimize, gate, None
        self._load(lora)

    def _load(self, lora: str | None) -> None:
        import gc

        import torch
        from voxcpm import VoxCPM

        self.tts = None
        gc.collect()
        torch.cuda.empty_cache()
        t = time.time()
        # lora_weights_path: VoxCPM reads <dir>/lora_config.json (r, alpha, targets) and loads the weights
        self.tts = VoxCPM.from_pretrained(self.model, load_denoiser=False, optimize=self.optimize,
                                          lora_weights_path=lora)
        self.lora, self.cfg, self.sr = lora, lora_config(lora), self.tts.tts_model.sample_rate
        self.load_s = round(time.time() - t, 1)

    def _use(self, lora: str | None) -> None:
        if lora is None:
            if self.cfg is not None:
                self.tts.set_lora_enabled(False)  # LoRA scale 0: the base model's output
            return
        cfg = lora_config(lora)
        if self.cfg is None or cfg != self.cfg:  # base-only model, or a LoRA of another shape: load again
            if self.gate is not None:
                with self.gate:
                    self._load(lora)
            else:
                self._load(lora)
        elif lora != self.lora:
            self.tts.load_lora(lora)
            self.lora = lora
        self.tts.set_lora_enabled(True)

    def run(self, job: dict) -> dict:
        """One clip: the reference prepared (refprep.prep_ref_file, unless the job says ref_prep False), the output
        through pause_clean when the job says so (clean_pauses), then at refprep's level, its onsets checked
        (onset_info; recorded, not retried) -> {"duration", onset fields, "level_in": its active level before
        levelling (dBFS), "pause_clean" (cleaned clips)}."""
        import tempfile

        import numpy as np
        import soundfile as sf
        import torch
        from refprep import normalize_level, prep_ref_file

        self._use(job.get("lora"))
        ref = job.get("ref") or None
        if ref and job.get("ref_prep", True):
            ref = prep_ref_file(ref, Path(tempfile.gettempdir()) / "voxcpm-refprep")
        wav = self.tts.generate(text=job["text"], reference_wav_path=ref, cfg_value=job["cfg"],
                                inference_timesteps=job["ts"], normalize=False, seed=job["seed"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wav, pc = np.asarray(wav, dtype=np.float32), None
        if job.get("pause_clean"):
            wav, pc = clean_pauses(wav, self.sr)
        wav, lv = normalize_level(wav, self.sr)
        info = {**onset_info(check_onset(wav, self.sr, job["text"])), "level_in": lv["level_in"]}
        if pc is not None:
            info["pause_clean"] = pc
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        sf.write(str(tmp), wav, self.sr, format="WAV", subtype="PCM_16")
        os.replace(tmp, out)  # a killed worker leaves only a .part file
        return {"duration": len(wav) / self.sr, **info}

    def info(self) -> dict:
        import torch

        d = {"load_s": self.load_s, "lora": self.lora}
        if torch.cuda.is_available():
            d["gpu_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
            d["gpu_reserved_mib"] = round(torch.cuda.memory_reserved() / 2**20)
        return d


class PoolClosing(Exception):
    """The pool closed while this worker waited for its turn to load."""


class LoadGate:
    """A worker's turn to load VoxCPM2 (RAM: one load peaks at ~LOAD_RAM_GB): asks SynthPool for a token over the
    results queue, waits for it on its own ctl queue (None: the pool is closing) and hands it back after. The parent
    takes back the token of a worker that dies while loading; a shared semaphore would stay taken by a killed holder
    and leave the others waiting forever."""

    def __init__(self, wid: int, ctl, results):
        self.wid, self.ctl, self.results = wid, ctl, results

    def __enter__(self):
        self.results.put(("want", self.wid))
        if self.ctl.get() is None:
            raise PoolClosing
        return self

    def __exit__(self, *exc) -> bool:
        self.results.put(("loaded", self.wid))
        return False


def _worker(wid: int, model: str, backend: str, lora: str | None, optimize: bool, ctl, jobs, results) -> None:
    """One synthesis process: load the model once in its turn (LoadGate: at most --load-parallel at a time, RAM), then
    take jobs till None."""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    gate = LoadGate(wid, ctl, results)
    try:
        with gate:
            synth = FakeSynth(model) if backend == "fake" else VoxSynth(model, lora, optimize, gate)
        results.put(("ready", wid, synth.info()))
    except PoolClosing:
        return
    except BaseException as e:  # noqa: BLE001  (CUDA OOM, a missing file, ...): the other workers go on
        results.put(("dead", wid, f"{type(e).__name__}: {e}"[:500]))
        return
    while (job := jobs.get()) is not None:
        results.put(("start", wid, job["key"]))
        t = time.time()
        try:
            res = synth.run(job)
            took = round(time.time() - t, 3)
            results.put(("done", wid, job["key"], {**res, "duration": round(res["duration"], 3), "synth_seconds": took,
                                                   "worker": wid}))
        except Exception as e:
            results.put(("fail", wid, job["key"], f"{type(e).__name__}: {e}"[:500]))
    results.put(("exit", wid, synth.info()))


def gpu_query(field: str) -> list[str]:
    """nvidia-smi --query-gpu=<field> values (one per GPU); [] without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return [x.strip() for x in out.splitlines() if x.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def gpu_free_mib() -> float | None:
    v = gpu_query("memory.free")
    try:
        return float(v[0]) if v else None
    except ValueError:
        return None


def host_mem_mib() -> float | None:
    """Memory in use in this container (cgroup v2 / v1, page cache included), else on this machine (MemTotal -
    MemAvailable); None without either (Windows)."""
    for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return round(int(Path(p).read_text().split()[0]) / 2**20)
        except (OSError, ValueError, IndexError):
            pass
    try:
        info = {k: int(v.split()[0]) for k, v in (x.split(":", 1) for x in
                                                   Path("/proc/meminfo").read_text().splitlines() if ":" in x)}
        return round((info["MemTotal"] - info["MemAvailable"]) / 1024)
    except (OSError, ValueError, KeyError, IndexError):
        return None


def rss_mib(pids) -> float | None:
    """Sum of the resident memory (VmRSS) of these processes; None without /proc."""
    total, seen = 0, False
    for pid in pids:
        try:
            line = next(x for x in Path(f"/proc/{pid}/status").read_text().splitlines() if x.startswith("VmRSS:"))
            total, seen = total + int(line.split()[1]), True
        except (OSError, ValueError, IndexError, StopIteration):
            pass
    return round(total / 1024) if seen else None


class SynthPool:
    """`workers` synthesis processes on one GPU (spawn: CUDA cannot fork), fed from one queue (order_jobs), so a slow or
    dead worker never holds the others back. lora: the LoRA whose structure every worker loads (see the module doc).
    At most load_parallel workers load at a time (LoadGate tokens, handed out while run() runs); the pool is given up
    when no worker is ready after ready_s or, with jobs out, no clip finishes for STALL_S."""

    def __init__(self, workers: int, model: str, backend: str = "voxcpm", lora: str | None = None,
                 load_parallel: int = 2, optimize: bool = False, log: Callable[[str], None] = print,
                 ready_s: float = READY_S):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.jobs, self.results, self.log, self.workers, self.backend = ctx.Queue(), ctx.Queue(), log, workers, backend
        self.ctl = {w: ctx.Queue() for w in range(workers)}
        self.slots, self.loading, self.waiting = max(1, load_parallel), set(), deque()
        self.procs = {w: ctx.Process(target=_worker, args=(w, model, backend, lora, optimize, self.ctl[w], self.jobs,
                                                             self.results), daemon=True) for w in range(workers)}
        self.t0, self.ready_s = time.time(), ready_s
        for p in self.procs.values():
            p.start()
        self.ready, self.dead, self.exited, self.busy = {}, {}, {}, {}
        self.t_ready, self.mem_peak, self.host_peak, self.rss_peak = None, None, None, None
        self._stop = threading.Event()
        self.gpu = (gpu_query("name") or [None])[0]
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self) -> None:  # peak GPU and host memory while the pool lives
        while not self._stop.wait(5):
            v = gpu_query("memory.used")
            if v:
                try:
                    self.mem_peak = max(self.mem_peak or 0.0, float(v[0]))
                except ValueError:
                    pass
            if (h := host_mem_mib()) is not None:
                self.host_peak = max(self.host_peak or 0, h)
            if (r := rss_mib([os.getpid(), *(p.pid for p in self.procs.values())])) is not None:
                self.rss_peak = max(self.rss_peak or 0, r)

    def live(self) -> int:
        return sum(w not in self.dead and w not in self.exited for w in self.procs)

    def _grant(self) -> None:
        """Load tokens to the waiting workers while fewer than load_parallel hold one."""
        while self.waiting and len(self.loading) < self.slots:
            w = self.waiting.popleft()
            if w not in self.dead and w not in self.exited:
                self.loading.add(w)
                self.ctl[w].put(1)

    def _release(self, w: int) -> None:
        """Worker w's token back (it loaded, failed, or died), to the next waiting worker."""
        self.loading.discard(w)
        if w in self.waiting:
            self.waiting.remove(w)
        self._grant()

    def _handle(self, msg, done: Callable, failed: dict) -> int:
        """One worker message; returns how many jobs it finished (done or failed)."""
        kind, wid = msg[0], msg[1]
        if kind == "want":
            self.waiting.append(wid)
            self._grant()
        elif kind == "loaded":
            self._release(wid)
        elif kind == "ready":
            self.ready[wid] = msg[2]
            self.t_ready = self.t_ready or time.time()
            self.log(f"[pool] worker {wid} ready {msg[2]}")
        elif kind == "dead":
            self.dead[wid] = msg[2]
            self._release(wid)
            self.log(f"[pool] worker {wid} failed to start: {msg[2]}")
        elif kind == "start":
            if wid in self.dead:  # read after the worker was found dead: its job is lost
                failed[msg[2]] = f"worker {wid} died ({self.dead[wid]})"
                return 1
            self.busy[wid] = msg[2]
        elif kind == "done":
            self.busy.pop(wid, None)
            done(msg[2], msg[3])
            return 1
        elif kind == "fail":
            self.busy.pop(wid, None)
            failed[msg[2]] = msg[3]
            self.log(f"[pool] {msg[2]}: {msg[3]}")
            return 1
        elif kind == "exit":
            self.exited[wid] = msg[2]
        return 0

    def run(self, jobs: list[dict], feed_until: float = math.inf, hard_until: float = math.inf,
            on_done: Callable[[dict, dict], None] | None = None,
            on_fail: Callable[[dict, str], None] | None = None) -> dict:
        """Feed jobs (in order) while each could still finish before feed_until (estimated from the clips done so far),
        collect results (on_done(job, info), on_fail(job, error)); at hard_until, or when the workers do not get ready
        or stop making progress (class doc), they are killed. Returns synthesis stats (throughput)."""
        import queue

        by_key = {j["key"]: j for j in jobs}
        pending, outstanding, failed, n_done = deque(jobs), 0, {}, 0
        audio = busy_s = 0.0
        t_first = t_last = None
        job_rtf: list[float] = []
        stopped, last_msg = "", time.time()
        t_start = t_progress = last_msg

        def done(key: str, info: dict) -> None:
            nonlocal n_done, audio, busy_s, t_last
            n_done += 1
            audio += info["duration"]
            busy_s += info["synth_seconds"]
            t_last = time.time()
            if info["duration"] > 0:
                job_rtf.append(info["synth_seconds"] / info["duration"])
            if on_done:
                on_done(by_key[key], info)

        while pending or outstanding:
            now = time.time()
            if now > hard_until:
                stopped = "hard deadline: workers killed"
                self.kill()
                break
            if not self.ready and now - self.t0 > self.ready_s:
                stopped = f"no worker ready after {self.ready_s:.0f} s: workers killed"
                self.kill()
                break
            if outstanding and self.ready and now - max(t_progress, self.t_ready or 0.0) > STALL_S:
                stopped = f"no clip finished in {STALL_S:.0f} s: workers killed"
                self.kill()
                break
            live = self.live()
            if not live:
                stopped = "no worker left"
                break
            # each in-flight job is slowed by the others: wall per audio second of one job, measured or assumed
            per = median(job_rtf) or rtf(int(pending[0]["ts"]) if pending else 10) * live / speedup(live)
            while pending and outstanding < 2 * live:  # a job fed now may wait behind one per worker: 2 x its time
                if time.time() + 2 * pending[0]["est_s"] * per > feed_until:
                    stopped = stopped or "deadline: no time for more clips (the same command continues)"
                    break
                self.jobs.put(pending.popleft())
                outstanding += 1
                t_first = t_first or time.time()
            if stopped and not outstanding:
                break
            try:
                if finished := self._handle(self.results.get(timeout=2), done, failed):
                    outstanding -= finished
                    t_progress = time.time()
                last_msg = time.time()
            except queue.Empty:
                pass
            for w, p in self.procs.items():  # a worker that died without a word (segfault, OOM kill)
                if not p.is_alive() and w not in self.dead and w not in self.exited:
                    self.dead[w] = f"exit code {p.exitcode}"
                    self._release(w)  # a token it held while loading goes to the next worker
                    if key := self.busy.pop(w, None):
                        failed[key] = f"worker {w} died (exit code {p.exitcode})"
                        outstanding -= 1
                    self.log(f"[pool] worker {w} died (exit code {p.exitcode})")
            idle = [w for w in self.ready if w not in self.dead and w not in self.exited and w not in self.busy]
            if outstanding and not self.busy and idle and time.time() - last_msg > 120:
                # ready workers idle for 2 min with jobs out: those died with a worker before it said "start"
                failed["(lost)"] = f"{outstanding} jobs lost with a dead worker"
                outstanding = 0
        if on_fail:
            for key, err in failed.items():
                if key in by_key:
                    on_fail(by_key[key], err)
        t0 = max(self.t_ready, t_start) if self.t_ready else t_first  # a pool reused for a second round: from now
        wall = (t_last - t0) if (t_last and t0) else 0.0
        return {"backend": self.backend, "jobs": len(jobs), "done": n_done, "failed": len(failed),
                "failures": dict(list(failed.items())[:20]), "left": len(pending), "stopped": stopped,
                "workers": self.workers, "load_parallel": self.slots,
                "workers_ready": len(self.ready), "dead": dict(self.dead), "audio_s": round(audio, 1),
                "wall_s": round(wall, 1), "audio_per_wall": round(audio / wall, 3) if wall > 0 else None,
                "job_rtf_median": round(median(job_rtf), 3) if job_rtf else None,
                "busy_s": round(busy_s, 1), "gpu": self.gpu, "gpu_mem_peak_mib": self.mem_peak,
                "host_mem_peak_mib": self.host_peak, "rss_peak_mib": self.rss_peak, "worker_info": dict(self.ready)}

    def close(self, timeout: float = 60.0) -> None:
        """Stop the workers (frees their GPU memory): a None per worker (and to each one still waiting to load), then
        kill what is still there."""
        for w in self.procs:
            self.jobs.put(None)
            self.ctl[w].put(None)
        t = time.time() + timeout
        for p in self.procs.values():
            p.join(max(t - time.time(), 0.1))
        self.kill()

    def kill(self) -> None:
        for w, p in self.procs.items():
            if p.is_alive():
                p.kill()
                p.join(5)
            if w not in self.exited:  # a later run() finds no worker left
                self.dead.setdefault(w, f"stopped (exit code {p.exitcode})")
        for q in (self.jobs, *self.ctl.values()):
            q.cancel_join_thread()  # items nobody will take must not hold this process at exit
        self._stop.set()


def write_wav16(out: str | Path, chunks, sr: int) -> float:
    """Float samples in chunks (numpy arrays; lists of floats too) -> a mono 16-bit PCM WAV at `out`, as VoxSynth
    writes its clips (clipped to +-1): a .part file, renamed when whole. Returns its seconds."""
    import array
    import wave

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
    except ImportError:  # the tests' fake server sends lists
        np = None
    if np is not None:
        x = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks] or [np.zeros(0, np.float32)])
        n, pcm = len(x), np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    else:
        a = array.array("h", (int(round(max(-1.0, min(1.0, float(v))) * 32767.0)) for c in chunks for v in c))
        if sys.byteorder != "little":
            a.byteswap()
        n, pcm = len(a), a.tobytes()
    tmp = out.with_name(out.name + ".part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm)
    os.replace(tmp, out)  # a killed call leaves only a .part file
    return n / sr


def check_onset(wav, sr: int, text: str) -> dict:
    """onset.onset_flags of a clip; a detector error never costs the clip ({"covered": False, "error": ...})."""
    try:
        from onset import onset_flags

        return onset_flags(wav, sr, text)
    except Exception as e:  # noqa: BLE001
        return {"covered": False, "bad": False, "why": [], "error": f"{type(e).__name__}: {e}"[:200]}


def onset_info(first: dict, retries: int = 0, trimmed: bool = False) -> dict:
    """A clip's row fields of its onset check (first: the flags of its first whole try): onset_bad (None: its text
    is not covered), onset_why, onset_retries (tries made again for a flagged onset), onset_trimmed (still flagged
    after them: cut, onset.trim_onset)."""
    out = {"onset_bad": bool(first.get("bad")) if first.get("covered") else None,
           "onset_why": list(first.get("why") or []), "onset_retries": retries, "onset_trimmed": trimmed}
    if first.get("error"):
        out["onset_error"] = first["error"]
    return out


def clean_pauses(wav, sr: int):
    """render --pause-clean: pause_clean.clean of a synthesized clip, before refprep's level, the onset check and the
    scores -> (float32 audio, its row record "pause_clean": pauses, breaths, gated_s)."""
    from pause_clean import clean, row_record

    y, rep = clean(wav, sr)
    return y, row_record(rep)


def _levelled(chunks, sr: int, pause_clean: bool = False):
    """A Nano-vLLM try (one float chunk per generation step) as one clip (through clean_pauses first with pause_clean)
    at refprep's level, its active level before (dBFS; None when silent: render_corpus's emotion energy) and its
    "pause_clean" record (None without)."""
    import numpy as np
    from refprep import normalize_level

    x = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks] or [np.zeros(0, np.float32)])
    pc = None
    if pause_clean:
        x, pc = clean_pauses(x, sr)
    y, info = normalize_level(x, sr)
    return y, info["level_in"], pc


class NanoPool:
    """Synthesis on Nano-vLLM-VoxCPM (--backend nanovllm), with SynthPool's interface (run, close). The jobs are grouped
    by (LoRA, timesteps) (nano_groups: the timesteps are set per server, a LoRA is registered on it); each group gets
    one server (nanovllm_base.nano_server, as nanovllm_app.py's probe starts it), which encodes every reference clip
    once and takes the group's requests with at most `concurrency` in flight, and is stopped (its engine process gone,
    its GPU memory free) before the next group starts and before run() returns, so the scorers find the GPU free.
    A request is the official API's call on the job: its prompt as target_text (voice design / instruction in
    parentheses, as VoxSynth), cfg, seed (Nano-vLLM seeds each request on its own, whatever else is in the batch) and
    reference as ref_audio_latents (reference-only cloning; never prompt_latents / prompt_text; refprep.prep_ref's
    16 kHz, right-padded, levelled clip unless the job says ref_prep False), capped and retried as the official API
    does (NANO_TRIES, NANO_BAD_RATIO: seed + 1 per try, the last try kept). Each whole try goes through clean_pauses
    when the job says "pause_clean" (in a thread; info "pause_clean"), is levelled (refprep.normalize_level; its level
    before: info "level_in") and its onset checked (onset(wav, sr, text), default
    check_onset): a flagged one is made again with the next seed, at most NANO_TRIES times; still flagged, the best
    try is cut where its word starts
    (onset.trim_onset) and logged. The clip goes where VoxSynth writes it (write_wav16 at the server's sample rate).
    A group starts only with the jobs whose estimate fits before feed_until (nano_fit; none: not started), a request
    only if it could still finish by then; at hard_until the requests are cancelled and the server killed. A group
    whose server cannot take its LoRA (start or register_lora fails) is handed back: run()'s "lora_failed" {lora:
    error} (synthesize: the voxcpm backend instead). A server that does not start without a LoRA, or whose engine
    dies, ends the run (the rest is left for the next call).
    server(model, timesteps, max_num_seqs, lora_kwargs) and tokens(model) -> count(text): stand-ins for nano_server
    and nano_tokens (tests)."""

    def __init__(self, model: str, concurrency: int = NANO_CONCURRENCY, measured: list[dict] | None = None,
                 log: Callable[[str], None] = print, server: Callable | None = None,
                 tokens: Callable[[str], Callable[[str], int]] | None = None, onset: Callable | None = None):
        self.model, self.n, self.measured, self.log = model, max(1, int(concurrency)), measured or [], log
        self.server, self.tokens, self.count, self.mem_peak = server, tokens, None, None
        self.onset = onset or check_onset
        self.gpu = (gpu_query("name") or [None])[0]

    def close(self) -> None:
        """Nothing left to stop: run() stops every server it starts."""

    def _sample(self, stop: threading.Event) -> None:  # peak GPU memory while run() runs
        while not stop.wait(5):
            try:
                if v := gpu_query("memory.used"):
                    self.mem_peak = max(self.mem_peak or 0.0, float(v[0]))
            except ValueError:
                pass

    def run(self, jobs: list[dict], feed_until: float = math.inf, hard_until: float = math.inf,
            on_done: Callable[[dict, dict], None] | None = None,
            on_fail: Callable[[dict, str], None] | None = None) -> dict:
        """Make jobs (in order) group by group; on_done(job, info) as each clip is written, on_fail(job, error) at the
        end. Returns synthesis stats: SynthPool's keys, "groups" (one record per server) and "lora_failed"."""
        import asyncio

        stop = threading.Event()
        threading.Thread(target=self._sample, args=(stop,), daemon=True).start()
        try:
            return asyncio.run(self._run(jobs, feed_until, hard_until, on_done, on_fail))
        finally:
            stop.set()

    async def _run(self, jobs, feed_until, hard_until, on_done, on_fail) -> dict:
        failed: dict[str, str] = {}
        st = {"backend": "nanovllm", "jobs": len(jobs), "done": 0, "left": 0, "stopped": "", "workers": 1,
              "workers_ready": 0, "concurrency": self.n, "groups": [], "lora_failed": {}, "lora_jobs": 0}
        fatal = ""
        for (lora, ts), gjobs in nano_groups(jobs):
            if lora in st["lora_failed"]:  # the same LoRA at other timesteps: refused as well
                st["lora_jobs"] += len(gjobs)
                continue
            take = [] if fatal else nano_fit(gjobs, ts, feed_until - time.time(), self.measured)
            if not take:
                st["left"] += len(gjobs)
                st["stopped"] = fatal or "deadline: no time for more clips (the same command continues)"
                continue
            g = await self._group(lora, ts, take, feed_until, hard_until, on_done, failed)
            st["groups"].append(g)
            if "lora_error" in g:
                st["lora_failed"][lora] = g["lora_error"]
                st["lora_jobs"] += len(gjobs)
                continue
            st["done"] += g["done"]
            st["left"] += len(gjobs) - len(take) + g["left"]
            st["workers_ready"] = max(st["workers_ready"], int("ready_s" in g))
            if len(take) < len(gjobs) or g["left"]:
                st["stopped"] = st["stopped"] or "deadline: no time for more clips (the same command continues)"
            if g.get("error") or g.get("stopped"):
                fatal = st["stopped"] = g.get("stopped") or f"Nano-vLLM server: {g['error']}"
        by_key = {j["key"]: j for j in jobs}
        if on_fail:
            for key, err in failed.items():
                on_fail(by_key[key], err)
        audio, wall = sum(g["audio_s"] for g in st["groups"]), sum(g["wall_s"] for g in st["groups"])
        st.update(failed=len(failed), failures=dict(list(failed.items())[:20]), audio_s=round(audio, 1),
                  wall_s=round(wall, 1), audio_per_wall=round(audio / wall, 3) if wall > 0 else None,
                  server_s=round(sum(g.get("up_s", 0.0) for g in st["groups"]), 1), gpu=self.gpu,
                  gpu_mem_peak_mib=self.mem_peak)
        return st

    async def _group(self, lora: str | None, ts: int, jobs: list[dict], feed_until: float, hard_until: float,
                     on_done, failed: dict) -> dict:
        """One server: start (LoRA, references), all of jobs, stop. Returns its record (throughput.jsonl)."""
        import asyncio

        g = {"lora": lora, "timesteps": ts, "jobs": len(jobs), "ref_jobs": sum(bool(j.get("ref")) for j in jobs),
             "concurrency": self.n, "done": 0, "failed": 0, "left": 0, "audio_s": 0.0, "wall_s": 0.0}
        t0, server, name, before = time.time(), None, None, _gpu_used()
        self.log(f"[nano] server for {len(jobs)} clips, {ts} timesteps{', LoRA ' + lora if lora else ''}")
        stage, kw = "start", None
        try:
            try:
                ready_s = min(NANO_READY_S, feed_until - time.time() - NANO_STOP_S, hard_until - time.time())
                if ready_s <= 0:  # no time left to start a server whose clips could still be scored
                    g["left"] = len(jobs)
                    return g
                if self.count is None:
                    self.count = (self.tokens or nano_tokens)(self.model)
                kw = nano_lora_kwargs(lora_config(lora) or {}) if lora else None
                server = (self.server or nano_server)(self.model, ts, self.n, kw)
                await asyncio.wait_for(server.wait_for_ready(), ready_s)
                if lora:
                    stage = "register"
                    await server.register_lora("contest", lora)
                    name = "contest"
                stage = "info"
                info = await server.get_model_info()
                sr = int(info.get("output_sample_rate") or info["sample_rate"])
                patch = 4 * int(info.get("feat_dim") or 64) * int(info.get("patch_size") or 4)  # float32 bytes
            except Exception as e:  # noqa: BLE001  a server with a LoRA that fails goes back to the caller (its LoRA
                # entries to the official API); without one, or not ready in time, the run ends
                err = f"{type(e).__name__}: {e}"[:500]
                lora_fault = stage == "register" or (stage == "start" and kw is not None and "lora" in err.lower()
                                                     and not isinstance(e, TimeoutError))
                g["lora_error" if lora_fault else "error"] = err  # OOM, CUDA, timeouts: not the LoRA's fault
                self.log(f"[nano] server not usable: {err}")
                return g
            g["ready_s"] = round(time.time() - t0, 1)
            refs: dict[tuple[str, bool], bytes | str] = {}
            for p, prep in dict.fromkeys((j["ref"], j.get("ref_prep", True)) for j in jobs if j.get("ref")):
                try:  # reference-only cloning: the clip's latents, encoded once per server
                    if prep:
                        from refprep import prep_ref

                        data, fmt = prep_ref(p), "wav"
                    else:
                        data, fmt = Path(p).read_bytes(), Path(p).suffix.lstrip(".").lower()
                    refs[(p, prep)] = await server.encode_latents(data, fmt)
                except Exception as e:  # noqa: BLE001  (its jobs fail, the others go on)
                    refs[(p, prep)] = f"reference not encoded: {type(e).__name__}: {e}"[:500]
            self.log(f"[nano] ready in {g['ready_s']} s, {len(refs)} references")
            sem, dead, rtf = asyncio.Semaphore(self.n), [], _lin(NANO_REQ_RTF, ts)
            t_first, t_last = time.time(), None

            async def one(job: dict) -> None:
                nonlocal t_last
                async with sem:
                    if dead or time.time() + job["est_s"] * NANO_DUR * rtf > feed_until:
                        g["left"] += 1
                        return
                    ref = refs.get((job["ref"], job.get("ref_prep", True))) if job.get("ref") else None
                    t = time.time()
                    try:
                        if isinstance(ref, str):
                            raise RuntimeError(ref)
                        n_tok = self.count(job["text"])
                        prompt = n_tok + 1 + (len(ref) // patch + 2 if ref else 0)  # Nano-vLLM's max_model_len check
                        cap = max(1, min(int(n_tok * NANO_BAD_RATIO + 10), NANO_MAX_STEPS, MAX_MODEL_LEN - prompt))
                        best, first, runaway, redo = None, None, 0, 0
                        for k in range(2 * NANO_TRIES):  # bad cases: NANO_TRIES tries in all; onsets: NANO_TRIES more
                            chunks = []  # one per generation step
                            async for c in server.generate(target_text=job["text"], max_generate_length=cap,
                                                           cfg_value=job["cfg"], ref_audio_latents=ref, lora_name=name,
                                                           seed=job["seed"] + k):
                                chunks.append(c)
                            bad = len(chunks) >= n_tok * NANO_BAD_RATIO
                            runaway += bad
                            if bad and runaway < NANO_TRIES:
                                continue
                            if job.get("pause_clean"):  # CPU: off the event loop, the other requests stream on
                                wav, level, pc = await asyncio.to_thread(_levelled, chunks, sr, True)
                            else:
                                wav, level, pc = _levelled(chunks, sr)
                            fl = await asyncio.to_thread(self.onset, wav, sr, job["text"])
                            first = first or fl
                            if best is None or (bad, bool(fl.get("bad"))) <= best[:2]:  # not runaway, then no stray
                                best = (bad, bool(fl.get("bad")), wav, fl, level, pc)
                            if bad or not fl.get("bad") or redo >= NANO_TRIES:
                                break
                            redo += 1
                        bad, flagged, wav, fl, level, pc = best
                        trimmed = flagged and fl.get("cut_s") is not None
                        if trimmed:
                            from onset import PRE_S, trim_onset

                            wav = trim_onset(wav, sr, fl["cut_s"])
                        if flagged:
                            self.log(f"[nano] {job['key']}: stray onset ({', '.join(fl.get('why') or [])}) after "
                                     f"{redo} more tries: " + (f"cut before {max(0.0, fl['cut_s'] - PRE_S):.2f} s"
                                                                if trimmed else "kept, no cut point"))
                        dur = write_wav16(job["out"], [wav], sr)
                    except Exception as e:  # noqa: BLE001  (this request; a dead engine: the rest is left)
                        if "WorkerDied" in type(e).__name__:
                            dead.append(e)
                            g["left"] += 1
                        else:
                            failed[job["key"]] = f"{type(e).__name__}: {e}"[:500]
                            g["failed"] += 1
                        return
                    t_last = time.time()
                    info = {"duration": round(dur, 3), "synth_seconds": round(t_last - t, 3), "engine": "nanovllm",
                            **onset_info(first or {}, redo, trimmed), "level_in": level}
                    if pc is not None:
                        info["pause_clean"] = pc
                    if k:
                        info["tries"] = k + 1
                    if bad:
                        info["bad_case"] = True  # still runaway after NANO_TRIES: kept, as the official API keeps it
                    g["done"] += 1
                    g["audio_s"] += dur
                    if g["done"] % 50 == 0:
                        self.log(f"[nano] {g['done']}/{len(jobs)} clips ({time.time() - t_first:.0f} s)")
                    if on_done:
                        on_done(job, info)

            tasks = [asyncio.ensure_future(one(j)) for j in jobs]
            _, pending = await asyncio.wait(tasks, timeout=max(0.0, hard_until - time.time())
                                            if math.isfinite(hard_until) else None)
            if pending:
                for x in pending:
                    x.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                g["left"] += len(pending)
                g["stopped"] = "hard deadline: requests cancelled, server killed"
            if dead:
                g["error"] = f"engine died: {type(dead[0]).__name__}: {dead[0]}"[:500]
            g["wall_s"] = round((t_last or t_first) - t_first, 1)
            g["audio_s"] = round(g["audio_s"], 1)
            g["audio_per_wall"] = round(g["audio_s"] / g["wall_s"], 3) if g["wall_s"] > 0 else None
            return g
        finally:
            if server is not None:
                await self._stop(server, kill=bool(g.get("stopped")))
                if not g.get("stopped"):  # after the hard deadline no scorer follows: no waiting
                    await self._freed(before)
            g["up_s"] = round(time.time() - t0, 1)
            self.log(f"[nano] server stopped after {g['up_s']} s: {g['done']} clips, {g['failed']} failed, "
                     f"{g['left']} left, {g['audio_s']} s audio ({g.get('audio_per_wall')} audio s per GPU s)")

    async def _stop(self, server, kill: bool = False) -> None:
        """Stop a server so that its GPU memory is free: kill its engine first at the hard deadline, else server.stop()
        (graceful), then kill whatever engine process is still alive."""
        import asyncio

        procs = [p for s in (getattr(server, "servers", None) or [server]) if (p := getattr(s, "process", None))]
        if kill:  # hard deadline: no graceful stop (it would outlast the grace before the outer kill)
            for p in procs:
                p.kill()
                p.join(5)
            return
        try:
            await asyncio.wait_for(server.stop(), NANO_STOP_S + 20)
        except Exception as e:  # noqa: BLE001
            self.log(f"[nano] server stop: {type(e).__name__}: {e}")
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join(10)

    async def _freed(self, before: float | None, wait_s: int = 30) -> None:
        """Wait (at most wait_s) until the GPU memory in use is back near what it was before the server started, so the
        next server or a scorer finds it free (nvidia-smi; nothing to wait for without one)."""
        import asyncio

        for _ in range(wait_s):
            if before is None or (used := _gpu_used()) is None or used <= before + 1024:
                return
            await asyncio.sleep(1)
        self.log(f"[nano] {used:.0f} MiB of GPU memory still in use after the server stopped ({before:.0f} before)")


def _gpu_used() -> float | None:
    """MiB of GPU memory in use (nvidia-smi), None without one."""
    try:
        return float(gpu_query("memory.used")[0])
    except (IndexError, ValueError):
        return None


def log_throughput(path: str, mode: str, ts: int, stats: dict) -> None:
    """Lines of throughput.jsonl (what the cost estimates use) for what made enough audio to tell: one per pool run
    (voxcpm), one per Nano-vLLM server (its timesteps, reference mode, requests, concurrency, start-up, audio, wall)."""
    if not path:
        return
    if stats.get("backend") == "nanovllm":
        recs = [{"time": now_iso(), "mode": mode, "backend": "nanovllm", "timesteps": g["timesteps"],
                 "ref": 2 * g["ref_jobs"] >= g["jobs"], "lora": bool(g.get("lora")),
                 **{k: g.get(k) for k in ("jobs", "done", "concurrency", "ready_s", "up_s", "audio_s", "wall_s",
                                          "audio_per_wall")}, "gpu": stats.get("gpu"),
                 "gpu_mem_peak_mib": stats.get("gpu_mem_peak_mib")}
                for g in stats.get("groups") or [] if (g.get("audio_s") or 0) >= 30 and g.get("wall_s")]
    elif (stats.get("audio_s") or 0) >= 30 and stats.get("wall_s"):
        recs = [{"time": now_iso(), "mode": mode, "backend": stats.get("backend", "voxcpm"), "timesteps": ts,
                 "workers": stats["workers"],
                 **{k: stats.get(k) for k in ("load_parallel", "workers_ready", "audio_s", "wall_s", "audio_per_wall",
                                              "job_rtf_median", "gpu", "gpu_mem_peak_mib", "host_mem_peak_mib",
                                              "rss_peak_mib")}}]
    else:
        recs = []
    if not recs:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in recs))
    except OSError as e:
        print(f"[contest] throughput not logged: {e}", flush=True)


def read_measured(path: str) -> list[dict]:
    return read_rows(Path(path)) if path else []


# ---- scoring ------------------------------------------------------------------------------------------------------


class Scorers:
    """Whisper (WER/CER, language-ID), UTMOS and ECAPA, each loaded once on first use, in this process."""

    def __init__(self, device: str = "auto"):
        self.device, self._w, self._u, self._e = device, None, None, None

    def loaded(self, name: str) -> bool:
        return {"whisper": self._w, "utmos": self._u, "ecapa": self._e}[name] is not None

    def dev(self) -> str:
        if self.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def whisper(self):
        if self._w is None:
            from faster_whisper import WhisperModel

            cuda = self.dev() == "cuda"
            self._w = WhisperModel(WHISPER, device=self.dev(), compute_type="float16" if cuda else "int8")
        return self._w

    def utmos(self):
        if self._u is None:
            from score_utmos import load_predictor

            self._u = load_predictor(self.dev())
        return self._u

    def ecapa(self):
        if self._e is None:
            import torchaudio
            from huggingface_hub import snapshot_download

            if not hasattr(torchaudio, "list_audio_backends"):  # torchaudio >= 2.9 dropped it; speechbrain 1.0 calls it
                torchaudio.list_audio_backends = lambda: ["soundfile"]  # (clips are loaded with librosa here)
            from speechbrain.inference.classifiers import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy

            # on the Volume in the container (TTS_DATA set); elsewhere the HF cache
            d = snapshot_download(ECAPA_REPO, revision=ECAPA_REV, local_dir=ECAPA_DIR if os.environ.get("TTS_DATA")
                                  else None)
            # source = savedir = the downloaded copy, NO_LINK: speechbrain uses the files where they are, fetches,
            # copies and symlinks nothing (label_encoder.txt would be symlinked to .ckpt: not on Windows / Volumes)
            self._e = EncoderClassifier.from_hparams(source=d, savedir=d, overrides={"pretrained_path": d},
                                                     run_opts={"device": self.dev()},
                                                     local_strategy=LocalStrategy.NO_LINK)
        return self._e

    def embed(self, path: str | Path):
        """Unit ECAPA embedding (192) of a clip at 16 kHz."""
        import librosa
        import torch

        y, _ = librosa.load(str(path), sr=16000, mono=True)
        with torch.inference_mode():
            e = self.ecapa().encode_batch(torch.from_numpy(y).unsqueeze(0)).reshape(-1).float()
        return torch.nn.functional.normalize(e, dim=0).cpu()

    def close(self) -> None:
        self._w = self._u = self._e = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def no_time(sc: Scorers, name: str, until: float, label: str, n: int) -> bool:
    """True (and printed) when `until` leaves no time to load scorer `name` (unless it is loaded) and score a clip: the
    n clips it would score are left for the next call."""
    if time.time() + (0.0 if sc.loaded(name) else LOAD_S[name]) + SCORE_CLIP_S <= until:
        return False
    print(f"[{name}] {label}: no time left{'' if sc.loaded(name) else ' to load it'}, {n} left for the next call",
          flush=True)
    return True


def errors_from_hyp(rows: list[dict]) -> None:
    """errors / ref_words / char_errors / ref_chars of every row with a hyp (cheap; refs may have changed)."""
    for r in rows:
        if "hyp" in r:
            r["errors"], r["ref_words"], r["char_errors"], r["ref_chars"] = wer_cer(r["refs"], r["hyp"])


def whisper_pass(sc: Scorers, rows: list[dict], wav_of, label: str, rescore: bool = False,
                 until: float = math.inf, words: bool = False) -> int:
    """hyp (language tr, beam 5; word timestamps for soru rows, every row with words), lang and p_tr (Whisper's
    language-ID, not forced), then the error counts. Returns how many rows it left for lack of time."""
    todo = [r for r in rows if wav_of(r) is not None and (rescore or "hyp" not in r)]
    left = 0
    if todo and no_time(sc, "whisper", until, label, len(todo)):
        left, todo = len(todo), []
    if todo:
        from faster_whisper import decode_audio

        model, t0 = sc.whisper(), time.time()
        for i, r in enumerate(todo):
            if time.time() > until:
                left = len(todo) - i
                print(f"[whisper] {label}: deadline, {left} rows left for the next call", flush=True)
                break
            try:
                audio = decode_audio(str(wav_of(r)), sampling_rate=16000)
                lang, _, probs = model.detect_language(audio)
                timed = words or r["set"] == "soru"  # score_intonation's word timing; render_corpus's edges
                segs, _ = model.transcribe(audio, language="tr", beam_size=5, word_timestamps=timed)
                segs = list(segs)
                r["hyp"] = " ".join(s.text.strip() for s in segs).strip()
                r["lang"], r["p_tr"] = lang, round(float(dict(probs).get("tr", 0.0)), 4)
                if timed:
                    r["words"] = [[w.word.strip(), round(w.start, 3), round(w.end, 3)]
                                  for s in segs for w in (getattr(s, "words", None) or [])]
                r.pop("whisper_error", None)
            except Exception as e:  # one bad clip costs one row
                r["whisper_error"] = f"{type(e).__name__}: {e}"
                print(f"[whisper] {label}: {r['set']}/{r['id']}: {r['whisper_error']}", flush=True)
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f"[whisper] {label}: {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    errors_from_hyp(rows)
    return left


def utmos_pass(sc: Scorers, rows: list[dict], wav_of, label: str, rescore: bool = False,
               until: float = math.inf) -> int:
    from score_utmos import score_rows

    todo = [r for r in rows if wav_of(r) is not None and (rescore or r.get("utmos") is None)]
    if todo and no_time(sc, "utmos", until, label, len(todo)):
        return len(todo)
    for i in range(0, len(todo), 25):
        if time.time() > until:
            print(f"[utmos] {label}: deadline, {len(todo) - i} rows left for the next call", flush=True)
            return len(todo) - i
        score_rows(sc.utmos(), todo[i:i + 25], wav_of, sc.dev(), label)
    return 0


def sim_pass(sc: Scorers, groups: list[tuple[str, list[dict], Path | None]], wav_of, rescore: bool = False,
             until: float = math.inf, self_sim: bool = True) -> int:
    """sim (cosine to the group's reference clip) and self_sim (cosine to the centroid of the group's clips, >= 3) of
    every clip; groups: (label, rows, reference or None). Returns how many groups it left for lack of time."""
    for gi, (label, rows, ref) in enumerate(groups):
        have = [r for r in rows if wav_of(r) is not None]
        if not have or not (rescore or any((ref is not None and "sim" not in r) or (self_sim and "self_sim" not in r)
                                           for r in have)):
            continue
        if no_time(sc, "ecapa", until, label, len(groups) - gi):  # groups
            return len(groups) - gi
        sc.ecapa()  # a model that does not load fails the pass here, not once per clip
        ref_e, ref_err = None, None
        try:
            ref_e = sc.embed(ref) if ref is not None else None
        except Exception as e:  # noqa: BLE001
            ref_err = f"reference not embedded: {type(e).__name__}: {e}"
            print(f"[sim] {label}: {ref}: {ref_err}", flush=True)
        embs = {}
        for r in have:
            try:
                embs[id(r)] = sc.embed(wav_of(r))
                r.pop("sim_error", None)
            except Exception as e:  # noqa: BLE001
                r["sim_error"] = f"{type(e).__name__}: {e}"
        if ref_err:
            for r in have:
                r.setdefault("sim_error", ref_err)
        if ref_e is not None:
            for r in have:
                if id(r) in embs:
                    r["sim"] = round(float(ref_e @ embs[id(r)]), 4)
        if self_sim and len(embs) >= 3:
            import torch

            c = torch.nn.functional.normalize(torch.stack(list(embs.values())).mean(0), dim=0)
            for r in have:
                if id(r) in embs:
                    r["self_sim"] = round(float(c @ embs[id(r)]), 4)
        print(f"[sim] {label}: {len(embs)} clips", flush=True)
    return 0


def prosody(path: str | Path, model_text: str) -> dict:
    """Prosody of one clip: F0 median and 5-95 % range in st re 100 Hz (pyin 60-600 Hz, quarter-semitone bins, 10 ms;
    octave jumps inside the clip folded by score_intonation.fold_octaves; None under MIN_VOICED voiced frames), speech
    rate (syllables per second of speech: frames above score_intonation.SIL_DB re the loudest) and loudness (dB re full
    scale of the mean RMS of the speech frames). Not score_intonation.f0_track: its first pass sets the range of the
    second and, on short clips, can put the whole clip on a harmonic; a clip that is an octave off as a whole is folded
    back at the entry level (emotion_summary)."""
    import librosa
    import numpy as np
    import score_intonation as si

    y = si.load_audio(path)
    f0, vf, _ = librosa.pyin(y, fmin=60.0, fmax=600.0, sr=si.SR, frame_length=si.FRAME, hop_length=si.HOP,
                             resolution=0.25)
    rms = librosa.feature.rms(y=y, frame_length=si.FRAME, hop_length=si.HOP)[0]
    n = min(len(f0), len(rms))
    f0, vf, rms = f0[:n], vf[:n], rms[:n]
    speech = 20 * np.log10(np.maximum(rms, 1e-10) / max(float(rms.max()), 1e-10)) > si.SIL_DB
    if not speech.any():
        raise ValueError("no speech found")
    speech_s = float(speech.sum()) * si.HOP_S
    idx = np.flatnonzero(vf & np.isfinite(f0) & speech)
    rnd = lambda x: round(float(x), 3)  # noqa: E731
    out = {"voiced": int(len(idx)), "speech_s": rnd(speech_s),
           "rate_syl_s": rnd(sum(si.syllables(w) for w in model_text.split()) / max(speech_s, 0.1)),
           "energy_db": rnd(20 * np.log10(max(float(rms[speech].mean()), 1e-10))),
           "f0_median_st": None, "f0_range_st": None}
    if len(idx) >= MIN_VOICED:
        st = si.fold_octaves(12 * np.log2(f0[idx] / 100.0), idx)
        lo, hi = np.percentile(st, [5, 95])
        out.update(f0_median_st=rnd(np.median(st)), f0_range_st=rnd(hi - lo))
    return out


def fold_to(x: float | None, base: float | None, lo: float = -5.0, hi: float = 9.0) -> float | None:
    """x moved by whole octaves (12 st) into [base + lo, base + hi): undoes a clip whose pitch track sat on a harmonic
    or subharmonic throughout; a real shift inside that window (an emotion raising the voice by up to 9 st, lowering it
    by up to 5) stays."""
    if x is None or base is None:
        return x
    while x >= base + hi:
        x -= 12.0
    while x < base + lo:
        x += 12.0
    return round(x, 3)


CPU_KEYS = ("set", "id", "category", "text", "model_text", "prompt", "pattern", "peak", "peak_word", "pair_id",
            "words")


def _cpu_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"


def _cpu_job(item: tuple) -> tuple:
    """Prosody, (soru rows) intonation and (rows without one) the onset check of one clip, in a CPU process."""
    key, row, path, want_inton, want_pros, want_onset = item
    out = {}
    if want_onset:
        try:
            import soundfile as sf

            x, sr = sf.read(path, dtype="float32")
            info = onset_info(check_onset(x, sr, row.get("prompt") or row["model_text"]))
            out["onset"] = {k: info[k] for k in ("onset_bad", "onset_why", "onset_error") if k in info}
        except Exception as e:  # noqa: BLE001  (an unreadable clip)
            out["onset"] = {"onset_bad": None, "onset_error": f"{type(e).__name__}: {e}"[:200]}
    if want_pros:
        try:
            out["prosody"] = prosody(path, row["model_text"])
        except Exception as e:  # noqa: BLE001  (no voiced speech, ...)
            out["prosody_error"] = f"{type(e).__name__}: {e}"
    if want_inton:
        try:
            import score_intonation as si

            out["intonation"] = si.score_row(row, path)
        except Exception as e:  # noqa: BLE001
            out["intonation_error"] = f"{type(e).__name__}: {e}"
    return key, out


def cpu_items(groups: list[list[dict]], wav_of_g: list, rescore: bool = False) -> list[tuple]:
    """Work of the CPU pass: ((group, row index), copy of the row fields it needs, wav, intonation?, prosody?, onset?
    (only for a clip without one: a synthesis-time check also knows its first try, retries and cut))."""
    items = []
    for g, (rows, wav_of) in enumerate(zip(groups, wav_of_g)):
        for i, r in enumerate(rows):
            p = wav_of(r)
            if p is None:
                continue
            inton = r["set"] == "soru" and (rescore or "intonation" not in r)
            pros = rescore or "prosody" not in r
            ons = "onset_bad" not in r
            if inton or pros or ons:
                fields = {k: (list(r[k]) if isinstance(r[k], list) else r[k]) for k in CPU_KEYS if k in r}
                items.append(((g, i), fields, str(p), inton, pros, ons))
    return items


def apply_cpu(groups: list[list[dict]], results) -> None:
    for (g, i), out in results:
        r = groups[g][i]
        r.update(out.get("onset") or {})
        for k in ("prosody", "intonation"):
            if k in out:
                r[k] = out[k]
                r.pop(f"{k}_error", None)
            elif f"{k}_error" in out:
                r[f"{k}_error"] = out[f"{k}_error"]


class CpuPass:
    """The CPU pass in a process pool, beside the GPU scorers (procs 0: here, at get())."""

    def __init__(self, items: list[tuple], procs: int):
        self.items, self.pool, self.res = items, None, []
        if items and procs > 0:
            import multiprocessing as mp

            self.pool = mp.get_context("spawn").Pool(min(procs, len(items)), initializer=_cpu_init)
            self.res = [self.pool.apply_async(_cpu_job, (x,)) for x in items]  # one by one: a deadline keeps the done

    def get(self, until: float = math.inf) -> list:
        """The results finished by `until` (the rest: the next call); the pool is gone after."""
        out = []
        if self.pool is None:
            for x in self.items:
                if time.time() > until:
                    break
                out.append(_cpu_job(x))
        else:
            try:
                for r in self.res:
                    if (left := until - time.time()) <= 0:
                        break
                    r.wait(min(left, 3600.0))
                out = [r.get() for r in self.res if r.ready() and r.successful()]
            finally:
                if all(r.ready() for r in self.res):
                    self.pool.close()
                else:
                    self.pool.terminate()
                self.pool.join()
        if len(out) < len(self.items):
            print(f"[cpu] deadline: {len(self.items) - len(out)} clips left for the next call", flush=True)
        return out


# ---- summaries ----------------------------------------------------------------------------------------------------


def group_wer(rows: list[dict]) -> dict:
    """WER per set and set/category (synth_eval's layout) and "all", over the rows with error counts."""
    groups = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        if r.get("ref_words") is None:
            continue
        for key in ("all", r["set"], f"{r['set']}/{r['category']}"):
            g = groups[key]
            g[0] += r["errors"]
            g[1] += r["ref_words"]
            g[2] += 1
            g[3] += r["errors"] == 0
    return {k: {"wer": round(100 * e / max(n, 1), 2), "rows": c, "rows_correct": ok}
            for k, (e, n, c, ok) in sorted(groups.items())}


def lang_summary(rows: list[dict]) -> dict:
    ps = [r["p_tr"] for r in rows if r.get("p_tr") is not None]
    if not ps:
        return {}
    by = defaultdict(list)
    for r in rows:
        if r.get("p_tr") is not None:
            by[r["set"]].append(r["p_tr"])
    return {"p_tr_mean": mean(ps), "p_tr_min": round(min(ps), 4), "share_ok": round(sum(p >= P_TR_OK for p in ps) /
            len(ps), 3), "not_tr": sum(r.get("lang") not in (None, "tr") for r in rows), "rows": len(ps),
            "by_set": {k: mean(v) for k, v in sorted(by.items())}}


def emotion_summary(rows: list[dict]) -> dict:
    """Per emotion: mean prosody difference of its rows to the entry's neutral median (trzor rows; else the soru
    declaratives; F0 medians folded to the neutral octave first, fold_to) and "agree", the share of (row, expected
    direction) pairs it meets (EXPECTED); overall "agree"."""
    base_rows = [r for r in rows if r["set"] == "trzor" and r.get("prosody")] or [
        r for r in rows if r.get("pattern") == "fall" and r.get("prosody")]
    emo = [r for r in rows if r.get("emotion") and r.get("prosody") and r["set"] != "trzor"]
    if not base_rows or not emo:
        return {}
    raw = median(r["prosody"].get("f0_median_st") for r in base_rows)  # most neutral clips sit in the right octave
    pros = lambda r: {**r["prosody"], "f0_median_st": fold_to(r["prosody"].get("f0_median_st"), raw)}  # noqa: E731
    base = {f: median(pros(r).get(f) for r in base_rows) for f in PROSODY}
    by = defaultdict(list)
    for r in emo:
        by[r["emotion"]].append(r)
    out, hits, total = {}, 0, 0
    for e, rs in sorted(by.items()):
        deltas = [{f: p[f] - base[f] for f in PROSODY if p.get(f) is not None and base[f] is not None}
                  for p in (pros(r) for r in rs)]
        g = {"rows": len(rs), **{f: mean(d.get(f) for d in deltas) for f in PROSODY}}
        if e in EXPECTED:
            pairs = [d[f] * s > 0 for d in deltas for f, s in EXPECTED[e].items() if f in d]
            g["agree"] = round(sum(pairs) / len(pairs), 3) if pairs else None
            hits, total = hits + sum(pairs), total + len(pairs)
        out[e] = g
    return {"baseline": {f: (round(v, 3) if v is not None else None) for f, v in base.items()},
            "baseline_rows": len(base_rows), "by_emotion": out, "agree": round(hits / total, 3) if total else None}


def onset_summary(rows: list[dict]) -> dict | None:
    """Stray onsets of an entry's clips (onset.py; rows with onset_bad not None: texts it covers): how many were
    flagged at their first try (rate), made again (retried), cut (trimmed); None when no clip was checked."""
    cov = [r for r in rows if r.get("onset_bad") is not None]
    if not cov:
        return None
    bad = sum(bool(r["onset_bad"]) for r in cov)
    why = defaultdict(int)
    for r in cov:
        for w in r.get("onset_why") or []:
            why[w] += 1
    return {"covered": len(cov), "stray": bad, "rate": round(bad / len(cov), 3),
            "retried": sum(bool(r.get("onset_retries")) for r in cov),
            "trimmed": sum(bool(r.get("onset_trimmed")) for r in cov), "why": dict(sorted(why.items())),
            "not_covered": sum(1 for r in rows if "onset_bad" in r and r["onset_bad"] is None)}


def entry_summary(rows: list[dict], meta: dict) -> dict:
    """summary.json of one entry: spec, settings, coverage and every metric of its clips."""
    have = [r for r in rows if r.get("duration") is not None]
    synth = [r for r in have if r.get("synth_seconds")]
    s = {"entry": meta.get("name"), **{k: meta.get(k) for k in ("kind", "ref", "lora", "sim_ref", "description",
                                                                 "gender", "notes", "source", "settings")},
         "clips": len(rows), "synthesized": len(have), "audio_s": round(sum(r["duration"] for r in have), 1),
         "rtf_job": round(sum(r["synth_seconds"] for r in synth) / max(sum(r["duration"] for r in synth), 1e-9), 3)
         if synth else None,
         "wer": group_wer(rows), "lang": lang_summary(rows), "utmos": utmos_summary(rows)}
    cer = [r for r in rows if r.get("ref_chars")]
    s["cer"] = round(100 * sum(r["char_errors"] for r in cer) / sum(r["ref_chars"] for r in cer), 2) if cer else None
    sims = [r["sim"] for r in rows if r.get("sim") is not None]
    s["sim"] = {"mean": mean(sims), "min": round(min(sims), 4), "rows": len(sims)} if sims else None
    s["self_sim"] = mean(r.get("self_sim") for r in rows)
    if any(r.get("intonation") for r in rows):
        from score_intonation import intonation_summary

        s["intonation"] = intonation_summary(rows)
    s["emotion"] = emotion_summary(rows)
    s["onset"] = onset_summary(rows)
    s["pause_clean"] = pause_summary(rows)
    missing = {k: sum(1 for r in have if r.get(k) is None) for k in ("hyp", "utmos", "prosody")}
    missing["sim"] = sum(1 for r in have if r.get("sim") is None) if meta.get("sim_file") else 0
    missing["intonation"] = sum(1 for r in have if r["set"] == "soru" and r.get("intonation") is None)
    s["unscored"] = {k: v for k, v in missing.items() if v}
    s["complete"] = len(have) == len(rows) and not s["unscored"]
    return s


def headline(s: dict) -> dict:
    """One leaderboard line of an entry summary."""
    w, lang, inton = s.get("wer") or {}, s.get("lang") or {}, (s.get("intonation") or {}).get("soru") or {}
    sets = [v for k, v in (s.get("utmos") or {}).items() if "/" not in k]  # per-set means -> mean over all clips
    n_ut = sum(v["rows"] for v in sets)
    ut_all = round(sum(v["mean"] * v["rows"] for v in sets) / n_ut, 3) if n_ut else None
    return {"entry": s.get("entry"), "kind": s.get("kind"), "lora": bool(s.get("lora")),
            "wer_trzor": (w.get("trzor") or {}).get("wer"), "wer_all": (w.get("all") or {}).get("wer"),
            "rows_correct": f"{(w.get('all') or {}).get('rows_correct', 0)}/{(w.get('all') or {}).get('rows', 0)}",
            "cer": s.get("cer"), "p_tr": lang.get("p_tr_mean"), "p_tr_ok": lang.get("share_ok"), "utmos": ut_all,
            "sim": (s.get("sim") or {}).get("mean"), "self_sim": s.get("self_sim"),
            "q_pass": inton.get("questions_pass_rate"), "auc": inton.get("auc_peak"),
            "emo_agree": (s.get("emotion") or {}).get("agree"), "onset": (s.get("onset") or {}).get("rate"),
            "clips": f"{s.get('synthesized')}/{s.get('clips')}",
            "complete": s.get("complete"), "lang_ok": (lang.get("p_tr_mean") or 0) >= P_TR_OK}


def leaderboard(summaries: list[dict]) -> tuple[list[dict], str]:
    """Headlines sorted by UTMOS, the entries whose mean p(tr) is >= P_TR_OK first; and the same as a Markdown table."""
    lines = sorted((headline(s) for s in summaries), key=lambda h: (not h["lang_ok"], -(h["utmos"] or 0), h["entry"]))
    cols = [("entry", "entry"), ("kind", "kind"), ("lora", "LoRA"), ("utmos", "UTMOS"), ("wer_trzor", "WER TR-Zor"),
            ("wer_all", "WER all"), ("rows_correct", "rows ok"), ("cer", "CER"), ("p_tr", "p(tr)"),
            ("p_tr_ok", f"p(tr)>={P_TR_OK}"), ("sim", "sim"), ("self_sim", "self-sim"), ("q_pass", "questions"),
            ("auc", "AUC"), ("emo_agree", "emotion"), ("onset", "onset stray"), ("clips", "clips")]
    fmt = lambda v: "-" if v is None else ("yes" if v is True else "" if v is False else  # noqa: E731
                                           f"{v:.3f}" if isinstance(v, float) else str(v))
    md = ["| " + " | ".join(t for _, t in cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(h[k]) + ("" if k != "entry" or h["complete"] else " (incomplete)") for k, _ in cols)
           + " |" for h in lines]
    note = (f"\nUTMOS: mean over all clips. WER/CER %: Whisper {WHISPER}, trnorm forms. p(tr): Whisper language-ID "
            f"(not forced); entries below {P_TR_OK} come last. sim: ECAPA cosine to the reference; self-sim: to the "
            "entry's own centroid. questions / AUC: score_intonation on the soru rows. emotion: share of expected "
            "prosody directions met. onset stray: share of the clips of texts starting with a vowel, a voiced "
            "consonant or y whose first try had a stray onset (onset.py; Nano-vLLM made those again, summary.json "
            "\"onset\").\n")
    srcs = sorted(((s["entry"], s["source"]) for s in summaries if isinstance(s.get("source"), dict)),
                  key=lambda t: t[0])
    if srcs:
        note += "\nReference voices (reference-only cloning; sources.json next to each entry's clips):\n\n" + "".join(
            f"- {n}: {src.get('license')}; {src.get('attribution')}\n" for n, src in srcs)
    return lines, "\n".join(md) + "\n" + note


# ---- screen ranking -----------------------------------------------------------------------------------------------


def cer_of(r: dict) -> float | None:
    return r["char_errors"] / max(r["ref_chars"], 1) if r.get("ref_chars") is not None else None


def select_for_clone(cands: list[dict], top: int, per_desc: int = 1, max_cer: float = 0.02, min_p_tr: float = 0.0,
                     dur_range: tuple[float, float] = (0.0, math.inf)) -> list[dict]:
    """The candidates that get a clone test: scored, CER <= max_cer, p(tr) >= min_p_tr, duration in dur_range; the best
    by UTMOS, at most per_desc per description, top of them."""
    ok = [c for c in cands if c.get("utmos") is not None and (cer_of(c) is not None and cer_of(c) <= max_cer)
          and (c.get("p_tr") or 0.0) >= min_p_tr and dur_range[0] <= (c.get("duration") or 0.0) <= dur_range[1]]
    ok.sort(key=lambda c: (-c["utmos"], c["id"]))
    out, per = [], defaultdict(int)
    for c in ok:
        if len(out) >= top:
            break
        if per[c["desc_id"]] < per_desc:
            out.append(c)
            per[c["desc_id"]] += 1
    return out


def rank_voices(chosen: list[dict], clones: list[dict], n_clone: int, need_cer0: int = 3, min_sim: float = 0.4,
                min_p_tr: float = 0.9) -> list[dict]:
    """Clone-test result per chosen candidate, ranked: eligible first (all n_clone clones scored, CER 0 on >= need_cer0,
    mean sim >= min_sim, mean p(tr) >= min_p_tr), then by the MEAN clone UTMOS, then sim."""
    by = defaultdict(list)
    for r in clones:
        by[r["cand"]].append(r)
    out = []
    for c in chosen:
        cl = [r for r in by[c["id"]] if r.get("utmos") is not None and r.get("ref_chars") is not None]
        ut = [r["utmos"] for r in cl]
        sims = [r["sim"] for r in cl if r.get("sim") is not None]
        v = {"cand": c["id"], "desc_id": c["desc_id"], "seed": c.get("seed"), "gender": c.get("gender", ""),
             "description": c.get("description"), "cand_utmos": c.get("utmos"), "clones": len(cl),
             "utmos_mean": mean(ut), "utmos_min": round(min(ut), 4) if ut else None,
             "cer0": sum(r["char_errors"] == 0 for r in cl), "sim_mean": mean(sims),
             "sim_min": round(min(sims), 4) if sims else None, "p_tr_mean": mean(r.get("p_tr") for r in cl)}
        why = []
        if len(cl) < n_clone:
            why.append(f"{len(cl)}/{n_clone} clones scored")
        if v["cer0"] < need_cer0:
            why.append(f"CER 0 on {v['cer0']} clones (needs {need_cer0})")
        if v["sim_mean"] is None or v["sim_mean"] < min_sim:
            why.append(f"similarity {v['sim_mean']} < {min_sim}")
        if v["p_tr_mean"] is None or v["p_tr_mean"] < min_p_tr:
            why.append(f"p(tr) {v['p_tr_mean']} < {min_p_tr}")
        v["eligible"], v["why_not"] = not why, why
        out.append(v)
    out.sort(key=lambda v: (not v["eligible"], -(v["utmos_mean"] if v["utmos_mean"] is not None else -9.0),
                            -(v["sim_mean"] or 0.0), v["cand"]))
    for i, v in enumerate(out, 1):
        v["rank"] = i
    return out


def pick_voices(ranked: list[dict], keep: int, min_per_gender: int = 1) -> list[dict]:
    """The keep best eligible voices, with at least min_per_gender of each gender among the eligible (as far as keep
    allows), in rank order."""
    ok = [v for v in ranked if v["eligible"]]
    picks = []
    for g in sorted({v["gender"] for v in ok if v["gender"]}):
        picks += [v for v in ok if v["gender"] == g][:min_per_gender]
    for v in ok:
        if len(picks) >= keep:
            break
        if v not in picks:
            picks.append(v)
    return sorted(picks, key=lambda v: v["rank"])[:keep]


def designed_entries(picks: list[dict], cand_wav: Callable[[str], str]) -> list[dict]:
    """A render spec of the picked voices: each clones its candidate clip."""
    return [{"name": f"designed-{v['cand']}", "kind": "designed", "ref": cand_wav(v["cand"]),
             "description": v["description"], **({"gender": v["gender"]} if v["gender"] else {}),
             "notes": f"screen rank {v['rank']}: clone UTMOS mean {v['utmos_mean']} (min {v['utmos_min']}), CER 0 on "
                      f"{v['cer0']}/{v['clones']}, sim {v['sim_mean']}, p(tr) {v['p_tr_mean']}; candidate UTMOS "
                      f"{v['cand_utmos']}"} for v in picks]


def ranking_md(ranked: list[dict], picks: list[dict], params: dict, missing: int = 0) -> str:
    keep = {v["cand"] for v in picks}
    lines = [f"# Designed voices: clone test ({params.get('n_clone')} clones each)", ""]
    if missing:
        lines += [f"PROVISIONAL: {missing} clips not made or scored yet (ranking.json \"missing\"); no entries.json "
                  "until the same contest-screen command has finished them.", ""]
    lines += ["| rank | voice | gender | clone UTMOS mean | min | CER 0 | sim | p(tr) | cand UTMOS | picked | why not "
              "|", "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)  # noqa: E731
    for v in ranked:
        lines.append(f"| {v['rank']} | {v['cand']} | {v['gender']} | {f(v['utmos_mean'])} | {f(v['utmos_min'])} | "
                     f"{v['cer0']}/{v['clones']} | {f(v['sim_mean'])} | {f(v['p_tr_mean'])} | {f(v['cand_utmos'])} | "
                     f"{'yes' if v['cand'] in keep else ''} | {'; '.join(v['why_not'])} |")
    lines += ["", "Descriptions of the picked voices:", ""]
    lines += [f"- {v['cand']}: {v['description']}" for v in picks]
    return "\n".join(lines) + "\n"


# ---- run state ----------------------------------------------------------------------------------------------------


class Status:
    """<out>/status.json: phase, synthesis stats, errors (modal_app returns it)."""

    def __init__(self, path: Path, mode: str, dry: bool = False):
        self.path, self.dry = path, dry
        self.d = {"mode": mode, "started": now_iso(), "phase": "start", "errors": {}}

    def set(self, **kw) -> None:
        self.d.update(kw)
        self.write()

    def error(self, key: str, msg: str) -> None:
        self.d["errors"][key] = msg
        print(f"[contest] {key}: {msg}", flush=True)
        self.write()

    def write(self) -> None:
        if not self.dry:
            write_json(self.path, {**self.d, "updated": now_iso()})


def check_settings(path: Path, settings: dict, has_clips: bool) -> None:
    """Refuse to add clips made with other settings to a dir that has some (ValueError); record them otherwise."""
    old = read_json(path)
    if old and has_clips and any(old.get(k) != v for k, v in settings.items()):
        diff = {k: (old.get(k), v) for k, v in settings.items() if old.get(k) != v}
        raise ValueError(f"{path.parent} holds clips made with other settings {diff}: use another --run")
    write_json(path, settings)


def setup_entry(d: Path, e: dict, settings: dict, fallback: str | None = None) -> dict:
    """entry.json of an entry dir (ValueError when its clips were made with another reference, LoRA or settings), the
    reference copied next to its clips (ref<ext>, sim_ref<ext>) with the sources.json beside it. fallback: why the
    entry's engine (settings "backend") is not the call's (entry_backend), kept in entry.json for later calls."""
    key = entry_key(e, settings)
    old = read_json(d / "entry.json")
    if (diff := entry_conflict(old, e, settings)) and any(has_wav(p) for p in (d / "wav").rglob("*.wav")):
        raise ValueError(f"{d.name}: its clips were made otherwise ({diff}): use another --run (finals: a new --run "
                         f"with --screen {d.parent.name} for this run's designed voices)")
    d.mkdir(parents=True, exist_ok=True)
    meta = {**e, "settings": settings, "key": key, **({"fallback": fallback} if fallback else {})}
    for k in ("ref", "sim_ref"):
        if e.get(k):
            src = Path(e[k])
            dst = d / f"{k}{src.suffix.lower()}"
            if not has_wav(dst) or (old or {}).get(k) != e[k]:
                shutil.copyfile(src, dst)
            meta[f"{k}_file"] = dst.name
            if (s := src.parent / "sources.json").is_file():
                shutil.copyfile(s, d / "sources.json")
    meta["sim_file"] = meta.get("sim_ref_file") or meta.get("ref_file")
    write_json(d / "entry.json", meta)
    return meta


def entry_rows(rows: list[dict]) -> list[dict]:
    return [{**r, "prompt": compose(r.get("instruction"), r["model_text"]), "wav": f"wav/{r['set']}/{r['id']}.wav"}
            for r in rows]


def make_jobs(rows: list[dict], base: Path, group: str, ref: str | None, lora: str | None, ts: int, cfg: float,
              seed: int, backend: str = "voxcpm", ref_prep: bool = True, pause_clean: bool = False) -> list[dict]:
    """Synthesis jobs of the rows whose WAV is not there yet, for the engine `backend`: each with its row's seed
    (job_seed: --seed + crc32 of the text unless the row has its own), whether its reference is prepared
    (refprep.prep_ref; False: the raw clip) and whether its clip goes through clean_pauses (render --pause-clean)."""
    return [{"key": f"{group}/{r['set']}/{r['id']}", "group": group, "text": r["prompt"], "ref": r.get("ref", ref),
             "lora": lora, "cfg": cfg, "ts": ts, "seed": job_seed(r, seed), "out": str(base / r["wav"]),
             "est_s": round(est_audio_s(r["model_text"]), 2), "backend": backend, "ref_prep": ref_prep,
             "pause_clean": pause_clean}
            for r in rows if not has_wav(base / r["wav"])]


def fill_durations(rows: list[dict], base: Path) -> None:
    for r in rows:
        if r.get("duration") is None and has_wav(base / r["wav"]):
            r["duration"] = wav_duration(base / r["wav"])


def synthesize(args, jobs: list[dict], status: Status, mode: str, pool: SynthPool | None, feed_until: float,
               on_done, on_fail=None, fallback=None) -> tuple[SynthPool | None, dict]:
    """Run jobs on their engine (job "backend"): Nano-vLLM's on a NanoPool (every server stopped when it returns), the
    others (voxcpm / fake, LoRA fallbacks) after them on the SynthPool (started here if needed; returned for the
    caller to reuse or close). fallback(problems, jobs) -> the jobs of the LoRAs a Nano-vLLM server could not take
    (problems: {lora: error}) to make on the voxcpm backend instead; the others fail. Throughput -> status and the
    log."""
    if not jobs:
        return pool, {}
    nano = [j for j in jobs if j.get("backend") == "nanovllm"]
    rest = [j for j in jobs if j.get("backend") != "nanovllm"]
    stats = {}
    if nano:
        print(f"[contest] {len(nano)} clips on Nano-vLLM ({args.timesteps} timesteps, {args.concurrency} requests in "
              "flight, one server per LoRA)", flush=True)
        npool = NanoPool(args.model, args.concurrency, read_measured(args.throughput_log))
        stats = npool.run(order_jobs(nano), feed_until, args.deadline or math.inf, on_done, on_fail)
        print(f"[contest] {mode}: {stats['done']}/{stats['jobs']} clips, {stats['audio_s']} s audio in "
              f"{stats['wall_s']} s on {len(stats['groups'])} servers ({stats['audio_per_wall']} audio s per GPU s; "
              f"servers up {stats['server_s']} s, peak GPU {stats['gpu_mem_peak_mib']} MiB)"
              f"{'; ' + stats['stopped'] if stats['stopped'] else ''}", flush=True)
        status.set(**{f"synth_{mode}": stats})
        log_throughput(args.throughput_log, mode, args.timesteps, stats)
        if lost := stats["lora_failed"]:
            ljobs = [j for j in nano if j.get("lora") in lost]
            moved = fallback(lost, ljobs) if fallback else []
            keys = {j["key"] for j in moved}
            for j in ljobs:
                if j["key"] not in keys and on_fail:
                    on_fail(j, f"Nano-vLLM could not take LoRA {j['lora']}: {lost[j['lora']]}")
            rest += moved
    if rest:
        lora = next((j["lora"] for j in rest if j.get("lora")), None)
        if pool is None:
            print(f"[contest] {len(rest)} clips on {args.workers} workers ({rest[0]['backend']}, {args.timesteps} "
                  "timesteps)", flush=True)
            pool = SynthPool(args.workers, args.model, rest[0]["backend"], lora, args.load_parallel, args.optimize)
        vmode = f"{mode}-voxcpm" if nano else mode
        stats = pool.run(order_jobs(rest), feed_until, args.deadline or math.inf, on_done, on_fail)
        print(f"[contest] {vmode}: {stats['done']}/{stats['jobs']} clips, {stats['audio_s']} s audio in "
              f"{stats['wall_s']} s ({stats['audio_per_wall']} audio s per wall s, {stats['workers_ready']}/"
              f"{stats['workers']} workers, peak GPU {stats['gpu_mem_peak_mib']} MiB, host "
              f"{stats['host_mem_peak_mib']} MiB){'; ' + stats['stopped'] if stats['stopped'] else ''}", flush=True)
        status.set(**{f"synth_{vmode}": stats})
        log_throughput(args.throughput_log, vmode, args.timesteps, stats)
    return pool, stats


def _wav_of(base: Path):
    return lambda r: (base / r["wav"]) if has_wav(base / r["wav"]) else None


# ---- screen -------------------------------------------------------------------------------------------------------


SCORE_FIELDS = {"whisper": ("hyp", "whisper_error"), "utmos": ("utmos", "utmos_error"), "sim": ("sim", "sim_error")}


def unresolved(rows: list[dict], wav_of, metrics=("whisper", "utmos")) -> list[str]:
    """Ids of the rows a screen still waits for: no WAV yet (unless its synthesis failed in SYNTH_TRIES calls), or one
    of these metrics neither scored nor failed on the clip."""
    out = []
    for r in rows:
        if wav_of(r) is None:
            if r.get("synth_fails", 0) < SYNTH_TRIES:
                out.append(r["id"])
        elif any(r.get(k) is None and not r.get(err) for k, err in (SCORE_FIELDS[m] for m in metrics)):
            out.append(r["id"])
    return out


def row_callbacks(by_key: dict[str, dict]):
    """SynthPool.run's on_done / on_fail: a job's result on its row (synth_fails: the calls in which it failed)."""
    def on_done(job: dict, info: dict) -> None:
        if (row := by_key.get(job["key"])) is not None:
            row.update(info)
            row.pop("synth_error", None)
            row.pop("synth_fails", None)

    def on_fail(job: dict, err: str) -> None:
        if (row := by_key.get(job["key"])) is not None:
            row["synth_error"], row["synth_fails"] = err, row.get("synth_fails", 0) + 1

    return on_done, on_fail


def screen_rank(out: Path, cands: list[dict], clones: list[dict], params: dict) -> tuple[list[dict], list[str]]:
    """Rank the clone-tested voices -> ranking.json, ranking.md; and entries.json (the picked voices as a render spec)
    only when every candidate and every clone is made and scored (unresolved): else the ranking is provisional and an
    earlier entries.json is removed, so a render never takes picks from a partial pool. Returns (picks, missing)."""
    chosen_ids = params.get("clone_test", [])
    by = {c["id"]: c for c in cands}
    chosen = [by[i] for i in chosen_ids if i in by]
    wav_of = _wav_of(out)
    missing = [f"cand {i}" for i in unresolved(cands, wav_of)] + [
        f"clone {i}" for i in unresolved([r for r in clones if r.get("cand") in chosen_ids], wav_of,
                                         ("whisper", "utmos", "sim"))]
    ranked = rank_voices(chosen, clones, params["n_clone"], params["need_cer0"], params["min_sim"], params["min_p_tr"])
    picks = pick_voices(ranked, params["keep"], params["min_per_gender"])
    write_json(out / "ranking.json", {"params": params, "provisional": bool(missing), "missing": missing[:100],
                                      "voices": ranked, "picked": [v["cand"] for v in picks],
                                      "candidates": [{k: c.get(k) for k in ("id", "desc_id", "seed", "gender", "utmos",
                                                                            "p_tr", "char_errors", "ref_chars",
                                                                            "duration")} for c in cands]})
    if missing:
        (out / "entries.json").unlink(missing_ok=True)
    else:
        write_json(out / "entries.json", {"entries": designed_entries(picks, lambda c: str(out / "cand" / "wav" /
                                                                                      f"{c}.wav"))})
    write_atomic(out / "ranking.md", ranking_md(ranked, picks, params, len(missing)))
    print(f"[screen] {'provisional (' + str(len(missing)) + ' clips not made or scored yet), ' if missing else ''}"
          f"picked {len(picks)}: {', '.join(v['cand'] for v in picks) or '-'}", flush=True)
    return picks, missing


def screen(args) -> int:
    out: Path = args.out
    descs = load_descriptions(repo_path(args.descriptions))[: args.limit or None]
    seeds = parse_ints(args.seeds)
    clone_texts = [t.strip() for t in args.clone_texts.split("|") if t.strip()]
    if not clone_texts or args.need_cer0 > len(clone_texts):
        raise ValueError(f"--clone-texts has {len(clone_texts)} sentences, --need-cer0 is {args.need_cer0}")
    screen_mt = normalize(args.screen_text)
    # seed: the request's, per description (text_seed of its id) and SEED_STRIDE apart per --seeds value: no two
    # candidates share a first-step noise, and a retried onset (seed + k) never meets another candidate's seed
    cands = [{"set": "cand", "id": f"{d['id']}-s{s}", "category": d["gender"] or "?", "desc_id": d["id"],
              "seed": text_seed(args.seed + SEED_STRIDE * s, "cand", d["id"]), "gender": d["gender"],
              "description": d["text"], "text": args.screen_text, "model_text": screen_mt, "refs": [screen_mt],
              "prompt": compose(d["text"], screen_mt), "wav": f"cand/wav/{d['id']}-s{s}.wav"}
             for d in descs for s in seeds]
    est = plan_screen(sum(not has_wav(out / c["wav"]) for c in cands), clone_top(args.top, len(descs), args.per_desc,
                      len(seeds)), clone_texts, args.timesteps, args.workers, args.screen_text,
                      read_measured(args.throughput_log), backend=args.backend)
    on = "Nano-vLLM" if args.backend == "nanovllm" else f"{args.workers} workers ({args.backend})"
    print(f"[screen] {len(descs)} descriptions x {len(seeds)} seeds = {len(cands)} candidates, clone test {args.top} x "
          f"{len(clone_texts)}; ~{est / 60:.0f} min on {on}", flush=True)
    if args.dry_run:
        for c in cands[:3]:
            print(f"  {c['id']}: {c['prompt']}")
        return 0
    settings = screen_settings(args)
    check_settings(out / "settings.json", settings, any(has_wav(p) for p in (out / "cand" / "wav").glob("*.wav")))
    status = Status(out / "status.json", "screen")
    deadline = args.deadline or math.inf
    cands = merge_rows(cands, read_rows(out / "cand" / "rows.jsonl"), out)
    by_key = {f"cand/cand/{c['id']}": c for c in cands}
    on_done, on_fail = row_callbacks(by_key)
    sc, pool = Scorers(args.device), None
    params = {"top": args.top, "per_desc": args.per_desc, "max_cand_cer": args.max_cand_cer, "need_cer0":
              args.need_cer0, "min_sim": args.min_sim, "min_p_tr": args.min_p_tr, "keep": args.keep,
              "min_per_gender": args.min_per_gender, "n_clone": len(clone_texts), "clone_texts": clone_texts,
              "screen_text": args.screen_text, "seeds": seeds, **settings, "seed": args.seed}
    try:
        status.set(phase="candidates", candidates=len(cands))
        # each seed of a description is a request of its own (Nano-vLLM: an independent per-request seed)
        jobs = make_jobs(cands, out, "cand", None, None, args.timesteps, args.cfg, args.seed, args.backend,
                         not args.no_refprep)
        reserve = SCORE_LOAD_S + len(cands) * SCORE_CLIP_S + SUMMARY_S
        pool, _ = synthesize(args, jobs, status, "screen-cand", pool, deadline - reserve, on_done, on_fail)
        fill_durations(cands, out)
        write_rows(out / "cand" / "rows.jsonl", cands)
        if pool is not None and (free := gpu_free_mib()) is not None and free < 6000:
            pool.close()  # no room for the scorers next to the idle workers
            pool = None
        wav_of = _wav_of(out)
        if args.no_score:
            status.set(phase="candidates made (--no-score: nothing ranked)")
            return 0
        for name, fn in (("whisper", whisper_pass), ("utmos", utmos_pass)):
            try:
                fn(sc, cands, wav_of, "cand", args.rescore, deadline - SUMMARY_S)
            except Exception as e:  # noqa: BLE001
                status.error(f"cand {name}", f"{type(e).__name__}: {e}")
            write_rows(out / "cand" / "rows.jsonl", cands)
        if gaps := unresolved(cands, wav_of):  # the top --top of a partial pool are not the pool's: no clone test yet
            (out / "entries.json").unlink(missing_ok=True)  # an earlier call's picks are not this pool's either
            print(f"[screen] {len(gaps)} of {len(cands)} candidates not made or not scored yet ({', '.join(gaps[:5])}"
                  f"{', ...' if len(gaps) > 5 else ''}): no clone test in this call; the same command continues",
                  flush=True)
            status.set(phase="candidates incomplete", unresolved=len(gaps))
            return 0
        est_dur = est_audio_s(screen_mt)
        chosen = select_for_clone(cands, args.top, args.per_desc, args.max_cand_cer, args.min_p_tr,
                                  (0.4 * est_dur, 2.5 * est_dur))
        params["clone_test"] = [c["id"] for c in chosen]
        clones = [{"set": "clone", "id": f"{c['id']}-c{k}", "category": c["id"], "cand": c["id"], "text": t,
                   "model_text": normalize(t), "refs": [normalize(t)], "prompt": normalize(t),
                   "seed": text_seed(args.seed, "clone", f"c{k}"),  # a clone text's seed: the same in every voice
                   "ref": str(out / c["wav"]), "wav": f"clone/wav/{c['id']}/c{k}.wav"}
                  for c in chosen for k, t in enumerate(clone_texts)]
        clones = merge_rows(clones, read_rows(out / "clone" / "rows.jsonl"), out)
        by_key.update({f"clone/clone/{r['id']}": r for r in clones})
        status.set(phase="clone test", clone_test=len(chosen))
        jobs = make_jobs(clones, out, "clone", None, None, args.timesteps, args.cfg, args.seed, args.backend,
                         not args.no_refprep)
        loads = LOAD_S["ecapa"]  # Whisper and UTMOS are loaded already
        if jobs and args.backend == "nanovllm":
            sc.close()  # the clone server gets the GPU alone; Whisper and UTMOS load again for the clones
            loads = SCORE_LOAD_S
        pool, _ = synthesize(args, jobs, status, "screen-clone", pool,
                             deadline - (len(clones) * SCORE_CLIP_S + loads + 30 + SUMMARY_S), on_done, on_fail)
        if pool is not None:
            pool.close()  # free the GPU for the last scoring
            pool = None
        fill_durations(clones, out)
        write_rows(out / "clone" / "rows.jsonl", clones)
        for name, fn in (("whisper", whisper_pass), ("utmos", utmos_pass)):
            try:
                fn(sc, clones, wav_of, "clone", args.rescore, deadline - SUMMARY_S)
            except Exception as e:  # noqa: BLE001
                status.error(f"clone {name}", f"{type(e).__name__}: {e}")
        try:
            groups = [(c["id"], [r for r in clones if r["cand"] == c["id"]], out / c["wav"]) for c in chosen]
            sim_pass(sc, groups, wav_of, args.rescore, deadline - SUMMARY_S, self_sim=False)
        except Exception as e:  # noqa: BLE001
            status.error("clone sim", f"{type(e).__name__}: {e}")
        write_rows(out / "clone" / "rows.jsonl", clones)
        picks, missing = screen_rank(out, cands, clones, params)
        status.set(phase="clone test incomplete" if missing else "done", unresolved=len(missing),
                   **{"provisional" if missing else "picked": [v["cand"] for v in picks]},
                   scored={"cand": sum(c.get("utmos") is not None for c in cands),
                           "clone": sum(r.get("utmos") is not None for r in clones)})
    finally:
        if pool is not None:
            pool.close()
        sc.close()
    return 0


# ---- render -------------------------------------------------------------------------------------------------------


def list_refs_fs(root: Path) -> dict[str, list[str]]:
    if not root.is_dir():
        return {}
    return {d.name: sorted(p.name for p in d.iterdir() if p.is_file()) for d in sorted(root.iterdir()) if d.is_dir()}


def score_entries(items: list[tuple[Path, dict, list[dict]]], sc: Scorers, metrics: set[str], rescore: bool,
                  until: float, procs: int, status: Status | None = None) -> None:
    """Every metric of the clips of these entry dirs (dir, entry.json, rows), each model once over all of them;
    rows.jsonl after each pass, so a kill keeps what is done."""
    def err(k, e):
        msg = f"{type(e).__name__}: {e}"
        if status:
            status.error(k, msg)
        else:
            print(f"[score] {k}: {msg}", flush=True)

    def save():
        for d, _, rows in items:
            write_rows(d / "rows.jsonl", rows)

    wavs = [_wav_of(d) for d, _, _ in items]
    if "whisper" in metrics:
        for (d, _, rows), w in zip(items, wavs):
            try:
                whisper_pass(sc, rows, w, d.name, rescore, until)
            except Exception as e:  # noqa: BLE001  (e.g. faster-whisper missing): the other passes still run
                err(f"whisper {d.name}", e)
                break
        save()
    cpu = None
    if "cpu" in metrics and (todo := cpu_items([rows for _, _, rows in items], wavs, rescore)):
        if time.time() + CPU_MIN_S <= until:
            cpu = CpuPass(todo, procs)
        else:
            print(f"[cpu] no time left: {len(todo)} clips left for the next call", flush=True)
    if "utmos" in metrics:
        for (d, _, rows), w in zip(items, wavs):
            try:
                utmos_pass(sc, rows, w, d.name, rescore, until)
            except Exception as e:  # noqa: BLE001
                err(f"utmos {d.name}", e)
                break
        save()
    if "sim" in metrics:
        groups = [(d.name, rows, (d / meta["sim_file"]) if meta.get("sim_file") else None) for d, meta, rows in items]
        for (d, _, rows), w, g in zip(items, wavs, groups):
            try:
                sim_pass(sc, [g], w, rescore, until)
            except Exception as e:  # noqa: BLE001
                err(f"sim {d.name}", e)
                break
        save()
    if cpu is not None:
        try:
            apply_cpu([rows for _, _, rows in items], cpu.get(until))
        except Exception as e:  # noqa: BLE001
            err("prosody/intonation", e)
        save()


def write_summaries(items: list[tuple[Path, dict, list[dict]]], run_dir: Path | None) -> None:
    for d, meta, rows in items:
        write_json(d / "summary.json", entry_summary(rows, meta))
    if run_dir is None:
        return
    summaries = [s for p in sorted(run_dir.glob("*/summary.json")) if (s := read_json(p))]
    lines, md = leaderboard(summaries)
    write_json(run_dir / "contest_summary.json", {"updated": now_iso(), "entries": lines,
                                                  "summaries": {s["entry"]: s for s in summaries}})
    write_atomic(run_dir / "leaderboard.md", f"# Voice contest {run_dir.name}\n\n{md}")
    print(md, flush=True)


def render(args) -> int:
    out: Path = args.out
    fs, refs_root = LocalFS(), args.refs_root
    spec = read_json(args.spec) if args.spec else DEFAULT_SPEC
    if spec is None:
        raise ValueError(f"{args.spec}: not readable JSON")
    entries = resolve_spec(with_screen(spec, args.screen), out.name, lambda: list_refs_fs(Path(refs_root)),
                           lambda r: read_json(out.parent / r / "screen" / "entries.json"), refs_root,
                           warn=lambda m: print(f"[render] {m}", flush=True),
                           read_sources=lambda d: read_json(Path(refs_root) / d / "sources.json"))
    if args.only:
        want = [n.strip() for n in args.only.split(",") if n.strip()]
        if missing := set(want) - {e["name"] for e in entries}:
            raise ValueError(f"--only: no entry {', '.join(sorted(missing))} "
                             f"(have {', '.join(e['name'] for e in entries)})")
        entries = [e for e in entries if e["name"] in want]
    texts = render_texts(args)
    settings = render_settings(args)
    done = {e["name"]: {f"{r['set']}/{r['id']}" for r in texts
                        if has_wav(out / e["name"] / "wav" / r["set"] / f"{r['id']}.wav")} for e in entries}

    def engine(e: dict) -> tuple[str, str | None]:
        try:
            lora = resolve_lora(e["lora"], fs) if e.get("lora") else None
        except ValueError:  # resolve_entry reports it below
            return args.backend, None
        return entry_backend(args.backend, {**e, "lora": lora}, lora_config(lora),
                             read_json(out / e["name"] / "entry.json"))

    engines = {e["name"]: engine(e) for e in entries}
    plan = plan_render(entries, texts, done, args.timesteps, args.workers, math.inf, read_measured(args.throughput_log),
                       args.backend, {n: v[0] for n, v in engines.items()}, not args.no_score)
    per_set = ", ".join(f"{k} {v}" for k, v in sorted(_count(texts, "set").items()))
    print(f"[render] {len(entries)} entries x {len(texts)} texts ({per_set})"
          + (f", pause_clean {PAUSE_CLEAN} after synthesis" if args.pause_clean else ""), flush=True)
    for e, p in zip(entries, plan):
        print(f"  {e['name']:<34} {e['kind']:<8} {p['todo']:3d}/{p['clips']} to make, ~{p['est_s'] / 60:4.1f} min"
              f"{'  lora ' + e['lora'] if e.get('lora') else ''}{'  ref ' + e['ref'] if e.get('ref') else ''}"
              f"{'  on ' + p['engine'] if p['engine'] != args.backend else ''}")
        if why := engines[e["name"]][1]:
            print(f"    {why}: the official API makes it (voxcpm, --workers {args.workers})", flush=True)
    if args.dry_run:
        firsts = {}
        for r in entry_rows(texts):
            firsts.setdefault(r["set"], r)
        for r in firsts.values():
            print(f"  {r['set']}/{r['id']}: {r['prompt']}")
        return 0
    status = Status(out / "status.json", "render")
    status.set(phase="setup", entries=[e["name"] for e in entries])
    items: list[tuple[Path, dict, list[dict]]] = []
    for e in entries:
        backend, why = engines[e["name"]]
        try:
            e = resolve_entry(e, fs, refs_root)
            meta = setup_entry(out / e["name"], e, entry_settings(settings, e, backend), why)
        except (ValueError, OSError) as err:  # this entry only; the others go on
            status.error(e["name"], str(err))
            continue
        d = out / e["name"]
        rows = merge_rows([{**r, "ref": e.get("ref")} for r in entry_rows(texts)], read_rows(d / "rows.jsonl"), d)
        items.append((d, meta, rows))
    if fb := {d.name: meta["fallback"] for d, meta, _ in items if meta.get("fallback")}:
        status.set(fallback=fb)
    by_key = {f"{d.name}/{r['set']}/{r['id']}": r for d, _, rows in items for r in rows}
    on_done, on_fail = row_callbacks(by_key)
    jobs = [j for d, meta, rows in items
            for j in make_jobs(rows, d, d.name, meta.get("ref"), meta.get("lora"), args.timesteps, args.cfg, args.seed,
                               meta["settings"]["backend"], meta["settings"].get("ref_prep") != "raw",
                               bool(meta["settings"].get("pause_clean")))]

    def fallback(lost: dict[str, str], ljobs: list[dict]) -> list[dict]:
        """Entries whose LoRA a Nano-vLLM server refused: made on the voxcpm backend instead while none of their clips
        is there (entry.json records it, so later calls keep them there), else left failed (one engine per entry)."""
        moved = []
        for i, (d, meta, rows) in enumerate(items):
            mine = [j for j in ljobs if j["group"] == d.name]
            if not mine or any(has_wav(p) for p in (d / "wav").rglob("*.wav")):
                continue
            why = f"Nano-vLLM could not take LoRA {meta['lora']}: {lost[meta['lora']]}"
            e = {k: meta[k] for k in ENTRY_KEYS if k in meta}
            items[i] = (d, setup_entry(d, e, {**meta["settings"], "backend": "voxcpm"}, why), rows)
            print(f"[render] {d.name}: {why}: the official API makes it (voxcpm)", flush=True)
            status.set(fallback={**status.d.get("fallback", {}), d.name: why})
            moved += [{**j, "backend": "voxcpm"} for j in mine]
        return moved

    deadline = args.deadline or math.inf
    reserve = SCORE_LOAD_S + sum(len(rows) for _, _, rows in items) * SCORE_CLIP_S + SUMMARY_S
    status.set(phase="synthesis", clips=len(by_key), to_make=len(jobs))
    pool = None
    try:
        pool, _ = synthesize(args, jobs, status, "render", None, deadline - (0 if args.no_score else reserve), on_done,
                             on_fail, fallback)
    finally:
        if pool is not None:
            pool.close()  # free the TTS models before scoring
        for d, _, rows in items:
            fill_durations(rows, d)
            write_rows(d / "rows.jsonl", rows)
    try:
        if not args.no_score:
            status.set(phase="scoring")
            sc = Scorers(args.device)
            try:  # every pass stops at the deadline (scorer loads included): the summaries below always get written
                score_entries(items, sc, set(args.metrics.split(",")), args.rescore, deadline - SUMMARY_S,
                              args.cpu_procs, status)
            finally:
                sc.close()
    finally:
        write_summaries(items, out)
    complete = {d.name: (all(r.get("duration") is not None for r in rows) if args.no_score
                         else entry_summary(rows, meta)["complete"]) for d, meta, rows in items}
    status.set(phase="done" if all(complete.values()) else "incomplete", complete=complete)  # errors: status "errors"
    return 0


def _count(rows: list[dict], key: str) -> dict:
    c = defaultdict(int)
    for r in rows:
        c[r[key]] += 1
    return dict(c)


# ---- score --------------------------------------------------------------------------------------------------------


def score_dirs(args) -> int:
    """Rescore render runs, entry dirs and screen dirs in place."""
    metrics = set(args.metrics.split(","))
    sc = Scorers(args.device)
    try:
        for d in args.dirs:
            if (d / "cand" / "rows.jsonl").is_file():  # a screen dir
                rk = read_json(d / "ranking.json") or {}
                params = rk.get("params")
                cands, clones = read_rows(d / "cand" / "rows.jsonl"), read_rows(d / "clone" / "rows.jsonl")
                w = _wav_of(d)
                for rows, label, f in ((cands, "cand", d / "cand" / "rows.jsonl"),
                                       (clones, "clone", d / "clone" / "rows.jsonl")):
                    if "whisper" in metrics:
                        whisper_pass(sc, rows, w, label, args.rescore)
                    if "utmos" in metrics:
                        utmos_pass(sc, rows, w, label, args.rescore)
                    write_rows(f, rows)
                if "sim" in metrics:
                    by = defaultdict(list)
                    for r in clones:
                        by[r["cand"]].append(r)
                    sim_pass(sc, [(c, rs, d / "cand" / "wav" / f"{c}.wav") for c, rs in by.items()], w, args.rescore,
                             self_sim=False)
                    write_rows(d / "clone" / "rows.jsonl", clones)
                if params:
                    screen_rank(d, cands, clones, params)  # entries.json only when nothing is missing
                continue
            dirs = [d] if (d / "rows.jsonl").is_file() else sorted(p.parent for p in d.glob("*/rows.jsonl"))
            if not dirs:
                raise ValueError(f"{d}: no rows.jsonl (an entry dir, a render run or a screen dir)")
            items = [(x, read_json(x / "entry.json") or {"name": x.name}, read_rows(x / "rows.jsonl")) for x in dirs]
            score_entries(items, sc, metrics, args.rescore, math.inf, args.cpu_procs)
            run_dir = d if d not in dirs else d.parent if (d.parent / "leaderboard.md").exists() else None
            write_summaries(items, run_dir)
    finally:
        sc.close()
    return 0


# ---- CLI ----------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--model", default=MODEL_DIR, help="VoxCPM2 snapshot directory")
        p.add_argument("--timesteps", type=int, default=10, help="10 to screen, 16 for finals")
        p.add_argument("--cfg", type=float, default=2.0)
        p.add_argument("--seed", type=int, default=0, help="base seed: a clip's is --seed + crc32(<set>/<id>) "
                       "(text_seed; the same in every voice), unless its row has its own")
        p.add_argument("--no-refprep", action="store_true", help="references as they are (raw), not refprep.prep_ref's "
                       "16 kHz right-padded levelled clip (a spec entry's \"ref_prep\" overrides it)")
        p.add_argument("--workers", type=int, default=4, choices=range(1, MAX_WORKERS + 1), metavar="1-8",
                       help="voxcpm backend (and LoRA fallbacks): synthesis processes on the GPU (each holds VoxCPM2, "
                       "~7 GB)")
        p.add_argument("--concurrency", type=int, default=NANO_CONCURRENCY, choices=range(1, 257), metavar="1-256",
                       help="nanovllm backend: requests in flight per server (its max_num_seqs)")
        p.add_argument("--load-parallel", type=int, default=2, choices=range(1, MAX_LOAD_PARALLEL + 1),
                       metavar=f"1-{MAX_LOAD_PARALLEL}", help="workers loading VoxCPM2 at once (each peaks at "
                       f"~{LOAD_RAM_GB:.0f} GB RAM: fp32 build + weights; pool_ram_gb)")
        p.add_argument("--optimize", action="store_true", help="VoxCPM's torch.compile (untested here)")
        p.add_argument("--backend", choices=BACKENDS, default="nanovllm", help="nanovllm: Nano-vLLM-VoxCPM, one server "
                       "per LoRA (the GPU default); voxcpm: the official API in --workers processes; fake: silent "
                       "clips (tests)")
        p.add_argument("--device", default="auto", help="scorers: cuda, cpu or auto")
        p.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; 0 = none")
        p.add_argument("--throughput-log", default=THROUGHPUT_LOG, help="throughput.jsonl ('' = none)")
        p.add_argument("--no-score", action="store_true", help="synthesis only")
        p.add_argument("--rescore", action="store_true", help="score every clip again")
        p.add_argument("--dry-run", action="store_true", help="print the plan and the first prompts, write nothing")
        p.add_argument("--cpu-procs", type=int, default=4, help="processes of the prosody / intonation pass")

    s = sub.add_parser("screen", help="designed voices: candidates, clone test, ranking")
    common(s)
    s.add_argument("--descriptions", default=str(DESCRIPTIONS))
    s.add_argument("--seeds", default="0,1")
    s.add_argument("--limit", type=int, default=0, help="first N descriptions only")
    s.add_argument("--screen-text", default=SCREEN_TEXT)
    s.add_argument("--clone-texts", default="|".join(CLONE_TEXTS), help="'|'-separated")
    s.add_argument("--top", type=int, default=30, help="candidates that get the clone test")
    s.add_argument("--per-desc", type=int, default=1, help="of them at most this many per description")
    s.add_argument("--max-cand-cer", type=float, default=0.02)
    s.add_argument("--need-cer0", type=int, default=3, help="clones with CER 0 a voice needs")
    s.add_argument("--min-sim", type=float, default=0.4, help="mean ECAPA cosine of the clones to the candidate "
                   "(provisional: an Antalia LoRA's short clips vs a real Antalia clip measured 0.32-0.57)")
    s.add_argument("--min-p-tr", type=float, default=0.9, help="mean P(Turkish) (candidates and clones)")
    s.add_argument("--keep", type=int, default=5)
    s.add_argument("--min-per-gender", type=int, default=1)
    s.set_defaults(metrics="whisper,utmos,sim")

    r = sub.add_parser("render", help="entries x the contest texts -> clips + scores + leaderboard")
    common(r)
    r.add_argument("--spec", type=Path, help="entry spec (JSON); default: DEFAULT_SPEC (controls + refs + screen)")
    r.add_argument("--screen", default="", help="comma-separated runs whose screen winners join (replaces the spec's "
                   "\"screen\"; default: this run's), e.g. finals at --timesteps 16 under a new --run")
    r.add_argument("--texts", default=str(TEXTS))
    r.add_argument("--refs-root", default=REFS)
    r.add_argument("--only", default="", help="comma-separated entry names")
    r.add_argument("--limit", type=int, default=0, help="N texts per set (smoke test)")
    r.add_argument("--ids", default="", help="comma-separated text ids: only those")
    r.add_argument("--seeds", default="", help="comma-separated seeds: each text once per seed (id <id>-s<seed>, that "
                   "seed as it is), e.g. 0,1,2,3 to see what a seed does")
    r.add_argument("--metrics", default="whisper,utmos,sim,cpu")
    r.add_argument("--pause-clean", action="store_true", help="pause_clean.clean on every clip after synthesis, before "
                   "its level, onset check and scores (settings \"pause_clean\": clips made without it need another "
                   "--run)")

    c = sub.add_parser("score", help="rescore existing dirs")
    c.add_argument("dirs", nargs="+", type=Path)
    c.add_argument("--rescore", action="store_true")
    c.add_argument("--metrics", default="whisper,utmos,sim,cpu")
    c.add_argument("--device", default="auto")
    c.add_argument("--cpu-procs", type=int, default=4)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return {"screen": screen, "render": render, "score": score_dirs}[args.cmd](args)
    except ValueError as e:
        print(f"[contest] {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
