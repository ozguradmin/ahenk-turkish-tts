#!/usr/bin/env python3
"""Corpus render (docs/SES_PLANI.md step 3): a clean ~5 h Turkish TTS training corpus in ONE voice, rendered by base
VoxCPM2 cloning one reference clip (reference-only; Nano-vLLM cannot load our LoRA), every clip gated, in the layout
train_lora.py --data takes.

    python tts/voxcpm/render_corpus.py --out /data/f022corpus --hours 5 [--mix neutral=0.4,questions=0.25,...]
        [--backend nanovllm|voxcpm] [--deadline EPOCH] [--min-sim 0.7] [--laughter-usd 0.3] [--dry-run]
    python tts/voxcpm/render_corpus.py --out /data/f022corpus2 --hours 1.0 --mix long=0.47,neutral=0.47,cwords=0.06
        --min-sim 0.6 --laughter-usd 0 --refs-from /data/f022corpus --avoid-pool /data/f022corpus
        [--plain-render cwords]
    python tts/voxcpm/render_corpus.py --out /data/f022mix --merge /data/f022corpus /data/f022corpus2 [--dry-run]
        [--instructions en|both|tr]
    python tts/voxcpm/render_corpus.py --out /data/f022corpus --audit-c [--dry-run]
    (on Modal: modal_app.py --action corpus-render / corpus-fetch [--out DIR], corpus-merge, corpus-audit-c)

Voice: designed-f022-s0, the r1 screen candidate f022-s0 (--ref, its raw clip copied to ref/<voice>.raw.wav), given to
both engines as refprep.prep_ref makes it (16 kHz, right-padded to whole patches, -20 dBFS: the contest's ref_prep);
every clip at refprep's output level. ref/<voice>.wav: the prepared reference as 16 kHz PCM, a manifest ref_audio.

Texts (texts.jsonl, the pool: built on the first call, grown when the textbank or --domain-lines grows, never
reordered): "neutral" = sentences of cloud0day3/alania-domain-text-tr (CC BY 4.0, PatientDesk AI; pinned
DOMAIN_REV; text_raw through trnorm.normalize, 25-180 characters of spoken form, no "?" or question tag, no digit or
symbol left after normalising, Turkish language-ID >= LID_MIN), drawn by --domain-weights over the categories with
numbers, times, money, dates, phones, addresses, appointments, banking and cargo, --rich-share of each from lines with
digits or acronyms; questions / emotions / short / interjections = the rows of --textbank (build_textbank.py: id,
category, kind or emotion, text, spoken, instruction, instruction_split, pattern, peak_word = an index into the spoken
form (or the word), span, render_variants, render_texts, pair_id); emotion rows with a held-out instruction wording
(instruction_split "heldout": evaluation only) are left out, a question without a peak_word is not intonation-gated
when its pattern needs one. The spoken form is trnorm.normalize(text) (the textbank's "spoken" where it differs is
counted, not used). Leakage guard
(prepare_alania.build_guard): no text that is, or contains a >= 4-word sentence of, a row of benchmarks/tts/sets/*.jsonl
(text, spoken, alts, refs), of the probes (PROBE_GUARD: tts/voxcpm/probe_canina*.jsonl, probe_tr_instr.jsonl: our c and
Turkish-instruction tests), of
Antalia's held-out sentences (HELDOUT, by default: /data/antalia16k/voxcpm_val.jsonl, voxcpm-eval/antalia-gt/rows.jsonl,
whichever is here) or of an --exclude-manifest; duplicates dropped (key: the
normalize_for_eval form + the final mark "?" / "..." / statement, so the textbank's "Pardon!" / "Pardon?" pairs stay).

Training text (voxcpm_train.jsonl "text"): the spoken form; emotions "(instruction)" + spoken form (VoxCPM2's control
form; the official trainer tokenizes the text verbatim). Render text: the same, except that interjections try their
render_variants (each through trnorm) in turn until Whisper hears the canonical word (a variant that passed is kept
for the next tries); the training text stays the canonical one.

Long items (category "long", only with a --mix share; f022 is weak on 12-19 s sentences, its training clips
averaged 3.5 s): 1-3 CONSECUTIVE lines of one generation prompt of the domain text, --long-chars (120-300) characters
of spoken form (each run's window nearest the middle of --long-target: long_windows), the windows within --long-target
(190-280, ~12.5-17 s) drawn first (2180 real rows: median 14.5 s, all >= 12 s; the fewest lines that fit gave 9.9 s),
each line's sentence mark kept (a "." added where a line has none) so the model learns the pauses between sentences;
one line alone only when it is that long itself (a long sentence: most long_explanations lines). The dataset's
grouping (read 2026-10-12): every
line is its own doc_id "qwen3:s3-p<prompt>-<k>" (unit_idx 0, n_sent 1); the lines of one prompt share category and
subtopic, k is their order (doc_group); a line another filter drops (a question, low language-ID, ...), that
--avoid-pool / the pool already holds or a k not read breaks a run, so joined lines are always neighbours. Categories
by --long-weights; lines through trnorm one by one; the leakage guard on the whole text and on each line; --long-lines
(1500) items in the pool, drawn after the neutral sentences and never sharing a line with them. The sentence
boundaries the pause gate checks are the joins (the row's "breaks"), not every "." trnorm left (an initial "ka.", an
ordinal "beş.").

Numbers in neutral rows (--number-share, default 0.6): number_kinds' written-form checks of text_raw (phone, time,
money, date, percent, acronym, other digits or currency signs); the neutral draw takes lines with numbers first in each
category (--rich-share), spread over the kinds (kind_spread), and tops them up from any category to --number-share of
the draw; each wave orders the new neutral texts so that the expected kept seconds with numbers stay at
>= --number-share (number_order, from each class's measured pass rate once there is one). Both only as far as the pool
allows.

c words (category "cwords", only with a --mix share; under an English emotion instruction the model turns
word-initial Turkish "c" (/dʒ/) into /k/ or /d/): the rows of --cwords (textbank/cwords.jsonl, emotions.jsonl's schema:
each an emotion, a train instruction wording and a word-initial c) as emotion rows (textbank_row: held-out wordings
left out; the emotion's reference clip, its prosody gate, "(instruction)" in the training text; UTMOS calibrated per
"cwords/<emotion>"), guarded and de-duplicated like the textbank, --cwords-tries (8) tries a text (the base model makes
the error too), CWORDS_PRERENDER (4) of them rendered a wave. Its pass rate is a guess (0.35 a try) until measured, so
its first wave takes PROBE_TEXTS texts.

Plain render (--plain-render CATS, of the categories with an instruction: emotions, cwords; call 1 of /data/f022corpus2:
after the English "(instruction)" base VoxCPM2 read a word-initial c as k / ç / s, 26 of 590 cwords tries passed; the
same voice without one says it right): the rows of CATS render without the instruction (render_text), cloning their
emotion's reference clip as before (ref_for: it carries the emotion); everything else as it was: the training text
"(instruction)" + spoken form (so the LoRA learns that a c after an instruction stays /dʒ/), the c-check (the row has
an instruction), seeds, gates; tries and kept clips record "render": "plain". The emotion check is stricter
(plain_verdict: the clip has to show the emotion, verdict "expected", else gate PLAIN_FAIL), and such rows wait while
their emotion has no reference clip or the corpus has no baseline (plain_wait). The mode is the category's (state.json
"modes"): switching (switch_mode) keeps the tries so far under the mode they were made in (stats' "modes"), measures the
pass rate, the --min-pass stop and PROBE_TEXTS anew in the new mode (PLAIN_PASS until then), keeps the kept texts and
gives every other text a budget of tries in the new mode from its next try (try0: new seeds), the old mode's undecided
tries removed; a text gets one budget per mode ("mode_tries": back in a mode it had before, its tries there count, one
that used them up stays dropped); UTMOS calibrates plain tries on their own ("<category>:plain"). --plain-render is
given again by every later call whose --mix gives a plain category a share (a call without it is refused, --plain-render
none switches back); a category without a share keeps its mode and renders nothing; the reference, model, timesteps,
cfg and seed still may not change. A plain category under --min-pass after its probe leaves texts never tried: the
corpus stays incomplete (corpus-merge --allow-incomplete takes its kept clips; --min-pass 0 goes on).

Mix (--mix, by KEPT hours of --hours): neutral 40 %, questions 25 %, emotions 20 %, short 10 %, interjections 5 %, long
and cwords 0 % (a corpus of their own: --mix long=0.47,neutral=0.47,cwords=0.06). A category whose pool is used up
hands what it lacks to --fill (neutral).

Gates (all must pass; a failed try is made again with seed + SEED_STRIDE x try, --tries in all, interjections one more
per extra variant, then the text is dropped):
  synthesis  made (runaway bad case: fail); onset: not cut by NanoPool's onset retry (onset_trimmed) and the final
             clip not flagged by onset.py (CPU pass)
  duration   7-25 characters of spoken form per second (--rate); texts under SHORT_CHARS characters: 0.25-SHORT_MAX_S s
  text       Whisper (voice_contest's judge: large-v3-turbo, language tr, beam 5, word timestamps): CER 0 against the
             canonical spoken form (normalize_for_eval both sides), P(tr) >= --min-p-tr from its own language-ID
  c          rows with an instruction and a "c" in the spoken form (emotions, cwords; needs_c): phon_check's IPA model
             (no language model: Whisper writes "canına" for a /k/) hears a /dʒ/ for every run of c (c_check); a try it
             cannot read fails. Loaded only when a wave has such a row, kept off the GPU between waves. --no-c-check:
             off
  clip       the f022 pilot's gates (99 kept clips, a native listener's notes on 44; per gate in tts/voxcpm/README.md),
             from Whisper's word times and the clip (measure_clip, onset.py): tail: speech end = the last word's end
             through its own sound; a stray after a 150 ms gap (pilot clip 41, "Özür dilerim." + "n") is cut there +
             120 ms with a 20 ms fade (tail_cut_s; trailing silence too), one inside the last word or longer than
             0.35 s fails; head: a loud voiced burst before a voiceless start ("vlead") or > 150 ms of sound before the
             first word; stretch: the first word's s / syllable > --stretch-max (2.2) x the other words' median
             (interjections exempt); sibilance: the loudest sustained (40 ms) 2-8 kHz frication > --sib-max (-4.5 dB re
             the loud frames; clip 15's "her şey" -3.4, all clips not flagged by ear <= -5.7); speed: syllables per
             second of speech (word times, pauses out) > SPEED_MAX x --speed-scale (neutral / short / long 8, questions
             8.5, emotions 7.8, interjections 7.5; clip 9: 8.9, the fastest not flagged 8.3)
  pause      long clips only (pause_measures): the silence between two words = Whisper's gap or the clip's longest
             quiet run (onset.edges' pauses) nearest that gap, whichever is longer; at each join of two lines
             --pause-sent (0.12-0.9 s), elsewhere <= --pause-mid-max (0.6 s), anywhere <= --pause-max (1.2 s);
             words Whisper's times do not match: failed (nothing to check by). Long texts get --long-tries (6) tries
  speaker    ECAPA cosine to the raw reference >= --min-sim for clips of >= SIM_FULL_S s, --sim-short-slack less at 1 s
             (linear between): ECAPA scores short clips lower (f022 in r1 + r2: 0.61 under 2.5 s, 0.73 from 4 s)
  UTMOS      >= the --utmos-drop quantile (bottom 20 %) of the category's first tries that passed the text gate;
             emotions per emotion once it has --calib-n of them (UTMOS punishes energetic speech)
  intonation question kinds (mi_soru, degil_mi: peak before mI then a fall; ne_soru: peak on the wh-word, not under
             a word before it by > 2 st, not flat; yukselen: final rise): score_intonation's item passes; duz_eslesik
             must not pass as a final rise
  emotion    prosody delta to the voice's kept neutral clips (median; >= BASELINE_MIN of them; energy: level_db, the
             level before refprep levelled the clip, from synthesis): angry / cheerful / surprised dropped only when
             clearly opposite (an F0 range >= EMO_CLEAR_ST narrower); sad / calm kept only when slower AND lower in F0
             (and not louder by > EMO_LEVEL_SLACK dB): 7 of the pilot's 21 would have failed; a plain-rendered try
             (--plain-render) only when "expected" and plain_score > 0 (empathetic: lower and quieter, plain_score >
             0), gate PLAIN_FAIL
Gate order (cost): synthesis / duration, then Whisper, then the c-check, then the clip gates, then UTMOS + ECAPA + the
CPU pass on what passed.

Prelude (once, before the first wave; one server, then the scorers; prelude_jobs): emotion references: for each
emotion of the pool, --emo-ref-n renders of EMO_REF_TEXT in f022 (controllable cloning: the voice's reference + the
emotion's instruction) and EMO_REF_NEUTRAL plain ones; the candidate heard right with ECAPA >= EMO_REF_MIN_SIM whose
prosody goes furthest the emotion's way against the plain renders (emo_ref_score: sad lower, slower, quieter before
levelling; angry / cheerful / surprised wider and louder) -> refs/<emotion>.wav, cloned by that emotion's rows instead
of the voice's reference (state "emotion_refs"; none found: the voice's reference). Laughter trial (--laughter-usd, 0 =
off): textbank/laughter.jsonl x LAUGH_INSTRUCTIONS x LAUGH_SEEDS within the budget; Whisper CER after removing the
laughter from both sides (it drops "haha"), the burst detector (onset.laugh) in the span Whisper's words leave free,
AST AudioSet laughter (AST_MODEL) only when it is already in the HF cache; the LAUGH_KEEP best tries of each line ->
laughter_review/ (wav/, review.jsonl, laughter.html; corpus-fetch -> voxcpm-corpus/f022/laughter.html), never in the
manifest by itself.

Human-approved clips (--add-reviewed FILE, e.g. textbank/laughter_approved.json: {"<review id>": true, ..., "best":
{line: [ranks]}, "note"}; between corpus-render calls only, refused while status.json says a call is going or
state.json is newer than the report): each approved laughter_review/ clip -> wav/laugh_<id>.wav (16 kHz PCM16), a
row of reviewed.jsonl (category "laughter", source "human-approved"; the record every later finalize merges into
clips.jsonl, the manifest and the report, and reconcile keeps their WAVs), appended to clips.jsonl and
voxcpm_train.jsonl (text "(instruction)" + trnorm spoken form, the laughter kept as written: "Hahaha, ..."; ref_audio
for --ref-fraction of them by id), the report's per-category hours updated. Idempotent: a clip already there is not
added again. Nothing rendered, no GPU.

Waves (the GPU is shared): the texts still needed (open rows first, then new ones while the expected kept hours fall
short of each category's target, interleaved so that a cut wave keeps the mix) get --prerender tries rendered at once
(one Nano-vLLM server, --concurrency in flight, per-text seeds; NanoPool's own onset retry inside), the server is
stopped, the scorers load and score the tries in rounds (each text's next try only when the one before failed), the
scorers are freed, and the next wave renders the retries and tops up (cwords: CWORDS_PRERENDER (4) tries a wave, short
clips, so --cwords-tries 8 take no more waves than a long item's 6). A category whose pass rate is a guess (GUESSED:
long, cwords) has at most PROBE_TEXTS (40) texts in flight until MIN_MEASURED of its tries are decided, so one wave
measures it before the waves after it plan with it; a category whose measured pass rate is under --min-pass (0.05)
takes no new texts (logged and in the report; the corpus stays incomplete). Resume: state.json (rows, kept clips, the
calibration), clips.jsonl, attempts/<id>/t<k>.wav + .json (made, not yet decided: scored by the next call). A call
stops with time to finalize (--deadline, set by modal_app from --max-usd); the same command continues. --hours may
change between calls; the reference, model, timesteps, cfg and seed may not (another --out).

A second corpus beside a first (--out /data/f022corpus2): --refs-from DIR copies DIR's voice reference (ref/: raw and
prepared, refused when the raw clip is not --ref's or DIR's voice is another) and its emotion references (refs/<e>.wav
and their state records) when this corpus has none; --avoid-pool DIR leaves out every text (dedup_key) and domain line
(line_id, a long item's line_ids) of DIR's texts.jsonl, so the new corpus is new text. DIR is only read.

Merge (--merge SRC [SRC ...] --out DIR; no render, no GPU): DIR/voxcpm_train.jsonl = the union of the sources'
manifests (rows as they are: absolute WAV paths, ref_audio; a row whose audio an earlier row has, or whose text an
EARLIER source has, left out; seeded shuffle), DIR/prep_report.json ("merged", "complete", per-source manifest bytes and
sha256: modal_app's train preflight refuses a merge whose sources changed since) + PREP_REPORT.md, README.md (the
note), ATTRIBUTION.txt (the sources' lines). Refused while a source's corpus-render call is going (render_busy), for a
source whose mix is not complete (--allow-incomplete takes it), a WAV or ref_audio that is not there, or a DIR that
is a source or a render corpus. A source's c_audit.jsonl (--audit-c): its clips with "ok": false are left out (counted
and named in the report), the file's bytes and sha256 recorded (the train preflight hashes it again and refuses a
merge whose audit changed, appeared or went since); a source with instructed clips with a c that neither the gate
(clips.jsonl c_expected) nor its audit checked is refused (--allow-unaudited takes it: the first corpus before its
corpus-audit-c).
Instruction language (--instructions; after an English "(instruction)" the model reads a word-initial c
the English way, after a Turkish one it says it right): en (default) writes the rows as they are; both writes every
instructed row and a copy of it (same audio, duration, ref_audio) whose instruction is the Turkish wording of
INSTRUCTIONS_TR (tts/voxcpm/instructions_tr.json, english -> turkish), so the LoRA learns the Turkish wordings and keeps
the English ones; tr writes the Turkish copy only; rows without an instruction once (instruction_copies). An instructed
row whose wording the table lacks refuses the merge (no silent passthrough), the dry run too. The report records the
mode, the table's sha256 and the manifest rows per language ("instructions": rows plain / en / tr); its "selected" and
"by_category" count manifest rows (a clip in both languages twice), "audio" each clip once. A merge dir keeps its mode:
a merge with another --instructions goes to another --out (merge_out_for: /data/f022mix -> /data/f022mix_both,
/data/f022mix_tr), so an experiment trained on one never resumes on other texts; the train preflight (modal_app.py)
refuses a merge made with another table than the local one (table_sha256: the probe takes the table's wordings).

c audit (--audit-c --out DIR; no render): phon_check's IPA tokens of every manifest clip of DIR whose training text has
an instruction and a "c" (clips kept before the c-check: the first corpus's emotion rows, approved laughter) ->
DIR/c_audit.jsonl ({id, wav (the manifest's audio path), category, text, expected, got, ok, ipa}; "error" and ok false
for a clip it cannot read; when it cannot read SYSTEMIC_N and SYSTEMIC_SHARE of them, the model or the setup is
broken, not the clips: exit 1 and nothing written, an earlier audit stays). The manifest, clips.jsonl, state.json and
the report are not touched; refused while a corpus-render call is going (render_busy). On Modal: modal_app.py --action
corpus-audit-c (a small GPU, minutes).

Outputs (--out, train_lora.py's layout): wav/<id>.wav (16 kHz PCM16 mono), voxcpm_train.jsonl ({audio, text,
duration}; --ref-fraction of the rows with ref_audio / ref_duration: the prepared reference or another kept neutral
clip of 3-10 s, never itself), clips.jsonl (every score of every kept clip: words, edges, sib_db, first_stretch,
rate_ws, tail_cut_s, level_in, ref), attempts.jsonl (every decided try), prep_report.json + PREP_REPORT.md (mix, gate
pass rates, distributions, the clip gates, emotion references, the laughter trial, throughput), ATTRIBUTION.txt;
refs/<emotion>.wav, laughter_review/, reviewed.jsonl (--add-reviewed).

Throughput: each Nano-vLLM server's line in --throughput-log (voice_contest.log_throughput, mode "corpus"), and one
"corpus-score" line per wave (scored clips, audio, wall s, the model's estimate: score_est + c_est per clip without
the scoring factor, the scorer loads and the IPA model's measured load; "raw": true); the estimates use them.
Importing this module needs no numpy, torch, pyarrow or Nano-vLLM.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2]
for _p in (str(REPO / "src"), str(_HERE.parent)):  # also when imported by path (modal_app, tests)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import prepare_alania as pa  # noqa: E402  (guard, spread order, quantiles, WAV writer; no pyarrow / numpy at import)
import onset  # noqa: E402  (no numpy at import)
import phon_check as pc  # noqa: E402  (the c-check; no torch / transformers at import)
import score_intonation as si  # noqa: E402  (no numpy at import)
import voice_contest as vc  # noqa: E402  (no torch / numpy at import)
from build_textbank import LAUGH_RE, strip_laughter  # noqa: E402  (stdlib + trnorm)
from trnorm import normalize  # noqa: E402

DATA = os.environ.get("TTS_DATA", "/data")
OUT = f"{DATA}/f022corpus"
VOICE = "designed-f022-s0"
REF = f"{DATA}/contest/r1/screen/cand/wav/f022-s0.wav"
DESCRIPTION = ("Female speaker, in their fifties, a counsellor; low-pitched and velvety; patient and polite delivery "
               "at a measured pace; recorded close to the microphone in a quiet studio.")
TEXTBANK = Path("tts/voxcpm/textbank/textbank.jsonl")  # relative = under the repo
SETS_DIR = Path("benchmarks/tts/sets")
# the probes, never trained on (make_guard): the canına probes (our c test), the Turkish-instruction probe
PROBE_GUARD = ("tts/voxcpm/probe_canina*.jsonl", "tts/voxcpm/probe_tr_instr.jsonl")
CWORDS = Path("tts/voxcpm/textbank/cwords.jsonl")  # emotion rows with a word-initial c (category "cwords")
# Antalia's held-out sentences, always guarded when here: the Volume's manifest, the local copy of their eval rows
HELDOUT = (Path(f"{DATA}/antalia16k/voxcpm_val.jsonl"), REPO / "voxcpm-eval" / "antalia-gt" / "rows.jsonl")
DOMAIN_REPO = "cloud0day3/alania-domain-text-tr"
DOMAIN_REV = "20fd903fe656dc47edd63c010de15431fc747334"  # v1.2, 2026-10-03; current on 2026-10-11
DOMAIN_ATTRIBUTION = ("Alania Turkish Domain Text by PatientDesk AI "
                      "(https://huggingface.co/datasets/cloud0day3/alania-domain-text-tr), licensed under CC BY 4.0")
DOMAIN_COLUMNS = ["line_id", "text_raw", "category", "tags", "lid_tr", "doc_id", "unit_idx"]
# its categories with numbers, times, money, dates, phones, codes, addresses, acronyms (questions, empathetic lines,
# confirmations and long explanations left out: other categories cover them, or they are not neutral sentences)
DOMAIN_WEIGHTS = {"numbers_codes": 0.25, "appointments_clinics": 0.2, "banking": 0.2, "ecommerce_cargo": 0.15,
                  "addresses_names": 0.15, "customer_support": 0.05}
DOMAIN_LINES, DOMAIN_SCAN, RICH_SHARE, LID_MIN = 6000, 120_000, 0.7, 0.9
DOMAIN_CHARS = 80  # a drawn neutral sentence's spoken form, for estimates before the pool exists (25-180, rich lines)
MIN_CHARS, MAX_CHARS = 25, 180  # neutral lines, characters of spoken form
RICH = re.compile(r"\d|\b[A-ZÇĞİÖŞÜ]{2,}\b|[%₺$€£]")  # digits (numbers, times, money, dates, phones) or acronyms
JUNK = re.compile(r"[\d@#%&*/\\_=<>{}\[\]|~^$€£₺]")  # left in a spoken form: something trnorm did not read out
_MONTHS = "ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık"
# written-form checks of a domain line (number_kinds), the most specific first; any of them = RICH
NUMBER_KINDS = {
    "phone": re.compile(r"(?<![\d.,])(?:\+90[\s-]?)?0?\d{3}[\s-]\d{3}[\s-]?\d{2}[\s-]?\d{2}(?![\d.,])"
                        r"|(?<!\d)444[\s-]?\d{1,2}[\s-]?\d{2,3}(?!\d)"),
    "time": re.compile(r"(?<![\d.,:])(?:[01]?\d|2[0-3])[:.][0-5]\d(?![\d.,:])|\bsaat\s+\d", re.I),
    "money": re.compile(r"[₺$€£]\s?\d|\d\s?(?:TL|₺|lira|kuruş|dolar|euro|avro|USD|EUR)", re.I),
    "date": re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})|(?<!\d)\d{{1,2}}[./]\d{{1,2}}[./](?:\d{{4}}|\d{{2}})(?!\d)",
                       re.I),
    "percent": re.compile(r"%\s?\d|\d\s?%|\byüzde\s+\d", re.I),
    "acronym": re.compile(r"\b[A-ZÇĞİÖŞÜ]{2,}\b"),
    "digits": re.compile(r"\d"),
    "symbol": re.compile(r"[%₺$€£]"),
}
NUMBER_SHARE = 0.6  # neutral rows with numbers: of the draw and of the kept seconds, when the pool allows
CATEGORIES = ("neutral", "questions", "emotions", "short", "interjections", "long", "cwords")
TB_CATEGORIES = ("questions", "emotions", "short", "interjections")  # the textbank's
MIX = {"neutral": 0.40, "questions": 0.25, "emotions": 0.20, "short": 0.10, "interjections": 0.05, "long": 0.0,
       "cwords": 0.0}
FILL = "neutral"
EMO_CATS = ("emotions", "cwords")  # emotion rows: the emotion's reference clip, its prosody gate, UTMOS per emotion
CWORDS_TRIES, CWORDS_SOURCE = 8, "cwords"  # tries a cwords text gets (the base model errs under instructions too)
# cwords tries rendered per text and wave (at least --prerender): ~2.5 s clips, so the few never scored cost seconds of
# GPU, while 2 a wave would make 8 tries outlast a long item's 6 by a wave (a server start and the scorer loads, ~212 s)
CWORDS_PRERENDER = 4
# --plain-render (module doc "Plain render"): the categories that may render without their instruction (the ones with
# one; the others' mode is "instructed"), the gate a plain try fails when its prosody does not show the emotion
# (plain_verdict) and the per-try pass rate guessed until a plain category measures its own. The guess, from call 1 of
# /data/f022corpus2's 590 instructed cwords tries: 0.92 made without a stray onset; Whisper heard 0.585 of the neutral
# tries right (cwords' 0.27 was the c read as k / ç / s); 0.26 of the cwords tries past CER and the c-check passed the
# rest (ECAPA failed 59 of 100: emotion references clone less like the raw voice); the stricter emotion rule ~0.8 of
# that: ~0.11-0.15
PLAIN_CATS = EMO_CATS
PLAIN_FAIL, PLAIN_PASS = "emotion_plain", 0.15
# the c-check (phon_check): its gate's name; s per checked try (wav2vec2-large fp32 on the GPU, the clip read and
# resampled; a guess until measured) and the model's first load in a call (1.3 GB, from the Volume's HF cache)
C_FAIL, PHON_CLIP_S, PHON_AUDIO_S, PHON_LOAD_S = "c_dj", 0.05, 0.01, 45.0
C_AUDIT = "c_audit.jsonl"  # --audit-c's record in a corpus dir (corpus-merge leaves its failed clips out)
# --merge --instructions: the training texts' instruction language (module doc "Merge"); the English -> Turkish table
INSTRUCTION_MODES = ("en", "both", "tr")
INSTRUCTIONS_TR = Path("tts/voxcpm/instructions_tr.json")  # relative = under the repo
# long items (module doc): 1-3 consecutive lines of one generation prompt, LONG_CHARS characters of spoken form, drawn
# from LONG_TARGET (~12.5-17 s: f022 is weak on 12-19 s sentences; the fewest lines that fit made a median of 9.9 s)
LONG_WEIGHTS = {"long_explanations": 0.25, "customer_support": 0.15, "appointments_clinics": 0.15, "banking": 0.15,
                "ecommerce_cargo": 0.1, "numbers_codes": 0.1, "addresses_names": 0.1}
LONG_LINES, LONG_CHARS, LONG_JOIN, LONG_TRIES, LONG_TARGET = 1500, (120, 300), (1, 3), 6, (190, 280)
LONG_BAND = 30  # characters: the draw takes the windows within LONG_TARGET first, then the nearest, in steps of this
LONG_PLAN_CHARS, LONG_PLAN_SENT = 225, 3  # a long item in estimates before the pool exists (2180 real rows: medians)
LONG_PAUSE_S = 0.35  # est_dur: seconds per internal sentence boundary of a long item
LONG_RENDER = 1.3  # synthesis s per audio s of a long clip over a short one's (longer sequences, a wave's long tail)
GEN_ID = re.compile(r"^(?P<group>.+)-(?P<k>\d+)$")  # the domain text's doc_id: "qwen3:s3-p<prompt>-<k>"
PAUSE_SENT, PAUSE_MID_MAX, PAUSE_MAX = (0.12, 0.9), 0.6, 1.2  # the pause gate (long clips), s
EMO_UP, EMO_DOWN = ("angry", "cheerful", "surprised"), ("sad", "calm")
BASELINE_CATS, BASELINE_MIN, BASELINE_N = ("neutral", "short"), 20, 300  # kept clips the emotion deltas compare with

STATE_VERSION = 1
MARKER = ".render_corpus"
OWN_FILES = ("state.json", "texts.jsonl", "clips.jsonl", "attempts.jsonl", "voxcpm_train.jsonl", "ATTRIBUTION.txt",
             "prep_report.json", "PREP_REPORT.md", "status.json", "reviewed.jsonl")
WRAPPER_OPTS = ("--out", "--model", "--deadline", "--throughput-log", "--backend")  # modal_app owns these
KEY_FIELDS = ("voice", "ref_sha256", "model", "timesteps", "cfg", "seed", "ref_prep", "loudness")
SEED_STRIDE = 1000  # between a text's tries: NanoPool's own retries use seed + 1 .. + 5
TRIES, PRERENDER, REF_FRACTION = 4, 2, 0.3
REF_MIN_S, REF_MAX_S = pa.REF_MIN_S, pa.REF_MAX_S  # a ref_audio clip: 3-10 s
MIN_SIM, SIM_SHORT_SLACK, SIM_FULL_S = 0.70, 0.12, 4.0
RATE = (7.0, 25.0)
SHORT_CHARS, SHORT_MIN_S, SHORT_MAX_S = 20, 0.25, 3.0
UTMOS_DROP, CALIB_N = 0.2, 40
EMO_CLEAR_ST, EMO_CLEAR_RATE = 2.0, 0.6
# the pilot's gates (module doc "Gates"; calibrated on its 99 clips and a native listener's notes)
STRETCH_MAX = 2.2  # first word's s / syllable over the median of the others' (Whisper times; interjections exempt)
SPEED_MAX = {"neutral": 8.0, "questions": 8.5, "emotions": 7.8, "short": 8.0, "interjections": 7.5,
             "long": 8.0, "cwords": 7.8}  # syllables / s
EMO_LEVEL_SLACK = 1.0  # dB: a sad / calm clip may not be louder than the neutral clips (level before levelling)
# emotion references: one short neutral sentence in f022, each emotion's instruction (a train wording), several seeds
EMO_REF_TEXT = "Yarın sabah erkenden yola çıkacağız, eşyaları akşamdan hazırladım."
EMO_REF_INSTRUCTIONS = {"sad": "Sad and quiet, with a heavy heart.", "calm": "Calm, slow and soothing.",
                        "empathetic": "Warm and compassionate, as if comforting someone.",
                        "angry": "Angry tone, voice getting louder.",
                        "cheerful": "Happy and upbeat, bright and lively.",
                        "surprised": "Surprised, voice going up in disbelief."}
EMO_REF_N, EMO_REF_NEUTRAL, EMO_REF_MIN_SIM = 6, 4, 0.6
REFS_DIR = "refs"  # <out>/refs/<emotion>.wav
# laughter trial (never in the manifest): textbank/laughter.jsonl x LAUGH_INSTRUCTIONS x LAUGH_SEEDS, at most
# --laughter-usd of GPU time; the LAUGH_KEEP best tries of each line -> <out>/laughter_review/ for a listener
LAUGHTER = Path("tts/voxcpm/textbank/laughter.jsonl")
LAUGH_INSTRUCTIONS = ("Laughing heartily while speaking.", "Speaking with a cheerful laugh.", "Giggling, amused.")
LAUGH_SEEDS, LAUGH_USD, LAUGH_KEEP, LAUGH_AST_MIN = 2, 0.3, 3, 0.1
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"  # BSD-3-Clause; used only when already in the HF cache
REVIEW = "laughter_review"
# human-approved review clips in the corpus (--add-reviewed): their records, the category and source they carry
REVIEWED, REVIEWED_CAT, REVIEWED_SOURCE = "reviewed.jsonl", "laughter", "human-approved"
LAUGH_APPROVED = Path("tts/voxcpm/textbank/laughter_approved.json")  # the listener's verdicts on laughter_review/
RUNNING = ("start", "pool", "prelude", "waves")  # status.json phases of a call still going (or killed mid-call)

# estimates until state.json / throughput.jsonl measure them. Per-try pass rate of all gates, measured: /data/f022corpus
# (PREP_REPORT.md "pass/try" of 2026-10-12: 14019 tries at --min-sim 0.6, the pilot's gates on all but its first 902).
# "long" is not measured yet, a conservative guess at the same --min-sim: CER 0 over ~3x a neutral text's characters
# (neutral tries passed it 0.58 of the time: 0.58^3 = 0.2), the other gates as for neutral (0.7 of what passed CER),
# the pause gate ~0.7 -> 0.1 (a corpus's own pass rates replace these from MIN_MEASURED decided tries). "cwords": a
# guess too (0.35 a try; emotions measured 0.225 under the same gates without the c-check: it may be lower)
MEASURED_PASS = {"neutral": 0.407, "questions": 0.265, "emotions": 0.225, "short": 0.202, "interjections": 0.201,
                 "long": 0.10, "cwords": 0.35}
MEASURED_SIM = 0.6
# a category whose MEASURED_PASS is a guess takes at most PROBE_TEXTS texts in flight until MIN_MEASURED of its tries
# are decided (a wave measures it, the next waves plan with the measurement); a measured pass rate under --min-pass
# takes no new texts at all (its open ones finish; its target is not handed to --fill: the corpus stays incomplete)
GUESSED, PROBE_TEXTS, MIN_PASS = ("long", "cwords"), 40, 0.05
# share of the pilot's kept clips the gates added on 2026-10-12 pass (99 heard: questions 3 of 31 fail, emotions 2 of
# 20 + sad / calm 7 of 21 of all kept, short 1 of 12): scales pass rates a corpus measured before them (stats
# "decided" / "passed"); tries decided since count in "decided_g2" / "passed_g2" (long: under its gates from the start)
NEW_GATE_PASS = {"neutral": 0.97, "questions": 0.9, "emotions": 0.8, "short": 0.92, "interjections": 0.97, "long": 1.0,
                 "cwords": 1.0}
# share of f022's r1 + r2 clips over sim_floor at a --min-sim (the default --sim-short-slack): scales MEASURED_PASS
SIM_PASS = {0.5: 0.976, 0.55: 0.963, 0.6: 0.939, 0.65: 0.768, 0.7: 0.598, 0.75: 0.354, 0.8: 0.073}
CHARS_S, DUR_PAD_S = 16.5, 0.25  # f022 speaks ~16-18 characters of spoken form a second (r1, median by set)
MIN_MEASURED = 30  # decided tries / kept clips before a measurement replaces the prior
# scoring, s per scored clip: r1's render call scored 910 clips (2677 s of audio) in ~463 s (610 s call - 147 s
# server; Whisper + UTMOS + ECAPA, the CPU pass beside them, scorer loads); later the corpus-score lines (a wave's,
# from SCORE_LOG_MIN scored clips: fewer say more about the loads than the clips)
SCORE_CLIP_S, SCORE_AUDIO_S, SCORE_LOG_MIN = 0.25, 0.08, 20
NANO_SERVER_S = 92.0  # a Nano-vLLM server's start, stop and last batch beyond its generation (r1: 146.9 - 54.5 s)
ONSET_REDO = 0.2  # NanoPool's onset retries: ~16 % of first tries made again
POOL_S, FINAL_S, MIN_WAVE_S = 120.0, 30.0, 300.0  # pool build (first call); finalize; least time a wave starts with
# this many failures of one kind in a wave or round, and this share of it, is the engine's or a scorer's fault, not
# the texts': the call stops with an error and spends no try on them
SYSTEMIC_N, SYSTEMIC_SHARE = 10, 0.5
USD_PER_H = 2.50 + 8 * 0.047 + 32 * 0.008  # A100-80GB + 8 cores + 32 GiB (modal_app's corpus-render container)


# ---- small helpers ------------------------------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[corpus] {msg}", flush=True)


def collapse(s: str) -> str:
    return " ".join((s or "").split())


def spoken_of(text: str) -> str:
    """The spoken form a model reads (trnorm.normalize), whitespace collapsed."""
    return collapse(normalize(text or ""))


def clean_id(x) -> str:
    """A text id as a file name (letters, digits, . _ -), at most 64 characters; "" when nothing is left."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(x if x is not None else "")).strip("._-")[:64]


def rank(seed: int, key: str) -> str:
    """A deterministic shuffle key."""
    return hashlib.sha1(f"{seed}|{key}".encode()).hexdigest()


def quantile(xs: list[float], q: float) -> float | None:
    """The q-quantile of xs (nearest rank, lower); None when empty."""
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, max(0, int(math.floor(q * len(xs)))))] if xs else None


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    pa.write_atomic(p, text)


def append_rows(p: Path, rows: list[dict]) -> None:
    if rows:
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def parse_mix(s: str) -> dict[str, float]:
    """"neutral=0.4,questions=0.25" -> shares of the categories (normalised to 1; the others 0). ValueError on an
    unknown category, a negative share or none."""
    out = dict.fromkeys(CATEGORIES, 0.0)
    for item in (x.strip() for x in s.split(",") if x.strip()):
        k, sep, v = item.partition("=")
        try:
            val = float(v)
        except ValueError:
            val = -1.0
        if not sep or k.strip() not in out or not (math.isfinite(val) and val >= 0):
            raise ValueError(f"--mix {item!r}: want <category>=<share>, categories {', '.join(CATEGORIES)}")
        out[k.strip()] = val
    total = sum(out.values())
    if total <= 0:
        raise ValueError("--mix: no category with a share > 0")
    return {k: v / total for k, v in out.items()}


def parse_weights(s: str, opt: str = "--domain-weights") -> dict[str, float]:
    """"numbers_codes:0.25,banking:0.2" -> {category: weight > 0}."""
    out = {}
    for item in (x.strip() for x in s.split(",") if x.strip()):
        k, sep, v = item.partition(":")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            sep = ""
        if not sep or not k.strip() or not out.get(k.strip(), 0) > 0:
            raise ValueError(f"{opt} {item!r}: want <dataset category>:<weight > 0>")
    if not out:
        raise ValueError(f"{opt}: none")
    return out


def parse_pair(s: str, opt: str) -> tuple[float, float]:
    """"0.12,0.9" -> (0.12, 0.9), 0 <= lo < hi (ValueError otherwise)."""
    try:
        lo, hi = (float(x) for x in s.split(","))
    except ValueError:
        raise ValueError(f"{opt} {s!r}: want <low>,<high>") from None
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
        raise ValueError(f"{opt} {s!r}: want 0 <= low < high")
    return lo, hi


def parse_plain(s: str | None) -> set[str] | None:
    """--plain-render: None when not given, set() for "none", else its categories (PLAIN_CATS; ValueError on another
    or none)."""
    if s is None:
        return None
    items = {x.strip() for x in s.split(",") if x.strip()}
    if items == {"none"}:
        return set()
    if not items or (bad := items - set(PLAIN_CATS)):
        raise ValueError(f"--plain-render {s!r}: want 'none' or categories with an instruction: {', '.join(PLAIN_CATS)}"
                         + (f" (not {', '.join(sorted(bad))})" if items else ""))
    return items


def emotion_of(r: dict) -> str | None:
    e = r.get("emotion") or r.get("kind")
    return vc.EMOTION_ALIASES.get(str(e).lower(), str(e).lower()) if e else None


# ---- texts --------------------------------------------------------------------------------------------------------


def peak_index(words: list[str], peak_word: str | None, pattern: str) -> int | None:
    """Word index of an intonation row's peak in its spoken form (score_intonation.set_item checks it): peak_word's
    place (peak_fall: the one right before mI; wh_peak: the first; else the last), or without a peak_word the scorer's
    own reading of the text (score_intonation.infer_items); None when neither finds it."""
    cores = [si.core(w) for w in words]
    if peak_word:
        hits = [i for i, c in enumerate(cores) if c and c == si.core(peak_word)]
        if not hits:
            return None
        if pattern == "peak_fall":
            before = [i for i in hits if i + 1 < len(cores) and si.MI.match(cores[i + 1])]
            return (before or hits)[-1]
        return hits[0] if pattern == "wh_peak" else hits[-1]
    items = [it for it in si.infer_items(words) if it["pattern"] == pattern and not it.get("unscored")]
    return items[-1]["target"] if items else None


def textbank_row(r: dict) -> tuple[dict | None, str]:
    """A pool row of one textbank row (module doc), or (None, why it is left out); the second value also carries a
    note on a kept row (its "spoken" differs from trnorm's, a question kind without an intonation check)."""
    if not isinstance(r, dict):
        return None, "not an object"
    cat, text, rid = r.get("category"), r.get("text"), clean_id(r.get("id"))
    if cat not in TB_CATEGORIES:
        return None, f"category {cat!r}"
    if not isinstance(text, str) or not text.strip() or not rid:
        return None, "no id or text"
    spoken, note = spoken_of(text), ""
    if not spoken:
        return None, "empty spoken form"
    if isinstance(r.get("spoken"), str) and collapse(r["spoken"]) != spoken:
        note = "spoken differs from trnorm.normalize(text) (trnorm's used)"
    row = {"id": rid, "category": cat, "text": collapse(text), "spoken": spoken, "source": "textbank"}
    if r.get("kind"):
        row["kind"] = str(r["kind"])
    if r.get("pair_id") is not None:
        row["pair_id"] = str(r["pair_id"])
    if cat == "questions":
        pattern = r.get("pattern") if r.get("pattern") in si.NEEDS else si.PATTERNS.get(row.get("kind", ""))
        pw, words = r.get("peak_word"), spoken.split()
        if not pattern:
            note = f"questions: kind {row.get('kind')!r} has no intonation check"
        elif pw is None and pattern in ("peak_fall", "wh_peak"):
            note = "questions: no peak_word (the textbank was unsure): intonation not gated"
        else:
            if isinstance(pw, int) and not isinstance(pw, bool):  # build_textbank: an index into the spoken form
                k = pw if 0 <= pw < len(words) and si.core(words[pw]) else None
            else:
                k = peak_index(words, pw if isinstance(pw, str) else None, pattern)
            if k is None:
                return None, f"questions/{row.get('kind')}: peak_word {pw!r} not in the spoken form"
            row.update(pattern=pattern, peak=k, peak_word=si.core(words[k]))
    elif cat == "emotions":
        if r.get("instruction_split") == "heldout":
            return None, "emotions: held-out instruction wording (evaluation only)"
        try:
            instruction = vc.strip_instruction(r.get("instruction"))
        except ValueError as e:
            return None, f"emotions: {e}"
        if not instruction:
            return None, "emotions: no instruction"
        row.update(emotion=emotion_of(r), instruction=instruction)
    elif cat == "interjections":
        rts, span = r.get("render_texts"), r.get("span")
        if isinstance(rts, list) and rts:  # build_textbank: the spoken form with each variant in the span's place
            vs = [collapse(v) for v in rts if isinstance(v, str)]
        else:  # render_variants: spellings of the span (or of the whole text when there is no span in it)
            vs = [spoken_of(text.replace(span, v, 1) if isinstance(span, str) and span and span in text else v)
                  for v in r.get("render_variants") or [] if isinstance(v, str) and v.strip()]
        row["variants"] = list(dict.fromkeys(v for v in vs if v)) or [spoken]
        if r.get("interjection"):
            row["interjection"] = str(r["interjection"])
    return row, note


def load_textbank(path: Path) -> tuple[list[dict], collections.Counter, str]:
    """(pool rows, problems / notes by reason, sha256 of the file); ([], {"no textbank": 1}, "") without one."""
    if not path.is_file():
        return [], collections.Counter({f"no textbank at {path}": 1}), ""
    rows, bad, seen = [], collections.Counter(), set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad["not JSON"] += 1
            continue
        row, why = textbank_row(r)
        if why:
            bad[why] += 1
        if row is None:
            continue
        if row["id"] in seen:
            bad["duplicate id"] += 1
            continue
        seen.add(row["id"])
        rows.append(row)
    return rows, bad, sha256_file(path)


def load_cwords(path: Path) -> tuple[list[dict], collections.Counter, str]:
    """The cwords rows (module doc): each row of path read as an emotion row (textbank_row: held-out wordings, rows
    without an instruction left out), category "cwords", source CWORDS_SOURCE; a row whose spoken form has no "c" is
    left out. (rows, problems by reason, sha256); ([], {"no cwords file": 1}, "") without the file."""
    if not path.is_file():
        return [], collections.Counter({f"no cwords file at {path}": 1}), ""
    rows, bad, seen = [], collections.Counter(), set()
    for r in vc.read_rows(path):
        row, why = textbank_row({**r, "category": "emotions"} if isinstance(r, dict) else r)
        if row is not None and not pc.c_expected(row["spoken"]):
            row, why = None, "no c in the spoken form"
        if why:
            bad[why.replace("emotions: ", "cwords: ")] += 1
        if row is None:
            continue
        if row["id"] in seen:
            bad["duplicate id"] += 1
            continue
        seen.add(row["id"])
        rows.append({**row, "category": "cwords", "source": CWORDS_SOURCE})
    return rows, bad, sha256_file(path)


def protected_texts(p: Path) -> list[str]:
    """prepare_alania.protected_texts of a jsonl file (text, spoken, text_spoken, alts) plus each row's "refs"
    (contest.jsonl's references)."""
    out = pa.protected_texts(p)
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            refs = json.loads(line).get("refs") if line.strip() else None
        except (ValueError, AttributeError):
            continue
        out += [refs] if isinstance(refs, str) else [x for x in refs or [] if isinstance(x, str)]
    return out


def heldout_files(exclude: list[Path]) -> list[Path]:
    """Antalia's held-out sentences wherever they are here (HELDOUT: the Volume's manifest, the local eval copy), and
    the --exclude-manifest files, each once."""
    return list(dict.fromkeys([*(p for p in HELDOUT if p.is_file()), *exclude]))


def probe_files() -> list[Path]:
    """The probe sets the guard keeps out of training (PROBE_GUARD under the repo: the canına probes, the
    Turkish-instruction probe)."""
    return sorted({p for g in PROBE_GUARD for p in REPO.glob(g)})


def make_guard(sets_dir: Path, exclude: list[Path]) -> tuple[pa.Guard, list[str]]:
    """The leakage guard (prepare_alania.build_guard) over every benchmarks/tts/sets/*.jsonl row (refs too), the
    probes (probe_files: text and refs), Antalia's held-out rows (heldout_files) and the --exclude-manifest rows;
    and the files it read."""
    files = sorted(sets_dir.glob("*.jsonl")) + probe_files() + heldout_files(list(exclude))
    texts = [t for p in files for t in protected_texts(p)]
    return pa.build_guard(texts), [str(p) for p in files]


def final_mark(text: str) -> str:
    """The class of a text's final mark, as build_textbank keys it: "?", "..." or "." (statements: . ! or none), so a
    statement and the same words asked ("Pardon!" / "Pardon?") are two texts with two intonations."""
    t = (text or "").rstrip().rstrip("\"'”’)»").rstrip()
    return "?" if t.endswith("?") else "..." if t.endswith(("...", "…")) else "."


def dedup_key(spoken: str) -> str:
    """De-duplication key of a spoken form: its normalize_for_eval form + its final mark (final_mark)."""
    return f"{pa.eval_key(spoken)}|{final_mark(spoken)}"


class Admit:
    """Leakage guard (normalize_for_eval form of text and spoken form) and de-duplication (dedup_key) of pool
    texts."""

    def __init__(self, guard: pa.Guard, seen: set[str] | None = None):
        self.guard, self.seen = guard, set(seen or ())

    def why(self, row: dict) -> str:
        if self.guard.hit(pa.eval_key(row["spoken"]), pa.eval_key(row.get("text") or "")):
            return "leakage guard"
        key = dedup_key(row["spoken"])
        if key in self.seen:
            return "duplicate text"
        self.seen.add(key)
        return ""


def domain_reason(r: dict, weights: dict[str, float]) -> str:
    """Why a domain-text row is not a neutral candidate ("" = it is), before normalising (cheap)."""
    text = collapse(r.get("text_raw") or "")
    tags = r.get("tags") or []
    tags = tags if isinstance(tags, (list, tuple)) else str(tags).split(",")
    if r.get("category") not in weights:
        return "category"
    if not text or text.endswith("?") or "?" in text or any("question" in str(t) for t in tags):
        return "question"
    if r.get("lid_tr") is not None and not float(r["lid_tr"]) >= LID_MIN:
        return "language-ID"
    if not 15 <= len(text) <= 2 * MAX_CHARS:
        return "length"
    return ""


def number_kinds(text: str) -> list[str]:
    """The kinds of written numbers in a text (NUMBER_KINDS: phone, time, money, date, percent, acronym, digits,
    symbol), in that order; [] for none (non-empty exactly when RICH finds something)."""
    return [k for k, rx in NUMBER_KINDS.items() if rx.search(text or "")]


def has_numbers(row: dict) -> bool:
    """A pool row or kept clip with numbers in its written text (its "kinds", or number_kinds of its text)."""
    kinds = row.get("kinds")
    return bool(kinds) if isinstance(kinds, list) else bool(number_kinds(row.get("text") or ""))


def kind_spread(rows: list[dict]) -> list[dict]:
    """rows (domain lines with numbers, in their order) interleaved by their first number kind, so any prefix covers
    every kind the rows have: phones, times, money, dates and percentages as often as plain digits."""
    by = {k: collections.deque() for k in NUMBER_KINDS}
    for r in rows:
        by[(number_kinds(r.get("text_raw") or "") or ["digits"])[0]].append(r)
    out = []
    while any(by.values()):
        out += [q.popleft() for q in by.values() if q]
    return out


def domain_row(r: dict) -> tuple[dict | None, str]:
    """The pool row of a neutral candidate (spoken form 25-180 characters, nothing left unread), else (None, why)."""
    text = collapse(r["text_raw"])
    spoken = spoken_of(text)
    if not MIN_CHARS <= len(spoken) <= MAX_CHARS:
        return None, "spoken length"
    if JUNK.search(spoken):
        return None, "digits or symbols left"
    lid = clean_id(r.get("line_id"))
    if not lid:
        return None, "no line_id"
    kinds = number_kinds(text)
    return {"id": f"d{lid}"[:64], "category": "neutral", "text": text, "spoken": spoken, "source": DOMAIN_REPO,
            "line_id": str(r.get("line_id")), "domain_category": r["category"], "rich": bool(kinds),
            "kinds": kinds}, ""


def pick_domain(rows: list[dict], n: int, weights: dict[str, float], rich_share: float, admit: Admit, seed: int,
                have: set[str] | None = None, number_share: float = 0.0) -> tuple[list[dict], collections.Counter]:
    """n neutral pool rows: per dataset category round(n x weight) (what a category lacks goes to the others), of
    each --rich-share from lines with numbers first (number_kinds; spread over the kinds: kind_spread), in a seeded
    order; then, while lines with numbers are short of number_share of the draw, more of them from any category in
    place of the last plain ones; rows normalised, guarded and de-duplicated as they are taken. have: line ids
    already in the pool (taken, not counted). Returns (rows, drops)."""
    drops, by = collections.Counter(), collections.defaultdict(lambda: ([], []))
    for r in rows:
        if why := domain_reason(r, weights):
            drops[why] += 1
        elif str(r.get("line_id")) not in (have or set()):
            by[r["category"]][0 if RICH.search(r["text_raw"]) else 1].append(r)
    key = (lambda r: rank(seed, str(r["line_id"])))
    queues = {c: (collections.deque(kind_spread(sorted(rich, key=key))), collections.deque(sorted(plain, key=key)))
              for c, (rich, plain) in by.items()}
    total_w = sum(weights.values())
    quota = {c: round(n * w / total_w) for c, w in weights.items()}
    picked: list[dict] = []
    taken = collections.Counter()

    def take(c: str, k: int, which: int) -> int:
        got = 0
        lst = queues[c][which] if c in queues else collections.deque()
        while got < k and lst:
            row, why = domain_row(lst.popleft())
            if row is not None and not (why := admit.why(row)):
                picked.append(row)
                got += 1
            else:
                drops[why] += 1
        return got

    for c in weights:  # rich lines first, then plain ones; a short list hands its rest to the other
        want_rich = round(quota[c] * rich_share)
        got = take(c, want_rich, 0)
        got += take(c, quota[c] - got, 1)
        taken[c] = got + take(c, quota[c] - got, 0)
    for c in sorted(weights, key=lambda c: -weights[c]):  # categories that ran short: the rest from the others
        if len(picked) >= n:
            break
        take(c, n - len(picked), 0)
        take(c, n - len(picked), 1)
    picked = picked[:n]
    need = math.ceil(number_share * len(picked) - 1e-9) - sum(r["rich"] for r in picked)
    if need > 0:  # top up the lines with numbers from any category, the last plain lines make room
        start = len(picked)
        for c in sorted(weights, key=lambda c: -weights[c]):
            need -= take(c, need, 0)
            if need <= 0:
                break
        extra = len(picked) - start
        drop = {i for i in sorted((i for i, r in enumerate(picked[:start]) if not r["rich"]), reverse=True)[:extra]}
        picked = [r for i, r in enumerate(picked) if i not in drop][:n]
    return picked, drops


# ---- long items -----------------------------------------------------------------------------------------------------


def doc_group(r: dict) -> tuple[str, tuple[int, int]] | None:
    """(group, place) of a domain line: its generation prompt (doc_id "qwen3:s3-p<prompt>-<k>" without "-<k>") and
    (k, unit_idx); a doc_id without "-<k>" is a group of its own, in unit_idx order. None without a doc_id."""
    d = str(r.get("doc_id") or "")
    if not d:
        return None
    try:
        u = int(r.get("unit_idx") or 0)
    except (TypeError, ValueError):
        u = 0
    m = GEN_ID.match(d)
    return (m["group"], (int(m["k"]), u)) if m else (d, (0, u))


def long_runs(rows: list[dict], weights: dict[str, float], have: set[str]
              ) -> tuple[dict[str, list[tuple[str, list[dict]]]], collections.Counter]:
    """{dataset category: [(group, lines), ...]}: runs of neighbouring lines of one group (doc_group order) that may
    be joined: a line domain_reason rejects, one in `have`, or a k missing from the rows read ends a run. Rows of other
    categories are not looked at."""
    drops, groups = collections.Counter(), collections.defaultdict(list)
    for r in rows:
        if r.get("category") not in weights:
            continue
        g = doc_group(r)
        if g is None:
            drops["long: no doc_id"] += 1
            continue
        groups[(r["category"], g[0])].append((g[1], r))
    out = collections.defaultdict(list)
    for (cat, g), lines in groups.items():
        run: list[dict] = []
        prev = None
        for place, r in sorted(lines, key=lambda x: x[0]) + [((math.inf, 0), None)]:
            gap = prev is not None and place[0] > prev + 1
            prev = place[0]
            if gap and len(run) >= LONG_JOIN[0]:
                out[cat].append((g, run))
            run = [] if gap else run
            if r is not None and not domain_reason(r, weights) and str(r.get("line_id")) not in have:
                run.append(r)
                continue
            if len(run) >= LONG_JOIN[0]:
                out[cat].append((g, run))
            run = []
    return out, drops


def long_windows(run: list[dict], chars: tuple[float, float], join: tuple[int, int] = LONG_JOIN,
                 target: tuple[float, float] = LONG_TARGET, sizes: dict[int, int] | None = None) -> list[list[dict]]:
    """Non-overlapping windows of join[0]..join[1] neighbouring lines of a run, left to right: at each line the window
    whose spoken form (long_line of each, joined by spaces) lies within chars and is nearest the middle of target
    (fewer lines on a tie); a line no window starts at is passed. sizes: id(line) -> its long_line length (a cache
    across runs)."""
    sizes, mid = {} if sizes is None else sizes, (target[0] + target[1]) / 2
    out, i = [], 0
    while i < len(run):
        best, n = None, -1
        for size in range(1, min(join[1], len(run) - i) + 1):
            r = run[i + size - 1]
            if id(r) not in sizes:
                sizes[id(r)] = len(long_line(r))
            n += sizes[id(r)] + 1
            if n > chars[1]:
                break
            if size >= join[0] and n >= chars[0] and (best is None or abs(n - mid) < abs(best[1] - mid)):
                best = (size, n)
        if best:
            out.append(run[i:i + best[0]])
        i += best[0] if best else 1
    return out


def with_mark(text: str) -> str:
    """A sentence with its final mark: text, or text + "." when it ends without . ! ? or an ellipsis (a trailing
    , ; : or dash replaced)."""
    text = text.rstrip()
    return text if re.search(r"[.!?…][\"'”’)»]*$", text) else text.rstrip(",;:–—- ") + "."


def long_line(r: dict) -> str:
    """A domain line as a long item reads it: its spoken form (trnorm) with its sentence mark."""
    return with_mark(spoken_of(with_mark(collapse(r["text_raw"]))))


def long_row(lines: list[dict], cat: str, group: str, chars: tuple[float, float]
             ) -> tuple[dict | None, str, list[str]]:
    """The pool row of a long item (module doc) of these lines, or (None, why); and the lines' written + spoken forms
    (the leakage guard looks at each). breaks: the index (in spoken.split()) of each line's last word but the last
    line's, the sentence boundaries the pause gate checks (a "." trnorm left inside a line, e.g. an initial "ka." or an
    ordinal "beş.", is none); sentences: the lines."""
    texts = [with_mark(collapse(r["text_raw"])) for r in lines]
    spokens = [long_line(r) for r in lines]
    spoken, text = " ".join(spokens), " ".join(texts)
    if not chars[0] <= len(spoken) <= chars[1]:
        return None, "long: spoken length", []
    if JUNK.search(spoken):
        return None, "long: digits or symbols left", []
    ids = [str(r.get("line_id")) for r in lines]
    if not clean_id(ids[0]):
        return None, "long: no line_id", []
    ends = list(itertools.accumulate(len(s.split()) for s in spokens))
    row = {"id": f"L{clean_id(ids[0])}"[:64], "category": "long", "text": text, "spoken": spoken, "source": DOMAIN_REPO,
           "line_ids": ids, "doc_group": group, "domain_category": cat, "sentences": len(lines),
           "breaks": [k - 1 for k in ends[:-1]], "kinds": number_kinds(text)}
    return row, "", texts + spokens


def build_long(rows: list[dict], n: int, weights: dict[str, float], admit: Admit, seed: int,
               have: set[str] | None = None, chars: tuple[float, float] = LONG_CHARS,
               target: tuple[float, float] = LONG_TARGET) -> tuple[list[dict], collections.Counter]:
    """n long items: per dataset category round(n x weight) of its windows (long_runs, long_windows), the ones within
    target first, then the nearest in LONG_BAND steps (seeded within a step: the dataset offers far more windows than a
    pool takes), what a category lacks from the others; each line through the leakage guard, the whole text guarded and
    de-duplicated (Admit). have: line ids not to use (the pool's, --avoid-pool's). Returns (rows, drops)."""
    runs, drops = long_runs(rows, weights, set(have or ()))
    sizes: dict[int, int] = {}
    windows = {c: [(g, w) for g, run in rs for w in long_windows(run, chars, LONG_JOIN, target, sizes)]
               for c, rs in runs.items()}

    def order(x: tuple[str, list[dict]]) -> tuple[int, str]:  # within the target first, then the nearest; seeded
        size = sum(sizes[id(r)] for r in x[1]) + len(x[1]) - 1
        off = max(target[0] - size, size - target[1], 0)
        return math.ceil(off / LONG_BAND), rank(seed, str(x[1][0].get("line_id")))

    queues = {c: collections.deque(sorted(ws, key=order)) for c, ws in windows.items()}
    total_w = sum(weights.values())
    quota = {c: round(n * w / total_w) for c, w in weights.items()}
    picked: list[dict] = []

    def take(c: str, k: int) -> None:
        got, q = 0, queues.get(c, collections.deque())
        while got < k and q:
            g, lines = q.popleft()
            row, why, parts = long_row(lines, c, g, chars)
            if row is not None and admit.guard.hit(*(pa.eval_key(x) for x in parts)):
                row, why = None, "leakage guard"
            if row is not None and not (why := admit.why(row)):
                picked.append(row)
                got += 1
            else:
                drops[why] += 1

    for c in weights:
        take(c, quota[c])
    for c in sorted(weights, key=lambda c: -weights[c]):
        if len(picked) >= n:
            break
        take(c, n - len(picked))
    return picked[:n], drops


class DomainSource:
    """cloud0day3/alania-domain-text-tr at DOMAIN_REV on the Hub (HfFileSystem range reads of DOMAIN_COLUMNS only;
    pyarrow and huggingface_hub imported here)."""

    def __init__(self, revision: str = DOMAIN_REV):
        from huggingface_hub import HfFileSystem

        self.revision, self.fs = revision, HfFileSystem()

    def list_files(self) -> list[str]:
        from huggingface_hub import HfApi

        tree = HfApi().list_repo_tree(DOMAIN_REPO, path_in_repo="data", repo_type="dataset", revision=self.revision)
        return sorted(f.path for f in tree if f.path.endswith(".parquet"))

    def read(self, name: str, max_rows: int) -> list[dict]:
        """DOMAIN_COLUMNS of the row groups of `name` in a spread order until max_rows."""
        import pyarrow.parquet as pq

        with self.fs.open(f"datasets/{DOMAIN_REPO}@{self.revision}/{name}", cache_type="none") as f:
            pf = pq.ParquetFile(f, pre_buffer=True)
            groups, n = [], 0
            for g in pa.spread_order(pf.metadata.num_row_groups):
                groups.append(g)
                n += pf.metadata.row_group(g).num_rows
                if n >= max_rows:
                    break
            cols = [c for c in DOMAIN_COLUMNS if c in pf.schema_arrow.names]
            return pf.read_row_groups(sorted(groups), columns=cols).to_pylist()


def fetch_domain(source, scan: int) -> tuple[list[dict], dict]:
    """About `scan` rows spread over every file of the dataset; (rows, record of what was read)."""
    files = pa.retry(source.list_files, what="domain text files")
    per = max(1, math.ceil(scan / max(len(files), 1)))
    rows = []
    for name in files:
        rows += pa.retry(source.read, name, per, what=f"domain text {name}")
    return rows, {"repo": DOMAIN_REPO, "revision": getattr(source, "revision", DOMAIN_REV), "files": len(files),
                  "rows_read": len(rows)}


def line_ids_of(rows: list[dict]) -> set[str]:
    """The domain lines pool rows were made of (a neutral row's line_id, a long item's line_ids)."""
    out = set()
    for r in rows:
        out.update(str(x) for x in [r.get("line_id"), *(r.get("line_ids") or [])] if x is not None and x != "")
    return out


def avoid_of(dirs: list[Path]) -> tuple[set[str], set[str], dict]:
    """--avoid-pool: (domain line ids, dedup keys, record) of the texts.jsonl of each dir (another corpus's pool)."""
    ids, keys, rec = set(), set(), {}
    for d in dirs:
        rows = vc.read_rows(Path(d) / "texts.jsonl")
        ids |= line_ids_of(rows)
        keys |= {dedup_key(r["spoken"]) for r in rows if isinstance(r.get("spoken"), str)}
        rec[str(d)] = len(rows)
    return ids, keys, {"pools": rec, "line_ids": len(ids), "texts": len(keys)}


def build_pool(old: list[dict], textbank: list[dict], admit: Admit, domain_lines: int, weights: dict[str, float],
               rich_share: float, seed: int, domain_rows: Callable[[], tuple[list[dict], dict]] | None, *,
               long_lines: int = 0, long_weights: dict[str, float] | None = None,
               long_chars: tuple[float, float] = LONG_CHARS, long_target: tuple[float, float] = LONG_TARGET,
               number_share: float = 0.0, avoid: tuple[set[str], set[str]] | None = None,
               cwords: list[dict] | None = None) -> tuple[list[dict], dict]:
    """The pool grown from `old` (texts.jsonl; never reordered or dropped): textbank rows and then cwords rows with new
    ids (guarded, de-duplicated, in a seeded order) and, when it holds fewer than domain_lines neutral rows or
    long_lines long items, rows from domain_rows() (pick_domain, then build_long on the lines left; None: none
    fetched). avoid: (line ids, dedup keys) of other corpora's pools (--avoid-pool), never taken. Returns (pool, record
    of what was added and dropped)."""
    pool, ids = list(old), {r["id"] for r in old}
    avoid_ids, avoid_keys = avoid or (set(), set())
    admit.seen |= avoid_keys
    for r in old:
        admit.seen.add(dedup_key(r["spoken"]))
    rec = {"textbank_added": 0, "textbank_dropped": {}, "domain_added": 0, "domain_dropped": {}, "domain_read": None,
           "long_added": 0, "long_dropped": {}, "cwords_added": 0, "cwords_dropped": {}}
    for name, rows in (("textbank", textbank), ("cwords", cwords or [])):
        drops = collections.Counter()
        for row in sorted((r for r in rows if r["id"] not in ids), key=lambda r: rank(seed, r["id"])):
            if why := admit.why(row):
                drops[why] += 1
                continue
            pool.append(row)
            ids.add(row["id"])
            rec[f"{name}_added"] += 1
        rec[f"{name}_dropped"] = dict(drops)
    want = domain_lines - sum(r["category"] == "neutral" for r in pool)
    want_long = long_lines - sum(r["category"] == "long" for r in pool)
    if domain_rows is not None and (want > 0 or want_long > 0):
        rows, rec["domain_read"] = domain_rows()
        line_ids = line_ids_of(pool) | set(avoid_ids)
        if want > 0:
            got, ddrops = pick_domain(rows, want, weights, rich_share, admit, seed, line_ids, number_share)
            got = [r for r in got if r["id"] not in ids]
            pool += sorted(got, key=lambda r: rank(seed, r["id"]))
            ids |= {r["id"] for r in got}
            line_ids |= line_ids_of(got)
            rec["domain_added"], rec["domain_dropped"] = len(got), dict(ddrops)
        if want_long > 0:
            got, ldrops = build_long(rows, want_long, long_weights or LONG_WEIGHTS, admit, seed, line_ids, long_chars,
                                     long_target)
            got = [r for r in got if r["id"] not in ids]
            pool += sorted(got, key=lambda r: rank(seed, r["id"]))
            rec["long_added"], rec["long_dropped"] = len(got), dict(ldrops)
    return pool, rec


def train_text(row: dict) -> str:
    """The manifest text: the spoken form; "(instruction)" + spoken form on emotion rows (rendered plain too)."""
    return vc.compose(row.get("instruction"), row["spoken"])


def is_plain(row: dict) -> bool:
    """A row this call renders without its instruction (with_render: its category is in plain mode)."""
    return row.get("render") == "plain"


def max_tries(row: dict, tries: int, long_tries: int = 0, cwords_tries: int = 0) -> int:
    """Tries a text gets: --tries (long items --long-tries, cwords --cwords-tries when given), interjections one more
    per extra render variant."""
    own = {"long": long_tries, "cwords": cwords_tries}.get(row.get("category") or "")
    return (own or tries) + max(0, len(row.get("variants") or [None]) - 1)


def try_end(row: dict, st: dict | None, tries: int, long_tries: int = 0, cwords_tries: int = 0) -> int:
    """The try a text's budget ends before: max_tries from its first try in its render mode (try0: the tries an earlier
    mode used, switch_mode; their seeds are never used again)."""
    return int((st or {}).get("try0") or 0) + max_tries(row, tries, long_tries, cwords_tries)


def variant_of(row: dict, t: int, st: dict | None) -> int:
    """The render variant of try t: one that passed the text gate before, else variants in turn."""
    vs = row.get("variants") or []
    if not vs:
        return 0
    ok = (st or {}).get("var_ok")
    return ok if ok is not None and 0 <= ok < len(vs) else t % len(vs)


def render_spoken(row: dict, v: int) -> str:
    vs = row.get("variants") or []
    return vs[v] if vs else row["spoken"]


def render_text(row: dict, v: int) -> str:
    """What the model gets: the (variant) spoken form; "(instruction)" before it on emotion rows, not on a plain one
    (is_plain: its emotion's reference clip carries the emotion; the manifest keeps the instruction)."""
    return render_spoken(row, v) if is_plain(row) else vc.compose(row.get("instruction"), render_spoken(row, v))


def seed_of(base: int, row: dict, t: int) -> int:
    """Try t of a text: its own seed (voice_contest.text_seed of category/id) + SEED_STRIDE x t."""
    return vc.text_seed(base, row["category"], row["id"]) + SEED_STRIDE * t


def est_dur(row: dict, chars_s: float = CHARS_S) -> float:
    """Expected seconds of a text's clip (measured speech rate of the voice; a long item LONG_PAUSE_S more per
    internal sentence boundary)."""
    pauses = LONG_PAUSE_S * max(0, int(row.get("sentences") or 1) - 1) if row.get("category") == "long" else 0.0
    return DUR_PAD_S + len(render_spoken(row, 0)) / chars_s + pauses


def render_factor(cat: str) -> float:
    """Synthesis time per audio second of a category over the measured one (LONG_RENDER for long clips)."""
    return LONG_RENDER if cat == "long" else 1.0


# ---- gates --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Gates:
    min_sim: float = MIN_SIM
    sim_short_slack: float = SIM_SHORT_SLACK
    min_p_tr: float = vc.P_TR_OK
    rate_lo: float = RATE[0]
    rate_hi: float = RATE[1]
    utmos_drop: float = UTMOS_DROP
    calib_n: int = CALIB_N
    emo_clear_st: float = EMO_CLEAR_ST
    emo_clear_rate: float = EMO_CLEAR_RATE
    stretch_max: float = STRETCH_MAX
    sib_max: float = onset.SIB_MAX_DB
    speed_scale: float = 1.0
    pause_lo: float = PAUSE_SENT[0]
    pause_hi: float = PAUSE_SENT[1]
    pause_mid: float = PAUSE_MID_MAX
    pause_max: float = PAUSE_MAX
    c_check: bool = True


def sim_floor(g: Gates, dur: float) -> float:
    """The ECAPA similarity a clip of dur seconds needs: --min-sim from SIM_FULL_S, --sim-short-slack less at 1 s."""
    short = min(max((SIM_FULL_S - dur) / (SIM_FULL_S - 1.0), 0.0), 1.0)
    return round(g.min_sim - g.sim_short_slack * short, 4)


def rate_fails(n_chars: int, dur: float, g: Gates) -> list[str]:
    if n_chars < SHORT_CHARS:
        return [] if SHORT_MIN_S <= dur <= SHORT_MAX_S else ["duration"]
    return [] if g.rate_lo <= n_chars / max(dur, 1e-6) <= g.rate_hi else ["duration"]


def cheap_fails(att: dict, g: Gates) -> list[str]:
    """Synthesis-time gates: made, not runaway, onset not cut, a sane length."""
    if att.get("synth_error") or not att.get("duration"):
        return ["synthesis"]
    out = ["runaway"] if att.get("bad_case") else []
    if att.get("onset_trimmed"):
        out.append("onset")
    return out + rate_fails(len(att["model_text"]), att["duration"], g)


def text_fails(att: dict, g: Gates) -> list[str]:
    """Whisper gates: CER 0 against the canonical spoken form, P(tr)."""
    if att.get("whisper_error") or att.get("hyp") is None:
        return ["whisper"]
    out = [] if att.get("char_errors") == 0 else ["cer"]
    return out + ([] if (att.get("p_tr") or 0.0) >= g.min_p_tr else ["p_tr"])


def needs_c(row: dict, text: str | None = None) -> bool:
    """A row the c-check gates: an instruction (emotions, cwords) and a "c" in its spoken form (or text: the render
    variant a try read). Rows without an instruction (neutral, long, ...) were heard right without one: not checked (a
    check costs tries); plain-rendered rows keep theirs (render_text drops it from the model text only): checked."""
    return bool(row.get("instruction")) and pc.c_expected(text if text is not None else row.get("spoken") or "") > 0


def c_fails(a: dict, row: dict, g: Gates) -> list[str]:
    """The c-check (phon_check.c_check) of a try of a needs_c row: as many /dʒ/ phones as runs of c (c_got >=
    c_expected); a try without the counts (the IPA model could not read it) fails. [] when off or not needed."""
    if not (g.c_check and needs_c(row, a.get("model_text"))):
        return []
    ok = a.get("c_got") is not None and a.get("c_expected") is not None and a["c_got"] >= a["c_expected"]
    return [] if ok else [C_FAIL]


def clip_fails(a: dict, row: dict, g: Gates) -> list[str]:
    """The pilot's gates on a try's clip measures (measure_clip): tail (a stray inside the last word or too long to
    cut; a short one after it is cut, not failed), head (a voiced burst before a voiceless start, or sound before the
    first word), stretch (first word, not interjections), sibilance, speed (SPEED_MAX of the category)."""
    ed, out = a.get("edges") or {}, []
    if ed.get("tail") in ("inside", "long"):
        out.append("tail")
    if ed.get("head"):
        out.append("head")
    if row["category"] != "interjections" and (a.get("first_stretch") or 0.0) > g.stretch_max:
        out.append("stretch")
    if a.get("sib_db") is not None and a["sib_db"] > g.sib_max:
        out.append("sibilance")
    if (a.get("rate_ws") or 0.0) > SPEED_MAX.get(row["category"], math.inf) * g.speed_scale:
        out.append("speed")
    return out + pause_fails(a, row, g)


def pause_measures(words: list, spoken: str, pauses: list | None = None, breaks: list[int] | None = None) -> dict:
    """The silences of a clip between its spoken words (Whisper's word times matched by score_intonation.whisper_times):
    each gap's Whisper gap, or the longest of the clip's quiet runs (onset.edges' pauses) whose middle lies nearest that
    gap (between the two words' centres) when longer. pause_sent: the gaps at the sentence boundaries, in order: after
    the words `breaks` names (a long item's line joins, long_row), else after each clause-final mark
    (score_intonation.terminal); pause_mid: the longest other gap; pause_max: the longest gap or quiet run. {} when the
    words do not match (or there is one word)."""
    sw = spoken.split()
    times = si.whisper_times(sw, [w[:3] for w in words or []]) if words and len(sw) >= 2 else None
    if not times:
        return {}
    gaps = [max(0.0, times[k + 1][0] - times[k][1]) for k in range(len(sw) - 1)]
    mids = [(t0 + t1) / 2 for t0, t1 in times]
    runs = [(float(p0), float(p1)) for p0, p1 in pauses or []]
    for p0, p1 in runs:
        k = min(max(bisect.bisect_left(mids, (p0 + p1) / 2) - 1, 0), len(gaps) - 1)
        gaps[k] = max(gaps[k], p1 - p0)
    sent = (sorted({k for k in breaks if 0 <= k < len(gaps)}) if breaks is not None
            else [k for k in range(len(gaps)) if si.terminal(sw[k])])
    marks = set(sent)
    inside = [x for k, x in enumerate(gaps) if k not in marks]
    return {"pause_sent": [round(gaps[k], 3) for k in sent], "pause_mid": round(max(inside, default=0.0), 3),
            "pause_max": round(max([*gaps, *(p1 - p0 for p0, p1 in runs)], default=0.0), 3)}


def pause_fails(a: dict, row: dict, g: Gates) -> list[str]:
    """The pause gate of long clips (pause_measures): each internal sentence boundary pause_lo..pause_hi s, nothing
    inside a sentence over pause_mid, nothing anywhere over pause_max; a clip without the measures fails (its words
    did not match Whisper's: nothing to check by)."""
    if row["category"] != "long":
        return []
    if a.get("pause_sent") is None:
        return ["pause"]
    ok = (all(g.pause_lo <= x <= g.pause_hi for x in a["pause_sent"]) and a["pause_mid"] <= g.pause_mid
          and a["pause_max"] <= g.pause_max)
    return [] if ok else ["pause"]


def measure_clip(a: dict, row: dict, g: Gates) -> None:
    """Clip measures of a try Whisper heard (its words): edges (onset.edges, compact), sib_db / sib_ms
    (onset.sibilance), first_stretch / rate_ws / speech_s (word_measures), long clips pause_sent / pause_mid /
    pause_max (pause_measures); a stray tail after the last word is cut in the WAV itself (onset.trim_tail; tail_cut_s,
    duration and the .json beside it updated) when no gate fails. A measure that errs never costs the try
    (clip_error), except a long clip's: the pause gate has nothing to check by (score stops a round where most err)."""
    import soundfile as sf

    try:
        x, sr = sf.read(a["path"], dtype="float32", always_2d=False)
        ed = onset.edges(x, sr, a.get("words"), a["model_text"])
        a["edges"] = {k: ed[k] for k in ("speech_end_s", "head_s", "head", "tail", "tail_db", "tail_ms", "cut_s",
                                         "vlead_ms") if ed.get(k) not in (None, "")}
        a.update(onset.sibilance(x, sr))
        a.update({k: v for k, v in word_measures(a.get("words"), a["model_text"], ed).items()
                  if k in ("first_stretch", "rate_ws", "speech_s")})
        if row["category"] == "long":
            a.update(pause_measures(a.get("words"), a["model_text"], ed.get("pauses"), row.get("breaks")))
        if ed.get("cut_s") and not clip_fails(a, row, g):
            y = onset.trim_tail(x if x.ndim == 1 else x.mean(axis=1), sr, ed["cut_s"])
            dur = vc.write_wav16(a["path"], [y], sr)
            a["tail_cut_s"], a["duration"] = round(a["duration"] - dur, 3), round(dur, 3)
            side = Path(a["path"]).with_suffix(".json")
            vc.write_json(side, {**(vc.read_json(side) or {}), "duration": a["duration"],
                                 "tail_cut_s": a["tail_cut_s"]})
    except Exception as e:  # noqa: BLE001  (an unreadable clip, no voiced frame, ...)
        a["clip_error"] = f"{type(e).__name__}: {e}"[:200]


def utmos_cat(row: dict) -> str:
    """A row's UTMOS calibration category: its category, "<category>:plain" when rendered plain (another population:
    calibrated on its own first tries)."""
    return f"{row['category']}:plain" if is_plain(row) else row["category"]


def utmos_key(row: dict) -> str:
    return f"{utmos_cat(row)}/{row.get('emotion')}" if row["category"] in EMO_CATS else utmos_cat(row)


def utmos_thresholds(first: dict[str, list[float]], g: Gates, sizes: dict[str, int] | None = None) -> dict:
    """{key: UTMOS threshold}: the --utmos-drop quantile of the first tries of each category (from --calib-n of them,
    or all its pool when smaller, at least 5) and of each emotion (from --calib-n)."""
    out = {}
    for k, xs in first.items():
        need = g.calib_n if "/" in k else max(5, min(g.calib_n, (sizes or {}).get(k, g.calib_n)))
        if len(xs) >= need:
            out[k] = round(quantile(xs, g.utmos_drop), 4)
    return out


def utmos_threshold(row: dict, th: dict) -> float | None:
    return th.get(utmos_key(row), th.get(utmos_cat(row)))


def intonation_fails(att: dict, row: dict) -> list[str]:
    """Question kinds: the expected pattern's item passes; a matched declarative must not pass as a final rise."""
    items = att.get("intonation") or []
    if row.get("pattern") == "fall":
        return ["intonation"] if items and si.judge("final_rise", items[0]) else []
    return [] if items and items[0].get("pass") is True else ["intonation"]


def word_measures(words: list, spoken: str, edge: dict | None = None) -> dict:
    """From Whisper's word times matched to the spoken form (score_intonation.whisper_times) and the clip's edges
    (onset.edges: first sound, pauses): first_stretch = the first word's seconds per syllable (from where the sound
    starts) over the median of the other words' (>= 2 of them); rate_ws = syllables per second of speech from the first
    word's start to the last word's end, pauses of >= onset.PAUSE_S left out. {} when the words do not match."""
    sw = spoken.split()
    times = si.whisper_times(sw, [w[:3] for w in words or []]) if words else None
    if not times:
        return {}
    edge = edge or {}
    syl = [si.syllables(w) for w in sw]
    idx = [i for i, s in enumerate(syl) if s]
    if not idx:
        return {}
    start = max(times[idx[0]][0], edge.get("on_s") or 0.0)
    per = [(times[i][1] - (start if i == idx[0] else times[i][0])) / syl[i] for i in idx]
    out = {"first_word": si.core(sw[idx[0]]), "first_s_per_syl": round(per[0], 3)}
    if len(per) >= 3:
        out["first_stretch"] = round(per[0] / max(statistics.median(per[1:]), 1e-3), 3)
    t1 = times[idx[-1]][1]
    paused = sum(max(0.0, min(b, t1) - max(a, start)) for a, b in edge.get("pauses") or [])
    speech = t1 - start - paused
    if speech > 0.1:
        out.update(speech_s=round(speech, 3), rate_ws=round(sum(syl) / speech, 3))
    return out


EMO_PROSODY = (*vc.PROSODY, "level_db")  # level_db: the clip's active level before refprep levelled it (synthesis)


def baseline(pros: list[dict], min_n: int | None = None) -> dict | None:
    """Median prosody of the kept neutral clips (F0 medians folded to their own octave first, as
    voice_contest.emotion_summary), None under min_n (BASELINE_MIN); level_db only from min_n clips that have it
    (clips kept before it was recorded have none)."""
    need = BASELINE_MIN if min_n is None else min_n
    if len(pros) < need:
        return None
    raw = vc.median(p.get("f0_median_st") for p in pros)
    folded = [{**p, "f0_median_st": vc.fold_to(p.get("f0_median_st"), raw)} for p in pros]
    out = {"n": len(pros), "f0_median_raw": raw, **{f: vc.median(p.get(f) for p in folded) for f in EMO_PROSODY}}
    if sum(p.get("level_db") is not None for p in pros) < need:
        out["level_db"] = None
    return out


def prosody_delta(pros: dict, base: dict) -> dict:
    """Prosody minus a baseline (baseline()), field by field (EMO_PROSODY; F0 median folded to its octave)."""
    d = {}
    for f in EMO_PROSODY:
        x = pros.get(f)
        if f == "f0_median_st":
            x = vc.fold_to(x, base.get("f0_median_raw"))
        if x is not None and base.get(f) is not None:
            d[f] = round(x - base[f], 3)
    return d


def emotion_verdict(emotion: str | None, pros: dict | None, base: dict | None, g: Gates) -> tuple[str, dict]:
    """(verdict, deltas to the neutral baseline): expected | weak (neither way) | opposite | recorded (an emotion
    without a rule) | unchecked (no prosody or no baseline). Energy: level_db (before levelling) when both have it,
    else energy_db (levelled clips: little left of it). angry / cheerful / surprised: expected when louder or a wider
    F0 range. sad / calm: expected only when slower AND lower in F0 (and not louder than EMO_LEVEL_SLACK)."""
    if not pros or not base:
        return "unchecked", {}
    d = prosody_delta(pros, base)
    energy = d.get("level_db", d.get("energy_db", 0.0))
    if emotion in EMO_UP:
        if energy > 0 or d.get("f0_range_st", 0.0) > 0:
            return "expected", d
        opposite = d.get("f0_range_st") is not None and d["f0_range_st"] <= -g.emo_clear_st
        return ("opposite" if opposite else "weak"), d
    if emotion in EMO_DOWN:
        if (d.get("rate_syl_s", 0.0) < 0 and d.get("f0_median_st", 0.0) < 0
                and d.get("level_db", 0.0) <= EMO_LEVEL_SLACK):
            return "expected", d
        opposite = d.get("rate_syl_s") is not None and d["rate_syl_s"] >= g.emo_clear_rate
        return ("opposite" if opposite else "weak"), d
    return "recorded", d


def plain_score(emotion: str, d: dict) -> float:
    """emo_ref_score for a plain-rendered try; empathetic without its rate term, lower and quieter (its reference clip:
    -3.9 st, -1.6 dB): call 1's cwords tries were ~1 syllable / s slower than the neutral clips whatever their emotion
    (short sentences), so "slower, quieter" passed 37 of 42 surprised tries, "lower, quieter" 2 (empathetic 5 of 12)."""
    if emotion == "empathetic":
        return -(d.get("f0_median_st") or 0.0) / 2 - d.get("level_db", d.get("energy_db", 0.0)) / 2
    return emo_ref_score(emotion, d)


def plain_verdict(emotion: str | None, pros: dict | None, base: dict | None, g: Gates) -> tuple[str, dict]:
    """The emotion check of a plain-rendered try (no instruction asked for the emotion, only its reference clip carries
    it, so the clip has to show it; emotion_verdict lets a weak angry / cheerful / surprised clip and every empathetic
    one through, an instructed clip had the instruction): "expected" only when emotion_verdict says "expected" AND its
    deltas taken together go the emotion's way (plain_score > 0: emo_ref_score, the measure its reference clip was
    picked by; an "expected" surprised clip that is louder but lower and narrower is not); empathetic (no
    emotion_verdict rule: "recorded") by plain_score alone (lower, quieter). Else "weak" or "opposite"
    (emotion_verdict's), "unchecked" without prosody or a baseline, "recorded" for an emotion without any rule: all
    fail (PLAIN_FAIL)."""
    v, d = emotion_verdict(emotion, pros, base, g)
    if v == "expected" or (v == "recorded" and emotion == "empathetic"):
        v = "expected" if plain_score(emotion or "", d) > 0 else "weak"
    return v, d


def full_fails(att: dict, row: dict, g: Gates, th: dict, base: dict | None) -> list[str]:
    """The gates after Whisper: final onset, speaker similarity, UTMOS, intonation, emotion (records the floor, the
    threshold and the emotion verdict on att; a plain-rendered try: plain_verdict, PLAIN_FAIL unless "expected")."""
    out = ["onset"] if att.get("onset_final_bad") else []
    att["sim_floor"] = sim_floor(g, att["duration"])
    if att.get("sim") is None or att["sim"] < att["sim_floor"]:
        out.append("sim")
    att["utmos_threshold"] = thr = utmos_threshold(row, th)
    if att.get("utmos") is None or (thr is not None and att["utmos"] < thr):
        out.append("utmos")
    if row.get("pattern"):
        out += intonation_fails(att, row)
    if row["category"] in EMO_CATS:
        pros = {**att["prosody"], "level_db": att.get("level_in")} if att.get("prosody") else None
        verdict = plain_verdict if is_plain(row) else emotion_verdict
        att["emotion_verdict"], att["emotion_delta"] = verdict(row.get("emotion"), pros, base, g)
        if is_plain(row):
            out += [] if att["emotion_verdict"] == "expected" else [PLAIN_FAIL]
        elif att["emotion_verdict"] == "opposite" or (row.get("emotion") in EMO_DOWN
                                                      and att["emotion_verdict"] == "weak"):
            out.append("emotion")
    return out


# ---- state --------------------------------------------------------------------------------------------------------


def new_state(key: dict) -> dict:
    return {"version": STATE_VERSION, "key": key, "rows": {}, "kept": {}, "first_utmos": {}, "baseline": [],
            "emotion_refs": {}, "laughter": {}, "prelude": [],
            "stats": {}, "render": {"waves": 0, "jobs": 0, "done": 0, "audio_s": 0.0, "wall_s": 0.0, "server_s": 0.0},
            "scoring": {"clips": 0, "audio_s": 0.0, "wall_s": 0.0}, "calls": [], "pool": {}}


def key_conflicts(old: dict | None, key: dict) -> dict:
    """{field: "was -> now"} where a state was made otherwise ({} when it was not, or there is none)."""
    if not old:
        return {}
    return {k: f"{old.get(k)} -> {key.get(k)}" for k in KEY_FIELDS if old.get(k) != key.get(k)}


def new_counts() -> dict:
    """A category's (or a render mode's) try counters."""
    return {"decided": 0, "passed": 0, "audio_s": 0.0, "first": 0, "first_pass": 0, "kept": 0, "kept_s": 0.0,
            "dropped": 0, "fails": {}}


def cat_stats(state: dict, cat: str) -> dict:
    return state["stats"].setdefault(cat, new_counts())


# ---- render modes (--plain-render; module doc "Plain render") -------------------------------------------------------


def mode_of(state: dict, cat: str) -> str:
    """A category's render mode in this corpus: "plain" or "instructed" (state "modes"; none: instructed)."""
    return ((state.get("modes") or {}).get(cat) or {}).get("mode") or "instructed"


def mode_stats(state: dict, cat: str, create: bool = False) -> dict | None:
    """The counters of a category's current render mode (its stats' "modes"), None while it has no mode record (its
    totals are one mode's then); create: an empty one when missing."""
    if cat not in (state.get("modes") or {}):
        return None
    modes = cat_stats(state, cat).setdefault("modes", {}) if create else (state["stats"].get(cat) or {}).get("modes")
    m = mode_of(state, cat)
    return modes.setdefault(m, new_counts()) if create else (modes or {}).get(m) or {}


def with_render(pool: list[dict], state: dict) -> list[dict]:
    """The pool as a call renders it: the rows of a category in plain mode as copies marked "render": "plain"
    (render_text, the emotion check, UTMOS calibration, the records); texts.jsonl never has the mark."""
    plain = {c for c in PLAIN_CATS if mode_of(state, c) == "plain"}
    return [{**r, "render": "plain"} if r["category"] in plain else r for r in pool]


def mode_changes(state: dict, args) -> dict[str, str]:
    """{category: mode} this call switches to: of the categories with an instruction and a share in --mix,
    --plain-render's plain, the others instructed, where state has another mode ({} without --plain-render:
    mode_problem refuses a corpus with plain categories then). A category without a share keeps its mode (it renders
    nothing), so a call that leaves it out of --mix switches nothing."""
    want = parse_plain(getattr(args, "plain_render", None))
    if want is None:
        return {}
    share = parse_mix(args.mix)
    new = {c: "plain" if c in want else "instructed" for c in PLAIN_CATS if share[c] > 0}
    return {c: m for c, m in new.items() if m != mode_of(state, c)}


def mode_problem(state: dict, args) -> str:
    """Why this call may not render in a corpus ("" when it may): its state has plain categories with a share in --mix
    and --plain-render is not given, so the call would go back to the instruction silently (and mix the modes)."""
    if getattr(args, "plain_render", None) is not None:
        return ""
    share = parse_mix(args.mix)
    plain = [c for c in PLAIN_CATS if mode_of(state, c) == "plain" and share[c] > 0]
    if not plain:
        return ""
    since = "; ".join(f"{c} since call {state['modes'][c].get('call')} ({state['modes'][c].get('since')})"
                      for c in plain)
    return (f"{', '.join(plain)} render plain in this corpus (state.json \"modes\": {since}): give --plain-render "
            f"{','.join(plain)} again (the same command), or --plain-render none to render with the instruction again "
            "(its texts not kept get the tries left of their budget in that mode)")


def switch_mode(state: dict, pool: list[dict], cat: str, mode: str, budget: Callable[[dict], int],
                when: str = "") -> dict:
    """Start render mode `mode` for cat: its counters so far stay under the mode they were made in (its totals so far
    when it had no mode record; a mode it had before goes on with its own); each of its texts not kept (dropped: tries
    used up, open: its undecided tries on disk are the old mode's, reconcile removes them, or new) gets its budget
    (budget(row): max_tries) in `mode` from its next try: try0 = tries - the tries it had in `mode` before (row
    "mode_tries": {mode: tries}; none: a fresh budget, new seeds, none reused), status new, or dropped when that budget
    is used up already ("spent"); kept texts stay. state's "rows", "stats" and "modes" are replaced, not changed in
    place, so a shallow copy of state plans it (estimate). Returns what it did."""
    old = mode_of(state, cat)
    s = dict(state["stats"].get(cat) or new_counts())
    modes = {m: {**x, "fails": dict(x.get("fails") or {})} for m, x in (s.get("modes") or {}).items()}
    if old not in modes and s.get("decided"):
        modes[old] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in s.items() if k != "modes"}
    modes.setdefault(mode, new_counts())
    state["stats"] = {**state["stats"], cat: {**s, "modes": modes}}
    rows, n = dict(state["rows"]), collections.Counter()
    for r in pool:
        if r["category"] != cat:
            continue
        st = rows.get(r["id"])
        if not st or st.get("status") == "kept":
            n["kept" if st else "new"] += 1
            continue
        tries = int(st.get("tries") or 0)
        had = {**(st.get("mode_tries") or {}), old: tries - int(st.get("try0") or 0)}  # tries each mode has had
        spent = had.get(mode, 0) >= budget(r)
        rows[r["id"]] = {**st, "status": "dropped" if spent else "new", "try0": tries - had.get(mode, 0),
                         "mode_tries": had}
        n["spent" if spent else {"dropped": "reopened", "open": "restarted"}.get(st.get("status"), "new")] += 1
        n["back"] += bool(had.get(mode)) and not spent
    state["rows"] = rows
    prev = (state.get("modes") or {}).get(cat) or {}
    sw = {"category": cat, "from": old, "to": mode, "time": when, **{k: v for k, v in n.items() if v}}
    state["modes"] = {**(state.get("modes") or {}), cat: {
        "mode": mode, "since": when, "call": len(state.get("calls") or []) + 1,
        "switches": [*prev.get("switches", []), {k: v for k, v in sw.items() if k != "category"}]}}
    return sw


def apply_modes(state: dict, pool: list[dict], args, when: str = "") -> tuple[dict, list[dict]]:
    """(state with this call's mode_changes made: a shallow copy, the switches' records); (state, []) without any."""
    changes = mode_changes(state, args)
    if not changes:
        return state, []
    state = {**state}
    budget = (lambda r: max_tries(r, args.tries, args.long_tries, args.cwords_tries))
    return state, [switch_mode(state, pool, c, m, budget, when) for c, m in changes.items()]


def switch_line(sw: dict, budget: int) -> str:
    """A switch (switch_mode's record) as a printed line."""
    left = sw.get("reopened", 0) + sw.get("restarted", 0) + sw.get("new", 0)
    return (f"render mode {sw['category']}: {sw['from']} -> {sw['to']} ({left} texts not kept get a fresh budget of "
            f"{budget} tries from their next try, {sw.get('reopened', 0)} of them used up in {sw['from']}"
            + (f", {sw['back']} of them less the tries they had in {sw['to']} before" if sw.get("back") else "")
            + (f"; {sw['spent']} used up their tries in {sw['to']} before: dropped" if sw.get("spent") else "")
            + f"; {sw.get('kept', 0)} kept stay; its pass rate and the --min-pass stop are measured on its {sw['to']} "
            "tries alone"
            + ("; the model text without the instruction, its emotion's reference clip, the manifest keeps the "
               "instruction" if sw["to"] == "plain" else "") + ")")


def plain_wait(state: dict, row: dict) -> str:
    """Why a plain-rendered row may not render yet ("" when it may): no reference clip for its emotion (nothing would
    carry the emotion) or fewer than BASELINE_MIN kept neutral clips (plain_verdict has nothing to compare with)."""
    if not ((state.get("emotion_refs") or {}).get(row.get("emotion") or "") or {}).get("wav"):
        return f"no reference clip for {row.get('emotion')}"
    if len(state.get("baseline") or []) < BASELINE_MIN:
        return f"fewer than {BASELINE_MIN} kept neutral clips (the emotion check's baseline)"
    return ""


def kept_s(state: dict) -> dict[str, float]:
    out = dict.fromkeys(CATEGORIES, 0.0)
    for c in state["kept"].values():
        out[c["category"]] = out.get(c["category"], 0.0) + c["duration"]
    return out


def sim_pass(min_sim: float) -> float:
    """SIM_PASS at min_sim (linear between its points, its ends beyond them)."""
    pts = sorted(SIM_PASS.items())
    if min_sim <= pts[0][0] or min_sim >= pts[-1][0]:
        return pts[0][1] if min_sim <= pts[0][0] else pts[-1][1]
    (a, ya), (b, yb) = next((p, q) for p, q in zip(pts, pts[1:]) if p[0] <= min_sim <= q[0])
    return ya + (min_sim - a) * (yb - ya) / (b - a)


def measured_pass(state: dict, cat: str) -> float | None:
    """A category's own per-try pass rate (in its current render mode once it has a mode record: mode_stats; tries
    made in another mode do not count): under today's gates from MIN_MEASURED decided tries; else measured before the
    pilot's gates x NEW_GATE_PASS; None before."""
    s = mode_stats(state, cat)
    if s is None:
        s = state["stats"].get(cat) or {}
    if s.get("decided_g2", 0) >= MIN_MEASURED:
        return s["passed_g2"] / s["decided_g2"]
    if s.get("decided", 0) >= MIN_MEASURED:
        return s["passed"] / s["decided"] * NEW_GATE_PASS[cat]
    return None


def p_try(state: dict, cat: str, min_sim: float = MIN_SIM) -> float:
    """Per-try pass rate of a category: measured_pass (at least 0.02); else MEASURED_PASS (/data/f022corpus at
    MEASURED_SIM; PLAIN_PASS in plain mode; scaled by the speaker gate's SIM_PASS at min_sim)."""
    if (m := measured_pass(state, cat)) is not None:
        return max(0.02, m)
    prior = PLAIN_PASS if mode_of(state, cat) == "plain" else MEASURED_PASS[cat]
    return min(0.95, prior * sim_pass(min_sim) / sim_pass(MEASURED_SIM))


def stopped_cats(state: dict, min_pass: float) -> dict[str, float]:
    """{category: its measured_pass} of the categories under --min-pass: they take no new texts (GUESSED)."""
    return {c: round(m, 3) for c in CATEGORIES if (m := measured_pass(state, c)) is not None and m < min_pass}


def probing(state: dict, cat: str) -> bool:
    """A category whose pass rate is a guess (GUESSED, or in plain mode) and not measured yet: PROBE_TEXTS texts in
    flight at most."""
    return (cat in GUESSED or mode_of(state, cat) == "plain") and measured_pass(state, cat) is None


def class_pass(pool: list[dict], state: dict, numbers: bool) -> float | None:
    """The per-try pass rate of the neutral texts with (numbers=True) or without written numbers (has_numbers):
    kept texts / decided tries of that class, from MIN_MEASURED tries; None before."""
    kept = tries = 0
    for r in pool:
        st = state["rows"].get(r["id"])
        if r["category"] == "neutral" and st and has_numbers(r) == numbers:
            kept += st.get("status") == "kept"
            tries += int(st.get("tries") or 0)
    return max(0.02, kept / tries) if tries >= MIN_MEASURED else None


def number_order(rows: list[dict], base: tuple[float, float], share: float, p: dict[bool, float], tries: int,
                 cps: float) -> list[dict]:
    """New neutral texts in the order a wave takes them: one with written numbers whenever the expected kept seconds
    with numbers (base: (with numbers, all) kept or expected before these, + each taken text's expected length x
    chance to be kept in `tries` at its class's pass rate p[True / False]) are below `share` of all; else a plain one.
    Each class in its own (pool) order; when one runs out, the other."""
    nums = collections.deque(r for r in rows if has_numbers(r))
    plain = collections.deque(r for r in rows if not has_numbers(r))
    e_num, e_all, out = base[0], base[1], []
    while nums or plain:
        r = nums.popleft() if nums and (not plain or e_num <= share * e_all) else plain.popleft()
        y = est_dur(r, cps) * (1 - (1 - p[has_numbers(r)]) ** max_tries(r, tries))
        e_num, e_all = e_num + (y if has_numbers(r) else 0.0), e_all + y
        out.append(r)
    return out


def chars_s(state: dict) -> float:
    """The voice's measured characters of spoken form per second (kept clips, long items left out: their pauses),
    else CHARS_S."""
    kept = [c for c in state["kept"].values() if c.get("category") != "long"]
    if len(kept) < MIN_MEASURED:
        return CHARS_S
    return sum(len(c["render_spoken"]) for c in kept) / max(sum(c["duration"] - DUR_PAD_S for c in kept), 1.0)


def category_targets(hours: float, mix: dict[str, float], kept: dict[str, float], exhausted: set[str],
                     fill: str) -> dict[str, float]:
    """Seconds to keep per category: --hours x --mix; what a used-up category lacks goes to fill (unless it is used up
    too, or fill is "none")."""
    t = {c: hours * 3600 * mix.get(c, 0.0) for c in CATEGORIES}
    if fill in t and fill not in exhausted:
        for c in CATEGORIES:
            if c != fill and c in exhausted and kept.get(c, 0.0) < t[c]:
                t[fill] += t[c] - kept.get(c, 0.0)
                t[c] = kept.get(c, 0.0)
    return t


def row_status(state: dict, rid: str) -> str:
    return (state["rows"].get(rid) or {}).get("status", "new")


def exhausted_cats(pool: list[dict], state: dict) -> set[str]:
    """Categories without a text still to try (every pool row kept or dropped)."""
    live = {r["category"] for r in pool if row_status(state, r["id"]) in ("new", "open")}
    return set(CATEGORIES) - live


# ---- planning -----------------------------------------------------------------------------------------------------


def score_est(dur: float) -> float:
    """The scorers' seconds for one try, before the measured scoring factor."""
    return SCORE_CLIP_S + SCORE_AUDIO_S * dur


def c_est(dur: float) -> float:
    """The c-check's seconds for one try (phon_check on the GPU), before the measured scoring factor."""
    return PHON_CLIP_S + PHON_AUDIO_S * dur


@dataclass
class Cost:
    """The time model of a call: synthesis per audio s, scoring per clip, fixed parts."""
    render_per_audio: float
    score_factor: float = 1.0
    server_s: float = NANO_SERVER_S
    load_s: float = vc.SCORE_LOAD_S

    def score_s(self, dur: float) -> float:
        return score_est(dur) * self.score_factor

    def c_s(self, dur: float) -> float:
        """The c-check of one try, beside score_s."""
        return c_est(dur) * self.score_factor

    def render_s(self, audio_s: float, factor: float = 1.0) -> float:
        return audio_s * self.render_per_audio * (1 + ONSET_REDO) * factor


def make_cost(backend: str, ts: int, workers: int, measured: list[dict]) -> Cost:
    """Speeds from throughput.jsonl (Nano-vLLM servers with a reference; the official API's pool; the corpus-score
    lines: measured / estimated scoring time, the "raw" ones when there are any: older lines logged an estimate that
    already had the factor of their call in it), else voice_contest's probe numbers."""
    per = (vc.nano_wall_per_audio(ts, True, measured) if backend == "nanovllm"
           else vc.wall_per_audio(ts, workers, measured) if backend == "voxcpm" else 0.001)
    sc = [m for m in measured if m.get("mode") == "corpus-score" and (m.get("model_s") or 0) > 0
          and (m.get("clips") or 0) >= SCORE_LOG_MIN]
    sc = ([m for m in sc if m.get("raw")] or sc)[-5:]
    factor = sum(m["wall_s"] for m in sc) / sum(m["model_s"] for m in sc) if sc else 1.0
    server = NANO_SERVER_S if backend == "nanovllm" else vc.WORKERS_UP_S + 30.0 if backend == "voxcpm" else 0.0
    return Cost(per, factor, server)


def wave_rows(pool: list[dict], state: dict, targets: dict[str, float], tries: int, cps: float,
              on_disk: dict[str, list[int]] | None = None, min_sim: float = MIN_SIM, long_tries: int = 0,
              number_share: float = 0.0, min_pass: float = 0.0, cwords_tries: int = 0) -> list[dict]:
    """The texts of the next wave, in order: per category with a deficit (target - kept), its open rows, then new ones
    (neutral: number_order at number_share; a probing category up to PROBE_TEXTS in flight; one under min_pass none)
    while the expected kept seconds (expected length x chance to be kept in the tries left) fall short of it;
    interleaved by the share of the deficit each covers, so a wave cut short keeps the mix. Open rows whose category
    has no deficit wait, and so do the rows of a plain category that may not render yet (plain_wait)."""
    kept, stopped = kept_s(state), stopped_cats(state, min_pass)
    out = []
    for cat in CATEGORIES:
        deficit = targets.get(cat, 0.0) - kept.get(cat, 0.0)
        if deficit <= 0:
            continue
        p, have = p_try(state, cat, min_sim), 0.0
        rows = [r for r in pool if r["category"] == cat]
        if mode_of(state, cat) == "plain":
            rows = [r for r in rows if not plain_wait(state, r)]
        open_ = [r for r in rows if row_status(state, r["id"]) == "open"]
        open_.sort(key=lambda r: -len((on_disk or {}).get(r["id"], [])))  # tries already made first
        new = [r for r in rows if row_status(state, r["id"]) == "new"]
        if cat in stopped:
            new = []
        elif probing(state, cat):
            new = new[:max(0, PROBE_TEXTS - len(open_))]
        if cat == "neutral" and number_share > 0:
            pcls = {k: class_pass(pool, state, k) or p for k in (True, False)}
            base = [0.0, 0.0]
            for c in state["kept"].values():
                if c.get("category") == "neutral":
                    base[0] += c["duration"] if has_numbers(c) else 0.0
                    base[1] += c["duration"]
            for r in open_:
                st = state["rows"].get(r["id"]) or {}
                left = try_end(r, st, tries) - st.get("tries", 0)
                y = est_dur(r, cps) * (1 - (1 - pcls[has_numbers(r)]) ** max(left, 1))
                base[0] += y if has_numbers(r) else 0.0
                base[1] += y
            new = number_order(new, (base[0], base[1]), number_share, pcls, tries, cps)
        for r in open_ + new:
            if have >= deficit:
                break
            st = state["rows"].get(r["id"]) or {}
            left = try_end(r, st, tries, long_tries, cwords_tries) - st.get("tries", 0)
            have += est_dur(r, cps) * (1 - (1 - p) ** max(left, 1))
            out.append((have / deficit, r))
    out.sort(key=lambda x: x[0])
    return [r for _, r in out]


def prerender_of(row: dict, prerender: int) -> int:
    """Tries of a text rendered a wave: --prerender; cwords at least CWORDS_PRERENDER."""
    return max(prerender, CWORDS_PRERENDER) if row.get("category") == "cwords" else prerender


def wave_tries(row: dict, st: dict | None, tries: int, prerender: int, on_disk: list[int],
               long_tries: int = 0, cwords_tries: int = 0) -> tuple[list[int], int]:
    """(tries to render, tries available in all) of a text in a wave: from the next undecided try, prerender_of of them
    (those already on disk are not made again), never past its budget (try_end)."""
    first = (st or {}).get("tries", 0)
    last = min(first + max(prerender_of(row, prerender), len([t for t in on_disk if t >= first])),
               try_end(row, st, tries, long_tries, cwords_tries))
    want = list(range(first, last))
    return [t for t in want if t not in on_disk], len(want)


def c_checked(row: dict, args) -> bool:
    """The c-check runs on this row's tries (needs_c, --no-c-check not given)."""
    return bool(getattr(args, "c_check", True)) and needs_c(row)


def fit_wave(rows: list[dict], state: dict, args, cost: Cost, budget_s: float, on_disk: dict[str, list[int]],
             cps: float, phon_load_s: float = PHON_LOAD_S) -> tuple[list[dict], float]:
    """The longest prefix of rows whose wave fits in budget_s (a server when anything is rendered, the scorer loads,
    the c-check's model when a row needs it: phon_load_s, 0 once the process has it, the renders, the expected scored
    tries); (rows, its estimated seconds)."""
    p = {c: p_try(state, c, args.min_sim) for c in CATEGORIES}
    t, render, phon, out = cost.load_s, False, False, []
    for r in rows:
        todo, avail = wave_tries(r, state["rows"].get(r["id"]), args.tries, args.prerender, on_disk.get(r["id"], []),
                                 args.long_tries, args.cwords_tries)
        d, cc = est_dur(r, cps), c_checked(r, args)
        need = (cost.render_s(len(todo) * d, render_factor(r["category"]))
                + sum((1 - p[r["category"]]) ** j for j in range(avail)) * (cost.score_s(d) + cc * cost.c_s(d)))
        if todo and not render:
            need += cost.server_s
        if cc and not phon:
            need += phon_load_s
        if t + need > budget_s:
            break
        t += need
        render, phon = render or bool(todo), phon or cc
        out.append(r)
        if args.max_wave_rows and len(out) >= args.max_wave_rows:
            break
    return out, t


def estimate(pool: list[dict], state: dict, args, cost: Cost, prelude: dict | None = None) -> dict:
    """What the rest of the corpus costs: per category the texts still needed (deficit / (expected length x chance to
    be kept)), tries scored and rendered, then synthesis, scoring, the c-check (the tries of needs_c rows, as if all
    passed Whisper; its model's load once) and per-wave overhead in seconds (the waves a text's tries take at
    prerender_of a wave, + 1; a probing category: one wave more); plus the prelude (prelude_jobs' plan) when given. A
    category under --min-pass: its open texts only, its deficit left (not handed to --fill). The render modes as this
    call would switch them (apply_modes on a copy: "switches")."""
    state, switches = apply_modes(state, pool, args)
    mix, kept = parse_mix(args.mix), kept_s(state)
    targets = category_targets(args.hours, mix, kept, exhausted_cats(pool, state), args.fill)
    stopped = stopped_cats(state, args.min_pass)
    cps = chars_s(state)
    by, render_a, render_t, scored, score_t = {}, 0.0, 0.0, 0.0, 0.0
    c_n, c_t, span = 0.0, 0.0, math.ceil(max(args.tries, 1) / max(args.prerender, 1))
    order = [c for c in CATEGORIES if c != args.fill] + [c for c in CATEGORIES if c == args.fill]
    for cat in order:  # the fill category last: it takes what the others' pools cannot give
        live = ("open",) if cat in stopped else ("new", "open")
        rows = [r for r in pool if r["category"] == cat and row_status(state, r["id"]) in live]
        deficit = max(0.0, targets[cat] - kept[cat])
        if not deficit:
            continue
        d = sum(est_dur(r, cps) for r in rows) / len(rows) if rows else DUR_PAD_S + 60 / cps
        p = p_try(state, cat, args.min_sim)
        n_try = max_tries(rows[0] if rows else {"category": cat}, args.tries, args.long_tries, args.cwords_tries)
        pre = max(prerender_of({"category": cat}, args.prerender), 1)
        if cat in ("long", "cwords"):  # waves: their own tries may take more
            span = max(span, math.ceil(n_try / pre))
        keep = 1 - (1 - p) ** n_try
        need = deficit / (d * keep)
        if need > len(rows):  # its texts run out: the rest goes to --fill
            if args.fill in targets and cat != args.fill and cat not in stopped:
                targets[args.fill] += (need - len(rows)) * d * keep
                targets[cat] -= (need - len(rows)) * d * keep
            need = float(len(rows))
        tries = need * keep / p  # expected tries scored
        rend = tries + need * keep * (pre - 1) / 2  # + pre-rendered tries never scored
        cfrac = sum(c_checked(r, args) for r in rows) / len(rows) if rows else 0.0
        by[cat] = {"target_s": round(targets[cat]), "kept_s": round(kept[cat]), "texts": math.ceil(need),
                   "pool": len(rows), "short_of_pool": need >= len(rows), "clip_s": round(d, 2),
                   "p_try": round(p, 3), "p_keep": round(keep, 3), "tries": round(tries),
                   "rendered_s": round(rend * d), **({"stopped": stopped[cat]} if cat in stopped else {}),
                   **({"probe": PROBE_TEXTS} if probing(state, cat) else {}),
                   **({"c_checked": round(tries * cfrac)} if cfrac else {}),
                   **({"render": "plain"} if mode_of(state, cat) == "plain" else {})}
        render_a += rend * d
        render_t += cost.render_s(rend * d, render_factor(cat))
        scored += tries
        score_t += tries * cost.score_s(d)
        c_n += tries * cfrac
        c_t += tries * cfrac * cost.c_s(d)
    waves = span + 1 if by else 0
    waves += any("probe" in b for b in by.values())  # its first wave measures it
    over = waves * (cost.server_s + cost.load_s)
    c_t += PHON_LOAD_S if c_n else 0.0
    return {"by": by, "targets": {c: round(v) for c, v in targets.items()}, "rendered_audio_s": round(render_a),
            "scored": round(scored), "render_s": round(render_t), "score_s": round(score_t), "waves": waves,
            "c_checked": round(c_n), "c_s": round(c_t), "overhead_s": round(over),
            "work_s": round(render_t + score_t + c_t + over + (prelude or {}).get("s", 0)), "prelude": prelude,
            "left_s": round(sum(max(0.0, targets[c] - kept[c]) for c in CATEGORIES if c not in stopped)),
            "stopped_left_s": round(sum(max(0.0, targets[c] - kept[c]) for c in stopped)),
            "kept_s": round(sum(kept.values())), "render_per_audio": cost.render_per_audio,
            "score_factor": round(cost.score_factor, 3), "min_pass": args.min_pass,
            "switches": [{**sw, "budget": max_tries({"category": sw["category"]}, args.tries, args.long_tries,
                                                    args.cwords_tries)} for sw in switches]}


def estimate_lines(est: dict, usd_per_h: float = USD_PER_H, budget_s: float | None = None) -> list[str]:
    """The plan as printed lines: per category, the sums and their arithmetic, what budget_s seconds of work make."""
    out = [switch_line(sw, sw["budget"]) for sw in est.get("switches") or []]
    out.append(f"kept {est['kept_s'] / 3600:.2f} h, {est['left_s'] / 3600:.2f} h to go "
               f"(targets: {', '.join(f'{c} {v / 3600:.2f} h' for c, v in est['targets'].items() if v)})")
    for c, b in est["by"].items():
        note = (f"  (STOPPED: {b['stopped']} of its tries pass, under --min-pass {est.get('min_pass')}: its "
                f"{b['pool']} open texts only, {est.get('stopped_left_s', 0) / 3600:.2f} h left unmade; look at its "
                "gate fails, --min-pass 0 goes on)" if "stopped" in b
                else f"  (all {b['pool']} of its texts: the rest goes to --fill)" if b["short_of_pool"] else "")
        if "probe" in b:
            note += (f"  (p(try) a guess: its first wave takes {b['probe']} texts and measures it, the next waves plan "
                     "with the measurement)")
        if b.get("c_checked"):
            note += f"  (c-check on {b['c_checked']} tries)"
        if b.get("render") == "plain":
            note += ("  (rendered plain: no instruction in the model text, the emotion's reference clip; the emotion "
                     "check wants \"expected\"" + (f"; p(try) guessed {PLAIN_PASS:g} until measured plain"
                                                   if "probe" in b else "") + ")")
        out.append(f"  {c:<13} {b['texts']:5d} texts x {b['clip_s']:4.1f} s x p(keep) {b['p_keep']:.2f} "
                   f"(p(try) {b['p_try']:.2f}): {b['tries']:5d} tries scored, {b['rendered_s'] / 3600:5.2f} h rendered"
                   + note)
    r, pre = est["render_per_audio"], est.get("prelude")
    if pre is not None:
        out.append(prelude_line(pre, usd_per_h))
    slow = (f" (long clips x {LONG_RENDER:g}: {est['by']['long']['rendered_s']} s of it)" if "long" in est["by"]
            else "")
    cc = (f"c-check {est['c_checked']} tries x ({PHON_CLIP_S} + {PHON_AUDIO_S} x clip s) x {est['score_factor']} + "
          f"{PHON_LOAD_S:g} s load = {est['c_s']} s; " if est.get("c_checked") else "")
    out.append(f"synthesis {est['rendered_audio_s']} s audio x {r:.4f} s/s x {1 + ONSET_REDO:g} (onset retries){slow}"
               f" = {est['render_s']} s; scoring {est['scored']} tries x ({SCORE_CLIP_S} + {SCORE_AUDIO_S} x clip s) x "
               f"{est['score_factor']} = {est['score_s']} s; " + cc + f"{est['waves']} waves x (server + scorer "
               f"loads) = {est['overhead_s']} s; " + (f"prelude {pre['s']} s; " if pre and pre.get("s") else "")
               + f"total {est['work_s']} s = {est['work_s'] / 3600:.2f} h, "
               f"~${est['work_s'] / 3600 * usd_per_h:.2f} at ${usd_per_h:.2f}/h")
    if budget_s is not None and est["work_s"] > 0:
        share = min(1.0, max(budget_s, 0.0) / est["work_s"])
        out.append(f"this call: {budget_s:.0f} s of work -> ~{share * est['left_s'] / 3600:.2f} h kept"
                   + ("" if share >= 1 else f"; the rest (~${(est['work_s'] - budget_s) / 3600 * usd_per_h:.2f} of "
                      "waves, before each call's own start-up) in the next call(s): the same command continues"))
    return out


# ---- files --------------------------------------------------------------------------------------------------------


def attempt_wav(out: Path, rid: str, t: int) -> Path:
    return out / "attempts" / rid / f"t{t}.wav"


def on_disk_tries(out: Path, rid: str) -> list[int]:
    """Tries of a text made and recorded (WAV + its .json), not yet decided."""
    d = out / "attempts" / rid
    if not d.is_dir():
        return []
    return sorted(int(p.stem[1:]) for p in d.glob("t*.wav") if p.stem[1:].isdigit() and vc.has_wav(p)
                  and p.with_suffix(".json").is_file())


def to_16k(src: Path, dst: Path) -> float:
    """A clip as 16 kHz PCM16 mono at dst (atomic; copied when it is one already); its seconds."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(src), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    if (sr, ch, sw) == (pa.RATE, 1, 2):
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        return n / sr
    import soundfile as sf

    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    y = pa.resample(x.mean(axis=1), sr, pa.RATE)
    pa.write_wav(dst, y, pa.RATE)
    return len(y) / pa.RATE


def prep_reference(raw: Path, dst: Path) -> float:
    """refprep.prep_audio of the reference (what both engines get) as 16 kHz PCM16 at dst; its seconds."""
    import librosa
    from refprep import SR, prep_audio

    x, _ = librosa.load(str(raw), sr=SR, mono=True)
    y = prep_audio(x)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pa.write_wav(dst, y, SR)
    return len(y) / SR


def foreign_files(out: Path) -> list[str]:
    """Files this script would rewrite in --out without having written them ([] with its marker)."""
    if (out / MARKER).is_file():
        return []
    return [n for n in (*OWN_FILES, *pa.OWN_FILES) if (out / n).exists()] + ["wav/*.wav"] * any(
        (out / "wav").glob("*.wav"))


def assign_refs(kept: list[dict], fraction: float, seed: int, ref: tuple[str, float] | None) -> dict[str, dict]:
    """{clip id: {"ref_audio", "ref_duration"}} for round(fraction x clips): the prepared reference (ref: path,
    seconds) or another kept neutral clip of REF_MIN_S-REF_MAX_S s, never the clip itself. Deterministic."""
    rng = random.Random(seed)
    kept = sorted(kept, key=lambda c: c["id"])
    cands = ref_donors(kept, ref)
    out = {}
    if not cands:
        return out
    for c in sorted(rng.sample(kept, min(len(kept), round(fraction * len(kept)))), key=lambda c: c["id"]):
        pick = [x for x in cands if x[0] != c["id"]]
        if pick:
            _, path, dur = rng.choice(pick)
            out[c["id"]] = {"ref_audio": path, "ref_duration": round(float(dur), 3)}
    return out


def ref_donors(kept: list[dict], ref: tuple[str, float] | None) -> list[tuple[str, str, float]]:
    """(clip id, wav, seconds) a ref_audio may be: kept neutral / short clips of REF_MIN_S-REF_MAX_S s (by id) and
    the prepared reference (id "")."""
    cands = [(c["id"], c["wav"], c["duration"]) for c in sorted(kept, key=lambda c: c["id"])
             if c["category"] in BASELINE_CATS and REF_MIN_S <= c["duration"] <= REF_MAX_S]
    if ref and REF_MIN_S <= ref[1] <= REF_MAX_S + 0.5:  # the prepared reference: padded to whole patches
        cands.append(("", ref[0], ref[1]))
    return cands


def reviewed_refs(reviewed: list[dict], kept: list[dict], fraction: float, seed: int,
                  ref: tuple[str, float] | None) -> dict[str, dict]:
    """assign_refs for the human-approved clips: each one gets a donor (ref_donors of the kept clips) with
    probability fraction, both drawn from its own id, so adding a clip never changes another's."""
    cands, out = ref_donors(kept, ref), {}
    for c in reviewed if cands else []:
        h = int(rank(seed, f"ref/{c['id']}"), 16)
        if h % 10_000 < fraction * 10_000:
            _, path, dur = cands[(h // 10_000) % len(cands)]
            out[c["id"]] = {"ref_audio": path, "ref_duration": round(float(dur), 3)}
    return out


def manifest_rows(kept: list[dict], refs: dict[str, dict], wav_dir: Path, seed: int) -> list[dict]:
    """voxcpm_train.jsonl rows (absolute paths; text = the training text), shuffled."""
    rows = []
    for c in sorted(kept, key=lambda c: c["id"]):
        r = {"audio": str(wav_dir / c["wav"]), "text": c["train_text"], "duration": float(c["duration"])}
        if c["id"] in refs:
            ref = refs[c["id"]]
            r["ref_audio"] = ref["ref_audio"] if os.path.isabs(ref["ref_audio"]) else str(wav_dir / ref["ref_audio"])
            r["ref_duration"] = ref["ref_duration"]
        rows.append(r)
    random.Random(seed + 1).shuffle(rows)
    return rows


# ---- scoring ------------------------------------------------------------------------------------------------------


def _wav_of(a: dict):
    return Path(a["path"]) if vc.has_wav(Path(a["path"])) else None


class Scoring:
    """voice_contest's scorers on rounds of tries: Whisper (hyp, P(tr), word timestamps: the clip gates), the c-check's
    IPA model (phon_check) on the tries that need it, then UTMOS and ECAPA similarity to the raw reference with the CPU
    pass beside them (onset of the final clip, prosody, intonation). Loaded on first use, freed by close() before the
    next server (the IPA model: moved to the CPU, the next wave moves it back)."""

    def __init__(self, ref: str, device: str = "auto", procs: int = 6):
        self.ref, self.sc, self.procs, self.pm = ref, vc.Scorers(device), procs, None
        self.phon_load_s = 0.0  # the IPA model's load in this wave (0 when the process had it): Run.score logs it

    def text(self, atts: list[dict], until: float) -> None:
        vc.whisper_pass(self.sc, atts, _wav_of, "corpus", until=until, words=True)

    def phon(self, atts: list[dict], until: float) -> None:
        """The c-check's measures of each try: ipa (space-separated phones), c_expected / c_got (phon_check.c_counts of
        its model text); c_error when the model cannot read the clip; tries left without either at the deadline."""
        if self.pm is None:
            t0, fresh = time.time(), not pc.loaded()
            self.pm = pc.get_model(self.sc.dev())
            self.phon_load_s = time.time() - t0 if fresh else 0.0
        for a in atts:
            if time.time() >= until:
                break
            try:
                toks = self.pm.ipa_file(a["path"])
            except Exception as e:  # noqa: BLE001  (an unreadable clip fails its try; systemic(): a broken model)
                a["c_error"] = f"{type(e).__name__}: {e}"[:200]
                continue
            a["ipa"] = pc.ipa_text(toks)
            a["c_expected"], a["c_got"] = pc.c_counts(a["model_text"], toks)

    def rest(self, atts: list[dict], until: float, cpu: list[tuple[dict, bool, bool]]) -> None:
        items = [(i, {k: a[k] for k in vc.CPU_KEYS if k in a}, a["path"], inton, pros, True)
                 for i, (a, inton, pros) in enumerate(cpu)]
        pool = vc.CpuPass(items, self.procs) if items and time.time() + vc.CPU_MIN_S <= until else None
        vc.utmos_pass(self.sc, atts, _wav_of, "corpus", until=until)
        vc.sim_pass(self.sc, [("corpus", atts, Path(self.ref))], _wav_of, until=until, self_sim=False)
        for i, out in (pool.get(until) if pool else []):
            a = cpu[i][0]
            a["cpu_done"] = True
            if "onset" in out:
                a["onset_final_bad"] = out["onset"].get("onset_bad")
                a["onset_final_why"] = out["onset"].get("onset_why") or []
            for k in ("prosody", "intonation", "prosody_error", "intonation_error"):
                if k in out:
                    a[k] = out[k]

    def close(self) -> None:
        if self.pm is not None:
            self.pm.to("cpu")  # off the GPU before Scorers.close empties its cache; the process keeps it
        self.sc.close()


# ---- emotion references and the laughter trial (the prelude) -----------------------------------------------------


def emo_ref_score(emotion: str, d: dict) -> float:
    """How far a reference candidate's prosody deltas (to neutral renders of the same sentence) go the emotion's way,
    in rough units (2 st of F0 median, 3 st of range, 0.5 syllables / s, 2 dB of level before levelling): sad lower,
    slower, quieter; calm and empathetic slower, quieter (calm also narrower); angry wider and louder; cheerful and
    surprised wider, louder and higher. > 0: the right way."""
    f0, rng, rate, lvl = ((d.get(k) or 0.0) for k in ("f0_median_st", "f0_range_st", "rate_syl_s", "level_db"))
    if emotion == "sad":
        return -f0 / 2 - rate / 0.5 - lvl / 2
    if emotion == "calm":
        return -rate / 0.5 - lvl / 2 - rng / 3
    if emotion == "empathetic":
        return -rate / 0.5 - lvl / 2
    if emotion == "angry":
        return rng / 3 + lvl / 2
    return rng / 3 + lvl / 2 + f0 / 2  # cheerful, surprised


def pick_emo_ref(emotion: str, cands: list[dict], base: dict | None) -> tuple[dict | None, str]:
    """The candidate (prelude try: char_errors, sim, onset_final_bad, prosody, level_in) whose prosody best matches the
    emotion (emo_ref_score > 0) among those Whisper heard right, with ECAPA similarity >= EMO_REF_MIN_SIM and no stray
    onset; (None, why) when there is none. Each candidate gets its "delta" and "score"."""
    if not base:
        return None, "no neutral render of the sentence to compare with"
    ok = []
    for c in cands:
        if c.get("prosody"):
            c["delta"] = prosody_delta({**c["prosody"], "level_db": c.get("level_in")}, base)
            c["score"] = round(emo_ref_score(emotion, c["delta"]), 3)
        if c.get("char_errors") == 0 and (c.get("sim") or 0.0) >= EMO_REF_MIN_SIM and not c.get("onset_final_bad") \
                and "score" in c:
            ok.append(c)
    if not ok:
        return None, f"none of {len(cands)} heard right with sim >= {EMO_REF_MIN_SIM}"
    best = max(ok, key=lambda c: c["score"])
    if best["score"] <= 0:
        return None, f"none of {len(ok)} goes the {emotion} way (best score {best['score']})"
    return best, ""


def load_laughter(path: Path, guard: pa.Guard | None = None) -> tuple[list[dict], collections.Counter]:
    """The laughter trial's lines (build_textbank checks them): {id, text, spoken, plain: spoken without the
    laughter}; ([], ...) without the file. Lines the leakage guard hits are left out."""
    rows, bad = [], collections.Counter()
    if not path.is_file():
        return rows, collections.Counter({f"no laughter lines at {path}": 1})
    for r in vc.read_rows(path):
        lid, text = clean_id(r.get("id")), r.get("text")
        if not lid or not isinstance(text, str) or not r.get("laugh"):
            bad["not a laughter line"] += 1
            continue
        spoken = spoken_of(text)
        if guard is not None and guard.hit(pa.eval_key(spoken), pa.eval_key(text)):
            bad["leakage guard"] += 1
            continue
        rows.append({"id": lid, "text": collapse(text), "spoken": spoken, "plain": strip_laughter(spoken)})
    return rows, bad


def laugh_span(words: list, spoken: str, dur: float) -> tuple[float, float]:
    """Where the laughter should be: the stretch Whisper's words leave free (words that are laughter or in brackets do
    not count) before the first word when the text starts with it, after the last one when it ends with it, else
    the longest free stretch."""
    ws = [(float(w[1]), float(w[2])) for w in words or []
          if not LAUGH_RE.fullmatch(str(w[0]).strip(" ,.!?…")) and not re.match(r"^\W*[(\[]", str(w[0]))]
    if not ws:
        return 0.0, dur
    gaps = [(0.0, ws[0][0])] + [(a[1], b[0]) for a, b in zip(ws, ws[1:])] + [(ws[-1][1], dur)]
    m = LAUGH_RE.search(spoken)
    if m and not spoken[:m.start()].strip(" ,"):
        return gaps[0]
    if m and not spoken[m.end():].strip(" ,.!?…"):
        return gaps[-1]
    return max(gaps, key=lambda g: g[1] - g[0])


class AstLaugh:
    """AudioSet's laughter classes (Laughter, Baby laughter, Giggle, Snicker, Belly laugh, Chuckle) from AST_MODEL, only
    when transformers and the model are already here (local_files_only: nothing is downloaded); else available False
    and why."""

    def __init__(self, name: str = AST_MODEL, device: str = "cpu"):
        self.why, self.model = "", None
        try:
            from transformers import ASTFeatureExtractor, ASTForAudioClassification

            self.fe = ASTFeatureExtractor.from_pretrained(name, local_files_only=True)
            self.model = ASTForAudioClassification.from_pretrained(name, local_files_only=True).eval().to(device)
            self.device = device
            self.ids = [i for i, lab in self.model.config.id2label.items()
                        if re.search(r"augh|iggle|nicker|huckle", lab)]
        except Exception as e:  # noqa: BLE001  (no transformers, no model in the cache)
            self.why = f"{type(e).__name__}: {e}"[:200]

    @property
    def available(self) -> bool:
        return self.model is not None

    def __call__(self, path: str | Path) -> float:
        """The highest laughter-class probability (sigmoid) of a clip."""
        import librosa
        import torch

        y, _ = librosa.load(str(path), sr=16000, mono=True)
        with torch.inference_mode():
            x = self.fe(y, sampling_rate=16000, return_tensors="pt").to(self.device)
            p = torch.sigmoid(self.model(**x).logits[0])
        return round(float(max(p[i] for i in self.ids)), 4)


def laugh_rank(a: dict) -> tuple:
    """A laughter try's place among its line's tries: heard right without the laughter, the voice, laughter found
    (AST when there, else the burst detector), how strongly."""
    lg = a.get("laugh") or {}
    ast = a.get("ast_laugh")
    found = ast >= LAUGH_AST_MIN if ast is not None else bool(lg.get("laugh"))
    return (a.get("cer_nolaugh") == 0, (a.get("sim") or 0.0) >= EMO_REF_MIN_SIM, found,
            ast if ast is not None else lg.get("score", 0.0))


def laughter_html(rows: list[dict], wav_dir: str = f"{REVIEW}/wav") -> str:
    """voxcpm-corpus/f022/laughter.html: one row per line, its kept tries side by side, a "gerçekten gülüyor" box
    under each, a comment per line, a button that shows and copies the result (Turkish UI)."""
    import html

    by = collections.defaultdict(list)
    for r in rows:
        by[r["line"]].append(r)
    L = ["<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
         "<title>f022 gülme denemesi</title><style>body{font-family:sans-serif;max-width:960px;margin:auto;"
         "padding:16px;background:#111;color:#eee}.r{background:#1d1d1d;margin:8px 0;padding:10px;border-radius:8px}"
         ".c{display:flex;gap:10px;flex-wrap:wrap}.k{flex:1;min-width:200px}.s{color:#9c9;font-size:12px}"
         "audio{width:100%;margin:4px 0}.y{width:90%;background:#222;color:#eee;border:1px solid #444}"
         "button{font-size:16px;padding:8px 14px;border-radius:8px;border:0;background:#3a6;color:#fff}</style>"
         "<h2>f022 gülme denemesi</h2><p>Her satırda aynı cümlenin en iyi denemeleri yan yana. Gerçekten gülen her "
         "kaydın altındaki kutuyu işaretle, istersen yorum yaz. Bitince alttaki düğmeye bas, çıkan yazıyı bana "
         "yapıştır (panoya da kopyalanır). Bu kayıtlar eğitime girmez.</p>"]
    for lid in sorted(by):
        cands = sorted(by[lid], key=lambda r: r.get("rank", 0))
        L.append(f"<div class=r data-l='{html.escape(lid)}'><b>{html.escape(lid)}</b> "
                 f"{html.escape(cands[0].get('text', ''))}<div class=c>")
        for c in cands:
            note = html.escape(f"{c.get('instruction', '')} (tohum {c.get('seed')})")
            L.append(f"<div class=k><div class=s>{note}</div><audio controls preload=none "
                     f"src=\"{html.escape(wav_dir)}/{html.escape(Path(c['wav']).name)}\"></audio><label><input "
                     f"type=checkbox data-c='{html.escape(c['id'])}'> gerçekten gülüyor</label></div>")
        L.append("</div><input class=y placeholder='yorum (isteğe bağlı)'></div>")
    L.append("<button onclick='go()'>Sonucu göster ve kopyala</button><pre id=o></pre><script>function go(){"
             "const n=String.fromCharCode(10);let s='';document.querySelectorAll('.r').forEach(r=>{"
             "const l=r.dataset.l;const ok=[...r.querySelectorAll('input[type=checkbox]:checked')]"
             ".map(x=>x.dataset.c);const y=r.querySelector('.y').value.trim();"
             "s+=l+': '+(ok.length?ok.join(', ')+' gerçekten gülüyor':'hiçbiri gülmüyor')+(y?' - '+y:'')+n});"
             "document.getElementById('o').textContent=s;if(navigator.clipboard)navigator.clipboard.writeText(s)"
             ".catch(()=>{})}</script>")
    return "".join(L) + "\n"


LAUGH_PAD_S, AST_CLIP_S = 0.8, 0.1  # the laughter itself beyond its letters; AST per clip on the GPU (estimates)


def prelude_jobs(out: Path, state: dict, args, pool: list[dict], laugh_rows: list[dict], cost: Cost, ref: str
                 ) -> tuple[list[dict], dict]:
    """The prelude's tries (one server before the waves): EMO_REF_NEUTRAL plain and --emo-ref-n instructed renders of
    EMO_REF_TEXT for each emotion of the pool still to render that has no reference yet (--emotion-refs, emotions in
    the mix); the laughter trial once (--laughter-usd > 0): line x LAUGH_INSTRUCTIONS x LAUGH_SEEDS, seed-major (every
    line once before any twice), as many as --laughter-usd pays for (render + scoring, and the server and scorer
    loads when no emotion reference shares them). Returns (jobs, plan: what and its estimated seconds)."""
    have = state.get("emotion_refs") or {}
    emos = []
    cats = [c for c in EMO_CATS if parse_mix(args.mix).get(c, 0) > 0]  # emotion rows of the mix (emotions, cwords)
    if args.emotion_refs and cats:
        live = {r.get("emotion") for r in pool if r["category"] in cats
                and row_status(state, r["id"]) in ("new", "open")}
        emos = sorted((live & set(EMO_REF_INSTRUCTIONS)) - set(have))
    cps = chars_s(state)
    jobs, work = [], 0.0

    def add(name: str, kind: str, text: str, spoken: str, seed: int, **extra) -> float:
        d = est_dur({"spoken": spoken}, cps) + (LAUGH_PAD_S if kind == "laugh" else 0.0)
        jobs.append({"key": f"prelude/{name}", "group": "corpus", "text": text, "ref": ref, "lora": None,
                     "cfg": args.cfg, "ts": args.timesteps, "seed": seed, "out": str(out / "prelude" / f"{name}.wav"),
                     "est_s": round(d, 2), "backend": args.backend, "ref_prep": True, "kind": kind, "spoken": spoken,
                     "plain": extra.pop("plain", spoken), **extra})
        return cost.render_s(d) + cost.score_s(d) + (AST_CLIP_S if kind == "laugh" else 0.0)

    ref_spoken = spoken_of(EMO_REF_TEXT)
    for k in range(EMO_REF_NEUTRAL if emos else 0):
        work += add(f"n{k}", "neutral", ref_spoken, ref_spoken, vc.text_seed(args.seed, "emoref", "neutral")
                    + SEED_STRIDE * k)
    for e in emos:
        for k in range(args.emo_ref_n):
            work += add(f"{e}{k}", "emo", vc.compose(EMO_REF_INSTRUCTIONS[e], ref_spoken), ref_spoken,
                        vc.text_seed(args.seed, "emoref", e) + SEED_STRIDE * k, emotion=e,
                        instruction=EMO_REF_INSTRUCTIONS[e])
    emo_s = work
    plan = {"emotions": emos, "emo_jobs": len(jobs), "laugh_lines": 0, "laugh_jobs": 0}
    if args.laughter_usd > 0 and laugh_rows and not (state.get("laughter") or {}).get("done"):
        budget = args.laughter_usd / USD_PER_H * 3600 - (0.0 if emos else cost.server_s + cost.load_s)
        spent, lines = 0.0, set()
        for s in range(LAUGH_SEEDS):
            for r in laugh_rows:
                for i, ins in enumerate(LAUGH_INSTRUCTIONS):
                    d = est_dur({"spoken": r["spoken"]}, cps) + LAUGH_PAD_S
                    c = cost.render_s(d) + cost.score_s(d) + AST_CLIP_S
                    if spent + c > budget:
                        break
                    spent += add(f"{r['id']}-i{i}-s{s}", "laugh", vc.compose(ins, r["spoken"]), r["spoken"],
                                 vc.text_seed(args.seed, "laughter", f"{r['id']}/{i}") + SEED_STRIDE * s,
                                 line=r["id"], line_text=r["text"], instruction=ins, plain=r["plain"])
                    lines.add(r["id"])
        work += spent
        plan.update(laugh_lines=len(lines), laugh_jobs=len(jobs) - plan["emo_jobs"])
    fixed = cost.server_s + cost.load_s if jobs else 0.0
    plan.update(audio_s=round(sum(j["est_s"] for j in jobs)), emo_s=round(emo_s + (fixed if emos else 0.0)),
                laugh_s=round(work - emo_s + (0.0 if emos or not jobs else fixed)), s=round(work + fixed))
    return jobs, plan


def prelude_line(plan: dict, usd_per_h: float = USD_PER_H) -> str:
    """The prelude's plan as one printed line."""
    if not plan.get("s"):
        return "prelude: nothing (every emotion has its reference, the laughter trial is done or off)"
    return (f"prelude: emotion references for {', '.join(plan['emotions']) or 'none'} ({plan['emo_jobs']} tries, "
            f"~{plan['emo_s']} s), laughter trial {plan['laugh_jobs']} tries of {plan['laugh_lines']} lines "
            f"(~{plan['laugh_s']} s, ~${plan['laugh_s'] / 3600 * usd_per_h:.2f}); {plan['audio_s']} s of audio, "
            f"~{plan['s']} s = ~${plan['s'] / 3600 * usd_per_h:.2f} in all (one server + scorer loads)")


# ---- the run ------------------------------------------------------------------------------------------------------


class Run:
    """One call: the pool, the state and the waves (render, score in rounds, decide). render(jobs, feed_until,
    on_done, on_fail) -> stats and scoring() -> a Scoring: stand-ins in the tests."""

    def __init__(self, args, out: Path, pool: list[dict], state: dict, gates: Gates, cost: Cost, ref_raw: Path,
                 render: Callable, scoring: Callable[[], Scoring], status: vc.Status | None = None):
        self.args, self.out, self.pool, self.state, self.g, self.cost = args, out, pool, state, gates, cost
        self.by_id = {r["id"]: r for r in pool}
        self.ref_raw, self.render, self.scoring, self.status = ref_raw, render, scoring, status
        self.deadline = args.deadline or math.inf
        self.sizes = collections.Counter(utmos_cat(r) for r in pool)  # UTMOS calibration: a category's pool
        self.added, self.ast_why = 0.0, ""
        self.stopped: set[str] = set()  # categories under --min-pass, logged once
        self.waiting: set[str] = set()  # plain_wait's reasons, logged once

    # -- bookkeeping
    def save(self) -> None:
        vc.write_json(self.out / "state.json", self.state)
        vc.write_rows(self.out / "clips.jsonl", all_clips(self.state, self.out))

    def st(self, rid: str) -> dict:
        return self.state["rows"].setdefault(rid, {"status": "open", "tries": 0, "fails": []})

    def reconcile(self) -> dict:
        """Undecided tries on disk kept (WAV + .json, a try not yet decided of an open text); everything else of
        attempts/ removed; WAVs in wav/ that no kept or human-approved clip (reviewed.jsonl) names removed; a kept clip
        whose WAV is gone reopened."""
        n = collections.Counter()
        for rid, c in list(self.state["kept"].items()):
            if not vc.has_wav(self.out / "wav" / c["wav"]):
                del self.state["kept"][rid]
                st = self.st(rid)
                st["status"] = "open"
                n["kept clips without a WAV: reopened"] += 1
        names = {c["wav"] for c in [*self.state["kept"].values(), *load_reviewed(self.out)]}
        for p in (self.out / "wav").glob("*"):
            if p.name not in names:
                p.unlink()
                n["orphan WAVs removed"] += 1
        root = self.out / "attempts"
        for d in (root.iterdir() if root.is_dir() else []):
            st = self.state["rows"].get(d.name)
            if d.name not in self.by_id or not st or st.get("status") != "open":
                shutil.rmtree(d, ignore_errors=True)
                n["attempt dirs removed"] += 1
                continue
            keep = set(on_disk_tries(self.out, d.name))
            for p in d.iterdir():
                t = p.stem[1:]
                if not (t.isdigit() and int(t) in keep and int(t) >= st["tries"] and p.suffix in (".wav", ".json")):
                    p.unlink()
        return dict(n)

    # -- the prelude: emotion references and the laughter trial
    def ref_for(self, row: dict) -> str:
        """The reference an emotion row clones: its emotion's (refs/<emotion>.wav) when chosen, else the voice's."""
        e = (self.state.get("emotion_refs") or {}).get(row.get("emotion") or "") or {}
        p = self.out / e["wav"] if row["category"] in EMO_CATS and e.get("wav") else None
        return str(p) if p is not None and vc.has_wav(p) else str(self.ref_raw)

    def prelude(self, laugh_rows: list[dict]) -> dict:
        """Render the prelude's tries (prelude_jobs) on one server, score them (Whisper with words, UTMOS, ECAPA to the
        voice, prosody; laughter: CER without the laughter, the burst detector in the span Whisper's words leave free,
        AST when it is here), then choose the emotion references (pick_emo_ref -> refs/<emotion>.wav) and keep the
        laughter trial's best tries (laughter_review/). Skipped when it does not fit in the time left. Its record."""
        jobs, plan = prelude_jobs(self.out, self.state, self.args, self.pool, laugh_rows, self.cost, str(self.ref_raw))
        if not jobs:
            return {}
        if time.time() + plan["s"] > self.deadline - FINAL_S - MIN_WAVE_S:
            log(f"prelude skipped: ~{plan['s']} s does not fit before the deadline (emotion rows clone the voice's "
                "reference; the next call tries again)")
            return {"skipped": plan}
        log(prelude_line(plan))
        shutil.rmtree(self.out / "prelude", ignore_errors=True)
        made, failed, t0 = {}, {}, time.time()
        score_s = sum(self.cost.score_s(j["est_s"]) for j in jobs) + self.cost.load_s
        stats = self.render(jobs, self.deadline - score_s - FINAL_S, lambda j, info: made.__setitem__(j["key"], info),
                            lambda j, e: failed.__setitem__(j["key"], e)) or {}
        atts = [{"rid": j["key"], "t": 0, "set": "prelude", "id": j["key"].replace("/", "-"), "category": j["kind"],
                 "refs": [j["plain"]], "model_text": j["spoken"], "prompt": j["text"], "path": j["out"],
                 **made[j["key"]], **{k: j[k] for k in ("kind", "emotion", "line", "line_text", "instruction", "seed",
                                                        "plain") if k in j}}
                for j in jobs if j["key"] in made]
        until = self.deadline - FINAL_S
        sc = self.scoring() if atts else None
        laughs = []
        try:
            if sc is not None:
                sc.text(atts, until)
                heard = [a for a in atts if a.get("hyp") is not None]
                for a in heard:
                    if a["kind"] == "laugh":
                        hyp = strip_laughter(re.sub(r"[(\[][^)\]]*[)\]]", " ", a["hyp"]))
                        a["cer_nolaugh"] = vc.wer_cer([a["plain"]], hyp)[2]
                sc.rest(heard, until, [(a, False, a["kind"] != "laugh") for a in heard])
                laughs = [a for a in heard if a["kind"] == "laugh"]
                dev = sc.sc.dev() if isinstance(getattr(sc, "sc", None), vc.Scorers) else "cpu"
                ast = AstLaugh(device=dev) if laughs and self.args.laugh_model == "auto" else None
                if ast is not None and ast.available:
                    for a in laughs:
                        try:
                            a["ast_laugh"] = ast(a["path"])
                        except Exception as e:  # noqa: BLE001
                            a["ast_error"] = f"{type(e).__name__}: {e}"[:200]
                self.ast_why = "" if ast is None else ast.why
        finally:
            if sc is not None:
                sc.close()
        for a in laughs:
            try:
                import soundfile as sf

                x, sr = sf.read(a["path"], dtype="float32")
                a["laugh_span"] = [round(t, 2) for t in laugh_span(a.get("words"), a["model_text"], len(x) / sr)]
                a["laugh"] = onset.laugh(x, sr, a["laugh_span"])
            except Exception as e:  # noqa: BLE001
                a["laugh_error"] = f"{type(e).__name__}: {e}"[:200]
        rec = {"jobs": len(jobs), "made": len(made), "failed": len(failed), "plan": plan,
               "render_audio_s": stats.get("audio_s"), "wall_s": round(time.time() - t0, 1)}
        rec["emotion_refs"] = self.choose_refs(atts, plan["emotions"])
        if plan["laugh_jobs"]:
            rec["laughter"] = self.keep_laughter(laughs, plan)
        shutil.rmtree(self.out / "prelude", ignore_errors=True)
        self.state.setdefault("prelude", []).append({k: v for k, v in rec.items() if k != "emotion_refs"})
        self.save()
        return rec

    def choose_refs(self, atts: list[dict], emotions: list[str]) -> dict:
        """pick_emo_ref for each emotion against the plain renders' prosody; the chosen try -> refs/<emotion>.wav
        (16 kHz PCM); state["emotion_refs"][emotion] records it (or why none: that emotion then clones the voice's
        reference; an emotion whose tries were not made is tried again by the next call)."""
        plain = [{**{k: a["prosody"].get(k) for k in vc.PROSODY}, "level_db": a.get("level_in")}
                 for a in atts if a["kind"] == "neutral" and a.get("prosody")]
        base = baseline(plain, 1)
        refs = self.state.setdefault("emotion_refs", {})
        out = {}
        for e in emotions:
            cands = [a for a in atts if a["kind"] == "emo" and a.get("emotion") == e and a.get("hyp") is not None]
            if not cands:
                continue
            best, why = pick_emo_ref(e, cands, base)
            rec = {"candidates": len(cands), "scores": sorted((c.get("score") for c in cands
                                                               if c.get("score") is not None), reverse=True)}
            if best is not None:
                dur = to_16k(Path(best["path"]), self.out / REFS_DIR / f"{e}.wav")
                rec.update(wav=f"{REFS_DIR}/{e}.wav", duration=round(dur, 3), seed=best["seed"],
                           instruction=best.get("instruction"), text=EMO_REF_TEXT, sim=best.get("sim"),
                           utmos=best.get("utmos"), score=best["score"], delta=best.get("delta"))
                log(f"emotion reference {e}: seed {best['seed']}, sim {best.get('sim')}, score {best['score']}, "
                    f"delta {best.get('delta')}")
            else:
                rec.update(wav=None, why=why)
                log(f"emotion reference {e}: none ({why}); its rows clone the voice's reference")
            refs[e] = out[e] = rec
        return out

    def keep_laughter(self, laughs: list[dict], plan: dict) -> dict:
        """The LAUGH_KEEP best tries of each line (laugh_rank) -> laughter_review/wav/<line>-<k>.wav, review.jsonl and
        laughter.html beside them; never the manifest. state["laughter"]: done and the trial's summary."""
        d = self.out / REVIEW
        shutil.rmtree(d, ignore_errors=True)
        by = collections.defaultdict(list)
        for a in laughs:
            by[a["line"]].append(a)
        rows = []
        for lid in sorted(by):
            for k, a in enumerate(sorted(by[lid], key=laugh_rank, reverse=True)[:LAUGH_KEEP], 1):
                name = f"{lid}-{k}"
                dur = to_16k(Path(a["path"]), d / "wav" / f"{name}.wav")
                rows.append({"id": name, "line": lid, "rank": k, "text": a.get("line_text"),
                             "instruction": a.get("instruction"), "seed": a.get("seed"), "wav": f"wav/{name}.wav",
                             "duration": round(dur, 3), "hyp": a.get("hyp"), "cer_nolaugh": a.get("cer_nolaugh"),
                             "sim": a.get("sim"), "utmos": a.get("utmos"), "laugh": a.get("laugh"),
                             "laugh_span": a.get("laugh_span"), "ast_laugh": a.get("ast_laugh")})
        vc.write_rows(d / "review.jsonl", rows)
        write_text(d / "laughter.html", laughter_html(rows, "wav"))
        found = [laugh_rank(a)[2] for a in laughs]
        summ = {"done": True, "lines": len(by), "tries": len(laughs), "planned": plan["laugh_jobs"], "kept": len(rows),
                "heard_right": round(sum(a.get("cer_nolaugh") == 0 for a in laughs) / len(laughs), 3) if laughs
                else None, "laugh_found": round(sum(found) / len(found), 3) if found else None,
                "detector": "ast" if any(a.get("ast_laugh") is not None for a in laughs) else "bursts",
                "ast_unavailable": getattr(self, "ast_why", "") or None, "est_s": plan["laugh_s"]}
        self.state["laughter"] = summ
        log(f"laughter trial: {summ}")
        return summ

    # -- one wave
    def plan(self) -> tuple[list[dict], dict[str, float]]:
        kept = kept_s(self.state)
        targets = category_targets(self.args.hours, parse_mix(self.args.mix), kept,
                                   exhausted_cats(self.pool, self.state), self.args.fill)
        on_disk = {r["id"]: on_disk_tries(self.out, r["id"]) for r in self.pool
                   if row_status(self.state, r["id"]) == "open"}
        rows = wave_rows(self.pool, self.state, targets, self.args.tries, chars_s(self.state), on_disk,
                         self.args.min_sim, self.args.long_tries, self.args.number_share, self.args.min_pass,
                         self.args.cwords_tries)
        for c, m in stopped_cats(self.state, self.args.min_pass).items():
            if c not in self.stopped:
                self.stopped.add(c)
                log(f"{c}: {m} of its tries pass, under --min-pass {self.args.min_pass:g}: no new {c} texts (its open "
                    f"ones finish; its target is not handed to --fill). Look at its gate fails in PREP_REPORT.md; "
                    f"--min-pass 0 goes on anyway")
        for r in self.pool:
            if is_plain(r) and row_status(self.state, r["id"]) in ("new", "open") \
                    and (why := plain_wait(self.state, r)) and why not in self.waiting:
                self.waiting.add(why)
                log(f"{r['category']} renders plain: its {r.get('emotion')} texts wait ({why})")
        return rows, targets

    def wave(self, rows: list[dict]) -> dict:
        """Render the wave's tries, score and decide them; the wave's record."""
        a, out = self.args, self.out
        if gone := [r for r in rows if is_plain(r) and self.ref_for(r) == str(self.ref_raw)]:
            log(f"plain rows whose emotion's reference clip is not here wait: {', '.join(r['id'] for r in gone)}")
            rows = [r for r in rows if r not in gone]
        on_disk = {r["id"]: on_disk_tries(out, r["id"]) for r in rows}
        budget = self.deadline - time.time() - FINAL_S
        cps = chars_s(self.state)
        phon_load = 0.0 if pc.loaded() else PHON_LOAD_S  # the IPA model loads once a process (later waves move it)
        rows, est = fit_wave(rows, self.state, a, self.cost, budget, on_disk, cps, phon_load)
        rec = {"rows": len(rows), "est_s": round(est), "t0": time.time()}
        if not rows:
            return rec
        for r in rows:  # their tries on disk survive a kill (reconcile keeps the tries of open texts)
            st = self.st(r["id"])
            if st["status"] == "new":  # a text a render-mode switch made new again (switch_mode)
                st["status"] = "open"
        self.save()
        jobs, failed = [], {}
        for r in rows:
            st = self.state["rows"].get(r["id"])
            todo, _ = wave_tries(r, st, a.tries, a.prerender, on_disk[r["id"]], a.long_tries, a.cwords_tries)
            for t in todo:
                v = variant_of(r, t, st)
                jobs.append({"key": f"{r['id']}/t{t}", "group": "corpus", "text": render_text(r, v),
                             "ref": self.ref_for(r), "lora": None, "cfg": a.cfg, "ts": a.timesteps,
                             "seed": seed_of(a.seed, r, t), "out": str(attempt_wav(out, r["id"], t)),
                             "est_s": round(vc.est_audio_s(render_spoken(r, v)), 2), "backend": a.backend,
                             "ref_prep": True, "rid": r["id"], "try": t, "variant": v})
        if jobs:
            cc = [r for r in rows if c_checked(r, a)]
            score_need = (sum(self.cost.score_s(est_dur(r, cps)) for r in rows) + self.cost.load_s
                          + sum(self.cost.c_s(est_dur(r, cps)) for r in cc) + (phon_load if cc else 0.0))
            stats = self.render(jobs, self.deadline - score_need - FINAL_S, self._on_done,
                                lambda j, e: failed.__setitem__(j["key"], e)) or {}
            for k in ("jobs", "done"):
                self.state["render"][k] += int(stats.get(k) or 0)
            for k in ("audio_s", "wall_s", "server_s"):
                self.state["render"][k] = round(self.state["render"][k] + float(stats.get(k) or 0.0), 1)
            rec.update(rendered=stats.get("done"), render_audio_s=stats.get("audio_s"),
                       audio_per_gpu_s=stats.get("audio_per_wall"), render_stopped=stats.get("stopped") or "")
            if len(failed) >= max(SYSTEMIC_N, SYSTEMIC_SHARE * len(jobs)):  # the engine, not the texts: no try spent
                raise RuntimeError(f"synthesis failed for {len(failed)} of {len(jobs)} tries, e.g. "
                                   f"{next(iter(failed.values()))}")
        self.state["render"]["waves"] += 1
        rec.update(self.score(rows, failed))
        return rec

    def _on_done(self, job: dict, info: dict) -> None:
        """A made try's record next to its WAV (what the decision needs: its text, seed, variant, synthesis info)."""
        side = {"render_text": job["text"], "seed": job["seed"], "variant": job["variant"], "try": job["try"],
                **{k: info[k] for k in ("duration", "onset_bad", "onset_why", "onset_retries", "onset_trimmed",
                                        "engine", "tries", "bad_case", "onset_error", "level_in") if k in info}}
        if job["ref"] != str(self.ref_raw):
            side["ref"] = Path(job["ref"]).name  # an emotion reference
        vc.write_json(Path(job["out"]).with_suffix(".json"), side)

    def attempt(self, row: dict, t: int) -> dict:
        """A try as the scorers take it (voice_contest's row fields) with its synthesis record."""
        wav = attempt_wav(self.out, row["id"], t)
        side = vc.read_json(wav.with_suffix(".json")) or {}
        v = side.get("variant", 0)
        a = {"rid": row["id"], "t": t, "set": "soru" if row.get("pattern") else row["category"],
             "id": f"{row['id']}-t{t}", "category": row.get("kind") or row["category"], "refs": [row["spoken"]],
             "model_text": render_spoken(row, v), "prompt": side.get("render_text") or render_text(row, v),
             "path": str(wav), **side}
        if row.get("pattern"):
            a.update({k: row[k] for k in ("pattern", "peak", "peak_word", "pair_id") if k in row})
        return a

    def score(self, rows: list[dict], failed: dict[str, str]) -> dict:
        """Rounds: each open text's next made try; cheap gates, Whisper, the rest; decide; until no try is left or
        the deadline. Returns the counts."""
        until = self.deadline - FINAL_S
        n = collections.Counter()
        sc, t0, scored_audio, model_s = None, time.time(), 0.0, 0.0
        pending = {r["id"]: on_disk_tries(self.out, r["id"]) for r in rows}
        for key in failed:  # synthesis failures: a decided (failed) try each
            rid, t = key.rsplit("/t", 1)
            if rid in self.by_id and int(t) not in pending.get(rid, []):
                pending.setdefault(rid, []).append(int(t))
        pending = {rid: sorted(ts) for rid, ts in pending.items()}
        try:
            while time.time() < until:
                batch = []
                for rid, ts in pending.items():
                    st = self.state["rows"].get(rid) or {}
                    if st.get("status", "new") not in ("new", "open"):
                        continue
                    nxt = [t for t in ts if t >= st.get("tries", 0)]
                    if nxt:
                        a = self.attempt(self.by_id[rid], nxt[0])
                        if f"{rid}/t{nxt[0]}" in failed:
                            a["synth_error"] = failed[f"{rid}/t{nxt[0]}"]
                        batch.append(a)
                if not batch:
                    break
                for a in batch:
                    a["fails"] = cheap_fails(a, self.g)
                live = [a for a in batch if not a["fails"]]
                if live:
                    sc = sc or self.scoring()
                    sc.text(live, until)
                    for a in live:
                        if a.get("hyp") is not None or a.get("whisper_error"):
                            a["fails"] = text_fails(a, self.g)
                        else:
                            a["undecided"] = True  # the deadline cut Whisper short: the next call
                    systemic(live, "whisper_error")
                    heard = [a for a in live if not a["fails"] and not a.get("undecided")]
                    cc = [a for a in heard if self.g.c_check and needs_c(self.by_id[a["rid"]], a["model_text"])]
                    if cc:  # the c-check after CER: phon_check's IPA model, loaded only for these
                        sc.phon(cc, until)
                        for a in cc:
                            if a.get("c_got") is None and not a.get("c_error"):
                                a["undecided"] = True  # the deadline cut it short
                        systemic(cc, "c_error")
                        model_s += sum(c_est(a["duration"]) for a in cc)
                    for a in heard:
                        if a.get("undecided"):
                            continue
                        a["fails"] = c_fails(a, self.by_id[a["rid"]], self.g)
                        if not a["fails"]:  # edges (a stray tail cut here), sibilance, word timing
                            measure_clip(a, self.by_id[a["rid"]], self.g)
                            a["fails"] = clip_fails(a, self.by_id[a["rid"]], self.g)
                    systemic([a for a in live if "edges" in a or "clip_error" in a], "clip_error")
                    full = [a for a in live if not a["fails"] and not a.get("undecided")]
                    if full:
                        cpu = [(a, bool(self.by_id[a["rid"]].get("pattern")), self.wants_prosody(a)) for a in full]
                        sc.rest(full, until, cpu)
                        for a in full:  # a score neither made nor failed: cut short at the deadline
                            if any(a.get(k) is None and not a.get(f"{k}_error") for k in ("utmos", "sim")) \
                                    or not a.get("cpu_done"):
                                a["undecided"] = True
                        systemic(full, "utmos_error")
                        systemic(full, "sim_error")
                    scored_audio += sum(a["duration"] for a in live)
                    model_s += sum(score_est(a["duration"]) for a in live)  # without the factor: make_cost's divisor
                decided = self.decide([a for a in batch if not a.get("undecided")])
                n.update(decided)
                self.save()
                if any(a.get("undecided") for a in batch):
                    break
        finally:
            if sc is not None:
                sc.close()
        wall = time.time() - t0
        clips = n["kept"] + n["failed"]
        self.state["scoring"]["clips"] += clips
        self.state["scoring"]["audio_s"] = round(self.state["scoring"]["audio_s"] + scored_audio, 1)
        self.state["scoring"]["wall_s"] = round(self.state["scoring"]["wall_s"] + wall, 1)
        if self.args.throughput_log and clips >= SCORE_LOG_MIN and sc is not None:
            phon_load = float(getattr(sc, "phon_load_s", 0.0) or 0.0)  # measured: in wall, not the factor's business
            line = {"time": vc.now_iso(), "mode": "corpus-score", "backend": "scorers", "clips": clips,
                    "audio_s": round(scored_audio, 1), "wall_s": round(wall, 1),
                    "model_s": round(model_s + self.cost.load_s + phon_load, 1), "raw": True}
            append_rows(Path(self.args.throughput_log), [{**line, **({"phon_load_s": round(phon_load, 1)}
                                                                     if phon_load else {})}])
        return {"kept": n["kept"], "failed": n["failed"], "dropped": n["dropped"], "score_wall_s": round(wall, 1)}

    def wants_prosody(self, a: dict) -> bool:
        cat = self.by_id[a["rid"]]["category"]
        return cat in EMO_CATS or (cat in BASELINE_CATS and len(self.state["baseline"]) < BASELINE_N)

    def decide(self, batch: list[dict]) -> collections.Counter:
        """Calibrate UTMOS on the round's first tries, decide the neutral texts (their prosody: the emotion baseline),
        then the emotions; keep, retry or drop each."""
        n = collections.Counter()
        for a in batch:
            row = self.by_id[a["rid"]]
            first = a["t"] == (self.state["rows"].get(a["rid"]) or {}).get("try0", 0)  # in its render mode
            if first and not a["fails"] and a.get("utmos") is not None:
                for k in {utmos_key(row), utmos_cat(row)}:
                    self.state["first_utmos"].setdefault(k, []).append(a["utmos"])
        th = utmos_thresholds(self.state["first_utmos"], self.g, self.sizes)
        for a in sorted(batch, key=lambda a: self.by_id[a["rid"]]["category"] in EMO_CATS):
            row = self.by_id[a["rid"]]
            if row["category"] in EMO_CATS and not a["fails"]:
                base = baseline(self.state["baseline"])
            else:
                base = None
            if not a["fails"]:
                a["fails"] = full_fails(a, row, self.g, th, base)
            n.update(self.apply(a, row))
        return n

    def apply(self, a: dict, row: dict) -> list[str]:
        st, cat = self.st(row["id"]), row["category"]
        # the category's totals and, once it has a mode record, its render mode's counters (measured_pass reads those)
        counts = [cat_stats(self.state, cat), *filter(None, [mode_stats(self.state, cat, create=True)])]
        ok, first = not a["fails"], a["t"] == st.get("try0", 0)  # first: the text's first try in its render mode
        for s in counts:
            s["decided"] += 1
            s["audio_s"] = round(s["audio_s"] + (a.get("duration") or 0.0), 2)
            s["passed"] += ok
            s["decided_g2"] = s.get("decided_g2", 0) + 1  # under the pilot's gates (p_try)
            s["passed_g2"] = s.get("passed_g2", 0) + ok
            s["first"] += first
            s["first_pass"] += first and ok
            for f in a["fails"]:
                s["fails"][f] = s["fails"].get(f, 0) + 1
        if a.get("variant") is not None and a.get("hyp") is not None and "cer" not in a["fails"] \
                and "whisper" not in a["fails"]:
            st["var_ok"] = a["variant"]  # Whisper heard the canonical text with this variant
        append_rows(self.out / "attempts.jsonl", [attempt_record(a, row)])
        out = []
        if ok:
            dur = to_16k(Path(a["path"]), self.out / "wav" / f"{row['id']}.wav")
            self.state["kept"][row["id"]] = clip_record(a, row, dur)
            st.update(status="kept", tries=a["t"] + 1)
            for s in counts:
                s["kept"] += 1
                s["kept_s"] = round(s["kept_s"] + dur, 2)
            self.added += dur
            if cat in BASELINE_CATS and a.get("prosody") and len(self.state["baseline"]) < BASELINE_N:
                self.state["baseline"].append({**{k: a["prosody"].get(k) for k in vc.PROSODY},
                                               "level_db": a.get("level_in")})
            out.append("kept")
        else:
            st["tries"] = a["t"] + 1
            st["fails"].append(a["fails"])
            out.append("failed")
            if st["tries"] >= try_end(row, st, self.args.tries, self.args.long_tries, self.args.cwords_tries):
                st["status"] = "dropped"
                for s in counts:
                    s["dropped"] += 1
                out.append("dropped")
        if st["status"] != "open":
            shutil.rmtree(self.out / "attempts" / row["id"], ignore_errors=True)
        else:
            for p in (Path(a["path"]), Path(a["path"]).with_suffix(".json")):
                p.unlink(missing_ok=True)
        return out

    def loop(self) -> list[dict]:
        """Waves until every category has its hours (or its texts are used up) or the deadline."""
        waves = []
        while True:
            rows, targets = self.plan()
            if not rows:
                log("every category has its hours, or its texts are used up"
                    + (f" (or under --min-pass: {', '.join(sorted(self.stopped))})" if self.stopped else ""))
                break
            if self.deadline - time.time() - FINAL_S < MIN_WAVE_S and waves:
                log("deadline: no time for another wave (the same command continues)")
                break
            rec = self.wave(rows)
            rec["t0"] = round(time.time() - rec["t0"], 1)
            waves.append(rec)
            k = kept_s(self.state)
            log(f"wave {len(waves)}: {rec}; kept {sum(k.values()) / 3600:.2f} h "
                f"({', '.join(f'{c} {v / 3600:.2f}/{targets[c] / 3600:.2f}' for c, v in k.items() if targets[c])})")
            if self.status:
                self.status.set(phase="waves", waves=len(waves), kept_h=round(sum(k.values()) / 3600, 3))
            if not rec["rows"] or (not rec.get("kept") and not rec.get("failed") and not rec.get("rendered")):
                log("nothing fitted in the time left (the same command continues)")
                break
        return waves


def systemic(atts: list[dict], err: str) -> None:
    """RuntimeError when a scorer failed on SYSTEMIC_N and SYSTEMIC_SHARE of a round's clips (it, not the clips)."""
    bad = [a for a in atts if a.get(err)]
    if len(bad) >= max(SYSTEMIC_N, SYSTEMIC_SHARE * len(atts)):
        raise RuntimeError(f"{err} on {len(bad)} of {len(atts)} clips, e.g. {bad[0][err]}")


CLIP_KEYS = ("edges", "sib_db", "sib_ms", "first_stretch", "rate_ws", "speech_s", "tail_cut_s", "clip_error",
             "pause_sent", "pause_mid", "pause_max")
C_KEYS = ("c_expected", "c_got", "ipa", "c_error")  # the c-check's measures of a try (Scoring.phon)


def attempt_record(a: dict, row: dict) -> dict:
    """attempts.jsonl: one decided try with its scores and the gates it failed."""
    keys = ("t", "seed", "variant", "duration", "render_text", "engine", "onset_bad", "onset_retries", "onset_trimmed",
            "onset_final_bad", "hyp", "char_errors", "ref_chars", "p_tr", "lang", "utmos", "utmos_threshold", "sim",
            "sim_floor", "emotion_verdict", "emotion_delta", "synth_error", "whisper_error", "level_in", "ref",
            *CLIP_KEYS, *C_KEYS)
    out = {"id": row["id"], "category": row["category"], **{k: a[k] for k in keys if a.get(k) is not None},
           "fails": a["fails"]}
    for k in ("kind", "emotion", "render"):  # render: "plain" (rendered without its instruction)
        if row.get(k):
            out[k] = row[k]
    if a.get("intonation"):
        it = a["intonation"][0]
        out["intonation"] = {k: it.get(k) for k in ("pattern", "pass", "peak_st", "drop_st", "end_st", "slope_st_s",
                                                    "peak_at", "lead_st", "range_st")}
    return out


def clip_record(a: dict, row: dict, dur: float) -> dict:
    """clips.jsonl: a kept clip, its texts and every score."""
    rec = {"id": row["id"], "category": row["category"], "wav": f"{row['id']}.wav", "duration": round(dur, 3),
           "text": row["text"], "spoken": row["spoken"], "train_text": train_text(row),
           "render_text": a.get("render_text"), "render_spoken": a["model_text"], "source": row["source"],
           **{k: row[k] for k in ("kind", "emotion", "instruction", "pair_id", "line_id", "domain_category",
                                  "line_ids", "doc_group", "sentences", "breaks", "kinds", "render")
              if row.get(k) is not None}}
    for k in ("t", "seed", "variant", "engine", "hyp", "char_errors", "ref_chars", "p_tr", "utmos", "utmos_threshold",
              "sim", "sim_floor", "onset_bad", "onset_retries", "onset_trimmed", "prosody", "emotion_verdict",
              "emotion_delta", "level_in", "ref", "words", *CLIP_KEYS, *C_KEYS):
        if a.get(k) is not None:
            rec["try" if k == "t" else k] = a[k]
    if a.get("intonation"):
        rec["intonation"] = a["intonation"][0]
    return rec


# ---- human-approved clips (--add-reviewed) ------------------------------------------------------------------------


def load_reviewed(out: Path) -> list[dict]:
    """reviewed.jsonl's clips whose WAV is in wav/ (a clip without one is in no manifest)."""
    return [c for c in vc.read_rows(out / REVIEWED) if vc.has_wav(out / "wav" / c["wav"])]


def all_clips(state: dict, out: Path) -> list[dict]:
    """clips.jsonl: the kept clips by id, then the human-approved ones."""
    return sorted(state["kept"].values(), key=lambda c: c["id"]) + load_reviewed(out)


def reviewed_record(r: dict, dur: float, best: bool) -> dict:
    """clips.jsonl / reviewed.jsonl: an approved laughter_review/ clip; training text "(its instruction)" + the trnorm
    spoken form of its line (the laughter as written: trnorm leaves "Hahaha" alone)."""
    cid = clean_id(f"laugh_{r['id']}")
    spoken = spoken_of(r["text"])
    rec = {"id": cid, "category": REVIEWED_CAT, "wav": f"{cid}.wav", "duration": round(dur, 3), "text": r["text"],
           "spoken": spoken, "train_text": vc.compose(r.get("instruction"), spoken), "source": REVIEWED_SOURCE,
           "review_id": r["id"], **({"best": True} if best else {})}
    for k in ("instruction", "line", "rank", "seed", "hyp", "cer_nolaugh", "sim", "utmos", "ast_laugh"):
        if r.get(k) is not None:
            rec[k] = r[k]
    return rec


def manifest_stats(clips: list[dict], ref_paths: list[str]) -> dict:
    """The report's "manifest": rows, rows with ref_audio (the prepared reference among them), with an instruction."""
    return {"rows": len(clips), "with_ref_audio": len(ref_paths),
            "ref_is_reference": sum(1 for p in ref_paths if p.endswith(".wav") and "/ref/" in p.replace("\\", "/")),
            "with_instruction": sum(c["train_text"] != c["spoken"] for c in clips)}


def fold_reviewed(rep: dict, kept: list[dict], reviewed: list[dict]) -> dict:
    """The report with the human-approved clips: their own category (REVIEWED_CAT, no mix share or target),
    selected hours / clips and every category's share over all clips. Idempotent (the category is replaced);
    "reached" / "complete" stay the rendered clips' (the mix's targets)."""
    by = {c: b for c, b in rep["by_category"].items() if c != REVIEWED_CAT}
    secs = collections.Counter()
    for c in [*kept, *reviewed]:
        secs[c["category"]] += c["duration"]
    if reviewed:
        by[REVIEWED_CAT] = {"clips": len(reviewed), "hours": round(secs[REVIEWED_CAT] / 3600, 3), "share": 0.0,
                            "mix": 0.0, "target_hours": 0.0, "pool": {}, "tries": 0, "pass_rate": None,
                            "first_pass_rate": None, "fails": {}, "duration_s": _q([c["duration"] for c in reviewed]),
                            "source": REVIEWED_SOURCE}
    total = sum(secs.values())
    for c, b in by.items():
        b["share"] = round(secs[c] / max(total, 1e-9), 3)
    rep["by_category"] = by
    rep["selected"] = {**rep["selected"], "hours": round(total / 3600, 3), "clips": len(kept) + len(reviewed)}
    rep["reviewed"] = {"clips": len(reviewed), "seconds": round(secs[REVIEWED_CAT], 2), "file": REVIEWED,
                       "category": REVIEWED_CAT, "source": REVIEWED_SOURCE,
                       "best": sum(bool(c.get("best")) for c in reviewed)} if reviewed else {}
    return rep


def render_busy(out: Path) -> str:
    """Why a corpus-render call may be writing out now ("" when none seems to): status.json's phase is not done /
    error (a call going, or one killed mid-call: the same corpus-render command finishes it), or state.json is newer
    than prep_report.json (modal_app.report_current)."""
    st = vc.read_json(out / "status.json") or {}
    if st.get("phase") in RUNNING:
        return f"status.json says phase {st['phase']!r} (updated {st.get('updated')})"
    rep, sj = out / "prep_report.json", out / "state.json"
    if not rep.is_file() or not sj.is_file():
        return "no prep_report.json / state.json: no finished corpus-render call"
    if sj.stat().st_mtime > rep.stat().st_mtime:
        return "state.json is newer than prep_report.json: a call cut short or still going"
    return ""


def add_reviewed(out: Path, approved_path: Path, dry: bool = False) -> int:
    """--add-reviewed (module doc): the approved laughter_review/ clips -> wav/, reviewed.jsonl, clips.jsonl,
    voxcpm_train.jsonl, ATTRIBUTION.txt, the report. 0 (also when nothing is new), 2 when refused."""
    if not (out / MARKER).is_file():
        log(f"--out {out}: not a render_corpus.py corpus (no {MARKER})")
        return 2
    if why := render_busy(out):
        log(f"refused: {why}; add reviewed clips only between corpus-render calls (a killed call: run the same "
            "corpus-render command again, it finishes and writes the report)")
        return 2
    try:
        approved = json.loads(approved_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log(f"--add-reviewed {approved_path}: {e}")
        return 2
    ids = sorted(k for k, v in approved.items() if v is True) if isinstance(approved, dict) else []
    marked = approved.get("best") if isinstance(approved, dict) else None
    best = {f"{line}-{k}" for line, ks in (marked if isinstance(marked, dict) else {}).items()
            for k in (ks if isinstance(ks, list) else [ks])}
    review = {r["id"]: r for r in vc.read_rows(out / REVIEW / "review.jsonl")}
    missing = [i for i in ids if i not in review or not vc.has_wav(out / REVIEW / review[i]["wav"])]
    if not ids or missing:
        log(f"--add-reviewed {approved_path}: " + (f"not in {REVIEW}/review.jsonl or no WAV: {missing}" if missing
                                                   else 'no approved ids ({"<review id>": true})'))
        return 2
    state = vc.read_json(out / "state.json") or {}
    rep = vc.read_json(out / "prep_report.json") or {}
    kept = sorted((state.get("kept") or {}).values(), key=lambda c: c["id"])
    old = {c["id"]: c for c in vc.read_rows(out / REVIEWED)}
    have = {c["review_id"] for c in load_reviewed(out)}
    new = [i for i in ids if i not in have]
    if gone := sorted(c["review_id"] for c in old.values() if c["review_id"] not in ids):
        log(f"in {REVIEWED} but not approved in {approved_path.name} (kept; remove them by hand): {gone}")
    secs = sum(float(review[i]["duration"]) for i in new)
    log(f"{len(ids)} approved, {len(ids) - len(new)} already in the corpus; adding {len(new)} clips, ~{secs:.1f} s "
        f"(category {REVIEWED_CAT})")
    if dry:
        return 0
    added = []
    for i in new:
        rec = reviewed_record(review[i], 0.0, i in best)
        rec["duration"] = round(to_16k(out / REVIEW / review[i]["wav"], out / "wav" / rec["wav"]), 3)
        old[rec["id"]] = rec
        added.append(rec)
    if added:
        vc.write_rows(out / REVIEWED, list(old.values()))  # the record later finalizes merge: before the files below
    reviewed = load_reviewed(out)
    # every approved clip clips.jsonl / the manifest lack (the new ones; also any an interrupted call left out)
    ids_there = {c["id"] for c in vc.read_rows(out / "clips.jsonl")}
    add_clips = [c for c in reviewed if c["id"] not in ids_there]
    settings = rep.get("settings") or {}
    voice = (state.get("key") or {}).get("voice") or settings.get("voice") or VOICE
    prepared = out / "ref" / f"{voice}.wav"
    ref = (str(prepared), vc.wav_duration(prepared)) if vc.has_wav(prepared) else None
    refs = reviewed_refs(reviewed, kept, float(settings.get("ref_fraction", REF_FRACTION)),
                         int(settings.get("seed", 0)), ref)
    rows = vc.read_rows(out / "voxcpm_train.jsonl")
    there = {r["audio"] for r in rows}
    add_rows = sorted((r for r in manifest_rows(reviewed, refs, out / "wav", 0) if r["audio"] not in there),
                      key=lambda r: r["audio"])
    if not (add_clips or add_rows):
        log("nothing to add: every approved clip is in clips.jsonl and the manifest")
        return 0
    append_rows(out / "clips.jsonl", add_clips)
    append_rows(out / "voxcpm_train.jsonl", add_rows)
    rows += add_rows
    write_text(out / "ATTRIBUTION.txt", "".join(x + "\n" for x in attribution_lines(
        [*kept, *reviewed], voice, (rep.get("voice") or {}).get("description") or DESCRIPTION)))
    if rep.get("by_category"):  # the report last (modal_app.report_current: a report newer than the state)
        fold_reviewed(rep, kept, reviewed)
        rep["manifest"] = manifest_stats([*kept, *reviewed], [r["ref_audio"] for r in rows if r.get("ref_audio")])
        rep["reviewed"]["added"] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "clips": len(added),
                                    "seconds": round(sum(c["duration"] for c in added), 2),
                                    "approved": approved_path.name}
        vc.write_json(out / "prep_report.json", rep)
        write_text(out / "PREP_REPORT.md", render_markdown(rep))
    log(f"added {len(added)} clips, {sum(c['duration'] for c in added):.1f} s ({len(add_rows)} manifest rows); "
        f"{REVIEWED_CAT} now {len(reviewed)} clips, {sum(c['duration'] for c in reviewed):.1f} s; corpus "
        f"{rep.get('selected', {}).get('hours')} h -> {out}")
    return 0


# ---- a second corpus: references from the first (--refs-from), the merge (--merge) --------------------------------


def refs_problem(src: Path, voice: str, ref_in: Path) -> str:
    """Why --refs-from src cannot lend its references ("" when it can): no ref/<voice>.raw.wav, a state of another
    voice, or a raw reference that is not --ref's clip (when --ref is here)."""
    raw = src / "ref" / f"{voice}.raw.wav"
    if not vc.has_wav(raw):
        return f"no ref/{raw.name} (a render_corpus.py corpus of voice {voice})"
    was = ((vc.read_json(src / "state.json") or {}).get("key") or {}).get("voice")
    if was and was != voice:
        return f"its voice is {was}, this corpus's {voice}"
    if ref_in.is_file() and sha256_file(raw) != sha256_file(ref_in):
        return f"its reference ({raw}) is another clip than --ref {ref_in}"
    return ""


def reuse_ref_files(src: Path, out: Path, voice: str) -> list[str]:
    """src's ref/<voice>.raw.wav and ref/<voice>.wav (prepared) and refs/*.wav -> out where out has none; the copied
    paths (relative)."""
    names = [f"ref/{voice}.raw.wav", f"ref/{voice}.wav"]
    names += [f"{REFS_DIR}/{p.name}" for p in sorted((src / REFS_DIR).glob("*.wav"))]
    done = []
    for n in names:
        if vc.has_wav(src / n) and not vc.has_wav(out / n):
            (out / n).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / n, out / f"{n}.part")
            os.replace(out / f"{n}.part", out / n)
            done.append(n)
    return done


def reuse_emotion_refs(src: Path, out: Path, state: dict) -> list[str]:
    """src's emotion references (state.json "emotion_refs" with a WAV that out now has) into state where it has none
    with a WAV ("from": src); the emotions taken. The prelude then renders none for them."""
    have = state.setdefault("emotion_refs", {})
    took = []
    for e, rec in sorted(((vc.read_json(src / "state.json") or {}).get("emotion_refs") or {}).items()):
        if isinstance(rec, dict) and rec.get("wav") and vc.has_wav(out / rec["wav"]) \
                and not (have.get(e) or {}).get("wav"):
            have[e] = {**rec, "from": str(src)}
            took.append(e)
    return took


def lent_refs(src: Path) -> dict:
    """The emotion references a --refs-from corpus lends, for a plan (dry_run): its state.json "emotion_refs" with a
    WAV, else its prep_report.json's (corpus-fetch brings the report, not the state); {} when neither is here."""
    for name in ("state.json", "prep_report.json"):
        refs = {e: r for e, r in ((vc.read_json(src / name) or {}).get("emotion_refs") or {}).items()
                if isinstance(r, dict) and r.get("wav")}
        if refs:
            return refs
    return {}


MERGED_KEYS = ("audio", "text", "duration", "ref_audio", "ref_duration")  # a manifest row (manifest_rows)


def load_instructions_tr(path: Path) -> dict[str, str]:
    """INSTRUCTIONS_TR: {english wording: turkish wording}. ValueError unless every wording on both sides is one
    stripped line that voice_contest.strip_instruction takes as it is (no parentheses: they would end VoxCPM's
    instruction early), no Turkish wording is its English one or another English key, and each Turkish wording is its
    own."""
    table = vc.read_json(path)
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{path}: want a JSON object {{english wording: turkish wording, ...}}")
    for en, tr in table.items():
        for s in (en, tr):
            try:
                ok = isinstance(s, str) and "\n" not in s and vc.strip_instruction(s) == s
            except ValueError:
                ok = False
            if not ok:
                raise ValueError(f"{path}: {s!r} is not an instruction wording (one stripped line, no parentheses)")
    if same := sorted(set(table.values()) & set(table)):
        raise ValueError(f"{path}: Turkish wordings that are English keys too, e.g. {same[:3]}")
    if dup := sorted(t for t, n in collections.Counter(table.values()).items() if n > 1):
        raise ValueError(f"{path}: Turkish wordings given to more than one English one, e.g. {dup[:3]}")
    return dict(table)


def instruction_copies(rows: list[dict], mode: str, table: dict[str, str]
                       ) -> tuple[list[dict], collections.Counter]:
    """--merge --instructions (module doc "Merge"): the manifest rows of a mode, each with "_lang" (plain: no
    instruction; en / tr), and the instruction wordings table lacks (wording -> rows: the merge refuses them). en: the
    rows as they are, the table not read; both: each instructed row, then its copy with the table's Turkish wording in
    place of the English one (all else the same); tr: the copy only. A row whose instruction already is one of the
    table's Turkish wordings stays once ("tr")."""
    turkish, out, unknown = set(table.values()), [], collections.Counter()
    for r in rows:
        m = pc.CONTROL.match(r["text"])
        ins = m.group(1).strip() if m else ""
        if not ins:
            out.append({**r, "_lang": "plain"})
        elif mode == "en":
            out.append({**r, "_lang": "en"})
        elif ins in turkish:
            out.append({**r, "_lang": "tr"})
        elif ins in table:
            tr = {**r, "text": r["text"][:m.start(1)] + table[ins] + r["text"][m.end(1):], "_lang": "tr"}
            out += [{**r, "_lang": "en"}, tr] if mode == "both" else [tr]
        else:
            unknown[ins] += 1
    return out, unknown


def merge_out_for(out: str, was: str, mode: str) -> str:
    """The --out to suggest for a merge with --instructions mode when out holds one with was: out without its "_<was>"
    ending, then "_<mode>" (/data/f022mix holding en, both -> /data/f022mix_both; /data/f022mix_both holding both,
    tr -> /data/f022mix_tr)."""
    base = out[:-len(was) - 1] if out.endswith(f"_{was}") else out
    return f"{base}_{mode}"


def instructions_problem(out: Path, mode: str) -> str:
    """Why a merge with --instructions mode may not go to out ("" when it may): out holds a merge made with another
    mode (prep_report.json "instructions" mode; none: en, a merge from before the option)."""
    rep = vc.read_json(out / "prep_report.json") or {}
    was = ((rep.get("instructions") or {}).get("mode") or "en") if rep.get("merged") else mode
    return "" if was == mode else (f"--out {out} holds a merge with --instructions {was}: a merge with --instructions "
                                   f"{mode} goes to a dir of its own (e.g. {merge_out_for(str(out), was, mode)}), so "
                                   f"an experiment trained on {out} never resumes on other texts")


def merge_sources(sources: list[Path], allow_incomplete: bool, allow_unaudited: bool = False
                  ) -> tuple[list[dict], list[str]]:
    """Each source's record (dir, report, manifest rows, its bytes / sha256, clip categories, its c audit: audit_of)
    and the problems that refuse the merge (module doc "Merge")."""
    recs, bad = [], []
    for src in sources:
        name = str(src)
        if not (src / MARKER).is_file():
            bad.append(f"{name}: not a render_corpus.py corpus (no {MARKER})")
            continue
        if why := render_busy(src):
            bad.append(f"{name}: {why}")
            continue
        rep = vc.read_json(src / "prep_report.json") or {}
        if rep.get("complete") is False and not allow_incomplete:
            bad.append(f"{name}: its mix is not complete ({(rep.get('selected') or {}).get('hours')} of "
                       f"{rep.get('target_hours')} h; --allow-incomplete takes it as it is"
                       + (f"; {', '.join(rep['held_by_stop'])} stopped under --min-pass: no call adds to it"
                          if rep.get("held_by_stop") else "") + ")")
        man = src / "voxcpm_train.jsonl"
        rows = vc.read_rows(man)
        if not rows:
            bad.append(f"{name}: no voxcpm_train.jsonl rows")
            continue
        paths = {p for r in rows for p in (r.get("audio"), r.get("ref_audio")) if p is not None}
        lost = sorted(str(p) for p in paths if not isinstance(p, str) or not (p.startswith("/") or os.path.isabs(p))
                      or not vc.has_wav(Path(p)))
        if lost:
            bad.append(f"{name}: {len(lost)} WAV / ref_audio paths not absolute or not there, e.g. {lost[:3]}")
        clips = [c for c in vc.read_rows(src / "clips.jsonl") if isinstance(c.get("wav"), str)]
        cats = {str(src / "wav" / c["wav"]): c.get("category") for c in clips}
        checked = {str(src / "wav" / c["wav"]) for c in clips if c.get("c_expected") is not None}
        rec = {"dir": name, "report": rep, "rows": rows, "bytes": man.stat().st_size, "sha256": sha256_file(man),
               "cats": cats, "attribution": (src / "ATTRIBUTION.txt").read_text(encoding="utf-8").splitlines()
               if (src / "ATTRIBUTION.txt").is_file() else [], **audit_of(src, rows, checked)}
        if rec["c_unchecked"] and not allow_unaudited:
            bad.append(f"{name}: {len(rec['c_unchecked'])} instructed clips with a c were never c-checked (kept "
                       f"before the gate, not in its {C_AUDIT}), e.g. {rec['c_unchecked'][:3]}: --audit-c it first "
                       "(--allow-unaudited takes them as they are)")
        recs.append(rec)
    return recs, bad


def c_unchecked(rows: list[dict], checked: set[str], audited: set[str]) -> list[str]:
    """Audio paths of the manifest rows audit_rows takes that were neither c-checked when rendered (checked: their
    clips.jsonl record has c_expected) nor are in the corpus's C_AUDIT (audited)."""
    return [r["audio"] for r in audit_rows(rows) if r["audio"] not in checked and r["audio"] not in audited]


def audit_of(src: Path, rows: list[dict], checked: set[str] | None = None) -> dict:
    """A merge source's c audit (C_AUDIT, --audit-c): {"c_failed": {audio path: clip id} of its rows with "ok": false,
    "c_audit": None without the file, else {rows, failed, bytes, sha256, unaudited}, "c_unchecked": c_unchecked of its
    manifest}."""
    p = src / C_AUDIT
    audit = vc.read_rows(p) if p.is_file() else []
    left = c_unchecked(rows, checked or set(), {x.get("wav") for x in audit})
    if not p.is_file():
        return {"c_failed": {}, "c_audit": None, "c_unchecked": left}
    failed = {x["wav"]: str(x.get("id") or Path(x["wav"]).stem) for x in audit
              if x.get("ok") is False and isinstance(x.get("wav"), str)}
    return {"c_failed": failed, "c_unchecked": left,
            "c_audit": {"rows": len(audit), "failed": len(failed), "bytes": p.stat().st_size, "sha256": sha256_file(p),
                        "unaudited": len(left)}}


def merge_corpora(out: Path, sources: list[Path], allow_incomplete: bool = False, dry: bool = False,
                  allow_unaudited: bool = False, instructions: str = "en") -> int:
    """--merge (module doc): out's manifest, report, note and attribution from the sources, the instructions in the
    language(s) of --instructions (instruction_copies). 0, or 2 when refused."""
    if len(set(sources)) != len(sources) or len(sources) < 2:
        log("--merge: give two or more different corpus dirs")
        return 2
    if any(out == s or out in s.parents or s in out.parents for s in sources):
        log(f"--out {out}: a source or inside one; the merge goes to a dir of its own")
        return 2
    if (out / MARKER).is_file() or (out / "state.json").exists() or (
            (out / "voxcpm_train.jsonl").exists() and not (vc.read_json(out / "prep_report.json") or {}).get("merged")):
        log(f"--out {out}: holds a corpus that is not a merge; give a new dir")
        return 2
    if instructions not in INSTRUCTION_MODES:
        log(f"--instructions {instructions!r}: one of {', '.join(INSTRUCTION_MODES)}")
        return 2
    if why := instructions_problem(out, instructions):
        log(why)
        return 2
    table, table_path = {}, vc.repo_path(INSTRUCTIONS_TR)
    if instructions != "en":
        try:
            table = load_instructions_tr(table_path)
        except ValueError as e:
            log(f"--instructions {instructions}: {e}")
            return 2
    recs, bad = merge_sources(sources, allow_incomplete, allow_unaudited)
    if bad:
        for b in bad:
            log(f"refused: {b}")
        return 2
    rows, audio, texts, per = [], set(), set(), []
    for rec in recs:
        mine, gone, start, c_ids = set(), collections.Counter(), len(rows), []
        for r in rec["rows"]:
            if r["audio"] in rec["c_failed"]:  # its c heard wrong (C_AUDIT): its text stays free for a later source
                gone[f"c-check failed ({C_AUDIT})"] += 1
                c_ids.append(rec["c_failed"][r["audio"]])
            elif r["audio"] in audio:
                gone["same audio"] += 1
            elif r["text"] in texts:
                gone["same text as an earlier source"] += 1
            else:
                rows.append({**{k: r[k] for k in MERGED_KEYS if k in r}, "_cat": rec["cats"].get(r["audio"], "?")})
                audio.add(r["audio"])
                mine.add(r["text"])
        texts |= mine  # a source's own repeated texts stay (approved laughter: several takes of one line)
        took = rows[start:]
        ca = rec["c_audit"]
        per.append({"dir": rec["dir"], "rows": len(rec["rows"]), "kept_rows": len(took),
                    "hours": round(sum(x["duration"] for x in took) / 3600, 3), "dropped": dict(gone),
                    "complete": rec["report"].get("complete"), "report_time": rec["report"].get("time"),
                    "manifest_bytes": rec["bytes"], "manifest_sha256": rec["sha256"],
                    "c_audit_bytes": ca["bytes"] if ca else None, "c_audit_sha256": ca["sha256"] if ca else None,
                    "c_audit": {**ca, "dropped_ids": sorted(c_ids)} if ca else None,
                    "c_unchecked": len(rec["c_unchecked"])})
        log(f"{rec['dir']}: {len(rec['rows'])} rows, {len(took)} taken, {per[-1]['hours']} h"
            + (f"; left out {dict(gone)}" if gone else ""))
        if ca:
            log(f"{rec['dir']}/{C_AUDIT}: {ca['failed']} of its {ca['rows']} audited clips fail the c-check, "
                f"{len(c_ids)} left out" + (f": {', '.join(sorted(c_ids))}" if c_ids else ""))
        if rec["c_unchecked"]:
            log(f"{rec['dir']}: {len(rec['c_unchecked'])} instructed clips with a c never c-checked, taken as they are "
                "(--allow-unaudited)")
    audio = {"clips": len(rows), "hours": round(sum(x["duration"] for x in rows) / 3600, 3)}
    rows, unknown = instruction_copies(rows, instructions, table)
    if unknown:
        log(f"refused: --instructions {instructions}: {sum(unknown.values())} instructed rows have a wording "
            f"{table_path} lacks (add its Turkish wording there): "
            + "; ".join(f"{w!r} x{n}" for w, n in sorted(unknown.items())))
        return 2
    langs = collections.Counter(x["_lang"] for x in rows)
    log(f"instructions {instructions}: {langs['plain']} rows without an instruction, {langs['en']} English, "
        f"{langs['tr']} Turkish")
    secs = collections.Counter()
    for x in rows:
        secs[x["_cat"]] += x["duration"]
    total = sum(secs.values())
    log(f"merged: {len(rows)} rows, {total / 3600:.3f} h -> {out}" + (" (dry run: nothing written)" if dry else ""))
    if dry:
        return 0
    random.Random(0).shuffle(rows)
    out.mkdir(parents=True, exist_ok=True)
    vc.write_rows(out / "voxcpm_train.jsonl", [{k: x[k] for k in MERGED_KEYS if k in x} for x in rows])
    write_text(out / "ATTRIBUTION.txt", "".join(x + "\n" for x in dict.fromkeys(
        line for rec in recs for line in rec["attribution"] if line.strip())))
    incomplete = [p["dir"] for p in per if p["complete"] is False]
    rep = {"merged": True, "complete": True, "reached": True, "target_hours": round(total / 3600, 3),
           "selected": {"hours": round(total / 3600, 3), "clips": len(rows), "voices": 1},
           "by_category": {c: {"clips": sum(x["_cat"] == c for x in rows), "hours": round(s / 3600, 3),
                               "share": round(s / max(total, 1e-9), 3)} for c, s in secs.most_common()},
           "sources": per, "allow_incomplete": allow_incomplete, "incomplete_sources": incomplete,
           "allow_unaudited": allow_unaudited,
           "manifest": {"rows": len(rows), "with_ref_audio": sum(bool(x.get("ref_audio")) for x in rows)},
           "audio": audio, "instructions": {
               "mode": instructions, "rows": {k: langs[k] for k in ("plain", "en", "tr")},
               **({"table": INSTRUCTIONS_TR.as_posix(), "table_sha256": sha256_file(table_path), "wordings": len(table)}
                  if instructions != "en" else {})},
           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_text(out / "README.md", merged_note(rep))
    write_text(out / "PREP_REPORT.md", merged_markdown(rep))
    vc.write_json(out / "prep_report.json", rep)  # last: a report newer than the manifest is a finished merge
    return 0


def instructions_line(rep: dict) -> str:
    """A merge's instruction language(s) in a sentence (README.md, PREP_REPORT.md)."""
    ins = rep.get("instructions") or {}
    mode, n, a = ins.get("mode") or "en", ins.get("rows") or {}, rep.get("audio") or {}
    how = {"en": "the instructions as rendered (English)",
           "both": "each instructed row twice, with its English instruction and with the table's Turkish wording",
           "tr": "each instructed row with the table's Turkish wording of its instruction"}.get(mode, mode)
    return (f"Instructions (`--instructions {mode}`): {how}"
            + (f" (`{ins['table']}`, sha256 {ins['table_sha256'][:12]})" if ins.get("table_sha256") else "")
            + f"; manifest rows: {n.get('plain', '?')} without an instruction, {n.get('en', '?')} English, "
            f"{n.get('tr', '?')} Turkish" + (f"; audio: {a['clips']} clips, {a['hours']} h" if a else "") + ".")


def merged_note(rep: dict) -> str:
    """README.md of a merged corpus."""
    src = "".join(f"- `{p['dir']}`: {p['kept_rows']} of its {p['rows']} manifest rows, {p['hours']} h"
                  + (f" (left out: {p['dropped']})" if p["dropped"] else "")
                  + (" **(its mix was not complete)**" if p["complete"] is False else "") + "\n"
                  for p in rep["sources"])
    return (f"# Merged training corpus ({rep['selected']['hours']} h, {rep['selected']['clips']} rows)\n\n"
            f"Written by `tts/voxcpm/render_corpus.py --merge` on {rep['time']}: `voxcpm_train.jsonl` is the union of "
            "the sources' manifests, rows unchanged (absolute WAV paths into the sources, ref_audio kept), a row "
            "whose audio an earlier row has, or whose text an earlier source has, left out, and a clip its source's "
            f"`{C_AUDIT}` (`--audit-c`) marks with a c not heard as /dʒ/; no WAV is copied, so the sources must stay "
            f"where they are.\n\n{src}\n{instructions_line(rep)}\n\n"
            "Train on it with `--action train --args \"--data <this dir> ...\"`. The train preflight refuses it once a "
            f"source's manifest or `{C_AUDIT}` changes (a later corpus-render, corpus-add-reviewed or corpus-audit-c): "
            "run `--action corpus-merge` again. Attribution: `ATTRIBUTION.txt` (the sources' lines).\n")


def merged_markdown(rep: dict) -> str:
    """PREP_REPORT.md of a merged corpus."""
    L = ["# Merged corpus report (generated by `tts/voxcpm/render_corpus.py --merge`)\n",
         f"**{rep['selected']['hours']} h, {rep['selected']['clips']} rows**, {rep['manifest']['with_ref_audio']} "
         f"with ref_audio; {rep['time']}.\n", f"{instructions_line(rep)}\n", "| category | rows | hours | share |",
         "|---|---:|---:|---:|"]
    L += [f"| {c} | {b['clips']} | {b['hours']} | {b['share']} |" for c, b in rep["by_category"].items()]
    L.append("\n| source | rows | taken | hours | left out | complete | manifest sha256 |\n"
             "|---|---:|---:|---:|---|---|---|")
    L += [f"| {p['dir']} | {p['rows']} | {p['kept_rows']} | {p['hours']} | {p['dropped'] or '-'} | {p['complete']} | "
          f"{p['manifest_sha256'][:12]} |" for p in rep["sources"]]
    for p in rep["sources"]:
        if ca := p.get("c_audit"):
            L.append(f"\n`{p['dir']}/{C_AUDIT}` (sha256 {ca['sha256'][:12]}): {ca['failed']} of {ca['rows']} audited "
                     f"clips fail the c-check, left out: {', '.join(ca['dropped_ids']) or '-'}")
        if p.get("c_unchecked"):
            L.append(f"\n`{p['dir']}`: **{p['c_unchecked']} instructed clips with a c never c-checked** "
                     "(--allow-unaudited)")
    return "\n".join(L) + "\n"


# ---- the c audit (--audit-c) --------------------------------------------------------------------------------------


def audit_rows(rows: list[dict]) -> list[dict]:
    """The manifest rows the c audit reads: an instruction in the training text and a "c" in its spoken part."""
    out = []
    for r in rows:
        ins, spoken = pc.spoken_part(r.get("text") or "")
        if ins and pc.c_expected(spoken) and isinstance(r.get("audio"), str):
            out.append(r)
    return out


def audit_c(out: Path, model: Callable | None = None, dry: bool = False) -> int:
    """--audit-c (module doc): out/C_AUDIT from phon_check's IPA tokens of each audit_rows clip of its manifest;
    nothing else written. model: () -> an object with ipa_file(path) (tests; default phon_check.get_model). 0; 2 when
    refused; 1, nothing written, when the model could not read SYSTEMIC_N and SYSTEMIC_SHARE of the clips (it or the
    setup is broken: a merge would leave out every clip)."""
    if not (out / MARKER).is_file():
        log(f"--out {out}: not a render_corpus.py corpus (no {MARKER}); audit the corpora a merge is made of")
        return 2
    if why := render_busy(out):
        log(f"refused: {why}; audit only between corpus-render calls")
        return 2
    rows = audit_rows(vc.read_rows(out / "voxcpm_train.jsonl"))
    ids = {str(out / "wav" / c["wav"]): c for c in vc.read_rows(out / "clips.jsonl") if isinstance(c.get("wav"), str)}
    secs = sum(float(r.get("duration") or 0.0) for r in rows)
    log(f"c audit of {out}: {len(rows)} manifest clips with an instruction and a c, {secs:.0f} s of audio "
        f"(~{len(rows) * PHON_CLIP_S + secs * PHON_AUDIO_S + PHON_LOAD_S:.0f} s on a GPU with the model's load)"
        + (" (dry run: nothing written)" if dry else ""))
    if dry or not rows:
        return 0
    pm, recs = (model or pc.get_model)(), []
    for r in rows:
        c = ids.get(r["audio"]) or {}
        _, spoken = pc.spoken_part(r["text"])
        rec = {"id": c.get("id") or Path(r["audio"]).stem, "wav": r["audio"], "category": c.get("category"),
               "text": r["text"], "expected": pc.c_expected(spoken)}
        try:
            toks = pm.ipa_file(r["audio"])
        except Exception as e:  # noqa: BLE001  (an unreadable clip: not trained on unchecked)
            recs.append({**rec, "got": None, "ok": False, "ipa": None, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        ok, _ = pc.c_check(spoken, toks)
        recs.append({**rec, "got": pc.dj_count(toks), "ok": ok, "ipa": pc.ipa_text(toks)})
    errs = [x for x in recs if "error" in x]
    if len(errs) >= max(SYSTEMIC_N, SYSTEMIC_SHARE * len(recs)):
        log(f"refused: the IPA model could not read {len(errs)} of {len(recs)} clips, e.g. {errs[0]['error']}: the "
            "model or the setup, not the clips; nothing written" + (f" ({out / C_AUDIT} stays as it was)"
                                                                  if (out / C_AUDIT).is_file() else ""))
        return 1
    vc.write_rows(out / C_AUDIT, recs)
    bad = [x for x in recs if not x["ok"]]
    by = collections.Counter(x["category"] for x in bad)
    log(f"{len(bad)} of {len(recs)} fail the c-check ({dict(by) or 'none'}; {len(errs)} unreadable) -> "
        f"{out / C_AUDIT}" + (f": {', '.join(x['id'] for x in bad)}" if bad else "") + "; corpus-merge leaves them out")
    for x in bad:  # what the model heard: read these before a merge (a c before e / i / ö / ü was never calibrated)
        log(f"  {x['id']}: {pc.spoken_part(x['text'])[1]!r} -> " + (x["ipa"] if x.get("ipa") is not None
                                                                    else x["error"]))
    return 0


# ---- report -------------------------------------------------------------------------------------------------------


def _q(xs: list[float]) -> dict:
    return pa.quantiles([x for x in xs if x is not None])


def build_report(state: dict, pool: list[dict], args, attempts: list[dict], refs: dict, extra: dict,
                 reviewed: list[dict] | None = None) -> dict:
    """prep_report.json; refs: the manifest's ref_audio of the kept and the human-approved clips (reviewed:
    fold_reviewed)."""
    kept = list(state["kept"].values())
    hours = round(sum(c["duration"] for c in kept) / 3600, 3)
    mix = parse_mix(args.mix)
    exhausted = exhausted_cats(pool, state)
    targets = category_targets(args.hours, mix, kept_s(state), exhausted, args.fill)
    left = {c: max(0.0, targets[c] - kept_s(state)[c]) for c in CATEGORIES}
    complete = all(v <= 0 or c in exhausted for c, v in left.items())
    stopped = stopped_cats(state, args.min_pass)
    # short of their target with only texts never tried left, under --min-pass: the same command adds nothing to them
    short = [c for c, v in left.items() if v > 0 and c not in exhausted]
    held = sorted(short) if short and all(c in stopped and not any(
        row_status(state, r["id"]) == "open" for r in pool if r["category"] == c) for c in short) else []
    by_cat = {}
    for c in CATEGORIES:
        cs = [x for x in kept if x["category"] == c]
        s = state["stats"].get(c) or {}
        statuses = collections.Counter(row_status(state, r["id"]) for r in pool if r["category"] == c)
        if not (cs or s or statuses or mix[c]):  # a category this corpus never had (long in a corpus without it)
            continue
        by_cat[c] = {"clips": len(cs), "hours": round(sum(x["duration"] for x in cs) / 3600, 3),
                     "share": round(sum(x["duration"] for x in cs) / max(hours * 3600, 1e-9), 3),
                     "mix": round(mix[c], 3), "target_hours": round(targets[c] / 3600, 3),
                     "pool": dict(statuses), "tries": s.get("decided", 0),
                     "pass_rate": round(s["passed"] / s["decided"], 3) if s.get("decided") else None,
                     "first_pass_rate": round(s["first_pass"] / s["first"], 3) if s.get("first") else None,
                     "fails": dict(sorted((s.get("fails") or {}).items(), key=lambda kv: -kv[1])),
                     "duration_s": _q([x["duration"] for x in cs])}
        if s.get("modes"):  # per render mode (--plain-render): tries made in each, the mode the category is in now
            by_cat[c].update(render=mode_of(state, c), modes={m: {
                "tries": x.get("decided", 0), "kept": x.get("kept", 0), "kept_s": x.get("kept_s", 0.0),
                "dropped": x.get("dropped", 0),
                "pass_rate": round(x["passed"] / x["decided"], 3) if x.get("decided") else None,
                "first_pass_rate": round(x["first_pass"] / x["first"], 3) if x.get("first") else None,
                "fails": dict(sorted((x.get("fails") or {}).items(), key=lambda kv: -kv[1]))}
                for m, x in s["modes"].items()})
    scored = [a for a in attempts if a.get("sim") is not None]
    sim = {"kept": _q([c.get("sim") for c in kept]), "scored": _q([a["sim"] for a in scored]),
           "by_duration": {b: _q([a["sim"] for a in scored if lo <= a["duration"] < hi])
                           for b, lo, hi in (("<2s", 0, 2), ("2-4s", 2, 4), (">=4s", 4, 1e9))},
           "min_sim": args.min_sim, "sim_short_slack": args.sim_short_slack,
           "pass_share": round(sum(a["sim"] >= a.get("sim_floor", args.min_sim) for a in scored) / len(scored), 3)
           if scored else None}
    utmos = {c: {"scored": _q([a.get("utmos") for a in attempts if a["category"] == c]),
                 "kept": _q([x.get("utmos") for x in kept if x["category"] == c])} for c in CATEGORIES}
    th = utmos_thresholds(state["first_utmos"], gates_of(args), collections.Counter(utmos_cat(r) for r in pool))
    text = [a for a in attempts if a.get("hyp") is not None]
    inton = collections.defaultdict(lambda: [0, 0])
    for a in attempts:
        if a.get("intonation"):
            g = inton[a.get("kind") or "?"]
            g[0] += 1
            g[1] += not ("intonation" in a["fails"])
    emo = collections.defaultdict(collections.Counter)
    for a in attempts:
        if a.get("emotion_verdict"):
            k = (a.get("emotion") or "?") + (" (plain)" if a.get("render") == "plain" else "")
            emo[k][a["emotion_verdict"]] += 1
    first = [a for a in attempts if a.get("t") == 0 and a.get("onset_bad") is not None]
    measured = [a for a in attempts if a.get("edges") is not None or a.get("sib_db") is not None]
    clip = {"tries_measured": len(measured),
            "fails": dict(collections.Counter(f for a in attempts for f in a.get("fails") or []
                                              if f in ("tail", "head", "stretch", "sibilance", "speed", "pause"))),
            "tails_cut": sum(1 for c in kept if c.get("tail_cut_s")),
            "sib_db": _q([a.get("sib_db") for a in measured]),
            "first_stretch": _q([a.get("first_stretch") for a in measured]),
            "rate_ws": {c: _q([a.get("rate_ws") for a in measured if a["category"] == c]) for c in CATEGORIES},
            "limits": {"stretch_max": args.stretch_max, "sib_max": args.sib_max,
                       "speed_max": {c: round(v * args.speed_scale, 2) for c, v in SPEED_MAX.items()}}}
    npool, nkept = [x for x in pool if x["category"] == "neutral"], [x for x in kept if x["category"] == "neutral"]
    nsec = sum(x["duration"] for x in nkept)
    numbers = {"target": args.number_share,
               "pool_share": round(sum(map(has_numbers, npool)) / len(npool), 3) if npool else None,
               "kept_share_s": round(sum(x["duration"] for x in nkept if has_numbers(x)) / nsec, 3) if nsec else None,
               "kept_share_clips": round(sum(map(has_numbers, nkept)) / len(nkept), 3) if nkept else None,
               "kinds_kept": dict(collections.Counter(k for x in nkept for k in (
                   x["kinds"] if isinstance(x.get("kinds"), list) else number_kinds(x.get("text") or ""))))}
    lm = [a for a in attempts if a["category"] == "long" and a.get("pause_sent") is not None]
    lkept = [x for x in kept if x["category"] == "long"]
    long_d = {"tries_measured": len(lm), "pause_sent": _q([x for a in lm for x in a["pause_sent"]]),
              "pause_mid": _q([a.get("pause_mid") for a in lm]), "pause_max": _q([a.get("pause_max") for a in lm]),
              "pause_fails": sum("pause" in (a.get("fails") or []) for a in attempts if a["category"] == "long"),
              "kept_s": _q([x["duration"] for x in lkept]), "kept_sentences": _q([x.get("sentences") for x in lkept]),
              "tries": args.long_tries or args.tries,
              "limits": {"pause_sent": [args.pause_lo, args.pause_hi], "pause_mid_max": args.pause_mid,
                         "pause_max": args.pause_max}} if "long" in by_cat else {}
    checked = [a for a in attempts if a.get("c_expected") is not None or a.get("c_error")]
    c_by = collections.defaultdict(lambda: {"checked": 0, "failed": 0})
    for a in checked:  # plain-rendered tries apart: does leaving the instruction out make the c right?
        k = a["category"] + (" (plain)" if a.get("render") == "plain" else "")
        c_by[k]["checked"] += 1
        c_by[k]["failed"] += C_FAIL in (a.get("fails") or [])
    c_d = {"on": args.c_check, "model": f"{pc.IPA_MODEL}@{pc.IPA_REV[:10]}", "tries_checked": len(checked),
           "failed": sum(C_FAIL in (a.get("fails") or []) for a in attempts),
           "errors": sum(bool(a.get("c_error")) for a in attempts), "by_category": dict(c_by),
           "kept_checked": sum(x.get("c_expected") is not None for x in kept)}
    r = state["render"]
    rendered_s, kept_total = r["audio_s"], sum(c["duration"] for c in kept)
    sc = state["scoring"]
    rep = {
        "voice": {"name": args.voice, "description": args.description, "ref": args.ref,
                  "ref_sha256": state["key"].get("ref_sha256"), "ref_prep": state["key"].get("ref_prep"),
                  "loudness": state["key"].get("loudness"), "model": state["key"].get("model"),
                  "timesteps": args.timesteps, "cfg": args.cfg, "seed": args.seed},
        "selected": {"hours": hours, "clips": len(kept), "voices": 1},
        "target_hours": args.hours, "reached": hours * 3600 >= args.hours * 3600 - 1, "complete": complete,
        "exhausted": sorted(c for c in exhausted if c in by_cat), "stopped": stopped, "held_by_stop": held,
        "by_category": by_cat, "render_modes": state.get("modes") or {},
        "gates": {**asdict(gates_of(args)), "tries": args.tries, "prerender": args.prerender,
                  "utmos_thresholds": th, "rate": [args.rate_lo, args.rate_hi], "short_chars": SHORT_CHARS,
                  "short_s": [SHORT_MIN_S, SHORT_MAX_S], "sim_full_s": SIM_FULL_S},
        "distributions": {"sim": sim, "utmos": utmos, "p_tr": _q([a.get("p_tr") for a in text]),
                          "cer0_share": round(sum(a.get("char_errors") == 0 for a in text) / len(text), 3)
                          if text else None,
                          "intonation_pass": {k: {"items": v[0], "passed": v[1]} for k, v in sorted(inton.items())},
                          "emotion_verdicts": {k: dict(v) for k, v in sorted(emo.items())},
                          "onset_first_try_stray": round(sum(bool(a["onset_bad"]) for a in first) / len(first), 3)
                          if first else None,
                          "onset_trimmed": sum(bool(a.get("onset_trimmed")) for a in attempts), "clip": clip,
                          "numbers": numbers, "long": long_d, "c_check": c_d},
        "emotion_refs": state.get("emotion_refs") or {}, "laughter": state.get("laughter") or {},
        "render": {**r, "audio_per_gpu_s": round(r["audio_s"] / r["wall_s"], 2) if r["wall_s"] else None,
                   "tries_decided": len(attempts), "rendered_per_kept_audio": round(rendered_s / kept_total, 2)
                   if kept_total else None},
        "scoring": {**sc, "s_per_clip": round(sc["wall_s"] / sc["clips"], 3) if sc["clips"] else None},
        "manifest": manifest_stats([*kept, *(reviewed or [])], [x["ref_audio"] for x in refs.values()]),
        "pool": state.get("pool", {}), "baseline": baseline(state["baseline"]),
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                     if k not in ("deadline", "dry_run")},
        "calls": state["calls"][-20:], **extra}
    return fold_reviewed(rep, kept, reviewed or [])


def gates_of(args) -> Gates:
    return Gates(args.min_sim, args.sim_short_slack, args.min_p_tr, args.rate_lo, args.rate_hi, args.utmos_drop,
                 args.calib_n, stretch_max=args.stretch_max, sib_max=args.sib_max, speed_scale=args.speed_scale,
                 pause_lo=args.pause_lo, pause_hi=args.pause_hi, pause_mid=args.pause_mid, pause_max=args.pause_max,
                 c_check=args.c_check)


def incomplete_note(rep: dict, em: str = "") -> str:
    """What an incomplete corpus's report says: the same command continues, unless only a --min-pass stop holds it
    (held_by_stop: texts never tried, no call adds to them)."""
    if not rep.get("held_by_stop"):
        return f"{em}incomplete: the same corpus-render command continues{em}"
    return (f"{em}incomplete: {', '.join(rep['held_by_stop'])} stopped under --min-pass with texts never tried, the "
            "same command adds nothing (--min-pass 0 goes on; corpus-merge --allow-incomplete takes the kept clips)"
            f"{em}")


def render_markdown(rep: dict) -> str:
    L = []
    a = L.append
    v, sel, g = rep["voice"], rep["selected"], rep["gates"]
    a("# f022 corpus report (generated by `tts/voxcpm/render_corpus.py`)\n")
    a(f"Voice `{v['name']}`: {v['description']}  \nReference `{v['ref']}` (sha256 {str(v['ref_sha256'])[:12]}, "
      f"{v['ref_prep']}), base VoxCPM2, {v['timesteps']} timesteps, cfg {v['cfg']}, seed {v['seed']}.\n")
    status = "reached" if rep["reached"] else "**not reached**"
    a(f"**{sel['hours']} h, {sel['clips']} clips** (target {rep['target_hours']} h: {status}; "
      f"{'complete' if rep['complete'] else incomplete_note(rep, '**')}"
      f"{'; texts used up: ' + ', '.join(rep['exhausted']) if rep['exhausted'] else ''}"
      + "".join(f"; **{c} stopped: {m} of its tries pass, under --min-pass {rep['settings'].get('min_pass')}**"
                for c, m in (rep.get("stopped") or {}).items()) + ").\n")
    a("## Mix (kept hours)\n\n| category | clips | hours | share | mix | target h | tries | pass/try | first try | "
      "pool (new/open/kept/dropped) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for c, b in rep["by_category"].items():
        p = b["pool"]
        a(f"| {c} | {b['clips']} | {b['hours']} | {b['share']} | {b['mix']} | {b['target_hours']} | {b['tries']} | "
          f"{b['pass_rate']} | {b['first_pass_rate']} | {p.get('new', 0)}/{p.get('open', 0)}/{p.get('kept', 0)}/"
          f"{p.get('dropped', 0)} |")
    a("\n## Failed gates (tries; a try can fail several)\n")
    for c, b in rep["by_category"].items():
        if b["fails"]:
            a(f"- {c}: " + ", ".join(f"{k} {n}" for k, n in b["fails"].items()))
    if modes := {c: b for c, b in rep["by_category"].items() if b.get("modes")}:
        a("\n## Render modes (--plain-render)\n\nPlain: the model text without the instruction, the emotion's "
          "reference clip carries the emotion, the manifest keeps \"(instruction)\" + text; a plain try's prosody must "
          f"show the emotion (verdict \"expected\", else gate {PLAIN_FAIL}). Pass rate, --min-pass stop and tries a "
          "text gets count in the mode the category is in now.\n\n| category | mode | tries | kept | kept s | dropped "
          "| pass/try | first try | failed gates |\n|---|---|---:|---:|---:|---:|---:|---:|---|")
        for c, b in modes.items():
            rec = (rep.get("render_modes") or {}).get(c) or {}
            for m, x in b["modes"].items():
                now = f" (now, since call {rec.get('call')})" if m == b.get("render") else ""
                a(f"| {c} | {m}{now} | {x['tries']} | {x['kept']} | {x['kept_s']} | {x['dropped']} | {x['pass_rate']} "
                  f"| {x['first_pass_rate']} | " + ", ".join(f"{k} {n}" for k, n in x["fails"].items()) + " |")
    d = rep["distributions"]
    a(f"\n## Gates and distributions\n\nGates: CER 0, p(tr) >= {g['min_p_tr']}, ECAPA sim >= {g['min_sim']} from "
      f"{g['sim_full_s']} s ({g['sim_short_slack']} less at 1 s), UTMOS bottom {g['utmos_drop']} per category dropped "
      f"(thresholds {g['utmos_thresholds']}), {g['rate'][0]}-{g['rate'][1]} chars/s, {g['tries']} tries.\n")
    a(f"- sim: kept {d['sim']['kept']}  \n  scored {d['sim']['scored']} (pass share {d['sim']['pass_share']})  \n"
      + "".join(f"  {k}: {q}  \n" for k, q in d["sim"]["by_duration"].items()))
    a(f"- p(tr) of the tries Whisper heard: {d['p_tr']}; CER 0 share {d['cer0_share']}")
    a("- UTMOS kept: " + "; ".join(f"{c} median {u['kept'].get('median')}" for c, u in d["utmos"].items()
                                   if u["kept"]))
    a(f"- intonation (tries scored / passed): {d['intonation_pass']}")
    a(f"- emotion verdicts: {d['emotion_verdicts']}")
    a(f"- stray onsets at a first try: {d['onset_first_try_stray']}; cut by NanoPool: {d['onset_trimmed']}")
    c = d["clip"]
    a(f"- clip gates (pilot): fails {c['fails']}; stray tails cut {c['tails_cut']}; limits {c['limits']}  \n"
      f"  sibilance (dB re loud frames) {c['sib_db']}  \n  first-word stretch {c['first_stretch']}  \n"
      f"  syllables / s of speech: " + "; ".join(f"{k} {q}" for k, q in c["rate_ws"].items() if q))
    if (nb := d.get("numbers") or {}).get("pool_share") is not None:
        a(f"- written numbers in neutral rows (target >= {nb['target']}): pool {nb['pool_share']}, kept "
          f"{nb['kept_share_s']} of the seconds ({nb['kept_share_clips']} of the clips); kinds kept {nb['kinds_kept']}")
    if lg := d.get("long"):
        a(f"- long clips ({lg['tries']} tries a text; limits {lg['limits']}): pause at a sentence boundary "
          f"{lg['pause_sent']}  \n  longest inside a sentence {lg['pause_mid']}  \n  longest {lg['pause_max']}; pause "
          f"gate failed {lg['pause_fails']} tries  \n  kept: seconds {lg['kept_s']}, sentences {lg['kept_sentences']}")
    if (cd := d.get("c_check") or {}).get("tries_checked") or cd.get("on") is False:
        a(f"- c-check (instructed rows with a c heard as /dʒ/, {cd.get('model')}): "
          + (f"{cd['tries_checked']} tries checked, {cd['failed']} failed ({cd['errors']} unreadable); by category "
             f"{cd['by_category']}" if cd.get("on") else "off (--no-c-check)"))
    refs = rep["emotion_refs"]
    a("- emotion references: " + ("; ".join(
        f"{e} " + (f"{x['wav']} (seed {x['seed']}, sim {x['sim']}, score {x['score']}, delta {x['delta']})"
                   if x.get("wav") else f"none ({x.get('why')})") for e, x in sorted(refs.items())) or "none"))
    a(f"- laughter trial (not in the manifest by itself; {REVIEW}/): {rep['laughter'] or 'not run'}")
    if rv := rep.get("reviewed"):
        a(f"- human-approved clips in the manifest ({rv['file']}, category {rv['category']}): {rv['clips']} clips, "
          f"{rv['seconds']} s ({rv['best']} marked best)")
    r, s, m = rep["render"], rep["scoring"], rep["manifest"]
    a(f"\n## Throughput\n\nSynthesis: {r['jobs']} tries asked, {r['done']} made, {r['audio_s']} s of audio in "
      f"{r['wall_s']} s ({r['audio_per_gpu_s']} audio s per GPU s), {r['waves']} waves; rendered / kept audio "
      f"{r['rendered_per_kept_audio']}.  \nScoring: {s['clips']} tries in {s['wall_s']} s "
      f"({s['s_per_clip']} s each).\n")
    a(f"Manifest: {m['rows']} rows, {m['with_ref_audio']} with ref_audio ({m['ref_is_reference']} the reference), "
      f"{m['with_instruction']} with an instruction.\n")
    p = rep["pool"]
    a(f"Pool: {p}\n")
    return "\n".join(L) + "\n"


def attribution_lines(kept: list[dict], voice: str, description: str) -> list[str]:
    out = [f"Synthetic speech (AI-generated) in one designed voice, {voice}: VoxCPM2 (OpenBMB, Apache-2.0) voice "
           f"design, \"{description}\" (no real person's voice), cloned by reference with base VoxCPM2."]
    if any(c.get("source") == DOMAIN_REPO for c in kept):
        joined = " and long passages (1-3 consecutive lines joined)" if any(c.get("category") == "long"
                                                                             for c in kept) else ""
        out.append(f"Neutral sentences{joined}: {DOMAIN_ATTRIBUTION}; read in normalised form (trnorm).")
    if any(c.get("source") == "textbank" for c in kept):
        out.append("Questions, emotional lines, short lines and interjections: our own text bank "
                   "(tts/voxcpm/textbank/textbank.jsonl).")
    if any(c.get("source") == CWORDS_SOURCE for c in kept):
        out.append("Emotional lines with a word-initial c: our own text bank (tts/voxcpm/textbank/cwords.jsonl).")
    if any(c.get("source") == REVIEWED_SOURCE for c in kept):
        out.append("Laughter lines: our own text bank (tts/voxcpm/textbank/laughter.jsonl), each clip approved by a "
                   "listener (tts/voxcpm/textbank/laughter_approved.json).")
    return out


# ---- main ---------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="corpus directory (e.g. /data/f022corpus)")
    ap.add_argument("--hours", type=float, default=5.0, help="kept hours to stop at [%(default)s]")
    ap.add_argument("--mix", default=",".join(f"{k}={v}" for k, v in MIX.items() if v),
                    help=f"kept-hour shares of {', '.join(CATEGORIES)} [%(default)s]")
    ap.add_argument("--fill", choices=[*CATEGORIES, "none"], default=FILL,
                    help="what a used-up category lacks goes here [%(default)s]")
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--description", default=DESCRIPTION, help="the voice's design description (report, attribution)")
    ap.add_argument("--ref", default=REF, help="the voice's reference clip [%(default)s]")
    ap.add_argument("--model", default=vc.MODEL_DIR, help="VoxCPM2 snapshot directory")
    ap.add_argument("--textbank", default=str(TEXTBANK), help="relative = under the repo [%(default)s]")
    ap.add_argument("--exclude-manifest", type=Path, nargs="+", action="extend", default=[],
                    help="jsonl files whose texts must never be trained on (Antalia's voxcpm_val.jsonl); "
                         "benchmarks/tts/sets/*.jsonl always")
    ap.add_argument("--domain-lines", type=int, default=DOMAIN_LINES, help="neutral sentences in the pool")
    ap.add_argument("--domain-scan", type=int, default=DOMAIN_SCAN, help="domain-text rows read to draw them from")
    ap.add_argument("--domain-weights", default=",".join(f"{k}:{v}" for k, v in DOMAIN_WEIGHTS.items()))
    ap.add_argument("--rich-share", type=float, default=RICH_SHARE, help="of each category, lines with digits or "
                    "acronyms [%(default)s]")
    ap.add_argument("--number-share", type=float, default=NUMBER_SHARE, help="neutral rows with written numbers, at "
                    "least, of the draw and of the kept seconds when the pool allows; 0 = no weighting [%(default)s]")
    ap.add_argument("--long-lines", type=int, default=LONG_LINES, help="long items in the pool (only with a long "
                    "share in --mix) [%(default)s]")
    ap.add_argument("--long-chars", default=f"{LONG_CHARS[0]},{LONG_CHARS[1]}", help="a long item's characters of "
                    "spoken form [%(default)s]")
    ap.add_argument("--long-target", default=f"{LONG_TARGET[0]},{LONG_TARGET[1]}", help="characters of spoken form "
                    "the long items are drawn from first (each run's window nearest its middle), within --long-chars "
                    "[%(default)s]")
    ap.add_argument("--long-weights", default=",".join(f"{k}:{v}" for k, v in LONG_WEIGHTS.items()),
                    help="dataset categories of the long items")
    ap.add_argument("--long-tries", type=int, default=LONG_TRIES, help="tries per long text [%(default)s]")
    ap.add_argument("--cwords", default=str(CWORDS), help="emotion rows with a word-initial c (category cwords, only "
                    "with a cwords share in --mix); relative = under the repo [%(default)s]")
    ap.add_argument("--cwords-tries", type=int, default=CWORDS_TRIES, help="tries per cwords text [%(default)s]")
    ap.add_argument("--no-c-check", dest="c_check", action="store_false", help="no c-check gate (phon_check: every "
                    "c of an instructed row heard as /dʒ/)")
    ap.add_argument("--plain-render", metavar="CATS", help=f"categories ({', '.join(PLAIN_CATS)}) whose rows the model "
                    "reads without their instruction (module doc \"Plain render\"); 'none': all with it. Kept in "
                    "state.json: a later call gives it again (refused without it)")
    ap.add_argument("--pause-sent", default=f"{PAUSE_SENT[0]:g},{PAUSE_SENT[1]:g}", help="long clips: s of silence at "
                    "each internal sentence boundary [%(default)s]")
    ap.add_argument("--pause-mid-max", type=float, default=PAUSE_MID_MAX, help="long clips: longest pause inside a "
                    "sentence, s [%(default)s]")
    ap.add_argument("--pause-max", type=float, default=PAUSE_MAX, help="long clips: longest pause, s [%(default)s]")
    ap.add_argument("--avoid-pool", type=Path, nargs="+", action="extend", default=[], metavar="DIR",
                    help="corpus dirs whose texts and domain lines (texts.jsonl) this pool leaves out (only read)")
    ap.add_argument("--refs-from", type=Path, metavar="DIR", help="a corpus dir whose voice reference (ref/) and "
                    "emotion references (refs/, state.json) this corpus reuses (only read)")
    ap.add_argument("--no-domain", action="store_true", help="no domain text (an offline pool of the textbank only)")
    ap.add_argument("--tries", type=int, default=TRIES, help="tries per text (interjections + extra variants)")
    ap.add_argument("--prerender", type=int, default=PRERENDER, help="tries rendered per text and wave")
    ap.add_argument("--min-pass", type=float, default=MIN_PASS, help="a category whose measured pass rate per try is "
                    "under this takes no new texts (0 = never stop) [%(default)s]")
    ap.add_argument("--min-sim", type=float, default=MIN_SIM, help=f"ECAPA cosine to the reference, clips >= "
                    f"{SIM_FULL_S:g} s [%(default)s]")
    ap.add_argument("--sim-short-slack", type=float, default=SIM_SHORT_SLACK, help="less for a 1 s clip [%(default)s]")
    ap.add_argument("--min-p-tr", type=float, default=vc.P_TR_OK)
    ap.add_argument("--rate", default=f"{RATE[0]:g},{RATE[1]:g}", help="characters per second [%(default)s]")
    ap.add_argument("--utmos-drop", type=float, default=UTMOS_DROP, help="bottom share per category [%(default)s]")
    ap.add_argument("--calib-n", type=int, default=CALIB_N, help="first tries before a UTMOS threshold [%(default)s]")
    ap.add_argument("--stretch-max", type=float, default=STRETCH_MAX, help="first word's s / syllable over the "
                    "other words' median [%(default)s]")
    ap.add_argument("--sib-max", type=float, default=onset.SIB_MAX_DB, help="loudest sustained sibilant, dB re the "
                    "loud frames [%(default)s]")
    ap.add_argument("--speed-scale", type=float, default=1.0, help="x the syllables / s limits of SPEED_MAX")
    ap.add_argument("--no-emotion-refs", dest="emotion_refs", action="store_false",
                    help="emotion rows clone the voice's reference (no refs/<emotion>.wav)")
    ap.add_argument("--emo-ref-n", type=int, default=EMO_REF_N, help="tries per emotion reference [%(default)s]")
    ap.add_argument("--laughter-usd", type=float, default=LAUGH_USD, help="the laughter trial's GPU budget, once; "
                    "0 = none [%(default)s]")
    ap.add_argument("--laughter-bank", default=str(LAUGHTER), help="relative = under the repo [%(default)s]")
    ap.add_argument("--laugh-model", choices=("auto", "none"), default="auto",
                    help=f"auto: {AST_MODEL} when it is already in the HF cache; none: the burst detector only")
    ap.add_argument("--ref-fraction", type=float, default=REF_FRACTION, help="manifest rows with ref_audio")
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0, help="base seed (per text: + crc32, per try: + 1000 x try)")
    ap.add_argument("--backend", choices=vc.BACKENDS, default="nanovllm")
    ap.add_argument("--concurrency", type=int, default=vc.NANO_CONCURRENCY, choices=range(1, 257), metavar="1-256")
    ap.add_argument("--workers", type=int, default=2, choices=range(1, vc.MAX_WORKERS + 1), metavar="1-8",
                    help="--backend voxcpm: VoxCPM2 processes")
    ap.add_argument("--load-parallel", type=int, default=1, choices=range(1, vc.MAX_LOAD_PARALLEL + 1), metavar="1-4")
    ap.add_argument("--device", default="auto", help="scorers: cuda, cpu or auto")
    ap.add_argument("--cpu-procs", type=int, default=6, help="processes of the CPU pass")
    ap.add_argument("--max-wave-rows", type=int, default=0, help="texts per wave at most; 0 = no cap")
    ap.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; 0 = none")
    ap.add_argument("--throughput-log", default=vc.THROUGHPUT_LOG, help="throughput.jsonl ('' = none)")
    ap.add_argument("--add-reviewed", type=Path, metavar="APPROVED_JSON",
                    help=f"no render: add the approved {REVIEW}/ clips ({{\"<review id>\": true}}, e.g. "
                         f"{LAUGH_APPROVED}) to --out's manifest (relative = under the repo); only between "
                         "corpus-render calls")
    ap.add_argument("--merge", type=Path, nargs="+", metavar="SRC", help="no render: --out = the union of these "
                    "finished corpora's manifests (module doc \"Merge\")")
    ap.add_argument("--allow-incomplete", action="store_true", help="--merge: take a source whose mix is not complete")
    ap.add_argument("--allow-unaudited", action="store_true", help=f"--merge: take a source whose instructed clips "
                    f"with a c were never c-checked (kept before the gate, not in its {C_AUDIT})")
    ap.add_argument("--instructions", choices=INSTRUCTION_MODES, help=f"--merge: the training texts' instructions: en "
                    f"as rendered (default), both: + a copy of each instructed row with the Turkish wording of "
                    f"{INSTRUCTIONS_TR}, tr: the Turkish copy only (module doc \"Merge\")")
    ap.add_argument("--audit-c", action="store_true", help=f"no render: --out/{C_AUDIT}, the c-check of the manifest's "
                    "instructed clips with a c (module doc \"c audit\"); nothing else written")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    return ap


def parse_args(argv=None) -> argparse.Namespace:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        parse_mix(args.mix)
        parse_weights(args.domain_weights)
        parse_weights(args.long_weights, "--long-weights")
        args.rate_lo, args.rate_hi = (float(x) for x in args.rate.split(","))
        args.long_lo, args.long_hi = parse_pair(args.long_chars, "--long-chars")
        args.target_lo, args.target_hi = parse_pair(args.long_target, "--long-target")
        args.pause_lo, args.pause_hi = parse_pair(args.pause_sent, "--pause-sent")
        if none := [c for c in parse_plain(args.plain_render) or () if parse_mix(args.mix)[c] <= 0]:
            raise ValueError(f"--plain-render {args.plain_render}: {', '.join(sorted(none))} has no share in --mix")
    except ValueError as e:
        ap.error(str(e))
    if args.instructions and not args.merge:
        ap.error("--instructions goes with --merge (the language of a merged manifest's instructions)")
    args.pause_mid = args.pause_mid_max
    for name, ok in (("--hours", args.hours > 0), ("--tries", args.tries >= 1), ("--prerender", args.prerender >= 1),
                     ("--ref-fraction", 0 <= args.ref_fraction <= 1), ("--rich-share", 0 <= args.rich_share <= 1),
                     ("--min-sim", -1 <= args.min_sim <= 1), ("--utmos-drop", 0 <= args.utmos_drop < 1),
                     ("--rate", 0 < args.rate_lo < args.rate_hi), ("--domain-lines", args.domain_lines >= 0),
                     ("--speed-scale", args.speed_scale > 0), ("--emo-ref-n", args.emo_ref_n >= 1),
                     ("--laughter-usd", 0 <= args.laughter_usd <= 2),
                     ("--number-share", 0 <= args.number_share <= 1), ("--long-lines", args.long_lines >= 0),
                     ("--long-tries", args.long_tries >= 1), ("--long-chars", args.long_lo >= MIN_CHARS),
                     ("--cwords-tries", args.cwords_tries >= 1),
                     ("--long-target", args.long_lo <= args.target_lo <= args.target_hi <= args.long_hi),
                     ("--min-pass", 0 <= args.min_pass < 1),
                     ("--pause-mid-max", args.pause_mid > 0), ("--pause-max", args.pause_max >= args.pause_hi)):
        if not ok:
            ap.error(f"bad {name}")
    args.optimize = False  # voice_contest.synthesize's SynthPool option
    return args


def default_render(args, status: vc.Status) -> Callable:
    """voice_contest.synthesize: Nano-vLLM (one server, stopped before it returns) or the official API's pool (closed
    here, so the scorers find the GPU free); throughput lines in --throughput-log."""
    def render(jobs, feed_until, on_done, on_fail) -> dict:
        pool, stats = vc.synthesize(args, jobs, status, "corpus", None, feed_until, on_done, on_fail)
        if pool is not None:
            pool.close()
        return stats

    return render


def pool_from_files(args, out: Path, guard_files: list[Path], domain: Callable | None) -> tuple[list[dict], dict]:
    """The pool (texts.jsonl, grown) and its record (textbank and cwords problems, guard, what was added)."""
    tb_rows, tb_bad, tb_sha = load_textbank(vc.repo_path(args.textbank))
    cw_rows, cw_bad, cw_sha = cwords_wanted(args)
    guard, files = make_guard(vc.repo_path(SETS_DIR), guard_files)
    old = vc.read_rows(out / "texts.jsonl")
    ids, keys, avoided = avoid_of([p for p in args.avoid_pool if (p / "texts.jsonl").is_file()])
    pool, rec = build_pool(old, tb_rows, Admit(guard), args.domain_lines, parse_weights(args.domain_weights),
                           args.rich_share, args.seed, None if args.no_domain else domain,
                           long_lines=long_wanted(args), long_weights=parse_weights(args.long_weights),
                           long_chars=(args.long_lo, args.long_hi), long_target=(args.target_lo, args.target_hi),
                           number_share=args.number_share,
                           avoid=(ids, keys), cwords=cw_rows)
    was = {o["id"]: o["spoken"] for o in old}
    changed = sum(1 for r in tb_rows if r["id"] in was and was[r["id"]] != r["spoken"])
    rec.update(textbank=str(args.textbank), textbank_sha256=tb_sha, textbank_rows=len(tb_rows),
               textbank_problems=dict(tb_bad), textbank_changed_rows_kept_old=changed, guard_files=files,
               guard_keys=len(guard.exact), by_category=dict(collections.Counter(r["category"] for r in pool)),
               **({"avoided": avoided} if args.avoid_pool else {}),
               **({"cwords": str(args.cwords), "cwords_sha256": cw_sha, "cwords_rows": len(cw_rows),
                   "cwords_problems": dict(cw_bad)} if parse_mix(args.mix)["cwords"] > 0 else {}))
    return pool, rec


def long_wanted(args) -> int:
    """Long items the pool should hold: --long-lines when --mix gives long a share, else 0 (none drawn)."""
    return args.long_lines if parse_mix(args.mix)["long"] > 0 else 0


def cwords_wanted(args) -> tuple[list[dict], collections.Counter, str]:
    """load_cwords of --cwords when --mix gives cwords a share, else ([], {}, "") (none in the pool)."""
    if parse_mix(args.mix)["cwords"] <= 0:
        return [], collections.Counter(), ""
    return load_cwords(vc.repo_path(args.cwords))


def plan_pool(pool: list[dict], textbank: list[dict], args, cwords: list[dict] | None = None) -> list[dict]:
    """The pool an estimate counts: texts.jsonl's rows (pool) or, before the first call, the textbank's, and the cwords
    rows not in it yet (unguarded: an estimate); plus a placeholder for each neutral sentence still to be drawn
    (--domain-lines; DOMAIN_CHARS characters each) and each long item (long_wanted; LONG_PLAN_CHARS characters,
    LONG_PLAN_SENT sentences)."""
    rows = list(pool or textbank)
    have = {r["id"] for r in rows}
    rows += [r for r in cwords or [] if r["id"] not in have]
    n = 0 if args.no_domain else max(0, args.domain_lines - sum(r["category"] == "neutral" for r in rows))
    n_long = 0 if args.no_domain else max(0, long_wanted(args) - sum(r["category"] == "long" for r in rows))
    return (rows + [{"id": f"d-plan-{i}", "category": "neutral", "spoken": "x" * DOMAIN_CHARS, "text": ""}
                    for i in range(n)]
            + [{"id": f"L-plan-{i}", "category": "long", "spoken": "x" * LONG_PLAN_CHARS, "text": "",
                "sentences": LONG_PLAN_SENT} for i in range(n_long)])


def sample_clips(clips: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n kept clips drawn at random (seeded), each category in proportion to its clips and at least one of each."""
    rng, by = random.Random(seed), collections.defaultdict(list)
    for c in sorted(clips, key=lambda c: c["id"]):
        by[c["category"]].append(c)
    n = min(n, len(clips))
    want = {k: max(1, round(n * len(v) / max(len(clips), 1))) for k, v in by.items()}
    while sum(want.values()) > n:  # rounding: take from the largest
        want[max(want, key=lambda k: want[k])] -= 1
    out = [c for k in sorted(by) for c in rng.sample(by[k], min(want[k], len(by[k])))]
    return sorted(out, key=lambda c: (c["category"], c["id"]))


def dry_run(args, out: Path) -> int:
    """The plan from what is here: the textbank, an earlier pool and state in --out; the domain text is counted as
    --domain-lines sentences (not fetched)."""
    missing = [str(p) for p in args.exclude_manifest if not p.is_file()]
    pool, rec = pool_from_files(args, out, [p for p in args.exclude_manifest if p.is_file()], None)
    state = vc.read_json(out / "state.json") or new_state({})
    lent = None  # what --refs-from lends (the run takes it, reuse_emotion_refs): no prelude for it, as modal_app plans
    if args.refs_from and not state.get("emotion_refs"):
        lent = lent_refs(Path(args.refs_from))
        state = {**state, "emotion_refs": lent}
    if why := mode_problem(state, args):
        log(f"refused: {why}")
        return 2
    planned = plan_pool(pool, [], args)
    cost = make_cost(args.backend, args.timesteps, args.workers, vc.read_measured(args.throughput_log))
    laugh, _ = load_laughter(vc.repo_path(args.laughter_bank))
    est = estimate(planned, state, args, cost, prelude_jobs(out, state, args, planned, laugh, cost, args.ref)[1])
    pool = with_render(pool, apply_modes(state, pool, args)[0])  # the render texts below as this call makes them
    n_long = sum(r["id"].startswith("L-plan-") for r in planned)
    log(f"dry run: --out {out}, voice {args.voice}, reference {args.ref}")
    log(f"pool: {rec['by_category']} + {len(planned) - len(pool) - n_long} neutral sentences to draw from "
        f"{DOMAIN_REPO}@{DOMAIN_REV[:10]} (~{DOMAIN_CHARS} characters each in this plan)"
        + (f" + {n_long} long items (~{LONG_PLAN_CHARS} characters, {LONG_PLAN_SENT} sentences each)" if n_long
           else "") + f"; textbank {rec['textbank']} ({rec['textbank_rows']} "
        f"rows; problems {rec['textbank_problems'] or 'none'}); leakage guard: {len(rec['guard_files'])} files, "
        f"{rec['guard_keys']} sentences, dropped {rec['textbank_dropped'] or 'none'}"
        + (f"; --exclude-manifest not here (the container has it): {missing}" if missing else ""))
    if "cwords" in rec:
        log(f"cwords: {rec['cwords']} ({rec['cwords_rows']} rows; problems {rec['cwords_problems'] or 'none'}; "
            f"added {rec['cwords_added']}, dropped {rec['cwords_dropped'] or 'none'}); c-check "
            + ("on" if args.c_check else "OFF (--no-c-check)"))
    if args.avoid_pool or args.refs_from:
        gone = [str(p) for p in args.avoid_pool if not (p / "texts.jsonl").is_file()]
        log(f"--avoid-pool: {rec.get('avoided', {})}" + (f"; not here (the container has them): {gone}" if gone
                                                         else "")
            + (f"; --refs-from {args.refs_from}: " + ("here" if (args.refs_from / "ref").is_dir()
                                                       else "not here (the container has it)")
               if args.refs_from else ""))
    if lent is not None:
        log(f"--refs-from {args.refs_from}: emotion references {', '.join(sorted(lent))} lent, no prelude for them"
            if lent else f"--refs-from {args.refs_from}: no state.json / prep_report.json with its emotion references "
            "here, so the prelude below counts them all; the run borrows those it has (modal_app's plan reads them)")
    for line in estimate_lines(est):
        log(line)
    firsts = {}
    for r in pool:
        firsts.setdefault(r["category"], r)
    for c, r in firsts.items():
        log(f"  {c}/{r['id']}: train {train_text(r)!r}; render {render_text(r, 0)!r}")
    return 0


def run(args, render: Callable | None = None, scoring: Callable[[], Scoring] | None = None,
        domain: Callable | None = None) -> int:
    """The call (module doc). render / scoring / domain: stand-ins (tests); default the engines, the scorers and the
    Hub."""
    t_start, out = time.time(), Path(os.path.abspath(args.out))  # not resolve(): a Volume's mount path stays /data
    if args.audit_c:
        return audit_c(out, dry=args.dry_run)
    if args.add_reviewed:
        return add_reviewed(out, vc.repo_path(args.add_reviewed), args.dry_run)
    if args.merge:
        return merge_corpora(out, [Path(os.path.abspath(p)) for p in args.merge], args.allow_incomplete,
                             args.dry_run, args.allow_unaudited, args.instructions or "en")
    if args.dry_run:
        return dry_run(args, out)
    if missing := [str(p) for p in args.exclude_manifest if not p.is_file()]:
        log(f"--exclude-manifest not found: {missing} (the leakage guard needs them)")
        return 2
    if foreign := foreign_files(out):
        log(f"--out {out} holds {', '.join(foreign)} but no {MARKER}: another corpus; use an empty or new directory")
        return 2
    tb, mix = vc.repo_path(args.textbank), parse_mix(args.mix)
    if not tb.is_file() and (want := [c for c in TB_CATEGORIES if mix[c] > 0]):
        log(f"no textbank at {tb} (build_textbank.py) but --mix wants {', '.join(want)}: build it first, or --mix "
            "neutral=1")
        return 2
    if mix["cwords"] > 0 and not (cw := cwords_wanted(args))[0]:
        log(f"--mix gives cwords a share but {args.cwords} has no usable rows ({dict(cw[1])})")
        return 2
    others = [Path(os.path.abspath(p)) for p in args.avoid_pool] + ([Path(os.path.abspath(args.refs_from))]
                                                                     if args.refs_from else [])
    if out in others:
        log("--avoid-pool / --refs-from name --out itself: give the other corpus")
        return 2
    if gone := [str(p) for p in args.avoid_pool if not (p / "texts.jsonl").is_file()]:
        log(f"--avoid-pool: no texts.jsonl in {gone}")
        return 2
    ref_in = Path(args.ref)
    src = Path(os.path.abspath(args.refs_from)) if args.refs_from else None
    if src is not None and (why := refs_problem(src, args.voice, ref_in)):
        log(f"--refs-from {src}: {why}")
        return 2
    old = vc.read_json(out / "state.json")
    raw = out / "ref" / f"{args.voice}.raw.wav"
    src_raw = src / "ref" / f"{args.voice}.raw.wav" if src is not None else None
    if not vc.has_wav(raw) and not ref_in.is_file() and src_raw is None:
        log(f"--ref {ref_in}: not found")
        return 2
    sha = sha256_file(ref_in if ref_in.is_file() else raw if vc.has_wav(raw) else src_raw)
    key = {"voice": args.voice, "ref_sha256": sha, "model": Path(args.model).name, "timesteps": args.timesteps,
           "cfg": args.cfg, "seed": args.seed, "ref_prep": vc.REF_PREP, "loudness": vc.LOUDNESS}
    if old and (diff := key_conflicts(old.get("key"), key)):
        log(f"{out} was rendered otherwise ({diff}): use another --out")
        return 2
    if old and (why := mode_problem(old, args)):  # --plain-render: a switch is given, never implied
        log(f"refused: {why}")
        return 2
    (out / "wav").mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text("written by tts/voxcpm/render_corpus.py\n", encoding="utf-8")
    copied = reuse_ref_files(src, out, args.voice) if src is not None else []
    if not vc.has_wav(raw):
        raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref_in, raw)
    prepared = out / "ref" / f"{args.voice}.wav"
    ref_dur = vc.wav_duration(prepared) if vc.has_wav(prepared) else prep_reference(raw, prepared)
    state = old or new_state(key)
    if src is not None:
        emos = reuse_emotion_refs(src, out, state)
        state["refs_from"] = {"dir": str(src), "files": copied, "emotion_refs": emos}
        log(f"--refs-from {src}: copied {copied or 'nothing (here already)'}; emotion references {emos or 'none new'}")
    status = vc.Status(out / "status.json", "corpus")
    status.set(phase="pool", out=str(out))
    domain = domain or (lambda: fetch_domain(DomainSource(), args.domain_scan))
    if not heldout_files(list(args.exclude_manifest)):
        log(f"warning: no Antalia held-out sentences here ({', '.join(map(str, HELDOUT))}, --exclude-manifest): "
            "the leakage guard covers the benchmark sets only")
    pool, rec = pool_from_files(args, out, list(args.exclude_manifest), domain)
    vc.write_rows(out / "texts.jsonl", pool)
    prev = state.get("pool") or {}
    if not rec["domain_read"] and prev.get("domain_read"):  # this call drew none: the first call's record stays
        rec.update(domain_read=prev["domain_read"], domain_dropped=prev.get("domain_dropped"),
                   long_dropped=prev.get("long_dropped") or {})
    state["pool"] = rec
    log(f"pool: {rec['by_category']} (added: textbank {rec['textbank_added']}, domain {rec['domain_added']}, long "
        f"{rec['long_added']}; dropped: {rec['textbank_dropped']} {rec['domain_dropped']} {rec['long_dropped']})")
    state, switches = apply_modes(state, pool, args, vc.now_iso())  # --plain-render (before reconcile: see switch_mode)
    for sw in switches:
        log(switch_line(sw, max_tries({"category": sw["category"]}, args.tries, args.long_tries, args.cwords_tries)))
    pool = with_render(pool, state)  # texts.jsonl above has no mark
    cost = make_cost(args.backend, args.timesteps, args.workers, vc.read_measured(args.throughput_log))
    r = Run(args, out, pool, state, gates_of(args), cost, raw, render or default_render(args, status),
            scoring or (lambda: Scoring(str(raw), args.device, args.cpu_procs)), status)
    fixed = r.reconcile()
    if fixed:
        log(f"resume: {fixed}")
    r.save()
    laugh, laugh_bad = load_laughter(vc.repo_path(args.laughter_bank),
                                     make_guard(vc.repo_path(SETS_DIR), list(args.exclude_manifest))[0])
    if laugh_bad:
        log(f"laughter lines left out: {dict(laugh_bad)}")
    for line in estimate_lines(estimate(pool, state, args, cost, prelude_jobs(out, state, args, pool, laugh, cost,
                                                                               str(raw))[1])):
        log(line)
    waves, err = [], ""
    status.set(phase="prelude")
    try:
        r.prelude(laugh)
        status.set(phase="waves")
        waves = r.loop()
    except Exception as e:  # noqa: BLE001  (a scorer that does not load, ...): what is decided stays, report written
        err = f"{type(e).__name__}: {e}"[:500]
        status.error("waves", err)
    state["calls"].append({"start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)), "s": round(
        time.time() - t_start), "waves": len(waves), "kept_added_h": round(r.added / 3600, 3), "backend": args.backend,
        "hours": args.hours, "mix": args.mix, "min_sim": args.min_sim, **({"error": err} if err else {}),
        **({"plain_render": args.plain_render} if args.plain_render is not None else {})})
    finalize(args, out, state, pool, (str(prepared), ref_dur), waves)
    status.set(phase="done" if not err else "error", kept_h=round(sum(kept_s(state).values()) / 3600, 3))
    return 1 if err else 0


def finalize(args, out: Path, state: dict, pool: list[dict], ref: tuple[str, float], waves: list[dict]) -> dict:
    """state.json and clips.jsonl, then the manifest, ATTRIBUTION.txt and the report (last: modal_app's preflight
    takes a report newer than the state as a finished call)."""
    vc.write_json(out / "state.json", state)
    kept = sorted(state["kept"].values(), key=lambda c: c["id"])
    reviewed = load_reviewed(out)  # --add-reviewed's clips: in every manifest
    vc.write_rows(out / "clips.jsonl", kept + reviewed)
    refs = {**assign_refs(kept, args.ref_fraction, args.seed, ref),
            **reviewed_refs(reviewed, kept, args.ref_fraction, args.seed, ref)}
    vc.write_rows(out / "voxcpm_train.jsonl", manifest_rows(kept + reviewed, refs, out / "wav", args.seed))
    write_text(out / "ATTRIBUTION.txt", "".join(x + "\n" for x in attribution_lines(kept + reviewed, args.voice,
                                                                                    args.description)))
    rep = build_report(state, pool, args, vc.read_rows(out / "attempts.jsonl"), refs,
                       {"waves_this_call": waves, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, reviewed)
    vc.write_json(out / "prep_report.json", rep)
    write_text(out / "PREP_REPORT.md", render_markdown(rep))
    per = ", ".join(f"{c} {b['hours']} h" for c, b in rep["by_category"].items() if b["clips"])
    log(f"{rep['selected']['hours']} h, {rep['selected']['clips']} clips ({per}); {len(refs)} with ref_audio; "
        f"{'complete' if rep['complete'] else incomplete_note(rep)} -> {out}")
    return rep


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
