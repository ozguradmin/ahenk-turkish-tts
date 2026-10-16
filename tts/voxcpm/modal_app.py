"""Fine-tune VoxCPM2 (Apache-2.0) on the Antalia voice (CC-BY-4.0) on Modal GPUs, and score it.

LoRA fine-tuning with the official VoxCPM recipe (pinned commit), driven by tts/voxcpm/train_lora.py.
Corpus, base model, checkpoints and evaluations live in the Modal Volume `turkish-tts-data`
(the one the Piper recipe uses), so every step can be stopped and resumed.

Steps, from the repo root (PowerShell or bash, same commands):

    python -m modal run tts/voxcpm/modal_app.py --action prepare            # CPU ~15 min: 16 kHz corpus + base model (~5 GB)
    python -m modal run tts/voxcpm/modal_app.py --action probe              # GPU ~12 min: real s/step -> cost plan
    python -m modal run --detach tts/voxcpm/modal_app.py --action train --max-usd 8 --steps 1000
    python -m modal run tts/voxcpm/modal_app.py --action status
    python -m modal run tts/voxcpm/modal_app.py --action eval --ckpt step_0000600,trendyol --max-usd 2.2
    (full evaluation, ~17 + 23 min on A100: eval first lists each run's estimate and refuses when the default $2
     does not cover them all; then copies each run's WAVs to ./voxcpm-eval/<name>/ and uploads the score files to
     a private HF repo; `--action fetch --fetch step_0000600` does only that)
    python -m modal run tts/voxcpm/modal_app.py --action eval --ckpt step_0000600 --limit 30 --grid 10:2.0,16:2.0,16:2.5 --max-usd 2.4
    (inference sweep: one run per checkpoint and timesteps:cfg point, out dir eval/<ckpt>@ts<ts>-cfg<cfg>; --limit 30
     = 30 TR-Zor rows spread over all categories + the 20 held-out rows, long sentences that alone take ~5 min to
     synthesize: --args "--sets trzor" makes the same sweep fit in --max-usd 1.4. --limit / --args need --grid.)
    python -m modal run tts/voxcpm/modal_app.py --action fetch-ckpt --fetch step_0000600
    (the LoRA files only, -> ./voxcpm-lora/<ckpt>/: lora_weights.safetensors, lora_config.json, training_state.json)
    python -m modal run tts/voxcpm/modal_app.py --action push-ckpt --fetch step_0000600
    (./voxcpm-lora/<ckpt>/ -> the Volume of the ACTIVE Modal profile, e.g. to evaluate on another workspace;
     never overwrites)

Two-stage training: stage 1 strengthens Turkish on ~25 h of Alania, stage 2 continues that LoRA
on Antalia so the voice stays Antalia's. After `--action prepare` (Antalia's held-out sentences must be there):

    python -m modal run tts/voxcpm/modal_app.py --action prepare-alania --max-usd 1
    (CPU, 8 cores: tts/voxcpm/prepare_alania.py -> /data/alania16k, a filtered subset of
     cloud0day3/alania-synthetic-speech-tr (CC BY 4.0), Antalia's held-out and TR-Zor sentences excluded;
     --args "--hours 25 --min-utmos 3.8" pass through; at the --max-usd deadline it stops, the same command continues)
    python -m modal run tts/voxcpm/modal_app.py --action probe --args "--data /data/alania16k"
    python -m modal run --detach tts/voxcpm/modal_app.py --action train --exp-name voxcpm2-alania-s1 --steps N --args "--data /data/alania16k"
    python -m modal run --detach tts/voxcpm/modal_app.py --action train --exp-name voxcpm2-alania-s2 --steps M --args "--init-lora /data/exp/voxcpm2-alania-s1/ckpt/step_XXXXXXX"
    (stage 2, on Antalia: step 0 from the stage-1 LoRA, fresh optimizer and LR schedule, same --rank; --init-lora
     counts only while voxcpm2-alania-s2 has no checkpoint, later runs of the same command resume. Stage 1 continues
     with the same command, --args included: before any GPU starts, train / probe check --args with train_lora.py's
     parser, refuse --data whose prepare-alania was cut short, and refuse to resume an experiment on another corpus
     (--switch-data allows it) or LoRA rank than its checkpoints')
    python -m modal run tts/voxcpm/modal_app.py --action eval --exp-name voxcpm2-alania-s2 --ckpt step_... --max-usd 2

Stage 1 on REAL human speech instead (same layout; train with --args "--data /data/human16k"):

    python -m modal run tts/voxcpm/modal_app.py --action prepare-human --max-usd 2
    (TTS_HUMAN_GPU=L4|T4 (default L4) + 8 cores + 16 GiB: tts/voxcpm/prepare_human.py -> /data/human16k, Common Voice
     17.0 Turkish (CC0) clips scored with UTMOS on the GPU, Antalia's held-out and TR-Zor sentences excluded;
     --args "--min-utmos 3.0 --sources cv,issai" pass through; at the --max-usd deadline it stops, the same command
     continues; a later --args "... --restart" with another --min-utmos reuses the scores in human16k/scores.jsonl)
    (fetch, fetch-ckpt and push-ckpt of an --exp-name other than voxcpm2-antalia use ./voxcpm-eval/<exp>/ and
     ./voxcpm-lora/<exp>/ (and <exp>/ in the HF score repo), so its step_* never overwrite the Antalia ones)

Voice contest (docs/SES_PLANI.md step 1; tts/voxcpm/voice_contest.py, results in /data/contest/<run>/):

    python -m modal run tts/voxcpm/modal_app.py --action contest-refs-push
    (./voxcpm-refs/<voice>/ (tts/voxcpm/fetch_refs.py: clips + sources.json) -> /data/contest/refs/<voice>/ on the
     ACTIVE profile's Volume; a folder whose sources.json lacks the licence / consent / attribution / reference-only
     record (voice_contest.check_sources), or one already there, is refused: nothing is replaced)
    python -m modal run tts/voxcpm/modal_app.py --action contest-screen --run r1 --max-usd 1.2
    (designed voices: tts/voxcpm/voice_descriptions.jsonl x 2 seeds by voice design, the best 30 clone 4 sentences,
     ranked by their mean UTMOS -> screen/entries.json, ranking.md; --args "--limit 10" for a cheap first look; the
     clone test waits until every candidate is made and scored, entries.json until every clone is: run the same
     command again while the result says "the same command continues")
    python -m modal run tts/voxcpm/modal_app.py --action contest-render --run r1 --max-usd 1.5 [--spec my.json]
    (entries x benchmarks/tts/sets/contest.jsonl: the controls + every pushed reference + this run's screen winners
     (voice_contest.DEFAULT_SPEC) or a local --spec; per entry and in total "~min" and "at most $X" first, an entry
     that does not fit is skipped (with the default $2: refused); nothing starts when every entry is made and scored)
    python -m modal run tts/voxcpm/modal_app.py --action contest-render --run r1-final --max-usd 0.9 --args "--timesteps 16 --screen r1 --only a,b"
    (finals: a run's clips all have one --timesteps / --cfg / --seed, so finals go to a new --run; --screen r1 brings
     r1's designed winners along; an entry whose clips on the Volume were made otherwise is left out before any GPU)
    python -m modal run tts/voxcpm/modal_app.py --action contest-render --run onset-v1 --max-usd 0.5 --spec tts/voxcpm/verify_onset.json --args "--ids sira_01,soru_01,duz_eslesik_01,mi_soru_01,emo_angry_01,emo_empathetic_01 --seeds 0,1,2,3 --no-score"
    python -m modal run tts/voxcpm/modal_app.py --action contest-render --run onset-v1-api --backend voxcpm --max-usd 0.5 --spec tts/voxcpm/verify_onset.json --args "--only masha,dima,designed-f037-s1 --ids sira_01,soru_01,duz_eslesik_01,mi_soru_01,emo_angry_01,emo_empathetic_01 --seeds 0,1 --workers 2 --no-score"
    (the stray-onset check: masha, dima, designed-f037-s1 x the 6 Y-initial texts x seeds 0-3, the reference
     prepared (refprep) and raw (-raw), on Nano-vLLM (~$0.33 estimated, 144 clips); seeds 0-1 on the official API
     (~$0.39 estimated, 36 clips); no scoring: each clip's onset check comes from synthesis (rows.jsonl onset_*,
     leaderboard "onset stray"))
    python -m modal run tts/voxcpm/modal_app.py --action contest-fetch --run r1 [--args --all]
    (clips + scores + leaderboard.md -> ./voxcpm-contest/r1/; contest-screen / -render fetch at their end too)
    (TTS_GPU's container + 8 cores + 48 GiB, on nanovllm_app.py's image (nanovllm_base.base_image) + SpeechBrain.
     --backend nanovllm (default): Nano-vLLM-VoxCPM, one server per (LoRA, timesteps), --args "--concurrency N"
     requests in flight (64; the probe measured ~50 audio s per GPU s at 10 timesteps, 36-38 at 16), every server
     stopped before the next and before the scorers load; a LoRA it cannot load goes to the official API for those
     entries only. --backend voxcpm: several VoxCPM2 processes of the official API share the GPU (--args "--workers N
     --load-parallel M", also the LoRA fallback's: refused when voice_contest.pool_ram_gb exceeds the container's
     RAM). Throughput and peak memory are measured into /data/contest/throughput.jsonl and later estimates use it;
     --max-usd (default 2) is a hard deadline of the call as for prepare-alania: the script starts no server and no
     clip it could not finish in time to score them, the same command continues)

Training corpus in the chosen voice (docs/SES_PLANI.md step 3; tts/voxcpm/render_corpus.py -> /data/f022corpus):

    python -m modal run tts/voxcpm/modal_app.py --action corpus-render --max-usd 4 [--args "--hours 5 --min-sim 0.7"]
    (TTS_GPU's container + 8 cores + 32 GiB on the contest image: base VoxCPM2 cloning designed-f022-s0 on Nano-vLLM
     (--backend voxcpm: the official API), the texts of tts/voxcpm/textbank/ + cloud0day3/alania-domain-text-tr,
     every clip gated (onset, Whisper CER 0 and p(tr), ECAPA, UTMOS, intonation, emotion) and retried with new seeds,
     in waves (render, stop the server, score); first the estimate of the rest of the corpus from the Volume's state
     and throughput.jsonl and "at most $X"; --max-usd (default 4) is a hard deadline, the same command continues; at
     the end the report and a sample are fetched)
    python -m modal run tts/voxcpm/modal_app.py --action corpus-fetch [--args "--sample 100 --seed 0"]
    (report, clips.jsonl, manifest and a random sample of kept clips -> ./voxcpm-corpus/f022/)
    python -m modal run tts/voxcpm/modal_app.py --action corpus-add-reviewed [--args "--add-reviewed FILE --dry-run"]
    (the laughter trial's clips a listener approved, tts/voxcpm/textbank/laughter_approved.json by default, into the
     corpus: render_corpus.py --add-reviewed on the Volume, a CPU call of seconds (~$0.0001); only between
     corpus-render calls: refused while status.json says one is going; adding again adds nothing new)
    python -m modal run --detach tts/voxcpm/modal_app.py --action train --exp-name voxcpm2-f022 --steps N --args "--data /data/f022corpus"
    (refused while the corpus is cut short or not complete)

A second corpus beside it (long items + number-heavy neutral rows; render_corpus.py module doc), then one manifest:

    python -m modal run tts/voxcpm/modal_app.py --action corpus-render --out /data/f022corpus2 --max-usd 2 --args "--hours 1.0 --mix long=0.47,neutral=0.47,cwords=0.06 --min-sim 0.6 --laughter-usd 0 --refs-from /data/f022corpus --avoid-pool /data/f022corpus --dry-run"
    (--out: a corpus folder right under /data, default /data/f022corpus; corpus-fetch --out brings it to
     ./voxcpm-corpus/<folder>/; the same command without --dry-run renders, again: it continues. The plan prints
     this call's cost and the whole rest's: how many calls of --max-usd, each with its start-up and a wave's server +
     scorer loads again (corpus_calls). The long items' and cwords' pass rates are guesses until the first wave
     measures them on 40 texts; under --min-pass (0.05) they stop. cwords: emotion rows with a word-initial c, every
     instructed try with a c c-checked (phon_check; render_corpus.py module doc). --args "... --plain-render cwords":
     cwords render without the instruction, a fresh budget of tries measured anew (render_corpus.py "Plain render");
     a later call without it is refused here before any GPU starts)
    python -m modal run tts/voxcpm/modal_app.py --action corpus-merge [--out /data/f022mix] [--args "--sources /data/f022corpus,/data/f022corpus2 --dry-run"]
    (CPU, seconds: render_corpus.py --merge -> /data/f022mix/voxcpm_train.jsonl, the union of the sources' manifests,
     + prep_report.json / README.md; refused while a source's corpus-render call is going or its mix is not complete
     (--args "--allow-incomplete" takes it); train --args "--data /data/f022mix" then refuses a merge whose sources
     changed since: merge again)
    python -m modal run tts/voxcpm/modal_app.py --action corpus-merge --out /data/f022mix_both --args "--instructions both --dry-run"
    (the same merge with Turkish instructions (render_corpus.py "Merge"): both = each instructed row also with the
     Turkish wording of tts/voxcpm/instructions_tr.json, tr = the Turkish one only (--out /data/f022mix_tr); refused
     here, before the call, when a source's manifest has a wording the table lacks or --out holds a merge with another
     --instructions; train --data on it refuses it once the table changed: merge again, same --out)
    python -m modal run tts/voxcpm/modal_app.py --action corpus-audit-c [--out /data/f022corpus] [--args "--dry-run"]
    (AUDIT_GPU (L4) + 4 cores + 16 GiB, minutes: render_corpus.py --audit-c, phon_check's IPA model over the
     corpus's manifest clips with an instruction and a "c" -> <out>/c_audit.jsonl (nothing else written; refused while
     a corpus-render call is going); the plan first: clips, seconds of audio, "at most $X"; the file is fetched to
     ./voxcpm-corpus/<folder>/. corpus-merge then leaves the failed clips out (and refuses a source whose instructed
     clips with a c neither the render gate nor an audit checked: --args "--allow-unaudited" takes it), and the train
     preflight refuses a merge whose source's c_audit.jsonl changed since)

GPU: TTS_GPU=A100-80GB|L40S|H100 (default A100-80GB), read at import time.
Cost control: `train` turns --max-usd (default 8) into a hard time limit; the trainer then saves and stops, and
the next `train` continues from the last checkpoint (--args may not set train_lora.py's --hours / --steps / --exp /
--model: the wrapper's win anyway; a run train_lora.py refuses, exit code 2, is not retried). `prepare-alania` turns --max-usd (default 1) into a hard
deadline of its CPU call (8 cores + 16 GiB, ~$0.50/h); `prepare-human` (default 2) the same for its GPU call (L4
+ 8 cores + 16 GiB, ~$1.30/h). `eval` turns --max-usd (default 2) into a hard deadline for
the whole call. A run starts only if its estimated time (run_seconds) still fits, else it is "skipped (budget)" and
costs nothing; one still going at the deadline is killed and leaves eval/<name> as it was. Ledgers on the Volume
count the time that preempted attempts already used. GPU containers stop 2 s after their call (scaledown_window).
Also set a workspace spending limit in Modal.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import posixpath
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Callable

import modal

REPO = "/repo"
DATA = "/data"
VOXCPM_SRC = "/opt/voxcpm"
VOXCPM_COMMIT = "f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629"  # OpenBMB/VoxCPM, 2026-09-30
_HERE = Path(__file__).resolve()
# repo root on the local machine; inside the container this file sits at /root/modal_app.py (no repo
# around it), where the value is unused but must not raise
LOCAL_REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
for _d in (_HERE.parent, Path(REPO, "tts", "voxcpm")):  # nanovllm_base.py: next to this file, in the container /repo
    if (_d / "nanovllm_base.py").is_file() and str(_d) not in sys.path:
        sys.path.append(str(_d))

import nanovllm_base  # noqa: E402  (the contest's Nano-vLLM image recipe; stdlib only)

GPU = os.environ.get("TTS_GPU", "A100-80GB")
GPU_PRICE = {"H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95}  # USD/h, Modal list prices (2026)
GPU_SIDE_PRICE = 4 * 0.047 + 32 * 0.008  # 4 cores + 32 GiB of the GPU container
CPU_JOB_PRICE = 4 * 0.047 + 16 * 0.008
ALANIA_CPU_PRICE = 8 * 0.047 + 16 * 0.008  # prepare_alania: 8 cores + 16 GiB
EXP = "voxcpm2-antalia"
MODEL_DIR = f"{DATA}/models/VoxCPM2"
CORPUS = f"{DATA}/antalia16k"
ALANIA = f"{DATA}/alania16k"  # stage-1 corpus (prepare_alania.py)
TRAIN_MAX_USD, EVAL_MAX_USD, ALANIA_MAX_USD = 8.0, 2.0, 1.0  # --max-usd 0 = these
ALANIA_TIMEOUT_S = 4 * 3600  # Modal's own limit of one prepare_alania call
ALANIA_START_S, ALANIA_TAIL_S = 60, 120  # billed start-up; after the kill: ledger, final volume commit
ALANIA_MIN_S = 60  # with less time left the script is not started
HUMAN = f"{DATA}/human16k"  # stage-1 corpus of real human speech (prepare_human.py)
SCORE_GPU_PRICE = {"L4": 0.80, "T4": 0.59}  # USD/h, Modal list ($0.000222 / $0.000164 per s, modal.com/pricing 2026-10-10)
HUMAN_GPU_PRICE = {**SCORE_GPU_PRICE, **GPU_PRICE}  # prepare-human only: train / probe / eval keep GPU_PRICE
HUMAN_GPU = os.environ.get("TTS_HUMAN_GPU", "L4")  # prepare-human's UTMOS GPU, read at import time
HUMAN_SIDE_PRICE = 8 * 0.047 + 16 * 0.008  # prepare_human: 8 cores + 16 GiB next to the GPU
HUMAN_MAX_USD, HUMAN_TIMEOUT_S = 2.0, 4 * 3600  # --max-usd 0 = this; Modal's own limit of one call
HUMAN_START_S = 90  # billed GPU container start-up (deadline accounting; tail and minimum as for prepare-alania)
EVAL_TIMEOUT_S = 3 * 3600  # Modal's own limit of one evaluate call
EVAL_START_S = 90  # billed container start-up before evaluate() runs, taken off its budget
EVAL_TAIL_S = 60  # after the last run: kill, ledger, volume commit
IDLE_S = 2  # scaledown_window of the GPU functions: billed idle after a call returns (Modal's default: 60 s)
# run_seconds, rough A100 time of one synth_eval run: a fixed part (VoxCPM2 / Whisper / UTMOS loading, judging; a full
# step_0000600 run took ~16 min, 10 of it synthesis) + synthesis per set at 10 timesteps, set -> (s, rows): measured
# 301 s for TR-Zor's 155 rows and 291 s for the 20 long held-out sentences; it grows with the timesteps (Trendyol, 16:
# 494 / 422 s)
EVAL_FIXED_S = 7 * 60
EVAL_SYNTH_S = {"trzor": (310.0, 155), "heldout": (300.0, 20)}
SCORE_FILES = ("rows.jsonl", "summary.json")  # synth_eval writes them last, in this order
CKPT_FILES = ("lora_weights.safetensors", "lora_config.json", "training_state.json")  # first two required
LORA_DIR = LOCAL_REPO / "voxcpm-lora"  # fetch-ckpt / push-ckpt, local side
WRAPPER_OPTS = ("--model", "--exp", "--steps", "--hours")  # train_lora.py options train / probe set (train_cmd)
PREP_FILES = ("prep_report.json", "voxcpm_train.jsonl", "state.json", "clips.jsonl")  # report_current
CONTEST = f"{DATA}/contest"  # voice contest (voice_contest.py): <run>/<entry>/, <run>/screen/, refs/, throughput.jsonl
# synthesis workers + the prosody pool; RAM: voice_contest.pool_ram_gb (~3 GB per loaded worker + ~13 GB per VoxCPM2
# load, --load-parallel at once; estimates: throughput.jsonl records host_mem_peak_mib), at most CONTEST_RAM_USE of it
CONTEST_CPU, CONTEST_MEM, CONTEST_RAM_USE = 8, 49152, 0.9
CONTEST_SIDE_PRICE = CONTEST_CPU * 0.047 + CONTEST_MEM / 1024 * 0.008  # $/h next to the GPU
CONTEST_MAX_USD, CONTEST_TIMEOUT_S = 2.0, 3 * 3600  # --max-usd 0 = this; Modal's own limit of one call
CONTEST_START_S, CONTEST_GRACE_S = 90, 30  # billed start-up; the script's own deadline this long before the kill
CONTEST_DIR, REFS_DIR = LOCAL_REPO / "voxcpm-contest", LOCAL_REPO / "voxcpm-refs"  # contest-fetch / contest-refs-push
REFS_PUSH_EXT, REFS_PUSH_MAX_MB = (".wav", ".flac", ".mp3", ".ogg", ".json", ".txt", ".md"), 50
F022 = f"{DATA}/f022corpus"  # corpus-render (render_corpus.py): the designed-f022-s0 training corpus
# one Nano-vLLM server or the scorers at a time + the CPU pass (no VoxCPM2 worker pool unless --backend voxcpm)
CORPUS_CPU, CORPUS_MEM = 8, 32768
CORPUS_SIDE_PRICE = CORPUS_CPU * 0.047 + CORPUS_MEM / 1024 * 0.008  # $/h next to the GPU (render_corpus.USD_PER_H)
CORPUS_MAX_USD, CORPUS_TIMEOUT_S, CORPUS_START_S = 4.0, 6 * 3600, 90  # --max-usd 0 = this; Modal's limit of a call
CORPUS_SCRIPT_S = 15  # render_corpus.py's own start in a call before its first wave (imports, state, the pool read)
CORPUS_DIR, CORPUS_SAMPLE = LOCAL_REPO / "voxcpm-corpus", 100  # corpus-fetch: ./voxcpm-corpus/f022/, clips to listen
# corpus-add-reviewed: render_corpus.py --add-reviewed on the Volume (copies a few WAVs, appends rows: seconds)
REVIEWED_CPU, REVIEWED_MEM, REVIEWED_TIMEOUT_S, REVIEWED_START_S = 1, 1024, 600, 30
REVIEWED_PRICE = REVIEWED_CPU * 0.047 + REVIEWED_MEM / 1024 * 0.008  # $/h
# corpus-merge: render_corpus.py --merge (reads manifests, stats each WAV, writes one manifest: seconds)
MERGE_OUT, MERGE_SOURCES = f"{DATA}/f022mix", (F022, f"{DATA}/f022corpus2")
MERGE_CPU, MERGE_MEM, MERGE_TIMEOUT_S, MERGE_START_S = 1, 2048, 900, 30
MERGE_PRICE = MERGE_CPU * 0.047 + MERGE_MEM / 1024 * 0.008  # $/h
# corpus-audit-c: render_corpus.py --audit-c (phon_check's wav2vec2 on a small GPU, the contest image's transformers as
# the gate's); its timeout is the hard limit
AUDIT_GPU, AUDIT_CPU, AUDIT_MEM, AUDIT_TIMEOUT_S, AUDIT_START_S = "L4", 4, 16384, 840, 120
AUDIT_PRICE = SCORE_GPU_PRICE[AUDIT_GPU] + AUDIT_CPU * 0.047 + AUDIT_MEM / 1024 * 0.008  # $/h
# folders right under /data a corpus-render --out / corpus-merge may never be (the other datasets, models, runs)
CORPUS_RESERVED = ("antalia", "antalia16k", "alania16k", "human16k", "contest", "models", "exp", "modal_ledger",
                   "hf_home", "torch_home")

app = modal.App("turkish-tts-voxcpm")
volume = modal.Volume.from_name("turkish-tts-data", create_if_missing=True)
SECRETS = [modal.Secret.from_name(os.environ["TTS_HF_SECRET"])] if os.environ.get("TTS_HF_SECRET") else []

_IGNORE = [".git", ".venv*", "data", "exports", "build", "dist", "**/__pycache__", "**/*.pyc", "**/*.ckpt",
           "**/*.onnx", "**/*.wav", "stt/icefall", "tts/piper1-gpl", "tts/voxcpm-src", "benchmarks/**/results",
           "benchmarks/**/logs", "antalia-export", "voxcpm-eval", "voxcpm-lora", "hf-publish", "voxcpm-refs",
           "voxcpm-contest", "voxcpm-corpus"]

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .run_commands(
        f"git clone https://github.com/OpenBMB/VoxCPM {VOXCPM_SRC} && cd {VOXCPM_SRC} && git checkout {VOXCPM_COMMIT}",
        # one resolver run with torch pinned, so no dependency can swap the CUDA build of torch
        "pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0 "
        f"-e {VOXCPM_SRC} 'datasets>=3,<4' tensorboardX faster-whisper==1.2.1 jiwer soxr",
        "python -c \"import torch, voxcpm, faster_whisper; print('torch', torch.__version__, torch.version.cuda)\"",
        # ctranslate2 (faster-whisper) loads cuBLAS/cuDNN by soname: point the loader at the pip CUDA libs
        "python -c \"import nvidia.cublas, nvidia.cudnn; print(nvidia.cublas.__path__[0] + '/lib'); "
        "print(nvidia.cudnn.__path__[0] + '/lib')\" > /etc/ld.so.conf.d/nvidia-pip.conf && ldconfig",
    )
)
# TORCH_HOME: UTMOS (torch.hub tarepan/SpeechMOS, needs only torch + torchaudio) cached on the Volume
_ENV = {"TTS_DATA": DATA, "VOXCPM_SRC": VOXCPM_SRC, "PYTHONPATH": f"{REPO}/src", "HF_HOME": f"{DATA}/hf_home",
        "TORCH_HOME": f"{DATA}/torch_home", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
image = base_image.env(_ENV).add_local_dir(LOCAL_REPO, REPO, ignore=_IGNORE)
# voice contest: nanovllm_app.py's image (nanovllm_base.base_image: Nano-vLLM-VoxCPM + flash-attn next to the official
# API at the same VoxCPM commit, faster-whisper, torch 2.8.0+cu128), so both backends (--backend nanovllm / voxcpm)
# run on it, + SpeechBrain 1.0.3 (Apache-2.0) for ECAPA speaker similarity (speechbrain/spkrec-ecapa-voxceleb,
# Apache-2.0); UTMOS comes through torch.hub (torch + torchaudio). Train / eval keep their own image. torch pinned
# again so pip keeps the CUDA build. `modal run` builds every function's image: the speechbrain check only logs, so
# a speechbrain problem can never block train / eval (the contest then reports "sim" as an error and goes on)
contest_image = (
    nanovllm_base.base_image(modal)
    .run_commands(
        f"pip install {nanovllm_base.CU128} {nanovllm_base.TORCH_PINS} speechbrain==1.0.3",
        "python -c \"import torch, speechbrain; from speechbrain.inference.classifiers import EncoderClassifier; "
        "print('torch', torch.__version__, 'speechbrain', speechbrain.__version__)\" "
        "|| echo 'WARNING: speechbrain does not import: the contest will have no speaker similarity'",
    )
    .env(_ENV)
    .add_local_dir(LOCAL_REPO, REPO, ignore=_IGNORE)
)


def _run(cmd: list[str], check: bool = True) -> int:
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if check and rc not in (0, 75):  # 75 = stopped at the time limit, resumable
        raise RuntimeError(f"exit code {rc}: {' '.join(cmd)}")
    return rc


def _kill(p: subprocess.Popen, wait: bool = True) -> None:
    """SIGKILL p and its process group (started with start_new_session), so nothing keeps the GPU busy."""
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):  # no killpg (Windows tests) or already gone
        p.kill()
    if wait:
        p.wait()


def _run_until(cmd: list[str], deadline: float, live: list | None = None) -> int | None:
    """_run(check=False) with a deadline (time.time()): the process is killed then and None returned."""
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=REPO, start_new_session=True)
    if live is not None:
        live.append(p)
    try:  # timeout=inf overflows in Popen.wait: None = no deadline
        return p.wait(timeout=max(deadline - time.time(), 0.0) if math.isfinite(deadline) else None)
    except subprocess.TimeoutExpired:
        _kill(p)
        return None
    except BaseException:  # SIGINT on preemption etc.: never leave a synthesis running
        _kill(p)
        raise
    finally:
        if live is not None:
            live.remove(p)


def _names(s: str) -> list[str]:
    """"a, b,,a" -> ["a", "b"]"""
    return list(dict.fromkeys(c.strip() for c in s.split(",") if c.strip()))


def parse_grid(grid: str) -> list[tuple[int, float]]:
    """"10:1.5,16:2.0" -> [(10, 1.5), (16, 2.0)]: (inference timesteps, CFG) points, duplicates dropped."""
    points: list[tuple[int, float]] = []
    for item in _names(grid):
        ts, sep, cfg = item.partition(":")
        try:
            p = (int(ts), float(cfg))
        except ValueError:
            p = None
        if not sep or p is None or not 1 <= p[0] <= 100 or not (math.isfinite(p[1]) and 0 < p[1] <= 10):
            raise ValueError(f"bad --grid point {item!r}: want timesteps:cfg (1-100 : 0-10), e.g. 16:2.0")
        if p not in points:
            points.append(p)
    return points


def run_name(ck: str, ts: int | None = None, cfg: float | None = None) -> str:
    """Out dir of one eval run: the checkpoint, or <ckpt>@ts<ts>-cfg<cfg> for a grid point."""
    return ck if ts is None else f"{ck}@ts{ts}-cfg{cfg:g}"


def eval_runs(ckpts: str, grid: str = "") -> list[tuple[str, str, list[str]]]:
    """(out dir name, checkpoint, synth_eval args appended last) per run: each checkpoint x each grid point."""
    points = parse_grid(grid)
    if not points:
        return [(ck, ck, []) for ck in _names(ckpts)]
    return [(run_name(ck, ts, cfg), ck, ["--timesteps", str(ts), "--cfg", f"{cfg:g}"])
            for ck in _names(ckpts) for ts, cfg in points]


def usd_to_seconds(usd: float, usd_per_h: float) -> float:
    return usd / usd_per_h * 3600


def overwrites_full_eval(limit: int, extra: list[str] | None, grid: str) -> bool:
    """--limit / --args without a grid would write eval/<ckpt>, replacing that checkpoint's full evaluation."""
    return bool(limit or extra) and not parse_grid(grid)


def synth_opts(ck: str, limit: int, extra: list[str], point: list[str]) -> list[str]:
    """synth_eval options of one run after --model/--val/--out/--lora/--ref, in evaluate()'s order (argparse: the
    last one wins): --args override --limit, Trendyol's 16 timesteps override --args, the grid point overrides all."""
    return ["--limit", str(limit), *extra, *(["--timesteps", "16"] if ck == "trendyol" else []), *point]


def _last_opt(args: list[str], opt: str, default: str) -> str:
    """Value of the last `opt V` / `opt=V` in args, as argparse takes it."""
    v = default
    for i, a in enumerate(args):
        if a == opt and i + 1 < len(args):
            v = args[i + 1]
        elif a.startswith(opt + "="):
            v = a[len(opt) + 1:]
    return v


def run_seconds(opts: list[str]) -> float:
    """Rough A100 seconds of one synth_eval run with these options (--limit, --timesteps, --sets; see EVAL_SYNTH_S).
    ValueError on a non-integer --limit / --timesteps."""
    limit, ts = int(_last_opt(opts, "--limit", "0")), int(_last_opt(opts, "--timesteps", "10"))
    sets = _last_opt(opts, "--sets", "trzor,heldout").split(",")
    synth = sum(s * (min(limit, n) if limit > 0 else n) / n for k, (s, n) in EVAL_SYNTH_S.items() if k in sets)
    return EVAL_FIXED_S + synth * max(ts, 1) / 10


def eval_plan(runs: list[tuple[str, str, list[str]]], limit: int, extra: list[str],
              max_s: float) -> list[tuple[str, float, bool]]:
    """(name, estimated s, starts) per run, by evaluate()'s rule: a run starts only if its estimate fits in what
    start-up, the earlier runs, the tail and the idle window leave of max_s; a run that does not fit costs nothing."""
    left, plan = max_s - EVAL_START_S - EVAL_TAIL_S - IDLE_S, []
    for name, ck, point in runs:
        need = run_seconds(synth_opts(ck, limit, extra, point))
        plan.append((name, need, need <= left))
        left -= need if need <= left else 0.0
    return plan


def _scores(s: dict) -> dict:
    """Per-set WER of a synth_eval summary.json, plus "utmos": per-set means when it was scored, and "utmos_error"
    when some or all rows have no UTMOS (pending, failed, bad clips), so a partial mean never passes as complete."""
    out = {k: v["wer"] for k, v in s.get("wer", {}).items() if "/" not in k}
    if utmos := {k: v["mean"] for k, v in s.get("utmos", {}).items() if "/" not in k}:
        out["utmos"] = utmos
    if s.get("utmos_error"):
        out["utmos_error"] = s["utmos_error"]
    return out


def _opt_args(args: list[str], *opts: str) -> list[str]:
    """The items of args that set one of opts (`--opt`, `--opt=V`, or a prefix argparse expands, such as --ou)."""
    return [a for a in args
            if len(o := a.split("=", 1)[0]) > 2 and o.startswith("--") and any(f.startswith(o) for f in opts)]


def alania_cmd(extra: list[str]) -> list[str]:
    """prepare_alania.py command line: output dir and the Antalia held-out exclusion fixed, --args passed through (a
    further --exclude-manifest adds to the Antalia one: action="extend"). ValueError when --args set --out."""
    if bad := _opt_args(extra, "--out"):
        raise ValueError(f"--args may not set {', '.join(bad)}: prepare-alania always writes {ALANIA} (status, train "
                         "--args \"--data ...\" and the docs expect it there)")
    return ["python", "tts/voxcpm/prepare_alania.py", "--out", ALANIA,
            "--exclude-manifest", f"{CORPUS}/voxcpm_val.jsonl", *extra]


def human_cmd(extra: list[str]) -> list[str]:
    """prepare_human.py command line, as alania_cmd: output dir and the Antalia held-out exclusion fixed, --args passed
    through. ValueError when --args set --out."""
    if bad := _opt_args(extra, "--out"):
        raise ValueError(f"--args may not set {', '.join(bad)}: prepare-human always writes {HUMAN} (status, train "
                         "--args \"--data ...\" and the docs expect it there)")
    return ["python", "tts/voxcpm/prepare_human.py", "--out", HUMAN,
            "--exclude-manifest", f"{CORPUS}/voxcpm_val.jsonl", *extra]


def report_current(mtimes: dict[str, float]) -> bool:
    """prepare_alania.py's last run finished: voxcpm_train.jsonl and prep_report.json (its final writes) exist and
    neither state.json nor clips.jsonl changed after the report (it rewrites both when it starts): else a later run
    was cut short or is still going, and manifest and report are an older selection. mtimes: name -> mtime of the
    files that exist."""
    rep = mtimes.get("prep_report.json")
    return (rep is not None and "voxcpm_train.jsonl" in mtimes
            and all(mtimes.get(n, rep) <= rep for n in ("state.json", "clips.jsonl")))


def corpus_summary(d: Path, report: bool = True) -> dict | None:
    """Headline of prepare_alania.py's prep_report.json (selected hours / clips / voices, target, revision) when its
    last run finished (report_current, and the report counts the clips of clips.jsonl); else (report=False too: the
    run was cut short) the progress in clips.jsonl, plus the older selection that voxcpm_train.jsonl still holds
    under "last_finished"; None when d has neither."""
    head = None
    try:
        rep = json.loads((d / "prep_report.json").read_text(encoding="utf-8"))
        sel, ds = rep.get("selected") or {}, rep.get("dataset") or {}
        head = ({**{k: sel[k] for k in ("hours", "clips", "voices", "speakers") if k in sel},
                 **{k: rep[k] for k in ("target_hours", "reached", "time") if k in rep},
                 **({"revision": ds["revision"]} if "revision" in ds else {})}
                or {k: v for k, v in rep.items() if isinstance(v, (str, int, float, bool))})
        if rep.get("merged") and isinstance(rep.get("audio"), dict):  # --merge: "selected" counts manifest rows
            head = {**{("rows" if k == "clips" else k): v for k, v in head.items()}, "audio": rep["audio"],
                    "instructions": (rep.get("instructions") or {}).get("mode") or "en"}
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    prog = None
    try:
        lines = (d / "clips.jsonl").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        pass
    else:
        n = s = 0
        for x in lines:
            try:
                s, n = s + float(json.loads(x)["duration"]), n + 1
            except (ValueError, KeyError, TypeError):  # a line cut short by a kill
                pass
        prog = {"unfinished": True, "clips": n, "hours": round(s / 3600, 2)}
    try:
        mt = {n: (d / n).stat().st_mtime for n in PREP_FILES if (d / n).exists()}
    except OSError:
        mt = {}
    if (head is not None and report and report_current(mt)
            and (prog is None or prog["clips"] == head.get("clips", prog["clips"]))):
        return head
    if head is None:
        return prog
    return {**(prog or {"unfinished": True, "clips": 0, "hours": 0.0}), "last_finished": head}


def local_dir(base: Path, exp_name: str) -> Path:
    """Local side of fetch / fetch-ckpt / push-ckpt: base itself for the Antalia experiment (as before), base/<exp_name>
    for any other, so e.g. voxcpm2-alania-s2's step_0000600 never overwrites voxcpm2-antalia's."""
    return base if exp_name == EXP else base / exp_name


def _promote(tmp: Path, dst: Path) -> None:
    """A finished synth_eval --out dir tmp -> dst (an older dst is kept as <dst>.replaced-<UTC time>); the rows'
    wav paths follow. shutil.move: a plain rename, or copy + delete where the Volume cannot rename a directory."""
    if dst.exists():
        shutil.move(dst, dst.with_name(f"{dst.name}.replaced-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"))
    shutil.move(tmp, dst)
    rows, old = dst / "rows.jsonl", str(tmp)
    if rows.exists():
        out = []
        for x in rows.read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            if isinstance(r.get("wav"), str) and r["wav"].startswith(old):
                r["wav"] = str(dst) + r["wav"][len(old):]
            out.append(json.dumps(r, ensure_ascii=False) + "\n")
        part = rows.with_name(rows.name + ".part")
        part.write_text("".join(out), encoding="utf-8")
        os.replace(part, rows)


class _PeriodicCommit:
    """Commit the volume every few minutes so checkpoints survive preemption."""

    def __init__(self, every_s: int = 300):
        self.every_s, self._stop = every_s, threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(self.every_s):
            try:
                volume.commit()
            except Exception as e:  # keep working even if one commit fails
                print(f"[modal] volume commit failed: {e}", flush=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=120)  # let an in-flight periodic commit finish first
        volume.commit()


@app.function(image=image, cpu=4, memory=16384, volumes={DATA: volume}, timeout=2 * 3600, secrets=SECRETS)
def prepare() -> None:
    """Antalia -> 16 kHz WAV (VoxCPM2's AudioVAE encoder rate) + manifests; VoxCPM2 weights."""
    from huggingface_hub import snapshot_download

    with _PeriodicCommit(300):
        _run(["python", "tts/prepare_antalia.py", "--out", CORPUS, "--sample-rate", "16000", "--no-phonemes",
              "--workers", "4"])
        _run(["python", "tts/voxcpm/make_manifest.py", "--data", CORPUS])
        snapshot_download("openbmb/VoxCPM2", local_dir=MODEL_DIR)
        print(f"[prepare] model in {MODEL_DIR}: {sorted(p.name for p in Path(MODEL_DIR).iterdir())}")


def _deadline_job(tag: str, cmd: list[str] | Callable[[float], list[str]], max_s: float, run_id: str, out: str,
                  start_s: float, need_val: bool = True, report: Callable[[int | None], dict] | None = None,
                  tmp_hf_home: bool = True) -> str:
    """Body of prepare_alania / prepare_human / contest: run `cmd` (a script) with a hard deadline of the whole call
    incl. start-up (start_s) and attempts a preemption cut short (ledger <run_id>.json on the Volume); the script's
    process group is killed then and the next call continues from its state file; a watchdog ends the container if
    anything else hangs. cmd may be a function of the kill time (time.time() scale) -> the command line (contest: the
    script's own --deadline). need_val: refuse without Antalia's held-out sentences (the corpus scripts exclude them);
    tmp_hf_home: HF downloads off the Volume (parquet, xet cache; contest keeps Whisper's weights on it). Returns {"rc"
    (None: killed at the deadline), "used_s", **report(rc) (default "corpus": corpus_summary(out))} or {"error",
    "used_s"}."""
    t0 = time.time()
    if need_val and not Path(CORPUS, "voxcpm_val.jsonl").is_file():
        return json.dumps({"error": f"no {CORPUS}/voxcpm_val.jsonl (the Antalia sentences to exclude): run "
                                    "--action prepare first", "used_s": start_s})
    ledger = Path(DATA) / "modal_ledger" / f"{run_id}.json" if run_id else None
    used0 = (json.loads(ledger.read_text())["used_s"] if ledger and ledger.exists() else 0.0) + start_s
    end = t0 + max_s - used0 - IDLE_S  # the container must be gone by then
    if end - ALANIA_TAIL_S - time.time() < ALANIA_MIN_S:
        return json.dumps({"error": f"budget of {max_s:.0f} s used up by earlier attempts", "used_s": round(used0)})
    if tmp_hf_home:
        os.environ["HF_HOME"] = "/tmp/hf_home"  # parquet downloads and the hf_xet chunk cache stay off the Volume
    live, stop = [], threading.Event()

    def save_ledger():
        if not ledger:
            return
        try:
            ledger.parent.mkdir(parents=True, exist_ok=True)
            tmp = ledger.with_suffix(".tmp")
            tmp.write_text(json.dumps({"used_s": used0 + time.time() - t0, "max_s": max_s}))
            os.replace(tmp, ledger)
        except Exception as e:  # the deadline itself does not depend on the ledger
            print(f"[{tag}] ledger not written: {e}", flush=True)

    def bookkeeping():
        while not stop.wait(30):
            save_ledger()

    def hard_stop():
        print(f"[{tag}] hard deadline ({max_s:.0f} s budget) reached: killing everything", flush=True)
        for p in list(live):
            _kill(p, wait=False)
        os._exit(124)

    watchdog = threading.Timer(max(end - time.time(), 15.0), hard_stop)
    watchdog.daemon = True
    watchdog.start()
    threading.Thread(target=bookkeeping, daemon=True).start()
    kill = end - ALANIA_TAIL_S
    try:
        with _PeriodicCommit(300):
            try:
                rc = _run_until(cmd(kill) if callable(cmd) else cmd, kill, live)
            finally:
                stop.set()
                save_ledger()  # before _PeriodicCommit's final commit
    finally:
        watchdog.cancel()
    if rc is None:
        print(f"[{tag}] stopped at the budget deadline; the same command continues from here", flush=True)
    res = report(rc) if report else {"corpus": corpus_summary(Path(out), report=rc == 0)}
    return json.dumps({"rc": rc, "used_s": round(used0 + time.time() - t0), **res}, indent=1, ensure_ascii=False)


@app.function(image=image, cpu=8, memory=16384, volumes={DATA: volume}, timeout=ALANIA_TIMEOUT_S,
              scaledown_window=IDLE_S, secrets=SECRETS)
def prepare_alania(args: list[str], max_s: float, run_id: str = "") -> str:
    """Stage-1 corpus: tts/voxcpm/prepare_alania.py -> ALANIA (alania_cmd: Antalia's held-out sentences excluded, `args`
    passed through). max_s: hard deadline of the whole call incl. start-up and attempts a preemption cut short (ledger
    <run_id>.json on the Volume); the script's process group is killed then and the next call continues from its state
    file; a watchdog ends the container if anything else hangs. Returns {"rc" (None: killed at the deadline), "used_s",
    "corpus": corpus_summary} or {"error", "used_s"}."""
    try:  # main checks this before the call
        cmd = alania_cmd(args)
    except ValueError as e:
        return json.dumps({"error": str(e), "used_s": ALANIA_START_S})
    return _deadline_job("prepare-alania", cmd, max_s, run_id, ALANIA, ALANIA_START_S)


@app.function(image=image, gpu=HUMAN_GPU if HUMAN_GPU in HUMAN_GPU_PRICE else "L4", cpu=8, memory=16384,
              volumes={DATA: volume}, timeout=HUMAN_TIMEOUT_S, scaledown_window=IDLE_S, secrets=SECRETS)
def prepare_human(args: list[str], max_s: float, run_id: str = "") -> str:
    """Stage-1 corpus of real human speech: tts/voxcpm/prepare_human.py -> HUMAN (human_cmd: Antalia's held-out
    sentences excluded, `args` passed through), UTMOS on this container's GPU; deadline, ledger, watchdog and result
    as prepare_alania (_deadline_job)."""
    try:  # main checks this before the call
        cmd = human_cmd(args)
    except ValueError as e:
        return json.dumps({"error": str(e), "used_s": HUMAN_START_S})
    return _deadline_job("prepare-human", cmd, max_s, run_id, HUMAN, HUMAN_START_S)


def contest_out(mode: str, run: str) -> str:
    """Volume dir of a contest call: /data/contest/<run> (render) or /data/contest/<run>/screen (screen); ValueError on
    another mode or a run name that is not a plain folder name (or is "refs", the references' folder, or
    "nanovllm-probe", nanovllm_app.py's folder next to the runs in ./voxcpm-contest/)."""
    if mode not in ("screen", "render"):
        raise ValueError(f"unknown contest mode {mode!r}")
    if not (run and len(run) <= 64 and run.isascii() and run[0].isalnum()
            and all(c.isalnum() or c in "._-" for c in run)) or run in ("refs", "nanovllm-probe"):
        raise ValueError(f"--run {run!r}: name the contest run, letters / digits / . _ - (not refs or nanovllm-probe), "
                         "e.g. --run r1")
    return f"{CONTEST}/{run}" + ("/screen" if mode == "screen" else "")


CONTEST_BACKENDS = ("nanovllm", "voxcpm")  # voice_contest.py's real engines (its "fake" is for tests)


def contest_cmd(mode: str, run: str, extra: list[str], deadline: float, spec_file: str = "",
                backend: str = "nanovllm") -> list[str]:
    """voice_contest.py command line: --args first, then what the wrapper owns (voice_contest.WRAPPER_OPTS; argparse:
    the last one wins, and main refuses them in --args too): model, output dir, the script's own deadline, the
    throughput log the estimates read, the engine (main's --backend: nanovllm or voxcpm), the resolved entry spec of a
    render."""
    if backend not in CONTEST_BACKENDS:
        raise ValueError(f"--backend {backend!r}: one of {', '.join(CONTEST_BACKENDS)}")
    return ["python", "tts/voxcpm/voice_contest.py", mode, *extra, "--model", MODEL_DIR,
            "--out", contest_out(mode, run), "--deadline", f"{deadline:.0f}",
            "--throughput-log", f"{CONTEST}/throughput.jsonl", "--backend", backend,
            *(["--spec", spec_file] if spec_file else [])]


@app.function(image=contest_image, gpu=GPU, cpu=CONTEST_CPU, memory=CONTEST_MEM, volumes={DATA: volume},
              timeout=CONTEST_TIMEOUT_S, scaledown_window=IDLE_S, secrets=SECRETS)
def contest(mode: str, run: str, args: list[str], max_s: float, run_id: str = "", spec: str = "",
            backend: str = "nanovllm") -> str:
    """Voice contest call: tts/voxcpm/voice_contest.py <mode> (contest_cmd) under a hard deadline of the whole call, as
    prepare_alania (_deadline_job: ledger <run_id>.json, process group killed at the deadline, watchdog: Nano-vLLM's
    engine processes are in that group too); the script gets its own deadline CONTEST_GRACE_S earlier, so it stops
    synthesis in time to score and write its summaries, and the next call continues where it stopped. spec: a
    render's resolved entry spec (-> /data/contest/<run>/entries.json). Returns {"rc", "used_s", "status": the
    script's status.json} or {"error", "used_s"}."""
    try:
        out = contest_out(mode, run)
        contest_cmd(mode, run, args, 0.0, "", backend)
    except ValueError as e:  # main checks this before the call
        return json.dumps({"error": str(e), "used_s": CONTEST_START_S})
    if not Path(MODEL_DIR, "model.safetensors").is_file():
        return json.dumps({"error": f"no {MODEL_DIR}: run --action prepare first", "used_s": CONTEST_START_S})
    spec_file = ""
    if spec:
        p = Path(CONTEST, run, "entries.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(spec, encoding="utf-8")
        spec_file = str(p)

    def report(rc: int | None) -> dict:
        try:
            return {"status": json.loads(Path(out, "status.json").read_text(encoding="utf-8"))}
        except (OSError, ValueError):
            return {"status": None}

    return _deadline_job(f"contest-{mode}",
                         lambda kill: contest_cmd(mode, run, args, kill - CONTEST_GRACE_S, spec_file, backend), max_s,
                         run_id, out, CONTEST_START_S, need_val=False, report=report, tmp_hf_home=False)


def corpus_dir(path: str, default: str = F022) -> str:
    """A corpus folder on the Volume (corpus-render / corpus-fetch / corpus-merge --out, a merge source): path, or
    default when empty; a plain folder right under DATA (letters, digits, . _ -), not one of CORPUS_RESERVED.
    ValueError otherwise."""
    p = posixpath.normpath(path or default) if (path or default).startswith("/") else (path or default)
    name = p[len(DATA) + 1:] if p.startswith(DATA + "/") else ""
    if (not name or "/" in name or len(name) > 64 or not name[0].isalnum() or not name.isascii()
            or not all(c.isalnum() or c in "._-" for c in name) or name in CORPUS_RESERVED):
        raise ValueError(f"{path!r}: give a corpus folder right under {DATA} (letters, digits, . _ -; not "
                         f"{', '.join(CORPUS_RESERVED)}), e.g. {DATA}/f022corpus2")
    return p


def corpus_local(out: str) -> Path:
    """corpus-fetch's local folder of a corpus: ./voxcpm-corpus/f022/ for F022 (as before), else /<its folder name>/."""
    return CORPUS_DIR / ("f022" if out == F022 else PurePosixPath(out).name)


def corpus_cmd(extra: list[str], deadline: float, backend: str = "nanovllm", out: str = F022) -> list[str]:
    """render_corpus.py command line: --args first, then what the wrapper owns (render_corpus.WRAPPER_OPTS; main
    refuses them in --args): the corpus dir (main's --out, default F022), the model, the script's own deadline, the
    throughput log the estimates read, the engine; Antalia's held-out sentences always excluded (a further
    --exclude-manifest adds)."""
    if backend not in CONTEST_BACKENDS:
        raise ValueError(f"--backend {backend!r}: one of {', '.join(CONTEST_BACKENDS)}")
    return ["python", "tts/voxcpm/render_corpus.py", *extra, "--out", corpus_dir(out), "--model", MODEL_DIR,
            "--deadline", f"{deadline:.0f}", "--throughput-log", f"{CONTEST}/throughput.jsonl", "--backend", backend,
            "--exclude-manifest", f"{CORPUS}/voxcpm_val.jsonl"]


@app.function(image=contest_image, gpu=GPU, cpu=CORPUS_CPU, memory=CORPUS_MEM, volumes={DATA: volume},
              timeout=CORPUS_TIMEOUT_S, scaledown_window=IDLE_S, secrets=SECRETS)
def corpus(args: list[str], max_s: float, run_id: str = "", backend: str = "nanovllm", out: str = F022) -> str:
    """Corpus render call: tts/voxcpm/render_corpus.py (corpus_cmd) -> out (default F022) under a hard deadline of the
    whole call, as contest (_deadline_job: ledger <run_id>.json, process group killed at the deadline, watchdog); the
    script gets its own deadline CONTEST_GRACE_S earlier, so it stops a wave in time to write the manifest and report,
    and the next call continues. Returns {"rc", "used_s", "status": the script's status.json, "corpus":
    corpus_summary} or {"error", "used_s"}."""
    try:
        corpus_cmd(args, 0.0, backend, out)
    except ValueError as e:  # main checks this before the call
        return json.dumps({"error": str(e), "used_s": CORPUS_START_S})
    if not Path(MODEL_DIR, "model.safetensors").is_file():
        return json.dumps({"error": f"no {MODEL_DIR}: run --action prepare first", "used_s": CORPUS_START_S})

    def report(rc: int | None) -> dict:
        try:
            st = json.loads(Path(out, "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = None
        return {"status": st, "corpus": corpus_summary(Path(out), report=rc == 0)}

    return _deadline_job("corpus-render", lambda kill: corpus_cmd(args, kill - CONTEST_GRACE_S, backend, out), max_s,
                         run_id, out, CORPUS_START_S, need_val=True, report=report, tmp_hf_home=False)


def merge_cmd(sources: list[str], out: str = MERGE_OUT, allow_incomplete: bool = False, dry: bool = False,
              allow_unaudited: bool = False, instructions: str = "en") -> list[str]:
    """render_corpus.py --merge command line (ValueError for a folder corpus_dir refuses)."""
    return ["python", "tts/voxcpm/render_corpus.py", "--out", corpus_dir(out, MERGE_OUT), "--merge",
            *(corpus_dir(s) for s in sources), *(["--allow-incomplete"] if allow_incomplete else []),
            *(["--allow-unaudited"] if allow_unaudited else []),
            *(["--instructions", instructions] if instructions != "en" else []), *(["--dry-run"] if dry else [])]


@app.function(image=image, cpu=MERGE_CPU, memory=MERGE_MEM, volumes={DATA: volume}, timeout=MERGE_TIMEOUT_S,
              scaledown_window=IDLE_S)
def corpus_merge(sources: list[str], out: str = MERGE_OUT, allow_incomplete: bool = False,
                 allow_unaudited: bool = False, instructions: str = "en") -> str:
    """corpus-merge: render_corpus.py --merge on the Volume (it refuses while a source's corpus-render call is going,
    a source whose mix is not complete or whose instructed clips with a c were never c-checked, a WAV that is not
    there, an instruction wording instructions_tr.json lacks when instructions is not en); the Volume committed.
    Returns {"rc", "used_s", "corpus": corpus_summary(out)}."""
    t0 = time.time()
    try:
        cmd = merge_cmd(sources, out, allow_incomplete, False, allow_unaudited, instructions)
    except ValueError as e:  # main checks this before the call
        return json.dumps({"rc": 2, "error": str(e), "used_s": MERGE_START_S})
    rc = _run(cmd, check=False)
    if rc == 0:
        volume.commit()
    return json.dumps({"rc": rc, "used_s": round(time.time() - t0 + MERGE_START_S),
                       "corpus": corpus_summary(Path(out))}, indent=1, ensure_ascii=False)


def audit_cmd(out: str = F022, dry: bool = False) -> list[str]:
    """render_corpus.py --audit-c command line (ValueError for a folder corpus_dir refuses)."""
    return ["python", "tts/voxcpm/render_corpus.py", "--out", corpus_dir(out), "--audit-c",
            *(["--dry-run"] if dry else [])]


@app.function(image=contest_image, gpu=AUDIT_GPU, cpu=AUDIT_CPU, memory=AUDIT_MEM, volumes={DATA: volume},
              timeout=AUDIT_TIMEOUT_S, scaledown_window=IDLE_S, secrets=SECRETS)
def corpus_audit_c(out: str = F022) -> str:
    """corpus-audit-c: render_corpus.py --audit-c on the Volume (it refuses while a corpus-render call is going; the
    IPA model's weights go to the HF cache on the Volume, HF_HOME); the Volume committed. Returns {"rc", "used_s",
    "audit": {rows, failed, ids}}."""
    t0 = time.time()
    try:
        cmd = audit_cmd(out)
    except ValueError as e:  # main checks this before the call
        return json.dumps({"rc": 2, "error": str(e), "used_s": AUDIT_START_S})
    rc = _run(cmd, check=False)
    rows = []
    if rc == 0:
        volume.commit()
        p = Path(out, "c_audit.jsonl")
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []
    bad = [r.get("id") for r in rows if r.get("ok") is False]
    return json.dumps({"rc": rc, "used_s": round(time.time() - t0 + AUDIT_START_S),
                       "audit": {"rows": len(rows), "failed": len(bad), "ids": bad}}, indent=1, ensure_ascii=False)


def add_reviewed_cmd(approved: str, dry: bool = False) -> list[str]:
    """render_corpus.py --add-reviewed command line on F022 (approved: the uploaded file's path in the container)."""
    return ["python", "tts/voxcpm/render_corpus.py", "--out", F022, "--add-reviewed", approved,
            *(["--dry-run"] if dry else [])]


@app.function(image=image, cpu=REVIEWED_CPU, memory=REVIEWED_MEM, volumes={DATA: volume},
              timeout=REVIEWED_TIMEOUT_S, scaledown_window=IDLE_S)
def corpus_add_reviewed(approved: str, name: str) -> str:
    """corpus-add-reviewed: the approved file's text (uploaded from the local repo) -> /tmp/<name>, then
    render_corpus.py --add-reviewed on F022 (it refuses while a corpus-render call is going: status.json's phase,
    state.json newer than the report); the Volume committed. Returns {"rc", "used_s", "corpus": corpus_summary}."""
    t0 = time.time()
    src = Path("/tmp") / Path(name).name
    src.write_text(approved, encoding="utf-8")
    rc = _run(add_reviewed_cmd(str(src)), check=False)
    if rc == 0:
        volume.commit()
    return json.dumps({"rc": rc, "used_s": round(time.time() - t0 + REVIEWED_START_S),
                       "corpus": corpus_summary(Path(F022))}, indent=1, ensure_ascii=False)


def train_cmd(steps: int, hours: float, exp_name: str, extra: list[str]) -> list[str]:
    """train_lora.py command line: --data CORPUS unless --args give another (argparse: the last one wins); the options
    the wrapper owns (WRAPPER_OPTS: model, experiment, steps, time limit = the budget) come after --args, so --args can
    never lift the budget (main refuses them too)."""
    return ["python", "tts/voxcpm/train_lora.py", "--data", CORPUS, *extra, "--model", MODEL_DIR,
            "--exp", f"{DATA}/exp/{exp_name}", "--steps", str(steps), "--hours", f"{hours:.4f}"]


def _train(steps: int, hours: float, exp_name: str, extra: list[str], check: bool = True) -> int:
    return _run(train_cmd(steps, hours, exp_name, extra), check=check)


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, volumes={DATA: volume}, timeout=24 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30.0), scaledown_window=IDLE_S, secrets=SECRETS)
def train(hours: float, steps: int, exp_name: str, extra: list[str], run_id: str) -> None:
    ledger = Path(DATA) / "modal_ledger" / f"{run_id}.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    used = json.loads(ledger.read_text())["used_h"] if ledger.exists() else 0.0
    left = hours - used - 0.05  # start-up, final save, volume commit
    if left < 0.05:
        print(f"[modal] budget of run {run_id} used ({used:.2f} of {hours:.2f} h); nothing to do")
        return
    t0, stop = time.time(), threading.Event()

    def bookkeeping():
        while not stop.wait(60):
            ledger.write_text(json.dumps({"used_h": used + (time.time() - t0) / 3600, "limit_h": hours}))

    threading.Thread(target=bookkeeping, daemon=True).start()
    try:
        with _PeriodicCommit(300):
            rc = _train(steps, left, exp_name, extra, check=False)
    finally:
        stop.set()
        ledger.write_text(json.dumps({"used_h": used + (time.time() - t0) / 3600, "limit_h": hours}))
        volume.commit()
    if rc == 2:  # train_lora.py refused (arguments, --init-lora, resume guard): a retry would pay to fail again
        print("[modal] train_lora.py refused the run (exit code 2, see above); not retried", flush=True)
    elif rc not in (0, 75):
        raise RuntimeError(f"train_lora.py exit code {rc}")


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, volumes={DATA: volume}, timeout=3600,
              scaledown_window=IDLE_S, secrets=SECRETS)
def probe(minutes: float = 10.0, extra: list[str] | None = None) -> str:
    """A short real training run: seconds per optimizer step and peak GPU memory. extra: train_lora.py options, e.g.
    ["--data", ALANIA] (its own checkpoint cadence, every 25 steps, always wins)."""
    exp = Path(DATA) / "exp" / f"probe-{GPU.lower()}"
    shutil.rmtree(exp, ignore_errors=True)
    t0, peak, done = time.time(), [0], threading.Event()

    def sample_gpu_memory():  # the trainer runs in a subprocess: sample nvidia-smi
        while not done.wait(5):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True).stdout
                peak[0] = max(peak[0], max(int(x) for x in out.split()))
            except Exception:
                pass

    threading.Thread(target=sample_gpu_memory, daemon=True).start()
    with _PeriodicCommit(600):
        _train(1000, minutes / 60, exp.name, [*(extra or []), "--save-every", "25"])
    done.set()
    # seconds per optimizer step from the times the periodic checkpoints were written
    saves = sorted((int(p.name[5:]), p.stat().st_mtime) for p in (exp / "ckpt").glob("step_*") if p.name[5:].isdigit())
    saves = [s for s in saves if s[0] > 0 and s[0] % 25 == 0]
    s_per_step = (saves[-1][1] - saves[0][1]) / (saves[-1][0] - saves[0][0]) if len(saves) >= 2 else None
    log = exp / "ckpt" / "train.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    return json.dumps({"gpu": GPU, "wall_s": round(time.time() - t0, 1), "steps_saved": [s[0] for s in saves],
                       "s_per_step": s_per_step, "gpu_mem_peak_GiB": round(peak[0] / 1024, 1),
                       "log_tail": text.splitlines()[-6:]})


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, volumes={DATA: volume}, timeout=EVAL_TIMEOUT_S,
              scaledown_window=IDLE_S, secrets=SECRETS)
def evaluate(ckpts: str, exp_name: str = EXP, limit: int = 0, extra: list[str] | None = None, grid: str = "",
             max_s: float = 0.0, run_id: str = "") -> str:
    """synth_eval.py for each checkpoint (x each timesteps:cfg point of `grid`, out dir eval/<ckpt>@ts<ts>-cfg<cfg>).
    "base" = VoxCPM2 cloning an Antalia training clip; "trendyol" = Trendyol/Trendyol-TTS (VoxCPM2 + a LoRA on
    ~20 h of private Turkish speech, MIT) as a reference point, at its recommended 16 timesteps; ours use VoxCPM2's
    default 10. Grid points come last, so they override both and --args. --limit / --args need a grid (main refuses
    them otherwise): they would replace eval/<ckpt>, the checkpoint's full evaluation.
    Each run writes eval/<name>.partial, which becomes eval/<name> only once synth_eval finished (an older eval/<name>
    is kept as <name>.replaced-<UTC time>), so a killed or failed run never touches an earlier evaluation.
    max_s > 0: hard deadline for the whole call, incl. start-up, the idle window and attempts that a preemption cut
    short (ledger <run_id>.json on the Volume, which also keeps finished runs from running twice). A run starts only
    if its estimate (run_seconds, scaled up when a run of this call took longer than estimated) fits before the
    deadline, else it is "skipped (budget)"; one still going at the deadline is killed anyway; a watchdog ends the
    container if anything else hangs. Returns {"used_s", "runs": {name: {rtf, <set>: WER, utmos: {<set>: mean}, min}
    | reason}}."""
    t0, extra = time.time(), extra or []
    runs = eval_runs(ckpts, grid)
    if overwrites_full_eval(limit, extra, grid):  # main refuses this before the call; never overwrite a full eval
        return json.dumps({"used_s": EVAL_START_S, "runs": {n: "refused: --limit / --args need --grid"
                                                             for n, _, _ in runs}})
    exp = Path(DATA) / "exp" / exp_name
    ref = json.loads(Path(CORPUS, "voxcpm_train.jsonl").read_text(encoding="utf-8").splitlines()[0])["audio"]
    ledger = Path(DATA) / "modal_ledger" / f"{run_id}.json" if run_id else None
    state = json.loads(ledger.read_text()) if ledger and ledger.exists() else {"used_s": 0.0, "done": {}}
    used0 = state["used_s"] + EVAL_START_S  # this attempt's start-up counts as used
    end = t0 + max_s - used0 - IDLE_S if max_s > 0 else math.inf  # the container must be gone by then
    out, live, stop, lock, slow = {}, [], threading.Event(), threading.Lock(), 1.0  # slow: >= actual / estimated

    def save_ledger():  # main thread and bookkeeping thread
        if not ledger:
            return
        with lock:
            try:
                ledger.parent.mkdir(parents=True, exist_ok=True)
                tmp = ledger.with_suffix(".tmp")
                tmp.write_text(json.dumps({**state, "used_s": used0 + time.time() - t0, "max_s": max_s}))
                os.replace(tmp, ledger)
            except Exception as e:  # keep evaluating; the deadline itself does not depend on the ledger
                print(f"[eval] ledger not written: {e}", flush=True)

    def bookkeeping():
        while not stop.wait(30):
            save_ledger()

    def hard_stop():  # only if something hangs past the deadline outside a run (download, commit)
        print(f"[eval] hard deadline ({max_s:.0f} s budget) reached: killing everything", flush=True)
        for p in list(live):
            _kill(p, wait=False)
        os._exit(124)

    # >= 15 s: a restart that finds the budget used up reports that instead of dying (costs < 2 cents)
    watchdog = threading.Timer(max(end - time.time(), 15.0), hard_stop) if end < math.inf else None
    if watchdog:
        watchdog.daemon = True
        watchdog.start()
    threading.Thread(target=bookkeeping, daemon=True).start()
    try:
        with _PeriodicCommit(60):
            try:
                for name, ck, point in runs:
                    if name in state["done"]:
                        out[name] = state["done"][name]
                        print(f"[eval] {name}: finished before a restart, not run again", flush=True)
                        continue
                    opts = synth_opts(ck, limit, extra, point)
                    est = run_seconds(opts)
                    if (left := end - EVAL_TAIL_S - time.time()) < est * slow:  # it could not finish: costs nothing
                        out[name] = "skipped (budget)"
                        print(f"[eval] {name}: skipped, needs ~{est * slow / 60:.0f} min, {max(left, 0) / 60:.1f} left",
                              flush=True)
                        continue
                    model = MODEL_DIR
                    if ck == "trendyol":
                        from huggingface_hub import snapshot_download

                        model = snapshot_download("Trendyol/Trendyol-TTS", local_dir=f"{DATA}/models/Trendyol-TTS")
                    own = {"trendyol": [], "base": ["--ref", ref]}.get(ck, ["--lora", str(exp / "ckpt" / ck)])
                    dst, tmp = exp / "eval" / name, exp / "eval" / f"{name}.partial"
                    shutil.rmtree(tmp, ignore_errors=True)  # left by a killed or failed attempt
                    t = time.time()
                    rc = _run_until(["python", "tts/voxcpm/synth_eval.py", "--model", model, "--val",
                                     f"{CORPUS}/voxcpm_val.jsonl", "--out", str(tmp), *own, *opts],
                                    end - EVAL_TAIL_S, live)
                    if rc is None:
                        out[name] = (f"killed at the budget deadline after {time.time() - t:.0f} s "
                                     f"(eval/{name} untouched)")
                    elif rc == 0 and (tmp / "summary.json").exists():  # exit 0: final summary (WER, then UTMOS)
                        s, took = json.loads((tmp / "summary.json").read_text()), time.time() - t
                        slow = max(slow, took / est)
                        out[name] = {"rtf": s["rtf"], **_scores(s), "min": round(took / 60, 1)}
                        try:
                            _promote(tmp, dst)
                        except Exception as e:  # the scores are in the .partial dir all the same
                            print(f"[eval] {name}: finished, but left in eval/{name}.partial: {e}", flush=True)
                            out[name]["dir"] = tmp.name
                    else:
                        out[name] = f"failed (exit code {rc}, see log; eval/{name} untouched)"
                    if rc is not None:
                        with lock:  # the bookkeeping thread serializes `state`
                            state["done"][name] = out[name]
                    save_ledger()
                    try:  # each finished run survives a later kill
                        volume.commit()
                    except Exception as e:
                        print(f"[modal] volume commit failed: {e}", flush=True)
            finally:
                stop.set()
                save_ledger()  # before _PeriodicCommit's final commit
    finally:
        if watchdog:
            watchdog.cancel()
    return json.dumps({"used_s": round(used0 + time.time() - t0), "runs": out}, indent=1, ensure_ascii=False)


@app.function(image=image, cpu=1, memory=2048, volumes={DATA: volume}, timeout=300)
def status(exp_name: str = EXP) -> str:
    exp = Path(DATA) / "exp" / exp_name
    out = {"exp": str(exp), "exists": exp.exists(), "corpus": Path(CORPUS, "voxcpm_train.jsonl").exists(),
           "model": Path(MODEL_DIR, "model.safetensors").exists()}
    if (exp / "ckpt").exists():
        out["checkpoints"] = sorted(p.name for p in (exp / "ckpt").glob("step_*"))
        log = exp / "ckpt" / "train.log"
        if log.exists():
            out["train_log_tail"] = log.read_text(errors="replace").splitlines()[-5:]
    if (exp / "runs.jsonl").exists():
        out["runs"] = [json.loads(x) for x in (exp / "runs.jsonl").read_text().splitlines() if x.strip()]
    if (exp / "init_lora.json").exists():  # stage 2: the LoRA it started from (train_lora.py --init-lora)
        out["init_lora"] = json.loads((exp / "init_lora.json").read_text())
    for s in sorted((exp / "eval").glob("*/summary.json")) if (exp / "eval").exists() else []:
        if not s.parent.name.endswith(".partial") and ".replaced-" not in s.parent.name:  # unfinished / superseded
            out.setdefault("eval", {})[s.parent.name] = _scores(json.loads(s.read_text()))
    if (alania := corpus_summary(Path(ALANIA))) is not None:  # stage-1 corpus (prepare-alania)
        out["alania16k"] = alania
    if (human := corpus_summary(Path(HUMAN))) is not None:  # stage-1 corpus of real speech (prepare-human)
        out["human16k"] = human
    if (f022 := corpus_summary(Path(F022))) is not None:  # the designed-f022-s0 corpus (corpus-render)
        out["f022corpus"] = f022
    for d in (*MERGE_SOURCES[1:], MERGE_OUT):  # the second corpus (corpus-render --out) and their merge
        if (more := corpus_summary(Path(d))) is not None:
            out[Path(d).name] = more
    return json.dumps(out, indent=1, ensure_ascii=False)


def _download(path: str, size: int, out: Path) -> None:
    """Runs on the local machine: one Volume file -> out, whole or not at all (.part, size check, rename)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    try:
        with open(part, "wb") as f:
            for chunk in volume.read_file(path):
                f.write(chunk)
        if (got := part.stat().st_size) != size:
            raise RuntimeError(f"{path}: got {got} of {size} bytes")
        os.replace(part, out)
    finally:
        part.unlink(missing_ok=True)


def _has_utmos(summary: bytes) -> bool:
    try:
        return bool(json.loads(summary).get("utmos"))
    except (ValueError, AttributeError):
        return False


def _fetch(ckpts: list[str], exp_name: str, dest: Path, hf_repo: str) -> None:
    """Runs on the local machine: copy eval/<name>/ (WAVs + scores) from the Volume into dest/<name>/, then
    upload the score files (json/jsonl, a few KB) to a private Hugging Face repo with the local HF login.
    <name> is a checkpoint or a grid run (<ckpt>@ts<ts>-cfg<cfg>). Each file lands whole (_download), the score
    files after the WAVs. A local copy whose summary has UTMOS (e.g. rescored by score_utmos.py) is kept when the
    Volume's has none: it is the same run or an older one; rename the local dir to fetch the Volume's anyway.
    Experiments other than EXP go to dest/<exp_name>/ and <exp_name>/ in the repo (local_dir)."""
    root = local_dir(dest, exp_name)
    for ck in ckpts:
        remote, local = f"exp/{exp_name}/eval/{ck}", root / ck
        try:
            files = {PurePosixPath(e.path.lstrip("/")).relative_to(remote): e
                     for e in volume.listdir(remote, recursive=True) if e.type == modal.volume.FileEntryType.FILE}
            summ = files.get(PurePosixPath("summary.json"))
            if ((local / "summary.json").is_file() and _has_utmos((local / "summary.json").read_bytes())
                    and not (summ and _has_utmos(b"".join(volume.read_file(summ.path))))):
                print(f"[fetch] {ck}: kept the local copy: it has UTMOS scores, the Volume's has none "
                      f"(rename {local} to fetch it anyway)")
                continue
            last = {n: i + 1 for i, n in enumerate(SCORE_FILES)}  # WAVs, then rows.jsonl, then summary.json
            for rel, e in sorted(files.items(), key=lambda kv: last.get(str(kv[0]), 0)):
                _download(e.path, e.size, local.joinpath(*rel.parts))
        except Exception as e:  # not on the Volume, a broken connection, ...: the other runs still come
            print(f"[fetch] {remote}: {e}")
            continue
        print(f"[fetch] {len(files)} files -> {local}")
    if not hf_repo:
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(hf_repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=hf_repo, folder_path=str(root), allow_patterns=["*.json", "*.jsonl"],
                          **({} if root == dest else {"path_in_repo": exp_name}))
        print(f"[fetch] scores uploaded to https://huggingface.co/{hf_repo} (private)")
    except Exception as e:  # no HF login on this machine: the local copy is still there
        print(f"[fetch] Hugging Face upload skipped: {e}")


def _fetch_ckpt(ckpts: list[str], exp_name: str, dest: Path) -> None:
    """Runs on the local machine: the LoRA of ckpt/<ckpt>/ (CKPT_FILES only, no optimizer/scheduler state) from the
    Volume into dest/<ckpt>/ (dest/<exp_name>/<ckpt>/ for experiments other than EXP). Each file lands complete or not
    at all (_download)."""
    dest = local_dir(dest, exp_name)
    for ck in ckpts:
        remote = f"exp/{exp_name}/ckpt/{ck}"
        try:
            entries = {PurePosixPath(e.path).name: e for e in volume.listdir(remote)
                       if e.type == modal.volume.FileEntryType.FILE}
        except Exception as e:
            print(f"[fetch-ckpt] {remote}: {e}")
            continue
        missing = [n for n in CKPT_FILES if n not in entries]
        if any(n in missing for n in CKPT_FILES[:2]):
            print(f"[fetch-ckpt] {remote}: no {', '.join(missing)} on the Volume; skipped")
            continue
        for name in (n for n in CKPT_FILES if n in entries):
            _download(entries[name].path, entries[name].size, dest / ck / name)
        print(f"[fetch-ckpt] {ck} -> {dest / ck}: " + ", ".join(f"{n} ({entries[n].size / 1e6:.1f} MB)"
                                                             for n in CKPT_FILES if n in entries)
              + (f"; not on the Volume: {', '.join(missing)}" if missing else ""))


def _push_ckpt(ckpts: list[str], exp_name: str, src: Path) -> None:
    """Runs on the local machine: src/<ckpt>/ (CKPT_FILES) -> ckpt/<ckpt>/ on the Volume of the ACTIVE Modal
    profile, e.g. to evaluate step_0000600 on another workspace. Refuses if anything is there already. src/<exp_name>/
    for experiments other than EXP, as _fetch_ckpt writes them."""
    profile, src = getattr(modal.config, "_profile", "?"), local_dir(src, exp_name)
    for ck in ckpts:
        files = [src / ck / n for n in CKPT_FILES if (src / ck / n).is_file()]
        if not all((src / ck / n).is_file() for n in CKPT_FILES[:2]):
            print(f"[push-ckpt] {src / ck}: needs {' and '.join(CKPT_FILES[:2])} (--action fetch-ckpt); skipped")
            continue
        remote = f"exp/{exp_name}/ckpt/{ck}"
        try:
            existing = volume.listdir(remote)
        except modal.exception.NotFoundError:
            existing = []
        if existing:
            print(f"[push-ckpt] {remote} already exists on the Volume of Modal profile {profile} "
                  f"({len(existing)} entries); nothing uploaded")
            continue
        try:
            with volume.batch_upload(force=False) as batch:  # force=False: the server refuses to overwrite too
                for p in files:
                    batch.put_file(p, f"/{remote}/{p.name}")
        except FileExistsError as e:
            print(f"[push-ckpt] {remote}: refused, files exist: {e}")
            continue
        print(f"[push-ckpt] {', '.join(p.name for p in files)} -> {remote} (Modal profile {profile})")


def _vc():
    """voice_contest.py as a module, on the local machine (stdlib + trnorm at import): spec, texts, cost plans."""
    import importlib.util
    import sys

    if (mod := sys.modules.get("voice_contest")) is not None:
        return mod
    spec = importlib.util.spec_from_file_location("voice_contest", _HERE.with_name("voice_contest.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["voice_contest"] = mod
    spec.loader.exec_module(mod)
    return mod


class _VolFS:
    """voice_contest.resolve_ref's view of the Volume (paths under DATA), on the local machine."""

    @staticmethod
    def _rel(p: str) -> str | None:
        p = p.rstrip("/")
        return p[len(DATA) + 1:] if p.startswith(DATA + "/") else None

    def kind(self, p: str) -> str | None:
        if not (rel := self._rel(p)):
            return None
        parent, _, name = rel.rpartition("/")
        e = _vol_files(parent or "/").get(name)
        return None if e is None else "dir" if e.type == modal.volume.FileEntryType.DIRECTORY else "file"

    def listdir(self, p: str) -> list[str]:
        return sorted(_vol_files(rel)) if (rel := self._rel(p)) else []

    def read_text(self, p: str) -> str | None:
        if not (rel := self._rel(p)):
            return None
        try:
            return b"".join(volume.read_file(rel)).decode("utf-8")
        except Exception:  # not there
            return None


def _vol_rows(rel: str) -> list[dict]:
    """The rows of a jsonl file on the Volume; [] when it is not there."""
    text = _VolFS().read_text(f"{DATA}/{rel}") or ""
    out = []
    for x in text.splitlines():
        try:
            out.append(json.loads(x))
        except ValueError:  # a line cut short
            pass
    return [r for r in out if isinstance(r, dict)]


def _vol_wavs(rel: str) -> set[str]:
    """<sub path without .wav> of the finished WAVs (more than a header) under a Volume dir; empty when not there."""
    try:
        entries = volume.listdir(rel, recursive=True)
    except Exception:  # NotFoundError
        return set()
    return {str(PurePosixPath(p).relative_to(rel).with_suffix("")) for e in entries
            if e.type == modal.volume.FileEntryType.FILE and (p := e.path.lstrip("/")).endswith(".wav") and e.size > 44}


def _without_opt(args: list[str], opt: str) -> list[str]:
    """args without `opt V` / `opt=V` (or a prefix argparse expands to opt)."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif _opt_args([a], opt):
            skip = "=" not in a
        else:
            out.append(a)
    return out


def _contest_entries(spec_path: str, run: str, vc, screen: str = "") -> list[dict]:
    """A render's entries from --spec (a local JSON file; default voice_contest.DEFAULT_SPEC; screen: render --screen)
    resolved on the Volume: discovered refs (with a complete sources.json), screen winners, arms; each ref / sim_ref a
    clip there, each lora a checkpoint there. An entry whose files are not there is left out (printed)."""
    spec = vc.DEFAULT_SPEC
    if spec_path:
        try:
            spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise SystemExit(f"--spec {spec_path}: {e}")

    def list_refs() -> dict[str, list[str]]:
        return {n: sorted(_vol_files(f"contest/refs/{n}")) for n, e in _vol_files("contest/refs").items()
                if e.type == modal.volume.FileEntryType.DIRECTORY}

    entries = vc.resolve_spec(vc.with_screen(spec, screen), run, list_refs,
                              lambda r: _vol_json(f"contest/{r}/screen/entries.json"), f"{CONTEST}/refs",
                              warn=lambda m: print(f"[modal] {m}"),
                              read_sources=lambda d: _vol_json(f"contest/refs/{d}/sources.json"))
    out, fs = [], _VolFS()
    for e in entries:
        try:
            out.append(vc.resolve_entry(e, fs, f"{CONTEST}/refs"))
        except ValueError as err:
            print(f"[modal] entry {e['name']} left out: {err}")
    return out


def _contest(action: str, run: str, spec_path: str, extra: list[str], max_usd: float,
             backend: str = "nanovllm") -> None:
    """contest-screen / contest-render on the local machine: check --args with voice_contest.py's parser, plan the call
    (estimates per entry on its engine: Nano-vLLM's measured speeds and a server start per LoRA, the voxcpm pool's;
    the measured throughput when there is one), refuse or skip what does not fit, print "at most $X", run the GPU
    call, fetch the results to ./voxcpm-contest/<run>/."""
    vc, mode = _vc(), action.split("-", 1)[1]
    try:
        contest_out(mode, run)
        contest_cmd(mode, run, [], 0.0, "", backend)
    except ValueError as e:
        raise SystemExit(str(e))
    if bad := _opt_args(extra, *vc.WRAPPER_OPTS):
        raise SystemExit(f"--args may not set {', '.join(bad)}: {action} sets the model, the output dir (--run), the "
                         "deadline (--max-usd), the throughput log, the backend (--backend), the entry spec (--spec) "
                         f"and the references ({CONTEST}/refs, contest-refs-push)")
    try:
        ns = vc.build_parser().parse_args([mode, *extra, "--out", "x"])
    except SystemExit as e:
        if e.code in (0, None):  # --help
            raise
        raise SystemExit(f"voice_contest.py would refuse --args \"{' '.join(extra)}\" (see above); nothing started")
    for opt in ("descriptions", "texts"):
        if (p := getattr(ns, opt, None)) and Path(p).is_absolute():
            raise SystemExit(f"--{opt} {p}: give a path inside the repo (the container has it at {REPO})")
    ram, have = vc.pool_ram_gb(ns.workers, ns.load_parallel), CONTEST_RAM_USE * CONTEST_MEM / 1024
    if ram > have:  # an OOM kill mid-load costs the whole budget for nothing
        raise SystemExit(f"--workers {ns.workers} with --load-parallel {ns.load_parallel} needs ~{ram:.0f} GiB of RAM "
                         f"while loading (voice_contest.pool_ram_gb), over {have:.0f} of the container's "
                         f"{CONTEST_MEM // 1024} GiB: lower --load-parallel or --workers; nothing started")
    ns.model, ns.backend = MODEL_DIR, backend  # what contest_cmd passes: the settings the container records
    rate = GPU_PRICE[GPU] + CONTEST_SIDE_PRICE
    budget = max_usd or CONTEST_MAX_USD
    max_s = min(usd_to_seconds(budget, rate), CONTEST_TIMEOUT_S)
    overhead = CONTEST_START_S + ALANIA_TAIL_S + IDLE_S + CONTEST_GRACE_S  # start-up, kill tail, idle, the grace
    if max_s < overhead + ALANIA_MIN_S:
        raise SystemExit(f"--max-usd {budget:g} buys {max_s:.0f} s; {action} needs at least "
                         f"${math.ceil((overhead + ALANIA_MIN_S) / 3600 * rate * 100) / 100:.2f}")
    measured = _vol_rows("contest/throughput.jsonl")
    per = vc.wall_per_audio(ns.timesteps, ns.workers, measured)
    how = (("measured" if any(m.get("backend", "voxcpm") == "voxcpm" and m.get("timesteps") == ns.timesteps
                              for m in measured) else "assumed")
           + f" throughput of the official API: {per:.2f} wall s per audio s with {ns.workers} workers")
    if backend == "nanovllm":
        how = vc.nano_speed_note(ns.timesteps, measured) + (f"; LoRA fallbacks: {how}" if mode == "render" else "")
    spec_json, cmd = "", list(extra)
    try:
        if mode == "screen":
            descs = vc.load_descriptions(vc.repo_path(ns.descriptions))[: ns.limit or None]
            seeds = vc.parse_ints(ns.seeds)
            clone_texts = [t.strip() for t in ns.clone_texts.split("|") if t.strip()]
            have = _vol_wavs(f"contest/{run}/screen/cand/wav")
            old = _vol_json(f"contest/{run}/screen/settings.json")
            if have and old and (diff := {k: f"{old.get(k)} -> {v}" for k, v in vc.screen_settings(ns).items()
                                          if old.get(k) != v}):  # the container would refuse it (check_settings)
                raise ValueError(f"its candidates on the Volume were made otherwise ({diff}): continue it with the "
                                 f"same settings (e.g. --backend {old.get('backend')}) or screen under a new --run")
            todo = sum(f"{d['id']}-s{s}" not in have for d in descs for s in seeds)
            top = vc.clone_top(ns.top, len(descs), ns.per_desc, len(seeds))
            need = vc.plan_screen(todo, top, clone_texts, ns.timesteps, ns.workers, ns.screen_text, measured,
                                  len(_vol_wavs(f"contest/{run}/screen/clone/wav")), backend)
            print(f"[modal] contest-screen {run}: {len(descs)} descriptions x {len(seeds)} seeds ({todo} still to "
                  f"make), clone test {top} x {len(clone_texts)}, {ns.timesteps} timesteps on {GPU} ({how}): "
                  f"~{need / 60:.0f} min + {overhead / 60:.0f} min start / stop")
            if need > max_s - overhead:
                all_usd = math.ceil((need + overhead) / 3600 * rate * 10) / 10
                if not max_usd:  # the default budget never runs a screen it cannot finish
                    raise SystemExit(f"[modal] the default ${CONTEST_MAX_USD:g} does not cover it; --max-usd "
                                     f"{all_usd:g} covers it (or a smaller --max-usd: it stops there and the same "
                                     "command continues)")
                print(f"[modal] does not fit in --max-usd {budget:g} (~${all_usd:g} needed): it stops at the deadline, "
                      "the same command continues")
            expected = min(need + overhead, max_s)
        else:
            entries = _contest_entries(spec_path, run, vc, ns.screen)
            cmd = _without_opt(cmd, "--screen")  # the spec sent holds the screen winners already
            if ns.only:
                want = [n.strip() for n in ns.only.split(",") if n.strip()]
                if missing := set(want) - {e["name"] for e in entries}:
                    raise ValueError(f"--only: no entry {', '.join(sorted(missing))}")
                entries = [e for e in entries if e["name"] in want]
                cmd = _without_opt(cmd, "--only")  # the spec sent holds only these
            if not entries:
                raise ValueError("no entry to render")
            rows = vc.render_texts(ns)  # --limit, --ids, --seeds
            done = {e["name"]: _vol_wavs(f"contest/{run}/{e['name']}/wav") for e in entries}
            settings, refused, engines, fs = vc.render_settings(ns), [], {}, _VolFS()
            for e in entries:  # each entry's engine (a LoRA Nano-vLLM cannot load: the official API), as the container
                old = _vol_json(f"contest/{run}/{e['name']}/entry.json")
                engines[e["name"]], why = vc.entry_backend(backend, e, vc.lora_config(e.get("lora"), fs.read_text),
                                                           old)
                if why:
                    print(f"[modal] {e['name']}: {why}: the official API makes it (voxcpm, --workers {ns.workers})")
                # the container refuses these (setup_entry): leave them out before any GPU starts
                if done[e["name"]] and (diff := vc.entry_conflict(old, e, vc.entry_settings(settings, e,
                                                                                             engines[e["name"]]))):
                    print(f"[modal] {e['name']}: left out, its clips in {run} were made otherwise ({diff})")
                    refused.append(e["name"])
            if refused:
                entries = [e for e in entries if e["name"] not in refused]
                hint = (f"render them under a new --run (finals: --run {run}-ts{ns.timesteps} --args \"--timesteps "
                        f"{ns.timesteps} --screen {run} --only ...\")")
                if not entries:
                    raise ValueError(f"every entry was made with other settings in {run}: {hint}")
                print(f"[modal] {len(refused)} entries left out: {hint}")
            plan = vc.plan_render(entries, rows, done, ns.timesteps, ns.workers, max_s - overhead, measured, backend,
                                  engines, not ns.no_score)
            print(f"[modal] contest-render {run}: {len(entries)} entries x {len(rows)} texts, {ns.timesteps} timesteps "
                  f"on {GPU} ({how}); rough minutes:")
            for e, p in zip(entries, plan):
                print(f"  {p['name']:34s} {e['kind']:8s} {p['todo']:3d}/{p['clips']} clips to make "
                      f"~{p['est_s'] / 60:5.1f}" + ("" if p["engine"] == backend else f"  on {p['engine']}")
                      + ("" if p["fits"] else "  -> skipped: does not fit in the budget"))
            fixed = (0 if ns.no_score else vc.SCORE_LOAD_S) + vc.SUMMARY_S + overhead
            print(f"  (+ {fixed / 60:.0f} min fixed: start-up, {'' if ns.no_score else 'scorer loads, '}summaries, stop)")
            keep = [e for e, p in zip(entries, plan) if p["fits"]]
            if skipped := [p["name"] for p in plan if not p["fits"]]:
                all_s = fixed + sum(p["est_s"] for p in plan)
                hint = (f"--max-usd {math.ceil(all_s / 3600 * rate * 10) / 10:g} covers all {len(plan)}"
                        if all_s <= CONTEST_TIMEOUT_S else "they do not all fit in one call (3 h): split them "
                        "(--args \"--only a,b\")")
                if not keep:
                    raise SystemExit(f"[modal] no entry fits in ${budget:g}; {hint}")
                if not max_usd:  # the default budget never drops entries silently
                    raise SystemExit(f"[modal] the default ${CONTEST_MAX_USD:g} covers {len(keep)} of {len(plan)} "
                                     f"entries; {hint}, or --max-usd {CONTEST_MAX_USD:g} renders only those")
                print(f"[modal] {len(skipped)} entries skipped (they cost nothing): {', '.join(skipped)} ({hint})")
            if not ns.rescore and all(p["todo"] == 0 for p in plan if p["fits"]) and (ns.no_score or all(
                    (_vol_json(f"contest/{run}/{e['name']}/summary.json") or {}).get("complete") for e in keep)):
                print(f"[modal] contest-render {run}: every entry is made and scored already; no GPU started")
                _contest_fetch(run, CONTEST_DIR)
                return
            spec_json = json.dumps({"entries": keep, "refs": False, "screen": []}, ensure_ascii=False)
            expected = min(fixed + sum(p["est_s"] for p in plan if p["fits"]), max_s)
    except ValueError as e:
        raise SystemExit(f"[modal] {action}: {e}")
    print(f"[modal] {action} on {GPU} + {CONTEST_CPU} CPU cores + {CONTEST_MEM // 1024} GiB: at most "
          f"${max_s / 3600 * rate:.2f} (--max-usd {budget:g}): hard limit {max_s / 60:.0f} min at ${rate:.2f}/h incl. "
          f"CPU/RAM; expected ~${expected / 3600 * rate:.2f}"
          + (f"; voice_contest.py args: {' '.join(cmd)}" if cmd else ""))
    if ns.dry_run:  # the plan above is all a dry run needs: no GPU
        print("[modal] --dry-run: nothing started")
        return
    try:
        res = json.loads(contest.remote(mode, run, cmd, max_s, f"contest-{mode}-{run}-{int(time.time())}", spec_json,
                                        backend))
    except Exception as e:  # e.g. the watchdog's hard stop: what is done is on the Volume all the same
        print(f"[modal] {action} ended with an error: {e}; what is done is on the Volume, the same command continues")
    else:
        st = res.get("status") or {}
        for k, v in st.items():
            if not (k.startswith("synth_") and isinstance(v, dict)):
                continue
            if v.get("backend") == "nanovllm":
                print(f"[modal] {k[6:]}: {v.get('done')}/{v.get('jobs')} clips, {v.get('audio_s')} s audio in "
                      f"{v.get('wall_s')} s on {len(v.get('groups') or [])} Nano-vLLM servers "
                      f"({v.get('audio_per_wall')} audio s per GPU s; servers up {v.get('server_s')} s, peak GPU "
                      f"{v.get('gpu_mem_peak_mib')} MiB)"
                      f" {v.get('stopped') or ''}")
                continue
            print(f"[modal] {k[6:]}: {v.get('done')}/{v.get('jobs')} clips, {v.get('audio_s')} s audio in "
                  f"{v.get('wall_s')} s ({v.get('audio_per_wall')} audio s per wall s, {v.get('workers_ready')}/"
                  f"{v.get('workers')} workers, peak GPU {v.get('gpu_mem_peak_mib')} MiB, host "
                  f"{v.get('host_mem_peak_mib')} MiB) {v.get('stopped') or ''}")
        print(json.dumps({"rc": res.get("rc"), "error": res.get("error"), "phase": st.get("phase"),
                          "errors": st.get("errors"), **{k: st[k] for k in ("picked", "provisional", "unresolved",
                                                                             "complete", "fallback") if k in st}},
                         indent=1, ensure_ascii=False))
        used, rc, phase = res.get("used_s", 0), res.get("rc"), st.get("phase")  # used_s: incl. start-up per attempt
        note = ("" if "error" in res else f"; voice_contest.py failed (exit code {rc}), see the log above"
                if rc not in (0, None) else "; stopped at the deadline: the same command continues" if rc is None
                else f"; not finished (phase {phase!r}): the same command continues" if phase != "done" else "")
        print(f"[modal] GPU container ~{used / 60:.1f} min, ~${used / 3600 * rate:.2f}{note}")
    _contest_fetch(run, CONTEST_DIR)


def _contest_fetch(run: str, dest: Path, every: bool = False) -> None:
    """Runs on the local machine: /data/contest/<run>/ -> dest/<run>/ (clips, references, scores, leaderboard); a WAV
    already there with the same size is not fetched again, the score files come last. Of the screen's candidate clips
    only those of the clone test come, unless every."""
    remote = f"contest/{run}"
    try:
        files = [e for e in volume.listdir(remote, recursive=True) if e.type == modal.volume.FileEntryType.FILE]
    except Exception as e:  # not on the Volume, ...
        print(f"[contest-fetch] {remote}: {e}")
        return
    keep = None if every else set(((_vol_json(f"{remote}/screen/ranking.json") or {}).get("params") or {})
                                  .get("clone_test") or [])
    todo, skipped, have = [], 0, 0
    for e in files:
        rel = PurePosixPath(e.path.lstrip("/")).relative_to(remote)
        if rel.name.endswith((".part", ".tmp")):  # a write cut short
            continue
        if keep is not None and rel.parts[:3] == ("screen", "cand", "wav") and rel.stem not in keep:
            skipped += 1
            continue
        local = dest / run / Path(*rel.parts)
        if rel.suffix != ".json" and rel.suffix != ".jsonl" and local.is_file() and local.stat().st_size == e.size:
            have += 1
            continue
        todo.append((rel.suffix in (".json", ".jsonl", ".md"), str(rel), e, local))
    mb = sum(e.size for *_, e, _ in todo) / 1e6
    print(f"[contest-fetch] {len(todo)} files ({mb:.0f} MB) -> {dest / run}"
          + (f"; {have} already here" if have else "")
          + (f"; {skipped} screen candidates outside the clone test left on the Volume (--args --all)" if skipped
             else ""))
    for _, rel, e, local in sorted(todo, key=lambda t: t[:2]):
        try:
            _download(e.path, e.size, local)
        except Exception as err:  # a broken connection: the next fetch continues
            print(f"[contest-fetch] {rel}: {err}")
    if (dest / run / "leaderboard.md").is_file():
        print(f"[contest-fetch] {dest / run / 'leaderboard.md'}")


def _contest_refs_push(src: Path) -> None:
    """Runs on the local machine: each src/<voice>/ (fetch_refs.py's clips + sources.json) ->
    /data/contest/refs/<voice>/ on the Volume of the ACTIVE Modal profile. A folder without a clip or a complete
    sources.json (voice_contest.check_sources: licence, consent, attribution, reference-only use), with a file over
    REFS_PUSH_MAX_MB, or already on the Volume is refused, so a reference is never overwritten."""
    vc, profile = _vc(), getattr(modal.config, "_profile", "?")
    dirs = sorted(d for d in src.iterdir() if d.is_dir()) if src.is_dir() else []
    if not dirs:
        raise SystemExit(f"no {src}/<voice>/ folders: run tts/voxcpm/fetch_refs.py first")
    for d in dirs:
        files = [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in REFS_PUSH_EXT]
        try:
            bad = vc.check_sources(json.loads((d / "sources.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            bad = ["readable JSON object"]
        why = ("not a usable entry name" if not vc.NAME_RE.match(d.name) or d.name in vc.RESERVED
               else "no sources.json (licence + consent record)" if not (d / "sources.json").is_file()
               else f"sources.json lacks {', '.join(bad)}" if bad
               else "no audio clip" if not any(p.suffix.lower() in vc.AUDIO_EXT for p in files)
               else f"a file over {REFS_PUSH_MAX_MB} MB"
               if any(p.stat().st_size > REFS_PUSH_MAX_MB * 1e6 for p in files) else "")
        if why:
            print(f"[contest-refs-push] {d.name}: skipped, {why}")
            continue
        remote = f"contest/refs/{d.name}"
        if existing := _vol_files(remote):
            print(f"[contest-refs-push] {remote} already exists on the Volume of Modal profile {profile} "
                  f"({len(existing)} entries); nothing uploaded (a reference is never replaced: use a new folder name)")
            continue
        try:
            with volume.batch_upload(force=False) as batch:  # force=False: the server refuses to overwrite too
                for p in files:
                    batch.put_file(p, f"/{remote}/{p.name}")
        except FileExistsError as e:
            print(f"[contest-refs-push] {remote}: refused, files exist: {e}")
            continue
        print(f"[contest-refs-push] {', '.join(p.name for p in files)} -> {remote} (Modal profile {profile})")


def _rc():
    """render_corpus.py as a module, on the local machine (stdlib + trnorm at import): parser, pool, estimates."""
    import importlib.util
    import sys

    if (mod := sys.modules.get("render_corpus")) is not None:
        return mod
    _vc()  # its voice_contest import finds this one
    spec = importlib.util.spec_from_file_location("render_corpus", _HERE.with_name("render_corpus.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_corpus"] = mod
    spec.loader.exec_module(mod)
    return mod


def corpus_calls(work_s: float, first_s: float, per_call_s: float, call_s: float, wave_s: float) -> tuple[int, float]:
    """(calls, billed s) of the rest of a corpus (work_s of waves, first_s of pool building in the first call) in calls
    whose script works per_call_s each: each call bills call_s more (start-up, the script's start, finalize, idle),
    each call after the first wave_s more (its first wave starts a server and loads the scorers again) and so may take
    one call more."""
    calls = 1
    while work_s + first_s + (calls - 1) * wave_s > calls * max(per_call_s, 1.0):
        calls += 1
    return calls, work_s + first_s + calls * call_s + (calls - 1) * wave_s


def _corpus(extra: list[str], max_usd: float, backend: str = "nanovllm", out: str = F022) -> None:
    """corpus-render on the local machine: check --args with render_corpus.py's parser, refuse what the container
    would refuse (no textbank for the mix, RAM of a voxcpm pool, an --avoid-pool / --refs-from corpus not on the
    Volume), estimate the rest of the corpus in out from the Volume's pool, state and throughput.jsonl (Nano-vLLM's
    measured speed, the measured scoring time; render_corpus.estimate), print "at most $X" and what this call should
    make, run the GPU call, fetch the report and a sample."""
    rc = _rc()
    try:
        out = corpus_dir(out)
        corpus_cmd(extra, 0.0, backend, out)
    except ValueError as e:
        raise SystemExit(str(e))
    if bad := _opt_args(extra, *rc.WRAPPER_OPTS):
        raise SystemExit(f"--args may not set {', '.join(bad)}: corpus-render sets the output dir (--out, default "
                         f"{F022}), the model, the deadline (--max-usd), the throughput log and the backend "
                         "(--backend)")
    try:
        ns = rc.parse_args([*extra, "--out", "x"])
    except SystemExit as e:
        if e.code in (0, None):  # --help
            raise
        raise SystemExit(f"render_corpus.py would refuse --args \"{' '.join(extra)}\" (see above); nothing started")
    if ns.add_reviewed:
        raise SystemExit("--add-reviewed is its own action: --action corpus-add-reviewed (a CPU call, no GPU)")
    if ns.merge:
        raise SystemExit("--merge is its own action: --action corpus-merge (a CPU call, no GPU)")
    if ns.audit_c:
        raise SystemExit("--audit-c is its own action: --action corpus-audit-c (a small GPU, minutes)")
    for opt, val in (("--textbank", ns.textbank), ("--laughter-bank", ns.laughter_bank), ("--cwords", ns.cwords)):
        if Path(val).is_absolute():
            raise SystemExit(f"{opt} {val}: give a path inside the repo (the container has it at {REPO})")
    for opt, val in [*(("--avoid-pool", p) for p in ns.avoid_pool), *([("--refs-from", ns.refs_from)]
                                                                      if ns.refs_from else [])]:
        try:
            d = corpus_dir(_vol_path(val))
        except ValueError as e:
            raise SystemExit(f"{opt}: {e}")
        need = "texts.jsonl" if opt == "--avoid-pool" else "ref"
        if d == out or need not in _vol_files(d[len(DATA) + 1:]):
            raise SystemExit(f"{opt} {d}: " + ("--out itself" if d == out else f"no {need} there on the Volume")
                             + "; nothing started")
    tb_rows, tb_bad, _ = rc.load_textbank(rc.vc.repo_path(ns.textbank))
    mix = rc.parse_mix(ns.mix)
    if not tb_rows and (want := [c for c in rc.TB_CATEGORIES if mix[c] > 0]):
        raise SystemExit(f"no textbank rows at {ns.textbank} ({dict(tb_bad)}) but --mix wants {', '.join(want)}: "
                         "build it (tts/voxcpm/build_textbank.py) or --args \"--mix neutral=1\"; nothing started")
    cw_rows, cw_bad, _ = rc.cwords_wanted(ns)  # the cwords rows (a cwords share only), for the plan
    if mix["cwords"] > 0 and not cw_rows:
        raise SystemExit(f"--mix gives cwords a share but {ns.cwords} has no usable rows ({dict(cw_bad)}); nothing "
                         "started")
    if backend == "voxcpm" and (ram := rc.vc.pool_ram_gb(ns.workers, ns.load_parallel)) > (
            have := CONTEST_RAM_USE * CORPUS_MEM / 1024):
        raise SystemExit(f"--workers {ns.workers} with --load-parallel {ns.load_parallel} needs ~{ram:.0f} GiB of RAM "
                         f"while loading, over {have:.0f} of the container's {CORPUS_MEM // 1024} GiB; nothing started")
    ns.backend, ns.model = backend, MODEL_DIR
    rate = GPU_PRICE[GPU] + CORPUS_SIDE_PRICE
    budget = max_usd or CORPUS_MAX_USD
    max_s = min(usd_to_seconds(budget, rate), CORPUS_TIMEOUT_S)
    overhead = CORPUS_START_S + ALANIA_TAIL_S + IDLE_S + CONTEST_GRACE_S + rc.FINAL_S
    if max_s < overhead + rc.MIN_WAVE_S:
        raise SystemExit(f"--max-usd {budget:g} buys {max_s:.0f} s; corpus-render needs at least "
                         f"${math.ceil((overhead + rc.MIN_WAVE_S) / 3600 * rate * 100) / 100:.2f}")
    rel = out[len(DATA) + 1:]
    state = _vol_json(f"{rel}/state.json") or rc.new_state({})
    if why := rc.mode_problem(state, ns):  # the container refuses it too (render_corpus "Plain render")
        raise SystemExit(f"{why}; nothing started")
    pool = _vol_rows(f"{rel}/texts.jsonl")
    cost = rc.make_cost(backend, ns.timesteps, ns.workers, _vol_rows("contest/throughput.jsonl"))
    planned = rc.plan_pool(pool, tb_rows, ns, cw_rows)
    if ns.refs_from and not state.get("emotion_refs"):  # the prelude renders no reference --refs-from lends
        state = {**state, "emotion_refs": (_vol_json(f"{_vol_path(ns.refs_from)[len(DATA) + 1:]}/state.json")
                                           or {}).get("emotion_refs") or {}}
    laugh, _ = rc.load_laughter(rc.vc.repo_path(ns.laughter_bank))  # the prelude: emotion references, laughter trial
    est = rc.estimate(planned, state, ns, cost, rc.prelude_jobs(Path(out), state, ns, planned, laugh, cost, ns.ref)[1])
    first = 0.0 if pool else rc.POOL_S  # the first call draws the neutral sentences (and the long items)
    print(f"[modal] corpus-render -> {out}: {ns.hours:g} h of {ns.voice} ({ns.mix}; {GPU}, {backend}, "
          f"{ns.timesteps} timesteps; pool {'on the Volume' if pool else 'built in this call'}):")
    for line in rc.estimate_lines(est, rate, max_s - overhead - first):
        print(f"[modal]   {line}")
    if est["left_s"] <= 0:
        print("[modal] corpus-render: every category has its hours already; no GPU started")
        _corpus_fetch(corpus_local(out), out=out)
        return
    expected = min(est["work_s"] + overhead + first, max_s)
    print(f"[modal] corpus-render on {GPU} + {CORPUS_CPU} CPU cores + {CORPUS_MEM // 1024} GiB: at most "
          f"${max_s / 3600 * rate:.2f} (--max-usd {budget:g}): hard limit {max_s / 60:.0f} min at ${rate:.2f}/h incl. "
          f"CPU/RAM; expected ~${expected / 3600 * rate:.2f}"
          + (f"; render_corpus.py args: {' '.join(extra)}" if extra else ""))
    # the whole rest: a call's script stops when a wave no longer fits (half a MIN_WAVE_S left on average)
    per_call = max_s - overhead - CORPUS_SCRIPT_S - rc.MIN_WAVE_S / 2
    call_s = CORPUS_START_S + CORPUS_SCRIPT_S + rc.FINAL_S + IDLE_S
    wave_s = cost.server_s + cost.load_s + (rc.PHON_LOAD_S if est.get("c_checked") else 0.0)  # + the IPA model
    calls, billed = corpus_calls(est["work_s"], first, per_call, call_s, wave_s)
    print(f"[modal] the rest of {out}: ~{calls} call(s) of --max-usd {budget:g} (~{per_call / 60:.0f} min of waves "
          f"each): ~{billed / 3600:.2f} h = ~${billed / 3600 * rate:.2f} in all ({calls} x {call_s:.0f} s start-up, "
          f"script start, finalize, idle; {calls - 1} x {wave_s:.0f} s server + scorer loads again)")
    if ns.dry_run:  # the plan above is all a dry run needs: no GPU
        print("[modal] --dry-run: nothing started")
        return
    try:
        res = json.loads(corpus.remote(extra, max_s, f"corpus-render-{int(time.time())}", backend, out))
    except Exception as e:  # e.g. the watchdog's hard stop: what is decided is on the Volume all the same
        print(f"[modal] corpus-render ended with an error: {e}; what is done is on the Volume, the same command "
              "continues")
    else:
        st, cs = res.get("status") or {}, res.get("corpus") or {}
        print(json.dumps({"rc": res.get("rc"), "error": res.get("error"), "phase": st.get("phase"),
                          "errors": st.get("errors"), "corpus": cs}, indent=1, ensure_ascii=False))
        used, code = res.get("used_s", 0), res.get("rc")
        note = ("" if "error" in res else f"; render_corpus.py failed (exit code {code}), see the log above"
                if code not in (0, None) else "; stopped at the deadline: the same command continues" if code is None
                else "" if cs.get("reached") else "; not complete: the same command continues")
        print(f"[modal] GPU container ~{used / 60:.1f} min, ~${used / 3600 * rate:.2f}{note}")
    _corpus_fetch(corpus_local(out), out=out)


def _corpus_fetch(dest: Path, n: int = CORPUS_SAMPLE, seed: int = 0, out: str = F022) -> None:
    """Runs on the local machine: the corpus's (out, default F022) report, scores and manifest (prep_report.json,
    PREP_REPORT.md, ATTRIBUTION.txt, clips.jsonl, voxcpm_train.jsonl, status.json), its reference clips (ref/, the
    emotion references refs/) and n kept clips drawn at random (render_corpus.sample_clips: seeded, every category in
    proportion) -> dest/wav/, with sample.jsonl (their texts and scores) to listen with; the laughter trial's review
    clips (laughter_review/) with dest/laughter.html (render_corpus.laughter_html) to mark. A WAV already there with
    the same size is not fetched again."""
    rc, rel = _rc(), out[len(DATA) + 1:]
    files = _vol_files(rel)
    if not files:
        print(f"[corpus-fetch] no {out} on the Volume (run --action corpus-render)")
        return
    for name in ("prep_report.json", "PREP_REPORT.md", "ATTRIBUTION.txt", "clips.jsonl", "voxcpm_train.jsonl",
                 "status.json", rc.C_AUDIT):
        if name in files:
            try:
                _download(files[name].path, files[name].size, dest / name)
            except Exception as e:  # a broken connection: the next fetch continues
                print(f"[corpus-fetch] {name}: {e}")
    clips = []
    if (dest / "clips.jsonl").is_file():
        clips = [json.loads(x) for x in (dest / "clips.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    pick = rc.sample_clips(clips, n, seed)
    wavs = _vol_files(f"{rel}/wav") if pick else {}
    todo = [(f"wav/{c['wav']}", wavs[c["wav"]]) for c in pick if c["wav"] in wavs]
    todo += [(f"ref/{k}", e) for k, e in _vol_files(f"{rel}/ref").items()]
    todo += [(f"{rc.REFS_DIR}/{k}", e) for k, e in _vol_files(f"{rel}/{rc.REFS_DIR}").items()]
    review = _vol_files(f"{rel}/{rc.REVIEW}")
    if "review.jsonl" in review:
        todo += [(f"{rc.REVIEW}/review.jsonl", review["review.jsonl"])]
        todo += [(f"{rc.REVIEW}/wav/{k}", e) for k, e in _vol_files(f"{rel}/{rc.REVIEW}/wav").items()]
    have = 0
    for local_rel, e in todo:
        local = dest / local_rel
        if local.is_file() and local.stat().st_size == e.size and local.suffix == ".wav":
            have += 1
            continue
        try:
            _download(e.path, e.size, local)
        except Exception as err:
            print(f"[corpus-fetch] {local_rel}: {err}")
    keys = ("id", "category", "kind", "emotion", "duration", "train_text", "render_text", "hyp", "utmos", "sim",
            "p_tr", "try", "emotion_verdict", "ref", "sib_db", "first_stretch", "rate_ws", "tail_cut_s", "kinds",
            "sentences", "pause_sent", "pause_mid")
    (dest / "sample.jsonl").write_text("".join(json.dumps({k: c[k] for k in keys if k in c}, ensure_ascii=False)
                                               + "\n" for c in pick), encoding="utf-8")
    print(f"[corpus-fetch] {len(clips)} kept clips; a sample of {len(pick)} -> {dest / 'wav'} (sample.jsonl"
          + (f"; {have} files already here" if have else "") + f"); report {dest / 'PREP_REPORT.md'}")
    if (rv := dest / rc.REVIEW / "review.jsonl").is_file():
        rows = [json.loads(x) for x in rv.read_text(encoding="utf-8").splitlines() if x.strip()]
        (dest / "laughter.html").write_text(rc.laughter_html(rows), encoding="utf-8", newline="\n")
        print(f"[corpus-fetch] laughter trial: {len(rows)} clips of {len({r['line'] for r in rows})} lines -> "
              f"{dest / 'laughter.html'} (not in the training manifest)")


def _corpus_add_reviewed(extra: list[str]) -> None:
    """corpus-add-reviewed on the local machine: --args "[--add-reviewed FILE] [--dry-run]" (default
    render_corpus.LAUGH_APPROVED); refuse while the Volume's status.json says a corpus-render call is going; print what
    is new (approved ids not yet in reviewed.jsonl, their seconds from laughter_review/review.jsonl) and the cost; run
    the CPU call with the file's text (render_corpus.py checks everything again on the Volume)."""
    rc, rel = _rc(), F022[len(DATA) + 1:]
    path = Path(_last_opt(extra, "--add-reviewed", str(rc.LAUGH_APPROVED)))
    path = path if path.is_absolute() else LOCAL_REPO / path
    bad = [a for a in _without_opt(extra, "--add-reviewed") if a != "--dry-run"]
    if bad:
        raise SystemExit(f"corpus-add-reviewed: --args takes --add-reviewed FILE and --dry-run, not {bad}")
    try:
        text = path.read_text(encoding="utf-8")
        approved = json.loads(text)
    except (OSError, ValueError) as e:
        raise SystemExit(f"corpus-add-reviewed: {path}: {e}")
    ids = sorted(k for k, v in approved.items() if v is True) if isinstance(approved, dict) else []
    if not ids:
        raise SystemExit(f'corpus-add-reviewed: no approved ids in {path} ({{"<review id>": true, ...}})')
    st = _vol_json(f"{rel}/status.json") or {}
    if st.get("phase") in rc.RUNNING:
        raise SystemExit(f"corpus-add-reviewed: {F022}/status.json says phase {st['phase']!r} (updated "
                         f"{st.get('updated')}): a corpus-render call is going (or was killed mid-call: run the same "
                         "corpus-render command again first); nothing started")
    review = {r.get("id"): r for r in _vol_rows(f"{rel}/{rc.REVIEW}/review.jsonl")}
    have = {r.get("review_id") for r in _vol_rows(f"{rel}/{rc.REVIEWED}")}
    if missing := [i for i in ids if i not in review]:
        raise SystemExit(f"corpus-add-reviewed: not in {rc.REVIEW}/review.jsonl on the Volume: {missing}")
    new = [i for i in ids if i not in have]
    secs = sum(float(review[i].get("duration") or 0) for i in new)
    print(f"[modal] corpus-add-reviewed {path.name} -> {F022}: {len(ids)} approved, {len(ids) - len(new)} already "
          f"there; {len(new)} clips, ~{secs:.1f} s of {rc.REVIEWED_CAT} to add")
    if "--dry-run" in extra:
        print("[modal] --dry-run: nothing started")
        return
    print(f"[modal] CPU container ({REVIEWED_CPU} core + {REVIEWED_MEM // 1024} GiB, ${REVIEWED_PRICE:.3f}/h): a few "
          f"seconds, at most ${REVIEWED_TIMEOUT_S / 3600 * REVIEWED_PRICE:.3f} (its {REVIEWED_TIMEOUT_S // 60} min "
          "timeout)")
    res = json.loads(corpus_add_reviewed.remote(text, path.name))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    used = res.get("used_s", 0)
    note = "" if res.get("rc") == 0 else f"; render_corpus.py refused or failed (exit code {res.get('rc')}), see above"
    print(f"[modal] CPU container ~{used} s, ~${used / 3600 * REVIEWED_PRICE:.4f}{note}; --action corpus-fetch brings "
          "the report home")


def merged_stale(rep: dict, sizes: dict[str, int | None], audits: dict[str, str | None] | None = None) -> list[str]:
    """Why a merged corpus's manifest no longer is its sources' union ([] when it is): a source's voxcpm_train.jsonl
    gone, or of another size than when it was merged (prep_report.json "sources": manifest_bytes); with audits, a
    source's c_audit.jsonl (render_corpus --audit-c: the clips the merge left out) that appeared, went or changed since
    (c_audit_sha256: an audit made again can keep its size). sizes: source dir -> the manifest's bytes now; audits:
    source dir -> the audit's sha256 now (None: not there)."""
    out = []
    for p in rep.get("sources") or []:
        d, now = str(p.get("dir")), sizes.get(str(p.get("dir")))
        if now is None:
            out.append(f"{d}/voxcpm_train.jsonl is not there any more")
        elif now != p.get("manifest_bytes"):
            out.append(f"{d}/voxcpm_train.jsonl changed since the merge ({p.get('manifest_bytes')} -> {now} bytes)")
        if audits is not None and (a := audits.get(d)) != (was := p.get("c_audit_sha256")):
            out.append(f"{d}/c_audit.jsonl " + ("appeared" if was is None else "is gone" if a is None
                                                else f"changed (sha256 {was[:12]} -> {a[:12]})") + " since the merge")
    return out or ([] if rep.get("sources") else ["no sources in its prep_report.json"])


def merged_table_problem(ins: dict) -> str:
    """Why a merge with --instructions both / tr (prep_report.json "instructions") no longer has the local
    tts/voxcpm/instructions_tr.json's Turkish wordings ("" when it has): the table changed since the merge (its
    table_sha256), or is not here. The probe (probe_tr_instr.jsonl) takes the table's wordings: an experiment trained on
    the older ones would be probed on wordings it never saw."""
    rc = _rc()
    path = _vc().repo_path(rc.INSTRUCTIONS_TR)
    try:
        now = rc.sha256_file(path)
    except OSError as e:
        return f"merged with {ins.get('table')}, but the local {path} cannot be read ({e})"
    if now == (was := ins.get("table_sha256")):
        return ""
    return (f"merged with {ins.get('table')} sha256 {str(was)[:12]}, the local one is sha256 {now[:12]}: its Turkish "
            "instructions are not the table's")


def merge_source_problem(name: str, files: dict, status: dict | None, rep: dict | None, allow_incomplete: bool,
                         running: tuple[str, ...]) -> str:
    """Why corpus-merge must not take a source ("" when it may): no manifest, a corpus-render call going (status.json
    phase in `running`) or cut short (report_current), a mix not complete (unless allow_incomplete)."""
    if "voxcpm_train.jsonl" not in files or "prep_report.json" not in files:
        return f"no {name}/voxcpm_train.jsonl + prep_report.json on the Volume (--action corpus-render --out {name})"
    if (status or {}).get("phase") in running:
        return (f"{name}/status.json says phase {status['phase']!r} (updated {status.get('updated')}): a corpus-render "
                "call is going (or was killed mid-call: run it again first)")
    if not report_current({n: e.mtime for n, e in files.items()}):
        return f"{name}: its last corpus-render call was cut short (state.json newer than the report): run it again"
    if (rep or {}).get("complete") is False and not allow_incomplete:
        sel = (rep or {}).get("selected") or {}
        return (f"{name}: {sel.get('hours', '?')} of its {rep.get('target_hours')} h, its mix not complete (--args "
                "\"--allow-incomplete\" takes it as it is)")
    return ""


def _corpus_merge(extra: list[str], out: str = MERGE_OUT) -> None:
    """corpus-merge on the local machine: --args "[--sources A,B] [--allow-incomplete] [--allow-unaudited]
    [--instructions en|both|tr] [--dry-run]" (default MERGE_SOURCES -> MERGE_OUT); refuse a source
    merge_source_problem names, one whose instructed clips with a c were never c-checked (render_corpus.c_unchecked: no
    corpus-audit-c yet; --allow-unaudited takes it), with --instructions both / tr a source manifest's instruction
    wording render_corpus.INSTRUCTIONS_TR lacks, or an --out that holds another corpus or a merge with another
    --instructions; print the sources' hours, their c audits, the rows per instruction language and the cost; run the
    CPU call (render_corpus.py --merge checks everything again)."""
    rc = _rc()
    srcs = [s.strip() for s in _last_opt(extra, "--sources", ",".join(MERGE_SOURCES)).split(",") if s.strip()]
    allow, dry, unaudited = "--allow-incomplete" in extra, "--dry-run" in extra, "--allow-unaudited" in extra
    instructions = _last_opt(extra, "--instructions", "en")
    if bad := [a for a in _without_opt(_without_opt(extra, "--sources"), "--instructions")
               if a not in ("--allow-incomplete", "--allow-unaudited", "--dry-run")]:
        raise SystemExit(f"corpus-merge: --args takes --sources A,B, --allow-incomplete, --allow-unaudited, "
                         f"--instructions en|both|tr and --dry-run, not {bad}")
    if instructions not in rc.INSTRUCTION_MODES or extra[-1:] == ["--instructions"] or any(
            a != "--instructions" and not a.startswith("--instructions=") for a in _opt_args(extra, "--instructions")):
        raise SystemExit(f"corpus-merge: --instructions takes {', '.join(rc.INSTRUCTION_MODES)} (spelt out in full), "
                         f"not {instructions!r}")
    table = {}
    if instructions != "en":
        try:
            table = rc.load_instructions_tr(_vc().repo_path(rc.INSTRUCTIONS_TR))
        except ValueError as e:
            raise SystemExit(f"corpus-merge --instructions {instructions}: {e}")
    try:
        merge_cmd(srcs, out, allow)
        out, srcs = corpus_dir(out, MERGE_OUT), [corpus_dir(s) for s in srcs]
    except ValueError as e:
        raise SystemExit(f"corpus-merge: {e}")
    if out in srcs or len(set(srcs)) != len(srcs) or len(srcs) < 2:
        raise SystemExit(f"corpus-merge: two or more different sources, none of them --out {out}; nothing started")
    hours = 0.0
    for s in srcs:
        rel = s[len(DATA) + 1:]
        rep = _vol_json(f"{rel}/prep_report.json")
        if why := merge_source_problem(s, _vol_files(rel), _vol_json(f"{rel}/status.json"), rep, allow, rc.RUNNING):
            raise SystemExit(f"corpus-merge: {why}; nothing started")
        sel = (rep or {}).get("selected") or {}
        hours += float(sel.get("hours") or 0.0)
        audit = _vol_rows(f"{rel}/{rc.C_AUDIT}") if rc.C_AUDIT in _vol_files(rel) else None
        checked = {f"{s}/wav/{c['wav']}" for c in _vol_rows(f"{rel}/clips.jsonl")
                   if c.get("c_expected") is not None and isinstance(c.get("wav"), str)}
        man = _vol_rows(f"{rel}/voxcpm_train.jsonl")
        left = rc.c_unchecked(man, checked, {r.get("wav") for r in audit or []})
        copies, unknown = rc.instruction_copies(man, instructions, table)
        langs = collections.Counter(r["_lang"] for r in copies)
        print(f"[modal] corpus-merge source {s}: {sel.get('hours')} h, {sel.get('clips')} clips"
              + (" (its mix is not complete)" if (rep or {}).get("complete") is False else "")
              + (f"; {rc.C_AUDIT}: {sum(r.get('ok') is False for r in audit)} of {len(audit)} audited clips fail the "
                 "c-check (left out)" if audit is not None else "")
              + (f"; {len(left)} instructed clips with a c never c-checked" if left else "")
              + f"; --instructions {instructions}: {langs['plain']} rows without an instruction, {langs['en']} "
              f"English, {langs['tr']} Turkish")
        if left and not unaudited:
            raise SystemExit(f"corpus-merge: {s} has {len(left)} instructed clips with a c that were never c-checked: "
                             f"run --action corpus-audit-c --out {s} first (--args \"--allow-unaudited\" takes them "
                             "as they are); nothing started")
        if unknown:
            raise SystemExit(f"corpus-merge: {s} has {sum(unknown.values())} instructed rows whose wording "
                             f"{rc.INSTRUCTIONS_TR} lacks (add its Turkish wording there): "
                             + "; ".join(f"{w!r} x{n}" for w, n in sorted(unknown.items())) + "; nothing started")
    there = _vol_files(out[len(DATA) + 1:])
    merged = (_vol_json(f"{out[len(DATA) + 1:]}/prep_report.json") or {}) if "prep_report.json" in there else {}
    if rc.MARKER in there or ("voxcpm_train.jsonl" in there and not merged.get("merged")):
        raise SystemExit(f"corpus-merge: {out} holds a corpus that is not a merge; give another --out")
    if merged.get("merged") and (was := (merged.get("instructions") or {}).get("mode") or "en") != instructions:
        raise SystemExit(f"corpus-merge: {out} holds a merge with --instructions {was}; a merge with --instructions "
                         f"{instructions} goes to its own --out (e.g. {rc.merge_out_for(out, was, instructions)}), so "
                         f"an experiment trained on {out} never resumes on other texts; nothing started")
    print(f"[modal] corpus-merge -> {out}: ~{hours:.2f} h before duplicates; CPU container ({MERGE_CPU} core + "
          f"{MERGE_MEM // 1024} GiB, ${MERGE_PRICE:.3f}/h): seconds, at most "
          f"${MERGE_TIMEOUT_S / 3600 * MERGE_PRICE:.3f} (its {MERGE_TIMEOUT_S // 60} min timeout)")
    if dry:
        print("[modal] --dry-run: nothing started")
        return
    res = json.loads(corpus_merge.remote(srcs, out, allow, unaudited, instructions))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    used = res.get("used_s", 0)
    note = "" if res.get("rc") == 0 else f"; render_corpus.py refused or failed (exit code {res.get('rc')}), see above"
    print(f"[modal] CPU container ~{used} s, ~${used / 3600 * MERGE_PRICE:.4f}{note}"
          + ("" if note else f"; train with --args \"--data {out} ...\""))


def audit_plan(rows: list[dict]) -> dict:
    """corpus-audit-c's plan from a corpus's manifest rows: the clips render_corpus.audit_rows takes, their seconds,
    the estimated container seconds (start-up, the model's load, phon_check per clip) and USD, the hard limit's USD."""
    rc = _rc()
    take = rc.audit_rows(rows)
    secs = sum(float(r.get("duration") or 0.0) for r in take)
    est = AUDIT_START_S + rc.PHON_LOAD_S + len(take) * rc.PHON_CLIP_S + secs * rc.PHON_AUDIO_S + IDLE_S
    return {"clips": len(take), "audio_s": round(secs, 1), "est_s": round(est), "est_usd": est / 3600 * AUDIT_PRICE,
            "max_usd": (AUDIT_TIMEOUT_S + IDLE_S) / 3600 * AUDIT_PRICE}


def _corpus_audit_c(extra: list[str], out: str = F022) -> None:
    """corpus-audit-c on the local machine: --args "[--dry-run]"; refuse a folder that is no corpus-render corpus on
    the Volume or whose status.json says a call is going; print the plan (audit_plan) and "at most $X"; run the GPU
    call (render_corpus.py --audit-c checks everything again); fetch c_audit.jsonl to ./voxcpm-corpus/<folder>/."""
    rc = _rc()
    if bad := [a for a in extra if a != "--dry-run"]:
        raise SystemExit(f"corpus-audit-c: --args takes --dry-run only, not {bad}")
    try:
        out = corpus_dir(out)
    except ValueError as e:
        raise SystemExit(f"corpus-audit-c: {e}")
    rel = out[len(DATA) + 1:]
    files = _vol_files(rel)
    if rc.MARKER not in files or "voxcpm_train.jsonl" not in files:
        raise SystemExit(f"corpus-audit-c: {out} is no corpus-render corpus on the Volume (a merge: audit its sources)")
    st = _vol_json(f"{rel}/status.json") or {}
    if st.get("phase") in rc.RUNNING:
        raise SystemExit(f"corpus-audit-c: {out}/status.json says phase {st['phase']!r} (updated {st.get('updated')}): "
                         "a corpus-render call is going (or was killed mid-call: run it again first); nothing started")
    plan = audit_plan(_vol_rows(f"{rel}/voxcpm_train.jsonl"))
    print(f"[modal] corpus-audit-c {out}: {plan['clips']} manifest clips with an instruction and a c, "
          f"{plan['audio_s']:.0f} s of audio; {AUDIT_GPU} + {AUDIT_CPU} cores + {AUDIT_MEM // 1024} GiB at "
          f"${AUDIT_PRICE:.2f}/h: ~{plan['est_s']} s (~${plan['est_usd']:.2f}; start-up, the model's load "
          f"{rc.PHON_LOAD_S:g} s, {rc.PHON_CLIP_S} + {rc.PHON_AUDIO_S} x clip s each), at most ${plan['max_usd']:.2f} "
          f"(its {AUDIT_TIMEOUT_S // 60} min timeout)" + (f"; it replaces {rc.C_AUDIT}" if rc.C_AUDIT in files else ""))
    if "--dry-run" in extra or not plan["clips"]:
        print("[modal] " + ("--dry-run" if plan["clips"] else "nothing to audit") + ": nothing started")
        return
    res = json.loads(corpus_audit_c.remote(out))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    used = res.get("used_s", 0)
    note = "" if res.get("rc") == 0 else f"; render_corpus.py refused or failed (exit code {res.get('rc')}), see above"
    print(f"[modal] {AUDIT_GPU} container ~{used} s, ~${used / 3600 * AUDIT_PRICE:.3f}{note}")
    if res.get("rc") == 0 and (e := _vol_files(rel).get(rc.C_AUDIT)) is not None:
        _download(e.path, e.size, corpus_local(out) / rc.C_AUDIT)
        print(f"[modal] {rc.C_AUDIT} -> {corpus_local(out) / rc.C_AUDIT}; corpus-merge leaves its failed clips out")


def _train_lora():
    """train_lora.py (stdlib only) as a module, on the local machine: its parser and resume guard."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_lora", _HERE.with_name("train_lora.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _vol_path(p: Path | None) -> str:
    """A path train_lora.py's parser returned (a local Path, WindowsPath on Windows) as the container's POSIX path."""
    return posixpath.normpath(p.as_posix()) if p is not None else ""


def _vol_files(rel: str) -> dict:
    """name -> FileEntry of a Volume directory; {} when it is not there."""
    try:
        return {PurePosixPath(e.path).name: e for e in volume.listdir(rel)}
    except Exception:  # NotFoundError: no such directory
        return {}


def _vol_bytes(rel: str) -> bytes | None:
    """A Volume file's bytes; None when it is not there."""
    try:
        return b"".join(volume.read_file(rel))
    except Exception:  # NotFoundError
        return None


def _vol_json(rel: str) -> dict | None:
    try:
        d = json.loads(b"".join(volume.read_file(rel)))
    except Exception:  # not there, not JSON
        return None
    return d if isinstance(d, dict) else None


def _vol_has_ckpt(exp_name: str) -> bool:
    """train_lora.has_checkpoint on the Volume: a step_* folder, or ckpt/latest with LoRA weights."""
    ck = _vol_files(f"exp/{exp_name}/ckpt")
    return (any(n.startswith("step_") and e.type == modal.volume.FileEntryType.DIRECTORY for n, e in ck.items())
            or ("latest" in ck and bool({CKPT_FILES[0], "lora_weights.ckpt"} & set(_vol_files(
                f"exp/{exp_name}/ckpt/latest")))))


def _preflight_init_lora(src: str, rank: int, exp_name: str) -> None:
    """The --init-lora folder must be on the Volume with both LoRA files and the run's --rank."""
    if not src.startswith(DATA + "/"):
        raise SystemExit(f"--init-lora {src or '?'}: give a checkpoint folder on the Volume in full, e.g. "
                         f"--init-lora {DATA}/exp/voxcpm2-alania-s1/ckpt/step_0001000")
    if not (names := _vol_files(src[len(DATA) + 1:])):
        raise SystemExit(f"--init-lora {src}: not on the Volume")
    try:
        r = int(json.loads(b"".join(volume.read_file(names["lora_config.json"].path)))["lora_config"]["r"])
    except Exception as e:  # not on the Volume, no lora_config.json, ...
        raise SystemExit(f"--init-lora {src}: no usable LoRA checkpoint ({type(e).__name__}: {e})")
    if CKPT_FILES[0] not in names:
        raise SystemExit(f"--init-lora {src}: no {CKPT_FILES[0]} on the Volume")
    if r != rank:
        raise SystemExit(f"--init-lora {src} is a rank {r} LoRA, the run's --rank is {rank}: add --rank {r} to --args")
    print(f"[modal] {exp_name} starts at step 0 from {src} (rank {r}), fresh optimizer and LR schedule")


def _preflight_train(extra: list[str], exp_name: str, steps: int, probe: bool = False):
    """Runs on the local machine before `train` / `probe` start a GPU (train_lora.py refuses the same only after a paid
    start-up, and `train` retries other failures). --args are parsed by train_lora.py's own parser (prefixes, `=`,
    the last one wins) and may not set WRAPPER_OPTS; --data must be a corpus on the Volume whose preparation finished
    (prepare-alania cut short: refused; a merge whose sources changed since (merged_stale) or, with --instructions
    both / tr, whose Turkish table is not the local one (merged_table_problem): refused); an experiment with checkpoints keeps its corpus and LoRA rank
    (train_lora.resume_conflicts); --init-lora must be usable (_preflight_init_lora), outside the experiment itself,
    and is ignored once the experiment has checkpoints. Returns train_lora.py's parsed arguments."""
    if bad := _opt_args(extra, *WRAPPER_OPTS):
        raise SystemExit(f"--args may not set {', '.join(bad)}: {'probe' if probe else 'train'} sets the model, the "
                         "experiment (--exp-name), the steps (--steps) and the time limit (--max-usd / --hours)")
    tl = _train_lora()
    try:
        ns = tl.build_parser().parse_args(train_cmd(steps, 1.0, exp_name, extra)[2:])
    except SystemExit as e:
        if e.code in (0, None):  # --help
            raise
        raise SystemExit(f"train_lora.py would refuse --args \"{' '.join(extra)}\" (see above); nothing started")
    data, exp = _vol_path(ns.data), f"{DATA}/exp/{exp_name}"
    if ns.init_lora is not None and exp_name == EXP and not probe:
        raise SystemExit(f"--init-lora starts a new experiment: name it with --exp-name (e.g. voxcpm2-alania-s2); "
                         f"{EXP} has its own checkpoints")
    if not data.startswith(DATA + "/"):
        raise SystemExit(f"--data {data}: give a corpus folder on the Volume in full, e.g. --data {ALANIA}")
    if data != CORPUS and exp_name == EXP and not probe:
        raise SystemExit(f"--data {data}: name the experiment with --exp-name (e.g. voxcpm2-alania-s1); {EXP} is the "
                         f"one trained on {CORPUS}")
    files = _vol_files(data[len(DATA) + 1:])
    merged = (_vol_json(f"{data[len(DATA) + 1:]}/prep_report.json") or {}) if "prep_report.json" in files else {}
    merged = merged if merged.get("merged") else {}
    prep = ({ALANIA: "prepare-alania", HUMAN: "prepare-human", F022: "corpus-render"}.get(data)
            or ("corpus-merge" + ("" if data == MERGE_OUT else f" --out {data}") if merged
                else f"corpus-render --out {data}" if ".render_corpus" in files else None))
    how = (f"run --action {prep} (again: the same command continues) until it ends with rc 0" if prep
           else "run --action prepare first" if data == CORPUS else "prepare it first")
    if "voxcpm_train.jsonl" not in files:
        raise SystemExit(f"no {data}/voxcpm_train.jsonl on the Volume: {how}")
    if merged:  # render_corpus.py --merge: its sources' manifests and c audits as they were then
        sizes, audits = {}, {}
        for p in merged.get("sources") or []:
            d = str(p.get("dir") or "")
            got = _vol_files(d[len(DATA) + 1:]) if d.startswith(DATA + "/") else {}
            sizes[d] = getattr(got.get("voxcpm_train.jsonl"), "size", None)
            raw = _vol_bytes(f"{d[len(DATA) + 1:]}/c_audit.jsonl") if "c_audit.jsonl" in got else None  # KBs
            audits[d] = hashlib.sha256(raw).hexdigest() if raw is not None else None
        ins = merged.get("instructions") or {}  # render_corpus.py --merge --instructions
        mode, again = ins.get("mode") or "en", f"run --action {prep} again first"
        again += f" (--args \"--instructions {mode}\")" if mode != "en" else ""
        if stale := merged_stale(merged, sizes, audits):
            raise SystemExit(f"{data}: {'; '.join(stale)}: {again}")
        if mode != "en" and (why := merged_table_problem(ins)):
            raise SystemExit(f"{data}: {why}: {again} (the same --out takes it), or git pull when the local table is "
                             "the older one")
        sel, aud = merged.get("selected") or {}, merged.get("audio") or {}
        print(f"[modal] corpus {data} (merged from {', '.join(sizes)}): {sel.get('hours', '?')} h, "
              f"{sel.get('clips', '?')} {'rows' if aud else 'clips'}"
              + (f" (audio: {aud.get('clips', '?')} clips, {aud.get('hours', '?')} h)" if aud else "")
              + (f", instructions {mode}: " + ", ".join(f"{n} {k}" for k, n in (ins.get("rows") or {}).items())
                 + " rows" if mode != "en" else ""))
    if "state.json" in files:  # prepare_alania.py's / prepare_human.py's / render_corpus.py's layout
        if not report_current({n: e.mtime for n, e in files.items()}):
            raise SystemExit(f"{data}: its last {prep or 'prepare-alania'} run was cut short (or is still going), so "
                             f"voxcpm_train.jsonl is an older selection: {how}")
        rep = _vol_json(f"{data[len(DATA) + 1:]}/prep_report.json") or {}
        sel = rep.get("selected") or {}
        if rep.get("complete") is False:  # render_corpus.py: hours or mix not there yet, texts left to try
            raise SystemExit(f"{data}: {sel.get('hours', '?')} of its {rep.get('target_hours')} h, its mix not "
                             f"complete: {how} (a smaller --args \"--hours N\" closes it sooner)")
        print(f"[modal] corpus {data}: {sel.get('hours', '?')} h, {sel.get('clips', '?')} clips"
              + ("" if rep.get("reached", True) else f" (short of its {rep.get('target_hours')} h target)"))
    has = not probe and _vol_has_ckpt(exp_name)  # probe: a fresh experiment every time
    if has:
        conflicts = tl.resume_conflicts(_vol_json(f"exp/{exp_name}/lora.yaml"), tl.ckpt_rank(
            _vol_json(f"exp/{exp_name}/ckpt/latest/lora_config.json"), _vol_json(f"exp/{exp_name}/init_lora.json")),
            f"{data}/voxcpm_train.jsonl", ns.rank, ns.switch_data)
        if conflicts:
            raise SystemExit(f"{exp_name} has checkpoints and would resume, but {'; '.join(conflicts)} "
                             f"(train_lora.py options go in --args, a new experiment is --exp-name)")
    if ns.init_lora is not None:
        src = _vol_path(ns.init_lora)
        if src == exp or src.startswith(exp + "/"):
            raise SystemExit(f"--init-lora {src} lies inside {exp_name} itself: stage 2 needs its own --exp-name "
                             "(e.g. voxcpm2-alania-s2)")
        if has:
            print(f"[modal] {exp_name} already has checkpoints: training resumes from them, --init-lora is ignored")
        else:
            _preflight_init_lora(src, ns.rank, exp_name)
    return ns


def _eval_runs_or_exit(ckpts: str, grid: str, limit: int = 0,
                       extra: list[str] | None = None) -> list[tuple[str, str, list[str]]]:
    try:
        runs = eval_runs(ckpts, grid)
    except ValueError as e:
        raise SystemExit(str(e))
    if not runs:
        raise SystemExit("no checkpoints given (--ckpt / --fetch)")
    if overwrites_full_eval(limit, extra, grid):
        raise SystemExit("--limit / --args without --grid would write eval/<ckpt> and replace that checkpoint's full "
                         "evaluation; add --grid 10:2.0 (VoxCPM2's default; Trendyol's: 16:2.0), so the runs go to "
                         "eval/<ckpt>@ts10-cfg2")
    return runs


@app.local_entrypoint()
def main(action: str = "status", max_usd: float = 0.0, hours: float = 0.0, steps: int = 1000,
         exp_name: str = EXP, ckpt: str = "base,latest", minutes: float = 10.0, limit: int = 0, args: str = "",
         fetch: str = "", out: str = "", hf_repo: str = "ozgurguler/voxcpm2-antalia-eval", grid: str = "",
         run: str = "", spec: str = "", backend: str = "nanovllm"):
    price = GPU_PRICE.get(GPU)
    if action in ("probe", "train", "eval", "contest-screen", "contest-render", "corpus-render") and price is None:
        raise SystemExit(f"unknown GPU {GPU}; choose from {sorted(GPU_PRICE)}")
    if not (math.isfinite(max_usd) and max_usd >= 0) or limit < 0:
        raise SystemExit("--max-usd and --limit must be >= 0 (--max-usd 0 = the action's default: train 8, eval 2, "
                         "prepare-alania 1, prepare-human 2, contest-screen / contest-render 2, corpus-render 4)")
    total = (price or 0) + GPU_SIDE_PRICE
    extra = args.split() if args else []
    if action == "prepare":
        print(f"[modal] prepare on 4 CPU cores (~${CPU_JOB_PRICE:.2f}/h)")
        prepare.remote()
    elif action == "prepare-alania":
        try:
            alania_cmd(extra)
        except ValueError as e:
            raise SystemExit(str(e))
        budget = max_usd or ALANIA_MAX_USD
        max_s = min(usd_to_seconds(budget, ALANIA_CPU_PRICE), ALANIA_TIMEOUT_S)
        if max_s < (need := ALANIA_START_S + ALANIA_TAIL_S + IDLE_S + ALANIA_MIN_S):
            raise SystemExit(f"--max-usd {budget:g} buys {max_s:.0f} s; prepare-alania needs at least "
                             f"${math.ceil(need / 3600 * ALANIA_CPU_PRICE * 100) / 100:.2f}")
        print(f"[modal] prepare-alania on 8 CPU cores + 16 GiB: at most ${max_s / 3600 * ALANIA_CPU_PRICE:.2f} "
              f"(--max-usd {budget:g}): hard limit {max_s / 60:.0f} min at ${ALANIA_CPU_PRICE:.2f}/h; stopped there, "
              f"the same command continues" + (f"; prepare_alania.py args: {' '.join(extra)}" if extra else ""))
        try:
            res = json.loads(prepare_alania.remote(extra, max_s, f"prepare-alania-{int(time.time())}"))
        except Exception as e:  # e.g. the watchdog's hard stop: what is done is on the Volume all the same
            print(f"[modal] prepare-alania ended with an error: {e}; run the same command again to continue")
        else:
            print(json.dumps(res, indent=1, ensure_ascii=False))
            used, rc = res.get("used_s", 0), res.get("rc")  # used_s: incl. a start-up allowance per attempt
            note = ("" if rc == 0 or "error" in res else "; stopped at the deadline: the same command continues"
                    if rc is None else f"; prepare_alania.py failed (exit code {rc}), see the log above")
            print(f"[modal] CPU container ~{used / 60:.1f} min, ~${used / 3600 * ALANIA_CPU_PRICE:.2f}{note}")
    elif action == "prepare-human":
        if HUMAN_GPU not in HUMAN_GPU_PRICE:
            raise SystemExit(f"unknown TTS_HUMAN_GPU {HUMAN_GPU}; choose from {sorted(HUMAN_GPU_PRICE)}")
        try:
            human_cmd(extra)
        except ValueError as e:
            raise SystemExit(str(e))
        rate = HUMAN_GPU_PRICE[HUMAN_GPU] + HUMAN_SIDE_PRICE
        budget = max_usd or HUMAN_MAX_USD
        max_s = min(usd_to_seconds(budget, rate), HUMAN_TIMEOUT_S)
        if max_s < (need := HUMAN_START_S + ALANIA_TAIL_S + IDLE_S + ALANIA_MIN_S):
            raise SystemExit(f"--max-usd {budget:g} buys {max_s:.0f} s; prepare-human needs at least "
                             f"${math.ceil(need / 3600 * rate * 100) / 100:.2f}")
        print(f"[modal] prepare-human on {HUMAN_GPU} + 8 CPU cores + 16 GiB: at most ${max_s / 3600 * rate:.2f} "
              f"(--max-usd {budget:g}): hard limit {max_s / 60:.0f} min at ${rate:.2f}/h incl. CPU/RAM; stopped "
              f"there, the same command continues" + (f"; prepare_human.py args: {' '.join(extra)}" if extra else ""))
        try:
            res = json.loads(prepare_human.remote(extra, max_s, f"prepare-human-{int(time.time())}"))
        except Exception as e:  # e.g. the watchdog's hard stop: what is done is on the Volume all the same
            print(f"[modal] prepare-human ended with an error: {e}; run the same command again to continue")
        else:
            print(json.dumps(res, indent=1, ensure_ascii=False))
            used, rc = res.get("used_s", 0), res.get("rc")  # used_s: incl. a start-up allowance per attempt
            note = ("" if rc == 0 or "error" in res else "; stopped at the deadline: the same command continues"
                    if rc is None else f"; prepare_human.py failed (exit code {rc}), see the log above")
            print(f"[modal] {HUMAN_GPU} container ~{used / 60:.1f} min, ~${used / 3600 * rate:.2f}{note}")
    elif action == "probe":
        _preflight_train(extra, f"probe-{GPU.lower()}", 1000, probe=True)
        print(f"[modal] ~{minutes + 4:.0f}-minute probe on {GPU} (~${total * (minutes + 4) / 60:.2f})"
              + (f"; train_lora args: {' '.join(extra)}" if extra else ""))
        r = json.loads(probe.remote(minutes, extra))
        print(json.dumps(r, indent=1, ensure_ascii=False))
        if r.get("s_per_step"):
            print(f"[probe] {r['s_per_step']:.2f} s/step at batch 16, peak GPU memory {r['gpu_mem_peak_GiB']} GiB; "
                  f"${total:.2f}/h incl. CPU/RAM")
            for n in (300, 500, 1000, 1500, 2000):
                h = n * r["s_per_step"] / 3600
                print(f"  {n:5d} steps: {h:5.2f} h  ~${h * total:5.2f}")
    elif action == "train":
        init = _preflight_train(extra, exp_name, steps).init_lora is not None
        budget = max_usd or TRAIN_MAX_USD
        h = min(hours, budget / total) if hours else budget / total
        print(f"[modal] LoRA training on {GPU}: at most {h:.2f} h, i.e. <= ${h * total:.2f}; target {steps} steps; "
              f"resumes from the last checkpoint of {exp_name}"
              + ("; --init-lora counts only while it has none" if init else ""))
        train.remote(h, steps, exp_name, extra, f"{exp_name}-{int(time.time())}")
    elif action == "eval":
        runs = _eval_runs_or_exit(ckpt, grid, limit, extra)
        budget = max_usd or EVAL_MAX_USD
        max_s = min(usd_to_seconds(budget, total), EVAL_TIMEOUT_S)
        try:
            plan = eval_plan(runs, limit, extra, max_s)
        except ValueError as e:
            raise SystemExit(f"bad --args: {e}")
        overhead = EVAL_START_S + EVAL_TAIL_S + IDLE_S
        print(f"[modal] eval: {len(runs)} run(s) on {GPU}, {limit or 'all'} rows per set (rough A100 minutes):")
        for n, need, ok in plan:
            print(f"  {n:34s} ~{need / 60:4.1f} min" + ("" if ok else "  -> skipped: does not fit in the budget"))
        fit_usd = (sum(need for _, need, ok in plan if ok) + overhead) / 3600 * total
        print(f"[modal] at most ${max_s / 3600 * total:.2f} (--max-usd {budget:g}): hard limit {max_s / 60:.1f} min at "
              f"${total:.2f}/h incl. CPU/RAM; expected ~${fit_usd:.2f}")
        if skipped := [n for n, _, ok in plan if not ok]:
            all_s = sum(need for _, need, _ in plan) + overhead
            hint = (f"--max-usd {math.ceil(all_s / 3600 * total * 10) / 10:g} covers "
                    + ("it" if len(plan) == 1 else f"all {len(plan)}")
                    if all_s <= EVAL_TIMEOUT_S else f"all {len(plan)} do not fit in one call (3 h): split them")
            if len(skipped) == len(plan):
                raise SystemExit(f"[modal] no run fits in ${budget:g}; {hint}")
            if not max_usd:  # the default budget never drops runs silently
                raise SystemExit(f"[modal] the default ${EVAL_MAX_USD:g} covers {len(plan) - len(skipped)} of "
                                 f"{len(plan)} runs; {hint}, or --max-usd {EVAL_MAX_USD:g} runs only those")
            print(f"[modal] {len(skipped)} run(s) will be skipped: {', '.join(skipped)} ({hint})")
        try:
            res = json.loads(evaluate.remote(ckpt, exp_name, limit, extra, grid, max_s,
                                             f"eval-{exp_name}-{int(time.time())}"))
        except Exception as e:  # e.g. the watchdog's hard stop: finished runs are on the Volume all the same
            print(f"[modal] evaluate ended with an error: {e}")
            res = {"runs": {n: "?" for n, _, _ in runs}}
        else:
            print(json.dumps(res["runs"], indent=1, ensure_ascii=False))
            used = res["used_s"]  # incl. a start-up allowance of EVAL_START_S per attempt
            print(f"[modal] GPU container ~{used / 60:.1f} min, ~${used / 3600 * total:.2f}")
        # bring the WAVs home (to listen) and the scores to HF: the runs that finished (or may have: "?") and any in
        # --fetch; a killed / failed run left eval/<name> as it was
        done = [n for n, r in res["runs"].items() if isinstance(r, dict) or r == "?"]
        _fetch(_names(",".join([*done, fetch])), exp_name, Path(out or "voxcpm-eval"), hf_repo)
    elif action == "fetch":  # --grid too: the grid runs of the named checkpoints
        _fetch([n for n, _, _ in _eval_runs_or_exit(fetch or ckpt, grid)], exp_name, Path(out or "voxcpm-eval"),
               hf_repo)
    elif action in ("fetch-ckpt", "push-ckpt"):
        cks = [c for c in _names(fetch) if c not in ("base", "trendyol")]
        if not cks:
            raise SystemExit(f"name the checkpoints: --action {action} --fetch step_0000600[,step_...]")
        (_fetch_ckpt if action == "fetch-ckpt" else _push_ckpt)(cks, exp_name, LORA_DIR)
    elif action in ("contest-screen", "contest-render"):
        if limit:  # main's --limit is eval's: voice_contest.py's own goes through --args
            raise SystemExit(f"{action}: --limit is eval's; give voice_contest.py's in --args \"--limit {limit}\"")
        if spec and action == "contest-screen":
            raise SystemExit("contest-screen takes no --spec (contest-render does)")
        _contest(action, run, spec, extra, max_usd, backend)
    elif action == "contest-fetch":
        try:
            contest_out("render", run or fetch)
        except ValueError as e:
            raise SystemExit(str(e))
        _contest_fetch(run or fetch, CONTEST_DIR, every="--all" in extra)
    elif action == "contest-refs-push":
        _contest_refs_push(REFS_DIR)
    elif action == "corpus-render":
        if limit or run or spec:  # main's --limit is eval's, --run / --spec the contest's
            raise SystemExit("corpus-render takes --out, --max-usd, --backend and --args (render_corpus.py's options)")
        _corpus(extra, max_usd, backend, out or F022)
    elif action == "corpus-fetch":
        try:
            n, seed = int(_last_opt(extra, "--sample", str(CORPUS_SAMPLE))), int(_last_opt(extra, "--seed", "0"))
        except ValueError:
            raise SystemExit("corpus-fetch: --args \"--sample N --seed S\" take integers")
        try:
            dest = corpus_dir(out or F022)
        except ValueError as e:
            raise SystemExit(f"corpus-fetch --out: {e}")
        _corpus_fetch(corpus_local(dest), n, seed, dest)
    elif action == "corpus-merge":
        if limit or run or spec or max_usd:
            raise SystemExit("corpus-merge takes --out and --args \"[--sources A,B] [--allow-incomplete] "
                             "[--allow-unaudited] [--instructions en|both|tr] [--dry-run]\" only (a CPU call of "
                             "seconds)")
        _corpus_merge(extra, out or MERGE_OUT)
    elif action == "corpus-audit-c":
        if limit or run or spec or max_usd:
            raise SystemExit("corpus-audit-c takes --out and --args \"[--dry-run]\" only (its timeout is its limit)")
        _corpus_audit_c(extra, out or F022)
    elif action == "corpus-add-reviewed":
        if limit or run or spec or max_usd:
            raise SystemExit("corpus-add-reviewed takes --args \"[--add-reviewed FILE] [--dry-run]\" only (a CPU call "
                             "of seconds)")
        _corpus_add_reviewed(extra)
    elif action == "status":
        print(status.remote(exp_name))
    else:
        raise SystemExit("action must be one of: prepare, prepare-alania, prepare-human, probe, train, eval, fetch, "
                         "fetch-ckpt, push-ckpt, contest-screen, contest-render, contest-fetch, contest-refs-push, "
                         "corpus-render, corpus-fetch, corpus-add-reviewed, corpus-merge, corpus-audit-c, status")
