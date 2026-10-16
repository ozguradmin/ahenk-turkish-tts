"""Nano-vLLM-VoxCPM throughput probe on Modal: is batched VoxCPM2 synthesis fast enough, and the same, for the voice
contest and the ~5 h data generation (docs/SES_PLANI.md, steps 1 and 3)?

Nano-vLLM-VoxCPM (github.com/a710128/nanovllm-voxcpm, MIT, pinned NANO_COMMIT) serves VoxCPM2 with continuous batching,
CUDA graphs and FlashAttention. This separate app (own modal.App, the Volume "turkish-tts-data" of modal_app.py) loads
/data/models/VoxCPM2 (modal_app.py --action prepare), optionally a LoRA from the Volume, and for each inference
timesteps value (set per server: one server start each) and mode (noref: no reference; ref: a reference clip as
ref_audio_latents, VoxCPM2's reference-only cloning, never prompt continuation) runs contest-like Turkish sentences at
each concurrency N (--rounds x N requests, at most N in flight). Per scenario: audio seconds per GPU second, all and
usable (a request that hit the length cap is runaway audio the official API would re-render: not usable), USD per usable
audio hour, request latency and time to first audio (p50 / p90), per-request RTF, peak GPU memory and mean utilization
(nvidia-smi), requests that hit the length cap. Then the official API (voxcpm VoxCPM.generate as synth_eval.py calls
it, same commit as modal_app.py) renders --compare of the same sentences per timesteps x mode in the same container:
its RTF gives the speed-up, and both engines' clips are judged as synth_eval does (faster-whisper large-v3-turbo WER
after trnorm, UTMOS22).

    python tts/voxcpm/nanovllm_app.py                                 # the plan, locally, no Modal: sentences, estimate
    python -m modal run tts/voxcpm/nanovllm_app.py --action plan      # the same through Modal (no GPU)
    python -m modal run tts/voxcpm/nanovllm_app.py --action probe     # A100-80GB, at most $0.80 (--max-usd)
    python -m modal run tts/voxcpm/nanovllm_app.py --action probe --lora /data/exp/voxcpm2-alania-s1/ckpt/step_0002000
    python -m modal run tts/voxcpm/nanovllm_app.py --action probe --sentences benchmarks/tts/sets/contest.jsonl
    python -m modal run tts/voxcpm/nanovllm_app.py --action fetch --run probe-20261011-120000

Text: synth_eval.load_sets rows (TR-Zor stratified over its 13 categories, tr_soru stratified over its 5) and six
instruction lines "(English instruction)" + trnorm(sentence), interleaved so that every prefix mixes them; --sentences
FILE: a contest-style JSONL instead (id, text; optional set, refs, instruction). Model text = trnorm.normalize as in
synth_eval (VoxCPM's own normalizer is off). Requests: cfg 2.0, seed 0, max_generate_length = 6 x text tokens + 10, the
official API's first-attempt cap (Nano-vLLM has no bad-case retry). Reference: --ref (a WAV on the Volume), else the
first Antalia training clip of 6-12 s.
Server: max_num_seqs = the highest concurrency, CUDA graphs on; it reserves --gpu-mem (default 0.85) of the card for its
KV cache, so nvidia-smi's peak mostly shows that reservation: a lower --gpu-mem tells whether N still fits in less.
Cost: --max-usd (default 0.8) is a hard deadline of the whole GPU call incl. start-up (ledger on the Volume for attempts
a preemption cut short). The worker runs in priority order (Nano-vLLM per timesteps value, then the official renders,
then judging), skips what its estimate says no longer fits, and is killed at the deadline anyway. Results:
/data/nanovllm-probe/<run>/ (results.json rewritten after every scenario, compare WAVs); the same --run continues an
unfinished probe (finished scenarios are not run again; one with failed requests is, and a dead engine ends its server's
scenarios); a resume with other sentences, GPU or build is refused; they come home to
./voxcpm-contest/nanovllm-probe/<run>/.
GPU: TTS_GPU=A100-80GB|L40S|H100 (default A100-80GB), read at import time.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path, PurePosixPath

import modal

REPO, DATA = "/repo", "/data"
_HERE = Path(__file__).resolve()
# repo root on the local machine; in the container this file is /root/nanovllm_app.py, where the value is unused
LOCAL_REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
for _d in (_HERE.parent, Path(REPO, "tts", "voxcpm")):  # nanovllm_base.py: next to this file, in the container /repo
    if (_d / "nanovllm_base.py").is_file() and str(_d) not in sys.path:
        sys.path.append(str(_d))

# the pins, image recipe and LoRA kwargs the voice contest (modal_app.py, voice_contest.py) shares
from nanovllm_base import (FLASH_ATTN, NANO_COMMIT, NANO_REPO, PY_TAG, TORCH, TORCHCODEC,  # noqa: E402,F401
                           VOXCPM_COMMIT, VOXCPM_SRC, base_image, flash_attn_wheel, nano_lora_kwargs)

GPU = os.environ.get("TTS_GPU", "A100-80GB")
GPU_PRICE = {"H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95}  # USD/h, Modal list prices (2026), as modal_app.py
SIDE_PRICE = 4 * 0.047 + 32 * 0.008  # 4 cores + 32 GiB next to the GPU
MODEL_DIR = f"{DATA}/models/VoxCPM2"
MODEL_FILES = ("config.json", "model.safetensors", "audiovae.pth")  # Nano-vLLM loads *.safetensors + audiovae.pth
CORPUS = f"{DATA}/antalia16k"  # default reference: an Antalia training clip of REF_SECONDS
PROBE_ROOT = f"{DATA}/nanovllm-probe"
LOCAL_OUT = "voxcpm-contest/nanovllm-probe"
PROBE_MAX_USD, PROBE_TIMEOUT_S = 0.8, 3600  # --max-usd default; Modal's own limit of one probe call
START_S = 90  # billed container start-up (image, Volume), taken off the budget
TAIL_S = 45  # after the worker: kill, results, ledger, Volume commit
IDLE_S = 2  # scaledown_window: billed idle after the call returns
MIN_S = 240  # with less left for the worker it is not started (one server start + one batch)
REF_SECONDS = (6.0, 12.0)
# rough A100 times for the plan and the worker's skip decisions (the worker uses its measured server start after the
# first): Nano-vLLM throughput is a guess until this probe measures it (RTX 4090, 64 short prompts: ~100 audio-s/s)
ENV_S, SERVER_START_S, SERVER_STOP_S, WARMUP_S = 20.0, 90.0, 15.0, 15.0
AUDIO_S_EST, NANO_AUDIO_PER_S = 4.5, 25.0  # audio s per contest-like request; audio-s per GPU-s at 10 timesteps
OFFICIAL_LOAD_S, JUDGE_S, JUDGE_CLIP_S = 45.0, 75.0, 1.0
MAX_GEN_CAP, MAX_GEN_RATIO = 2000, 6.0  # Nano-vLLM's default max_generate_length; the official API's bad-case ratio
MODES = ("noref", "ref")
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
EMOTION_LINES = (  # (id, English instruction in VoxCPM's demo style, original Turkish sentence)
    ("emo_angry", "Angry tone, voice getting louder.", "Bu kaçıncı kez aynı hatayı yapıyorsun, artık yeter!"),
    ("emo_cheerful", "Cheerful and excited tone.", "Harika haber, sınavı birincilikle geçmişsin!"),
    ("emo_sad", "Sad, quiet and slow voice.", "Onu bir daha göremeyeceğimi bilmek çok zor."),
    ("emo_surprised", "Surprised tone, rising pitch.", "Gerçekten mi, bütün yolu yürüyerek mi geldin?"),
    ("emo_calm", "Calm and soothing voice.", "Derin bir nefes al, her şey yoluna girecek."),
    ("emo_empathetic", "Warm, empathetic and gentle tone.", "Ne kadar yorulduğunu anlıyorum, yanındayım."),
)
RESUME_KEYS = ("lora", "ref", "cfg", "seed", "gpu_mem", "rounds", "max_num_seqs", "sentences", "items_sha1", "compare",
               "gpu", "torch", "flash_attn", "nano_commit", "voxcpm_commit")
DEAD_ERROR_SHARE = 0.5  # a batch with this share of failed requests (or a WorkerDiedError): its server is gone


app = modal.App("turkish-tts-nanovllm-probe")
volume = modal.Volume.from_name("turkish-tts-data", create_if_missing=True)
SECRETS = [modal.Secret.from_name(os.environ["TTS_HF_SECRET"])] if os.environ.get("TTS_HF_SECRET") else []

_IGNORE = [".git", ".venv*", "data", "exports", "build", "dist", "**/__pycache__", "**/*.pyc", "**/*.ckpt",
           "**/*.onnx", "**/*.wav", "stt/icefall", "tts/piper1-gpl", "tts/voxcpm-src", "benchmarks/**/results",
           "benchmarks/**/logs", "antalia-export", "voxcpm-eval", "voxcpm-lora", "voxcpm-contest", "voxcpm-refs",
           "hf-publish"]
image = (
    base_image(modal)
    # the Whisper (HF_HOME) and UTMOS (TORCH_HOME) caches modal_app.py's eval filled on the Volume
    .env({"TTS_DATA": DATA, "VOXCPM_SRC": VOXCPM_SRC, "PYTHONPATH": f"{REPO}/src", "HF_HOME": f"{DATA}/hf_home",
          "TORCH_HOME": f"{DATA}/torch_home", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(LOCAL_REPO, REPO, ignore=_IGNORE)
)


# ---------------------------------------------------------------- pure helpers (no torch / numpy / GPU)

def parse_ints(s: str, lo: int, hi: int, what: str) -> list[int]:
    """"16, 32,,16" -> [16, 32] (sorted, unique); ValueError outside lo..hi or when empty."""
    out = set()
    for x in (t.strip() for t in str(s).split(",")):
        if not x:
            continue
        try:
            v = int(x)
        except ValueError:
            raise ValueError(f"--{what}: {x!r} is not an integer") from None
        if not lo <= v <= hi:
            raise ValueError(f"--{what}: {v} is outside {lo}-{hi}")
        out.add(v)
    if not out:
        raise ValueError(f"--{what}: give at least one value")
    return sorted(out)


def parse_modes(s: str) -> list[str]:
    """"ref,noref" -> ["noref", "ref"] (MODES order); ValueError for anything else."""
    got = {x.strip() for x in s.split(",") if x.strip()}
    if not got or got - set(MODES):
        raise ValueError(f"--modes: choose from {', '.join(MODES)} (comma-separated), got {s!r}")
    return [m for m in MODES if m in got]


def scenario_key(ts: int, mode: str, n: int) -> str:
    return f"ts{ts}/{mode}/c{n}"


def max_gen_len(n_tokens: int, cap: int = MAX_GEN_CAP, ratio: float = MAX_GEN_RATIO) -> int:
    """Generation steps allowed for a text of n_tokens: the official API's first-attempt cap (ratio x tokens + 10)."""
    return max(1, min(int(n_tokens * ratio + 10), cap))


def percentile(xs: list[float], q: float) -> float | None:
    """Linear interpolation between order statistics (numpy's default); None for no values."""
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * min(max(q, 0.0), 100.0) / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


def pick_ref(rows: list[dict], lo: float = REF_SECONDS[0], hi: float = REF_SECONDS[1]) -> dict | None:
    """The first manifest row whose duration is within lo-hi s, else the one closest to the middle; None if no rows."""
    rows = [r for r in rows if isinstance(r.get("duration"), (int, float)) and r.get("audio")]
    if not rows:
        return None
    return next((r for r in rows if lo <= r["duration"] <= hi), None) or min(
        rows, key=lambda r: abs(r["duration"] - (lo + hi) / 2))


def interleave(*groups: list) -> list:
    """Merge lists so that every prefix holds each list in proportion: item i of a list of n sits at (i + 0.5) / n."""
    keyed = [((i + 0.5) / len(g), gi, i, x) for gi, g in enumerate(groups) for i, x in enumerate(g)]
    return [k[3] for k in sorted(keyed, key=lambda k: k[:3])]


def _repo_modules(root: Path) -> None:
    """synth_eval / score_utmos (tts/voxcpm) and trnorm (src) importable."""
    for p in (root / "tts" / "voxcpm", root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def build_items(n_trzor: int = 42, n_soru: int = 16, root: Path = LOCAL_REPO) -> list[dict]:
    """Contest-like probe sentences: synth_eval.load_sets rows (TR-Zor and tr_soru, stratified; model_text =
    trnorm.normalize, refs as synth_eval scores them) and EMOTION_LINES ("(instruction)" + trnorm(sentence), scored
    against the sentence alone), interleaved. Each item: id, set, category, text, model_text, refs."""
    _repo_modules(root)
    from synth_eval import load_sets
    from trnorm import normalize

    keep = ("id", "set", "category", "text", "model_text", "refs")
    trzor = [{k: r[k] for k in keep} for r in load_sets(None, ["trzor"], n_trzor)]
    soru = [{k: r[k] for k in keep} for r in load_sets(None, ["soru"], n_soru)]
    emo = [{"id": i, "set": "emo", "category": i[4:], "text": s, "model_text": f"({instr}){normalize(s)}",
            "refs": [s]} for i, instr, s in EMOTION_LINES]
    return interleave(trzor, soru, emo)


def load_sentences(path: Path, root: Path = LOCAL_REPO) -> list[dict]:
    """A contest-style JSONL (id, text; optional set, category, refs, instruction) -> probe items, interleaved by set;
    model_text = "(instruction)" + trnorm(text), or trnorm(text) without one, as the contest renders them.
    ValueError for a row without id / text, duplicate ids or no rows."""
    _repo_modules(root)
    from trnorm import normalize

    groups: dict[str, list[dict]] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if not (isinstance(r.get("id"), str) and isinstance(r.get("text"), str) and r["text"].strip()):
            raise ValueError(f"{path}:{n}: needs an id and a text")
        instr = r.get("instruction") or ""
        groups.setdefault(r.get("set") or "file", []).append({
            "id": r["id"], "set": r.get("set") or "file", "category": r.get("category") or "", "text": r["text"],
            "model_text": (f"({instr})" if instr else "") + normalize(r["text"]), "refs": r.get("refs") or [r["text"]]})
    items = interleave(*groups.values())
    if not items or len({it["id"] for it in items}) != len(items):
        raise ValueError(f"{path}: no sentences, or duplicate ids")
    return items


def items_sha1(items: list[dict]) -> str:
    """Digest of what the probe renders and judges (id, model_text, refs, in request order): results.json keeps the
    spec without its items, so a resume compares this instead (an edited sentences file must not mix in)."""
    rows = [[it["id"], it["model_text"], it["refs"]] for it in items]
    return hashlib.sha1(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


def compare_ids(items: list[dict], k: int, pool: int) -> list[str]:
    """k item ids for the official-API comparison, round-robin over the sets in items[:pool] (the requests of the
    lowest concurrency's batch, whose Nano-vLLM audio is kept), so each set is in it from k = number of sets."""
    by: dict[str, list[str]] = {}
    for it in items[:pool]:
        by.setdefault(it["set"], []).append(it["id"])
    out, i = [], 0
    while len(out) < k and any(i < len(v) for v in by.values()):
        out += [v[i] for v in by.values() if i < len(v)][:k - len(out)]
        i += 1
    return out


def build_spec(run: str = "", timesteps: str = "10,16", concurrency: str = "16,32,64", modes: str = "noref,ref",
               rounds: int = 2, compare: int = 4, cfg: float = 2.0, gpu_mem: float = 0.85, lora: str = "",
               ref: str = "", sentences: str = "", root: Path = LOCAL_REPO) -> dict:
    """The probe's whole configuration (validated; ValueError otherwise), sent to the GPU worker as JSON. sentences:
    a contest-style JSONL (load_sentences; relative to the repo root) instead of build_items' mix."""
    run = run or time.strftime("probe-%Y%m%d-%H%M%S", time.gmtime())
    if not RUN_RE.match(run):
        raise ValueError(f"--run {run!r}: letters, digits, '.', '_', '-' (at most 64)")
    ts, conc, md = parse_ints(timesteps, 1, 50, "timesteps"), parse_ints(concurrency, 1, 256, "concurrency"), \
        parse_modes(modes)
    if not 1 <= rounds <= 10:
        raise ValueError("--rounds: 1-10 (requests per scenario = rounds x concurrency)")
    if not (math.isfinite(cfg) and 0 < cfg <= 10):
        raise ValueError("--cfg: 0-10")
    if not 0.3 <= gpu_mem <= 0.95:
        raise ValueError("--gpu-mem: 0.3-0.95 (Nano-vLLM's gpu_memory_utilization: KV cache sizing)")
    for opt, p in (("lora", lora), ("ref", ref)):
        if p and not p.startswith(DATA + "/"):
            raise ValueError(f"--{opt} {p}: give a path on the Volume in full, e.g. {DATA}/...")
    if ref and "ref" not in md:
        raise ValueError("--ref without the ref mode (--modes ...,ref) would never be used")
    if sentences:
        try:
            items = load_sentences(Path(root, sentences), root)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"--sentences {sentences}: {e}") from None
    else:
        items = build_items(root=root)
    pool = rounds * min(conc)
    if not 0 <= compare <= min(pool, len(items)):
        raise ValueError(f"--compare: 0-{min(pool, len(items))} (sentences of the lowest concurrency's batch)")
    return {"run": run, "out": f"{PROBE_ROOT}/{run}", "gpu": GPU, "timesteps": ts, "concurrency": conc, "modes": md,
            "rounds": rounds, "cfg": cfg, "seed": 0, "gpu_mem": gpu_mem, "max_num_seqs": max(conc),
            "lora": lora.rstrip("/"), "ref": ref, "sentences": sentences or "builtin", "items_sha1": items_sha1(items),
            "compare": compare_ids(items, compare, pool), "items": items,
            "nano_commit": NANO_COMMIT, "voxcpm_commit": VOXCPM_COMMIT, "flash_attn": FLASH_ATTN, "torch": TORCH}


def resume_conflicts(old: dict, new: dict) -> list[str]:
    """Settings an existing run's results.json was made with that this spec changes (its scenarios would mix)."""
    return [f"{k}: {old.get(k)!r} -> {new.get(k)!r}" for k in RESUME_KEYS if old.get(k) != new.get(k)]


def scenario_s(n: int, rounds: int, ts: int) -> float:
    """Estimated seconds of one scenario: rounds x n requests at the guessed throughput, slower with the timesteps."""
    return rounds * n * AUDIO_S_EST / (NANO_AUDIO_PER_S * 10 / max(ts, 1)) + 5.0


def official_rtf(ts: int) -> float:
    """The official API's single-request RTF on an A100 (measured: ~0.9 at 10 timesteps, 1.7-2.3 at 16)."""
    return max(0.3, 0.9 + 0.185 * (ts - 10))


def official_clip_s(ts: int) -> float:
    return AUDIO_S_EST * official_rtf(ts)


def estimate(spec: dict) -> list[tuple[str, float]]:
    """(phase, rough A100 seconds) in the worker's order, start-up and tail not included."""
    out = [("env check", ENV_S)]
    for ts in spec["timesteps"]:
        s = SERVER_START_S + SERVER_STOP_S + len(spec["modes"]) * WARMUP_S
        s += sum(scenario_s(n, spec["rounds"], ts) for n in spec["concurrency"]) * len(spec["modes"])
        out.append((f"nano-vllm ts{ts}", s))
    if k := len(spec["compare"]):
        out.append(("official API", OFFICIAL_LOAD_S + sum(k * len(spec["modes"]) * official_clip_s(ts)
                                                          for ts in spec["timesteps"])))
        out.append(("judge (Whisper, UTMOS)", JUDGE_S + JUDGE_CLIP_S * 2 * k * len(spec["modes"])
                    * len(spec["timesteps"])))
    return out


def usd_to_seconds(usd: float, usd_per_h: float) -> float:
    return usd / usd_per_h * 3600


def parse_smi(text: str) -> tuple[int, int] | None:
    """First line of `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits` -> (MiB, %)."""
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                return int(float(parts[0])), int(float(parts[1]))
            except ValueError:
                return None
    return None


def window_stats(samples: list[tuple[float, int, int]], t0: float, t1: float) -> dict:
    """Peak memory (MiB) and mean utilization (%) of the (time, MiB, %) samples within t0..t1; {} without any."""
    w = [s for s in samples if t0 <= s[0] <= t1]
    if not w:
        return {}
    return {"gpu_mem_peak_mib": max(s[1] for s in w), "gpu_util_mean": round(sum(s[2] for s in w) / len(w))}


def batch_stats(reqs: list[dict], sr: int, wall_s: float, usd_per_h: float = 0.0) -> dict:
    """One scenario from run_batch's request records (t0, first, t1 from perf_counter, samples, steps, cap; or an
    "error"): audio seconds per GPU second over the batch wall time, all and usable (requests below their cap: a capped
    one is runaway audio, ~6x a normal line, that the official API would throw away and re-render), USD per usable
    audio hour, latency / TTFB percentiles, per-request RTF ((t1 - first) / audio, as Nano-vLLM's benchmark), requests
    whose steps reached their cap and their share of the audio."""
    ok = [r for r in reqs if "error" not in r]
    audio = [r["samples"] / sr for r in ok]
    usable = sum(a for r, a in zip(ok, audio) if r["steps"] < r["cap"])
    lat, ttfb = [r["t1"] - r["t0"] for r in ok], [r["first"] - r["t0"] for r in ok]
    rtf = [(r["t1"] - r["first"]) / a for r, a in zip(ok, audio) if a > 0]
    thr, thr_ok = (sum(audio) / wall_s, usable / wall_s) if wall_s > 0 else (None, None)
    out = {"requests": len(reqs), "ok": len(ok), "wall_s": round(wall_s, 2), "audio_s": round(sum(audio), 1),
           "usable_audio_s": round(usable, 1), "audio_s_per_gpu_s": _r(thr, 2),
           "usable_audio_s_per_gpu_s": _r(thr_ok, 2),
           "usd_per_usable_audio_h": _r(usd_per_h / thr_ok, 4) if thr_ok and usd_per_h else None,
           "latency_p50_s": _r(percentile(lat, 50), 2), "latency_p90_s": _r(percentile(lat, 90), 2),
           "latency_max_s": _r(max(lat, default=None), 2), "ttfb_p50_s": _r(percentile(ttfb, 50)),
           "ttfb_p90_s": _r(percentile(ttfb, 90)), "rtf_req_mean": _r(sum(rtf) / len(rtf) if rtf else None),
           "audio_s_mean": _r(sum(audio) / len(audio) if audio else None, 2),
           "hit_cap": sum(r["steps"] >= r["cap"] for r in ok),
           "cap_audio_share": _r(1 - usable / sum(audio)) if sum(audio) > 0 else None}
    if errors := [r for r in reqs if "error" in r]:
        out["errors"], out["first_error"] = len(errors), errors[0]["error"]
    return out


def scenario_done(s: dict | None) -> bool:
    """A stored scenario counts (resume skips it, the report may pick it) only when none of its requests failed."""
    return bool(s) and not s.get("incomplete") and not s.get("errors")


def server_failure(reqs: list[dict]) -> str | None:
    """Why the Nano-vLLM server of a batch must not be measured further (its engine process died: WorkerDiedError,
    or DEAD_ERROR_SHARE of the requests failed), else None. Nano-vLLM's engine loop has no try/except around step():
    after a crash (e.g. CUDA OOM) every later request fails at once."""
    errors = [r["error"] for r in reqs if "error" in r]
    if not errors:
        return None
    died = next((e for e in errors if "WorkerDiedError" in e), None)
    if died or len(errors) >= DEAD_ERROR_SHARE * len(reqs):
        return f"{len(errors)} of {len(reqs)} requests failed ({died or errors[0]})"
    return None


def compare_summary(rows: dict) -> dict:
    """Judged comparison rows ({key: {engine, ts, mode, id, duration, synth_s?, errors?, ref_words?, utmos?,
    hit_cap?}}) -> {"<engine>/ts<ts>/<mode>": {clips, capped (Nano-vLLM clips that ran to the length cap), wer,
    rows_correct, utmos, duration_s, rtf (official)}} and per "ts<ts>/<mode>" the paired delta over the sentences both
    engines rendered: WER and UTMOS of each, the mean duration ratio."""
    groups: dict[str, list[dict]] = {}
    for r in rows.values():
        groups.setdefault(f"{r['engine']}/ts{r['ts']}/{r['mode']}", []).append(r)
    out: dict = {}
    for g, rs in sorted(groups.items()):
        judged, scored = [r for r in rs if "errors" in r], [r["utmos"] for r in rs if r.get("utmos") is not None]
        s = {"clips": len(rs), "duration_s": round(sum(r["duration"] for r in rs) / len(rs), 2)}
        if capped := sum(bool(r.get("hit_cap")) for r in rs):
            s["capped"] = capped
        if judged:
            n = sum(r["ref_words"] for r in judged)
            s.update(wer=round(100 * sum(r["errors"] for r in judged) / max(n, 1), 2), judged=len(judged),
                     rows_correct=sum(r["errors"] == 0 for r in judged))
        if scored:
            s["utmos"] = round(sum(scored) / len(scored), 3)
        if (synth := [r for r in rs if r.get("synth_s") is not None]) and sum(r["duration"] for r in synth) > 0:
            s["rtf"] = round(sum(r["synth_s"] for r in synth) / sum(r["duration"] for r in synth), 3)
        out[g] = s
    pairs: dict[str, list[tuple[dict, dict]]] = {}
    nano = {(r["ts"], r["mode"], r["id"]): r for r in rows.values() if r["engine"] == "nano"}
    for r in rows.values():
        if r["engine"] == "official" and (n := nano.get((r["ts"], r["mode"], r["id"]))):
            pairs.setdefault(f"ts{r['ts']}/{r['mode']}", []).append((n, r))
    delta = {}
    for g, ps in sorted(pairs.items()):
        d = {"pairs": len(ps)}
        if capped := sum(bool(p[0].get("hit_cap")) for p in ps):
            d["nano_capped"] = capped
        for name, i in (("nano", 0), ("official", 1)):
            if all("errors" in p[i] for p in ps):
                d[f"wer_{name}"] = round(100 * sum(p[i]["errors"] for p in ps) / max(sum(p[i]["ref_words"]
                                                                                       for p in ps), 1), 2)
            if all(p[i].get("utmos") is not None for p in ps):
                d[f"utmos_{name}"] = round(sum(p[i]["utmos"] for p in ps) / len(ps), 3)
        ratios = [p[0]["duration"] / p[1]["duration"] for p in ps if p[1]["duration"] > 0]
        if ratios:
            d["duration_ratio"] = round(sum(ratios) / len(ratios), 3)
        delta[g] = d
    if delta:
        out["nano_vs_official"] = delta
    return out


def format_report(res: dict, usd_per_h: float) -> list[str]:
    """Human-readable lines of a results.json (partial ones too); priced at its own GPU (spec.gpu), else usd_per_h.
    Throughput and $ are per usable audio; the speed-up picks only scenarios without failed requests."""
    gpu = (res.get("spec") or {}).get("gpu")
    usd_per_h = GPU_PRICE[gpu] + SIDE_PRICE if gpu in GPU_PRICE else usd_per_h
    env, lines = res.get("env") or {}, []
    if env:
        lines.append("[probe] " + ", ".join(f"{k} {v}" for k, v in env.items()))
    for k, s in sorted((res.get("servers") or {}).items()):
        mem = s.get("mem_ready_mib")
        lines.append(f"[probe] server {k}: ready in {s.get('ready_s', '?')} s"
                     + (f", {mem / 1024:.1f} GiB in use after start" if mem else "")
                     + (f", LoRA {s['lora']}" if s.get("lora") else ""))
    sc = res.get("scenarios") or {}
    if sc:
        lines.append(f"{'scenario':<18} {'req':>4} {'audio-s/GPU-s':>13} {'usable':>6} {'$/audio-h':>9} "
                     f"{'lat p50/p90 s':>13} {'TTFB p50':>8} {'RTF/req':>7} {'mem GiB':>7} {'util%':>5} {'cap':>3} "
                     f"{'cap%':>4} {'err':>3}")

        def f(x, nd=2):
            return "-" if x is None else f"{x:.{nd}f}"

        for k in sorted(sc, key=lambda k: (int(k.split("/")[0][2:]), k.split("/")[1], int(k.split("/")[2][1:]))):
            s = sc[k]
            mem, share = s.get("gpu_mem_peak_mib"), s.get("cap_audio_share")
            lines.append(f"{k:<18} {s['requests']:>4} {f(s.get('audio_s_per_gpu_s')):>13} "
                         f"{f(s.get('usable_audio_s_per_gpu_s')):>6} {f(s.get('usd_per_usable_audio_h'), 3):>9} "
                         f"{f(s.get('latency_p50_s'), 1) + '/' + f(s.get('latency_p90_s'), 1):>13} "
                         f"{f(s.get('ttfb_p50_s')):>8} {f(s.get('rtf_req_mean')):>7} "
                         f"{'-' if mem is None else f'{mem / 1024:.1f}':>7} {s.get('gpu_util_mean', '-'):>5} "
                         f"{s.get('hit_cap', 0):>3} {'-' if share is None else f'{100 * share:.0f}':>4} "
                         f"{s.get('errors', 0):>3}" + ("  incomplete: runs again" if not scenario_done(s) else ""))
        lines.append("  usable: audio of requests below the length cap (a capped one is runaway audio the official API "
                     "re-renders); $/audio-h per usable hour")
    cs = res.get("compare_summary") or compare_summary(res.get("compare") or {})  # a killed worker wrote none
    for g, s in sorted(cs.items()):
        if not g.startswith("official/") or s.get("rtf") is None:
            continue
        ts, mode = g.split("/")[1:]
        best = max(((v.get("usable_audio_s_per_gpu_s") or 0, k) for k, v in sc.items()
                    if k.startswith(f"{ts}/{mode}/") and scenario_done(v)), default=None)
        line = (f"[probe] official API {ts}/{mode}: RTF {s['rtf']:.2f} on {s['clips']} clips, one request at a time "
                f"(~${usd_per_h * s['rtf']:.2f} per audio hour{f' on {gpu}' if gpu else ''})")
        if best and best[0]:
            line += f"; Nano-vLLM best {best[0]:.1f} usable audio-s/GPU-s ({best[1]}) = {best[0] * s['rtf']:.0f}x"
        lines.append(line)
    if delta := cs.get("nano_vs_official"):
        lines.append("[probe] same sentences, Whisper WER % / UTMOS (nano vs official), nano/official duration:")
        for g, d in delta.items():
            lines.append(f"  {g:<12} {d['pairs']} pairs  WER {d.get('wer_nano', '-')} vs {d.get('wer_official', '-')}"
                         f"  UTMOS {d.get('utmos_nano', '-')} vs {d.get('utmos_official', '-')}"
                         f"  duration x{d.get('duration_ratio', '-')}"
                         + (f"  ({d['nano_capped']} nano clips hit the length cap)" if d.get("nano_capped") else ""))
    lines += [f"[probe] skipped: {s}" for s in res.get("skipped") or []]
    lines += [f"[probe] error: {e}" for e in res.get("errors") or []]
    if res and not res.get("finished"):
        lines.append("[probe] unfinished: run the same command with --run " + str((res.get("spec") or {}).get("run"))
                     + " to continue")
    return lines


# ---------------------------------------------------------------- GPU worker (runs in the container)

async def run_batch(server, items: list[dict], n: int, total: int, *, cfg: float, seed: int | None,
                    ref_latents: bytes | None = None, lora_name: str | None = None,
                    keep: tuple | list | set = ()) -> tuple[list[dict], dict, float]:
    """`total` requests over items (request i: items[i % len]) with at most n in flight on a Nano-vLLM server pool
    (its async generate). Returns (request records for batch_stats, the kept one marked "kept"; {id: audio chunks} of
    the first request of each id in keep, wall seconds)."""
    sem, kept = asyncio.Semaphore(n), {}

    async def one(i: int) -> dict:
        it = items[i % len(items)]
        cap = it.get("max_len", MAX_GEN_CAP)
        async with sem:
            chunks = None
            if it["id"] in keep and it["id"] not in kept:
                chunks = kept[it["id"]] = []
            t0, first, samples, steps = time.perf_counter(), None, 0, 0
            try:
                async for c in server.generate(target_text=it["model_text"], max_generate_length=cap, cfg_value=cfg,
                                               ref_audio_latents=ref_latents, lora_name=lora_name, seed=seed):
                    if first is None:
                        first = time.perf_counter()
                    samples, steps = samples + len(c), steps + 1
                    if chunks is not None:
                        chunks.append(c)
            except Exception as e:  # one failed request is counted, the batch goes on
                if chunks is not None:
                    kept.pop(it["id"], None)
                return {"i": i, "id": it["id"], "error": f"{type(e).__name__}: {e}"}
            t1 = time.perf_counter()
        return {"i": i, "id": it["id"], "t0": t0, "first": t1 if first is None else first, "t1": t1,
                "samples": samples, "steps": steps, "cap": cap, "kept": chunks is not None}

    t = time.perf_counter()
    reqs = await asyncio.gather(*(one(i) for i in range(total)))
    return list(reqs), kept, time.perf_counter() - t


def _smi() -> tuple[int, int] | None:
    try:
        return parse_smi(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                        timeout=5).stdout)
    except Exception:
        return None


class _GpuSampler:
    """nvidia-smi every every_s seconds in a thread: (time.time(), MiB used, % utilization). Nano-vLLM's engine runs in
    its own process, so torch's counters of this one would see nothing."""

    def __init__(self, every_s: float = 0.5):
        self.every_s, self.samples, self._stop = every_s, [], threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(self.every_s):
            if (s := _smi()) is not None:
                self.samples.append((time.time(), *s))

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


class _Probe:
    """The GPU work of one probe call, in priority order; results.json is rewritten after every step."""

    def __init__(self, spec: dict, deadline: float):
        self.spec, self.deadline = spec, deadline
        self.out = Path(spec["out"])
        self.items = {it["id"]: it for it in spec["items"]}
        self.price = GPU_PRICE.get(spec.get("gpu", GPU), GPU_PRICE["A100-80GB"]) + SIDE_PRICE
        self.server_s = SERVER_START_S
        p = self.out / "results.json"
        try:
            self.res = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.res = {}
        spec_head = {k: v for k, v in spec.items() if k != "items"}
        self.res = {"spec": spec_head, "env": {}, "servers": {}, "scenarios": {}, "compare": {},
                    **{k: v for k, v in self.res.items() if k in ("env", "servers", "scenarios", "compare")},
                    "skipped": [], "errors": [], "finished": False}

    def left(self) -> float:
        return self.deadline - time.time()

    def save(self) -> None:
        _write_atomic(self.out / "results.json", json.dumps(self.res, indent=1, ensure_ascii=False) + "\n")

    def skip(self, what: str, need: float) -> None:
        msg = f"{what} (needs ~{need:.0f} s, {max(self.left(), 0):.0f} s left)"
        self.res["skipped"].append(msg)
        print(f"[probe] skipped: {msg}", flush=True)

    def error(self, what: str, e: BaseException) -> None:
        traceback.print_exc()
        self.res["errors"].append(f"{what}: {type(e).__name__}: {e}")

    def finished(self) -> bool:
        """Nothing skipped or failed, and every scenario of the spec measured without a failed request."""
        sp = self.spec
        return not self.res["skipped"] and not self.res["errors"] and all(
            scenario_done(self.res["scenarios"].get(scenario_key(ts, m, n)))
            for ts in sp["timesteps"] for m in sp["modes"] for n in sp["concurrency"])

    def run(self) -> int:
        with _GpuSampler() as self.smi:
            try:
                self.env_check()
            except Exception as e:  # flash-attn / CUDA / package mismatch: nothing else can work, stop paying now
                self.error("env check", e)
                self.save()
                return 3
            self.save()
            for ts in self.spec["timesteps"]:
                asyncio.run(self.nano(ts))
            if self.spec["compare"]:
                self.official()
                self.judge()
            self.res["compare_summary"] = compare_summary(self.res["compare"])
            self.res["finished"] = self.finished()
            self.save()
        print("\n".join(format_report(self.res, self.price)), flush=True)
        return 0

    def env_check(self) -> None:
        import importlib.metadata as md

        import flash_attn  # noqa: F401  (a wheel built for another torch / ABI fails here: undefined symbol)
        import torch
        from flash_attn import flash_attn_varlen_func  # noqa: F401

        env = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
               "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI), "cuda_available": torch.cuda.is_available()}
        for pkg in ("flash_attn", "nano-vllm-voxcpm", "voxcpm", "triton", "transformers", "faster-whisper"):
            try:
                env[pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                env[pkg] = None
        try:  # without creating a CUDA context in this process (Nano-vLLM's engine is another process)
            q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10).stdout.splitlines()[0].split(",")
            env["gpu"], env["gpu_mem_total_mib"] = q[0].strip(), int(float(q[1]))
        except Exception as e:
            env["gpu"] = f"? ({type(e).__name__})"
        if not env["cuda_available"]:
            raise RuntimeError("torch sees no CUDA device")
        missing = [f for f in MODEL_FILES if not Path(MODEL_DIR, f).is_file()]
        if missing:
            raise FileNotFoundError(f"{MODEL_DIR}: no {', '.join(missing)} (modal_app.py --action prepare)")
        from transformers import LlamaTokenizerFast

        from nanovllm_voxcpm.models.voxcpm2.utils import mask_multichar_chinese_tokens

        tok = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(MODEL_DIR))
        for it in self.spec["items"]:  # the cap the official API's first attempt has (Nano-vLLM: no retry)
            it["max_len"] = max_gen_len(len(tok(it["model_text"])))
        self.res["env"] = env
        print(f"[probe] {json.dumps(env)}", flush=True)

    async def nano(self, ts: int) -> None:
        sp = self.spec
        todo = [(m, n) for m in sp["modes"] for n in sp["concurrency"]
                if not scenario_done(self.res["scenarios"].get(scenario_key(ts, m, n)))]
        if not todo:
            print(f"[probe] nano-vllm ts{ts}: all scenarios done before a restart", flush=True)
            return
        if self.left() < (need := self.server_s + SERVER_STOP_S + WARMUP_S + scenario_s(todo[0][1], sp["rounds"], ts)):
            return self.skip(f"nano-vllm ts{ts} ({len(todo)} scenarios)", need)
        server, t = None, time.time()
        try:
            from nanovllm_voxcpm import VoxCPM as NanoVoxCPM

            kw = {"model": MODEL_DIR, "inference_timesteps": ts, "max_num_seqs": sp["max_num_seqs"],
                  "max_model_len": 4096, "gpu_memory_utilization": sp["gpu_mem"], "devices": [0]}
            if sp["lora"]:
                from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

                kw["lora_config"] = LoRAConfig(**sp["lora_kwargs"])
            server = NanoVoxCPM.from_pretrained(**kw)
            await server.wait_for_ready()
            srv = {"ready_s": round(time.time() - t, 1), "mem_ready_mib": (_smi() or (None,))[0]}
            self.server_s = max(srv["ready_s"], 30.0)
            lora_name = None
            if sp["lora"]:
                await server.register_lora("probe", sp["lora"])
                lora_name, srv["lora"] = "probe", sp["lora"]
            sr = int((await server.get_model_info())["output_sample_rate"])
            srv["sample_rate"] = sr
            ref_lat = await server.encode_latents(Path(sp["ref"]).read_bytes(), "wav") \
                if any(m == "ref" for m, _ in todo) else None
            self.res["servers"][f"ts{ts}"] = srv
            self.save()
            print(f"[probe] nano-vllm ts{ts}: {srv}", flush=True)
            for mode in sp["modes"]:
                ns = [n for m, n in todo if m == mode]
                if not ns:
                    continue
                lat = ref_lat if mode == "ref" else None
                if self.left() < (need := WARMUP_S + SERVER_STOP_S + scenario_s(ns[0], sp["rounds"], ts)):
                    self.skip(f"nano-vllm ts{ts}/{mode}", need)
                    continue
                w = min(8, max(ns))  # first requests: Triton autotuning, CUDA graph warm-up; not measured
                warm, _, _ = await run_batch(server, sp["items"], w, w, cfg=sp["cfg"], seed=sp["seed"], ref_latents=lat,
                                             lora_name=lora_name)
                if dead := server_failure(warm):
                    raise RuntimeError(f"warm-up of ts{ts}/{mode}: {dead}; this server's scenarios are not run")
                for n in ns:
                    key = scenario_key(ts, mode, n)
                    if self.left() < (need := 1.5 * scenario_s(n, sp["rounds"], ts) + SERVER_STOP_S):
                        self.skip(key, need)
                        continue
                    keep = sp["compare"] if n == min(sp["concurrency"]) else ()
                    t0 = time.time()
                    reqs, kept, wall = await run_batch(server, sp["items"], n, sp["rounds"] * n, cfg=sp["cfg"],
                                                       seed=sp["seed"], ref_latents=lat, lora_name=lora_name, keep=keep)
                    st = {**batch_stats(reqs, sr, wall, self.price), **window_stats(self.smi.samples, t0, time.time()),
                          "gpu": sp["gpu"]}
                    if st.get("errors"):  # kept for the report; a resume runs it again, the best pick skips it
                        st["incomplete"] = True
                    self.res["scenarios"][key] = st
                    self.keep_wavs("nano", ts, mode, kept, sr,
                                   capped={r["id"] for r in reqs if r.get("kept") and r["steps"] >= r["cap"]})
                    self.save()
                    print(f"[probe] {key}: {json.dumps(st)}", flush=True)
                    if dead := server_failure(reqs):  # a dead engine fails every later request at once
                        raise RuntimeError(f"{key}: {dead}; this server's remaining scenarios are not run")
                    if st.get("errors"):
                        self.res["errors"].append(f"{key}: {st['errors']} of {st['requests']} requests failed "
                                                  f"({st['first_error']})")
        except Exception as e:  # recorded; the next timesteps value and the official API still run
            self.error(f"nano-vllm ts{ts}", e)
        finally:
            if server is not None:
                try:
                    await asyncio.wait_for(server.stop(), SERVER_STOP_S)
                except Exception as e:
                    print(f"[probe] server stop: {type(e).__name__}: {e}", flush=True)
            self.save()

    def keep_wavs(self, engine: str, ts: int, mode: str, kept: dict, sr: int, capped: set | tuple = ()) -> None:
        """The kept clips as WAVs and compare rows (hit_cap: the request ran to its length cap)."""
        if not (kept := {i: c for i, c in kept.items() if c}):
            return
        import numpy as np
        import soundfile as sf

        for i, chunks in kept.items():
            wav = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks])
            path = self.out / "wav" / engine / f"ts{ts}-{mode}" / f"{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, wav, sr)
            row = {"engine": engine, "ts": ts, "mode": mode, "id": i, "set": self.items[i]["set"], "wav": str(path),
                   "duration": round(len(wav) / sr, 3), "gpu": self.spec["gpu"]}
            if i in capped:
                row["hit_cap"] = True
            self.res["compare"][f"{engine}/ts{ts}/{mode}/{i}"] = row

    def official(self) -> None:
        sp = self.spec
        todo = [(ts, m, i) for ts in sp["timesteps"] for m in sp["modes"] for i in sp["compare"]
                if f"official/ts{ts}/{m}/{i}" not in self.res["compare"]]
        if not todo:
            return
        if self.left() < (need := OFFICIAL_LOAD_S + official_clip_s(todo[0][0]) + JUDGE_S):
            return self.skip(f"official API ({len(todo)} clips)", need)
        tts = None
        try:  # as synth_eval.py: no denoiser, no torch.compile, VoxCPM's normalizer off, seed per clip
            import numpy as np
            import soundfile as sf
            import torch
            from voxcpm import VoxCPM

            t = time.time()
            tts = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False, optimize=False,
                                         lora_weights_path=sp["lora"] or None)
            tts.generate(text="Merhaba.", cfg_value=sp["cfg"], inference_timesteps=10, normalize=False, seed=0)
            sr = tts.tts_model.sample_rate
            self.res["official_load_s"] = round(time.time() - t, 1)
            for n, (ts, mode, i) in enumerate(todo):
                if self.left() < (need := 1.5 * official_clip_s(ts) + JUDGE_S + JUDGE_CLIP_S * 2 * len(todo)):
                    self.skip(f"official API: {len(todo) - n} of {len(todo)} clips", need)
                    break
                t = time.time()
                wav = tts.generate(text=self.items[i]["model_text"], reference_wav_path=sp["ref"] if mode == "ref"
                                   else None, cfg_value=sp["cfg"], inference_timesteps=ts, normalize=False,
                                   seed=sp["seed"])
                torch.cuda.synchronize()
                dt = time.time() - t
                path = self.out / "wav" / "official" / f"ts{ts}-{mode}" / f"{i}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                wav = np.asarray(wav, dtype=np.float32)
                sf.write(path, wav, sr)
                self.res["compare"][f"official/ts{ts}/{mode}/{i}"] = {
                    "engine": "official", "ts": ts, "mode": mode, "id": i, "set": self.items[i]["set"],
                    "wav": str(path), "duration": round(len(wav) / sr, 3), "synth_s": round(dt, 3), "gpu": sp["gpu"]}
                self.save()
        except Exception as e:
            self.error("official API", e)
        finally:
            del tts  # its GPU memory goes before Whisper loads
            if "torch" in sys.modules:
                sys.modules["torch"].cuda.empty_cache()
            self.save()

    def judge(self) -> None:
        """synth_eval's judge on the comparison clips: WER against the closest ref, then UTMOS."""
        rows = list(self.res["compare"].values())
        if not rows:
            return
        if self.left() < (need := JUDGE_S + JUDGE_CLIP_S * len(rows)):
            return self.skip(f"judge ({len(rows)} clips)", need)
        try:
            from faster_whisper import WhisperModel
            from synth_eval import wer_parts

            judge = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
            for r in (r for r in rows if "errors" not in r):
                segs, _ = judge.transcribe(r["wav"], language="tr", beam_size=5)
                r["hyp"] = " ".join(s.text.strip() for s in segs).strip()
                refs = self.items[r["id"]]["refs"]
                best = min(refs, key=lambda x: wer_parts(x, r["hyp"])[0] / max(wer_parts(x, r["hyp"])[1], 1))
                r["errors"], r["ref_words"] = wer_parts(best, r["hyp"])
            del judge
            self.save()
        except Exception as e:
            self.error("judge: Whisper", e)
        try:
            from score_utmos import load_predictor, score_wav

            pred = load_predictor("cuda")
            for r in (r for r in rows if r.get("utmos") is None):
                try:
                    r["utmos"] = round(score_wav(pred, r["wav"], "cuda"), 4)
                except Exception as e:  # one bad clip costs one row
                    r["utmos_error"] = f"{type(e).__name__}: {e}"
        except Exception as e:
            self.error("judge: UTMOS", e)
        self.save()


def worker_main(argv: list[str] | None = None) -> int:
    """In the container: python -c '<WORKER_BOOT>' --spec <out>/spec.json --deadline <epoch s>."""
    ap = argparse.ArgumentParser(description="nanovllm_app GPU worker")
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--deadline", type=float, default=math.inf, help="time.time() by which the worker must be done")
    a = ap.parse_args(argv)
    _repo_modules(Path(REPO))
    return _Probe(json.loads(a.spec.read_text(encoding="utf-8")), a.deadline).run()


# ---------------------------------------------------------------- Modal: the GPU call, its deadline, local actions

def _kill(p: subprocess.Popen, wait: bool = True) -> None:
    """SIGKILL p and its process group (start_new_session): the worker and Nano-vLLM's engine processes."""
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        p.kill()
    if wait:
        p.wait()


def _run_until(cmd: list[str], deadline: float, live: list) -> int | None:
    """Run cmd in its own session until deadline (time.time()); killed then and None returned."""
    print("$ " + " ".join(cmd[:3]) + " ... " + " ".join(cmd[4:]), flush=True)
    p = subprocess.Popen(cmd, cwd=REPO, start_new_session=True)
    live.append(p)
    try:
        return p.wait(timeout=max(deadline - time.time(), 0.0))
    except subprocess.TimeoutExpired:
        _kill(p)
        return None
    except BaseException:  # SIGINT on preemption etc.: never leave the GPU busy
        _kill(p)
        raise
    finally:
        live.remove(p)


class _PeriodicCommit:
    """Commit the volume every few minutes so results survive preemption."""

    def __init__(self, every_s: int = 60):
        self.every_s, self._stop = every_s, threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(self.every_s):
            try:
                volume.commit()
            except Exception as e:
                print(f"[modal] volume commit failed: {e}", flush=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=120)
        volume.commit()


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, volumes={DATA: volume}, timeout=PROBE_TIMEOUT_S,
              scaledown_window=IDLE_S, secrets=SECRETS)
def probe(spec: dict, max_s: float, run_id: str) -> str:
    """The worker (worker_main, its own process group) with a hard deadline of the whole call incl. start-up (START_S)
    and attempts a preemption cut short (ledger <run_id>.json on the Volume); killed then, and a watchdog ends the
    container if anything else hangs. Returns {"rc" (None: killed at the deadline), "used_s", "results"} or
    {"error", "used_s"}."""
    t0 = time.time()
    ledger = Path(DATA) / "modal_ledger" / f"{run_id}.json"
    used0 = (json.loads(ledger.read_text())["used_s"] if ledger.exists() else 0.0) + START_S
    end = t0 + max_s - used0 - IDLE_S  # the container must be gone by then
    if end - TAIL_S - time.time() < MIN_S:
        return json.dumps({"error": f"budget of {max_s:.0f} s used up by earlier attempts", "used_s": round(used0)})
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    _write_atomic(out / "spec.json", json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
    live, stop = [], threading.Event()

    def save_ledger():
        try:
            ledger.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(ledger, json.dumps({"used_s": used0 + time.time() - t0, "max_s": max_s}))
        except Exception as e:  # the deadline itself does not depend on the ledger
            print(f"[probe] ledger not written: {e}", flush=True)

    def bookkeeping():
        while not stop.wait(30):
            save_ledger()

    def hard_stop():
        print(f"[probe] hard deadline ({max_s:.0f} s budget) reached: killing everything", flush=True)
        for p in list(live):
            _kill(p, wait=False)
        os._exit(124)

    watchdog = threading.Timer(max(end - time.time(), 15.0), hard_stop)
    watchdog.daemon = True
    watchdog.start()
    threading.Thread(target=bookkeeping, daemon=True).start()
    boot = (f"import sys; sys.path.insert(0, {str(_HERE.parent)!r}); import nanovllm_app as m; "
            "sys.exit(m.worker_main(sys.argv[1:]))")
    try:
        with _PeriodicCommit(60):
            try:
                rc = _run_until(["python", "-u", "-c", boot, "--spec", str(out / "spec.json"), "--deadline",
                                 f"{end - TAIL_S:.1f}"], end - TAIL_S, live)
            finally:
                stop.set()
                save_ledger()  # before _PeriodicCommit's final commit
    finally:
        watchdog.cancel()
    if rc is None:
        print("[probe] stopped at the budget deadline; the same --run continues", flush=True)
    try:
        res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        res = {}
    return json.dumps({"rc": rc, "used_s": round(used0 + time.time() - t0), "results": res}, ensure_ascii=False)


def _vol_files(rel: str) -> dict:
    """name -> FileEntry of a Volume directory; {} when it is not there."""
    try:
        return {PurePosixPath(e.path).name: e for e in volume.listdir(rel)}
    except Exception:
        return {}


def _vol_read(rel: str) -> bytes | None:
    try:
        return b"".join(volume.read_file(rel))
    except Exception:
        return None


def _rel(p: str) -> str:
    return p[len(DATA) + 1:]


def _preflight(spec: dict) -> None:
    """Runs on the local machine before a GPU starts: the model on the Volume, the LoRA (its lora_config.json ->
    spec["lora_kwargs"]), the reference clip (default: the first Antalia training clip of REF_SECONDS), and an
    earlier results.json of this --run made with the same settings."""
    if missing := [f for f in MODEL_FILES if f not in _vol_files(_rel(MODEL_DIR))]:
        raise SystemExit(f"{MODEL_DIR}: no {', '.join(missing)} on the Volume: run modal_app.py --action prepare")
    if spec["lora"]:
        names = _vol_files(_rel(spec["lora"]))
        raw = _vol_read(f"{_rel(spec['lora'])}/lora_config.json")
        if "lora_weights.safetensors" not in names or raw is None:
            raise SystemExit(f"--lora {spec['lora']}: needs lora_weights.safetensors and lora_config.json there")
        try:
            spec["lora_kwargs"] = nano_lora_kwargs(json.loads(raw))
        except ValueError as e:
            raise SystemExit(f"--lora {spec['lora']}: {e}")
    if "ref" in spec["modes"]:
        if not spec["ref"]:
            raw = _vol_read(f"{_rel(CORPUS)}/voxcpm_train.jsonl")
            rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()] if raw else []
            if (r := pick_ref(rows)) is None:
                raise SystemExit(f"no {CORPUS}/voxcpm_train.jsonl for the default reference: give --ref "
                                 f"{DATA}/....wav or --modes noref")
            spec["ref"], spec["ref_duration"] = r["audio"], r["duration"]
        elif PurePosixPath(spec["ref"]).name not in _vol_files(str(PurePosixPath(_rel(spec["ref"])).parent)):
            raise SystemExit(f"--ref {spec['ref']}: not on the Volume")
    if raw := _vol_read(f"{_rel(spec['out'])}/results.json"):
        try:
            old = json.loads(raw).get("spec") or {}
        except ValueError:
            old = {}
        if bad := resume_conflicts(old, spec):
            raise SystemExit(f"--run {spec['run']} exists with other settings ({'; '.join(bad)}): choose another "
                             "--run, or the same settings to continue it")
        print(f"[probe] --run {spec['run']} exists: finished scenarios are kept, the rest runs")


def _download(path: str, size: int, out: Path) -> None:
    """One Volume file -> out, whole or not at all (.part, size check, rename)."""
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


def _fetch(run: str, dest: Path) -> dict:
    """Runs on the local machine: nanovllm-probe/<run>/ (results.json, spec.json, compare WAVs) -> dest/<run>/,
    results.json last; returns the results (or {})."""
    remote, local = f"{_rel(PROBE_ROOT)}/{run}", dest / run
    try:
        files = {PurePosixPath(e.path.lstrip("/")).relative_to(remote): e
                 for e in volume.listdir(remote, recursive=True) if e.type == modal.volume.FileEntryType.FILE}
        for rel, e in sorted(files.items(), key=lambda kv: str(kv[0]) == "results.json"):
            _download(e.path, e.size, local.joinpath(*rel.parts))
    except Exception as e:  # not there, a broken connection, ...
        print(f"[fetch] {remote}: {e}")
        return {}
    print(f"[fetch] {len(files)} files -> {local}")
    try:
        return json.loads((local / "results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _plan_lines(spec: dict, max_s: float, price: float) -> list[str]:
    phases = estimate(spec)
    sets = Counter(it["set"] for it in spec["items"])
    lines = [f"[probe] run {spec['run']}: {len(spec['items'])} sentences of {spec.get('sentences', 'builtin')} ("
             + ", ".join(f"{s} {n}" for s, n in sets.items())
             + f"), timesteps {spec['timesteps']}, modes {spec['modes']}, concurrency {spec['concurrency']} x "
             f"{spec['rounds']} rounds, compare {len(spec['compare'])} sentences {spec['compare']}"
             + (f", LoRA {spec['lora']}" if spec["lora"] else "") + (f", ref {spec['ref']}" if spec["ref"] else "")]
    left = max_s - START_S - TAIL_S - IDLE_S
    for name, s in phases:
        left -= s
        lines.append(f"  {name:<24} ~{s / 60:4.1f} min" + ("" if left >= 0 else "  -> may be skipped (budget)"))
    total = sum(s for _, s in phases) + START_S + TAIL_S + IDLE_S
    lines.append(f"[probe] at most ${max_s / 3600 * price:.2f}: hard limit {max_s / 60:.1f} min on {GPU} at "
                 f"${price:.2f}/h incl. CPU/RAM; rough estimate {total / 60:.1f} min (~${total / 3600 * price:.2f})")
    return lines


@app.local_entrypoint()
def main(action: str = "plan", max_usd: float = PROBE_MAX_USD, run: str = "", lora: str = "", ref: str = "",
         timesteps: str = "10,16", concurrency: str = "16,32,64", modes: str = "noref,ref", rounds: int = 2,
         compare: int = 4, cfg: float = 2.0, gpu_mem: float = 0.85, sentences: str = "",
         out: str = LOCAL_OUT):
    if GPU not in GPU_PRICE:
        raise SystemExit(f"unknown GPU {GPU}; choose from {sorted(GPU_PRICE)}")
    price = GPU_PRICE[GPU] + SIDE_PRICE
    if action == "fetch":
        if not run:
            raise SystemExit("name the probe: --action fetch --run probe-...")
        print("\n".join(format_report(_fetch(run, Path(out)), price)))
        return
    if action not in ("plan", "probe"):
        raise SystemExit("action must be one of: plan, probe, fetch")
    if not (math.isfinite(max_usd) and max_usd > 0):
        raise SystemExit("--max-usd must be > 0")
    try:
        spec = build_spec(run, timesteps, concurrency, modes, rounds, compare, cfg, gpu_mem, lora, ref, sentences)
    except ValueError as e:
        raise SystemExit(str(e))
    max_s = min(usd_to_seconds(max_usd, price), PROBE_TIMEOUT_S)
    if max_s < (need := START_S + TAIL_S + IDLE_S + MIN_S):
        raise SystemExit(f"--max-usd {max_usd:g} buys {max_s:.0f} s; the probe needs at least "
                         f"${math.ceil(need / 3600 * price * 100) / 100:.2f}")
    if action == "probe":
        _preflight(spec)
    print("\n".join(_plan_lines(spec, max_s, price)))
    if action == "plan":
        return
    try:
        res = json.loads(probe.remote(spec, max_s, f"nanovllm-probe-{spec['run']}-{int(time.time())}"))
    except Exception as e:  # e.g. the watchdog's hard stop: what is done is on the Volume all the same
        print(f"[probe] ended with an error: {e}")
    else:
        if "error" in res:
            print(f"[probe] {res['error']}")
        used = res.get("used_s", 0)  # incl. a start-up allowance per attempt
        note = {None: "; stopped at the deadline: the same --run continues", 0: "", 3: "; env check failed"}.get(
            res.get("rc"), f"; worker exit code {res.get('rc')}")
        print(f"[probe] GPU container ~{used / 60:.1f} min, ~${used / 3600 * price:.2f}{note}")
    print("\n".join(format_report(_fetch(spec["run"], Path(out)), price)))


def cli(argv: list[str] | None = None) -> int:
    """python tts/voxcpm/nanovllm_app.py [--timesteps 10,16 ...]: the plan of a probe, locally, without Modal (free);
    with --spec: the GPU worker (as probe() starts it)."""
    argv = sys.argv[1:] if argv is None else argv
    if any(a == "--spec" or a.startswith("--spec=") for a in argv):
        return worker_main(argv)
    ap = argparse.ArgumentParser(description=cli.__doc__)
    for opt, default in (("--run", ""), ("--timesteps", "10,16"), ("--concurrency", "16,32,64"),
                         ("--modes", "noref,ref"), ("--lora", ""), ("--ref", ""), ("--sentences", "")):
        ap.add_argument(opt, default=default)
    for opt, typ, default in (("--max-usd", float, PROBE_MAX_USD), ("--rounds", int, 2), ("--compare", int, 4),
                              ("--cfg", float, 2.0), ("--gpu-mem", float, 0.85)):
        ap.add_argument(opt, type=typ, default=default)
    a = ap.parse_args(argv)
    try:
        spec = build_spec(a.run, a.timesteps, a.concurrency, a.modes, a.rounds, a.compare, a.cfg, a.gpu_mem, a.lora,
                          a.ref, a.sentences)
    except ValueError as e:
        ap.error(str(e))
    price = GPU_PRICE.get(GPU, GPU_PRICE["A100-80GB"]) + SIDE_PRICE
    print("\n".join(_plan_lines(spec, min(usd_to_seconds(a.max_usd, price), PROBE_TIMEOUT_S), price)))
    for it in spec["items"]:
        print(f"  {it['id']:<18} {it['model_text']}")
    return 0


if __name__ == "__main__":  # the Modal CLI imports this file (`modal run`), it never runs it as a script
    raise SystemExit(cli())
