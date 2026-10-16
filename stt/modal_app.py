"""Run the Turkish streaming-ASR recipe on Modal (https://modal.com) GPUs.

The same scripts as on the laptop (prepare.sh, train.sh, export_and_test.sh) run
inside a pinned image built by setup_env.sh. Data, features and checkpoints live
in a persistent Modal Volume, so every step can be stopped and resumed.

Setup (once):   pip install modal && modal setup
                (or export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET)

Steps, from the repo root:

    modal run stt/modal_app.py --action prepare                 # CPU, ~3-5 h, ~$3-5
    modal run stt/modal_app.py --action probe                   # GPU 10 min, prints real speed + cost plan
    modal run --detach stt/modal_app.py --action train --max-usd 20
    modal run stt/modal_app.py --action export                  # CPU: ONNX fp32/int8 + WER/CER
    modal volume get turkish-stt-data exp/<run>/<export-dir> ./exports/

GPU type: STT_GPU=H100|A100-80GB|L40S|L4 (default A100-80GB), read at import time,
e.g. `STT_GPU=H100 modal run --detach stt/modal_app.py --action train`.

Cost control: Modal bills per second. `train` turns --max-usd into a hard time
limit (train.sh --hours) using the prices below, so a run cannot spend more than
that (plus container start-up). Also set a workspace spending limit in Modal.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

REPO = "/repo"
DATA = "/data"
VENV = "/opt/venv-stt"
ICEFALL = "/opt/icefall"
LOCAL_REPO = Path(__file__).resolve().parents[1]

GPU = os.environ.get("STT_GPU", "A100-80GB")
# USD per hour, Modal list prices (2026); check modal.com/pricing before large runs.
GPU_PRICE = {"H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95, "L4": 0.80}
# CPU + memory of the GPU container (4 cores, 16 GiB) and of the CPU containers (8 cores, 32 GiB)
GPU_SIDE_PRICE = 4 * 0.047 + 16 * 0.008
CPU_JOB_PRICE = 8 * 0.047 + 32 * 0.008
# seconds of audio per batch that fit each card for the two presets
MAX_DURATION = {
    "H100": {"small": 1000, "medium": 1000},
    "A100-80GB": {"small": 1000, "medium": 1000},
    "L40S": {"small": 1000, "medium": 800},
    "L4": {"small": 650, "medium": 450},
}

app = modal.App("turkish-stt")
volume = modal.Volume.from_name("turkish-stt-data", create_if_missing=True)
# optional: a Modal secret holding HF_TOKEN (higher Hugging Face download limits),
# created with `modal secret create huggingface HF_TOKEN=...`; pass its name in STT_HF_SECRET
SECRETS = [modal.Secret.from_name(os.environ["STT_HF_SECRET"])] if os.environ.get("STT_HF_SECRET") else []

_IGNORE = [
    ".git",
    ".venv*",
    "data",
    "build",
    "dist",
    "**/__pycache__",
    "**/*.pyc",
    "stt/icefall",
    "stt/.env.local",
    "benchmarks/**/results",
    "benchmarks/**/logs",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "ca-certificates", "libsndfile1", "ffmpeg", "build-essential")
    # only what the environment build needs, so code edits do not rebuild the venv
    .add_local_file(LOCAL_REPO / "pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
    .add_local_file(LOCAL_REPO / "README.md", f"{REPO}/README.md", copy=True)
    .add_local_file(LOCAL_REPO / "LICENSE", f"{REPO}/LICENSE", copy=True)
    .add_local_dir(LOCAL_REPO / "src", f"{REPO}/src", copy=True, ignore=["**/__pycache__"])
    .add_local_file(LOCAL_REPO / "stt/setup_env.sh", f"{REPO}/stt/setup_env.sh", copy=True)
    .add_local_file(LOCAL_REPO / "stt/path.sh", f"{REPO}/stt/path.sh", copy=True)
    .add_local_file(LOCAL_REPO / "stt/requirements-stt.txt", f"{REPO}/stt/requirements-stt.txt", copy=True)
    .add_local_file(LOCAL_REPO / "stt/constraints-stt.txt", f"{REPO}/stt/constraints-stt.txt", copy=True)
    .add_local_dir(LOCAL_REPO / "stt/locks", f"{REPO}/stt/locks", copy=True)
    .run_commands(f"cd {REPO} && bash stt/setup_env.sh --venv {VENV} --icefall-dir {ICEFALL} --no-verify")
    .env({"VENV": VENV, "ICEFALL_DIR": ICEFALL, "STT_DATA": DATA, "PYTHONUNBUFFERED": "1"})
    # the rest of the code is mounted at start, so edits apply without a rebuild
    .add_local_dir(LOCAL_REPO, REPO, ignore=_IGNORE)
)


def _sh(cmd: str, threads: int) -> None:
    """Run a recipe command with the pinned environment, streaming its output."""
    full = (
        f"cd {REPO} && export VENV={VENV} ICEFALL_DIR={ICEFALL} STT_DATA={DATA} "
        f"OMP_NUM_THREADS={threads} && source stt/path.sh && {cmd}"
    )
    print(f"$ {cmd}", flush=True)
    rc = subprocess.run(["bash", "-lc", full]).returncode
    if rc not in (0, 75):  # 75 = train.sh stopped cleanly at its --hours limit
        raise RuntimeError(f"command failed with exit code {rc}: {cmd}")


class _PeriodicCommit:
    """Commit the volume every few minutes so checkpoints survive preemption."""

    def __init__(self, every_s: int = 300):
        self.every_s = every_s
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.every_s):
            try:
                volume.commit()
            except Exception as e:  # keep training even if a commit fails
                print(f"[modal] volume commit failed: {e}", flush=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        volume.commit()


@app.function(image=image, cpu=8, memory=32768, volumes={DATA: volume}, timeout=24 * 3600, secrets=SECRETS)
def prepare(args: str = "--cv-train-from validated --num-workers 8 --fbank-device cpu") -> None:
    with _PeriodicCommit(600):
        _sh(f"bash ./stt/prepare.sh {args}", threads=8)


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=16384,
    volumes={DATA: volume},
    timeout=24 * 3600,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
)
def train(
    preset: str, hours: float, max_duration: int, num_epochs: int = 0, exp_name: str = "", extra: str = "", run_id: str = ""
) -> None:
    exp = f"--exp-dir {DATA}/exp/{exp_name}" if exp_name else ""
    epochs = f"--num-epochs {num_epochs}" if num_epochs else ""
    tail = f" -- {extra}" if extra else ""
    # Budget ledger: GPU time already used by earlier attempts of this run (preemption retries),
    # so a retry continues the same budget instead of starting a fresh --hours window.
    ledger = Path(DATA) / "modal_ledger" / f"{run_id or 'default'}.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    used = json.loads(ledger.read_text())["used_h"] if ledger.exists() else 0.0
    left = hours - used
    if left < 0.02:
        print(f"[modal] budget for run {run_id} already used ({used:.2f} of {hours:.2f} h); nothing to do")
        return
    t0 = time.time()
    stop = threading.Event()

    def bookkeeping() -> None:
        while not stop.wait(60):
            ledger.write_text(json.dumps({"used_h": used + (time.time() - t0) / 3600, "limit_h": hours}))

    threading.Thread(target=bookkeeping, daemon=True).start()
    try:
        with _PeriodicCommit(300):
            # train.sh resumes from the newest checkpoint on its own (also after a preemption retry)
            _sh(
                f"bash ./stt/train.sh --preset {preset} --hours {left:.3f} --no-guard --max-duration {max_duration} "
                f"--num-workers 4 {epochs} {exp}{tail}",
                threads=4,
            )
    finally:
        stop.set()
        ledger.write_text(json.dumps({"used_h": used + (time.time() - t0) / 3600, "limit_h": hours}))
        volume.commit()


@app.function(image=image, cpu=4, memory=16384, volumes={DATA: volume}, timeout=60)
def eta(exp_name: str) -> str:
    out = subprocess.run(
        ["bash", "-lc", f"cd {REPO} && source stt/path.sh && python stt/tools/eta.py --exp-dir {DATA}/exp/{exp_name}"],
        capture_output=True,
        text=True,
    )
    return out.stdout + out.stderr


@app.function(image=image, cpu=8, memory=32768, volumes={DATA: volume}, timeout=6 * 3600)
def export(preset: str = "small", args: str = "") -> None:
    with _PeriodicCommit(600):
        _sh(f"bash ./stt/export_and_test.sh --preset {preset} --num-threads 8 {args}", threads=8)


@app.local_entrypoint()
def main(
    action: str = "probe",
    preset: str = "small",
    max_usd: float = 20.0,
    hours: float = 0.0,
    num_epochs: int = 0,
    exp_name: str = "",
    args: str = "",
):
    price = GPU_PRICE.get(GPU)
    if action == "prepare":
        print(f"[modal] prepare on 8 CPU cores (~${CPU_JOB_PRICE:.2f}/h)")
        prepare.remote(args) if args else prepare.remote()
    elif action == "probe":
        if price is None:
            raise SystemExit(f"unknown GPU {GPU}; choose from {sorted(GPU_PRICE)}")
        name = f"probe-{preset}-{GPU.lower()}"
        print(f"[modal] 10-minute speed probe on {GPU} (~${(price + GPU_SIDE_PRICE) / 6:.2f})")
        train.remote(preset, 1 / 6, MAX_DURATION[GPU][preset], 0, name, args, f"{name}-{int(time.time())}")
        print(eta.remote(name))
        print("[modal] plug the measured hours/epoch into: cost = hours/epoch x epochs x "
              f"${price + GPU_SIDE_PRICE:.2f}/h")
    elif action == "train":
        if price is None:
            raise SystemExit(f"unknown GPU {GPU}; choose from {sorted(GPU_PRICE)}")
        cap = max_usd / (price + GPU_SIDE_PRICE)
        h = min(hours, cap) if hours else cap
        print(f"[modal] training {preset} on {GPU}: at most {h:.2f} h, i.e. <= ${h * (price + GPU_SIDE_PRICE):.2f}")
        train.remote(preset, h, MAX_DURATION[GPU][preset], num_epochs, exp_name, args, f"train-{int(time.time())}")
    elif action == "export":
        export.remote(preset, args)
    else:
        raise SystemExit("action must be one of: prepare, probe, train, export")
