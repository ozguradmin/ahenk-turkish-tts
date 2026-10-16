"""Train the Turkish Piper voice (tts/) on Modal (https://modal.com) GPUs.

The same scripts as on a laptop (prepare_antalia.py, train.sh, export_and_test.sh) run inside a
pinned image built by setup_env.sh (CUDA 12.8). Corpus, Piper cache, base checkpoint and
experiments live in the persistent Modal Volume `turkish-tts-data`, so every step can be stopped
and resumed. The local entrypoint only calls Modal functions (pure Python), so it works the same
from Windows PowerShell, macOS or Linux; everything else runs in Linux containers.

Setup (once):   pip install modal && modal setup

Steps, from the repo root (PowerShell or bash, same commands):

    modal run tts/modal_app.py --action prepare                    # CPU, ~10-15 min, ~$0.05
    modal run tts/modal_app.py --action probe                      # GPU ~12 min: real speed, memory, cost plan
    modal run --detach tts/modal_app.py --action train --max-usd 30 --max-epochs 1000
    modal run tts/modal_app.py --action status                     # progress (cheap, CPU)
    modal run tts/modal_app.py --action export                     # CPU: ONNX + .onnx.json + samples + RTF
    modal volume get turkish-tts-data exp/antalia-medium/export/last ./antalia-export

GPU type: TTS_GPU=A100-80GB|L40S|L4|H100 (default A100-80GB), read at import time:
    PowerShell: $env:TTS_GPU="L4"; modal run tts/modal_app.py --action probe
    bash:       TTS_GPU=L4 modal run tts/modal_app.py --action probe

Cost control: Modal bills per second. `train` turns --max-usd into a hard time limit
(train.sh --hours) from the prices below, so a run cannot spend more than that (plus container
start-up, ~1-2 min). A preempted run is retried and continues from last.ckpt; a ledger on the Volume
counts the time earlier attempts used. Also set a workspace spending limit in Modal.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

REPO = "/repo"          # local repo, mounted at container start (code edits need no rebuild)
WORK = "/work"          # writable copy with LF line endings (Windows checkouts may carry CRLF)
DATA = "/data"          # Modal Volume
VENV = "/opt/venv-piper"
PIPER_DIR = "/opt/piper1-gpl"
LOCAL_REPO = Path(__file__).resolve().parents[1]

GPU = os.environ.get("TTS_GPU", "A100-80GB")
# USD per hour, Modal list prices (2026); check modal.com/pricing before large runs.
GPU_PRICE = {"H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95, "L4": 0.80}
# CPU + memory of the GPU container (4 cores, 16 GiB) and of the CPU containers (4 cores, 8 GiB)
GPU_SIDE_PRICE = 4 * 0.047 + 16 * 0.008
CPU_JOB_PRICE = 4 * 0.047 + 8 * 0.008
# train.sh --preset per card (batch size: tts/README.md, section 3)
PRESET = {"H100": "h100", "A100-80GB": "a100", "L40S": "l40s", "L4": "l4"}
EXP = "antalia-medium"

app = modal.App("turkish-tts-piper")
volume = modal.Volume.from_name("turkish-tts-data", create_if_missing=True)
# optional: a Modal secret holding HF_TOKEN (higher Hugging Face download limits), created with
# `modal secret create huggingface HF_TOKEN=...`; pass its name in TTS_HF_SECRET
SECRETS = [modal.Secret.from_name(os.environ["TTS_HF_SECRET"])] if os.environ.get("TTS_HF_SECRET") else []

_IGNORE = [
    ".git",
    ".venv*",
    "data",
    "exports",
    "build",
    "dist",
    "**/__pycache__",
    "**/*.pyc",
    "**/*.ckpt",
    "**/*.onnx",
    "**/*.wav",
    "stt/icefall",
    "stt/.env.local",
    "tts/piper1-gpl",
    "tts/.env.local",
    "benchmarks/**/results",
    "benchmarks/**/logs",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ca-certificates", "build-essential", "ninja-build", "cmake", "libsndfile1")
    # only what the environment build needs, so code edits do not rebuild the venv
    .add_local_file(LOCAL_REPO / "pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
    .add_local_file(LOCAL_REPO / "README.md", f"{REPO}/README.md", copy=True)
    .add_local_file(LOCAL_REPO / "LICENSE", f"{REPO}/LICENSE", copy=True)
    .add_local_dir(LOCAL_REPO / "src", f"{REPO}/src", copy=True, ignore=["**/__pycache__"])
    .add_local_file(LOCAL_REPO / "tts" / "setup_env.sh", f"{REPO}/tts/setup_env.sh", copy=True)
    .add_local_file(LOCAL_REPO / "tts" / "requirements-tts.txt", f"{REPO}/tts/requirements-tts.txt", copy=True)
    .add_local_file(LOCAL_REPO / "tts" / "constraints-tts.txt", f"{REPO}/tts/constraints-tts.txt", copy=True)
    .run_commands(
        f"sed -i 's/\\r$//' {REPO}/tts/setup_env.sh {REPO}/tts/requirements-tts.txt {REPO}/tts/constraints-tts.txt"
        f" && cd {REPO} && bash tts/setup_env.sh --venv {VENV} --piper-dir {PIPER_DIR} --no-verify"
    )
    .env({"VENV": VENV, "PIPER_DIR": PIPER_DIR, "TTS_DATA": DATA, "HF_HOME": f"{DATA}/hf_home",
          "TORCH_HOME": f"{DATA}/torch_home", "PYTHONUNBUFFERED": "1"})
    # the rest of the code is mounted at start, so edits apply without a rebuild
    .add_local_dir(LOCAL_REPO, REPO, ignore=_IGNORE)
)


def container_cmd(cmd: str, threads: int, repo: str = REPO, work: str = WORK, venv: str = VENV,
                  piper_dir: str = PIPER_DIR, data: str = DATA) -> str:
    """The bash line every container step runs: a writable LF-only copy of the mounted repo, the
    pinned environment (path.sh), then `cmd`. tts/tools/modal_dryrun.py runs the same lines on a
    local CPU with local paths."""
    return (
        f"rm -rf {work} && cp -a {repo} {work} && find {work} -name '*.sh' -exec sed -i 's/\\r$//' {{}} + "
        # files mounted from a Windows checkout carry no executable bit
        f"&& find {work} -name '*.sh' -exec chmod +x {{}} + "
        f"&& cd {work} && export VENV={venv} PIPER_DIR={piper_dir} TTS_DATA={data} OMP_NUM_THREADS={threads} "
        f"&& source tts/path.sh && {cmd}"
    )


# --- the recipe commands of each step (also used by tts/tools/modal_dryrun.py) ---------------
def prepare_cmds(args: str = "") -> list[str]:
    return [
        f"python tts/prepare_antalia.py --workers 4 --piper-cache {args}",
        "./tts/train.sh --fetch-base-only",
        # val_mos weights (UTMOS via torch.hub); piper disables val_mos if this fails
        "python -c \"import torch; torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)\" || true",
    ]


def train_cmd(preset: str, hours: float, max_epochs: int, exp_dir: str, extra: str = "") -> str:
    return (f"./tts/train.sh --preset {preset} --hours {hours:.3f} --max-epochs {max_epochs} --exp-dir {exp_dir} "
            f"--no-guard --ckpt-every 10 --val-every 5 --snapshot-every 100 --num-workers 4 {extra}").strip()


def probe_cmds(preset: str, minutes: float, exp_dir: str, bench_out: str, batches=(16, 32), extra: str = "") -> list[str]:
    bs = " ".join(str(b) for b in batches)
    return [
        f"rm -rf {exp_dir}; for b in {bs}; do python tts/tools/step_bench.py --batch $b --seconds 20.24 --ids 763 "
        f"--steps 3 | tail -1 >> {bench_out} || true; done",
        f"./tts/train.sh --preset {preset} --hours {minutes / 60:.3f} --max-epochs 1000 --exp-dir {exp_dir} --no-guard "
        f"--ckpt-every 1000 --val-every 5 --snapshot-every 0 --num-workers 4 {extra}".strip(),
    ]


def export_cmd(exp_dir: str, ckpt: str = "", args: str = "", threads: int = 4) -> str:
    ck = f"--ckpt {exp_dir}/checkpoints/{ckpt}" if ckpt else ""
    return f"./tts/export_and_test.sh --exp-dir {exp_dir} {ck} --threads {threads} {args}".strip()


def _sh(cmd: str, threads: int, check: bool = True) -> int:
    """Run a recipe command in the container, streaming its output."""
    print(f"$ {cmd}", flush=True)
    rc = subprocess.run(["bash", "-c", container_cmd(cmd, threads)]).returncode
    if check and rc not in (0, 75):  # 75 = train.sh stopped cleanly at its --hours limit
        raise RuntimeError(f"command failed with exit code {rc}: {cmd}")
    return rc


class _PeriodicCommit:
    """Commit the volume every few minutes so checkpoints survive preemption."""

    def __init__(self, every_s: int = 600):
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


@app.function(image=image, cpu=4, memory=8192, volumes={DATA: volume}, timeout=3 * 3600, secrets=SECRETS)
def prepare(args: str = "") -> None:
    """Corpus -> 22.05 kHz WAV + metadata.csv + Piper cache; base checkpoint; UTMOS weights."""
    with _PeriodicCommit(300):
        for cmd in prepare_cmds(args):
            _sh(cmd, threads=4)


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=16384,
    volumes={DATA: volume},
    timeout=24 * 3600,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
    secrets=SECRETS,
)
def train(hours: float, max_epochs: int, exp_name: str, extra: str = "", run_id: str = "") -> None:
    # Budget ledger: GPU time already used by earlier attempts of this run (preemption retries),
    # so a retry continues the same budget instead of starting a fresh --hours window.
    ledger = Path(DATA) / "modal_ledger" / f"{run_id or 'default'}.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    used = json.loads(ledger.read_text())["used_h"] if ledger.exists() else 0.0
    left = hours - used - 0.05  # ~3 min for start-up, the final save and the volume commit
    if left < 0.05:
        print(f"[modal] budget for run {run_id} already used ({used:.2f} of {hours:.2f} h); nothing to do")
        return
    t0 = time.time()
    stop = threading.Event()

    def bookkeeping() -> None:
        while not stop.wait(60):
            ledger.write_text(json.dumps({"used_h": used + (time.time() - t0) / 3600, "limit_h": hours}))

    threading.Thread(target=bookkeeping, daemon=True).start()
    try:
        with _PeriodicCommit(600):
            # train.sh resumes from last.ckpt on its own (also after a preemption retry)
            _sh(train_cmd(PRESET[GPU], left, max_epochs, f"{DATA}/exp/{exp_name}", extra), threads=4)
    finally:
        stop.set()
        ledger.write_text(json.dumps({"used_h": used + (time.time() - t0) / 3600, "limit_h": hours}))
        volume.commit()


@app.function(image=image, gpu=GPU, cpu=4, memory=16384, volumes={DATA: volume}, timeout=3600, secrets=SECRETS)
def probe(minutes: float = 10.0) -> str:
    """Peak GPU memory at the preset batch (worst-case clips) + a short real training run."""
    name = f"probe-{GPU.lower()}"
    bench = "/tmp/step_bench.jsonl"
    bench_cmd, train_probe = probe_cmds(PRESET[GPU], minutes, f"{DATA}/exp/{name}", bench)
    _sh(bench_cmd, threads=4, check=False)
    with _PeriodicCommit(600):
        _sh(train_probe, threads=4, check=False)
    lines = [ln for ln in Path(bench).read_text().splitlines() if ln.startswith("{")] if Path(bench).exists() else []
    epochs = [json.loads(ln) for ln in (Path(DATA) / "exp" / name / "epochs.jsonl").read_text().splitlines()
              if ln.strip()] if (Path(DATA) / "exp" / name / "epochs.jsonl").exists() else []
    return json.dumps({"gpu": GPU, "step_bench": [json.loads(x) for x in lines], "epochs": epochs})


@app.function(image=image, cpu=1, memory=2048, volumes={DATA: volume}, timeout=300)
def status(exp_name: str = EXP) -> str:
    exp = Path(DATA) / "exp" / exp_name
    out = {"exp": str(exp), "exists": exp.exists()}
    if (exp / "RESUME.json").exists():
        out["resume"] = json.loads((exp / "RESUME.json").read_text())
    if (exp / "epochs.jsonl").exists():
        rows = [json.loads(ln) for ln in (exp / "epochs.jsonl").read_text().splitlines() if ln.strip()]
        out["epochs_logged"] = len(rows)
        out["last_epochs"] = rows[-3:]
    if (exp / "checkpoints").exists():
        out["checkpoints"] = sorted(p.name for p in (exp / "checkpoints").glob("*.ckpt"))
    if (exp / "export").exists():
        out["exports"] = sorted(str(p.relative_to(exp)) for p in (exp / "export").glob("*/*.onnx"))
    for ledger in sorted((Path(DATA) / "modal_ledger").glob("*.json")) if (Path(DATA) / "modal_ledger").exists() else []:
        out.setdefault("ledgers", {})[ledger.stem] = json.loads(ledger.read_text())
    return json.dumps(out, indent=1, ensure_ascii=False)


@app.function(image=image, cpu=4, memory=8192, volumes={DATA: volume}, timeout=3600)
def export(exp_name: str = EXP, ckpt: str = "", args: str = "") -> str:
    with _PeriodicCommit(600):
        _sh(export_cmd(f"{DATA}/exp/{exp_name}", ckpt, args), threads=4)
    summaries = sorted((Path(DATA) / "exp" / exp_name / "export").glob("*/samples/summary.json"),
                       key=lambda p: p.stat().st_mtime)
    return summaries[-1].read_text() if summaries else "no summary found"


def _fmt_probe(raw: str, price: float) -> str:
    r = json.loads(raw)
    lines = [f"[probe] GPU {r['gpu']}, ${price:.2f}/h incl. CPU/RAM"]
    for b in r["step_bench"]:
        lines.append(f"  worst-case batch {b['batch']} x 20.24 s: {b['s_per_step']} s/step, peak "
                     f"{b.get('peak_alloc_GiB')} GiB allocated / {b.get('peak_reserved_GiB')} GiB reserved")
    ep = [e for e in r["epochs"] if e.get("batches")]
    if ep:
        tail = ep[1:] or ep  # the first epoch includes start-up effects
        train_s = sum(e["seconds"] - e.get("validation_seconds", 0) for e in tail) / len(tail)
        val = [e["validation_seconds"] for e in tail if e.get("validation_seconds")]
        per_epoch = train_s + (sum(val) / len(val) / 5 if val else 0)  # validation every 5 epochs
        lines.append(f"  {len(ep)} epochs measured: {train_s:.1f} s training/epoch, "
                     f"{(sum(val) / len(val)) if val else 0:.1f} s per validation, ~{per_epoch:.1f} s/epoch overall")
        for n in (300, 500, 1000):
            h = n * per_epoch / 3600
            lines.append(f"  {n:4d} epochs: {h:5.1f} h  ~${h * price:6.2f}")
    else:
        lines.append("  no epoch finished; see the container log")
    return "\n".join(lines)


@app.local_entrypoint()
def main(
    action: str = "status",
    max_usd: float = 10.0,
    hours: float = 0.0,
    max_epochs: int = 1000,
    exp_name: str = EXP,
    ckpt: str = "",
    minutes: float = 10.0,
    args: str = "",
):
    price = GPU_PRICE.get(GPU)
    if action in ("probe", "train") and price is None:
        raise SystemExit(f"unknown GPU {GPU}; choose from {sorted(GPU_PRICE)}")
    if action == "prepare":
        print(f"[modal] prepare on 4 CPU cores (~${CPU_JOB_PRICE:.2f}/h)")
        prepare.remote(args)
    elif action == "probe":
        total = price + GPU_SIDE_PRICE
        print(f"[modal] ~{minutes + 2:.0f}-minute probe on {GPU} (~${total * (minutes + 3) / 60:.2f})")
        print(_fmt_probe(probe.remote(minutes), total))
    elif action == "train":
        total = price + GPU_SIDE_PRICE
        cap = max_usd / total
        h = min(hours, cap) if hours else cap
        print(f"[modal] training on {GPU} (preset {PRESET[GPU]}): at most {h:.2f} h, i.e. <= ${h * total:.2f}; "
              f"target {max_epochs} epochs; resumes from last.ckpt if {exp_name} exists")
        train.remote(h, max_epochs, exp_name, args, f"{exp_name}-{int(time.time())}")
    elif action == "status":
        print(status.remote(exp_name))
    elif action == "export":
        print(export.remote(exp_name, ckpt, args))
    else:
        raise SystemExit("action must be one of: prepare, probe, train, status, export")
