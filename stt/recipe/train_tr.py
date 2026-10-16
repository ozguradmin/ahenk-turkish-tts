#!/usr/bin/env python3
"""Train a causal (streaming) Zipformer transducer on the Turkish data mix.

This is a thin wrapper around icefall's egs/librispeech/ASR/zipformer/train.py
(pinned commit, see setup_env.sh). The model, loss, optimizer and checkpoint
format are icefall's own; the wrapper only

  * swaps the LibriSpeech data module for TurkishAsrDataModule (tr_datamodule.py);
  * adds a wall-clock session limit: --session-hours N saves a checkpoint and
    exits with code 75 once N hours have passed (also on `touch <exp>/STOP`,
    SIGTERM or SIGUSR1), so a laptop can train in several sittings;
  * resumes automatically from the newest epoch-*.pt / checkpoint-*.pt in
    --exp-dir unless --start-epoch / --start-batch is given (--auto-resume 0
    disables this);
  * lets you override log/valid intervals and the number of CPU threads.

Every other option is icefall's (python train_tr.py --help).
Single GPU only (the laptop case); for multi-GPU use icefall's train.py directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ICEFALL_DIR = Path(os.environ.get("ICEFALL_DIR", HERE.parent / "icefall")).resolve()
ZIP_DIR = ICEFALL_DIR / "egs" / "librispeech" / "ASR" / "zipformer"
for _p in (str(ICEFALL_DIR), str(ZIP_DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXIT_PAUSED = 75  # "not finished, run again to resume"

# ----------------------------------------------------------- wrapper options
_w = argparse.ArgumentParser(add_help=False)
_w.add_argument("--session-hours", type=float, default=0.0)
_w.add_argument("--valid-interval", type=int, default=0)
_w.add_argument("--log-interval", type=int, default=0)
_w.add_argument("--num-threads", type=int, default=0)
_w.add_argument("--auto-resume", type=int, default=1)
WARGS, _rest = _w.parse_known_args(sys.argv[1:])
sys.argv = [sys.argv[0]] + _rest

_e = argparse.ArgumentParser(add_help=False)
_e.add_argument("--exp-dir", type=str, default="zipformer/exp")
_e.add_argument("--num-epochs", type=int, default=30)
_e.add_argument("--start-epoch", type=int, default=None)
_e.add_argument("--start-batch", type=int, default=None)
_e.add_argument("--world-size", type=int, default=1)
EARGS, _ = _e.parse_known_args(sys.argv[1:])
EXP_DIR = Path(EARGS.exp_dir)
if EARGS.world_size != 1:
    sys.exit("train_tr.py supports --world-size 1 only")


def _latest_checkpoint(exp_dir: Path):
    """Newest of epoch-E.pt / checkpoint-N.pt by mtime -> resume arguments."""
    best = None
    for f in exp_dir.glob("*.pt"):
        m = re.fullmatch(r"(epoch|checkpoint)-(\d+)\.pt", f.name)
        if m and (best is None or f.stat().st_mtime > best[0]):
            best = (f.stat().st_mtime, m.group(1), int(m.group(2)))
    return best


if WARGS.auto_resume and EARGS.start_epoch is None and EARGS.start_batch is None:
    latest = _latest_checkpoint(EXP_DIR)
    if latest is not None:
        _, kind, n = latest
        if kind == "epoch":
            if n >= EARGS.num_epochs:
                print(f"[train_tr] {EXP_DIR}/epoch-{n}.pt exists and --num-epochs is {EARGS.num_epochs}: "
                      "training is complete.")
                sys.exit(0)
            sys.argv += ["--start-epoch", str(n + 1)]
        else:
            sys.argv += ["--start-batch", str(n)]
        print(f"[train_tr] auto-resume: {' '.join(sys.argv[-2:])}")

import torch  # noqa: E402

import train as zt  # noqa: E402  (icefall zipformer/train.py)
from tr_datamodule import TurkishAsrDataModule  # noqa: E402

zt.LibriSpeechAsrDataModule = TurkishAsrDataModule
if WARGS.num_threads > 0:
    torch.set_num_threads(WARGS.num_threads)  # icefall sets 1 at import


class SessionEnd(Exception):
    pass


DEADLINE = time.time() + WARGS.session_hours * 3600 if WARGS.session_hours > 0 else None
_SIGNALLED = {"why": None}
CTX: dict = {}


def _on_signal(signum, _frame):
    _SIGNALLED["why"] = signal.Signals(signum).name


signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGUSR1, _on_signal)


def _stop_reason():
    if _SIGNALLED["why"]:
        return _SIGNALLED["why"]
    stop_file = EXP_DIR / "STOP"
    if stop_file.exists():
        stop_file.unlink(missing_ok=True)
        return "STOP file"
    if DEADLINE is not None and time.time() >= DEADLINE:
        return f"session limit of {WARGS.session_hours} h"
    return None


def _write_resume(info: dict):
    info["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (EXP_DIR / "RESUME.json").write_text(json.dumps(info, indent=1))


# -- params overrides --------------------------------------------------------
_orig_get_params = zt.get_params


def _get_params():
    p = _orig_get_params()
    if WARGS.valid_interval > 0:
        p.valid_interval = WARGS.valid_interval
    if WARGS.log_interval > 0:
        p.log_interval = WARGS.log_interval
    return p


zt.get_params = _get_params

# -- mid-epoch resume fix -----------------------------------------------------
# icefall runs scan_pessimistic_batches_for_oom() after restoring the sampler
# state. lhotse's find_pessimistic_batches() iterates the restored sampler to
# its end, which clears the "just restored" flag, so the training loop then
# starts the epoch from batch 0 again (verified with lhotse 1.33 + icefall
# 3f848bb). With multi-hour laptop sittings shorter than an epoch, the epoch
# would never finish. The scan already ran when the epoch started, so skip it
# when resuming from a mid-epoch checkpoint.
_orig_scan = zt.scan_pessimistic_batches_for_oom


def _scan_pessimistic_batches_for_oom(**kw):
    if kw["params"].start_batch > 0:
        logging.info("[train_tr] mid-epoch resume: skipping the OOM scan (it would rewind the sampler)")
        return
    return _orig_scan(**kw)


zt.scan_pessimistic_batches_for_oom = _scan_pessimistic_batches_for_oom

# -- remember the training objects so we can checkpoint from inside the loop --
_orig_train_one_epoch = zt.train_one_epoch


def _train_one_epoch(**kw):
    CTX.clear()
    CTX.update(kw)
    return _orig_train_one_epoch(**kw)


zt.train_one_epoch = _train_one_epoch

# -- every 10 batches icefall calls set_batch_count(): our stop hook ---------
_orig_set_batch_count = zt.set_batch_count


def _set_batch_count(model, batch_count):
    _orig_set_batch_count(model, batch_count)
    if not CTX:
        return
    why = _stop_reason()
    if why is None:
        return
    params = CTX["params"]
    n = params.batch_idx_train
    if n > 0:
        logging.info(f"[train_tr] {why}: saving checkpoint-{n}.pt and stopping")
        zt.save_checkpoint_with_global_batch_idx(
            out_dir=params.exp_dir,
            global_batch_idx=n,
            model=CTX["model"],
            model_avg=CTX["model_avg"],
            params=params,
            optimizer=CTX["optimizer"],
            scheduler=CTX["scheduler"],
            sampler=CTX["train_dl"].sampler,
            scaler=CTX["scaler"],
            rank=CTX["rank"],
        )
        zt.remove_checkpoints(out_dir=params.exp_dir, topk=params.keep_last_k, rank=CTX["rank"])
    _write_resume({"reason": why, "epoch": params.cur_epoch, "batch_idx_train": n,
                   "resume_args": f"--start-batch {n}" if n > 0 else "(none)"})
    raise SessionEnd(why)


zt.set_batch_count = _set_batch_count

# -- after each epoch checkpoint: stop if the session is over ----------------
_orig_save_checkpoint = zt.save_checkpoint


def _save_checkpoint(**kw):
    _orig_save_checkpoint(**kw)
    params = kw["params"]
    why = _stop_reason()
    if why is not None and params.cur_epoch < params.num_epochs:
        _write_resume({"reason": why, "epoch": params.cur_epoch, "batch_idx_train": params.batch_idx_train,
                       "resume_args": f"--start-epoch {params.cur_epoch + 1}"})
        raise SessionEnd(why)


zt.save_checkpoint = _save_checkpoint


def main():
    t0 = time.time()
    try:
        zt.main()
    except SessionEnd as e:
        logging.info(f"[train_tr] session ended ({e}) after {(time.time() - t0) / 3600:.2f} h; "
                     f"run the same command again to resume")
        sys.exit(EXIT_PAUSED)
    _write_resume({"reason": "finished", "resume_args": "(training complete)"})
    logging.info(f"[train_tr] training finished in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
