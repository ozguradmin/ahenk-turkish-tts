#!/usr/bin/env python3
"""Piper training entry point for the Turkish Antalia voice (called by tts/train.sh).

This is `python -m piper.train fit` from the pinned piper1-gpl commit (same LightningCLI, same
argument links, same VitsModel / VitsDataModule), plus what a resumable, budget-capped cloud run
needs. Nothing in the model, loss or data pipeline is changed.

1. Fixed validation split: `--data.heldout_list FILE` (wav names, one per line; written by
   prepare_antalia.py). Upstream draws `validation_split` (10 %) + `num_test_examples` clips at
   random on every start, so a resumed session would validate on clips an earlier session trained
   on, and 10 % of a 5-hour corpus is a lot. Here exactly the held-out clips are validated
   (val_mel, val_loss) and never trained on; the first `--data.num_test_examples` of them are
   synthesized at each validation for TensorBoard audio and the UTMOS score (val_mos).
2. Resume checkpoint `<default_root_dir>/checkpoints/last.ckpt`: full training state (weights,
   optimizers, LR schedulers, loop state) written atomically (temp file + rename, so a preemption
   mid-write never corrupts it) every `--ckpt_every` epochs, at the end, at a time-limit stop and on
   an interrupt/exception. Lightning 2.6's own `save_last` only writes when a top-k file is saved in
   the same step and is not atomic.
3. `--time_limit_hours H`: stops at an epoch boundary before H hours (including the time an epoch
   takes), saves, and writes `RESUME.json` with reason "time_limit" (train.sh exits with 75).
   A file `<default_root_dir>/STOP` (train.sh --stop) does the same after the current batch.
4. Selection checkpoints (weights only, ~0.28 GB each, in the same directory): best 3 by val_mel,
   best 3 by val_mos, and a snapshot every `--snapshot_every` epochs for listening tests.
5. `epochs.jsonl`: seconds and optimizer steps per epoch (speed, ETA and cost estimates).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from lightning.pytorch.callbacks import Checkpoint, ModelCheckpoint
from torch.utils.data import Subset

from piper.train.__main__ import VitsLightningCLI
from piper.train.vits.dataset import VitsDataModule
from piper.train.vits.lightning import VitsModel
from piper.train.vits.utils import get_cache_id

_LOGGER = logging.getLogger("piper_fit")


class FixedSplitDataModule(VitsDataModule):
    """VitsDataModule whose validation/test clips come from a list instead of a random split."""

    def __init__(self, *args, heldout_list: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heldout_list = heldout_list

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if not self.heldout_list:
            _LOGGER.warning("no --data.heldout_list: using upstream's random split (validation_split=%s)",
                            self.validation_split)
            return
        if self.is_multispeaker:
            raise ValueError("--data.heldout_list supports single-speaker data only")
        names = {line.strip() for line in open(self.heldout_list, encoding="utf-8") if line.strip()}
        held_ids = set()
        with open(self.csv_path, encoding="utf-8") as f:  # same row numbering as VitsDataModule
            for row_number, row in enumerate(csv.reader(f, delimiter="|"), start=1):
                if row and row[0] in names:
                    held_ids.add(get_cache_id(row_number, row[-1]))
        full = self.train_dataset.dataset  # the upstream Subsets all wrap one VitsDataset in CSV order
        suffix = ".phonemes.pt"
        train_idx, val_idx = [], []
        for i, utt in enumerate(full.utts):
            cache_id = utt.phoneme_ids_path.name[: -len(suffix)]
            (val_idx if cache_id in held_ids else train_idx).append(i)
        if len(val_idx) != len(names):
            raise ValueError(f"{self.heldout_list}: {len(names)} names, {len(val_idx)} found in {self.csv_path}")
        self.train_dataset = Subset(full, train_idx)
        self.val_dataset = Subset(full, val_idx)
        self.test_dataset = Subset(full, val_idx[: min(self.num_test_examples, len(val_idx))])
        _LOGGER.info("fixed split: %d train, %d validation (%d synthesized per validation) from %s",
                     len(train_idx), len(val_idx), len(self.test_dataset), self.heldout_list)


class RootCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that defaults to <default_root_dir>/checkpoints (shared by all sessions)
    instead of the per-session lightning_logs/version_N/checkpoints.

    optional=True: if the monitored metric is missing, skip quietly. Lightning 2.6 raises
    MisconfigurationException when a monitored key is absent after validation has run, which is
    what happens to val_mos whenever the UTMOS predictor cannot be downloaded (piper then disables
    it); upstream's default val_mos checkpoint would stop the run there."""

    def __init__(self, *args, optional: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.optional = optional
        self._warned_missing = False

    def setup(self, trainer, pl_module, stage) -> None:
        if self.dirpath is None:
            self.dirpath = os.path.realpath(os.path.join(trainer.default_root_dir, "checkpoints"))
        super().setup(trainer, pl_module, stage)

    def _save_topk_checkpoint(self, trainer, monitor_candidates) -> None:
        if self.optional and self.monitor is not None and self.monitor not in monitor_candidates:
            if not self._warned_missing:
                _LOGGER.warning("%s is not logged (e.g. UTMOS unavailable); no %s checkpoints", self.monitor, self.monitor)
                self._warned_missing = True
            return
        super()._save_topk_checkpoint(trainer, monitor_candidates)


class ResumeCheckpoint(Checkpoint):
    """Atomic last.ckpt every N epochs + time limit + per-epoch timing log (see module docstring)."""

    def __init__(self, every_n_epochs: int = 5, time_limit_hours: float = 0.0) -> None:
        self.every = max(1, int(every_n_epochs))
        self.limit_s = float(time_limit_hours) * 3600.0
        self._t0 = time.monotonic()
        self._epoch_t0 = self._t0
        self._step0 = 0
        self._last_saved_step = -1
        self._durations: list[float] = []
        self._val_s = 0.0
        self._val_t0 = self._t0
        self.stop_reason = ""

    # Lightning stores callback state under this key; keep it independent of the arguments.
    @property
    def state_key(self) -> str:
        return "ResumeCheckpoint"

    def setup(self, trainer, pl_module, stage) -> None:
        self.root = Path(trainer.default_root_dir)
        self.dir = self.root / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "last.ckpt"
        self._t0 = time.monotonic()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_t0 = time.monotonic()
        self._step0 = trainer.global_step
        self._val_s = 0.0

    def on_validation_start(self, trainer, pl_module) -> None:
        self._val_t0 = time.monotonic()

    def on_validation_end(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._val_s += time.monotonic() - self._val_t0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        stop = self.root / "STOP"  # `train.sh --stop`: save and exit after the current batch
        if not self.stop_reason and stop.exists():
            stop.unlink(missing_ok=True)
            self.stop_reason = "stop_file"
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        now = time.monotonic()
        seconds = now - self._epoch_t0
        self._durations.append(seconds)
        done = trainer.current_epoch + 1  # epochs completed, counting earlier sessions
        steps = (trainer.global_step - self._step0) // 2  # manual optimization: g + d step per batch
        train_s = seconds - getattr(self, "_val_s", 0.0)
        self._log_epoch({"epoch": done, "seconds": round(seconds, 2), "validation_seconds": round(seconds - train_s, 2),
                         "batches": steps, "s_per_batch": round(train_s / max(1, steps), 3),
                         "global_step": trainer.global_step, "session_hours": round((now - self._t0) / 3600, 4),
                         "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if self.limit_s > 0:
            recent = self._durations[-5:]
            expected_next = max(recent) * 1.1  # epochs with validation take longer
            if (now - self._t0) + expected_next > self.limit_s:
                self.stop_reason = "time_limit"
                trainer.should_stop = True
        max_epochs = trainer.max_epochs if (trainer.max_epochs or 0) > 0 else None
        if done % self.every == 0 or trainer.should_stop or (max_epochs and done >= max_epochs):
            self._save(trainer, self.stop_reason or ("max_epochs" if max_epochs and done >= max_epochs else "periodic"))

    def on_train_end(self, trainer, pl_module) -> None:
        if self._last_saved_step != trainer.global_step:
            self._save(trainer, self.stop_reason or "train_end")

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        # Ctrl-C / SIGTERM (e.g. a cloud preemption): keep the progress of the current epoch
        if trainer.global_step > 0 and self._last_saved_step != trainer.global_step:
            try:
                self._save(trainer, f"exception:{type(exception).__name__}")
            except Exception as e:  # noqa: BLE001
                _LOGGER.error("could not save %s after %s: %s", self.path, type(exception).__name__, e)

    def _save(self, trainer, reason: str) -> None:
        tmp = self.path.with_name(f".last.ckpt.tmp{os.getpid()}")
        t = time.monotonic()
        trainer.save_checkpoint(tmp)
        os.replace(tmp, self.path)
        self._last_saved_step = trainer.global_step
        status = {"reason": reason, "epochs_done": trainer.current_epoch + (0 if reason.startswith("exception") else 1),
                  "global_step": trainer.global_step, "max_epochs": trainer.max_epochs,
                  "checkpoint": str(self.path), "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "save_seconds": round(time.monotonic() - t, 2)}
        tmp_status = self.root / ".RESUME.json.tmp"
        tmp_status.write_text(json.dumps(status, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp_status, self.root / "RESUME.json")
        _LOGGER.info("saved %s (%s, epoch %s, step %s, %.1f s)", self.path, reason, status["epochs_done"],
                     trainer.global_step, status["save_seconds"])

    def _log_epoch(self, row: dict) -> None:
        with open(self.root / "epochs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


class AntaliaCLI(VitsLightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument("--ckpt_every", type=int, default=5, help="epochs between resume checkpoints")
        parser.add_argument("--snapshot_every", type=int, default=100,
                            help="epochs between weights-only snapshots (0 = none)")
        parser.add_argument("--time_limit_hours", type=float, default=0.0,
                            help="stop at an epoch boundary before this many hours (0 = no limit)")

    def before_instantiate_classes(self) -> None:
        cfg = self.config[self.subcommand] if self.subcommand else self.config
        val_every = int(cfg.trainer.check_val_every_n_epoch or 1)
        callbacks = [
            # selection checkpoints, weights only; evaluated whenever validation runs
            RootCheckpoint(monitor="val_mel", mode="min", save_top_k=3, save_weights_only=True,
                           filename="best-epoch={epoch}-val_mel={val_mel:.4f}", auto_insert_metric_name=False),
            RootCheckpoint(monitor="val_mos", mode="max", save_top_k=3, save_weights_only=True, optional=True,
                           filename="best-epoch={epoch}-val_mos={val_mos:.4f}", auto_insert_metric_name=False),
        ]
        if cfg.snapshot_every > 0:
            if cfg.snapshot_every % val_every:
                raise ValueError("--snapshot_every must be a multiple of --trainer.check_val_every_n_epoch")
            callbacks.append(RootCheckpoint(every_n_epochs=cfg.snapshot_every, save_top_k=-1, save_weights_only=True,
                                            filename="snapshot-epoch={epoch}", auto_insert_metric_name=False))
        # Last: Lightning runs Checkpoint callbacks in list order (after all other callbacks), and
        # last.ckpt must contain the top-k bookkeeping the callbacks above update in the same epoch;
        # otherwise a resumed run forgets the newest best-* file and never deletes it.
        callbacks.append(ResumeCheckpoint(every_n_epochs=cfg.ckpt_every, time_limit_hours=cfg.time_limit_hours))
        self.trainer_defaults["callbacks"] = callbacks


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True  # as upstream piper.train.__main__
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    AntaliaCLI(VitsModel, FixedSplitDataModule, trainer_defaults={"max_epochs": -1},
               save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
