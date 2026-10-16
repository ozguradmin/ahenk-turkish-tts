# Copyright 2021 Piotr Żelasko, 2022 Xiaomi Corp. (icefall, Apache-2.0) -- adapted
# for the Turkish data mix. Same interface as icefall's LibriSpeechAsrDataModule
# so that icefall's zipformer/train.py can use it unchanged (see train_tr.py).
"""Lhotse data module for the Turkish ASR mix (precomputed 80-dim fbank)."""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (  # noqa: F401
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import AudioSamples, OnTheFlyFeatures  # noqa: F401
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


def _default_manifest_dir() -> Path:
    data = os.environ.get("STT_DATA") or str(Path(__file__).resolve().parents[2] / "data" / "stt")
    return Path(data) / "fbank"


class TurkishAsrDataModule:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        g = parser.add_argument_group(title="Turkish ASR data options")
        g.add_argument("--manifest-dir", type=Path, default=_default_manifest_dir(),
                       help="Directory with the fbank cut manifests.")
        g.add_argument("--train-cuts", type=str, default="train_mix_cuts.jsonl.gz",
                       help="Training cuts (relative to --manifest-dir or absolute).")
        g.add_argument("--dev-cuts", type=str, default="dev_mix_cuts.jsonl.gz",
                       help="Validation cuts (relative to --manifest-dir or absolute).")
        # Kept only for compatibility with icefall's librispeech train.py
        g.add_argument("--full-libri", type=str2bool, default=True, help=argparse.SUPPRESS)
        g.add_argument("--mini-libri", type=str2bool, default=False, help=argparse.SUPPRESS)
        g.add_argument("--max-duration", type=float, default=300.0,
                       help="Seconds of audio per batch. Main knob for GPU memory.")
        g.add_argument("--bucketing-sampler", type=str2bool, default=True)
        g.add_argument("--num-buckets", type=int, default=30)
        g.add_argument("--concatenate-cuts", type=str2bool, default=False)
        g.add_argument("--duration-factor", type=float, default=1.0)
        g.add_argument("--gap", type=float, default=1.0)
        g.add_argument("--on-the-fly-feats", type=str2bool, default=False)
        g.add_argument("--shuffle", type=str2bool, default=True)
        g.add_argument("--drop-last", type=str2bool, default=True)
        g.add_argument("--return-cuts", type=str2bool, default=True)
        g.add_argument("--num-workers", type=int, default=2,
                       help="DataLoader worker processes for the training set.")
        g.add_argument("--enable-spec-aug", type=str2bool, default=True)
        g.add_argument("--spec-aug-time-warp-factor", type=int, default=80)
        g.add_argument("--enable-musan", type=str2bool, default=False,
                       help="Mix MUSAN noise (needs --musan-cuts). Off: alania's degraded "
                       "channel already provides room/mic/noise variety.")
        g.add_argument("--musan-cuts", type=Path, default=None)
        g.add_argument("--input-strategy", type=str, default="PrecomputedFeatures")

    def _path(self, name: str) -> Path:
        p = Path(name)
        return p if p.is_absolute() else self.args.manifest_dir / p

    def train_dataloaders(self, cuts_train: CutSet,
                          sampler_state_dict: Optional[Dict[str, Any]] = None) -> DataLoader:
        transforms = []
        if self.args.enable_musan:
            assert self.args.musan_cuts, "--enable-musan needs --musan-cuts"
            logging.info(f"Enable MUSAN from {self.args.musan_cuts}")
            transforms.append(CutMix(cuts=load_manifest(self.args.musan_cuts), p=0.5,
                                     snr=(10, 20), preserve_id=True))
        if self.args.concatenate_cuts:
            transforms = [CutConcatenate(duration_factor=self.args.duration_factor,
                                         gap=self.args.gap)] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            num_frame_masks = 10
            if inspect.signature(SpecAugment.__init__).parameters["num_frame_masks"].default == 1:
                num_frame_masks = 2
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        if self.args.on_the_fly_feats:
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )
        else:
            train = K2SpeechRecognitionDataset(
                input_strategy=eval(self.args.input_strategy)(),
                cut_transforms=transforms,
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:
            sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
            )
        else:
            sampler = SimpleCutSampler(cuts_train, max_duration=self.args.max_duration,
                                       shuffle=self.args.shuffle)
        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            sampler.load_state_dict(sampler_state_dict)

        seed = torch.randint(0, 100000, ()).item()
        return DataLoader(
            train,
            sampler=sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=_SeedWorkers(seed),
        )

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = K2SpeechRecognitionDataset(return_cuts=self.args.return_cuts)
        sampler = DynamicBucketingSampler(cuts_valid, max_duration=self.args.max_duration, shuffle=False)
        return DataLoader(validate, sampler=sampler, batch_size=None,
                          num_workers=min(2, self.args.num_workers), persistent_workers=False)

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )
        sampler = DynamicBucketingSampler(cuts, max_duration=self.args.max_duration, shuffle=False)
        return DataLoader(test, batch_size=None, sampler=sampler, num_workers=self.args.num_workers)

    # ---- names used by icefall's librispeech zipformer/train.py -------------
    @lru_cache()
    def train_cuts(self) -> CutSet:
        path = self._path(self.args.train_cuts)
        logging.info(f"About to get train cuts from {path}")
        return load_manifest_lazy(path)

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        path = self._path(self.args.dev_cuts)
        logging.info(f"About to get dev cuts from {path}")
        return load_manifest_lazy(path)

    def train_all_shuf_cuts(self) -> CutSet:
        return self.train_cuts()

    def train_clean_100_cuts(self) -> CutSet:
        return self.train_cuts()

    def dev_clean_cuts(self) -> CutSet:
        return self.dev_cuts()

    def dev_other_cuts(self) -> CutSet:
        return CutSet.from_cuts([])
