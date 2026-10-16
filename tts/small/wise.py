#!/usr/bin/env python3
"""Weight-space interpolation of a fine-tuned student with its init (WiSE-FT, Wortsman et al. 2022, arXiv 2109.01903):
w = alpha * w_init + (1 - alpha) * w_ft, written as a train.py-style checkpoint (its "ema" and "model" keys both hold
the result) that eval_student.py / student.load_student read.

    <venv-student>/python tts/small/wise.py data/small/runs/ft-a/ckpt_00020000.pt --alpha 0.3 \
        --out data/small/runs/wise/ft-a20k-a0.30.pt

The init is ema.pt's EMA weights (style embedding zero) unless --init names another student checkpoint. The
fine-tuned side is the checkpoint's EMA weights (--weights model for the raw ones).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0common as C  # noqa: E402

import torch  # noqa: E402

import student as S  # noqa: E402


def interpolate(init: dict, ft: dict, alpha: float) -> dict:
    if set(init) != set(ft):
        raise ValueError(f"state dicts differ: {sorted(set(init) ^ set(ft))[:5]}")
    return {k: (alpha * init[k].float() + (1 - alpha) * ft[k].float()).to(ft[k].dtype) if ft[k].is_floating_point()
            else ft[k] for k in ft}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--alpha", type=float, required=True, help="share of the init weights (0 = the fine-tuned model)")
    ap.add_argument("--init", default="ema.pt")
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root = lambda p: Path(p) if Path(p).is_absolute() else C.REPO / p  # noqa: E731
    ck = torch.load(str(root(args.ckpt)), map_location="cpu", weights_only=False)
    if args.init == "ema.pt":
        init = S.build("ema", "cpu")[0].state_dict()
    else:
        init = torch.load(str(root(args.init)), map_location="cpu", weights_only=False)["ema"]
    w = interpolate(init, ck[args.weights], args.alpha)
    out = root(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": w, "model": w, "cfg": ck["cfg"], "vocab": ck["vocab"], "styles": ck.get("styles"),
                "wise": {"ft": args.ckpt, "init": args.init, "alpha": args.alpha, "weights": args.weights}}, str(out))
    print(f"{out}: alpha {args.alpha}")


if __name__ == "__main__":
    main()
