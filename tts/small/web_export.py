#!/usr/bin/env python3
"""Export Ahenk Cep (student.pt + decoder.pt) to ONNX for the in-browser demo (site/). Nothing is uploaded.

    python tts/small/web_export.py --weights hf-publish/ahenk-cep-tr --out site/models

The sampling loop of tts/small/ahenk_cep.py is cut into four graphs; the page (site/js/engine.js) runs the loop:

    text.onnx     ids [1, L] int64, style [1] int64            -> h [1, L, 224], dur [1, L]   (text stage)
    cond.onnx     h, dur, cw, wstart [1, L], fw, fp [1, T]     -> cond [1, T, 224]            (aligner)
    step.onnx     x [1, T, 64], cond, t, dt, cfg [1] float     -> x + dt * v_cfg              (one Euler step, CFG
                                                                                                batched inside)
    decoder.onnx  z [1, 64, F]                                 -> audio [1, F * 1920]         (one decoder window)

Batch is always 1 and nothing is padded, so every mask of the PyTorch code is all ones (cond derives it from cw >= 0,
exactly as the engine pads cw with -1). Variants written next to the fp32 files:

    *.f16.onnx    every float weight stored as fp16 plus a Cast to fp32: half the download, fp32 arithmetic
    *.int8.onnx   MatMul weights quantized to int8 (dynamic quantization); only for text / cond / step

ahenk-cep.json (the manifest the page reads) lists vocab, styles, rates and every file with its size and sha256.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]

from ahenk_cep import STYLES, SAMPLE_RATE, load_student  # noqa: E402
from ema_lightning.decoder import load_decoder  # noqa: E402
from ema_lightning.model import rope  # noqa: E402

OPSET = 18
GRAPHS = ("text", "cond", "step", "decoder")


class TextGraph(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, style):
        mask = ids != 0
        h = self.m.features(ids, mask, style)
        return h, self.m.chardur(h, mask)


class CondGraph(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, h, dur, cw, wstart, fw, fp):
        mask = cw >= 0
        return self.m.condition(h, dur, mask, cw, wstart, fw, fp)


class StepGraph(nn.Module):
    """x + dt * (vu + cfg * (vc - vu)): StudentAcoustic.velocity with the Euler update of sound_stage."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, cond, t, dt, cfg):
        T = x.shape[1]
        fmask = torch.ones_like(x[..., 0], dtype=torch.bool)
        cos, sin = rope(torch.arange(T, device=x.device), self.m.dh)
        v = self.m.backbone(torch.cat([x, x]), torch.cat([cond, torch.zeros_like(cond)]), torch.cat([t, t]),
                            torch.cat([fmask, fmask]), cos, sin)
        vc, vu = v.chunk(2)
        return x + dt * (vu + cfg * (vc - vu))


class SubpixelUp(nn.Module):
    """ConvTranspose1d with kernel 2 * stride, as a kernel-2 Conv1d into stride phases, interleaved.

    Same arithmetic as the transposed convolution: output q = m * s + r (before cropping the padding) is
    x[m] @ W[:, :, r] + x[m - 1] @ W[:, :, r + s]. onnxruntime-web 1.30's WebGPU ConvTranspose gets some of these
    shapes wrong (128 -> 64, kernel 12, stride 6 among them); a plain Conv, Reshape and Transpose run right
    everywhere, so the decoder graph uses this form on every backend.
    """

    def __init__(self, up: nn.ConvTranspose1d):
        super().__init__()
        s, k, p = up.stride[0], up.kernel_size[0], up.padding[0]
        if k != 2 * s or up.groups != 1 or up.dilation[0] != 1 or up.output_padding[0] != 0:
            raise ValueError("SubpixelUp needs kernel == 2 * stride, no groups, dilation or output padding")
        w = up.weight.detach()  # [C_in, C_out, k]
        c_in, c_out = w.shape[0], w.shape[1]
        conv = nn.Conv1d(c_in, c_out * s, 2, padding=1)
        wc = torch.stack([w[:, :, s:], w[:, :, :s]], -1)  # [C_in, C_out, s, 2]: taps for x[m - 1], x[m]
        conv.weight.data.copy_(wc.permute(1, 2, 0, 3).reshape(c_out * s, c_in, 2))
        conv.bias.data.copy_(up.bias.detach().repeat_interleave(s))
        self.conv, self.s, self.p, self.c_out = conv, s, p, c_out
        self.stride, self.padding, self.kernel_size = up.stride, up.padding, up.kernel_size

    def forward(self, x):
        B, _, T = x.shape
        z = self.conv(x)  # [B, C_out * s, T + 1]
        z = z.view(B, self.c_out, self.s, T + 1).transpose(2, 3).reshape(B, self.c_out, (T + 1) * self.s)
        return z[:, :, self.p:(T + 1) * self.s - self.p]


class DecoderGraph(nn.Module):
    def __init__(self, d):
        super().__init__()
        d = copy.deepcopy(d)
        d.ups = nn.ModuleList(SubpixelUp(up) for up in d.ups)
        self.d = d

    def forward(self, z):
        return self.d(z, None)


def examples(L=37, T=120, F=116):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(2, 49, (1, L), generator=g)
    cw = torch.arange(L)[None] // 6
    wstart = (cw * 6).clamp(max=L - 1)
    fw = torch.repeat_interleave(torch.arange(int(cw.max()) + 1), T // (int(cw.max()) + 1) + 1)[:T][None]
    fp = torch.rand(1, T, generator=g)
    return {
        "text": (ids, torch.tensor([3])),
        "cond": (torch.randn(1, L, 224, generator=g), torch.rand(1, L, generator=g) * 4 + 0.5, cw, wstart, fw, fp),
        "step": (torch.randn(1, T, 64, generator=g), torch.randn(1, T, 224, generator=g), torch.tensor([0.25]),
                 torch.tensor([1 / 16]), torch.tensor([2.0])),
        "decoder": (torch.randn(1, 64, F, generator=g),),
    }


NAMES = {
    "text": (["ids", "style"], ["h", "dur"], {"ids": {1: "L"}, "h": {1: "L"}, "dur": {1: "L"}}),
    "cond": (["h", "dur", "cw", "wstart", "fw", "fp"], ["cond"],
             {"h": {1: "L"}, "dur": {1: "L"}, "cw": {1: "L"}, "wstart": {1: "L"}, "fw": {1: "T"}, "fp": {1: "T"},
              "cond": {1: "T"}}),
    "step": (["x", "cond", "t", "dt", "cfg"], ["x_next"], {"x": {1: "T"}, "cond": {1: "T"}, "x_next": {1: "T"}}),
    "decoder": (["z"], ["audio"], {"z": {2: "F"}, "audio": {1: "S"}}),
}


def modules(student, decoder):
    return {"text": TextGraph(student), "cond": CondGraph(student), "step": StepGraph(student),
            "decoder": DecoderGraph(decoder)}


def _expm1(g, x):  # the TorchScript exporter has no aten::expm1; exp(x) - 1 differs by < 1e-6 at these values
    return g.op("Sub", g.op("Exp", x), g.op("Constant", value_t=torch.tensor(1.0)))


def export(module, args, path: Path, name: str):
    inputs, outputs, axes = NAMES[name]
    torch.onnx.register_custom_op_symbolic("aten::expm1", _expm1, OPSET)
    torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs, dynamic_axes=axes,
                      opset_version=OPSET, dynamo=False, do_constant_folding=True)


def to_f16_storage(src: Path, dst: Path):
    """Every float32 initializer stored as fp16, read back through a Cast to fp32 (ORT folds it at load)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(src))
    g = model.graph
    keep, casts = [], []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims or [1])) >= 16:
            arr = numpy_helper.to_array(init).astype(np.float16)
            half = numpy_helper.from_array(arr, init.name + "__f16")
            keep.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT,
                                          name=init.name + "__cast"))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def to_int8(src: Path, dst: Path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8,
                     per_channel=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_graph(path: Path, module, args, name) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {n: a.numpy() for n, a in zip(NAMES[name][0], args, strict=True)}
    got = sess.run(None, feeds)
    with torch.inference_mode():
        ref = module(*args)
    ref = ref if isinstance(ref, tuple) else (ref,)
    return max(float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref, strict=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=Path("hf-publish/ahenk-cep-tr"))
    ap.add_argument("--out", type=Path, default=Path("site/models"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    student = load_student(a.weights / "student.pt")
    decoder = load_decoder(str(a.weights / "decoder.pt"), "cpu")
    mods, ex = modules(student, decoder), examples()
    refs = {**mods, "decoder": lambda z: decoder(z, None)}  # the decoder graph is checked against the original
    files = {}
    for name in GRAPHS:
        path = a.out / f"{name}.onnx"
        export(mods[name], ex[name], path, name)
        err = check_graph(path, refs[name], ex[name], name)
        print(f"[web_export] {path.name}: {path.stat().st_size / 1e6:.2f} MB, max |onnx - torch| = {err:.2e}")
        f16 = a.out / f"{name}.f16.onnx"
        to_f16_storage(path, f16)
        print(f"[web_export] {f16.name}: {f16.stat().st_size / 1e6:.2f} MB, max diff {check_graph(f16, refs[name], ex[name], name):.2e}")
        variants = {"fp32": path, "f16": f16}
        if name != "decoder":
            q = a.out / f"{name}.int8.onnx"
            to_int8(path, q)
            print(f"[web_export] {q.name}: {q.stat().st_size / 1e6:.2f} MB, max diff {check_graph(q, refs[name], ex[name], name):.2e}")
            variants["int8"] = q
        files[name] = {k: {"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for k, p in variants.items()}
    manifest = {
        "name": "Ahenk Cep",
        "version": 1,
        "sample_rate": SAMPLE_RATE,
        "hop": int(decoder.hop),
        "fps": 25,
        "latent_dim": int(student.latent_dim),
        "d": int(student.d),
        "vocab": list(student.vocab),
        "styles": list(STYLES),
        "defaults": {"steps": 16, "cfg": 2.0},
        "limits": {"max_word_frames": 250, "max_frames": 3000, "window": 100, "context": 8},
        "files": files,
    }
    (a.out / "ahenk-cep.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[web_export] wrote {a.out / 'ahenk-cep.json'}")


if __name__ == "__main__":
    main()
