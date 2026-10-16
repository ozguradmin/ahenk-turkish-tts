"""VoxCPM2 AudioVAE (v2) encoder + decoder, standalone, for the cached openbmb/VoxCPM2 audiovae.pth.

Classes copied from OpenBMB/VoxCPM src/voxcpm/modules/audiovae/audio_vae_v2.py (Apache-2.0) via an
earlier local check (now tts/small/latent_check.py). Encoder: 16 kHz mono in, 64-dim mu latents at
25 Hz out (hop 640, causal: right zero-padding never changes earlier frames, so padded batches are exact).
Decoder: 48 kHz out. encode() returns the raw, unnormalised mu, as VoxCPM2's AudioVAE.encode does.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

HOP = 640
FPS = 25


def WNConv1d(*a, **k):
    return weight_norm(nn.Conv1d(*a, **k))


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, padding=0, output_padding=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding, self.__output_padding = padding, output_padding

    def forward(self, x):
        return super().forward(F.pad(x, (self.__padding * 2 - self.__output_padding, 0)))


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding=0, output_padding=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding, self.__output_padding = padding, output_padding

    def forward(self, x):
        return super().forward(x)[..., : -(self.__padding * 2 - self.__output_padding)]


def WNCausalConv1d(*a, **k):
    return weight_norm(CausalConv1d(*a, **k))


def WNCausalTransposeConv1d(*a, **k):
    return weight_norm(CausalTransposeConv1d(*a, **k))


def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class CausalResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1, kernel=7, groups=1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(Snake1d(dim), WNCausalConv1d(dim, dim, kernel_size=kernel, dilation=dilation,
                                                                padding=pad, groups=groups),
                                   Snake1d(dim), WNCausalConv1d(dim, dim, kernel_size=1))

    def forward(self, x):
        return x + self.block(x)


class CausalEncoderBlock(nn.Module):
    def __init__(self, output_dim=16, input_dim=None, stride=1, groups=1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            CausalResidualUnit(input_dim, dilation=1, groups=groups),
            CausalResidualUnit(input_dim, dilation=3, groups=groups),
            CausalResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNCausalConv1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride,
                           padding=math.ceil(stride / 2), output_padding=stride % 2))

    def forward(self, x):
        return self.block(x)


class CausalEncoder(nn.Module):
    def __init__(self, d_model=64, latent_dim=32, strides=(2, 4, 8, 8), depthwise=False):
        super().__init__()
        block = [WNCausalConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            block += [CausalEncoderBlock(output_dim=d_model, stride=stride, groups=groups)]
        self.fc_mu = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.fc_mu(self.block(x))


class CausalDecoderBlock(nn.Module):
    def __init__(self, input_dim=16, output_dim=8, stride=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNCausalTransposeConv1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride,
                                    padding=math.ceil(stride / 2), output_padding=stride % 2),
            CausalResidualUnit(output_dim, dilation=1, groups=groups),
            CausalResidualUnit(output_dim, dilation=3, groups=groups),
            CausalResidualUnit(output_dim, dilation=9, groups=groups))
        self.input_channels = input_dim

    def forward(self, x):
        return self.block(x)


class SRCond(nn.Module):
    def __init__(self, input_dim, buckets):
        super().__init__()
        self.scale_embed = nn.Embedding(buckets, input_dim)
        self.bias_embed = nn.Embedding(buckets, input_dim)
        self.out_layer = nn.Identity()

    def forward(self, x, c):
        return x * self.scale_embed(c).unsqueeze(-1) + self.bias_embed(c).unsqueeze(-1)


class CausalDecoder(nn.Module):
    def __init__(self, input_channel, channels, rates, depthwise, sr_bin_boundaries):
        super().__init__()
        layers = [WNCausalConv1d(input_channel, input_channel, kernel_size=7, padding=3, groups=input_channel),
                  WNCausalConv1d(input_channel, channels, kernel_size=1)] if depthwise else \
                 [WNCausalConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            inp, out = channels // 2 ** i, channels // 2 ** (i + 1)
            layers += [CausalDecoderBlock(inp, out, stride, groups=out if depthwise else 1)]
        layers += [Snake1d(out), WNCausalConv1d(out, 1, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.ModuleList(layers)
        self.register_buffer("sr_bin_boundaries", torch.tensor(sr_bin_boundaries, dtype=torch.int32))
        nb = len(sr_bin_boundaries) + 1
        self.sr_cond_model = nn.ModuleList(
            SRCond(l.input_channels, nb) if isinstance(l, CausalDecoderBlock) else None for l in layers)

    def forward(self, x, sr):
        c = torch.bucketize(sr, self.sr_bin_boundaries)
        for layer, cl in zip(self.model, self.sr_cond_model):
            if cl is not None:
                x = cl(x, c)
            x = layer(x)
        return x


class AudioVAE(nn.Module):
    def __init__(self, with_decoder: bool = True):
        super().__init__()
        self.encoder = CausalEncoder(128, 64, [2, 5, 8, 8], depthwise=True)
        self.decoder = CausalDecoder(64, 2048, [8, 6, 5, 2, 2, 2], True, [20000, 30000, 40000]) if with_decoder else None

    @torch.no_grad()
    def encode(self, x16k: torch.Tensor) -> torch.Tensor:
        """[B, 1, N] 16 kHz -> [B, 64, ceil(N / 640)] mu."""
        pad = math.ceil(x16k.shape[-1] / HOP) * HOP - x16k.shape[-1]
        return self.encoder(F.pad(x16k, (0, pad)))

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, 64, T] -> [B, 1, T * 1920] at 48 kHz."""
        return self.decoder(z, torch.tensor([48000], dtype=torch.int32, device=z.device))


def load_audiovae(path, device="cpu", with_decoder: bool = True) -> AudioVAE:
    vae = AudioVAE(with_decoder)
    sd = torch.load(str(path), map_location="cpu", weights_only=True)["state_dict"]
    if not with_decoder:
        sd = {k: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "sr_bin_boundaries" not in k]
    if missing or unexpected:
        raise RuntimeError(f"audiovae.pth does not match: missing {missing[:6]} unexpected {unexpected[:6]}")
    return vae.to(device).eval().requires_grad_(False)
