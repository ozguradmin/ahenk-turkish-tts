"""Nano-vLLM-VoxCPM (github.com/a710128/nanovllm-voxcpm, MIT) for VoxCPM2: the pins, the Modal image recipe and the
server / LoRA helpers that nanovllm_app.py (the throughput probe), modal_app.py (the voice contest's GPU image) and
voice_contest.py (its "nanovllm" backend) share, so all three run the same build.

Importing this needs nothing beyond the standard library: the image recipe takes the `modal` module as an argument, and
nano_server / nano_tokens import Nano-vLLM, transformers and torch only when called (in the GPU container). The Modal
apps find this file next to them locally and under /repo/tts/voxcpm in their containers (their add_local_dir).

What the pinned source (NANO_COMMIT) does with a LoRA, checked against our checkpoints (lora_weights.safetensors +
lora_config.json of the official VoxCPM trainer, e.g. voxcpm2-alania-s2 step 600: r 64, alpha 64, base_lm /
residual_lm / feat_decoder.estimator q/k/v/o, no projections): models/voxcpm2/lora_loader.py reads lora_config.json's
"lora_config" (r, alpha or lora_alpha; scaling alpha / r, as VoxCPM's), takes the tensors named <module>.lora_A /
<module>.lora_B (the official trainer's keys; PEFT's .lora_A.weight is refused) and packs q/k/v into qkv_proj
(VoxCPM2Model.packed_modules_mapping); register_lora refuses a module the server built no LoRA slot for
(engine/model_runner.py validate_lora_payload) and a rank above max_lora_rank (engine/lora_manager.py). Our LoRAs fit:
nano_lora_kwargs builds the server's LoRAConfig from the checkpoint's own config (its rank, its targets), and
nano_lora_problem names what would not fit, for the contest to hand such a LoRA to the official API instead.
"""

from __future__ import annotations

from typing import Callable

VOXCPM_SRC = "/opt/voxcpm"
VOXCPM_COMMIT = "f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629"  # OpenBMB/VoxCPM, as modal_app.py: the official API
NANO_REPO = "https://github.com/a710128/nanovllm-voxcpm"
NANO_COMMIT = "c702631ea33db63932f0c2288cc09a543e68a3ef"  # main 2026-09-24: v2.0.4 + worker fail-fast fixes
TORCH, TORCHCODEC = "2.8.0", "0.7.0"  # torchcodec matches torch 1:1 (Nano-vLLM's CI: torch 2.8 <-> torchcodec 0.7)
FLASH_ATTN = "2.8.3"  # its GitHub release has prebuilt wheels: cu12 x torch 2.4-2.8 x C++11 ABI TRUE/FALSE x cp39-cp313
PY_TAG = "cp311"  # the image's Python
CU128 = "--extra-index-url https://download.pytorch.org/whl/cu128"
GPU_MEM, MAX_MODEL_LEN = 0.85, 4096  # the probe's gpu_memory_utilization (weights + KV cache) and max_model_len
TORCH_PINS = f"torch=={TORCH} torchaudio=={TORCH} torchcodec=={TORCHCODEC} 'transformers>=4.51,<5'"
# the layers Nano-vLLM can put a LoRA slot on (models/voxcpm2/model.py: Cpm4Attention / Cpm4MLP of base_lm,
# residual_lm and the DiT; the four projections of VoxCPM2Model)
LAYER_TARGETS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})
PROJ_TARGETS = frozenset({"enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj", "fusion_concat_proj"})


def flash_attn_wheel(torch_version: str = TORCH, cxx11_abi: bool | str = True, py_tag: str = PY_TAG,
                     version: str = FLASH_ATTN) -> str:
    """URL of flash-attn's prebuilt Linux wheel for this torch (major.minor), C++ ABI and Python (no source build);
    a str cxx11_abi goes in verbatim (the image build's shell substitution of torch's own flag)."""
    mm = ".".join(torch_version.split("+")[0].split(".")[:2])
    abi = cxx11_abi if isinstance(cxx11_abi, str) else "TRUE" if cxx11_abi else "FALSE"
    name = f"flash_attn-{version}+cu12torch{mm}cxx11abi{abi}-{py_tag}-{py_tag}-linux_x86_64.whl"
    return f"https://github.com/Dao-AILab/flash-attention/releases/download/v{version}/{name.replace('+', '%2B')}"


def base_image(modal):
    """The image both apps build on (each adds its env and the repo): Python 3.11, the official API (VoxCPM at
    VOXCPM_COMMIT) and the judges (faster-whisper, jiwer) with torch pinned in every resolver run, flash-attn's
    prebuilt wheel for torch's C++ ABI, Nano-vLLM-VoxCPM at NANO_COMMIT, cuBLAS / cuDNN on the loader path for
    ctranslate2. The probe ran on exactly these layers (probe-20261011-110126)."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "ffmpeg", "libsndfile1")
        .run_commands(
            f"git clone https://github.com/OpenBMB/VoxCPM {VOXCPM_SRC} && cd {VOXCPM_SRC} "
            f"&& git checkout {VOXCPM_COMMIT}",
            # one resolver run with torch pinned (the official API, the judges), as modal_app.py's image
            f"pip install {CU128} {TORCH_PINS} -e {VOXCPM_SRC} faster-whisper==1.2.1 jiwer",
            # flash-attn's prebuilt wheel for the C++ ABI this torch was built with (pip wheels >= 2.7: the C++11 one)
            "python -c \"import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')\" "
            "> /tmp/torch_abi "
            f"&& pip install --no-deps \"{flash_attn_wheel(cxx11_abi='$(cat /tmp/torch_abi)')}\"",
            f"pip install {CU128} {TORCH_PINS} 'nano-vllm-voxcpm @ git+{NANO_REPO}@{NANO_COMMIT}'",
            "python -c \"import importlib.metadata as m, torch; print('torch', torch.__version__, torch.version.cuda, "
            "'flash_attn', m.version('flash_attn'), 'nano-vllm-voxcpm', m.version('nano-vllm-voxcpm'), "
            "'transformers', m.version('transformers'))\"",
            # ctranslate2 (faster-whisper) loads cuBLAS/cuDNN by soname: point the loader at the pip CUDA libs
            "python -c \"import nvidia.cublas, nvidia.cudnn; print(nvidia.cublas.__path__[0] + '/lib'); "
            "print(nvidia.cudnn.__path__[0] + '/lib')\" > /etc/ld.so.conf.d/nvidia-pip.conf && ldconfig",
        )
    )


def nano_lora_kwargs(cfg: dict) -> dict:
    """Nano-vLLM LoRAConfig kwargs (extra keys are refused there) for one of our checkpoints' lora_config.json (the
    official trainer's: {"lora_config": {r, alpha, enable_*, target_*}}): its rank as the slot rank, one slot.
    ValueError when r / alpha are missing or not positive (Nano-vLLM scales by alpha / r, as VoxCPM does)."""
    lc = cfg.get("lora_config", cfg) if isinstance(cfg, dict) else None
    try:
        r, alpha = int(lc["r"]), float(lc.get("alpha", lc.get("lora_alpha")))
    except (KeyError, TypeError, ValueError):
        raise ValueError("lora_config.json: needs lora_config.r and lora_config.alpha") from None
    if r <= 0 or alpha <= 0:
        raise ValueError(f"lora_config.json: r {r} and alpha {alpha} must be > 0")
    out = {"enable_lm": bool(lc.get("enable_lm", True)), "enable_dit": bool(lc.get("enable_dit", True)),
           "enable_proj": bool(lc.get("enable_proj", False)), "max_loras": 1, "max_lora_rank": r}
    for k in ("target_modules_lm", "target_modules_dit", "target_proj_modules"):
        if isinstance(lc.get(k), list):
            out[k] = [str(x) for x in lc[k]]
    return out


def nano_lora_problem(cfg) -> str | None:
    """Why a Nano-vLLM server cannot take the LoRA checkpoint whose lora_config is cfg (the "lora_config" object of its
    lora_config.json; {"auto": True} or None: there is none), else None: its loader needs r and alpha > 0 from that
    file, and every enabled target must be a layer it builds LoRA slots on (LAYER_TARGETS; PROJ_TARGETS for the
    projections), or register_lora refuses the tensors."""
    if not isinstance(cfg, dict) or cfg.get("auto"):
        return "no readable lora_config.json (its loader reads r and alpha there)"
    try:
        nano_lora_kwargs(cfg)
    except ValueError as e:
        return str(e)
    for flag, key, known, default in (("enable_lm", "target_modules_lm", LAYER_TARGETS, True),
                                      ("enable_dit", "target_modules_dit", LAYER_TARGETS, True),
                                      ("enable_proj", "target_proj_modules", PROJ_TARGETS, False)):
        if cfg.get(flag, default) and (bad := sorted({str(x) for x in cfg.get(key) or []} - known)):
            return f"{key} {bad}: no LoRA slot for them in Nano-vLLM"
    return None


def nano_server(model: str, timesteps: int, max_num_seqs: int, lora_kwargs: dict | None = None,
                gpu_mem: float = GPU_MEM, max_model_len: int = MAX_MODEL_LEN):
    """A Nano-vLLM-VoxCPM server pool on GPU 0 for VoxCPM2 at `model`, as nanovllm_app.py's probe starts it: inference
    timesteps fixed per server, max_num_seqs requests at once, CUDA graphs on, gpu_mem of the card for weights and its
    KV cache, a LoRA slot built from lora_kwargs (nano_lora_kwargs; None: no LoRA). Call it inside a running event loop
    (the async pool: await wait_for_ready(), register_lora, encode_latents, generate, stop); the engine is a spawned
    process of its own."""
    from nanovllm_voxcpm import VoxCPM as NanoVoxCPM

    kw = {"model": model, "inference_timesteps": timesteps, "max_num_seqs": max_num_seqs,
          "max_model_len": max_model_len, "gpu_memory_utilization": gpu_mem, "devices": [0]}
    if lora_kwargs is not None:
        from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

        kw["lora_config"] = LoRAConfig(**lora_kwargs)
    return NanoVoxCPM.from_pretrained(**kw)


def nano_tokens(model: str) -> Callable[[str], int]:
    """Text tokens of a prompt as Nano-vLLM's engine counts them (and the official API: the cap of a request and its
    bad-case test are multiples of this)."""
    from nanovllm_voxcpm.models.voxcpm2.utils import mask_multichar_chinese_tokens
    from transformers import LlamaTokenizerFast

    tok = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(model))
    return lambda text: len(tok(text))
