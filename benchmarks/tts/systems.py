"""CPU wrappers for small open Turkish TTS systems.

Every wrapper has the same interface:

    tts = load_system("ema-lightning", native=False)   # native=True: system's own normalizer on
    audio, sample_rate = tts.synthesize(text)          # float32 mono numpy array, Hz
    tts.model_text(text)                               # exact text the model consumes
    tts.reset()                                        # call before each utterance (not timed)

`native=False` bypasses a system's bundled text normalizer where that is possible, so an external
front-end (or none, for `raw`) decides how numbers and symbols are read. What every system still
does to its input in both modes (mandatory input encoding, not normalization):

  ema-lightning       Turkish lowercasing + alphabet filter (EMA Frontend.alphabet): characters
                      outside the model's letters/punctuation (digits included) become spaces.
  pocket-tts-turkish  alphabet fit (w->v, x->ks, q->k; characters the SentencePiece model lacks,
                      digits included, become spaces), sentence split, first word lowercased.
                      native=True runs the package's full TextFrontend (normalizer.py + spelling
                      of stray digits).
  piper-tr-dfki       none; espeak-ng phonemizes and ALWAYS reads digits itself (raw == native).
  piper-antalia       same as piper-tr-dfki; our voice from tts/, opt-in: only registered when
                      PIPER_ANTALIA_ONNX points to its exported .onnx (see PiperAntalia).
  mms-tts-tur         I->ı / İ->i, then the VitsTokenizer lowercases and drops out-of-vocabulary
                      characters (its vocabulary has the digits 0-4 and 6 but not 5, 7, 8, 9).
  qwen3-tts-tiny-tr   I->ı / İ->i, as its model card requires; no normalizer exists.

Determinism: every utterance uses the same fixed seed (default 0), set right before synthesis, so a
row's audio does not depend on run order (piper: the ONNX session is re-created after
onnxruntime.set_seed, because its RandomNormalLike ops are seeded at session creation).
"""

from __future__ import annotations

import common  # noqa: F401  (thread caps + HF_HOME before torch/onnxruntime are imported)

import json
import os
import re
import warnings
from hashlib import sha1, sha256
from pathlib import Path

import numpy as np

from common import SEED, THREADS


class NotApplicable(Exception):
    """The requested front-end makes no sense for this system (e.g. `native` without a normalizer)."""


def _cap_torch_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # can only be set once, before any parallel work
        pass


def _tr_capitals(text: str) -> str:
    """Turkish dotted/dotless capitals (the only casing a plain .lower() gets wrong)."""
    return text.replace("İ", "i").replace("I", "ı")


class TTSSystem:
    name = "?"
    model_id = "?"
    params = "?"
    license = "?"
    # "separable": bundled normalizer, can be switched off; "always_on": cannot be bypassed
    # (raw == native); "none": the system has no normalizer.
    native_mode = "none"
    native_note = ""
    sample_rate = 0
    default_threads: int | None = None  # None -> THREADS (2)

    def __init__(self, native: bool = False, seed: int = SEED, threads: int | None = None):
        if native and self.native_mode == "none":
            raise NotApplicable(f"{self.name} has no bundled text normalizer")
        if native and self.native_mode == "always_on":
            raise NotApplicable(f"{self.name}: raw == native ({self.native_note})")
        self.native = native
        self.seed = seed
        # None: the system's own default (pocket-tts: 1), never above the global cap THREADS
        self.threads = threads if threads is not None else min(self.default_threads or THREADS, THREADS)

    # --- interface -------------------------------------------------------------------------
    def model_text(self, text: str) -> str:
        raise NotImplementedError

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def reset(self) -> None:
        """Called before each utterance, outside the timer."""

    def extra(self, text: str) -> dict:
        """Optional diagnostics recorded with each utterance."""
        return {}

    def config(self) -> dict:
        return {"system": self.name, "model": self.model_id, "native": self.native, "seed": self.seed,
                "threads": self.threads}

    def config_hash(self) -> str:
        return sha1(json.dumps(self.config(), sort_keys=True).encode()).hexdigest()[:12]

    @staticmethod
    def _empty() -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


# ------------------------------------------------------------------------------------- EMA
class EMALightning(TTSSystem):
    name = "ema-lightning"
    model_id = "canberkkkkkk/ema-lightning"
    params = "8.6M (5.6M DiT + 3M decoder)"
    license = "Apache-2.0 (code and weights)"
    native_mode = "separable"
    native_note = "EMA always runs normalizer-tr (policy 'fallback') in EMA.say(); raw replaces Frontend.spoken with the identity"
    sample_rate = 48000

    def __init__(self, native=False, seed=SEED, threads=None):
        super().__init__(native, seed, threads)
        import torch  # noqa: F401
        from ema_lightning import EMA, __version__

        self.version = __version__
        self.tts = EMA(device="cpu")
        _cap_torch_threads(self.threads)
        if not native:
            # Frontend.__call__ = alphabet(spoken(text)); spoken() is the normalizer-tr call.
            self.tts._frontend.spoken = lambda text: text

    def model_text(self, text):
        return self.tts._frontend(text)

    def synthesize(self, text):
        if not self.model_text(text):
            return self._empty(), self.sample_rate
        speech = self.tts.say(text, seed=self.seed)
        return speech.audio.astype(np.float32), speech.sample_rate

    def config(self):
        return {**super().config(), "package": f"ema-lightning=={self.version}", "speed": 1.0}


# ------------------------------------------------------------------------------ pocket-tts
class PocketTTSTurkish(TTSSystem):
    name = "pocket-tts-turkish"
    model_id = "wite-tech/pocket-tts-turkish"
    params = "~110M (incl. codec)"
    license = "CC-BY-4.0 (weights); pocket-tts-turkish package Apache-2.0"
    native_mode = "separable"
    native_note = "package TextFrontend = normalizer.py + alphabet fit (+ stray digits spelled one by one)"
    sample_rate = 24000
    voice = "female_1"  # the package's default voice
    # pocket-tts pins torch to 1 CPU thread itself; on this shared 4-core box 2 threads were ~3x slower
    # (RTF 1.8 vs 0.62, load average ~6.5), so it keeps its own default.
    default_threads = 1

    def __init__(self, native=False, seed=SEED, threads=None):
        super().__init__(native, seed, threads)
        from pocket_tts_turkish import TurkishTTS, __version__

        self.version = __version__
        self.tts = TurkishTTS.from_pretrained(device="cpu")
        _cap_torch_threads(self.threads)  # pocket-tts sets torch to 1 thread on load; make it explicit
        if not native:
            self.tts.frontend = _pocket_bare_frontend(self.tts.frontend)

    def model_text(self, text):
        return " | ".join(self.tts.prepare(text))

    def synthesize(self, text):
        if not self.tts.prepare(text):
            return self._empty(), self.sample_rate
        audio = self.tts.generate(text, voice=self.voice, seed=self.seed)
        return audio.astype(np.float32), self.tts.sample_rate

    def config(self):
        return {**super().config(), "package": f"pocket-tts-turkish=={self.version}", "voice": self.voice,
                "emotion": None}


def _pocket_bare_frontend(base):
    """pocket-tts-turkish TextFrontend.prepare minus normalize() and minus stray-digit spelling."""
    from pocket_tts_turkish import frontend as F

    letters = {k: v for k, v in F._TRANSLITERATE.items() if k.isalpha()}  # w/x/q only

    class BareFrontend(F.TextFrontend):
        def __init__(self, base):
            self.__dict__.update(base.__dict__)

        def fit_alphabet(self, text):
            out = [letters.get(ch, ch if self.knows(ch) else " ") for ch in text]
            fitted = re.sub(r"\s+", " ", "".join(out)).strip()
            return re.sub(r"\s+([.,;:!?])", r"\1", fitted)

        def prepare(self, text, emotion=None):
            tag = F.resolve_emotion(emotion)
            sentences = []
            for sentence in F.split_sentences(self.fit_alphabet(text)):
                if not re.search(r"\w", sentence):
                    continue
                sentence = self.guard_first_word(sentence, tagged=tag is not None)
                sentences.append(f"[{tag}] {sentence}" if tag else sentence)
            return sentences

    return BareFrontend(base)


# ----------------------------------------------------------------------------------- piper
class PiperDFKI(TTSSystem):
    name = "piper-tr-dfki"
    model_id = "rhasspy/piper-voices tr/tr_TR/dfki/medium"
    params = "~15M (VITS medium)"
    license = "voice data CC-BY-NC-SA-4.0 (dfki-ot-data); piper-tts code GPL-3.0-or-later (bundles espeak-ng, GPL-3.0)"
    native_mode = "always_on"
    native_note = "espeak-ng's Turkish rules read digits, '%', thousands/decimal separators internally; cannot be bypassed"
    sample_rate = 22050
    _file = "tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx"

    def __init__(self, native=False, seed=SEED, threads=None):
        super().__init__(native, seed, threads)
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        self._ort = ort
        model = hf_hub_download("rhasspy/piper-voices", self._file)
        cfg = hf_hub_download("rhasspy/piper-voices", self._file + ".json")
        self.voice = PiperVoice.load(model, cfg)
        with open(model, "rb") as f:
            self._model_bytes = f.read()
        self._opts = ort.SessionOptions()
        self._opts.intra_op_num_threads = self.threads
        self._opts.inter_op_num_threads = 1
        self._opts.add_session_config_entry("session.intra_op.allow_spinning", "0")  # see OMP_WAIT_POLICY
        self.sample_rate = self.voice.config.sample_rate
        self.reset()

    def reset(self):
        self._ort.set_seed(self.seed)
        self.voice.session = self._ort.InferenceSession(
            self._model_bytes, sess_options=self._opts, providers=["CPUExecutionProvider"])

    def model_text(self, text):
        return text

    def extra(self, text):
        return {"phonemes": " | ".join("".join(p) for p in self.voice.phonemize(text))}

    def synthesize(self, text):
        from piper import SynthesisConfig

        chunks = [c.audio_float_array for c in self.voice.synthesize(text, syn_config=SynthesisConfig())]
        if not chunks:
            return self._empty(), self.sample_rate
        return np.concatenate(chunks).astype(np.float32), self.sample_rate

    def config(self):
        from importlib import metadata

        c = self.voice.config
        return {**super().config(), "package": f"piper-tts=={metadata.version('piper-tts')}",
                "noise_scale": c.noise_scale, "length_scale": c.length_scale, "noise_w": c.noise_w_scale}


class PiperAntalia(PiperDFKI):
    """Our Turkish Piper voice (tts/ recipe: LJSpeech medium fine-tuned on the Antalia corpus),
    loaded from a local export. Registered only when PIPER_ANTALIA_ONNX is set, so the default
    system list (speed.py, score.py metadata) and every existing result stay unchanged:

        PIPER_ANTALIA_ONNX=.../tr_TR-antalia-medium.onnx  # .onnx.json next to it, or PIPER_ANTALIA_CONFIG
        .venv-tts/bin/python benchmarks/tts/synthesize.py --set tr_zor --system piper-antalia --frontend trnorm

    Set the variable for judge.py/score.py runs too. The ONNX file's sha256 is part of config(), so a
    newly exported voice is synthesized again instead of reusing cached rows.
    """

    name = "piper-antalia"
    model_id = "local export of tts/ (tr_TR-antalia-medium)"
    params = "15.7M (VITS medium, fine-tuned from LJSpeech medium)"
    license = ("weights CC-BY-4.0 (Antalia voice corpus CC-BY-4.0, base LJSpeech public domain); "
               "piper-tts GPL-3.0-or-later (bundles espeak-ng, GPL-3.0)")
    native_note = "espeak-ng's Turkish rules read digits, '%', thousands/decimal separators internally; cannot be bypassed"

    def __init__(self, native=False, seed=SEED, threads=None):
        TTSSystem.__init__(self, native, seed, threads)
        import onnxruntime as ort
        from piper import PiperVoice

        model = os.environ.get("PIPER_ANTALIA_ONNX", "")
        cfg = os.environ.get("PIPER_ANTALIA_CONFIG") or model + ".json"
        if not (model and Path(model).is_file() and Path(cfg).is_file()):
            raise SystemExit(f"piper-antalia: PIPER_ANTALIA_ONNX={model!r} / config {cfg!r} not found")
        self._ort = ort
        self.voice = PiperVoice.load(model, cfg)
        with open(model, "rb") as f:
            self._model_bytes = f.read()
        self.model_file, self.model_sha256 = str(Path(model).resolve()), sha256(self._model_bytes).hexdigest()
        self._opts = ort.SessionOptions()
        self._opts.intra_op_num_threads = self.threads
        self._opts.inter_op_num_threads = 1
        self._opts.add_session_config_entry("session.intra_op.allow_spinning", "0")  # see OMP_WAIT_POLICY
        self.sample_rate = self.voice.config.sample_rate
        self.reset()

    def config(self):
        return {**super().config(), "model_file": Path(self.model_file).name, "model_sha256": self.model_sha256}


# ------------------------------------------------------------------------------------- MMS
class MMSTur(TTSSystem):
    name = "mms-tts-tur"
    model_id = "facebook/mms-tts-tur"
    params = "36M (VITS)"
    license = "CC-BY-NC-4.0"
    native_mode = "none"
    native_note = "no normalizer; VitsTokenizer lowercases and drops out-of-vocabulary characters"
    sample_rate = 16000

    def __init__(self, native=False, seed=SEED, threads=None):
        super().__init__(native, seed, threads)
        import torch
        from transformers import AutoTokenizer, VitsModel

        self._torch = torch
        self.model = VitsModel.from_pretrained(self.model_id).eval()
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.sample_rate = self.model.config.sampling_rate
        _cap_torch_threads(self.threads)

    def model_text(self, text):
        return self.tok.decode(self.tok(_tr_capitals(text))["input_ids"]).strip()

    def synthesize(self, text):
        if not self.model_text(text):
            return self._empty(), self.sample_rate
        inputs = self.tok(_tr_capitals(text), return_tensors="pt")
        self._torch.manual_seed(self.seed)
        with self._torch.inference_mode():
            wav = self.model(**inputs).waveform[0].cpu().numpy()
        return wav.astype(np.float32), self.sample_rate

    def config(self):
        from importlib import metadata

        return {**super().config(), "package": f"transformers=={metadata.version('transformers')}",
                "noise_scale": self.model.config.noise_scale,
                "noise_scale_duration": self.model.config.noise_scale_duration}


# ----------------------------------------------------------------------------- Qwen3 tiny
class Qwen3TinyTR(TTSSystem):
    name = "qwen3-tts-tiny-tr"
    model_id = "erkamk/qwen3-tts-tiny-tr"
    params = "199M talker + 171M speech tokenizer"
    license = "Apache-2.0 (weights, inherited from Qwen3-TTS); qwen-tts code Apache-2.0"
    native_mode = "none"
    native_note = "no normalizer; model card: write numbers as words and I/İ as ı/i"
    sample_rate = 24000
    ref_file = "samples/paragraph-01.wav"  # the model card's own sample, first 10 s, as voice reference
    ref_seconds = 10.0
    gen = dict(temperature=0.7, subtalker_temperature=0.6, repetition_penalty=1.0)  # model-card settings

    def __init__(self, native=False, seed=SEED, threads=None):
        super().__init__(native, seed, threads)
        import logging

        import soundfile as sf
        import torch
        from huggingface_hub import hf_hub_download

        logging.getLogger("transformers").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        from qwen_tts import Qwen3TTSModel

        self._torch = torch
        self.model = Qwen3TTSModel.from_pretrained(self.model_id, device_map="cpu", dtype=torch.float32)
        _cap_torch_threads(self.threads)
        ref, sr = sf.read(hf_hub_download(self.model_id, self.ref_file), dtype="float32")
        if ref.ndim > 1:
            ref = ref.mean(axis=1)
        ref = ref[: int(self.ref_seconds * sr)]
        self.voice = self.model.create_voice_clone_prompt(ref_audio=(ref, sr), x_vector_only_mode=True)

    def model_text(self, text):
        return _tr_capitals(text).strip()

    def synthesize(self, text):
        t = self.model_text(text)
        if not t:
            return self._empty(), self.sample_rate
        self._torch.manual_seed(self.seed)
        wavs, rate = self.model.generate_voice_clone(
            text=t, language="Auto", voice_clone_prompt=self.voice, non_streaming_mode=True,
            max_new_tokens=25 + 2 * len(t), **self.gen)
        return np.asarray(wavs[0], dtype=np.float32), int(rate)

    def config(self):
        from importlib import metadata

        return {**super().config(), "package": f"qwen-tts=={metadata.version('qwen-tts')}",
                "voice_ref": f"{self.ref_file}[:{self.ref_seconds}s] x_vector_only", **self.gen}


SYSTEMS: dict[str, type[TTSSystem]] = {
    cls.name: cls for cls in (EMALightning, PocketTTSTurkish, PiperDFKI, MMSTur, Qwen3TinyTR)
}
if os.environ.get("PIPER_ANTALIA_ONNX"):  # opt-in, see PiperAntalia
    SYSTEMS[PiperAntalia.name] = PiperAntalia


def load_system(name: str, native: bool = False, seed: int = SEED, threads: int | None = None) -> TTSSystem:
    if name not in SYSTEMS:
        raise SystemExit(f"unknown system {name!r}; choose from {sorted(SYSTEMS)}")
    return SYSTEMS[name](native=native, seed=seed, threads=threads)


def native_status(name: str) -> tuple[str, str]:
    cls = SYSTEMS[name]
    return cls.native_mode, cls.native_note
