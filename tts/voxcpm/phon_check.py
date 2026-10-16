"""The c-check: is every Turkish "c" (/dʒ/) of a clip heard as /dʒ/?

With an English emotion instruction ("(Astonished, with rising intonation.)", "(Pleasantly surprised.)") VoxCPM2 and
our f022 LoRA turn word-initial Turkish "c" into /k/ or /d/: "Vay canına" -> "Vay kanına" (f022-1500: 0 of 18 right
over 6 spellings), "Cam kırılmış" -> "Kam" / "Dam"; without an instruction it is right (8 of 8). Whisper large-v3
often hides it (its language model writes the real word "canına" for a /k/ clip), so CER 0 is not enough.
facebook/wav2vec2-xlsr-53-espeak-cv-ft (Apache-2.0; wav2vec2 XLSR-53 fine-tuned to espeak IPA phones with CTC, no
language model) agreed 8 of 8 with our calibrated listener: a right c -> the phone "dʒ", a wrong one -> "k", "d" or "ɡ"
("v aɪ dʒ a n ə n a ..." vs "v aɪ k a n ə n a ..."; voxcpm-contest/probe-vay, probe-canina2).

    tokens = PhonModel().ipa_file("clip.wav")          # ["v", "aɪ", "dʒ", "a", "n", "ə", "n", "a", ...]
    ok, detail = c_check("Vay canına!", tokens)        # (True, "dʒ 1 of 1 c")

c_check: expected = the runs of the letter c in the Turkish spoken form (Turkish lowercasing; "ç" is another letter; a
geminate "cc", as in "hacca", is one long phone for CTC), got = the /dʒ/ phones (DJ: "dʒ", its long and palatalised
forms, X-SAMPA "dZ", "ʤ"); ok when got >= expected. It cannot tell which c went wrong, only that one did.

The model: IPA_MODEL at the pinned IPA_REV, loaded once per process (get_model; PhonModel.to moves it off the GPU and
back), fp32 as calibrated; its CTC output decoded here from vocab.json (ctc_tokens: repeats merged, blanks and special
tokens dropped: what Wav2Vec2PhonemeCTCTokenizer.batch_decode gives, without phonemizer / espeak). HF cache: HF_HOME
(on Modal /data/hf_home, the Volume, as Whisper's). Users: render_corpus.py (the gate on instructed rows with a c,
--audit-c), phon_probe.py (a folder of WAVs on a CPU container). Importing this module needs only the standard
library; torch, transformers, huggingface_hub, numpy, soundfile and librosa load in PhonModel.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

IPA_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"  # Apache-2.0
IPA_REV = "2c733782da5604684829819a5eb744c193fe9398"  # main since 2021-12-10 (read 2026-10-12)
IPA_FILES = ["config.json", "preprocessor_config.json", "vocab.json", "pytorch_model.bin"]  # its whole snapshot
IPA_SR = 16000
DJ = frozenset({"dʒ", "dʒː", "dʒʲ", "dZ", "ʤ"})  # /dʒ/ in the model's vocabulary (ʤ: the ligature, not in it)
SPECIAL = frozenset({"<pad>", "<s>", "</s>", "<unk>", "|"})  # CTC blank, specials, the word delimiter (not in vocab)
CONTROL = re.compile(r"^\s*\(([^()]*)\)\s*")  # VoxCPM2's "(instruction)text" (prepare_alania.CONTROL)


def tr_lower(s: str) -> str:
    """Turkish lowercasing of an NFC text: I -> ı, İ -> i, the rest str.lower."""
    return unicodedata.normalize("NFC", s or "").replace("I", "ı").replace("İ", "i").lower()


def spoken_part(text: str) -> tuple[str | None, str]:
    """(instruction or None, the spoken form) of a training text "(instruction)spoken"."""
    m = CONTROL.match(text or "")
    return (m.group(1).strip() or None, (text or "")[m.end():]) if m else (None, text or "")


def c_expected(spoken: str) -> int:
    """How many /dʒ/ a Turkish spoken form asks for: its runs of the letter c (tr_lower; "ç" does not count)."""
    return len(re.findall("c+", tr_lower(spoken)))


def dj_count(tokens: list[str]) -> int:
    """The /dʒ/ phones (DJ) among IPA tokens."""
    return sum(t in DJ for t in tokens)


def c_counts(spoken: str, tokens: list[str]) -> tuple[int, int]:
    """(expected, got): c_expected of the spoken form, dj_count of the tokens."""
    return c_expected(spoken), dj_count(tokens)


def c_check(spoken: str, tokens: list[str]) -> tuple[bool, str]:
    """(ok, detail): every c of the spoken form heard as /dʒ/, i.e. dj_count(tokens) >= c_expected(spoken)."""
    expected, got = c_counts(spoken, tokens)
    return got >= expected, f"dʒ {got} of {expected} c"


def ctc_tokens(ids: list[int], vocab: dict[int, str], blank: int = 0) -> list[str]:
    """Greedy CTC decoding of frame argmax ids: repeats merged, then the blank and SPECIAL tokens dropped."""
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            t = vocab.get(i, "<unk>")
            if t not in SPECIAL:
                out.append(t)
        prev = i
    return out


def ipa_text(tokens: list[str]) -> str:
    """The tokens as batch_decode writes them: space-separated phones."""
    return " ".join(tokens)


def load_vocab(path: str | Path) -> dict[int, str]:
    """vocab.json (token -> id) as id -> token."""
    return {int(i): t for t, i in json.loads(Path(path).read_text(encoding="utf-8")).items()}


class PhonModel:
    """IPA_MODEL at IPA_REV (the snapshot from the HF cache, downloaded there once) on device ("auto": cuda when
    there), fp32, eval mode."""

    def __init__(self, device: str = "auto", repo: str = IPA_MODEL, revision: str = IPA_REV):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        d = snapshot_download(repo, revision=revision, allow_patterns=IPA_FILES)
        self.repo, self.revision, self.device = repo, revision, device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(d)
        self.model = Wav2Vec2ForCTC.from_pretrained(d).eval().to(device)
        self.vocab = load_vocab(Path(d) / "vocab.json")
        self.blank = self.model.config.pad_token_id or 0

    def to(self, device: str) -> "PhonModel":
        """The model on another device (render_corpus: off the GPU while a Nano-vLLM server runs)."""
        if device != self.device:
            self.model.to(device)
            self.device = device
        return self

    def ipa(self, wav, sr: int) -> list[str]:
        """IPA tokens of a clip (samples, mono or channels last, any rate: resampled to 16 kHz as calibrated)."""
        import numpy as np
        import torch

        x = np.asarray(wav, dtype=np.float32)
        x = x.mean(axis=1) if x.ndim > 1 else x
        if sr != IPA_SR:
            import librosa

            x = librosa.resample(x, orig_sr=sr, target_sr=IPA_SR)
        inp = self.fe(x, sampling_rate=IPA_SR, return_tensors="pt").input_values.to(self.device)
        with torch.inference_mode():
            ids = self.model(inp).logits[0].argmax(-1).tolist()
        return ctc_tokens(ids, self.vocab, self.blank)

    def ipa_file(self, path: str | Path) -> list[str]:
        """ipa of a WAV file."""
        import soundfile as sf

        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return self.ipa(x, sr)


_MODELS: dict[str, PhonModel] = {}


def loaded() -> bool:
    """The process's PhonModel is loaded (get_model then only moves it: render_corpus reserves its load once)."""
    return "m" in _MODELS


def get_model(device: str = "auto") -> PhonModel:
    """The process's one PhonModel (loaded on first use), moved to device ("auto" on first use only: then where it
    is)."""
    if "m" not in _MODELS:
        _MODELS["m"] = PhonModel(device)
    return _MODELS["m"] if device == "auto" else _MODELS["m"].to(device)


def ipa(wav, sr: int) -> list[str]:
    """IPA tokens of a clip with the process's model (get_model)."""
    return get_model().ipa(wav, sr)
