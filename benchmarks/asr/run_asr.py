#!/usr/bin/env python3
"""Decode an ASR eval set with one system and save RAW hypotheses (no normalisation).

    python benchmarks/asr/run_asr.py --set fleurs_tr_test --system whisper-small [--limit N]

Writes benchmarks/asr/results/<set>/<system>.jsonl, one line per utterance:
    {"id", "ref", "hyp", "duration", "decode_seconds", "cpu_seconds"}
plus <system>.meta.json (decoding config, versions, CPU). Resumable: ids already in
the output file are skipped, so an interrupted run can simply be restarted, and
`--limit 5` followed by a full run reuses the first five results.

Scoring is a separate step (score.py) so normalisation can change without re-decoding.
decode_seconds is wall-clock time of the decode call only (audio already in memory),
measured with --threads CPU threads (default 2).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "data" / "asr_eval"
MODELS_DIR = REPO_ROOT / "data" / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SR = 16_000

# HF repos are pinned to the commit used for the published numbers.
WHISPER_DECODE = dict(language="tr", beam_size=5, condition_on_previous_text=False, vad_filter=False)
SYSTEMS: dict[str, dict] = {
    "whisper-tiny": dict(kind="faster-whisper", repo="Systran/faster-whisper-tiny",
                         revision="d90ca5fe260221311c53c58e660288d3deb8d356"),
    "whisper-base": dict(kind="faster-whisper", repo="Systran/faster-whisper-base",
                         revision="ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"),
    "whisper-small": dict(kind="faster-whisper", repo="Systran/faster-whisper-small",
                          revision="536b0662742c02347bc0e980a01041f333bce120"),
    "whisper-medium": dict(kind="faster-whisper", repo="Systran/faster-whisper-medium",
                           revision="08e178d48790749d25932bbc082711ddcfdfbc4f"),
    # faster-whisper's "large-v3-turbo" alias (mobiuslabsgmbh/...) now redirects here.
    "whisper-large-v3-turbo": dict(kind="faster-whisper",
                                   repo="dropbox-dash/faster-whisper-large-v3-turbo",
                                   revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"),
    # BuzzASR/turkish (whisper-large-v3 fine-tune, MIT) converted locally to CTranslate2 int8
    # (see README.md, "BuzzASR conversion"). Its generation_config has return_timestamps=False
    # (fine-tuned on <|notimestamps|> targets), so it is decoded without timestamps; the other
    # settings match the vanilla Whisper runs.
    "buzzasr-turkish": dict(kind="faster-whisper", path=str(MODELS_DIR / "buzzasr-turkish-ct2"),
                            source="BuzzASR/turkish@928b47596e31776dab9daf1e606ecf76234b9ecd",
                            decode=dict(without_timestamps=True)),
    # Streaming zipformer transducer (icefall export, chunk 32 / left 128). Settings follow
    # the repo's transcribe.py: int8, modified beam search, 12 active paths, 1 s tail pad.
    "duxx-zipformer": dict(
        kind="sherpa-online-transducer", repo="duxx/turkish-stt-zipformer",
        revision="cebec199e0a9a1bdcd195b9124922b4a33e98bc8",
        tokens="tokens.txt",
        encoder="encoder-epoch-1-avg-1-chunk-32-left-128.int8.onnx",
        decoder="decoder-epoch-1-avg-1-chunk-32-left-128.int8.onnx",
        joiner="joiner-epoch-1-avg-1-chunk-32-left-128.int8.onnx",
        decoding_method="modified_beam_search", max_active_paths=12, tail_padding_s=1.0),
    # Streaming transducer; settings follow the repo's eval/run_eval.py: int8 encoder+joiner,
    # fp32 decoder, modified beam search (sherpa default 4 paths), 0.5 s tail pad.
    # Licence (mihu Community License) forbids training on its outputs -> eval only.
    "mihuai-turkish-stt": dict(
        kind="sherpa-online-transducer", repo="mihuai/turkish-stt",
        revision="4f868b7a4a7cfaccb97fdaf12cdc8fdfabc8caad",
        tokens="tokens.txt", encoder="encoder.int8.onnx", decoder="decoder.onnx",
        joiner="joiner.int8.onnx",
        decoding_method="modified_beam_search", max_active_paths=4, tail_padding_s=0.5),
    "vosk-tr": dict(kind="vosk", url="https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip",
                    model_dir=str(MODELS_DIR / "vosk-model-small-tr-0.3")),
}

log = logging.getLogger("run_asr")


# --------------------------------------------------------------------------- systems
class FasterWhisper:
    def __init__(self, cfg: dict, threads: int):
        from faster_whisper import WhisperModel

        if "path" in cfg:
            if not Path(cfg["path"]).is_dir():
                raise SystemExit(f"converted model not found: {cfg['path']} (see SYSTEMS comment)")
            self.model = WhisperModel(cfg["path"], device="cpu", compute_type="int8",
                                      cpu_threads=threads, num_workers=1)
        else:
            self.model = WhisperModel(cfg["repo"], revision=cfg["revision"], device="cpu",
                                      compute_type="int8", cpu_threads=threads, num_workers=1)
        self.decode = {**WHISPER_DECODE, **cfg.get("decode", {})}
        self.config = dict(compute_type="int8", **self.decode)

    def __call__(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, **self.decode)
        return " ".join(s.text.strip() for s in segments).strip()  # generator: decodes here


class SherpaOnlineTransducer:
    def __init__(self, cfg: dict, threads: int):
        import sherpa_onnx
        from huggingface_hub import hf_hub_download

        files = {k: hf_hub_download(cfg["repo"], cfg[k], revision=cfg["revision"])
                 for k in ("tokens", "encoder", "decoder", "joiner")}
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            **files, num_threads=threads, sample_rate=SR, feature_dim=80,
            decoding_method=cfg["decoding_method"], max_active_paths=cfg["max_active_paths"],
            enable_endpoint_detection=False, provider="cpu")
        self.tail = np.zeros(int(cfg["tail_padding_s"] * SR), dtype=np.float32)
        self.config = {k: cfg[k] for k in ("decoding_method", "max_active_paths", "tail_padding_s",
                                           "encoder", "decoder", "joiner")}

    def __call__(self, audio: np.ndarray) -> str:
        stream = self.rec.create_stream()
        stream.accept_waveform(SR, audio)
        stream.accept_waveform(SR, self.tail)
        stream.input_finished()
        while self.rec.is_ready(stream):
            self.rec.decode_stream(stream)
        result = self.rec.get_result(stream)
        return (result if isinstance(result, str) else result.text).strip()


class Vosk:
    def __init__(self, cfg: dict, threads: int):
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        model_dir = Path(cfg["model_dir"])
        if not model_dir.is_dir():
            download_and_unzip(cfg["url"], model_dir.parent)
        self.model = Model(str(model_dir))
        self.config = {"model": model_dir.name, "chunk_samples": 4000, "note": "single-threaded"}

    def __call__(self, audio: np.ndarray) -> str:
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self.model, SR)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        parts = []
        step = 4000 * 2  # 0.25 s of int16
        for i in range(0, len(pcm), step):
            if rec.AcceptWaveform(pcm[i:i + step]):  # vosk's endpointer closed a segment
                parts.append(json.loads(rec.Result()).get("text", ""))
        parts.append(json.loads(rec.FinalResult()).get("text", ""))
        return " ".join(p for p in parts if p).strip()


def download_and_unzip(url: str, dest_parent: Path) -> None:
    dest_parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest_parent / Path(url).name
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as r, open(zip_path, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_parent)  # zipfile strips absolute / '..' member paths
    zip_path.unlink()


KINDS = {"faster-whisper": FasterWhisper, "sherpa-online-transducer": SherpaOnlineTransducer,
         "vosk": Vosk}


# --------------------------------------------------------------------------- helpers
def load_manifest(set_name: str) -> tuple[Path, list[dict]]:
    path = EVAL_DIR / set_name / "manifest.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found - run benchmarks/asr/prepare_sets.py first")
    with open(path, encoding="utf-8") as f:
        return path.parent, [json.loads(line) for line in f if line.strip()]


def load_done(out_path: Path) -> set[str]:
    """Ids already decoded. Drops a truncated trailing line left by a killed process."""
    if not out_path.exists():
        return set()
    good, bad = [], 0
    for line in out_path.read_text(encoding="utf-8").splitlines():
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        log.warning("%s: dropping %d unparsable line(s)", out_path, bad)
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in good),
                       encoding="utf-8")
        tmp.replace(out_path)
    elif good and not out_path.read_bytes().endswith(b"\n"):
        with open(out_path, "a", encoding="utf-8") as f:  # so the next append starts a new line
            f.write("\n")
    return {r["id"] for r in good}


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(audio, dtype=np.float32)


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("faster_whisper", "ctranslate2", "sherpa_onnx", "vosk", "onnxruntime"):
        if mod in sys.modules:
            out[mod] = getattr(sys.modules[mod], "__version__", "?")
    return out


def fmt_min(seconds: float) -> str:
    return f"{seconds / 60:.1f}min"


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="eval set name (dir under data/asr_eval/)")
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N manifest entries")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for the decoder")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    set_dir, manifest = load_manifest(args.set)
    if args.limit:
        manifest = manifest[: args.limit]
    out_dir = RESULTS_DIR / args.set
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.system}.jsonl"
    done = load_done(out_path)
    todo = [u for u in manifest if u["id"] not in done]
    log.info("%s / %s: %d utts selected, %d already done, %d to decode -> %s", args.set,
             args.system, len(manifest), len(manifest) - len(todo), len(todo), out_path)
    if not todo:
        return

    cfg = SYSTEMS[args.system]
    t0 = time.perf_counter()
    system = KINDS[cfg["kind"]](cfg, args.threads)
    load_s = time.perf_counter() - t0
    system(np.zeros(SR, dtype=np.float32))  # warm-up (lazy init), excluded from timing
    log.info("model loaded in %.1fs", load_s)

    meta_path = out_dir / f"{args.system}.meta.json"
    meta = {
        "system": args.system, "set": args.set,
        "model": {k: v for k, v in cfg.items() if k != "kind"} | {"kind": cfg["kind"]},
        "decode_config": system.config, "threads": args.threads,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "cpu": cpu_name(),
        "n_cpus": os.cpu_count(), "versions": versions(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    total_audio = sum(u["duration"] for u in todo)
    dec_sum = dur_sum = 0.0
    t_start = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, utt in enumerate(todo, 1):
            audio_path = Path(utt["audio_path"])
            if not audio_path.is_absolute():
                audio_path = set_dir / audio_path
            audio = load_audio(audio_path)
            t, c = time.perf_counter(), time.process_time()
            hyp = system(audio)
            dt, dc = time.perf_counter() - t, time.process_time() - c
            # cpu_seconds = process CPU time over all decoder threads (less sensitive to
            # other load on a shared machine than wall-clock decode_seconds).
            rec = {"id": utt["id"], "ref": utt["ref_text"], "hyp": hyp,
                   "duration": utt["duration"], "decode_seconds": round(dt, 4),
                   "cpu_seconds": round(dc, 4)}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            dec_sum += dt
            dur_sum += utt["duration"]
            if i % args.log_every == 0 or i == len(todo) or i <= 3:
                rtf = dec_sum / max(dur_sum, 1e-9)
                eta = (total_audio - dur_sum) * rtf
                log.info("[%d/%d] rtf=%.3f elapsed=%s eta=%s | %s | %s", i, len(todo), rtf,
                         fmt_min(time.perf_counter() - t_start), fmt_min(eta), utt["id"], hyp[:100])

    meta.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), n_decoded_this_run=len(todo),
                rtf_this_run=round(dec_sum / max(dur_sum, 1e-9), 4), model_load_seconds=round(load_s, 1))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    log.info("done: %d utts, %.1f min audio, rtf=%.3f -> %s", len(todo), dur_sum / 60,
             dec_sum / max(dur_sum, 1e-9), out_path)


if __name__ == "__main__":
    main()
