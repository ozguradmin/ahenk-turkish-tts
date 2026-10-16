"""Phone probe: a folder of WAVs -> the IPA phones phon_check hears (facebook/wav2vec2-xlsr-53-espeak-cv-ft, no
language model) and Whisper large-v3's transcript of each, as JSON, on a Modal CPU container with a hard timeout.

    python -m modal run tts/voxcpm/phon_probe.py --src voxcpm-contest/probe-canina2/f022-1500/wav --out phon.json \
        [--rows voxcpm-contest/probe-canina2/f022-1500/rows.jsonl] [--no-whisper]

Why both: Whisper's language model writes the real word ("canına") for a clip that says /k/; the CTC phone model has
none, so a wrong c shows as "k" / "d" / "ɡ" in place of "dʒ" (agreed 8 of 8 with our calibrated
listener on probe-vay and probe-canina2). --rows (a contest rows.jsonl: id = the WAV's stem, refs[0] or text the
canonical sentence) adds phon_check.c_check of each clip: expected c, dʒ heard, ok. WAVs: every *.wav under --src
(stems unique), sent with the call (at most MAX_MB). Container: PROBE_CPU cores + PROBE_MEM, the Volume
"turkish-tts-data" for the HF cache (/data/hf_home, as modal_app.py's: the IPA model downloaded once for the corpus gate
and this probe); clips are processed until PROBE_TAIL_S before the timeout, the rest listed as "skipped". Cost: at most
PROBE_TIMEOUT_S of PROBE_PRICE (printed first). The model code is phon_check's (PhonModel, ctc_tokens); Whisper pinned
at WHISPER_REV through transformers' pipeline, language tr.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
if str(_HERE.parent) not in sys.path:  # phon_check next to this file (in the container: /root, added below)
    sys.path.insert(0, str(_HERE.parent))

import phon_check as pc  # noqa: E402  (stdlib at import)

DATA = "/data"
WHISPER, WHISPER_REV = "openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"  # main, 2024-08-12
PROBE_CPU, PROBE_MEM, PROBE_TIMEOUT_S, PROBE_TAIL_S = 8, 12288, 900, 60
PROBE_PRICE = PROBE_CPU * 0.047 + PROBE_MEM / 1024 * 0.008  # $/h, Modal list (CPU core, GiB)
MAX_MB = 200  # WAV bytes one call takes

app = modal.App("phon-probe")
volume = modal.Volume.from_name("turkish-tts-data", create_if_missing=True)
# the stack the 2026-10-12 probe ran on (phonemizer / espeak-ng no longer needed: phon_check decodes vocab.json itself)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("transformers==4.44.2", "librosa==0.10.2.post1", "soundfile", "numpy<2")
         .env({"HF_HOME": f"{DATA}/hf_home", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_file(str(_HERE.with_name("phon_check.py")), "/root/phon_check.py"))


def collect(src: Path) -> dict[str, Path]:
    """stem -> path of every *.wav under src (sorted); ValueError on two WAVs with one stem."""
    out: dict[str, Path] = {}
    for p in sorted(src.rglob("*.wav")):
        if p.stem in out:
            raise ValueError(f"two WAVs named {p.stem}: {out[p.stem]} and {p}")
        out[p.stem] = p
    return out


def canonical_texts(rows: Path) -> dict[str, str]:
    """id -> the canonical sentence (refs[0], else text) of a contest rows.jsonl / texts file."""
    out = {}
    for line in rows.read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else {}
        refs = r.get("refs") if isinstance(r.get("refs"), list) else []
        text = refs[0] if refs and isinstance(refs[0], str) else r.get("text")
        if r.get("id") is not None and isinstance(text, str):
            out[str(r["id"])] = text
    return out


def annotate(res: dict[str, dict], texts: dict[str, str]) -> dict[str, dict]:
    """Each clip's result with phon_check.c_check of its canonical text (texts[stem], else texts[stem without its last
    "-part"]: "k7-s0" -> "k7"): text, c_expected, c_got, c_ok."""
    for k, r in res.items():
        text = texts.get(k, texts.get(k.rsplit("-", 1)[0]))
        if text is not None and r.get("tokens") is not None:
            ok, _ = pc.c_check(text, r["tokens"])
            r.update(text=text, c_expected=pc.c_expected(text), c_got=pc.dj_count(r["tokens"]), c_ok=ok)
    return res


@app.function(image=image, cpu=PROBE_CPU, memory=PROBE_MEM, volumes={DATA: volume}, timeout=PROBE_TIMEOUT_S)
def run(blobs: dict, whisper: bool = True) -> dict:
    """IPA tokens (phon_check.PhonModel on the CPU) and, with whisper, Whisper's transcript of each WAV (bytes), until
    PROBE_TAIL_S before the timeout; {stem: {"ipa", "tokens", "whisper"}, "_skipped": [...], "_s": seconds}."""
    import io

    import librosa
    import soundfile as sf
    import torch

    t0 = time.time()
    torch.set_num_threads(PROBE_CPU)
    pm = pc.PhonModel("cpu")
    asr = None
    if whisper:
        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", model=WHISPER, revision=WHISPER_REV, device="cpu")
    out, skipped = {}, []
    for k, b in sorted(blobs.items()):
        if time.time() > t0 + PROBE_TIMEOUT_S - PROBE_TAIL_S:
            skipped.append(k)
            continue
        y, sr = sf.read(io.BytesIO(b), dtype="float32")
        y = y.mean(axis=1) if y.ndim > 1 else y
        y = librosa.resample(y, orig_sr=sr, target_sr=pc.IPA_SR)
        toks = pm.ipa(y, pc.IPA_SR)
        out[k] = {"ipa": pc.ipa_text(toks), "tokens": toks}
        if asr is not None:
            out[k]["whisper"] = asr(y.copy(), generate_kwargs={"language": "turkish", "task": "transcribe"})[
                "text"].strip()
        print(k, out[k]["ipa"], "|", out[k].get("whisper", ""), flush=True)
    volume.commit()  # the HF cache for the next call
    return {**out, "_skipped": skipped, "_s": round(time.time() - t0)}


@app.local_entrypoint()
def main(src: str, out: str, rows: str = "", no_whisper: bool = False):
    wavs = collect(Path(src))
    size = sum(p.stat().st_size for p in wavs.values())
    if not wavs or size > MAX_MB * 2 ** 20:
        raise SystemExit(f"{src}: {len(wavs)} WAVs, {size / 2 ** 20:.0f} MB (want 1 to {MAX_MB} MB)")
    texts = canonical_texts(Path(rows)) if rows else {}
    print(f"[phon-probe] {len(wavs)} WAVs ({size / 2 ** 20:.1f} MB), {PROBE_CPU} CPU cores + {PROBE_MEM // 1024} GiB: "
          f"at most ${(PROBE_TIMEOUT_S + 2) / 3600 * PROBE_PRICE:.2f} (its {PROBE_TIMEOUT_S // 60} min timeout)"
          + (f"; c-check against {len(texts)} texts of {rows}" if texts else ""))
    res = run.remote({k: p.read_bytes() for k, p in wavs.items()}, not no_whisper)
    meta = {k: res.pop(k) for k in ("_skipped", "_s")}
    res = annotate(res, texts)
    Path(out).write_text(json.dumps({**res, **meta}, ensure_ascii=False, indent=1), encoding="utf-8")
    checked = [r for r in res.values() if "c_ok" in r]
    print(f"[phon-probe] {len(res)} clips in {meta['_s']} s (~${meta['_s'] / 3600 * PROBE_PRICE:.3f}) -> {out}"
          + (f"; c-check: {sum(r['c_ok'] for r in checked)} of {len(checked)} ok" if checked else "")
          + (f"; skipped at the deadline: {meta['_skipped']}" if meta["_skipped"] else ""))
