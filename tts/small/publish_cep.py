#!/usr/bin/env python3
"""Stage the Ahenk Cep model repo (small on-device student) for Hugging Face. Stages only: nothing is uploaded.

    python tts/small/publish_cep.py --weights data/small/runs/final --samples-dir data/release-samples
        [--repo ozgurguler/ahenk-cep-tr] [--out hf-publish/ahenk-cep-tr]
    python tts/small/publish_cep.py --check hf-publish/ahenk-cep-tr

Staging dir: README.md rendered from MODEL_CARD.ahenk-cep.template.md (sizes and sha256 of the staged files),
student.pt + decoder.pt (copied as they are), ahenk_cep.py (the inference module, copied from this folder) and
samples/<SAMPLES>. README.md is written last, so a stage that stops halfway leaves no card. check_staging() lists what
blocks an upload: placeholders left, a missing or changed file, card links and samples/*.wav that differ, an em dash.
Upload by hand after review, e.g. `hf upload ozgurguler/ahenk-cep-tr hf-publish/ahenk-cep-tr . --private`
(create the repo private first).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEMPLATE = HERE / "MODEL_CARD.ahenk-cep.template.md"
WEIGHTS = ("student.pt", "decoder.pt")
CODE = "ahenk_cep.py"
SAMPLES = ("small_01_question_garantisi.wav", "small_02_question_saat.wav", "small_03_long_heldout583922.wav")
DEFAULT_REPO = "ozgurguler/ahenk-cep-tr"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def render(template: str, values: dict) -> str:
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(values.get(m.group(1), m.group(0))), template)


def stage(weights: Path, samples_dir: Path, out: Path, repo: str = DEFAULT_REPO,
          template: Path = TEMPLATE) -> list[str]:
    missing = [str(p) for p in [*(weights / w for w in WEIGHTS), *(samples_dir / s for s in SAMPLES)]
               if not p.is_file()]
    if missing:
        raise SystemExit("[publish_cep] missing: " + ", ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").unlink(missing_ok=True)
    for w in WEIGHTS:
        shutil.copyfile(weights / w, out / w)
    shutil.copyfile(HERE / CODE, out / CODE)
    shutil.rmtree(out / "samples", ignore_errors=True)
    (out / "samples").mkdir()
    for s in SAMPLES:
        shutil.copyfile(samples_dir / s, out / "samples" / s)
    values = {"repo": repo,
              "student_mb": f"{(out / 'student.pt').stat().st_size / 1e6:.1f}",
              "student_sha256": sha256(out / "student.pt"),
              "decoder_mb": f"{(out / 'decoder.pt').stat().st_size / 1e6:.1f}",
              "decoder_sha256": sha256(out / "decoder.pt"),
              "code_kb": f"{(out / CODE).stat().st_size / 1e3:.1f}", "code_sha256": sha256(out / CODE)}
    card = render(template.read_text(encoding="utf-8-sig"), values)
    (tmp := out / "README.md.tmp").write_text(card, encoding="utf-8", newline="\n")
    os.replace(tmp, out / "README.md")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    print(f"[publish_cep] staged {out} for {repo} (nothing uploaded):")
    for p in files:
        print(f"  {p.relative_to(out).as_posix():44s} {p.stat().st_size / 1e6:8.2f} MB")
    print(f"  total {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
    problems = check_staging(out)
    for p in problems:
        print(f"[publish_cep] NOT READY: {p}")
    return problems


def check_staging(out: Path) -> list[str]:
    readme = out / "README.md"
    if not readme.is_file():
        return [f"{readme} missing (stage first)"]
    text = readme.read_text(encoding="utf-8")
    problems = []
    if left := sorted(set(re.findall(r"\{\{[^}]*\}\}", text))):
        problems.append(f"README.md still has placeholders: {', '.join(left)}")
    if "—" in text:
        problems.append("README.md has an em dash")
    for name in (*WEIGHTS, CODE):
        p = out / name
        if not p.is_file() or p.stat().st_size == 0:
            problems.append(f"{name} missing or empty")
        elif f"`{sha256(p)}`" not in text:
            problems.append(f"README.md does not give the sha256 of the staged {name}: stage again")
    if (out / CODE).is_file() and (out / CODE).read_bytes() != (HERE / CODE).read_bytes():
        problems.append(f"{CODE} differs from tts/small/{CODE}: stage again")
    repo = (re.findall(r'snapshot_download\("([^"]+)"', text) or [""])[0]  # links into other repos are not ours
    linked = {n for r, n in re.findall(r"huggingface\.co/([^/\s]+/[^/\s]+)/resolve/main/samples/([^\"'\s)<>]+\.wav)",
                                       text) if r == repo}
    if not repo:
        problems.append("README.md has no snapshot_download(\"<repo>\") usage line")
    wavs = {p.name for p in (out / "samples").glob("*.wav")}
    if linked - wavs:
        problems.append(f"README.md links samples that are not staged: {', '.join(sorted(linked - wavs))}")
    if wavs - linked:
        problems.append(f"samples/ holds WAVs README.md does not link: {', '.join(sorted(wavs - linked))}")
    extra = sorted(p.name for p in out.iterdir() if p.is_file() and p.name not in {"README.md", *WEIGHTS, CODE})
    if extra:
        problems.append(f"unexpected files in the staging dir: {', '.join(extra)}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="dir with student.pt and decoder.pt")
    ap.add_argument("--samples-dir", help=f"dir with {', '.join(SAMPLES)}")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", help="staging dir [hf-publish/<repo name>]")
    ap.add_argument("--check", metavar="DIR", help="only check a staging dir")
    args = ap.parse_args(argv)
    if args.check:
        problems = check_staging(Path(args.check))
        print("\n".join(f"[publish_cep] NOT READY: {p}" for p in problems) or "[publish_cep] ready")
        return 1 if problems else 0
    if not (args.weights and args.samples_dir):
        ap.error("staging needs --weights and --samples-dir")
    out = Path(args.out or REPO / "hf-publish" / args.repo.split("/")[-1])
    return 1 if stage(Path(args.weights), Path(args.samples_dir), out, args.repo) else 0


if __name__ == "__main__":
    raise SystemExit(main())
