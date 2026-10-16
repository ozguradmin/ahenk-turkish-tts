#!/usr/bin/env python3
"""Stage (and, with --upload, publish) a VoxCPM2 LoRA (Antalia, Alania/Antalia, f022) as a Hugging Face model repo.

    python tts/voxcpm/publish_hf.py --name "Ad" --repo ozgurguler/ad --lora voxcpm-lora/step_0000600 \
        --evals voxcpm-eval/step_0000600 voxcpm-eval/trendyol        # stage only -> hf-publish/ad/
        [--exp-name voxcpm2-antalia]                                  # Modal experiment of the checkpoint
    python tts/voxcpm/publish_hf.py --repo ozgurguler/ad --upload      # upload hf-publish/ad/ as staged, PRIVATE repo
    python tts/voxcpm/publish_hf.py --repo ozgurguler/ad --upload --public

Staging dir (--out, default hf-publish/<repo name>): README.md rendered from MODEL_CARD.template.md,
lora_weights.safetensors + lora_config.json of the checkpoint (optimizer/scheduler/training state stay home),
samples/*.wav: --samples clips of the FIRST eval dir, a third held-out and the rest TR-Zor, one per category in file
order, each the shortest clip >= --min-s of its group, picked without looking at the ASR result.
Card numbers come only from files: r/alpha/targets/parameter count from the LoRA files, WER/rows_correct/UTMOS/RTF/
inference settings and TR-Soru intonation (summary "intonation", templates that use {{intonation_table}}) from each
eval dir's summary.json; a missing value renders as "—". The first --evals dir is the
published model, the rest are references (DIR or DIR=Label); its summary "lora" must end in
exp/<--exp-name>/ckpt/<--lora dir name> (fetch/fetch-ckpt name local dirs by checkpoint only, so step_0000600 of two
experiments would collide). The usage snippet takes cfg/timesteps from the first summary (synth_eval's defaults
2.0/10 when it has none). Re-staging deletes README.md first and writes it last (via README.md.tmp), so a stage that
stops halfway leaves no card and --upload refuses.

Package files (--template f022s3 or ahenk, TEMPLATE_EXTRAS): the card's usage code needs files beside the LoRA, copied from
tts/voxcpm as they are with the tts/voxcpm modules they import (bundle: pause_clean.py -> onset.py) to the staging dir's
root: instruction_pick.py (instruction language per line: Turkish when one of the first two words starts with "c",
else English; stdlib), pause_clean.py + onset.py (breath noise in pauses, after synthesis; numpy + scipy) and
instructions_tr.json (the English -> Turkish instruction table). {{extra_files}} lists each with its sha256;
{{instructions_tr_table}} renders the table's Turkish training wordings per emotion (build_textbank.INSTRUCTIONS split
"train" and render_corpus.LAUGH_INSTRUCTIONS, read without importing them) and how many of its wordings those are;
{{pause_clean_note}} says whether the first eval's clips went through pause_clean (its summary "pause_clean":
synth_eval --pause-clean). A stage with another template removes package files an earlier stage left.

--upload uploads the staging dir AS IT IS (stage, review/edit README.md, then upload): it refuses while README.md
still holds "{{" placeholders or "TODO(final" markers, the LoRA files are missing or broken, README.md's sha256 or LoRA
r/alpha do not match the staged weights, its sample links and samples/*.wav differ, or its package file table and the
staged package files differ (a listed file missing or changed since staging, an unlisted one staged, a staged .py
whose tts/voxcpm import is not staged, a .py that does not parse, an instructions_tr.json that is not a {str: str}
table, an instruction_pick.py whose EMOTIONS pairs are not that table's). It sends only README.md, the two LoRA
files, samples/*.wav and the listed package files and, in the same commit, deletes remote samples/*.wav and package
files that are no longer staged; creates the repo PRIVATE unless --public is given, and refuses to upload into an
existing public repo without --public (with --public an existing private repo is made public after the upload).

Card usage verified against OpenBMB/VoxCPM f0c787f: from_pretrained(hub id, lora_weights_path=<local dir with
lora_weights.safetensors + lora_config.json>) reads r/alpha from lora_config.json["lora_config"]; PyPI voxcpm 2.0.3
(latest) does not (falls back to r=8) and has no generate(seed=), so the snippet passes lora_config explicitly.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEMPLATE = HERE / "MODEL_CARD.template.md"
DEFAULT_LABEL = "VoxCPM2 + Antalia LoRA"
LORA_LABEL = DEFAULT_LABEL  # results-table label of our row; main() sets it from --template (TEMPLATE_LABELS)
TEMPLATE_LABELS = {"f022": "VoxCPM2 + f022 LoRA (designed voice)",  # template file name part -> LORA_LABEL
                   "f022s3": "VoxCPM2 + f022-s3tr LoRA (designed voice)",  # the longest part in the name wins
                   "two-stage": "VoxCPM2 + Alania/Antalia LoRA",
                   "ahenk": "VoxCPM2 2B + LoRA"}  # the public "Ahenk" card (same release as f022s3)
# package files of a template's release beside the LoRA (from tts/voxcpm, + the modules they import: bundle)
TEMPLATE_EXTRAS = {"f022s3": ("instruction_pick.py", "pause_clean.py", "instructions_tr.json"),
                   "ahenk": ("instruction_pick.py", "pause_clean.py", "instructions_tr.json")}
EXTRA_INFO = {  # {{extra_files}} contents column
    "instruction_pick.py": "talimatın dilini seçer: ilk iki kelimeden biri c ile başlıyorsa Türkçe, yoksa "
                           "İngilizce; Vay -> Vaay (yalnız standart kütüphane) / picks the instruction language: "
                           "Turkish when one of the first two words starts with c, else English; Vay -> Vaay (stdlib "
                           "only)",
    "pause_clean.py": "sentezden sonra duraklardaki nefes sesini kısar (numpy + scipy) / turns breath noise in pauses "
                      "down after synthesis (numpy + scipy)",
    "onset.py": "pause_clean.py'nin kullandığı çerçeve ve perde yardımcıları / the frame and pitch helpers "
                "pause_clean.py imports",
    "instructions_tr.json": "İngilizce talimat -> Türkçe karşılığı, {n} ifade / English instruction -> its Turkish "
                            "wording, {n} wordings",
}
EXTRA_SUFFIXES = (".py", ".json")  # root files of a staging dir that are package files (lora_config.json aside)
NO_EM_DASH = {"ahenk"}  # templates whose rendered card uses no em dash: missing values "-", " — " joins ", "
TODO_MARK = "TODO(final"  # a template's "fill in for the released checkpoint" marker: no upload while one is left
VOXCPM_COMMIT = "f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629"  # same pin as modal_app.py (training + eval)
LORA_FILES = ("lora_weights.safetensors", "lora_config.json")
UPLOAD_PATTERNS = ["README.md", *LORA_FILES, "samples/*.wav"]  # + the staged package files (upload)
# remote samples / package files no longer staged go in the same commit (re-uploaded ones stay)
DELETE_PATTERNS = ["samples/*.wav", "*.py", *sorted({n for ns in TEMPLATE_EXTRAS.values() for n in ns
                                                     if not n.endswith(".py")})]
EMOTION_NAMES = {"angry": "kızgın / angry", "cheerful": "neşeli / cheerful", "sad": "üzgün / sad",
                 "surprised": "şaşkın / surprised", "calm": "sakin / calm", "empathetic": "empatik / empathetic",
                 "laughter": "gülme / laughter"}
EXP = "voxcpm2-antalia"  # modal_app.py EXP (not imported: it imports modal)
DASH = "—"
DEFAULT_CFG, DEFAULT_TIMESTEPS = 2.0, 10  # synth_eval.py defaults
KNOWN_LABELS = {  # eval dir name (before any "@" grid suffix) -> (results table label, category table header)
    "trendyol": ("[Trendyol-TTS](https://huggingface.co/Trendyol/Trendyol-TTS) (VoxCPM2 + LoRA, ~20 sa gizli veri / "
                 "~20 h private data)", "Trendyol-TTS"),
    "base": ("VoxCPM2, ince ayarsız (yeniden ölçüm) / not fine-tuned (re-run)", "VoxCPM2 (base)"),
}
CATEGORY_NAMES = {  # TR-Zor categories (benchmarks/tts/sets/tr_zor.jsonl)
    "sayi": "sayı / numbers", "sira": "sıra sayısı / ordinals", "tarih": "tarih / dates", "saat": "saat / times",
    "para": "para / money", "yuzde": "yüzde / percentages", "birim": "birim / units",
    "numara": "telefon numarası / phone numbers", "kisaltma": "kısaltma (Prof. Dr.) / abbreviations",
    "akronim": "akronim (TBMM, THY) / acronyms", "kisa": "kısa cevap / short replies", "soru": "soru / questions",
    "ozel": "özel yazım (kâğıt, hâlâ) / special spellings",
}


def _assigned(path: Path, name: str) -> ast.expr:
    """The value node of a module-level `name = ...` (or `name: T = ...`) of a source file, read without importing it
    (heavy imports); RuntimeError when there is none."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        if any(getattr(t, "id", "") == name for t in targets):
            return node.value
    raise RuntimeError(f"{name} not found in {path.name}")


def attribution() -> str:
    """The exact ATTRIBUTION string of tts/prepare_antalia.py, read without importing it (heavy imports)."""
    return ast.literal_eval(_assigned(REPO / "tts" / "prepare_antalia.py", "ATTRIBUTION"))


def training_instructions() -> dict[str, list[str]]:
    """English instruction wordings the f022 corpora trained, per emotion in file order (the first: the "mid" one the
    cards recommend): build_textbank.INSTRUCTIONS with split "train" (the held-out ones were never trained) and
    "laughter": render_corpus.LAUGH_INSTRUCTIONS (the approved laughter clips' wordings); read without importing."""
    out = {}
    node = _assigned(HERE / "build_textbank.py", "INSTRUCTIONS")
    for k, v in zip(node.keys, node.values):  # "emotion": _para("emotion", [(text, level, split), ...])
        out[ast.literal_eval(k)] = [t for t, _, split in ast.literal_eval(v.args[1]) if split == "train"]
    out["laughter"] = list(ast.literal_eval(_assigned(HERE / "render_corpus.py", "LAUGH_INSTRUCTIONS")))
    return out


def instructions_tr_md(table_path: Path = HERE / "instructions_tr.json") -> str:
    """{{instructions_tr_table}}: per emotion the Turkish wordings (instructions_tr.json) of its training wordings
    (training_instructions), then how many of the table's wordings were trained; SystemExit when it lacks one."""
    table = json.loads(table_path.read_text(encoding="utf-8"))
    lines = ["| Duygu / Emotion | Önerilen / Recommended | İngilizce aslı / English original | Diğer eğitim ifadeleri "
             "/ other training wordings |", "|---|---|---|---|"]
    trained = training_instructions()
    for emo, ens in trained.items():
        if missing := [e for e in ens if e not in table]:
            raise SystemExit(f"[publish] {table_path.name} has no Turkish wording for {missing}")
        tr = [table[e] for e in ens]
        lines.append(f"| {EMOTION_NAMES.get(emo, emo)} | `{tr[0]}` | `{ens[0]}` | "
                     + "<br>".join(f"`{t}`" for t in tr[1:]) + " |")
    n, k = len(table), len({e for ens in trained.values() for e in ens})
    lines += ["", f"Tablo, `{table_path.name}`'daki {n} ifadeden eğitimde kullanılan {k} ifadeyi gösterir; kalan "
                  f"{n - k} ifade (ayrılmış ifadeler ve dinleme yarışmasının ifadeleri) eğitimde yoktu / the table "
                  f"shows the {k} of the {n} wordings in `{table_path.name}` that training used; the other {n - k} "
                  "(held-out wordings and the listening contest's) were never trained."]
    return "\n".join(lines)


def local_imports(path: Path) -> set[str]:
    """The tts/voxcpm modules (HERE/<name>.py) a Python file imports anywhere in it (lazy imports too)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return {n for n in names if (HERE / f"{n}.py").is_file()}


def bundle(names) -> list[str]:
    """Package files of tts/voxcpm: names, each .py followed by the modules it imports from there (transitively,
    once each): ("pause_clean.py", "instructions_tr.json") -> pause_clean.py, onset.py, instructions_tr.json."""
    out, todo = [], list(names)
    while todo:
        n = todo.pop(0)
        if n in out:
            continue
        if not (HERE / n).is_file():
            raise SystemExit(f"[publish] package file {HERE / n} not found")
        out.append(n)
        if n.endswith(".py"):
            todo[:0] = [f"{m}.py" for m in sorted(local_imports(HERE / n))]
    return out


def extra_info(path: Path) -> str:
    info = EXTRA_INFO.get(path.name, DASH)
    if "{n}" in info:
        info = info.replace("{n}", str(len(json.loads(path.read_text(encoding="utf-8")))))
    return info


def extras_md(out: Path, names: list[str]) -> str:
    """{{extra_files}}: one row per staged package file with its sha256 (check_staging reads them back)."""
    if not names:
        return DASH
    return "\n".join(["| Dosya / File | İçerik / Contents | sha256 |", "|---|---|---|",
                      *(f"| `{n}` | {extra_info(out / n)} | `{sha256(out / n)}` |" for n in names)])


def pause_note(summary: dict) -> str:
    """{{pause_clean_note}}: whether the eval's clips went through pause_clean (synth_eval summary "pause_clean")."""
    pc = summary.get("pause_clean")
    if not isinstance(pc, dict):
        return ("Bu ölçümün klipleri `pause_clean.py`'den geçmedi / the clips of this evaluation did not go through "
                "`pause_clean.py`.")
    gated = fmt(pc.get("gated_s"), 1)
    return (f"Bu ölçümün klipleri sentezden sonra `pause_clean.py`'den geçti ({pc.get('clips')} klip, "
            f"{pc.get('breaths')} nefes kısıldı, toplam {gated.replace('.', ',')} sn; ayar `{pc.get('settings')}`) / "
            f"the clips of this evaluation went through `pause_clean.py` after synthesis ({pc.get('clips')} clips, "
            f"{pc.get('breaths')} breaths turned down, {gated} s in all; settings "
            f"`{pc.get('settings')}`).")


def resolve_template(spec: str | Path) -> Path:
    """--template: a file path, or a short name ("ahenk", "f022s3") for tts/voxcpm/MODEL_CARD.<name>.template.md."""
    p = Path(spec)
    if p.is_file():
        return p
    short = HERE / f"MODEL_CARD.{spec}.template.md"
    if "/" not in str(spec) and "\\" not in str(spec) and short.is_file():
        return short
    raise SystemExit(f"[publish] --template {spec}: no such file, and no {short.name} in {HERE}")


def template_key(template: str | Path) -> str | None:
    """The TEMPLATE_LABELS part a template's file name holds (the longest: "f022s3" over "f022"), else None."""
    name = Path(template).name
    return next((k for k in sorted(TEMPLATE_LABELS, key=len, reverse=True) if k in name), None)


def safetensors_header(path: Path) -> dict:
    """{tensor name: {"dtype", "shape", "data_offsets"}} from the file header; raises on a broken file."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if not 2 <= n <= min(size - 8, 100 << 20):
            raise ValueError(f"{path.name}: bad safetensors header length {n} (file {size} bytes)")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    end = max((v["data_offsets"][1] for v in header.values()), default=0)
    if 8 + n + end != size:
        raise ValueError(f"{path.name}: header says {8 + n + end} bytes, file has {size} (truncated?)")
    return header


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def lora_info(lora_dir: Path) -> dict:
    """Card values from a checkpoint dir: r/alpha/targets from lora_config.json, size/params/rank from the weights."""
    cfg = json.loads((lora_dir / "lora_config.json").read_text(encoding="utf-8"))
    lc = cfg.get("lora_config") or {}
    header = safetensors_header(lora_dir / "lora_weights.safetensors")
    keys = [k for k in header if "lora_" in k]
    if not keys:
        raise ValueError(f"{lora_dir / 'lora_weights.safetensors'} holds no lora_ tensors: {list(header)[:3]}")
    ranks = sorted({header[k]["shape"][0] for k in keys if k.endswith("lora_A")})
    if lc.get("r") is not None and ranks and ranks != [lc["r"]]:
        raise ValueError(f"lora_config.json says r={lc['r']} but lora_A tensors have rank {ranks}")
    targets = []
    for flag, part, mods in (("enable_lm", "LM", "target_modules_lm"), ("enable_dit", "DiT", "target_modules_dit"),
                             ("enable_proj", "proj", "target_proj_modules")):
        if lc.get(flag):
            targets.append(f"{part} ({', '.join(lc.get(mods) or [])})" if lc.get(mods) else part)
    w = lora_dir / "lora_weights.safetensors"
    state = lora_dir / "training_state.json"
    step = json.loads(state.read_text(encoding="utf-8")).get("step") if state.exists() else None
    if step is None and (m := re.match(r"step_(\d+)$", lora_dir.name)):
        step = int(m.group(1))
    params = sum(math.prod(header[k]["shape"]) for k in keys)
    return {"checkpoint": lora_dir.name, "step": step, "lora_r": lc.get("r"), "lora_alpha": lc.get("alpha"),
            "lora_targets": " + ".join(targets) or None,
            "lora_params": f"{params / 1e6:.1f} M" if params >= 1e5 else f"{params:,}",
            "lora_mb": f"{w.stat().st_size / 2**20:.1f}", "lora_sha256": sha256(w),
            "base_model": cfg.get("base_model")}


def load_eval(spec: str) -> dict:
    path, _, label = spec.partition("=")
    d = Path(path)
    if not (d / "summary.json").exists():
        raise SystemExit(f"[publish] {d / 'summary.json'} not found")
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    rows = []
    if (d / "rows.jsonl").exists():
        rows = [json.loads(x) for x in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return {"dir": d, "label": label, "summary": summary, "rows": rows}


def fmt(v, nd: int = 2) -> str:
    return DASH if v is None else f"{v:.{nd}f}"


def _get(summary: dict, metric: str, key: str, field: str):
    return ((summary.get(metric) or {}).get(key) or {}).get(field)


def _correct(summary: dict, key: str) -> str:
    ok, n = _get(summary, "wer", key, "rows_correct"), _get(summary, "wer", key, "rows")
    return DASH if ok is None or n is None else f"{ok}/{n}"


def _settings(summary: dict) -> str:
    ts, cfg = summary.get("timesteps"), summary.get("cfg")
    return DASH if ts is None or cfg is None else f"{ts} / {float(cfg)}"


def _subset(summary: dict) -> str:
    return f" — örneklem / subset: ≤ {summary['limit']} satır/rows per set ({summary.get('sample', 'head')})" \
        if summary.get("limit") else ""


def label_of(ev: dict, name: str, checkpoint: str, ours: bool, short: bool = False) -> str:
    if ev["label"]:
        return ev["label"]
    if ours:
        return f"**{name}**" if short else f"**{name}** ({LORA_LABEL}, `{checkpoint}`)"
    known = KNOWN_LABELS.get(ev["dir"].name.split("@")[0])
    return known[short] if known else f"`{ev['dir'].name}`"


def results_rows(evals: list[dict], name: str, checkpoint: str) -> str:
    out = []
    for i, ev in enumerate(evals):
        s = ev["summary"]
        cells = [label_of(ev, name, checkpoint, i == 0) + _subset(s), _settings(s),
                 fmt(_get(s, "wer", "trzor", "wer")), _correct(s, "trzor"),
                 fmt(_get(s, "wer", "heldout", "wer")), _correct(s, "heldout"),
                 fmt(_get(s, "utmos", "trzor", "mean")), fmt(_get(s, "utmos", "heldout", "mean")), fmt(s.get("rtf"))]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def trzor_categories() -> list[str]:
    cats = []
    for line in (REPO / "benchmarks" / "tts" / "sets" / "tr_zor.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip() and (c := json.loads(line)["category"]) not in cats:
            cats.append(c)
    return cats


def category_table(evals: list[dict], name: str, checkpoint: str) -> str:
    found = {k.split("/", 1)[1] for ev in evals for k in (ev["summary"].get("wer") or {}) if k.startswith("trzor/")}
    if not found:
        return DASH
    order = [c for c in trzor_categories() if c in found] + sorted(found - set(trzor_categories()))
    heads = [label_of(ev, name, checkpoint, i == 0, short=True) for i, ev in enumerate(evals)]
    lines = ["| Kategori / Category | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(evals)]
    for c in order:
        cells = []
        for ev in evals:
            w = _get(ev["summary"], "wer", f"trzor/{c}", "wer")
            cells.append(DASH if w is None else f"{w:.2f} ({_correct(ev['summary'], f'trzor/{c}')})")
        lines.append(f"| {CATEGORY_NAMES.get(c, c)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def intonation_table(evals: list[dict], name: str, checkpoint: str) -> str:
    """TR-Soru question intonation (score_intonation.py, summary["intonation"]["soru"]) of the evals that have it."""
    lines = []
    for i, ev in enumerate(evals):
        s = ev["summary"]
        if not (it := (s.get("intonation") or {}).get("soru")):
            continue
        acc, n = it.get("pair_acc"), it.get("pairs")
        cells = [label_of(ev, name, checkpoint, i == 0) + _subset(s), _settings(s),
                 DASH if acc is None or not n else f"{round(acc * n)}/{n}", fmt(it.get("auc_peak")),
                 fmt(it.get("questions_pass_rate")), fmt(_get(s, "wer", "soru", "wer")), _correct(s, "soru")]
        lines.append("| " + " | ".join(cells) + " |")
    if not lines:
        return DASH
    return "\n".join(["| Model | Çıkarım / Inference (timesteps / cfg) | Eşler / pairs | AUC | Kalıba uyan sorular / "
                      "questions matching the pattern | TR-Soru WER | TR-Soru hatasız / error-free |",
                      "|---|---|---:|---:|---:|---:|---:|", *lines])


def wav_of(ev: dict, row: dict) -> Path:
    """rows.jsonl "wav" holds the container path (/data/exp/...): resolve it inside the local eval dir."""
    return ev["dir"] / "wav" / row["set"] / f"{row['id']}.wav"


def pick_samples(ev: dict, n: int, min_s: float) -> list[dict]:
    """n rows: n // 3 (>= 1) held-out, the rest TR-Zor round-robin over categories in file order; per group the
    shortest clip >= min_s first (then shorter ones). Rows whose WAV is missing locally are skipped."""
    groups = defaultdict(list)
    for r in ev["rows"]:
        if r.get("duration") is not None and wav_of(ev, r).exists():
            groups[(r["set"], r["category"])].append(r)
    for g in groups.values():
        g.sort(key=lambda r: (r["duration"] < min_s, r["duration"], r["id"]))
    held = groups.pop(("heldout", "heldout"), [])
    order = {c: i for i, c in enumerate(trzor_categories())}
    trzor = sorted((k for k in groups if k[0] == "trzor"), key=lambda k: (order.get(k[1], len(order)), k[1]))
    queues = [groups[k] for k in trzor]
    n_held = min(len(held), n if not queues else (max(1, n // 3) if n > 1 else 0))
    picked = held[:n_held]
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop(0))
    picked += held[n_held:n_held + n - len(picked)]  # fewer TR-Zor rows than asked: top up with held-out
    return picked


def sample_name(row: dict) -> str:
    return f"{row['id']}.wav" if row["id"].startswith(row["set"]) else f"{row['set']}_{row['id']}.wav"


def samples_md(picked: list[dict], repo: str, min_s: float) -> str:
    if not picked:
        return f"Örnek yok / no samples ({DASH})."
    lines = [f"Seçim / selection: her gruptan en az {min_s:g} sn olan en kısa klip, ASR sonucuna bakılmadan; TR-Zor "
             f"metni modele trnorm'dan geçerek verildi / the shortest clip of at least {min_s:g} s of each group, "
             f"picked without looking at the ASR result; TR-Zor text was fed through trnorm.", ""]
    for r in picked:
        where = ("Ayrılmış / held-out" if r["set"] == "heldout"
                 else f"TR-Zor / {CATEGORY_NAMES.get(r['category'], r['category'])}")
        url = f"https://huggingface.co/{repo}/resolve/main/samples/{sample_name(r)}"
        lines += [f"- **{where}** ({r['duration']:.1f} s): {html.escape(r['text'], quote=False)}",
                  f"  <audio controls src=\"{url}\"></audio>"]
    return "\n".join(lines)


def extra_samples_md(rows: list[dict], repo: str) -> str:
    """--extra-samples rows (title, instruction, text, wav) as the card's list; staged as samples/<wav name>."""
    if not rows:
        return f"Örnek yok / no samples ({DASH})."
    lines = []
    for r in rows:
        ins = f" — `({html.escape(r['instruction'], quote=False)})`" if r.get("instruction") else ""
        url = f"https://huggingface.co/{repo}/resolve/main/samples/{Path(r['wav']).name}"
        lines += [f"- **{html.escape(r['title'], quote=False)}**{ins}: {html.escape(r['text'], quote=False)}",
                  f"  <audio controls src=\"{url}\"></audio>"]
    return "\n".join(lines)


def _cell(text: str) -> str:
    return html.escape(text, quote=False).replace("|", "\\|").replace("\n", " ")


def extra_samples_table_md(rows: list[dict], repo: str) -> str:
    """--extra-samples rows as a table (the {{extra_samples_table}} placeholder): what it shows, the text as typed
    (+ the spoken form the model read, when "spoken" differs), the instruction, a player and how it was picked
    ("pick")."""
    if not rows:
        return f"Örnek yok / no samples ({DASH})."
    lines = ["| Ne gösteriyor / What it shows | Metin / Text | Talimat / Instruction | Ses / Audio | Seçim / Picked by |",
             "|---|---|---|---|---|"]
    for r in rows:
        text = _cell(r["text"])
        if r.get("spoken") and r["spoken"].strip() != r["text"].strip():
            text += f"<br>→ *{_cell(r['spoken'])}*"
        ins = f"`({_cell(r['instruction'])})`" if r.get("instruction") else DASH
        url = f"https://huggingface.co/{repo}/resolve/main/samples/{Path(r['wav']).name}"
        lines.append(f"| {_cell(r['title'])} | {text} | {ins} | <audio controls src=\"{url}\"></audio> | "
                     f"{_cell(r.get('pick') or DASH)} |")
    return "\n".join(lines)


def load_extra_samples(path: str | None) -> list[dict]:
    """--extra-samples JSONL: one {title, text, wav, instruction?} per line, wav relative to the file; ValueError on a
    missing field or WAV, or a name clash with another row."""
    if not path:
        return []
    base, rows, seen = Path(path).parent, [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if not all(isinstance(r.get(k), str) and r[k].strip() for k in ("title", "text", "wav")):
            raise ValueError(f"{path}:{n}: want title, text and wav")
        wav = (base / r["wav"]).resolve()
        if not wav.is_file() or wav.name in seen:
            raise ValueError(f"{path}:{n}: {wav} is missing or its name is used twice")
        seen.add(wav.name)
        rows.append({**r, "wav": str(wav)})
    return rows


def render(template: str, values: dict) -> str:
    """{{key}} -> value ("—" for None); unknown placeholders stay, so check_staging() catches them."""
    return re.sub(r"\{\{(\w+)\}\}",
                  lambda m: (DASH if values[m.group(1)] is None else str(values[m.group(1)]))
                  if m.group(1) in values else m.group(0), template)


def staged_extras(out: Path) -> list[Path]:
    """Package files of a staging dir: its root .py / .json files but lora_config.json (EXTRA_SUFFIXES)."""
    if not out.is_dir():
        return []
    return sorted(p for p in out.iterdir() if p.is_file() and p.suffix in EXTRA_SUFFIXES and p.name not in LORA_FILES)


def staged_files(out: Path) -> list[Path]:
    return [p for p in [out / "README.md", *(out / f for f in LORA_FILES)] if p.exists()] + staged_extras(out) + \
        sorted((out / "samples").glob("*.wav"))


def listed_extras(text: str) -> dict[str, str]:
    """{file: sha256} of a card's package file table ({{extra_files}} rows: | `name` | ... | `sha256` |)."""
    rows = re.findall(r"^\| `([^`|]+)` \|.*\| `([0-9a-f]{64})` \|[ \t]*$", text, re.M)
    return {n: h for n, h in rows if Path(n).suffix in EXTRA_SUFFIXES and n not in LORA_FILES}


def extra_problems(out: Path, text: str) -> list[str]:
    """check_staging's package file checks: README.md's table and the staged files agree, every staged .py parses and
    has the tts/voxcpm modules it imports staged too, instructions_tr.json is a {str: str} table."""
    problems, listed = [], listed_extras(text)
    staged = {p.name: p for p in staged_extras(out)}
    for n, h in sorted(listed.items()):
        if n not in staged:
            problems.append(f"README.md lists {n}, which is not staged: stage again")
        elif sha256(staged[n]) != h:
            problems.append(f"{n} differs from the sha256 README.md gives ({h[:16]}...): stage again")
    if extra := sorted(set(staged) - set(listed)):
        problems.append(f"staged package files README.md does not list (delete them or stage again): "
                        f"{', '.join(extra)}")
    for n, p in sorted(staged.items()):
        try:
            if p.suffix == ".py":
                if missing := sorted(m for m in local_imports(p) if f"{m}.py" not in staged):
                    problems.append(f"{n} imports {', '.join(missing)}, not staged: stage again")
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if n == "instructions_tr.json" and not (isinstance(d, dict) and d and all(
                    isinstance(k, str) and k and isinstance(v, str) and v for k, v in d.items())):
                problems.append(f"{n} is not a {{english wording: turkish wording}} table")
        except (SyntaxError, ValueError) as e:  # a .py that does not parse, broken JSON or UTF-8
            problems.append(f"{n} unreadable: {type(e).__name__}: {e}")
    if "instruction_pick.py" in staged and "instructions_tr.json" in staged:
        problems += pick_problems(staged["instruction_pick.py"], staged["instructions_tr.json"])
    return problems


def pick_problems(pick_py: Path, table_json: Path) -> list[str]:
    """instruction_pick.EMOTIONS ({emotion: {"en", "tr"}}) must pair each English wording with its Turkish one in
    instructions_tr.json; unreadable files are extra_problems' to report."""
    try:
        table = json.loads(table_json.read_text(encoding="utf-8"))
        node = _assigned(pick_py, "EMOTIONS")
    except (SyntaxError, ValueError):
        return []
    except RuntimeError as e:
        return [f"instruction_pick.py: {e}"]
    try:
        emotions = ast.literal_eval(node)
        bad = sorted(e for e, w in emotions.items() if not isinstance(table, dict) or table.get(w["en"]) != w["tr"])
    except (ValueError, TypeError, KeyError, AttributeError):
        return ["instruction_pick.py: EMOTIONS is not a {emotion: {\"en\": ..., \"tr\": ...}} table"]
    return [f"instruction_pick.py EMOTIONS {', '.join(bad)}: the Turkish wording is not instructions_tr.json's for "
            f"the English one"] if bad else []


def check_staging(out: Path) -> list[str]:
    """Problems that block an upload (empty list = ready): README.md must describe exactly the staged files."""
    problems = []
    readme = out / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    if not readme.exists():
        problems.append(f"{readme} missing (stage first)")
    elif "{{" in text:
        left = ", ".join(sorted(set(re.findall(r"\{\{[^}]*\}\}", text)))) or "{{"
        problems.append(f"README.md still has placeholders: {left}")
    for f in LORA_FILES:
        if not (out / f).is_file() or (out / f).stat().st_size == 0:
            problems.append(f"{out / f} missing or empty")
    if problems:
        return problems
    if n := text.count(TODO_MARK):  # the other checks still run: the review sees every problem at once
        problems.append(f"README.md still has {n} {TODO_MARK}...) markers: fill them in for the released checkpoint")
    try:
        info = lora_info(out)
    except Exception as e:  # broken header, rank mismatch, bad JSON
        return [*problems, f"LoRA files unusable: {type(e).__name__}: {e}"]
    if not info["lora_r"]:
        problems.append("lora_config.json has no lora_config.r")
    if info["lora_sha256"] not in text:  # e.g. a re-stage that died after copying new weights
        problems.append(f"README.md does not name the staged weights (sha256 {info['lora_sha256'][:16]}...): "
                        f"stage again")
    want = (str(info["lora_r"]), DASH if info["lora_alpha"] is None else str(info["lora_alpha"]))
    if (said := set(re.findall(r"LoRA r=([^,\s]+), alpha=([^;,\s]+)", text))) != {want}:
        problems.append(f"README.md says LoRA r/alpha {', '.join('/'.join(x) for x in sorted(said)) or DASH}, "
                        f"staged lora_config.json r={want[0]}, alpha={want[1]}")
    linked = set(re.findall(r"/resolve/main/samples/([^\"'\s)<>]+\.wav)", text))
    wavs = {p.name for p in (out / "samples").glob("*.wav")}
    if linked - wavs:
        problems.append(f"README.md links samples that are not staged: {', '.join(sorted(linked - wavs))}")
    if wavs - linked:
        problems.append(f"samples/ holds WAVs README.md does not link (delete them or stage again): "
                        f"{', '.join(sorted(wavs - linked))}")
    return problems + extra_problems(out, text)


def stage(args) -> Path:
    lora = Path(args.lora)
    missing = [f for f in LORA_FILES if not (lora / f).is_file()]
    if missing:
        exp = "" if args.exp_name == EXP else f" --exp-name {args.exp_name}"
        raise SystemExit(f"[publish] {lora}: missing {', '.join(missing)}. Fetch them first, from the repo root (reads "
                         f"the Modal Volume only, starts no GPU):\n  python -m modal run tts/voxcpm/modal_app.py "
                         f"--action fetch-ckpt --fetch {lora.name}{exp}")
    info = lora_info(lora)
    evals = [load_eval(s) for s in args.evals]
    ours = evals[0]["summary"]
    measured = (ours.get("lora") or "").replace("\\", "/").rstrip("/")
    want = f"exp/{args.exp_name}/ckpt/{lora.name}"
    if not f"/{measured}".endswith(f"/{want}"):  # the card's numbers must belong to the published weights
        raise SystemExit(f"[publish] first eval dir {evals[0]['dir']} measured lora {ours.get('lora')!r}, "
                         f"not .../{want} (--exp-name {args.exp_name}, --lora {lora})")
    print(f"[publish] weights {lora} (r={info['lora_r']}, sha256 {info['lora_sha256'][:16]}...); card numbers from "
          f"{evals[0]['dir']}, measured with {measured}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").unlink(missing_ok=True)  # written last: a stage that stops halfway leaves no card to upload

    shutil.copyfile(lora / "lora_weights.safetensors", out / "lora_weights.safetensors")
    cfg = json.loads((lora / "lora_config.json").read_text(encoding="utf-8"))
    if cfg.get("base_model") != args.base_model:  # a local training path is useless (and noisy) on the Hub
        print(f"[publish] lora_config.json base_model {cfg.get('base_model')!r} -> {args.base_model!r} "
              f"in the staged copy")
        cfg["base_model"] = args.base_model
        (out / "lora_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        shutil.copyfile(lora / "lora_config.json", out / "lora_config.json")

    if (out / "samples").exists():
        try:
            shutil.rmtree(out / "samples")
        except OSError as e:  # Windows: a WAV still open in a media player
            raise SystemExit(f"[publish] cannot clear {out / 'samples'} ({e}); close whatever plays it and stage again")
    picked = pick_samples(evals[0], args.samples, args.min_s) if args.samples > 0 else []
    if picked:
        (out / "samples").mkdir()
    for r in picked:
        shutil.copyfile(wav_of(evals[0], r), out / "samples" / sample_name(r))
    try:
        extra_rows = load_extra_samples(getattr(args, "extra_samples", None))
    except ValueError as e:
        raise SystemExit(f"[publish] --extra-samples: {e}")
    if extra_rows:
        (out / "samples").mkdir(exist_ok=True)
        clash = {sample_name(r) for r in picked} & {Path(r["wav"]).name for r in extra_rows}
        if clash:
            raise SystemExit(f"[publish] --extra-samples: {', '.join(sorted(clash))} clash with the eval samples")
    for r in extra_rows:
        shutil.copyfile(r["wav"], out / "samples" / Path(r["wav"]).name)
    if len(picked) < args.samples:
        print(f"[publish] only {len(picked)} of {args.samples} samples: WAVs missing under {evals[0]['dir'] / 'wav'}?")
    extras = bundle(getattr(args, "extras", ()))
    for p in staged_extras(out):  # another template's package files
        if p.name not in extras:
            p.unlink()
    for n in extras:
        shutil.copyfile(HERE / n, out / n)
    template = resolve_template(args.template).read_text(encoding="utf-8-sig")

    notes = []
    if any(ev["summary"].get("limit") for ev in evals):
        notes.append("Not / note: \"örneklem\" satırları setlerin yalnızca bir kısmında ölçüldü, tam ölçümle "
                     "doğrudan karşılaştırılamaz / \"subset\" rows were measured on part of each set only and are not "
                     "directly comparable with full runs.")
    if any(ev["dir"].name.split("@")[0] == "trendyol" and ev["summary"].get("utmos") for ev in evals[1:]):
        notes.append("Trendyol-TTS burada kendi sesiyle konuşur (referans klip verilmedi); UTMOS farkının bir kısmı "
                     "modelden değil ses ve kayıttan gelebilir / Trendyol-TTS speaks in its own voice here (no "
                     "reference clip), so part of the UTMOS gap may come from the voice and recording, not the model.")
    note = " ".join(notes)
    values = {**info, "name": args.name, "repo": args.repo, "exp": args.exp_name, "voxcpm_commit": VOXCPM_COMMIT,
              "cfg": float(ours.get("cfg") or DEFAULT_CFG),
              "timesteps": int(ours.get("timesteps") or DEFAULT_TIMESTEPS),
              "rtf": fmt(ours.get("rtf")), "eval_gpu": args.gpu, "attribution": attribution(),
              "results_rows": results_rows(evals, args.name, info["checkpoint"]),
              "category_table": category_table(evals, args.name, info["checkpoint"]),
              "intonation_table": intonation_table(evals, args.name, info["checkpoint"]),
              "samples": samples_md(picked, args.repo, args.min_s), "eval_note": note,
              "extra_files": extras_md(out, extras), "pause_clean_note": pause_note(ours)}
    if "{{extra_samples}}" in template:
        values["extra_samples"] = extra_samples_md(extra_rows, args.repo)
    if "{{extra_samples_table}}" in template:
        values["extra_samples_table"] = extra_samples_table_md(extra_rows, args.repo)
    if extra_rows and not {"extra_samples", "extra_samples_table"} & set(values):
        raise SystemExit(f"[publish] --extra-samples given but {args.template} has no "
                         "{{extra_samples}} or {{extra_samples_table}} placeholder")
    if "{{instructions_tr_table}}" in template:
        values["instructions_tr_table"] = instructions_tr_md()
    card = render(template, values)
    if template_key(args.template) in NO_EM_DASH:
        card = re.sub(r"(?<=\| )—(?= \|)", "-", card)  # a missing table value first, then joins
        card = card.replace(" — ", ", ").replace(DASH, "-")
    (tmp := out / "README.md.tmp").write_text(card, encoding="utf-8", newline="\n")
    os.replace(tmp, out / "README.md")  # only now that the weights and samples it describes are in place

    files = staged_files(out)
    print(f"[publish] staged {out} (nothing uploaded); would upload to {args.repo} (private unless --public):")
    for p in files:
        print(f"  {p.relative_to(out).as_posix():48s} {p.stat().st_size / 1e6:8.2f} MB")
    print(f"  total {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
    problems = check_staging(out)
    for p in problems:
        print(f"[publish] NOT READY: {p}")
    if not problems or all(TODO_MARK in p for p in problems):
        print(f"[publish] review {out / 'README.md'}{' and fill in its ' + TODO_MARK + ') markers' if problems else ''}"
              f", then: python tts/voxcpm/publish_hf.py --repo {args.repo} --out {out} --upload")
    return out


def upload(out: Path, repo: str, public: bool = False) -> str:
    """Upload the staged files of out to the model repo; refuses before any network call when not ready."""
    problems = check_staging(out)
    if not problems and f'snapshot_download("{repo}"' not in (out / "README.md").read_text(encoding="utf-8"):
        problems.append(f"README.md was not staged for {repo} (usage snippet, sample URLs): stage again with --repo")
    if problems:
        raise SystemExit("[publish] refusing to upload:\n  " + "\n  ".join(problems))
    from huggingface_hub import HfApi

    api = HfApi()
    make_public = False
    if api.repo_exists(repo, repo_type="model"):
        private = api.repo_info(repo, repo_type="model").private
        if not private and not public:
            raise SystemExit(f"[publish] {repo} already exists and is PUBLIC; refusing without --public")
        make_public = bool(private and public)
    else:
        api.create_repo(repo, repo_type="model", private=not public)  # explicit: private=None would mean public
    files = staged_files(out)
    print(f"[publish] uploading {len(files)} files ({sum(p.stat().st_size for p in files) / 1e6:.1f} MB) to {repo}")
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(out),
                      allow_patterns=UPLOAD_PATTERNS + [p.name for p in staged_extras(out)],
                      delete_patterns=DELETE_PATTERNS,
                      commit_message="LoRA weights, model card and samples (tts/voxcpm/publish_hf.py)")
    if make_public:
        api.update_repo_settings(repo, private=False, repo_type="model")
    url = f"https://huggingface.co/{repo}"
    print(f"[publish] done: {url} ({'public' if public else 'private'})")
    return url


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="Hugging Face model repo, e.g. ozgurguler/<slug>")
    ap.add_argument("--name", help="display name in the model card (staging)")
    ap.add_argument("--lora", help="checkpoint dir with lora_weights.safetensors + lora_config.json (staging)")
    ap.add_argument("--exp-name", default=EXP, help=f"Modal experiment of the checkpoint (modal_app.py --exp-name); "
                    f"the first eval's summary lora must be .../exp/<this>/ckpt/<--lora dir name> [{EXP}]")
    ap.add_argument("--evals", nargs="+", default=[],
                    help="eval dirs (summary.json, rows.jsonl, wav/): ours first, then references; DIR or DIR=Label")
    ap.add_argument("--samples", type=int, default=6, help="sample WAVs from the first eval dir [6]")
    ap.add_argument("--min-s", type=float, default=2.0, help="prefer sample clips at least this long [2.0]")
    ap.add_argument("--gpu", default="A100 80 GB", help="GPU the RTF numbers were measured on [A100 80 GB]")
    ap.add_argument("--base-model", default="openbmb/VoxCPM2", help="base_model written to lora_config.json")
    ap.add_argument("--template", default=str(TEMPLATE),
                    help="model card template [MODEL_CARD.template.md; two-stage LoRA: MODEL_CARD.two-stage.template.md; "
                         "f022 designed voice: MODEL_CARD.f022.template.md; its s3 release with English + Turkish "
                         "instructions, instruction_pick and pause_clean: MODEL_CARD.f022s3.template.md; the public "
                         "Ahenk card of that release: MODEL_CARD.ahenk.template.md; a short name such as 'ahenk' "
                         "picks tts/voxcpm/MODEL_CARD.<name>.template.md]")
    ap.add_argument("--extra-samples", help="JSONL of hand-picked samples {title, text, wav, instruction?, spoken?, "
                    "pick?} (wav relative to the file), staged as samples/<wav name> and listed at the template's "
                    "{{extra_samples}} (a list) or {{extra_samples_table}} (a table)")
    ap.add_argument("--out", help="staging dir [hf-publish/<repo name>]")
    ap.add_argument("--upload", action="store_true", help="upload the staging dir as it is (no re-staging)")
    ap.add_argument("--public", action="store_true", help="with --upload: public repo (default private)")
    args = ap.parse_args(argv)
    global LORA_LABEL  # reset on every call: a previous run's template must not leak into this card
    if not args.upload:
        args.template = str(resolve_template(args.template))
    key = template_key(args.template)
    LORA_LABEL = TEMPLATE_LABELS.get(key, DEFAULT_LABEL)
    args.extras = TEMPLATE_EXTRAS.get(key, ())
    if args.out is None:
        args.out = str(REPO / "hf-publish" / args.repo.split("/")[-1])
    if args.upload:
        if args.name or args.lora or args.evals:
            print("[publish] --upload sends the staging dir as it is; --name/--lora/--evals are ignored")
        upload(Path(args.out), args.repo, args.public)
        return 0
    if args.public:
        ap.error("--public only makes sense with --upload")
    if not (args.name and args.lora and args.evals):
        ap.error("staging needs --name, --lora and --evals")
    stage(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
