#!/usr/bin/env python3
"""Blind listening page for the small model (P1): per sentence, the student vs EMA Lightning vs Antalia-2 Mini vs the
teacher, in random order, each clip under a code that is new for every sentence and a random file name; loudness
normalised; the listener picks the best clip per sentence and copies the result text back.

    python tts/small/listen_page.py --bench voxcpm-contest/small-bench --out voxcpm-contest/small-bench/dinleme \
        --systems st-final ema antalia teacher

The key (code -> system per sentence) goes to OUT/../dinleme-anahtar.json, outside the page folder, so the page
itself reveals nothing; decode a pasted result with --decode "<pasted text>".
Loudness: ITU-R BS.1770-4 integrated loudness (K-weighting at 48 kHz, 400 ms blocks with 75 % overlap, absolute gate
-70 LUFS, relative gate -10 LU), every clip set to --lufs (default -20) with its sample peak kept under -1 dBFS.
Needs numpy, scipy, soundfile, soxr (no torch).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SR = 48000
# 10 sentences of the P0/P1 benchmark subset: 3 plain, 3 with numbers, 2 questions, 2 long (held-out)
SENTENCES = [("trzor", "ozel_04"), ("trzor", "ozel_08"), ("trzor", "ozel_10"),
             ("trzor", "sayi_13"), ("trzor", "para_09"), ("trzor", "tarih_05"),
             ("trzor", "soru_07"), ("trzor", "soru_03"),
             ("heldout", "heldout_d4d2e9f7f74f4e83"), ("heldout", "heldout_6df62be70e7fa49d")]
CODE_CHARS = "ABCDEFGHJKLMNPRSTUVYZ"  # no I/O/Q/W/X (confusable or not Turkish)


def k_weight(x, sr: int = SR):
    """BS.1770 K-weighting (pre-filter + RLB high-pass), the 48 kHz coefficients of the standard."""
    from scipy.signal import lfilter

    if sr != 48000:
        raise ValueError("K-weighting coefficients are for 48 kHz; resample first")
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    return lfilter(b2, a2, lfilter(b1, a1, x))


def lufs(x, sr: int = SR) -> float:
    """Integrated loudness (mono) in LUFS; -inf for silence or a clip shorter than one 400 ms block."""
    import numpy as np

    y = k_weight(np.asarray(x, dtype=np.float64), sr)
    n, hop = int(0.4 * sr), int(0.1 * sr)
    if len(y) < n:
        return float("-inf")
    ms = np.array([np.mean(y[i:i + n] ** 2) for i in range(0, len(y) - n + 1, hop)])
    lk = -0.691 + 10 * np.log10(np.maximum(ms, 1e-20))
    keep = lk > -70
    if not keep.any():
        return float("-inf")
    rel = -0.691 + 10 * np.log10(ms[keep].mean()) - 10
    keep &= lk > rel
    return float(-0.691 + 10 * np.log10(ms[keep].mean()))


def normalize(x, sr: int = SR, target: float = -20.0, peak_db: float = -1.0):
    """x scaled to `target` LUFS, then down (never up) so that its sample peak is <= peak_db dBFS."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    L = lufs(x, sr)
    g = 1.0 if not np.isfinite(L) else 10 ** ((target - L) / 20)
    y = x * g
    pk = np.abs(y).max() if len(y) else 0.0
    lim = 10 ** (peak_db / 20)
    if pk > lim:
        y *= lim / pk
    return y.astype(np.float32)


def load_48k(path):
    import numpy as np
    import soundfile as sf
    import soxr

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = soxr.resample(x, sr, SR, quality="VHQ")
    return np.ascontiguousarray(x, dtype=np.float32)


def assign(sentences: list[dict], systems: list[str], rng: random.Random) -> list[dict]:
    """Per sentence: the systems in random order, each with a 2-letter code unique on the page and a random file
    name (s/<12 hex>.wav). sentences: [{key, text}] -> [{key, text, clips: [{code, system, file}]}]."""
    used, out = set(), []
    for s in sentences:
        clips = []
        for sys_ in rng.sample(systems, len(systems)):
            while True:
                code = rng.choice(CODE_CHARS) + rng.choice("23456789")
                if code not in used:
                    used.add(code)
                    break
            clips.append({"code": code, "system": sys_, "file": f"s/{rng.getrandbits(48):012x}.wav"})
        out.append({**s, "clips": clips})
    return out


def page_html(items: list[dict], title: str) -> str:
    """The page: one sentence at a time, 'En iyisi bu' per clip, optional comment, copyable result. No system names."""
    data = json.dumps([{"text": it["text"], "clips": [[c["code"], c["file"]] for c in it["clips"]]} for it in items],
                      ensure_ascii=False)
    return f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{title}</title>
<style>body{{font-family:sans-serif;max-width:760px;margin:auto;padding:16px;background:#111;color:#eee}}.r{{display:flex;align-items:center;gap:10px;background:#1d1d1d;margin:8px 0;padding:8px;border-radius:8px}}
.r audio{{flex:1;min-width:0}}button{{font-size:16px;padding:8px 14px;border-radius:8px;border:0;background:#3a6;color:#fff;cursor:pointer}}textarea{{width:100%;box-sizing:border-box;height:60px;margin-top:10px;background:#1d1d1d;color:#eee}}
.t{{font-size:20px;margin:14px 0}}.k{{color:#aaa}}pre{{white-space:pre-wrap}}</style>
<h2>{title}</h2><div class=k>Her cümlede birkaç ses var, sıraları ve adları her cümlede değişiyor. Hepsini dinle, en iyi bulduğunun yanındaki düğmeye bas. Ses düzeyleri eşitlendi.</div><div class=k id=n></div><div class=t id=t></div><div id=L></div>
<textarea id=y placeholder="İstersen yorum yaz, sonra 'En iyisi bu'ya bas (ör: 2. ses robotik, 3. seste kelime yanlış...)"></textarea><button id=c style="display:none" onclick="navigator.clipboard.writeText(o.textContent)">Kopyala</button><pre id=o></pre>
<script>const T={data};const NL=String.fromCharCode(10);let i=0,res=[];
function show(){{if(i>=T.length){{done();return}}const x=T[i];n.textContent="Cümle "+(i+1)+" / "+T.length;t.textContent="“"+x.text+"”";
L.innerHTML=x.clips.map((c,k)=>'<div class=r><b>'+(k+1)+'</b><audio controls preload=none src="'+c[1]+'"></audio><button onclick="pick('+k+')">En iyisi bu</button></div>').join("");y.value=""}}
function pick(k){{const x=T[i];res.push([x.clips[k][0],x.clips.map(c=>c[0]).join(","),y.value.trim()]);i++;show()}}
function done(){{L.innerHTML="";y.style.display="none";c.style.display="";t.textContent="Bitti! Aşağıdaki yazıyı kopyalayıp bana yapıştır.";
let s="KUCUK MODEL DINLEME: "+res.length+" cümle"+NL;res.forEach((r,j)=>{{s+=(j+1)+": "+r[0]+"  ["+r[1]+"]"+(r[2]?"  yorum: "+r[2]:"")+NL}});o.textContent=s}}
show()</script>
"""


def decode(text: str, key: dict) -> dict:
    """Pasted result -> wins per system (each line 'N: CODE  [CODES]  yorum: ...')."""
    code2sys = {c["code"]: c["system"] for it in key["items"] for c in it["clips"]}
    wins: dict[str, int] = {}
    picks = []
    for m in re.finditer(r"^\s*(\d+):\s*([A-Z][2-9])\b(.*)$", text, re.M):
        s = code2sys.get(m.group(2))
        if s is None:
            continue
        wins[s] = wins.get(s, 0) + 1
        picks.append({"n": int(m.group(1)), "system": s, "rest": m.group(3).strip()})
    return {"wins": dict(sorted(wins.items(), key=lambda kv: -kv[1])), "picks": picks}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="voxcpm-contest/small-bench")
    ap.add_argument("--out", default="voxcpm-contest/small-bench/dinleme")
    ap.add_argument("--systems", nargs="+", default=["st-final", "ema", "antalia", "teacher"])
    ap.add_argument("--lufs", type=float, default=-20.0)
    ap.add_argument("--seed", type=int, default=None, help="default: a fresh random seed (stored in the key)")
    ap.add_argument("--title", default="Küçük model: hangisi daha iyi?")
    ap.add_argument("--decode", default=None, help="a pasted result text: print wins per system and exit")
    args = ap.parse_args(argv)
    bench = REPO / args.bench
    out = REPO / args.out
    keyp = out.parent / f"{out.name}-anahtar.json"
    if args.decode is not None:
        print(json.dumps(decode(args.decode, json.loads(keyp.read_text(encoding="utf-8"))), ensure_ascii=False,
                         indent=1))
        return 0
    import soundfile as sf

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    subset = {(r["set"], r["id"]): r for r in map(json.loads, (bench / "subset.jsonl").read_text(
        encoding="utf-8").splitlines())}
    sentences = [{"key": list(k), "text": subset[k]["text"]} for k in SENTENCES]
    items = assign(sentences, args.systems, random.Random(seed))
    (out / "s").mkdir(parents=True, exist_ok=True)
    for old in (out / "s").glob("*.wav"):
        old.unlink()
    levels = []
    for it in items:
        for c in it["clips"]:
            x = load_48k(bench / c["system"] / it["key"][0] / f"{it['key'][1]}.wav")
            y = normalize(x, SR, args.lufs)
            sf.write(str(out / c["file"]), y, SR, subtype="PCM_16")
            c["lufs_in"] = round(lufs(x), 2)
            levels.append(c["lufs_in"])
    (out / "index.html").write_text(page_html(items, args.title), encoding="utf-8")
    keyp.write_text(json.dumps({"seed": seed, "systems": args.systems, "lufs": args.lufs, "items": items},
                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{out / 'index.html'}: {len(items)} sentences x {len(args.systems)} clips; key {keyp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
