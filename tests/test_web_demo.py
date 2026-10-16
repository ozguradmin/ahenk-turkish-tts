"""site/ (the in-browser Ahenk Cep demo) and its Python side: tts/small/web_export.py, web_pipeline.py, web_parity.py.

Stdlib-only checks always run: the vendored trnorm copy equals src/trnorm, the Pyodide file list matches site/py,
every CDN URL is pinned to an exact version, the page has no em dashes, the models stay out of git, and
site/py/ahenk_front.py gives exactly what tts/small/ahenk_cep.py feeds the model. The JavaScript checks (noise,
frame timeline, decoder windows) run where `node` is installed; the torch / onnxruntime checks where those are
installed and site/models has been exported (python tts/small/web_export.py)."""

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"
SMALL = REPO / "tts" / "small"
sys.path[:0] = [str(SITE / "py"), str(SMALL), str(REPO / "src")]

import ahenk_front as F  # noqa: E402

VOCAB = ["<pad>", "<unk>", " ", "!", '"', "%", "&", "'", "(", ")", ",", "-", ".", "/", ":", ";", "?",
         *"abcdefghijklmnopqrstuvwxyz", *"çöüğış"]
SETS = REPO / "benchmarks" / "tts" / "sets"


def set_texts(limit=None):
    out = []
    for p in sorted(SETS.glob("*.jsonl")):
        out += [json.loads(x)["text"] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return out[:limit]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- the static site ----------------------------------------------------------------------------------------------
def test_vendored_trnorm_is_src_trnorm():
    vendored = sorted(p.name for p in (SITE / "py" / "trnorm").glob("*.py"))
    expected = sorted(p.name for p in (REPO / "src" / "trnorm").glob("*.py") if p.name != "cli.py")
    assert vendored == expected
    for name in vendored:
        assert (SITE / "py" / "trnorm" / name).read_bytes() == (REPO / "src" / "trnorm" / name).read_bytes(), (
            f"site/py/trnorm/{name} differs from src/trnorm/{name}: copy it again")


def test_pyodide_file_list_matches_site_py():
    js = (SITE / "js" / "front.js").read_text(encoding="utf-8")
    listed = re.findall(r"'([\w/]+\.py)'", js[js.index("PY_FILES"):js.index("];")])
    on_disk = sorted(p.relative_to(SITE / "py").as_posix() for p in (SITE / "py").rglob("*.py")
                     if "__pycache__" not in p.parts)
    assert sorted(listed) == on_disk


def test_cdn_urls_are_pinned_and_allowed():
    text = "\n".join(p.read_text(encoding="utf-8") for p in [SITE / "index.html", *(SITE / "js").glob("*.js")])
    urls = re.findall(r"https://cdn\.jsdelivr\.net/npm/[^'\"`\s)]+", text)
    assert urls, "the page loads onnxruntime-web and pyodide from jsdelivr"
    for u in urls:
        assert re.search(r"@(\d+\.\d+\.\d+|\$\{VERSIONS\.\w+\})/", u), u
    cfg = (SITE / "js" / "config.js").read_text(encoding="utf-8")
    versions = dict(re.findall(r"(\w+): '(\d+\.\d+\.\d+)'", cfg[cfg.index("VERSIONS"):cfg.index("};")]))
    assert versions == {"ort": "1.30.0", "pyodide": "0.27.7"}
    hosts = set(re.findall(r"https://([\w.-]+)/", text))
    assert hosts <= {"cdn.jsdelivr.net", "huggingface.co", "github.com"}, hosts


def test_page_text_has_no_em_dash_and_keeps_the_notices():
    for p in [SITE / "index.html", SITE / "js" / "app.js", SITE / "js" / "player.js", SITE / "css" / "style.css",
              SITE / "README.md"]:
        assert "—" not in p.read_text(encoding="utf-8"), p
    app = (SITE / "js" / "app.js").read_text(encoding="utf-8")
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert "Bu ses yapaydır." in html and "Bu ses yapaydır." in app and "This voice is synthetic." in app
    assert "destek@ozgurguler.tech" in html and "ozguradmin/ahenk-turkish-tts" in html
    assert "ozgurguler/ahenk-cep-tr" in html


def test_no_emotion_selector_and_the_neutral_note():
    """The small model's styles are too weak to offer: the page always asks for neutral and says where emotions are."""
    app = (SITE / "js" / "app.js").read_text(encoding="utf-8")
    html = (SITE / "index.html").read_text(encoding="utf-8")
    for text in (app, html):
        assert 'name="style"' not in text and "STYLES" not in text and "styleLabel" not in text
    assert "const STYLE = 'neutral';" in app and "style: STYLE" in app
    note = "Ahenk Cep şimdilik nötr konuşur; duygulu konuşma (kızgın, neşeli, şaşkın, gülme...) büyük model Ahenk'te var"
    assert note in html and note in app
    assert "https://huggingface.co/ozgurguler/ahenk-tts-tr" in html
    assert "https://huggingface.co/ozgurguler/ahenk-tts-tr/tree/main/samples" in html


SAMPLES = SITE / "samples"


def test_pre_rendered_examples_are_small_committed_and_match_trnorm():
    from trnorm import normalize

    meta = json.loads((SAMPLES / "samples.json").read_text(encoding="utf-8"))
    assert 5 <= len(meta["examples"]) <= 8 and meta["seed"] == 0
    total = (SAMPLES / "samples.json").stat().st_size
    for ex in meta["examples"]:
        for key in ("ogg", "mp3"):
            f = SAMPLES / ex[key]
            assert f.is_file(), f
            total += f.stat().st_size
            r = subprocess.run(["git", "check-ignore", "-q", f"site/samples/{ex[key]}"], cwd=REPO)
            assert r.returncode == 1, f"{ex[key]} is gitignored"
        assert ex["spoken"] == normalize(ex["text"]), ex["id"]
        assert len(ex["peaks"]) == 160 and max(ex["peaks"]) == 1.0 and ex["seconds"] > 0.5
        assert ex["tr"] and ex["en"]
    assert total < 1_500_000, total
    kinds = " ".join(ex["id"] for ex in meta["examples"])
    for kind in ("tarih", "soru", "telefon", "para", "uzun"):
        assert kind in kinds
    big = {b["file"] for b in meta["big"]}
    assert big == {"big_11_laughing.wav", "big_07_cheerful.wav", "big_09_calm.wav"}
    r = subprocess.run(["git", "check-ignore", "-q", "site/samples/big/big_09_calm.wav"], cwd=REPO)
    assert r.returncode == 0, "the local copy of the big model's samples must stay out of git"
    cfg = (SITE / "js" / "config.js").read_text(encoding="utf-8")
    assert "https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/" in cfg


def test_models_stay_out_of_git():
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "site/models/" in ignore and "*.onnx" in ignore
    r = subprocess.run(["git", "check-ignore", "-q", "site/models/step.onnx"], cwd=REPO)
    assert r.returncode == 0


def test_model_base_rules_in_config():
    cfg = (SITE / "js" / "config.js").read_text(encoding="utf-8")
    assert "https://huggingface.co/ozgurguler/ahenk-cep-tr/resolve/main/onnx/" in cfg
    assert "local: 'models/'" in cfg and "localhost" in cfg


def test_service_worker_sits_at_the_site_root():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert '<script src="coi-serviceworker.js"></script>' in html  # scope "/" needs the file at the root
    assert (SITE / "coi-serviceworker.js").is_file() and (SITE / "coi-serviceworker.LICENSE").is_file()


# ---- the text front-end (site/py/ahenk_front.py) ------------------------------------------------------------------
def test_prepare_shapes_and_bounds():
    out = json.loads(F.prepare("Randevunuz 14.10.2026 saat 09:30'da, 3. katta.", 1.0, VOCAB))
    assert out["spoken"].startswith("Randevunuz on dört Ekim")
    (p,) = out["pieces"]
    assert p["text"] == "randevunuz on dört ekim iki bin yirmi altı saat dokuz otuzda, üçüncü katta."
    assert len(p["ids"]) == len(p["cw"]) == len(p["wstart"]) == len(p["text"])
    stoi = {c: i for i, c in enumerate(VOCAB)}
    assert p["ids"] == [stoi[c] for c in p["text"]] and p["pause"] == 0.0
    for i, w in enumerate(p["cw"]):
        assert p["text"][p["wstart"][i]] != " " or i == 0
        assert p["cw"][p["wstart"][i]] == w


def test_prepare_edges():
    assert json.loads(F.prepare("", 1.0, VOCAB))["pieces"] == []
    assert json.loads(F.prepare("123", 1.0, VOCAB))["pieces"][0]["text"] == "yüz yirmi üç."
    assert json.loads(F.prepare("...", 1.0, VOCAB))["pieces"] == []
    long = json.loads(F.prepare("Bu cümle uzun, " * 30 + "bitti.", 1.0, VOCAB))["pieces"]
    assert len(long) > 1 and all(len(p["text"]) <= 180 for p in long)
    assert all(p["text"][-1] in ".!?" for p in long) and long[-1]["pause"] == 0.0 and long[0]["pause"] > 0


def _engine_piece(text):
    starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
    bounds = [0] + starts[1:] + [len(text)]
    cw, wstart = [], []
    for w, (a, b) in enumerate(pairwise(bounds)):
        cw += [w] * (b - a)
        wstart += [a] * (b - a)
    return cw, wstart


def test_front_matches_ema_lightning():
    pytest.importorskip("torch")
    ema = pytest.importorskip("ema_lightning")
    from ema_lightning.chunker import chunk
    from ema_lightning.frontend import Frontend
    from trnorm import normalize

    assert ema.__version__ == "1.1.2"
    fe = Frontend.__new__(Frontend)
    fe.vocab = frozenset(VOCAB)
    fe.spoken = lambda t: t
    texts = set_texts() + ["Çok güzel 😀 bir gün!", "naïve café", "Iğdır İSTANBUL", "Bu cümle uzun, " * 30]
    for t in texts:
        ref = chunk(fe(normalize(t)), 1.0)
        got = json.loads(F.prepare(t, 1.0, VOCAB))["pieces"]
        assert [(p["text"], p["pause"]) for p in got] == ref, t
        for p in got:
            assert (p["cw"], p["wstart"]) == _engine_piece(p["text"])


def test_front_speed_changes_the_cut():
    t = "Bu cümle uzun, " * 12 + "bitti."
    slow = json.loads(F.prepare(t, 0.7, VOCAB))["pieces"]
    fast = json.loads(F.prepare(t, 1.4, VOCAB))["pieces"]
    assert len(slow) > len(fast)


# ---- noise, timeline, windows (Python reference) ------------------------------------------------------------------
np = pytest.importorskip("numpy")
W = _load("web_pipeline", SMALL / "web_pipeline.py")


def test_mt19937_matches_the_standard():
    g = W.MT19937(5489)  # std::mt19937's default seed; the 10000th output is fixed by the C++ standard
    for _ in range(9999):
        g.next_u32()
    assert g.next_u32() == 4123659995


def test_plan_rounds_half_to_even_and_clamps():
    dur = np.array([0.5, 1.0, 0.5, 1.0, 1.5, 300.0], np.float32)
    cw = np.array([0, 0, 1, 2, 2, 3])
    d, fw, fp = W.plan(dur, cw, 1.0)
    assert np.bincount(fw).tolist() == [2, 1, 2, 250]  # 1.5 -> 2, 0.5 -> 0 -> clamp 1, 2.5 -> 2, 300 -> 250
    assert fp[:2].tolist() == [0.0, 0.5] and fp.dtype == np.float32
    d2, fw2, _ = W.plan(dur, cw, 2.0)
    assert np.allclose(d2, dur / 2) and np.bincount(fw2).tolist() == [1, 1, 1, 150]


def test_windows_cover_every_frame_once():
    for n in (1, 99, 100, 101, 250, 3000):
        spans = W.windows(n)
        assert spans[0][0] == 0 and spans[-1][1] == n and all(b == c for (_, b), (c, _) in pairwise(spans))
        assert all(e - s <= 100 for s, e in spans)


def test_piece_seed_matches_ema():
    assert W.piece_seed(0, 0) == 0 and W.piece_seed(3, 2) == 3 * 1_000_003 + 2
    assert W.piece_seed(2 ** 62, 1) == (2 ** 62 * 1_000_003 + 1) % 2 ** 63


def test_torch_randn_port():
    torch = pytest.importorskip("torch")
    for seed in (0, 7, 2 ** 40 + 7):
        for frames in (1, 37):
            ref = torch.randn((4, frames, 64), generator=torch.Generator().manual_seed(seed))[0].reshape(-1).numpy()
            assert np.abs(W.torch_randn(seed, ref.size) - ref).max() < 1e-5


# ---- the page's JavaScript, in node ---------------------------------------------------------------------------------
NODE = shutil.which("node")


def run_node(code: str):
    script = "import { pathToFileURL } from 'node:url';\n" + code
    r = subprocess.run([NODE, "--input-type=module", "-e", script], cwd=SITE, capture_output=True, text=True,
                       encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_randn_and_mt_match_python():
    out = run_node(
        "const m = await import(pathToFileURL('js/randn.js').href);\n"
        "const g = new m.MT19937(5489); let v; for (let i = 0; i < 10000; i++) v = g.next();\n"
        "console.log(JSON.stringify({ v, a: Array.from(m.torchRandn(7n, 64 * 5)),"
        " b: Array.from(m.torchRandn(2n ** 40n + 7n, 64)), s: String(m.pieceSeed(2n ** 62n, 1)) }));")
    assert out["v"] == 4123659995
    assert np.abs(np.array(out["a"], np.float32) - W.torch_randn(7, 64 * 5)).max() < 1e-6
    assert np.abs(np.array(out["b"], np.float32) - W.torch_randn(2 ** 40 + 7, 64)).max() < 1e-6
    assert int(out["s"]) == W.piece_seed(2 ** 62, 1)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_plan_and_windows_match_python():
    rng = np.random.default_rng(0)
    dur = (rng.random(40) * 6).astype(np.float32)
    dur[:4] = [0.5, 1.5, 2.5, 0.25]
    cw = np.sort(rng.integers(0, 9, 40))
    cw = np.unique(cw, return_inverse=True)[1]
    out = run_node(
        "const e = await import(pathToFileURL('js/engine.js').href);\n"
        f"const dur = Float32Array.from({json.dumps(dur.tolist())}); const cw = {json.dumps(cw.tolist())};\n"
        "const r = [1.0, 0.7, 1.35].map((s) => { const p = e.plan(dur, cw, s);"
        " return { dur: Array.from(p.dur), fw: Array.from(p.fw, Number), fp: Array.from(p.fp) }; });\n"
        "const w = [1, 100, 101, 250].map((n) => e.windows(n));\n"
        "const rh = [0.5, 1.5, 2.5, 2.4999, 3.5001].map(e.roundHalfEven);\n"
        "console.log(JSON.stringify({ r, w, rh }));")
    for res, speed in zip(out["r"], (1.0, 0.7, 1.35), strict=True):
        d, fw, fp = W.plan(dur, cw, speed)
        assert np.array_equal(np.array(res["dur"], np.float32), d)
        assert res["fw"] == fw.tolist() and np.array_equal(np.array(res["fp"], np.float32), fp)
    assert out["w"] == [[list(s) for s in W.windows(n)] for n in (1, 100, 101, 250)]
    assert out["rh"] == [0, 2, 2, 2, 4]


# ---- ONNX export and the end-to-end pipeline -----------------------------------------------------------------------
def test_subpixel_up_equals_conv_transpose():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ema_lightning")
    export = _load("web_export", SMALL / "web_export.py")
    for ci, co, k, s in [(256, 128, 16, 8), (128, 64, 12, 6), (64, 32, 10, 5), (16, 8, 4, 2)]:
        up = torch.nn.ConvTranspose1d(ci, co, k, s, padding=(k - s) // 2)
        x = torch.randn(1, ci, 23)
        with torch.no_grad():
            assert (up(x) - export.SubpixelUp(up)(x)).abs().max() < 1e-5
    with pytest.raises(ValueError):
        export.SubpixelUp(torch.nn.ConvTranspose1d(4, 4, 3, 2))


MODELS = SITE / "models"
WEIGHTS = REPO / "hf-publish" / "ahenk-cep-tr"


@pytest.mark.skipif(not (MODELS / "ahenk-cep.json").is_file() or not (WEIGHTS / "student.pt").is_file(),
                    reason="needs site/models (tts/small/web_export.py) and the weights")
def test_onnx_pipeline_matches_pytorch():
    pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    pytest.importorskip("ema_lightning")
    from ahenk_cep import AhenkCep

    manifest = json.loads((MODELS / "ahenk-cep.json").read_text(encoding="utf-8"))
    assert manifest["vocab"] == VOCAB and manifest["hop"] == 1920 and manifest["sample_rate"] == 48000
    torch.set_num_threads(4)
    pt = AhenkCep(WEIGHTS / "student.pt", WEIGHTS / "decoder.pt")
    pt.engine._batch_size = 1
    ox = W.OnnxCep(MODELS, "fp32", threads=4)
    for text, style, seed in [("Bu ürünün garantisi var mı?", "neutral", 0), ("Saat 09:30'da gel.", "sad", 3)]:
        a, _ = pt.say(text, style, seed=seed)
        b, _ = ox.say(text, style, seed=seed)
        assert len(a) == len(b)
        assert 10 * np.log10((a ** 2).sum() / ((a - b) ** 2).sum()) > 60  # dB


@pytest.mark.skipif(not (MODELS / "ahenk-cep.json").is_file(), reason="needs site/models")
def test_manifest_lists_every_file_with_its_checksum():
    import hashlib

    manifest = json.loads((MODELS / "ahenk-cep.json").read_text(encoding="utf-8"))
    for graph in ("text", "cond", "step", "decoder"):
        kinds = manifest["files"][graph]
        assert {"fp32", "f16"} <= set(kinds)
        for f in kinds.values():
            p = MODELS / f["file"]
            assert p.stat().st_size == f["bytes"]
            assert hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"]
