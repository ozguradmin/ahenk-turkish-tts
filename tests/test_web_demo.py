"""site/ (the in-browser Ahenk Cep demo) and its Python side: tts/small/web_export.py, web_pipeline.py, web_parity.py.

Stdlib-only checks always run: the vendored trnorm copy equals src/trnorm, the Pyodide file list matches site/py,
every CDN URL is pinned to an exact version, the page has no em dashes, the models stay out of git and every file the
page imports is in it, the service worker is the reviewed upstream file set to require-corp, the page survives a
worker that fails or hangs, fonts do not block the first paint, and site/py/ahenk_front.py gives exactly what
tts/small/ahenk_cep.py feeds the model. The JavaScript checks (noise, frame timeline, decoder windows, backend choice,
WebGPU and thread fallbacks) run where `node` is installed; the torch / onnxruntime checks where those are installed
and site/models has been exported (python tts/small/web_export.py)."""

import hashlib
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
    # the site itself (og:url and og:image for link previews): GitHub Pages, with a copy on Cloudflare Pages
    assert hosts <= {"cdn.jsdelivr.net", "huggingface.co", "github.com", "ozguradmin.github.io", "ahenk-tts.pages.dev"}, hosts


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


def test_every_local_file_the_page_loads_is_in_git():
    """A new module left out of a commit would ship a page whose app.js cannot load."""
    html = (SITE / "index.html").read_text(encoding="utf-8")
    wanted = set(re.findall(r'<(?:script|link)[^>]* (?:src|href)="([^":#]+)"', html))
    for js in (SITE / "js").glob("*.js"):
        text = js.read_text(encoding="utf-8")
        for m in re.findall(r"from '\./([\w.-]+\.js)'|new URL\('\./([\w.-]+\.js)', import\.meta\.url\)", text):
            wanted.add(f"js/{m[0] or m[1]}")
    assert {"js/app.js", "js/backend.js", "js/worker.js", "coi-serviceworker.js", "css/style.css"} <= wanted, wanted
    for rel in sorted(wanted):
        assert (SITE / rel).is_file(), rel
    if shutil.which("git") and (REPO / ".git").exists():
        r = subprocess.run(["git", "ls-files", "--error-unmatch", *(f"site/{rel}" for rel in sorted(wanted))], cwd=REPO,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"not in git (git add it): {r.stderr.strip()}"


# gzuidhof/coi-serviceworker at 509a799 (master, 2023-12-06; its header still says v0.1.7), read before vendoring
COI_SHA256 = "e97bbac6017322d48aa54a5bc7ce473c725a4b7376ff245002e0ba71c3dcdd7e"


def test_service_worker_forces_require_corp_and_reloads_at_most_once():
    """npm's 0.1.7 picked COEP credentialless in every browser without window.chrome (all of iOS), which Safari does
    not support, and could reload in a loop. The page now uses upstream's newer file with require-corp everywhere."""
    sw = (SITE / "coi-serviceworker.js").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(sw).hexdigest() == COI_SHA256, "coi-serviceworker.js changed: review it, then update the hash"
    text = sw.decode()
    assert 'sessionStorage.setItem("coiReloadedBySelf"' in text and "shouldRegister: () => !reloadedBySelf" in text
    html = (SITE / "index.html").read_text(encoding="utf-8")
    cfg = html[html.index("window.coi = {"):html.index('<script src="coi-serviceworker.js">')]
    assert "coepCredentialless: () => false" in cfg and "coepDegrade: () => false" in cfg
    assert "shouldRegister" not in cfg, "keep upstream's default: it is the reload guard"
    assert "coiReloadedBySelf" in cfg  # read before the service worker script removes it (js/app.js uses it)
    # its reload comes while the service worker installs: wait until it is active, or the reloaded page is not isolated
    assert "navigator.serviceWorker.ready" in cfg[cfg.index("doReload"):]


def test_page_survives_a_worker_that_fails_or_hangs():
    app = (SITE / "js" / "app.js").read_text(encoding="utf-8")
    assert "w.onerror = " in app and "loadFailed('worker'" in app  # a worker script or import that cannot load
    assert "STALL_MS" in app and "loadFailed('timeout'" in app  # a load that stops making progress
    assert "OPEN_MS" in app and "post({ type: 'alive' })" in (SITE / "js" / "worker.js").read_text(encoding="utf-8")
    # threads that fail get one more try on one thread, once; WebGPU that failed is not tried again in this tab
    assert "threads > 1 && !avoid.threads" in app and "maxThreads: avoid.threads ? 1 : 0" in app
    assert "avoid.webgpu ? 'wasm'" in app
    # js/app.js itself failing to load is not silent either
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert "window.ahenkStarted = true;" in app and "if (window.ahenkStarted) return;" in html
    retry = app[app.index("$('retry').addEventListener"):]
    retry = retry[:retry.index("});")]
    assert "startWorker()" in retry
    assert "location.reload" not in app, "restart the worker, not the page (phones lose memory on every reload)"
    # short human messages, the technical line folded away in a <details>; never the raw onnxruntime dump
    assert "el('details'" in app and "loadErrorHint" not in app
    for key in ("errNetwork", "errEngine", "errEngineIos", "errMemory", "errTimeout", "errWorker", "tapPlay"):
        assert app.count(f"    {key}: ") == 2, key  # Turkish and English
    assert "mainPlayer.unlock();" in app[app.index("function speak()"):app.index("function finish(")]
    worker = (SITE / "js" / "worker.js").read_text(encoding="utf-8")
    assert "openWithFallback(gpu, open" in worker and "pickBackend(" in worker and "toWasm" in worker


def test_fonts_do_not_block_the_first_paint():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    head = re.sub(r"<noscript>.*?</noscript>", "", html[:html.index("</head>")], flags=re.S)
    assert not re.search(r'<link rel="stylesheet" href="https://', head)
    assert head.count('rel="preload" as="style"') == 3 and head.count("this.rel='stylesheet'") == 3
    css = (SITE / "css" / "style.css").read_text(encoding="utf-8")
    # iOS zooms into a focused field under 16px
    assert re.search(r'@media \(max-width: 560px\), \(pointer: coarse\) \{\s*\.field select, '
                     r'\.field input\[type="number"\] \{ font-size: 16px; \}', css)


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


PYTHON_ERROR = ('Traceback (most recent call last):\n  File "/lib/python312.zip/_pyodide/_base.py", line 523, in '
                'eval_code\n    .run(globals, locals)\nValueError: could not read this text')

IPHONE_ERROR = (  # what the page showed on an iPhone (X's in-app browser, 2026-10-17), cut short
    "failed to call OrtRun(). ERROR_CODE: 1, ERROR_MESSAGE: /mnt/azure_nvme_temp/_work/1/s/onnxruntime/core/providers/"
    "webgpu/buffer_manager.cc:562 WGPUBuffer onnxruntime::webgpu::BufferManager::Create(size_t, wgpu::BufferUsage, "
    "bool) const [ONNXRuntimeError] : 1 : FAIL : Failed to create a WebGPU compute pipeline: Failed to compile the "
    "shader source, generated metal: #include <metal_stdlib>" + "\nusing namespace metal;" * 300)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_backend_choice_and_webgpu_fallback():
    navs = {
        "iphone": {"userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, "
                                "like Gecko) Mobile/15E148 Twitter for iPhone", "vendor": "Apple Computer, Inc.",
                   "maxTouchPoints": 5},
        "ipad": {"userAgent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)"
                              " Version/18.6 Safari/605.1.15", "vendor": "Apple Computer, Inc.", "maxTouchPoints": 5},
        "mac_safari": {"userAgent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like "
                                    "Gecko) Version/18.6 Safari/605.1.15", "vendor": "Apple Computer, Inc.",
                       "maxTouchPoints": 0},
        "android": {"userAgent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/141.0.0.0 Mobile Safari/537.36", "vendor": "Google Inc.", "maxTouchPoints": 5,
                    "userAgentData": {"mobile": True}},
        "desktop": {"userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/141.0.0.0 Safari/537.36", "vendor": "Google Inc.", "maxTouchPoints": 0,
                    "userAgentData": {"mobile": False}},
        # Chrome's "desktop site" on an Android tablet, and a touch-screen laptop with a mouse (pointer: fine)
        "android_desktop": {"userAgent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                         "Chrome/141.0.0.0 Safari/537.36", "vendor": "Google Inc.", "maxTouchPoints": 5,
                            "userAgentData": {"mobile": False}, "coarse": True},
        "touch_laptop": {"userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/141.0.0.0 Safari/537.36", "vendor": "Google Inc.", "maxTouchPoints": 10,
                         "userAgentData": {"mobile": False}, "coarse": False},
    }
    out = run_node(
        "const b = await import(pathToFileURL('js/backend.js').href);\n"
        f"const navs = {json.dumps(navs)};\n"
        "const cores = { iphone: 2, ipad: 2, mac_safari: 8, android: 8, desktop: 16, android_desktop: 8, touch_laptop: 12 };\n"
        "const dev = {}; const pick = {};\n"
        "for (const [k, n] of Object.entries(navs)) {\n"
        "  dev[k] = b.deviceInfo(n, !!n.coarse);\n"
        "  for (const isolated of [false, true]) for (const ep of ['auto', 'wasm', 'webgpu'])\n"
        "    pick[`${k} ${isolated ? 'isolated' : 'plain'} ${ep}`] = b.pickBackend({ ep, isolated, cores: cores[k], device: dev[k] });\n"
        "}\n"
        "const run = async (gpu, failAt) => { const calls = []; const errs = [];\n"
        "  const open = async (onGpu) => { calls.push(onGpu); if (onGpu && failAt) { const e = new Error(failAt);\n"
        "    if (failAt === 'front') e.where = 'front'; throw e; } return onGpu ? 'gpu-engine' : 'wasm-engine'; };\n"
        "  try { const r = await b.openWithFallback(gpu, open, (e) => errs.push(e.message)); return { ...r, calls, errs }; }\n"
        "  catch (e) { return { thrown: e.message, calls, errs }; } };\n"
        "const fb = { ok: await run(true), create: await run(true, 'create'), warm: await run(true, 'warm-up'),\n"
        "  front: await run(true, 'front'), wasm: await run(false) };\n"
        "const hang = await b.timed(new Promise(() => {}), 30, 'requestAdapter').catch((e) => e.message);\n"
        "const quick = await b.timed(Promise.resolve(7), 1000, 'x');\n"
        # the clock starts once `after` resolves (the .wasm has downloaded); an uncaught error in `scope` fails it
        "const t0 = Date.now();\n"
        "const late = await b.timed(new Promise(() => {}), 30, 'threads', { after: new Promise((r) => setTimeout(r, 150)) })\n"
        "  .catch((e) => ({ m: e.message, ms: Date.now() - t0 }));\n"
        "const scope = new EventTarget(); const pending = new Promise(() => {});\n"
        "const guarded = b.timed(pending, 60000, '2 WebAssembly threads', { after: pending, scope }).catch((e) => e);\n"
        "const ev = new Event('error', { cancelable: true }); ev.error = { message: 'worker sent an error!' };\n"
        "scope.dispatchEvent(ev);\n"
        "const g = await guarded; const thread = { m: g.message, kind: b.errorKind(g), prevented: ev.defaultPrevented };\n"
        "const again = new Event('error', { cancelable: true }); scope.dispatchEvent(again); thread.later = again.defaultPrevented;\n"
        f"const msg = {json.dumps(IPHONE_ERROR)};\n"
        f"const pyMsg = {json.dumps(PYTHON_ERROR)};\n"
        "const net = b.tag(new TypeError('Failed to fetch'), { kind: 'network' });\n"
        "const py = b.tag(new Error(pyMsg), { where: 'front' });\n"
        "const abort = new Error('RuntimeError: Aborted(). Build with -sASSERTIONS for more info.');\n"
        "const oom = new RangeError('WebAssembly.Memory(): could not allocate memory');\n"
        "const plans = { cap: b.pickBackend({ isolated: true, cores: 16, maxThreads: 1 }),\n"
        "  capIos: b.pickBackend({ isolated: true, cores: 2, device: dev.iphone, maxThreads: 1 }) };\n"
        "const pyodide404 = \"Failed to load 'https://cdn.jsdelivr.net/npm/pyodide@0.27.7/pyodide.asm.wasm': request failed.\";\n"
        "console.log(JSON.stringify({ dev, pick, fb, hang, quick, late, thread, plans, short: b.shortError(new Error(msg)),\n"
        "  shortPy: b.shortError(py), shortPage: b.shortError('x'.repeat(500)),\n"
        "  broken: [b.engineBroken(abort), b.engineBroken(oom), b.engineBroken(py), b.engineBroken(new Error(msg)),\n"
        "    b.engineBroken(new Error('RuntimeError: unreachable'))],\n"
        "  kinds: [b.errorKind(new Error(msg)), b.errorKind(net), b.errorKind(new TypeError('Load failed')),\n"
        "    b.errorKind(new Error('step.f16.onnx: HTTP 404')), b.errorKind(oom),\n"
        "    b.errorKind(b.tag('plain string', { kind: 'network' })), b.errorKind(new Error(pyodide404)),\n"
        "    b.errorKind(new Error('std::bad_alloc')), b.errorKind(new Error('Cannot enlarge memory arrays to size 2147483648 bytes')),\n"
        "    b.errorKind(b.tag(new Error('2 WebAssembly threads: Failed to fetch dynamically imported module'), { kind: 'engine' }))] }));")
    dev = out["dev"]
    assert dev["iphone"] == {"ios": True, "mobile": True, "webkit": True}
    assert dev["ipad"] == {"ios": True, "mobile": True, "webkit": True}  # iPadOS says "Macintosh"
    assert dev["mac_safari"] == {"ios": False, "mobile": False, "webkit": True}
    assert dev["android"] == {"ios": False, "mobile": True, "webkit": False}
    assert dev["desktop"] == {"ios": False, "mobile": False, "webkit": False}
    assert dev["android_desktop"]["mobile"] and not dev["touch_laptop"]["mobile"]
    pick = out["pick"]
    for k in ("iphone", "ipad", "mac_safari", "android"):  # WebGPU is never picked by itself there
        assert not pick[f"{k} plain auto"]["gpu"] and not pick[f"{k} isolated auto"]["gpu"], k
        assert pick[f"{k} plain webgpu"]["gpu"], k  # ?ep=webgpu stays an explicit override
    assert pick["iphone plain auto"] == {"threads": 1, "gpu": False}
    assert pick["iphone isolated auto"] == {"threads": 2, "gpu": False}
    assert pick["android isolated auto"]["threads"] == 4 and pick["mac_safari isolated auto"]["threads"] == 6
    assert pick["desktop plain auto"] == {"threads": 1, "gpu": True}  # no threads: WebGPU first, as before
    assert pick["desktop isolated auto"] == {"threads": 8, "gpu": False}
    assert not pick["desktop plain wasm"]["gpu"]
    assert not pick["android_desktop plain auto"]["gpu"] and pick["android_desktop isolated auto"]["threads"] == 4
    assert pick["touch_laptop plain auto"]["gpu"] and pick["touch_laptop isolated auto"]["threads"] == 8
    assert out["plans"] == {"cap": {"threads": 1, "gpu": True}, "capIos": {"threads": 1, "gpu": False}}
    fb = out["fb"]
    assert fb["ok"] == {"engine": "gpu-engine", "backend": "webgpu", "calls": [True], "errs": []}
    for case in ("create", "warm"):  # session creation or the warm-up run fails on WebGPU: the model reopens on wasm
        assert fb[case]["backend"] == "wasm" and fb[case]["calls"] == [True, False] and len(fb[case]["errs"]) == 1
    assert fb["front"] == {"thrown": "front", "calls": [True], "errs": []}  # not a WebGPU problem
    assert fb["wasm"]["calls"] == [False] and fb["wasm"]["backend"] == "wasm"
    assert out["hang"] == "requestAdapter: no answer in 0.03 s" and out["quick"] == 7
    assert out["late"]["m"] == "threads: no answer in 0.03 s" and out["late"]["ms"] >= 150
    assert out["thread"] == {"m": "2 WebAssembly threads: worker sent an error!", "kind": "engine", "prevented": True,
                             "later": False}  # the listener is gone once it settled
    short = out["short"]
    assert "\n" not in short and len(short) <= 240 and "/mnt/" not in short and short.startswith("failed to call OrtRun()")
    # the C++ location and signature go, what failed stays
    assert short == ("failed to call OrtRun(). ERROR_CODE: 1, ERROR_MESSAGE: buffer_manager.cc:562: Failed to create a "
                     "WebGPU compute pipeline: Failed to compile the shader source, generated metal: #include <metal_stdlib>")
    assert out["shortPy"] == "ValueError: could not read this text"  # the end of a traceback says what went wrong
    assert len(out["shortPage"]) == 240 and out["shortPage"].endswith("...")
    assert out["broken"] == [True, True, False, False, True]
    assert out["kinds"] == ["engine", "network", "network", "network", "memory", "network", "network", "memory",
                            "memory", "engine"]


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
