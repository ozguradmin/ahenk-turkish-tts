# Ahenk Cep in the browser (site/)

A static page that runs Ahenk Cep (8.6M parameters, EMA Lightning architecture) entirely in the visitor's browser:
no server, no GPU of ours, the text never leaves the device. Turkish UI with an English toggle, light and dark
themes. The page always uses the neutral style: the small model's other styles are too weak to offer yet, so it says
so and points to the big model, Ahenk, for expressive speech.

## Open it locally

```bash
python tts/small/web_export.py --weights hf-publish/ahenk-cep-tr --out site/models   # once: writes the ONNX files
python tools/web/serve.py --isolate          # http://localhost:8000/  (multi-threaded WebAssembly)
python tools/web/serve.py                    # the same without COOP/COEP headers, as GitHub Pages serves it
python -m http.server 8000 --directory site  # also fine (no headers)
```

Use the venv that has torch and ema-lightning for the export; `serve.py` needs only the standard library. On
localhost the page reads the model from `site/models/`; anywhere else from the Hugging Face model repo (see below).
URL switches: `?models=local|hf` (also picks the big model's samples), `?variant=fp32|f16|int8`, `?ep=wasm|webgpu`,
`?lang=en`.

## Where it is published

| Address | Host | Deploy |
|---|---|---|
| https://ozguradmin.github.io/ahenk-turkish-tts/ (main link) | GitHub Pages | `.github/workflows/pages.yml`, on every push to `main` that touches `site/` |
| https://ahenk-tts.pages.dev/ | Cloudflare Pages | `wrangler pages deploy <copy of site/ without models/ and samples/big/> --project-name ahenk-tts` |

Both hosts serve the same files and get cross-origin isolation from `coi-serviceworker.js`; the model always comes
from the Hugging Face model repo. Some Turkish ISPs block every `*.pages.dev` address (on 2026-10-17 TurkNet's
resolver answered NXDOMAIN for all of them), so links for Turkish readers go to GitHub Pages. All paths in the page
are relative, so it works at `/` and under `/ahenk-turkish-tts/`.

## How it works

| Piece | File |
|---|---|
| UI, WAV download (16-bit, 48 kHz, RIFF INFO comment saying the speech is synthetic) | `index.html`, `css/style.css`, `js/app.js`, `js/wav.js` |
| Audio player: canvas waveform, seek by pointer or keyboard (a slider), one clip plays at a time | `js/player.js` |
| Pre-rendered examples and the big model's samples (see below) | `samples/`, `tools/web/render_samples.py` |
| Fonts: Newsreader and Hanken Grotesk (Fontsource 5.3.0 on jsdelivr, OFL, with the latin-ext subset for Turkish) | `index.html` |
| Worker: Pyodide, onnxruntime-web, download with sha256 check, Cache Storage | `js/worker.js` |
| Sampling loop: frame timeline, 16 Euler steps with CFG 2, 100-frame decoder windows, pauses | `js/engine.js` |
| torch.randn's CPU noise from a seed (mt19937 + ATen's Box-Muller), so a seed gives the PyTorch voice | `js/randn.js` |
| Text front-end in Pyodide: trnorm (copy of `src/trnorm`) and EMA's alphabet, chunker and word bounds | `js/front.js`, `py/` |
| Cross-origin isolation on GitHub Pages (MIT, gzuidhof/coi-serviceworker 0.1.7) | `coi-serviceworker.js` |
| URLs and versions (onnxruntime-web 1.30.0, Pyodide 0.27.7, jsdelivr) | `js/config.js` |

The model is cut into four ONNX graphs by `tts/small/web_export.py`: `text` (letters and style to features and
durations), `cond` (aligner), `step` (one Euler step, CFG batched inside) and `decoder` (one window). The decoder's
transposed convolutions are exported as kernel-2 convolutions plus an interleave (same arithmetic), because
onnxruntime-web 1.30's WebGPU ConvTranspose gives wrong output for some of these shapes.

Backend: multi-threaded WebAssembly when the page is cross-origin isolated (up to 8 threads), otherwise WebGPU when
the browser has it, otherwise one WebAssembly thread.

## Examples and samples

Every example sentence plays at once from a pre-rendered clip, so a visitor can listen before the model is ready.
`python tools/web/render_samples.py` (venv with onnxruntime, ffmpeg on PATH) renders them with the page's own pipeline
(ONNX f16, neutral, 16 steps, seed 0) and writes `samples/cep_*.ogg` (Opus 40 kbps) and `samples/cep_*.mp3` (fallback
for Safari before 18.4) plus `samples/samples.json` (texts, spoken forms, waveform peaks). They are committed: about
0.5 MB in all. "Use this text" puts the sentence in the text box; with the default settings and seed 0 the browser
gives the same voice.

The section on expressive speech plays three of the big model's samples (laughing, cheerful, calm) straight from its
model card, `ozgurguler/ahenk-tts-tr/resolve/main/samples/`. While that repo is private they answer 401 and the cards
say the recording cannot be opened yet. For local testing, `render_samples.py` copies them from
`data/release-samples` to `samples/big/` (gitignored), which the page tries first on localhost.

## Model files (not in git)

`site/models/` is gitignored. Before the site goes public, upload its files to the model repo under `onnx/`
(`ahenk-cep.json` plus the `.onnx` files) and, ideally, pin `MODELS.hf` in `js/config.js` to that commit instead of
`main`. The page checks every file against the sha256 in `ahenk-cep.json`.

| Variant | text + cond + step + decoder | Total | Notes |
|---|---|---:|---|
| `f16` (default) | fp16-stored weights, fp32 arithmetic | 17.4 MB | same Whisper WER as fp32, UTMOS -0.004 |
| `fp32` | as exported | 34.5 MB | reference |
| `int8` | MatMul weights int8 (text, cond, step) + f16 decoder | 13.3 MB | same WER, UTMOS -0.035, voice a bit further from PyTorch |

## Measurements (2026-10-16, laptop: 16 logical cores, RTX 5060 Laptop; model files from localhost)

First visit downloads about 26 MB: the model (17.4 MB, f16), onnxruntime-web's WebAssembly (3.1 MB compressed;
the WebGPU build 5.5 MB) and Pyodide (5.6 MB). A cold visit in headless Chrome 155 was ready in 8.8 s (including the
service worker's one reload), a second visit in 2.6 s (model from Cache Storage, verified by sha256).

Real-time factor (synthesis time / audio length; lower is faster), after the first request:

| Backend | 16 steps | 8 steps | 4 steps |
|---|---:|---:|---:|
| WebAssembly, 8 threads (Chrome 155, isolated through the service worker) | 0.11-0.17 | 0.07 | 0.055 |
| WebGPU, not isolated (Claude browser pane, Chrome 152) | 0.14-0.17 | 0.08 | 0.07 |
| WebAssembly, 1 thread, not isolated | 0.42-0.56 | 0.28 | 0.21 |

## Checks

- `tests/test_web_demo.py`: vendored trnorm equals `src/trnorm`, the front-end equals EMA Lightning's, the JS noise /
  timeline / windows equal the Python ones (needs `node`), the ONNX pipeline equals PyTorch (needs `site/models`).
- `tts/small/web_parity.py`: runs the page's own JS and Python in Node (same Pyodide and onnxruntime-web) over the
  316 rows of `benchmarks/tts/sets` plus 19 edge cases, and compares text, noise, audio and Whisper WER with PyTorch.
  It needs `npm install onnxruntime-web@1.30.0 pyodide@0.27.7` in a scratch folder passed as `--deps`.
  Last run: 335/335 rows give the same spoken form and pieces as CPython and as EMA Lightning's own front-end; noise
  within 1.8e-6 of torch.randn; fp32 audio the same length as PyTorch on 24/24 rows at 84-105 dB SNR; WER 1.52 /
  CER 0.51 (22/24 rows exact) for PyTorch, fp32, f16 and int8 alike; UTMOS 3.472 (PyTorch, fp32), 3.468 (f16),
  3.437 (int8); the page's WAV within 1 LSB of `write_wav`.
- `tools/web/browser_bench.mjs`: a cold and a warm visit in a real Chrome (puppeteer-core), timings as JSON.

Contact: destek@ozgurguler.tech
