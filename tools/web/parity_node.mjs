// Runs the page's own code (site/js/*.js, site/py/*) in Node with the same Pyodide and onnxruntime-web the page
// loads, and writes what it produced for tts/small/web_parity.py to compare with CPython and PyTorch.
//
//   node tools/web/parity_node.mjs --deps <dir with node_modules/{pyodide,onnxruntime-web}> --texts rows.jsonl
//        --out <dir> [--models site/models] [--variant fp32] [--audio 8] [--threads 4]
//
// rows.jsonl: {"id", "text", "style"?, "seed"?}. Writes front.jsonl ({"id", "spoken", "pieces"} for every row),
// audio/<id>.f32 (raw float32, the first --audio rows), first.wav (the page's WAV of row 1), randn.json and
// timing.json.

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SITE = path.resolve(HERE, '..', '..', 'site');

function args() {
  const a = { models: path.join(SITE, 'models'), variant: 'fp32', audio: 8, threads: 4 };
  const v = process.argv.slice(2);
  for (let i = 0; i < v.length; i += 2) a[v[i].replace(/^--/, '')] = v[i + 1];
  for (const k of ['deps', 'texts', 'out']) if (!a[k]) throw new Error(`--${k} is required`);
  return a;
}

const a = args();
const nm = path.resolve(a.deps, 'node_modules');
const { loadPyodide } = await import(pathToFileURL(path.join(nm, 'pyodide', 'pyodide.mjs')).href);
const ort = await import(pathToFileURL(path.join(nm, 'onnxruntime-web', 'dist', 'ort.node.min.mjs')).href);
const { setupFront } = await import(pathToFileURL(path.join(SITE, 'js', 'front.js')).href);
const { CepEngine, variantFiles } = await import(pathToFileURL(path.join(SITE, 'js', 'engine.js')).href);
const { torchRandn } = await import(pathToFileURL(path.join(SITE, 'js', 'randn.js')).href);
const { encodeWav } = await import(pathToFileURL(path.join(SITE, 'js', 'wav.js')).href);

fs.mkdirSync(path.join(a.out, 'audio'), { recursive: true });
const manifest = JSON.parse(fs.readFileSync(path.join(a.models, 'ahenk-cep.json'), 'utf8'));
const t0 = performance.now();
const pyodide = await loadPyodide({ indexURL: path.join(nm, 'pyodide') + path.sep });
const front = await setupFront(pyodide, async (p) => fs.readFileSync(path.join(SITE, 'py', p), 'utf8'), manifest.vocab);
const tFront = performance.now() - t0;

const rows = fs.readFileSync(a.texts, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));
const lines = [];
for (const r of rows) {
  const p = front.prepare(r.text, 1.0);
  lines.push(JSON.stringify({ id: r.id, spoken: p.spoken, pieces: p.pieces }));
}
fs.writeFileSync(path.join(a.out, 'front.jsonl'), lines.join('\n') + '\n');

const randn = {};
for (const seed of ['0', '1', '12345', String(2 ** 40 + 7)]) randn[seed] = Array.from(torchRandn(BigInt(seed), 64 * 37));
fs.writeFileSync(path.join(a.out, 'randn.json'), JSON.stringify(randn));

ort.env.wasm.numThreads = Number(a.threads);
const sessions = {};
const t1 = performance.now();
for (const f of variantFiles(manifest, a.variant)) {
  sessions[f.graph] = await ort.InferenceSession.create(fs.readFileSync(path.join(a.models, f.file)),
    { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
}
const tLoad = performance.now() - t1;
const engine = new CepEngine(ort, sessions, manifest, async (text, speed) => front.prepare(text, speed));
const timing = { front_ms: tFront, sessions_ms: tLoad, threads: Number(a.threads), variant: a.variant, rows: [] };
for (const r of rows.slice(0, Number(a.audio))) {
  const s = performance.now();
  const out = await engine.say(r.text, { style: r.style || 'neutral', seed: r.seed || 0 });
  const ms = performance.now() - s;
  fs.writeFileSync(path.join(a.out, 'audio', `${r.id.replace(/[^\w.-]/g, '_')}.f32`), Buffer.from(out.audio.buffer));
  if (!timing.rows.length) { // the page's WAV file for the first row, for web_parity.py to read back
    const wav = encodeWav(out.audio, out.sampleRate, { ISFT: 'Ahenk Cep', ICMT: 'Yapay zeka ile üretilmiş ses.' });
    fs.writeFileSync(path.join(a.out, 'first.wav'), Buffer.from(await wav.arrayBuffer()));
  }
  timing.rows.push({ id: r.id, ms, seconds: out.audio.length / out.sampleRate, rtf: ms / 1000 / (out.audio.length / out.sampleRate) });
  console.log(`${r.id}: ${(out.audio.length / out.sampleRate).toFixed(2)} s audio in ${(ms / 1000).toFixed(2)} s`);
}
fs.writeFileSync(path.join(a.out, 'timing.json'), JSON.stringify(timing, null, 1));
console.log(`front ${tFront.toFixed(0)} ms, sessions ${tLoad.toFixed(0)} ms`);
