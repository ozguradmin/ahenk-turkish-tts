// Everything heavy runs here, off the page's main thread: Pyodide with trnorm (text to spoken form), the four ONNX
// graphs on onnxruntime-web (WebGPU when asked and available, else WebAssembly), the model download with a
// sha256 check and a Cache Storage copy for the next visit.
//
// Messages in:  {type: 'init', modelBase, pyBase, variant, ep}  {type: 'say', id, text, style, steps, speed, seed}
// Messages out: {type: 'status', stage, ...} {type: 'ready', info} {type: 'progress', id, ...}
//               {type: 'audio', id, audio, sampleRate, spoken, ms} {type: 'error', id?, message}

import { CDN } from './config.js';
import { CepEngine, variantFiles } from './engine.js';
import { setupFront } from './front.js';

const CACHE = 'ahenk-cep-models-v1';
let engine = null;
let ready = null;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function sha256(buf) {
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', buf));
  return Array.from(d, (b) => b.toString(16).padStart(2, '0')).join('');
}

async function openCache() {
  try {
    return await caches.open(CACHE);
  } catch {
    return null; // private mode or blocked storage: download every visit
  }
}

// One model file: from Cache Storage if a verified copy is there, else downloaded with progress and verified.
async function fetchModel(cache, url, file, onBytes) {
  const key = `${url}?sha256=${file.sha256}`;
  if (cache) {
    try {
      const hit = await cache.match(key);
      if (hit) {
        const buf = new Uint8Array(await hit.arrayBuffer());
        if ((await sha256(buf)) === file.sha256) {
          onBytes(buf.length, true);
          return buf;
        }
      }
    } catch { /* fall through to the network */ }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${file.file}: HTTP ${res.status}`);
  const buf = new Uint8Array(file.bytes);
  let off = 0;
  if (res.body && res.body.getReader) {
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (off + value.length > buf.length) throw new Error(`${file.file}: larger than expected`);
      buf.set(value, off);
      off += value.length;
      onBytes(value.length, false);
    }
  } else {
    const all = new Uint8Array(await res.arrayBuffer());
    buf.set(all.subarray(0, buf.length));
    off = all.length;
    onBytes(off, false);
  }
  if (off !== file.bytes || (await sha256(buf)) !== file.sha256) throw new Error(`${file.file}: checksum mismatch`);
  if (cache) {
    try {
      await cache.put(key, new Response(buf, { headers: { 'content-type': 'application/octet-stream' } }));
    } catch { /* quota: fine, just not cached */ }
  }
  return buf;
}

async function webgpuAvailable() {
  try {
    return !!(self.navigator.gpu && (await self.navigator.gpu.requestAdapter()));
  } catch {
    return false;
  }
}

async function loadFront(pyBase, vocab) {
  const t0 = performance.now();
  const { loadPyodide } = await import(`${CDN.pyodide}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL: CDN.pyodide });
  const front = await setupFront(pyodide, async (p) => {
    const r = await fetch(new URL(p, pyBase));
    if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
    return r.text();
  }, vocab);
  post({ type: 'status', stage: 'front-ready', ms: performance.now() - t0 });
  return front;
}

async function init({ modelBase, pyBase, variant, ep }) {
  const t0 = performance.now();
  // Backend. Multi-threaded WebAssembly (needs cross-origin isolation, which coi-serviceworker.js provides) was as
  // fast as WebGPU on a laptop RTX 5060 and starts faster, so it wins when available; without threads, WebGPU is
  // about twice as fast as one WebAssembly thread. ?ep=wasm|webgpu forces one.
  const cores = self.navigator.hardwareConcurrency || 2;
  const threads = self.crossOriginIsolated ? Math.max(1, Math.min(8, Math.floor((cores * 3) / 4))) : 1;
  const gpu = ep === 'webgpu' || (ep === 'auto' && threads < 2) ? await webgpuAvailable() : false;
  const ort = await import(`${CDN.ortDist}${gpu ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs'}`);
  ort.env.wasm.wasmPaths = CDN.ortDist;
  ort.env.wasm.numThreads = threads;
  const mres = await fetch(new URL('ahenk-cep.json', modelBase));
  if (!mres.ok) throw new Error(`ahenk-cep.json: HTTP ${mres.status}`);
  const manifest = await mres.json();
  const files = variantFiles(manifest, variant);
  const total = files.reduce((n, f) => n + f.bytes, 0);
  post({ type: 'status', stage: 'download', loaded: 0, total });
  const frontP = loadFront(pyBase, manifest.vocab);
  const cache = await openCache();
  let loaded = 0;
  let fromCache = true;
  const buffers = await Promise.all(files.map((f) => fetchModel(cache, new URL(f.file, modelBase).href, f, (n, cached) => {
    loaded += n;
    fromCache = fromCache && cached;
    post({ type: 'status', stage: 'download', loaded, total });
  })));
  const tDownload = performance.now() - t0;
  post({ type: 'status', stage: 'sessions' });
  const opts = { executionProviders: gpu ? ['webgpu', 'wasm'] : ['wasm'], graphOptimizationLevel: 'all' };
  const sessions = {};
  for (let i = 0; i < files.length; i++) sessions[files[i].graph] = await ort.InferenceSession.create(buffers[i], opts);
  const front = await frontP;
  engine = new CepEngine(ort, sessions, manifest, async (text, speed) => front.prepare(text, speed));
  const warm = performance.now();
  await engine.say('Merhaba.', { steps: 2 }); // compiles kernels and allocates once, before the first real request
  const info = {
    backend: gpu ? 'webgpu' : 'wasm', threads, variant, bytes: total, fromCache,
    isolated: !!self.crossOriginIsolated, ms: performance.now() - t0, downloadMs: tDownload,
    warmupMs: performance.now() - warm, ort: ort.env.versions?.web || null, sampleRate: manifest.sample_rate,
  };
  post({ type: 'ready', info });
}

async function say({ id, text, style, steps, speed, seed }) {
  await ready;
  const t0 = performance.now();
  const out = await engine.say(text, {
    style, steps, cfg: 2.0, speed, seed,
    onProgress: (p) => post({ type: 'progress', id, ...p }),
  });
  const ms = performance.now() - t0;
  post({ type: 'audio', id, audio: out.audio, sampleRate: out.sampleRate, spoken: out.spoken, ms }, [out.audio.buffer]);
}

self.onmessage = ({ data }) => {
  if (data.type === 'init') {
    ready = init(data);
    ready.catch((e) => post({ type: 'error', stage: 'init', message: String(e && e.message ? e.message : e) }));
  } else if (data.type === 'say') {
    say(data).catch((e) => post({ type: 'error', id: data.id, message: String(e && e.message ? e.message : e) }));
  }
};
