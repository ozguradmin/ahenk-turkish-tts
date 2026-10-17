// Everything heavy runs here, off the page's main thread: Pyodide with trnorm (text to spoken form), the four ONNX
// graphs on onnxruntime-web (WebAssembly; WebGPU only on desktops without threads, see js/backend.js), the model
// download with a sha256 check and a Cache Storage copy for the next visit. If WebGPU fails, at session creation, on
// the warm-up or on a later request, the same model buffers are opened again on WebAssembly. If threaded WebAssembly
// fails to start, the error says how many threads it had and the page tries again on one.
//
// Messages in:  {type: 'init', modelBase, pyBase, variant, ep, device, maxThreads}
//               {type: 'say', id, text, style, steps, speed, seed}
// Messages out: {type: 'status', stage, ...} {type: 'alive'} {type: 'ready', info} {type: 'backend', info}
//               {type: 'progress', id, ...} {type: 'audio', id, audio, sampleRate, spoken, ms}
//               {type: 'error', stage?, id?, kind, message, threads, broken}

import { engineBroken, errorKind, openWithFallback, pickBackend, shortError, tag, timed } from './backend.js';
import { CDN } from './config.js';
import { CepEngine, variantFiles } from './engine.js';
import { setupFront } from './front.js';

const CACHE = 'ahenk-cep-models-v1';
const GPU_MS = 40000; // the longest WebGPU may take to create one session or to run the warm-up
const THREADS_MS = 30000; // the longest threaded WebAssembly may take to start once its .wasm has arrived
const BEAT_MS = 10000;
let engine = null;
let ready = null;
let info = null;
let toWasm = null; // while on WebGPU: () => the same model opened on WebAssembly

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
    return !!(self.navigator.gpu && (await timed(self.navigator.gpu.requestAdapter(), 5000, 'requestAdapter')));
  } catch {
    return false;
  }
}

// The WebGPU and the WebAssembly builds are separate modules, each with its own WebAssembly instance: switching
// means importing the other one and creating the sessions again.
async function importOrt(gpu, threads) {
  let ort;
  try {
    ort = await import(`${CDN.ortDist}${gpu ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs'}`);
  } catch (e) {
    throw tag(e, { kind: 'network' });
  }
  ort.env.wasm.wasmPaths = CDN.ortDist;
  ort.env.wasm.numThreads = threads;
  return ort;
}

async function loadFront(pyBase, vocab) {
  const t0 = performance.now();
  const { loadPyodide } = await import(`${CDN.pyodide}pyodide.mjs`).catch((e) => { throw tag(e, { kind: 'network' }); });
  const pyodide = await loadPyodide({ indexURL: CDN.pyodide });
  const front = await setupFront(pyodide, async (p) => {
    const r = await fetch(new URL(p, pyBase));
    if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
    return r.text();
  }, vocab);
  post({ type: 'status', stage: 'front-ready', ms: performance.now() - t0 });
  return front;
}

// When onnxruntime's .wasm, fetched from now on, has downloaded (resource timing), so that the time limits below do
// not count the download. A browser that cannot tell starts the clock at once.
function wasmArrived() {
  const since = performance.now();
  return new Promise((resolve) => {
    try {
      if (!(self.PerformanceObserver?.supportedEntryTypes || []).includes('resource')) return resolve();
      const obs = new PerformanceObserver((list) => {
        if (list.getEntries().some((x) => x.name.startsWith(CDN.ortDist) && x.name.endsWith('.wasm') && x.startTime >= since)) {
          obs.disconnect();
          resolve();
        }
      });
      obs.observe({ type: 'resource', buffered: true });
    } catch {
      resolve();
    }
  });
}

function releaseAll(sessions) {
  for (const s of Object.values(sessions)) Promise.resolve().then(() => s.release()).catch(() => {});
}

// The four sessions, then a short warm-up run (compiles kernels and allocates once, before the first real request).
// On WebGPU each step has a time limit. The first session also starts onnxruntime's WebAssembly: the .wasm and, with
// threads, the nested workers that run them; its limit starts once the .wasm has arrived, and with threads an
// uncaught error (a thread that could not start) fails it too. On failure whatever was created is released.
async function openEngine(ort, gpu, threads, buffers, files, manifest, frontP) {
  // logSeverityLevel 3: errors only (WebGPU sessions otherwise print node-placement warnings as console errors)
  const opts = { executionProviders: gpu ? ['webgpu', 'wasm'] : ['wasm'], graphOptimizationLevel: 'all', logSeverityLevel: 3 };
  const sessions = {};
  try {
    for (let i = 0; i < files.length; i++) {
      const arrived = i === 0 ? wasmArrived() : null; // before create, which fetches the .wasm
      let create = ort.InferenceSession.create(buffers[i], opts);
      if (gpu) create = timed(create, GPU_MS, `WebGPU ${files[i].graph}`, { after: arrived });
      else if (arrived && threads > 1) create = timed(create, THREADS_MS, `${threads} WebAssembly threads`, { after: arrived, scope: self });
      sessions[files[i].graph] = await create;
    }
    const front = await frontP;
    const prepare = async (text, speed) => {
      try {
        return front.prepare(text, speed);
      } catch (e) {
        throw tag(e, { where: 'front' }); // a Python exception: not the backend's fault
      }
    };
    const eng = new CepEngine(ort, sessions, manifest, prepare);
    const warm = eng.say('Merhaba.', { steps: 2 });
    await (gpu ? timed(warm, GPU_MS, 'WebGPU warm-up') : warm);
    return eng;
  } catch (e) {
    releaseAll(sessions);
    throw e;
  }
}

async function init({ modelBase, pyBase, variant, ep, device = {}, maxThreads = 0 }) {
  const t0 = performance.now();
  post({ type: 'status', stage: 'start' }); // this worker and its imports have loaded
  const cores = self.navigator.hardwareConcurrency || 2;
  const plan = pickBackend({ ep, isolated: !!self.crossOriginIsolated, cores, device, maxThreads });
  const { threads } = plan;
  const gpu = plan.gpu && (await webgpuAvailable());
  let ort = await importOrt(gpu, threads);
  const mres = await fetch(new URL('ahenk-cep.json', modelBase)).catch((e) => { throw tag(e, { kind: 'network' }); });
  if (!mres.ok) throw new Error(`ahenk-cep.json: HTTP ${mres.status}`);
  const manifest = await mres.json();
  const files = variantFiles(manifest, variant);
  const total = files.reduce((n, f) => n + f.bytes, 0);
  post({ type: 'status', stage: 'download', loaded: 0, total });
  const frontP = loadFront(pyBase, manifest.vocab).catch((e) => { throw tag(e, { where: 'front' }); });
  frontP.catch(() => {}); // awaited in openEngine; if the download fails first, that is the error reported
  const cache = await openCache();
  let loaded = 0;
  let fromCache = true;
  const buffers = await Promise.all(files.map((f) => fetchModel(cache, new URL(f.file, modelBase).href, f, (n, cached) => {
    loaded += n;
    fromCache = fromCache && cached;
    post({ type: 'status', stage: 'download', loaded, total });
  }).catch((e) => { throw tag(e, { kind: 'network' }); })));
  const tDownload = performance.now() - t0;
  post({ type: 'status', stage: 'sessions', threads });
  // Opening says nothing for a while (onnxruntime's and Pyodide's own downloads, compiling, the warm-up): a beat
  // tells the page that this worker is still alive, so a slow connection is not taken for a dead worker.
  const beat = setInterval(() => post({ type: 'alive' }), BEAT_MS);
  const open = async (onGpu) => {
    try {
      if (gpu && !onGpu) ort = await importOrt(false, threads);
      return await openEngine(ort, onGpu, threads, buffers, files, manifest, frontP);
    } catch (e) {
      // threads that failed to start or to run (not a download, not the text front-end): one thread may still work
      if (!onGpu && threads > 1 && e.where !== 'front' && errorKind(e) !== 'network') throw tag(e, { threads });
      throw e;
    }
  };
  let gpuError = null;
  let opened;
  try {
    opened = await openWithFallback(gpu, open, (e) => {
      gpuError = shortError(e);
      console.warn('Ahenk Cep: WebGPU failed, using WebAssembly instead.', e);
    });
  } finally {
    clearInterval(beat);
  }
  engine = opened.engine;
  toWasm = opened.backend === 'webgpu' ? () => open(false) : null;
  info = {
    backend: opened.backend, threads, variant, bytes: total, fromCache, gpuError,
    isolated: !!self.crossOriginIsolated, ms: performance.now() - t0, downloadMs: tDownload,
    ort: ort.env.versions?.web || null, sampleRate: manifest.sample_rate,
  };
  post({ type: 'ready', info });
}

async function say({ id, text, style, steps, speed, seed }) {
  await ready;
  const t0 = performance.now();
  const run = () => engine.say(text, {
    style, steps, cfg: 2.0, speed, seed,
    onProgress: (p) => post({ type: 'progress', id, ...p }),
  });
  let out;
  try {
    out = await run();
  } catch (e) {
    if (!toWasm || e.where === 'front') throw e;
    // WebGPU passed the warm-up but failed on this request: move to WebAssembly for good and try once more.
    console.warn('Ahenk Cep: WebGPU failed, using WebAssembly instead.', e);
    const reopen = toWasm;
    toWasm = null;
    const old = engine;
    engine = await reopen();
    releaseAll(old.s);
    info = { ...info, backend: 'wasm', gpuError: shortError(e) };
    post({ type: 'backend', info });
    out = await run();
  }
  const ms = performance.now() - t0;
  post({ type: 'audio', id, audio: out.audio, sampleRate: out.sampleRate, spoken: out.spoken, ms }, [out.audio.buffer]);
}

// The page gets a kind (what to tell the visitor) and one short line; the whole error goes to the console. threads:
// the threads it failed on, when one thread may do better; broken: the engine is unusable, start a new worker.
function failure(e) {
  console.error(e);
  return { kind: errorKind(e), message: shortError(e), threads: (e && e.threads) || 0, broken: engineBroken(e) };
}

self.onmessage = ({ data }) => {
  if (data.type === 'init') {
    ready = init(data);
    ready.catch((e) => post({ type: 'error', stage: 'init', ...failure(e) }));
  } else if (data.type === 'say') {
    say(data).catch((e) => post({ type: 'error', id: data.id, ...failure(e) }));
  }
};
