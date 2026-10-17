// Which backend to run on, and what to do when it fails. No DOM and no onnxruntime here, so tests run it in Node.
//
// WebAssembly works everywhere (one thread without cross-origin isolation, several with it). WebGPU is only tried
// on desktops without threads, where it is about twice as fast as one WebAssembly thread; never on phones and
// tablets, nor in Safari or any iOS browser: onnxruntime-web's WebGPU shaders failed to compile on an iPhone (Metal),
// at run time, after the sessions had been created. Whenever WebGPU fails, the model is opened again on WebAssembly.

// What the worker needs to know about the device (navigator.maxTouchPoints does not exist in workers). `nav` is
// window.navigator, `coarse` whether the main pointer is a finger (matchMedia('(pointer: coarse)')). iPadOS says
// "Macintosh" in its user agent and Android tablets in desktop mode say "Linux": their touch points give them away.
export function deviceInfo(nav, coarse = false) {
  const ua = nav.userAgent || '';
  const touch = (nav.maxTouchPoints || 0) > 1;
  const ios = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && touch);
  const mobile = ios || !!(nav.userAgentData && nav.userAgentData.mobile) || /Android|Mobi/i.test(ua) || (touch && coarse);
  const webkit = ios || /Apple/.test(nav.vendor || '');
  return { ios, mobile, webkit };
}

// -> {threads, gpu}; gpu: try WebGPU first. ep is ?ep=wasm|webgpu|auto; webgpu is an explicit override.
// Threads: a quarter of the cores stays with the page. Phones and tablets get at most 4 and iOS 2: each thread costs
// memory and iOS gives a web page little of it. (WebKit rounds the core count it reports, so it says little there;
// every iPhone since 2016 has two fast cores.) maxThreads: a lower cap, after threads failed to start in this tab.
export function pickBackend({ ep = 'auto', isolated = false, cores = 2, device = {}, maxThreads = 0 } = {}) {
  let threads = 1;
  if (isolated) threads = Math.max(1, Math.min(device.mobile ? 4 : 8, device.ios ? 2 : Math.floor((cores * 3) / 4)));
  if (maxThreads > 0) threads = Math.min(threads, maxThreads);
  const gpu = ep === 'webgpu' || (ep === 'auto' && threads < 2 && !device.mobile && !device.webkit);
  return { threads, gpu };
}

// open(onGpu) -> engine; it creates the sessions and runs the warm-up, so a failure in either moves to WebAssembly.
// A failure of the text front-end (err.where === 'front') has nothing to do with the backend and is passed on.
export async function openWithFallback(gpu, open, onGpuError = () => {}) {
  if (gpu) {
    try {
      return { engine: await open(true), backend: 'webgpu' };
    } catch (e) {
      if (e && e.where === 'front') throw e;
      onGpuError(e);
    }
  }
  return { engine: await open(false), backend: 'wasm' };
}

// The promise, or an error after `ms`. A late failure of the promise is not left unhandled. Options: `after`, a
// promise the clock waits for (onnxruntime's .wasm has downloaded, so a slow connection is not taken for a hang);
// `scope`, an event target whose uncaught errors also fail it (a WebAssembly thread that cannot start throws one
// into the worker and leaves onnxruntime waiting for good). Those errors are marked kind 'engine'.
export function timed(promise, ms, what, { after = null, scope = null } = {}) {
  let timer = 0;
  let done = false;
  let onError = null;
  const limit = new Promise((_, reject) => {
    const arm = () => {
      if (!done) timer = setTimeout(() => reject(new Error(`${what}: no answer in ${ms / 1000} s`)), ms);
    };
    if (after) after.then(arm, arm);
    else arm();
    if (scope) {
      onError = (ev) => {
        if (ev.preventDefault) ev.preventDefault();
        const m = (ev.error && ev.error.message) || ev.message || 'uncaught error';
        reject(tag(new Error(`${what}: ${m}`), { kind: 'engine' }));
      };
      scope.addEventListener('error', onError);
    }
  });
  promise.catch(() => {});
  return Promise.race([promise, limit]).finally(() => {
    done = true;
    clearTimeout(timer);
    if (onError) scope.removeEventListener('error', onError);
  });
}

// Mark where an error came from: err.kind ('network', 'engine') for errorKind, err.where ('front') for
// openWithFallback, err.threads for the page (threads that failed: it tries again on one).
export function tag(e, fields) {
  const err = e instanceof Error ? e : new Error(String(e));
  for (const [k, v] of Object.entries(fields)) if (err[k] === undefined) err[k] = v;
  return err;
}

// What the page should tell the visitor: 'memory', 'network' or 'engine' (the model could not run here).
export function errorKind(e) {
  const m = String((e && e.message) || e);
  if (/out of memory|could not allocate|failed to allocate|allocation failed|bad_alloc|enlarge memory|\bOOM\b/i.test(m)) return 'memory';
  if (e && (e.kind === 'network' || e.kind === 'engine')) return e.kind;
  if (/failed to fetch|networkerror|load failed|request failed|dynamically imported module|importing a module script failed|HTTP \d{3}|checksum mismatch|larger than expected/i.test(m)) {
    return 'network';
  }
  return 'engine';
}

// Whether the WebAssembly module is likely unusable after this error (an abort or a trap): the page then starts a
// new worker. Errors of the text front-end (a Python exception) leave it intact.
export function engineBroken(e) {
  if (e && e.where === 'front') return false;
  return errorKind(e) === 'memory' || /Aborted\(|\bRuntimeError\b|unreachable|out of bounds/i.test(String((e && e.message) || e));
}

// One line of an error, at most `max` characters: the first (the last for a Python traceback), without onnxruntime's
// build paths and C++ signatures. onnxruntime's WebGPU errors run to thousands of characters of shader source; the
// page shows this short form, folded away.
export function shortError(e, max = 240) {
  const text = String((e && e.message) || e || 'unknown error');
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);
  let line = (/^Traceback/.test(lines[0] || '') ? lines[lines.length - 1] : lines[0]) || 'unknown error';
  line = line.replace(/\/\S*?\/onnxruntime\/(?=\S)/g, '');
  // "core/providers/webgpu/buffer_manager.cc:562 WGPUBuffer onnxruntime::...(...) const [ONNXRuntimeError] : 1 :
  // FAIL : Failed to create..." -> "buffer_manager.cc:562: Failed to create..."
  line = line.replace(/(?:\S+\/)?([\w.-]+\.(?:cc|cpp|h)):(\d+) .*?\[ONNXRuntimeError\] : \d+ : \w+ : /, '$1:$2: ');
  return line.length > max ? `${line.slice(0, max - 3)}...` : line;
}
