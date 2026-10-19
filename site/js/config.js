// Where the page gets its code and its model. Every third-party file has an exact version.
//
// Model files: on a deployed site they come from the Hugging Face model repo (MODELS.hf). On localhost the page
// uses site/models/ (written by tts/small/web_export.py), because the model repo stays private until release.
// Override with ?models=local or ?models=hf. Variant: ?variant=fp32|f16|int8. Backend: ?ep=wasm|webgpu. Text
// front-end: ?front=js|py.

export const VERSIONS = { ort: '1.30.0', pyodide: '0.27.7' };

export const CDN = {
  ortDist: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VERSIONS.ort}/dist/`,
  pyodide: `https://cdn.jsdelivr.net/npm/pyodide@${VERSIONS.pyodide}/`,
};

export const MODELS = {
  hf: 'https://huggingface.co/ozgurguler/ahenk-cep-tr/resolve/main/onnx/',
  local: 'models/',
};

// Pre-rendered clips. Ahenk Cep's examples live in site/samples/ (tools/web/render_samples.py). The big model's
// emotional samples come from its model card; on localhost a local copy in site/samples/big/ (gitignored, written by
// render_samples.py) is tried first, because the model repo is private until release.
export const SAMPLES = {
  examples: 'samples/',
  bigHf: 'https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/',
  bigLocal: 'samples/big/',
};

export const DEFAULT_VARIANT = 'f16';

// The text front-end: 'js' is the JavaScript port (js/front-native.js and js/trnorm/), with the Python as its
// fallback in this browser or for one text; 'py' runs the Python (site/py: trnorm and ahenk_front.py) in Pyodide from
// the start. ?front=js|py overrides it for one visit.
export const FRONT = 'js';

export const MAX_CHARS = 600;

export const LINKS = {
  github: 'https://github.com/ozguradmin/ahenk-turkish-tts',
  hfCep: 'https://huggingface.co/ozgurguler/ahenk-cep-tr',
  hfBig: 'https://huggingface.co/ozgurguler/ahenk-tts-tr',
  hfBigSamples: 'https://huggingface.co/ozgurguler/ahenk-tts-tr/tree/main/samples',
  ema: 'https://huggingface.co/canberkkkkkk/ema-lightning',
  okur: 'https://ozguradmin.github.io/ahenk-cep-pdfreader/', // Ahenk Okur: whole documents, same model and model cache
  mail: 'destek@ozgurguler.tech',
};

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]', '::1']);

// The model base URL for this page, absolute. `loc` is window.location.
export function modelBase(loc) {
  const q = new URLSearchParams(loc.search).get('models');
  const local = new URL(MODELS.local, loc.href).href;
  if (q === 'local') return local;
  if (q === 'hf') return MODELS.hf;
  return LOCAL_HOSTS.has(loc.hostname) ? local : MODELS.hf;
}

// Where to look for the big model's samples, in order. ?models=hf skips the local copy.
export function bigSampleBases(loc) {
  const local = new URL(SAMPLES.bigLocal, loc.href).href;
  const q = new URLSearchParams(loc.search).get('models');
  if (q === 'hf') return [SAMPLES.bigHf];
  return LOCAL_HOSTS.has(loc.hostname) || q === 'local' ? [local, SAMPLES.bigHf] : [SAMPLES.bigHf];
}

export function pageOptions(loc) {
  const q = new URLSearchParams(loc.search);
  const variant = ['fp32', 'f16', 'int8'].includes(q.get('variant')) ? q.get('variant') : DEFAULT_VARIANT;
  const ep = ['wasm', 'webgpu'].includes(q.get('ep')) ? q.get('ep') : 'auto';
  const front = ['js', 'py'].includes(q.get('front')) ? q.get('front') : FRONT;
  return { variant, ep, front, modelBase: modelBase(loc) };
}
