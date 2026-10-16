// Where the page gets its code and its model. Every third-party file has an exact version.
//
// Model files: on a deployed site they come from the Hugging Face model repo (MODELS.hf). On localhost the page
// uses site/models/ (written by tts/small/web_export.py), because the model repo stays private until release.
// Override with ?models=local or ?models=hf. Variant: ?variant=fp32|f16|int8. Backend: ?ep=wasm|webgpu.

export const VERSIONS = { ort: '1.30.0', pyodide: '0.27.7' };

export const CDN = {
  ortDist: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VERSIONS.ort}/dist/`,
  pyodide: `https://cdn.jsdelivr.net/npm/pyodide@${VERSIONS.pyodide}/`,
};

export const MODELS = {
  hf: 'https://huggingface.co/ozgurguler/ahenk-cep-tr/resolve/main/onnx/',
  local: 'models/',
};

export const DEFAULT_VARIANT = 'f16';
export const MAX_CHARS = 600;

export const LINKS = {
  github: 'https://github.com/ozguradmin/ahenk-turkish-tts',
  hfCep: 'https://huggingface.co/ozgurguler/ahenk-cep-tr',
  hfBig: 'https://huggingface.co/ozgurguler/ahenk-tts-tr',
  ema: 'https://huggingface.co/canberkkkkkk/ema-lightning',
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

export function pageOptions(loc) {
  const q = new URLSearchParams(loc.search);
  const variant = ['fp32', 'f16', 'int8'].includes(q.get('variant')) ? q.get('variant') : DEFAULT_VARIANT;
  const ep = ['wasm', 'webgpu'].includes(q.get('ep')) ? q.get('ep') : 'auto';
  return { variant, ep, modelBase: modelBase(loc) };
}
