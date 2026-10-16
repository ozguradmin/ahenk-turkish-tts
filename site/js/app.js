// The page: Turkish first, English on request. All synthesis happens in js/worker.js.

import { LINKS, MAX_CHARS, pageOptions } from './config.js';
import { encodeWav } from './wav.js';

const STYLES = [
  { id: 'neutral', tr: 'Nötr', en: 'Neutral' },
  { id: 'cheerful', tr: 'Neşeli', en: 'Cheerful' },
  { id: 'sad', tr: 'Üzgün', en: 'Sad' },
  { id: 'angry', tr: 'Kızgın', en: 'Angry' },
  { id: 'surprised', tr: 'Şaşkın', en: 'Surprised' },
  { id: 'calm', tr: 'Sakin', en: 'Calm' },
  { id: 'empathetic', tr: 'Empatik', en: 'Empathetic' },
];

const EXAMPLES = [
  { text: "Randevunuz 14.10.2026 saat 09:30'da, 3. katta.", style: 'neutral', tr: 'tarih ve saat', en: 'date and time' },
  { text: 'Bu ürünün garantisi var mı?', style: 'neutral', tr: 'soru', en: 'question' },
  { text: "SGK'ye 15.00'te başvurun; ücret 1.250,75 TL, %20'si peşin.", style: 'neutral', tr: 'kısaltma, para, yüzde', en: 'abbreviation, money, percent' },
  { text: 'Siparişiniz 3 iş günü içinde, 249,90 TL karşılığında kapınızda olacak!', style: 'cheerful', tr: 'sayı, neşeli', en: 'numbers, cheerful' },
  { text: 'Ne kadar uğraştıysam da yetişemedim, onunla son kez vedalaşamadım.', style: 'sad', tr: 'üzgün', en: 'sad' },
  { text: 'Derin bir nefes alın, her şeyi adım adım birlikte halledeceğiz.', style: 'calm', tr: 'sakin', en: 'calm' },
  { text: 'Saat kaçta buluşalım? Dr. Kaya 2. kattaki 205 numaralı odada bekliyor.', style: 'neutral', tr: 'soru, unvan, sıra sayısı', en: 'question, title, ordinal' },
];

const T = {
  tr: {
    skip: 'Metin alanına geç',
    title: 'Türkçe metni tarayıcınızda seslendirin',
    lede: 'Ahenk Cep, 8,6 milyon parametreli küçük bir Türkçe ses modeli. Bu sayfada tamamen sizin cihazınızda çalışır: yazdığınız metin hiçbir sunucuya gitmez.',
    noticeTitle: 'Bu ses yapaydır.',
    noticeBody: 'Duyacağınız ses (f022) bir metin tarifinden tasarlandı; gerçek bir kişinin sesi ya da kaydı değildir. Ürettiğiniz sesi paylaşırken yapay olduğunu belirtin ve kimseyi taklit etmek ya da kandırmak için kullanmayın.',
    ttsHeading: 'Seslendirme',
    loadStart: 'Model hazırlanıyor…',
    loadDownload: 'Model indiriliyor',
    loadCache: 'Model önbellekten açılıyor',
    loadSessions: 'Model açılıyor',
    loadFront: 'Metin işleyici yükleniyor',
    loadWarm: 'Son hazırlık',
    loadError: 'Model yüklenemedi',
    textLabel: 'Metin',
    textHint: 'Sayılar, tarihler, saat, para ve kısaltmalar okunuşa çevrilir.',
    styleLabel: 'Duygu',
    styleHint: 'Bu küçük modelde duygular hafiftir: örneğin "Üzgün" sesi biraz daha pes ve yavaş yapar, "Sakin" biraz yavaşlatır. Büyük bir fark beklemeyin.',
    settings: 'Ayarlar',
    steps: 'Kalite',
    stepsBest: 'En iyi (16 adım)',
    stepsFast: 'Hızlı (8 adım)',
    stepsFaster: 'Çok hızlı (4 adım, daha az doğal)',
    speed: 'Hız',
    seed: 'Tohum',
    seedHint: 'Aynı metin, duygu ve tohum her zaman aynı sesi verir.',
    go: 'Seslendir',
    busy: 'Seslendiriliyor…',
    waiting: 'Model hazır olunca başlayacak…',
    empty: 'Önce bir metin yazın.',
    tooLong: `En çok ${MAX_CHARS} karakter.`,
    nothing: 'Bu metinde okunacak bir şey bulunamadı.',
    progress: (p) => p.decoding ? `Ses oluşturuluyor${p.pieces > 1 ? ` (parça ${p.piece + 1}/${p.pieces})` : ''}` : `Seslendiriliyor${p.pieces > 1 ? `: parça ${p.piece + 1}/${p.pieces},` : ':'} adım ${p.step}/${p.steps}`,
    download: 'WAV olarak indir',
    spokenLabel: 'Modelin okuduğu metin',
    stats: (s) => `${fmt(s.seconds, 1)} sn ses, ${fmt(s.wall, 1)} sn'de üretildi (gerçek zaman oranı ${fmt(s.rtf, 2)})`,
    examples: 'Örnek cümleler',
    aboutTitle: 'Nasıl çalışıyor?',
    about1: 'İlk açılışta tarayıcınız modeli (yaklaşık <span id="model-size">17 MB</span>) ve metni okunuşa çeviren Python kodunu (Pyodide ile, yaklaşık 6 MB) indirir; sonraki ziyaretlerde önbellekten açılır. Metin önce <a href="https://github.com/ozguradmin/ahenk-turkish-tts/blob/main/docs/TRNORM.md">trnorm</a> ile okunuş formuna çevrilir ("14.10.2026" olur "on dört ekim iki bin yirmi altı"), sonra model sesi 48 kHz olarak üretir.',
    about2: 'Model, büyük Ahenk modelinin sesinden öğrenen bir öğrencidir ve <a href="https://huggingface.co/canberkkkkkk/ema-lightning">EMA Lightning</a> mimarisini kullanır. Sınırları: tek tasarlanmış kadın sesi; duygular hafif; sonu yükselen sorular ("Yarın geliyorsun?") yükselmez; uzun metinde doğallık büyük modelin gerisinde kalır.',
    linkCode: 'Kod (GitHub)',
    linkCep: 'Ahenk Cep modeli (Hugging Face)',
    linkBig: 'Büyük model Ahenk (Hugging Face)',
    contact: 'İletişim:',
    license: 'Kod Apache-2.0, model ağırlıkları CC-BY-4.0. Temel model EMA Lightning (Apache-2.0).',
    tech: (i) => `${i.backend === 'webgpu' ? 'WebGPU' : `WebAssembly, ${i.threads} iş parçacığı`} · ${i.variant} · ${fmt(i.bytes / 1e6, 1)} MB`,
    langButton: 'EN',
    langLabel: 'Switch to English',
    wavComment: 'Yapay zeka ile üretilmiş ses (Ahenk Cep, tasarlanmış ses f022; gerçek bir kişi değildir).',
  },
  en: {
    skip: 'Skip to the text box',
    title: 'Turkish text to speech, in your browser',
    lede: 'Ahenk Cep is a small Turkish speech model with 8.6 million parameters. On this page it runs entirely on your device: the text you type never goes to a server.',
    noticeTitle: 'This voice is synthetic.',
    noticeBody: 'The voice you hear (f022) was designed from a text description; it is not a real person or a recording of one. Say that it is AI-generated when you share what you make, and never use it to impersonate or deceive anyone.',
    ttsHeading: 'Synthesis',
    loadStart: 'Preparing the model…',
    loadDownload: 'Downloading the model',
    loadCache: 'Opening the model from cache',
    loadSessions: 'Opening the model',
    loadFront: 'Loading the text processor',
    loadWarm: 'Almost ready',
    loadError: 'The model could not be loaded',
    textLabel: 'Text (Turkish)',
    textHint: 'Numbers, dates, times, money and abbreviations are read out in words.',
    styleLabel: 'Emotion',
    styleHint: 'Emotions are mild in this small model: "Sad" makes the voice a little lower and slower, "Calm" a little slower. Do not expect a big difference.',
    settings: 'Settings',
    steps: 'Quality',
    stepsBest: 'Best (16 steps)',
    stepsFast: 'Fast (8 steps)',
    stepsFaster: 'Fastest (4 steps, less natural)',
    speed: 'Speed',
    seed: 'Seed',
    seedHint: 'The same text, emotion and seed always give the same audio.',
    go: 'Speak',
    busy: 'Speaking…',
    waiting: 'Will start when the model is ready…',
    empty: 'Type some text first.',
    tooLong: `At most ${MAX_CHARS} characters.`,
    nothing: 'Nothing in this text can be read aloud.',
    progress: (p) => p.decoding ? `Making the audio${p.pieces > 1 ? ` (part ${p.piece + 1}/${p.pieces})` : ''}` : `Speaking${p.pieces > 1 ? `: part ${p.piece + 1}/${p.pieces},` : ':'} step ${p.step}/${p.steps}`,
    download: 'Download WAV',
    spokenLabel: 'What the model reads',
    stats: (s) => `${fmt(s.seconds, 1)} s of audio made in ${fmt(s.wall, 1)} s (real-time factor ${fmt(s.rtf, 2)})`,
    examples: 'Example sentences',
    aboutTitle: 'How it works',
    about1: 'On the first visit your browser downloads the model (about <span id="model-size">17 MB</span>) and the Python code that turns text into its spoken form (with Pyodide, about 6 MB); later visits open it from the cache. The text first goes through <a href="https://github.com/ozguradmin/ahenk-turkish-tts/blob/main/docs/TRNORM.md">trnorm</a> ("14.10.2026" becomes "on dört ekim iki bin yirmi altı"), then the model makes 48 kHz audio.',
    about2: 'The model is a student that learned from the voice of the big Ahenk model and uses the <a href="https://huggingface.co/canberkkkkkk/ema-lightning">EMA Lightning</a> architecture. Limits: one designed female voice; mild emotions; questions that rise only at the end ("Yarın geliyorsun?") do not rise; long texts sound less natural than the big model.',
    linkCode: 'Code (GitHub)',
    linkCep: 'Ahenk Cep model (Hugging Face)',
    linkBig: 'The big model, Ahenk (Hugging Face)',
    contact: 'Contact:',
    license: 'Code Apache-2.0, model weights CC-BY-4.0. Base model EMA Lightning (Apache-2.0).',
    tech: (i) => `${i.backend === 'webgpu' ? 'WebGPU' : `WebAssembly, ${i.threads} thread${i.threads > 1 ? 's' : ''}`} · ${i.variant} · ${fmt(i.bytes / 1e6, 1)} MB`,
    langButton: 'TR',
    langLabel: 'Türkçeye geç',
    wavComment: 'AI-generated speech (Ahenk Cep, designed voice f022; not a real person).',
  },
};

let lang = 'tr';
const $ = (id) => document.getElementById(id);
function fmt(x, d) {
  return Number(x).toLocaleString(lang === 'tr' ? 'tr-TR' : 'en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
}
const t = (k, ...a) => (typeof T[lang][k] === 'function' ? T[lang][k](...a) : T[lang][k]);

const state = { ready: false, info: null, busy: false, pending: null, nextId: 1, url: null, loadStage: 'start', loaded: 0, total: 0 };

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch { /* storage blocked: the page still works */ }
  return null;
}

// ---- rendering ---------------------------------------------------------------------------------------------------
function renderChips() {
  const box = $('chips');
  const current = box.querySelector('input:checked')?.value || 'neutral';
  box.innerHTML = '';
  for (const s of STYLES) {
    const label = document.createElement('label');
    label.className = 'chip';
    label.innerHTML = `<input type="radio" name="style" value="${s.id}"${s.id === current ? ' checked' : ''}><span>${s[lang]}</span>`;
    box.appendChild(label);
  }
}

function renderExamples() {
  const ul = $('example-list');
  ul.innerHTML = '';
  for (const ex of EXAMPLES) {
    const li = document.createElement('li');
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'example';
    const style = STYLES.find((s) => s.id === ex.style)[lang];
    b.innerHTML = '<span class="ex-text"></span><span class="tag"></span>';
    b.querySelector('.ex-text').textContent = ex.text;
    b.querySelector('.tag').textContent = `${ex[lang]} · ${style}`;
    b.addEventListener('click', () => {
      $('text').value = ex.text;
      updateCount();
      const r = document.querySelector(`input[name="style"][value="${ex.style}"]`);
      if (r) r.checked = true;
      $('text').scrollIntoView({ behavior: 'smooth', block: 'center' });
      speak();
    });
    li.appendChild(b);
    ul.appendChild(li);
  }
}

function renderLoad() {
  const el = $('load');
  if (state.ready) {
    el.classList.add('done');
    return;
  }
  const key = { start: 'loadStart', download: state.fromCache ? 'loadCache' : 'loadDownload', sessions: 'loadSessions', front: 'loadFront', warm: 'loadWarm', error: 'loadError' }[state.loadStage];
  $('load-text').textContent = t(key) + (state.loadError ? `: ${state.loadError}` : '');
  el.classList.toggle('error', state.loadStage === 'error');
  if (state.total) {
    const pct = Math.min(100, (100 * state.loaded) / state.total);
    $('load-num').textContent = `${fmt(state.loaded / 1e6, 1)} / ${fmt(state.total / 1e6, 1)} MB`;
    $('load-bar').style.width = `${state.loadStage === 'download' ? pct : 100}%`;
    el.querySelector('.bar').setAttribute('aria-valuenow', String(Math.round(pct)));
  }
}

function applyLang() {
  document.documentElement.lang = lang;
  for (const el of document.querySelectorAll('[data-i18n]')) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll('[data-i18n-html]')) el.innerHTML = t(el.dataset.i18nHtml);
  $('lang').textContent = t('langButton');
  $('lang').setAttribute('aria-label', t('langLabel'));
  $('speed-val').textContent = fmt(Number($('speed').value), 2);
  if (state.info) {
    $('tech').textContent = t('tech', state.info);
    const ms = $('model-size');
    if (ms) ms.textContent = `${fmt(state.info.bytes / 1e6, 0)} MB`;
  }
  if (state.lastStats) $('stats').textContent = t('stats', state.lastStats);
  renderChips();
  renderExamples();
  renderLoad();
  if (!state.busy) setStatus('');
}

function setStatus(msg, error = false) {
  $('status').textContent = msg;
  $('status').classList.toggle('error', error);
}

function updateCount() {
  const n = $('text').value.length;
  $('count').textContent = `${n} / ${MAX_CHARS}`;
}

// ---- worker ------------------------------------------------------------------------------------------------------
const opts = pageOptions(window.location);
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

worker.onmessage = ({ data }) => {
  if (data.type === 'status') {
    if (data.stage === 'download') {
      state.loadStage = 'download';
      state.loaded = data.loaded;
      state.total = data.total;
    } else if (data.stage === 'sessions') state.loadStage = 'sessions';
    renderLoad();
  } else if (data.type === 'ready') {
    state.ready = true;
    state.info = data.info;
    $('go').disabled = false;
    $('tech').textContent = t('tech', data.info);
    const ms = $('model-size');
    if (ms) ms.textContent = `${fmt(data.info.bytes / 1e6, 0)} MB`;
    window.__ahenk = { info: data.info, runs: [] };
    renderLoad();
    if (state.pending) {
      const p = state.pending;
      state.pending = null;
      send(p);
    }
  } else if (data.type === 'progress') {
    if (data.id === state.current) setStatus(t('progress', data));
  } else if (data.type === 'audio') {
    if (data.id !== state.current) return;
    finish(data);
  } else if (data.type === 'error') {
    if (data.stage === 'init') {
      state.loadStage = 'error';
      state.loadError = data.message;
      renderLoad();
      setStatus(t('loadError'), true);
      $('go').disabled = true;
    } else {
      state.busy = false;
      $('go').disabled = false;
      $('go').textContent = t('go');
      setStatus(data.message, true);
    }
  }
};

worker.postMessage({ type: 'init', modelBase: opts.modelBase, pyBase: new URL('py/', document.baseURI).href, variant: opts.variant, ep: opts.ep });

function send(req) {
  state.current = req.id;
  state.busy = true;
  $('go').disabled = true;
  $('go').textContent = t('busy');
  setStatus(state.ready ? '' : t('waiting'));
  worker.postMessage({ type: 'say', ...req });
}

function speak() {
  if (state.busy) return;
  const text = $('text').value.trim();
  if (!text) return setStatus(t('empty'), true);
  if (text.length > MAX_CHARS) return setStatus(t('tooLong'), true);
  const style = document.querySelector('input[name="style"]:checked')?.value || 'neutral';
  const seed = Math.max(0, Math.min(2147483647, Math.floor(Number($('seed').value) || 0)));
  const req = { id: state.nextId++, text, style, steps: Number($('steps').value), speed: Number($('speed').value), seed };
  state.req = req;
  if (!state.ready) {
    state.pending = req;
    state.current = req.id;
    state.busy = true;
    $('go').textContent = t('busy');
    setStatus(t('waiting'));
    return;
  }
  send(req);
}

function finish(data) {
  state.busy = false;
  $('go').disabled = false;
  $('go').textContent = t('go');
  const seconds = data.audio.length / data.sampleRate;
  if (!data.audio.length) {
    setStatus(t('nothing'), true);
    return;
  }
  setStatus('');
  const blob = encodeWav(data.audio, data.sampleRate, { ISFT: 'Ahenk Cep', ICMT: t('wavComment') });
  if (state.url) URL.revokeObjectURL(state.url);
  state.url = URL.createObjectURL(blob);
  const player = $('player');
  player.src = state.url;
  $('download').href = state.url;
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  $('download').download = `ahenk-cep-${state.req.style}-${stamp}.wav`;
  state.lastStats = { seconds, wall: data.ms / 1000, rtf: data.ms / 1000 / seconds };
  $('stats').textContent = t('stats', state.lastStats);
  $('spoken').textContent = data.spoken;
  $('result').hidden = false;
  if (window.__ahenk) window.__ahenk.runs.push({ ...state.req, seconds, ms: data.ms, rtf: state.lastStats.rtf, samples: data.audio.length });
  player.play().catch(() => { /* autoplay blocked: the controls are there */ });
}

// ---- wiring ------------------------------------------------------------------------------------------------------
const asked = new URLSearchParams(location.search).get('lang');
const saved = store('ahenk-lang');
if (asked === 'en' || asked === 'tr') lang = asked;
else if (saved === 'en' || saved === 'tr') lang = saved;

$('lang').addEventListener('click', () => {
  lang = lang === 'tr' ? 'en' : 'tr';
  store('ahenk-lang', lang);
  applyLang();
});
$('go').addEventListener('click', speak);
$('text').addEventListener('input', updateCount);
$('text').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    speak();
  }
});
$('speed').addEventListener('input', () => { $('speed-val').textContent = fmt(Number($('speed').value), 2); });
$('go').disabled = false; // a click before the model is ready queues the request
$('link-github').href = LINKS.github;
$('link-cep').href = LINKS.hfCep;
$('link-big').href = LINKS.hfBig;

updateCount();
applyLang();
