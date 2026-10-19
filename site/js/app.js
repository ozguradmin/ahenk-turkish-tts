// The page: Turkish first, English on request. All synthesis happens in js/worker.js; js/player.js draws and plays.
// Ahenk Cep is used in its neutral style only: the small model's other styles are too weak to offer yet.

import { deviceInfo, shortError } from './backend.js';
import { LINKS, MAX_CHARS, SAMPLES, bigSampleBases, pageOptions } from './config.js';
import { WavePlayer, redrawAll } from './player.js';
import { encodeWav } from './wav.js';

const STYLE = 'neutral';

const T = {
  tr: {
    skip: 'Metin alanına geç',
    navLabel: 'Sayfa',
    navExamples: 'Örnekler',
    navBig: 'Duygulu konuşma',
    navHow: 'Nasıl çalışır',
    eyebrow: 'Türkçe metinden sese',
    title: 'Türkçe metni, <em>kendi cihazınızda</em> sese çevirin.',
    lede: 'Ahenk Cep, 8,6 milyon parametrelik küçük bir Türkçe ses modeli. Bu sayfada tamamen tarayıcınızda çalışır; yazdığınız metin hiçbir sunucuya gönderilmez.',
    fact1h: 'Yerel',
    fact1: 'Metin de ses de cihazınızda kalır.',
    fact2h: 'Hafif',
    fact2: 'Model yaklaşık <span class="model-size num">17 MB</span>, bir kez iner.',
    fact3h: 'Okunuşa duyarlı',
    fact3: 'Tarih, saat, para ve telefon numaralarını okunuşa çevirir.',
    noticeTitle: 'Bu ses yapaydır.',
    noticeBody: 'Duyacağınız ses (f022) bir metin tarifinden tasarlandı; gerçek bir kişinin sesi ya da kaydı değildir. Ürettiğiniz sesi paylaşırken yapay olduğunu belirtin ve kimseyi taklit etmek ya da kandırmak için kullanmayın.',
    loadStart: 'Model hazırlanıyor',
    loadDownload: 'Model indiriliyor',
    loadCache: 'Model önbellekten açılıyor',
    loadSessions: 'Model açılıyor',
    loadFront: 'Metin işleyici yükleniyor',
    loadWarm: 'Son hazırlık',
    loadReady: 'Model hazır',
    loadError: 'Model yüklenemedi',
    errNetwork: 'Model indirilemedi. Bağlantınızı kontrol edip tekrar deneyin; hazır örnekleri yine de dinleyebilirsiniz.',
    errEngine: 'Model bu tarayıcıda çalıştırılamadı. Sayfayı başka bir tarayıcıda açmayı deneyin; hazır örnekleri yine de dinleyebilirsiniz.',
    errEngineIos: "Model bu tarayıcıda çalıştırılamadı. Sayfayı bir uygulamanın içinden açtıysanız Safari'de açmayı deneyin; hazır örnekleri yine de dinleyebilirsiniz.",
    errMemory: 'Cihazın belleği modeli açmaya yetmedi. Diğer sekmeleri kapatıp tekrar deneyin.',
    errTimeout: 'Model hazırlığı çok uzun sürdüğü için durduruldu. Tekrar deneyin; olmazsa bağlantınızı kontrol edin.',
    errWorker: 'Sayfanın bir parçası yüklenemedi. Tekrar deneyin, olmazsa sayfayı yenileyin.',
    detail: 'Ayrıntı',
    retry: 'Tekrar dene',
    textLabel: 'Seslendirilecek metin',
    placeholder: 'Bir şeyler yazın ya da aşağıdaki örneklerden birini seçin.',
    textHint: '<kbd>Ctrl</kbd> + <kbd>Enter</kbd> ile de seslendirebilirsiniz.',
    okurHint: `Daha uzun bir metin ya da PDF, Word, EPUB belge mi? <a href="${LINKS.okur}">Ahenk Okur</a> ile tamamını dinleyin ve indirin.`,
    settings: 'Ayarlar',
    steps: 'Kalite',
    stepsBest: 'En iyi (16 adım)',
    stepsFast: 'Hızlı (8 adım)',
    stepsFaster: 'Çok hızlı (4 adım, daha az doğal)',
    speed: 'Hız',
    seed: 'Tohum',
    seedHint: 'Aynı metin ve tohum her zaman aynı sesi verir.',
    go: 'Seslendir',
    busy: 'Seslendiriliyor',
    waiting: 'Model hazır olunca başlayacak.',
    empty: 'Önce bir metin yazın.',
    tooLong: `En çok ${MAX_CHARS} karakter yazabilirsiniz.`,
    nothing: 'Bu metinde okunacak bir şey bulunamadı.',
    failed: 'Seslendirme başarısız oldu.',
    failedMemory: 'Seslendirme başarısız oldu: cihazın belleği yetmedi. Daha kısa bir metin deneyin.',
    failedStuck: 'Seslendirme yanıt vermedi. Model yeniden hazırlanıyor, birazdan tekrar deneyin.',
    tapPlay: 'Ses hazır. Dinlemek için oynat düğmesine basın.',
    playFailed: 'Ses burada çalınamadı. WAV dosyasını indirip dinleyebilirsiniz.',
    progress: (p) => p.decoding ? `Ses oluşturuluyor${p.pieces > 1 ? ` (parça ${p.piece + 1}/${p.pieces})` : ''}` : `Seslendiriliyor${p.pieces > 1 ? `: parça ${p.piece + 1}/${p.pieces},` : ':'} adım ${p.step}/${p.steps}`,
    emptyPlayer: 'Seslendirdiğiniz metin burada dinlenebilir.',
    play: 'Oynat',
    pause: 'Duraklat',
    seek: 'Konum',
    listen: (x) => `Dinle: ${x}`,
    stop: (x) => `Duraklat: ${x}`,
    download: 'WAV indir',
    spokenLabel: 'Modelin okuduğu metin',
    stats: (s) => `${fmt(s.seconds, 1)} sn ses, ${fmt(s.wall, 1)} sn'de üretildi`,
    neutralNote: "Ahenk Cep şimdilik nötr konuşur; duygulu konuşma (kızgın, neşeli, şaşkın, gülme...) büyük model Ahenk'te var.",
    neutralCard: 'Model kartı',
    neutralSamples: 'Örnek kayıtlar',
    examples: 'Hazır örnekler',
    examplesLede: 'Her biri Ahenk Cep ile önceden seslendirildi; dinlemek için modelin inmesini beklemeniz gerekmez.',
    examplesError: 'Örnek kayıtlar yüklenemedi. Metni yazıp kendiniz seslendirebilirsiniz.',
    spokenPrefix: 'Okunuşu:',
    use: 'Bu metni kullan',
    useLabel: (x) => `Bu metni kullan: ${x}`,
    bigTitle: 'Duygulu konuşma büyük modelde',
    bigLede: "Ahenk Cep, büyük Ahenk modelinin sesinden öğrendi. Ahenk aynı tasarlanmış sesle gülerek, neşeyle ya da sakin bir tonla konuşabilir; şaşkınlık, kızgınlık ve üzüntü de var. Bu kayıtlar Ahenk'in model kartından.",
    bigUnavailable: 'Bu kayıt şu an açılamıyor. Model sayfası herkese açıldığında burada dinlenebilecek.',
    bigMore: 'Ahenk model kartı (Hugging Face)',
    howTitle: 'Nasıl çalışır',
    how1h: 'Okunuş',
    how1: 'Metin önce <a href="https://github.com/ozguradmin/ahenk-turkish-tts/blob/main/docs/TRNORM.md">trnorm</a> ile okunuşa çevrilir: “14.10.2026” olur “on dört Ekim iki bin yirmi altı”.',
    how2h: 'Ses',
    how2: 'Model, okunuş metninden 48 kHz ses üretir. Aynı metin ve tohum her zaman aynı sesi verir.',
    how3h: 'Cihazda',
    how3: 'Hepsi tarayıcınızda, WebAssembly ya da WebGPU ile çalışır. İlk açılışta model (<span class="model-size num">17 MB</span>) ve metin işleyici (yaklaşık 6 MB) iner; sonraki ziyaretlerde önbellekten açılır.',
    limits: 'Model, büyük Ahenk modelinin sesinden öğrenen bir öğrencidir ve <a href="https://huggingface.co/canberkkkkkk/ema-lightning">EMA Lightning</a> mimarisini kullanır. Sınırları: tek bir tasarlanmış kadın sesi var; şimdilik yalnızca nötr konuşur; sonu yükselen sorular (“Yarın geliyorsun?”) yükselmez; uzun metinlerde doğallık büyük modelin gerisinde kalır.',
    linkCode: 'Kod',
    linkCep: 'Ahenk Cep modeli',
    linkBig: 'Büyük model Ahenk',
    contact: 'İletişim:',
    license: 'Kod Apache-2.0, model ağırlıkları CC-BY-4.0. Temel model EMA Lightning (Apache-2.0).',
    tech: (i) => `${i.backend === 'webgpu' ? 'WebGPU' : `WebAssembly, ${i.threads} iş parçacığı`} · ${i.variant} · ${fmt(i.bytes / 1e6, 1)} MB`,
    langButton: 'EN',
    langLabel: 'Switch to English',
    themeDark: 'Koyu temaya geç',
    themeLight: 'Açık temaya geç',
    wavComment: 'Yapay zeka ile üretilmiş ses (Ahenk Cep, tasarlanmış ses f022; gerçek bir kişi değildir).',
  },
  en: {
    skip: 'Skip to the text box',
    navLabel: 'Page',
    navExamples: 'Examples',
    navBig: 'Expressive speech',
    navHow: 'How it works',
    eyebrow: 'Turkish text to speech',
    title: 'Turn Turkish text into speech, <em>on your own device</em>.',
    lede: 'Ahenk Cep is a small Turkish speech model with 8.6 million parameters. On this page it runs entirely in your browser; the text you type is never sent to a server.',
    fact1h: 'Local',
    fact1: 'Your text and the audio stay on your device.',
    fact2h: 'Light',
    fact2: 'The model is about <span class="model-size num">17 MB</span> and downloads once.',
    fact3h: 'Reads it right',
    fact3: 'Dates, times, money and phone numbers are turned into words.',
    noticeTitle: 'This voice is synthetic.',
    noticeBody: 'The voice you hear (f022) was designed from a text description; it is not a real person or a recording of one. Say that it is AI-generated when you share what you make, and never use it to impersonate or deceive anyone.',
    loadStart: 'Preparing the model',
    loadDownload: 'Downloading the model',
    loadCache: 'Opening the model from cache',
    loadSessions: 'Opening the model',
    loadFront: 'Loading the text processor',
    loadWarm: 'Almost ready',
    loadReady: 'Model ready',
    loadError: 'The model could not be loaded',
    errNetwork: 'The model could not be downloaded. Check your connection and try again; the ready-made examples still play.',
    errEngine: 'The model could not run in this browser. Try opening the page in another browser; the ready-made examples still play.',
    errEngineIos: 'The model could not run in this browser. If you opened the page inside an app, try opening it in Safari; the ready-made examples still play.',
    errMemory: 'This device ran out of memory while opening the model. Close other tabs and try again.',
    errTimeout: 'Preparing the model took too long and was stopped. Try again; if that fails, check your connection.',
    errWorker: 'Part of the page could not be loaded. Try again, or reload the page.',
    detail: 'Details',
    retry: 'Try again',
    textLabel: 'Text to speak (Turkish)',
    placeholder: 'Type something, or pick one of the examples below.',
    textHint: '<kbd>Ctrl</kbd> + <kbd>Enter</kbd> also speaks.',
    okurHint: `A longer text, or a PDF, Word or EPUB document? <a href="${LINKS.okur}">Ahenk Okur</a> reads all of it and lets you download it.`,
    settings: 'Settings',
    steps: 'Quality',
    stepsBest: 'Best (16 steps)',
    stepsFast: 'Fast (8 steps)',
    stepsFaster: 'Fastest (4 steps, less natural)',
    speed: 'Speed',
    seed: 'Seed',
    seedHint: 'The same text and seed always give the same audio.',
    go: 'Speak',
    busy: 'Speaking',
    waiting: 'Will start when the model is ready.',
    empty: 'Type some text first.',
    tooLong: `At most ${MAX_CHARS} characters.`,
    nothing: 'Nothing in this text can be read aloud.',
    failed: 'Speaking failed.',
    failedMemory: 'Speaking failed: the device ran out of memory. Try a shorter text.',
    failedStuck: 'Speaking got stuck. The model is being prepared again; try again in a moment.',
    tapPlay: 'The audio is ready. Press play to listen.',
    playFailed: 'The audio could not be played here. You can download the WAV and listen to it.',
    progress: (p) => p.decoding ? `Making the audio${p.pieces > 1 ? ` (part ${p.piece + 1}/${p.pieces})` : ''}` : `Speaking${p.pieces > 1 ? `: part ${p.piece + 1}/${p.pieces},` : ':'} step ${p.step}/${p.steps}`,
    emptyPlayer: 'What you speak will play here.',
    play: 'Play',
    pause: 'Pause',
    seek: 'Position',
    listen: (x) => `Play: ${x}`,
    stop: (x) => `Pause: ${x}`,
    download: 'Download WAV',
    spokenLabel: 'What the model reads',
    stats: (s) => `${fmt(s.seconds, 1)} s of audio made in ${fmt(s.wall, 1)} s`,
    neutralNote: 'Ahenk Cep speaks in a neutral tone for now; expressive speech (angry, cheerful, surprised, laughing...) is in the big model, Ahenk.',
    neutralCard: 'Model card',
    neutralSamples: 'Sample recordings',
    examples: 'Ready-made examples',
    examplesLede: 'Each was spoken in advance by Ahenk Cep, so you can listen without waiting for the model.',
    examplesError: 'The sample recordings could not be loaded. You can still type the text and speak it yourself.',
    spokenPrefix: 'Read as:',
    use: 'Use this text',
    useLabel: (x) => `Use this text: ${x}`,
    bigTitle: 'Expressive speech lives in the big model',
    bigLede: "Ahenk Cep learned from the voice of the big Ahenk model. Ahenk can speak in the same designed voice while laughing, cheerfully or calmly, and also surprised, angry or sad. These recordings come from Ahenk's model card.",
    bigUnavailable: 'This recording cannot be opened right now. It will play here once the model page is public.',
    bigMore: 'Ahenk model card (Hugging Face)',
    howTitle: 'How it works',
    how1h: 'Spoken form',
    how1: 'The text first goes through <a href="https://github.com/ozguradmin/ahenk-turkish-tts/blob/main/docs/TRNORM.md">trnorm</a>: “14.10.2026” becomes “on dört Ekim iki bin yirmi altı”.',
    how2h: 'Voice',
    how2: 'The model turns the spoken form into 48 kHz audio. The same text and seed always give the same audio.',
    how3h: 'On device',
    how3: 'Everything runs in your browser with WebAssembly or WebGPU. The first visit downloads the model (<span class="model-size num">17 MB</span>) and the text processor (about 6 MB); later visits open them from the cache.',
    limits: 'The model is a student that learned from the voice of the big Ahenk model and uses the <a href="https://huggingface.co/canberkkkkkk/ema-lightning">EMA Lightning</a> architecture. Limits: one designed female voice; neutral tone only for now; questions that rise only at the end (“Yarın geliyorsun?”) do not rise; long texts sound less natural than the big model.',
    linkCode: 'Code',
    linkCep: 'Ahenk Cep model',
    linkBig: 'The big model, Ahenk',
    contact: 'Contact:',
    license: 'Code Apache-2.0, model weights CC-BY-4.0. Base model EMA Lightning (Apache-2.0).',
    tech: (i) => `${i.backend === 'webgpu' ? 'WebGPU' : `WebAssembly, ${i.threads} thread${i.threads > 1 ? 's' : ''}`} · ${i.variant} · ${fmt(i.bytes / 1e6, 1)} MB`,
    langButton: 'TR',
    langLabel: 'Türkçeye geç',
    themeDark: 'Switch to dark theme',
    themeLight: 'Switch to light theme',
    wavComment: 'AI-generated speech (Ahenk Cep, designed voice f022; not a real person).',
  },
};

let lang = 'tr';
const $ = (id) => document.getElementById(id);
function fmt(x, d) {
  return Number(x).toLocaleString(lang === 'tr' ? 'tr-TR' : 'en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
}
const t = (k, ...a) => (typeof T[lang][k] === 'function' ? T[lang][k](...a) : T[lang][k]);

const state = { ready: false, info: null, busy: false, pending: null, nextId: 1, url: null, loadStage: 'start', loaded: 0, total: 0, heard: 0 };

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch { /* storage blocked: the page still works */ }
  return null;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

// A label that follows the language: both strings live on the element.
function bi(tag, cls, tr, en) {
  const e = el(tag, cls, lang === 'tr' ? tr : en);
  e.dataset.tr = tr;
  e.dataset.en = en;
  return e;
}

function playerMarkup(withTime = true) {
  const p = el('div', 'player');
  p.innerHTML = `<button class="play" type="button" disabled data-role="play"><svg class="ic ic-play" aria-hidden="true"><use href="#i-play"/></svg><svg class="ic ic-pause" aria-hidden="true"><use href="#i-pause"/></svg></button><div class="wave" data-role="wave" role="slider" tabindex="-1" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0"><canvas aria-hidden="true"></canvas></div>${withTime ? '<span class="time num" data-role="time">0:00</span>' : ''}`;
  return p;
}

// ---- players -----------------------------------------------------------------------------------------------------
const mainPlayer = new WavePlayer($('main-player'), {
  labels: () => ({ play: t('play'), pause: t('pause'), seek: t('seek') }),
  onState: (s) => {
    if (s === 'playing' && state.tapHint) {
      state.tapHint = false;
      setStatus('');
    }
  },
});

const cardPlayers = [];

function cardPlayer(card, label, opts = {}) {
  const pl = playerMarkup();
  card.appendChild(pl);
  const name = () => (typeof label === 'function' ? label() : label);
  const player = new WavePlayer(pl, {
    ...opts,
    labels: () => ({ play: t('listen', name()), pause: t('stop', name()), seek: `${t('seek')}: ${name()}` }),
    onState: (s) => {
      card.classList.toggle('is-playing', s === 'playing');
      if (opts.onState) opts.onState(s);
    },
  });
  cardPlayers.push(player);
  return player;
}

// ---- examples and the big model's samples ------------------------------------------------------------------------
const canOpus = (() => {
  try {
    return new Audio().canPlayType('audio/ogg; codecs=opus') !== '';
  } catch {
    return false;
  }
})();

function useText(text) {
  $('text').value = text;
  updateCount();
  if (!state.busy) clearStatus();
  $('composer').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center' });
  $('text').focus({ preventScroll: true });
}

function renderExamples(meta) {
  const ul = $('example-list');
  const base = new URL(SAMPLES.examples, document.baseURI);
  for (const ex of meta.examples) {
    const li = el('li', 'card');
    const top = el('div', 'card-top');
    top.append(bi('span', 'tag', ex.tr, ex.en));
    const text = el('p', 'card-text', ex.text);
    text.lang = 'tr';
    const spoken = el('p', 'card-spoken');
    const pre = el('b', '', t('spokenPrefix'));
    pre.dataset.i18n = 'spokenPrefix';
    const sp = el('span', '', ` ${ex.spoken}`);
    sp.lang = 'tr';
    spoken.append(pre, sp);
    li.append(top, text, spoken);
    const label = () => (lang === 'tr' ? ex.tr : ex.en);
    const player = cardPlayer(li, label);
    player.setSources([new URL(canOpus ? ex.ogg : ex.mp3, base).href]);
    player.setPeaks(ex.peaks, ex.seconds);
    const use = el('button', 'ghost use');
    use.type = 'button';
    use.innerHTML = '<svg class="ic" aria-hidden="true"><use href="#i-use"/></svg><span data-i18n="use"></span>';
    use.querySelector('span').textContent = t('use');
    use.dataset.ariaFn = 'useLabel';
    use.dataset.ariaArgTr = ex.tr;
    use.dataset.ariaArgEn = ex.en;
    use.setAttribute('aria-label', t('useLabel', label()));
    use.addEventListener('click', () => useText(ex.text));
    top.appendChild(use);
    ul.appendChild(li);
  }
}

async function firstReachable(urls) {
  for (const u of urls) {
    try {
      const r = await fetch(u, { method: 'HEAD', mode: 'cors', credentials: 'omit', cache: 'no-store' });
      if (r.ok) return u;
    } catch { /* try the next one */ }
  }
  return null;
}

function renderBig(meta) {
  const ul = $('big-list');
  const bases = bigSampleBases(window.location);
  for (const b of meta.big) {
    const li = el('li', 'card');
    const top = el('div', 'card-top');
    top.append(bi('p', 'big-label', b.tr, b.en));
    const text = el('p', 'card-text', b.text);
    text.lang = 'tr';
    li.append(top, text);
    const state = el('p', 'state');
    state.dataset.i18n = 'bigUnavailable';
    state.hidden = true;
    const markUnavailable = () => {
      li.classList.add('unavailable');
      state.textContent = t('bigUnavailable');
      state.hidden = false;
    };
    const player = cardPlayer(li, () => (lang === 'tr' ? b.tr : b.en), {
      crossOrigin: true,
      onState: (s) => { if (s === 'error') markUnavailable(); },
    });
    li.appendChild(state);
    if (b.peaks) player.setPeaks(b.peaks, b.seconds);
    ul.appendChild(li);
    firstReachable(bases.map((x) => new URL(b.file, x).href)).then((url) => {
      if (!url) return markUnavailable();
      player.setSources([url]);
      if (!b.peaks) player.setPeaks(new Float32Array(64).fill(0.15), b.seconds || 1);
      player.btn.disabled = false;
    });
  }
}

async function loadSamples() {
  try {
    const r = await fetch(new URL(`${SAMPLES.examples}samples.json`, document.baseURI));
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const meta = await r.json();
    renderExamples(meta);
    renderBig(meta);
  } catch {
    $('examples-error').hidden = false;
    $('duygu').hidden = true;
  }
}

// ---- rendering ---------------------------------------------------------------------------------------------------
function renderLoad() {
  const box = $('load');
  const bar = box.querySelector('.bar');
  box.classList.remove('indeterminate', 'error', 'done');
  $('retry').hidden = true;
  if (state.ready) {
    box.classList.add('done');
    $('load-text').textContent = t('loadReady');
    $('load-num').textContent = state.info ? `${state.info.backend === 'webgpu' ? 'WebGPU' : 'WebAssembly'} · ${fmt(state.info.bytes / 1e6, 1)} MB` : '';
    bar.setAttribute('aria-valuenow', '100');
    $('load-bar').style.width = '100%';
    return;
  }
  const key = { start: 'loadStart', download: state.fromCache ? 'loadCache' : 'loadDownload', sessions: 'loadSessions', front: 'loadFront', warm: 'loadWarm', error: 'loadError' }[state.loadStage];
  $('load-text').textContent = t(key);
  if (state.loadStage === 'error') {
    box.classList.add('error');
    $('load-num').textContent = '';
    $('retry').hidden = false;
    return;
  }
  if (state.loadStage === 'download' && state.total) {
    const pct = Math.min(100, (100 * state.loaded) / state.total);
    $('load-num').textContent = `${fmt(state.loaded / 1e6, 1)} / ${fmt(state.total / 1e6, 1)} MB`;
    $('load-bar').style.width = `${pct}%`;
    bar.setAttribute('aria-valuenow', String(Math.round(pct)));
  } else {
    box.classList.add('indeterminate');
    if (state.loadStage !== 'download') $('load-num').textContent = '';
  }
}

function setModelSize() {
  if (!state.info) return;
  for (const e of document.querySelectorAll('.model-size')) e.textContent = `${fmt(state.info.bytes / 1e6, 0)} MB`;
}

function effectiveTheme() {
  const th = document.documentElement.dataset.theme;
  if (th === 'light' || th === 'dark') return th;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function renderTheme() {
  $('theme').setAttribute('aria-label', effectiveTheme() === 'dark' ? t('themeLight') : t('themeDark'));
  redrawAll();
}

function applyLang() {
  document.documentElement.lang = lang;
  for (const e of document.querySelectorAll('[data-i18n]')) e.textContent = t(e.dataset.i18n);
  for (const e of document.querySelectorAll('[data-i18n-html]')) e.innerHTML = t(e.dataset.i18nHtml);
  for (const e of document.querySelectorAll('[data-i18n-placeholder]')) e.placeholder = t(e.dataset.i18nPlaceholder);
  for (const e of document.querySelectorAll('[data-i18n-aria]')) e.setAttribute('aria-label', t(e.dataset.i18nAria));
  for (const e of document.querySelectorAll('[data-tr]')) e.textContent = e.dataset[lang];
  for (const e of document.querySelectorAll('[data-aria-fn]')) e.setAttribute('aria-label', t(e.dataset.ariaFn, lang === 'tr' ? e.dataset.ariaArgTr : e.dataset.ariaArgEn));
  for (const e of document.querySelectorAll('.unavailable .state')) e.textContent = t('bigUnavailable');
  $('lang').textContent = t('langButton');
  $('lang').setAttribute('aria-label', t('langLabel'));
  $('speed-val').textContent = fmt(Number($('speed').value), 2);
  if (state.info) $('tech').textContent = t('tech', state.info);
  setModelSize();
  if (state.lastStats) $('stats').textContent = t('stats', state.lastStats);
  $('go').textContent = state.busy ? t('busy') : t('go');
  mainPlayer.sync();
  for (const p of cardPlayers) p.sync();
  renderLoad();
  renderTheme();
  if (state.tapHint) setStatus(t('tapPlay'));
  else if (!state.busy) clearStatus();
}

// A short message under Seslendir. `detail` (a technical error, cut to one line) goes into a closed <details>.
function setStatus(msg, error = false, detail = '') {
  const box = $('status');
  box.textContent = msg;
  box.classList.toggle('error', error);
  if (detail) {
    const d = el('details', 'detail');
    d.append(el('summary', '', t('detail')), el('code', '', shortError(detail)));
    box.append(' ', d);
  }
}

// Empty the status line, except that a failed load keeps saying so.
function clearStatus() {
  state.tapHint = false;
  if (state.loadStage === 'error') showLoadError();
  else setStatus('');
}

function updateCount() {
  const n = $('text').value.length;
  $('count').textContent = `${n} / ${MAX_CHARS}`;
  $('count').classList.toggle('over', n >= MAX_CHARS);
  $('okur-hint').classList.toggle('over', n >= MAX_CHARS); // at the limit, point to Ahenk Okur more clearly
}

function setBusy(busy, frac = 0) {
  state.busy = busy;
  const go = $('go');
  go.classList.toggle('busy', busy);
  go.style.setProperty('--p', `${Math.round(frac * 100)}%`);
  go.textContent = busy ? t('busy') : t('go');
  go.setAttribute('aria-busy', busy ? 'true' : 'false');
}

// ---- worker ------------------------------------------------------------------------------------------------------
// "Tekrar dene" starts a new worker instead of reloading the page: on phones a reload costs more memory than it frees.
// A watchdog turns silence into an error: no word from the worker for STALL_MS while the model loads or while it
// speaks means something is stuck (a hung download, a worker that died). While the model opens, after its download,
// the worker sends a beat every 10 s; that stage gets OPEN_MS in all (onnxruntime's and Pyodide's files come then).
const opts = pageOptions(window.location);
const device = deviceInfo(navigator, matchMedia('(pointer: coarse)').matches);
const STALL_MS = 120000;
const OPEN_MS = 600000;
const ERRORS = { network: 'errNetwork', engine: 'errEngine', memory: 'errMemory', timeout: 'errTimeout', worker: 'errWorker' };
let worker = null;

// What this tab has learned to avoid, in sessionStorage so that "Tekrar dene" and a reload remember it: threads that
// failed to start (then one thread), WebGPU after it failed.
const avoid = (() => {
  try {
    return JSON.parse(sessionStorage.getItem('ahenk-avoid')) || {};
  } catch {
    return {};
  }
})();
function remember(what) {
  avoid[what] = true;
  try {
    sessionStorage.setItem('ahenk-avoid', JSON.stringify(avoid));
  } catch { /* storage blocked: this page still remembers */ }
}

function onWorkerMessage(data) {
  state.heard = performance.now();
  if (data.type === 'status') {
    if (data.stage === 'download') {
      state.loadStage = 'download';
      state.loaded = data.loaded;
      state.total = data.total;
      if (data.loaded >= data.total && data.total) state.loadStage = 'sessions';
    } else if (data.stage === 'sessions') Object.assign(state, { loadStage: 'sessions', opening: true, threads: data.threads || 1 });
    else if (data.stage === 'front-ready') state.loadStage = 'warm';
    renderLoad();
  } else if (data.type === 'ready' || data.type === 'backend') {
    // 'backend': WebGPU failed on a request and the worker moved to WebAssembly
    if (data.info.gpuError) remember('webgpu');
    state.info = data.info;
    $('tech').textContent = t('tech', data.info);
    setModelSize();
    window.__ahenk = { runs: [], ...window.__ahenk, info: data.info };
    if (data.type === 'backend') return renderLoad();
    state.ready = true;
    $('go').disabled = false;
    renderLoad();
    if (state.pending) {
      const p = state.pending;
      state.pending = null;
      send(p);
    }
  } else if (data.type === 'progress') {
    if (data.id !== state.current || !state.busy) return;
    setStatus(t('progress', data));
    const per = data.decoding ? 1 : data.step / data.steps;
    setBusy(true, (data.piece + per * 0.92) / data.pieces);
  } else if (data.type === 'audio') {
    if (data.id !== state.current) return;
    finish(data);
  } else if (data.type === 'error') {
    if (data.stage === 'init') loadFailed(data.kind, data.message, data.threads);
    else if (data.id === state.current && state.busy) sayFailed(data.kind, data.message, data.broken);
  }
}

function startWorker() {
  if (worker) worker.terminate();
  worker = null;
  Object.assign(state, {
    ready: false, info: null, loadStage: 'start', loaded: 0, total: 0, fromCache: false, loadError: null,
    opening: false, openMs: 0, threads: 0,
  });
  state.heard = performance.now();
  renderLoad();
  let w;
  try {
    w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  } catch (e) {
    loadFailed('worker', String((e && e.message) || e));
    return;
  }
  worker = w;
  let started = false;
  w.onmessage = ({ data }) => {
    started = true;
    if (w === worker) onWorkerMessage(data);
  };
  // A worker whose script or imports cannot load (missing, blocked, broken) fires this before it says anything.
  // Once it has started, an uncaught error in it is not fatal by itself: its own messages or the watchdog tell.
  w.onerror = (e) => {
    if (w !== worker) return;
    const message = (e && e.message) || 'js/worker.js did not start';
    if (!started) loadFailed('worker', message);
    else console.warn('Ahenk Cep: uncaught error in the worker:', message);
  };
  w.postMessage({
    type: 'init', modelBase: opts.modelBase, pyBase: new URL('py/', document.baseURI).href, variant: opts.variant,
    ep: avoid.webgpu ? 'wasm' : opts.ep, device, maxThreads: avoid.threads ? 1 : 0,
  });
}

// A request that hung or broke the engine: a new worker (the model comes from the cache). A click meanwhile waits.
function restartWorker() {
  if (state.info && state.info.backend === 'webgpu') remember('webgpu');
  startWorker();
}

function showLoadError() {
  const { kind, message } = state.loadError;
  setStatus(t(kind === 'engine' && device.ios ? 'errEngineIos' : ERRORS[kind] || 'errEngine'), true, message);
}

// threads: how many WebAssembly threads the model failed to open on. Threads are new on iOS and untested there, so
// a failure or a hang with more than one gets one more try, at once, on a single thread.
function loadFailed(kind, message, threads = 0) {
  if (worker) worker.terminate(); // frees its memory; "Tekrar dene" starts a new one
  worker = null;
  if (threads > 1 && !avoid.threads) {
    console.warn(`Ahenk Cep: the model did not open on ${threads} threads (${message || kind}); trying one.`);
    remember('threads');
    startWorker();
    return;
  }
  state.ready = false;
  state.loadStage = 'error';
  state.loadError = { kind, message: message || '' };
  renderLoad();
  if (state.busy) {
    state.pending = null;
    setBusy(false);
  }
  $('go').disabled = true;
  showLoadError();
}

// broken: an abort or running out of memory left the WebAssembly module unusable; the next try gets a new one.
function sayFailed(kind, message, broken = false) {
  setBusy(false);
  $('go').disabled = false;
  setStatus(t(kind === 'memory' ? 'failedMemory' : 'failed'), true, message);
  if (broken) restartWorker();
}

setInterval(() => {
  if (!worker || document.hidden) return;
  const silent = performance.now() - state.heard >= STALL_MS;
  if (!state.ready) {
    if (state.opening) state.openMs += 5000;
    if (!silent && state.openMs < OPEN_MS) return;
    const why = silent ? `no word from the worker in ${STALL_MS / 1000} s` : `not ready ${OPEN_MS / 60000} min after the download`;
    loadFailed('timeout', why, state.opening ? state.threads : 0);
  } else if (state.busy && silent) {
    setBusy(false);
    restartWorker();
    $('go').disabled = false;
    setStatus(t('failedStuck'), true);
  }
}, 5000);
// Background tabs are throttled or frozen: that time does not count.
document.addEventListener('visibilitychange', () => { state.heard = performance.now(); });

// On a first visit coi-serviceworker.js registers itself and then reloads the page once to make it cross-origin
// isolated (see index.html). Starting the model before that reload would only download and allocate it twice (and
// iOS may not give the memory back), so wait: the reload is asked for as soon as the registration settles. Pages it
// already controls, or cannot help, start at once; a reload that never comes, after 10 s; anything else after 15 s.
function afterCoiReload() {
  const sw = navigator.serviceWorker;
  const coi = window.coiState || {};
  // the same conditions under which coi-serviceworker.js registers itself
  if (window.crossOriginIsolated !== false || !window.isSecureContext || !sw || sw.controller || coi.reloaded) return Promise.resolve();
  return new Promise((resolve) => {
    const t0 = performance.now();
    let settled = 0;
    const iv = setInterval(() => {
      const now = performance.now();
      const go = coi.reloading ? now - coi.reloading > 10000 : sw.controller || (settled && now - settled > 250) || now - t0 > 15000;
      if (go) {
        clearInterval(iv);
        resolve();
      }
    }, 50);
    // The same registration as coi-serviceworker.js's (an equal request joins it), which asks for the reload as soon
    // as it settles. If it fails, no reload is coming.
    const done = () => { settled = performance.now(); };
    sw.register(new URL('coi-serviceworker.js', document.baseURI).href).then(done, done);
  });
}

function send(req) {
  state.current = req.id;
  state.heard = performance.now();
  setBusy(true);
  $('go').disabled = true;
  setStatus(state.ready ? '' : t('waiting'));
  worker.postMessage({ type: 'say', ...req });
}

function speak() {
  if (state.busy) return;
  const text = $('text').value.trim();
  if (!text) {
    setStatus(t('empty'), true);
    $('text').focus();
    return;
  }
  if (text.length > MAX_CHARS) return setStatus(t('tooLong'), true);
  if (state.loadStage === 'error') return showLoadError();
  state.tapHint = false;
  mainPlayer.btn.classList.remove('nudge');
  mainPlayer.unlock(); // inside the tap: iOS would block the result, which starts playing after an await
  const seed = Math.max(0, Math.min(2147483647, Math.floor(Number($('seed').value) || 0)));
  const req = { id: state.nextId++, text, style: STYLE, steps: Number($('steps').value), speed: Number($('speed').value), seed };
  state.req = req;
  if (!state.ready) {
    state.pending = req;
    state.current = req.id;
    setBusy(true);
    setStatus(t('waiting'));
    return;
  }
  send(req);
}

function finish(data) {
  setBusy(false);
  $('go').disabled = false;
  const seconds = data.audio.length / data.sampleRate;
  if (!data.audio.length) {
    setStatus(t('nothing'), true);
    return;
  }
  setStatus('');
  const blob = encodeWav(data.audio, data.sampleRate, { ISFT: 'Ahenk Cep', ICMT: t('wavComment') });
  if (state.url) URL.revokeObjectURL(state.url);
  state.url = URL.createObjectURL(blob);
  $('result').classList.remove('is-empty');
  mainPlayer.setAudio(data.audio, data.sampleRate, state.url);
  $('download').href = state.url;
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  $('download').download = `ahenk-cep-${stamp}.wav`;
  state.lastStats = { seconds, wall: data.ms / 1000, rtf: data.ms / 1000 / seconds };
  $('stats').textContent = t('stats', state.lastStats);
  $('spoken').textContent = data.spoken;
  $('result-row').hidden = false;
  $('spoken-box').hidden = false;
  if (window.__ahenk) window.__ahenk.runs.push({ ...state.req, seconds, ms: data.ms, rtf: state.lastStats.rtf, samples: data.audio.length });
  // Never fail silently: if the browser wants a tap first, say so (the play button gets a ring).
  mainPlayer.play().then((r) => {
    if (r === 'blocked') {
      state.tapHint = true;
      setStatus(t('tapPlay'));
    } else if (r === 'failed') setStatus(t('playFailed'), true);
  });
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
$('theme').addEventListener('click', () => {
  const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  store('ahenk-theme', next);
  renderTheme();
});
matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', renderTheme);
$('settings-btn').addEventListener('click', () => {
  const open = $('settings').hidden;
  $('settings').hidden = !open;
  $('settings-btn').setAttribute('aria-expanded', String(open));
});
$('retry').addEventListener('click', () => {
  $('go').disabled = false;
  setStatus('');
  startWorker(); // last: if it fails at once, its error stays on screen
});
$('go').addEventListener('click', speak);
$('text').addEventListener('input', () => {
  updateCount();
  // a speaking error goes once the text changes; a failed load stays as it is (no redraw on every key)
  if ($('status').classList.contains('error') && !state.busy && state.loadStage !== 'error') clearStatus();
});
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
$('link-big-card').href = LINKS.hfBig;
$('link-big-more').href = LINKS.hfBig;
$('link-big-samples').href = LINKS.hfBigSamples;

updateCount();
applyLang();
loadSamples();
afterCoiReload().then(startWorker);
window.ahenkStarted = true; // index.html says the page did not load if this never runs
