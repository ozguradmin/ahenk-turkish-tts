// A small audio player with a drawn waveform: play/pause button, a seekable canvas (pointer and keyboard, exposed
// as a slider) and a time readout. One player plays at a time on the page.
//
//   const p = new WavePlayer(rootEl, { labels });   // rootEl holds [data-role=play|wave|time] and a <canvas>
//   p.setPeaks(peaks, seconds); p.setSources([url, ...]);    // pre-rendered clip
//   p.setAudio(float32, sampleRate, blobUrl);               // generated audio
//   p.unlock();                                             // in a tap, before audio that will play after an await

import { encodeWav } from './wav.js';

const players = new Set();
let silence = null;
const silenceUrl = () => (silence ||= URL.createObjectURL(encodeWav(new Float32Array(800), 8000))); // 0.1 s

export function fmtTime(s) {
  if (!Number.isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, '0')}`;
}

// Peak absolute value in n equal slices of a Float32Array, scaled so the loudest slice is 1.
export function computePeaks(audio, n = 600) {
  const out = new Float32Array(n);
  if (!audio.length) return out;
  const step = audio.length / n;
  let max = 1e-9;
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * step);
    const b = Math.max(a + 1, Math.floor((i + 1) * step));
    let p = 0;
    for (let j = a; j < b && j < audio.length; j++) {
      const v = Math.abs(audio[j]);
      if (v > p) p = v;
    }
    out[i] = p;
    if (p > max) max = p;
  }
  for (let i = 0; i < n; i++) out[i] /= max;
  return out;
}

function resample(peaks, n) {
  const out = new Float32Array(n);
  const step = peaks.length / n;
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * step);
    const b = Math.max(a + 1, Math.floor((i + 1) * step));
    let p = 0;
    for (let j = a; j < b && j < peaks.length; j++) p = Math.max(p, peaks[j]);
    out[i] = p;
  }
  return out;
}

const reduceMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export class WavePlayer {
  constructor(root, { labels, onState, crossOrigin = false } = {}) {
    this.root = root;
    this.btn = root.querySelector('[data-role=play]');
    this.wave = root.querySelector('[data-role=wave]');
    this.timeEl = root.querySelector('[data-role=time]');
    this.canvas = this.wave.querySelector('canvas');
    this.labels = labels || (() => ({ play: 'Play', pause: 'Pause', seek: 'Position' }));
    this.onState = onState || (() => {});
    this.audio = new Audio();
    this.audio.preload = 'none';
    if (crossOrigin) this.audio.crossOrigin = 'anonymous';
    this.peaks = null;
    this.duration = 0;
    this.grow = 1;
    this.sources = [];
    this.srcIndex = 0;
    players.add(this);

    this.btn.addEventListener('click', () => this.toggle());
    this.audio.addEventListener('play', () => this.sync());
    this.audio.addEventListener('pause', () => this.sync());
    this.audio.addEventListener('ended', () => { this.audio.currentTime = 0; this.sync(); });
    this.audio.addEventListener('waiting', () => this.btn.classList.add('loading'));
    this.audio.addEventListener('playing', () => this.btn.classList.remove('loading'));
    this.audio.addEventListener('loadedmetadata', () => {
      if (Number.isFinite(this.audio.duration) && this.audio.duration > 0) this.duration = this.audio.duration;
      this.updateTime();
    });
    this.audio.addEventListener('error', () => this.nextSource());

    let dragging = false;
    const seekAt = (e) => {
      const r = this.wave.getBoundingClientRect();
      this.seek(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * this.duration);
    };
    this.wave.addEventListener('pointerdown', (e) => {
      if (!this.ready()) return;
      dragging = true;
      this.wave.setPointerCapture(e.pointerId);
      seekAt(e);
    });
    this.wave.addEventListener('pointermove', (e) => { if (dragging) seekAt(e); });
    this.wave.addEventListener('pointerup', () => { dragging = false; });
    this.wave.addEventListener('pointercancel', () => { dragging = false; });
    this.wave.addEventListener('keydown', (e) => {
      if (!this.ready()) return;
      const t = this.audio.currentTime;
      const big = Math.max(1, this.duration / 10);
      const k = { ArrowRight: t + 1, ArrowUp: t + 1, ArrowLeft: t - 1, ArrowDown: t - 1, PageUp: t + big, PageDown: t - big, Home: 0, End: this.duration }[e.key];
      if (k !== undefined) {
        e.preventDefault();
        this.seek(k);
      } else if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        this.toggle();
      }
    });

    if ('ResizeObserver' in window) new ResizeObserver(() => this.draw()).observe(this.wave);
    this.sync();
  }

  ready() { return !!this.peaks && this.duration > 0 && this.sources.length > 0; }

  setPeaks(peaks, seconds, animate = false) {
    this.peaks = peaks instanceof Float32Array ? peaks : Float32Array.from(peaks);
    this.duration = seconds;
    this.wave.tabIndex = 0;
    this.btn.disabled = !this.sources.length;
    if (animate && !reduceMotion()) {
      const t0 = performance.now();
      const tick = (now) => {
        this.grow = Math.min(1, (now - t0) / 520);
        this.grow = 1 - (1 - this.grow) ** 3;
        this.draw();
        if (this.grow < 1) requestAnimationFrame(tick);
      };
      this.grow = 0;
      requestAnimationFrame(tick);
    } else {
      this.grow = 1;
      this.draw();
    }
    this.updateTime();
  }

  setSources(urls) {
    this.pause();
    this.silent = false;
    this.sources = urls;
    this.srcIndex = 0;
    if (urls.length) this.audio.src = urls[0];
    else this.audio.removeAttribute('src');
    this.btn.disabled = !urls.length || !this.peaks;
  }

  setAudio(samples, sampleRate, url) {
    this.setSources([url]);
    this.audio.preload = 'auto';
    this.setPeaks(computePeaks(samples), samples.length / sampleRate, true);
    this.btn.disabled = false;
  }

  nextSource() {
    if (!this.sources.length) return;
    this.srcIndex += 1;
    if (this.srcIndex < this.sources.length) {
      this.audio.src = this.sources[this.srcIndex];
      if (this.wantPlay) this.play();
    } else {
      this.btn.classList.remove('loading');
      this.btn.disabled = true;
      this.onState('error');
    }
  }

  // -> a promise of 'playing', 'blocked' (the browser wants a tap first; the button gets a ring), 'aborted' (paused
  // or replaced before it started) or 'failed'.
  play() {
    for (const p of players) if (p !== this) p.pause();
    this.wantPlay = true;
    this.btn.classList.add('loading');
    const r = this.audio.play();
    if (!r || !r.then) return Promise.resolve('playing');
    return r.then(() => {
      this.unlocked = true;
      return 'playing';
    }, (e) => {
      this.btn.classList.remove('loading');
      this.wantPlay = false;
      this.sync();
      const name = e && e.name;
      if (name === 'NotAllowedError') this.btn.classList.add('nudge');
      return name === 'NotAllowedError' ? 'blocked' : name === 'AbortError' ? 'aborted' : 'failed';
    });
  }

  // iOS lets a media element start by itself later (after an await, as when generated audio arrives) only if it
  // already played inside a tap. Called from the tap: a tenth of a second of silence plays and the element stays
  // allowed. A clip that is already loaded (one that was blocked) plays muted for a moment instead, and stays.
  unlock() {
    if (this.unlocked) return;
    this.unlocked = true;
    this.silent = true;
    const clip = !!this.audio.getAttribute('src');
    if (clip) this.audio.muted = true;
    else this.audio.src = silenceUrl();
    const r = this.audio.play();
    const end = (ok) => {
      if (!ok) this.unlocked = false;
      if (clip) {
        if (this.silent) {
          this.audio.pause();
          this.audio.currentTime = 0;
          this.silent = false;
        }
        this.audio.muted = false;
        this.sync();
      }
    };
    if (r && r.then) r.then(() => end(true), () => end(false));
    else end(true);
  }

  pause() {
    this.wantPlay = false;
    this.btn.classList.remove('loading');
    if (!this.audio.paused) this.audio.pause();
  }

  toggle() {
    if (this.audio.paused) this.play();
    else this.pause();
  }

  seek(t) {
    if (!this.duration) return;
    this.audio.currentTime = Math.min(this.duration, Math.max(0, t));
    this.updateTime();
    this.draw();
  }

  sync() {
    const playing = !this.audio.paused && !this.silent;
    this.btn.classList.toggle('playing', playing);
    if (playing) this.btn.classList.remove('loading', 'nudge');
    const L = this.labels();
    this.btn.setAttribute('aria-label', playing ? L.pause : L.play);
    this.wave.setAttribute('aria-label', L.seek);
    this.onState(playing ? 'playing' : 'paused');
    if (playing && !this.raf) {
      const loop = () => {
        this.updateTime();
        this.draw();
        this.raf = this.audio.paused ? 0 : requestAnimationFrame(loop);
      };
      this.raf = requestAnimationFrame(loop);
    }
    this.updateTime();
    this.draw();
  }

  updateTime() {
    const cur = this.audio.currentTime || 0;
    const started = !this.audio.paused || cur > 0;
    this.timeEl.textContent = started ? `${fmtTime(cur)} / ${fmtTime(this.duration)}` : fmtTime(this.duration);
    this.wave.setAttribute('aria-valuemax', String(Math.round(this.duration * 10) / 10));
    this.wave.setAttribute('aria-valuenow', String(Math.round(cur * 10) / 10));
    this.wave.setAttribute('aria-valuetext', `${fmtTime(cur)} / ${fmtTime(this.duration)}`);
  }

  draw() {
    const c = this.canvas;
    const w = this.wave.clientWidth;
    const h = this.wave.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    if (!this.peaks) return;
    const css = getComputedStyle(this.root);
    const played = css.getPropertyValue('--accent').trim() || '#a8432a';
    const rest = css.getPropertyValue('--wave').trim() || '#c9c0ae';
    const bar = w < 260 ? 2 : 2.5;
    const gap = w < 260 ? 2 : 2;
    const n = Math.max(8, Math.floor((w + gap) / (bar + gap)));
    const pk = resample(this.peaks, n);
    const frac = this.duration ? (this.audio.currentTime || 0) / this.duration : 0;
    const mid = h / 2;
    for (let i = 0; i < n; i++) {
      const x = i * (bar + gap);
      const amp = Math.max(1.5, pk[i] ** 0.8 * (h * 0.92) * this.grow);
      g.fillStyle = (i + 0.5) / n <= frac ? played : rest;
      const y = mid - amp / 2;
      if (g.roundRect) {
        g.beginPath();
        g.roundRect(x, y, bar, amp, bar / 2);
        g.fill();
      } else g.fillRect(x, y, bar, amp);
    }
  }
}

export function redrawAll() {
  for (const p of players) p.draw();
}
