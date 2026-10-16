// Ahenk Cep's sampling loop around four ONNX graphs. Mirrors tts/small/ahenk_cep.py inside EMA Lightning's engine
// (frame timeline, 16 Euler steps with classifier-free guidance, 100-frame decoder windows with 8 frames of
// context, pauses between pieces) and tts/small/web_pipeline.py, which it is checked against.
// Works in a browser worker and in Node (tools/web/parity_node.mjs): `ort` is onnxruntime-web passed in.

import { pieceSeed, torchRandn } from './randn.js';

export const GRAPHS = ['text', 'cond', 'step', 'decoder'];

// Which file of each graph a variant uses (manifest.files[graph][kind]).
export const VARIANTS = {
  fp32: { text: 'fp32', cond: 'fp32', step: 'fp32', decoder: 'fp32' },
  f16: { text: 'f16', cond: 'f16', step: 'f16', decoder: 'f16' },
  int8: { text: 'int8', cond: 'int8', step: 'int8', decoder: 'f16' },
};

export function variantFiles(manifest, variant) {
  const pick = VARIANTS[variant] || VARIANTS.fp32;
  return GRAPHS.map((g) => ({ graph: g, ...manifest.files[g][pick[g]] }));
}

// round half to even, like torch.round
export function roundHalfEven(x) {
  const fl = Math.floor(x);
  const d = x - fl;
  if (d > 0.5) return fl + 1;
  if (d < 0.5) return fl;
  return fl % 2 === 0 ? fl : fl + 1;
}

// Engine.plan for one unpadded piece: duration / speed, frames per word, the word (fw) and in-word position (fp)
// of every frame.
export function plan(dur, cw, speed, maxWord = 250, maxFrames = 3000) {
  const L = cw.length;
  const s = Math.fround(speed);
  const d = new Float32Array(L);
  for (let i = 0; i < L; i++) d[i] = dur[i] / s;
  const nWords = cw[L - 1] + 1;
  const sums = new Float32Array(nWords);
  for (let i = 0; i < L; i++) sums[cw[i]] = sums[cw[i]] + d[i];
  const n = new Int32Array(nWords);
  let total = 0;
  for (let w = 0; w < nWords; w++) {
    n[w] = Math.min(maxWord, Math.max(1, roundHalfEven(sums[w])));
    total += n[w];
  }
  const T = Math.min(total, maxFrames);
  const fw = new BigInt64Array(T);
  const fp = new Float32Array(T);
  let k = 0;
  for (let w = 0; w < nWords && k < T; w++) {
    for (let j = 0; j < n[w] && k < T; j++, k++) {
      fw[k] = BigInt(w);
      fp[k] = j / n[w];
    }
  }
  return { dur: d, fw, fp, frames: T };
}

export function windows(frames, first = 100, size = 100) {
  const spans = [];
  let s = 0;
  while (s < frames) {
    const e = Math.min(frames, s + (s === 0 ? first : size));
    spans.push([s, e]);
    s = e;
  }
  return spans;
}

const i64 = (arr) => BigInt64Array.from(arr, (v) => BigInt(v));

export class CepEngine {
  // sessions: {text, cond, step, decoder} onnxruntime InferenceSessions; prepare(text, speed) -> {spoken, pieces}
  constructor(ort, sessions, manifest, prepare) {
    this.ort = ort;
    this.s = sessions;
    this.m = manifest;
    this.prepare = prepare;
  }

  tensor(type, data, dims) {
    return new this.ort.Tensor(type, data, dims);
  }

  async latents(p, style, speed, seed, steps, cfg, tick) {
    const L = p.ids.length;
    const ids = this.tensor('int64', i64(p.ids), [1, L]);
    const cw = this.tensor('int64', i64(p.cw), [1, L]);
    const wstart = this.tensor('int64', i64(p.wstart), [1, L]);
    const t1 = await this.s.text.run({ ids, style: this.tensor('int64', BigInt64Array.of(BigInt(style)), [1]) });
    const pl = plan(t1.dur.data, p.cw, speed, this.m.limits.max_word_frames, this.m.limits.max_frames);
    const T = pl.frames;
    const c = await this.s.cond.run({
      h: t1.h, dur: this.tensor('float32', pl.dur, [1, L]), cw, wstart,
      fw: this.tensor('int64', pl.fw, [1, T]), fp: this.tensor('float32', pl.fp, [1, T]),
    });
    const D = this.m.latent_dim;
    let x = this.tensor('float32', torchRandn(seed, T * D), [1, T, D]);
    const dt = this.tensor('float32', Float32Array.of(1 / steps), [1]);
    const g = this.tensor('float32', Float32Array.of(cfg), [1]);
    for (let k = 0; k < steps; k++) {
      const out = await this.s.step.run({ x, cond: c.cond, t: this.tensor('float32', Float32Array.of(k / steps), [1]), dt, cfg: g });
      x = out.x_next;
      if (tick) tick(k + 1, steps);
    }
    return { latents: x.data, frames: T };
  }

  async decode(lat, frames) {
    const { window, context } = this.m.limits;
    const D = this.m.latent_dim;
    const hop = this.m.hop;
    const parts = [];
    for (const [s, e] of windows(frames, window, window)) {
      const a = Math.max(0, s - context);
      const b = Math.min(frames, e + context);
      const len = b - a;
      const z = new Float32Array(D * len);
      for (let t = 0; t < len; t++) for (let c = 0; c < D; c++) z[c * len + t] = lat[(a + t) * D + c];
      const out = await this.s.decoder.run({ z: this.tensor('float32', z, [1, D, len]) });
      parts.push(out.audio.data.slice((s - a) * hop, (e - a) * hop));
    }
    return parts;
  }

  // -> {audio: Float32Array, sampleRate, spoken, pieces}
  async say(text, { style = 'neutral', steps = 16, cfg = 2.0, speed = 1.0, seed = 0, onProgress } = {}) {
    const styleId = this.m.styles.indexOf(style);
    if (styleId < 0) throw new Error(`unknown style ${style}`);
    const prep = await this.prepare(text, speed);
    const sr = this.m.sample_rate;
    const chunks = [];
    const n = prep.pieces.length;
    for (let i = 0; i < n; i++) {
      const p = prep.pieces[i];
      const tick = onProgress ? (k, K) => onProgress({ piece: i, pieces: n, step: k, steps: K }) : null;
      const { latents, frames } = await this.latents(p, styleId, speed, pieceSeed(seed, i), steps, cfg, tick);
      if (onProgress) onProgress({ piece: i, pieces: n, step: steps, steps, decoding: true });
      chunks.push(...(await this.decode(latents, frames)));
      if (p.pause) chunks.push(new Float32Array(Math.round(p.pause * sr)));
    }
    let total = 0;
    for (const c of chunks) total += c.length;
    const audio = new Float32Array(total);
    let o = 0;
    for (const c of chunks) {
      audio.set(c, o);
      o += c.length;
    }
    return { audio, sampleRate: sr, spoken: prep.spoken, pieces: prep.pieces.map((q) => q.text) };
  }
}
