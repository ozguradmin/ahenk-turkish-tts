// torch.randn's CPU noise from a seed, so a seed gives the same voice here as in PyTorch (tts/small/ahenk_cep.py).
// mt19937 seeded like at::mt19937 (low 32 bits of the seed), 24-bit uniforms, then ATen's normal_fill: each block of
// 16 uniforms becomes 16 normals by Box-Muller, pairing value j with j + 8, in float32.
// Checked against torch in tests/test_web_demo.py (through tools/web/parity_node.mjs).

export class MT19937 {
  constructor(seed32) {
    this.mt = new Uint32Array(624);
    this.mt[0] = seed32 >>> 0;
    for (let j = 1; j < 624; j++) {
      const p = this.mt[j - 1] ^ (this.mt[j - 1] >>> 30);
      this.mt[j] = (Math.imul(1812433253, p) + j) >>> 0;
    }
    this.index = 624;
  }

  twist() {
    const mt = this.mt;
    for (let i = 0; i < 624; i++) {
      const y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
      let v = mt[(i + 397) % 624] ^ (y >>> 1);
      if (y & 1) v ^= 0x9908b0df;
      mt[i] = v >>> 0;
    }
    this.index = 0;
  }

  next() {
    if (this.index >= 624) this.twist();
    let y = this.mt[this.index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
}

const f = Math.fround;

// The first n values (n a multiple of 16) of torch.randn(shape, generator=Generator().manual_seed(seed)).
export function torchRandn(seed, n) {
  if (n % 16 !== 0) throw new Error('n must be a multiple of 16');
  const low = Number(BigInt(seed) & 0xffffffffn);
  const gen = new MT19937(low);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (gen.next() & 0xffffff) * 5.9604644775390625e-8; // 2 ** -24
  for (let b = 0; b < n; b += 16) {
    for (let j = 0; j < 8; j++) {
      const u1 = f(1 - out[b + j]);
      const u2 = out[b + j + 8];
      const radius = f(Math.sqrt(f(-2 * f(Math.log(u1)))));
      const theta = f(2 * Math.PI * u2);
      out[b + j] = f(radius * f(Math.cos(theta)));
      out[b + j + 8] = f(radius * f(Math.sin(theta)));
    }
  }
  return out;
}

// EMA's per-piece seed: (seed * 1_000_003 + i) mod 2**63.
export function pieceSeed(seed, i) {
  return (BigInt(seed) * 1000003n + BigInt(i)) % (1n << 63n);
}
