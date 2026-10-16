// 16-bit mono PCM WAV, scaled like ema_lightning.audio.write_wav, with a RIFF INFO chunk that says the speech is
// synthetic (ISFT: software, ICMT: comment), so the label travels with the file.

function infoChunk(fields) {
  const enc = new TextEncoder();
  const parts = [];
  for (const [id, text] of Object.entries(fields)) {
    let bytes = enc.encode(text + '\0');
    if (bytes.length % 2) bytes = Uint8Array.from([...bytes, 0]);
    parts.push({ id, bytes });
  }
  const size = 4 + parts.reduce((n, p) => n + 8 + p.bytes.length, 0);
  const out = new Uint8Array(8 + size);
  const v = new DataView(out.buffer);
  out.set(enc.encode('LIST'), 0);
  v.setUint32(4, size, true);
  out.set(enc.encode('INFO'), 8);
  let o = 12;
  for (const p of parts) {
    out.set(enc.encode(p.id), o);
    v.setUint32(o + 4, p.bytes.length, true);
    out.set(p.bytes, o + 8);
    o += 8 + p.bytes.length;
  }
  return out;
}

export function encodeWav(audio, sampleRate, info = {}) {
  const list = Object.keys(info).length ? infoChunk(info) : new Uint8Array(0);
  const dataBytes = audio.length * 2;
  const buf = new ArrayBuffer(44 + list.length + dataBytes);
  const v = new DataView(buf);
  const enc = new TextEncoder();
  const str = (o, s) => new Uint8Array(buf, o, s.length).set(enc.encode(s));
  str(0, 'RIFF');
  v.setUint32(4, 36 + list.length + dataBytes, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  new Uint8Array(buf, 36, list.length).set(list);
  const d = 36 + list.length;
  str(d, 'data');
  v.setUint32(d + 4, dataBytes, true);
  for (let i = 0; i < audio.length; i++) {
    const s = Math.max(-1, Math.min(1, audio[i]));
    v.setInt16(d + 8 + i * 2, Math.round(s * 32767), true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}
