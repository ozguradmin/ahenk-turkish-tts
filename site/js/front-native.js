// Ahenk Cep's text side in plain JavaScript: mirrors site/py/ahenk_front.py (prepare, pieces, piece, chunk, finish,
// frontend, alphabet, blocks), which copies EMA Lightning 1.1.2's frontend.py, chunker.py and Engine.piece.
// From ozguradmin/ahenk-cep-pdfreader (docs/js/front-native.js, Apache-2.0); js/trnorm/ is ozguradmin/trnorm-js
// 0.1.2 (npm @ozgurguler/trnorm, src/, Apache-2.0) as published. tests/test_web_demo.py checks both against the
// Python.
//
//   prepare(text, speed, vocab) -> {spoken, pieces: [{text, pause, ids, cw, wstart}]}
//
// is the same object as JSON.parse(ahenk_front.prepare(text, speed, vocab)) for every input, and it throws where
// the Python raises. Python's meaning is spelled out wherever JavaScript's differs: lengths, indices and slices
// count code points (not UTF-16 units), whitespace is str.isspace() (what str.split/strip and re's \s use; JS's \s
// and trim() differ), LETTER is re's Unicode [^\W\d_], lower() is Python's (trnorm/py.js: Unicode 15.0's case data,
// final sigma), and unicodedata.combining is a table taken from Python. Works in browsers (Chrome, Firefox,
// Safari 16.4+) and Node; no Node APIs.
//
// Every error thrown on purpose carries python: true: the Python raises there too (a ValueError for a number of more
// than 4300 digits, a TypeError for a bad vocabulary) or never returns (a speed under 1/180, a RuntimeError here), so
// asking the Python again gives the same error, or hangs. The one other is an EngineError: this browser's RegExp
// misreads the text (Safari 26.0 to 26.5, trnorm/py.js U_PAIR_BUG), and the Python reads it right.

import { normalize, normalize_for_frontend } from './trnorm/index.js';
import { lower } from './trnorm/py.js';
import { UNASSIGNED } from './trnorm/unassigned.js';

export { normalize };

// ---- Python's str and float semantics ---------------------------------------------------------------------------

// str.isspace(): bidi class WS, B or S, or category Zs. Checked against Python over every code point.
const SPACE = '\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const SPACES = new RegExp(`[${SPACE}]+`, 'g'); // re.sub(r"\s+", ...)
const WORD = new RegExp(`[^${SPACE}]+`, 'g'); // the words of str.split()
const NOT_SPACE = new RegExp(`[^${SPACE}]`);

const is_space = (c) => (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c === 0x85 || c === 0xa0
  || c === 0x1680 || (c >= 0x2000 && c <= 0x200a) || c === 0x2028 || c === 0x2029 || c === 0x202f
  || c === 0x205f || c === 0x3000;

// str.strip() (no surrogate is a space, so code units are enough)
function strip(s) {
  let a = 0;
  let b = s.length;
  while (a < b && is_space(s.charCodeAt(a))) a++;
  while (b > a && is_space(s.charCodeAt(b - 1))) b--;
  return a === 0 && b === s.length ? s : s.slice(a, b);
}

// str.rstrip(chars) for ASCII chars
function rstrip(s, chars) {
  let b = s.length;
  while (b > 0 && chars.includes(s[b - 1])) b--;
  return s.slice(0, b);
}

// str.strip() of the code points cps[a:b], as a span
function strip_span(cps, a, b) {
  while (a < b && is_space(cps[a].charCodeAt(0))) a++;
  while (b > a && is_space(cps[b - 1].charCodeAt(0))) b--;
  return [a, b];
}

// Where s[:i] ends (and s[i:] starts) in a sequence of n, for any int i: negative counts from the end.
const slice_end = (i, n) => (i < 0 ? Math.max(0, n + i) : Math.min(i, n));

// A Python exception: an Error named after its type, so a caller can tell which one Python raised.
function py_error(type, message) {
  const e = type === 'TypeError' ? new TypeError(message) : new Error(message);
  e.name = type;
  e.python = true;
  return e;
}

// float(x). A string goes through _PyUnicode_TransformDecimalAndSpaceToASCII first: below U+007F a character stays,
// above it whitespace becomes a space and a decimal digit of any script (Unicode 15.0, Pyodide's) its ASCII digit,
// anything else fails. Only ASCII whitespace is stripped after that ("\x1c1.5" fails, "\u30001.5" and "١.٥" do not).
const FLOAT = /^[+-]?(?:(?:\d(?:_?\d)*)?\.\d(?:_?\d)*|\d(?:_?\d)*\.?)(?:e[+-]?\d(?:_?\d)*)?$/i;
const FLOAT_WORD = /^([+-]?)(inf|infinity|nan)$/i;
const ASCII_SPACE = /^[\t\n\v\f\r ]+|[\t\n\v\f\r ]+$/g;
const ND = /^\p{Nd}$/u;
// a code point Unicode 15.0 leaves unassigned: no digit, no decomposition, no case to Python; maybe more to this engine
const NEWER = new RegExp(`[${UNASSIGNED}]`, 'u');

// the ASCII digit of a decimal digit c (\p{Nd}, which Unicode encodes in runs of ten), '' for any other character
function decimal_digit(c) {
  if (!ND.test(c) || NEWER.test(c)) return '';
  let zero = c.codePointAt(0);
  while (ND.test(String.fromCodePoint(zero - 1))) zero--;
  return String((c.codePointAt(0) - zero) % 10);
}

function float_ascii(s) {
  let out = '';
  for (const c of s) {
    const u = c.codePointAt(0);
    if (u < 0x7f) out += c;
    else if (is_space(u)) out += ' ';
    else {
      const d = decimal_digit(c);
      if (!d) return null;
      out += d;
    }
  }
  return out;
}

function py_float(x) {
  if (typeof x === 'number') return x;
  if (typeof x === 'boolean') return x ? 1 : 0;
  if (typeof x === 'bigint') {
    const f = Number(x); // rounds half to even, like int.__float__
    if (!Number.isFinite(f)) throw py_error('OverflowError', 'int too large to convert to float');
    return f;
  }
  if (typeof x === 'string') {
    const s = float_ascii(x)?.replace(ASCII_SPACE, '');
    if (s != null && FLOAT.test(s)) return Number(s.replaceAll('_', ''));
    const w = s == null ? null : FLOAT_WORD.exec(s);
    if (w) return w[2].toLowerCase() === 'nan' ? NaN : (w[1] === '-' ? -Infinity : Infinity);
    throw py_error('ValueError', `could not convert string to float: ${JSON.stringify(x)}`);
  }
  const type = x === null || x === undefined ? 'NoneType' : typeof x;
  throw py_error('TypeError', `float() argument must be a string or a real number, not '${type}'`);
}

// int(x) of a float: truncates; inf and nan raise
function py_int(x) {
  if (Number.isNaN(x)) throw py_error('ValueError', 'cannot convert float NaN to integer');
  if (!Number.isFinite(x)) throw py_error('OverflowError', 'cannot convert float infinity to integer');
  return Math.trunc(x) + 0; // + 0: no -0
}

// min(a, b) keeps a unless b < a (so a NaN b loses)
const py_min = (a, b) => (b < a ? b : a);

// list(vocab), and the TypeError that {ch: i ...} raises for an unhashable item (a list or a dict from JSON)
function py_list(vocab) {
  if (typeof vocab === 'string' || (vocab != null && typeof vocab[Symbol.iterator] === 'function')) {
    const items = Array.from(vocab);
    for (const ch of items) {
      if (ch !== null && typeof ch === 'object') {
        throw py_error('TypeError', `unhashable type: '${Array.isArray(ch) ? 'list' : 'dict'}'`);
      }
    }
    return items;
  }
  const type = vocab === null || vocab === undefined ? 'NoneType' : typeof vocab;
  throw py_error('TypeError', `'${type}' object is not iterable`);
}

// ---- ema_lightning/frontend.py ------------------------------------------------------------------------------------
export const BLOCK_BYTES = 8 * 1024;
const TURKISH = new Set('çğıöşüÇĞİÖŞÜ');
const TYPOGRAPHY = {
  '’': "'", '‘': "'", 'ʼ': "'", '´': "'", '`': "'", '“': '"', '”': '"', '„': '"',
  '«': '"', '»': '"', '–': '-', '—': '-', '−': '-', '…': '...',
};
const TYPOGRAPHY_CHARS = /[’‘ʼ´`“”„«»–—−…]/g;
const UNSAFE = /[\x00-\x08\x0b-\x1f\x7f-\x9f؜‎‏‪-‮⁦-⁩]/g;
// what encode("utf-8", "ignore") drops: a surrogate that is not half of a pair
const LONE_SURROGATE = /[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/g;

// Code points with unicodedata.combining(c) != 0 (canonical combining class), from Python 3.13 (Unicode 15.1).
// JavaScript has no such property, and \p{M} is not the same set: many marks have class 0.
const COMBINING = new RegExp(`[${[
  '̀-͎͐-ͯ҃-֑҇-ׇֽֿׁׂׅׄؐ-ؚ',
  'ً-ٰٟۖ-ۜ۟-۪ۤۧۨ-ܑۭܰ-݊߫-߳',
  '߽ࠖ-࠙ࠛ-ࠣࠥ-ࠧࠩ-࡙࠭-࡛࢘-࢟࣊-࣡',
  'ࣣ-़्ࣿ॑-়਼઼଼఼্੍્୍்॔৾',
  '಼్್഻഼്්ౕౖุ-ฺ่-๋ຸ-຺',
  '່-໋ཱིེུ༹༘༙༵༷-ཽྀྂ-྄',
  '့္်࿆ႍ྆྇፝-᜔᜕᜴្᤹ᢩ፟៝-᤻',
  '᩠ᨘᨗ᩵-᩿᩼᪰-᪽ᪿ-᬴᭄ᫎ᭫-᮪᮫᭳',
  '᯦᰷᯲᯳᳐-᳔᳒-᳢᳠-᳨᳭᳴᳸᳹᷀-᷿',
  '⃐-⃥⃜⃡-⃰⳯-⵿⳱ⷠ-〪ⷿ-゙゚〯꙯',
  'ꙴ-꠆꠬꣄꙽ꚞꚟ꛰꛱꣠-꤫꣱-꦳꥓꧀꤭',
  'ꪰꪲ-꫶꯭ﬞꪴꪷꪸꪾ꪿꫁︠-︯\u{101fd}\u{102e0}',
  '\u{10376}-\u{1037a}\u{10a0d}\u{10a0f}\u{10a38}-\u{10a3a}\u{10a3f}\u{10ae5}\u{10ae6}\u{10d24}-\u{10d27}',
  '\u{10eab}\u{10eac}\u{10efd}-\u{10eff}\u{10f46}-\u{10f50}\u{10f82}-\u{10f85}\u{11046}\u{11070}\u{1107f}',
  '\u{110b9}\u{110ba}\u{11100}-\u{11102}\u{11133}\u{11134}\u{11173}\u{111c0}\u{111ca}\u{11235}\u{11236}',
  '\u{112e9}\u{112ea}\u{1133b}\u{1133c}\u{1134d}\u{11366}-\u{1136c}\u{11370}-\u{11374}\u{11442}\u{11446}',
  '\u{1145e}\u{114c2}\u{114c3}\u{115bf}\u{115c0}\u{1163f}\u{116b6}\u{116b7}\u{1172b}\u{11839}\u{1183a}',
  '\u{1193d}\u{1193e}\u{11943}\u{119e0}\u{11a34}\u{11a47}\u{11a99}\u{11c3f}\u{11d42}\u{11d44}\u{11d45}',
  '\u{11d97}\u{11f41}\u{11f42}\u{16af0}-\u{16af4}\u{16b30}-\u{16b36}\u{16ff0}\u{16ff1}\u{1bc9e}',
  '\u{1d165}-\u{1d169}\u{1d16d}-\u{1d172}\u{1d17b}-\u{1d182}\u{1d185}-\u{1d18b}\u{1d1aa}-\u{1d1ad}',
  '\u{1d242}-\u{1d244}\u{1e000}-\u{1e006}\u{1e008}-\u{1e018}\u{1e01b}-\u{1e021}\u{1e023}\u{1e024}',
  '\u{1e026}-\u{1e02a}\u{1e08f}\u{1e130}-\u{1e136}\u{1e2ae}\u{1e2ec}-\u{1e2ef}\u{1e4ec}-\u{1e4ef}',
  '\u{1e8d0}-\u{1e8d6}\u{1e944}-\u{1e94a}',
].join('')}]`, 'gu');

// len(s.encode("utf-8")) of a string without lone surrogates: a pair is 2 + 2 bytes
function utf8_len(s) {
  let n = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    n += c < 0x80 ? 1 : c < 0x800 || (c >= 0xd800 && c <= 0xdfff) ? 2 : 3;
  }
  return n;
}

export function blocks(text) {
  const words = text.match(WORD) || [];
  const out = [];
  let block = [];
  let size = 0;
  for (const word of words) {
    const n = utf8_len(word) + 1;
    if (block.length && size + n > BLOCK_BYTES) {
      out.push(block.join(' '));
      block = [];
      size = 0;
    }
    block.push(word);
    size += n;
  }
  if (block.length) out.push(block.join(' '));
  return out;
}

// One code point of the lowered text: Turkish letters stay, others lose their accents (NFKD, then every code point
// of nonzero combining class); a result with anything outside the vocabulary, or nothing left, is a space. A
// character newer than Unicode 15.0 has no decomposition to Python (this engine may know one) and stays as it is.
function alphabet_char(ch, vocab) {
  if (NEWER.test(ch)) return vocab.has(ch) ? ch : ' ';
  if (!TURKISH.has(ch)) ch = ch.normalize('NFKD').replace(COMBINING, '');
  if (!ch) return ' ';
  for (const c of ch) if (!vocab.has(c)) return ' ';
  return ch;
}

// vocab: a Set of the vocabulary's items (the Python's frozenset)
export function alphabet(text, vocab) {
  text = lower(text.replace(TYPOGRAPHY_CHARS, (c) => TYPOGRAPHY[c]).replaceAll('İ', 'i').replaceAll('I', 'ı'));
  const seen = new Map(); // the same code point always comes out the same; documents repeat few of them
  const out = [];
  for (const ch of text) {
    let r = seen.get(ch);
    if (r === undefined) {
      r = alphabet_char(ch, vocab);
      seen.set(ch, r);
    }
    out.push(r);
  }
  return strip(out.join('').replace(SPACES, ' '));
}

// EMA's Frontend.__call__ with spoken() = identity (trnorm already gave the spoken form)
export function frontend(text, vocab) {
  text = text.replace(LONE_SURROGATE, '').replace(UNSAFE, ' ');
  if (!NOT_SPACE.test(text)) return '';
  return alphabet(blocks(text).join(' '), vocab);
}

// ---- ema_lightning/chunker.py -------------------------------------------------------------------------------------
export const LETTERS_PER_SECOND = 18.0;
export const MAX_SECONDS = 10.0;
export const MAX_LETTERS = 250;
export const SENTENCE_PAUSE = 0.25;
export const CLAUSE_PAUSE = 0.12;
const CUTS = [
  [/[.!?]+["')]*(?= )/g, SENTENCE_PAUSE],
  [/[,;:](?= )/g, CLAUSE_PAUSE],
  [new RegExp(`[^${SPACE}](?= )`, 'gu'), CLAUSE_PAUSE], // \S(?= )
];
const LETTER = /[\p{L}\p{Nl}\p{No}]/u; // [^\W\d_]: alphabetic, or numeric but not a decimal digit
// LETTER.search(piece), a letter of Unicode 15.0 (one newer is none to Python)
const has_letter = (piece) => LETTER.test(piece)
  && (!NEWER.test(piece) || [...piece].some((c) => LETTER.test(c) && !NEWER.test(c)));

// m.end() of the last match of pattern.finditer(win) (win has n code points), in code points; -1 if none
function last_end(pattern, win, n) {
  let end = -1;
  for (const m of win.matchAll(pattern)) end = m.index + m[0].length;
  if (end < 0 || win.length === n) return end;
  return Array.from(win.slice(0, end)).length;
}

export function chunk(text, speed) {
  const limit = py_int(py_min(MAX_LETTERS, LETTERS_PER_SECOND * MAX_SECONDS * speed));
  const pieces = [];
  const cps = Array.from(text); // rest is cps[s:e]
  let [s, e] = strip_span(cps, 0, cps.length);
  while (s < e) {
    const n = e - s;
    let cut = n;
    let pause = 0.0;
    if (n > limit) {
      cut = limit;
      // finditer(rest, 0, limit + 1): the matches in rest[:limit + 1], whose end the lookahead cannot see past
      const w = Math.max(0, Math.min(limit + 1, n));
      const win = cps.slice(s, s + w).join('');
      for (const [pattern, gap] of CUTS) {
        const end = last_end(pattern, win, w);
        if (end >= 0) {
          cut = end;
          pause = gap;
          break;
        }
      }
    }
    const at = s + slice_end(cut, n);
    const [a, b] = strip_span(cps, s, at);
    const next = strip_span(cps, at, e);
    // A cut of 0 (or a negative one that keeps all of rest) leaves rest as it was: the Python loops forever there
    // (speed under 1/180, so limit <= 0). Fail instead of hanging the page.
    if (next[0] === s && next[1] === e) {
      throw py_error('RuntimeError', `chunk: no progress at speed ${speed} (limit ${limit}), Python never returns`);
    }
    [s, e] = next;
    const piece = cps.slice(a, b).join('');
    if (has_letter(piece)) pieces.push([finish(piece), pause]);
  }
  if (pieces.length) pieces[pieces.length - 1] = [pieces[pieces.length - 1][0], 0.0];
  return pieces;
}

export function finish(piece) {
  // [-1:] is the last code point; a pair's low half is never one of these, so the last unit answers the same
  if (['.', '!', '?'].includes(rstrip(piece, '"\')').slice(-1))) return piece;
  return rstrip(piece, ',;:- ') + '.';
}

// ---- ema_lightning/engine.py: Engine.piece ------------------------------------------------------------------------
export function piece(text, stoi) {
  const chars = Array.from(text);
  const ids = chars.map((ch) => stoi.get(ch) ?? 1);
  let starts = [];
  chars.forEach((ch, i) => {
    if (ch !== ' ' && (i === 0 || chars[i - 1] === ' ')) starts.push(i);
  });
  if (!starts.length) starts = [0];
  const bounds = [0, ...starts.slice(1), chars.length];
  const cw = [];
  const wstart = [];
  for (let w = 0; w + 1 < bounds.length; w++) {
    const a = bounds[w];
    for (let i = a; i < bounds[w + 1]; i++) {
      cw.push(w);
      wstart.push(a);
    }
  }
  return [ids, cw, wstart];
}

export function pieces(spoken, speed, vocab) {
  vocab = py_list(vocab);
  const stoi = new Map(); // {ch: i}: a repeated item keeps its last index
  vocab.forEach((ch, i) => stoi.set(ch, i));
  const letters = new Set(vocab);
  const out = [];
  for (const [text, pause] of chunk(frontend(spoken, letters), speed)) {
    const [ids, cw, wstart] = piece(text, stoi);
    out.push({ text, pause, ids, cw, wstart });
  }
  return out;
}

// V8 compiles a pattern for one-byte (Latin-1) and for two-byte subjects separately, each when first used, and some
// paths only when a text reaches them: the first text with ş, ğ or ı (any real Turkish sentence) paid ~50 ms more after
// the worker's Latin-1 warm-up call. So the first prepare() call also reads this text through every pass, as it is and
// with its non-Latin-1 characters swapped for Latin-1 ones (both kinds of string).
const WARM_UP = "Şişli'de 5 TL mi? Iğdır'da 3 kişi, 21 Kasım 2025'te 14.30'da, saat 09:30. Google'da ara: " +
  'ali.veli@ornek.com.tr, https://www.ornek.gov.tr/a?b=1 ve +90 (212) 555-12-34, TR12 0006 1005 1978 6457 8413 26, ' +
  "34 ABC 123. XX. Yüzyılda 4'üncü sırada; Dr. I. Smith, GPT-3.5 ile COVID-19, 25°C, 1,5 kg, 10-12 km/s, x² + 10³, " +
  "IV. Murat, SGK'ye 2 müdür mü? İSTİKLAL CADDESİ 50000 KİŞİYLE DOLDU. No: 12/4, ..., 34728 Kadıköy. 1.2.3 ve 3.x, " +
  "007, 1990'lardaki 12.345.678,90 ₺, 2023-24 sezonu %15, 3/4 ve 1.250,50 EUR. Bugün hava çok güzel, 25 derece; " +
  'yarın 3 gün sonra 10.40 metre, 2 bin kişi ve 15 öğrenci geldi.';
const LATIN1 = { ş: 's', ğ: 'g', ı: 'i', İ: 'I', Ş: 'S', Ğ: 'G', '₺': '$' };
let warm = false;

export function prepare(text, speed, vocab) {
  if (!warm) {
    warm = true;
    normalize_for_frontend(WARM_UP);
    normalize_for_frontend(WARM_UP.replace(/[^\0-\xff]/g, (c) => LATIN1[c] ?? ' '));
  }
  // spoken is normalize(text); pieces read it as the Python string holds it, whose lone surrogates frontend() drops
  // even where a pass brought two together (one astral character in a JS string)
  const [spoken, front_text] = normalize_for_frontend(text);
  return { spoken, pieces: pieces(front_text, py_float(speed), vocab) };
}
