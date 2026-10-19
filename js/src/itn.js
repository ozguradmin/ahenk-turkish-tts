// trnorm/itn.py in JavaScript. Inverse text normalization: spoken Turkish -> written form with digits.
//
// For ASR output, which comes out as words:
//
//     inverse_normalize("saat on dört otuzda yüzde on beşlik indirim")
//     -> "saat 14.30'da %15'lik indirim"
//
// A run of number words is split into the longest *canonical* numbers (a word sequence is a number iff
// number_to_words(value) reproduces it), so "sıfır beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi" becomes
// "0555 555 55 55". Small numbers (<= 10) standing alone stay as words, as Turkish style guides recommend ("bir gün",
// "üç kişi"), unless the context asks for digits ("saat üçte", "yüzde beş", "beş lira").
//
// Values are BigInts, as Python's ints are unbounded (number_to_words throws where Python raises, past 4300 digits).
// Tokens come from Python's split() and every word the module looks for is BMP, so UTF-16 slicing of a token's start
// is Python's code point slicing there.

import * as L from './lexicon.js';
import { tr_lower } from './casing.js';
import { ONES, TENS, VOWELS, number_to_words, ordinal_to_words } from './numbers.js';
import { pin_newer, unpin_newer } from './normalizer.js';
import { B, D, END, U_PAIR_BUG, U_PAIR_RE, first_cp, isupper, re, rstrip, split, strip } from './py.js';

const _SCALE_WORDS = ['yüz', 'bin', 'milyon', 'milyar', 'trilyon', 'katrilyon', 'kentilyon'];
export const NUMBER_WORDS = new Set([...ONES, ...TENS.filter(Boolean), ..._SCALE_WORDS]);
// "altmış" before "altı". Among words of one length Python's order is the set's (hash order, which varies from run
// to run); it cannot matter, as two different words of one length are never both prefixes of a token.
const _NUMBER_WORDS_SORTED = [...NUMBER_WORDS].sort((a, b) => b.length - a.length);
const _SINGLE_VALUE = new Map();
ONES.forEach((w, i) => _SINGLE_VALUE.set(w, BigInt(i)));
TENS.forEach((w, i) => { if (w) _SINGLE_VALUE.set(w, BigInt(i * 10)); });
_SCALE_WORDS.forEach((w, i) => _SINGLE_VALUE.set(w, i ? 10n ** BigInt(3 * i) : 100n));

// all BMP and no negated class: no u flag (see py.js Pattern)
const _ORDINAL_RE = re.compile('^(?:ı|i|u|ü)?nc(?:ı|i|u|ü)', null, false);

// plural? + possessive? + case? + ki?  (enough for suffixes that follow numbers)
const _V4 = '[ıiuü]';
const _V2 = '[ae]';
const _SUFFIX_GRAMMAR = re.compile(
  `^(?:l${_V2}r)?`
  + `(?:y?${_V4}|s${_V4}|${_V4}m${_V4}z|${_V4}n${_V4}z|${_V4}m|${_V4}n|l${_V4}k|l${_V4}|ş?${_V2}r)?`
  + `(?:n?[dt]${_V2}n?|n?[dt]${_V2}ki|y?${_V2}|n?${_V2}|y?${_V4}|n?${_V4}|n${_V4}n|${_V4}n|y?l${_V2}|[cç]${_V2})?`
  + END,
  null,
  false,
);

const _MONTHS_LOWER = new Set(L.MONTHS.filter(Boolean).map(tr_lower));
const _UNIT_WORDS = new Set([
  'lira', 'liraya', 'lirayı', 'liralık', 'kuruş', 'dolar', 'avro', 'euro', 'sterlin', 'sent',
  'kilometre', 'metre', 'santimetre', 'milimetre', 'kilogram', 'kilo', 'gram', 'miligram',
  'litre', 'mililitre', 'derece', 'metrekare', 'metreküp', 'saniye', 'dakika', 'gigabayt',
  'megabayt', 'kilobayt', 'terabayt', 'volt', 'vat', 'kilovat', 'hertz', 'virgül', 'buçuk',
]);
const _DIGIT_CONTEXT_BEFORE = new Set([
  'saat', 'yüzde', 'binde', 'numara', 'numarası', 'numaralı', 'no', 'kod', 'kodu', 'kodunuz',
  'daire', 'sayfa', 'madde', 'plaka', 'hat', 'dahili', 'virgül', 'eksi', 'artı',
]);
const PUNCT = ',.;:!?';

// token -> [base number word, suffix, is_ordinal] or null.
//
// "beşte" -> ["beş", "te", false]; "dördüncüsü" -> ["dört", "sü", true]
function _split_number_word(token) {
  for (const w of _NUMBER_WORDS_SORTED) {
    let rest;
    if (token.startsWith(w)) rest = token.slice(w.length);
    else if (w === 'dört' && token.startsWith('dörd') && VOWELS.has(token[4])) rest = token.slice(4);
    else continue;
    if (!rest) return [w, '', false];
    const m = _ORDINAL_RE.match(rest);
    if (m) {
      const tail = rest.slice(m[0].length);
      if (!tail || _SUFFIX_GRAMMAR.match(tail)) return [w, tail, true];
      continue;
    }
    if (L.BARE_NUMBER_SUFFIXES.has(rest) || _SUFFIX_GRAMMAR.match(rest)) return [w, rest, false];
  }
  return null;
}

// Value of a word list if it is a canonical Turkish cardinal, else null.
function _canonical_value(words) {
  if (!words.length || words.some((w) => !NUMBER_WORDS.has(w))) return null;
  let total = 0n;
  let current = 0n;
  for (const w of words) {
    if (ONES.includes(w)) current += BigInt(ONES.indexOf(w));
    else if (TENS.includes(w)) current += BigInt(TENS.indexOf(w) * 10);
    else if (w === 'yüz') current = (current || 1n) * 100n;
    else {
      total += (current || 1n) * 10n ** BigInt(3 * _SCALE_WORDS.indexOf(w));
      current = 0n;
    }
  }
  const value = total + current;
  return number_to_words(value) === words.join(' ') ? value : null;
}

// f"{v:,}".replace(",", ".") from 10000 on; a canonical value is at most 10**21 - 1, far below the 4300 digits
function _format_int(v) {
  const s = v.toString();
  return v >= 10000n ? s.replace(/\B(?=(\d{3})+$)/g, '.') : s;
}

// -> [base, suffix, is_ordinal, raw, has trailing punctuation] or null
function _parse_token(raw) {
  const core = rstrip(raw, PUNCT);
  if (!core) return null;
  const low = tr_lower(core);
  const parsed = _split_number_word(low);
  if (parsed === null) return null;
  // capitalized tokens are usually names ("Kırklareli"); accept only exact number words
  if (isupper(first_cp(core)) && parsed[1] && !L.BARE_NUMBER_SUFFIXES.has(parsed[1])) return null;
  return [parsed[0], parsed[1], parsed[2], raw, core !== raw];
}

// An Error for a text this engine cannot read right (py.js U_PAIR_BUG): not a Python exception.
function _engine_error() {
  return Object.assign(new Error("this browser's RegExp misreads a pair of characters here (Safari 26.0 to 26.5)"), { name: 'EngineError' });
}

// Convert spoken-form Turkish to written form with digits.
//
// Characters Unicode 15.0 (Python's) does not assign are swapped for a placeholder that this engine, like Python,
// sees as no letter, digit, space or cased character, and swapped back at the end (nothing here drops or reorders
// a character): a newer engine would take a Unicode 16 digit for one in _post_patterns.
export function inverse_normalize(text) {
  if (typeof text !== 'string') {
    throw Object.assign(new TypeError(`'${typeof text}' object has no attribute 'split'`), { name: 'AttributeError', python: true });
  }
  // _post_patterns' \b and \d run with the u flag; a buggy engine reads such a pair as one character there
  if (U_PAIR_BUG && U_PAIR_RE.test(text)) throw _engine_error();
  const [pinned, unassigned, lone] = pin_newer(text);
  const out = _inverse(pinned);
  return unassigned.length || lone.length ? unpin_newer(out, unassigned, lone, false) ?? _inverse(text) : out;
}

function _inverse(text) {
  const tokens = split(text);
  const out = [];
  const n = tokens.length;
  let i = 0;
  while (i < n) {
    const run = [];
    let j = i;
    while (j < n) {
      const p = _parse_token(tokens[j]);
      if (p === null) break;
      run.push(p);
      j += 1;
      if (p[1] || p[2] || p[4]) break; // suffix, ordinal or punctuation closes the number
    }
    if (!run.length) {
      out.push(tokens[i]);
      i += 1;
      continue;
    }
    out.push(..._render_run(run, out, j < n ? tokens[j] : ''));
    i = j;
  }
  return _post_patterns(out.join(' '));
}

const _TRAIL_RE = /[,.;:!?]+$/;
const pad2 = (v) => (v < 10n ? `0${v}` : String(v));

function _render_run(run, prev_out, next_token) {
  const bases = run.map((r) => r[0]);
  const groups = [];
  let k = 0;
  while (k < bases.length) {
    let end = k + 1;
    for (let e = bases.length; e > k; e--) {
      if (_canonical_value(bases.slice(k, e)) !== null) {
        end = e;
        break;
      }
    }
    groups.push([k, end]);
    k = end;
  }
  const values = groups.map(([a, b]) => _canonical_value(bases.slice(a, b)));

  const trail_m = _TRAIL_RE.exec(run[run.length - 1][3]);
  const trail = trail_m ? trail_m[0] : '';
  const [, last_suffix, last_ord] = run[run.length - 1];
  const suf = last_suffix ? `'${last_suffix}` : '';
  const prev_word = prev_out.length ? strip(tr_lower(prev_out[prev_out.length - 1]), PUNCT) : '';
  const next_word = strip(tr_lower(next_token), PUNCT);

  if (values.every((v) => v !== null)) {
    // identifiers read digit by digit: three or more single digits
    if (groups.length >= 3 && values.every((v, x) => v < 10n && groups[x][1] - groups[x][0] === 1)) {
      return [values.join('') + suf + trail];
    }
    // phone numbers: sıfır + area code + groups
    if (groups.length >= 4 && values[0] === 0n && values[1] >= 100n && values[1] <= 999n) {
      const parts = [`0${values[1]}`, ...values.slice(2).map(pad2)];
      return [parts.join(' ') + suf + trail];
    }
  }

  // "on bir yirmide" -> "11.20'de" (hour + minutes with a suffix)
  if (groups.length === 2 && values.every((v) => v !== null) && last_suffix && !last_ord) {
    const [h, mi] = values;
    if (h <= 24n && mi >= 10n && mi <= 59n && groups[1][1] - groups[1][0] <= 2) return [`${h}.${pad2(mi)}${suf}${trail}`];
  }

  const rendered = [];
  groups.forEach(([a, b], idx) => {
    const v = values[idx];
    const last = idx === groups.length - 1;
    const originals = run.slice(a, b).map((r) => r[3]);
    if (v === null) {
      rendered.push(...originals);
      return;
    }
    const ordinal = last_ord && last;
    const single = b - a === 1;
    const small = single && (v <= 10n || bases[a] === 'yüz' || bases[a] === 'bin');
    const digit_ctx = _DIGIT_CONTEXT_BEFORE.has(prev_word)
      || (last && (_UNIT_WORDS.has(next_word) || _MONTHS_LOWER.has(next_word)))
      || groups.length > 1;
    if (small && !digit_ctx) {
      rendered.push(...originals);
      return;
    }
    let s;
    if (ordinal) {
      if (last_suffix) {
        const ord_tail = ordinal_to_words(_SINGLE_VALUE.get(bases[b - 1])).slice(bases[b - 1].length);
        s = `${v}'${ord_tail}${last_suffix}`;
      } else s = `${v}.`;
    } else s = _format_int(v) + (last ? suf : '');
    rendered.push(s + (last ? trail : ''));
  });
  return rendered;
}

// Python's \d and \b are Unicode: a run of digits of any script ("٣ virgül ٤" -> "٣,٤")
const _POST = [
  [re.compile(`${B}(${D}+) virgül (${D}+)${B}`), (m) => `${m[1]},${m[2]}`],
  [re.compile(`${B}(${D}+) buçuk${B}`), (m) => `${m[1]},5`],
  [re.compile(`${B}yüzde (${D}+(?:,${D}+)?)`), (m) => `%${m[1]}`],
  [re.compile(`${B}binde (${D}+(?:,${D}+)?)`), (m) => `‰${m[1]}`],
  [re.compile(`${B}eksi (${D})`), (m) => `-${m[1]}`],
  // times: "saat 14 30'da" -> "saat 14.30'da"
  [re.compile(`${B}([Ss]aat) ([01]?${D}|2[0-4]) ([0-5]${D})${B}`), (m) => `${m[1]} ${m[2]}.${m[3]}`],
];

function _post_patterns(text) {
  for (const [rx, repl] of _POST) text = rx.sub(repl, text);
  return text;
}
