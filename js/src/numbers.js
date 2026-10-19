// trnorm/numbers.py in JavaScript: Turkish number reading and suffix (vowel/consonant) harmony.
//
// number_to_words(1923)   -> "bin dokuz yüz yirmi üç"
// ordinal_to_words(4)     -> "dördüncü"
// attach_suffix("dört", "e")      -> "dörde"
// attach_suffix("on beş", "de")   -> "on beşte"   (re-agrees d/t)
// attach_suffix("altı", "a")      -> "altıya"     (inserts buffer y)
// harmonize_clitic("lira", "mi")  -> "mı"         (separate question particle / dA)
//
// Numbers are BigInts (Python ints are unbounded); number_to_words and ordinal_to_words also take a JS number (see
// _py_number for which Python value it stands for).

import {
  digit_value, first_cp, int_str, isdecimal, last_cp, lower, lstrip, plen, py_int, split, type_error,
} from './py.js';

export const ONES = ['sıfır', 'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz'];
export const TENS = ['', 'on', 'yirmi', 'otuz', 'kırk', 'elli', 'altmış', 'yetmiş', 'seksen', 'doksan'];
export const SCALES = [
  [10n ** 18n, 'kentilyon'],
  [10n ** 15n, 'katrilyon'],
  [10n ** 12n, 'trilyon'],
  [10n ** 9n, 'milyar'],
  [10n ** 6n, 'milyon'],
  [10n ** 3n, 'bin'],
];
export const MAX_NUMBER = 10n ** 21n - 1n;

export const VOWELS = new Set('aeıioöuüâîû');
export const BACK_VOWELS = new Set('aıouâû');
export const ROUNDED_VOWELS = new Set('oöuüû');
export const VOICELESS = new Set('fstkçşhp');

// Loanword stems whose suffixes take front harmony despite a back last vowel.
const _FRONT_EXCEPTIONS = new Set(['saat', 'kontrol', 'gol', 'rol', 'alkol', 'petrol', 'protokol', 'hal', 'kalp']);

// n: a number below 1000
function _below_thousand(n) {
  const words = [];
  const hundreds = Math.floor(n / 100);
  const rest = n % 100;
  if (hundreds) {
    if (hundreds > 1) words.push(ONES[hundreds]);
    words.push('yüz');
  }
  const tens = Math.floor(rest / 10);
  const ones = rest % 10;
  if (tens) words.push(TENS[tens]);
  if (ones) words.push(ONES[ones]);
  return words;
}

// The Python value an argument of number_to_words stands for: a BigInt or a safe integer is an int, true and false
// are 1 and 0 (a bool is an int), any other number a float (2 ** 53 + 1 is already 9007199254740992 in JS, 1e21 a
// float in both). Anything else raises the TypeError Python's n < 0 raises.
function _py_number(n) {
  if (typeof n === 'bigint') return n;
  if (typeof n === 'number') return Number.isSafeInteger(n) ? BigInt(n) : n;
  if (typeof n === 'boolean') return n ? 1n : 0n;
  const type = typeof n === 'string' ? 'str' : n == null ? 'NoneType' : Array.isArray(n) ? 'list' : typeof n;
  throw type_error(`'<' not supported between instances of '${type}' and 'int'`);
}

// divmod(x, y) of floats, x >= 0 and y > 0, as CPython computes it (_float_div_mod)
function _float_divmod(x, y) {
  const mod = x % y; // fmod
  const div = (x - mod) / y;
  let floor = 0;
  if (div) {
    floor = Math.floor(div);
    if (div - floor > 0.5) floor += 1;
  }
  return [floor, mod];
}

const _float_index = () => type_error('list indices must be integers or slices, not float');

function _below_thousand_float(n) {
  const [hundreds, rest] = _float_divmod(n, 100);
  if (hundreds > 1) throw _float_index(); // ONES[hundreds]
  const [tens, ones] = _float_divmod(rest, 10);
  if (tens || ones) throw _float_index(); // TENS[tens], ONES[ones]
  return hundreds ? ['yüz'] : [];
}

// number_to_words of a Python float: the same steps on floats, so a float reads only where no list is indexed with
// one (1e20 -> "yüz kentilyon"), and from 1e21 on (MAX_NUMBER) as the digits of its repr ("1e+21" -> "bir iki bir")
function _float_words(x) {
  if (Number.isNaN(x)) throw _float_index(); // every comparison fails, divmod gives nan, TENS[nan]
  if (x < 0) return `eksi ${_float_words(-x)}`;
  if (x === 0) return 'sıfır';
  if (x >= 1e21) return digits_to_words(String(x)); // repr(): the same digits ("1.2345e+22"; inf has none)
  const words = [];
  let n = x;
  for (const [value, name] of SCALES) {
    let count;
    [count, n] = _float_divmod(n, Number(value));
    if (count) {
      if (!(value === 1000n && count === 1)) words.push(..._below_thousand_float(count));
      words.push(name);
    }
  }
  if (n) words.push(..._below_thousand_float(n));
  return words.join(' ');
}

// Cardinal reading of a non-negative (or negative) integer.
export function number_to_words(n) {
  n = _py_number(n);
  if (typeof n === 'number') return _float_words(n);
  if (n < 0n) return `eksi ${number_to_words(-n)}`;
  if (n === 0n) return 'sıfır';
  if (n > MAX_NUMBER) return digits_to_words(int_str(n));
  const words = [];
  for (const [value, name] of SCALES) {
    const count = n / value;
    n %= value;
    if (count) {
      if (!(value === 1000n && count === 1n)) words.push(..._below_thousand(Number(count))); // "bin", never "bir bin"
      words.push(name);
    }
  }
  if (n) words.push(..._below_thousand(Number(n)));
  return words.join(' ');
}

// Read a digit string one digit at a time: "054" -> "sıfır beş dört".
export function digits_to_words(digits) {
  const out = [];
  for (const d of digits) if (isdecimal(d)) out.push(ONES[digit_value(d)]);
  return out.join(' ');
}

function _last_vowel(word) {
  const chars = [...word];
  for (let i = chars.length - 1; i >= 0; i--) if (VOWELS.has(chars[i])) return chars[i];
  return null;
}

// [is_back, is_rounded] for the stem's last vowel.
function _harmony_state(stem) {
  const words = split(stem);
  const last_word = words.length ? words[words.length - 1] : stem;
  if (_FRONT_EXCEPTIONS.has(last_word) || last_word.endsWith('saat')) { // also "kilovatsaat" (kWh)
    const v = _last_vowel(last_word) || 'e';
    return [false, ROUNDED_VOWELS.has(v)];
  }
  const v = _last_vowel(stem);
  if (v === null) return [false, false]; // e.g. bare consonant: default to front unrounded
  return [BACK_VOWELS.has(v), ROUNDED_VOWELS.has(v)];
}

function _vowel4(back, rounded) {
  if (back) return rounded ? 'u' : 'ı';
  return rounded ? 'ü' : 'i';
}

const _vowel2 = (back) => (back ? 'a' : 'e');

// Re-agree a suffix with the *spoken* stem.
//
// Writers harmonize suffixes with the reading they have in mind ("5'te", "15.00'te"), but often get it wrong
// ("15.00'de", "SGK'ye"). We rebuild vowels and the d/t, c/ç alternations from the stem. Invariant morphemes (-ki
// after a locative, -yor, -ken) are kept as written.
export function harmonize_suffix(stem, suffix) {
  if (!suffix) return suffix;
  let s = [...lower(suffix)]; // code points: s[i] is Python's s[i]
  const stem_l = lower(stem);
  const stem_last = last_cp(stem_l);
  const stem_ends_vowel = VOWELS.has(stem_last);

  // Buffer consonants: drop after consonant-final stems, add after vowel-final.
  if (!stem_ends_vowel && s.length >= 2 && 'ysnm'.includes(s[0])) {
    if (s[0] === 'y' && VOWELS.has(s[1])) { // "5'ye" -> "beşe", "5'yle" -> "beşle"
      s = s.slice(1);
    } else if (s[0] === 's' && 'ıiuü'.includes(s[1])) { // possessive: "5'si" -> "beşi" (not conditional -sA)
      s = s.slice(1);
    } else if (s[0] === 'n' && 'ıiuü'.includes(s[1]) && s[2] === 'n') { // genitive: "5'nin" -> "beşin"
      s = s.slice(1);
    } else if (
      ('mn'.includes(s[0]) && !VOWELS.has(s[1]) && s[0] + s[1] !== 'nc') || /^[mn][ıiuü]z/u.test(s.join(''))
    ) {
      // possessive -(I)m, -(I)mIz, -(I)nIz after "kuruş": "TL'mda", "TL'nız" -> "kuruşumda", "kuruşunuz"
      s = ['ı', ...s];
    }
  } else if (!stem_ends_vowel && s.length === 1 && (s[0] === 'm' || s[0] === 'n')) {
    s = ['ı', ...s]; // possessive -m/-n: "25,50 TL'm" -> "... kuruşum"
  } else if (stem_ends_vowel && VOWELS.has(s[0])) {
    // genitive -(n)In takes n ("6'ın" -> "altının"), everything else takes y
    if ('ıiuü'.includes(s[0]) && s[1] === 'n' && !VOWELS.has(s[2] ?? '')) s = ['n', ...s];
    else s = ['y', ...s];
  }

  // -ki after a locative never harmonizes and sets the harmony of what follows:
  // "2023'teki", "1990'lardakiler", "2026'dakidir" (written -kı is re-spelled -ki)
  const fixed_i = new Set();
  for (let i = 2; i < s.length - 1; i++) {
    if (s[i] === 'k' && 'ıi'.includes(s[i + 1]) && 'dt'.includes(s[i - 2]) && 'ae'.includes(s[i - 1])) {
      fixed_i.add(i + 1);
    }
  }

  let [back, rounded] = _harmony_state(stem_l);
  let prev = stem_last;
  const out = [];
  s.forEach((c, i) => {
    let ch = c;
    if (fixed_i.has(i)) {
      ch = 'i';
      back = false;
      rounded = false;
    } else if ('aeâ'.includes(ch)) {
      ch = _vowel2(back);
      back = BACK_VOWELS.has(ch);
      rounded = false;
    } else if ('ıiuü'.includes(ch)) {
      ch = _vowel4(back, rounded);
      rounded = ROUNDED_VOWELS.has(ch);
    } else if ('oö'.includes(ch)) {
      back = ch === 'o';
      rounded = true;
    } else if ('dt'.includes(ch) && i === 0) {
      ch = VOICELESS.has(prev) ? 't' : 'd';
    } else if ('cç'.includes(ch) && i === 0) {
      ch = VOICELESS.has(prev) ? 'ç' : 'c';
    }
    out.push(ch);
    prev = ch;
  });
  return out.join('');
}

// Re-agree a separately written clitic (question mI and its inflections, dA) with the spoken stem.
//
// Only vowels change; consonants and buffer y are kept: ("lira", "mi") -> "mı", ("altı", "de") -> "da",
// ("dokuz", "misiniz") -> "musunuz". Both args lowercase.
export function harmonize_clitic(stem, clitic) {
  let [back, rounded] = _harmony_state(stem);
  const out = [];
  for (let ch of clitic) {
    if ('ae'.includes(ch)) {
      ch = _vowel2(back);
      back = BACK_VOWELS.has(ch);
      rounded = false;
    } else if ('ıiuü'.includes(ch)) {
      ch = _vowel4(back, rounded);
      rounded = ROUNDED_VOWELS.has(ch);
    }
    out.push(ch);
  }
  return out.join('');
}

// Attach a (re-harmonized) suffix to a spoken stem, handling dört->dörd.
export function attach_suffix(stem, suffix) {
  if (!suffix) return stem;
  const suf = harmonize_suffix(stem, suffix);
  if (suf && VOWELS.has(first_cp(suf)) && stem.endsWith('dört')) stem = `${stem.slice(0, -4)}dörd`;
  return stem + suf;
}

// Ordinal reading: 1 -> birinci, 4 -> dördüncü, 20 -> yirminci.
export function ordinal_to_words(n) {
  const cardinal = number_to_words(n);
  let stem = cardinal;
  const [back, rounded] = _harmony_state(stem);
  const v = _vowel4(back, rounded);
  // stem[-1]: the float inf reads as '' (its repr has no digits), and Python's index fails there
  if (!stem) throw Object.assign(new RangeError('string index out of range'), { name: 'IndexError', python: true });
  const suf = VOWELS.has(last_cp(stem)) ? `nc${v}` : `${v}nc${v}`;
  if (VOWELS.has(suf[0]) && stem.endsWith('dört')) stem = `${stem.slice(0, -4)}dörd`;
  return stem + suf;
}

// Read the part after the decimal comma: "5" -> beş, "05" -> sıfır beş, "25" -> yirmi beş.
export function read_decimal_fraction(digits) {
  if (!digits) return '';
  if (plen(digits) > 3) return digits_to_words(digits);
  const rest = lstrip(digits, '0');
  const parts = Array(digits.length - rest.length).fill('sıfır'); // "0" is one UTF-16 unit
  if (rest) parts.push(number_to_words(py_int(rest)));
  return parts.join(' ');
}

// Read a digit group keeping leading zeros: "05" -> "sıfır beş", "693" -> "altı yüz doksan üç".
export function read_digit_group(group) {
  const rest = lstrip(group, '0');
  const parts = Array(group.length - rest.length).fill('sıfır');
  if (rest) parts.push(number_to_words(py_int(rest)));
  return parts.join(' ');
}

const _GROUP_PATTERNS = new Map([
  [5, [3, 2]],
  [6, [3, 3]],
  [7, [3, 2, 2]],
  [8, [3, 3, 2]],
  [9, [3, 3, 3]],
  [10, [3, 3, 2, 2]],
  [11, [3, 3, 3, 2]],
]);

// Read an identifier-like digit string in speakable groups (3-3-2-2 style).
export function read_long_digits(digits) {
  const chars = [...digits];
  const n = chars.length;
  if (n <= 4) return read_digit_group(digits);
  let sizes = _GROUP_PATTERNS.get(n);
  if (sizes === undefined) {
    sizes = Array(Math.floor(n / 3)).fill(3);
    if (n % 3 === 1) {
      sizes[sizes.length - 1] = 2;
      sizes.push(2);
    } else if (n % 3 === 2) {
      sizes.push(2);
    }
  }
  const groups = [];
  let i = 0;
  for (const size of sizes) {
    groups.push(chars.slice(i, i + size).join(''));
    i += size;
  }
  return groups.map(read_digit_group).join(', ');
}
