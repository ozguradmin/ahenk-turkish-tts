// trnorm/normalizer.py in JavaScript: written Turkish -> spoken Turkish (normalize), and the canonical form for scoring
// ASR and TTS (normalize_for_eval).
//
// The normalizer is a sequence of regex passes, from the most specific patterns (URLs, phone numbers, dates, times,
// money) to the most generic (bare numbers, acronyms, symbols). Every numeric pass understands an optional apostrophe
// suffix and re-agrees it with the spoken form:
//
//     "15.00'de"  -> "on beşte"      "SGK'ye"   -> "se ge kaya"
//     "4'üncü"    -> "dördüncü"      "%15'lik"  -> "yüzde on beşlik"
//
// A separately written question particle (mI, miydi, misiniz...) or clitic dA right after an expanded token is
// re-agreed the same way: "5 TL mi" -> "beş lira mı".
//
// The port keeps Python's names, order and patterns. Patterns are written with py.js's fragments: W/WC for \w, D for
// \d, S/SC for \s, B for \b, END for $, lb/nlb for a lookbehind on \w or \d. m[n] is match.group(n) (undefined for
// None), m.index and end(m) its span in UTF-16 units, so every slice Python sizes in characters goes through cp_back /
// cp_fwd / plen. Numbers are BigInts. Where a Python pattern takes O(n²) on a long run (the URL and e-mail patterns
// on "1.1.1...", the alnum lookaheads on "a-a-a...", a postcode search over everything before it), the port gets the
// same matches without the quadratic part (finditer objects below, each argued at its definition).

import * as L from './lexicon.js';
import { is_all_caps, tr_capitalize, tr_lower, tr_upper } from './casing.js';
import {
  VOWELS,
  attach_suffix,
  digits_to_words,
  harmonize_clitic,
  harmonize_suffix,
  number_to_words,
  ordinal_to_words,
  read_decimal_fraction,
  read_digit_group,
  read_long_digits,
} from './numbers.js';
import {
  ANY, B, CI_A_Z, D, END, S, SC, U_PAIR_BUG, U_PAIR_RE, W, WC, char_at, char_before, ci, cp_back, cp_fwd, end, first_cp,
  has, has_digit, in_table, isalnum, isalpha, isdecimal, isdigit, islower, isupper, last_cp, lb, lower, nlb, plen, py_int,
  re, rstrip, split, split1, strip, type_error, upper,
} from './py.js';
import { UNASSIGNED } from './unassigned.js';

const LOW = 'a-zçğıöşüâîû';
const UP = 'A-ZÇĞİÖŞÜÂÎÛ';
const LET = LOW + UP;

// optional apostrophe suffix: "'te", "'DA" (uppercase suffixes in all-caps text)
const SUF = `(?:'([${LET}]+))?`;
// integer with optional dot thousands, optional comma decimal
const INT_TH = String.raw`${D}{1,3}(?:\.${D}{3})+`;
const NUM = `(?:${INT_TH}|${D}+)(?:,${D}+)?`;
// number in unit/currency context: also accepts dot decimals ("1.5 kg")
const NUMU = `(?:${INT_TH}(?:,${D}+)?|${D}+[.,]${D}+|${D}+)`;
const NB = nlb(`${WC}.,`); // number must not continue a word or another number
// NB where the number starts with a digit (or a sign): the same, with that first character tested first (faster)
const NBD = `(?=${D})${NB}`;
const NBS = `(?=[\\-${D}])${NB}`;

const _ORDINAL_SUFFIX_RE = re.compile(`^(?:[ıiuü]?nc[ıiuü])(${ANY}*)${END}`);

// a small int (a regex group of at most a few digits) as a JS number
const int = (s) => Number(py_int(s));

// --------------------------------------------------------------------------- helpers

function _suffix(stem, suffix) {
  if (!suffix) return stem;
  return attach_suffix(stem, tr_lower(suffix));
}

// Attach a suffix to a spoken integer; ordinal suffixes become ordinals.
function _attach_number_suffix(n, words, suffix) {
  if (!suffix) return words;
  const s = tr_lower(suffix);
  const m = _ORDINAL_SUFFIX_RE.match(s);
  if (m) { // 4'üncü, 2'nci, 1'inci, 3'ncü (misspelt) -> ordinal (+ rest: "si", "de")
    return m[1] ? _suffix(ordinal_to_words(n), m[1]) : ordinal_to_words(n);
  }
  return attach_suffix(words, s);
}

// Read "1.500", "3,5", "1.250,75", "1.5" (dot decimal), "-5".
export function read_number(s, dot_word = 'nokta') {
  s = strip(s);
  const neg = '-−'.includes(s.slice(0, 1)); // s[:1] in "-−" ("" is in every string)
  if (neg) s = s.slice(1);
  let out;
  if (s.includes(',')) {
    const [ip, fp] = split1(s, ',');
    const ipv = py_int(ip.replaceAll('.', '') || '0');
    if (fp === '5' && ipv > 0n) out = `${number_to_words(ipv)} buçuk`; // 1,5 -> bir buçuk
    else out = `${number_to_words(ipv)} virgül ${read_decimal_fraction(fp)}`;
  } else if (s.includes('.')) {
    if (re.fullmatch(INT_TH, s)) {
      out = number_to_words(py_int(s.replaceAll('.', '')));
    } else {
      const [ip, fp] = split1(s, '.');
      if (dot_word === 'virgül' && fp === '5' && py_int(ip || '0') > 0n) out = `${number_to_words(py_int(ip))} buçuk`;
      else out = `${number_to_words(py_int(ip || '0'))} ${dot_word} ${read_decimal_fraction(fp.replaceAll('.', ''))}`;
    }
  } else {
    out = plen(s) <= 21 ? number_to_words(py_int(s)) : digits_to_words(s);
  }
  return neg ? `eksi ${out}` : out;
}

function _int_value(s) {
  s = s.replaceAll('.', '');
  return isdecimal(s) ? py_int(s) : null;
}

const _BETWEEN_RE = re.compile(String.raw`[ \t]+(?:arası|arasında|arasındaki|aralığı|aralığında)${B}`);

// Range word: "ila" ("4 ila 5 dakika"), but "ile" before "arası(nda)" ("13 ile 17 arasında").
function _ila(text, end_) { // end_: Python's end (end() is the match helper here)
  return _BETWEEN_RE.match(text, end_) ? 'ile' : 'ila';
}

function _read_range(a, b, dot_word = 'nokta', word = 'ila') {
  return `${read_number(a, dot_word)} ${word} ${read_number(b, dot_word)}`;
}

function _letters(token) {
  const letters = [...token].filter(isalpha);
  // a Latin I inside a spelled acronym is the letter i, not ı: IBM, HDMI, IETT typed without Turkish keys
  return letters.map((ch) => (ch === 'I' && letters.length > 1 ? 'i' : (L.LETTER_NAMES.get(ch) ?? tr_lower(ch))))
    .join(' ');
}

function _pronounceable(word) {
  const w = tr_lower(word);
  const flags = [...w].filter(isalpha).map((c) => VOWELS.has(c));
  const n = flags.length;
  if (n <= 2 || !flags.some(Boolean)) return false;
  let run = 0;
  let maxrun = 0;
  for (const v of flags) {
    run = v ? 0 : run + 1;
    maxrun = Math.max(maxrun, run);
  }
  if (n >= 5) return maxrun <= 3 && flags.filter(Boolean).length / n >= 0.3;
  const starts_cc = !flags[0] && !flags[1];
  const ends_cc = !flags[n - 1] && !flags[n - 2];
  return maxrun <= 2 && !starts_cc && !ends_cc;
}

// --------------------------------------------------------------------------- clitics (mI, dA)

// Private-use char put after an expansion that is followed by a clitic; _pass_clitics consumes it.
const _MARK = '\u{e000}';
const _I4 = '[ıiuü]';
const _A2 = '[ae]';
const _PERS = `(?:m|n(?:${_I4}z)?|k|l${_A2}r)`; // -m, -n, -nIz, -k, -lAr after -DI / -sA
// question particle mI and its copula forms (miyim, misiniz, midir, miydi, miymiş, miyse...), clitic dA
const _CLITIC_RE = re.compile(
  `d${_A2}|m${_I4}(?:y${_I4}[mz]|s${_I4}n(?:${_I4}z)?|d${_I4}r(?:l${_A2}r)?|yd${_I4}${_PERS}?` +
  `|ym${_I4}ş(?:${_I4}[mz]|s${_I4}n(?:${_I4}z)?|l${_A2}r)?|ys${_A2}${_PERS}?)?`,
);
const _WORD_AFTER_RE = re.compile(String.raw`([ \t]+)([${LET}]+)(?![${WC}'])`);
const _QUESTION_NEXT_RE = re.compile(String.raw`${S}*["'’”»)\]]*${S}*\?`);
// source token of a plain cardinal / range / list: "2", "1.000", "2-3"
const _BARE_COUNT_RE = re.compile(String.raw`[${D}.,${SC}\-]+`);

// Match " <clitic>" at pos: all-lowercase or all-caps mI / dA forms only.
// count: the word before is a bare cardinal, after which "müdür" is the noun ("2 müdür?").
function _clitic_at(text, pos, count = false) {
  const m = _WORD_AFTER_RE.match(text, pos);
  if (!m) return null;
  const word = m[2];
  const low = tr_lower(word);
  if ((word !== low && word !== tr_upper(low)) || !_CLITIC_RE.fullmatch(low)) return null;
  // "müdür(ler)" is also a noun ("2 müdür, 3 memur", "İK müdür yardımcısı"): mI+dIr only right before "?"
  if (low.startsWith('müdür') && (count || !_QUESTION_NEXT_RE.match(text, end(m)))) return null;
  return m;
}

function _last_word(s) {
  const toks = split(s);
  return toks.length ? tr_lower(toks[toks.length - 1]) : '';
}

// rx.sub(repl, text), marking expansions followed by a clitic so _pass_clitics re-agrees it.
// Only a changed last word counts as an expansion: "NATO" -> "nato" keeps the writer's "NATO mu".
function _sub(rx, repl, text) {
  const out = [];
  let pos = 0;
  for (const m of rx.finditer(text)) {
    const neu = repl(m);
    out.push(text.slice(pos, m.index), neu);
    const count = Boolean(_BARE_COUNT_RE.fullmatch(m[0]));
    if (_last_word(neu) && _last_word(neu) !== _last_word(m[0]) && _clitic_at(text, end(m), count)) out.push(_MARK);
    pos = end(m);
  }
  out.push(text.slice(pos));
  return out.join('');
}

// Re-agree mI / dA after a marked expansion with its last spoken vowel: "beş lira mi" -> "beş lira mı".
// The only chain, dA + mI, is re-agreed left to right: "beş lira de mi" -> "beş lira da mı".
function _pass_clitics(text) {
  const [head, ...parts] = text.split(_MARK);
  let out = head;
  for (const part of parts) {
    let pos = 0;
    let prev = '';
    let m;
    while ((m = _clitic_at(part, pos)) && (!prev || ((prev === 'da' || prev === 'de') && tr_lower(m[2])[0] === 'm'))) {
      const word = m[2];
      prev = harmonize_clitic(tr_lower(out.slice(cp_back(out, out.length, 48)) + m[1]), tr_lower(word));
      out += m[1] + (word !== tr_lower(word) ? tr_upper(prev) : prev);
      pos = end(m);
    }
    out += part.slice(pos);
  }
  return out;
}

function _context_before(text, pos, width = 32) {
  return tr_lower(text.slice(cp_back(text, pos, width), pos));
}

// the text after the last digit (re.split(r"\d", context)[-1]) and its lowercase words
const _LOW_WORD_RE = re.compile(`[${LOW}]+`);
const _DIGIT_RE = re.compile(D);
const _tail_words = (context) => _LOW_WORD_RE.findall(_DIGIT_RE.split(context).pop());

// True if one of the last `last` words before a number starts with one of `words`.
// Digits break the window: in "kodunu gir, 21 Kasım 2025", the year is not a code.
function _has_word(context, words, last = 3) {
  const toks = _tail_words(context);
  if (toks.length && _MONTHS_LOWER.has(toks[toks.length - 1])) return false;
  return toks.slice(-last).some((t) => words.some((w) => t.startsWith(w)));
}

// nominal suffixes after a cue word: possessive / plural / -lI / -sIz, then a case ending
// ("numara-nız", "kod-u-nu", "numara-sı-nda", "hesab-ı-na", "kod-lu", "kişi-ye", "yıl-ı-nda", "lira-lık")
const _NOMINAL_SUFFIX_RE = re.compile(
  '(?:[ıiuü]|s[ıiuü]|[ıiuü]?m(?:[ıiuü]z)?|[ıiuü]?n(?:[ıiuü]z)?|l[ae]r(?:[ıi](?:m[ıi]z|n[ıi]z|[mn])?)?' +
  '|l[ıiuü]k?|s[ıiuü]z)?' +
  '(?:n?[ıiuü]n|y?[ae]|n[ae]|n?[dt][ae]n?|y?l[ae]|y?[ıiuü]|n[ıiuü])?',
);

// tok is one of `cues`, as a whole word or with nominal suffixes ("numaranız", "kişiye"); never a word that merely
// starts with a cue ("normalde" is not "no", "kartal" not "kart", "takipçi" not "takip").
function _cue_word(tok, cues) {
  return cues.some(
    (c) => tok === c || (!L.ID_EXACT.has(c) && tok.startsWith(c) && Boolean(_NOMINAL_SUFFIX_RE.fullmatch(tok, c.length))),
  );
}

// Like _has_word, with whole-word cue matching (_cue_word).
function _has_cue(context, cues, last) {
  const toks = _tail_words(context);
  if (toks.length && _MONTHS_LOWER.has(toks[toks.length - 1])) return false;
  return toks.slice(-last).some((t) => _cue_word(t, cues));
}

const _NEXT_LOW_WORD_RE = re.compile(`[ \\t]*([${LOW}]+)`);
// A count, not an identifier: "Müşteri sayısı 25000", "Telefonla 15000 kişiye", "Kart limiti 25000 lira".
function _quantity(context, after) {
  const toks = _tail_words(context);
  if (toks.length && L.QUANTITY_BEFORE.has(toks[toks.length - 1])) return true;
  const nxt = _NEXT_LOW_WORD_RE.match(after);
  return Boolean(nxt) && _cue_word(nxt[1], L.QUANTITY_AFTER);
}

const _MONTHS_LOWER = new Set(L.MONTHS.filter(Boolean).map(tr_lower));
// spelled-out unit words after a dot decimal make it a quantity: "10.40 metre" -> on virgül kırk metre
const _UNIT_WORDS_RE = [...new Set([...L.UNITS.values(), 'kilo', 'lira', 'dolar', 'avro', 'kuruş', 'derece', 'saat',
  'dakika', 'saniye', 'gün', 'yıl', 'ay', 'hafta'])].sort((a, b) => plen(b) - plen(a)).join('|');

// --------------------------------------------------------------------------- passes

// emoji, pictographs, flags, skin tones, keycap marks: not read aloud (a raw emoji reaches the TTS model otherwise).
//   [\U0001F000-\U0001FAFF\U0001FC00-\U0001FFFD\u2600-\u27BF\u2B00-\u2BFF\u2300-\u23FF\uFE0E\uFE0F\u20E3]+
// Written without the u flag, the astral ranges as surrogate pairs (the text has no lone surrogate here): this and
// _INVISIBLE_RE delete characters, which can bring a pair Safari 26 misreads together, so they run before the check.
const _EMOJI_RE = new RegExp(
  String.raw`(?:[\u2600-\u27bf\u2b00-\u2bff\u2300-\u23ff\ufe0e\ufe0f\u20e3]|\ud83c[\udc00-\udfff]` +
  String.raw`|\ud83d[\udc00-\udfff]|\ud83e[\udc00-\udeff]|\ud83f[\udc00-\udffd])+`, 'g',
);

// the patterns of _preclean (compiled once; Python's re caches its inline patterns the same way)
const _INVISIBLE_RE = /[\u200b\u200c\u200d\ufeff]/g;
const _APOSTROPHE_RE = re.compile(String.raw`(?=[’‘\x60´ʼ′])${lb(WC)}[’‘\x60´ʼ′](?=${W})`, has('’‘`´ʼ′'));
const _QUOTE_RE = re.compile('[“”„«»]');
const _SINGLE_QUOTE_RE = re.compile('[’‘]');
const _DASH_NUM_RE = re.compile(`(?=[${SC}–—])${lb(D)}${S}?[–—]${S}?(?=${D})`, has('–—'));
const _DASH_RE = re.compile(`${S}[–—]${S}`, has('–—'));
const _BLANKS_RE = re.compile(String.raw`[ \t]+`);

// An Error for a text this engine cannot read right (py.js U_PAIR_BUG): not a Python exception, so the worker asks
// the Python front-end, which reads it.
function _engine_error(text) {
  const at = text.search(U_PAIR_RE);
  const pair = [...text.slice(at, at + 2)].map((c) => `U+${c.codePointAt(0).toString(16).toUpperCase()}`).join(' ');
  const e = new Error(`this browser's RegExp misreads ${pair} (Safari 26.0 to 26.5)`);
  e.name = 'EngineError';
  return e;
}

function _preclean(text) {
  text = text.normalize('NFC').replaceAll(_MARK, '');
  text = text.replace(_EMOJI_RE, ' ');
  text = text.replace(_INVISIBLE_RE, '');
  // from here on a pass brings such a pair together only by deleting what stands between: _pass_symbols drops "#"
  // and "@" (checked again there); the rest replace a match whose last character was already next to the following one
  if (U_PAIR_BUG && U_PAIR_RE.test(text)) throw _engine_error(text);
  text = text.replaceAll('\u{a0}', ' ').replaceAll('\u{202f}', ' ');
  // apostrophes between word characters -> '
  text = _APOSTROPHE_RE.sub("'", text);
  text = _QUOTE_RE.sub('"', text);
  text = _SINGLE_QUOTE_RE.sub("'", text);
  text = text.replaceAll('…', '...');
  // en/em dash between numbers -> hyphen (ranges); elsewhere a pause
  text = _DASH_NUM_RE.sub('-', text);
  text = _DASH_RE.sub(', ', text);
  text = text.replaceAll('−', '-');
  text = _BLANKS_RE.sub(' ', text);
  return strip(text);
}

// Written forms of a FOREIGN_NAMES key: as written, ALL CAPS, and unless exact-case also lower/Capitalized.
// An exact-case key gets no Turkish ALL CAPS form either: "GEMİNİ" is the Turkish word "gemini" in capitals.
function _name_forms(key) {
  if (L.FOREIGN_NAMES_EXACT_CASE.has(key)) return new Set([key, upper(key)]);
  const low = lower(key);
  const head = first_cp(low);
  const tail = low.slice(head.length);
  // ALL CAPS (Latin and Turkish: "INSTAGRAM", "İNSTAGRAM"), youtube, Youtube, İnstagram
  return new Set([key, upper(key), tr_upper(key), low, upper(head) + tail, tr_upper(head) + tail]);
}

// Among keys of one length the order differs from Python's (a set's order there), which no match depends on: two
// different literals of one length never both match at one place.
function _build_names() {
  const table = new Map();
  for (const [key, spoken] of L.FOREIGN_NAMES) {
    for (const form of _name_forms(key)) if (!table.has(form)) table.set(form, spoken);
  }
  for (const [key, spoken] of L.FOREIGN_NAMES) table.set(key, spoken); // an exact key beats another key's generated form
  const alt = [...table.keys()].sort((a, b) => plen(b) - plen(a)).map(re.escape).join('|');
  const first = [...new Set([...table.keys()].map(first_cp))].join('').replace(/[\\\]^-]/g, '\\$&');
  // not inside a word, URL, e-mail address or handle ("google.com", "@google", "/google"); an optional
  // apostrophe suffix ("Google'da"), then a word end ((?=[first]): a first letter of some name, tested first)
  const rx = re.compile(
    String.raw`(?=[${first}])${nlb(String.raw`${WC}@./:\\`)}(?<!(?=${W}).')(${alt})(?:'([${LET}]+))?` +
    String.raw`(?![${WC}@+]|'[${LET}]|[./:]${W})`,
  );
  return [table, rx];
}

const [_NAMES, _NAMES_RE] = _build_names();

// Brands and loanwords to their Turkish pronunciation: "iPhone'umdan" -> "ayfonumdan", "Google'da" -> "Gugılda".
// A Capitalized name stays capitalized (Turkish rules: "Instagram" -> "İnstagram", never a dotless ı);
// ALL-CAPS and lowercase ones come out lowercase, like spelled acronyms.
function _pass_names(text) {
  const repl = (m) => {
    const written = m[1];
    const spoken = _suffix(_NAMES.get(written), m[2]);
    return isupper(first_cp(written)) && !is_all_caps(written) ? tr_capitalize(spoken) : spoken;
  };
  return _sub(_NAMES_RE, repl, text);
}

// re.I: ci() spells each letter the way Python's IGNORECASE matches it
const _HTTP = `${ci('http')}${ci('s')}?://`;
const _URL_TLDS = ['com', 'net', 'org', 'gov', 'edu', 'io', 'ai', 'tr', 'co', 'info', 'biz', 'me', 'app', 'dev'];
const _URL_TLD_ALT = _URL_TLDS.map(ci).join('|');
const _URL_SRC = String.raw`${B}(?:${_HTTP})?(?:${ci('www')}\.)?[${WC}\-]+(?:\.[${WC}\-]+)*\.(?:${_URL_TLD_ALT})` +
  String.raw`(?:\.[${CI_A_Z}]{2})?(?:/[^${SC}]*)?`;
const _EMAIL_SRC = String.raw`${B}[${WC}.+\-]+@[${WC}\-]+(?:\.[${WC}\-]+)+${B}`;
const _LABEL_CHAR = new RegExp(`^[${WC}\\-]$`, 'u'); // [\w-]: a character of a domain label
const _LOCAL_CHAR = new RegExp(`^[${WC}.+\\-]$`, 'u'); // [\w.+-]: of an e-mail address before the "@"

// The labels joined by single dots ([\w-]+(?:\.[\w-]+)*, as long as it goes) around the dot at q, which has a label
// character before it and after it: [start, end).
function _chain(text, q) {
  let a = q;
  for (;;) {
    while (a > 0 && _LABEL_CHAR.test(char_before(text, a))) a = cp_back(text, a, 1);
    if (a >= 2 && text[a - 1] === '.' && _LABEL_CHAR.test(char_before(text, a - 1))) a--;
    else break;
  }
  let e = q;
  while (text[e] === '.' && _LABEL_CHAR.test(char_at(text, e + 1))) {
    e++;
    while (e < text.length && _LABEL_CHAR.test(char_at(text, e))) e = cp_fwd(text, e, 1);
  }
  return [a, e];
}

// re.compile(URL, re.I).finditer, the same matches without its O(n²) part. The labels before ".com" are
// [\w-]+(?:\.[\w-]+)*, so Python's search, at every word boundary of a long chain like "1.1.1..." or "a-a-a...", takes
// in the rest of the chain and gives it back looking for a ".tld". Here the pattern runs only where it can match.
// A match ends its labels at a "site", a dot with a label character before it and a TLD after it, and starts in the
// chain of that site (or up to 8 characters before it: "https://"). From a start p the labels take the chain to its
// end and give back to the last site after p, so p matches if any site of the chain comes after it (and the word
// boundary holds), and a start that fails because no site is left fails for every later start of the chain too. So
// for the first site after pos the starts from max(pos, chain start - 8) to the chain's last site are tried in order
// (the first word boundary before a label matches); if none matches, the chain is done.
const _URL_RE = {
  sticky: new RegExp(_URL_SRC, 'uy'),
  sites: new RegExp(`${lb(`${WC}\\-`)}\\.(?=${_URL_TLD_ALT})`, 'gu'),
  *finditer(text) {
    const sites = [];
    this.sites.lastIndex = 0;
    for (let m; (m = this.sites.exec(text));) sites.push(m.index);
    let pos = 0;
    let k = 0;
    while (k < sites.length) {
      if (sites[k] <= pos) { // a match from pos on ends its labels after pos
        k++;
        continue;
      }
      const [start, stop] = _chain(text, sites[k]);
      let last = k;
      while (last + 1 < sites.length && sites[last + 1] < stop) last++;
      let m = null;
      for (let p = Math.max(pos, cp_back(text, start, 8)); p < sites[last] && !m; p = cp_fwd(text, p, 1)) {
        this.sticky.lastIndex = p;
        m = this.sticky.exec(text);
      }
      if (m) {
        yield m;
        pos = end(m);
      } else {
        k = last + 1;
      }
    }
  },
};

// _EMAIL_RE.finditer, the same matches without its O(n²) part. The part before "@" is [\w.+-]+, so Python's search
// takes in a whole run of those characters from each of its word boundaries ("ş+ş+ş+...", "1.1.1...") and fails at
// the end when no "@" follows. A match starts in the run right before an "@": from any start in it, that part runs to
// the "@" and the rest of the match does not depend on the start, so only the run's first word boundary (at or after
// pos) needs trying; if it fails, every start before that "@" does.
const _EMAIL_RE = {
  sticky: new RegExp(_EMAIL_SRC, 'uy'),
  boundary: new RegExp(B, 'gu'),
  *finditer(text) {
    let pos = 0;
    for (let at = text.indexOf('@'); at >= 0; at = text.indexOf('@', at + 1)) {
      if (at <= pos) continue;
      let a = at;
      while (a > pos && _LOCAL_CHAR.test(char_before(text, a))) a = cp_back(text, a, 1);
      if (a === at) continue;
      this.boundary.lastIndex = a;
      const b = this.boundary.exec(text);
      if (!b || b.index >= at) continue;
      this.sticky.lastIndex = b.index;
      const m = this.sticky.exec(text);
      if (m) {
        yield m;
        pos = end(m);
      }
    }
  },
};

const _TLD = new Map(Object.entries({ com: 'kom', tr: 'te re', net: 'net', org: 'org', gov: 'gov', edu: 'edu', io: 'ay o',
  ai: 'ey ay' }));

function _read_web_token(tok) {
  const parts = [];
  for (const piece of re.split(String.raw`([./@_\-])`, tok)) {
    if (!piece) continue;
    if (piece === '.') parts.push('nokta');
    else if (piece === '/') parts.push('eğik çizgi');
    else if (piece === '@') parts.push('et');
    else if (piece === '_') parts.push('alt çizgi');
    else if (piece === '-') parts.push('tire');
    else if (lower(piece) === 'www') parts.push('ve ve ve');
    else if (_TLD.has(lower(piece))) parts.push(_TLD.get(lower(piece)));
    else if (isdecimal(piece)) { // not "²" ("¹".isdigit() is True)
      parts.push(plen(piece) <= 4 ? read_digit_group(piece) : digits_to_words(piece));
    } else parts.push(tr_lower(piece));
  }
  return parts.join(' ');
}

function _pass_web(text) {
  text = _sub(_EMAIL_RE, (m) => _read_web_token(m[0]), text);

  const url = (m) => {
    const tok = rstrip(re.sub(`^${_HTTP}`, '', m[0]), '/');
    return _read_web_token(tok);
  };

  return _sub(_URL_RE, url, text);
}

// ABBREVIATIONS, and each title also in capitals and without the spaces after its dots
const _ABBR_FORMS = (() => {
  const forms = new Map(L.ABBREVIATIONS);
  for (const key of L.TITLE_ABBREVIATIONS) {
    const spoken = L.ABBREVIATIONS.get(key);
    for (const form of [key.replaceAll('. ', '.'), tr_upper(key), tr_upper(key).replaceAll('. ', '.')]) {
      if (!forms.has(form)) forms.set(form, spoken);
    }
  }
  return forms;
})();

function _build_abbr_re() {
  const keys = [..._ABBR_FORMS.keys()].sort((a, b) => plen(b) - plen(a));
  const alt = keys.map(re.escape).join('|');
  const first = [...new Set(keys.map(first_cp))].join('');
  // every key has a "." or a ":" ("Dr.", "No:")
  return re.compile(`(?=[${first}])${nlb(`${WC}.`)}(${alt})(?=[${SC},;:!?)"']|${END})`, has('.:'));
}

const _ABBR_RE = _build_abbr_re();
const _PAGE_RE = re.compile(String.raw`(?=[sc])${nlb(`${WC}.`)}([sc])\.${S}?(?=${D})`, has_digit);
const _NUMBER_LABEL_RE = re.compile(`(?=N)${nlb(`${WC}.`)}(No|Nu|NO)${S}?[:.]${S}?(?=${D})`, has_digit);
const _LABEL_RE = re.compile(`(?=[KDBTG])${nlb(`${WC}.`)}(Kat|Daire|Blok|D|Tel|GSM)${S}?:${S}?(?=${D})`, has(':'));
const _LABEL_WORDS = new Map([['D', 'daire'], ['Tel', 'telefon'], ['GSM', 'ge se me']]);
const _SPACE_CAPITAL_RE = re.compile(`${S}+[${UP}]`);

function _pass_abbreviations(text) {
  const repl = (m) => {
    const key = m[1];
    let out = _ABBR_FORMS.get(key);
    if (L.SENTENCE_FINAL_ABBR.has(key)) {
      const rest = text.slice(end(m));
      if (!strip(rest) || _SPACE_CAPITAL_RE.match(rest)) out += '.';
    }
    return out;
  };

  text = _sub(_ABBR_RE, repl, text);
  text = _NUMBER_LABEL_RE.sub('numara ', text);
  text = _LABEL_RE.sub((m) => `${_LABEL_WORDS.get(m[1]) ?? m[1]} `, text);
  return _PAGE_RE.sub((m) => (m[1] === 's' ? 'sayfa ' : 'cilt '), text);
}

const _PRODUCT_RE = re.compile(`(?=[ \\tx×*])${lb(D)}[ \\t]?[x×*][ \\t]?(?=[${D}₺$€£])`, has('x×*'));

// x, ×, * between numbers is "çarpı" ("12 x 833,25 TL", "4*5"); runs before the numbers become words.
function _pass_products(text) {
  const repl = (m) => {
    const s = m.index;
    // hexadecimal 0x1F (text[s - 1] is "0", one UTF-16 unit, so s < 2 counts characters too)
    if (strip(m[0]) === 'x' && text[s - 1] === '0' && (s < 2 || !isalnum(char_before(text, s - 1)))) return m[0];
    return ' çarpı ';
  };

  return _PRODUCT_RE.sub(repl, text);
}

const _PHONE_RE = re.compile(
  String.raw`(?=[+(${D}])${nlb(`${WC}+`)}(\+90[${SC}.\-]?|0[${SC}.\-]?)?\(?([2-58]${D}{2})\)?[${SC}.\-]?` +
  String.raw`(${D}{3})[${SC}.\-]?` +
  String.raw`(${D}{2})[${SC}.\-]?(${D}{2})${SUF}(?!${W})`, has_digit,
);
const _PHONE_444_RE = re.compile(
  String.raw`(?=4)${nlb(WC)}(444)[${SC}.\-]?(${D})[${SC}.\-]?(${D}{3})(?!${D})`, (t) => t.includes('444'),
);

function _pass_phones(text) {
  const repl = (m) => {
    const [raw, prefix, area, a, b, c, suf] = m;
    // without a 0/+90 prefix, require separators so plain 10-digit ids stay ids
    if (!prefix && !re.search(`[${SC}.()\\-]`, raw)) return raw;
    let parts;
    if (prefix && prefix.startsWith('+')) parts = ['artı doksan', number_to_words(py_int(area))];
    else if (prefix) parts = [`sıfır ${number_to_words(py_int(area))}`];
    else parts = [number_to_words(py_int(area))];
    parts.push(read_digit_group(a), read_digit_group(b), _suffix(read_digit_group(c), suf));
    return parts.join(', ');
  };

  text = _sub(_PHONE_RE, repl, text);
  return _sub(
    _PHONE_444_RE,
    (m) => `dört yüz kırk dört, ${read_digit_group(m[2])}, ${read_digit_group(m[3])}`,
    text,
  );
}

const _IBAN_RE = re.compile(
  String.raw`(?=T)${B}TR${S}?(${D}{2})((?:${S}?${D}{4}){5})${S}?(${D}{2})${B}`, (t) => t.includes('TR'),
);
const _FOUR_DIGITS_RE = re.compile(`${D}{4}`);

function _pass_iban(text) {
  const repl = (m) => {
    // check digits and the last pair as numbers, the four-digit groups digit by digit (easy to write down)
    const groups = [read_digit_group(m[1])];
    for (const g of _FOUR_DIGITS_RE.findall(m[2])) groups.push(digits_to_words(g));
    groups.push(read_digit_group(m[3]));
    return `te re ${groups.join(', ')}`;
  };

  return _sub(_IBAN_RE, repl, text);
}

const _PLATE_RE = re.compile(
  String.raw`(?=[0-8])${nlb(WC)}(0[1-9]|[1-7]${D}|8[01])${S}([${UP}]{1,3})${S}(${D}{2,5})${SUF}(?!${W})`, has_digit,
);

function _pass_plates(text) {
  const repl = (m) => {
    const [, city, letters, num, suf] = m;
    let tail;
    if (plen(num) === 4) {
      const d = [...num];
      tail = `${read_digit_group(d.slice(0, 2).join(''))} ${read_digit_group(d.slice(2).join(''))}`;
    } else {
      tail = read_digit_group(num);
    }
    return _suffix(`${read_digit_group(city)} ${_letters(letters)} ${tail}`, suf);
  };

  return _sub(_PLATE_RE, repl, text);
}

function _date_words(d, mo, y) {
  let out = `${number_to_words(d)} ${L.MONTHS[mo]}`;
  if (y) out += ` ${number_to_words(py_int(y))}`;
  return out;
}

const _DATE_RE = re.compile(
  String.raw`${NBD}(${D}{1,2})([./\-])(${D}{1,2})\2(${D}{4}|${D}{2})${SUF}(?!${W}|[.,]${D})`, has_digit,
);
const _MONTH_YEAR_RE = re.compile(
  String.raw`${NBD}(0?[1-9]|1[0-2])[./]((?:19|20)${D}{2})${SUF}(?!${W}|[.,]${D})`, has_digit,
);
const _ISO_DATE_RE = re.compile(String.raw`${NBD}(${D}{4})-(${D}{2})-(${D}{2})${SUF}(?!${W}|[.,]${D})`, has_digit);
// year first with slashes or dots: 2026/12/31, 2026/1/5, 2026.12.31
const _YMD_RE = re.compile(
  String.raw`${NBD}((?:1[89]|20)${D}{2})(?:/(${D}{1,2})/(${D}{1,2})|\.(${D}{2})\.(${D}{2}))${SUF}(?!${W}|[.,/]${D})`,
  has_digit,
);

function _pass_dates(text) {
  const iso = (m) => {
    const [, y, mo, d, suf] = m;
    if (int(mo) >= 1 && int(mo) <= 12 && int(d) >= 1 && int(d) <= 31) {
      return _suffix(_date_words(int(d), int(mo), y), suf);
    }
    return m[0];
  };

  const ymd = (m) => {
    const [, y, mo1, d1, mo2, d2, suf] = m;
    const mo = int(mo1 || mo2);
    const d = int(d1 || d2);
    if (mo >= 1 && mo <= 12 && d >= 1 && d <= 31) return _suffix(_date_words(d, mo, y), suf);
    return m[0];
  };

  text = _sub(_YMD_RE, ymd, text);

  const dmy = (m) => {
    const [, d, , mo, y, suf] = m;
    if (int(mo) >= 1 && int(mo) <= 12 && int(d) >= 1 && int(d) <= 31) {
      return _suffix(_date_words(int(d), int(mo), y), suf);
    }
    return m[0];
  };

  text = _sub(_DATE_RE, dmy, _sub(_ISO_DATE_RE, iso, text));

  const month_year = (m) => _suffix(`${L.MONTHS[int(m[1])]} ${number_to_words(py_int(m[2]))}`, m[3]);

  return _sub(_MONTH_YEAR_RE, month_year, text);
}

function _time_words(h, mi, s = null) {
  let out = number_to_words(py_int(h));
  if (s != null) return `${out} ${read_digit_group(mi)} ${read_digit_group(s)}`;
  if (mi !== '00') out += ` ${read_digit_group(mi)}`;
  return out;
}

const _TIME_COLON_RE = re.compile(
  String.raw`${NBD}([01]?${D}|2[0-4]):([0-5]${D})(?::([0-5]${D}))?${SUF}(?!${W}|[.,]${D})`, has_digit,
);
const _TIME_DOT_RE = re.compile(String.raw`${NBD}([01]?${D}|2[0-4])\.([0-5]${D})${SUF}(?!${W}|[.,]${D})`, has_digit);
const _TIME_RANGE_RE = re.compile(
  String.raw`${NBD}([01]?${D}|2[0-4])[:.]([0-5]${D})${S}?-${S}?([01]?${D}|2[0-4])[:.]([0-5]${D})${SUF}` +
  String.raw`(?!${W}|[.,]${D})`,
  has_digit,
);

const _SIGN_BEFORE_RE = re.compile(`[%‰₺$€£]${S}?${END}`);
const _TIME_WORD_AFTER_RE = re.compile(`${S}+(?:${L.TIME_CONTEXT.join('|')})`);

function _pass_times(text) {
  text = _sub(
    _TIME_RANGE_RE,
    (m) => _suffix(`${_time_words(m[1], m[2])} ${_ila(text, end(m))} ${_time_words(m[3], m[4])}`, m[5]),
    text,
  );
  text = _sub(_TIME_COLON_RE, (m) => _suffix(_time_words(m[1], m[2], m[3]), m[4]), text);

  const dot = (m) => {
    const [, h, mi, suf] = m;
    if (_SIGN_BEFORE_RE.search(text.slice(cp_back(text, m.index, 2), m.index))) return m[0];
    const before = _context_before(text, m.index);
    const after = tr_lower(text.slice(end(m), cp_fwd(text, end(m), 16)));
    let timey = Boolean(suf) || _has_word(before, L.TIME_CONTEXT) || mi === '00' || mi === '30';
    timey = timey || Boolean(_TIME_WORD_AFTER_RE.match(after));
    if (!timey) return m[0];
    return _suffix(_time_words(h, mi), suf);
  };

  return _sub(_TIME_DOT_RE, dot, text);
}

const _CUR_SYMBOLS = String.raw`₺|\$|€|£|¥|₽`;
const _CUR_CODES = 'TL|TRY|USD|EUR|GBP|JPY|RUB|CHF';
const _CUR_PREFIX_RE = re.compile(
  `(?=[₺$€£¥₽])${nlb(WC)}(${_CUR_SYMBOLS})${S}?(${NUMU})${SUF}(?!${W}|[.,]${D})`, has('₺$€£¥₽'),
);
const _CUR_POSTFIX_RE = re.compile(
  `${NBD}(${NUMU})(?:${S}?-${S}?(${NUMU}))?${S}?(${_CUR_SYMBOLS}|(?:${_CUR_CODES})(?!${W}))${SUF}(?!${W}|[.,]${D})`,
  has_digit,
);
const _SCALE_WORDS = /bin|milyon|milyar|trilyon/;
const _SCALE_CUR_RE = re.compile(
  `(?=[bmt])${B}(bin|milyon|milyar|trilyon)${S}(${_CUR_SYMBOLS}|${_CUR_CODES})(?!${W})${SUF}`,
  (t) => _SCALE_WORDS.test(t),
);
const _AMOUNT_RE = re.compile(`(${INT_TH}|${D}+)[.,](${D}{2})`);

function _money_words(amount, cur_word) {
  const sub = L.CURRENCY_SUBUNIT.get(cur_word);
  const m = _AMOUNT_RE.fullmatch(amount);
  if (sub && m && !re.fullmatch(INT_TH, amount)) {
    const major = number_to_words(py_int(m[1].replaceAll('.', '')));
    const minor = int(m[2]);
    return `${major} ${cur_word}${minor ? ` ${number_to_words(minor)} ${sub}` : ''}`;
  }
  return `${read_number(amount, 'virgül')} ${cur_word}`;
}

function _pass_money(text) {
  const prefix = (m) => {
    const [, sym, amount, suf] = m;
    return _suffix(_money_words(amount, L.CURRENCY_WORDS.get(sym)), suf);
  };

  const postfix = (m) => {
    const [, a, b, cur, suf] = m;
    const word = L.CURRENCY_WORDS.get(cur);
    if (b) return _suffix(`${read_number(a, 'virgül')} ${_ila(text, end(m))} ${_money_words(b, word)}`, suf);
    return _suffix(_money_words(a, word), suf);
  };

  text = _sub(_CUR_PREFIX_RE, prefix, text);
  text = _sub(_CUR_POSTFIX_RE, postfix, text);
  return _sub(_SCALE_CUR_RE, (m) => _suffix(`${m[1]} ${L.CURRENCY_WORDS.get(m[2])}`, m[3]), text);
}

const _PCT_PREFIX_RE = re.compile(
  `(?=[%‰])${nlb(WC)}([%‰])${S}?(${NUMU})(?:${S}?-${S}?%?${S}?(${NUMU}))?${SUF}(?!${W}|[.,]${D})`, has('%‰'),
);
const _PCT_POSTFIX_RE = re.compile(`${NBD}(${NUMU})${S}?%${SUF}(?!${W})`, has('%'));

function _pass_percent(text) {
  const pre = (m) => {
    const [, sym, a, b, suf] = m;
    const word = sym === '%' ? 'yüzde' : 'binde';
    const body = !b ? read_number(a, 'virgül') : _read_range(a, b, 'virgül', _ila(text, end(m)));
    return _suffix(`${word} ${body}`, suf);
  };

  text = _sub(_PCT_PREFIX_RE, pre, text);
  return _sub(_PCT_POSTFIX_RE, (m) => _suffix(`yüzde ${read_number(m[1], 'virgül')}`, m[2]), text);
}

const _DEG_RE = re.compile(`${NBS}(-?${NUMU})${S}?°${S}?([CF])?(?![${LET}])${SUF}`, has('°'));

function _pass_degrees(text) {
  const repl = (m) => {
    const [, num, scale, suf] = m;
    let out = `${read_number(num, 'virgül')} derece`;
    if (scale === 'F') out += ' fahrenhayt';
    return _suffix(out, suf);
  };

  return _sub(_DEG_RE, repl, text);
}

const _by_length = (keys) => [...keys].sort((a, b) => plen(b) - plen(a)); // sorted(key=len, reverse=True)

function _build_unit_re() {
  const alt = _by_length(L.UNITS.keys()).map(re.escape).join('|');
  return re.compile(`${NBS}(-?${NUMU})(?:${S}?-${S}?(${NUMU}))?${S}?(${alt})(?![${WC}²³/])${SUF}`, has_digit);
}

const _UNIT_RE = _build_unit_re();
const _SPEED_RE = re.compile(
  `${NBD}(${NUMU})(?:${S}?-${S}?(${NUMU}))?${S}?(${_by_length(L.SPEED_UNITS.keys()).map(re.escape).join('|')})` +
  `(?!${W})${SUF}`, has_digit,
);

function _pass_units(text) {
  const speed = (m) => {
    const [, a, b, unit, suf] = m;
    const [per, word] = L.SPEED_UNITS.get(unit);
    const num = !b ? read_number(a, 'virgül') : _read_range(a, b, 'virgül', _ila(text, end(m)));
    return _suffix(`${per} ${num} ${word}`, suf);
  };

  const unit = (m) => {
    const [, a, b, u, suf] = m;
    const num = !b ? read_number(a, 'virgül') : _read_range(a, b, 'virgül', _ila(text, end(m)));
    return _suffix(`${num} ${L.UNITS.get(u)}`, suf);
  };

  text = _sub(_SPEED_RE, speed, text);
  return _sub(_UNIT_RE, unit, text);
}

// tokens mixing letters and digits: TR42648163, A4, 5G, MP3, H2O, COVID-19, F-16
// ...with a decimal version at the end: GPT-3.5
//   (?<![\w'])(?=[LET\d-]*\d)(?=[LET\d-]*[UP])([LET\d]+(?:-[LET\d]+)*(?:(?<=\d)[.,]\d+)?)SUF(?![\w]|[.,]\d)
// The two lookaheads scan the run of [LET\d-] from each start: on a long run without a digit or a capital ("a-a-a...",
// "-----") Python tries every position of it, O(n²). finditer runs the pattern without them and tests them on what it
// matches: they hold for the same run from a later start only if they hold from this one (its rest of the run is
// shorter), so when they fail, no match starts before the run's end. A text without "-" has only one start per run
// (the others follow a letter or digit), so there the pattern as Python writes it is linear and runs as it is.
const _ALNUM_SRC = `${nlb(`${WC}'`)}(?=[${LET}${D}\\-]*${D})(?=[${LET}${D}\\-]*[${UP}])` +
  `([${LET}${D}]+(?:-[${LET}${D}]+)*(?:${lb(D)}[.,]${D}+)?)${SUF}(?!${W}|[.,]${D})`;
const _HAS_CAPITAL = new RegExp(`[${UP}]`);
const _ALNUM_RE = {
  plain: re.compile(`(?=[${LET}${D}])${_ALNUM_SRC}`),
  rx: new RegExp(
    `(?=[${LET}${D}])${nlb(`${WC}'`)}([${LET}${D}]+(?:-[${LET}${D}]+)*(?:${lb(D)}[.,]${D}+)?)${SUF}` +
    `(?!${W}|[.,]${D})`, 'gu',
  ),
  digit: new RegExp(`[${LET}${D}\\-]*${D}`, 'uy'),
  capital: new RegExp(`[${LET}${D}\\-]*[${UP}]`, 'uy'),
  run: new RegExp(`[${LET}${D}\\-]*`, 'uy'),
  *finditer(text) {
    if (!has_digit(text) || !_HAS_CAPITAL.test(text)) return; // every match has both
    if (!text.includes('-')) {
      yield* this.plain.finditer(text);
      return;
    }
    let pos = 0;
    for (;;) {
      this.rx.lastIndex = pos;
      const m = this.rx.exec(text);
      if (!m) return;
      this.digit.lastIndex = this.capital.lastIndex = this.run.lastIndex = m.index;
      if (this.digit.test(text) && this.capital.test(text)) {
        yield m;
        pos = end(m);
      } else {
        this.run.exec(text);
        pos = this.run.lastIndex;
      }
    }
  },
};

const _ALNUM_PIECE_RE = re.compile(`[${LET}]+|${D}+(?:[.,]${D}+)?`);

function _read_alnum(tok) {
  const out = [];
  for (const piece of _ALNUM_PIECE_RE.findall(tok)) {
    if (!isdigit(piece) && isdigit(first_cp(piece))) out.push(read_number(piece)); // GPT-3.5
    else if (isdigit(piece)) out.push(plen(piece) >= 5 ? digits_to_words(piece) : read_digit_group(piece));
    else if (L.ACRONYMS.has(piece)) out.push(L.ACRONYMS.get(piece));
    else if (is_all_caps(piece) || plen(piece) === 1) out.push(_pronounceable(piece) ? tr_lower(piece) : _letters(piece));
    else out.push(piece);
  }
  return out.join(' ');
}

function _pass_alnum(text) {
  // plain words with a trailing number like "Ankara2023" are rare; require caps letters
  const repl = (m) => _suffix(_read_alnum(m[1]), m[2]);

  return _sub(_ALNUM_RE, repl, text);
}

// either end may be negative: "-10 - 5 derece", "-5 - -2 gün" (a sign only after a space or the start)
const _RANGE_RE = re.compile(
  `(?=[\\-${D}])(?:${nlb(`${WC}.,\\-`)}(-)|${NB})(${NUM})${S}?-${S}?(-?)(${NUM})${SUF}(?!${W}|[.,]${D})`,
  (t) => t.includes('-') && has_digit(t),
);

const _SCORE_AFTER_RE = re.compile(`${S}*(?:${L.SCORE_CONTEXT.join('|')})`);

function _pass_ranges(text) {
  const repl = (m) => {
    const [, sa, a, sb, b, suf] = m;
    const before = _context_before(text, m.index, 40);
    const after = tr_lower(text.slice(end(m), cp_fwd(text, end(m), 30)));
    let av = _int_value(a.split(',')[0]);
    let bv = _int_value(b.split(',')[0]);
    if (sa && av !== null) av = -av;
    if (sb && bv !== null) bv = -bv;
    const first = read_number((sa || '') + a);

    const last = (v) => {
      if (v !== null && suf && !sb) return _attach_number_suffix(v, read_number(b), suf);
      return _suffix(read_number(sb + b), suf);
    };

    const descending = av !== null && bv !== null && bv <= av; // ranges go up; "3-1", "6-6", "2023-24" don't
    if (descending || _has_word(before, L.SCORE_CONTEXT) || _SCORE_AFTER_RE.match(after)) {
      return `${first} ${last(bv)}`;
    }
    return `${first} ${_ila(text, end(m))} ${last(_int_value(b))}`;
  };

  return _sub(_RANGE_RE, repl, text);
}

// N/M is a building / apartment number: after "No", "numara", "kapı", "daire"... or before "nolu".
const _LET_WORD_RE = re.compile(`[${LET}]+`);
const _NOLU_RE = re.compile(`${S}*(?:nolu|no'lu|numaralı)${B}`);
function _address_number(text, start, end_) {
  const words = _LET_WORD_RE.findall(_context_before(text, start, 24));
  if (words.length && L.ADDRESS_NUMBER_CONTEXT.has(words[words.length - 1])) return true;
  return _NOLU_RE.match(tr_lower(text.slice(end_, cp_fwd(text, end_, 16)))) !== null;
}

const _has_slash = (t) => t.includes('/');
const _FRACTION_RE = re.compile(
  `(?=${D})${nlb(`${WC}./,`)}(${D}{1,3})/(${D}{1,3})${SUF}(?![${WC}/]|[.,]${D})`, _has_slash,
);
const _RATIO_RE = re.compile(`${NBD}(${D}+,${D}+)/(${D}{1,3})${SUF}(?![${WC}/]|[.,]${D})`, _has_slash);

function _pass_fractions(text) {
  const repl = (m) => {
    const a = int(m[1]);
    const b = int(m[2]);
    const suf = m[3];
    if (a === 24 && b === 7) return _suffix('yirmi dört yedi', suf);
    if (_address_number(text, m.index, end(m))) { // "No: 12/4", "Daire 3/1", "10/2 nolu": never a date
      return _suffix(`${number_to_words(a)} bölü ${number_to_words(b)}`, suf);
    }
    if (b === 0) return m[0];
    if (a >= b && a >= 1 && a <= 31 && b >= 1 && b <= 12) return _suffix(_date_words(a, b, null), suf);
    if (a < b) return _suffix(`${attach_suffix(number_to_words(b), 'de')} ${number_to_words(a)}`, suf);
    return _suffix(`${number_to_words(a)} bölü ${number_to_words(b)}`, suf);
  };

  const ratio = (m) => _suffix(`${read_number(m[1])} bölü ${number_to_words(py_int(m[2]))}`, m[3]);

  text = _sub(_RATIO_RE, ratio, text);
  return _sub(_FRACTION_RE, repl, text);
}

const _ROMAN = 'M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})';
const _ROMAN_RE = re.compile(`(?=[MDCLXVI.])${nlb(`${WC}.`)}(${_ROMAN})\\.(?=${S}+([${LET}][${LET}']*))`, has('.'));
const _ROMAN_VALUES = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000 };

function _roman_to_int(s) {
  let total = 0;
  for (let i = 0; i < s.length; i++) {
    const v = _ROMAN_VALUES[s[i]];
    if (i + 1 < s.length && _ROMAN_VALUES[s[i + 1]] > v) total -= v;
    else total += v;
  }
  return total;
}

const _ROMAN_SUFFIX_RE = re.compile(`(?=[IVXLC])${nlb(`${WC}.`)}([IVXLC]{2,}|[IVX])'([${LOW}]+)(?!${W})`, has("'"));

// A capitalized word after which "N." / "IV." is an ordinal: "Lig", "Murat", "Yüzyılda" (also "LİG").
function _ordinal_noun(word) {
  if (is_all_caps(word)) word = tr_capitalize(tr_lower(word));
  return L.ORDINAL_CAP_NOUNS.has(word.split("'")[0]) || L.ORDINAL_CAP_PREFIXES.some((p) => word.startsWith(p));
}

const _WORD_BEFORE_RE = re.compile(`([${LET}]+)[ \\t]+${END}`);
// a title ("Dr.", "doktor", "Sayın") or another initial ("J. I. Smith") right before a lone I. / X.
// (?i:...) is spelled out by ci()
const _TITLE_BEFORE_RE = re.compile(
  `(?:${nlb(WC)}(?:${L.INITIAL_TITLES.map(ci).join('|')})\\.?|${nlb(`${WC}.`)}[${UP}]\\.)[ \\t]+${END}`,
);

function _pass_roman(text) {
  const suffixed = (m) => {
    const [, tok, suf] = m;
    if (!re.fullmatch(_ROMAN, tok)) return m[0];
    if (plen(tok) === 1) {
      // the written suffix tells which reading the writer had in mind: "V'te" (beşte) and "Ek I'de"
      // (birde) are numerals, "X'te" (ikste) and "V'ye" (veye) letters; when it fits both or neither,
      // a regnal name or "Madde" before it makes it the numeral ("Louis X'un", "Madde V'e")
      const s = tr_lower(suf);
      const n = _roman_to_int(tok);
      const as_number = harmonize_suffix(number_to_words(n), s) === s;
      const as_letter = harmonize_suffix(L.LETTER_NAMES.get(tok), s) === s;
      if (as_letter && !as_number) return m[0];
      if (as_letter === as_number) {
        const prev = _WORD_BEFORE_RE.search(text, cp_back(text, m.index, 24), m.index);
        if (!(prev && _ordinal_noun(prev[1]))) return m[0];
      }
    }
    const n = _roman_to_int(tok);
    return _attach_number_suffix(n, number_to_words(n), suf);
  };

  text = _sub(_ROMAN_SUFFIX_RE, suffixed, text);

  const repl = (m) => {
    const tok = m[1];
    const nxt = m[2];
    if (!tok || (plen(tok) === 1 && !'IVX'.includes(tok))) return m[0];
    // before a capitalized word that is not a known noun, a lone V. is an initial ("V. Öztürk", but
    // "V. Murat"); I. and X. are ordinals ("I. Balkan Savaşı", "I. TBMM") unless a title or another
    // initial comes before them ("Dr. I. Yılmaz")
    if (plen(tok) === 1 && !(islower(first_cp(nxt)) || _ordinal_noun(nxt))) {
      if (tok === 'V' || _TITLE_BEFORE_RE.search(text, cp_back(text, m.index, 24), m.index)) return m[0];
    }
    return ordinal_to_words(_roman_to_int(tok));
  };

  return _sub(_ROMAN_RE, repl, text);
}

const _ORDINAL_DOT_RE = re.compile(`${NBD}(${D}{1,4})\\.(?=${S}+([${LET}][${LET}']*))`, has_digit);

function _pass_ordinals(text) {
  const repl = (m) => {
    const n = py_int(m[1]);
    const nxt = m[2];
    if (islower(first_cp(nxt)) || _ordinal_noun(nxt)) return ordinal_to_words(n);
    return m[0];
  };

  return _sub(_ORDINAL_DOT_RE, repl, text);
}

const _NUMBER_RE = re.compile(
  `${NBS}(-)?((?:${INT_TH}|${D}+)(?:,${D}+|\\.${D}+)?)(?:'([${LET}]+)|([${LOW}]+))?(?!${W})`, has_digit,
);
const _LIST_RE = re.compile(`${NBD}${D}+(?:,${D}+){2,}(?![${WC}.,])`, (t) => t.includes(',') && has_digit(t));

// dotted versions and addresses: 1.2.3, 10.15.7, 192.168.1.1 (thousands groups like 1.250.000 are numbers)
const _VERSION_RE = re.compile(`${NBD}${D}+(?:\\.${D}+){2,}(?!${W}|[.,]${D})`, (t) => t.includes('.') && has_digit(t));
// a version wildcard: 3.x, 2.7.x
const _VERSION_X_RE = re.compile(`${NBD}(${D}+(?:\\.${D}+)*)\\.[xX](?!${W})`, has('xX'));
// a word, a hyphen, a number: "klod-3.5", "covid-19" (capitalized ones are read by _pass_alnum)
const _WORD_HYPHEN_NUM_RE = re.compile(`(?<=[${LET}]{2})-(?=${D})`, (t) => t.includes('-'));

function _dotted(s) {
  return s.split('.').map((p) => (plen(p) <= 3 ? read_digit_group(p) : digits_to_words(p))).join(' nokta ');
}

const _NUMBER_WORDS = new Set(split(
  'sıfır bir iki üç dört beş altı yedi sekiz dokuz on yirmi otuz kırk elli altmış yetmiş seksen doksan ' +
  'yüz bin milyon milyar trilyon katrilyon',
));

// re.search(rf"([{LET}]+)\.?$", head).group(1), or '' when it does not match. Searching from the start, as Python does,
// costs the whole text before every postcode candidate (O(n²) on an address list); the match can only be the letters
// that end head, before an optional "." and the final newline $ allows, so they are read from the end.
function _letters_at_end(head) {
  let r = head.length;
  if (head.endsWith('\n')) r--;
  if (head[r - 1] === '.') r--;
  let a = r;
  while (a > 0 && _LET_CHAR.test(head[a - 1])) a--;
  return head.slice(a, r);
}
const _LET_CHAR = new RegExp(`[${LET}]`); // LET is all BMP: one UTF-16 unit each

// A bare 5-digit Turkish postcode inside an address, before the place name: "..., 34728 Kadıköy".
//
// The address is named by a whole address word (ADDRESS_CONTEXT: "Mah.", "sokak", "No"; not "katılan",
// "mahallede"), and the number comes right after a comma, slash, colon, address word or door number
// ("No:5 D:3 34728 Kadıköy"), never after an ordinary word ("Mahallede 15000 Suriyeli yaşıyor").
// In ALL CAPS only a door number counts: "İSTİKLAL CADDESİ 50000 KİŞİYLE DOLDU" is a count.
const _NEXT_NAME_RE = re.compile(`[ \\t]+([${UP}][${LET}]*)`);
const _ADDRESS_WORD_RE = re.compile(`[${LOW}]+(?:'[${LOW}]+)?`);
function _postcode(text, m, num) {
  const nxt = _NEXT_NAME_RE.match(text, end(m)); // re.match(..., text[m.end():])
  const d = [...num];
  // the first two characters against ASCII bounds: UTF-16 order agrees with code point order there
  const first2 = d.slice(0, 2).join('');
  if (d.length !== 5 || !(first2 >= '01' && first2 <= '81') || !nxt) return false;
  const words = _ADDRESS_WORD_RE.findall(_context_before(text, m.index, 100));
  if (!words.some((w) => L.ADDRESS_CONTEXT.has(w))) return false;
  const head = rstrip(text.slice(0, m.index), ' \t');
  const prev = tr_lower(_letters_at_end(head));
  const door = isdecimal(last_cp(head)) || _NUMBER_WORDS.has(prev);
  if (is_all_caps(nxt[1])) return door;
  return door || !head || ',/:;\n'.includes(last_cp(head)) || L.ADDRESS_CONTEXT.has(prev);
}

const _ID_WORD_AFTER_RE = re.compile(`${S}*(?:numara|no${B}|nolu|kodu|kodlu)`);
const _SCALE_AFTER_RE = re.compile(`${S}+(?:bin|milyon|milyar|trilyon)${B}`);
const _UNIT_WORD_AFTER_RE = re.compile(`${S}+(?:${_UNIT_WORDS_RE})`);

function _pass_numbers(text) {
  text = _sub(_LIST_RE, (m) => m[0].split(',').map((x) => number_to_words(py_int(x))).join(', '), text);
  text = _sub(_VERSION_X_RE, (m) => `${_dotted(m[1])} nokta iks`, text);
  text = _sub(_VERSION_RE, (m) => (re.fullmatch(INT_TH, m[0]) ? m[0] : _dotted(m[0])), text);
  text = _WORD_HYPHEN_NUM_RE.sub(' ', text);

  const spoken = (m, sign, num, suf) => {
    const neg = Boolean(sign) && (m.index === 0 || !isalnum(char_before(text, m.index)));
    const prefix = neg ? 'eksi ' : sign ? '-' : '';
    const after = tr_lower(text.slice(end(m), cp_fwd(text, end(m), 24)));
    if (isdigit(num)) {
      const ctx = _context_before(text, m.index);
      const count = _quantity(ctx, after);
      const len = plen(num);
      const idlike = len >= 5 && (
        num.startsWith('0')
        || len >= 10
        || (!count && _has_cue(ctx, L.ID_CONTEXT, 3))
        || _ID_WORD_AFTER_RE.match(after) !== null
        || _postcode(text, m, num)
      );
      const codelike = len >= 4 && !count && _has_cue(ctx, L.CODE_CONTEXT, 2);
      if (idlike || codelike) {
        const words = len <= 8 ? digits_to_words(num) : read_long_digits(num);
        return prefix + _suffix(words, suf);
      }
      if (num.startsWith('0') && len > 1) return prefix + _suffix(read_digit_group(num), suf);
      const n = py_int(num);
      return prefix + _attach_number_suffix(n, number_to_words(n), suf);
    }
    const ctx_before = _context_before(text, m.index, 24);
    const dot_word = (
      _SCALE_AFTER_RE.match(after)
      || _UNIT_WORD_AFTER_RE.match(after)
      || _has_word(ctx_before, ['yüzde', 'oran', 'faiz', 'puan', 'not', 'ortalama'])
    ) ? 'virgül' : 'nokta';
    const words = read_number(num, dot_word);
    const val = !num.includes(',') && re.fullmatch(INT_TH, num) ? _int_value(num) : null;
    if (val !== null) return prefix + _attach_number_suffix(val, words, suf);
    return prefix + _suffix(words, suf);
  };

  const repl = (m) => {
    const [, sign, num, suf, bare] = m;
    if (bare !== undefined && !L.BARE_NUMBER_SUFFIXES.has(tr_lower(bare))) {
      // not a suffix ("4o", "2x", "1975yılında"): the number, then the letters as a word of their own
      return `${spoken(m, sign, num, null)} ${bare === 'x' ? 'iks' : bare}`;
    }
    return spoken(m, sign, num, bare === undefined ? suf : bare);
  };

  return _sub(_NUMBER_RE, repl, text);
}

// Safety net for "no digit is left": a digit run no pass recognized is read plainly ("on iki.31" never reaches TTS).
//
// A period after a number word is a decimal point ("on iki.31" -> on iki nokta otuz bir); after any other word
// it ends an abbreviation ("Tablo.3'te" -> Tablo üçte, "md.5" -> md beş). A suffix stays on its number.
function _pass_leftover_digits(text) {
  text = _WORD_DOT_DIGIT_RE.sub((m) => m[1] + (_NUMBER_WORDS.has(tr_lower(m[1])) ? ' nokta ' : ' '), text);

  const read = (m) => {
    const [, digits, suf] = m;
    if (plen(digits) > 21) return ` ${_suffix(digits_to_words(digits), suf)} `;
    const words = read_digit_group(digits);
    if (digits.startsWith('0')) return ` ${_suffix(words, suf)} `; // "05" keeps its zero: no ordinal
    return ` ${_attach_number_suffix(py_int(digits), words, suf)} `;
  };

  return _DIGITS_RE.sub(read, text);
}
// (?<![LET]): a match takes the whole run of letters before the "."; Python, failing at the start of a long run of
// letters without one, tries again from every letter of it (O(n²)), and every one of those fails the same way
const _WORD_DOT_DIGIT_RE = re.compile(`(?<![${LET}])([${LET}]+)\\.(?=${D})`, (t) => t.includes('.') && has_digit(t));
const _DIGITS_RE = re.compile(`(${D}+)(?:'([${LET}]+))?`, has_digit);

const _POWER_RE = re.compile(`(?:${nlb(WC)}([A-Za-z]{1,2}|${D}+(?:[.,]${D}+)?)|(\\)))([²³])(?!${W})`, has('²³'));

// x² -> iks kare, a³ -> a küp, 10² -> 10 kare, (a+b)² -> (a+b) kare, m² (no number before it) -> metrekare.
function _pass_powers(text) {
  const repl = (m) => {
    const [, b, paren, sup] = m;
    let base = b || paren;
    if (L.UNITS.has(base + sup)) return L.UNITS.get(base + sup);
    const power = sup === '²' ? 'kare' : 'küp';
    if (isalpha(base)) {
      if (plen(base) > 1) return m[0];
      base = L.LETTER_NAMES.get(tr_upper(base)) ?? base;
    }
    return `${base} ${power}`;
  };

  return _sub(_POWER_RE, repl, text);
}

const _ACRONYM_RE = re.compile(`(?=[${UP}])${nlb(`${WC}'`)}([${UP}][${UP}]+|PhD)(?:'([${LET}]+))?(?!${W})`);
const _SINGLE_LETTER_RE = re.compile(`(?=[A-NPR-ZÇĞİÖŞÜ])${nlb(`${WC}'.`)}([A-NPR-ZÇĞİÖŞÜ])(?=${S}+[${LOW}])`);
// a capital letter with a suffix: "X'te" -> ikste, "A'dan Z'ye" -> adan zeye (not the pronoun "O'nun"); the
// suffix is lowercase or ALL CAPS ("X'TE"), so a name with an elided article stays: "D'Artagnan", "L'Occitane"
const _SINGLE_LETTER_SUFFIX_RE = re.compile(
  `(?=[A-NPR-ZÇĞİÖŞÜ])${nlb(`${WC}'.`)}([A-NPR-ZÇĞİÖŞÜ])'([${LOW}]+|[${UP}]+)(?![${WC}'])`, has("'"),
);
// an initial before a name: "V. Öztürk" -> "ve. Öztürk", "J. K. Rowling"; the period stays (it may end a sentence)
const _INITIAL_RE = re.compile(`(?=[${UP}])${nlb(`${WC}'.`)}([${UP}])\\.(?=${S}+[${UP}])`, has('.'));
const _SENTENCE_SPLIT_RE = re.compile(`(?<=[.!?])${S}+`, null, false);

// lowercase words the number passes wrote ("on dört otuzda", "elli bin", "yüzde", "lira"): in "SAAT 14:30'DA
// BAŞLAYACAK DÖRT ETKİNLİK" they do not make the headline a mixed-case sentence
const _EXPANSION_STEMS = [...new Set([..._NUMBER_WORDS, 'dörd', 'nokta', 'virgül', 'buçuk', 'yüzde', 'binde', 'eksi',
  'artı', 'çarpı', 'bölü', 'ila', 'lira', 'kuruş', 'dolar', 'avro', 'sent', 'sterlin', 'derece'])].sort();

// A word of [LET] letters that is_all_caps: two or more, all capitals. A sentence needs three for a headline.
const _CAPS_WORD_RE = new RegExp(`(?<![${LET}])[${UP}]{2,}(?![${LET}])`, 'g');
const _three_caps_words = (sent) => {
  _CAPS_WORD_RE.lastIndex = 0;
  return _CAPS_WORD_RE.test(sent) && _CAPS_WORD_RE.test(sent) && _CAPS_WORD_RE.test(sent);
};

// Lowercase ALL-CAPS sentences, keeping known acronyms for the acronym pass.
function _lower_caps_headlines(text) {
  const out = [];
  for (let sent of _SENTENCE_SPLIT_RE.split(text)) {
    // caps_words below is the [LET] words that are two or more capitals: fewer than three, and it is no headline
    if (!_three_caps_words(sent)) {
      out.push(sent);
      continue;
    }
    const words = _LET_WORD_RE.findall(sent)
      .filter((w) => !(islower(w) && _EXPANSION_STEMS.some((p) => w.startsWith(p))));
    const letters = words.flatMap((w) => [...w]);
    const caps_words = words.filter((w) => is_all_caps(w));
    const upper = letters.filter((c) => isupper(c)).length;
    if (letters.length >= 12 && upper / letters.length > 0.8 && caps_words.length >= 3) {
      sent = _LET_WORD_RE.sub(
        (m) => (L.ACRONYMS.has(m[0]) || !_pronounceable_or_word(m[0]) ? m[0] : tr_lower(m[0])), sent,
      );
    }
    out.push(sent);
  }
  return out.join(' ');
}

// consonant clusters a Turkish word (or a loanword) can start or end with: "KRİZ", "SPOR", "TREN", "TÜRK",
// "DÖRT", "GENÇ", "HALK", "FİLM", "ÜST"
const _ONSETS = new Set(['bl', 'br', 'dr', 'fl', 'fr', 'gl', 'gr', 'kl', 'kr', 'pl', 'pr', 'sf', 'sk', 'sl', 'sm', 'sn',
  'sp', 'st', 'şp', 'şt', 'tr', 'skr', 'spr', 'str']);
const _CODAS = new Set(['rk', 'rt', 'rç', 'rs', 'rş', 'rz', 'rp', 'rf', 'rm', 'rn', 'rd', 'rb', 'rg', 'nk', 'nç', 'nt',
  'ns', 'nd', 'ng', 'nz', 'lk', 'lt', 'lç', 'lp', 'lm', 'lf', 'ls', 'şk', 'şt', 'st', 'sk', 'kt', 'ks', 'ft', 'pt',
  'ps', 'mp']);
const _CONSONANT_RUN_RE = re.compile('[^aeıioöuüâîû]+');

// Spelled like a Turkish word or loanword: a vowel, a known cluster at each end, at most four consonants in a
// row inside. "KONTROL", "TÜRK", "KRİZ" are; "AVM" (vm), "İETT" (tt), "DHA" (dh), "AİHM" (hm), "TBMM" are not.
function _word_shaped(w) {
  const low = tr_lower(w);
  if (![...low].some((c) => VOWELS.has(c))) return false;
  const lead = re.match('[^aeıioöuüâîû]*', low)[0];
  const trail = re.search(`[^aeıioöuüâîû]*${END}`, low)[0];
  if ((plen(lead) > 1 && !_ONSETS.has(lead)) || (plen(trail) > 1 && !_CODAS.has(trail))) return false;
  return _CONSONANT_RUN_RE.findall(strip(low, 'bcçdfgğhjklmnprsştvyzqwx')).every((r) => plen(r) <= 4);
}

// In an ALL-CAPS headline: read as a word ("DÖRT KİŞİ", "KRİZ", "KONTROL") unless it is an acronym.
function _pronounceable_or_word(w) {
  if (L.CAPS_WORDS.has(w) || _pronounceable(w) || plen(w) === 1) return true;
  return !L.EVAL_ACRONYMS.has(w) && _word_shaped(w);
}

function _pass_acronyms(text) {
  const repl = (m) => {
    const [, tok, suf] = m;
    let spoken;
    if (L.ACRONYMS.has(tok)) {
      spoken = L.ACRONYMS.get(tok);
    } else if (L.CAPS_WORDS.has(tok)) {
      return tr_lower(tok) + (suf ? `'${tr_lower(suf)}` : '');
    } else if (_pronounceable(tok)) {
      return tr_lower(tok) + (suf ? `'${tr_lower(suf)}` : '');
    } else {
      spoken = _letters(tok);
    }
    if (!suf) return spoken;
    if (!spoken.includes(' ')) return `${spoken}'${tr_lower(suf)}`; // read as a word: keep the apostrophe
    return _suffix(spoken, suf);
  };

  text = _sub(_ACRONYM_RE, repl, text);
  // lone capital letters used as labels: "C vitamini", "B planı", "A blok" (not the pronoun "O")
  return _sub(_SINGLE_LETTER_RE, (m) => L.LETTER_NAMES.get(m[1]) ?? m[1], text);
}

// A capital letter with a suffix or as an initial is read by its name: "X'te" -> ikste, "V. Öztürk" -> ve. Öztürk.
function _pass_letter_names(text) {
  text = _sub(_SINGLE_LETTER_SUFFIX_RE, (m) => _suffix(L.LETTER_NAMES.get(m[1]), m[2]), text);
  return _INITIAL_RE.sub((m) => `${L.LETTER_NAMES.get(m[1]) ?? tr_lower(m[1])}.`, text);
}

// _pass_symbols' patterns, each with the one character it cannot match without
const _SYMBOLS = [
  [re.compile(`(?=[${SC}&])(?:${S}&${S}|${lb(WC)}&(?=${W}))`, has('&')), ' ve '],
  [re.compile(`(?=[${SC}x×])${lb(D)}${S}?[x×]${S}?(?=${D})`, has('x×')), ' çarpı '],
  [re.compile(`${S}?=${S}?`, has('=')), ' eşittir '],
  [re.compile(`(?=[${SC}+])(?:${lb(WC)}${S}\\+${S}(?=${W})|${lb(D)}\\+(?=${D}))`, has('+')), ' artı '],
  [re.compile('±', has('±')), ' artı eksi '],
  [re.compile(`[~≈]${S}?(?=${W})`, has('~≈')), 'yaklaşık '],
  [re.compile(`(?=#)${nlb(WC)}#(?=${W})`, has('#')), ''],
  [re.compile(`(?=@)${nlb(WC)}@(?=${W})`, has('@')), ''],
  [re.compile(`§${S}?`, has('§')), 'paragraf '],
  [re.compile(`№${S}?`, has('№')), 'numara '],
  [re.compile('[*_|•→←▪►✓✔]', has('*_|•→←▪►✓✔')), ' '],
];
// Kadıköy/İstanbul -> Kadıköy, İstanbul
const _CITY_SLASH_RE = re.compile(`(?=[${UP}])${B}([${UP}][${LOW}]+)/(?=[${UP}][${LOW}])`, _has_slash);
const _WORD_SLASH_RE = re.compile(`(?=/)${lb(WC)}/(?=${W})`, _has_slash);
const _REPEATED_MARK_RE = re.compile(String.raw`([!?])\1+`, has('!?'));

function _pass_symbols(text) {
  for (const [rx, word] of _SYMBOLS) text = rx.sub(word, text);
  if (U_PAIR_BUG && U_PAIR_RE.test(text)) throw _engine_error(text); // a "#" or "@" between them is gone
  text = _CITY_SLASH_RE.sub((m) => `${m[1]}, `, text);
  text = _WORD_SLASH_RE.sub(' ', text);
  text = _REPEATED_MARK_RE.sub((m) => m[1], text);
  return text;
}

// (?<!\s): the match takes a whole run of spaces; Python retries a run that no punctuation ends from each of its
// characters (O(n²) on a long run of newlines), and all of them fail. All BMP: no u flag (Safari 16.4 is O(n²) with it)
const _SPACE_PUNCT_RE = re.compile(`(?<![${SC}])${S}+([,.;:!?)])`, null, false);
const _PAREN_SPACE_RE = re.compile(`([(])${S}+`, has('('), false);
const _COMMAS_RE = re.compile(`,${S}*,`, has(','), false);

function _cleanup(text) {
  text = _SPACE_PUNCT_RE.sub((m) => m[1], text);
  text = _PAREN_SPACE_RE.sub((m) => m[1], text);
  text = _COMMAS_RE.sub(',', text);
  text = _BLANKS_RE.sub(' ', text);
  return strip(text);
}

const _PASSES = [
  _pass_names,
  _pass_web,
  _pass_abbreviations,
  _pass_products,
  _pass_iban,
  _pass_phones,
  _pass_plates,
  _pass_dates,
  _pass_times,
  _pass_money,
  _pass_percent,
  _pass_degrees,
  _pass_units,
  _pass_powers,
  _pass_alnum,
  _pass_ranges,
  _pass_fractions,
  _pass_roman,
  _pass_ordinals,
  _pass_numbers,
  _pass_leftover_digits,
  _lower_caps_headlines,
  _pass_acronyms,
  _pass_letter_names,
  _pass_symbols,
  _pass_clitics,
  _cleanup,
];

// Python (CPython 3.12 in Pyodide) knows Unicode 15.0: to it a later character is unassigned, no letter, digit or
// space, without case or composition. A newer JS engine knows such characters (Unicode 16 added letters, digits and
// marks), so they are swapped for a noncharacter, which every engine treats as Python treats them, and swapped back
// at the end. No pass drops, adds or reorders one (only a URL path takes one in, and keeps it), so the n-th
// placeholder out is the n-th character in; the placeholder itself is in the class, so one in the input round-trips.
const _NONCHAR = '\u{fdd0}';
// A lone surrogate is a code point of its own to Python, and stays one when a pass deletes what stood between it and
// another ("\ud83d\u200d\ude00" loses its ZWJ, "5\ud800\ue000\udc00" its mark); a JS string would then hold one astral
// character there, a letter to \w if it is one. So lone surrogates get a noncharacter of their own for the passes
// (no letter, digit, space or cased character, as a surrogate is to Python) and come back at the end. U+FDD1 is
// unassigned, so one in the input is swapped for U+FDD0 with the rest: every U+FDD1 in the output is a placeholder.
const _SURROGATE = '\u{fdd1}';
const _PLACEHOLDER_RE = /[\ufdd0\ufdd1]/g;
// Both swaps are one pass of a pattern without the u flag, before any u-flag pattern sees the text: Safari 26
// (WebKit 2215 to 2311) reads a lone high surrogate and a following U+FC00..U+FFFF as one code point under the u flag
// (py.js U_PAIR_BUG; a U+FEFF after one was not removed). The pass takes every surrogate pair (swapped when it is
// unassigned), every other surrogate (a lone one) and every unassigned BMP character: UNASSIGNED split into a BMP
// class and astral ranges.
const [_BMP_UNASSIGNED, _ASTRAL_UNASSIGNED] = (() => {
  const hex = (n) => `\\u${n.toString(16).padStart(4, '0')}`;
  let bmp = '';
  const astral = [];
  for (const [, a, b = a] of UNASSIGNED.matchAll(/\\u\{([0-9a-f]+)\}(?:-\\u\{([0-9a-f]+)\})?/g)) {
    const lo = parseInt(a, 16);
    const hi = parseInt(b, 16);
    if (lo <= 0xffff) bmp += `${hex(lo)}-${hex(Math.min(hi, 0xffff))}`;
    if (hi > 0xffff) astral.push(Math.max(lo, 0x10000), hi);
  }
  return [bmp, Uint32Array.from(astral)];
})();
const _PIN_RE = new RegExp(`[\\ud800-\\udbff][\\udc00-\\udfff]|[\\ud800-\\udfff]|[${_BMP_UNASSIGNED}]`, 'g');

// [text with the placeholders, the unassigned characters they replace, the lone surrogates]
function _pin(text) {
  const unassigned = [];
  const lone = [];
  const pinned = text.replace(_PIN_RE, (c) => {
    if (c.length === 2) {
      if (!in_table(_ASTRAL_UNASSIGNED, c.codePointAt(0))) return c;
      unassigned.push(c);
      return _NONCHAR;
    }
    if (c >= '\ud800' && c <= '\udfff') {
      lone.push(c);
      return _SURROGATE;
    }
    unassigned.push(c);
    return _NONCHAR;
  });
  return [pinned, unassigned, lone];
}

// The placeholders of out back to the characters they stand for, in order; drop: leave out the surrogates. null when
// a count changed (no pass is known to do that), as the placeholders would land in the wrong places.
function _unpin(out, unassigned, lone, drop) {
  let i = 0;
  let j = 0;
  const back = out.replace(_PLACEHOLDER_RE, (c) => {
    if (c === _NONCHAR) return unassigned[i++] ?? c;
    j++;
    return drop ? '' : (lone[j - 1] ?? c);
  });
  return i === unassigned.length && j === lone.length ? back : null;
}

// [spoken, the spoken form without Python's surrogates]: normalize(text), and what ahenk_front's frontend() reads of
// it (it starts with encode("utf-8", "ignore"), which drops every surrogate code point; two that a pass brought
// together are one astral character in the JS string, so frontend() cannot tell them apart any more).
function _normalize(text) {
  if (typeof text !== 'string') throw type_error(`normalize() argument must be str, not ${typeof text}`);
  const [pinned, unassigned, lone] = _pin(text);
  if (unassigned.length || lone.length) {
    const out = _run(pinned, _PASSES);
    const spoken = _unpin(out, unassigned, lone, false);
    if (spoken !== null) return [spoken, lone.length ? _unpin(out, unassigned, lone, true) : spoken];
  }
  const out = _run(text, _PASSES);
  return [out, out];
}

// for itn.js, which has the same need
export { _pin as pin_newer, _unpin as unpin_newer };

// For tools/parity/matchers_check.mjs: each linear matcher with the pattern it stands for, as Python writes it
export const _testing = {
  url: [_URL_RE, _URL_SRC],
  email: [_EMAIL_RE, _EMAIL_SRC],
  alnum: [_ALNUM_RE, _ALNUM_SRC],
  leftover: [re.compile(`(?<![${LET}])([${LET}]+)\\.(?=${D})`), `([${LET}]+)\\.(?=${D})`],
  cleanup: [_SPACE_PUNCT_RE, `${S}+([,.;:!?)])`],
  letters_at_end: [_letters_at_end, `([${LET}]+)\\.?${END}`],
  preclean: _preclean,
  passes: _PASSES,
};

// Convert written Turkish into its spoken form (for TTS input).
//
// Case, punctuation and apostrophes on proper nouns are kept, because TTS models use them for prosody.
// Throws where Python raises (a ValueError for a number of more than 4300 digits).
export function normalize(text) {
  return _normalize(text)[0];
}

// For ahenk_front.prepare(): [normalize(text), the text its frontend() reads], the second without the surrogate code
// points the Python string holds (frontend() drops them first thing).
export function normalize_for_frontend(text) {
  return _normalize(text);
}

function _run(text, passes) {
  text = _preclean(text);
  for (const p of passes) text = p(text);
  return text;
}

// ---- normalize_for_eval ------------------------------------------------------------------------------------------

// The eval form keeps foreign names and single letters as written: ASR output and references spell "iPhone", "John
// F. Kennedy", "X'te", and lowercase ASR output ("x'te", "f") must still match them.
const _EVAL_PASSES = _PASSES.filter((p) => p !== _pass_names && p !== _pass_letter_names);
const _CIRCUMFLEX = { â: 'a', î: 'i', û: 'u', Â: 'a', Î: 'i', Û: 'u' };
const _CIRCUMFLEX_RE = /[âîûÂÎÛ]/g;

const _LOWER_ACRONYM_RE = re.compile(`${nlb(`${WC}'`)}([${LOW}]{2,6})(?:'([${LOW}]+))?(?!${W})`);

// Lowercase ASR output and references write "abd", "tbmm"; treat them like "ABD", "TBMM".
function _uppercase_known_acronyms(text) {
  return _LOWER_ACRONYM_RE.sub((m) => {
    const up = tr_upper(m[1]);
    if (!L.EVAL_ACRONYMS.has(up)) return m[0];
    return up + (m[2] ? `'${m[2]}` : '');
  }, text);
}

const _BUCUK_RE = re.compile(`${B}buçuk${B}`, has('ç'));
const _INNER_APOSTROPHE_RE = re.compile(`${lb(WC)}'(?=${W})`, has("'"));
const _NOT_WORD_RE = re.compile(`[^${WC}${SC}]|_`);

// Canonical form for WER/CER: spoken form, Turkish lowercase, no punctuation. Apply the same function to references
// and hypotheses.
//
// The passes run on the text with its unassigned characters and lone surrogates swapped for placeholders, as in
// normalize(); none of them survives the last steps (to Python they are neither \w nor \s, so each becomes a space,
// as the placeholder does), so nothing is swapped back.
export function normalize_for_eval(text) {
  if (typeof text !== 'string') throw type_error(`expected string or bytes-like object, got '${typeof text}'`);
  const [pinned, unassigned, lone] = _pin(text);
  let out = _run(_uppercase_known_acronyms(pinned), _EVAL_PASSES);
  if ((unassigned.length || lone.length) && _unpin(out, unassigned, lone, false) === null) {
    out = _run(_uppercase_known_acronyms(text), _EVAL_PASSES); // a count changed (no pass is known to do that)
  }
  out = tr_lower(out).replace(_CIRCUMFLEX_RE, (c) => _CIRCUMFLEX[c]);
  // "bir buçuk" and "bir virgül beş" are the same quantity
  out = _BUCUK_RE.sub('virgül beş', out);
  out = _INNER_APOSTROPHE_RE.sub('', out);
  // the apostrophe gone, two characters meet that a buggy engine reads as one (py.js U_PAIR_BUG)
  if (U_PAIR_BUG && U_PAIR_RE.test(out)) throw _engine_error(out);
  out = _NOT_WORD_RE.sub(' ', out);
  return split(out).join(' ');
}
