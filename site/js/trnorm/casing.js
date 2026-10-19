// trnorm/casing.py in JavaScript: Turkish-aware casing helpers.
//
// Python's str.lower()/upper() are locale-agnostic: "I".lower() == "i" and "İ".lower() == "i̇" (i + combining dot).
// Turkish needs I<->ı and İ<->i. lower() and upper() are py.js's: JS's toLowerCase()/toUpperCase() (never the
// toLocale* forms) with Python's Unicode 15.0 where the engine's data is newer, the final sigma rule included
// (a URL path "ornek.com/ΟΔΟ\u{1171E}Σ" is read with ς, as in Python).

import { first_cp, isalpha, isupper, lower, upper } from './py.js';

const LOWER_MAP = { I: 'ı', İ: 'i' };
const UPPER_MAP = { i: 'İ', ı: 'I' };

// Lowercase with Turkish dotted/dotless i rules.
export function tr_lower(text) {
  return lower(text.replace(/[Iİ]/g, (c) => LOWER_MAP[c])).replaceAll('i\u{307}', 'i');
}

// Uppercase with Turkish dotted/dotless i rules.
export function tr_upper(text) {
  return upper(text.replace(/[iı]/g, (c) => UPPER_MAP[c]));
}

// Uppercase the first letter only (Turkish rules), keep the rest.
export function tr_capitalize(word) {
  if (!word) return word;
  const first = first_cp(word);
  return tr_upper(first) + word.slice(first.length);
}

// True when the word has at least two letters and every letter is uppercase.
export function is_all_caps(word) {
  const letters = [...word].filter(isalpha);
  return letters.length >= 2 && letters.every(isupper);
}
