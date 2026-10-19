// trnorm/__init__.py in JavaScript: Turkish text normalization for speech, a pure JS port of the Python package
// (same output for the same input; see py.js for how Python's str, re and int semantics are kept).
//
// Public API:
//     normalize(text)          written Turkish -> spoken form (for TTS front-ends)
//     normalize_for_frontend   [spoken form, the text ahenk_front's frontend() reads of it] (for front-native.js)
//     tr_lower / tr_upper      Turkish-aware casing
//     number_to_words(n)       cardinal number reading (n: a BigInt or a safe integer is an int; another number is
//                              read as the Python float it is, true / false as 1 / 0, anything else: TypeError)
//     ordinal_to_words(n)      ordinal number reading (the same arguments)
//     normalize_for_eval(text) canonical form for ASR/TTS scoring (WER/CER)
//     inverse_normalize(text)  spoken form -> digits (for ASR output display)

export { tr_lower, tr_upper } from './casing.js';
export { number_to_words, ordinal_to_words } from './numbers.js';
export { normalize, normalize_for_eval, normalize_for_frontend } from './normalizer.js';
export { inverse_normalize } from './itn.js';

export const __version__ = '0.1.0';
