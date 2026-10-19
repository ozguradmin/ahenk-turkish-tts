// Types for trnorm (JavaScript). Every function takes and returns what the Python trnorm does.

/** Written Turkish -> spoken form, for TTS input. Case, punctuation and apostrophes on proper nouns are kept. */
export function normalize(text: string): string;

/** Canonical form for WER/CER scoring: spoken form, Turkish lowercase, no punctuation. Apply it to both sides. */
export function normalize_for_eval(text: string): string;

/** Spoken Turkish -> written form with digits (for ASR output): "saat on dört otuzda" -> "saat 14.30'da". */
export function inverse_normalize(text: string): string;

/** Ahenk Cep's front-end: [normalize(text), the same without lone surrogates]. */
export function normalize_for_frontend(text: string): [string, string];

/** Turkish-aware lowercase ("I" -> "ı", "İ" -> "i"). */
export function tr_lower(text: string): string;

/** Turkish-aware uppercase ("i" -> "İ", "ı" -> "I"). */
export function tr_upper(text: string): string;

/**
 * Cardinal reading: 1923 -> "bin dokuz yüz yirmi üç". A bigint or a safe integer is an int; any other number is read
 * as the Python float it is; true / false are 1 / 0.
 */
export function number_to_words(n: bigint | number | boolean): string;

/** Ordinal reading: 4 -> "dördüncü". The same arguments as number_to_words. */
export function ordinal_to_words(n: bigint | number | boolean): string;

export const __version__: string;
