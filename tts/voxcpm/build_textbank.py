#!/usr/bin/env python3
"""Text bank for our own Turkish TTS corpus (docs/SES_PLANI.md step 3): questions, emotions, interjections, short replies.

The texts in tts/voxcpm/textbank/*.jsonl are original (written for this repo, not taken from a dataset). VoxCPM2 reads
them in one designed voice; the clips fine-tune a LoRA that should read Turkish well, ask questions with Turkish
intonation and express an emotion on request. This script checks the four source files and merges them:

    python tts/voxcpm/build_textbank.py            # -> tts/voxcpm/textbank/textbank.jsonl + counts
    python tts/voxcpm/build_textbank.py --check    # validate only, write nothing

Sources (one JSON object per line; `id` unique over all files):
  questions.jsonl      id, kind, text, pair_id, peak_word
      kind: mi_soru (yes/no with the particle mI), duz_eslesik (its matched declarative: same pair_id), ne_soru
      (wh-word), degil_mi ("..., değil mi?"), yukselen (no mI, no wh-word: final rise), hizmet (polite service).
      peak_word: 0-based index into trnorm.normalize(text).split() of the word that carries the pitch peak: the word
      before mI (mi_soru, hizmet with one mI), "değil" (degil_mi), the wh-word (ne_soru, hizmet without mI), the last
      word (yukselen), the question's peak word (duz_eslesik); null when unsure (mI after a clitic dA, two mI).
      pair_id: required for mi_soru / duz_eslesik (exactly one of each per pair), optional for degil_mi / yukselen
      (the same sentence as the pair's declarative), null for ne_soru / hizmet.
  emotions.jsonl       id, emotion, intensity (mild/strong), text (4-20 words), instruction, instruction_id,
                       instruction_split (train/heldout)
      instruction: one of INSTRUCTIONS[emotion] (English VoxCPM2 control text, no parentheses: the renderer writes
      "(instruction)" + spoken). 20 % of the paraphrases are held out: rows using them have instruction_split
      "heldout" (evaluation of unseen wording), and no train row uses one.
  interjections.jsonl  id, interjection (canonical, lowercase), span (as written in text), text, render_variants,
                       emotion (hint or null)
      render_variants: other spellings of the span to try when RENDERING only (first = span), for interjections the
      model may mispronounce (all VoxCPM voices said "vey kenına" for "Vay canına"); null when none. Training text
      stays `text`.
  short.jsonl          id, kind, text: short replies and confirmations.

Per row: trnorm.normalize(text) -> `spoken`. A row is dropped (and reported) when normalization fails or leaves a
digit or one of % ₺ $ € £ # @; when it repeats an earlier row (key: trnorm.normalize_for_eval(text) plus the class of
the final mark: "?", "..." or statement ". !"; so "Yarın geliyorsun." and "Yarın geliyorsun?" are two rows with two
intonations, "Olur." and "Olur!" are one); or when the leakage guard hits. Leakage guard: prepare_alania's contamination guard, imported (build_guard / Guard),
on every sentence of benchmarks/tts/sets/*.jsonl (TR-Zor, TR-Soru, contest, ...: text, spoken, alts, refs) and of
Antalia's held-out rows (voxcpm-eval/antalia-gt/rows.jsonl when present, plus --exclude files): a row may not equal a
protected sentence nor contain one of >= 4 words (normalize_for_eval form of text and of spoken). Shared 4-word runs
that are not whole protected sentences are only reported here; tests/test_textbank.py also keeps the bank free of
those with a word other than a number reading ("bin beş yüz lira" may be shared, "ne kadar zor olduğunu" may not).

Output textbank.jsonl, one row per kept source row, in file order, with the same fields for every category:
  id, category, kind, emotion, intensity, instruction, instruction_id, instruction_split, pair_id, peak_word, pattern,
  interjection, span, render_variants, render_texts, text, spoken, features
  pattern (questions): peak_fall / wh_peak / final_rise / fall, as score_intonation.py names them; render_texts
  (interjections with variants): the spoken form of text with each variant in place of the span.
Schema errors stop the build (exit 2, nothing written); --strict also exits 1 when any row was dropped.

laughter.jsonl (id l_NNN, emotion cheerful, intensity, text, laugh: true, laugh_span: the laughter as written, e.g.
"Hahaha"): the lines of render_corpus.py's laughter trial (--laughter-usd; candidates for human review, never in the
training manifest). Checked here (laughter_errors: schema, a laugh_span LAUGH_RE reads as laughter, 2-14 words, the
spoken form, de-duplication against the bank, the leakage guard), never merged into textbank.jsonl.
Importing this module needs only the standard library, trnorm and prepare_alania.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

from prepare_alania import Guard, build_guard, eval_key, protected_texts  # noqa: E402
from trnorm import normalize, tr_lower, tr_upper  # noqa: E402

BANK = HERE / "textbank"
SOURCES = {"questions": "questions.jsonl", "emotions": "emotions.jsonl", "interjections": "interjections.jsonl",
           "short": "short.jsonl"}
OUT_NAME = "textbank.jsonl"
LAUGHTER = "laughter.jsonl"  # the laughter trial's lines: checked, not merged
# laughter as written: two or more h-syllables ("Hahaha", "Ha ha ha", "Hehe", "Ahahaha", "Hihihi"); render_corpus
# strips it before comparing with Whisper (which drops or respells it)
LAUGH_RE = re.compile(r"\b[aA]?[hH][aeıiAEIİ](?:\s?[hH][aeıiAEIİ])+[hH]?\b")
LAUGH_MIN_WORDS, LAUGH_MAX_WORDS = 2, 14
BENCH_DIR = REPO / "benchmarks" / "tts" / "sets"
ANTALIA_HELDOUT = REPO / "voxcpm-eval" / "antalia-gt" / "rows.jsonl"

# targets (rows); tests require each count within +-15 %
TARGETS = {"questions": 700, "emotions": 1020, "interjections": 250, "short": 250}
KIND_TARGETS = {"mi_soru": 250, "duz_eslesik": 250, "ne_soru": 120, "degil_mi": 40, "yukselen": 30, "hizmet": 60}
EMOTION_TARGET = 170

QUESTION_KINDS = tuple(KIND_TARGETS)
PATTERNS = {"mi_soru": "peak_fall", "degil_mi": "peak_fall", "ne_soru": "wh_peak", "yukselen": "final_rise",
            "duz_eslesik": "fall"}
SHORT_KINDS = ("onay", "ret", "tesekkur", "bekleme", "belirsiz", "selam", "ozur", "tepki", "soru", "dilek")
EMOTIONS = ("angry", "cheerful", "sad", "surprised", "calm", "empathetic")
INTENSITIES = ("mild", "strong")
EMO_MIN_WORDS, EMO_MAX_WORDS = 4, 20
SHORT_MAX_WORDS = 7


def _para(emotion: str, rows: list[tuple[str, str, str]]) -> list[dict]:
    return [{"id": f"{emotion}_{i:02d}", "text": t, "level": lv, "split": sp} for i, (t, lv, sp) in enumerate(rows, 1)]


# English control instructions in VoxCPM2's demo style, 10 per emotion; 2 (20 %) held out. level: which intensity of
# sentence a paraphrase fits (mid fits both).
INSTRUCTIONS: dict[str, list[dict]] = {
    "angry": _para("angry", [
        ("Angry tone, voice getting louder.", "mid", "train"),
        ("Irritated and stern, with clipped words.", "mid", "train"),
        ("Angry and indignant, voice rising with each word.", "mid", "train"),
        ("Very angry and frustrated, speaking forcefully.", "strong", "train"),
        ("Furious, sharp and loud, almost shouting.", "strong", "train"),
        ("Very upset, speaking fast with a hard edge in the voice.", "strong", "train"),
        ("Slightly annoyed, firm and impatient.", "mild", "train"),
        ("A little angry, cold and tense.", "mild", "train"),
        ("Mildly irritated, speaking curtly.", "mild", "heldout"),
        ("Exasperated and very angry, raising her voice.", "strong", "heldout"),
    ]),
    "cheerful": _para("cheerful", [
        ("Cheerful and warm, smiling while speaking.", "mid", "train"),
        ("Happy and upbeat, bright and lively.", "mid", "train"),
        ("Bubbly and playful, with a lively rhythm.", "mid", "train"),
        ("Excited and joyful, speaking fast with a bright voice.", "strong", "train"),
        ("Very happy and energetic, full of enthusiasm.", "strong", "train"),
        ("Very excited, almost laughing with joy.", "strong", "train"),
        ("Slightly cheerful, light and friendly.", "mild", "train"),
        ("Pleasantly happy, smiling softly while speaking.", "mild", "train"),
        ("Gently happy, with a cheerful lilt.", "mild", "heldout"),
        ("Overjoyed and thrilled, voice bursting with energy.", "strong", "heldout"),
    ]),
    "sad": _para("sad", [
        ("Sad and quiet, with a heavy heart.", "mid", "train"),
        ("Melancholic and quiet, speaking slowly.", "mid", "train"),
        ("Sorrowful and subdued, with long pauses.", "mid", "train"),
        ("Very sad, speaking slowly and softly, almost whispering.", "strong", "train"),
        ("Deeply sad, voice low and trembling.", "strong", "train"),
        ("Very sad, on the verge of tears.", "strong", "train"),
        ("Slightly sad and tired, low energy.", "mild", "train"),
        ("Disappointed and downcast, soft and slow.", "mild", "train"),
        ("A little sad and weary, with a sigh in the voice.", "mild", "heldout"),
        ("Heartbroken, speaking very quietly and slowly.", "strong", "heldout"),
    ]),
    "surprised": _para("surprised", [
        ("Astonished, with rising intonation.", "mid", "train"),
        ("Surprised, voice going up in disbelief.", "mid", "train"),
        ("Amazed, as if hearing unexpected news.", "mid", "train"),
        ("Very surprised, with a sharp rise in pitch.", "strong", "train"),
        ("Shocked, speaking quickly in wide-eyed disbelief.", "strong", "train"),
        ("Utterly astonished, voice jumping higher.", "strong", "train"),
        ("Slightly surprised, curious and lively.", "mild", "train"),
        ("Pleasantly surprised, bright and rising.", "mild", "train"),
        ("A little taken aback, with a questioning lift.", "mild", "heldout"),
        ("Very startled and amazed, with an exclamation in the voice.", "strong", "heldout"),
    ]),
    "calm": _para("calm", [
        ("Calm, slow and soothing.", "mid", "train"),
        ("Soft and serene, with gentle pauses.", "mid", "train"),
        ("Calm and reassuring, low and even.", "mid", "train"),
        ("Very calm and relaxed, speaking gently and slowly.", "strong", "train"),
        ("Tranquil and unhurried, like a guided relaxation.", "strong", "train"),
        ("Very peaceful, quiet and measured.", "strong", "train"),
        ("Calm and steady, slightly slower than usual.", "mild", "train"),
        ("Composed and gentle, with an even pace.", "mild", "train"),
        ("Quietly composed, smooth and unhurried.", "mild", "heldout"),
        ("Deeply relaxed and mellow, speaking very softly.", "strong", "heldout"),
    ]),
    "empathetic": _para("empathetic", [
        ("Warm and compassionate, as if comforting someone.", "mid", "train"),
        ("Empathetic and sincere, speaking softly.", "mid", "train"),
        ("Warm, patient and reassuring, full of understanding.", "mid", "train"),
        ("Very gentle and caring, soft and patient.", "strong", "train"),
        ("Deeply sympathetic, tender and slow.", "strong", "train"),
        ("Very compassionate, like consoling a close friend.", "strong", "train"),
        ("Slightly concerned, kind and understanding.", "mild", "train"),
        ("Kind and apologetic, soft and sincere.", "mild", "train"),
        ("Caring and attentive, with quiet warmth.", "mild", "heldout"),
        ("Deeply heartfelt and tender, soothing a grieving person.", "strong", "heldout"),
    ]),
}
INSTRUCTION_BY_ID = {p["id"]: {**p, "emotion": e} for e, ps in INSTRUCTIONS.items() for p in ps}
HELDOUT_SHARE = 0.2

# rendering-only spellings of interjections the model may mispronounce (first = canonical); others: none
RENDER_VARIANTS = {
    "vay": ["vay", "vaay", "vây", "vaaay"],
    "vay canına": ["vay canına", "vaay canına", "vay caanına", "vây cânına"],
    "ay": ["ay", "aay", "aaay"],
    "aa": ["aa", "aaa", "aaaa"],
    "of": ["of", "off", "ooof"],
    "offf": ["offf", "ooff", "ooof"],
    "öf": ["öf", "öff", "ööf"],
    "oh": ["oh", "ooh"],
    "oh be": ["oh be", "ooh be", "oh bee"],
    "ah": ["ah", "aah"],
    "eh": ["eh", "ehh", "eeh"],
    "ha": ["ha", "haa"],
    "hmm": ["hmm", "hımm", "hmmm", "hıım"],
    "tüh": ["tüh", "tüüh"],
    "yuh": ["yuh", "yuuh"],
    "maşallah": ["maşallah", "maaşallah"],
    "inşallah": ["inşallah", "inşaallah"],
}
INTERJECTIONS = ("vay", "vay canına", "ay", "of", "öf", "oh", "oh be", "aa", "eyvah", "tüh", "yaşasın", "hadi ya",
                 "hey", "haydi", "hmm", "ha", "aman", "eh", "yuh", "maşallah", "inşallah", "offf", "ah")

FIELDS_OUT = ("id", "category", "kind", "emotion", "intensity", "instruction", "instruction_id", "instruction_split",
              "pair_id", "peak_word", "pattern", "interjection", "span", "render_variants", "render_texts", "text",
              "spoken", "features")
REQUIRED = {
    "questions": ("id", "kind", "text", "pair_id", "peak_word"),
    "emotions": ("id", "emotion", "intensity", "text", "instruction", "instruction_id", "instruction_split"),
    "interjections": ("id", "interjection", "span", "text", "render_variants", "emotion"),
    "short": ("id", "kind", "text"),
}

# --------------------------------------------------------------------------- words of the spoken form
PUNCT = "\"'’“”«».,;:!?…()[]-–—"
_VOWELS = "aeıioöuüâîû"
_MI_RE = re.compile(r"^m[ıiuü](?:y[ıiuü]m|s[ıiuü]n|y[ıiuü]z|s[ıiuü]n[ıiuü]z|d[ıiuü]r|y(?:d[ıiuü](?:m|n|k|n[ıiuü]z|lar|ler)?"
                    r"|m[ıiuü]ş|s[ae]))?$")
WH_WORDS = frozenset("""ne neden niye niçin nasıl nasıldı nasılsın nasılsınız nerede nereye nereden neresi neresinde nereli
    kim kime kimi kimin kimden kimle kiminle kimler kimlere kimlerle kaç kaçta kaçı kaçıncı kaçar kaçtan hangi hangisi
    hangisini hangisine hangisinde neyi neye neyle neler nedir""".split())
NOT_MI = frozenset({"müdür"})  # the noun (manager), not mü + dür
CLITIC_DA = frozenset({"de", "da", "te", "ta"})
# back-vowel words that take front suffixes ("saat mi", "kalp mi", "gol mü"): harmony check exceptions
FRONT_HARMONY = frozenset("""saat dikkat kalp gol rol alkol kontrol normal hal hâl ihtimal sual kabul meşgul usul mahsul hayal
    metal global total ideal petrol protokol futbol harf şefkat itaat hakikat kemal hilal celal bilal""".split())


def core(tok: str) -> str:
    """A word of the spoken form without punctuation, lowercase (apostrophe suffixes kept: "kasım'da")."""
    return tr_lower(tok.strip(PUNCT))


def is_mi(tok: str) -> bool:
    c = core(tok)
    return bool(_MI_RE.match(c)) and c not in NOT_MI


def is_wh(tok: str) -> bool:
    return core(tok) in WH_WORDS


def expected_peak(kind: str, spoken: str) -> int | None:
    """peak_word of a question row by its kind (duz_eslesik: see duz_peak); None = unsure. ValueError when the kind's
    shape is missing (no mI in mi_soru, no wh-word in ne_soru, ...)."""
    w = spoken.split()
    mis = [i for i, t in enumerate(w) if is_mi(t)]
    whs = [i for i, t in enumerate(w) if is_wh(t)]
    if kind == "mi_soru":
        if not mis or mis[0] == 0:
            raise ValueError("mi_soru without a mI particle after a word")
        return None if len(mis) > 1 or core(w[mis[0] - 1]) in CLITIC_DA else mis[0] - 1
    if kind == "degil_mi":
        if len(w) < 3 or core(w[-2]) != "değil" or not is_mi(w[-1]):
            raise ValueError("degil_mi does not end in 'değil mi?'")
        return len(w) - 2
    if kind == "ne_soru":
        if mis:
            raise ValueError("ne_soru with a mI particle")
        if not whs:
            raise ValueError("ne_soru without a wh-word")
        return whs[0]
    if kind == "yukselen":
        if mis or whs:
            raise ValueError("yukselen with a mI particle or a wh-word")
        return len(w) - 1
    if kind == "hizmet":
        if mis:
            if mis[0] == 0:
                raise ValueError("hizmet starts with mI")
            return None if len(mis) > 1 or core(w[mis[0] - 1]) in CLITIC_DA else mis[0] - 1
        return whs[0] if whs else len(w) - 1
    raise ValueError(f"no peak rule for kind {kind!r}")


def question_pattern(kind: str, spoken: str) -> str | None:
    if kind != "hizmet":
        return PATTERNS[kind]
    w = spoken.split()
    mis = [t for t in w if is_mi(t)]
    if len(mis) > 1:
        return None  # alternative question ("... mi yoksa ... mi?"): no single shape
    if mis:
        return "peak_fall"
    return "wh_peak" if any(is_wh(t) for t in w) else "final_rise"


def duz_peak(q_spoken: str, d_spoken: str, q_peak: int | None) -> int | None:
    """The declarative's peak: the same word as its question's (same words before it, a shared stem at it)."""
    if q_peak is None:
        return None
    q, d = q_spoken.split(), d_spoken.split()
    if len(d) <= q_peak or [core(t) for t in d[:q_peak]] != [core(t) for t in q[:q_peak]]:
        raise ValueError("declarative does not start with the question's words before the peak")
    a, b = core(q[q_peak]), core(d[q_peak])
    if not (a == b or (len(a) >= 3 and b.startswith(a[:3]))):
        raise ValueError(f"declarative word {b!r} is not the question's peak word {a!r}")
    return q_peak


def _last_vowel(word: str) -> str | None:
    vs = [c for c in tr_lower(word) if c in _VOWELS]
    return vs[-1] if vs else None


def harmony_issues(spoken: str) -> list[str]:
    """mI particles whose vowel does not follow the word before them ("geliyor mi")."""
    out, w = [], spoken.split()
    want = {"a": "ı", "ı": "ı", "â": "ı", "o": "u", "u": "u", "û": "u", "e": "i", "i": "i", "î": "i", "ö": "ü", "ü": "ü"}
    for i in range(1, len(w)):
        if not is_mi(w[i]):
            continue
        prev = core(w[i - 1]).replace("'", "")  # the suffix decides: "Kasım'da mı", "saatte mi"
        v = _last_vowel(prev)
        if v is None:
            continue
        exp = want[v]
        if prev in FRONT_HARMONY:  # bare word only
            exp = {"ı": "i", "u": "ü"}.get(exp, exp)
        if core(w[i])[1] != exp:
            out.append(f"{w[i - 1]} {w[i]}")
    return out


# --------------------------------------------------------------------------- features of the written text
_MONTHS = "Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık"
FEATURE_RES = {
    "number": re.compile(r"\d"),
    "time": re.compile(r"(?<![\d.,])(?:[01]?\d|2[0-4])[:.][0-5]\d(?!\d|[.,]\d)|\b[Ss]aat \d{1,2}'"),
    "money": re.compile(r"₺|\$|€|£|\bTL\b|\d\s?(?:bin |milyon )?(?:lira|kuruş|dolar|euro)"),
    "phone": re.compile(r"\b0\s?\(?\d{3}\)?\s\d{3}\s\d{2}\s\d{2}\b"),
    "date": re.compile(rf"\b\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}\b|\b\d{{1,2}} (?:{_MONTHS})\b|\b(?:19|20)\d{{2}}\b"),
    "percent": re.compile(r"%"),
    "ordinal": re.compile(r"\d\.(?!\d)"),
    "acronym": re.compile(r"(?<![\w])[A-ZÇĞİÖŞÜ]{2,}(?![a-zçğıöşü])"),
    "address_email": re.compile(r"\b(?:Mah|Sok|Cad)\.|\bNo:|@"),
}
NUMERIC_FEATURES = ("number", "time", "money", "phone", "date", "percent", "acronym", "address_email")


def features(text: str) -> list[str]:
    return [k for k, rx in FEATURE_RES.items() if rx.search(text)]


# --------------------------------------------------------------------------- io + schema
def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}:{n}: {e}") from None
    return rows


def load_sources(bank: Path = BANK) -> dict[str, list[dict]]:
    return {cat: read_jsonl(bank / name) for cat, name in SOURCES.items()}


def final_mark(text: str) -> str:
    t = text.rstrip()
    if t.endswith(("...", "…")):
        return "…"
    return t[-1] if t and t[-1] in ".!?" else ""


def dedup_key(text: str) -> tuple[str, str]:
    m = final_mark(text)
    return eval_key(text), ("." if m in (".", "!") else m)


def spoken_of(text: str) -> tuple[str, str]:
    """(spoken form, reject reason or "")."""
    try:
        s = normalize(text)
    except Exception as e:  # noqa: BLE001
        return "", f"normalize failed: {type(e).__name__}: {e}"
    s = " ".join(s.split())
    if not s:
        return "", "normalize gave an empty text"
    if re.search(r"\d", s):
        return s, "digit left in spoken form"
    if re.search(r"[%₺$€£#@]", s):
        return s, "symbol left in spoken form"
    return s, ""


def _clean(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return "empty text"
    if text != " ".join(text.split()):
        return "extra whitespace"
    if not final_mark(text):
        return "no final punctuation (. ! ? ...)"
    if ("(" in text or ")" in text) and not re.search(r"\(\d{3}\)", text):  # only a phone area code: 0 (232) ...
        return "parenthesis in text (VoxCPM2 reads '(...)' as a control instruction)"
    return ""


def schema_errors(sources: dict[str, list[dict]]) -> list[str]:
    """Shape of every row and of the bank as a whole (pairs, peaks, instructions); spoken forms are computed here."""
    errs, ids = [], collections.Counter()
    for cat, rows in sources.items():
        for r in rows:
            ids[r.get("id")] += 1
            where = f"{cat}/{r.get('id')}"
            miss = [k for k in REQUIRED[cat] if k not in r]
            if miss:
                errs.append(f"{where}: missing {miss}")
                continue
            if (e := _clean(r["text"])):
                errs.append(f"{where}: {e}: {r['text']!r}")
                continue
            errs += [f"{where}: {e}" for e in _row_errors(cat, r)]
    errs += [f"duplicate id {i!r} ({n}x)" for i, n in ids.items() if n > 1]
    errs += _pair_errors(sources["questions"])
    errs += _instruction_errors(sources["emotions"])
    return errs


def _row_errors(cat: str, r: dict) -> list[str]:
    text, errs = r["text"], []
    spoken, _ = spoken_of(text)
    if cat == "questions":
        kind = r["kind"]
        if kind not in QUESTION_KINDS:
            return [f"kind {kind!r} not in {QUESTION_KINDS}"]
        if (kind == "duz_eslesik") == text.endswith("?"):
            errs.append("duz_eslesik must not end in '?', every other kind must")
        if kind in ("mi_soru", "duz_eslesik") and not r["pair_id"]:
            errs.append(f"{kind} needs a pair_id")
        if kind in ("ne_soru", "hizmet") and r["pair_id"] is not None:
            errs.append(f"{kind} has a pair_id")
        pk = r["peak_word"]
        if pk is not None and (not isinstance(pk, int) or isinstance(pk, bool) or not 0 <= pk < len(spoken.split())):
            errs.append(f"peak_word {pk!r} outside the {len(spoken.split())} spoken words")
        if kind != "duz_eslesik":
            try:
                exp = expected_peak(kind, spoken)
                if pk is not None and pk != exp:
                    errs.append(f"peak_word {pk} but the {kind} rule gives {exp} in {spoken!r}")
            except ValueError as e:
                errs.append(f"{e}: {spoken!r}")
        elif any(is_mi(t) for t in spoken.split()):
            errs.append("duz_eslesik with a mI particle")
        errs += [f"vowel harmony: {h!r}" for h in harmony_issues(spoken)]
    elif cat == "emotions":
        if r["emotion"] not in EMOTIONS:
            errs.append(f"emotion {r['emotion']!r} not in {EMOTIONS}")
        if r["intensity"] not in INTENSITIES:
            errs.append(f"intensity {r['intensity']!r} not in {INTENSITIES}")
        n = len(text.split())
        if not EMO_MIN_WORDS <= n <= EMO_MAX_WORDS:
            errs.append(f"{n} words, not {EMO_MIN_WORDS}-{EMO_MAX_WORDS}")
    elif cat == "interjections":
        if r["interjection"] not in INTERJECTIONS:
            errs.append(f"interjection {r['interjection']!r} not in INTERJECTIONS")
        span = r["span"]
        if not isinstance(span, str) or span not in text or tr_lower(span) != r["interjection"]:
            errs.append(f"span {span!r} is not the interjection {r['interjection']!r} as written in the text")
        rv = r["render_variants"]
        if rv is not None:
            if not (isinstance(rv, list) and len(rv) >= 2 and all(isinstance(v, str) and v for v in rv)):
                errs.append("render_variants must be null or a list of >= 2 spellings")
            elif rv[0] != span or len(set(rv)) != len(rv):
                errs.append(f"render_variants {rv!r}: first must be the span {span!r}, no repeats")
        if r["emotion"] is not None and r["emotion"] not in EMOTIONS:
            errs.append(f"emotion hint {r['emotion']!r} not in {EMOTIONS}")
    elif cat == "short":
        if r["kind"] not in SHORT_KINDS:
            errs.append(f"kind {r['kind']!r} not in {SHORT_KINDS}")
        if len(text.split()) > SHORT_MAX_WORDS:
            errs.append(f"more than {SHORT_MAX_WORDS} words")
    return errs


def _pair_errors(rows: list[dict]) -> list[str]:
    errs, by = [], collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("pair_id") is not None:
            by[r["pair_id"]][r.get("kind")].append(r)
    for pid, kinds in by.items():
        if len(kinds.get("mi_soru", [])) != 1 or len(kinds.get("duz_eslesik", [])) != 1:
            errs.append(f"pair {pid}: needs exactly one mi_soru and one duz_eslesik, has "
                        f"{ {k: len(v) for k, v in kinds.items()} }")
            continue
        q, d = kinds["mi_soru"][0], kinds["duz_eslesik"][0]
        qs, ds = spoken_of(q["text"])[0], spoken_of(d["text"])[0]
        try:
            exp = duz_peak(qs, ds, q["peak_word"])
            if d["peak_word"] != exp:
                errs.append(f"pair {pid}: duz_eslesik peak_word {d['peak_word']} but the question gives {exp}")
        except ValueError as e:
            errs.append(f"pair {pid}: {e}: {qs!r} / {ds!r}")
        stem = d["text"][:-1]
        for k, tail in (("degil_mi", ", değil mi?"), ("yukselen", "?")):
            for r in kinds.get(k, []):
                if r["text"] != stem + tail:
                    errs.append(f"pair {pid}: {r['id']} is not the declarative + {tail!r}")
    return errs


def _instruction_errors(rows: list[dict]) -> list[str]:
    errs, used = [], collections.Counter()
    for r in rows:
        p = INSTRUCTION_BY_ID.get(r.get("instruction_id"))
        where = f"emotions/{r.get('id')}"
        if p is None or p["emotion"] != r.get("emotion"):
            errs.append(f"{where}: instruction_id {r.get('instruction_id')!r} is not a {r.get('emotion')} paraphrase")
            continue
        used[p["id"]] += 1
        if r["instruction"] != p["text"]:
            errs.append(f"{where}: instruction text differs from {p['id']}")
        if r["instruction_split"] != p["split"]:
            errs.append(f"{where}: instruction_split {r['instruction_split']!r} but {p['id']} is {p['split']}")
        if p["level"] != "mid" and p["level"] != r.get("intensity"):
            errs.append(f"{where}: a {p['level']} paraphrase on a {r.get('intensity')} sentence")
    for e, ps in INSTRUCTIONS.items():
        for p in ps:
            if "(" in p["text"] or ")" in p["text"]:
                errs.append(f"instruction {p['id']} has a parenthesis")
            if rows and not used[p["id"]]:
                errs.append(f"instruction {p['id']} is never used")
        held = sum(p["split"] == "heldout" for p in ps)
        if held != round(HELDOUT_SHARE * len(ps)):
            errs.append(f"{e}: {held} of {len(ps)} paraphrases held out, want {HELDOUT_SHARE:.0%}")
    return errs


# --------------------------------------------------------------------------- leakage guard
def protected_files(extra: list[Path] | None = None) -> list[Path]:
    files = sorted(BENCH_DIR.glob("*.jsonl"))
    if ANTALIA_HELDOUT.exists():
        files.append(ANTALIA_HELDOUT)
    return files + [Path(p) for p in extra or []]


def protected_sentences(files: list[Path]) -> list[str]:
    """prepare_alania.protected_texts of each file plus the `refs` of benchmark rows (contest.jsonl's spoken forms)."""
    out = []
    for f in files:
        out += protected_texts(f)
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out += [t for t in json.loads(line).get("refs") or [] if isinstance(t, str)]
    return out


def leak_guard(extra: list[Path] | None = None) -> tuple[Guard, list[str]]:
    texts = protected_sentences(protected_files(extra))
    return build_guard(texts), texts


def ngrams(key: str, n: int = 4) -> set[tuple[str, ...]]:
    w = key.split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


# words of number, money, date and time readings: a shared 4-word run made only of these ("bin beş yüz lira",
# "mart iki bin yirmi") is how Turkish reads numbers, not shared wording
NUMERAL_WORDS = frozenset("""sıfır bir iki üç dört beş altı yedi sekiz dokuz on yirmi otuz kırk elli altmış yetmiş seksen
    doksan yüz bin milyon milyar virgül buçuk ila yüzde lira kuruş dolar euro gram kilo kilogram derece numara saat
    ocak şubat mart nisan mayıs haziran temmuz ağustos eylül ekim kasım aralık""".split())


def content_ngram(g: tuple[str, ...]) -> bool:
    return any(w not in NUMERAL_WORDS for w in g)


# --------------------------------------------------------------------------- build
def render_texts(text: str, span: str, variants: list[str] | None) -> list[str] | None:
    if not variants:
        return None
    i = text.index(span)
    return [spoken_of(text[:i] + v + text[i + len(span):])[0] for v in variants]


def merged_row(cat: str, r: dict, spoken: str) -> dict:
    out = dict.fromkeys(FIELDS_OUT)
    out.update(id=r["id"], category=cat, text=r["text"], spoken=spoken, features=features(r["text"]))
    if cat == "questions":
        out.update(kind=r["kind"], pair_id=r["pair_id"], peak_word=r["peak_word"],
                   pattern=question_pattern(r["kind"], spoken))
    elif cat == "emotions":
        out.update({k: r[k] for k in ("emotion", "intensity", "instruction", "instruction_id", "instruction_split")})
    elif cat == "interjections":
        out.update(kind="unlem", emotion=r["emotion"], interjection=r["interjection"], span=r["span"],
                   render_variants=r["render_variants"],
                   render_texts=render_texts(r["text"], r["span"], r["render_variants"]))
    else:
        out.update(kind=r["kind"])
    return out


def build_bank(sources: dict[str, list[dict]], guard: Guard, protected: list[str] | None = None) -> dict:
    """Normalize, dedup and guard the (schema-valid) sources. Returns rows (kept, merged), drops [(id, reason)],
    ngram_overlaps [(id, 4-word runs shared with protected sentences)] and content_overlaps (those runs with a word
    that is not a number reading; report only here, tests/test_textbank.py keeps them at zero)."""
    prot_ngrams = set().union(*(ngrams(eval_key(t)) for t in protected)) if protected else set()
    rows, drops, overlaps, content, seen = [], [], [], [], {}
    for cat, src in sources.items():
        for r in src:
            spoken, why = spoken_of(r["text"])
            key = dedup_key(r["text"])
            if not why and key in seen:
                why = f"duplicate of {seen[key]}"
            if not why and guard.hit(eval_key(r["text"]), eval_key(spoken)):
                why = "leakage: equals or contains a protected benchmark/held-out sentence"
            if why:
                drops.append((r["id"], why))
                continue
            seen[key] = r["id"]
            shared = (ngrams(eval_key(r["text"])) | ngrams(eval_key(spoken))) & prot_ngrams
            if shared:
                overlaps.append((r["id"], sorted(" ".join(g) for g in shared)))
            if any(content_ngram(g) for g in shared):
                content.append((r["id"], sorted(" ".join(g) for g in shared if content_ngram(g))))
            rows.append(merged_row(cat, r, spoken))
    return {"rows": rows, "drops": drops, "ngram_overlaps": overlaps, "content_overlaps": content}


def load_laughter(bank: Path = BANK) -> list[dict]:
    p = bank / LAUGHTER
    return read_jsonl(p) if p.is_file() else []


def strip_laughter(text: str) -> str:
    """text without its laughter (LAUGH_RE) and what is left of the punctuation around it."""
    t = re.sub(rf"[,;:]?\s*(?:{LAUGH_RE.pattern})[,;:]?", " ", text or "")
    t = re.sub(r"\s+([.!?…])", r"\1", " ".join(t.split()))
    return re.sub(r"^[\s,;:.!?…-]+", "", t)


def laughter_errors(rows: list[dict], guard: Guard | None = None, bank_rows: list[dict] | None = None) -> list[str]:
    """Problems of laughter.jsonl (module doc): schema, the laugh span, length, spoken form, ids and texts repeated
    (within it or against the merged bank's rows), the leakage guard."""
    errs, ids, seen = [], collections.Counter(r.get("id") for r in rows), {}
    for r in bank_rows or []:
        seen[dedup_key(r["text"])] = r["id"]
    for r in rows:
        where = f"laughter/{r.get('id')}"
        miss = [k for k in ("id", "emotion", "intensity", "text", "laugh", "laugh_span") if k not in r]
        if miss:
            errs.append(f"{where}: missing {miss}")
            continue
        text, span = r["text"], r["laugh_span"]
        if (e := _clean(text)):
            errs.append(f"{where}: {e}: {text!r}")
            continue
        if not re.fullmatch(r"l_\d{3}", str(r["id"])) or r["laugh"] is not True or r["emotion"] != "cheerful":
            errs.append(f"{where}: want id l_NNN, laugh true, emotion cheerful")
        if r["intensity"] not in INTENSITIES:
            errs.append(f"{where}: intensity {r['intensity']!r} not in {INTENSITIES}")
        if not isinstance(span, str) or span not in text or not LAUGH_RE.fullmatch(span):
            errs.append(f"{where}: laugh_span {span!r} is not laughter (LAUGH_RE) as written in the text")
        if not strip_laughter(text):
            errs.append(f"{where}: nothing but laughter")
        if not LAUGH_MIN_WORDS <= len(text.split()) <= LAUGH_MAX_WORDS:
            errs.append(f"{where}: {len(text.split())} words, not {LAUGH_MIN_WORDS}-{LAUGH_MAX_WORDS}")
        spoken, why = spoken_of(text)
        if why:
            errs.append(f"{where}: {why}")
        key = dedup_key(text)
        if key in seen:
            errs.append(f"{where}: duplicate of {seen[key]}")
        seen.setdefault(key, r["id"])
        if guard is not None and guard.hit(eval_key(text), eval_key(spoken)):
            errs.append(f"{where}: leakage: equals or contains a protected benchmark/held-out sentence")
    return errs + [f"laughter: duplicate id {i!r} ({n}x)" for i, n in ids.items() if n > 1]


def stats(rows: list[dict]) -> dict:
    by_cat = collections.Counter(r["category"] for r in rows)
    out = {"rows": len(rows), "by_category": dict(by_cat), "by_kind": {}, "numeric_share": {}, "features": {}}
    for cat in SOURCES:
        rs = [r for r in rows if r["category"] == cat]
        key = "emotion" if cat == "emotions" else ("interjection" if cat == "interjections" else "kind")
        out["by_kind"][cat] = dict(collections.Counter(r[key] for r in rs))
        if rs:
            out["numeric_share"][cat] = round(sum(bool(set(r["features"]) & set(NUMERIC_FEATURES)) for r in rs)
                                              / len(rs), 3)
            out["features"][cat] = {f: sum(f in r["features"] for r in rs) for f in FEATURE_RES}
    out["numeric_share"]["all"] = round(sum(bool(set(r["features"]) & set(NUMERIC_FEATURES)) for r in rows)
                                        / max(1, len(rows)), 3)
    emo = [r for r in rows if r["category"] == "emotions"]
    out["instruction_split"] = dict(collections.Counter(r["instruction_split"] for r in emo))
    out["paired_questions"] = len({r["pair_id"] for r in rows if r["kind"] == "mi_soru"})
    return out


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", type=Path, default=BANK, help="folder of the source files [%(default)s]")
    ap.add_argument("--out", type=Path, default=None, help=f"merged output [<bank>/{OUT_NAME}]")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="more protected jsonl files (e.g. Antalia's voxcpm_val.jsonl)")
    ap.add_argument("--check", action="store_true", help="validate and report only, write nothing")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any row is dropped")
    ap.add_argument("--show-overlaps", type=int, default=10, help="list this many rows sharing 4-word runs [10]")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    missing = [str(p) for p in a.exclude if not p.exists()]
    if missing:
        print(f"[textbank] --exclude not found: {missing}")
        return 2
    sources = load_sources(a.bank)
    errs = schema_errors(sources)
    if errs:
        print(f"[textbank] {len(errs)} schema errors, nothing written:")
        for e in errs[:200]:
            print("  " + e)
        return 2
    files = protected_files(a.exclude)
    guard, protected = leak_guard(a.exclude)
    res = build_bank(sources, guard, protected)
    st = stats(res["rows"])
    print(f"[textbank] protected: {len(files)} files, {len(guard.exact)} exact keys, {len(guard.contain)} contained "
          f"(digest {guard.digest()})" + ("" if ANTALIA_HELDOUT.exists() else f"; {ANTALIA_HELDOUT} not found"))
    print(f"[textbank] kept {st['rows']} rows, dropped {len(res['drops'])}")
    for i, why in res["drops"]:
        print(f"  drop {i}: {why}")
    for cat in SOURCES:
        n = st["by_category"].get(cat, 0)
        print(f"  {cat:14s} {n:5d} (target ~{TARGETS[cat]})  numbers/times/money/phones/acronyms: "
              f"{st['numeric_share'].get(cat, 0):.0%}  {st['by_kind'][cat]}")
    print(f"  all            {st['rows']:5d}  numbers/times/money/phones/acronyms: {st['numeric_share']['all']:.0%}")
    for cat, fs in st["features"].items():
        print(f"  features {cat}: " + ", ".join(f"{k} {v}" for k, v in fs.items() if v))
    print(f"  emotion instruction split: {st['instruction_split']}; question pairs: {st['paired_questions']}")
    ov, co = res["ngram_overlaps"], res["content_overlaps"]
    print(f"  rows sharing a 4-word run with a protected sentence (not leakage, report only): {len(ov)}, "
          f"of which with words other than number readings: {len(co)}")
    for i, g in co[:a.show_overlaps]:
        print(f"    {i}: {g}")
    laugh = load_laughter(a.bank)
    if lerrs := laughter_errors(laugh, guard, res["rows"]):
        print(f"[textbank] {LAUGHTER}: {len(lerrs)} problems, nothing written:")
        for e in lerrs[:200]:
            print("  " + e)
        return 2
    print(f"  {LAUGHTER}: {len(laugh)} lines checked (the laughter trial; not merged)")
    if not a.check:
        out = a.out or a.bank / OUT_NAME
        write_atomic(out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res["rows"]))
        print(f"[textbank] wrote {out}")
    return 1 if a.strict and res["drops"] else 0


if __name__ == "__main__":
    sys.exit(main())
