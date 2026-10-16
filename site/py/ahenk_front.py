"""Ahenk Cep's text side, pure Python: written Turkish in, model-ready pieces out. Runs in the browser (Pyodide) and
in CPython (tests, tts/small/web_pipeline.py).

    prepare(text, speed, vocab) -> JSON {"spoken": ..., "pieces": [{"text", "pause", "ids", "cw", "wstart"}]}

It does exactly what tts/small/ahenk_cep.py does before the model: trnorm.normalize, then EMA Lightning's own steps
with its normalizer switched off (ahenk_cep sets Frontend.spoken to the identity): the unsafe-character filter, the
8 KB blocks, the alphabet step (Turkish lowercase, accents dropped, letters outside the vocabulary become spaces),
greedy chunking into pieces of at most ~10 s with their pauses, and each piece's letter ids and word bounds. The
EMA Lightning parts are copied from ema-lightning 1.1.2 (frontend.py, chunker.py, engine.Engine.piece; Apache-2.0,
canberkkkkkk/ema-lightning); tests/test_web_demo.py checks them against the installed package.
"""

from __future__ import annotations

import json
import re
import unicodedata
from itertools import pairwise

from trnorm import normalize

# ---- ema_lightning/frontend.py ----------------------------------------------------------------------------------
BLOCK_BYTES = 8 * 1024
TURKISH = frozenset("çğıöşüÇĞİÖŞÜ")
TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'", "“": '"', "”": '"', "„": '"',
                            "«": '"', "»": '"', "–": "-", "—": "-", "−": "-", "…": "..."})
UNSAFE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f؜‎‏‪-‮⁦-⁩]")


def blocks(text):
    words, block, size = text.split(), [], 0
    for word in words:
        n = len(word.encode("utf-8")) + 1
        if block and size + n > BLOCK_BYTES:
            yield " ".join(block)
            block, size = [], 0
        block.append(word)
        size += n
    if block:
        yield " ".join(block)


def alphabet(text, vocab):
    text = text.translate(TYPOGRAPHY).replace("İ", "i").replace("I", "ı").lower()
    out = []
    for ch in text:
        if ch not in TURKISH:
            ch = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        out.append(ch if ch and all(c in vocab for c in ch) else " ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def frontend(text, vocab):
    """EMA's Frontend.__call__ with spoken() = identity (trnorm already gave the spoken form)."""
    text = UNSAFE.sub(" ", text.encode("utf-8", "ignore").decode("utf-8"))
    if not text.strip():
        return ""
    return alphabet(" ".join(blocks(text)), vocab)


# ---- ema_lightning/chunker.py -----------------------------------------------------------------------------------
LETTERS_PER_SECOND = 18.0
MAX_SECONDS = 10.0
MAX_LETTERS = 250
SENTENCE_PAUSE = 0.25
CLAUSE_PAUSE = 0.12
CUTS = ((re.compile(r"[.!?]+[\"')]*(?= )"), SENTENCE_PAUSE), (re.compile(r"[,;:](?= )"), CLAUSE_PAUSE),
        (re.compile(r"\S(?= )"), CLAUSE_PAUSE))
LETTER = re.compile(r"[^\W\d_]")


def chunk(text, speed):
    limit = int(min(MAX_LETTERS, LETTERS_PER_SECOND * MAX_SECONDS * speed))
    pieces, rest = [], text.strip()
    while rest:
        cut, pause = len(rest), 0.0
        if len(rest) > limit:
            cut = limit
            for pattern, gap in CUTS:
                ends = [m.end() for m in pattern.finditer(rest, 0, limit + 1)]
                if ends:
                    cut, pause = ends[-1], gap
                    break
        piece, rest = rest[:cut].strip(), rest[cut:].strip()
        if LETTER.search(piece):
            pieces.append((finish(piece), pause))
    if pieces:
        pieces[-1] = (pieces[-1][0], 0.0)
    return pieces


def finish(piece):
    if piece.rstrip("\"')")[-1:] in (".", "!", "?"):
        return piece
    return piece.rstrip(",;:- ") + "."


# ---- ema_lightning/engine.py: Engine.piece ------------------------------------------------------------------------
def piece(text, stoi):
    ids = [stoi.get(ch, 1) for ch in text]
    starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
    bounds = [0] + starts[1:] + [len(text)]
    cw, wstart = [], []
    for w, (a, b) in enumerate(pairwise(bounds)):
        cw += [w] * (b - a)
        wstart += [a] * (b - a)
    return ids, cw, wstart


def pieces(spoken, speed, vocab):
    vocab = list(vocab)
    stoi, letters = {ch: i for i, ch in enumerate(vocab)}, frozenset(vocab)
    out = []
    for text, pause in chunk(frontend(spoken, letters), speed):
        ids, cw, wstart = piece(text, stoi)
        out.append({"text": text, "pause": pause, "ids": ids, "cw": cw, "wstart": wstart})
    return out


def prepare(text, speed, vocab):
    spoken = normalize(text)
    return json.dumps({"spoken": spoken, "pieces": pieces(spoken, float(speed), vocab)}, ensure_ascii=False)
