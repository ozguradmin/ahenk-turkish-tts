#!/usr/bin/env python3
"""Train the SentencePiece token model (icefall convention: unigram LM,
<blk>=0, <sos/eos>=1, <unk>=2, character_coverage=1.0) on the training
transcripts, optionally topped up with sentences from
cloud0day3/alania-domain-text-tr (CC-BY-4.0) for coverage of numbers, dates,
codes and service-domain words.

    python stt/prepare/make_bpe.py --vocab-size 1000
    python stt/prepare/make_bpe.py --vocab-size 1000 --extra-text alania-domain --extra-max-lines 300000
    python stt/prepare/make_bpe.py --analyze 500,1000,1500,2000   # compare sizes, then exit

Writes $STT_DATA/lang_bpe_<V>/{bpe.model,bpe.vocab,tokens.txt,transcript.txt,stats.json}.
``--analyze`` reports tokens/word and, for cuts with durations, tokens per
second against the 25 Hz encoder output rate (pruned RNN-T needs T >= U).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with python -I
from common import asr_text, iter_jsonl_gz, setup_logging, stt_data_dir  # noqa: E402

DOMAIN_REPO = "cloud0day3/alania-domain-text-tr"
DOMAIN_REVISION = "20fd903fe656dc47edd63c010de15431fc747334"  # v1.2, 2026-10-03
DOMAIN_PARTS = [f"data/part-{i:05d}.parquet" for i in range(6)]
log = setup_logging("bpe")


def cut_texts(path: Path) -> list:
    out = []
    for d in iter_jsonl_gz(path):
        sup = d["supervisions"][0]
        if sup.get("text"):
            out.append((sup["text"], float(d["duration"])))
    return out


def domain_texts(data_dir: Path, max_lines: int, seed: int) -> list:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    raw = []
    for part in DOMAIN_PARTS:
        f = hf_hub_download(DOMAIN_REPO, part, repo_type="dataset", revision=DOMAIN_REVISION,
                            local_dir=str(data_dir / "downloads" / "alania_domain_text"))
        raw.extend(pq.read_table(f, columns=["text_model"]).column("text_model").to_pylist())
    random.Random(seed).shuffle(raw)
    lines = []
    for t in raw:  # normalise only as many lines as needed (trnorm is ~10k lines/s)
        t = asr_text(t)
        if t:
            lines.append(t)
            if len(lines) >= max_lines:
                break
    return lines


def train_spm(text_file: Path, vocab_size: int, model_type: str, prefix: Path, threads: int = 1) -> Path:
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=str(text_file),
        vocab_size=vocab_size,
        model_type=model_type,
        model_prefix=str(prefix),
        input_sentence_size=100_000_000,
        character_coverage=1.0,
        user_defined_symbols=["<blk>", "<sos/eos>"],
        unk_id=2,
        bos_id=-1,
        eos_id=-1,
        num_threads=threads,  # 1 = bit-for-bit reproducible model (multi-threaded EM is not)
        minloglevel=2,
    )
    return prefix.with_suffix(".model")


def token_stats(model: Path, texts_with_dur: list, plain_texts: list) -> dict:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model))
    words = toks = 0
    for t in plain_texts:
        words += len(t.split())
        toks += len(sp.encode(t))
    rates, too_dense = [], 0
    for t, dur in texts_with_dur:
        u = len(sp.encode(t))
        rates.append(u / dur)
        frames = ((int(dur * 100) - 7) // 2 + 1) // 2  # encoder frames (see train.py)
        too_dense += (frames - 2) < u
    res = {"tokens_per_word": round(toks / max(words, 1), 3)}
    if rates:
        res.update({
            "tokens_per_sec_median": round(statistics.median(rates), 2),
            "tokens_per_sec_p99": round(sorted(rates)[int(0.99 * (len(rates) - 1))], 2),
            "utts_failing_T_ge_U": too_dense,
        })
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="default: $STT_DATA")
    p.add_argument("--vocab-size", type=int, default=1000)
    p.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    p.add_argument("--train-cuts", default=None, help="default: fbank/train_mix_cuts.jsonl.gz")
    p.add_argument("--extra-text", default="none", help="none | alania-domain | path to a text file")
    p.add_argument("--extra-max-lines", type=int, default=300_000)
    p.add_argument("--analyze", default=None, help="comma list of vocab sizes to compare, then exit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1,
                   help="SentencePiece trainer threads; >1 is faster but not bit-reproducible")
    a = p.parse_args()
    data_dir = stt_data_dir(a.data_dir)
    train_cuts = Path(a.train_cuts) if a.train_cuts else data_dir / "fbank" / "train_mix_cuts.jsonl.gz"

    td = cut_texts(train_cuts)
    texts = [t for t, _ in td]
    log.info(f"{len(texts)} transcripts from {train_cuts}")
    extra = []
    if a.extra_text == "alania-domain":
        extra = domain_texts(data_dir, a.extra_max_lines, a.seed)
    elif a.extra_text != "none":
        extra = [t for t in (asr_text(x) for x in Path(a.extra_text).read_text().splitlines()) if t]
        extra = extra[: a.extra_max_lines]
    log.info(f"{len(extra)} extra text lines ({a.extra_text})")

    if a.analyze:
        # hold out 10% of transcripts to measure segmentation of unseen text
        rng = random.Random(a.seed)
        held = set(rng.sample(range(len(texts)), max(1, len(texts) // 10)))
        fit = [t for i, t in enumerate(texts) if i not in held] + extra
        eval_texts = [texts[i] for i in held] + extra[-5000:]
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.txt"
            corpus.write_text("\n".join(fit) + "\n", encoding="utf-8")
            for v in [int(x) for x in a.analyze.split(",")]:
                try:
                    m = train_spm(corpus, v, a.model_type, Path(tmp) / f"m{v}", a.threads)
                except RuntimeError as e:
                    log.warning(f"vocab {v}: {e}")
                    continue
                rows.append({"vocab": v, **token_stats(m, td, eval_texts)})
                log.info(json.dumps(rows[-1]))
        print(json.dumps(rows, indent=1))
        return

    # train from a fixed path so that the model file (which records its input
    # path) is bit-identical across runs with --threads 1
    lang = data_dir / f"lang_bpe_{a.vocab_size}"
    lang.mkdir(parents=True, exist_ok=True)
    corpus = lang / "transcript.txt"
    corpus.write_text("\n".join(texts + extra) + "\n", encoding="utf-8")
    train_spm(corpus, a.vocab_size, a.model_type, lang / "bpe", a.threads)

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(lang / "bpe.model"))
    assert sp.piece_to_id("<blk>") == 0 and sp.piece_to_id("<sos/eos>") == 1 and sp.unk_id() == 2
    with open(lang / "tokens.txt", "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")
    st = {"vocab_size": a.vocab_size, "model_type": a.model_type, "transcripts": len(texts),
          "extra_lines": len(extra), **token_stats(lang / "bpe.model", td, texts[:20000])}
    (lang / "stats.json").write_text(json.dumps(st, indent=2))
    log.info(f"wrote {lang}: {st}")


if __name__ == "__main__":
    main()
