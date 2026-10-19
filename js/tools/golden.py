#!/usr/bin/env python3
"""Writes test/golden.json: what the Python trnorm (../src/trnorm) gives for a fixed set of inputs, for the
JavaScript port's tests (test/golden.test.mjs).

    python3.12 js/tools/golden.py

Inputs: every string in tests/test_normalizer.py (the Python's own tests), the examples in docs/TRNORM.md, a seeded
set of sentences with numbers, dates, times, money, phones, units and acronyms, and their spoken and eval forms as
input to inverse_normalize (ASR-like text: words where the text had digits). Numbers: 0..2000 and seeded values up
to 10**30 for number_to_words / ordinal_to_words. Run it with Python 3.12, whose Unicode (15.0) the port follows;
the same Python always writes the same file.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.dont_write_bytecode = True

import trnorm  # noqa: E402


def strings_in(path):
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.strip()]


def doc_examples(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    return [m.group(1) for m in re.finditer(r'`([^`\n]{3,200})`', text)]


def generated(n, seed=7):
    rnd = random.Random(seed)
    words = ['Toplantı', 'saat', 'fiyatı', 'ABD', "TBMM'de", 'arayın', 'yüzde', 'kat', 'gelin', 'ödendi', 'mi?', 'NATO', 'km', 'kg']
    shapes = [
        lambda: str(rnd.randint(0, 10 ** rnd.randint(1, 12))),
        lambda: f'{rnd.randint(0, 23):02d}:{rnd.randint(0, 59):02d}' + rnd.choice(['', "'de", "'da", "'te"]),
        lambda: f'{rnd.randint(1, 28)}.{rnd.randint(1, 12):02d}.{rnd.randint(1900, 2100)}',
        lambda: f'{rnd.randint(1, 99999):,}'.replace(',', '.') + ',' + f'{rnd.randint(0, 99):02d} TL',
        lambda: f'%{rnd.randint(1, 100)}' + rnd.choice(['', "'lik", "'e", "'si"]),
        lambda: f'0{rnd.randint(500, 559)} {rnd.randint(100, 999)} {rnd.randint(10, 99)} {rnd.randint(10, 99)}',
        lambda: f'{rnd.randint(1, 30)}.' + rnd.choice([' kat', ' sırada', ' yüzyıl']),
        lambda: f'{rnd.randint(1, 500)} ' + rnd.choice(['km', 'kg', 'm²', '°C', 'MB', 'km/sa']),
        lambda: f"{rnd.randint(1, 9)}'{rnd.choice(['üncü', 'inci', 'de', 'ye'])}",
        lambda: rnd.choice(words),
    ]
    return [' '.join(rnd.choice(shapes)() for _ in range(rnd.randint(1, 8))) + rnd.choice(['', '.', '?', '!']) for _ in range(n)]


def call(fn, x):
    try:
        return fn(x)
    except Exception as e:  # noqa: BLE001 - the type is the result
        return {'error': type(e).__name__}


def main():
    texts = strings_in(os.path.join(ROOT, 'tests', 'test_normalizer.py'))
    texts += doc_examples(os.path.join(ROOT, 'docs', 'TRNORM.md'))
    texts += generated(1500)
    texts = list(dict.fromkeys(texts))
    cases = []
    for t in texts:
        spoken = call(trnorm.normalize, t)
        ev = call(trnorm.normalize_for_eval, t)
        cases.append({'text': t, 'normalize': spoken, 'eval': ev, 'itn': call(trnorm.inverse_normalize, t)})
        for form in (spoken, ev):
            if isinstance(form, str):
                cases.append({'text': form, 'itn': call(trnorm.inverse_normalize, form)})
    rnd = random.Random(11)
    nums = list(range(0, 2001)) + [rnd.randint(0, 10 ** rnd.randint(4, 30)) for _ in range(500)] + [-5, -1234, 10 ** 21 - 1, 10 ** 21]
    numbers = [{'n': str(n), 'cardinal': call(trnorm.number_to_words, n), 'ordinal': call(trnorm.ordinal_to_words, n)} for n in nums]
    out = {'python': sys.version.split()[0], 'trnorm': trnorm.__version__, 'texts': cases, 'numbers': numbers}
    with open(os.path.join(os.path.dirname(HERE), 'test', 'golden.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
        f.write('\n')
    print(f'{len(cases)} texts, {len(numbers)} numbers (Python {out["python"]})')


if __name__ == '__main__':
    main()
