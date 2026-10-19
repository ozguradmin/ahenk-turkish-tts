// The port against what the Python trnorm gives (test/golden.json, written by tools/golden.py from ../src/trnorm).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { test } from 'node:test';
import * as T from '../src/index.js';

const golden = JSON.parse(fs.readFileSync(new URL('./golden.json', import.meta.url), 'utf8'));

const call = (fn, x) => {
  try {
    return fn(x);
  } catch (e) {
    return { error: e.name };
  }
};

function compare(name, fn, rows, input, key) {
  test(`${name}: ${rows.length} cases from Python ${golden.python}`, () => {
    const bad = [];
    for (const r of rows) {
      if (!(key in r)) continue;
      const got = call(fn, input(r));
      if (JSON.stringify(got) !== JSON.stringify(r[key])) bad.push({ input: input(r), python: r[key], js: got });
    }
    assert.deepEqual(bad.slice(0, 5), [], `${bad.length} differ`);
  });
}

compare('normalize', T.normalize, golden.texts, (r) => r.text, 'normalize');
compare('normalize_for_eval', T.normalize_for_eval, golden.texts, (r) => r.text, 'eval');
compare('inverse_normalize', T.inverse_normalize, golden.texts, (r) => r.text, 'itn');
compare('number_to_words', T.number_to_words, golden.numbers, (r) => BigInt(r.n), 'cardinal');
compare('ordinal_to_words', T.ordinal_to_words, golden.numbers, (r) => BigInt(r.n), 'ordinal');

test('casing', () => {
  assert.equal(T.tr_upper('istanbul ılık'), 'İSTANBUL ILIK');
  assert.equal(T.tr_lower('İSTANBUL ILIK'), 'istanbul ılık');
});

test('README examples', () => {
  assert.equal(T.normalize("Saat 09:30'da 3. kata gelin, ücret 1.250,50 TL."), "Saat dokuz otuzda üçüncü kata gelin, ücret bin iki yüz elli lira elli kuruş.");
  assert.equal(T.inverse_normalize('saat on dört otuzda yüzde on beşlik indirim'), "saat 14.30'da %15'lik indirim");
  assert.equal(T.normalize_for_eval("ABD'de 5 TL mi?"), 'a be dede beş lira mı');
  assert.equal(T.number_to_words(1923), 'bin dokuz yüz yirmi üç');
  assert.equal(T.ordinal_to_words(4), 'dördüncü');
});
