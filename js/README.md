# trnorm (JavaScript)

Yazılı Türkçeyi okunuş formuna çeviren [trnorm](../src/trnorm)'un JavaScript sürümü: tarayıcıda ve Node'da, bağımlılıksız,
Python'la **aynı çıktıyı** verir. Sayı, tarih, saat, para, yüzde, birim, kısaltma, telefon, IBAN, plaka, URL; ekler
okunuşa uyar (`15.00'de` → "on beşte", `SGK'ye` → "se ge kaya").

```js
import { normalize, normalize_for_eval, inverse_normalize, number_to_words } from 'trnorm';

normalize("Saat 09:30'da 3. kata gelin, ücret 1.250,50 TL.");
// "Saat dokuz otuzda üçüncü kata gelin, ücret bin iki yüz elli lira elli kuruş."
normalize_for_eval("ABD'de 5 TL mi?");                 // "a be dede beş lira mı"  (WER/CER için)
inverse_normalize('saat on dört otuzda yüzde on beşlik indirim'); // "saat 14.30'da %15'lik indirim"
number_to_words(1923);                                 // "bin dokuz yüz yirmi üç"
```

```sh
npm install trnorm
```

Kurallar ve örnekler Python sürümüyle ortaktır: [docs/TRNORM.md](../docs/TRNORM.md).

## API

| İşlev | Ne yapar |
|---|---|
| `normalize(text)` | Yazılı metin → okunuş (TTS girdisi). Büyük/küçük harf ve noktalama korunur. |
| `normalize_for_eval(text)` | Ölçüm için kanonik biçim: okunuş, Türkçe küçük harf, noktalamasız. Referansa ve çıktıya aynen uygulanır. |
| `inverse_normalize(text)` | Okunuş → rakamlı yazı (ASR çıktısı için). |
| `number_to_words(n)`, `ordinal_to_words(n)` | Sayının okunuşu: `4` → "dört", "dördüncü". `n`: bigint ya da tam sayı; başka bir sayı Python'daki float gibi okunur. |
| `tr_lower(text)`, `tr_upper(text)` | Türkçe büyük/küçük harf (I ↔ ı, İ ↔ i). |

Python bir hata verdiğinde bu sürüm de hata fırlatır (örneğin 4300 haneden uzun bir sayıda `ValueError`).

## Python'la aynı mı?

Evet; amaç birebir aynılık. Sürüm, Python'un `str`, `re` ve `int` davranışını taklit eder (`src/py.js`): karakter sayımı
kod noktasıyla, `\w`/`\d`/`\s`/`\b` Python'un Unicode tanımıyla, `$`'ın son satır sonundan önce de eşleşmesi, sınırsız
tam sayılar (BigInt), Unicode 15.0'ın büyük/küçük harf tabloları (`src/case_tables.js`), Python 3.12'nin tanımadığı
yeni karakterler (`src/unassigned.js`) ve tek başına kalmış vekil (surrogate) karakterler.

- `npm test`: `test/golden.json`'daki 7.566 metin ve 2.505 sayı. Bu dosyayı `tools/golden.py` aynı depodaki Python
  trnorm'dan üretir (`npm run golden`, Python 3.12 ile); Python'daki bir değişiklikten sonra dosya yeniden üretilip
  test çalıştırılır.
- Daha geniş eşitlik testi [ahenk-cep-pdfreader](https://github.com/ozguradmin/ahenk-cep-pdfreader/tree/claude/tender-cori-4vvwp6/tools/parity)
  deposunda (`tools/parity`): testler, belgeler ve rastgele üretilmiş zor girdiler (dev sayılar, görünmez
  karakterler, birleşik işaretler, emoji, karışık alfabeler, tek başına vekil karakterler, Python'un tanımadığı yeni
  Unicode karakterleri). Son çalıştırmada üç işlevin (ve okunuş biçimlerinin `inverse_normalize`'ı) 742.716 sonucu
  Python 3.12'yle birebir aynı çıktı. `npm test` V8'de (Node) ve JavaScriptCore'da (Safari'nin motoru) geçiyor. Bu
  sürüm, o sitenin (Ahenk Okur) tarayıcıdaki metin ön ucudur.

**Safari 26.0–26.5:** Bu sürümlerin RegExp motoru, U+F800–U+FBFF ya da vekil aralığındaki bir karakterle ardından gelen
U+FC00–U+FFFF aralığındaki bir karakteri (örneğin `ﬁ５`) tek karakter sanıyor. Böyle bir çift içeren nadir bir metinde
işlevler yanlış sonuç vermek yerine `name === 'EngineError'` olan bir hata fırlatır; gerekirse metni sunucuya ya da
Python'a gönderin.

Gereken: ES2022 ve Unicode özellik kaçışları olan bir JavaScript motoru (Node 18+, güncel tarayıcılar, Safari 16.4+).

## English

A JavaScript port of the Python [trnorm](../src/trnorm), with the same output for the same input: Turkish written →
spoken form for TTS (`normalize`), the canonical form for WER/CER scoring (`normalize_for_eval`), and spoken → digits
for ASR output (`inverse_normalize`), plus `number_to_words`, `ordinal_to_words`, `tr_lower`, `tr_upper`. No
dependencies; browsers and Node 18+. It keeps Python's `str`, `re` and `int` semantics (code-point indexing, Unicode
`\w \d \s \b`, Python's `$`, BigInt, Unicode 15.0 case tables, characters newer than Python 3.12's Unicode, lone
surrogates) and throws where Python raises. `npm test` compares it with the Python in this repository on 7,566 texts
and 2,505 numbers (`tools/golden.py` writes the file); the full parity harness lives in ahenk-cep-pdfreader, where the
last run gave the same result as Python 3.12 on all 742,716 results (corpus, fuzz and adversarial inputs). On
Safari 26.0–26.5, whose RegExp engine misreads certain character pairs, such rare inputs throw an `EngineError`
instead of giving a wrong answer.

Apache-2.0, like the rest of this repository.
