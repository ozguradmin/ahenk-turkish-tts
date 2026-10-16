# Metin bankası: soru, duygu, ünlem, kısa cevap

Kendi sesimizin (`docs/SES_PLANI.md` adım 3) eğitim derleminin metinleri. VoxCPM2 bunları tek bir tasarlanmış
kadın sesiyle ("f022": ellili yaşlarda, danışman; kalın, kadifemsi, sabırlı) okur; kliplerle eğitilen LoRA'nın Türkçeyi
iyi okuması, soruyu soru gibi sorması ve istenen duyguyu vermesi beklenir. Dinleme notları: sorular soru gibi
duyulmuyordu, duygular (özellikle üzgün) zayıftı, bütün sesler "Vay canına"yı "vey kenına" diye okudu.

| Dosya | Satır | İçerik |
|---|---:|---|
| `questions.jsonl` | 753 | 251 evet-hayır sorusu (mI her konumda, her kişi ve zamanda) + eşli düz cümlesi (`pair_id`), 121 soru sözcüklü (ne, neden, nasıl, nerede, kim, kaç, hangi), 40 "..., değil mi?", 30 mI'siz yükselen soru, 60 kibar hizmet sorusu |
| `emotions.jsonl` | 1.018 | 6 duygu × ~170: kızgın, neşeli, üzgün, şaşkın, sakin, empatik; her satırda İngilizce VoxCPM2 talimatı |
| `interjections.jsonl` | 259 | 23 ünlem: vay, vay canına, ay, of, öf, offf, oh, oh be, aa, ah, eyvah, tüh, yaşasın, hadi ya, hey, haydi, hmm, ha, aman, eh, yuh, maşallah, inşallah |
| `short.jsonl` | 255 | kısa cevap ve onaylar (onay, ret, teşekkür, bekleme, belirsizlik, selam, özür, tepki, kısa soru, dilek); . ! ? ... çeşitliliği |
| `textbank.jsonl` | 2.285 | `build_textbank.py`'nin birleşik çıktısı (aşağıda) |
| `laughter.jsonl` | 52 | gülme denemesi: içinde gülme geçen özgün satırlar ("Hahaha, çok komiksin!"); `emotion` cheerful, `intensity`, `laugh: true`, `laugh_span` (gülmenin metindeki yazımı). `build_textbank.py` denetler, `textbank.jsonl`'a katmaz; `render_corpus.py --laughter-usd` yalnız dinleme için üretir, eğitime girmez |

Konular gündelik: ev, aile, alışveriş, sağlık randevusu, banka, kargo, seyahat, okul, iş, yemek, hava, spor, arkadaşlar.
Satırların %16'sında (sorularda %28) rakamla yazılmış sayı, saat, para, tarih, telefon, adres, yüzde ya da kısaltma var;
trnorm bunları okunuş formuna çevirir.

## Alanlar

- **questions**: `kind` (mi_soru, duz_eslesik, ne_soru, degil_mi, yukselen, hizmet), `pair_id` (soru ile eşli düz cümle;
  bazı değil_mi / yükselen satırlar da aynı cümleyle aynı grupta), `peak_word`: `trnorm.normalize(text).split()`
  içinde perde tepesini taşıyan kelimenin 0'dan sıra numarası. mI'den önceki kelime, "değil", soru sözcüğü, yükselende
  son kelime; düz cümlede sorusununkiyle aynı kelime. Emin olunmayan yerde (dA'dan sonra mI, iki mI'li seçme sorusu)
  `null`. Birleşik çıktıdaki `pattern`, `score_intonation.py`'nin adlarıdır (peak_fall, wh_peak, final_rise, fall).
- **emotions**: `emotion`, `intensity` (mild/strong), `instruction` (parantezsiz; okuyucu `(talimat)` + okunuş formu
  verir), `instruction_id`, `instruction_split`. Her duygunun 10 talimat çeşidinden 2'si (%20) **ayrılmıştır**
  (`heldout`): o satırlar eğitime girmez, modelin görmediği talimat ifadesine uyup uymadığını ölçer. Hafif cümleye
  "slightly / a little", güçlüye "very / deeply" türü talimat düşer. Talimat listesi `build_textbank.py`'de
  (`INSTRUCTIONS`). Üzgün satırların çoğu sessiz, yavaş, yorgun bağlamlarda; talimatları da öyle.
- **interjections**: `interjection` (kanonik), `span` (metinde yazıldığı hâli), `render_variants`: yalnız **seslendirirken**
  denenecek yazımlar (ilk eleman metindeki hâl), ör. "Vay canına" → "Vaay canına", "Vay caanına", "Vây cânına". Eğitim
  metni her zaman `text`; okuyucu, doğru okunan çeşidin sesini kanonik metinle eşler. Birleşik çıktıda `render_texts`
  her çeşidin okunuş formudur. `emotion`: isteğe bağlı duygu ipucu.
- **short**: `kind`, `text`.

## Derleme ve denetim

```powershell
python tts/voxcpm/build_textbank.py            # textbank.jsonl'ı yazar, sayıları ve rakamlı satır oranını basar
python tts/voxcpm/build_textbank.py --check    # yalnız denetim
python -m pytest tests/test_textbank.py
```

- Şema, eşler, tepe kelimesi, mI ünlü uyumu ve talimat bölünmesi denetlenir; hata varsa hiçbir şey yazılmaz.
- Her metin `trnorm.normalize` ile okunuş formuna (`spoken`) çevrilir. Rakam ya da sembol kalan, normalizasyonu
  başarısız olan satır atılır.
- Tekrar: `normalize_for_eval` + son işaretin sınıfı (? / ... / . !). "Yarın geliyorsun." ile "Yarın geliyorsun?"
  ayrı satırdır.
- **Sızıntı koruması**: `prepare_alania.py`'nin koruması (aynısı, içe aktarılır). `benchmarks/tts/sets/*.jsonl`
  (TR-Zor, TR-Soru, yarışma, ...) ve Antalia'nın ayrılmış cümleleri (`voxcpm-eval/antalia-gt/rows.jsonl`) korunur: bir
  satır korunan bir cümleye eşit olamaz, ≥4 kelimelik birini içeremez. Bu yüzden TR-Zor'daki tek kelimelik "Evet.",
  "Tamam.", "Anladım.", "Peki." ve freya setindeki "Bir saniye lütfen." bankada yok; yerine "Evet efendim.",
  "Tamam, anladım." gibi biçimler var. Test ayrıca, sayı okunuşu dışında hiçbir 4 kelimelik dizinin korunan cümlelerle
  ortak olmamasını ister.

## Lisans

Metinlerin hepsi bu depo için yazıldı; hiçbir veri setinden alınmadı. Deponun lisansı (Apache-2.0) geçerlidir;
ayrıca hak iddia etmeden CC0 1.0 olarak da kullanılabilir. Üretilen ses yapaydır ve öyle belirtilmelidir.

Note: in the public copy, the example phone numbers in seven question rows (`q_*_025`, `q_*_071`, `q_degil_040`,
`q_*_110`) were replaced with obviously fictitious ones (`0555 555 55 55`, `0 (222) 222 22 22`, `0216 555 50 50`)
with the same spoken word count, and `textbank.jsonl` was rebuilt. The released model was trained on corpora rendered
from the earlier wording of these seven rows.
