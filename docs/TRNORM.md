# trnorm: kurallar ve okuma tercihleri

`trnorm`, yazılı Türkçeyi TTS modellerine verilecek okunuş formuna çevirir (`normalize`), ASR ölçümü için kanonik bir biçim üretir (`normalize_for_eval`) ve ASR çıktısını rakamlı yazıya geri çevirir (`inverse_normalize`). Bu belge hangi durumda neyi neden seçtiğimizi anlatır. Bir tercihe katılmıyorsan, `src/trnorm/lexicon.py`'deki listeler ve `tests/test_normalizer.py`'deki örnekler değiştirilecek yerlerdir.

## Genel ilkeler
1. **Ekler okunuşa göre yeniden uyumlanır.** Yazar `15.00'de` yazmış olsa da okunuş "on beş" olduğundan çıktı "on beşte" olur. Ünlü uyumu, d/t ve c/ç benzeşmesi, kaynaştırma harfleri (y, n, s), ünsüzle biten okunuşta iyelik ekinin bağlayıcı ünlüsü (`25,50 TL'm` → "... elli kuruşum", `12,40 TL'nız` → "... kuruşunuz") ve `dört → dörd-` yumuşaması uygulanır. Değişmeyen ekler (`-ki`, `-yor`) korunur; `-ki`'den sonraki ekler de onun ünlüsüne uyar (`1990'lardakiler` → "... doksanlardakiler", `2026'dakidir` → "... altıdakidir").
   - **Ayrı yazılan soru eki ve "de/da" da uyumlanır.** Açılan bir ifadenin (sayı, para, birim, yüzde, saat, tarih, kısaltma, akronim...) hemen ardındaki `mi/mı/mu/mü` ve çekimli biçimleri (`miyim`, `misin`, `misiniz`, `midir`, `miydi`, `miymiş`, `miyse`...) ile "de/da" bağlacı, okunuşun son ünlüsüne göre yeniden yazılır: `5 TL mi?` → "beş lira mı?", `49,90 TL mi?` → "... doksan kuruş mu?", `SGK mi?` → "se ge ka mı?", `Dr. de` → "doktor da", `5 TL de mi?` → "beş lira da mı?". Önceki kelime açılmadıysa (ya da yalnızca küçük harfe çevrildiyse: `NATO mi` → "nato mi") yazarın tercihi korunur. `ki`, kelime içindeki "mi" ve diğer kelimelere dokunulmaz. Klitik ya tamamen küçük ya tamamen büyük harfle yazılmış olmalı (`Mİ` olur, `Mi` olmaz).
2. **Rakam bırakılmaz.** Okunuşu belirsiz bir ifade bile en olası okunuşla yazıya çevrilir. TTS modelleri rakamı ya hiç okuyamıyor ya da yanlış okuyor. Hiçbir kuralın tanımadığı bir rakam dizisi de son adımda düz okunur (`802.11b` → "sekiz yüz iki nokta on bir b", `6inçlik` → "altı inçlik"); çıktıda rakam kalmaz. Bu adımda nokta yalnızca bir sayı kelimesinden sonra "nokta" okunur; bir kısaltmadan sonra duraktır ve ek sayıya bağlanır (`Tablo.3'te` → "Tablo üçte", `md.5` → "md beş").
3. **Büyük/küçük harf ve noktalama korunur** (`normalize`). TTS modelleri bunları tonlama için kullanıyor. Kesme işareti özel isimlerde kalır (`Ankara'ya`). Sayı yazıya çevrilince kesme kalkar (`5'te` → "beşte"), TDK yazımına uygun olarak. Okunuşa çevrilen yabancı ad büyük harfle yazılmışsa okunuşu da büyük harfle başlar, Türkçe kurala göre (`Google'da` → "Gugılda", `Instagram` → "İnstagram"; EMA Lightning'in alfabe adımı `I`'yı `ı`, `İ`'yi `i` yapar).
4. **Değerlendirme formu iki tarafa da aynen uygulanır** (`normalize_for_eval`): okunuş formu, Türkçe küçük harf (I→ı, İ→i), noktalama yok, kesmeler birleşik (`Ankara'ya` → "ankaraya"), şapkalar düz (`kâr` → "kar"), "buçuk" ↔ "virgül beş" eşit. Yabancı adlar ve tek harfler (baş harf, `X'te`) burada yazıldığı gibi kalır (`iPhone'umdan` → "iphoneumdan", `John F. Kennedy` → "john f kennedy"): ASR çıktısı ve referanslar bunları yazılı biçimle verir, STT eğitim metinleri de bu fonksiyondan geçer.

## Sayılar
| Durum | Örnek | Çıktı | Not |
|---|---|---|---|
| Binlik nokta | `52.000` | elli iki bin | Tam üç haneli gruplar binlik sayılır |
| Ondalık virgül | `3,25` | üç virgül yirmi beş | |
| ,5 | `1,5 saat`, `%2,5` | bir buçuk saat, yüzde iki buçuk | Konuşmadaki doğal okunuş; ölçümde "virgül beş" ile eşit sayılır |
| Noktalı ondalık | `Python 3.12` / `1.5 milyon` | üç nokta on iki / bir buçuk milyon | Ölçek kelimesi, yüzde, oran, faiz bağlamında "virgül" |
| Sıra sayısı | `3. kat`, `4'üncü`, `XIX. yüzyıl`, `3. Yüzyıl` | üçüncü kat, dördüncü, on dokuzuncu yüzyıl, üçüncü Yüzyıl | `N.` ancak ardından küçük harfli kelime ya da `Lig`, `Dünya`, `Yüzyıl`, `Murat` gibi bilinen bir isim gelirse sıra sayısıdır; cümle sonundaki `2023.` asıl sayıdır |
| Aralık | `4-5 dakika`, `1914-1918`, `-10 - 5 derece` | dört ila beş dakika, eksi on ila beş derece | Skor bağlamında (`yendi`, `maç`, `skor`...) "üç bir". Ardından `arası`/`arasında` gelirse "ile": `13:00-17:00 arasında` → on üç ile on yedi arasında |
| Kesir | `1/2`, `3/4'ü` | ikide bir, dörtte üçü | `24/7` → yirmi dört yedi; gün/ay olabilen `15/03` → on beş Mart |
| Adres numarası | `No:12/4`, `Daire 3/1`, `Kapı no 15/3`, `10/2 nolu` | numara on iki bölü dört, Daire üç bölü bir, ... | `No`, `numara`, `kapı`, `daire`, `kat`, `blok`, `apartman`, `cadde`, `sokak`, `bulvar` sonrasında ya da `nolu`/`numaralı` öncesinde `N/M` tarih ya da kesir değil, "bölü"dür |
| Çarpma | `12 x 833,25 TL`, `12*2`, `3×4`, `4*5=20` | on iki çarpı sekiz yüz otuz üç lira yirmi beş kuruş, ... | İki sayı arasındaki `x`, `×`, `*` "çarpı"dır (`0x1F` gibi onaltılık sayılar hariç) |
| Sürüm, IP | `3.x`, `1.2.3`, `192.168.1.1`, `GPT-3.5` | üç nokta iks, bir nokta iki nokta üç, ..., ge pe te üç nokta beş | Üç haneli gruplardan oluşan `1.250.000` binlik sayıdır |
| Kuvvet | `x²`, `10²`, `(a+b)²`, `a³` | iks kare, on kare, (a+b) kare, a küp | Sayıdan sonraki `m²`, `cm³` birimdir (metrekare) |
| Negatif | `-5 derece` | eksi beş derece | |
| Dağıtma / topluluk | `4'er`, `2'şer`, `1990'lı` | dörder, ikişer, bin dokuz yüz doksanlı | |

## Kimlikler, kodlar, telefonlar
| Durum | Örnek | Çıktı |
|---|---|---|
| Kimlik bağlamı (`numara`, `no`, `sipariş`, `takip`, `kod`...), 5–8 hane | `Sipariş no 584213`, `Hesabınız 12345678` | beş sekiz dört iki bir üç (rakam rakam) |
| ...ama sayım değil | `Normalde 25000 kişi`, `Müşteri sayısı 25000`, `Kart limiti 25000 lira`, `Kodlama 2026 yılında` | yirmi beş bin kişi, ..., iki bin yirmi altı yılında |
| 9+ hane | `1429614680` | yüz kırk iki, dokuz yüz altmış bir, kırk altı, seksen (3-3-2-2) |
| Tek kullanımlık kod, PIN (4+ hane) | `Doğrulama kodunuz 4821`, `PIN 4821` | dört sekiz iki bir |
| Sıfırla başlayan | `07339` | sıfır yedi üç üç dokuz |
| Cep / sabit telefon | `0555 555 55 55` | sıfır beş yüz elli beş, beş yüz elli beş, elli beş, elli beş |
| Uluslararası | `+90 555 555 55 55` | artı doksan, beş yüz elli beş, ... |
| 444 hatları | `444 0 333` | dört yüz kırk dört, sıfır, üç yüz otuz üç |
| Plaka | `34 ABC 123`, `52 KH 2334` | otuz dört a be ce yüz yirmi üç; dört haneliler ikişer okunur: yirmi üç otuz dört |
| IBAN | `TR12 0006 4000 ...` | te re on iki, sıfır sıfır sıfır altı, dört sıfır sıfır sıfır, ... (dörtlü gruplar rakam rakam: dinleyen yazabilsin) |
| Posta kodu | `Posta kodu 06100`; adreste yer adından önce `34728 Kadıköy` | sıfır altı bir sıfır sıfır; üç dört yedi iki sekiz (rakam rakam) |

- Kimlik bağlamı kelimesi ya tam kelime ya da ekli hâliyle sayılır (`numaranız`, `kodu`, `hesabınız`); onunla başlayan başka bir kelime sayılmaz (`normalde` "no" değil, `televizyonda` "tel" değil, `Kartal'da` "kart" değil, `takipçi` "takip" değil). Hemen önce `sayısı`, `limiti`, `borcu`, `tutarı` gibi bir miktar kelimesi ya da hemen sonra `kişi`, `lira`, `adet`, `yıl` gibi bir birim/sayılan isim varsa sayı kimlik değil, sayımdır.
- Posta kodu: 5 hane, `01`–`81` ile başlar, önceki 100 karakterde tam bir adres kelimesi var (`Mah.`, `Cad.`, `Sok.`, `No`, `Daire`, `adres`...; `katılan`, `kategoride`, `mahallede` sayılmaz), sayı virgül/eğik çizgi/iki nokta, bir adres kelimesi ya da kapı/daire numarasından hemen sonra gelir ve ardından büyük harfle başlayan bir yer adı gelir: `No:5 D:3 34728 Kadıköy`, `Gül Sok. 34728 Kadıköy`. Büyük harfli başlıkta yalnızca kapı numarasından sonra: `İSTİKLAL CADDESİ 50000 KİŞİYLE DOLDU` → "elli bin kişiyle".

## Tarih ve saat
- `26.09.2027`, `26/09/2027`, `2027-09-26` → "yirmi altı Eylül iki bin yirmi yedi". Ay adları TDK'ya uygun olarak büyük harfle yazılır.
- Yıl başta: `2026/12/31`, `2026/1/5`, `2026.12.31` → "otuz bir Aralık iki bin yirmi altı".
- `12.2027` → "Aralık iki bin yirmi yedi".
- `14:30` → "on dört otuz", `09:00` → "dokuz", `06:05` → "altı sıfır beş". "İki buçuk", "dokuzu beş geçe" gibi gündelik okunuşlar seçilmez çünkü 24 saat biçimini birebir taşımazlar.
- Noktalı saat (`10.40`) yalnızca bağlam varsa saat sayılır: ek (`10.00'a`), yakında `saat/sabah/akşam/kadar/itibaren...` kelimeleri ya da `:00`, `:30` dakikaları. Diğer durumlarda ondalık okunur.

## Para, yüzde, birimler
- `1.250,75 TL`, `₺1.250,75` → "bin iki yüz elli lira yetmiş beş kuruş". `$3,50` → "üç dolar elli sent". İki haneli ondalık alt birimle okunur, diğerleri "virgül" ile.
- Sayıdan sonra gelen `TL` "lira" okunur; tek başına `TL` "te le" okunur (`TL cinsinden`).
- `€` → "avro" (TDK).
- `%20'si` → "yüzde yirmisi"; `20%` → "yüzde yirmi"; `‰5` → "binde beş".
- Birimler yalnızca sayıdan sonra açılır: `5 km'den` → "beş kilometreden", `250 mg'lık` → "iki yüz elli miligramlık", `120 km/sa` → "saatte yüz yirmi kilometre", `38°C'ye` → "otuz sekiz dereceye".

## Kısaltmalar ve akronimler
- **Noktalı kısaltmalar** sözlükle açılır (`Prof. Dr.`, `Doç.`, `vb.`, `vs.`, `bkz.`, `Cad.`, `Sok.`, `Mah.`, `No:`, `A.Ş.`, `T.C.`...). Cümle sonundaki `vb.` gibi kısaltmalarda nokta cümle sonu olarak korunur.
- **Akronimler:** önce sözlük (`THY` → "te ha ye", `SMS` → "esemes", `IBAN` → "iban", `CEO` → "si i o", `QR` → "kü ar"). Sözlükte yoksa:
  - **Okunabiliyorsa kelime gibi okunur:** en az bir ünlü, başta/sonda çift ünsüz yok, en çok iki ünsüz yan yana (`NATO`, `MEB`, `TÜBİTAK`, `ASELSAN`). 5+ harfli sözcüklerde gevşek kural uygulanır (`ÖZTRAK`).
  - **Okunamıyorsa harf harf okunur:** `TBMM` → "te be me me", `ABD` → "a be de", `ÖSYM` → "ö se ye me".
  - K harfi "ka" okunur (`KDV` → "ka de ve"), H harfi "he" okunur (`CHP` → "ce he pe"). Yaygın istisnalar (`THY`, `HDP`) sözlüktedir.
  - Ek, son harfin adına uyumlanır: `SGK'ye` → "se ge kaya" (yazar "ke" okunuşunu düşünmüş olsa bile).
  - Harf harf okunan akronimdeki Latin `I` "i" okunur, "ı" değil: `IBM` → "i be me", `HDMI` → "he de me i", `IP` → "i pe" (Türk klavyesi olmadan yazılmış `IETT` de "i e te te"). Kelime gibi okunan büyük harfli Türkçe kelimeler `ı`'yı korur (`ISPARTA'DA` → "ısparta'da", `TIR` → "tır"); yabancı olanlar sözlüktedir: `ISO` → "iso", `BIOS` → "bayos", `ID` → "ay di", `BIST`/`BİST` → "bist", `AI` → "ey ay", `PDF` → "pe de ef" (ek buna uyar: `PDF'ten` ve `PDF'den` → "pe de eften"), `BMW` → "be me ve".
- **Büyük harfli başlıklar** (`ÇOK ÖNEMLİ BİR DUYURU`) küçük harfe çevrilir; içlerindeki bilinen akronimler korunur. Başlıkta Türkçe kelime biçimindeki her kelime kelime olarak okunur: başında ya da sonunda Türkçede olağan bir ünsüz kümesi olanlar da (`DÖRT KİŞİ`, `TÜRK`, `KRİZ`, `GENÇ`, `HALK`, `ÜST`, `SPOR`, `KONTROL` → "dört kişi", "türk", "kriz"...). Ünlüsüz ya da Türkçe kelimede olmayan kümeli olanlar akronim sayılıp harf harf okunur (`AVM`, `İETT`, `DHA`, `AİHM`, `TBMM`), `EVAL_ACRONYMS`'teki `EFT` gibi akronimler de. Sayıların yazıya çevrilmiş hâli başlığı karışık harfli yapmaz (`SAAT 14:30'DA BAŞLAYACAK DÖRT ETKİNLİK` → "saat on dört otuzda başlayacak dört etkinlik").
- **Tek büyük harf etiketleri:** `C vitamini`, `B planı` → "ce vitamini", "be planı". Zamir olan `O` dokunulmadan kalır.
- **Ekli tek harf** harfin adıyla okunur: `X'te` (platform) → "ikste", `A'dan Z'ye` → "adan zeye" (`O'nun` dokunulmaz; ek küçük ya da tamamı büyük harfle yazılmalı, `D'Artagnan`, `L'Occitane` ad olarak kalır). `I`, `V`, `X` için yazılan ek hangi okunuşun kastedildiğini söyler: yalnızca sayıya uyuyorsa Romen rakamıdır (`Ek I'de` → "Ek birde", `Şekil V'te` → "Şekil beşte"), yalnızca harfe uyuyorsa harftir (`X'te` → "ikste", `V'ye` → "veye"). İkisine de uyuyorsa ya da hiçbirine uymuyorsa, önceki kelime bir hükümdar adı ya da `Madde` gibi bilinen bir isimse Romen rakamıdır: `Louis X'un` → "Louis onun", `Madde V'e` → "Madde beşe".
- **Baş harfler:** büyük harfli bir kelimeden önceki `V.`, `F.` harfin adıyla okunur, nokta kalır (cümle sonu da olabilir): `V. Öztürk` → "ve. Öztürk", `Dr. I. Yılmaz` → "doktor ı. Yılmaz", `Vitamin C. Sonra` → "Vitamin ce. Sonra".
  - Tek harfli `V.` ancak ardından küçük harfli kelime ya da bilinen bir isim (`Murat`, `Dünya`, `Yüzyıl`, `Charles`, `Daire`, `Hukuk`...) gelirse sıra sayısıdır: `V. Murat` → "beşinci Murat", `Yargıtay V. Hukuk Dairesi` → "Yargıtay beşinci Hukuk Dairesi".
  - Tek harfli `I.` ve `X.` sıra sayısıdır (`I. Balkan Savaşı`, `I. Meşrutiyet`, `I. TBMM`, `I. DÜNYA SAVAŞI` → "birinci ..."); ancak önünde bir unvan (`Dr.`, `doktor`, `Prof.`, `Av.`, `Sayın`, `Bay`...) ya da başka bir baş harf varsa baş harftir: `Dr. I. Yılmaz`, `Av. X. Demir`, `J. I. Smith`.

## Yabancı adlar ve markalar
Türkçe okuyanın yanlış okuyacağı yabancı adlar (marka, uygulama, ürün) ve iş/internet dilindeki İngilizce kelimeler sözlükle (`FOREIGN_NAMES`, `src/trnorm/lexicon.py`) Türkçe yazımlı okunuşa çevrilir. Kesmeden sonraki ek okunuşa göre yeniden uyumlanır:

| Girdi | Çıktı |
|---|---|
| `iPhone'umdan`, `iPhone'larımıza` | ayfonumdan, ayfonlarımıza |
| `ChatGPT'ye`, `OpenAI'a`, `Claude'u` | Çet ci pi tiye, Opın ey aya, Klodu |
| `Google'da`, `YouTube'um`, `WhatsApp'tan` | Gugılda, Yutubum, Vatsaptan |
| `Instagram'e` (yanlış ek), `Intel` | İnstagrama, İntel |
| `Visual Studio Code'un`, `GitHub Copilot'ımızda` | Vijuıl stüdyo kodun, Githab Kopaylıtımızda |
| `Coca-Cola`, `McDonald's'ta`, `Disney+`, `Wi-Fi` | Koka kola, Mekdonıldsta, Disni plas, Vayfay |
| `CHATGPT`, `NVIDIA`, `GITHUB` (büyük harf) | çet ci pi ti, envidya, githab |
| `Online meeting'e`, `feature'ı deploy`, `mail'den forward` | Onlayn mitinge, fiçırı diploy, meylden forvırd |

- Anahtar yazıldığı gibi eşleşir. Tamamı büyük, tamamı küçük ve yalnızca baş harfi büyük biçimleri de eşleşir (`YOUTUBE`, `youtube`, `Youtube`). `React`, `Rust`, `Java`, `Office`, `Teams`, `Plus`, `Lightning`, `Gemini` yalnızca yazıldığı gibi ve Latin büyük harfle (`GEMINI`) eşleşir, çünkü küçük harfli biçimleri sıradan bir İngilizce ya da Türkçe kelimedir: `Kaptan, gemini terk etme` ve `KAPTAN GEMİNİ TERK ETME` dokunulmaz, `Gemini'ye` → "Ceminaya".
- Türkçe yazımıyla zaten doğru okunan adlar sözlükte yok, dokunulmaz: `Trendyol`, `Getir`, `Migros`, `TikTok`, `Samsung`, `Amazon`, `Tesla`, `BİM`, `A101`.
- Adresler, e-postalar ve kullanıcı adları içindeki adlar web okumasına bırakılır (`www.google.com`, `ali@gmail.com`, `@google`).
- Sözlüğe ekleme: `FOREIGN_NAMES`'e `"Yazılış": "türkçe okunuş"` (küçük harf, Türkçe harflerle) ve `tests/test_normalizer.py`'ye bir örnek.

## Web, semboller
`www.ornek.com.tr/abc` → "ve ve ve nokta ornek nokta kom nokta te re eğik çizgi abc"; `ali@ornek.com` → "ali et ornek nokta kom".
- Emojiler okunmaz, metinden çıkarılır (`Harika 🙂` → "Harika"); TTS modeline ham emoji gitmez.
- İki yer adı arasındaki eğik çizgi duraktır: `Kadıköy/İstanbul` → "Kadıköy, İstanbul".

## Bilinen sınırlar
- Bağlam penceresi kısa (birkaç kelime), anlam çözümlemesi yok. `10.40` gibi belirsiz yazımlar her zaman isabetli okunmaz.
- Yabancı adların okunuşu yalnızca sözlükteki adlar için düzeltilir (`FOREIGN_NAMES`, yaklaşık 150 ad ve kelime). Sözlükte olmayan yabancı adlar yazıldığı gibi kalır, onları TTS modeli okur. Büyük `I` ile başlayan böyle bir ad EMA Lightning'de `ı` ile okunur.
- Kesmesiz eklenmiş yabancı kelime eşleşmez (`maili`, `Googleda`); `mail`'in eski Türkçe anlamı ("eğilimli", `sola mail`) da "meyl" okunur.
- Ürün kodlarındaki harfler Türkçe harf adıyla okunur (`Galaxy S24` → "Geleksi se yirmi dört", `PS5` → "pe se beş"); İngilizce harf adları ("es") için ürün sözlüğü gerekir.
- Posta kodu ancak adres içinde ve yer adından önce tanınır (yukarıdaki kural); tek başına `34728`, yer adından sonra gelen (`Kadıköy 34728 İstanbul`) ya da büyük harfli bir adres satırında sokak adından hemen sonra gelen posta kodu sayı olarak okunur.
- Büyük harfli başlıkta Türkçe kelimeye benzeyen kısa akronimler kelime gibi okunur (`ERP`, `UPS` → "erp", "ups"); bunlar için `ACRONYMS` sözlüğüne giriş gerekir. Sözlükte olmayan büyük harfli yabancı adlar (`SCHENGEN`, `MARTELLY`) harf harf okunabilir.
- Cümle sonundaki tek harfli `X.` ardından büyük harfli bir kelime gelirse sıra sayısı okunur (`Malcolm X. Daha sonra` → "Malcolm onuncu Daha sonra"); `X'in`, `X'te` gibi ekli yazımlar doğru okunur.
- Adres içindeki alan adı yazıldığı gibi okunur (`turkiye.gov.tr` → "turkiye", "türkiye" değil).
- Vurgu ve tonlama bilgisi üretilmez.
- Açılan ifadeden sonra ayrı yazılan "de" her zaman "de/da" bağlacı sayılır; "söyle" anlamındaki `6 de` de "altı da" olur. `müdür(ler)` ancak hemen ardından `?` gelirse ve önceki ifade yalın bir sayı değilse soru eki sayılır (`5 TL müdür?` → "beş lira mıdır?"); isim olarak korunur: `2 müdür, 3 memur`, `5 müdür?`, `İK müdür yardımcısı`. Ardışık klitiklerden yalnızca "de/da + mi" zinciri uyumlanır (`5 TL mi D blokta?` → "... mı de blokta?").
- Romen rakamı ancak noktayla biterse (`IV.`) ve ardından kelime gelirse, ya da kesmeyle ek alırsa (`XIV'üncü`) çevrilir. Tek harfli `C.`, `D.`, `M.` baş harf olabileceği için Romen rakamı sayılmaz; tek harfli `I.`, `V.`, `X.` için yukarıdaki baş harf kuralı geçerlidir. Tek başına `II`, `III` Romen rakamı sayılmaz, harf harf okunur.

## ITN (okunuş → yazı) tercihleri
- Tek başına duran 0–10 arası sayılar, `yüz` ve `bin` kelime olarak kalır (TDK önerisi; ayrıca "bir"in tanımlık, "yüz"ün yüz anlamı, "bin"in fiil olması). Bağlam varsa (`saat`, `yüzde`, birim, ay adı) rakama çevrilir.
- Ard arda üç ve daha fazla tek rakam kimlik olarak birleştirilir (`beş sekiz dört` → `584`).
- `sıfır` + alan kodu + gruplar telefon biçimine çevrilir (`0555 555 55 55`).
- Büyük harfle başlayan kelimeler yalnızca tam sayı kelimesiyse sayı sayılır (`Kırklareli` dokunulmaz).
