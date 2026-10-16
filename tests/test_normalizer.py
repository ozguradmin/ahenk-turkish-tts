import pytest

from trnorm import normalize, normalize_for_eval, number_to_words, ordinal_to_words, tr_lower, tr_upper
from trnorm.numbers import attach_suffix, read_long_digits


@pytest.mark.parametrize(
    "n, words",
    [
        (0, "sıfır"),
        (1, "bir"),
        (10, "on"),
        (11, "on bir"),
        (100, "yüz"),
        (101, "yüz bir"),
        (1000, "bin"),
        (1001, "bin bir"),
        (1923, "bin dokuz yüz yirmi üç"),
        (2000, "iki bin"),
        (10500, "on bin beş yüz"),
        (100000, "yüz bin"),
        (1000000, "bir milyon"),
        (1001000, "bir milyon bin"),
        (2500000000, "iki milyar beş yüz milyon"),
        (-5, "eksi beş"),
    ],
)
def test_cardinals(n, words):
    assert number_to_words(n) == words


@pytest.mark.parametrize(
    "n, words",
    [
        (1, "birinci"),
        (2, "ikinci"),
        (3, "üçüncü"),
        (4, "dördüncü"),
        (6, "altıncı"),
        (9, "dokuzuncu"),
        (10, "onuncu"),
        (20, "yirminci"),
        (40, "kırkıncı"),
        (60, "altmışıncı"),
        (90, "doksanıncı"),
        (100, "yüzüncü"),
        (1000, "bininci"),
        (21, "yirmi birinci"),
    ],
)
def test_ordinals(n, words):
    assert ordinal_to_words(n) == words


@pytest.mark.parametrize(
    "stem, suffix, out",
    [
        ("beş", "te", "beşte"),
        ("on beş", "de", "on beşte"),  # writer error re-agreed
        ("altı", "a", "altıya"),  # buffer y inserted
        ("beş", "ye", "beşe"),  # buffer y dropped
        ("beş", "si", "beşi"),
        ("beş", "se", "beşse"),  # conditional keeps s
        ("dört", "e", "dörde"),
        ("dört", "üncü", "dördüncü"),
        ("doksan", "lerde", "doksanlarda"),
        ("iki bin yirmi üç", "teki", "iki bin yirmi üçteki"),
        ("doksan", "daki", "doksandaki"),
        ("doksan", "lardakiler", "doksanlardakiler"),  # -ki keeps its vowel and fronts what follows
        ("altı", "dakidir", "altıdakidir"),
        ("altı", "dakıdır", "altıdakidir"),  # misspelt -kı
        ("kuruş", "m", "kuruşum"),  # possessive after a consonant takes the linking vowel
        ("kuruş", "n", "kuruşun"),
        ("kuruş", "nız", "kuruşunuz"),
        ("kuruş", "mda", "kuruşumda"),
        ("lira", "m", "liram"),
        ("saat", "de", "saatte"),
        ("ka", "ye", "kaya"),
        ("lira", "ye", "liraya"),
        ("miligram", "lık", "miligramlık"),
        ("dört", "er", "dörder"),
    ],
)
def test_suffix_harmony(stem, suffix, out):
    assert attach_suffix(stem, suffix) == out


def test_turkish_casing():
    assert tr_lower("IŞIK İSTANBUL") == "ışık istanbul"
    assert tr_upper("istanbul ışık") == "İSTANBUL IŞIK"


def test_long_digits_grouping():
    assert read_long_digits("1429614680") == "yüz kırk iki, dokuz yüz altmış bir, kırk altı, seksen"


@pytest.mark.parametrize(
    "written, spoken",
    [
        # numbers with suffixes
        ("5'te görüşürüz.", "beşte görüşürüz."),
        ("1923'te Cumhuriyet ilan edildi.", "bin dokuz yüz yirmi üçte Cumhuriyet ilan edildi."),
        ("100'den fazla", "yüzden fazla"),
        ("1980'lerde", "bin dokuz yüz seksenlerde"),
        ("90'lar", "doksanlar"),
        ("4'er kişi", "dörder kişi"),
        ("3'üncü", "üçüncü"),
        ("40'ıncı", "kırkıncı"),
        ("2.000.000 kişi", "iki milyon kişi"),
        ("1000000", "bir milyon"),
        # ordinals
        ("3. kat, daire 31'de", "üçüncü kat, daire otuz birde"),
        ("Ahmet 2. oldu.", "Ahmet ikinci oldu."),
        ("1. Dünya Savaşı", "birinci Dünya Savaşı"),
        ("XX. yüzyıl", "yirminci yüzyıl"),
        ("II. Abdülhamit", "ikinci Abdülhamit"),
        ("21. yüzyılda", "yirmi birinci yüzyılda"),
        # decimals / fractions / ranges
        ("3,5 kg elma aldım.", "üç buçuk kilogram elma aldım."),
        ("0,05 oranında", "sıfır virgül sıfır beş oranında"),
        ("1/2 oranında", "ikide bir oranında"),
        ("3/4'ü", "dörtte üçü"),
        ("4-5 dakika", "dört ila beş dakika"),
        ("Fenerbahçe 3-1 yendi.", "Fenerbahçe üç bir yendi."),
        ("Python 3.12 sürümü", "Paytın üç nokta on iki sürümü"),  # the name: FOREIGN_NAMES
        ("1.5 milyon kişi", "bir buçuk milyon kişi"),
        # dates and times
        ("26.09.2027'de sabah 10.00'a gelin", "yirmi altı Eylül iki bin yirmi yedide sabah ona gelin"),
        ("Saat 09:30'da toplantı var.", "Saat dokuz otuzda toplantı var."),
        ("21:00'e kadar", "yirmi bire kadar"),
        ("Siparişiniz 15.00’de hazır", "Siparişiniz on beşte hazır"),
        ("09.00-17.00 arası", "dokuz ile on yedi arası"),  # "ile" before arası/arasında
        ("01.01.2025", "bir Ocak iki bin yirmi beş"),
        ("2024-03-15", "on beş Mart iki bin yirmi dört"),
        # money, percent, units
        ("10.500 TL ödeme", "on bin beş yüz lira ödeme"),
        ("₺1.250,75", "bin iki yüz elli lira yetmiş beş kuruş"),
        ("$100", "yüz dolar"),
        ("12.000 TL'ye", "on iki bin liraya"),
        ("%15'lik indirim", "yüzde on beşlik indirim"),
        ("%20'si", "yüzde yirmisi"),
        ("%3,5 faiz", "yüzde üç buçuk faiz"),
        ("250 mg'lık doz", "iki yüz elli miligramlık doz"),
        ("120 km/sa hızla", "saatte yüz yirmi kilometre hızla"),
        ("-5 derece", "eksi beş derece"),
        ("25°C", "yirmi beş derece"),
        ("45 m2 daire", "kırk beş metrekare daire"),
        # phones, ids, plates
        ("0555 555 55 55'i arayın", "sıfır beş yüz elli beş, beş yüz elli beş, elli beş, elli beşi arayın"),
        ("+90 555 555 55 55", "artı doksan, beş yüz elli beş, beş yüz elli beş, elli beş, elli beş"),
        ("Sipariş no 250909", "Sipariş no iki beş sıfır dokuz sıfır dokuz"),
        ("TR42648163", "te re dört iki altı dört sekiz bir altı üç"),
        ("34 ABC 123", "otuz dört a be ce yüz yirmi üç"),
        # abbreviations and acronyms
        ("Dr. Ahmet Bey TBMM'de konuştu.", "doktor Ahmet Bey te be me mede konuştu."),
        ("Prof. Dr. Ayşe", "profesör doktor Ayşe"),
        ("vb. şeyler", "ve benzeri şeyler"),
        ("ABD'ye gitti", "a be deye gitti"),
        ("THY'nin uçağı", "te ha yenin uçağı"),
        ("SGK'ye başvurdu", "se ge kaya başvurdu"),
        ("NATO'ya katıldı", "nato'ya katıldı"),
        ("ÖTV dahil", "ö te ve dahil"),
        ("İSTANBUL'DA", "istanbul'da"),
        ("ÇOK ÖNEMLİ BİR HABER VE DUYURU", "çok önemli bir haber ve duyuru"),
        ("COVID-19 salgını", "kovid on dokuz salgını"),
        ("5G teknolojisi", "beş ge teknolojisi"),
        ("Atatürk Cad. No:15", "Atatürk caddesi numara on beş"),
        ("Günde 500 mg C vitamini", "Günde beş yüz miligram ce vitamini"),
        ("B planı hazır.", "be planı hazır."),
        ("O gelmedi.", "O gelmedi."),
        # web
        ("www.google.com", "ve ve ve nokta google nokta kom"),
        ("ali@ornek.com", "ali et ornek nokta kom"),
        # untouched
        ("Kâr ettik", "Kâr ettik"),
        ("e-Devlet", "e-Devlet"),
        ("Ankara’ya gidiyorum.", "Ankara'ya gidiyorum."),
    ],
)
def test_normalize(written, spoken):
    assert normalize(written) == spoken


def test_no_digits_left_on_common_inputs():
    text = "Toplam 3 ürün, 2 adet 12,90 TL ve 1 adet 7,50 TL; kargo 3-5 iş günü, takip no 123456789."
    assert not any(ch.isdigit() for ch in normalize(text))


@pytest.mark.parametrize(
    "a, b",
    [
        ("1,5 saat", "bir buçuk saat"),
        ("Saat 15:00'te", "saat on beşte"),
        ("%20", "yüzde yirmi"),
        ("Ankara'ya", "ankaraya"),
        ("Kâr", "kar"),
        ("İstanbul, İzmir!", "istanbul izmir"),
    ],
)
def test_eval_normalization_equivalence(a, b):
    assert normalize_for_eval(a) == normalize_for_eval(b)


def test_year_after_month_is_not_a_code():
    assert normalize("İndirim kodunu gir, 21 Kasım 2025'e kadar") == "İndirim kodunu gir, yirmi bir Kasım iki bin yirmi beşe kadar"


from trnorm import inverse_normalize  # noqa: E402


@pytest.mark.parametrize(
    "spoken, written",
    [
        ("saat on dört otuzda yüzde on beşlik indirim", "saat 14.30'da %15'lik indirim"),
        ("bin dokuz yüz yirmi üçte", "1923'te"),
        ("sıfır beş yüz elli beş beş yüz elli beş elli beş elli beş", "0555 555 55 55"),
        ("sipariş numaranız beş sekiz dört iki bir üç", "sipariş numaranız 584213"),
        ("bir gün üç kişi geldi", "bir gün üç kişi geldi"),
        ("masanın altında kitap var", "masanın altında kitap var"),
        ("onlar yarın gelecek", "onlar yarın gelecek"),
        ("Kırklareli'ye gittik", "Kırklareli'ye gittik"),
        ("yüzünü yıkadı", "yüzünü yıkadı"),
        ("yirmi birinci yüzyılda", "21. yüzyılda"),
        ("on beş mart iki bin yirmi dört", "15 mart 2024"),
        ("üç virgül beş kilogram", "3,5 kilogram"),
        ("bir buçuk saat", "1,5 saat"),
        ("eksi beş derece", "-5 derece"),
        ("elli iki bin seyirci", "52.000 seyirci"),
        ("saat altıda kalkıyor", "saat 6'da kalkıyor"),
        ("on bir yirmide başladı", "11.20'de başladı"),
        ("doksanlı yıllarda", "90'lı yıllarda"),
    ],
)
def test_inverse_normalize(spoken, written):
    assert inverse_normalize(spoken) == written


@pytest.mark.parametrize(
    "written, spoken",
    [
        ("III'ün", "üçün"),
        ("6-6'ya geldi", "altı altıya geldi"),
        ("2023-24 sezonu", "iki bin yirmi üç yirmi dört sezonu"),
    ],
)
def test_roman_suffix_and_descending_ranges(written, spoken):
    assert normalize(written) == spoken


@pytest.mark.parametrize(
    "a, b",
    [
        ("ABD'ye gitti", "abd'ye gitti"),
        ("TBMM'de", "tbmm'de"),
        ("100 TL", "100 tl"),
    ],
)
def test_eval_lowercase_acronyms_match_uppercase(a, b):
    assert normalize_for_eval(a) == normalize_for_eval(b)


def test_eval_leaves_real_words_alone():
    assert normalize_for_eval("ok attı, at bindi") == "ok attı at bindi"


from trnorm.numbers import harmonize_clitic  # noqa: E402


@pytest.mark.parametrize(
    "stem, clitic, out",
    [
        ("lira", "mi", "mı"),
        ("kuruş", "mi", "mu"),
        ("dördüncü", "misin", "müsün"),
        ("dokuz", "misiniz", "musunuz"),
        ("lira", "miydi", "mıydı"),
        ("lira", "miymiş", "mıymış"),
        ("kilogram", "miyse", "mıysa"),
        ("saat", "mı", "mi"),  # front-harmony loanword, like its suffixes
        ("kilovatsaat", "mı", "mi"),  # ...and its compounds
        ("altı", "de", "da"),
        ("beş", "da", "de"),
    ],
)
def test_clitic_harmony(stem, clitic, out):
    assert harmonize_clitic(stem, clitic) == out


@pytest.mark.parametrize(
    "written, spoken",
    [
        # money: the separate question particle follows the spoken currency word
        ("5 TL mi?", "beş lira mı?"),
        ("49,90 TL mi?", "kırk dokuz lira doksan kuruş mu?"),
        ("$100 mü?", "yüz dolar mı?"),
        ("€5 mi?", "beş avro mu?"),
        # percent, time, units, numbers, ordinals
        ("%10 mi?", "yüzde on mu?"),
        ("%5 mı?", "yüzde beş mi?"),
        ("Saat 10:00 mı?", "Saat on mu?"),
        ("21:00 mı?", "yirmi bir mi?"),
        ("6'da mi?", "altıda mı?"),
        ("3 kg mi?", "üç kilogram mı?"),
        ("5 sa mı?", "beş saat mi?"),
        ("5 kWh mi?", "beş kilovatsaat mi?"),
        ("5 kWh de", "beş kilovatsaat de"),
        ("5 kWh'e", "beş kilovatsaate"),
        ("6 mi?", "altı mı?"),
        ("Sen 4. misin?", "Sen dördüncü müsün?"),
        ("26.09.2027 mı?", "yirmi altı Eylül iki bin yirmi yedi mi?"),
        # abbreviations, acronyms, letter labels
        ("Dr. mi?", "doktor mu?"),
        ("THY mi?", "te ha ye mi?"),
        ("SGK mı?", "se ge ka mı?"),
        ("SGK mi?", "se ge ka mı?"),
        ("SGK'ye mi?", "se ge kaya mı?"),
        ("A mı B mi?", "a mı be mi?"),
        # inflected forms
        ("9. misiniz?", "dokuzuncu musunuz?"),
        ("50 TL midir?", "elli lira mıdır?"),
        ("100 TL miydi?", "yüz lira mıydı?"),
        ("10 TL miymiş?", "on lira mıymış?"),
        ("5 kg miyse", "beş kilogram mıysa"),
        ("Bu 5 TL müdür?", "Bu beş lira mıdır?"),
        ('Bu 5 TL müdür?"', 'Bu beş lira mıdır?"'),
        # dA ("too"), and a dA + mI chain
        ("5 TL de yeter.", "beş lira da yeter."),
        ("6 de geldi.", "altı da geldi."),
        ("Dr. de geldi.", "doktor da geldi."),
        ("THY da zam yaptı.", "te ha ye de zam yaptı."),
        ("5 TL de mi?", "beş lira da mı?"),
        ("5 TL mi D blokta?", "beş lira mı de blokta?"),  # only dA + mI chains; "de" here is the letter D
        # all-caps clitic
        ("5 TL Mİ?", "beş lira mı?"),
        ("SGK DA ZAM YAPTI.", "se ge ka da zam yaptı."),
        # untouched: unexpanded or case-only-changed word, other words
        ("Kitap mi?", "Kitap mi?"),
        ("NATO mi?", "nato mi?"),
        ("Ona evet de.", "Ona evet de."),
        ("Ali de geldi.", "Ali de geldi."),
        ("5 TL ki az", "beş lira ki az"),
        ("6 midye", "altı midye"),
        ("5 milyon mu?", "beş milyon mu?"),
        ("İK müdür yardımcısı", "i ka müdür yardımcısı"),
        # the noun "müdür": not before "?", or after a bare count
        ("2 müdür, 3 memur", "iki müdür, üç memur"),
        ("Katılan: 5 müdür.", "Katılan: beş müdür."),
        ("TOPLANTIYA 2 MÜDÜR, 5 UZMAN KATILDI.", "toplantıya iki müdür, beş uzman katıldı."),
        ("Kaç kişi, 5 müdür?", "Kaç kişi, beş müdür?"),
        ("2-3 müdür?", "iki ila üç müdür?"),
    ],
)
def test_clitics_after_expansions(written, spoken):
    assert normalize(written) == spoken


def test_eval_clitic_spelling_matches_spoken():
    assert normalize_for_eval("5 TL mi?") == normalize_for_eval("beş lira mı?") == "beş lira mı"


def test_eval_count_noun_mudur_matches_spoken():
    ref = "Toplantıya 2 müdür, 5 uzman katıldı."
    assert normalize_for_eval(ref) == normalize_for_eval("toplantıya iki müdür beş uzman katıldı")


def test_clitic_marker_never_leaks():
    assert "\ue000" not in normalize("5 TL\ue000 mi, 6 da\ue000 mı?")


# --------------------------------------------------------------------------- EMA Lightning issue reports
# Each case is a reading EMA Lightning users reported as wrong (GitHub #1/#8, HF discussions #5/#9,
# normalizer-tr #4 and PRs #1/#2/#7), checked against trnorm.


@pytest.mark.parametrize(
    "written, spoken",
    [
        # brands and products: Turkish pronunciation, suffix re-agreed with it, initial capital kept
        (
            "akşam iPhone'umdan ChatGPT ile Claude'u karşılaştırdım, sonra YouTube'da bir Netflix fragmanı izledim.",
            "akşam ayfonumdan Çet ci pi ti ile Klodu karşılaştırdım, sonra Yutubda bir Netfliks fragmanı izledim.",
        ),
        ("ChatGPT'ye sordum.", "Çet ci pi tiye sordum."),
        ("Instagram'da gördüm, WhatsApp'tan yazdım.", "İnstagramda gördüm, Vatsaptan yazdım."),
        ("iPhone'larımıza", "ayfonlarımıza"),
        ("YouTube'um", "Yutubum"),
        ("Google'ım", "Gugılım"),
        ("GitHub Copilot'ımızda", "Githab Kopaylıtımızda"),
        ("Instagram'e", "İnstagrama"),  # writer's harmony re-agreed with the spoken stem
        ("Visual Studio Code'un", "Vijuıl stüdyo kodun"),
        ("VS Code", "Vi es kod"),
        ("CHATGPT", "çet ci pi ti"),  # ALL CAPS: lowercase, like spelled acronyms
        ("OPENAI", "opın ey ay"),
        ("OpenAI'a", "Opın ey aya"),
        ("GITHUB", "githab"),
        ("NVIDIA", "envidya"),
        ("EMA Lightning", "Ema laytning"),
        ("Hugging Face", "Haging feys"),
        ("Microsoft Windows", "Maykrosoft Vindovs"),
        ("Python ve JavaScript", "Paytın ve Cavaskript"),
        ("LC Waikiki'den mont aldım.", "el si Vaykikiden mont aldım."),
        ("Coca-Cola'nın", "Koka kolanın"),
        ("McDonald's'ta yedik.", "Mekdonıldsta yedik."),
        ("Disney+ üyeliği", "Disni plas üyeliği"),
        ("Bluetooth'u aç.", "Blututu aç."),
        ("Wi-Fi şifresi nedir?", "Vayfay şifresi nedir?"),
        ("App Store'dan indir.", "Ep stordan indir."),
        ("iOS 18 çıktı.", "ay o es on sekiz çıktı."),
        ("macOS 10.15", "mek o es on nokta on beş"),
        ("Turkcell'den", "Türkselden"),
        ("Tesla ve Intel", "Tesla ve İntel"),  # never a dotless ı from a Latin capital I
        ("youtube'da, instagram'dan", "yutubda, instagramdan"),
        ("Claude-3.5", "Klod üç nokta beş"),
        ("ChatGPT-4o", "Çet ci pi ti dört o"),
        ("iPhone mi?", "ayfon mu?"),  # a separate question particle follows the spoken name
        ("Rekor yıl 2023. Apple ise geride kaldı.", "Rekor yıl iki bin yirmi üç. Epıl ise geride kaldı."),
        # English loanwords of office talk
        ("Toplantının link'ini mail'den forward eder misin?", "Toplantının link'ini meylden forvırd eder misin?"),
        ("Bu feature'ı yarın deploy edeceğiz.", "Bu fiçırı yarın diploy edeceğiz."),
        ("Online meeting'e katıldım.", "Onlayn mitinge katıldım."),
        ("Xbox'ta oynadık.", "Eksboksta oynadık."),
        ("Gemini'ye sordum.", "Ceminaya sordum."),
        # an exact-case key: the Turkish word "gemini" (gemi + n + i) is left alone, in capitals too
        ("Kaptan, gemini terk etme.", "Kaptan, gemini terk etme."),
        ("KAPTAN GEMİNİ TERK ETME", "kaptan gemini terk etme"),
        # a name with an elided article is not a letter with a suffix
        ("D'Artagnan ve L'Occitane", "D'Artagnan ve L'Occitane"),
        # read right by Turkish spelling: left alone
        (
            "Trendyol'dan, Getir'den, Migros'a, TikTok'ta, Samsung, Amazon'dan",
            "Trendyol'dan, Getir'den, Migros'a, TikTok'ta, Samsung, Amazon'dan",
        ),
        # inside addresses: left to the web pass
        ("www.google.com", "ve ve ve nokta google nokta kom"),
        ("ali@gmail.com", "ali et gmail nokta kom"),
    ],
)
def test_foreign_names(written, spoken):
    assert normalize(written) == spoken


def test_eval_keeps_written_names_and_letters():
    # ASR output and references spell names as written; normalize_for_eval also builds STT transcripts
    assert normalize_for_eval("iPhone'umdan Google'a") == "iphoneumdan googlea"
    assert normalize_for_eval("John F. Kennedy") == normalize_for_eval("john f kennedy") == "john f kennedy"


@pytest.mark.parametrize(
    "written, spoken",
    [
        # a Latin I in a spelled or foreign acronym is i, not ı
        ("API ve IP, HDMI", "a pe i ve i pe, he de me i"),
        ("IBM'in", "i be menin"),
        ("ISO 9001", "iso dokuz bin bir"),
        ("ID numarası", "ay di numarası"),
        ("BIOS", "bayos"),
        ("BIST 100 endeksi", "bist yüz endeksi"),
        ("BİST", "bist"),
        ("AI destekli", "ey ay destekli"),
        ("PDF'ten", "pe de eften"),
        ("PDF dosyası", "pe de ef dosyası"),
        ("BMW", "be me ve"),
        # Turkish words in capitals keep ı
        ("ISPARTA'DA HAVA", "ısparta'da hava"),
        ("TIR şoförü", "tır şoförü"),
    ],
)
def test_acronyms_with_latin_i(written, spoken):
    assert normalize(written) == spoken


@pytest.mark.parametrize(
    "written, spoken",
    [
        # a lone V. before a name is an initial; before a lowercase word or a known noun, an ordinal
        ("V. Öztürk", "ve. Öztürk"),
        ("V. Murat", "beşinci Murat"),
        ("V. Daire", "beşinci Daire"),
        ("Yargıtay V. Hukuk Dairesi", "Yargıtay beşinci Hukuk Dairesi"),
        # a lone I., X. is an ordinal unless a title or another initial comes before it
        ("Dr. I. Yılmaz", "doktor ı. Yılmaz"),
        ("Av. X. Demir", "avukat iks. Demir"),
        ("I. Balkan Savaşı", "birinci Balkan Savaşı"),
        ("I. Meşrutiyet döneminde", "birinci Meşrutiyet döneminde"),
        ("I. TBMM döneminde", "birinci te be me me döneminde"),
        ("I. DÜNYA SAVAŞI", "birinci dünya savaşı"),
        ("5. Hukuk Dairesi", "beşinci Hukuk Dairesi"),
        ("I. Dünya Savaşı", "birinci Dünya Savaşı"),
        ("X. Yüzyılda", "onuncu Yüzyılda"),
        ("3. Yüzyıl", "üçüncü Yüzyıl"),
        ("Vitamin C. Sonra gel.", "Vitamin ce. Sonra gel."),
        ("J. K. Rowling", "je. ka. Rowling"),
        ("A.B.D. Başkanı", "A.B.D. Başkanı"),
        # a capital letter with a suffix is the letter; after a regnal name or "Madde" it is the numeral
        ("Bunu X'te paylaştım.", "Bunu ikste paylaştım."),
        ("X'e yazdım.", "ikse yazdım."),
        ("A'dan Z'ye", "adan zeye"),
        ("Louis X'un", "Louis onun"),
        ("Madde V'e göre", "Madde beşe göre"),
        # the written suffix tells which one the writer meant: a numeral ("beşte", "birde") or the letter ("veye")
        ("Ek I'de", "Ek birde"),
        ("Tablo I'de", "Tablo birde"),
        ("Şekil V'te", "Şekil beşte"),
        ("V'ye", "veye"),
        ("X'TE", "ikste"),
        ("O'nun sözü", "O'nun sözü"),
    ],
)
def test_initials_letters_and_roman_numerals(written, spoken):
    assert normalize(written) == spoken


@pytest.mark.parametrize(
    "written, spoken",
    [
        # address numbers are never dates or fractions
        (
            "Adresiniz Atatürk Mah. Cumhuriyet Cad. No:12/4 Kat:3 Daire:7 Kadıköy/İstanbul.",
            "Adresiniz Atatürk mahallesi Cumhuriyet caddesi numara on iki bölü dört Kat üç Daire yedi Kadıköy, İstanbul.",
        ),
        ("No:3/1", "numara üç bölü bir"),
        ("No: 5/2", "numara: beş bölü iki"),
        ("Kapı no 15/3", "Kapı no on beş bölü üç"),
        ("10/2 nolu daire", "on bölü iki nolu daire"),
        ("Daire 12/4", "Daire on iki bölü dört"),
        ("3/4'ü", "dörtte üçü"),
        ("15/03'te", "on beş Martta"),
        # a postcode inside an address is read digit by digit
        (
            "Teslimat adresi: Bağdat Cd. 245. Sk. No: 5 D: 2, 34728 Kadıköy.",
            "Teslimat adresi: Bağdat caddesi iki yüz kırk beşinci sokak numara: beş daire iki, "
            "üç dört yedi iki sekiz Kadıköy.",
        ),
        ("Stadyumda 34728 kişi vardı.", "Stadyumda otuz dört bin yedi yüz yirmi sekiz kişi vardı."),
        (
            "Atatürk Mah. Gül Sok. No:5 D:3 34728 Kadıköy/İstanbul",
            "Atatürk mahallesi Gül sokak numara beş daire üç üç dört yedi iki sekiz Kadıköy, İstanbul",
        ),
        ("Gül Sok. 34728 Kadıköy", "Gül sokak üç dört yedi iki sekiz Kadıköy"),
        # ...but a count is not a postcode, even with an address-like word nearby or in capitals
        ("MİTİNGE KATILAN 50000 KİŞİ", "mitinge katılan elli bin kişi"),
        ("Mahallede 15000 Suriyeli yaşıyor.", "Mahallede on beş bin Suriyeli yaşıyor."),
        ("Bu kategoride 25000 Samsung telefon var.", "Bu kategoride yirmi beş bin Samsung telefon var."),
        ("İSTİKLAL CADDESİ 50000 KİŞİYLE DOLDU", "istiklal caddesi elli bin kişiyle doldu"),
        ("Moda Caddesi 5/2", "Moda Caddesi beş bölü iki"),
        # IBAN: four-digit groups digit by digit
        (
            "IBAN numaranız TR12 0006 4000 0011 2345 6789 01.",
            "iban numaranız te re on iki, sıfır sıfır sıfır altı, dört sıfır sıfır sıfır, sıfır sıfır bir bir, "
            "iki üç dört beş, altı yedi sekiz dokuz, sıfır bir.",
        ),
        # x, ×, * between numbers
        (
            "Aylık taksitiniz 12 x 833,25 TL şeklinde hesaplanmıştır.",
            "Aylık taksitiniz on iki çarpı sekiz yüz otuz üç lira yirmi beş kuruş şeklinde hesaplanmıştır.",
        ),
        ("12*2*833,25 TL", "on iki çarpı iki çarpı sekiz yüz otuz üç lira yirmi beş kuruş"),
        ("1.234 x 2", "bin iki yüz otuz dört çarpı iki"),
        ("3×4", "üç çarpı dört"),
        ("4*5=20", "dört çarpı beş eşittir yirmi"),
        # year-first dates
        (
            "Üyeliğiniz 2026/12/31 tarihine kadar geçerlidir.",
            "Üyeliğiniz otuz bir Aralık iki bin yirmi altı tarihine kadar geçerlidir.",
        ),
        ("2026/01/05'te", "beş Ocak iki bin yirmi altıda"),
        ("2026.12.31", "otuz bir Aralık iki bin yirmi altı"),
        # -ki and the possessive after kuruş
        ("2026'dakidir", "iki bin yirmi altıdakidir"),
        ("1990'lardakiler", "bin dokuz yüz doksanlardakiler"),
        ("5'tekiler", "beştekiler"),
        ("25,50 TL'm", "yirmi beş lira elli kuruşum"),
        ("12,40 TL'n", "on iki lira kırk kuruşun"),
        ("25 TL'm", "yirmi beş liram"),
        # ranges: negative ends, "ile" before arası/arasında
        ("-10 - 5 derece", "eksi on ila beş derece"),
        ("-5 - -2 gün", "eksi beş ila eksi iki gün"),
        ("10 - 15 kişi", "on ila on beş kişi"),
        (
            "Servis ekibimiz 13:00-17:00 arasında adresinize gelecek.",
            "Servis ekibimiz on üç ile on yedi arasında adresinize gelecek.",
        ),
        ("10-15 TL arasında", "on ile on beş lira arasında"),
        ("%10-20 arası", "yüzde on ile yirmi arası"),
        ("4-5 dakika", "dört ila beş dakika"),
        # web: "/" is eğik çizgi
        (
            "Detaylar www.ema.com.tr/abc adresinde.",
            "Detaylar ve ve ve nokta ema nokta kom nokta te re eğik çizgi abc adresinde.",
        ),
        # emoji are not read; powers are
        ("Ahmet'e 🙂 dedim.", "Ahmet'e dedim."),
        ("Süper ❤️👍", "Süper"),
        ("Merhaba 👋 nasılsın?", "Merhaba nasılsın?"),
        ("x² + y²", "iks kare artı ye kare"),
        ("a² + b² = c²", "a kare artı be kare eşittir ce kare"),
        ("10²", "on kare"),
        ("m² başına", "metrekare başına"),
        ("45 m² daire", "kırk beş metrekare daire"),
        # versions
        ("3.x sürümü", "üç nokta iks sürümü"),
        ("1.2.3 sürümü", "bir nokta iki nokta üç sürümü"),
        ("192.168.1.1", "yüz doksan iki nokta yüz altmış sekiz nokta bir nokta bir"),
        ("GPT-3.5", "ge pe te üç nokta beş"),
        ("1.250.000 kişi", "bir milyon iki yüz elli bin kişi"),
    ],
)
def test_notation_reported_for_ema(written, spoken):
    assert normalize(written) == spoken


@pytest.mark.parametrize(
    "text",
    ["2026.12.31", "Claude-3.5", "3.x", "802.11b", "2x hızlı", "6inçlik", "ChatGPT-4o", "1975yılında", "8.6.9..Dabul", "x² + 10³"],
)
def test_no_digit_is_ever_left(text):
    assert not any(ch.isdecimal() for ch in normalize(text)), normalize(text)


@pytest.mark.parametrize(
    "written, spoken",
    [
        # an abbreviation's period before a number is a pause, not "nokta"; the suffix stays on the number
        ("Tablo.3'te", "Tablo üçte"),
        ("Fig.2'de", "Fig ikide"),
        ("Kat.3", "Kat üç"),
        ("md.5'inci", "md beşinci"),
    ],
)
def test_leftover_digits_after_an_abbreviation(written, spoken):
    assert normalize(written) == spoken


@pytest.mark.parametrize(
    "written, spoken",
    [
        # a word of an ALL-CAPS headline is read as a word when it is shaped like one, clusters included
        ("DÖRT KİŞİ HAYATINI KAYBETTİ", "dört kişi hayatını kaybetti"),
        ("EKONOMİDE KRİZ BÜYÜYOR", "ekonomide kriz büyüyor"),
        ("BU BİR TÜRK MALIDIR", "bu bir türk malıdır"),
        ("GENÇ YETENEKLER ENSTRÜMAN ÇALDI", "genç yetenekler enstrüman çaldı"),
        ("TREN KAZASINDA ÜST ÜSTE", "tren kazasında üst üste"),
        ("SPOR TOTO SÜPER LİG", "spor toto süper lig"),
        # spelled-out numbers do not make a headline mixed-case
        ("SAAT 14:30'DA BAŞLAYACAK DÖRT ETKİNLİK", "saat on dört otuzda başlayacak dört etkinlik"),
        # acronyms are still spelled
        ("AVM YANGINI KONTROL ALTINDA", "a ve me yangını kontrol altında"),
        ("CHP VE AKP ARASINDA KRİZ", "ce he pe ve a ka pe arasında kriz"),
        ("EFT LİMİTİ ARTTI", "e fe te limiti arttı"),
        ("İETT OTOBÜSÜ KAZA YAPTI", "i e te te otobüsü kaza yaptı"),
        ("DHA MUHABİRİ YARALANDI", "de he a muhabiri yaralandı"),
    ],
)
def test_caps_headline_words(written, spoken):
    assert normalize(written) == spoken


@pytest.mark.parametrize(
    "written, spoken",
    [
        # an identifier cue is a whole word or the word with suffixes, never the start of another word
        ("Normalde 25000 kişi gelir.", "Normalde yirmi beş bin kişi gelir."),
        ("Televizyonda 15000 kişi izledi.", "Televizyonda on beş bin kişi izledi."),
        ("Hatay'da 15000 kişi yaşıyor.", "Hatay'da on beş bin kişi yaşıyor."),
        ("Kodlama 2026 yılında başladı.", "Kodlama iki bin yirmi altı yılında başladı."),
        ("Onaylanan 2500 başvuru var.", "Onaylanan iki bin beş yüz başvuru var."),
        # a count is not an identifier
        ("Takipçi sayısı 25000'i geçti.", "Takipçi sayısı yirmi beş bini geçti."),
        ("Kart limiti 25000 lira.", "Kart limiti yirmi beş bin lira."),
        ("Telefonla 15000 kişiye ulaştık.", "Telefonla on beş bin kişiye ulaştık."),
        # identifiers and codes
        ("Hesabınız 12345678", "Hesabınız bir iki üç dört beş altı yedi sekiz"),
        ("Takip kodunuz 123456", "Takip kodunuz bir iki üç dört beş altı"),
        ("Onay kodu 4821", "Onay kodu dört sekiz iki bir"),
        ("PIN 4821", "pin dört sekiz iki bir"),
    ],
)
def test_identifier_or_count(written, spoken):
    assert normalize(written) == spoken


def test_superscript_digits_do_not_crash_the_web_pass():
    assert normalize("abc@1.2.3") and normalize(":¹3.5abc@1.2.3")
