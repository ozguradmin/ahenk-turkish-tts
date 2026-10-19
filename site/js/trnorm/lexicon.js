// trnorm/lexicon.py in JavaScript: the word lists used by the normalizer, entry for entry.
// A Python dict is a Map (insertion order kept, and no inherited keys: a word like "constructor" is not an entry),
// a set is a Set, a tuple or list an array.

const dict = (entries) => new Map(Object.entries(entries)); // no key here looks like an array index, so order holds

export const MONTHS = [
  '',
  'Ocak',
  'Şubat',
  'Mart',
  'Nisan',
  'Mayıs',
  'Haziran',
  'Temmuz',
  'Ağustos',
  'Eylül',
  'Ekim',
  'Kasım',
  'Aralık',
];

// Letter names used when spelling acronyms (TBMM -> te be me me).
export const LETTER_NAMES = dict({
  'A': 'a', 'B': 'be', 'C': 'ce', 'Ç': 'çe', 'D': 'de', 'E': 'e', 'F': 'fe',
  'G': 'ge', 'Ğ': 'yumuşak ge', 'H': 'he', 'I': 'ı', 'İ': 'i', 'J': 'je',
  'K': 'ka', 'L': 'le', 'M': 'me', 'N': 'ne', 'O': 'o', 'Ö': 'ö', 'P': 'pe',
  'Q': 'kü', 'R': 're', 'S': 'se', 'Ş': 'şe', 'T': 'te', 'U': 'u', 'Ü': 'ü',
  'V': 've', 'W': 'çift ve', 'X': 'iks', 'Y': 'ye', 'Z': 'ze',
  'Â': 'a', 'Î': 'i', 'Û': 'u',
});

// Acronyms whose usual reading differs from plain spelling / plain word reading.
export const ACRONYMS = dict({
  'THY': 'te ha ye',
  'HDP': 'ha de pe',
  'AKP': 'a ka pe',
  'AK': 'ak',
  'CHP': 'ce he pe',
  'MHP': 'me he pe',
  'NTV': 'en te ve',
  'CNN': 'si en en',
  'BBC': 'bi bi si',
  'CEO': 'si i o',
  'DİSK': 'disk',
  'TOGG': 'tog',
  'WC': 've ce',
  'COVID': 'kovid',
  'UNICEF': 'yunisef',
  'RTÜK': 're te ü ka',
  'İTÜ': 'i te ü',
  'YTÜ': 'ye te ü',
  'KTÜ': 'ka te ü',
  'ODTÜ': 'o de te ü',
  'MS': 'milattan sonra',
  'MÖ': 'milattan önce',
  'TL': 'te le',
  'IMF': 'ay em ef',
  'AI': 'ey ay',
  'PhD': 'pi eyç di',
  'OK': 'okey',
  'SOS': 'es o es',
  'FBI': 'ef bi ay',
  'CIA': 'si ay ey',
  'UFO': 'ufo',
  'LGBT': 'le ge be te',
  'WWE': 'dabılyu dabılyu i',
  'HTML': 'he te me le',
  'CSS': 'ce se se',
  'API': 'a pe i',
  'URL': 'u re le',
  'VIP': 'vip',
  'IBAN': 'iban',
  'SMS': 'esemes',
  'FAST': 'fast',
  'SIM': 'sim',
  'POS': 'pos',
  'MHRS': 'me he re se',
  'PIN': 'pin',
  'OTP': 'o te pe',
  'IKEA': 'ikea',
  'QR': 'kü ar',
  'LC': 'el si',
  'ISBN': 'i se be ne',
  'IQ': 'ay kü',
  'IP': 'i pe',
  'IT': 'ay ti',
  'ID': 'ay di',
  'ISO': 'iso',
  'BIOS': 'bayos',
  'BIST': 'bist',
  'BİST': 'bist',
  'BMW': 'be me ve',
  'PDF': 'pe de ef',
});

// Foreign names (brands, apps, products) and English loanwords whose spelling misleads a Turkish
// reader, mapped to their usual Turkish pronunciation in lowercase Turkish spelling. normalize()
// replaces them (the eval form keeps the written word, as ASR output and references do), re-agrees
// an apostrophe suffix with the pronunciation and keeps a written initial capital:
// "iPhone'umdan" -> "ayfonumdan", "Google'da" -> "Gugılda", "Instagram" -> "İnstagram".
// Matching is case-sensitive; besides each key, its ALL-CAPS, all-lowercase and Capitalized forms are
// matched too ("YOUTUBE", "youtube", "Youtube"), except for the keys in FOREIGN_NAMES_EXACT_CASE.
// Only names a Turkish reader gets wrong belong here: Trendyol, Getir, Migros, TikTok, Samsung, Amazon,
// Tesla... read right by Turkish spelling and are left alone. Longest key wins ("Visual Studio Code").
export const FOREIGN_NAMES = dict({
  // AI
  'ChatGPT': 'çet ci pi ti',
  'OpenAI': 'opın ey ay',
  'Claude': 'klod',
  'Gemini': 'ceminay',
  'Anthropic': 'entropik',
  'Copilot': 'kopaylıt',
  'Codex': 'kodeks',
  'DeepSeek': 'dipsik',
  'Midjourney': 'midcörni',
  'Perplexity': 'pörpleksiti',
  'Hugging Face': 'haging feys',
  'EMA Lightning': 'ema laytning',
  'ema-lightning': 'ema laytning',
  'Lightning': 'laytning',
  // software, developer tools
  'GitHub': 'githab',
  'GitLab': 'gitleb',
  'Docker': 'dokır',
  'Kubernetes': 'kubırnetis',
  'Python': 'paytın',
  'JavaScript': 'cavaskript',
  'TypeScript': 'taypskript',
  'Java': 'cava',
  'React': 'riekt',
  'Rust': 'rast',
  'Linux': 'linuks',
  'Visual Studio Code': 'vijuıl stüdyo kod',
  'Visual Studio': 'vijuıl stüdyo',
  'VS Code': 'vi es kod',
  'Stack Overflow': 'stek ovırflov',
  'Jira': 'cira',
  'Notion': 'noşın',
  'Slack': 'slek',
  'Zoom': 'zum',
  'Microsoft': 'maykrosoft',
  'Windows': 'vindovs',
  'Office': 'ofis',
  'Excel': 'eksel',
  'PowerPoint': 'pavırpoynt',
  'Outlook': 'autluk',
  'Teams': 'tims',
  'OneDrive': 'vandrayv',
  'Xbox': 'eksboks',
  'PlayStation': 'pleysteyşın',
  'Google': 'gugıl',
  'Gmail': 'cimeyl',
  'Chrome': 'krom',
  'Firefox': 'fayırfoks',
  'Google Play Store': 'gugıl pley stor',
  'Google Play': 'gugıl pley',
  'Play Store': 'pley stor',
  'App Store': 'ep stor',
  // devices, platforms
  'Apple': 'epıl',
  'iPhone': 'ayfon',
  'iPad': 'aypad',
  'iCloud': 'ayklaud',
  'iOS': 'ay o es',
  'macOS': 'mek o es',
  'MacBook': 'mekbuk',
  'AirPods': 'erpods',
  'Galaxy': 'geleksi',
  'Huawei': 'huavey',
  'Xiaomi': 'şaomi',
  'NVIDIA': 'envidya',
  'Intel': 'intel',  // "Intel": a Latin capital I is not the Turkish ı
  'Bluetooth': 'blutut',
  'Wi-Fi': 'vayfay',
  'WiFi': 'vayfay',
  'Max': 'maks',
  'Plus': 'plas',
  // social, media, services
  'YouTube': 'yutub',
  'WhatsApp': 'vatsap',
  'Instagram': 'instagram',  // Latin capital I: "ınstagram" otherwise
  'Facebook': 'feysbuk',
  'Twitter': 'tivitır',
  'LinkedIn': 'linkedin',
  'Snapchat': 'snepçet',
  'Pinterest': 'pintırest',
  'Netflix': 'netfliks',
  'Spotify': 'spotifay',
  'Disney+': 'disni plas',
  'Disney': 'disni',
  'PayPal': 'peypal',
  'Mastercard': 'mastırkard',
  'Airbnb': 'erbienbi',
  'Turkcell': 'türksel',
  'Vodafone': 'vodafon',
  // food, retail, cars
  'Coca-Cola': 'koka kola',
  'Coca Cola': 'koka kola',
  'McDonald\'s': 'mekdonılds',
  'McDonalds': 'mekdonılds',
  'Burger King': 'börgır king',
  'Starbucks': 'starbaks',
  'Nescafé': 'neskafe',
  'Nescafe': 'neskafe',
  'Nike': 'nayki',
  'Waikiki': 'vaykiki',  // "LC Waikiki": LC is an acronym
  'Mercedes-Benz': 'mersedes benz',
  'Mercedes': 'mersedes',
  'Volkswagen': 'folksvagen',
  'Renault': 'reno',
  'Peugeot': 'pejo',
  'Citroën': 'sitroen',
  'Citroen': 'sitroen',
  'Chevrolet': 'şevrole',
  'Porsche': 'porşe',
  'Jeep': 'cip',
  'Range Rover': 'reync rovır',
  'Land Rover': 'lend rovır',
  // English loanwords of office and internet talk ("Online meeting'e", "feature'ı deploy etmek")
  'online': 'onlayn',
  'offline': 'oflayn',
  'meeting': 'miting',
  'e-mail': 'imeyl',
  'email': 'imeyl',
  'mail': 'meyl',
  'forward': 'forvırd',
  'feature': 'fiçır',
  'deploy': 'diploy',
  'update': 'apdeyt',
  'upgrade': 'apgreyd',
  'download': 'davnlod',
  'upload': 'aplod',
  'feedback': 'fidbek',
  'deadline': 'dedlayn',
  'brief': 'brif',
  'team': 'tim',
  'weekend': 'vikend',
  'laptop': 'leptop',
  'software': 'softver',
  'hardware': 'hardver',
  'startup': 'startap',
  'influencer': 'influensır',
  'podcast': 'podkast',
  'streaming': 'striming',
  'story': 'stori',
  'follower': 'folovır',
  'password': 'pasvord',
  'browser': 'bravzır',
  'server': 'sörvır',
  'backend': 'bekend',
  'debug': 'dibag',
  'release': 'rilis',
  'commit': 'komit',
  'chat': 'çet',
  'gaming': 'geyming',
  'cloud': 'klaud',
});
// keys matched only as written and in Latin ALL CAPS ("GEMINI", not the Turkish "GEMİNİ"): their lowercase
// form is an ordinary English word, or a Turkish one ("gemini terk etme": gemi + n + i)
export const FOREIGN_NAMES_EXACT_CASE = new Set([
  'React', 'Rust', 'Java', 'Office', 'Teams', 'Plus', 'Lightning', 'Gemini',
]);

// Short Turkish words that appear in ALL-CAPS headlines and must not be spelled.
export const CAPS_WORDS = new Set([
  'VE', 'DE', 'DA', 'BU', 'ŞU', 'Mİ', 'MI', 'MU', 'MÜ', 'NE', 'EN', 'İLE',
  'Kİ', 'YA', 'AMA', 'BİR', 'İKİ', 'ÜÇ', 'SON', 'İLK', 'HER', 'HİÇ', 'YOK',
  'VAR', 'ÇOK', 'AZ', 'İÇ', 'DIŞ', 'BEN', 'SEN', 'BİZ', 'SİZ', 'ŞOK', 'ÖN',
  'AÇ', 'AT', 'AL', 'VER', 'GEL', 'GİT', 'OL', 'YAP', 'UN', 'SU', 'ON', 'AY',
  'YIL', 'GÜN', 'İŞ', 'EV', 'EL', 'AN', 'ÖZ',
]);

// Titles that are also written in capitals or without the spaces ("DR. ÖĞR. ÜYESİ", "DR.ÖĞR.ÜYESİ", "Prof.Dr."):
// those forms are read the same (normalizer.js _ABBR_FORMS). Only titles: "AV." or "SOK." in capitals could be other words.
export const TITLE_ABBREVIATIONS = [
  'Prof. Dr.', 'Doç. Dr.', 'Yrd. Doç. Dr.', 'Dr. Öğr. Üyesi', 'Öğr. Gör.', 'Arş. Gör.', 'Uzm. Dr.', 'Op. Dr.',
  'Yrd. Doç.', 'Prof.', 'Doç.', 'Dr.', 'Uzm.', 'Müh.',
];

// Dotted / colon abbreviations (case-sensitive). Longest match wins.
export const ABBREVIATIONS = dict({
  'Prof. Dr.': 'profesör doktor',
  'Doç. Dr.': 'doçent doktor',
  'Yrd. Doç. Dr.': 'yardımcı doçent doktor',
  'Dr. Öğr. Üyesi': 'doktor öğretim üyesi',
  'Öğr. Gör.': 'öğretim görevlisi',
  'Arş. Gör.': 'araştırma görevlisi',
  'Uzm. Dr.': 'uzman doktor',
  'Op. Dr.': 'operatör doktor',
  'Yrd. Doç.': 'yardımcı doçent',
  'Ltd. Şti.': 'limitet şirketi',
  'San. ve Tic.': 'sanayi ve ticaret',
  'Tic. Ltd. Şti.': 'ticaret limitet şirketi',
  'T.C.': 'te ce',
  'A.Ş.': 'anonim şirketi',
  'M.Ö.': 'milattan önce',
  'M.S.': 'milattan sonra',
  'Prof.': 'profesör',
  'Doç.': 'doçent',
  'Dr.': 'doktor',
  'Av.': 'avukat',
  'Uzm.': 'uzman',
  'Müh.': 'mühendis',
  'Hz.': 'hazreti',
  'Sn.': 'sayın',
  'Alb.': 'albay',
  'Yzb.': 'yüzbaşı',
  'Bşk.': 'başkanı',
  'Gn.': 'genel',
  'Md.': 'müdürü',
  'Mah.': 'mahallesi',
  'Mh.': 'mahallesi',
  'Cad.': 'caddesi',
  'Cd.': 'caddesi',
  'Sok.': 'sokak',
  'Sk.': 'sokak',
  'Bulv.': 'bulvarı',
  'Blv.': 'bulvarı',
  'Apt.': 'apartmanı',
  'Ap.': 'apartmanı',
  'Şti.': 'şirketi',
  'Ltd.': 'limitet',
  'Tic.': 'ticaret',
  'San.': 'sanayi',
  'vb.': 've benzeri',
  'vs.': 'vesaire',
  'vd.': 've diğerleri',
  'bkz.': 'bakınız',
  'Bkz.': 'bakınız',
  'örn.': 'örneğin',
  'Örn.': 'örneğin',
  'yy.': 'yüzyıl',
  'yak.': 'yaklaşık',
  'ort.': 'ortalama',
  'maks.': 'maksimum',
  'Tel.': 'telefon',
  'No.': 'numara',
  'Nu.': 'numara',
  'No:': 'numara:',
  'Tel:': 'telefon:',
  'Faks:': 'faks:',
  'Fax:': 'faks:',
  'GSM:': 'ge se me:',
});
// abbreviations that can also end a sentence (their period doubles as a full stop)
export const SENTENCE_FINAL_ABBR = new Set(['vb.', 'vs.', 'vd.', 'yy.', 'A.Ş.', 'Şti.', 'Ltd.', 'M.Ö.', 'M.S.']);

export const CURRENCY_WORDS = dict({
  'TL': 'lira', '₺': 'lira', 'TRY': 'lira', 'lira': 'lira',
  '$': 'dolar', 'USD': 'dolar', 'dolar': 'dolar',
  '€': 'avro', 'EUR': 'avro', 'avro': 'avro', 'euro': 'avro', 'Euro': 'avro',
  '£': 'sterlin', 'GBP': 'sterlin', 'sterlin': 'sterlin',
  '¥': 'yen', 'JPY': 'yen',
  '₽': 'ruble', 'RUB': 'ruble',
  'CHF': 'isviçre frangı',
});
// currency -> subunit name used for two-digit decimals
export const CURRENCY_SUBUNIT = dict({ 'lira': 'kuruş', 'dolar': 'sent', 'avro': 'sent', 'sterlin': 'peni' });

export const UNITS = dict({
  // length / area / volume
  'km': 'kilometre', 'm': 'metre', 'cm': 'santimetre', 'mm': 'milimetre',
  'µm': 'mikrometre', 'μm': 'mikrometre', 'nm': 'nanometre',
  'km²': 'kilometrekare', 'km2': 'kilometrekare', 'm²': 'metrekare', 'm2': 'metrekare',
  'cm²': 'santimetrekare', 'm³': 'metreküp', 'm3': 'metreküp', 'cm³': 'santimetreküp',
  'cc': 'santimetreküp', 'ha': 'hektar', 'dönüm': 'dönüm',
  // mass
  'kg': 'kilogram', 'g': 'gram', 'gr': 'gram', 'mg': 'miligram', 'µg': 'mikrogram',
  'mcg': 'mikrogram', 'ton': 'ton',
  // volume
  'lt': 'litre', 'l': 'litre', 'L': 'litre', 'ml': 'mililitre', 'mL': 'mililitre', 'cl': 'santilitre',
  // time
  'sn': 'saniye', 'sn.': 'saniye', 'dk': 'dakika', 'dk.': 'dakika', 'dak': 'dakika',
  'sa': 'saat', 'ms': 'milisaniye',
  // energy / electricity / frequency
  'kWh': 'kilovatsaat', 'kW': 'kilovat', 'MW': 'megavat', 'GW': 'gigavat', 'W': 'vat',
  'V': 'volt', 'mV': 'milivolt', 'kV': 'kilovolt', 'mA': 'miliamper', 'mAh': 'miliamper saat',
  'Hz': 'hertz', 'kHz': 'kilohertz', 'MHz': 'megahertz', 'GHz': 'gigahertz',
  'dB': 'desibel', 'kcal': 'kilokalori', 'cal': 'kalori', 'bar': 'bar', 'rpm': 'devir',
  // data
  'KB': 'kilobayt', 'kB': 'kilobayt', 'MB': 'megabayt', 'GB': 'gigabayt', 'TB': 'terabayt',
  'Mbps': 'megabit', 'Gbps': 'gigabit', 'px': 'piksel', 'MP': 'megapiksel',
  'mmHg': 'milimetre cıva',
});

// speed units are read "saatte 120 kilometre"
export const SPEED_UNITS = dict({
  'km/sa': ['saatte', 'kilometre'],
  'km/s': ['saatte', 'kilometre'],
  'km/h': ['saatte', 'kilometre'],
  'm/sn': ['saniyede', 'metre'],
  'm/s': ['saniyede', 'metre'],
});

// words that, among the last three before a 5+ digit number, mark it as an identifier (read in digit
// groups). A cue matches as a whole word or with nominal suffixes ("numaranız", "kodu", "hesabınız"), never
// as the start of another word ("normalde", "televizyonda", "kartal", "takipçi")
export const ID_CONTEXT = [
  'numara', 'no', 'sipariş', 'takip', 'kod', 'hesap', 'hesab', 'kimlik', 'kimliğ',
  'tc', 'iban', 'ıban', 'telefon', 'tel', 'gsm', 'hat', 'hatt', 'fatura', 'referans', 'pnr',
  'seri', 'şifre', 'pin', 'pın', 'posta', 'plaka', 'müşteri', 'abone', 'sicil', 'dosya',
  'poliçe', 'kart', 'başvuru', 'rezervasyon', 'barkod', 'ürün',
];
// cues that match only as whole words: "hat" (not "hata", "hatta", "Hatay"), "seri" (not "serin")
export const ID_EXACT = new Set(['no', 'tc', 'tel', 'gsm', 'hat', 'pin', 'pın', 'pnr', 'seri', 'otp']);
// ...but a count is not an identifier: the word right before it names a quantity ("Müşteri sayısı 25000"),
// or the word after it is a unit or a counted noun ("Telefonla 15000 kişiye", "Vergi borcu 25000 lira")
export const QUANTITY_BEFORE = new Set([
  'sayı', 'sayısı', 'adet', 'adedi', 'tutar', 'tutarı', 'borç', 'borcu', 'limit', 'limiti', 'miktar',
  'miktarı', 'bedel', 'bedeli', 'ücret', 'ücreti', 'fiyat', 'fiyatı', 'bakiye', 'bakiyesi', 'toplam',
  'toplamı', 'değer', 'değeri',
]);
export const QUANTITY_AFTER = [
  'kişi', 'adet', 'tane', 'lira', 'kuruş', 'dolar', 'avro', 'euro', 'sterlin', 'yıl', 'gün', 'hafta', 'saat',
  'dakika', 'saniye', 'metre', 'kilometre', 'kilo', 'kilogram', 'gram', 'ton', 'litre', 'kez', 'defa',
  'personel', 'çalışan', 'öğrenci', 'üye', 'takipçi', 'abone', 'başvuru', 'konut', 'araç', 'bit', 'bayt',
];

// capitalized words after which "N." is still an ordinal ("3. Lig", "II. Dünya Savaşı")
export const ORDINAL_CAP_NOUNS = new Set([
  'Lig', 'Dünya', 'Ordu', 'Kolordu', 'Tümen', 'Cadde', 'Sokak', 'Bölge', 'Sınıf', 'Kat',
  'Murat', 'Selim', 'Mehmet', 'Ahmet', 'Mahmut', 'Abdülhamit', 'Abdülhamid', 'Bayezid',
  'Beyazıt', 'Mustafa', 'Osman', 'Süleyman', 'Dönem', 'Yasama', 'Meclis', 'Ulusal',
  'Uluslararası', 'Asliye', 'Ağır', 'İdare', 'Sulh', 'İcra', 'Noter', 'Etap', 'Kısım',
  'Bölüm', 'Ünite', 'Sezon', 'Hafta', 'Tur', 'Yıl', 'Baskı', 'Cilt', 'Kitap', 'Madde',
  'Fıkra', 'Bent', 'Ordusu', 'Filo', 'Tugay', 'Alay', 'Lige', 'Ligde', 'Dünya\'da',
  'Kongre', 'Kurultay', 'Olağan', 'Olağanüstü', 'Festival', 'Uluslararası', 'Yüzyıl',
  'Murad', 'Mehmed', 'Ahmed', 'Mahmud', 'Orhan', 'Abdülaziz', 'Abdülmecid', 'Abdülmecit',
  'Louis', 'Charles', 'Henry', 'George', 'Edward', 'Elizabeth', 'Wilhelm', 'Friedrich',
  'Frederick', 'Ferdinand', 'Karl', 'Carlos', 'Felipe', 'Napolyon', 'Petro', 'Katerina',
  'Nikola', 'Aleksandr', 'Konstantin', 'Leo', 'Pius', 'Paul', 'John', 'Juan',
  'Daire', 'Derece', 'Kademe', 'Grup', 'Hukuk', 'Ceza', 'Haçlı', 'Kalkınma',
]);

// inflected forms of ORDINAL_CAP_NOUNS: "XX. Yüzyılda", "3. Ligde"
export const ORDINAL_CAP_PREFIXES = ['Yüzyıl', 'Dünya', 'Lig'];

// titles (as written or as read out) after which a lone I. or X. is an initial, not an ordinal:
// "Dr. I. Yılmaz", "Av. X. Demir" (a lone V. is an initial unless a lowercase word or an
// ORDINAL_CAP_NOUNS word follows: "V. Öztürk", "V. Murat")
export const INITIAL_TITLES = [
  'dr', 'doktor', 'prof', 'profesör', 'doç', 'doçent', 'av', 'avukat', 'op', 'operatör', 'uzm', 'uzman',
  'müh', 'mühendis', 'sn', 'sayın', 'bay', 'bayan', 'mr', 'mrs', 'ms', 'görevlisi', 'üyesi', 'hazreti',
];

// accept these lowercase suffixes on a number even without an apostrophe ("1990larda")
export const BARE_NUMBER_SUFFIXES = new Set([
  'de', 'da', 'te', 'ta', 'den', 'dan', 'ten', 'tan', 'deki', 'daki', 'teki', 'taki',
  'e', 'a', 'ye', 'ya', 'i', 'ı', 'u', 'ü', 'yi', 'yı', 'yu', 'yü',
  'in', 'ın', 'un', 'ün', 'nin', 'nın', 'nun', 'nün',
  'le', 'la', 'yle', 'yla', 'ler', 'lar', 'lerde', 'larda', 'lerden', 'lardan',
  'lere', 'lara', 'leri', 'ları', 'lerin', 'ların', 'li', 'lı', 'lu', 'lü',
  'lik', 'lık', 'luk', 'lük', 'er', 'ar', 'şer', 'şar',
  'inci', 'ıncı', 'uncu', 'üncü', 'nci', 'ncı', 'ncu', 'ncü',
]);

// words right before "N/M" that make it an address number ("No: 12/4", "Daire 3/1", "kapı no 15/3"),
// read "on iki bölü dört" rather than a date or a fraction; also "N/M nolu"
export const ADDRESS_NUMBER_CONTEXT = new Set([
  'no', 'nu', 'numara', 'numarası', 'numaralı', 'kapı', 'daire', 'dairesi', 'kat', 'blok', 'bina',
  'apt', 'apartman', 'apartmanı', 'cadde', 'caddesi', 'sokak', 'sokağı', 'bulvar', 'bulvarı',
]);
// whole words that mark an address (not "katılan", "kategori", "mahallede", "postaneye"): a bare 5-digit
// number in one, before a place name, is a postcode read digit by digit ("..., 34728 Kadıköy")
export const ADDRESS_CONTEXT = new Set([
  'adres', 'adresi', 'adresim', 'adresimiz', 'adresiniz', 'mahalle', 'mahallesi', 'mah', 'mh', 'cadde',
  'caddesi', 'cad', 'cd', 'sokak', 'sokağı', 'sok', 'sk', 'bulvar', 'bulvarı', 'bulv', 'blv', 'no', 'nu',
  'numara', 'numarası', 'kapı', 'daire', 'dairesi', 'kat', 'apartman', 'apartmanı', 'apt', 'ap', 'blok',
  'site', 'sitesi', 'posta',
]);

export const SCORE_CONTEXT = [
  'yendi', 'mağlup', 'skor', 'skorla', 'galip', 'berabere', 'maç', 'maçı', 'maçta',
  'set', 'seti', 'galibiyet', 'yenilgi', 'kazandı', 'kaybetti', 'durumunda', 'farkla',
];

export const TIME_CONTEXT = [
  'saat', 'sabah', 'akşam', 'öğle', 'öğlen', 'öğleden', 'gece', 'itibaren', 'kadar',
  'sularında', 'civarında', 'arası', 'arasında', 'randevu', 'seans',
];

// one-time codes and PINs are read digit by digit even when short ("kodunuz 4821", "Onay kodu 4821");
// matched like ID_CONTEXT, so "Kodlama 2026 yılında" and "Onaylanan 2500 başvuru" stay numbers
export const CODE_CONTEXT = ['kod', 'şifre', 'pin', 'pın', 'otp', 'doğrulama']; // "pın": "PIN" in Turkish lowercase

// Acronyms that are spelled out when read. normalize_for_eval() also recognizes their
// lowercase spelling ("abd", "tbmm") so that lowercase ASR output and references score
// the same as "ABD", "TBMM". Only tokens that are not ordinary Turkish words belong here.
export const EVAL_ACRONYMS = new Set([
  'ABD', 'AB', 'BM', 'TBMM', 'CHP', 'AKP', 'MHP', 'HDP', 'DEM', 'SGK', 'KDV', 'ÖTV', 'TL', 'TRT', 'PTT',
  'THY', 'TCDD', 'TC', 'AVM', 'ATM', 'SMS', 'PDF', 'USB', 'GPS', 'DNA', 'MR', 'EKG', 'AİHM', 'AYM',
  'YSK', 'ÖSYM', 'YKS', 'LGS', 'KPSS', 'TSK', 'TÜİK', 'BDDK', 'SPK', 'TCMB', 'İBB', 'İETT', 'ABB',
  'PKK', 'IMF', 'NBA', 'CEO', 'TV', 'KKTC', 'AİHS', 'HSK', 'MSB', 'EGM', 'BTK', 'EPDK', 'RTÜK', 'KOBİ',
  'SSK', 'TMSF', 'KVKK', 'EFT', 'TBB', 'TFF', 'UEFA', 'FIFA', 'NATO', 'AFAD', 'MEB', 'YÖK', 'MİT',
]);
