# Gerçek insan sesi kaynakları (araştırma, 2026-10-14)

Yalnız araştırma: veri indirilmedi, ücretli hizmet kullanılmadı. Önceki listeler: `docs/SES_PLANI.md` (Bulgular),
`docs/ARASTIRMA.md` §4 (izinli gerçek konuşma havuzu).

## Türkçe özet

**Kısa cevap: İndirilip doğrudan "tek ses" olarak ince ayar yapılabilecek, ticari lisanslı, rızası belgeli,
stüdyo kalitesinde, saatlerce süren tek kişilik bir Türkçe kayıt hâlâ yok.** Elimizdeki tek örnek Antalia'nın
kendi korpusu (5 sa, CC-BY-4.0, profesyonel seslendirme sanatçısı, sözleşmeli). Bu, 2026-10-11'deki bulguyla
aynı. O tarihten beri iki yeni ve önemli kaynak çıktı. İkisi de çok konuşmacılı, yani "bizim sesimiz" olamaz:

1. **Basis Global Emotions (8 Ekim 2026), Türkçe bölümü:** 6,12 sa, 72 konuşmacı, 48 kHz. 36 duygu ve
   söyleyiş biçimi var (kızgın, üzgün, neşeli, fısıltı, bağırma, gülme...). Her katılımcı kayda ve AI
   eğitimine izin vermiş, metin-okuma ve ses klonlama sistemlerinin eğitimi de buna dahil. Lisans telifsiz;
   bu veriyle eğitilen model satılabilir. Kayıtların kendisini yeniden dağıtmak ve belirli bir kişiyi taklit
   etmek yasak. Erişim insan onayıyla veriliyor. **Duygu ve tonlama için şu an en iyi aday.** Konuşmacı başına
   ortalama ~5 dk düşüyor. Bu yüzden aşama-1'de çok konuşmacılı "duygu öğretmeni" olarak kullanılır, ardından
   aşama-2'de f022 sesine dönülür.
2. **Basis Conversations 1500, Türkçe bölümü:** 19,6 sa, 40 konuşmacı, 27 karşılıklı konuşma. 48 kHz, her
   konuşmacının ayrı kanalı var. Rıza yine kayıt sırasında alınmış. Ancak lisans metni (LICENSE) erişim
   kapısının arkasında, ticari model eğitimine izin verdiği kartta açıkça yazmıyor. Kullanmadan önce okunmalı.
   Doğal konuşma ritmi için değerli. Transkriptler gürültülü, kart bunlarla eğitim yapılmamasını söylüyor.

Diğer adaylar ve durumları:

- **Common Voice 25 TR (CC0):** 135 sa, 1.816 kişi. Rızası en temiz büyük havuz bu, ama MP3 ve gürültülü
  (bizim ölçtüğümüz UTMOS medyanı 2,5). Konuşmacının kimliğini bulmaya çalışmak ve veriyi yeniden barındırmak
  yasak. Küçük modelin ön eğitimi ve ASR için uygun. Sidon ile onarılmış hâli (CV22-Sidon, 48 kHz FLAC)
  denenebilir. Tek ses olarak kullanılmaz.
- **Risale-i Nur Text–Audio (CC-BY-4.0, yeni):** 203 sa, 2 okuyucu, 16 kHz. Yayınevinin (RNK Neşriyat) yazılı
  izni var, ama okuyucuların rızası yazmıyor. Metin dinî; Osmanlıca ve Arapça kelimeler içeriyor. Ön eğitim
  için kullanılabilir. Okuyucunun sesini klonlamak için kullanılmamalı. Aynı yayıncının daha büyük seti
  ticari olmayan lisanslı ve kullanılamaz.
- **YODAS2/YODAS3 TR (CC-BY-3.0, YouTube):** Lisansı videoyu yükleyen vermiş, konuşmacıların rızası yok.
  Yalnız ön eğitim ve ASR için. Alania'nın stil altyazıları (CC-BY) bu kayıtlardan duygulu/canlı parçaları
  seçmeye yarar.
- **ISSAI TSC (MIT, 218 sa):** Kaynağı açıklanmamış, rıza bilgisi yok. Ön eğitim ve ASR için.
- **FLEURS (CC-BY-4.0, ~11 sa, 16 kHz)** ve **MediaSpeech (CC-BY-4.0, 10 sa, YouTube medya kanalları):**
  Değerlendirme ve ASR için.
- **Kullanılamaz olanlar:** turkish-tts-audiobooks ve kiraat (kaynak kayıtların hakları temizlenmemiş),
  WorldSpeech ve risale-nur-audio (ticari değil), afkfatih/omersaidd "combined" setleri (kaynağı ve rızası
  belirsiz; içindeki Khan Academy kayıtları aslında ticari olmayan lisanslı), Anilosan15 "sıla" (lisans yok),
  cubukcum (kaynak yok), erenfazlioglu (Microsoft TTS çıktısı, ticari değil), Silencio örneği (ticari değil).
  **LibriVox'ta Türkçe kitap projesi yok** (API'de 22.414 proje tarandı, 0 Türkçe). Gazi Üniversitesi'nin
  12,6 sa'lik tek erkek konuşmacılı korpusunun yayınlandığına ya da lisansına dair bir iz bulunamadı.
- **Ücretli ve lisanslı:** Shaip Turkish (2.027 sa, 48 kHz, mobil uygulamayla kaydedilmiş, çok konuşmacı;
  fiyat ve lisans teklifle). Basis'ten özel lisans da istenebilir.

**En büyük kaldıraç hâlâ kendi kaydımız.** Rıza veren tek bir anadil konuşuru (tercihen profesyonel
seslendirme sanatçısı) şunları okur: 2-5 sa düz okuma, %15-20 soru, %10-20 ünlem ve 6 duygu. Gereken belgeler:
- FSEK m.52'ye uygun yazılı sözleşme. İcracı hakları (m.80) tek tek sayılmalı; "tüm haklarımı devrettim" gibi
  genel bir ifade geçersiz.
- KVKK açık rızası. Ses, biyometrik ve özel nitelikli veri sayılabilir.
- Açık klonlama ve ticari kullanım izni, ve taklit yasağı maddesi.

Microsoft'un kılavuzu: ~300 cümle makul bir ses için yeter, karakterli/duygulu bir ses için 1.000-2.000 cümle
gerekir. Bir sanatçı günde en fazla 2-3 saat kayıt yapabilir.

**Maliyet:** Türkiye için yayımlanmış birim fiyat bulamadık. Elimizdeki göstergeler: Armut'ta tek iş
350-7.750 TL; seslendirme sanatçısının ortalama aylık maaşı 42.800 TL; ABD'de yapay zekâ klon sözleşmelerinde
asgari ~1.500 $ oturum ücreti (doğrulanmamış). **Tahminimiz: sanatçı, stüdyo ve AI lisansı dahil 3-5 sa'lik
bitmiş veri için kabaca 1.000-3.000 $.** Bu doğrulanmış bir fiyat değil; 2-3 sanatçıdan teklif alınmalı.

**Önerilen sıra:**
1. Basis Global Emotions'a erişim iste ($0). LICENSE'ı oku. Konuşmacıların taklit edilmediği, çok konuşmacılı
   bir duygu aşaması dene.
2. Basis Conversations'ın LICENSE'ını oku.
3. Kendi kaydımız için 2-3 sanatçıdan teklif al.
4. Küçük modelin ön eğitimi için CV + YODAS + TSC'yi kalite filtresinden geçirerek kullan.

---

## Details (English)

### Criteria
(1) Commercial use and model training permitted (CC0, CC-BY, MIT, Apache, ODC-By or a written commercial
licence; not NC, ND or research-only). (2) The speakers consented to this kind of use. (3) Clean recordings at
22-48 kHz, ideally one or a few speakers with hours each. Expressive or conversational data is a plus.

### A. Usable with consent documented (best fit)

#### A1. Basis Global Emotions — Turkish (tr-TR)
- Link: https://huggingface.co/datasets/basis-ai/basis-global-emotions (gated, created 2026-10-08)
- Licence: `basis-data-license-1.0`. Card: "The Basis Data License grants a worldwide, royalty-free, perpetual
  license". Card: "Two things are not allowed: reselling or redistributing the recordings themselves", and
  "identifying, locating or contacting speakers, impersonating a specific person, or reconstructing removed
  information." Models built on the data may be sold "with no royalties, no revenue sharing and no further
  permission from Basis." The full LICENSE text sits behind the gate; read it before use.
- Consent: "Every participant agreed at signup to be recorded and to license their recordings for AI training".
  The consent section names "training of speech recognition, text-to-speech and voice-cloning systems".
  Speakers also waived publicity rights for AI training.
- Size: tr-TR has 6.12 h, 1,601 recordings, 643 series and 72 speakers (≈5 min per speaker).
- Audio: WAV, 16-bit, 48 kHz, mono, "exactly as recorded, no post-hoc denoising, enhancement, loudness
  normalization, resampling". Recorded on consumer apps. Recordings must pass minimum NISQA scores; whisper and
  laughter have their own acoustic check.
- Content: 36 "steers", each an emotion (e.g. angry, sad, amused, afraid) or a delivery direction (whispering,
  shouting, screaming, singing, mumbling). Laughter is included. These are exactly the gaps our Alania-based
  instruction data has (SES_PLANI: "kahkaha/öfke/ağlama yok").
- Quality caveat: consumer microphones, not studio. Measure UTMOS/DNSMOS per clip before use.
- Use:
  - Big model: a multi-speaker emotion stage, with `(Angry tone)` style instruction prefixes mapped from the
    steer labels, then re-anchor on f022 in stage 2.
  - Small model: emotion and style tokens.
  - Never clone an individual Basis speaker. Their terms forbid impersonation.

#### A2. Basis Conversations 1500 — Turkish
- Link: https://huggingface.co/datasets/basis-ai/basis-conversations-1500 (gated)
- Licence: `basis-data-license-1.0`. The card says only "The terms are in `LICENSE`" (behind the gate). Custom
  licences are available via data@withbasis.co. **Commercial training is not confirmed from the card.**
- Consent: same signup consent ("agreed at signup to be recorded and to license their recordings for AI
  training"), plus a publicity-rights waiver.
- Size: tr-TR has 19.6 h, 27 conversations and 40 speakers. 3.2 h are human-annotated.
- Audio: FLAC, 16-bit, mono, 48 kHz, "as recorded". Each participant's own microphone is on its own track,
  with a shared clock. Recorded "in a mobile and web app over live calls". Some noisy real-world tracks remain.
- Caveat: the card warns that the transcripts are noisy. Re-transcribe before use.
- Use: conversational rhythm, backchannels and turn-taking. Pretraining for the small model or a duplex model.
  Not a single voice.

#### A3. Antalia voice corpus (already in use)
- Link: https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus
- Licence: CC-BY-4.0. Attribute "Antalia (Patientdesk.ai)".
- Consent: a professional voice actor who "recorded this material for the project under a commercial
  agreement". She later signed an addendum permitting public redistribution, on the understanding that her
  voice would not be used to impersonate her or to pass synthetic speech off as a genuine recording. The card
  does not explicitly address voice cloning.
- Size and audio: 5.008 h, 1,073 clips. Recorded at 48 kHz, published as 24 kHz mono 16-bit. Median SNR 34.9 dB.
- Note: this is still the only real single-speaker, licensed and consented Turkish TTS set found. Our measured
  UTMOS for it is 3.44 (SES_PLANI).

#### A4. Mozilla Common Voice — Turkish (Scripted Speech 25.0)
- Link: https://datacollective.mozillafoundation.org/datasets/cmn2e7kbl01k2mm07gm5n1bc9 (now served from
  mozilladatacollective.com). Older HF mirror `fixie-ai/common_voice_17_0` (ARASTIRMA.md).
- Licence: CC0-1.0. The datasheet terms say re-hosting or re-sharing is forbidden, and downloaders agree "to not
  determine the identity of speakers" (seen on the CV 25 datasheets). A Mozilla staff reply reads this as barring
  "processes that require identification of individual contributors through matching and grouping their
  separate contributed clips" (https://discourse.mozilla.org/t/speaker-ids-for-speaker-identification-model/114358).
- Consent: contributors donate recordings under the CC0 dedication.
- Size: ~126,500 clips, ~135 h, 1,816 speakers. The datasheet notes speakers with foreign accents, "mostly
  immigrants".
- Audio: MP3 (https://datacollective.mozillafoundation.org/datasets/cmflnuzw6jhrlhdrkr2lqtq0g). Mozilla calls
  48 kHz "optimal for interoperability" (https://www.mozillafoundation.org/en/blog/guidance-for-splinter-datasets-on-mcv).
  Our measurement: UTMOS median 2.5; ~2 h pass ≥3.3 and ~9 h pass ≥3.0.
- Restored version: CV22-Sidon (https://huggingface.co/datasets/sarulab-speech/commonvoice22_sidon), CC0-1.0,
  48 kHz mono FLAC, gated, same no-identification clause.
- Use:
  - Small-model pretraining and ASR filters.
  - A/B as stage-1 human data after a UTMOS ≥3.3 filter.
  - Not for cloning one contributor: grouping by `client_id` to build one person's voice runs against the
    identification clause.

### B. Licence OK, but speaker consent for voice use is NOT documented (pretraining/ASR only, never cloning)

| Source | Licence (quote/tag) | Size | Audio | Consent / provenance | Use |
|---|---|---|---|---|---|
| Risale-i Nur Text–Audio, https://huggingface.co/datasets/risaleinur/risalei-nur-text-audio | CC-BY-4.0: "may be shared, adapted and used for any purpose, including commercially, with attribution"; "RNK Neşriyat — yazılı izinle · used with written permission" | 203.02 h, 91,792 clips, 2 readers (credited "A.Köseoğlu" in the viewer) | 16 kHz mono PCM16, ASR-verified alignment | Publisher permission only. Narrator consent not stated | Long-form reading prosody for small-model pretraining. Ask the publisher whether the narrators' performer rights (FSEK m.80) were transferred before using it in the big model. Domain: religious text with Ottoman and Arabic vocabulary; needs trnorm care |
| YODAS2 / YODAS3 TR, https://huggingface.co/datasets/espnet/yodas2, https://huggingface.co/datasets/espnet/yodas3 | CC-BY-3.0: "only consisted of videos with CC licenses" | YODAS2 tr ≈500 usable h (measured, ARASTIRMA.md). YODAS3 TR ≈2,398 h captioned by Alania | YouTube, mixed | Uploader licence. No speaker consent; removal by e-mail | Pretraining and ASR. Use Alania speech captions (CC-BY-4.0, real YODAS3 speech, https://huggingface.co/datasets/cloud0day3/alania-speech-captions-tr) to pick expressive, clean spans |
| ISSAI Turkish Speech Corpus, https://huggingface.co/datasets/issai/Turkish_Speech_Corpus | MIT (card tag) | 218.2 h, 186,171 utterances | not stated on card | Card and repo do not describe the sources or consent | Pretraining and ASR |
| FLEURS tr, https://huggingface.co/datasets/google/fleurs | CC-BY-4.0 | ~12 h per language (https://arxiv.org/abs/2205.12446) | 16 kHz | Read FLoRes sentences, 3 native speakers per sentence. Recruitment/pay not stated on the card | Evaluation and ASR |
| MediaSpeech tr, https://openslr.org/108/ | "distributed under the Creative Commons Attribution 4.0 International License" | 10 h | — | "collected from the official youtube channels of media" (https://arxiv.org/abs/2103.16193). No consent | Evaluation only |

### C. Rejected

- **serdarcaglar/turkish-tts-audiobooks** (2,724 h, 16 kHz): "The rights of the source recordings are not
  cleared; you accept legal responsibility for your use". https://huggingface.co/datasets/serdarcaglar/turkish-tts-audiobooks
- **serdarcaglar/kiraat**: built from audiobook YouTube channels. Already excluded in ARASTIRMA.md §7.
- **risaleinur/risale-nur-audio**: licence tag "rnk-attribution-noncommercial".
  https://huggingface.co/datasets/risaleinur/risale-nur-audio
- **disco-eth/WorldSpeech**: non-commercial (ARASTIRMA.md).
- **afkfatih/turkish-tts-combined-raw** and its copies (Taklaxbr, projectkaira, mshasir, Hm12cbbcbx):
  - The overall tag is CC-BY-SA-3.0.
  - The Mazlum Kiper, Ahmet Deniz and Nisan Kumru parts come from `omersaidd/*`. These are tagged MIT, with no
    provenance or consent statement.
  - The Khan Academy part (`ysdede/khanacademy-turkish`, 78 h, tagged CC-BY-SA-3.0) comes from Khan Academy
    videos, which Khan Academy licenses as CC BY-NC-SA
    (https://support.khanacademy.org/hc/en-us/articles/202262954-Can-I-use-Khan-Academy-s-videos-name-materials-links-).
    The HF tag is therefore unreliable.
- **Anilosan15/Turkish_TTS_Data** (one voice "sıla", 48 kHz, 30,606 rows): "README.md exists but content is
  empty". No licence and no source.
- **cubukcum/TurkishVoiceDataset**: tagged CC0, but every source field reads "[More Information Needed]".
- **erenfazlioglu/turkishvoicedataset**: generated with Microsoft TTS, CC-BY-NC-4.0.
- **jml2026/multilingual-accent-speech** (Silencio sample): CC-BY-NC-4.0.
- **falan42/*-tts-dataset** (Bentropi, Tunç M., Mert H.): no card or licence. The names suggest YouTubers or
  public figures, i.e. no consent.
- **LibriVox**: the LibriVox API (`/api/feed/audiobooks`, all 22,414 projects listed on 2026-10-14) has **zero**
  projects with language "Turkish". Yunus Emre has an author page (https://librivox.org/author/8697) but no
  Turkish-language project. SES_PLANI already notes that LibriVox derivatives lack cloning consent.
- **Gazi Univ. 12.6 h male TTS corpus** (doi:10.17671/gazibtd.1159289): 12 h 38 min, one male speaker, Istanbul
  Turkish, 22,050 Hz (https://search.trdizin.gov.tr/yayin/detay/1192280). No public release or licence found.
  One could ask the authors.
- **TurEV-DB** (emotion; 82 words, amateur actors; https://aclanthology.org/2020.sltu-1.52) and **TurES**
  (film audio): too small or no rights.
- **Spoken Wikipedia TR, Turkish CC podcasts**: none found.

### D. Paid / licensed

- **Shaip "Turkish (Turkey)"**: 2,027 h, 2,833 speakers, 48 kHz mono WAV. Recorded via a mobile app.
  https://shaip.com/offerings/speech-data-catalog/turkish-turkey-dataset
  Price and licence are by quote. Shaip claims every recording has "explicit informed consent"
  (https://www.shaip.com/offerings/tts-dataset). Ask for the licence, the consent form, and whether TTS voice
  cloning is allowed.
- **LDC**: METU Turkish Microphone Speech (LDC2006S33) and Turkish Broadcast News (LDC2012S06,
  https://catalog.ldc.upenn.edu/LDC2012S06). LDC licence terms, paid. Broadcast news has no cloning consent.
- **Basis custom licence**: data@withbasis.co (named on the Conversations card).
- **Fiverr seller** (studio 48 kHz Turkish): restricted to internal training, no TTS cloning for resale.
  https://www.fiverr.com/s/R71AjEx Unsuitable.

### E. Recording our own voice (recommended)

**Legal minimum (Turkey):**
- FSEK m.80 gives the performer the exclusive right to authorise fixation, reproduction, distribution and
  communication of the performance.
- FSEK m.52 requires the contract to be written, with each right listed separately. A clause like "tüm
  haklarımı devrettim" is not valid. Source: https://www.erdem-erdem.av.tr/bilgi-bankasi/eser-sahibinin-mali-haklarinin-devri
- KVKK lists biometric data as special-category data
  (https://cottgroup.com/tr/blog/kvkk-gdpr/item/biyometrik-verilerin-kvkk-ve-gdpr-bakimindan-islenmesi), so take
  explicit, separate consent for voice data.
- Contract checklist:
  - training a TTS model and releasing model weights;
  - commercial use and synthetic speech in her timbre;
  - territory and duration worldwide and perpetual, or a defined term with a clear revocation effect;
  - a no-impersonation and no-deception clause (as in the Antalia addendum);
  - an AI-disclosure and watermark commitment (EU AI Act Art. 50, SES_PLANI);
  - a recorded spoken consent statement (Microsoft requires one for professional voices).
- Have a lawyer review the contract. This is not legal advice.

**How much data:**
- Microsoft: "in most cases you can build a reasonable custom voice with 300 utterances". For a "voice persona
  with unique characteristics... at least 1000 (preferably 2000) utterances"
  (https://learn.microsoft.com/en-us/azure/ai-services/speech-service/record-custom-voice-samples).
- Our own data point: 5 h of Antalia was enough for a LoRA that matched its voice.
- Target:
  - **1 h minimum** (≈600 utterances): voice identity on top of the existing stage-1 LoRA.
  - **3-5 h ideal** (≈2,000-3,000 utterances): solid prosody plus emotion.
  - Add a separate **30-60 min of question and emotion lines** (SES_PLANI's "en temiz ileri adım").

**Script design** (adapted from Microsoft's guide):
- 70-80% statements.
- 10-20% questions, both rising and falling. For Turkish, cover the evet-hayır questions with mI (peak on the
  syllable before mI), the ne/kim questions, and questions without mI.
- 10-20% exclamations and ~10% short phrases.
- Six emotions, recorded as separate blocks.
- Numbers, dates, addresses and phones written in spoken form. Generate them with trnorm and check by hand.
- Draw sentences from `alania-domain-text-tr` (CC-BY-4.0) plus TR-Zor-style hard items.
- 2-15 s per line, no duplicates.

**Recording spec** (Microsoft):
- Studio booth; 48 kHz/24-bit capture, deliver ≥24 kHz/16-bit mono WAV.
- SNR > 35 dB; peaks -3 to -6 dB; compression no more than 4:1.
- 100-200 ms of lead and tail silence; 5 s of room tone at the start.
- Constant mic distance, a pop filter, and a "match file" repeated every page for consistency.
- Talent works 2-3 h/day, 3-4 days/week. A 3-5 h finished set is therefore ~2-3 weeks of sessions.

**Cost in Turkey:** no published per-minute rate card found.
- Armut gives job ranges, e.g. "350 TL ile 7.750 TL" (https://armut.com/kazan-seslendirme).
- eleman.net 2026 average voice-actor salary: 42,800 TL/month (https://www.eleman.net/meslek/seslendirme-sanatcisi/maas).
- US context: a ≥ $1,500 session fee for AI voice-clone contracts is cited by a blog
  (https://voxbooster.com/tr/blog/voiceover-rates-statistics-2026/), unverified.
- Royalty model: $0.03-0.20 per 1,000 characters (https://www.camb.ai/blog-post/how-voice-actors-license-their-voice-to-ai).
- **Estimate (ours, unverified): 1,000-3,000 USD all-in for 3-5 finished hours.** Breakdown:
  - roughly 10-15 studio hours plus engineer;
  - talent fee with a perpetual AI-voice buyout;
  - editing and segmentation (which we can do ourselves).
- Get 2-3 written quotes, priced per finished hour with the AI licence itemised. Royalty or revenue-share
  options lower the upfront cost.
- A cheaper path: a home-studio professional who meets the SNR spec, recording in sessions we direct remotely.
