# Kendi sesimiz: araştırma özeti ve plan (2026-10-11)

Amaç: Antalia yerine **bize ait**, daha doğal bir ses; soruları soru gibi okuyan ve duygu (kızgın, neşeli, üzgün,
şaşkın, sakin, empatik; ileride gülme) taşıyan bir Türkçe TTS. Taban model VoxCPM2 kalıyor (araştırılan açık modeller
arasında Türkçede ticari lisanslı, belirgin şekilde daha iyi bir taban bulunamadı).

## Bulgular (kaynaklı ayrıntılar araştırma raporlarında; ölçüm çıktıları yerel, depoda değil)

- **Tavan sestedir, modelde değil.** Antalia'nın gerçek kayıtları UTMOS 3.44; modelimiz 3.40-3.57. Trendyol 3.82.
- **VoxCPM2 ile tasarlanmış sesler daha yükseğe çıkabiliyor.** Alania'da kadın seslerin ~%14'ü ses ortalamasında
  UTMOS ≥ 3.9, en iyiler 4.1-4.2 (26 dosya, ~66 bin klip; Alania `qc_utmos` bizim ölçerle r=0.998 uyumlu).
  Ama: tek klibe bakıp seçmek yanıltıcı (ilk klibi ≥ 4.1 olan seslerin diğer klipleri ortalama 3.88); adayları
  birkaç klip ortalamasıyla sırala. Klonlama ~0.17 UTMOS kaybettirir. Model, kendi ürettiği sesin üstüne çıkamaz:
  gerçekçi hedef doğallıkta **≈ Trendyol seviyesi**.
- **Gerçek, izinli, ticari kullanıma açık, Antalia'dan iyi kaydedilmiş tek kişilik Türkçe ses bulunamadı.**
  Common Voice gürültülü (UTMOS medyanı 2.5). Ünlü/YouTuber setleri izinsiz; Appen 700h lisansı belirsiz.
- **Başka dildeki izinli sesler Türkçe konuşturulabilir** (VoxCPM2 referanstan klonlama, yalnız `reference_wav_path`
  modu; "Ultimate"/devam modu aksanı taşır, issue #321). Adaylar (rıza ve ticari izin açık):
  Dialogs-ru Masha/Dima (OpenRAIL, yazılı rıza, çok duygulu; Rus aksanı riski), KazakhTTS2 F1/M1 (Türki, rıza var,
  ticari izin makalede; düşük aksan riski, düz haber okuma), Thorsten-Voice (CC0, ses bağışı, erkek, Alman aksanı
  riski). VCTK, LibriVox türevleri (Hi-Fi TTS, LibriTTS-R), LJSpeech: lisans serbest ama **klonlama rızası yok**, kullanma.
- **Ses dönüştürme (VC)**: gerçek Türkçe konuşmanın tonlamasını tek bir tınıya taşır. Ticari lisanslı: Chatterbox VC
  (MIT, filigranlı), RVC v2 (MIT), OpenVoice v2 (MIT). Seed-VC GPL + NC veri; Vevo/EZ-VC NC. Kaynak konuşmacıların
  rızası yok (YODAS): yalnız iç test, yayınlanmaz.
- **Onarılmış gerçek Türkçe**: Sidon (MIT) gürültüyü/yankıyı temizler, tonlamayı canlandırmaz. CV22-Sidon (CC0, HF
  kapısı) aşama 1 için Alania'ya alternatif olarak A/B denenebilir (~$5-6).
- **Soru tonlaması**: Türkçe evet-hayır sorusunda tepe "mI"den önceki hecede, sonra düşüş (soruların çoğu alçak biter);
  ne/kim sorusunda tepe soru sözcüğünde; mI'siz sorular sonda yükselir. Bizim model ortalamada doğru ama tutarsız
  (7 sorudan 3'ü doğru; AUC 0.73, Trendyol 0.85).
- **Duygular**: VoxCPM2'de talimat metnin başına parantezle yazılır: `(Angry tone)metin`. Eğitim betiği metni olduğu
  gibi işlediği için `(talimat)+trnorm(cümle)` + `ref_audio` satırları kod değişmeden talimat öğretir. Yerel deney:
  kızgın 3/3, mutlu 3/3, üzgün 0/3, gülme yalnız zayıf kıkırdama. Alania'da ~78 sa talimatlı kadın sesi var ama
  hafif duygular, 38 farklı talimat; kahkaha/öfke/ağlama yok. ElevenLabs çıktısıyla eğitim kullanım şartlarıyla yasak.
- **Hız**: resmi API cümle cümle RTF 0.9-2.3; 10 sa veri üretmek ~$100 eder. Toplu üretim (Nano-vLLM-VoxCPM, MIT)
  A100'de ölçüldü (2026-10-11, `nanovllm_app.py`): 64 istek aynı anda saniyede ~50 sn ses (10 adım), ~37 (16 adım);
  kalite resmi API ile aynı (WER 0, UTMOS gürültü içinde, klipler %7-20 uzun). Ses yarışması artık bununla çalışır
  (`voice_contest.py --backend nanovllm`); veri üretimi de bunu kullanmalı.
- **Lisans zinciri**: Alania metni/sesi için PatientDesk atfı; FineWeb-2 (ODC-By, Common Crawl şartları) aşama 1'de var.
  AB Yapay Zekâ Yasası 50. madde: sentetik ses makinece işaretlenmeli (AudioSeal, MIT).

## Plan

| # | Adım | Maliyet |
|---|---|---|
| 0 | trnorm mI/dA uyumu, TR-Soru-30 seti + tonlama puanlayıcı, Alania talimat desteği | $0 |
| 1 | **Ses yarışması**: tasarlanmış sesler (~600 tasarım → çok klip ortalamasıyla en iyi 30 → 4-5), Masha, Dima, KazakhTTS2 F1/M1, Thorsten; aynı 70 cümle (TR-Zor, sorular, duygulu satırlar); WER, Türkçe dil-ID ≥ 0.95, UTMOS, benzerlik; Nano-vLLM ile | ~$4 |
| 2 | **Kör dinleme** (kullanıcı + 1-4 anadil konuşuru): "hangisi daha insan gibi", "yabancı aksan var mı" | $0 |
| 3 | Kazanan sesle ~5 sa veri, Nano-vLLM ile (hız testi yapıldı; %30 soru, tonlamasına göre seçilmiş; 6 duygu) | ~$5-8 |
| 4 | Eğitim (aşama 1 LoRA'sından) + değerlendirme + kör A/B | ~$5 |

Kural: her ücretli adım `--max-usd` ile sınırlı; kör dinlemede kazanan yoksa dururuz, mevcut model (TR-Zor 3.42) kalır.
En temiz ileri adım: rıza veren bir anadil konuşurunun 30-60 dk soru/duygu kaydı (yazılı izinle).

## Sonra: küçük model (öğretmen-öğrenci), büyük model bittikten sonra

Kullanıcı kararı (2026-10-11): önce büyük model (VoxCPM2 + seçilen ses) eğitilecek; o bitince aynı sesle küçük, telefonda ve
tarayıcıda çalışan bir model yapılacak.

- Neden: EMA Lightning (8.6M parametre, Apache-2.0, `src/trtts`) TR-Zor'da trnorm ile %1.9 WER (bizim 2B model %3.42).
  Küçük, otoregresif olmayan modeller kelime atlamaz/eklemez ama doğallık, duygu ve ses kontrolü sınırlı; WER yargıcı düz
  okumayı kolay anlar.
- Yöntem: büyük model öğretmen olur. Seçilen sesle 20-50 sa temiz konuşma üretilir (Nano-vLLM ile ~$0.06/sa ses), CER 0,
  başta fazla ses yok, tonlama kontrolü geçen klipler tutulur. Bu veriyle küçük öğrenci model eğitilir: EMA Lightning'in
  kendisi (eğitim kodu açık mı kontrol edilecek) ya da benzer küçük mimari. Duygu için birkaç temel stil etiketi denenebilir.
- Beklenti: doğallıkta öğretmene yaklaşır, geçemez. Eski Piper denemesi (4.8 sa gerçek veri, espeak fonemleri, %24.6 WER)
  başarısızdı; bu sefer çok daha fazla temiz veri ve trnorm var.
- Ürün: büyük model (sunucu, en doğal, duygulu) + küçük model (cihazda, hızlı, aynı ses).
