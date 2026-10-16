# Ahenk — proje durumu (Ekim 2026)

Başvurularda kullanılmak üzere kısa ve dürüst özet. Sayılar aynı ölçüm düzeneğinden (TR-Zor 155 zor cümle,
Antalia'nın ayrılmış 20 uzun cümlesi, Whisper large-v3-turbo + trnorm normalizasyonu, UTMOS22). İnsan dinlemesi tek
dinleyiciyle (anadili Türkçe) kör yapıldı; sistematik bir MOS testi henüz yapılmadı.

## Ne yaptık

1. **Ahenk (büyük model)** — VoxCPM2 (2B, Apache-2.0) üzerine Türkçe LoRA, tek bir **tasarlanmış (yapay)**
   kadın sesi. Eğitim verisi tamamen sentetik ve ticari kullanıma açık lisanslı (CC BY 4.0); gerçek bir kişinin sesi
   kullanılmadı. Soru tonlaması, 6 duygu (kızgın, neşeli, üzgün, şaşkın, sakin, empatik; İngilizce ve Türkçe talimat),
   gülme, sayı/tarih/para/telefon okuma. Hugging Face'te (şimdilik gizli) yayın paketi hazır.
2. **trnorm** — açık kaynak Türkçe metin normalleştirici (sayılar, tarihler, saat, para, kısaltmalar, ekler). EMA
   Lightning'in TR-Zor hata oranını %14,0'dan %1,9'a indirdi.
3. **Ahenk Cep: küçük cihaz modeli (deneme)** — 8,6 milyon parametre, 34 MB (fp32), CPU'da gerçek zamandan ~5 kat hızlı; büyük modelin
   sesinden öğrenen öğrenci. Henüz yayınlanmadı.
4. Yardımcı araçlar: telaffuz denetimi (IPA tabanlı "c" kontrolü), nefes temizleyici (pause_clean), değerlendirme
   düzeneği ve kör dinleme sayfaları; hepsi açık kaynak.

## Türkçe açık kaynak seslerle karşılaştırma

| Model | Boyut | TR-Zor WER (hatasız) | Ayrılmış WER | UTMOS kısa / uzun | Not |
|---|---|---|---|---|---|
| **Ahenk** | 2B + LoRA | **2,54** (141/155) | 0,82 | 3,70 / 3,14 | açık ağırlık, açık sentetik veri |
| Trendyol-TTS | 2B + LoRA | 3,98 (134/155) | 0,68 | 3,82 / 3,73 | ~20 sa kapalı veri |
| **Ahenk Cep (deneme)** | 8.6M | **2,32** | — | 3,41 / 3,18 | aynı ses, cihaz üstü |
| EMA Lightning | 8,6M | 2,43 | — | 3,35 / 2,95 | Apache-2.0 |
| Antalia-2 Mini | 7,6M | 3,30 | — | 3,22 / 2,94 | Apache-2.0 |

(Küçük model satırları ayrı bir oturumda, aynı düzenekle ölçüldü; 60/155 satırlık alt kümelerde farklar birkaç
kelimedir.)

- **Güçlü yanımız:** zor metni doğru okuma (sayı, tarih, para, telefon, kısaltma), soru tonlaması, duygu kontrolü,
  tamamen açık ve lisans zinciri temiz veri, aynı sesle büyük + küçük model ailesi.
- **Zayıf yanımız:** uzun cümlelerde doğallık Trendyol-TTS'in gerisinde (UTMOS 3,14'e karşı 3,73); duygulu seslerde
  zaman zaman "telefon gibi" tını; Türkçe talimatla duygu daha hafif.
- **Kör dinleme (tek dinleyici):** büyük model EMA Lightning'e karşı 20 cümlede 12'ye 2; küçük model aynı boyuttaki
  EMA ve Antalia-2 Mini'ye karşı 10 cümlede hiç kaybetmedi (öğretmen 8, öğrenci 2, diğerleri 0).

## GPU ile ne yapılacak

| İş | Tahmini A100 saati |
|---|---|
| Küçük model için 20-50 saat temiz öğretmen verisi üretimi ve denetimi | 30-60 |
| Küçük modelin büyük veriyle eğitimi + az adımlı damıtma (telefon hızı) | 20-50 |
| Büyük model iyileştirmeleri (soru ve duygu verisi, yeniden eğitim) | 10-20 |
| İzinli gerçek ses (seslendirme sanatçısı) ile LoRA, varsa | 5-10 |
| Değerlendirme, sistematik dinleme testleri | 10-20 |
| **Toplam** | **~75-160** |

Bugüne kadar tüm çalışma küçük aylık bulut GPU kredileri ve bir dizüstü GPU'su (RTX 5060, 8 GB) ile yapıldı.

---

## English summary

Ahenk is an open Turkish TTS project: a LoRA on VoxCPM2 (2B, Apache-2.0) with one designed synthetic voice,
trained only on CC BY synthetic data, with question intonation, six emotions, laughter and a Turkish text normaliser
(trnorm, which cut EMA Lightning's hard-text WER from 14.0 % to 1.9 %). On our TR-Zor benchmark it has the lowest
word error rate of the open Turkish models we measured (2.54 % vs Trendyol-TTS 3.98 %), while its predicted
naturalness on long sentences is lower (UTMOS 3.14 vs 3.73). An 8.6M-parameter (34 MB) on-device student in the same voice (Ahenk Cep)
already matches or beats EMA Lightning and Antalia-2 Mini on our automatic metrics and won every blind comparison
against them (single listener). GPU time (~75-160 A100 hours) would let us scale the student's data to 20-50 hours,
distil it for phones, and improve the large model; so far everything ran on small monthly cloud GPU credits and a
laptop GPU.
