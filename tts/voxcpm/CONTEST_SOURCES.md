# Ses yarışması: kaynaklar, lisanslar, rıza / Voice contest: sources, licences, consent

Bu belge yarışmadaki her sesin nereden geldiğini, hangi lisansla kullanılabildiğini, konuşmacının rızasının nasıl
belgelendiğini ve kazanan ses bir modele dönüşürse model kartına taşınacak atıf metnini toplar. Plan:
[docs/SES_PLANI.md](../../docs/SES_PLANI.md), adım 1. Lisanslar 2026-10-11'de, aşağıdaki sabit revizyonlarda okundu.

## Özet

- **Tasarlanmış sesler** (VoxCPM2 ile metinden üretilen, gerçek kişi yok): lisans sorunu yok, rıza gerekmez. En
  temiz yol.
- **Thorsten** (Almanca, CC0, sesini kendisi bağışladı): hukuken en sağlam yabancı ses. Alman aksanı riski.
- **Masha / Dima** (Rusça, OpenRAIL, yazılı rıza, ticari serbest): model ve **çıktıları** OpenRAIL kullanım
  kısıtlamalarını taşır; modeli dağıtırken lisans metni eklenmeli. Kısıtlama 10: **tıbbi tavsiye vermek yasak**. Ürün
  sağlık alanında konuşacaksa bu sesi seçme ya da Langswap'tan ticari sözleşme al.
- **KazakhTTS F1 / M1** (Kazakça, CC BY 4.0, rıza formu var): atıf zorunlu. Rıza formunun metni yayımlanmamış; başka
  dilde klonlamayı kapsayıp kapsamadığı bilinmiyor. Kazanırsa ISSAI'den yazılı teyit al, eğitim verisini resmi
  kaynaktan (issai/KazakhTTS) indir.
- Hepsi yalnızca **referanstan klonlama** (`reference_wav_path`) ile kullanılır; devam modu (`prompt_wav_path` +
  `prompt_text`) aksanı taşır, kullanılmaz.
- Çıktı her durumda **sentetik ses** olarak belirtilmeli (AB Yapay Zekâ Yasası md. 50: makinece okunur işaret, ör.
  AudioSeal, MIT).

## How the references are made

`python tts/voxcpm/fetch_refs.py` (local, no GPU, nothing uploaded) writes one clean neutral 6-12 s clip per
cross-lingual voice, plus an expressive clip for Masha (happy) and Thorsten (amused), to `voxcpm-refs/<entry>/`
(`neutral.wav`, `expressive.wav`, `sources.json`). The files are the corpus's own bytes (no resampling, trimming or
gain). Each clip is the best of up to 8 candidates from the pinned dataset revision by UTMOS among the ones without
gross faults (duration, clipping, noise floor, cut-off ending); `sources.json` records the dataset, revision, file,
transcript, licence, consent wording, attribution and the measures of every candidate. UTMOS is an English-trained
predictor: its values on Russian, Kazakh and German recordings rank candidates of one voice, they are not comparable
across languages or with the Turkish outputs. Downloads: ~105 MB for all entries; the UTMOS22 weights (411 MB from
GitHub) come on top only with `--utmos on` and an empty torch hub cache: the default `--utmos auto` uses them only
when cached, else ranks by the noise floor. An entry is replaced whole or not at all: if a clip cannot be downloaded,
the run exits 1, leaves the folder as it was, and the next run tries again.

Clips fetched on 2026-10-11 (sha256 prefix; `voxcpm-refs/<entry>/sources.json` has the full record):

| entry | clip | corpus file | s | Hz | UTMOS | sha256 |
|---|---|---|---:|---:|---:|---|
| `masha` | neutral | `wavs/masha_dima_part2_229.wav` | 7.61 | 44100 | 3.65 | `0a6eadf76e46` |
| `masha` | expressive (happy) | `wavs/masha_sveta_part5_82.wav` | 10.02 | 44100 | 3.48 | `2710793135c5` |
| `dima` | neutral | `wavs/masha_dima_part8_202.wav` | 9.31 | 44100 | 3.69 | `0a7f625d5339` |
| `kazakh_f1` | neutral (news) | `inform_1112_03.wav` (f1 shard 0, row group 3, row 0) | 9.79 | 22050 | 3.10 | `8012d7fe6dc1` |
| `kazakh_m1` | neutral (news) | `inform_1039_029.wav` (m1 shard 6, row group 1, row 4) | 8.66 | 22050 | 3.13 | `926233e3cff7` |
| `thorsten` | neutral | TV-2022.10-Neutral `dd01c488-…---33f19d167fe2689c605c98506d2d61e3.wav` | 6.08 | 44100 | 3.61 | `2cc786aef6ae` |
| `thorsten` | expressive (amused) | TV-2021.06-Emotional `680eb50a-…---0cf91be5ce94ff566520aa47a5b573c8.wav` | 7.16 | 44100 | 3.00 | `e5506d6a725f` |

All entries are rendered by VoxCPM2 with reference-only cloning: `generate(text, reference_wav_path=<clip>, ...)`,
never `prompt_wav_path` / `prompt_text` (continuation carries the source accent, OpenBMB/VoxCPM issue #321). The
transcripts are kept for provenance only.

## (a) Designed voices

- **What:** VoxCPM2 voice design: `"(" + description + ")" + Turkish sentence`, no reference audio. The ~150 English
  descriptions are ours (`tts/voxcpm/voice_descriptions.jsonl`, written for this project, Apache-2.0 like the repo);
  the best screen clips become the cloning references of the designed entries.
- **Licence:** [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), Apache-2.0: commercial use, modification and
  redistribution allowed; keep the licence and NOTICE, state changes.
- **Consent:** not applicable: no recording of a real person is used. The descriptions name no real person and ask for
  no imitation; a designed voice is a new synthetic voice. Residual risk: an accidental resemblance to a real person,
  as with any generated voice.
- **Attribution / model card:** "Voice designed from a text description with VoxCPM2 (OpenBMB,
  https://huggingface.co/openbmb/VoxCPM2, Apache-2.0); no real person's voice." / "Ses, VoxCPM2 ile metin tarifinden
  tasarlandı; gerçek bir kişinin sesi değildir."

## (b) Cross-lingual consented voices

### `masha`, `dima`: Dialogs (Russian)

- **Corpus:** [langswap/dialogs-ru-emotional-conversations](https://huggingface.co/datasets/langswap/dialogs-ru-emotional-conversations),
  revision `09b2b30d2afe` (`09b2b30d2afe5fe21d1dc191c4415ea76c16ef97`). 20.6 h of scripted dialogs by three
  professional theatre actors, recorded face to face in a studio, 12 style/emotion labels. Masha = speaker `M`
  (female, 9.9 h), Dima = speaker `D` (male, 6.2 h). 1,800 of the 11,796 rows in `metadata.csv` have no WAV in the
  repo; fetch_refs only picks files that exist.
- **Licence:** OpenRAIL (the BigScience / CreativeML OpenRAIL-M template applied to a dataset):
  https://huggingface.co/datasets/langswap/dialogs-ru-emotional-conversations/blob/09b2b30d2afe5fe21d1dc191c4415ea76c16ef97/LICENSE.md.
  Perpetual, worldwide, royalty-free, including commercial use.
- **Consent:** dataset card and licence footer: the voice actors "gave written informed consent for open release and
  lawful use, including commercial". The consent forms are not published.
- **Obligations:** models trained on or derived from the data, and their outputs, are Derivatives (Section 1).
  Distributing a model or its outputs requires (a) shipping the licence with its use restrictions (Section 2 /
  Attachment A) and (b) binding recipients to them (Section 3); own terms may be added but not weaken them. Among the
  restrictions: no unlawful use, no harm to minors, no harmful disinformation, no harassment or defamation, no fully
  automated decisions against a person's legal rights, and **no medical advice or medical results interpretation**
  (restriction 10). Langswap offers a paid commercial agreement with indemnification (dataset card).
- **Risk:** Russian accent in Turkish; conversational, not read-style recordings.
- **Attribution / model card:** "Voice reference: Dialogs: A Studio-Quality Expressive Conversational Russian Speech
  Corpus for Dialog Assistants, I. Shigabeev and I. Latyshev (Langswap), Interspeech 2026,
  https://huggingface.co/datasets/langswap/dialogs-ru-emotional-conversations, licensed under OpenRAIL; speaker Masha
  (or Dima), who gave written consent for open release and commercial use. This model and its outputs are subject to
  the OpenRAIL use restrictions (included in LICENSE)." / "Ses referansı Dialogs veri setinden (Langswap, OpenRAIL);
  konuşmacı yazılı rıza verdi. Model ve çıktıları OpenRAIL kullanım kısıtlamalarına tabidir (tıbbi tavsiye dahil)."

### `kazakh_f1`, `kazakh_m1`: KazakhTTS2 (Kazakh)

- **Corpus:** ISSAI KazakhTTS / KazakhTTS2, Institute of Smart Systems and Artificial Intelligence, Nazarbayev
  University. Official copies: https://huggingface.co/datasets/issai/KazakhTTS (`ISSAI_KazakhTTS2.tar.gz`, 36 GB in
  six parts) and https://github.com/IS2AI/Kazakh_TTS. F1: female, 44, 14 years of narration; M1: male, 46, 12 years
  (news reading; both from the first KazakhTTS release, 22.05 kHz).
- **Read from:** [ai4kazakh/ISSAI_KazakhTTS2](https://huggingface.co/datasets/ai4kazakh/ISSAI_KazakhTTS2), revision
  `bcf754ffb407` (`bcf754ffb4074e48e6dae6aa7e0b83630fd5410c`): a third-party parquet copy without a card, used because
  an HTTP range request can reach one clip in it (the official tar.gz cannot be read partially). The file names
  (`inform_<article>_<segment>.wav`) are the corpus's own. If the voice wins, take the training data from the official
  release.
- **Licence:** CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ (KazakhTTS paper: free for academic and
  commercial use; LICENSE of github.com/IS2AI/Kazakh_TTS). The HF repo `issai/KazakhTTS` carries no licence tag.
- **Consent:** KazakhTTS (Interspeech 2021, https://arxiv.org/abs/2104.08459, section 3): the project had the approval
  of the Nazarbayev University research ethics committee, and each speaker participated voluntarily and was "informed
  of the data collection and use protocols through a consent form". KazakhTTS2 (LREC 2022,
  https://arxiv.org/abs/2201.05771, section 3.3) says the same of its speakers. The form itself is not published: it
  is not known whether it covers cloning the voice to speak another language. Before a product voice is built on F1 or
  M1, ask ISSAI for written confirmation.
- **Obligations:** CC BY 4.0 section 3(a): attribution including a link to the material (the official release
  above), keep any copyright and disclaimer notice supplied with it, a link to the licence, indicate changes. CC BY 4.0
  does not license personality rights (section 2(b)(1)); those rest on the speakers' consent above.
- **Risk:** low accent risk (Turkic language, read speech); flat news-reading style.
- **Attribution / model card:** "Voice reference: KazakhTTS2 corpus, speaker F1 (or M1). Mussakhojayeva, S.,
  Janaliyeva, A., Mirzakhmetov, A., Khassanov, Y., Varol, H.A. (2021) KazakhTTS: An Open-Source Kazakh Text-to-Speech
  Synthesis Dataset. Proc. Interspeech 2021, 2786-2790, doi: 10.21437/Interspeech.2021-2124; Mussakhojayeva, S.,
  Khassanov, Y., Varol, H.A. (2022) KazakhTTS2: Extending the Open-Source Kazakh TTS Corpus With More Data, Speakers,
  and Topics. Proc. LREC 2022, 5404-5411. The Institute of Smart Systems and Artificial Intelligence
  (issai.nu.edu.kz), Nazarbayev University, Kazakhstan. https://github.com/IS2AI/Kazakh_TTS (dataset:
  https://huggingface.co/datasets/issai/KazakhTTS). Licensed under CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/); changes: <list>." / "Ses referansı KazakhTTS2 (ISSAI, Nazarbayev
  Üniversitesi, https://github.com/IS2AI/Kazakh_TTS), CC BY 4.0; konuşmacı rıza formuyla katıldı."

### `thorsten`: Thorsten-Voice (German)

- **Corpus:** [Thorsten-Voice/TV-44kHz-Full](https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full), revision
  `2b61b98fa8f9` (`2b61b98fa8f99abd1ce1587b4bf413d6ebc217d5`), doi 10.57967/hf/3290. One male speaker, 44.1 kHz,
  denoised, -24 dB. Neutral clip from TV-2022.10-Neutral (natural flow, Rode Podcaster); expressive clip from
  TV-2021.06-Emotional, style "amused" (the card warns that emotional recordings may end early; fetch_refs checks the
  last 50 ms).
- **Licence:** CC0 1.0, https://creativecommons.org/publicdomain/zero/1.0/ (dataset card and the CC0-1.0 LICENSE of
  github.com/thorstenMueller/Thorsten-Voice). The waiver is the speaker's own, and CC0 1.0 covers performer and
  publicity rights as far as he holds them.
- **Consent:** Thorsten Müller recorded and released his own voice; in the card's disclaimer he describes himself as
  "just an open source speech technology enthusiast who donates his voice", and the project's goal is a free German
  TTS voice for every project without licence restrictions (GitHub README).
- **Obligations:** none. Attribution is a courtesy.
- **Risk:** German accent in Turkish; not a professional speaker (his own words).
- **Attribution / model card:** "Voice reference: Thorsten-Voice dataset by Thorsten Müller
  (https://www.Thorsten-Voice.de), TV-44kHz-Full, doi: 10.57967/hf/3290, CC0 1.0. Thank you, Thorsten." / "Ses
  referansı Thorsten Müller'in kendi bağışladığı kayıtlardan (CC0)."

## (c) Controls

| control | voice | licences and attribution |
|---|---|---|
| then-released model (s2): LoRA `/data/exp/voxcpm2-alania-s2/ckpt/step_0000600`, no reference | Antalia | VoxCPM2 (Apache-2.0); Antalia (Patientdesk.ai), Antalia Turkish single-speaker scripted speech, CC BY 4.0, https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus (professional voice actor, commercial agreement and an addendum permitting public redistribution); Alania Turkish Synthetic Speech by PatientDesk AI (https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr), CC BY 4.0, with FineWeb-2 text (ODC-By 1.0, Common Crawl terms). As in `MODEL_CARD.two-stage.template.md`. |
| base VoxCPM2 cloning an Antalia clip | Antalia | VoxCPM2 (Apache-2.0); the Antalia attribution above. |
| stage-1 LoRA `/data/exp/voxcpm2-alania-s1/ckpt/step_0002000` + a designed reference | designed | VoxCPM2 (Apache-2.0); the Alania attribution above; (a) for the reference. |

## Model card carry-over (whichever voice wins)

1. The attribution text of the winning entry above, plus VoxCPM2 (Apache-2.0) and, if the stage-1 LoRA is in the
   chain, Alania (CC BY 4.0) and the FineWeb-2 note.
2. Disclosure: "Bu bir sentetik (yapay) sestir." / "This is a synthetic voice." Mark generated audio machine-readably
   (EU AI Act Article 50; e.g. AudioSeal, MIT).
3. Acceptable use as in the current card: no impersonation of the speaker or any real person, no passing synthetic
   speech off as a real recording, no fraud or political robocalls. For Masha / Dima these are licence conditions
   (OpenRAIL, flow-down to users and outputs), not only requests: ship the Dialogs LICENSE.md text with the model.
4. KazakhTTS: written confirmation from ISSAI on cloning F1 / M1 into Turkish before publishing a product voice.

## Not in the contest, and why

- VCTK, LibriVox derivatives (Hi-Fi TTS, LibriTTS-R), LJSpeech: permissive licences but no consent of the speakers to
  voice cloning.
- Celebrity / YouTuber sets, Appen Turkish 700 h: no consent or unclear licence.
- Common Voice Turkish: CC0 but noisy (UTMOS median 2.5); usable for training stage 1, not as a voice.
- ElevenLabs outputs: their terms forbid training other models on them.
