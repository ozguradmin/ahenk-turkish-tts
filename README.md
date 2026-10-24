# Ahenk: Turkish text-to-speech

**English** · [Türkçe](README.tr.md)

Open Turkish speech tools: Ahenk (large model, GPU), Ahenk Cep (8.6M-parameter small model, CPU), trnorm (Turkish
text normalisation) and the TR-Zor / TR-Soru test sets.

> **Synthetic voice.** Ahenk and Ahenk Cep speak in one designed voice (f022); it is not a real person's voice or
> recording. Say that it is AI-generated when you share what you make (EU AI Act Art. 50).

[Try it in your browser](https://ozguradmin.github.io/ahenk-turkish-tts/) · [Try Ahenk (2B) on Hugging Face](https://huggingface.co/spaces/ozgurguler/ahenk-turkish-tts) · [Ahenk Okur: listen to documents](https://ozguradmin.github.io/ahenk-cep-pdfreader/) · [Ahenk model card](https://huggingface.co/ozgurguler/ahenk-tts-tr) ·
[Ahenk Cep model card](https://huggingface.co/ozgurguler/ahenk-cep-tr)

## Listen

GitHub does not play audio in the page: the links open the WAV files on Hugging Face. The model cards have the same
samples with players. You can try Ahenk Cep in your browser, no install: [ozguradmin.github.io/ahenk-turkish-tts](https://ozguradmin.github.io/ahenk-turkish-tts/).

| What it shows | Text (as written) | Model | Audio |
|---|---|---|---|
| Laughing | Hahaha, bu fıkrayı daha önce hiç duymamıştım! | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_11_laughing.wav) |
| Cheerful | Harika haber! Sınavı geçtim, akşam hep birlikte kutlayalım! | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_07_cheerful.wav) |
| Calm | Derin bir nefes alın, her şeyi adım adım birlikte halledeceğiz. | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_09_calm.wav) |
| Plain reading | Eskişehir'den İzmir'e otobüsle gittik. | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_01_plain_ozel04.wav) |
| Money (trnorm: "doksan dokuz lira doksan dokuz kuruşa") | İndirimle 99,99 TL'ye düştü. | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_02_numbers_para09.wav) |
| Question pair, the statement first | Cihazın garanti süresi 2 yıl. | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_03a_statement_garanti.wav) |
| Question pair, the same sentence with "mı?" | Cihazın garanti süresi 2 yıl mı? | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_03b_question_garanti.wav) |
| Long sentence, not in training (12 s) | Karşılaştırma isteyenlere eski ve yeni paketi yan yana anlatırız; farkı müşteri kendisi görür. Akşam olduğunda … | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_04_long_heldout583922.wav) |
| Angry | Yeter artık! Üç gündür aynı sorun için arıyorum ve hâlâ kimse ilgilenmedi! | Ahenk | [WAV](https://huggingface.co/ozgurguler/ahenk-tts-tr/resolve/main/samples/big_06_angry.wav) |
| The same long sentence | Karşılaştırma isteyenlere eski ve yeni paketi yan yana anlatırız; … | Ahenk Cep | [WAV](https://huggingface.co/ozgurguler/ahenk-cep-tr/resolve/main/samples/small_03_long_heldout583922.wav) |

How they were picked: the plain, money, laughing and emotion clips were chosen by the listener in blind listening.
The question pair and the long sentence were chosen by automatic scores (word-perfect, highest UTMOS). All are
existing evaluation clips, not rendered again.

## What is here

| Component | What it does | Size, hardware | Licence |
|---|---|---|---|
| [Ahenk](https://huggingface.co/ozgurguler/ahenk-tts-tr) | Turkish LoRA for VoxCPM2: question intonation, 6 emotions and laughter, Turkish or English instructions. Recipe: [`tts/voxcpm`](tts/voxcpm) | 2B + 36M LoRA, GPU | weights CC BY 4.0 |
| [Ahenk Cep](https://huggingface.co/ozgurguler/ahenk-cep-tr) | Small student model that learned from Ahenk's voice (EMA Lightning architecture). Code: [`tts/small`](tts/small) | 8.6M, 34 MB, RTF ~0.2 on CPU | weights CC BY 4.0 |
| [Ahenk Okur](https://ozguradmin.github.io/ahenk-cep-pdfreader/) | Reads PDF, Word, EPUB, plain text and photos (with OCR) aloud in the browser with Ahenk Cep. Book-length documents are made piece by piece and can be heard while they are being made; the work resumes after the tab is closed; the result downloads as MP3 or WAV. Code: [ahenk-cep-pdfreader](https://github.com/ozguradmin/ahenk-cep-pdfreader) | browser, on the device | Apache-2.0 |
| [`trnorm`](src/trnorm) | Turns written Turkish into its spoken form: numbers, dates, times, money, percentages, abbreviations, phone numbers; suffixes follow the spoken form. No dependencies. | pure Python | Apache-2.0 |
| [`trnorm-js`](https://github.com/ozguradmin/trnorm-js) | trnorm for the browser and Node; the same output as Python (`npm install @ozgurguler/trnorm`) | pure JavaScript | Apache-2.0 |
| [`trtts`](src/trtts) | EMA Lightning + trnorm: a ready Turkish TTS that runs on CPU | 8.6M, CPU | Apache-2.0 |
| [`benchmarks/tts`](benchmarks/tts) | TR-Zor (155 hard sentences, 13 categories), TR-Soru (35 rows of question intonation) and a harness that reads the audio back with ASR | | Apache-2.0 |
| [`benchmarks/asr`](benchmarks/asr) | Turkish ASR comparison on FLEURS-TR and MediaSpeech-TR ([table](benchmarks/asr/results/LEADERBOARD.md)) | CPU | Apache-2.0 |

The browser demo and Ahenk Okur process text with trnorm-js. It gave the same result as the Python version on 742,716
checks, so the browser no longer has to download Python: the first visit fell from 27 MB to 21.4 MB and the page is
ready about twice as fast. The Python version stays as a fallback.

## Quick start

Ahenk Cep, on a CPU:

```bash
pip install torch ema-lightning==1.1.2 "git+https://github.com/ozguradmin/ahenk-turkish-tts"
```

```python
import sys
import soundfile as sf
from huggingface_hub import snapshot_download

d = snapshot_download("ozgurguler/ahenk-cep-tr", allow_patterns=["*.pt", "ahenk_cep.py"])
sys.path.insert(0, d)
from ahenk_cep import AhenkCep

tts = AhenkCep(f"{d}/student.pt", f"{d}/decoder.pt")
audio, sr = tts.say("Randevunuz 14.10.2026 saat 09:30'da; ücret 1.250 TL.")
sf.write("randevu.wav", audio, sr)
```

Setup and emotions for Ahenk (GPU, VoxCPM2) are in its [model card](https://huggingface.co/ozgurguler/ahenk-tts-tr). To try it without a GPU, use the [Hugging Face demo](https://huggingface.co/spaces/ozgurguler/ahenk-turkish-tts).
Text normalisation only:

```bash
trnorm "Saat 09:30'da 3. kata gelin, ücret 1.250,50 TL."
# Saat dokuz otuzda üçüncü kata gelin, ücret bin iki yüz elli lira elli kuruş.
```

trnorm's rules and examples: [docs/TRNORM.md](docs/TRNORM.md).

## Results

TR-Zor: 155 hard sentences. Held-out: 20 long sentences (12-19 s) of the Antalia dataset, not in the training data of
any of our models. Judge: faster-whisper large-v3-turbo; reference and output normalised with
`trnorm.normalize_for_eval`. WER %, lower is better; in brackets, the number of sentences without a single word
error. UTMOS22 predicts naturalness (1-5, higher is better).

| Model | Size | TR-Zor WER | Held-out WER | UTMOS TR-Zor / held-out |
|---|---|---:|---:|---:|
| **Ahenk** | 2B + LoRA | 2.54 (141/155) | 0.82 (16/20) | 3.70 / 3.14 |
| [Trendyol-TTS](https://huggingface.co/Trendyol/Trendyol-TTS) | 2B + LoRA | 3.98 (134/155) | **0.68** (16/20) | **3.82 / 3.73** |
| **Ahenk Cep** | 8.6M | **2.32** (144/155) | 0.95 (17/20) | 3.44 / 3.18 |
| [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning) + trnorm | 8.6M | 2.43 (145/155) | 1.49 (16/20) | 3.36 / 2.95 |
| [Antalia-2 Mini](https://huggingface.co/cloud0day3/antalia-mini) | 7.6M | 3.30 (140/155) | 1.22 (16/20) | 3.23 / 2.94 |

- Ahenk and Trendyol-TTS were measured on an A100, the small models on a laptop GPU, with the same judge. Scored again
  on the second setup, Ahenk's clips came out the same within 0.01.
- **Where we lose:** Trendyol-TTS is ahead on predicted naturalness, most on long sentences (3.73 vs 3.14); it has a
  lower WER on the held-out sentences (0.68 vs 0.82) and is faster (RTF 1.37 vs 2.28 on an A100). Trendyol speaks in
  its own voice; part of the difference may come from the voice. On CPU, Ahenk Cep is slower than EMA and Antalia
  (RTF 0.20 vs 0.13 and 0.08 on 2 threads).
- trnorm's effect: EMA Lightning's TR-Zor error is 14.0 % with its own front-end and 1.9 % with trnorm (this row was
  measured on CPU with an int8 judge, do not compare it directly with the table above;
  [details](benchmarks/tts/results/tr_zor/RESULTS.md)).
- These are automatic metrics. No MOS study was done.

**Blind listening.** All done with one native listener (the project owner), choosing the best clip for each sentence;
it is a signal, not a statistical result.
- 10 sentences, four systems in random order, equal loudness: Ahenk 8, Ahenk Cep 2 (both questions), EMA Lightning 0,
  Antalia-2 Mini 0.
- Language of the emotion instruction, 20 decisions: for pure emotion the English instruction 6/6; for sentences with
  a word starting with "c" in their first two words, the Turkish instruction 10/10. `instruction_pick` in Ahenk works
  from this result.

## Limitations

- **Questions that should rise at the end do not:** questions without "mı", such as "Yarın geliyorsun?", end like a
  statement, in both models. Questions built with "mı" are right.
- **"c" after an English instruction:** in Ahenk a leading "c" may be read like /k/, /s/ or /z/ ("Cam" → "Kam").
  `instruction_pick` switches to a Turkish instruction; if needed, use `prefer="tr"`.
- **"Phone-like" timbre:** in some of Ahenk's emotional clips the voice is bright and crackly (7 of 20 clips in a probe
  with English instructions). Try another seed.
- **Breath in pauses:** stays in Ahenk; `pause_clean` in the package turns it down.
- **Ahenk Cep's styles are weak:** "sad" is only 1.9 semitones lower and 8 % slower; no laughter. Not yet distilled
  for phone speed.
- **Ahenk needs a GPU:** RTF 2.28 at 16 steps on an A100, slower than real time.
- **One voice, Turkish only.** trnorm leaves foreign names that are not in its lexicon as written.

## Responsible use

- Do not present generated speech as a real human recording; say that it is synthetic and, if you can, add a
  machine-readable mark (EU AI Act Art. 50). The models have no watermark; [AudioSeal](https://github.com/facebookresearch/audioseal)
  can be added.
- Do not impersonate anyone. VoxCPM2 can clone voices; do not clone anyone's voice without their consent.
- Do not present generated anger, sadness or warmth as a real person's reaction to put pressure on a listener.
- Do not use it for fraud, social engineering or robocalls.

## Licence and credit

- Code, docs, the TR-Zor and TR-Soru sets: [Apache-2.0](LICENSE). Third-party reference texts in the repo come with
  their own licences: [NOTICE](NOTICE).
- Ahenk and Ahenk Cep weights are **CC BY 4.0**. Credit is required when you use or redistribute them. Write:
  "Ahenk (https://github.com/ozguradmin/ahenk-turkish-tts), CC BY 4.0". Also keep the data attributions from the model
  cards: *Alania Turkish Synthetic Speech by PatientDesk AI* and *Alania Turkish Domain Text by PatientDesk AI*
  (CC BY 4.0).
- Base models: VoxCPM2 (OpenBMB) and EMA Lightning, both Apache-2.0.
- Academic citation: [CITATION.cff](CITATION.cff).
- GPU compute: thanks to [CloudRift](https://www.cloudrift.ai/) for sponsoring Ahenk with GPU credits through their AI Builder Grant.
- Not required, but if you use this in a project, we would be glad of a link back or a quick note (destek@ozgurguler.tech).

## Roadmap

- Make Ahenk Cep run in real time on phones (few-step distillation)
- More teacher data for the small model, stronger emotions and question intonation
- A demo page that runs in the browser
- A small, commercially usable Turkish speech-recognition (STT) model

## Repository layout

| Folder | Contents |
|---|---|
| `src/trnorm`, `src/trtts` | text normalisation, EMA Lightning + trnorm |
| `tts/voxcpm` | Ahenk training and evaluation recipe, `instruction_pick`, `pause_clean`, model card |
| `tts/small` | Ahenk Cep training, inference code (`ahenk_cep.py`), model card |
| `benchmarks` | TR-Zor, TR-Soru, the ASR comparison and results |
| `spaces` | Hugging Face demo apps |
| `docs` | working notes |

Contributing: open an issue; sentences that trnorm reads wrong help most (input, expected reading, the output you
got). `python -m pytest -q` must pass.

Contact: destek@ozgurguler.tech
