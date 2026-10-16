import gradio as gr

from trnorm import inverse_normalize, normalize, normalize_for_eval

EXAMPLES = [
    "Randevunuz 26.09.2027'de saat 10.00'da, 3. katta.",
    "SGK'ye 15.00'te başvurun; ücret 1.250,75 TL, %20'si peşin.",
    "Sipariş no 584213, kargo 0555 555 55 55'i arayacak.",
    "TBMM'de XIX. yüzyıl tartışıldı, ABD'den 4'er kişi geldi.",
    "Araç 120 km/sa hızla gidiyordu, hava 38°C'ydi.",
]

SPOKEN_EXAMPLES = [
    "saat on dört otuzda yüzde on beşlik indirim başlıyor",
    "sipariş numaranız beş sekiz dört iki bir üç",
    "bin dokuz yüz yirmi üçte cumhuriyet ilan edildi",
]


def run(text: str):
    return normalize(text), normalize_for_eval(text)


with gr.Blocks(title="trnorm") as demo:
    gr.Markdown(
        "# trnorm — Türkçe metin normalizasyonu\n"
        "Yazılı metni TTS için okunuş formuna çevirir, ekleri okunuşa göre uyumlar "
        "(`15.00'de` → *on beşte*, `SGK'ye` → *se ge kaya*). "
        "[Kaynak kod ve ölçümler](https://github.com/ozguradmin/ahenk-turkish-tts)"
    )
    with gr.Tab("Yazı → Okunuş (TTS)"):
        inp = gr.Textbox(label="Yazılı metin", lines=3, value=EXAMPLES[0])
        out = gr.Textbox(label="Okunuş formu (TTS girdisi)", lines=3)
        ev = gr.Textbox(label="Değerlendirme formu (WER/CER için)", lines=2)
        gr.Button("Normalize et", variant="primary").click(run, inp, [out, ev])
        gr.Examples(EXAMPLES, inp)
    with gr.Tab("Okunuş → Yazı (ASR çıktısı)"):
        sp = gr.Textbox(label="ASR çıktısı (okunuş formu)", lines=3, value=SPOKEN_EXAMPLES[0])
        wr = gr.Textbox(label="Rakamlı yazı", lines=3)
        gr.Button("Dönüştür", variant="primary").click(inverse_normalize, sp, wr)
        gr.Examples(SPOKEN_EXAMPLES, sp)

if __name__ == "__main__":
    demo.launch()
