"""tts/voxcpm/instruction_pick.py (instruction language per line: Turkish when one of the first two words starts with
"c", else English; "Vay" -> "Vaay" when instructed): the listening sentences, Turkish casing, punctuation, custom
instructions."""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VOXCPM_DIR = REPO / "tts" / "voxcpm"
sys.path.insert(0, str(VOXCPM_DIR))

import instruction_pick as ip  # noqa: E402

_spec = importlib.util.spec_from_file_location("publish_hf", VOXCPM_DIR / "publish_hf.py")
pub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pub)

PROBE = [json.loads(x) for x in (VOXCPM_DIR / "probe_tr_instr.jsonl").read_text(encoding="utf-8").splitlines()
         if x.strip()]
RISK = "an English instruction tends to misread it (/k/, /s/, /z/ or /tʃ/)"


def lang(p: ip.Pick, emotion: str) -> str:
    return next(k for k, v in ip.EMOTIONS[emotion].items() if v == p.instruction)


def test_emotions_are_training_wordings_with_their_table_side():
    trained = pub.training_instructions()  # per emotion in file order; the first: the "mid" one the card recommends
    table = json.loads((VOXCPM_DIR / "instructions_tr.json").read_text(encoding="utf-8"))
    assert list(ip.EMOTIONS) == ["angry", "cheerful", "sad", "calm", "surprised", "empathetic", "laughing"]
    for emo, w in ip.EMOTIONS.items():
        assert w["en"] == trained["laughter" if emo == "laughing" else emo][0] and w["tr"] == table[w["en"]]
    assert all(v in ip.EMOTIONS for v in ip.ALIASES.values())


def test_half_of_the_judged_wordings_were_not_emotions_own():
    # the docstring / card say so: only angry and the Cam / Cuma / Cesaret / Cidden lines used EMOTIONS' wordings
    own = {r["id"].split("_")[0] for r in PROBE if r["instruction"] == ip.EMOTIONS[r["emotion"]][r["lang"]]}
    assert own == {"angry", "cam", "cuma", "cesaret", "cidden"}


def test_stdlib_only_so_it_ships_alone():
    mods = {(n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0]
            for n in ast.walk(ast.parse((VOXCPM_DIR / "instruction_pick.py").read_text(encoding="utf-8")))
            if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert mods <= {"__future__", "argparse", "re", "sys", "typing", "unicodedata"}
    assert pub.local_imports(VOXCPM_DIR / "instruction_pick.py") == set()


@pytest.mark.parametrize("name, want", [
    ("canina", "tr"), ("cam", "tr"), ("cuma", "tr"), ("cesaret", "tr"), ("cidden", "tr"),  # Turkish won 10/10
    ("angry", "en"), ("cheerful", "en"), ("sad", "en"),  # English won 6/6
    ("acele", "en"),  # the c is inside the word; the listener took the English calm twice ("Asele" but calm)
    ("empathetic", "en"),  # no c; Turkish picked twice, but in round 2 "all three the same": the rule's 2 misses
])
def test_the_listening_sentences(name, want):
    row = next(r for r in PROBE if r["id"] == f"{name}_en")
    p = ip.pick(row["emotion"], row["text"])
    assert lang(p, row["emotion"]) == want and p.why.startswith(f"{want}:")
    if want == "tr":
        assert "starts with c" in p.why
    assert p.text == row["text"].replace("Vay canına", "Vaay canına")


def test_acele_gets_english_and_prefer_tr():
    text = next(r["text"] for r in PROBE if r["id"] == "acele_en")
    p = ip.pick("calm", text)
    assert p == ("Calm, slow and soothing.", text, "en: no word starting with c in the first two (English won "
                                                   "angry/cheerful/sad by ear); 'acele' has a c inside: English read "
                                                   "'Acele' as 'Asele' (prefer='tr' if the c matters)")
    assert ip.pick("calm", text, prefer="tr") == ("Sakin, yavaş ve yatıştırıcı.", text, "tr: prefer='tr'")
    p = ip.pick("surprised", "Cam kırıldı!", prefer="en")
    assert p.instruction == "Astonished, with rising intonation."
    assert p.why == f"en: prefer='en' ('cam' starts with c, {RISK})"
    assert ip.pick("angry", "Yeter artık!", prefer="tr").instruction == ip.EMOTIONS["angry"]["tr"]
    assert ip.compose(*ip.pick("surprised", "Vay canına!")[:2]) == "(Şaşkın, yükselen bir tonlamayla.)Vaay canına!"


def test_turkish_casing():
    assert ip.tr_lower("İSTANBUL IŞIK İi Iı") == "istanbul ışık ii ıı"
    assert ip.tr_lower("i̇") == "i"
    # str.lower makes "İşte" "i" + combining dot + "şte": two words, and "cevabı" would be the third
    assert lang(ip.pick("angry", "İşte cevabı buldum."), "angry") == "tr"
    assert lang(ip.pick("angry", "CİDDEN Mİ?"), "angry") == "tr"
    # decomposed input (NFC first): "C" + cedilla is "Ç", "I" + dot is "İ"
    assert ip.tr_lower("Çok İyi") == "çok iyi"
    assert lang(ip.pick("angry", "Çok güzel bir haber!"), "angry") == "en"
    assert lang(ip.pick("angry", "İşte cevabı buldum."), "angry") == "tr"
    assert ip.pick("KIZGIN", "Yeter!").instruction == ip.EMOTIONS["angry"]["en"]  # "kizgin" would be unknown
    assert ip.pick("Şaşkın", "Cam!").instruction == ip.EMOTIONS["surprised"]["tr"]
    assert ip.pick(" Laughter ", "Haha!").instruction == ip.EMOTIONS["laughing"]["en"]
    for name in ("SURPRISED", "LAUGHING", "EMPATHETIC", "Calm"):  # tr_lower would make "I" a dotless "ı"
        assert ip.pick(name, "Yeter!").instruction == ip.EMOTIONS[name.lower()]["en"]


@pytest.mark.parametrize("text, want", [
    ('"Cam kırılmış!" dedi.', "tr"), ("— Cidden mi?", "tr"), ("«Cuma» geliyorlar.", "tr"),
    ("... ve cam kırıldı.", "tr"), ("Can'ın evi yandı!", "tr"), ("3 cami var.", "tr"), ("Hahaha, cidden mi?", "tr"),
    ("Bu gece çok güzel.", "en"), ("Bu üçüncü aramam!", "en"),  # a c inside a word does not count, as in "Acele"
    ("Çok güzel bir haber!", "en"), ("Hiç beklemiyordum, cidden.", "en"), ("Ççç, sus!", "en"),  # ç is not c
])
def test_punctuation_and_c_not_cedilla(text, want):
    assert lang(ip.pick("angry", text), "angry") == want


@pytest.mark.parametrize("text, match", [("(?) ", "starts with"), ("", "nothing to read"), ("   ", "nothing to read"),
                                         ("... — !", "nothing to read")])
def test_instructed_text_needs_a_word_and_no_leading_bracket(text, match):
    with pytest.raises(ValueError, match=match):
        ip.pick("angry", text)


def test_vay_respelled_only_when_instructed():
    p = ip.pick("surprised", "Vay vay! VAY be, vay'ına. Vaybe Kovay Vaay")
    assert p.text == "Vaay vaay! VAAY be, vaay'ına. Vaybe Kovay Vaay" and p.why.endswith("'Vay' -> 'Vaay' x4")
    assert ip.pick("", " Vay canına! ") == ("", " Vay canına! ", "no instruction: plain text, as it is")
    assert ip.pick(None, "Vay").text == "Vay"
    assert ip.pick("", "") == ("", "", "no instruction: plain text, as it is")
    p = ip.pick("", "(Sad.)Merhaba")  # plain text is passed on as it is, with a warning
    assert p.text == "(Sad.)Merhaba" and p.why.endswith("it starts with '(': the model takes that for an instruction")
    assert ip.pick("angry", "  Yeter!").text == "Yeter!"  # "(instruction)text": no space, as trained
    assert ip.respell("Vaay") == ("Vaay", 0)


def test_custom_instruction_is_used_as_given():
    p = ip.pick("Fısıldayarak, çok alçak sesle.", "Vay, cam kırılmış!", prefer="en")
    assert p.instruction == "Fısıldayarak, çok alçak sesle." and p.text == "Vaay, cam kırılmış!"
    assert f"custom instruction, as given ('cam' starts with c, {RISK}; a Turkish one does not)" in p.why
    assert "prefer='en' ignored" in p.why
    assert ip.pick("(Whispering softly.)", "Yeter!").instruction == "Whispering softly."
    assert ip.pick("Fısıldayarak.", "Yeter!").why == "custom instruction, as given"
    assert ip.pick("Fısıldayarak!", "Yeter!").instruction == "Fısıldayarak!"
    for bad in ("angy", "Fısıldayarak", "()", "Sad (very) and slow."):
        with pytest.raises(ValueError):
            ip.pick(bad, "Yeter!")
    for bad in ("angry.", "(Angry)", "Laughing!", "Kızgın…"):  # an emotion name, not a custom instruction
        with pytest.raises(ValueError, match="names an emotion"):
            ip.pick(bad, "Yeter!")
    with pytest.raises(ValueError, match="prefer"):
        ip.pick("angry", "Yeter!", prefer="de")


def test_compose_and_cli(capsys):
    assert ip.compose("", "Metin.") == "Metin." and ip.compose("Sakin.", "Metin.") == "(Sakin.)Metin."
    assert ip.main(["surprised", "Cidden mi?"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["(Şaşkın, yükselen bir tonlamayla.)Cidden mi?", f"# tr: 'cidden' starts with c, {RISK}"]
    with pytest.raises(SystemExit):
        ip.main(["angy", "Yeter!"])
