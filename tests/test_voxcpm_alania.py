"""tts/voxcpm/prepare_alania.py: filters, contamination guard, selection order, refs, resume, on synthetic rows and a
fake Hub source (stdlib only: no pyarrow, numpy, soundfile or network)."""

import json
import math
import os
import subprocess
import sys
import wave
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VOXCPM = REPO / "tts" / "voxcpm"
sys.path[:0] = [str(VOXCPM), str(REPO / "src")]

import prepare_alania as pa  # noqa: E402

GOOD = {"render_mode": "reference", "qc_cer": 0.0, "qc_utmos": 4.0, "gender": "female", "duration": 5.0,
        "text_source": "domain", "text_category": "banking"}


def row(i, text=None, **kw):
    t = text or f"Bu cümle {i} numaralı satır için yazıldı."
    return {"id": f"r{i:04d}", "voice_id": f"v{i % 7}", "text": t, "text_spoken": t, **GOOD, **kw}


def irow(i, style="sad", instruction="Speak in a sad, quiet and subdued voice.", **kw):
    """A reference+instruction row (qc_utmos 4.0 unless given)."""
    return row(i, render_mode=pa.INSTRUCTION_MODE, style=style, instruction=instruction, **kw)


def test_import_needs_no_heavy_deps():
    code = (f"import sys; sys.path[:0] = [{str(VOXCPM)!r}, {str(REPO / 'src')!r}]; import prepare_alania; "
            "bad = {'pyarrow', 'numpy', 'soundfile', 'soxr', 'torch', 'torchaudio', 'scipy', 'huggingface_hub'} "
            "& set(sys.modules); assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_help_lists_interface():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run([sys.executable, str(VOXCPM / "prepare_alania.py"), "--help"], capture_output=True,
                         text=True, encoding="utf-8", env=env, check=True).stdout
    for flag in ("--out", "--hours", "--min-utmos", "--max-cer", "--gender", "--modes", "--min-dur", "--max-dur",
                 "--max-per-voice", "--fineweb-share", "--ref-fraction", "--exclude-manifest", "--revision",
                 "--workers", "--seed", "--max-files", "--row-groups", "--restart", "--styles",
                 "--min-utmos-instruction", "--max-style-share", "--instruction-format"):
        assert flag in out, flag


def test_columns_never_degraded_audio():
    assert "audio_channel" not in pa.META_COLUMNS + pa.AUDIO_COLUMNS
    assert "audio" not in pa.META_COLUMNS and pa.AUDIO_COLUMNS == ["id", "audio"]
    assert {"license", "text_doc_id"} <= set(pa.META_COLUMNS)  # per-clip provenance of the FineWeb-2 text
    assert {"instruction", "style"} <= set(pa.META_COLUMNS) and len(pa.KNOWN_STYLES) == 18
    assert len(pa.REVISION) == 40 and "CC BY 4.0" in pa.ATTRIBUTION and pa.REPO_ID in pa.ATTRIBUTION


@pytest.mark.parametrize("n", [0, 1, 2, 3, 10, 655])
def test_spread_order_is_a_spread_permutation(n):
    order = pa.spread_order(n)
    assert sorted(order) == list(range(n))
    if n == 655:
        assert order[:3] == [0, 404, 153]
        first = sorted(order[:20])  # 20 files cover the whole list: no gap above 2.5 x n/20
        assert max(b - a for a, b in zip(first, first[1:] + [n + first[0]])) < 2.5 * n / 20


def test_cheap_filters_each_reason():
    f = pa.Filters(gender="female")
    assert pa.cheap_reason(row(1), f) == ""
    cases = {"render_mode": {"render_mode": "voice-design"}, "qc_cer": {"qc_cer": 0.01},
             "qc_utmos": {"qc_utmos": 3.69}, "gender": {"gender": "male"}, "too_short": {"duration": 1.2},
             "too_long": {"duration": 20.5}, "empty_text": {"text_spoken": "  "}}
    for reason, kw in cases.items():
        assert pa.cheap_reason(row(1, **kw), f) == reason, reason
    for bad in ({"qc_utmos": math.nan}, {"qc_utmos": None}, {"qc_cer": math.nan}, {"qc_cer": None},
                {"duration": math.nan}, {"duration": None}):
        assert pa.cheap_reason(row(1, **bad), f), bad
    assert pa.cheap_reason(row(1, gender="male"), pa.Filters(gender="any")) == ""
    two = pa.Filters(modes=("reference", "reference+instruction"))
    assert pa.cheap_reason(row(1, render_mode="reference+instruction"), two) == ""


def test_modes_parse_commas_and_spaces(tmp_path):
    args = pa.parse_args(["--out", str(tmp_path), "--modes", "reference,reference+instruction", "voice-design"])
    assert args.modes == ["reference", "reference+instruction", "voice-design"]
    assert pa.parse_args(["--out", str(tmp_path)]).modes == ["reference"]
    with pytest.raises(SystemExit):
        pa.parse_args(["--out", str(tmp_path), "--fineweb-share", "1.5"])


def test_instruction_rows_cheap_filters():
    f = pa.Filters(modes=("reference", pa.INSTRUCTION_MODE))
    assert pa.cheap_reason(irow(1, qc_utmos=3.5), f) == ""  # --min-utmos-instruction 3.5 ...
    assert pa.cheap_reason(irow(1, qc_utmos=3.49), f) == "qc_utmos"
    assert pa.cheap_reason(row(1, qc_utmos=3.6), f) == "qc_utmos"  # ... plain rows keep --min-utmos 3.7
    assert pa.cheap_reason(irow(1), pa.Filters()) == "render_mode"  # default: reference rows only
    assert pa.cheap_reason(irow(1, qc_utmos=3.6), pa.Filters(modes=f.modes, min_utmos_instruction=3.7)) == "qc_utmos"
    low = pa.Filters(modes=f.modes, min_utmos=3.0)  # the default follows --min-utmos: never stricter than plain rows
    assert low.min_utmos_instruction == 2.8 and pa.Filters(min_utmos=3.9).min_utmos_instruction == 3.7
    assert pa.cheap_reason(irow(1, qc_utmos=2.85), low) == ""
    assert pa.cheap_reason(row(1, qc_utmos=2.95), low) == "qc_utmos"
    sad = pa.Filters(modes=f.modes, styles=("sad",))
    assert pa.cheap_reason(irow(1), sad) == "" and pa.cheap_reason(irow(1, style="cheerful"), sad) == "style"
    assert pa.cheap_reason(irow(1, style=None), sad) == "style"  # an instruction row without a style
    assert pa.cheap_reason(row(1), sad) == ""  # rows of other modes have no style: --modes decides
    assert pa.cheap_reason(irow(1, style="cheerful", qc_cer=0.5), sad) == "style"  # first failing filter


def test_instruction_options_parse_and_validate(tmp_path):
    base = ["--out", str(tmp_path)]
    a = pa.parse_args(base)
    assert (a.styles, a.min_utmos_instruction, a.max_style_share, a.instruction_format) == ([], 3.5, 1.0, "prefix")
    for utmos, want in ((["--min-utmos", "3.0"], 2.8), (["--min-utmos", "3.9"], 3.7),  # --min-utmos - 0.2 ...
                        (["--min-utmos", "3.0", "--min-utmos-instruction", "3.4"], 3.4)):  # ... unless given
        assert pa.parse_args(base + utmos).min_utmos_instruction == want, utmos
    a = pa.parse_args(base + ["--modes", "reference,reference+instruction", "--styles", "sad,cheerful", "calm",
                              "--max-style-share", "0.1", "--instruction-format", "drop"])
    assert a.styles == ["calm", "cheerful", "sad"] and a.max_style_share == 0.1 and a.instruction_format == "drop"
    only = ["--modes", "reference+instruction"]
    assert pa.parse_args(base + only + ["--styles", "sad,calm", "--max-style-share", "0.6"]).max_style_share == 0.6
    for bad in (only + ["--styles", "happy"],  # not a style of the dataset
                ["--styles", "sad"], ["--max-style-share", "0.5"],  # without reference+instruction rows
                ["--modes", "reference,reference+instruction", "--max-style-share", "0"],
                only + ["--styles", "sad,calm", "--max-style-share", "0.5"],  # instruction rows only: 2 x 0.5 <= 1
                only + ["--max-style-share", "0.05"],  # 18 styles x 0.05 <= 1
                ["--instruction-format", "suffix"]):
        with pytest.raises(SystemExit):
            pa.parse_args(base + bad)


def test_manifest_text_instruction_prefix_or_drop(tmp_path):
    ins = {"id": "i", "wav": "i.wav", "text_spoken": "Hesabınız kapandı.", "duration": 4.0, "voice_id": "v",
           "mode": pa.INSTRUCTION_MODE, "style": "sad", "instruction": "Speak in a sad (quiet) voice."}
    plain = {**ins, "id": "p", "wav": "p.wav", "mode": "reference", "style": None, "instruction": None}
    design = {**ins, "id": "d", "wav": "d.wav", "mode": "voice-design", "style": None, "instruction": "A deep voice."}
    old = {k: v for k, v in plain.items() if k not in ("style", "instruction")}  # a clip of a run before the columns
    assert pa.manifest_text(ins) == "(Speak in a sad quiet voice.)Hesabınız kapandı."  # VoxCPM2: no () inside
    assert pa.manifest_text(ins, "drop") == "Hesabınız kapandı." == pa.manifest_text({**ins, "instruction": " "})
    assert all(pa.manifest_text(c) == "Hesabınız kapandı." for c in (plain, design, old))  # no voice description
    rows = {Path(r["audio"]).stem: r["text"] for r in pa.manifest_rows([ins, plain, old], {}, tmp_path, 1)}
    assert rows == {"i": "(Speak in a sad quiet voice.)Hesabınız kapandı.", "p": "Hesabınız kapandı."}
    assert pa.strip_control(pa.manifest_text(ins)) == "Hesabınız kapandı."
    assert pa.strip_control("(Not a prefix) Metin.") == "(Not a prefix) Metin."  # a space after it: ordinary text


def test_contamination_guard_sees_the_sentence_not_the_instruction(tmp_path):
    held, short = "Randevunuz yarın saat onda başlıyor.", "Tamam, anlaşıldı."
    man = tmp_path / "val.jsonl"  # an instruction manifest as --exclude-manifest: its bare sentences are protected
    man.write_text("".join(json.dumps({"audio": "/x.wav", "text": f"(Speak calmly.){t}"}, ensure_ascii=False) + "\n"
                           for t in (held, short)), encoding="utf-8")
    texts = pa.protected_texts(man)
    assert {held, short, f"(Speak calmly.){held}"} <= set(texts)
    g = pa.build_guard(texts + ["Speak calmly and slowly, in a soothing voice."])
    sel = pa.Selector(pa.Filters(modes=("reference", pa.INSTRUCTION_MODE), max_per_voice=0, fineweb_share=1.0), g,
                      1e9)
    t = sel.take([irow(1, text=held, instruction="Speak calmly."),  # its sentence is protected, whatever the prefix
                  irow(2, text=short), row(3, text=short),  # a short protected sentence, whole
                  irow(4, text="Bugün hava çok güzel ve güneşli.",  # only its instruction is "protected": kept
                       instruction="Speak calmly and slowly, in a soothing voice.")])
    assert t.drops["contamination"] == 3 and [r["id"] for r in t.picked] == ["r0004"]
    no_strip = pa.build_guard([f"(Speak calmly.){x}" for x in (held, short)])  # why protected_texts strips the prefix
    assert not no_strip.hit(pa.eval_key(held)) and not no_strip.hit(pa.eval_key(short))


def test_style_share_holds_and_a_corpus_of_instruction_rows_only_grows():
    f = pa.Filters(modes=("reference", pa.INSTRUCTION_MODE), max_per_voice=0, fineweb_share=1.0, max_style_share=0.1)
    sel, picked = pa.Selector(f, pa.Guard(), 1e9), []
    for n in range(6):  # per file 20 plain rows, 20 sad, 10 calm, 10 warm: every style hits the cap
        t = sel.take([row(n * 100 + j, voice_id=f"v{j}", duration=2.0 + j % 5) if j % 3 == 0 else
                      irow(n * 100 + j, voice_id=f"v{j}", style="sad" if j % 2 == 0 else ["calm", "warm"][j // 2 % 2],
                           duration=2.0 + j % 5) for j in range(60)])
        picked += t.picked
        assert all(s <= 0.1 * sel.total_s + 6.0 for s in sel.style_s.values())  # one clip (<= 6 s) of overshoot
        assert t.drops["style_share"] > 0
    assert sel.style_s["sad"] > 0.09 * sel.total_s  # the share is used, not just respected
    assert sum(r["render_mode"] == "reference" for r in picked) == 120 and set(sel.style_s) == {"sad", "calm", "warm"}
    restored = pa.Selector(f, pa.Guard(), 1e9)  # a resumed run counts the committed clips' styles again
    restored.restore([pa.clip_record(r, r["duration"], "f") for r in picked])
    assert restored.style_s == sel.style_s and restored.total_s == sel.total_s
    styles = ["sad", "calm", "cheerful", "warm"]
    only = pa.Selector(pa.Filters(modes=(pa.INSTRUCTION_MODE,), max_per_voice=0, fineweb_share=1.0,
                                  max_style_share=0.3), pa.Guard(), 1e9)
    only.take([irow(i, voice_id=f"v{i}", style=styles[i % 4], duration=3.0 + i % 3) for i in range(400)])
    assert only.total_s > 1000 and all(s <= 0.3 * only.total_s + 5.0 for s in only.style_s.values())
    nocap = pa.Selector(pa.Filters(modes=(pa.INSTRUCTION_MODE,), max_per_voice=0, fineweb_share=1.0), pa.Guard(), 1e9)
    assert len(nocap.take([irow(i, voice_id=f"v{i}") for i in range(50)]).picked) == 50  # 1 = no cap


def test_selector_dedup_voice_cap_and_order():
    sel = pa.Selector(pa.Filters(max_per_voice=2, fineweb_share=1.0), pa.Guard(), 1e9)
    rows = [row(1, voice_id="a", qc_utmos=3.8), row(2, voice_id="a", qc_utmos=4.5),
            row(3, voice_id="a", qc_utmos=4.0),
            row(4, voice_id="b", text="Aynı cümle burada.", text_spoken="Aynı cümle burada."),
            row(5, voice_id="c", text="Aynı cümle, burada!", text_spoken="aynı cümle burada"),  # same normalized text
            row(6, voice_id="d", text="Hesap 5 TL.", text_spoken="Hesap beş lira."),
            row(7, voice_id="e", text="Hesap beş lira!", text_spoken="Hesap beş lira nin.")]  # text = 6's spoken
    t = sel.take(rows)
    ids = [r["id"] for r in t.picked]
    assert ids[:2] == ["r0002", "r0003"]  # best qc_utmos first; r0001 then hits the voice cap
    assert "r0001" not in ids and t.drops["voice_cap"] == 1
    assert ("r0004" in ids) != ("r0005" in ids) and ("r0006" in ids) != ("r0007" in ids)
    assert t.drops["duplicate_text"] == 2 and not t.truncated
    assert sel.take([row(2)]).drops == Counter({"duplicate_id": 1})  # the same id in a later file: never twice


def test_fineweb_share_holds_at_every_step_and_domain_comes_first():
    sel = pa.Selector(pa.Filters(max_per_voice=0, fineweb_share=0.25), pa.Guard(), 1e9)
    for f in range(5):
        rows = [row(f * 100 + j, voice_id=f"v{j}", text_source="fineweb2" if j % 2 else "domain",
                    duration=2.0 + (j % 5)) for j in range(40)]
        t = sel.take(rows)
        assert sel.web_s <= 0.25 * sel.total_s + 1e-6
        kinds = [pa.is_web(r) for r in t.picked]
        assert kinds == sorted(kinds)  # every domain row before the first fineweb row
        assert t.drops["fineweb_share"] > 0
    assert sel.web_s > 0.2 * sel.total_s  # the share is used, not just respected
    no_web = pa.Selector(pa.Filters(max_per_voice=0, fineweb_share=0.0), pa.Guard(), 1e9)
    assert not any(pa.is_web(r) for r in no_web.take([row(i, text_source="fineweb2") for i in range(5)]).picked)


def test_target_reached_inside_a_file():
    sel = pa.Selector(pa.Filters(max_per_voice=0), pa.Guard(), 12.0)
    t = sel.take([row(i, voice_id=f"v{i}", duration=5.0) for i in range(6)])
    assert len(t.picked) == 3 and t.truncated and t.drops["target_reached"] == 3 and sel.done()
    assert t.rows == 6 and t.seconds == 30.0 and t.drop_s["target_reached"] == 15.0


def test_contamination_guard_tr_zor_and_manifest(tmp_path):
    man = tmp_path / "val.jsonl"
    held = ("Randevunuz yarın saat onda başlıyor. Lütfen kimliğinizi ve sigorta kartınızı yanınızda getirin. "
            "Teşekkürler.")
    man.write_text(json.dumps({"audio": "/x.wav", "text": held, "duration": 9.0}, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    texts = pa.protected_texts(pa.TRZOR) + pa.protected_texts(man)
    first = json.loads(pa.TRZOR.read_text(encoding="utf-8").splitlines()[0])
    assert first["text"] in texts and first["spoken"] in texts
    g = pa.build_guard(texts)
    sel = pa.Selector(pa.Filters(max_per_voice=0, fineweb_share=1.0), g, 1e9)
    bad = [row(1, text="Toplantıya 45 kişi katıldı.", text_spoken="Toplantıya kırk beş kişi katıldı."),  # TR-Zor
           row(2, text="X", text_spoken="toplantıya kırk beş kişi katıldı"),  # spoken form, other punctuation
           row(3, text="Dün akşam toplantıya 45 kişi katıldı ve herkes memnun kaldı."),  # contains it (>= 4 words)
           row(4, text=held),  # the whole held-out text
           row(5, text="Bilginize: lütfen kimliğinizi ve sigorta kartınızı yanınızda getirin."),  # one of its sentences
           row(6, text="Evet.")]  # a short TR-Zor sentence, whole
    ok = [row(7, text="Evet, bu akşam size döneceğim."),  # short protected sentences only match whole rows
          row(8, text="Teşekkürler, iyi günler dilerim.")]
    t = sel.take(bad + ok)
    assert t.drops["contamination"] == 6 and [r["id"] for r in t.picked] == ["r0007", "r0008"]
    assert g.digest() == pa.build_guard(texts).digest() != pa.build_guard(texts[:-1]).digest()


def test_assign_refs_same_voice_3_to_10_s_never_itself():
    clips = [{"id": f"c{i:03d}", "voice_id": f"v{i % 4}", "duration": [2.0, 4.0, 8.0, 12.0, 5.0][i % 5]}
             for i in range(60)] + [{"id": "lonely", "voice_id": "solo", "duration": 5.0}]
    refs = pa.assign_refs(clips, 0.3, 1234)
    by = {c["id"]: c for c in clips}
    assert len(refs) == round(0.3 * len(clips)) and "lonely" not in refs
    for cid, rid in refs.items():
        assert cid != rid and by[cid]["voice_id"] == by[rid]["voice_id"] and 3 <= by[rid]["duration"] <= 10
    assert refs == pa.assign_refs(list(reversed(clips)), 0.3, 1234) != pa.assign_refs(clips, 0.3, 99)
    assert pa.assign_refs(clips, 0.0, 1) == {} and len(pa.assign_refs(clips, 1.0, 1)) == 60
    assert pa.assign_refs(clips, 0.3, 1234, pa.plain_ref) == refs  # clips without a mode: plain, the same refs


def test_assign_refs_never_takes_a_styled_ref():
    """Alania rendered every row of a voice, styled ones too, from the voice's plain reference: a styled clip is no
    ref_audio (it would pair a neutral target with a styled prompt)."""
    clips = [{"id": f"c{i:03d}", "voice_id": f"v{i % 3}", "duration": 4.0 + i % 5,
              "mode": pa.INSTRUCTION_MODE if i % 2 else "reference", "style": ["sad", "slow"][i % 4 // 2] if i % 2
              else None} for i in range(60)]
    clips += [{"id": f"s{i}", "voice_id": "styled-only", "duration": 5.0, "mode": pa.INSTRUCTION_MODE, "style": "sad"}
              for i in range(2)]
    by = {c["id"]: c for c in clips}
    refs = pa.assign_refs(clips, 1.0, 1234, pa.plain_ref)
    assert len(refs) == 60 and all(by[r]["mode"] == "reference" for r in refs.values())
    assert not {"s0", "s1"} & set(refs)  # a voice with styled clips only gets no ref
    assert sum(by[c]["mode"] == pa.INSTRUCTION_MODE for c in refs) == 30  # styled rows get a plain ref
    assert any(by[r]["mode"] == pa.INSTRUCTION_MODE for r in pa.assign_refs(clips, 1.0, 1234).values())  # unfiltered
    plain = [c for c in clips if c["mode"] == "reference"]  # every clip passes: the refs (and RNG draws) without it
    assert pa.assign_refs(plain, 0.3, 7, pa.plain_ref) == pa.assign_refs(plain, 0.3, 7)


def test_attribution_lines_add_fineweb_terms_only_for_web_text():
    domain = {"source": "domain", "license": "CC-BY-4.0"}
    web = {"source": "fineweb2", "license": "CC-BY-4.0 (text: ODC-By 1.0, FineWeb-2)"}
    assert pa.attribution_lines([domain]) == [pa.ATTRIBUTION]
    assert pa.attribution_lines([domain, web]) == [pa.ATTRIBUTION, pa.FINEWEB_NOTE]
    assert "ODC-By 1.0" in pa.FINEWEB_NOTE and "Common Crawl" in pa.FINEWEB_NOTE and "text_doc_id" in pa.FINEWEB_NOTE
    assert pa.attribution_lines([]) == [pa.ATTRIBUTION]


def test_manifest_rows(tmp_path):
    clips = [{"id": "a", "wav": "a.wav", "text_spoken": "bir iki", "duration": 4.0, "voice_id": "v"},
             {"id": "b", "wav": "b.wav", "text_spoken": "lira nin", "duration": 6.5, "voice_id": "v"}]
    rows = pa.manifest_rows(clips, {"b": "a"}, tmp_path, 1)
    by = {Path(r["audio"]).name: r for r in rows}
    assert by["a.wav"] == {"audio": str(tmp_path / "a.wav"), "text": "bir iki", "duration": 4.0}
    assert by["b.wav"]["text"] == "lira nin" and by["b.wav"]["ref_audio"] == str(tmp_path / "a.wav")
    assert by["b.wav"]["ref_duration"] == 4.0 and Path(by["b.wav"]["audio"]).is_absolute()


def test_settings_key_ignores_hours_but_not_filters():
    g = pa.Guard({"x"}, [])
    k = pa.settings_key(pa.Filters(), pa.REVISION, 0, g)
    assert k == pa.settings_key(pa.Filters(), pa.REVISION, 0, pa.Guard({"x"}, []))
    assert k != pa.settings_key(pa.Filters(min_utmos=3.8), pa.REVISION, 0, g)
    assert k != pa.settings_key(pa.Filters(), pa.REVISION, 1, g)
    assert k != pa.settings_key(pa.Filters(), pa.REVISION, 0, pa.Guard({"y"}, []))


def test_settings_key_without_instruction_rows_is_the_one_states_already_hold():
    g = pa.Guard({"x"}, [])
    legacy = ('{"filters": {"fineweb_share": 0.25, "gender": "any", "max_cer": 0.0, "max_dur": 20.0, '
              '"max_per_voice": 40, "min_dur": 1.5, "min_utmos": 3.7, "modes": ["reference"]}, '
              f'"guard": "{g.digest()}", "revision": "{pa.REVISION}", "row_groups": 0}}')
    assert pa.settings_key(pa.Filters(), pa.REVISION, 0, g) == legacy  # as written before the instruction options
    unused = {"styles": ("sad",), "min_utmos_instruction": 3.0, "max_style_share": 0.1}  # select nothing here
    assert pa.settings_key(pa.Filters(**unused), pa.REVISION, 0, g) == legacy
    design = pa.settings_key(pa.Filters(modes=("reference", "voice-design"), **unused), pa.REVISION, 0, g)
    assert not any(f in json.loads(design)["filters"] for f in pa.INSTRUCTION_FIELDS)
    ins = ("reference", pa.INSTRUCTION_MODE)
    k = pa.settings_key(pa.Filters(modes=ins), pa.REVISION, 0, g)
    assert all(f in json.loads(k)["filters"] for f in pa.INSTRUCTION_FIELDS)
    low = json.loads(pa.settings_key(pa.Filters(modes=ins, min_utmos=3.0), pa.REVISION, 0, g))["filters"]
    assert (json.loads(k)["filters"]["min_utmos_instruction"], low["min_utmos_instruction"]) == (3.5, 2.8)  # actual
    for kw in ({"styles": ("sad",)}, {"min_utmos_instruction": 3.6}, {"max_style_share": 0.5}):
        assert pa.settings_key(pa.Filters(modes=ins, **kw), pa.REVISION, 0, g) != k, kw


def test_resume_point_drops_uncommitted_and_redoes_a_cut_file():
    clips = [{"id": i, "duration": 10.0} for i in "abcde"]
    state = {"done": [{"file": "f1", "ids": ["a", "b"]}, {"file": "f2", "ids": ["c"], "truncated": True}]}
    done, kept = pa.resume_point(state, clips, 30.0)  # target met: keep f2 as it is; d, e were never committed
    assert [e["file"] for e in done] == ["f1", "f2"] and [c["id"] for c in kept] == ["a", "b", "c"]
    done, kept = pa.resume_point(state, clips, 100.0)  # --hours grew: f2 was cut by the old target, redo it
    assert [e["file"] for e in done] == ["f1"] and [c["id"] for c in kept] == ["a", "b"]
    with pytest.raises(RuntimeError):
        pa.resume_point(state, clips[:1], 30.0)


def test_report_and_markdown(tmp_path):
    clips = [{"id": f"c{i}", "voice_id": f"v{i % 3}", "gender": "female", "text_spoken": f"saat {i} buçuk",
              "category": "time", "source": "domain", "duration": 3600.0, "src_duration": 3600.0, "qc_utmos": 4.0,
              "license": "CC-BY-4.0"} for i in range(3)]
    done = [{"file": "data/cc-by/train-00000.parquet", "ids": ["c0", "c1", "c2"], "rows": 10, "seconds": 36000,
             "bytes": 2_000_000_000, "drops": {"qc_utmos": 7}, "drop_s": {"qc_utmos": 3600.0}}]
    args = pa.parse_args(["--out", str(tmp_path), "--hours", "3", "--exclude-manifest", str(tmp_path / "v.jsonl")])
    settings = {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if k == "exclude_manifest" else v
                for k, v in vars(args).items()}
    rep = pa.build_report(clips, done, {"c1": "c0"}, settings, pa.Guard({"a b"}, []), 655, True)
    sel = rep["selected"]
    assert sel["hours"] == 3.0 and sel["voices"] == 3 and sel["text_spoken_with_digits"] == 3
    assert rep["drops"] == {"qc_utmos": {"rows": 7, "hours": 1.0}} and rep["dataset"]["revision"] == pa.REVISION
    md = pa.render_markdown(rep)
    assert pa.ATTRIBUTION in md and "**3.0 h, 3 clips, 3 voices**" in md and "| qc_utmos | 7 | 1.0 |" in md
    assert "| CC-BY-4.0 | 3 | 3.0 |" in md and pa.FINEWEB_NOTE not in md and rep["dataset"]["attribution_extra"] == []
    assert sel["by_style"] == {"none": {"clips": 3, "hours": 3.0}} and "| none | 3 | 3.0 |" in md  # clips of old runs
    assert rep["manifest"]["with_instruction"] == 0 and pa.INSTRUCTION_MODE not in md
    web = [{**clips[0], "id": "w", "source": "fineweb2", "license": "CC-BY-4.0 (text: ODC-By 1.0, FineWeb-2)"}]
    rep = pa.build_report(clips + web, done, {}, settings, pa.Guard(), 655, True)
    assert rep["dataset"]["attribution_extra"] == [pa.FINEWEB_NOTE] and pa.FINEWEB_NOTE in pa.render_markdown(rep)
    json.dumps(rep)


# --------------------------------------------------------------------------- end to end with a fake Hub source
FILES, GROUPS, PER_GROUP = 8, 3, 10


class FakeSource:
    """FILES x GROUPS x PER_GROUP rows; fetch_audio writes silent 16 kHz WAVs of the metadata duration."""

    def __init__(self, fail_on: str = ""):
        self.fail_on, self.meta_reads, self.audio_reads = fail_on, [], []

    def list_files(self):
        return [f"data/cc-by/train-{i:05d}.parquet" for i in range(FILES)]

    def read_meta(self, name, row_groups):
        f, rows = int(name[-13:-8]), []
        for g in range(GROUPS if row_groups <= 0 else min(row_groups, GROUPS)):
            for j in range(PER_GROUP):
                k = f * 1000 + g * PER_GROUP + j
                text = f"Dosya {f} grup {g} satır {j} için özgün bir cümle." if j != 7 else "Ortak cümle her dosyada."
                rows.append({"id": f"{k:06x}-0-1", "voice_id": f"d1-{k * 7 % 61:05d}", "text": text,
                             "text_spoken": text, "duration": 2.0 + k % 7, "gender": "female" if j % 2 else "male",
                             "render_mode": "reference" if j % 5 else "voice-design", "style": None,
                             "instruction": None if j % 5 else f"A calm voice, number {k}.",  # voice description
                             "qc_cer": 0.0 if j % 6 else 0.02,
                             "qc_utmos": 3.5 + k % 9 / 10, "text_source": "fineweb2" if j % 3 == 0 else "domain",
                             "text_category": ["number", "money", "time"][j % 3], "row_group": g,
                             "license": "CC-BY-4.0 (text: ODC-By 1.0, FineWeb-2)" if j % 3 == 0 else "CC-BY-4.0",
                             "text_doc_id": f"fw2:<urn:uuid:{k}>" if j % 3 == 0 else f"qwen3:s3-p{k}"})
        self.meta_reads.append(name)
        return rows, None, 1000

    def fetch_audio(self, name, md, picked, wav_dir):
        if name == self.fail_on:
            raise ConnectionError("simulated kill")
        out = {}
        for r in picked:
            assert {"id", "row_group", "duration"} <= set(r)
            p = wav_dir / pa.wav_name(r["id"])
            if not p.exists():
                self.audio_reads.append(r["id"])
                with wave.open(str(p), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(pa.RATE)
                    w.writeframes(b"\0\0" * int(r["duration"] * pa.RATE))
            out[r["id"]] = pa.wav_seconds(p)
        return out, 5000


STYLES3 = {"sad": "Speak in a sad, quiet and subdued voice.", "cheerful": "Bright, upbeat and happy.",
           "calm": "Speak calmly and slowly, in a soothing voice."}


class InstructionSource(FakeSource):
    """FakeSource plus Alania's third mode: rows j % 5 in (2, 3) of a group are reference+instruction rows of one of
    three styles."""

    def read_meta(self, name, row_groups):
        rows, md, n = super().read_meta(name, row_groups)
        for i, r in enumerate(rows):
            if i % PER_GROUP % 5 in (2, 3):
                style = list(STYLES3)[int(r["id"][:6], 16) % 3]
                r.update(render_mode=pa.INSTRUCTION_MODE, style=style, instruction=STYLES3[style])
        return rows, md, n


@pytest.fixture
def no_retry(monkeypatch):
    monkeypatch.setattr(pa, "retry", lambda fn, *a, what="", tries=1: fn(*a))


def run(out, src, *extra):
    return pa.main(["--out", str(out), "--hours", "0.05", "--workers", "3", "--max-per-voice", "4", *extra], source=src)


def jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def manifest_view(out):
    """voxcpm_train.jsonl without the output directory in the paths."""
    return [{k: Path(v).name if k in ("audio", "ref_audio") else v for k, v in r.items()}
            for r in jsonl(out / "voxcpm_train.jsonl")]


def test_end_to_end_and_resume_selects_the_same(tmp_path, no_retry):
    full = tmp_path / "full"
    assert run(full, FakeSource()) == 0
    clips, state = jsonl(full / "clips.jsonl"), json.loads((full / "state.json").read_text(encoding="utf-8"))
    assert sum(c["src_duration"] for c in clips) >= 180 and state["done"][-1]["truncated"]
    order = [state["files"][i] for i in pa.spread_order(FILES)]
    assert len(state["done"]) >= 3 and [e["file"] for e in state["done"]] == order[:len(state["done"])]
    assert all(c["file"] == e["file"] for e in state["done"] for c in clips if c["id"] in e["ids"])
    assert len({pa.eval_key(c["text_spoken"]) for c in clips}) == len(clips)
    assert max(Counter(c["voice_id"] for c in clips).values()) <= 4
    assert all(c["mode"] == "reference" and c["qc_cer"] == 0 and c["qc_utmos"] >= 3.7 for c in clips)
    assert sum(c["duration"] for c in clips if c["source"] != "domain") <= 0.25 * sum(c["duration"] for c in clips)
    man = jsonl(full / "voxcpm_train.jsonl")
    assert len(man) == len(clips) and sum("ref_audio" in r for r in man) == round(0.3 * len(clips))
    assert all(Path(r["audio"]).is_file() and set(r) <= {"audio", "text", "duration", "ref_audio", "ref_duration"}
               for r in man)
    attribution = (full / "ATTRIBUTION.txt").read_text(encoding="utf-8").splitlines()
    assert any(c["source"] == "fineweb2" for c in clips) and attribution == [pa.ATTRIBUTION, pa.FINEWEB_NOTE]
    assert all(c["text_doc_id"].startswith("fw2:" if c["source"] == "fineweb2" else "qwen3:")
               and ("ODC-By" in c["license"]) == (c["source"] == "fineweb2") for c in clips)
    assert (full / pa.MARKER).is_file()
    rep = json.loads((full / "prep_report.json").read_text(encoding="utf-8"))
    assert rep["reached"] and rep["selected"]["clips"] == len(clips) and "render_mode" in rep["drops"]
    assert rep["seen"]["files"] == len(state["done"]) and rep["drops"]["duplicate_text"]["rows"] >= 1
    assert "Alania" in (full / "PREP_REPORT.md").read_text(encoding="utf-8")

    # killed while the 2nd file's audio was being written: the next call continues and ends with the same clips
    part = tmp_path / "part"
    with pytest.raises(ConnectionError):
        run(part, FakeSource(fail_on=order[1]))
    assert [e["file"] for e in json.loads((part / "state.json").read_text(encoding="utf-8"))["done"]] == order[:1]
    src = FakeSource()
    assert run(part, src) == 0 and order[0] not in src.meta_reads
    assert jsonl(part / "clips.jsonl") == clips and manifest_view(part) == manifest_view(full)

    # the same call again reads nothing; more --hours redo the file the old target cut, reusing its WAVs
    src = FakeSource()
    assert run(part, src) == 0 and src.meta_reads == []
    src = FakeSource()
    assert run(part, src, "--hours", "0.1") == 0
    cut = state["done"][-1]
    assert cut["file"] in src.meta_reads and not set(order[:len(state["done"]) - 1]) & set(src.meta_reads)
    assert not {c["id"] for c in clips} & set(src.audio_reads)
    more, n_keep = jsonl(part / "clips.jsonl"), len(clips) - len(cut["ids"])
    assert more[:n_keep] == clips[:n_keep] and len(more) > len(clips)
    assert sum(c["src_duration"] for c in more) >= 360


EXPR = ["--modes", "reference,reference+instruction", "--styles", "sad,cheerful", "--max-style-share", "0.2"]


def test_instruction_corpus_end_to_end_and_resume(tmp_path, no_retry):
    out = tmp_path / "expr"
    assert run(out, InstructionSource(), *EXPR) == 0
    clips, state = jsonl(out / "clips.jsonl"), json.loads((out / "state.json").read_text(encoding="utf-8"))
    styled = [c for c in clips if c["mode"] == pa.INSTRUCTION_MODE]
    plain = [c for c in clips if c["mode"] == "reference"]
    assert styled and plain and len(styled) + len(plain) == len(clips)
    assert {c["style"] for c in styled} <= {"sad", "cheerful"}
    assert all(c["instruction"] == STYLES3[c["style"]] for c in styled)
    assert all(c["style"] is None and c["instruction"] is None for c in plain)
    assert all(c["qc_utmos"] >= 3.5 for c in styled) and any(c["qc_utmos"] < 3.7 for c in styled)
    assert all(c["qc_utmos"] >= 3.7 for c in plain)
    total = sum(c["src_duration"] for c in clips)
    for s in ("sad", "cheerful"):  # <= 0.2 of the total + one clip (<= 8 s)
        assert sum(c["src_duration"] for c in styled if c["style"] == s) <= 0.2 * total + 8.0
    texts = {Path(r["audio"]).name: r["text"] for r in jsonl(out / "voxcpm_train.jsonl")}
    assert texts == {c["wav"]: f"({c['instruction']}){c['text_spoken']}" if c["style"] else c["text_spoken"]
                     for c in clips}
    by_wav = {c["wav"]: c for c in clips}
    refs = [r for r in jsonl(out / "voxcpm_train.jsonl") if "ref_audio" in r]
    assert refs and all(by_wav[Path(r["ref_audio"]).name]["mode"] == "reference" for r in refs)  # never styled
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    sel = rep["selected"]
    assert set(sel["by_style"]) <= {"sad", "cheerful", "none"} and sel["by_style"]["none"]["clips"] == len(plain)
    assert sum(v["hours"] for v in sel["by_style"].values()) == pytest.approx(sel["hours"], abs=0.003)
    assert sel["by_mode"][pa.INSTRUCTION_MODE]["clips"] == len(styled)
    assert sel["instructions"] == len({c["instruction"] for c in styled})
    assert rep["manifest"]["with_instruction"] == len(styled) and rep["drops"]["style"]["rows"] > 0
    md = (out / "PREP_REPORT.md").read_text(encoding="utf-8")
    assert "## Style" in md and "styles cheerful,sad" in md and f"{len(styled)} with an instruction prefix" in md
    assert json.loads(state["key"])["filters"]["styles"] == ["cheerful", "sad"]

    # killed while the 2nd file's audio was being written: the next call ends with the same clips and manifest
    order = [state["files"][i] for i in pa.spread_order(FILES)]
    part = tmp_path / "part"
    with pytest.raises(ConnectionError):
        run(part, InstructionSource(fail_on=order[1]), *EXPR)
    assert run(part, InstructionSource(), *EXPR) == 0
    assert jsonl(part / "clips.jsonl") == clips and manifest_view(part) == manifest_view(out)

    # --instruction-format only shapes the manifest: the state stays valid, nothing is read
    src = InstructionSource()
    assert run(out, src, *EXPR, "--instruction-format", "drop") == 0 and src.meta_reads == []
    assert jsonl(out / "clips.jsonl") == clips
    assert {Path(r["audio"]).name: r["text"] for r in jsonl(out / "voxcpm_train.jsonl")} == {
        c["wav"]: c["text_spoken"] for c in clips}
    assert json.loads((out / "prep_report.json").read_text(encoding="utf-8"))["manifest"]["with_instruction"] == 0
    # the instruction settings select rows: another value needs --restart
    for other in (EXPR[:-1] + ["0.3"], EXPR[:3] + ["sad"], EXPR + ["--min-utmos-instruction", "3.6"]):
        assert run(out, InstructionSource(), *other) == 2, other


def test_reference_run_keeps_its_state_key_whatever_the_instruction_options(tmp_path, no_retry):
    out = tmp_path / "o"
    assert run(out, InstructionSource()) == 0
    filters = json.loads(json.loads((out / "state.json").read_text(encoding="utf-8"))["key"])["filters"]
    assert filters["modes"] == ["reference"] and not set(pa.INSTRUCTION_FIELDS) & set(filters)
    clips = jsonl(out / "clips.jsonl")
    assert all(c["mode"] == "reference" and c["style"] is None and c["instruction"] is None for c in clips)
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    assert list(rep["selected"]["by_style"]) == ["none"] and rep["manifest"]["with_instruction"] == 0
    assert rep["drops"]["render_mode"]["rows"] > 0 and "style" not in rep["drops"]
    src = InstructionSource()  # options of instruction rows do not touch a reference run's selection
    assert run(out, src, "--min-utmos-instruction", "3.0", "--instruction-format", "drop") == 0 and not src.meta_reads
    assert jsonl(out / "clips.jsonl") == clips


def test_other_modes_never_replace_a_corpus_even_with_restart(tmp_path, no_retry, capsys):
    """A reference corpus (on Modal /data/alania16k, behind voxcpm2-alania-s1) is not turned into an expressive one in
    place: train_lora.py compares corpora by folder name, so an experiment on it would resume on other data."""
    out = tmp_path / "alania16k"
    assert run(out, InstructionSource()) == 0

    def files():
        return {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}

    before = files()
    for extra in ((), ("--restart",)):
        src = InstructionSource()
        assert run(out, src, *EXPR, *extra) == 2 and src.meta_reads == [] and files() == before, extra
        assert "its own --out" in capsys.readouterr().out
    expr = tmp_path / "alania_expr16k"
    assert run(expr, InstructionSource(), *EXPR) == 0
    assert run(expr, InstructionSource(), "--restart") == 2 and jsonl(expr / "clips.jsonl")  # the other way round too
    assert pa.state_modes(expr / "state.json") == ["reference", pa.INSTRUCTION_MODE]
    assert pa.state_modes(tmp_path / "none" / "state.json") is None


def test_settings_change_needs_restart_and_restart_drops_orphans(tmp_path, no_retry):
    out = tmp_path / "o"
    assert run(out, FakeSource()) == 0
    assert run(out, FakeSource(), "--min-utmos", "3.9") == 2
    before = {p.name for p in (out / "wav").glob("*.wav")}
    assert run(out, FakeSource(), "--min-utmos", "3.9", "--restart") == 0
    clips = jsonl(out / "clips.jsonl")
    assert clips and all(c["qc_utmos"] >= 3.9 for c in clips)
    after = {p.name for p in (out / "wav").glob("*.wav")}
    assert after == {c["wav"] for c in clips}
    removed = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))["orphan_wavs_removed"]
    assert removed == len(before - after) > 0


def test_missing_exclude_manifest_and_files_used_up(tmp_path, no_retry):
    assert run(tmp_path / "a", FakeSource(), "--exclude-manifest", str(tmp_path / "nope.jsonl")) == 2
    out = tmp_path / "b"
    assert run(out, FakeSource(), "--max-files", "1", "--row-groups", "1", "--hours", "5") == 0
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    assert not rep["reached"] and rep["seen"]["files"] == 1 and rep["seen"]["rows"] == PER_GROUP
    assert "NOT reached" in (out / "PREP_REPORT.md").read_text(encoding="utf-8")


def test_missing_wav_of_a_committed_clip_is_fetched_again(tmp_path, no_retry):
    out = tmp_path / "o"
    assert run(out, FakeSource()) == 0
    clips, man = jsonl(out / "clips.jsonl"), manifest_view(out)
    victim = clips[0]
    (out / "wav" / victim["wav"]).unlink()
    for refetched in (1, 0):  # the next call repairs it (no metadata read, one clip's audio); the one after has no work
        src = FakeSource()
        assert run(out, src) == 0 and src.meta_reads == []
        assert src.audio_reads == [victim["id"]] * refetched
        assert jsonl(out / "clips.jsonl") == clips and manifest_view(out) == man
        assert json.loads((out / "prep_report.json").read_text(encoding="utf-8"))["missing_wavs_refetched"] == refetched


def make_foreign_corpus(d):
    (d / "wav").mkdir(parents=True)
    for n in ("antalia_0001.wav", "antalia_val_0001.wav"):
        (d / "wav" / n).write_bytes(b"RIFF")
    (d / "clips.jsonl").write_text('{"wav": "antalia_0001.wav"}\n', encoding="utf-8")
    (d / "voxcpm_train.jsonl").write_text('{"audio": "wav/antalia_0001.wav", "text": "x"}\n', encoding="utf-8")
    return {p.relative_to(d): p.read_bytes() for p in d.rglob("*") if p.is_file()}


def test_foreign_out_is_refused_and_left_untouched(tmp_path, no_retry):
    other = tmp_path / "antalia16k"
    before = make_foreign_corpus(other)
    for extra in ((), ("--restart",)):  # --restart never overrides it
        assert run(other, FakeSource(), *extra) == 2
        assert {p.relative_to(other): p.read_bytes() for p in other.rglob("*") if p.is_file()} == before
    only_wavs = tmp_path / "w"
    (only_wavs / "wav").mkdir(parents=True)
    (only_wavs / "wav" / "x.wav").write_bytes(b"RIFF")
    assert run(only_wavs, FakeSource()) == 2 and (only_wavs / "wav" / "x.wav").exists()
    val = tmp_path / "fresh" / "voxcpm_val.jsonl"  # the held-out manifest inside --out: another corpus too
    val.parent.mkdir()
    val.write_text('{"audio": "wav/v.wav", "text": "Merhaba dünya nasılsın bugün."}\n', encoding="utf-8")
    assert run(val.parent, FakeSource(), "--exclude-manifest", str(val)) == 2 and not (val.parent / "wav").exists()


def test_own_out_is_recognised_with_or_without_the_marker(tmp_path, no_retry):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("kept", encoding="utf-8")  # unrelated files do not make a corpus
    assert run(empty, FakeSource()) == 0 and (empty / "notes.txt").exists() and (empty / pa.MARKER).is_file()
    (empty / pa.MARKER).unlink()  # a directory written before the marker existed: its state.json names the dataset
    assert pa.foreign_files(empty) == []
    src = FakeSource()
    assert run(empty, src) == 0 and src.meta_reads == [] and (empty / pa.MARKER).is_file()
    assert pa.foreign_files(tmp_path / "does-not-exist") == []



def test_train_lora_config_writes_exponent_floats_yaml_reads_as_floats(tmp_path):
    """lora.yaml is JSON read by yaml.safe_load (YAML 1.1): "5e-05" would be a string there and AdamW fails."""
    import argparse
    import importlib
    import json
    import re
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tts" / "voxcpm"))
    train_lora = importlib.import_module("train_lora")
    args = argparse.Namespace(data=tmp_path, model=tmp_path, sample_rate=16000, batch_size=8, grad_accum=2,
                              num_workers=2, steps=500, save_every=100, lr=5e-5, warmup=50, rank=64,
                              hf_model_id="openbmb/VoxCPM2")
    text = train_lora.write_config(args, tmp_path).read_text()
    assert '"learning_rate": 5.0e-05' in text and json.loads(text)["learning_rate"] == 5e-5
    assert '"weight_decay": 0.01' in text and '"max_steps": 500' in text
    yaml11_float = re.compile(r"^[-+]?([0-9][0-9_]*)?\.[0-9_]*([eE][-+][0-9]+)?$")  # PyYAML's float resolver
    assert yaml11_float.match("5.0e-05") and not yaml11_float.match("5e-05")
