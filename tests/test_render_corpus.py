"""tts/voxcpm/render_corpus.py: textbank schema, leakage guard, domain draw, mix accounting, gates (the pilot's clip
gates, the stricter sad / calm and wh-question rules), retries and interjection variants, the training text, the
manifest and its ref_audio, the prelude (emotion references, the laughter trial and its review page), the wave loop
end to end with a fake engine and fake scorers (resume, a cut-short call, a broken engine), the dry run, and
modal_app's corpus actions (no torch, no GPU; the end-to-end runs prepare the reference with refprep: numpy,
soundfile, librosa, else skipped)."""

import json
import math
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VOXCPM = REPO / "tts" / "voxcpm"
sys.path[:0] = [str(VOXCPM), str(REPO / "src")]

import render_corpus as rc  # noqa: E402
import voice_contest as vc  # noqa: E402


def _wav(p: Path, seconds: float, sr: int = 16000, amp: int = 2000) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * seconds)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(int(amp * math.sin(i / 7)).to_bytes(2, "little", signed=True) for i in range(n)))
    return p


TEXTBANK = [
    {"id": "q_mi_1", "category": "questions", "kind": "mi_soru", "text": "Bu akşam sinemaya gidiyor musun?",
     "spoken": "Bu akşam sinemaya gidiyor musun?", "pair_id": "p1", "peak_word": "gidiyor"},
    {"id": "q_duz_1", "category": "questions", "kind": "duz_eslesik", "text": "Bu akşam sinemaya gidiyorsun.",
     "pair_id": "p1", "peak_word": "gidiyorsun"},
    {"id": "q_ne_1", "category": "questions", "kind": "ne_soru", "text": "Toplantı saat kaçta başlıyor?",
     "peak_word": "kaçta"},
    {"id": "q_leak", "category": "questions", "kind": "mi_soru", "text": "Yarın sabah geliyor musun?",
     "peak_word": "geliyor"},  # a TR-Soru sentence: the leakage guard drops it
    {"id": "q_bad", "category": "questions", "kind": "mi_soru", "text": "Geliyor musun?", "peak_word": "gitmek"},
    {"id": "e_angry_1", "category": "emotions", "emotion": "angry", "text": "Bunu kaç kez söylemem gerekiyor!",
     "instruction": "Angry tone, voice getting louder."},
    {"id": "e_sad_1", "category": "emotions", "kind": "sadness", "text": "Haberi duyunca çok üzüldüm, gerçekten.",
     "instruction": "(Sad and quiet.)"},
    {"id": "s_1", "category": "short", "text": "Tamam, anladım."},
    {"id": "s_2", "category": "short", "text": "Evet, öyle."},
    {"id": "i_vay", "category": "interjections", "text": "Vay!", "render_variants": ["Vey!", "Vaay!"]},
    {"id": "i_dup", "category": "short", "text": "Tamam,  anladım."},  # the same text as s_1: de-duplicated
]

DOMAIN = [
    {"line_id": 1, "text_raw": "Randevunuz 12 Ekim saat 14:30'da başlayacak.", "category": "appointments_clinics",
     "tags": [], "lid_tr": 0.99},
    {"line_id": 2, "text_raw": "Kargonuz bugün dağıtıma çıktı, akşam teslim edilecek.", "category": "ecommerce_cargo",
     "tags": [], "lid_tr": 0.99},
    {"line_id": 3, "text_raw": "Hesabınıza 250 TL yatırıldı ve bakiyeniz güncellendi.", "category": "banking",
     "tags": [], "lid_tr": 0.99},
    {"line_id": 4, "text_raw": "Kartınızın şifresini değiştirmek ister misiniz?", "category": "banking",
     "tags": ["question"], "lid_tr": 0.99},  # a question: not neutral
    {"line_id": 5, "text_raw": "Bu kısa.", "category": "banking", "tags": [], "lid_tr": 0.99},  # too short
    {"line_id": 6, "text_raw": "Lütfen bize ulaşın, ekibimiz size en kısa sürede dönüş yapacak.",
     "category": "customer_support", "tags": [], "lid_tr": 0.99},
    {"line_id": 7, "text_raw": "Bugün hava çok güzel ve kuşlar şarkı söylüyor.", "category": "long_explanations",
     "tags": [], "lid_tr": 0.99},  # a category not drawn
    {"line_id": 8, "text_raw": "Adresiniz Atatürk Caddesi numara 5 olarak kaydedildi.",
     "category": "addresses_names", "tags": [], "lid_tr": 0.4},  # language-ID too low
    {"line_id": 9, "text_raw": "Fatura numaranız 12345 ve son ödeme tarihi 20 Ekim.", "category": "numbers_codes",
     "tags": [], "lid_tr": 0.99},
]


def _textbank(tmp_path: Path, rows=TEXTBANK) -> Path:
    p = tmp_path / "textbank.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p


def _args(tmp_path: Path, *extra: str):
    ref = _wav(tmp_path / "refsrc" / "f022-s0.wav", 4.0, sr=48000)
    return rc.parse_args(["--out", str(tmp_path / "corpus"), "--ref", str(ref), "--textbank",
                          str(_textbank(tmp_path)), "--throughput-log", "", "--domain-lines", "5", "--laughter-usd",
                          "0", "--laugh-model", "none", *extra])


# ---- import, CLI, small helpers -----------------------------------------------------------------------------------


def test_import_needs_no_heavy_deps():
    code = (f"import sys; sys.path[:0] = [{str(VOXCPM)!r}, {str(REPO / 'src')!r}]; import render_corpus; "
            "bad = {'torch', 'numpy', 'pyarrow', 'soundfile', 'librosa', 'faster_whisper', 'speechbrain', 'modal', "
            "'huggingface_hub', 'nanovllm_voxcpm', 'transformers'} & set(sys.modules); assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_parse_mix_weights_and_args():
    m = rc.parse_mix("neutral=2,questions=1,emotions=1")
    assert m["neutral"] == 0.5 and m["short"] == 0 and sum(m.values()) == pytest.approx(1)
    assert rc.parse_mix(",".join(f"{k}={v}" for k, v in rc.MIX.items())) == pytest.approx(rc.MIX)
    for bad in ("laughs=1", "neutral=-1", "neutral=0", "neutral"):
        with pytest.raises(ValueError):
            rc.parse_mix(bad)
    assert rc.parse_weights("banking:0.2,numbers_codes:1") == {"banking": 0.2, "numbers_codes": 1.0}
    with pytest.raises(ValueError):
        rc.parse_weights("banking:0")
    ns = rc.parse_args(["--out", "x"])
    assert (ns.hours, ns.tries, ns.prerender, ns.min_sim, ns.rate_lo, ns.rate_hi, ns.backend) == (
        5.0, 4, 2, 0.70, 7.0, 25.0, "nanovllm")
    for bad in (["--mix", "x=1"], ["--rate", "9"], ["--tries", "0"], ["--hours", "0"]):
        with pytest.raises(SystemExit):
            rc.parse_args(["--out", "x", *bad])
    for opt in ("--out", "--model", "--deadline", "--throughput-log", "--backend"):
        assert opt in rc.WRAPPER_OPTS


def test_quantile_rank_and_ids():
    assert rc.quantile([5, 1, 4, 2, 3], 0.2) == 2 and rc.quantile([], 0.2) is None
    assert rc.clean_id("a b/c:ü") == "a_b_c" and rc.clean_id("x y") == "x_y" and rc.clean_id(None) == ""
    assert rc.rank(0, "a") == rc.rank(0, "a") != rc.rank(1, "a")


# ---- texts --------------------------------------------------------------------------------------------------------


def test_textbank_rows_schema(tmp_path):
    rows, bad, sha = rc.load_textbank(_textbank(tmp_path))
    by = {r["id"]: r for r in rows}
    assert len(sha) == 64 and "q_bad" not in by
    assert any("peak_word 'gitmek'" in k for k in bad)
    q = by["q_mi_1"]
    assert (q["pattern"], q["peak"], q["peak_word"], q["pair_id"]) == ("peak_fall", 3, "gidiyor", "p1")
    assert by["q_duz_1"]["pattern"] == "fall" and by["q_ne_1"]["pattern"] == "wh_peak" and by["q_ne_1"]["peak"] == 2
    e = by["e_sad_1"]
    assert (e["emotion"], e["instruction"]) == ("sad", "Sad and quiet.")  # alias, parentheses stripped
    assert rc.train_text(e) == "(Sad and quiet.)Haberi duyunca çok üzüldüm, gerçekten."
    assert rc.train_text(by["s_1"]) == "Tamam, anladım."  # the spoken form, no instruction
    i = by["i_vay"]
    assert i["variants"] == ["Vey!", "Vaay!"] and i["spoken"] == "Vay!" and rc.train_text(i) == "Vay!"
    assert rc.load_textbank(tmp_path / "none.jsonl")[0] == []


def test_textbank_spoken_is_trnorm_and_numbers_are_read(tmp_path):
    row, note = rc.textbank_row({"id": "s9", "category": "short", "text": "Saat 3'te.", "spoken": "Saat 3'te."})
    assert row["spoken"] == rc.spoken_of("Saat 3'te.") != "Saat 3'te." and "trnorm" in note
    assert rc.textbank_row({"id": "x", "category": "laughs", "text": "Ha ha"})[0] is None
    assert rc.textbank_row({"id": "e", "category": "emotions", "text": "Off."})[1] == "emotions: no instruction"
    assert "parentheses" in rc.textbank_row({"id": "e", "category": "emotions", "text": "Off.",
                                             "instruction": "a (b) c"})[1]


def test_build_textbank_schema_index_peaks_render_texts_heldout():
    q = {"id": "q", "category": "questions", "kind": "mi_soru", "pattern": "peak_fall", "peak_word": 4,
         "text": "Akşam yemeğe annemler de geliyor mu?"}
    row, _ = rc.textbank_row(q)
    assert (row["peak"], row["peak_word"], row["pattern"]) == (4, "geliyor", "peak_fall")
    assert rc.textbank_row({**q, "peak_word": 9})[0] is None  # an index outside the spoken form
    row, note = rc.textbank_row({**q, "kind": "hizmet", "peak_word": None})
    assert "pattern" not in row and "not gated" in note  # the textbank was unsure: no intonation gate
    row, _ = rc.textbank_row({**q, "kind": "yukselen", "pattern": "final_rise", "peak_word": None,
                              "text": "Yarın sabah geliyorsun?"})
    assert row["peak"] == 2  # a final rise needs no peak: the scorer's reading
    i = {"id": "i", "category": "interjections", "interjection": "vay", "span": "Vay", "text": "Vay, bu ne kalabalık!",
         "render_variants": ["Vay", "Vaay"], "render_texts": ["Vay, bu ne kalabalık!", "Vaay, bu ne kalabalık!"]}
    row, _ = rc.textbank_row(i)
    assert row["variants"] == ["Vay, bu ne kalabalık!", "Vaay, bu ne kalabalık!"] and row["interjection"] == "vay"
    row, _ = rc.textbank_row({**i, "render_texts": None})  # variants of the span, put in its place
    assert row["variants"] == ["Vay, bu ne kalabalık!", "Vaay, bu ne kalabalık!"]
    assert rc.train_text(row) == "Vay, bu ne kalabalık!"
    e = {"id": "e", "category": "emotions", "emotion": "calm", "text": "Sakin olalım, her şey yoluna girecek.",
         "instruction": "Calm and soothing.", "instruction_split": "heldout"}
    assert rc.textbank_row(e) == (None, "emotions: held-out instruction wording (evaluation only)")
    assert rc.textbank_row({**e, "instruction_split": "train"})[0]["instruction"] == "Calm and soothing."


def test_the_repo_textbank_loads_when_present():
    p = REPO / rc.TEXTBANK
    if not p.is_file():
        pytest.skip("no tts/voxcpm/textbank/textbank.jsonl yet (build_textbank.py)")
    rows, bad, _ = rc.load_textbank(p)
    cats = {r["category"] for r in rows}
    assert cats == set(rc.TB_CATEGORIES) and not any("not in the spoken form" in k for k in bad), bad
    assert all(r["variants"] for r in rows if r["category"] == "interjections")


def test_peak_index_follows_score_intonation():
    w = "Sen de geliyor musun, geliyor musun?".split()
    assert rc.peak_index(w, "geliyor", "peak_fall") == 4  # the one right before the last mI
    assert rc.peak_index("Kim bunu kim yaptı?".split(), "kim", "wh_peak") == 0
    assert rc.peak_index("Yarın sabah geliyorsun?".split(), None, "final_rise") == 2  # the scorer's own reading
    assert rc.peak_index("Yarın sabah geliyorsun.".split(), None, "fall") == 1
    assert rc.peak_index("Yarın geliyor musun?".split(), "gidiyor", "peak_fall") is None


def test_leakage_guard_and_dedup_build_the_pool(tmp_path):
    rows, _, _ = rc.load_textbank(_textbank(tmp_path))
    held = tmp_path / "val.jsonl"
    held.write_text(json.dumps({"audio": "a.wav", "text": "Toplantı saat kaçta başlıyor?"}) + "\n", encoding="utf-8")
    guard, files = rc.make_guard(REPO / "benchmarks" / "tts" / "sets", [held])
    assert any(f.endswith("tr_soru.jsonl") for f in files) and files[-1] == str(held)
    pool, rec = rc.build_pool([], rows, rc.Admit(guard), 0, rc.DOMAIN_WEIGHTS, 0.7, 0, None)
    ids = {r["id"] for r in pool}
    assert "q_leak" not in ids and "q_ne_1" not in ids  # TR-Soru, and the held-out manifest's sentence
    assert ("s_1" in ids) != ("i_dup" in ids)  # one of the two equal texts
    assert rec["textbank_dropped"] == {"leakage guard": 2, "duplicate text": 1}
    # a second call keeps the pool's order and adds only new ids
    more = rows + [{"id": "s_new", "category": "short", "text": "Peki, görüşürüz.", "spoken": "Peki, görüşürüz.",
                    "source": "textbank"}]
    pool2, rec2 = rc.build_pool(pool, more, rc.Admit(guard), 0, rc.DOMAIN_WEIGHTS, 0.7, 0, None)
    assert pool2[:len(pool)] == pool and [r["id"] for r in pool2[len(pool):]] == ["s_new"]
    assert rec2["textbank_added"] == 1


def test_domain_draw_filters_quotas_and_rich_first():
    guard = rc.pa.build_guard(["Kargonuz bugün dağıtıma çıktı, akşam teslim edilecek."])
    got, drops = rc.pick_domain(DOMAIN, 10, rc.DOMAIN_WEIGHTS, 0.7, rc.Admit(guard), 0)
    ids = [r["line_id"] for r in got]
    assert set(ids) == {"1", "3", "6", "9"}
    assert drops["question"] == 1 and drops["category"] == 1 and drops["language-ID"] == 1
    assert drops["length"] == 1 and drops["leakage guard"] == 1
    r1 = next(r for r in got if r["line_id"] == "1")
    assert r1["id"] == "d1" and r1["category"] == "neutral" and r1["rich"] and not rc.JUNK.search(r1["spoken"])
    assert "on iki" in r1["spoken"] and r1["domain_category"] == "appointments_clinics"
    # quotas: of 2 lines, the heaviest categories first; the line ids already in the pool are not taken again
    two, _ = rc.pick_domain(DOMAIN, 2, {"banking": 1.0, "customer_support": 1.0}, 1.0, rc.Admit(rc.pa.Guard()), 0)
    assert sorted(r["domain_category"] for r in two) == ["banking", "customer_support"]
    none, _ = rc.pick_domain(DOMAIN, 2, {"banking": 1.0}, 1.0, rc.Admit(rc.pa.Guard()), 0, {"3"})
    assert none == []


def test_fetch_domain_spreads_the_scan_over_the_files():
    class Src:
        revision = "r"

        def list_files(self):
            return ["data/a.parquet", "data/b.parquet", "data/c.parquet"]

        def read(self, name, n):
            return [{"line_id": f"{name}-{i}"} for i in range(n)]

    rows, rec = rc.fetch_domain(Src(), 10)
    assert len(rows) == 12 and rec["files"] == 3 and rec["rows_read"] == 12


# ---- tries, seeds, variants, mix -------------------------------------------------------------------------------


def test_tries_seeds_and_interjection_variants():
    row = {"id": "i", "category": "interjections", "spoken": "Vay!", "variants": ["Vey!", "Vaay!", "Vay!"]}
    assert rc.max_tries(row, 4) == 6 and rc.max_tries({"id": "x", "category": "short", "spoken": "a"}, 4) == 4
    assert [rc.variant_of(row, t, {}) for t in range(4)] == [0, 1, 2, 0]  # in turn
    assert [rc.variant_of(row, t, {"var_ok": 1}) for t in range(3)] == [1, 1, 1]  # the one Whisper heard
    assert rc.render_text(row, 1) == "Vaay!" and rc.train_text(row) == "Vay!"
    emo = {"id": "e", "category": "emotions", "spoken": "Yeter!", "instruction": "Angry."}
    assert rc.render_text(emo, 0) == "(Angry.)Yeter!"
    s0, s1 = rc.seed_of(0, row, 0), rc.seed_of(0, row, 1)
    assert s0 == vc.text_seed(0, "interjections", "i") and s1 - s0 == rc.SEED_STRIDE > 2 * vc.NANO_TRIES


def test_category_targets_hand_a_used_up_category_to_fill():
    mix = rc.parse_mix("neutral=0.4,questions=0.25,emotions=0.2,short=0.1,interjections=0.05")
    t = rc.category_targets(1.0, mix, {"interjections": 60.0}, set(), "neutral")
    assert t["neutral"] == pytest.approx(1440) and t["interjections"] == pytest.approx(180)
    t = rc.category_targets(1.0, mix, {"interjections": 60.0}, {"interjections"}, "neutral")
    assert t["interjections"] == 60 and t["neutral"] == pytest.approx(1440 + 120)
    assert sum(t.values()) == pytest.approx(3600)
    assert rc.category_targets(1.0, mix, {}, {"interjections"}, "none")["interjections"] == pytest.approx(180)


def test_wave_rows_cover_each_deficit_by_expected_yield_and_interleave():
    pool = [{"id": f"n{i}", "category": "neutral", "spoken": "x" * 82} for i in range(50)]  # 5.22 s each
    pool += [{"id": f"s{i}", "category": "short", "spoken": "x" * 16} for i in range(50)]  # 1.22 s
    state = rc.new_state({})
    state["rows"]["n7"] = {"status": "open", "tries": 1, "fails": [["cer"]]}
    state["rows"]["n8"] = {"status": "kept", "tries": 1}
    state["kept"]["n8"] = {"id": "n8", "category": "neutral", "duration": 5.0, "render_spoken": "x"}
    rows = rc.wave_rows(pool, state, {"neutral": 20.0, "short": 3.0}, 4, rc.CHARS_S)
    neutral = [r["id"] for r in rows if r["category"] == "neutral"]
    assert neutral[0] == "n7" and "n8" not in neutral  # the open text first, the kept one never
    p = rc.p_try(state, "neutral")
    keep4, keep3 = 1 - (1 - p) ** 4, 1 - (1 - p) ** 3
    exp = 5.22 * keep3 + 5.22 * keep4 * (len(neutral) - 1)
    assert exp >= 15 and exp - 5.22 * keep4 < 15  # just enough to cover 20 - 5 s kept
    shorts = [r for r in rows if r["category"] == "short"]
    assert len(shorts) == math.ceil(3.0 / (1.22 * (1 - (1 - rc.p_try(state, "short")) ** 4)))
    assert rows.index(shorts[0]) < len(rows) - 1  # interleaved, not appended after all neutral texts
    assert rc.wave_rows(pool, state, {"neutral": 5.0}, 4, rc.CHARS_S) == []  # met: nothing, n7 waits


def test_wave_tries_and_fit():
    row = {"id": "a", "category": "neutral", "spoken": "x" * 60}
    assert rc.wave_tries(row, None, 4, 2, []) == ([0, 1], 2)
    assert rc.wave_tries(row, {"tries": 3}, 4, 2, []) == ([3], 1)  # never past max_tries
    assert rc.wave_tries(row, {"tries": 1}, 4, 2, [1, 2]) == ([], 2)  # on disk: scored, not made again
    args = rc.parse_args(["--out", "x"])
    cost = rc.Cost(render_per_audio=0.02)
    state = rc.new_state({})
    rows = [dict(row, id=f"r{i}") for i in range(10)]
    got, t = rc.fit_wave(rows, state, args, cost, math.inf, {}, rc.CHARS_S)
    assert len(got) == 10
    one = (t - cost.load_s - cost.server_s) / 10
    got, t2 = rc.fit_wave(rows, state, args, cost, cost.load_s + cost.server_s + 3.5 * one, {}, rc.CHARS_S)
    assert len(got) == 3 and t2 <= cost.load_s + cost.server_s + 3.5 * one
    args.max_wave_rows = 4
    assert len(rc.fit_wave(rows, state, args, cost, math.inf, {}, rc.CHARS_S)[0]) == 4


# ---- gates --------------------------------------------------------------------------------------------------------


G = rc.Gates()


def test_sim_floor_and_duration_gates():
    assert rc.sim_floor(G, 5.0) == 0.70 == rc.sim_floor(G, 4.0)
    assert rc.sim_floor(G, 1.0) == pytest.approx(0.58) == rc.sim_floor(G, 0.5)
    assert rc.sim_floor(G, 2.5) == pytest.approx(0.64)
    assert rc.rate_fails(60, 4.0, G) == [] and rc.rate_fails(60, 10.0, G) == ["duration"]  # 6 chars/s
    assert rc.rate_fails(60, 2.0, G) == ["duration"]  # 30 chars/s
    assert rc.rate_fails(4, 1.5, G) == [] and rc.rate_fails(4, 4.0, G) == ["duration"]  # a short text: its length


def test_cheap_and_text_gates():
    a = {"duration": 3.0, "model_text": "x" * 45}
    assert rc.cheap_fails(a, G) == []
    assert rc.cheap_fails({**a, "synth_error": "boom"}, G) == ["synthesis"]
    assert rc.cheap_fails({**a, "bad_case": True, "onset_trimmed": True}, G) == ["runaway", "onset"]
    hyp = {"hyp": "x", "char_errors": 0, "p_tr": 0.99}
    assert rc.text_fails(hyp, G) == [] and rc.text_fails({**hyp, "char_errors": 1, "p_tr": 0.5}, G) == ["cer", "p_tr"]
    assert rc.text_fails({"whisper_error": "x"}, G) == ["whisper"]


def test_utmos_thresholds_per_category_and_emotion():
    first = {"neutral": [float(x) for x in range(1, 101)], "short": [3.0] * 4, "emotions": [2.0 + i / 100 for i in
                                                                                           range(60)],
             "emotions/angry": [2.5] * 10, "emotions/sad": [3.5] * 45, "questions": [3.0, 3.1, 3.2, 3.3, 3.4]}
    th = rc.utmos_thresholds(first, G, {"questions": 5, "short": 50})
    assert th["neutral"] == 21.0 and "short" not in th  # 4 of a 50-text pool: not calibrated
    assert th["questions"] == 3.1  # the whole pool is 5 texts
    assert "emotions/angry" not in th and th["emotions/sad"] == 3.5  # per emotion from --calib-n
    assert rc.utmos_threshold({"category": "emotions", "emotion": "angry"}, th) == th["emotions"]
    assert rc.utmos_threshold({"category": "emotions", "emotion": "sad"}, th) == 3.5


def test_intonation_and_emotion_gates():
    q = {"category": "questions", "pattern": "peak_fall"}
    assert rc.intonation_fails({"intonation": [{"pass": True}]}, q) == []
    assert rc.intonation_fails({"intonation": [{"pass": None}]}, q) == ["intonation"]  # unscorable: fails
    assert rc.intonation_fails({}, q) == ["intonation"]
    duz = {"category": "questions", "pattern": "fall"}
    rise = {"pass": False, "end_st": 2.5, "slope_st_s": 8.0}
    assert rc.intonation_fails({"intonation": [rise]}, duz) == ["intonation"]  # sounds like a question
    assert rc.intonation_fails({"intonation": [{**rise, "end_st": 0.5}]}, duz) == []
    assert rc.intonation_fails({"intonation": [{"pass": None, "end_st": None, "slope_st_s": None}]}, duz) == []
    pros = [{"f0_median_st": 4.0, "f0_range_st": 6.0, "rate_syl_s": 5.0, "energy_db": -20.0}] * 25
    base = rc.baseline(pros)
    assert base["n"] == 25 and base["f0_range_st"] == 6.0 and rc.baseline(pros[:5]) is None
    v = lambda e, **d: rc.emotion_verdict(e, {**pros[0], **d}, base, G)[0]  # noqa: E731
    assert v("angry", f0_range_st=8.0) == "expected" and v("angry", f0_range_st=5.0) == "weak"
    assert v("angry", f0_range_st=3.5) == "opposite" and v("cheerful", energy_db=-18.0) == "expected"
    # sad / calm: slower AND lower than the neutral clips (pilot: sad was the weakest emotion)
    assert v("sad", rate_syl_s=4.5, f0_median_st=3.0) == "expected" and v("calm", rate_syl_s=5.7) == "opposite"
    assert v("sad", rate_syl_s=4.5, energy_db=-21.0) == "weak"  # slower, but not lower
    assert v("sad", rate_syl_s=5.3) == "weak" and v("empathetic", rate_syl_s=9.0) == "recorded"
    assert rc.baseline([{**pros[0], "level_db": -24.0}] * 5 + pros[:20])["level_db"] is None  # too few have it
    lvl = rc.baseline([{**pros[0], "level_db": -24.0}] * 25)  # levels before levelling: the energy that counts
    assert rc.emotion_verdict("sad", {**pros[0], "rate_syl_s": 4.5, "f0_median_st": 3.0, "level_db": -20.0}, lvl,
                              G)[0] == "weak"  # 4 dB louder than the neutral clips
    assert rc.emotion_verdict("angry", {**pros[0], "f0_range_st": 5.0, "level_db": -21.0, "energy_db": -30.0}, lvl,
                              G)[0] == "expected"  # louder before levelling, though not after
    assert rc.emotion_verdict("angry", None, base, G)[0] == "unchecked"
    assert rc.emotion_verdict("angry", pros[0], None, G)[0] == "unchecked"
    d = rc.emotion_verdict("angry", {**pros[0], "f0_median_st": 16.5}, base, G)[1]
    assert d["f0_median_st"] == 0.5  # an octave-off pitch track folded back first


def test_full_gates_record_floor_threshold_and_verdict():
    row = {"id": "e", "category": "emotions", "emotion": "angry"}
    a = {"duration": 1.0, "sim": 0.6, "utmos": 3.0, "cpu_done": True, "onset_final_bad": False,
         "prosody": {"f0_median_st": 4.0, "f0_range_st": 3.0, "rate_syl_s": 5.0, "energy_db": -20.0}}
    base = rc.baseline([{"f0_median_st": 4.0, "f0_range_st": 6.0, "rate_syl_s": 5.0, "energy_db": -20.0}] * 20)
    fails = rc.full_fails(a, row, G, {"emotions": 3.2}, base)
    assert fails == ["utmos", "emotion"] and a["sim_floor"] == pytest.approx(0.58)
    assert a["utmos_threshold"] == 3.2 and a["emotion_verdict"] == "opposite"
    sad = {**a, "utmos": 3.5, "prosody": {**a["prosody"], "f0_range_st": 6.0, "rate_syl_s": 4.5}}  # slower, same F0
    assert rc.full_fails(sad, {"id": "s", "category": "emotions", "emotion": "sad"}, G, {}, base) == ["emotion"]
    assert sad["emotion_verdict"] == "weak"  # weak sad / calm is dropped, weak angry is not
    b = {**a, "sim": 0.57, "onset_final_bad": True, "utmos": None}
    assert rc.full_fails(b, {"id": "s", "category": "short"}, G, {}, None) == ["onset", "sim", "utmos"]


# ---- manifest -----------------------------------------------------------------------------------------------------


def test_ref_audio_never_itself_and_manifest_rows(tmp_path):
    kept = [{"id": f"c{i}", "category": "neutral" if i < 8 else "emotions", "wav": f"c{i}.wav", "duration": 4.0 + i,
             "train_text": f"t{i}", "spoken": f"t{i}"} for i in range(12)]
    refs = rc.assign_refs(kept, 0.5, 0, ("/data/c/ref/v.wav", 9.92))
    assert len(refs) == 6 and refs == rc.assign_refs(kept, 0.5, 0, ("/data/c/ref/v.wav", 9.92))
    by = {c["wav"]: c for c in kept}
    for cid, r in refs.items():
        assert r["ref_audio"] != f"{cid}.wav"
        if r["ref_audio"] in by:  # another kept neutral clip of 3-10 s, never an emotional one
            c = by[r["ref_audio"]]
            assert c["category"] == "neutral" and 3 <= c["duration"] <= 10
    assert rc.assign_refs(kept[:1], 1.0, 0, None) == {}  # no other clip and no reference: none
    rows = rc.manifest_rows(kept, refs, tmp_path / "wav", 0)
    assert sorted(r["text"] for r in rows) == sorted(c["train_text"] for c in kept)
    with_ref = [r for r in rows if "ref_audio" in r]
    assert len(with_ref) == 6 and all(Path(r["ref_audio"]).is_absolute() and r["ref_duration"] for r in with_ref)
    assert all(r["audio"] == str(tmp_path / "wav" / Path(r["audio"]).name) for r in rows)


def test_sample_clips_stratified_and_seeded():
    clips = [{"id": f"n{i}", "category": "neutral"} for i in range(80)] + [
        {"id": f"q{i}", "category": "questions"} for i in range(19)] + [{"id": "i0", "category": "interjections"}]
    s = rc.sample_clips(clips, 10, 0)
    cats = [c["category"] for c in s]
    assert len(s) == 10 and cats.count("neutral") == 7 and cats.count("questions") == 2 and "interjections" in cats
    assert s == rc.sample_clips(clips, 10, 0) != rc.sample_clips(clips, 10, 1)
    assert len(rc.sample_clips(clips[:3], 10)) == 3 and rc.sample_clips([], 5) == []


def test_estimate_arithmetic():
    pool = [{"id": f"n{i}", "category": "neutral", "spoken": "x" * 82} for i in range(10000)]
    args = rc.parse_args(["--out", "x", "--hours", "1", "--mix", "neutral=1"])
    cost = rc.Cost(render_per_audio=1 / 49)
    est = rc.estimate(pool, rc.new_state({}), args, cost)
    b = est["by"]["neutral"]
    p, d = rc.p_try(rc.new_state({}), "neutral"), rc.DUR_PAD_S + 82 / rc.CHARS_S
    keep = 1 - (1 - p) ** 4
    assert b["texts"] == math.ceil(3600 / (d * keep)) and b["p_keep"] == round(keep, 3)
    tries = 3600 / (d * keep) * keep / p
    assert b["tries"] == round(tries) and est["score_s"] == round(tries * cost.score_s(d))
    rend = (tries + 3600 / d * 0.5) * d
    assert est["rendered_audio_s"] == round(rend) and est["render_s"] == round(rend / 49 * 1.2)
    assert est["waves"] == 3 and est["overhead_s"] == round(3 * (cost.server_s + cost.load_s))
    assert est["work_s"] == pytest.approx(est["render_s"] + est["score_s"] + est["overhead_s"], abs=2)
    lines = rc.estimate_lines(est, 3.0, est["work_s"] / 2)
    assert "this call" in lines[-1] and "0.50 h kept" in lines[-1]
    # before a corpus measures its own: /data/f022corpus's measured pass rates (at --min-sim 0.6), scaled to min_sim
    assert rc.p_try(rc.new_state({}), "neutral", 0.6) == 0.407 == rc.MEASURED_PASS["neutral"]
    assert rc.p_try(rc.new_state({}), "neutral", 0.7) == pytest.approx(0.407 * 0.598 / 0.939)
    assert rc.p_try(rc.new_state({}), "long", 0.6) == 0.10
    assert rc.sim_pass(0.675) == pytest.approx((0.768 + 0.598) / 2) and rc.sim_pass(0.9) == 0.073
    st = rc.new_state({})
    st["stats"]["neutral"] = {"decided": 40, "passed": 10}
    assert rc.p_try(st, "neutral", 0.6) == 0.25 * rc.NEW_GATE_PASS["neutral"]  # measured before the pilot's gates
    st["stats"]["neutral"].update(decided_g2=30, passed_g2=6)
    assert rc.p_try(st, "neutral", 0.6) == 0.2  # measured under today's gates wins
    measured = [{"mode": "corpus-score", "clips": 100, "wall_s": 30.0, "model_s": 60.0}]
    assert rc.make_cost("nanovllm", 10, 2, measured).score_factor == 0.5
    assert rc.make_cost("nanovllm", 10, 2, []).render_per_audio == pytest.approx(1 / 49)


# ---- the run: fake engine, fake scorers ---------------------------------------------------------------------------


class FakeEngine:
    """Renders 16 kHz clips 0.25 s + 1 s per 16 characters of spoken form; records every job. dead: every job
    fails (an engine that cannot run)."""

    def __init__(self, dead: bool = False):
        self.jobs, self.dead = [], dead

    def __call__(self, jobs, feed_until, on_done, on_fail):
        self.jobs += jobs
        for j in jobs:
            if self.dead:
                on_fail(j, "RuntimeError: no engine")
                continue
            dur = 0.25 + len(rc.spoken_of(j["text"].split(")")[-1])) / 16
            _wav(Path(j["out"]), dur, amp=1000)
            on_done(j, {"duration": round(dur, 3), "onset_bad": False, "onset_why": [], "onset_retries": 0,
                        "onset_trimmed": False, "engine": "fake"})
        return {"jobs": len(jobs), "done": 0 if self.dead else len(jobs), "audio_s": 1.0, "wall_s": 0.1}


class FakeScoring:
    """Whisper hears the spoken form (an interjection's "Vey" as "vey"); the IPA model each letter, c as /dʒ/ (/k/ at
    the first try of the texts in WRONG_C); sim 0.9 except the texts in LOW_SIM at their first try; UTMOS 3.9 for first
    tries (minus 1 for LOW_UTMOS), 4.2 for retries; intonation passes."""

    LOW_SIM, LOW_UTMOS, WRONG_C, CALLS = {"d1"}, {"d3"}, set(), []

    def phon(self, atts, until):
        FakeScoring.CALLS.append(("phon", [a["rid"] for a in atts]))
        for a in atts:
            k = a["rid"] in self.WRONG_C and a["t"] == 0
            toks = [("k" if k else "dʒ") if ch == "c" else ch for ch in rc.pc.tr_lower(a["model_text"]) if ch.isalpha()]
            a["ipa"] = " ".join(toks)
            a["c_expected"], a["c_got"] = rc.pc.c_counts(a["model_text"], toks)

    def text(self, atts, until):
        FakeScoring.CALLS.append(("text", len(atts)))
        for a in atts:
            hyp = a["model_text"].replace("Vaay", "Vay")
            a.update(hyp=hyp, lang="tr", p_tr=0.99)
            a["errors"], a["ref_words"], a["char_errors"], a["ref_chars"] = vc.wer_cer(a["refs"], hyp)
            w = hyp.split()  # words spread over the clip by syllables
            a["words"] = [[x, round(t0, 3), round(t1, 3)] for x, (t0, t1) in zip(w, rc.si._spread(
                0.0, a["duration"], [max(1, rc.si.syllables(x)) for x in w]))]

    def rest(self, atts, until, cpu):
        FakeScoring.CALLS.append(("rest", len(atts)))
        for a in atts:
            a["sim"] = 0.5 if a["rid"] in self.LOW_SIM and a["t"] == 0 else 0.9
            a["utmos"] = (3.9 - (a["rid"] in self.LOW_UTMOS)) if a["t"] == 0 else 4.2
        for a, inton, pros in cpu:
            a.update(cpu_done=True, onset_final_bad=False)
            if inton:
                rise = a["pattern"] == "fall"  # the matched declarative ends rising: it sounds like a question
                a["intonation"] = [{"pattern": a["pattern"], "pass": not rise, "end_st": 2.5 if rise else -1.0,
                                    "slope_st_s": 8.0 if rise else -2.0}]
            if pros:
                sad = "Sad" in a["prompt"]
                a["prosody"] = {"f0_median_st": 3.0 if sad else 4.0, "f0_range_st": 8.0 if "Angry" in a["prompt"]
                                else 6.0, "rate_syl_s": 4.0 if sad else 5.0, "energy_db": -20.0}

    def close(self):
        FakeScoring.CALLS.append(("close",))


def _domain():
    return DOMAIN, {"repo": rc.DOMAIN_REPO, "files": 1, "rows_read": len(DOMAIN)}


@pytest.fixture
def audio_libs():
    for m in ("numpy", "soundfile", "librosa"):
        pytest.importorskip(m)
    FakeScoring.CALLS.clear()


def test_run_end_to_end_gates_retries_variants_and_layout(tmp_path, audio_libs, monkeypatch):
    monkeypatch.setattr(rc, "BASELINE_MIN", 3)  # the emotion baseline from this pool's few neutral clips
    args = _args(tmp_path, "--hours", "1")
    eng = FakeEngine()
    assert rc.run(args, render=eng, scoring=FakeScoring, domain=_domain) == 0
    out = tmp_path / "corpus"
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    kept = state["kept"]
    # 1 h asks for more than the pool: every text is tried, the categories are used up
    assert set(kept) >= {"d1", "d3", "d6", "d9", "q_mi_1", "e_angry_1", "e_sad_1", "s_1", "i_vay"}
    assert state["rows"]["q_duz_1"]["status"] == "dropped"  # a matched declarative that sounds like a question
    assert state["rows"]["q_duz_1"]["fails"][0] == ["intonation"]
    # d1: its first try's speaker similarity failed, the second (seed + 1000) kept
    assert kept["d1"]["try"] == 1 and kept["d1"]["seed"] == rc.seed_of(0, {"id": "d1", "category": "neutral"}, 1)
    seeds = sorted(j["seed"] for j in eng.jobs if j.get("rid") == "d1")
    assert seeds[1] - seeds[0] == rc.SEED_STRIDE
    # the interjection: "Vey" not heard as "vay" (CER), the next variant "Vaay" heard: canonical training text
    iv = kept["i_vay"]
    assert (iv["try"], iv["variant"], iv["render_text"], iv["train_text"]) == (1, 1, "Vaay!", "Vay!")
    assert state["rows"]["i_vay"]["var_ok"] == 1
    # d3: its first try in the neutral texts' bottom 20 % of UTMOS (calibrated on the 5 first tries), retried
    assert kept["d3"]["try"] == 1 and state["kept"]["d3"]["utmos_threshold"] == 3.9
    assert kept["e_sad_1"]["train_text"] == "(Sad and quiet.)Haberi duyunca çok üzüldüm, gerçekten."
    assert kept["e_angry_1"]["emotion_verdict"] == "expected" == kept["e_sad_1"]["emotion_verdict"]
    assert kept["e_angry_1"]["emotion_delta"]["f0_range_st"] == 2.0  # vs the neutral clips' median
    assert len(state["baseline"]) >= 3 and "prosody" not in kept["q_mi_1"]
    # the prelude: a reference per emotion of the pool (both go their way here), cloned by that emotion's rows
    refs = state["emotion_refs"]
    assert set(refs) == {"angry", "sad"} and refs["sad"]["wav"] == "refs/sad.wav" and refs["sad"]["score"] > 0
    assert (out / "refs" / "sad.wav").is_file() and (out / "refs" / "angry.wav").is_file()
    assert {Path(j["ref"]).name for j in eng.jobs if j.get("rid") == "e_sad_1"} == {"sad.wav"}
    assert {Path(j["ref"]).name for j in eng.jobs if j.get("rid") == "d1"} == {"designed-f022-s0.raw.wav"}
    assert kept["e_sad_1"]["ref"] == "sad.wav" and "ref" not in kept["d1"]
    assert sum(j["key"].startswith("prelude/") for j in eng.jobs) == rc.EMO_REF_NEUTRAL + 2 * rc.EMO_REF_N
    assert not (out / "prelude").exists() and not (out / rc.REVIEW).exists()  # --laughter-usd 0
    # the layout train_lora.py --data takes
    rows = [json.loads(x) for x in (out / "voxcpm_train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(kept) and {r["text"] for r in rows} == {c["train_text"] for c in kept.values()}
    for r in rows:
        with wave.open(r["audio"]) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
            assert w.getnframes() / 16000 == pytest.approx(r["duration"], abs=1e-3)
    assert sorted(p.name for p in (out / "wav").glob("*.wav")) == sorted(c["wav"] for c in kept.values())
    with wave.open(str(out / "ref" / "designed-f022-s0.wav")) as w:
        assert w.getframerate() == 16000 and (w.getnframes() % 2560) == 0  # prepared: 16 kHz, whole patches
    clips = [json.loads(x) for x in (out / "clips.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(c.get("utmos") and c.get("sim") and c.get("p_tr") and "hyp" in c for c in clips)
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    assert rep["selected"]["clips"] == len(kept) and not rep["reached"] and rep["complete"]
    mine = set(rc.CATEGORIES) - {"long", "cwords"}  # long, cwords: not this corpus's (no share, no texts)
    assert set(rep["exhausted"]) == mine and set(rep["by_category"]) == mine
    assert rep["by_category"]["questions"]["fails"]["intonation"] >= 1
    assert rep["distributions"]["sim"]["scored"]["n"] >= len(kept)
    assert "Alania Turkish Domain Text by PatientDesk AI" in (out / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    assert "f022 corpus report" in (out / "PREP_REPORT.md").read_text(encoding="utf-8")
    assert not any((out / "attempts").rglob("*.wav"))  # decided tries are removed
    att = [json.loads(x) for x in (out / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(att) == sum(c["decided"] for c in state["stats"].values())
    texts = {r["id"] for r in map(json.loads, (out / "texts.jsonl").read_text(encoding="utf-8").splitlines())}
    assert "q_leak" not in texts and "d4" not in texts
    # nothing left to do: a second call renders nothing and changes no clip
    eng2 = FakeEngine()
    assert rc.run(args, render=eng2, scoring=FakeScoring, domain=_domain) == 0
    assert eng2.jobs == [] and json.loads((out / "state.json").read_text(encoding="utf-8"))["kept"] == kept
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    assert set(rep["emotion_refs"]) == {"angry", "sad"} and "clip" in rep["distributions"]
    assert "emotion references: angry refs/angry.wav" in (out / "PREP_REPORT.md").read_text(encoding="utf-8")


def test_run_mix_by_kept_hours_and_retries_on_utmos(tmp_path, audio_libs):
    args = _args(tmp_path, "--hours", "0.003", "--mix", "neutral=0.5,short=0.5", "--calib-n", "2")
    eng = FakeEngine()
    assert rc.run(args, render=eng, scoring=FakeScoring, domain=_domain) == 0
    state = json.loads((tmp_path / "corpus" / "state.json").read_text(encoding="utf-8"))
    k = rc.kept_s(state)
    assert k["neutral"] >= 0.0015 * 3600 and k["short"] > 0 and k["questions"] == k["emotions"] == 0
    assert not any(j["rid"].startswith(("q_", "e_", "i_")) for j in eng.jobs)  # no share: not rendered
    assert state["first_utmos"]["short"] and state["first_utmos"]["neutral"]
    rep = json.loads((tmp_path / "corpus" / "prep_report.json").read_text(encoding="utf-8"))
    assert rep["reached"] and rep["complete"]


def test_run_resumes_a_cut_short_call_without_rendering_again(tmp_path, audio_libs, monkeypatch):
    args = _args(tmp_path, "--hours", "1", "--mix", "short=1", "--fill", "none")

    class Cut(FakeScoring):  # the deadline arrives during Whisper: nothing decided
        def text(self, atts, until):
            FakeScoring.CALLS.append(("text", len(atts)))

    eng = FakeEngine()
    assert rc.run(args, render=eng, scoring=Cut, domain=_domain) == 0
    out = tmp_path / "corpus"
    st = json.loads((out / "state.json").read_text(encoding="utf-8"))
    assert st["kept"] == {} and st["rows"]["s_1"] == {"status": "open", "tries": 0, "fails": []}
    made = {j["key"] for j in eng.jobs}
    assert made and all((out / "attempts" / k.split("/")[0] / f"{k.split('/')[1]}.json").is_file() for k in made)
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    assert rep["complete"] is False  # modal_app's train preflight refuses it
    eng2 = FakeEngine()
    assert rc.run(args, render=eng2, scoring=FakeScoring, domain=_domain) == 0
    st = json.loads((out / "state.json").read_text(encoding="utf-8"))
    assert {"s_1", "s_2"} <= set(st["kept"]) and not eng2.jobs  # the tries on disk were scored, none made again
    assert st["kept"]["s_1"]["try"] == 0


def test_run_stops_on_a_broken_engine_without_spending_tries(tmp_path, audio_libs):
    args = _args(tmp_path, "--hours", "1", "--domain-lines", "0", "--mix", "questions=1")
    pool_rows = 4  # questions in the pool after the guard
    big = [{"id": f"q{i}", "category": "questions", "kind": "yukselen", "text": f"Yarın {i} kez geliyorsun?"}
           for i in range(12)]
    _textbank(tmp_path, TEXTBANK + big)
    assert rc.run(args, render=FakeEngine(dead=True), scoring=FakeScoring, domain=_domain) == 1
    out = tmp_path / "corpus"
    st = json.loads((out / "state.json").read_text(encoding="utf-8"))
    assert all(r["tries"] == 0 and r["status"] == "open" for r in st["rows"].values()) and pool_rows
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "error" and "synthesis failed" in status["errors"]["waves"]
    assert (out / "prep_report.json").is_file()  # the report is written all the same


def test_run_refuses_another_reference_a_foreign_dir_and_no_textbank(tmp_path, audio_libs, capsys):
    args = _args(tmp_path, "--hours", "0.001", "--mix", "short=1")
    assert rc.run(args, render=FakeEngine(), scoring=FakeScoring, domain=_domain) == 0
    other = _wav(tmp_path / "other.wav", 3.0)
    args2 = rc.parse_args(["--out", str(args.out), "--ref", str(other), "--textbank", args.textbank,
                           "--throughput-log", ""])
    assert rc.run(args2) == 2 and "rendered otherwise" in capsys.readouterr().out
    foreign = tmp_path / "alania"
    (foreign / "wav").mkdir(parents=True)
    (foreign / "state.json").write_text("{}")
    assert rc.run(rc.parse_args(["--out", str(foreign), "--ref", str(other)])) == 2
    assert rc.run(rc.parse_args(["--out", str(tmp_path / "x"), "--ref", str(other), "--textbank",
                                 str(tmp_path / "none.jsonl")])) == 2
    assert "no textbank" in capsys.readouterr().out


def test_reconcile_keeps_undecided_tries_of_open_texts_only(tmp_path):
    out = tmp_path / "c"
    pool = [{"id": "a", "category": "short", "spoken": "Tamam."}, {"id": "b", "category": "short", "spoken": "Evet."}]
    state = rc.new_state({})
    state["rows"] = {"a": {"status": "open", "tries": 1, "fails": [["cer"]]}, "b": {"status": "kept", "tries": 1}}
    state["kept"] = {"b": {"id": "b", "category": "short", "wav": "b.wav", "duration": 1.0},
                     "c": {"id": "c", "category": "short", "wav": "c.wav", "duration": 1.0}}
    state["rows"]["c"] = {"status": "kept", "tries": 2}
    pool.append({"id": "c", "category": "short", "spoken": "Hayır."})
    _wav(out / "wav" / "b.wav", 1.0)
    for t in (0, 1, 2):
        _wav(rc.attempt_wav(out, "a", t), 0.5)
        rc.attempt_wav(out, "a", t).with_suffix(".json").write_text("{}")
    _wav(rc.attempt_wav(out, "a", 3), 0.5)  # no .json: its record was not written
    _wav(rc.attempt_wav(out, "b", 1), 0.5)
    _wav(out / "wav" / "orphan.wav", 0.5)
    run = rc.Run(rc.parse_args(["--out", str(out)]), out, pool, state, rc.Gates(), rc.Cost(0.02), Path("r"),
                 FakeEngine(), FakeScoring)
    fixed = run.reconcile()
    assert rc.on_disk_tries(out, "a") == [1, 2] and not (out / "attempts" / "b").exists()
    assert not (out / "wav" / "orphan.wav").exists()
    assert state["rows"]["c"]["status"] == "open" and "c" not in state["kept"]  # its WAV is gone: reopened
    assert state["rows"]["b"]["status"] == "kept" and (out / "wav" / "b.wav").is_file()
    assert fixed["kept clips without a WAV: reopened"] == 1


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "c"
    assert rc.main(["--out", str(out), "--textbank", str(_textbank(tmp_path)), "--throughput-log", "",
                    "--dry-run", "--exclude-manifest", str(tmp_path / "missing.jsonl")]) == 0
    text = capsys.readouterr().out
    assert "6000 neutral sentences to draw" in text and "texts x" in text and "total" in text and "$" in text
    assert "questions/" in text and "emotions/" in text and "train '(" in text  # the instruction in the text
    assert "not here (the container has it)" in text
    assert "prelude: emotion references for angry, sad (" in text
    assert "laughter trial" in text and "; prelude " in text  # in the total
    assert not out.exists()


def test_scoring_wraps_the_contest_passes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(vc, "whisper_pass", lambda sc, rows, w, label, until, words: calls.append(
        ("whisper", len(rows), words)))
    monkeypatch.setattr(vc, "utmos_pass", lambda sc, rows, w, label, until: calls.append(("utmos", len(rows))))
    monkeypatch.setattr(vc, "sim_pass", lambda sc, groups, w, until, self_sim: calls.append(
        ("sim", groups[0][2].name, self_sim)))

    class Pool:
        def __init__(self, items, procs):
            calls.append(("cpu", [(it[3], it[4], it[5]) for it in items], procs))
            self.items = items

        def get(self, until):
            return [(it[0], {"onset": {"onset_bad": True, "onset_why": ["stray"]}}) for it in self.items]

    monkeypatch.setattr(vc, "CpuPass", Pool)
    sc = rc.Scoring(str(tmp_path / "ref.raw.wav"), "cpu", 3)
    a = {"path": str(tmp_path / "a.wav"), "model_text": "x", "prompt": "x", "set": "soru", "id": "a"}
    sc.text([a], math.inf)
    sc.rest([a], math.inf, [(a, True, False)])
    assert calls == [("whisper", 1, True), ("cpu", [(True, False, True)], 3), ("utmos", 1),
                     ("sim", "ref.raw.wav", False)]
    assert a["onset_final_bad"] is True and a["cpu_done"] and a["onset_final_why"] == ["stray"]


# ---- modal_app ----------------------------------------------------------------------------------------------------


def _modal_app():
    import importlib.util
    from unittest import mock

    spec = importlib.util.spec_from_file_location("modal_app_corpus_test", VOXCPM / "modal_app.py")
    ma = importlib.util.module_from_spec(spec)
    had, saved = "modal" in sys.modules, sys.modules.get("modal")
    sys.modules["modal"] = mm = mock.MagicMock()
    try:
        spec.loader.exec_module(ma)
    finally:
        if had:
            sys.modules["modal"] = saved
        else:
            del sys.modules["modal"]
    return ma, mm


def test_modal_app_corpus_actions():
    ma, mm = _modal_app()
    cmd = ma.corpus_cmd(["--hours", "5"], 123.4, "voxcpm")
    assert cmd[:4] == ["python", "tts/voxcpm/render_corpus.py", "--hours", "5"]
    assert cmd[cmd.index("--out") + 1] == ma.F022 == "/data/f022corpus"
    assert cmd[cmd.index("--deadline") + 1] == "123" and cmd[cmd.index("--backend") + 1] == "voxcpm"
    assert cmd[-2:] == ["--exclude-manifest", "/data/antalia16k/voxcpm_val.jsonl"]
    with pytest.raises(ValueError):
        ma.corpus_cmd([], 1.0, "fake")
    for opt in rc.WRAPPER_OPTS:  # the wrapper's own options come after --args: they win
        assert opt in cmd[4:]
    fn = [c for c in mm.App.return_value.function.call_args_list if c.kwargs.get("timeout") == ma.CORPUS_TIMEOUT_S]
    assert len(fn) == 1 and fn[0].kwargs["image"] is ma.contest_image and fn[0].kwargs["memory"] == ma.CORPUS_MEM
    assert fn[0].kwargs["cpu"] == 8 and fn[0].kwargs["gpu"] == ma.GPU
    assert ma.CORPUS_MEM != ma.CONTEST_MEM and ma.CORPUS_MAX_USD == 4.0
    assert ma.GPU_PRICE["A100-80GB"] + ma.CORPUS_SIDE_PRICE == pytest.approx(rc.USD_PER_H)
    assert "voxcpm-corpus" in ma._IGNORE
    src = (VOXCPM / "modal_app.py").read_text(encoding="utf-8")
    assert 'F022: "corpus-render"' in src and 'rep.get("complete") is False' in src  # the train preflight knows it


def test_modal_app_corpus_summary_reads_the_report(tmp_path, audio_libs):
    ma, _ = _modal_app()
    args = _args(tmp_path, "--hours", "0.001", "--mix", "short=1")
    assert rc.run(args, render=FakeEngine(), scoring=FakeScoring, domain=_domain) == 0
    out = tmp_path / "corpus"
    s = ma.corpus_summary(out)
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    assert s["clips"] == rep["selected"]["clips"] and s["reached"] is True and "unfinished" not in s
    mt = {n: (out / n).stat().st_mtime for n in ma.PREP_FILES if (out / n).exists()}
    assert ma.report_current(mt)  # the report is written after state.json and clips.jsonl
    time.sleep(0.05)
    (out / "state.json").write_text((out / "state.json").read_text(encoding="utf-8"), encoding="utf-8")
    mt["state.json"] = mt["prep_report.json"] + 1
    assert not ma.report_current(mt)  # a call cut short after it: refused by the preflight


def test_dedup_keeps_question_statement_pairs():
    assert rc.dedup_key("Yolculuk dokuz saat sürüyor.") != rc.dedup_key("Yolculuk dokuz saat sürüyor?")
    assert rc.dedup_key("Pardon!") != rc.dedup_key("Pardon?")
    assert rc.dedup_key("Tamam, anladım.") == rc.dedup_key("tamam anladım.")
    assert rc.dedup_key("Olur.") == rc.dedup_key("Olur!") != rc.dedup_key("Olur...")  # build_textbank's classes
    assert rc.final_mark('Gerçekten mi?"') == "?" and rc.final_mark("Bilmem…") == "..." and rc.final_mark("Ha") == "."
    rows = [{"id": "a", "category": "questions", "text": "Pardon!", "spoken": "Pardon!"},
            {"id": "b", "category": "questions", "text": "Pardon?", "spoken": "Pardon?"},
            {"id": "c", "category": "short", "text": "Pardon.", "spoken": "Pardon."}]
    pool, rec = rc.build_pool([], rows, rc.Admit(rc.pa.Guard()), 0, rc.DOMAIN_WEIGHTS, 0.7, 0, None)
    assert sorted(r["id"] for r in pool) in (["a", "b"], ["b", "c"]) and rec["textbank_dropped"] == {
        "duplicate text": 1}


def test_emotion_reference_sentence_and_laughter_lines_are_not_protected():
    guard, _ = rc.make_guard(REPO / "benchmarks" / "tts" / "sets", [])
    sp = rc.spoken_of(rc.EMO_REF_TEXT)
    assert not guard.hit(rc.pa.eval_key(sp), rc.pa.eval_key(rc.EMO_REF_TEXT))
    rows, bad = rc.load_laughter(REPO / rc.LAUGHTER, guard)
    assert len(rows) >= 50 and not bad and all(r["plain"] and r["plain"] != r["spoken"] for r in rows)


def test_guard_covers_contest_refs_and_antalia_heldout_by_default(tmp_path, monkeypatch):
    sets = tmp_path / "sets"
    sets.mkdir()
    (sets / "contest.jsonl").write_text(json.dumps({"id": "x", "text": "Kısa metin.", "refs": [
        "Randevunuz yarın sabah dokuzda başlayacak efendim."]}) + "\n", encoding="utf-8")
    held = tmp_path / "rows.jsonl"
    held.write_text(json.dumps({"text": "Kargonuz bugün akşam saatlerinde teslim edilecek."}) + "\n", encoding="utf-8")
    monkeypatch.setattr(rc, "HELDOUT", (tmp_path / "none.jsonl", held))  # the Volume's absent, the local copy here
    guard, files = rc.make_guard(sets, [])
    assert files == [str(sets / "contest.jsonl"), *map(str, rc.probe_files()), str(held)]
    admit = rc.Admit(guard)
    for text in ("Randevunuz yarın sabah dokuzda başlayacak efendim.", "Kargonuz bugün akşam saatlerinde teslim "
                 "edilecek, lütfen evde olun."):
        assert admit.why({"spoken": rc.spoken_of(text), "text": text}) == "leakage guard"
    assert rc.heldout_files([held, tmp_path / "x.jsonl"]) == [held, tmp_path / "x.jsonl"]  # each once


# ---- the pilot's gates, emotion references, the laughter trial ----------------------------------------------------


def test_clip_gates_and_word_measures():
    row = {"id": "q", "category": "questions"}
    a = {"edges": {"tail": "stray", "head": ""}, "first_stretch": 1.2, "sib_db": -8.0, "rate_ws": 8.4}
    assert rc.clip_fails(a, row, G) == []  # a stray after the last word is cut, not failed
    bad = {"edges": {"tail": "inside", "head": "vlead"}, "first_stretch": 2.5, "sib_db": -3.4, "rate_ws": 8.9}
    assert rc.clip_fails(bad, row, G) == ["tail", "head", "stretch", "sibilance", "speed"]
    assert rc.clip_fails(bad, {"id": "i", "category": "interjections"}, G) == ["tail", "head", "sibilance", "speed"]
    assert rc.clip_fails({"rate_ws": 8.9}, row, rc.Gates(speed_scale=1.1)) == []
    words = [["Bu", 0.0, 0.4], ["iki", 0.4, 0.6], ["elbiseden", 0.6, 1.2], ["hangisini", 1.2, 1.8]]
    m = rc.word_measures(words, "Bu iki elbiseden hangisini", {"on_s": 0.1, "pauses": [[0.6, 0.8]]})
    assert m["first_word"] == "bu" and m["first_s_per_syl"] == pytest.approx(0.3)  # from where the sound starts
    assert m["first_stretch"] == pytest.approx(0.3 / 0.15)  # the others: 0.1, 0.2, 0.15 s a syllable
    assert m["speech_s"] == pytest.approx(1.5) and m["rate_ws"] == pytest.approx(11 / 1.5, abs=1e-3)
    assert rc.word_measures([], "Bu iki", {}) == {} and "first_stretch" not in rc.word_measures(words[:2], "Bu iki")


def _voice(seconds: float, sr: int = 16000, f0: float = 160.0, amp: float = 0.2):
    np = pytest.importorskip("numpy")
    t = np.arange(int(sr * seconds)) / sr
    return (amp * sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 8))).astype(np.float32)


def test_measure_clip_cuts_a_stray_tail_and_rejects_one_inside(tmp_path, audio_libs):
    import numpy as np

    sr = 16000
    gap = np.zeros(int(0.28 * sr), np.float32)
    x = np.concatenate([_voice(0.7), gap, _voice(0.07), np.zeros(int(0.03 * sr), np.float32)])  # pilot clip 41
    p = tmp_path / "t0.wav"
    vc.write_wav16(p, [x], sr)
    (tmp_path / "t0.json").write_text("{}")
    row = {"id": "s", "category": "short"}
    a = {"path": str(p), "duration": len(x) / sr, "model_text": "Özür dilerim.",
         "words": [["Özür", 0.0, 0.32], ["dilerim.", 0.32, 0.7]]}
    rc.measure_clip(a, row, G)
    assert a["edges"]["tail"] == "stray" and a["edges"]["cut_s"] == pytest.approx(0.82, abs=0.02)
    assert rc.clip_fails(a, row, G) == [] and a["tail_cut_s"] == pytest.approx(len(x) / sr - 0.82, abs=0.02)
    side = json.loads((tmp_path / "t0.json").read_text())
    assert vc.wav_duration(p) == pytest.approx(a["duration"]) and side["tail_cut_s"] == a["tail_cut_s"]
    vc.write_wav16(p, [x], sr)  # Whisper put the stray into the last word: not cut, failed
    b = {**a, "duration": len(x) / sr, "words": [["Özür", 0.0, 0.32], ["dilerim.", 0.32, 1.05]]}
    b.pop("tail_cut_s")
    rc.measure_clip(b, row, G)
    assert b["edges"]["tail"] == "inside" and rc.clip_fails(b, row, G) == ["tail"] and "tail_cut_s" not in b
    c = {"path": str(tmp_path / "none.wav"), "duration": 1.0, "model_text": "x", "words": []}
    rc.measure_clip(c, row, G)
    assert "clip_error" in c and rc.clip_fails(c, row, G) == []  # a measure that errs never costs the try


def test_wh_question_needs_the_peak_on_the_wh_word():
    si = rc.si
    m = {"peak_st": 2.4, "drop_st": 9.1, "range_st": 14.8}
    assert si.judge("wh_peak", {**m, "lead_st": -3.9}) is False  # pilot clip 5: "Çocuklar" above "nasıl"
    assert si.judge("wh_peak", {**m, "lead_st": -0.02}) is True  # pilot clip 6, "Bu formda neyi ...": kept
    assert si.judge("wh_peak", {**m, "lead_st": None}) is True  # the wh-word first: nothing before it
    assert si.judge("wh_peak", {**m, "lead_st": 1.0, "range_st": 4.0}) is False  # flat
    assert si.judge("peak_fall", {"peak_st": 3, "drop_st": 4, "peak_at": -0.5, "lead_st": -9}) is True  # mI: as before


def test_emotion_reference_choice():
    base = rc.baseline([{"f0_median_st": 10.0, "f0_range_st": 8.0, "rate_syl_s": 6.0, "energy_db": -20.0,
                         "level_db": -24.0}] * 4, 1)

    def pros(**d):
        return {"f0_median_st": 10.0, "f0_range_st": 8.0, "rate_syl_s": 6.0, "energy_db": -20.0, **d}

    cands = [{"seed": 1, "char_errors": 0, "sim": 0.7, "prosody": pros(f0_median_st=8.0, rate_syl_s=5.0),
              "level_in": -27.0},
             {"seed": 2, "char_errors": 0, "sim": 0.55, "prosody": pros(f0_median_st=6.0, rate_syl_s=4.0)},  # voice
             {"seed": 3, "char_errors": 1, "sim": 0.8, "prosody": pros(f0_median_st=6.0)},  # misread
             {"seed": 4, "char_errors": 0, "sim": 0.75, "prosody": pros(rate_syl_s=5.8)}]
    best, why = rc.pick_emo_ref("sad", cands, base)
    assert best["seed"] == 1 and not why and best["delta"]["level_db"] == -3.0
    assert best["score"] == pytest.approx(1 + 2 + 1.5)
    assert rc.pick_emo_ref("angry", cands, base)[0] is None  # none wider or louder: the voice's reference stays
    assert rc.emo_ref_score("surprised", {"f0_range_st": 3.0, "f0_median_st": 2.0}) == pytest.approx(2.0)
    assert rc.pick_emo_ref("sad", cands, None) == (None, "no neutral render of the sentence to compare with")


def _laugh_bank(tmp_path: Path, n: int = 3) -> Path:
    p = tmp_path / "laughter.jsonl"
    texts = ["Hahaha, çok komiksin!", "Gülmekten karnım ağrıdı, hahaha.", "Ay, hahaha, yüzünü görmeliydin!"]
    rows = [{"id": f"l_{i:03d}", "emotion": "cheerful", "intensity": "mild", "laugh": True, "laugh_span": "Hahaha",
             "text": t} for i, t in enumerate(texts[:n], 1)]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p


def test_prelude_jobs_plan_and_laughter_budget(tmp_path):
    laugh, bad = rc.load_laughter(_laugh_bank(tmp_path))
    assert [r["plain"] for r in laugh] == ["çok komiksin!", "Gülmekten karnım ağrıdı.", "Ay yüzünü görmeliydin!"]
    args = rc.parse_args(["--out", "x", "--laughter-usd", "0.3"])
    cost = rc.Cost(render_per_audio=1 / 49)
    pool = [{"id": "e1", "category": "emotions", "emotion": "sad", "spoken": "x"},
            {"id": "e2", "category": "emotions", "emotion": "angry", "spoken": "x"}]
    state = rc.new_state({})
    state["emotion_refs"]["angry"] = {"wav": "refs/angry.wav"}
    jobs, plan = rc.prelude_jobs(tmp_path, state, args, pool, laugh, cost, "ref.wav")
    assert plan["emotions"] == ["sad"] and plan["emo_jobs"] == rc.EMO_REF_NEUTRAL + rc.EMO_REF_N
    assert plan["laugh_jobs"] == 3 * len(rc.LAUGH_INSTRUCTIONS) * rc.LAUGH_SEEDS and plan["laugh_lines"] == 3
    sad = [j for j in jobs if j.get("emotion") == "sad"]
    assert sad[0]["text"].startswith("(Sad and quiet") and len({j["seed"] for j in sad}) == rc.EMO_REF_N
    lj = [j for j in jobs if j["kind"] == "laugh"]
    assert lj[0]["text"] == "(Laughing heartily while speaking.)Hahaha, çok komiksin!"
    assert lj[0]["plain"] == "çok komiksin!"
    assert plan["s"] == pytest.approx(plan["emo_s"] + plan["laugh_s"], abs=2)
    assert plan["laugh_s"] < 0.3 / rc.USD_PER_H * 3600
    many = [{"id": f"l{i}", "text": "Hahaha, çok güzel.", "spoken": "Hahaha, çok güzel.", "plain": "çok güzel."}
            for i in range(400)]
    cheap = rc.parse_args(["--out", "x", "--laughter-usd", "0.25", "--mix", "neutral=1"])
    jobs, plan = rc.prelude_jobs(tmp_path, rc.new_state({}), cheap, pool, many, cost, "ref.wav")
    assert plan["emo_jobs"] == 0 and 0 < plan["laugh_jobs"] < 400  # the budget stops it, seed 0 first
    assert plan["s"] <= 0.25 / rc.USD_PER_H * 3600 + 1 and all(j["key"].endswith("-s0") for j in jobs)
    alone = rc.parse_args(["--out", "x", "--laughter-usd", "0.05", "--mix", "neutral=1"])
    assert rc.prelude_jobs(tmp_path, rc.new_state({}), alone, pool, many, cost, "r")[0] == []  # not even a server
    done = rc.new_state({})
    done["laughter"] = {"done": True}
    assert rc.prelude_jobs(tmp_path, done, cheap, pool, many, cost, "r")[0] == []  # once
    assert "nothing" in rc.prelude_line(rc.prelude_jobs(tmp_path, done, cheap, pool, many, cost, "r")[1])


def test_laugh_span_and_review_page():
    words = [["çok", 0.9, 1.2], ["komiksin!", 1.2, 1.8]]
    assert rc.laugh_span(words, "Hahaha, çok komiksin!", 2.0) == (0.0, 0.9)  # Whisper dropped the laughter
    assert rc.laugh_span([["Hahaha,", 0.0, 0.8]] + words, "Hahaha, çok komiksin!", 2.0) == (0.0, 0.9)  # or wrote it
    end = rc.laugh_span([["Gülmekten", 0, 0.5], ["ağrıdı,", 0.5, 1.0]], "Gülmekten ağrıdı, hahaha.", 2.2)
    assert end == (1.0, 2.2)
    a = {"cer_nolaugh": 0, "sim": 0.7, "laugh": {"laugh": True, "score": 2.4}}
    assert rc.laugh_rank(a) > rc.laugh_rank({**a, "laugh": {"laugh": False, "score": 0.5}})
    assert rc.laugh_rank({**a, "ast_laugh": 0.4}) > rc.laugh_rank({**a, "ast_laugh": 0.05})
    rows = [{"id": f"l_001-{k}", "line": "l_001", "rank": k, "text": "Hahaha, çok komiksin!", "seed": k,
             "instruction": rc.LAUGH_INSTRUCTIONS[k - 1], "wav": f"wav/l_001-{k}.wav"} for k in (1, 2)]
    page = rc.laughter_html(rows)
    assert "gerçekten gülüyor" in page and page.count("<audio") == 2 and "laughter_review/wav/l_001-2.wav" in page
    script = page[page.index("<script>"):page.index("</script>")]
    assert "\n" not in script and "String.fromCharCode(10)" in script and "clipboard" in script


def test_laughter_trial_end_to_end(tmp_path, audio_libs):
    args = _args(tmp_path, "--hours", "0.001", "--mix", "short=1", "--laughter-usd", "0.3", "--laughter-bank",
                 str(_laugh_bank(tmp_path)))
    eng = FakeEngine()
    assert rc.run(args, render=eng, scoring=FakeScoring, domain=_domain) == 0
    out = tmp_path / "corpus"
    laugh_jobs = [j for j in eng.jobs if j.get("kind") == "laugh"]
    assert len(laugh_jobs) == 3 * len(rc.LAUGH_INSTRUCTIONS) * rc.LAUGH_SEEDS
    assert not any(j.get("kind") == "emo" for j in eng.jobs)  # no emotions in this mix: no references
    review = [json.loads(x) for x in (out / rc.REVIEW / "review.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(review) == 3 * rc.LAUGH_KEEP and {r["rank"] for r in review} == {1, 2, 3}
    assert all(r["cer_nolaugh"] == 0 and (out / rc.REVIEW / r["wav"]).is_file() for r in review)
    assert (out / rc.REVIEW / "laughter.html").is_file() and not (out / "prelude").exists()
    st = json.loads((out / "state.json").read_text(encoding="utf-8"))
    assert st["laughter"]["done"] and st["laughter"]["tries"] == len(laugh_jobs)
    assert st["laughter"]["detector"] == "bursts"
    manifest = (out / "voxcpm_train.jsonl").read_text(encoding="utf-8") + (out / "clips.jsonl").read_text(
        encoding="utf-8")
    assert "aha" not in manifest and rc.REVIEW not in manifest  # never trained on
    eng2 = FakeEngine()
    assert rc.run(args, render=eng2, scoring=FakeScoring, domain=_domain) == 0
    assert not any(j.get("kind") == "laugh" for j in eng2.jobs)  # once


def test_modal_app_corpus_fetch_brings_the_laughter_review_home(tmp_path, monkeypatch):
    from types import SimpleNamespace

    ma, _ = _modal_app()
    rel = ma.F022[len(ma.DATA) + 1:]
    review = [{"id": "l_001-1", "line": "l_001", "rank": 1, "text": "Hahaha, çok komiksin!", "seed": 3,
               "instruction": rc.LAUGH_INSTRUCTIONS[0], "wav": "wav/l_001-1.wav"}]
    store = {f"{rel}/clips.jsonl": b"", f"{rel}/{rc.REVIEW}/review.jsonl": (json.dumps(review[0]) + "\n").encode(),
             f"{rel}/{rc.REVIEW}/wav/l_001-1.wav": b"RIFF....", f"{rel}/{rc.REFS_DIR}/sad.wav": b"RIFF.."}

    def files(d):
        return {k[len(d) + 1:]: SimpleNamespace(path=k, size=len(v)) for k, v in store.items()
                if k.startswith(d + "/") and "/" not in k[len(d) + 1:]}

    def download(path, size, out):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(store[path])

    monkeypatch.setattr(ma, "_vol_files", files)
    monkeypatch.setattr(ma, "_download", download)
    monkeypatch.setattr(ma, "_rc", lambda: rc)
    ma._corpus_fetch(tmp_path)
    assert (tmp_path / rc.REVIEW / "wav" / "l_001-1.wav").read_bytes() == b"RIFF...."
    assert (tmp_path / rc.REFS_DIR / "sad.wav").is_file()
    page = (tmp_path / "laughter.html").read_text(encoding="utf-8")
    assert "gerçekten gülüyor" in page and f"{rc.REVIEW}/wav/l_001-1.wav" in page


# ---- human-approved clips (--add-reviewed) ------------------------------------------------------------------------


REVIEW_ROWS = [("l_001", "Hahaha, çok komiksin!"), ("l_002", "Gülmekten karnım ağrıdı, hahaha."),
               ("l_003", "Ay, hahaha, yüzünü görmeliydin!")]


def _finished_corpus(tmp_path: Path, *extra: str, review_sr: int = 16000) -> tuple[Path, object]:
    """A corpus as a finished corpus-render call leaves it (finalize: state, clips, manifest, report; status done),
    with its laughter trial's review clips: 3 lines x LAUGH_KEEP ranks. No engine, no audio libraries."""
    out = tmp_path / "corpus"
    args = rc.parse_args(["--out", str(out), *extra])
    (out / "wav").mkdir(parents=True)
    (out / rc.MARKER).write_text("x\n", encoding="utf-8")
    ref = _wav(out / "ref" / "designed-f022-s0.wav", 4.0)
    state = rc.new_state({"voice": rc.VOICE})
    for i, (cid, cat, dur) in enumerate([("d1", "neutral", 5.0), ("d2", "neutral", 3.5), ("s_1", "short", 1.0),
                                         ("e_angry_1", "emotions", 4.0)]):
        _wav(out / "wav" / f"{cid}.wav", dur)
        spoken = f"Cümle {i}."
        state["kept"][cid] = {"id": cid, "category": cat, "wav": f"{cid}.wav", "duration": dur, "text": spoken,
                              "spoken": spoken, "source": "textbank",
                              "train_text": ("(Angry tone.)" if cat == "emotions" else "") + spoken}
    pool = [{"id": c["id"], "category": c["category"]} for c in state["kept"].values()]
    rc.finalize(args, out, state, pool, (str(ref), 4.0), [])
    vc.write_json(out / "status.json", {"mode": "corpus", "phase": "done"})
    rows = []
    for line, text in REVIEW_ROWS:
        for k in range(1, rc.LAUGH_KEEP + 1):
            name = f"{line}-{k}"
            _wav(out / rc.REVIEW / "wav" / f"{name}.wav", 1.5 + k / 10, sr=review_sr)
            rows.append({"id": name, "line": line, "rank": k, "text": text, "seed": k,
                         "instruction": rc.LAUGH_INSTRUCTIONS[k - 1], "wav": f"wav/{name}.wav",
                         "duration": 1.5 + k / 10, "hyp": "Çok komiksin.", "cer_nolaugh": 0, "sim": 0.6})
    vc.write_rows(out / rc.REVIEW / "review.jsonl", rows)
    return out, args


def _approved(tmp_path: Path, *ids: str, best: dict | None = None) -> Path:
    p = tmp_path / "approved.json"
    p.write_text(json.dumps({"note": "test", "best": best or {}, **dict.fromkeys(ids, True), "l_003-3": False}),
                 encoding="utf-8")
    return p


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_add_reviewed_appends_laughter_rows_once(tmp_path, capsys):
    out, _ = _finished_corpus(tmp_path)
    before = _rows(out / "voxcpm_train.jsonl")
    appr = _approved(tmp_path, "l_001-1", "l_002-2", best={"l_001": [1]})
    assert rc.main(["--out", str(out), "--add-reviewed", str(appr)]) == 0
    rows = _rows(out / "voxcpm_train.jsonl")
    assert rows[:len(before)] == before and len(rows) == len(before) + 2  # appended, nothing else touched
    new = {Path(r["audio"]).name: r for r in rows[len(before):]}
    assert set(new) == {"laugh_l_001-1.wav", "laugh_l_002-2.wav"}
    r1 = new["laugh_l_001-1.wav"]
    assert r1["text"] == "(Laughing heartily while speaking.)Hahaha, çok komiksin!"  # the laughter as written
    assert new["laugh_l_002-2.wav"]["text"] == "(Speaking with a cheerful laugh.)Gülmekten karnım ağrıdı, hahaha."
    assert r1["audio"] == str(out / "wav" / "laugh_l_001-1.wav") and r1["duration"] == pytest.approx(1.6)
    for r in new.values():
        with wave.open(r["audio"]) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
    clips = {c["id"]: c for c in _rows(out / "clips.jsonl")}
    c1 = clips["laugh_l_001-1"]
    assert (c1["category"], c1["source"], c1["review_id"], c1["best"]) == ("laughter", "human-approved", "l_001-1",
                                                                            True)
    assert "best" not in clips["laugh_l_002-2"] and c1["train_text"] == r1["text"]
    assert c1["spoken"] == REVIEW_ROWS[0][1]
    assert [c["id"] for c in _rows(out / rc.REVIEWED)] == ["laugh_l_001-1", "laugh_l_002-2"]
    # the report: laughter its own category, hours and shares over all clips
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    lb = rep["by_category"]["laughter"]
    assert lb["clips"] == 2 and lb["hours"] == round(3.3 / 3600, 3) and lb["mix"] == 0 and lb["target_hours"] == 0
    assert rep["selected"]["clips"] == 6 and rep["selected"]["hours"] == round((13.5 + 3.3) / 3600, 3)
    assert lb["share"] == round(3.3 / 16.8, 3) and rep["by_category"]["neutral"]["share"] == round(8.5 / 16.8, 3)
    assert rep["reviewed"]["clips"] == 2 and rep["reviewed"]["best"] == 1 and rep["reviewed"]["added"]["clips"] == 2
    assert rep["manifest"]["rows"] == len(rows) and rep["manifest"]["with_instruction"] == 3
    md = (out / "PREP_REPORT.md").read_text(encoding="utf-8")
    assert "| laughter | 2 |" in md and "human-approved clips in the manifest" in md
    assert "laughter_approved.json" in (out / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    # modal_app.corpus_summary takes it as a finished corpus: the report is newer and counts clips.jsonl's clips
    ma, _ = _modal_app()
    assert ma.corpus_summary(out)["clips"] == 6
    # again: nothing new, nothing written
    snap = {n: (out / n).read_bytes() for n in ("voxcpm_train.jsonl", "clips.jsonl", "prep_report.json", rc.REVIEWED)}
    capsys.readouterr()
    assert rc.main(["--out", str(out), "--add-reviewed", str(appr)]) == 0
    assert {n: (out / n).read_bytes() for n in snap} == snap
    assert "adding 0 clips" in capsys.readouterr().out
    # one more approved: only it is added
    assert rc.main(["--out", str(out), "--add-reviewed", str(_approved(tmp_path, "l_001-1", "l_002-2",
                                                                         "l_003-1"))]) == 0
    rows2 = _rows(out / "voxcpm_train.jsonl")
    assert rows2[:len(rows)] == rows and [Path(r["audio"]).name for r in rows2[len(rows):]] == ["laugh_l_003-1.wav"]
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    assert rep["by_category"]["laughter"]["clips"] == 3 and rep["reviewed"]["added"]["clips"] == 1
    assert sum(b["share"] for b in rep["by_category"].values()) == pytest.approx(1, abs=0.01)


def test_add_reviewed_restores_rows_an_interrupted_call_left_out(tmp_path):
    out, _ = _finished_corpus(tmp_path)
    appr = _approved(tmp_path, "l_001-1")
    assert rc.main(["--out", str(out), "--add-reviewed", str(appr)]) == 0
    full = (out / "voxcpm_train.jsonl").read_text(encoding="utf-8")
    cut = "".join(x + "\n" for x in full.splitlines() if "laugh_" not in x)
    (out / "voxcpm_train.jsonl").write_text(cut, encoding="utf-8")  # reviewed.jsonl written, the manifest not
    assert rc.main(["--out", str(out), "--add-reviewed", str(appr)]) == 0
    for name in ("voxcpm_train.jsonl", "clips.jsonl"):
        assert sum("laugh_l_001-1" in x for x in (out / name).read_text(encoding="utf-8").splitlines()) == 1


def test_add_reviewed_refuses_a_running_render_and_bad_input(tmp_path, capsys):
    out, _ = _finished_corpus(tmp_path)
    appr = _approved(tmp_path, "l_001-1")
    manifest = (out / "voxcpm_train.jsonl").read_bytes()
    vc.write_json(out / "status.json", {"mode": "corpus", "phase": "waves", "updated": "now"})
    assert rc.main(["--out", str(out), "--add-reviewed", str(appr)]) == 2
    assert "phase 'waves'" in capsys.readouterr().out
    vc.write_json(out / "status.json", {"mode": "corpus", "phase": "done"})
    time.sleep(0.05)
    (out / "state.json").write_bytes((out / "state.json").read_bytes())  # a call cut short after the report
    assert rc.main(["--out", str(out), "--add-reviewed", str(appr)]) == 2
    assert "newer than prep_report.json" in capsys.readouterr().out
    time.sleep(0.05)
    vc.write_json(out / "prep_report.json", json.loads((out / "prep_report.json").read_text(encoding="utf-8")))
    assert rc.main(["--out", str(out), "--add-reviewed", str(_approved(tmp_path, "l_009-1"))]) == 2  # no such clip
    assert rc.main(["--out", str(out), "--add-reviewed", str(_approved(tmp_path))]) == 2  # none approved
    (out / rc.MARKER).unlink()
    assert rc.main(["--out", str(out), "--add-reviewed", str(_approved(tmp_path, "l_001-1"))]) == 2  # not ours
    (out / rc.MARKER).write_text("x\n", encoding="utf-8")
    assert not (out / rc.REVIEWED).exists() and not list((out / "wav").glob("laugh_*"))
    assert (out / "voxcpm_train.jsonl").read_bytes() == manifest
    capsys.readouterr()
    appr = _approved(tmp_path, "l_001-1")  # (the bad ones above overwrote it)
    assert rc.main(["--out", str(out), "--add-reviewed", str(appr), "--dry-run"]) == 0  # a dry run writes nothing
    assert "adding 1 clips" in capsys.readouterr().out and not (out / rc.REVIEWED).exists()


def test_a_later_render_keeps_the_reviewed_clips(tmp_path):
    out, args = _finished_corpus(tmp_path, "--ref-fraction", "1")
    assert rc.main(["--out", str(out), "--add-reviewed", str(_approved(tmp_path, "l_001-1", "l_003-2"))]) == 0
    added = {r["audio"]: r for r in _rows(out / "voxcpm_train.jsonl") if "laugh_" in r["audio"]}
    assert len(added) == 2 and all(r.get("ref_audio") for r in added.values())  # --ref-fraction 1 from the report
    # the next corpus-render call: reconcile keeps their WAVs, save and finalize write them again
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    pool = [{"id": c["id"], "category": c["category"]} for c in state["kept"].values()]
    run = rc.Run(args, out, pool, state, rc.gates_of(args), None, out / "ref" / "x.raw.wav", None, None)
    assert "orphan WAVs removed" not in run.reconcile() and (out / "wav" / "laugh_l_001-1.wav").is_file()
    _wav(out / "wav" / "stray.wav", 1.0)
    assert run.reconcile()["orphan WAVs removed"] == 1
    run.save()
    assert sum(c["category"] == "laughter" for c in _rows(out / "clips.jsonl")) == 2
    rep = rc.finalize(args, out, state, pool, (str(out / "ref" / "designed-f022-s0.wav"), 4.0), [])
    again = {r["audio"]: r for r in _rows(out / "voxcpm_train.jsonl") if "laugh_" in r["audio"]}
    assert again == added  # the same texts, durations and ref_audio as add-reviewed gave them
    assert rep["by_category"]["laughter"]["clips"] == 2 and rep["selected"]["clips"] == 6
    assert len(_rows(out / "clips.jsonl")) == 6 and rep["manifest"]["rows"] == 6


def test_add_reviewed_converts_other_rates(tmp_path, audio_libs):
    out, _ = _finished_corpus(tmp_path, review_sr=24000)
    assert rc.main(["--out", str(out), "--add-reviewed", str(_approved(tmp_path, "l_002-1"))]) == 0
    with wave.open(str(out / "wav" / "laugh_l_002-1.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert w.getnframes() / 16000 == pytest.approx(1.6, abs=0.01)


def test_the_repo_laughter_approvals():
    d = json.loads((REPO / rc.LAUGH_APPROVED).read_text(encoding="utf-8"))
    ids = [k for k, v in d.items() if v is True]
    lines = {r["id"] for r in map(json.loads, (REPO / rc.LAUGHTER).read_text(encoding="utf-8").splitlines())}
    assert len(ids) == 48 and all(k.rsplit("-", 1)[0] in lines and k[-1] in "123" for k in ids)
    assert d["best"] == {"l_001": [1], "l_003": [1], "l_024": [1], "l_030": [2, 3]}
    assert all(f"{line}-{k}" in ids for line, ks in d["best"].items() for k in ks)
    assert "Laughing heartily while speaking." in d["note"]


def test_modal_app_corpus_add_reviewed(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    ma, mm = _modal_app()
    assert ma.add_reviewed_cmd("/tmp/a.json") == ["python", "tts/voxcpm/render_corpus.py", "--out", ma.F022,
                                                  "--add-reviewed", "/tmp/a.json"]
    fn = [c for c in mm.App.return_value.function.call_args_list
          if c.kwargs.get("timeout") == ma.REVIEWED_TIMEOUT_S]
    assert len(fn) == 1 and "gpu" not in fn[0].kwargs and fn[0].kwargs["cpu"] == ma.REVIEWED_CPU
    rel = ma.F022[len(ma.DATA) + 1:]
    vol = {f"{rel}/status.json": {"phase": "waves"},
           f"{rel}/{rc.REVIEW}/review.jsonl": [{"id": "l_001-1", "duration": 2.0}, {"id": "l_002-1", "duration": 3.0}],
           f"{rel}/{rc.REVIEWED}": [{"review_id": "l_001-1"}]}
    monkeypatch.setattr(ma, "_rc", lambda: rc)
    monkeypatch.setattr(ma, "_vol_json", lambda p: vol.get(p))
    monkeypatch.setattr(ma, "_vol_rows", lambda p: vol.get(p, []))
    calls = []
    monkeypatch.setattr(ma, "corpus_add_reviewed", SimpleNamespace(
        remote=lambda text, name: calls.append((text, name)) or json.dumps({"rc": 0, "used_s": 31})))
    appr = _approved(tmp_path, "l_001-1", "l_002-1")
    with pytest.raises(SystemExit, match="phase 'waves'"):
        ma._corpus_add_reviewed(["--add-reviewed", str(appr)])
    vol[f"{rel}/status.json"] = {"phase": "done"}
    ma._corpus_add_reviewed(["--add-reviewed", str(appr), "--dry-run"])
    assert "1 clips, ~3.0 s" in capsys.readouterr().out and not calls
    ma._corpus_add_reviewed(["--add-reviewed", str(appr)])
    assert calls == [(appr.read_text(encoding="utf-8"), "approved.json")]
    with pytest.raises(SystemExit, match="not in"):
        ma._corpus_add_reviewed(["--add-reviewed", str(_approved(tmp_path, "l_009-9"))])
    with pytest.raises(SystemExit, match="takes"):
        ma._corpus_add_reviewed(["--hours", "5"])
    with pytest.raises(SystemExit, match="corpus-add-reviewed"):  # never on the GPU call
        ma._corpus(["--add-reviewed", str(appr)], 4.0)


# ---- long items, numbers in neutral rows, the pause gate, a second corpus, the merge ------------------------------


def _dl(lid: str, doc: str, text: str, cat: str = "long_explanations") -> dict:
    """A domain-text row as the dataset has it (doc_id "qwen3:s3-p<prompt>-<k>", unit_idx 0)."""
    return {"line_id": lid, "text_raw": text, "category": cat, "tags": [], "lid_tr": 0.99, "doc_id": doc,
            "unit_idx": 0}


LONG_ROWS = [  # in file order; prompt p7's k=2 is a question: it splits the prompt's lines into two runs
    _dl("p9k1", "qwen3:s3-p9-1", "Kartınızın son ödeme tarihi 20 Ekim, ödemeyi 0850 222 33 44 numaralı hattan da "
                                 "yapabilirsiniz.", "banking"),
    _dl("p7k0", "qwen3:s3-p7-0", "Kredi başvurunuz değerlendirmeye alındı ve sonuç birkaç iş günü içinde size "
                                 "bildirilecek."),
    _dl("p7k1", "qwen3:s3-p7-1", "Bu süre içinde ek bir belge gerekirse sizinle telefonla iletişime geçeceğiz."),
    _dl("p7k2", "qwen3:s3-p7-2", "Başvurunuzla ilgili başka bir sorunuz var mı?"),
    _dl("p7k3", "qwen3:s3-p7-3", "Onay çıktığında sözleşme e-posta adresinize gönderilecek ve imzanız beklenecek"),
    _dl("p7k4", "qwen3:s3-p7-4", "İmzadan sonra kredi tutarı hesabınıza aynı gün içinde aktarılacak."),
    _dl("p7k5", "qwen3:s3-p7-5", "Ödeme planınızı mobil uygulamadan dilediğiniz zaman görebilirsiniz."),
    _dl("p9k0", "qwen3:s3-p9-0", "Hesabınıza 250 TL yatırıldı ve güncel bakiyeniz mobil uygulamada görünüyor.",
        "banking"),
    _dl("p8k0", "qwen3:s3-p8-0", "Bugün hava çok güzel ve kuşlar şarkı söylüyor, yürüyüşe çıkmak için ideal bir gün.",
        "questions"),  # a category not drawn for long items
]


def test_number_kinds_are_written_form_checks():
    cases = {"Randevunuz 12 Ekim saat 14:30'da.": ["time", "date", "digits"],
             "0555 555 55 55 numarasından arayın.": ["phone", "digits"],
             "Hesabınıza 1.250,50 TL yatırıldı.": ["money", "acronym", "digits"],
             "Faiz %20'ye çıktı, yani yüzde 20.": ["percent", "digits", "symbol"],
             "SGK ve PTT şubeleri açık.": ["acronym"], "26.09.2027 tarihinde gelin.": ["date", "digits"],
             "Sipariş numaranız 6493.": ["digits"], "Bugün hava çok güzel.": []}
    for text, kinds in cases.items():
        assert rc.number_kinds(text) == kinds, text
        assert bool(kinds) == bool(rc.RICH.search(text))  # the rich lines are exactly the ones with a kind
    assert rc.has_numbers({"text": "Saat 9'da."}) and not rc.has_numbers({"text": "x", "kinds": []})
    rows = [{"text_raw": t} for t in ("Kod 1234.", "Kod 5678.", "0212 444 55 66 arayın.", "Saat 10.30'da.", "Kod 9.")]
    assert [rc.number_kinds(r["text_raw"])[0] for r in rc.kind_spread(rows)] == ["phone", "time", "digits", "digits",
                                                                                 "digits"]


def test_neutral_draw_tops_up_lines_with_numbers():
    plain = [{"line_id": f"b{i}", "text_raw": f"Bankamızın şubeleri hafta içi her gün açık, uğrayabilirsiniz {w}.",
              "category": "banking", "tags": [], "lid_tr": 0.99} for i, w in enumerate(
        ["efendim", "lütfen", "dilerseniz", "her zaman", "rahatça", "isterseniz", "kolayca", "memnuniyetle",
         "gönül rahatlığıyla", "sabahtan"])]
    nums = [{"line_id": f"n{i}", "text_raw": t, "category": "numbers_codes", "tags": [], "lid_tr": 0.99}
            for i, t in enumerate(["Sipariş numaranız 4821 olarak kaydedildi.", "Kargonuz 14:30'da yola çıktı.",
                                   "Fatura tutarınız 340 TL olarak hesaplandı.", "0212 444 55 66 numarasını arayın.",
                                   "Randevunuz 12 Ekim günü sabah saatlerinde.", "İndirim oranı yüzde 15 oldu.",
                                   "Takip kodunuz 99812 ile sorgulanabilir."])]
    weights = {"banking": 1.0, "numbers_codes": 0.01}  # the quotas alone: all plain
    got, _ = rc.pick_domain(plain + nums, 10, weights, 0.7, rc.Admit(rc.pa.Guard()), 0)
    assert len(got) == 10 and sum(r["rich"] for r in got) == 0
    got, _ = rc.pick_domain(plain + nums, 10, weights, 0.7, rc.Admit(rc.pa.Guard()), 0, number_share=0.6)
    with_numbers = [r for r in got if r["rich"]]
    assert len(got) == 10 and len(with_numbers) == 6 and all(r["kinds"] for r in with_numbers)
    assert {r["kinds"][0] for r in with_numbers} >= {"phone", "time", "money", "date", "percent"}  # spread over kinds
    few, _ = rc.pick_domain(plain + nums[:2], 10, weights, 0.7, rc.Admit(rc.pa.Guard()), 0, number_share=0.6)
    assert len(few) == 10 and sum(r["rich"] for r in few) == 2  # only as far as the pool allows


def test_wave_order_keeps_the_share_of_numbers():
    pool = [{"id": f"n{i:02d}", "category": "neutral", "spoken": "x" * 80, "text": f"Kod {i}.", "kinds": ["digits"]}
            for i in range(20)]
    pool += [{"id": f"p{i:02d}", "category": "neutral", "spoken": "x" * 80, "text": "Düz cümle.", "kinds": []}
             for i in range(20)]
    pool.sort(key=lambda r: rc.rank(0, r["id"]))
    state = rc.new_state({})
    rows = rc.wave_rows(pool, state, {"neutral": 60.0}, 4, rc.CHARS_S, number_share=0.6)
    share = sum(rc.has_numbers(r) for r in rows) / len(rows)
    assert 0.55 <= share <= 0.7 and rc.has_numbers(rows[0])
    plain_order = rc.wave_rows(pool, state, {"neutral": 60.0}, 4, rc.CHARS_S)  # no weighting: the pool's order
    assert [r["id"] for r in plain_order] == [r["id"] for r in pool][:len(plain_order)]
    # kept seconds mostly plain: the next texts are the ones with numbers
    state["kept"] = {f"k{i}": {"id": f"k{i}", "category": "neutral", "duration": 5.0, "text": "Düz.", "kinds": []}
                     for i in range(6)}
    first = rc.wave_rows(pool, state, {"neutral": 90.0}, 4, rc.CHARS_S, number_share=0.6)[:6]
    assert all(rc.has_numbers(r) for r in first)
    # each class's measured pass rate once it has MIN_MEASURED decided tries
    for r in pool:
        state["rows"][r["id"]] = {"status": "kept" if r["id"] < "n05" else "dropped", "tries": 2}
    assert rc.class_pass(pool, state, True) == pytest.approx(5 / 40) and rc.class_pass(pool, state, False) == 0.02
    assert rc.class_pass(pool[:5], state, True) is None


def test_long_items_join_consecutive_lines_of_one_prompt():
    assert rc.doc_group(LONG_ROWS[1]) == ("qwen3:s3-p7", (0, 0)) and rc.doc_group({"doc_id": "x"}) == ("x", (0, 0))
    assert rc.doc_group({"line_id": 1}) is None
    items, drops = rc.build_long(LONG_ROWS, 5, rc.LONG_WEIGHTS, rc.Admit(rc.pa.Guard()), 0)
    by = {r["id"]: r for r in items}
    assert set(by) == {"Lp7k0", "Lp7k3", "Lp9k0"} and drops == {}  # p7k2 a question splits p7, p8 not drawn
    # p7k3-k5: two lines make 146 characters, three 214, nearer the middle of --long-target (190-280): three
    assert by["Lp7k0"]["line_ids"] == ["p7k0", "p7k1"] and by["Lp7k3"]["line_ids"] == ["p7k3", "p7k4", "p7k5"]
    assert by["Lp9k0"]["line_ids"] == ["p9k0", "p9k1"]  # in k order, not file order
    place = {r["line_id"]: rc.doc_group(r) for r in LONG_ROWS}
    for r in items:
        assert r["category"] == "long" and rc.LONG_CHARS[0] <= len(r["spoken"]) <= rc.LONG_CHARS[1]
        groups = {place[x][0] for x in r["line_ids"]}
        ks = [place[x][1][0] for x in r["line_ids"]]
        assert len(groups) == 1 and r["doc_group"] in groups and ks == list(range(ks[0], ks[0] + len(ks)))
        assert r["sentences"] == len(r["line_ids"]) == r["spoken"].count(". ") + 1  # every line's mark kept
        words = r["spoken"].split()
        assert r["breaks"] == [k for k, w in enumerate(words[:-1]) if w.endswith(".")]  # the joins
        assert not rc.JUNK.search(r["spoken"]) and r["source"] == rc.DOMAIN_REPO
    assert "imzanız beklenecek. İmzadan" in by["Lp7k3"]["text"] and "beklenecek. İmzadan" in by["Lp7k3"]["spoken"]
    assert by["Lp9k0"]["kinds"][:3] == ["phone", "money", "date"] and "iki yüz elli lira" in by["Lp9k0"]["spoken"]
    assert by["Lp7k0"]["domain_category"] == "long_explanations" and by["Lp9k0"]["domain_category"] == "banking"
    # the leakage guard looks at each line; lines in `have` (the pool's, --avoid-pool's) break a run
    guard = rc.pa.build_guard(["Bu süre içinde ek bir belge gerekirse sizinle telefonla iletişime geçeceğiz."])
    items, drops = rc.build_long(LONG_ROWS, 5, rc.LONG_WEIGHTS, rc.Admit(guard), 0)
    assert {r["id"] for r in items} == {"Lp7k3", "Lp9k0"} and drops == {"leakage guard": 1}
    items, _ = rc.build_long(LONG_ROWS, 5, rc.LONG_WEIGHTS, rc.Admit(rc.pa.Guard()), 0, have={"p9k1", "p7k4"})
    assert {r["id"] for r in items} == {"Lp7k0"}
    # so does a k the rows read do not have: p7k1 missing, p7k0 and p7k2.. are not neighbours
    runs, _ = rc.long_runs([r for r in LONG_ROWS if r["line_id"] != "p7k1"], rc.LONG_WEIGHTS, set())
    assert [[x["line_id"] for x in run] for _, run in runs["long_explanations"]] == [["p7k0"], ["p7k3", "p7k4", "p7k5"]]
    # lengths: 1-3 lines within --long-chars, each run's window nearest the middle of --long-target
    short = [_dl(f"s{k}", f"qwen3:s3-p5-{k}", f"Kısa bir cümle {w}.")
             for k, w in enumerate(["bir", "iki", "üç", "dört"])]
    assert rc.build_long(short, 5, rc.LONG_WEIGHTS, rc.Admit(rc.pa.Guard()), 0)[0] == []
    line = {"text_raw": "x" * 70}  # 71 characters with its mark: two make 143, three 215
    assert rc.long_windows([line] * 3, (120, 300)) == [[line] * 3]
    assert rc.long_windows([line] * 3, (120, 300), target=(120, 160)) == [[line] * 2]
    one = {"text_raw": "y" * 200}  # a long sentence alone (201) is nearer 235 than with the next line (273)
    assert rc.long_windows([one, line], (120, 300)) == [[one]]
    assert rc.long_windows([one] * 3, (120, 300), (2, 3)) == []  # two are too long
    assert rc.with_mark("Kargonuz yola çıktı,") == "Kargonuz yola çıktı." and rc.with_mark("Tamam! ") == "Tamam!"
    assert rc.with_mark('Dedi ki "tamam."') == 'Dedi ki "tamam."' and rc.with_mark("Ödeme:") == "Ödeme."
    # the draw: windows within --long-target first, whatever their seeded rank
    filler = "paketiniz bugün depodan çıktı ve yarın öğleden sonra adresinize teslim edilecek".split()
    rows = []
    for g, city in enumerate(["Ankara", "İzmir", "Bursa", "Konya", "Adana", "Samsun", "Trabzon", "Erzurum"]):
        rows += [_dl(f"g{g}k{k}", f"qwen3:s3-p{100 + g}-{k}", f"{city} {where} " + " ".join(filler[:6 + g % 4 * 2]))
                 for k, where in enumerate(["şubesindeki", "merkezindeki"])]
    lens = {r["line_ids"][0]: len(r["spoken"])
            for r in rc.build_long(rows, 99, {"long_explanations": 1}, rc.Admit(rc.pa.Guard()), 0)[0]}
    inside = {k for k, n in lens.items() if 190 <= n <= 280}
    assert len(inside) == 2 and len(lens) == 6  # two prompts too short; 144-184 characters outside
    for seed in range(5):
        got, _ = rc.build_long(rows, len(inside), {"long_explanations": 1}, rc.Admit(rc.pa.Guard()), seed)
        assert {r["line_ids"][0] for r in got} == inside


def test_long_item_breaks_are_the_joins_not_a_period_trnorm_left():
    lines = [_dl("a1", "qwen3:s3-p1-0", "Sayın Ahmet K. Yılmaz, randevunuz yarın"),
             _dl("a2", "qwen3:s3-p1-1", "Lütfen erken gelin.")]
    row, why, parts = rc.long_row(lines, "addresses_names", "qwen3:s3-p1", (10, 300))
    assert why == "" and row["spoken"] == "Sayın Ahmet ka. Yılmaz, randevunuz yarın. Lütfen erken gelin."
    assert row["breaks"] == [5] and row["sentences"] == 2 and len(parts) == 4
    # Whisper: the name read on, a pause after "yarın."
    words = _gapped("Sayın Ahmet ka Yılmaz, randevunuz yarın. Lütfen erken gelin.", 5.0, 0.4)
    m = rc.pause_measures(words, row["spoken"], breaks=row["breaks"])
    assert m["pause_sent"] == [0.4] and m["pause_mid"] == 0.0 and rc.pause_fails(m, row, G) == []
    old = rc.pause_measures(words, row["spoken"])  # every mark a boundary: "ka." asks for a pause in the name
    assert old["pause_sent"] == [0.0, 0.4] and rc.pause_fails(old, row, G) == ["pause"]


def test_pool_draws_long_items_after_the_neutral_lines_and_avoids_another_pool():
    rows = DOMAIN + LONG_ROWS
    pool, rec = rc.build_pool([], [], rc.Admit(rc.pa.Guard()), 3, rc.DOMAIN_WEIGHTS, 0.7, 0, lambda: (rows, {}),
                              long_lines=5, number_share=0.6)
    neutral = [r for r in pool if r["category"] == "neutral"]
    longs = [r for r in pool if r["category"] == "long"]
    assert len(neutral) == 3 and rec["long_added"] == len(longs) == 2
    assert "p9k1" in rc.line_ids_of(neutral) and {r["id"] for r in longs} == {"Lp7k0", "Lp7k3"}  # a banking line
    assert not rc.line_ids_of(neutral) & rc.line_ids_of(longs)  # drawn as a neutral row first: no Lp9k0
    # another corpus's pool: its texts and its lines are never taken
    avoid = ({"p7k0", "1", "3"}, {rc.dedup_key(rc.spoken_of(DOMAIN[5]["text_raw"]))})
    pool2, _ = rc.build_pool([], [], rc.Admit(rc.pa.Guard()), 3, rc.DOMAIN_WEIGHTS, 0.7, 0, lambda: (rows, {}),
                             long_lines=5, avoid=avoid)
    assert not rc.line_ids_of(pool2) & {"p7k0", "1", "3", "6"} and "Lp7k0" not in {r["id"] for r in pool2}
    assert rc.avoid_of([]) == (set(), set(), {"pools": {}, "line_ids": 0, "texts": 0})


def _gapped(hyp: str, dur: float, gap: float) -> list[list]:
    """Whisper-like words over a clip, spread by syllables, `gap` s of silence after each clause-final word."""
    w = hyp.split()
    marks = {i for i, x in enumerate(w[:-1]) if rc.si.terminal(x)}
    span = rc.si._spread(0.0, dur - gap * len(marks), [max(1, rc.si.syllables(x)) for x in w])
    out, shift = [], 0.0
    for i, (x, (t0, t1)) in enumerate(zip(w, span)):
        out.append([x, round(t0 + shift, 3), round(t1 + shift, 3)])
        shift += gap if i in marks else 0.0
    return out


def test_pause_gate_on_synthetic_timestamps():
    spoken = "Kredi başvurunuz alındı, sonuç yarın bildirilecek. Sorunuz olursa bizi arayın."
    long_row = {"category": "long"}
    ok = rc.pause_measures(_gapped(spoken, 6.0, 0.4), spoken)
    assert ok["pause_sent"] == [0.4] and ok["pause_mid"] == 0.0 and ok["pause_max"] == 0.4
    assert rc.pause_fails(ok, long_row, G) == []
    run_on = rc.pause_measures(_gapped(spoken, 6.0, 0.0), spoken)
    assert run_on["pause_sent"] == [0.0] and rc.pause_fails(run_on, long_row, G) == ["pause"]  # read on, no pause
    # Whisper left no gap but the clip is quiet there: the quiet run (onset.edges' pauses) counts
    words = _gapped(spoken, 6.0, 0.0)
    t = next(x[2] for x in words if x[0].endswith("."))
    heard = rc.pause_measures(words, spoken, [[t - 0.05, t + 0.25]])
    assert heard["pause_sent"] == [0.3] and rc.pause_fails(heard, long_row, G) == []
    assert rc.pause_fails(rc.pause_measures(_gapped(spoken, 6.0, 1.0), spoken), long_row, G) == ["pause"]  # > 0.9
    # inside a sentence: 0.5 s after "alındı," is fine, 0.7 s is not; anywhere over 1.2 s fails
    words = _gapped(spoken, 6.0, 0.4)
    comma = next(x for x in words if x[0] == "alındı,")
    mid = rc.pause_measures(words, spoken, [[comma[2], comma[2] + 0.5]])
    assert mid["pause_mid"] == 0.5 and rc.pause_fails(mid, long_row, G) == []
    mid = rc.pause_measures(words, spoken, [[comma[2], comma[2] + 0.7]])
    assert rc.pause_fails(mid, long_row, G) == ["pause"]
    assert rc.pause_fails(mid, long_row, rc.Gates(pause_mid=0.8)) == []
    assert rc.pause_fails({**ok, "pause_max": 1.3}, long_row, G) == ["pause"]
    # other categories have no pause gate; a long clip without the measures fails
    assert rc.pause_fails({}, {"category": "neutral"}, G) == [] and rc.pause_fails({}, long_row, G) == ["pause"]
    assert rc.pause_measures([["merhaba", 0.0, 1.0]], spoken) == {}  # the words do not match
    assert rc.clip_fails({**ok, "rate_ws": 5.0}, long_row, G) == [] and "pause" in rc.clip_fails(run_on, long_row, G)


def test_long_tries_durations_and_estimate():
    row = {"id": "L1", "category": "long", "spoken": "x" * 200, "sentences": 3}
    assert rc.max_tries(row, 4, 6) == 6 and rc.max_tries(row, 4) == 4 and rc.max_tries({"category": "short"}, 4, 6) == 4
    assert rc.est_dur(row) == pytest.approx(rc.DUR_PAD_S + 200 / rc.CHARS_S + 2 * rc.LONG_PAUSE_S)
    assert rc.est_dur({**row, "category": "neutral"}) == pytest.approx(rc.DUR_PAD_S + 200 / rc.CHARS_S)
    assert rc.wave_tries(row, {"tries": 4}, 4, 2, [], 6) == ([4, 5], 2)
    args = rc.parse_args(["--out", "x", "--hours", "1", "--mix", "long=0.5,neutral=0.5", "--min-sim", "0.6"])
    assert (args.long_tries, args.long_lo, args.long_hi, args.pause_lo, args.pause_hi, args.pause_mid,
            args.pause_max, args.number_share) == (6, 120.0, 300.0, 0.12, 0.9, 0.6, 1.2, 0.6)
    assert (args.target_lo, args.target_hi, args.min_pass) == (190.0, 280.0, 0.05)
    planned = rc.plan_pool([], [], args)
    assert sum(r["category"] == "long" for r in planned) == rc.LONG_LINES == rc.long_wanted(args)
    assert rc.long_wanted(rc.parse_args(["--out", "x"])) == 0  # no long share: none drawn
    cost = rc.Cost(render_per_audio=1 / 37.5)
    est = rc.estimate(planned, rc.new_state({}), args, cost)
    b, n = est["by"]["long"], est["by"]["neutral"]
    d = rc.DUR_PAD_S + rc.LONG_PLAN_CHARS / rc.CHARS_S + (rc.LONG_PLAN_SENT - 1) * rc.LONG_PAUSE_S
    keep = 1 - (1 - 0.10) ** 6
    assert b["p_try"] == 0.1 and b["p_keep"] == round(keep, 3) and b["clip_s"] == round(d, 2)
    assert b["texts"] == math.ceil(1800 / (d * keep)) and est["waves"] == 5  # 6 tries, 2 a wave, + 1, + the probe
    assert b["probe"] == rc.PROBE_TEXTS and "probe" not in est["by"]["neutral"]
    assert n["p_try"] == 0.407 and n["p_keep"] == round(1 - 0.593 ** 4, 3)
    render = (b["rendered_s"] * rc.LONG_RENDER + n["rendered_s"]) / 37.5 * 1.2
    assert est["render_s"] == pytest.approx(render, rel=1e-3)
    assert "long clips x 1.3" in rc.estimate_lines(est)[-1]
    for bad in (["--long-chars", "300,120"], ["--pause-sent", "0.9"], ["--number-share", "2"], ["--long-tries", "0"],
                ["--long-weights", "banking:0"], ["--pause-max", "0.5"], ["--long-target", "100,200"],
                ["--long-chars", "120,200", "--long-target", "150,250"], ["--min-pass", "1"]):
        with pytest.raises(SystemExit):
            rc.parse_args(["--out", "x", *bad])


def test_a_guessed_pass_rate_is_measured_first_and_a_hopeless_category_stops():
    pool = [{"id": f"L{i:03d}", "category": "long", "spoken": "x" * 220, "text": "", "sentences": 3}
            for i in range(100)]
    pool += [{"id": f"n{i:03d}", "category": "neutral", "spoken": "x" * 80, "text": "", "kinds": []} for i in range(50)]
    state, targets = rc.new_state({}), {"long": 3600.0, "neutral": 600.0}
    first = rc.wave_rows(pool, state, targets, 4, rc.CHARS_S, long_tries=6, min_pass=0.05)
    assert sum(r["category"] == "long" for r in first) == rc.PROBE_TEXTS  # the guess: one probe's worth in flight
    assert sum(r["category"] == "neutral" for r in first) > rc.PROBE_TEXTS  # neutral's rate is measured elsewhere
    for r in pool[:10]:
        state["rows"][r["id"]] = {"status": "open", "tries": 2, "fails": []}
    longs = [r for r in rc.wave_rows(pool, state, targets, 4, rc.CHARS_S, long_tries=6) if r["category"] == "long"]
    assert len(longs) == rc.PROBE_TEXTS and longs[:10] == pool[:10]  # the open ones first, 30 new
    s = rc.cat_stats(state, "long")
    s.update(decided_g2=40, passed_g2=8)  # measured: 0.2 a try, no cap
    assert rc.measured_pass(state, "long") == 0.2 and not rc.probing(state, "long")
    assert sum(r["category"] == "long" for r in rc.wave_rows(pool, state, targets, 4, rc.CHARS_S, long_tries=6,
                                                            min_pass=0.05)) > rc.PROBE_TEXTS
    s.update(decided_g2=40, passed_g2=1)  # 0.025 < --min-pass: its open texts only
    assert rc.stopped_cats(state, 0.05) == {"long": 0.025} and rc.stopped_cats(state, 0.0) == {}
    rows = rc.wave_rows(pool, state, targets, 4, rc.CHARS_S, long_tries=6, min_pass=0.05)
    assert [r for r in rows if r["category"] == "long"] == pool[:10]
    assert sum(r["category"] == "long" for r in rc.wave_rows(pool, state, targets, 4, rc.CHARS_S, long_tries=6)) > 10
    # the estimate: no new long texts, its deficit not handed to neutral, not counted as work to do
    args = rc.parse_args(["--out", "x", "--hours", "1", "--mix", "long=0.5,neutral=0.5"])
    est = rc.estimate(pool, state, args, rc.Cost(render_per_audio=1 / 37.5))
    b = est["by"]["long"]
    assert b["stopped"] == 0.025 and b["pool"] == 10 and b["texts"] <= 10 and "probe" not in b
    assert est["targets"]["neutral"] == 1800 and est["left_s"] == 1800 and est["stopped_left_s"] == 1800
    assert "STOPPED: 0.025 of its tries pass, under --min-pass 0.05" in " ".join(rc.estimate_lines(est))
    rep = rc.build_report(state, pool, args, [], {}, {})
    assert rep["stopped"] == {"long": 0.025} and not rep["complete"]


class GapScoring(FakeScoring):
    """FakeScoring with Whisper's words of long clips 0.4 s apart after each sentence; the first try of NO_GAP's
    texts read on without a pause."""

    NO_GAP = {"Lp7k0"}

    def text(self, atts, until):
        super().text(atts, until)
        for a in atts:
            if a["category"] == "long":
                gap = 0.0 if a["rid"] in self.NO_GAP and a["t"] == 0 else 0.4
                a["words"] = _gapped(a["hyp"], a["duration"], gap)


DOMAIN2 = [  # new neutral lines for the second corpus (the first one's are avoided)
    {"line_id": 21, "text_raw": "Siparişiniz 14:30'da kargoya verildi ve yarın teslim edilecek.",
     "category": "ecommerce_cargo", "tags": [], "lid_tr": 0.99},
    {"line_id": 22, "text_raw": "Faturanızın tutarı 340 TL olarak hesaplandı.", "category": "numbers_codes",
     "tags": [], "lid_tr": 0.99},
    {"line_id": 23, "text_raw": "Randevunuz cuma sabahı kardiyoloji polikliniğinde olacak.",
     "category": "appointments_clinics", "tags": [], "lid_tr": 0.99},
    {"line_id": 24, "text_raw": "Müşteri hizmetlerimize her gün ulaşabilirsiniz.", "category": "customer_support",
     "tags": [], "lid_tr": 0.99},
]


def _snapshot(d: Path) -> dict:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def test_a_second_corpus_with_long_items_beside_the_first_then_the_merge(tmp_path, audio_libs, monkeypatch, capsys):
    monkeypatch.setattr(rc, "BASELINE_MIN", 3)
    first = _args(tmp_path, "--hours", "1")
    assert rc.run(first, render=FakeEngine(), scoring=FakeScoring, domain=_domain) == 0
    a = tmp_path / "corpus"
    before = _snapshot(a)
    a_state = json.loads((a / "state.json").read_text(encoding="utf-8"))
    assert set(a_state["emotion_refs"]) == {"angry", "sad"}
    b = tmp_path / "corpus2"
    argv = ["--out", str(b), "--ref", first.ref, "--textbank", first.textbank, "--throughput-log", "",
            "--laughter-usd", "0", "--laugh-model", "none", "--mix", "long=0.5,neutral=0.5", "--hours", "1",
            "--domain-lines", "4", "--long-lines", "3", "--min-sim", "0.6", "--refs-from", str(a), "--avoid-pool",
            str(a), "--domain-weights", "numbers_codes:0.3,appointments_clinics:0.3,ecommerce_cargo:0.2,"
            "customer_support:0.2"]  # not banking: the long items' banking lines stay for them
    eng = FakeEngine()
    rows = DOMAIN + DOMAIN2 + LONG_ROWS
    assert rc.run(rc.parse_args(argv), render=eng, scoring=GapScoring,
                  domain=lambda: (rows, {"repo": rc.DOMAIN_REPO, "rows_read": len(rows)})) == 0
    assert _snapshot(a) == before  # the first corpus only read
    state = json.loads((b / "state.json").read_text(encoding="utf-8"))
    pool = _rows(b / "texts.jsonl")
    a_pool = _rows(a / "texts.jsonl")
    assert {r["category"] for r in pool} == {"neutral", "long"}  # the textbank's texts: all in the first pool
    assert not rc.line_ids_of(pool) & rc.line_ids_of(a_pool)
    assert not {rc.dedup_key(r["spoken"]) for r in pool} & {rc.dedup_key(r["spoken"]) for r in a_pool}
    assert state["pool"]["avoided"]["line_ids"] == len(rc.line_ids_of(a_pool))
    kept = state["kept"]
    assert {"Lp7k0", "Lp7k3", "Lp9k0", "d21", "d22", "d23", "d24"} <= set(kept)
    assert kept["Lp7k0"]["try"] == 1 and kept["Lp7k0"]["pause_sent"] == [0.4] and kept["Lp7k3"]["try"] == 0
    assert kept["Lp7k3"]["pause_sent"] == [0.4, 0.4] and len(kept["Lp7k3"]["breaks"]) == 2  # three lines
    att = _rows(b / "attempts.jsonl")
    assert next(x for x in att if x["id"] == "Lp7k0" and x["t"] == 0)["fails"] == ["pause"]
    assert kept["Lp9k0"]["train_text"] == kept["Lp9k0"]["spoken"] and kept["Lp9k0"]["sentences"] == 2
    assert kept["Lp9k0"]["line_ids"] == ["p9k0", "p9k1"] and kept["Lp9k0"]["duration"] > 10
    assert not any(j["key"].startswith("prelude/") for j in eng.jobs)  # no emotions, no laughter trial
    # the first corpus's references: the voice's (the same clip) and the emotions' (with their state records)
    raw = "ref/designed-f022-s0.raw.wav"
    assert (b / raw).read_bytes() == (a / raw).read_bytes()
    assert (b / "refs" / "sad.wav").read_bytes() == (a / "refs" / "sad.wav").read_bytes()
    assert state["emotion_refs"]["sad"]["from"] == str(a) and state["refs_from"]["emotion_refs"] == ["angry", "sad"]
    assert state["key"]["ref_sha256"] == a_state["key"]["ref_sha256"]
    rep = json.loads((b / "prep_report.json").read_text(encoding="utf-8"))
    assert set(rep["by_category"]) == {"neutral", "long"} and rep["complete"] and rep["stopped"] == {}
    assert rep["by_category"]["long"]["fails"] == {"pause": 1} and rep["distributions"]["long"]["pause_fails"] == 1
    assert rep["distributions"]["numbers"]["pool_share"] == 0.5 and rep["distributions"]["numbers"]["target"] == 0.6
    md = (b / "PREP_REPORT.md").read_text(encoding="utf-8")
    assert "| long |" in md and "pause at a sentence boundary" in md and "written numbers in neutral rows" in md
    assert "long passages (1-3 consecutive lines joined)" in (b / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    # refused: another reference than the first corpus's, a missing pool to avoid, --out itself
    other = _wav(tmp_path / "other.wav", 3.0)
    capsys.readouterr()
    assert rc.run(rc.parse_args([*argv[:2], "--ref", str(other), *argv[4:]])) == 2
    assert "another clip" in capsys.readouterr().out
    assert rc.run(rc.parse_args([*argv, "--avoid-pool", str(tmp_path / "nope")])) == 2
    assert rc.run(rc.parse_args([*argv, "--avoid-pool", str(b)])) == 2
    # the merge: one manifest of both, rows as they are
    mix = tmp_path / "mix"
    assert rc.main(["--out", str(mix), "--merge", str(a), str(b)]) == 0
    merged = _rows(mix / "voxcpm_train.jsonl")
    ra, rb = _rows(a / "voxcpm_train.jsonl"), _rows(b / "voxcpm_train.jsonl")
    assert sorted(map(json.dumps, merged)) == sorted(map(json.dumps, ra + rb))
    mrep = json.loads((mix / "prep_report.json").read_text(encoding="utf-8"))
    assert mrep["merged"] and mrep["complete"] and mrep["selected"]["clips"] == len(ra) + len(rb)
    assert mrep["by_category"]["long"]["clips"] == sum(c["category"] == "long" for c in kept.values())
    assert [p["manifest_bytes"] for p in mrep["sources"]] == [(a / "voxcpm_train.jsonl").stat().st_size,
                                                             (b / "voxcpm_train.jsonl").stat().st_size]
    attribution = (mix / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    assert "long passages" in attribution and "Questions, emotional lines" in attribution


def _tiny_corpus(d: Path, rows: list[tuple[str, str, float, str]], complete: bool = True) -> Path:
    """A finished corpus-render dir: marker, WAVs, manifest (absolute paths; the first clip the last one's
    ref_audio), clips.jsonl, attribution, report (after the state), status done."""
    (d / "wav").mkdir(parents=True)
    (d / rc.MARKER).write_text("x\n", encoding="utf-8")
    man, clips = [], []
    for cid, text, dur, cat in rows:
        _wav(d / "wav" / f"{cid}.wav", dur)
        man.append({"audio": str(d / "wav" / f"{cid}.wav"), "text": text, "duration": dur})
        clips.append({"id": cid, "wav": f"{cid}.wav", "category": cat, "duration": dur, "train_text": text})
    if len(man) > 1:
        man[-1].update(ref_audio=man[0]["audio"], ref_duration=rows[0][2])
    vc.write_rows(d / "clips.jsonl", clips)
    vc.write_rows(d / "voxcpm_train.jsonl", man)
    (d / "ATTRIBUTION.txt").write_text(f"source {d.name}\nshared line\n", encoding="utf-8")
    vc.write_json(d / "state.json", {"kept": {}})
    time.sleep(0.02)
    vc.write_json(d / "prep_report.json", {"complete": complete, "selected": {"hours": 0.01, "clips": len(rows)},
                                           "target_hours": 0.01, "time": "t"})
    vc.write_json(d / "status.json", {"phase": "done"})
    return d


def test_merge_union_without_duplicates_and_the_train_preflight(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    a = _tiny_corpus(tmp_path / "srcA", [("a1", "Bir.", 3.0, "neutral"), ("a2", "İki.", 2.0, "short"),
                                         ("laugh_1", "(Laughing.)Haha!", 1.5, "laughter"),
                                         ("laugh_2", "(Laughing.)Haha!", 1.6, "laughter")])
    b = _tiny_corpus(tmp_path / "srcB", [("b1", "Üç uzun cümle. Dört.", 12.0, "long"), ("b2", "İki.", 2.0, "neutral"),
                                         ("b3", "Beş.", 2.0, "neutral")])
    vc.write_rows(b / "voxcpm_train.jsonl", _rows(b / "voxcpm_train.jsonl") + [_rows(a / "voxcpm_train.jsonl")[0]])
    time.sleep(0.02)
    vc.write_json(b / "prep_report.json", json.loads((b / "prep_report.json").read_text(encoding="utf-8")))
    mix = tmp_path / "mix"
    assert rc.main(["--out", str(mix), "--merge", str(a), str(b), "--dry-run"]) == 0 and not mix.exists()
    assert rc.main(["--out", str(mix), "--merge", str(a), str(b)]) == 0
    rows = _rows(mix / "voxcpm_train.jsonl")
    texts = [r["text"] for r in rows]
    assert len(rows) == 6 and texts.count("(Laughing.)Haha!") == 2  # a source's own repeats stay
    assert texts.count("İki.") == 1 and len({r["audio"] for r in rows}) == 6  # b2: a2's text; a1 again: once
    assert all(Path(r["audio"]).is_absolute() and r["audio"].startswith(str(tmp_path)) for r in rows)
    assert sum("ref_audio" in r for r in rows) == 2 and all(set(r) <= set(rc.MERGED_KEYS) for r in rows)
    rep = json.loads((mix / "prep_report.json").read_text(encoding="utf-8"))
    assert rep["sources"][1]["dropped"] == {"same text as an earlier source": 1, "same audio": 1}
    assert rep["by_category"]["long"]["clips"] == 1 and rep["selected"]["clips"] == 6 and rep["complete"]
    assert (mix / "ATTRIBUTION.txt").read_text(encoding="utf-8") == "source srcA\nshared line\nsource srcB\n"
    assert "Merged training corpus" in (mix / "README.md").read_text(encoding="utf-8")
    assert "| long | 1 |" in (mix / "PREP_REPORT.md").read_text(encoding="utf-8")
    assert rc.main(["--out", str(mix), "--merge", str(a), str(b)]) == 0  # again: the merge replaces itself
    # refused: a render going, a source not complete (unless --allow-incomplete), a lost WAV, --out a source
    vc.write_json(b / "status.json", {"phase": "waves", "updated": "now"})
    capsys.readouterr()
    assert rc.main(["--out", str(tmp_path / "m2"), "--merge", str(a), str(b)]) == 2
    assert "phase 'waves'" in capsys.readouterr().out and not (tmp_path / "m2").exists()
    vc.write_json(b / "status.json", {"phase": "done"})
    c = _tiny_corpus(tmp_path / "srcC", [("c1", "Altı.", 2.0, "neutral")], complete=False)
    assert rc.main(["--out", str(tmp_path / "m2"), "--merge", str(a), str(c)]) == 2
    assert rc.main(["--out", str(tmp_path / "m2"), "--merge", str(a), str(c), "--allow-incomplete"]) == 0
    m2 = json.loads((tmp_path / "m2" / "prep_report.json").read_text(encoding="utf-8"))
    assert m2["incomplete_sources"] == [str(c)] and m2["complete"]
    (c / "wav" / "c1.wav").unlink()
    assert rc.main(["--out", str(tmp_path / "m3"), "--merge", str(a), str(c), "--allow-incomplete"]) == 2
    assert rc.main(["--out", str(a), "--merge", str(a), str(b)]) == 2
    assert rc.main(["--out", str(tmp_path / "m4"), "--merge", str(a)]) == 2
    assert rc.main(["--out", str(b / "sub"), "--merge", str(a), str(b)]) == 2
    # modal_app's train preflight takes the merge while its sources' manifests are what was merged
    ma, _ = _modal_app()
    vol = {"f022mix": mix, "f022corpus": a, "f022corpus2": b}
    for p in rep["sources"]:  # the Volume's paths
        p["dir"] = {str(a): "/data/f022corpus", str(b): "/data/f022corpus2"}[p["dir"]]
    vc.write_json(mix / "prep_report.json", rep)

    def files(rel):
        d = vol.get(rel)
        return {p.name: SimpleNamespace(path=f"{rel}/{p.name}", size=p.stat().st_size, mtime=p.stat().st_mtime,
                                        type=None) for p in d.iterdir()} if d is not None and d.is_dir() else {}

    def vjson(rel):
        d, _, name = rel.rpartition("/")
        return vc.read_json(vol[d] / name) if d in vol else None

    monkeypatch.setattr(ma, "_vol_files", files)
    monkeypatch.setattr(ma, "_vol_json", vjson)
    capsys.readouterr()
    ns = ma._preflight_train(["--data", "/data/f022mix"], "voxcpm2-f022-s3", 1500)
    assert ma._vol_path(ns.data) == "/data/f022mix" and "merged from" in capsys.readouterr().out
    assert ma.merged_stale(rep, {"/data/f022corpus": rep["sources"][0]["manifest_bytes"],
                                 "/data/f022corpus2": rep["sources"][1]["manifest_bytes"]}) == []
    with (b / "voxcpm_train.jsonl").open("a", encoding="utf-8") as f:  # the second corpus went on after the merge
        f.write(json.dumps({"audio": str(b / "wav" / "b3.wav"), "text": "Yedi.", "duration": 2.0}) + "\n")
    with pytest.raises(SystemExit, match="changed since the merge.*corpus-merge"):
        ma._preflight_train(["--data", "/data/f022mix"], "voxcpm2-f022-s3", 1500)
    assert ma.merged_stale(rep, {})[0].endswith("is not there any more")


def test_modal_app_corpus_out_and_merge_actions(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    ma, mm = _modal_app()
    assert ma.corpus_dir("") == ma.F022 and ma.corpus_dir("/data/f022corpus2/") == "/data/f022corpus2"
    for bad in ("/data/alania16k", "/data/x/y", "f022corpus2", "/tmp/x", "/data/../etc", "/data/a b"):
        with pytest.raises(ValueError):
            ma.corpus_dir(bad)
    cmd = ma.corpus_cmd(["--hours", "1.5"], 9.0, "nanovllm", "/data/f022corpus2")
    assert cmd[cmd.index("--out") + 1] == "/data/f022corpus2" and cmd[2:4] == ["--hours", "1.5"]
    assert ma.corpus_local("/data/f022corpus2") == ma.CORPUS_DIR / "f022corpus2"
    assert ma.corpus_local(ma.F022) == ma.CORPUS_DIR / "f022"
    assert ma.merge_cmd(list(ma.MERGE_SOURCES)) == ["python", "tts/voxcpm/render_corpus.py", "--out", ma.MERGE_OUT,
                                                    "--merge", "/data/f022corpus", "/data/f022corpus2"]
    assert ma.merge_cmd(["/data/a", "/data/b"], "/data/m", True)[-1] == "--allow-incomplete"
    fn = [c for c in mm.App.return_value.function.call_args_list if c.kwargs.get("timeout") == ma.MERGE_TIMEOUT_S]
    assert len(fn) == 1 and "gpu" not in fn[0].kwargs and fn[0].kwargs["cpu"] == ma.MERGE_CPU
    # corpus-merge on the local machine: a source whose render is going refuses, a dry run starts nothing
    entry = SimpleNamespace(size=10, mtime=2.0, type=None, path="p")
    old = SimpleNamespace(size=10, mtime=1.0, type=None, path="p")
    vol_files = {"f022corpus": {"voxcpm_train.jsonl": old, "prep_report.json": entry, "state.json": old},
                 "f022corpus2": {"voxcpm_train.jsonl": old, "prep_report.json": entry, "state.json": old}}
    vol_json = {"f022corpus/prep_report.json": {"complete": True, "selected": {"hours": 3.53, "clips": 3660}},
                "f022corpus2/prep_report.json": {"complete": True, "selected": {"hours": 1.5, "clips": 900}},
                "f022corpus2/status.json": {"phase": "waves"}}
    monkeypatch.setattr(ma, "_rc", lambda: rc)
    monkeypatch.setattr(ma, "_vol_files", lambda rel: vol_files.get(rel, {}))
    monkeypatch.setattr(ma, "_vol_json", lambda rel: vol_json.get(rel))
    calls = []
    monkeypatch.setattr(ma, "corpus_merge", SimpleNamespace(
        remote=lambda s, o, allow, unaudited, ins: calls.append((s, o, allow)) or json.dumps({"rc": 0, "used_s": 35})))
    with pytest.raises(SystemExit, match="phase 'waves'"):
        ma._corpus_merge([])
    vol_json["f022corpus2/status.json"] = {"phase": "done"}
    vol_json["f022corpus2/prep_report.json"]["complete"] = False
    with pytest.raises(SystemExit, match="not complete"):
        ma._corpus_merge([])
    ma._corpus_merge(["--allow-incomplete", "--dry-run"])
    assert "~5.03 h before duplicates" in capsys.readouterr().out and not calls
    ma._corpus_merge(["--allow-incomplete"])
    assert calls == [(["/data/f022corpus", "/data/f022corpus2"], "/data/f022mix", True)]
    vol_files["f022corpus2"]["state.json"] = SimpleNamespace(size=10, mtime=3.0, type=None, path="p")
    with pytest.raises(SystemExit, match="cut short"):
        ma._corpus_merge(["--allow-incomplete"])
    for bad in (["--hours", "5"], ["--sources", "/data/f022corpus"], ["--sources", "/data/f022corpus,/data/models"]):
        with pytest.raises(SystemExit):
            ma._corpus_merge(bad)
    with pytest.raises(SystemExit, match="corpus-merge"):
        ma._corpus(["--merge", "/data/a", "/data/b"], 4.0)
    # corpus-render --out: the plan of a second corpus (long items, numbers), refused without the corpus it reuses
    vol_files["f022corpus"].update({"texts.jsonl": entry, "ref": entry})
    vol_json["f022corpus/state.json"] = {"emotion_refs": {"sad": {"wav": "refs/sad.wav"}}}
    monkeypatch.setattr(ma, "_vol_rows", lambda rel: [])
    monkeypatch.setattr(ma, "corpus", SimpleNamespace(remote=lambda *a: pytest.fail("no GPU in a dry run")))
    argv = ["--hours", "1.5", "--mix", "long=0.5,neutral=0.5", "--min-sim", "0.6", "--laughter-usd", "0",
            "--refs-from", "/data/f022corpus", "--avoid-pool", "/data/f022corpus", "--dry-run"]
    ma._corpus(argv, 3.0, out="/data/f022corpus2")
    text = capsys.readouterr().out
    assert "corpus-render -> /data/f022corpus2" in text and "long " in text and "long clips x 1.3" in text
    assert "prelude" in text and "--dry-run: nothing started" in text
    assert re.search(r"the rest of /data/f022corpus2: ~\d+ call\(s\) of --max-usd 3 ", text)
    # the whole rest in calls: each bills its start-up etc., each later one a server + scorer loads again
    assert ma.corpus_calls(1000.0, 120.0, 2000.0, 137.0, 212.0) == (1, 1257.0)
    assert ma.corpus_calls(5816.0, 120.0, 1860.0, 137.0, 212.0) == (4, 5936.0 + 4 * 137 + 3 * 212)
    assert ma.corpus_calls(3700.0, 0.0, 1860.0, 137.0, 212.0)[0] == 3  # 3720 s of work but a restart: three
    with pytest.raises(SystemExit, match="no texts.jsonl"):
        ma._corpus([*argv, "--avoid-pool", "/data/f022other"], 3.0, out="/data/f022corpus2")
    with pytest.raises(SystemExit, match="--out itself"):
        ma._corpus(argv, 3.0, out="/data/f022corpus")
    with pytest.raises(SystemExit, match="corpus folder"):
        ma._corpus(argv, 3.0, out="/data/alania16k")


# ---- the c-check, cwords, the c audit -----------------------------------------------------------------------------


CWORDS_ROWS = [
    {"id": "e_surprised_c001", "emotion": "surprised", "text": "Vay canına, bu kadar hızlı mı bitirdin?",
     "instruction": "Astonished, with rising intonation.", "instruction_split": "train", "focus": "word-initial c"},
    {"id": "e_cheerful_c001", "emotion": "cheerful", "text": "Canım benim, seni görmek ne güzel!",
     "instruction": "Cheerful and warm, smiling while speaking.", "instruction_split": "train"},
    {"id": "e_angry_c001", "emotion": "angry", "text": "Cevap vermeyecek misin artık?",
     "instruction": "Angry tone, voice getting louder.", "instruction_split": "train"},
    {"id": "e_held_c001", "emotion": "surprised", "text": "Cüzdanım yine nerede kaldı?",
     "instruction": "Stunned beyond words.", "instruction_split": "heldout"},  # evaluation only
    {"id": "e_calm_x001", "emotion": "calm", "text": "Yavaşça nefes alalım.", "instruction": "Calm, slow.",
     "instruction_split": "train"},  # no c
    {"id": "e_probe_c001", "emotion": "surprised", "text": "Cam kırılmış, cadde de kapanmış! Şimdi ne yapacağız?",
     "instruction": "Astonished, with rising intonation.", "instruction_split": "train"},  # probe_canina2 k7
    {"id": "e_probe_c002", "emotion": "surprised", "text": "Bütün bu yemekleri tek başına hazırladığına inanamıyorum! "
     "Cidden mi?", "instruction": "Pleasantly surprised.", "instruction_split": "train"},  # holds a probe sentence
]
DOMAIN_C = [{"line_id": 31, "text_raw": "Cuma günü kargonuz caddedeki şubemize ulaşacak.",
             "category": "ecommerce_cargo", "tags": [], "lid_tr": 0.99}]  # a plain row with c's: never c-checked


def _cwords(tmp_path: Path, rows=CWORDS_ROWS) -> Path:
    p = tmp_path / "cwords.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p


def test_cwords_rows_pool_and_the_probe_guard(tmp_path):
    rows, bad, sha = rc.load_cwords(_cwords(tmp_path))
    assert [r["id"] for r in rows] == ["e_surprised_c001", "e_cheerful_c001", "e_angry_c001", "e_probe_c001",
                                       "e_probe_c002"] and len(sha) == 64
    assert bad == {"cwords: held-out instruction wording (evaluation only)": 1, "no c in the spoken form": 1}
    r = rows[0]
    assert (r["category"], r["source"], r["emotion"], r["instruction"]) == (
        "cwords", rc.CWORDS_SOURCE, "surprised", "Astonished, with rising intonation.")
    assert rc.train_text(r) == "(Astonished, with rising intonation.)Vay canına, bu kadar hızlı mı bitirdin?"
    assert rc.load_cwords(tmp_path / "none.jsonl")[0] == []
    # the canına probes are in the guard: a probe sentence, or a text holding one, never enters a pool
    guard, files = rc.make_guard(REPO / "benchmarks" / "tts" / "sets", [])
    assert [Path(f).name for f in files if "probe" in f] == ["probe_canina.jsonl", "probe_canina2.jsonl",
                                                             "probe_tr_instr.jsonl"]
    tb = [{"id": "e_angry_x", "category": "emotions", "text": "Cevap vermeyecek misin artık?",
           "spoken": "Cevap vermeyecek misin artık?", "emotion": "angry", "instruction": "Angry.",
           "source": "textbank"}]
    pool, rec = rc.build_pool([], tb, rc.Admit(guard), 0, rc.DOMAIN_WEIGHTS, 0.7, 0, None, cwords=rows)
    assert sorted(r["id"] for r in pool) == ["e_angry_x", "e_cheerful_c001", "e_surprised_c001"]
    assert rec["cwords_added"] == 2 and rec["cwords_dropped"] == {"leakage guard": 2, "duplicate text": 1}
    for p in rc.probe_files():  # every probe sentence (canonical and spelt) is guarded
        for x in vc.read_rows(p):
            for t in [x["text"], *x["refs"]]:
                assert rc.Admit(guard).why({"spoken": rc.spoken_of(t), "text": t}) == "leakage guard"
    # the repo's cwords: 90 rows, every one with a c, none guarded
    real, real_bad, _ = rc.load_cwords(REPO / rc.CWORDS)
    assert len(real) == 90 and not real_bad and all(rc.needs_c(x) for x in real)
    assert not any(guard.hit(rc.pa.eval_key(x["spoken"]), rc.pa.eval_key(x["text"])) for x in real)
    # only a cwords share puts them in the pool
    tbp = _textbank(tmp_path)
    for mix, n in (("neutral=1", 0), ("neutral=0.5,cwords=0.5", 3)):
        args = rc.parse_args(["--out", str(tmp_path / "c"), "--textbank", str(tbp), "--cwords",
                              str(_cwords(tmp_path)), "--mix", mix, "--no-domain"])
        pool, rec = rc.pool_from_files(args, tmp_path / "c", [], None)
        assert sum(r["category"] == "cwords" for r in pool) == n and ("cwords" in rec) == bool(n)


def test_the_c_gate_only_on_instructed_rows_with_a_c():
    emo = {"id": "e", "category": "emotions", "spoken": "Vay canına!", "instruction": "Astonished."}
    assert rc.needs_c(emo) and rc.needs_c({**emo, "category": "cwords"})
    assert not rc.needs_c({**emo, "spoken": "Vay be!"}) and not rc.needs_c({**emo, "instruction": None})
    assert not rc.needs_c({"id": "d", "category": "neutral", "spoken": "Cuma günü caddede."})
    assert not rc.needs_c(emo, "Vay be!") and rc.needs_c({**emo, "spoken": "x"}, "Cam!")  # the text a try read
    a = {"model_text": "Vay canına!", "c_expected": 1, "c_got": 0}
    assert rc.c_fails(a, emo, G) == [rc.C_FAIL] == ["c_dj"]
    assert rc.c_fails({**a, "c_got": 1}, emo, G) == [] and rc.c_fails({**a, "c_got": 2}, emo, G) == []
    assert rc.c_fails({"model_text": "Vay canına!", "c_error": "x"}, emo, G) == ["c_dj"]  # unreadable: fails
    assert rc.c_fails(a, {"id": "d", "category": "neutral", "spoken": "Vay canına!"}, G) == []
    assert rc.c_fails(a, emo, rc.Gates(c_check=False)) == []
    args = rc.parse_args(["--out", "x", "--no-c-check"])
    assert not args.c_check and not rc.gates_of(args).c_check and rc.parse_args(["--out", "x"]).c_check
    assert rc.max_tries({"category": "cwords"}, 4, 6, 8) == 8 and rc.max_tries({"category": "cwords"}, 4, 6) == 4
    assert rc.wave_tries({"category": "cwords"}, {"tries": 6}, 4, 2, [], 6, 8) == ([6, 7], 2)
    assert rc.utmos_key({"category": "cwords", "emotion": "surprised"}) == "cwords/surprised"
    assert rc.SPEED_MAX["cwords"] == rc.SPEED_MAX["emotions"] and "cwords" in rc.GUESSED
    for bad in (["--cwords-tries", "0"], ["--mix", "cwords=-1"]):
        with pytest.raises(SystemExit):
            rc.parse_args(["--out", "x", *bad])


def _cw_argv(tmp_path: Path, out: str, *extra: str) -> list[str]:
    ref = _wav(tmp_path / "refsrc" / "f022-s0.wav", 4.0, sr=48000)
    return ["--out", str(tmp_path / out), "--ref", str(ref), "--textbank", str(_textbank(tmp_path)), "--throughput-log",
            "", "--laughter-usd", "0", "--laugh-model", "none", "--no-emotion-refs", "--mix", "neutral=0.5,cwords=0.5",
            "--hours", "1", "--domain-lines", "10", "--cwords", str(_cwords(tmp_path)), *extra]


def test_run_gates_cwords_and_emotions_on_the_c_and_never_plain_rows(tmp_path, audio_libs, monkeypatch):
    monkeypatch.setattr(FakeScoring, "WRONG_C", {"e_surprised_c001"})  # its first try says "Vay kanına"
    rows = DOMAIN + DOMAIN_C
    dom = (lambda: (rows, {"repo": rc.DOMAIN_REPO, "rows_read": len(rows)}))
    eng = FakeEngine()
    assert rc.run(rc.parse_args(_cw_argv(tmp_path, "c1")), render=eng, scoring=FakeScoring, domain=dom) == 0
    out = tmp_path / "c1"
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    kept = state["kept"]
    assert {"e_surprised_c001", "e_cheerful_c001", "e_angry_c001", "d31"} <= set(kept)
    pool = _rows(out / "texts.jsonl")
    assert not {"e_probe_c001", "e_probe_c002", "e_held_c001", "e_calm_x001"} & {r["id"] for r in pool}
    # the c heard as /k/ at its first try: failed on the c-check (Whisper heard it right), kept at the second
    s = kept["e_surprised_c001"]
    assert s["try"] == 1 and s["c_expected"] == 1 and s["c_got"] == 1 and "dʒ" in s["ipa"]
    assert s["train_text"] == "(Astonished, with rising intonation.)Vay canına, bu kadar hızlı mı bitirdin?"
    att = [x for x in _rows(out / "attempts.jsonl") if x["id"] == "e_surprised_c001"]
    assert att[0]["fails"] == ["c_dj"] and att[0]["c_got"] == 0 and " k " in f" {att[0]['ipa']} "
    assert att[0]["char_errors"] == 0  # Whisper's CER did not see it
    # only instructed rows with a c went to the IPA model; the plain row with c's never did
    phoned = {rid for call in FakeScoring.CALLS if call[0] == "phon" for rid in call[1]}
    assert phoned == {"e_surprised_c001", "e_cheerful_c001", "e_angry_c001"}
    assert "c_expected" not in kept["d31"] and "ipa" not in kept["d31"]
    assert state["rows"]["e_surprised_c001"]["fails"] == [["c_dj"]]
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    assert rep["by_category"]["cwords"]["fails"] == {"c_dj": 1} and rep["by_category"]["cwords"]["clips"] == 3
    cd = rep["distributions"]["c_check"]
    assert cd["on"] and cd["failed"] == 1 and cd["tries_checked"] == 4 and cd["by_category"]["cwords"] == {
        "checked": 4, "failed": 1} and cd["kept_checked"] == 3
    assert "c-check (instructed rows with a c heard as /dʒ/" in (out / "PREP_REPORT.md").read_text(encoding="utf-8")
    assert "word-initial c" in (out / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    man = {r["text"] for r in _rows(out / "voxcpm_train.jsonl")}
    assert s["train_text"] in man and rep["manifest"]["with_instruction"] == 3
    # --no-c-check: no IPA model, the first try kept
    FakeScoring.CALLS.clear()
    assert rc.run(rc.parse_args(_cw_argv(tmp_path, "c2", "--no-c-check")), render=FakeEngine(), scoring=FakeScoring,
                  domain=dom) == 0
    assert json.loads((tmp_path / "c2" / "state.json").read_text(encoding="utf-8"))["kept"]["e_surprised_c001"][
        "try"] == 0
    assert not any(call[0] == "phon" for call in FakeScoring.CALLS)
    rep2 = json.loads((tmp_path / "c2" / "prep_report.json").read_text(encoding="utf-8"))
    assert rep2["distributions"]["c_check"]["on"] is False and "off (--no-c-check)" in (
        tmp_path / "c2" / "PREP_REPORT.md").read_text(encoding="utf-8")


def test_estimate_counts_the_c_check_and_the_cwords_guess(tmp_path, capsys):
    cw = rc.load_cwords(REPO / rc.CWORDS)[0]
    args = rc.parse_args(["--out", "x", "--hours", "1", "--mix", "long=0.47,neutral=0.47,cwords=0.06", "--min-sim",
                          "0.6"])
    planned = rc.plan_pool([], [], args, cw)
    assert sum(r["category"] == "cwords" for r in planned) == 90
    cost = rc.Cost(render_per_audio=1 / 37.5)
    est = rc.estimate(planned, rc.new_state({}), args, cost)
    b = est["by"]["cwords"]
    d = sum(rc.est_dur(r) for r in cw) / len(cw)
    keep = 1 - (1 - 0.35) ** 8
    assert b["p_try"] == 0.35 and b["p_keep"] == round(keep, 3) and b["probe"] == rc.PROBE_TEXTS
    assert b["texts"] == math.ceil(0.06 * 3600 / (d * keep)) and b["c_checked"] == b["tries"]
    assert "c_checked" not in est["by"]["neutral"] and "c_checked" not in est["by"]["long"]
    tries = 0.06 * 3600 / (d * keep) * keep / 0.35
    assert est["c_s"] == round(tries * cost.c_s(d) + rc.PHON_LOAD_S) and est["c_checked"] == round(tries)
    assert est["work_s"] == pytest.approx(est["render_s"] + est["score_s"] + est["c_s"] + est["overhead_s"], abs=2)
    lines = rc.estimate_lines(est)
    assert "c-check" in lines[-1] and "(c-check on" in " ".join(lines)
    # a share bigger than the pool: all of its texts, the rest goes to --fill
    big = rc.parse_args(["--out", "x", "--hours", "1", "--mix", "neutral=0.5,cwords=0.5", "--min-sim", "0.6"])
    est2 = rc.estimate(rc.plan_pool([], [], big, cw), rc.new_state({}), big, cost)
    assert est2["by"]["cwords"]["short_of_pool"] and est2["by"]["cwords"]["texts"] == 90
    assert est2["targets"]["neutral"] > 1800 and "(all 90 of its texts: the rest goes to --fill)" in " ".join(
        rc.estimate_lines(est2))
    off = rc.parse_args(["--out", "x", "--hours", "1", "--mix", "neutral=0.5,cwords=0.5", "--no-c-check"])
    assert rc.estimate(rc.plan_pool([], [], off, cw), rc.new_state({}), off, cost)["c_checked"] == 0
    # the dry run: the cwords rows, the c-check's time and the guessed pass rate
    assert rc.main(["--out", str(tmp_path / "d"), "--textbank", str(_textbank(tmp_path)), "--throughput-log", "",
                    "--dry-run", "--mix", "long=0.47,neutral=0.47,cwords=0.06", "--min-sim", "0.6",
                    "--laughter-usd", "0"]) == 0
    text = capsys.readouterr().out
    assert "cwords: tts" in text and "90 rows" in text and "c-check on" in text and "(p(try) 0.35)" in text
    assert "cwords/e_" in text and not (tmp_path / "d").exists()


def test_dry_run_counts_no_prelude_for_the_emotion_references_refs_from_lends(tmp_path, capsys):
    src = tmp_path / "first"
    refs = {e: {"wav": f"refs/{e}.wav", "seed": 1} for e in rc.EMO_REF_INSTRUCTIONS}
    vc.write_json(src / "prep_report.json", {"emotion_refs": refs})  # corpus-fetch's copy: the report, no state.json
    argv = ["--out", str(tmp_path / "d"), "--textbank", str(_textbank(tmp_path)), "--throughput-log", "", "--dry-run",
            "--mix", "long=0.47,neutral=0.47,cwords=0.06", "--min-sim", "0.6", "--laughter-usd", "0"]
    assert rc.main([*argv, "--refs-from", str(src)]) == 0
    text = capsys.readouterr().out
    assert "emotion references angry, calm, cheerful, empathetic, sad, surprised lent" in text
    assert "prelude: nothing" in text
    vc.write_json(src / "state.json", {"emotion_refs": {"sad": refs["sad"]}})  # the state wins over the report
    assert rc.main([*argv, "--refs-from", str(src)]) == 0
    text = capsys.readouterr().out
    assert "emotion references sad lent" in text and "prelude: emotion references for angry, calm, cheerful" in text
    assert rc.main([*argv, "--refs-from", str(tmp_path / "not-here")]) == 0
    text = capsys.readouterr().out
    assert "the prelude below counts them all" in text and "prelude: emotion references for angry" in text
    assert rc.main(argv) == 0 and "prelude: emotion references for angry" in capsys.readouterr().out
    assert not (tmp_path / "d").exists() and sorted(p.name for p in src.iterdir()) == ["prep_report.json",
                                                                                      "state.json"]


def test_cwords_render_four_tries_a_wave_so_their_eight_add_no_wave():
    row = {"id": "e_x", "category": "cwords", "spoken": "Cam kırık.", "instruction": "Surprised."}
    assert rc.prerender_of(row, 2) == rc.CWORDS_PRERENDER == 4 and rc.prerender_of(row, 6) == 6
    assert rc.prerender_of({"category": "long"}, 2) == 2 and rc.prerender_of({"category": "emotions"}, 2) == 2
    assert rc.wave_tries(row, None, 4, 2, [], 6, 8) == ([0, 1, 2, 3], 4)
    assert rc.wave_tries(row, {"tries": 4}, 4, 2, [5], 6, 8) == ([4, 6, 7], 4)
    assert rc.wave_tries({**row, "category": "long"}, None, 4, 2, [], 6, 8) == ([0, 1], 2)
    # the plan: a long item's 6 tries take 3 waves at 2 a wave, cwords' 8 take 2 at 4: + 1, + 1 for the probing
    cw = rc.load_cwords(REPO / rc.CWORDS)[0]
    cost = rc.Cost(render_per_audio=1 / 37.5)
    plans = {}
    for mix in ("long=0.47,neutral=0.47,cwords=0.06", "long=0.5,neutral=0.5"):
        args = rc.parse_args(["--out", "x", "--hours", "1", "--mix", mix, "--min-sim", "0.6"])
        plans[mix] = rc.estimate(rc.plan_pool([], [], args, cw), rc.new_state({}), args, cost)
    est = plans["long=0.47,neutral=0.47,cwords=0.06"]
    assert est["waves"] == plans["long=0.5,neutral=0.5"]["waves"] == 3 + 1 + 1
    b = est["by"]["cwords"]
    # rendered: the tries scored (kept x 1 / p) + the never scored ones of 4 a wave (kept x (4 - 1) / 2)
    assert b["rendered_s"] == pytest.approx(0.06 * 3600 * (1 / 0.35 + 3 / 2), abs=1)
    # --cwords-tries 16: 4 waves at 4, one more than a long item's 3
    args = rc.parse_args(["--out", "x", "--hours", "1", "--mix", "long=0.47,neutral=0.47,cwords=0.06", "--min-sim",
                          "0.6", "--cwords-tries", "16"])
    assert rc.estimate(rc.plan_pool([], [], args, cw), rc.new_state({}), args, cost)["waves"] == 4 + 1 + 1


def test_fit_wave_reserves_the_ipa_models_load_until_the_process_has_it():
    rows = [{"id": "e_x", "category": "cwords", "spoken": "Cam kırık.", "instruction": "Surprised.",
             "emotion": "surprised"}, {"id": "d1", "category": "neutral", "spoken": "Cuma günü gelin."}]
    args = rc.parse_args(["--out", "x", "--mix", "neutral=0.5,cwords=0.5"])
    st, cost = rc.new_state({}), rc.Cost(render_per_audio=1 / 37.5)
    got, first = rc.fit_wave(rows, st, args, cost, 1e9, {}, rc.CHARS_S)
    _, later = rc.fit_wave(rows, st, args, cost, 1e9, {}, rc.CHARS_S, 0.0)
    assert got == rows and first - later == pytest.approx(rc.PHON_LOAD_S)
    off = rc.parse_args(["--out", "x", "--mix", "neutral=0.5,cwords=0.5", "--no-c-check"])
    assert rc.fit_wave(rows, st, off, cost, 1e9, {}, rc.CHARS_S)[1] < later  # no c-check: neither load nor checks
    # a budget the load does not fit in any more but the rows do once the model is there
    assert rc.fit_wave(rows, st, args, cost, later + 1, {}, rc.CHARS_S)[0] == []
    assert rc.fit_wave(rows, st, args, cost, later + 1, {}, rc.CHARS_S, 0.0)[0] == rows


def test_the_scoring_line_logs_the_estimate_without_the_factor(tmp_path, audio_libs, monkeypatch):
    monkeypatch.setattr(rc, "SCORE_LOG_MIN", 1)
    rows = DOMAIN + DOMAIN_C
    dom = (lambda: (rows, {"repo": rc.DOMAIN_REPO, "rows_read": len(rows)}))
    old = [{"mode": "corpus-score", "clips": 100, "wall_s": 300.0, "model_s": 100.0, "raw": True}]  # factor 3
    lines = {}
    for name, before in (("f1", []), ("f3", old)):
        log = tmp_path / f"{name}.jsonl"
        vc.write_rows(log, before)
        argv = _cw_argv(tmp_path, name)
        argv[argv.index("--throughput-log") + 1] = str(log)
        assert rc.run(rc.parse_args(argv), render=FakeEngine(), scoring=FakeScoring, domain=dom) == 0
        lines[name] = [x for x in _rows(log) if x.get("mode") == "corpus-score"][len(before):]
    assert lines["f1"] and all(x["raw"] and "phon_load_s" not in x for x in lines["f1"])
    assert [x["model_s"] for x in lines["f3"]] == [x["model_s"] for x in lines["f1"]]  # the factor is not in it
    # make_cost: the raw lines when there are any (older ones had their call's factor in model_s)
    mixed = [{"mode": "corpus-score", "clips": 100, "wall_s": 30.0, "model_s": 60.0}, *old]
    assert rc.make_cost("nanovllm", 10, 2, mixed).score_factor == 3.0
    assert rc.make_cost("nanovllm", 10, 2, mixed[:1]).score_factor == 0.5


def test_scoring_phon_measures_the_ipa_models_first_load_only(monkeypatch):
    from types import SimpleNamespace

    class Pm:
        def ipa_file(self, path):
            return ["dʒ", "a", "m"]

    def get_model(dev):
        if not rc.pc._MODELS:
            clock[0] += 7.0  # the load
            rc.pc._MODELS["m"] = Pm()
        return rc.pc._MODELS["m"]

    clock = [100.0]
    monkeypatch.setattr(rc.pc, "_MODELS", {})
    monkeypatch.setattr(rc.pc, "get_model", get_model)
    monkeypatch.setattr(rc, "time", SimpleNamespace(time=lambda: clock[0]))
    for want in (7.0, 0.0):  # a wave, then the next one's Scoring: the process has the model
        sc = rc.Scoring.__new__(rc.Scoring)
        sc.sc, sc.pm, sc.phon_load_s = SimpleNamespace(dev=lambda: "cpu"), None, 0.0
        a = {"path": "x.wav", "model_text": "Cam."}
        sc.phon([a], 1e12)
        assert sc.phon_load_s == want and (a["c_expected"], a["c_got"]) == (1, 1)


class FakePhon:
    """An IPA model by WAV stem: WRONG ones say /k/ for every c, UNREADABLE ones raise; the rest /dʒ/."""

    WRONG, UNREADABLE = {"e1"}, {"e4"}

    def __init__(self, texts: dict[str, str]):
        self.texts, self.calls = texts, []

    def ipa_file(self, path):
        stem = Path(path).stem
        self.calls.append(stem)
        if stem in self.UNREADABLE:
            raise RuntimeError("no frames")
        _, spoken = rc.pc.spoken_part(self.texts[stem])
        return [("k" if stem in self.WRONG else "dʒ") if ch == "c" else ch for ch in rc.pc.tr_lower(spoken)
                if ch.isalpha()]


AUDIT_ROWS = [("e1", "(Astonished.)Vay canına!", 2.0, "emotions"),
              ("e2", "(Astonished.)Cam kırılmış, cadde de kapanmış!", 2.5, "emotions"),
              ("e3", "(Calm.)Yavaşça nefes al.", 2.0, "emotions"),  # no c: not audited
              ("n1", "Cuma günü cadde kapalı.", 2.0, "neutral"),  # no instruction: not audited
              ("l1", "(Laughing heartily while speaking.)Hahaha, canım!", 1.5, "laughter"),
              ("e4", "(Surprised.)Cidden mi?", 1.0, "emotions")]


def test_c_audit_writes_its_file_and_touches_nothing_else(tmp_path, capsys, monkeypatch):
    d = _tiny_corpus(tmp_path / "f022corpus", AUDIT_ROWS)
    before = _snapshot(d)
    assert rc.main(["--out", str(d), "--audit-c", "--dry-run"]) == 0
    assert "4 manifest clips with an instruction and a c" in capsys.readouterr().out and _snapshot(d) == before
    fake = FakePhon({cid: text for cid, text, _, _ in AUDIT_ROWS})
    assert rc.audit_c(d, model=lambda: fake) == 0
    after = _snapshot(d)
    assert set(after) - set(before) == {rc.C_AUDIT} and {k: v for k, v in after.items() if k != rc.C_AUDIT} == before
    assert sorted(fake.calls) == ["e1", "e2", "e4", "l1"]
    audit = {x["id"]: x for x in _rows(d / rc.C_AUDIT)}
    assert {k: x["ok"] for k, x in audit.items()} == {"e1": False, "e2": True, "l1": True, "e4": False}
    assert (audit["e1"]["expected"], audit["e1"]["got"], audit["e2"]["expected"], audit["e2"]["got"]) == (1, 0, 2, 2)
    assert audit["e1"]["wav"] == str(d / "wav" / "e1.wav") and audit["e1"]["category"] == "emotions"
    assert audit["e1"]["ipa"].startswith("v a y k a n") and audit["e4"]["got"] is None
    assert "no frames" in audit["e4"]["error"]
    assert audit["l1"]["category"] == "laughter" and audit["l1"]["text"] == AUDIT_ROWS[4][1]
    log = capsys.readouterr().out
    assert "2 of 4 fail the c-check" in log and "e1" in log and "corpus-merge leaves them out" in log
    assert "  e1: 'Vay canına!' -> v a y k a n" in log and "  e4: 'Cidden mi?' -> RuntimeError: no frames" in log
    # the model reads (almost) none: it, not the clips; nothing written, the audit before stays
    monkeypatch.setattr(rc, "SYSTEMIC_N", 3)
    broken = FakePhon({cid: text for cid, text, _, _ in AUDIT_ROWS})
    broken.UNREADABLE = {"e1", "e2", "l1"}
    audit_before = (d / rc.C_AUDIT).read_bytes()
    assert rc.audit_c(d, model=lambda: broken) == 1 and (d / rc.C_AUDIT).read_bytes() == audit_before
    assert "could not read 3 of 4 clips" in capsys.readouterr().out
    broken.UNREADABLE = {"e1", "e2"}  # under SYSTEMIC_N: two unreadable clips, failed and written
    assert rc.audit_c(d, model=lambda: broken) == 0
    assert [x["id"] for x in _rows(d / rc.C_AUDIT) if "error" in x] == ["e1", "e2"]
    # refused: a render going, a dir that is no render corpus
    vc.write_json(d / "status.json", {"phase": "waves", "updated": "now"})
    assert rc.audit_c(d, model=lambda: pytest.fail("no model")) == 2
    assert rc.audit_c(tmp_path / "nothing", model=lambda: pytest.fail("no model")) == 2


def test_merge_leaves_out_failed_audit_clips_and_the_preflight_sees_a_changed_audit(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    a = _tiny_corpus(tmp_path / "srcA", AUDIT_ROWS)
    b = _tiny_corpus(tmp_path / "srcB", [("b1", "(Astonished.)Vay canına!", 2.2, "cwords"),
                                         ("b2", "Beş.", 2.0, "neutral")])
    vc.write_rows(b / "clips.jsonl", [{**c, **({"c_expected": 1, "c_got": 1} if c["id"] == "b1" else {})}
                                      for c in _rows(b / "clips.jsonl")])  # b1 rendered under the c gate
    mix = tmp_path / "mix"
    # the first corpus before its audit: refused (its emotion clips were never c-checked), unless --allow-unaudited
    capsys.readouterr()
    assert rc.main(["--out", str(mix), "--merge", str(a), str(b)]) == 2 and not mix.exists()
    assert "4 instructed clips with a c were never c-checked" in capsys.readouterr().out
    assert rc.main(["--out", str(tmp_path / "m0"), "--merge", str(a), str(b), "--allow-unaudited"]) == 0
    m0 = json.loads((tmp_path / "m0" / "prep_report.json").read_text(encoding="utf-8"))
    assert m0["allow_unaudited"] and [p["c_unchecked"] for p in m0["sources"]] == [4, 0]
    assert "4 instructed clips with a c never c-checked" in (tmp_path / "m0" / "PREP_REPORT.md").read_text(
        encoding="utf-8")
    assert rc.audit_c(a, model=lambda: FakePhon({cid: t for cid, t, _, _ in AUDIT_ROWS})) == 0
    capsys.readouterr()
    assert rc.main(["--out", str(mix), "--merge", str(a), str(b)]) == 0
    log = capsys.readouterr().out
    assert f"{rc.C_AUDIT}: 2 of its 4 audited clips fail the c-check, 2 left out: e1, e4" in log
    rows = _rows(mix / "voxcpm_train.jsonl")
    assert {Path(r["audio"]).stem for r in rows} == {"e2", "e3", "n1", "l1", "b1", "b2"}  # e1 out: b1's text is free
    rep = json.loads((mix / "prep_report.json").read_text(encoding="utf-8"))
    sa, sb = rep["sources"]
    assert sa["dropped"] == {f"c-check failed ({rc.C_AUDIT})": 2} and sa["c_audit"]["dropped_ids"] == ["e1", "e4"]
    assert sa["c_audit_bytes"] == (a / rc.C_AUDIT).stat().st_size and sa["c_audit"]["unaudited"] == 0
    assert sa["c_audit_sha256"] == rc.sha256_file(a / rc.C_AUDIT) and sb["c_audit"] is None
    assert sb["c_audit_bytes"] is None and rep["selected"]["clips"] == 6
    assert "fail the c-check, left out: e1, e4" in (mix / "PREP_REPORT.md").read_text(encoding="utf-8")
    assert rc.C_AUDIT in (mix / "README.md").read_text(encoding="utf-8")
    # an instructed clip with a c added after the audit and never checked: refused again
    _wav(a / "wav" / "e5.wav", 1.0)
    vc.write_rows(a / "voxcpm_train.jsonl", _rows(a / "voxcpm_train.jsonl") + [
        {"audio": str(a / "wav" / "e5.wav"), "text": "(Surprised.)Ceviz mi?", "duration": 1.0}])
    capsys.readouterr()
    assert rc.main(["--out", str(tmp_path / "m2"), "--merge", str(a), str(b)]) == 2
    assert "1 instructed clips with a c were never c-checked" in capsys.readouterr().out
    assert rc.main(["--out", str(tmp_path / "m2"), "--merge", str(a), str(b), "--allow-unaudited"]) == 0
    assert "1 instructed clips with a c never c-checked, taken as they are" in capsys.readouterr().out
    # the train preflight: the merge as it was, then the audit changed / gone / new
    ma, _ = _modal_app()
    vol = {"f022mix": mix, "f022corpus": a, "f022corpus2": b}
    for p in rep["sources"]:
        p["dir"] = {str(a): "/data/f022corpus", str(b): "/data/f022corpus2"}[p["dir"]]
    sa["manifest_bytes"] = (a / "voxcpm_train.jsonl").stat().st_size  # the manifest as it is now
    vc.write_json(mix / "prep_report.json", rep)

    def files(rel):
        d = vol.get(rel)
        return {p.name: SimpleNamespace(path=f"{rel}/{p.name}", size=p.stat().st_size, mtime=p.stat().st_mtime,
                                        type=None) for p in d.iterdir()} if d is not None and d.is_dir() else {}

    def vjson(rel):
        d, _, name = rel.rpartition("/")
        return vc.read_json(vol[d] / name) if d in vol else None

    def vbytes(rel):
        d, _, name = rel.rpartition("/")
        return (vol[d] / name).read_bytes() if d in vol and (vol[d] / name).is_file() else None

    monkeypatch.setattr(ma, "_vol_files", files)
    monkeypatch.setattr(ma, "_vol_json", vjson)
    monkeypatch.setattr(ma, "_vol_bytes", vbytes)
    ma._preflight_train(["--data", "/data/f022mix"], "voxcpm2-f022-s3", 1500)
    # audited again since with one verdict turned: the same size, another file (P1 of the review: size alone missed it)
    was = (a / rc.C_AUDIT).read_bytes()
    turned = was.replace('"got": 2, "ok": true, "ipa": "dʒ'.encode(), '"got": 1, "ok": false, "ipa": "ɡ'.encode(), 1)
    assert turned != was and len(turned) == len(was)
    (a / rc.C_AUDIT).write_bytes(turned)
    assert [x["ok"] for x in _rows(a / rc.C_AUDIT) if x["id"] == "e2"] == [False]  # still an audit, e2 now failed
    with pytest.raises(SystemExit, match=r"c_audit.jsonl changed \(sha256 \w{12} -> \w{12}\) since the merge.*"
                                         "corpus-merge"):
        ma._preflight_train(["--data", "/data/f022mix"], "voxcpm2-f022-s3", 1500)
    (a / rc.C_AUDIT).write_bytes(was)
    ma._preflight_train(["--data", "/data/f022mix"], "voxcpm2-f022-s3", 1500)  # as it was: taken again
    with (a / rc.C_AUDIT).open("a", encoding="utf-8") as f:  # audited again since: another file
        f.write(json.dumps({"id": "e5", "wav": str(a / "wav" / "e5.wav"), "ok": False}) + "\n")
    with pytest.raises(SystemExit, match=r"c_audit.jsonl changed .* since the merge.*corpus-merge"):
        ma._preflight_train(["--data", "/data/f022mix"], "voxcpm2-f022-s3", 1500)
    (a / rc.C_AUDIT).unlink()
    with pytest.raises(SystemExit, match="c_audit.jsonl is gone since the merge"):
        ma._preflight_train(["--data", "/data/f022mix"], "voxcpm2-f022-s3", 1500)
    sizes = {"/data/f022corpus": sa["manifest_bytes"], "/data/f022corpus2": sb["manifest_bytes"]}
    assert ma.merged_stale(rep, sizes, {"/data/f022corpus": sa["c_audit_sha256"], "/data/f022corpus2": None}) == []
    assert ma.merged_stale(rep, sizes, {"/data/f022corpus": sa["c_audit_sha256"], "/data/f022corpus2": "ab" * 32}) == [
        "/data/f022corpus2/c_audit.jsonl appeared since the merge"]
    assert ma.merged_stale(rep, sizes) == []  # without audits: the manifests only (as before)


def test_modal_app_corpus_audit_c(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    ma, mm = _modal_app()
    assert ma.audit_cmd() == ["python", "tts/voxcpm/render_corpus.py", "--out", ma.F022, "--audit-c"]
    assert ma.audit_cmd("/data/f022corpus2", True)[-1] == "--dry-run"
    with pytest.raises(ValueError):
        ma.audit_cmd("/data/models")
    fn = [c for c in mm.App.return_value.function.call_args_list if c.kwargs.get("timeout") == ma.AUDIT_TIMEOUT_S]
    assert len(fn) == 1 and fn[0].kwargs["gpu"] == "L4" and fn[0].kwargs["image"] is ma.contest_image
    assert ma.AUDIT_PRICE == pytest.approx(0.80 + 4 * 0.047 + 16 * 0.008)
    src = (VOXCPM / "modal_app.py").read_text(encoding="utf-8")
    assert 'action == "corpus-audit-c"' in src and "_corpus_audit_c(extra, out or F022)" in src
    rows = [{"audio": f"/data/f022corpus/wav/{cid}.wav", "text": t, "duration": d} for cid, t, d, _ in AUDIT_ROWS]
    monkeypatch.setattr(ma, "_rc", lambda: rc)
    plan = ma.audit_plan(rows)
    assert plan["clips"] == 4 and plan["audio_s"] == 7.0
    assert plan["est_s"] == round(ma.AUDIT_START_S + rc.PHON_LOAD_S + 4 * rc.PHON_CLIP_S + 7.0 * rc.PHON_AUDIO_S
                                  + ma.IDLE_S)
    assert plan["max_usd"] < 0.3 and plan["est_usd"] < 0.06
    entry = SimpleNamespace(size=10, mtime=1.0, type=None, path="f022corpus/c_audit.jsonl")
    vol_files = {"f022corpus": {rc.MARKER: entry, "voxcpm_train.jsonl": entry}}
    vol_json = {"f022corpus/status.json": {"phase": "waves", "updated": "now"}}
    calls, got = [], []
    monkeypatch.setattr(ma, "_vol_files", lambda rel: vol_files.get(rel, {}))
    monkeypatch.setattr(ma, "_vol_json", lambda rel: vol_json.get(rel))
    monkeypatch.setattr(ma, "_vol_rows", lambda rel: rows if rel.endswith("voxcpm_train.jsonl") else [])
    monkeypatch.setattr(ma, "_download", lambda path, size, out: got.append((path, out)))
    monkeypatch.setattr(ma, "corpus_audit_c", SimpleNamespace(remote=lambda out: calls.append(out) or json.dumps(
        {"rc": 0, "used_s": 200, "audit": {"rows": 4, "failed": 2, "ids": ["e1", "e4"]}})))
    with pytest.raises(SystemExit, match="phase 'waves'"):
        ma._corpus_audit_c([])
    vol_json["f022corpus/status.json"] = {"phase": "done"}
    ma._corpus_audit_c(["--dry-run"])
    text = capsys.readouterr().out
    assert "4 manifest clips with an instruction and a c, 7 s of audio" in text and "at most $0.2" in text
    assert "--dry-run: nothing started" in text and not calls
    vol_files["f022corpus"][rc.C_AUDIT] = entry
    ma._corpus_audit_c([])
    assert calls == ["/data/f022corpus"] and got == [(entry.path, ma.corpus_local(ma.F022) / rc.C_AUDIT)]
    assert '"failed": 2' in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--dry-run only"):
        ma._corpus_audit_c(["--hours", "1"])
    with pytest.raises(SystemExit, match="no corpus-render corpus"):
        ma._corpus_audit_c([], "/data/f022mix")
    with pytest.raises(SystemExit, match="corpus-audit-c"):  # never on the render call
        ma._corpus(["--audit-c"], 4.0)
    # corpus-merge: a source with instructed clips with a c never c-checked is refused before any call
    old = SimpleNamespace(size=10, mtime=1.0, type=None, path="p")
    vol_files.update({d: {"voxcpm_train.jsonl": old, "prep_report.json": entry, "state.json": old}
                      for d in ("f022corpus", "f022corpus2")})
    vol_json.update({f"{d}/prep_report.json": {"complete": True, "selected": {"hours": 1.0, "clips": 6}}
                     for d in ("f022corpus", "f022corpus2")})
    merged = []
    monkeypatch.setattr(ma, "corpus_merge", SimpleNamespace(remote=lambda s, o, allow, unaudited, ins: merged.append(
        unaudited) or json.dumps({"rc": 0, "used_s": 30})))
    with pytest.raises(SystemExit, match="4 instructed clips with a c that were never c-checked.*corpus-audit-c"):
        ma._corpus_merge([])
    ma._corpus_merge(["--allow-unaudited"])
    assert merged == [True] and "4 instructed clips with a c never c-checked" in capsys.readouterr().out
    assert ma.merge_cmd(["/data/a", "/data/b"], "/data/m", False, False, True)[-1] == "--allow-unaudited"


# ---- Turkish instructions (--merge --instructions, instructions_tr.json, probe_tr_instr.jsonl) --------------------


def _wordings() -> dict[str, set[str]]:
    """Every instruction wording a training text can carry, and the contest set's (listening rounds, probes)."""
    def of(p: Path) -> set[str]:
        return {r["instruction"] for r in vc.read_rows(p) if r.get("instruction")}

    tb = REPO / "tts" / "voxcpm" / "textbank"
    return {"emotions": of(tb / "emotions.jsonl"), "textbank": of(tb / "textbank.jsonl"),
            "cwords": of(tb / "cwords.jsonl"), "emo_refs": set(rc.EMO_REF_INSTRUCTIONS.values()),
            "laughter": set(rc.LAUGH_INSTRUCTIONS),
            "contest": of(REPO / "benchmarks" / "tts" / "sets" / "contest.jsonl")}


def test_instructions_tr_table_covers_every_wording_once(tmp_path):
    table = rc.load_instructions_tr(REPO / rc.INSTRUCTIONS_TR)
    got = _wordings()
    heldout = {r["instruction"] for r in vc.read_rows(REPO / "tts/voxcpm/textbank/emotions.jsonl")
               if r.get("instruction_split") == "heldout"}
    assert len(heldout) == 12 and heldout <= set(table)  # the evaluation wordings too
    want = set().union(*got.values())
    assert not (missing := sorted(want - set(table))), f"no Turkish wording for {missing}"
    assert not (extra := sorted(set(table) - want)), f"stale wordings {extra}"
    assert got["emo_refs"] | got["cwords"] <= got["emotions"] and len(table) == len(want) == 77
    english = {"and", "with", "the", "voice", "speaking", "while", "very", "slightly", "as", "if"}
    for en, tr in table.items():
        assert tr.endswith(".") and len(tr) <= 90 and tr != en, (en, tr)
        assert not english & set(re.findall(r"[a-z]+", tr.lower())), (en, tr)  # Turkish, not a copy
    assert len(set(table.values())) == len(table)
    # a broken table refuses: parentheses, a Turkish wording twice or equal to an English key, not an object, a newline
    for bad in ({"A.": "(B.)"}, {"A.": "B.", "C.": "B."}, {"A.": "C.", "C.": "D."}, ["A."], {}, {"A.": "B.\nC."},
                {"A.": " B."}):
        p = tmp_path / "t.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            rc.load_instructions_tr(p)
    with pytest.raises(ValueError):
        rc.load_instructions_tr(tmp_path / "none.json")


def test_instruction_copies_both_tr_en():
    table = {"Angry.": "Öfkeli.", "Sad.": "Üzgün."}
    rows = [{"audio": "/a.wav", "text": "Düz.", "duration": 1.0},
            {"audio": "/b.wav", "text": "(Angry.)Yeter!", "duration": 2.0, "ref_audio": "/a.wav", "ref_duration": 1.0},
            {"audio": "/c.wav", "text": "( Sad. )Of.", "duration": 3.0},
            {"audio": "/d.wav", "text": "(Öfkeli.)Yine mi?", "duration": 1.5}]  # already Turkish: once
    both, unknown = rc.instruction_copies(rows, "both", table)
    assert not unknown and [(r["audio"], r["_lang"], r["text"]) for r in both] == [
        ("/a.wav", "plain", "Düz."), ("/b.wav", "en", "(Angry.)Yeter!"), ("/b.wav", "tr", "(Öfkeli.)Yeter!"),
        ("/c.wav", "en", "( Sad. )Of."), ("/c.wav", "tr", "(Üzgün.)Of."), ("/d.wav", "tr", "(Öfkeli.)Yine mi?")]
    assert both[2]["ref_audio"] == "/a.wav" and both[2]["duration"] == 2.0  # all else the same
    tr, _ = rc.instruction_copies(rows, "tr", table)
    assert [(r["audio"], r["_lang"]) for r in tr] == [("/a.wav", "plain"), ("/b.wav", "tr"), ("/c.wav", "tr"),
                                                     ("/d.wav", "tr")]
    en, _ = rc.instruction_copies(rows, "en", {})  # the table not read: rows as they are
    assert [r["text"] for r in en] == [r["text"] for r in rows] and [r["_lang"] for r in en][:2] == ["plain", "en"]
    _, unknown = rc.instruction_copies(rows + [{"audio": "/e.wav", "text": "(Giggling.)Hi.", "duration": 1.0}],
                                       "both", table)
    assert unknown == {"Giggling.": 1}
    assert rc.instruction_copies(rows, "en", table)[1] == {}  # en never refuses


def test_merge_with_turkish_instructions_and_the_train_preflight(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    table = rc.load_instructions_tr(REPO / rc.INSTRUCTIONS_TR)
    calm_tr = table["Calm, slow and soothing."]
    a = _tiny_corpus(tmp_path / "srcA", [("a1", "Bir iki üç.", 2.0, "neutral"),
                                         ("e1", "(Angry tone, voice getting louder.)Yeter artık!", 1.5, "emotions"),
                                         ("e2", "(Sad and quiet, with a heavy heart.)Hiç beklemiyordum.", 2.0,
                                          "emotions"),
                                         ("l1", "(Laughing heartily while speaking.)Hahaha, çok komiksin!", 1.5,
                                          "laughter")])
    b = _tiny_corpus(tmp_path / "srcB", [("b1", "Dört beş altı.", 2.0, "neutral"),
                                         ("b3", f"({calm_tr})Yavaş ol.", 1.0, "emotions"),
                                         ("b2", "(Astonished, with rising intonation.)Kar mı yağmış?", 1.5, "cwords")])
    en_rows = _rows(a / "voxcpm_train.jsonl") + _rows(b / "voxcpm_train.jsonl")
    # en (default): today's merge, the report counts the rows (instructed ones as English: the table is not read)
    mix = tmp_path / "mix"
    assert rc.main(["--out", str(mix), "--merge", str(a), str(b)]) == 0
    assert sorted(map(json.dumps, _rows(mix / "voxcpm_train.jsonl"))) == sorted(map(json.dumps, en_rows))
    rep = json.loads((mix / "prep_report.json").read_text(encoding="utf-8"))
    assert rep["instructions"] == {"mode": "en", "rows": {"plain": 2, "en": 5, "tr": 0}}
    assert rep["audio"] == {"clips": 7, "hours": round(11.5 / 3600, 3)} and rep["selected"]["clips"] == 7
    # both: every instructed row twice (English, then the table's Turkish), the Turkish one once, plain rows once
    mtr = tmp_path / "mix_both"
    capsys.readouterr()
    assert rc.main(["--out", str(mtr), "--merge", str(a), str(b), "--instructions", "both", "--dry-run"]) == 0
    assert "instructions both: 2 rows without an instruction, 4 English, 5 Turkish" in capsys.readouterr().out
    assert not mtr.exists()
    assert rc.main(["--out", str(mtr), "--merge", str(a), str(b), "--instructions", "both"]) == 0
    rows = _rows(mtr / "voxcpm_train.jsonl")
    by_audio: dict[str, list[dict]] = {}
    for r in rows:
        by_audio.setdefault(Path(r["audio"]).stem, []).append(r)
    assert len(rows) == 11 and {k: len(v) for k, v in by_audio.items()} == {
        "a1": 1, "b1": 1, "b3": 1, "e1": 2, "e2": 2, "l1": 2, "b2": 2}
    angry = "Angry tone, voice getting louder."
    assert sorted(r["text"] for r in by_audio["e1"]) == [f"({angry})Yeter artık!", f"({table[angry]})Yeter artık!"]
    assert f"({table['Laughing heartily while speaking.']})Hahaha, çok komiksin!" in {r["text"] for r in by_audio["l1"]}
    assert all(set(r) <= set(rc.MERGED_KEYS) for r in rows)
    b2 = sorted(by_audio["b2"], key=lambda r: r["text"])
    assert b2[0]["ref_audio"] == b2[1]["ref_audio"] == str(b / "wav" / "b1.wav")  # same audio, duration, ref_audio
    assert b2[0]["duration"] == b2[1]["duration"] == 1.5
    assert by_audio["b3"][0]["text"] == f"({calm_tr})Yavaş ol."
    for r in rows:  # every Turkish text is an instruction of the table, as VoxCPM reads it
        ins, spoken = rc.pc.spoken_part(r["text"])
        assert ins is None or ins in table or ins in table.values()
    rep = json.loads((mtr / "prep_report.json").read_text(encoding="utf-8"))
    ins = rep["instructions"]
    assert ins["mode"] == "both" and ins["rows"] == {"plain": 2, "en": 4, "tr": 5} and ins["wordings"] == 77
    assert ins["table"] == rc.INSTRUCTIONS_TR.as_posix()
    assert ins["table_sha256"] == rc.sha256_file(REPO / rc.INSTRUCTIONS_TR)
    assert rep["selected"]["clips"] == rep["manifest"]["rows"] == 11 and rep["audio"]["clips"] == 7
    assert rep["by_category"]["emotions"]["clips"] == 5 and rep["by_category"]["laughter"]["clips"] == 2
    assert [p["kept_rows"] for p in rep["sources"]] == [4, 3]  # per source: its clips, once
    note, md = (mtr / "README.md").read_text(encoding="utf-8"), (mtr / "PREP_REPORT.md").read_text(encoding="utf-8")
    assert "`--instructions both`" in note and "2 without an instruction, 4 English, 5 Turkish" in note
    assert "audio: 7 clips" in md and "| emotions | 5 |" in md
    assert rc.main(["--out", str(mtr), "--merge", str(a), str(b), "--instructions", "both"]) == 0  # again: replaced
    # tr: the Turkish copy only
    mt2 = tmp_path / "mix_tr"
    assert rc.main(["--out", str(mt2), "--merge", str(a), str(b), "--instructions", "tr"]) == 0
    rows2 = _rows(mt2 / "voxcpm_train.jsonl")
    assert len(rows2) == 7 and len({r["audio"] for r in rows2}) == 7
    assert sum(r["text"].startswith("(Angry") for r in rows2) == 0
    assert json.loads((mt2 / "prep_report.json").read_text(encoding="utf-8"))["instructions"]["rows"] == {
        "plain": 2, "en": 0, "tr": 5}
    # a merge dir keeps its mode: another --instructions goes elsewhere (an old merge without the record is en), the
    # refusal names a dir for it: out without its "_<mode>" + "_<new mode>"
    capsys.readouterr()
    for out, mode, other in ((mtr, "tr", mt2), (mtr, "en", tmp_path / "mix_en"), (mix, "both", mtr),
                             (mt2, "both", mtr)):
        assert rc.main(["--out", str(out), "--merge", str(a), str(b), *(["--instructions", mode] if mode != "en"
                                                                          else [])]) == 2
        assert f"goes to a dir of its own (e.g. {other})" in capsys.readouterr().out
    cases = (("/data/f022mix", "en", "both"), ("/data/f022mix_both", "both", "tr"), ("/data/f022mix_tr", "both", "en"),
             ("/d/x_tr", "en", "tr"))
    assert [rc.merge_out_for(*c) for c in cases] == ["/data/f022mix_both", "/data/f022mix_tr", "/data/f022mix_tr_en",
                                                     "/d/x_tr_tr"]
    old = json.loads((mix / "prep_report.json").read_text(encoding="utf-8"))
    del old["instructions"]
    vc.write_json(mix / "prep_report.json", old)
    assert rc.main(["--out", str(mix), "--merge", str(a), str(b), "--instructions", "tr"]) == 2
    assert rc.main(["--out", str(mix), "--merge", str(a), str(b)]) == 0
    # an instructed row whose wording the table lacks refuses both / tr (dry run too), never en
    c = _tiny_corpus(tmp_path / "srcC", [("c1", "(Laughing.)Haha!", 1.0, "laughter"), ("c2", "Yedi.", 1.0, "neutral")])
    capsys.readouterr()
    for mode in ("both", "tr"):
        for dry in ([], ["--dry-run"]):
            assert rc.main(["--out", str(tmp_path / "mx"), "--merge", str(a), str(c), "--instructions", mode,
                            *dry]) == 2
            assert "lacks" in (log := capsys.readouterr().out) and "'Laughing.' x1" in log
    assert not (tmp_path / "mx").exists()
    assert rc.main(["--out", str(tmp_path / "mx"), "--merge", str(a), str(c)]) == 0
    with pytest.raises(SystemExit):
        rc.parse_args(["--out", str(tmp_path / "x"), "--instructions", "both"])  # only with --merge
    with pytest.raises(SystemExit):
        rc.parse_args(["--out", str(tmp_path / "x"), "--merge", str(a), str(b), "--instructions", "de"])
    monkeypatch.setattr(rc, "INSTRUCTIONS_TR", tmp_path / "missing.json")
    assert rc.main(["--out", str(tmp_path / "my"), "--merge", str(a), str(b), "--instructions", "both"]) == 2
    monkeypatch.undo()
    # modal_app's corpus summary (corpus-merge's result, status) counts a merge's rows and its audio apart
    ma, _ = _modal_app()
    summ = ma.corpus_summary(mtr)
    assert (summ["rows"], summ["audio"], summ["instructions"]) == (11, {"clips": 7, "hours": round(11.5 / 3600, 3)},
                                                                   "both") and "clips" not in summ
    assert ma.corpus_summary(tmp_path / "srcA")["clips"] == 4  # a corpus-render dir as before
    # modal_app's train preflight takes /data/f022mix_both like any merge, and names its instructions
    rep = json.loads((mtr / "prep_report.json").read_text(encoding="utf-8"))
    for p in rep["sources"]:
        p["dir"] = {str(a): "/data/f022corpus", str(b): "/data/f022corpus2"}[p["dir"]]
    vc.write_json(mtr / "prep_report.json", rep)
    vol = {"f022mix_both": mtr, "f022corpus": a, "f022corpus2": b}

    def files(rel):
        d = vol.get(rel)
        return {p.name: SimpleNamespace(path=f"{rel}/{p.name}", size=p.stat().st_size, mtime=p.stat().st_mtime,
                                        type=None) for p in d.iterdir()} if d is not None and d.is_dir() else {}

    def vjson(rel):
        d, _, name = rel.rpartition("/")
        return vc.read_json(vol[d] / name) if d in vol else None

    monkeypatch.setattr(ma, "_vol_files", files)
    monkeypatch.setattr(ma, "_vol_json", vjson)
    capsys.readouterr()
    ns = ma._preflight_train(["--data", "/data/f022mix_both", "--lr", "5e-5"], "voxcpm2-f022-s3tr", 500)
    assert ma._vol_path(ns.data) == "/data/f022mix_both" and ns.lr == 5e-5
    assert (f"{rep['selected']['hours']} h, 11 rows (audio: 7 clips, {round(11.5 / 3600, 3)} h), instructions both: "
            "2 plain, 4 en, 5 tr rows") in capsys.readouterr().out
    # the table changed since the merge (or is not here): its Turkish wordings are not the probe's, merge again
    edited = tmp_path / "instructions_tr.json"
    vc.write_json(edited, {**table, "Angry tone, voice getting louder.": "Kızgın, sesi giderek yükselerek."})
    monkeypatch.setattr(rc, "INSTRUCTIONS_TR", edited)
    with pytest.raises(SystemExit, match=r"merged with tts/voxcpm/instructions_tr.json sha256 \w{12}, the local one is "
                                         r"sha256 \w{12}.*corpus-merge --out /data/f022mix_both again first \(--args "
                                         r"\"--instructions both\"\) \(the same --out takes it\), or git pull"):
        ma._preflight_train(["--data", "/data/f022mix_both"], "voxcpm2-f022-s3tr", 500)
    monkeypatch.setattr(rc, "INSTRUCTIONS_TR", tmp_path / "none.json")
    with pytest.raises(SystemExit, match="cannot be read"):
        ma._preflight_train(["--data", "/data/f022mix_both"], "voxcpm2-f022-s3tr", 500)
    monkeypatch.setattr(rc, "INSTRUCTIONS_TR", Path("tts/voxcpm/instructions_tr.json"))
    ma._preflight_train(["--data", "/data/f022mix_both"], "voxcpm2-f022-s3tr", 500)  # the same table again: taken
    with (b / "voxcpm_train.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"audio": str(b / "wav" / "b1.wav"), "text": "Sekiz.", "duration": 2.0}) + "\n")
    with pytest.raises(SystemExit, match=r"corpus-merge --out /data/f022mix_both again first \(--args \"--instructions "
                                         r"both\"\)"):
        ma._preflight_train(["--data", "/data/f022mix_both"], "voxcpm2-f022-s3tr", 500)


def test_modal_app_corpus_merge_with_instructions(monkeypatch, capsys):
    from types import SimpleNamespace

    ma, _ = _modal_app()
    table = rc.load_instructions_tr(REPO / rc.INSTRUCTIONS_TR)
    assert ma.merge_cmd(list(ma.MERGE_SOURCES), "/data/f022mix_both", instructions="both")[-2:] == [
        "--instructions", "both"]
    assert "--instructions" not in ma.merge_cmd(list(ma.MERGE_SOURCES), "/data/f022mix_both", instructions="en")
    entry = SimpleNamespace(size=10, mtime=2.0, type=None, path="p")
    old = SimpleNamespace(size=10, mtime=1.0, type=None, path="p")
    vol_files = {d: {"voxcpm_train.jsonl": old, "prep_report.json": entry, "state.json": old}
                 for d in ("f022corpus", "f022corpus2")}
    vol_json = {f"{d}/prep_report.json": {"complete": True, "selected": {"hours": 1.0, "clips": 2}}
                for d in ("f022corpus", "f022corpus2")}
    man = {"f022corpus": [{"audio": "/data/f022corpus/wav/a.wav", "text": "Düz.", "duration": 1.0},
                          {"audio": "/data/f022corpus/wav/e.wav", "text": "(Angry tone, voice getting louder.)Yeter!",
                           "duration": 1.0}],
           "f022corpus2": [{"audio": "/data/f022corpus2/wav/b.wav", "text": "(Giggling, amused.)Hihi.",
                            "duration": 1.0}]}
    monkeypatch.setattr(ma, "_rc", lambda: rc)
    monkeypatch.setattr(ma, "_vol_files", lambda rel: vol_files.get(rel, {}))
    monkeypatch.setattr(ma, "_vol_json", lambda rel: vol_json.get(rel))
    monkeypatch.setattr(ma, "_vol_rows", lambda rel: man.get(rel.split("/")[0], []) if rel.endswith(
        "voxcpm_train.jsonl") else [])
    calls = []
    monkeypatch.setattr(ma, "corpus_merge", SimpleNamespace(remote=lambda s, o, allow, unaudited, ins: calls.append(
        (o, ins)) or json.dumps({"rc": 0, "used_s": 30})))
    for bad in (["--instructions", "de"], ["--instr", "both"], ["--instructions"]):
        with pytest.raises(SystemExit, match="corpus-merge"):
            ma._corpus_merge(bad, "/data/f022mix_both")
    ma._corpus_merge(["--instructions", "both", "--dry-run"], "/data/f022mix_both")
    out = capsys.readouterr().out
    assert "--instructions both: 1 rows without an instruction, 1 English, 1 Turkish" in out and not calls
    ma._corpus_merge(["--instructions=both"], "/data/f022mix_both")
    assert calls == [("/data/f022mix_both", "both")]
    # a wording the table lacks: refused before the call
    man["f022corpus2"][0]["text"] = "(Giggling.)Hihi."
    assert "Giggling." not in table
    with pytest.raises(SystemExit, match=r"1 instructed rows whose wording .*instructions_tr.json lacks.*"
                                         r"'Giggling.' x1"):
        ma._corpus_merge(["--instructions", "tr"], "/data/f022mix_tr")
    ma._corpus_merge([], "/data/f022mix_en")  # en: as before, no table
    assert calls[-1] == ("/data/f022mix_en", "en")
    # --out holds a merge with another --instructions (or an old one, en): refused before the call
    man["f022corpus2"][0]["text"] = "(Giggling, amused.)Hihi."
    vol_files["f022mix"] = {"voxcpm_train.jsonl": entry, "prep_report.json": entry}
    vol_json["f022mix/prep_report.json"] = {"merged": True}
    with pytest.raises(SystemExit, match=r"holds a merge with --instructions en.*\(e\.g\. /data/f022mix_both\)"):
        ma._corpus_merge(["--instructions", "both"])
    vol_json["f022mix/prep_report.json"] = {"merged": True, "instructions": {"mode": "both"}}
    with pytest.raises(SystemExit, match=r"holds a merge with --instructions both.*\(e\.g\. /data/f022mix_en\)"):
        ma._corpus_merge([])
    with pytest.raises(SystemExit, match=r"\(e\.g\. /data/f022mix_tr\)"):
        ma._corpus_merge(["--instructions", "tr"])
    n = len(calls)
    ma._corpus_merge(["--instructions", "both"])
    assert calls[n:] == [("/data/f022mix", "both")]


def test_probe_tr_instr_pairs_each_text_with_its_english_and_turkish_instruction():
    table = rc.load_instructions_tr(REPO / rc.INSTRUCTIONS_TR)
    rows = vc.load_texts(VOXCPM / "probe_tr_instr.jsonl")
    raw = {r["id"]: r for r in vc.read_rows(VOXCPM / "probe_tr_instr.jsonl")}
    pairs: dict[str, dict[str, dict]] = {}
    for r in rows:
        base, _, lang = r["id"].rpartition("_")
        assert lang in ("en", "tr") and raw[r["id"]]["lang"] == lang and r["set"] == "emo"
        pairs.setdefault(base, {})[lang] = r
    c_words = {"canina", "acele", "cam", "cuma", "cesaret", "cidden"}
    assert len(rows) == 20 and all(set(p) == {"en", "tr"} for p in pairs.values()) and c_words <= set(pairs)
    for base, p in pairs.items():
        en, tr = p["en"], p["tr"]
        assert (en["text"], en["emotion"], en["refs"]) == (tr["text"], tr["emotion"], tr["refs"])
        assert tr["instruction"] == table[en["instruction"]]  # the table's wording: a changed table shows here
        if base in c_words:
            assert rc.pc.c_expected(rc.spoken_of(en["text"])) >= 1
    assert {p["en"]["emotion"] for p in pairs.values()} == {"angry", "cheerful", "sad", "surprised", "calm",
                                                            "empathetic"}
    trained = _wordings()["emotions"]
    assert {p["en"]["instruction"] for p in pairs.values()} & trained  # learnt wordings and new ones (generalising)
    assert {p["en"]["instruction"] for p in pairs.values()} - trained
    spec = vc.parse_entries(json.loads((VOXCPM / "probe_tr_spec.json").read_text(encoding="utf-8")))
    assert [e["name"] for e in spec] == ["f022s3-1500", "s3tr-300", "s3tr-500"]
    assert spec[0]["lora"] == "/data/exp/voxcpm2-f022-s3/ckpt/step_0001500"


# ---- plain render (--plain-render) --------------------------------------------------------------------------------


def test_plain_render_drops_the_instruction_from_the_model_text_only():
    row = {"id": "e_surprised_c001", "category": "cwords", "emotion": "surprised", "text": "Vay canına, bitti mi?",
           "spoken": "Vay canına, bitti mi?", "instruction": "Astonished.", "source": rc.CWORDS_SOURCE}
    plain = {**row, "render": "plain"}
    assert rc.render_text(row, 0) == "(Astonished.)Vay canına, bitti mi?"
    assert rc.render_text(plain, 0) == "Vay canına, bitti mi?" and rc.is_plain(plain) and not rc.is_plain(row)
    assert rc.train_text(plain) == rc.train_text(row) == "(Astonished.)Vay canına, bitti mi?"  # the manifest keeps it
    # the c-check still runs: the row has an instruction (in its training text)
    assert rc.needs_c(plain) and rc.c_checked(plain, rc.parse_args(["--out", "x"]))
    a = {"model_text": plain["spoken"], "c_expected": 1, "c_got": 0}
    assert rc.c_fails(a, plain, G) == ["c_dj"] and rc.c_fails({**a, "c_got": 1}, plain, G) == []
    # UTMOS calibrates plain tries on their own first tries
    assert rc.utmos_key(plain) == "cwords:plain/surprised" and rc.utmos_cat(plain) == "cwords:plain"
    assert rc.utmos_key(row) == "cwords/surprised" and rc.utmos_cat(row) == "cwords"
    assert rc.utmos_threshold(plain, {"cwords": 3.0, "cwords/surprised": 3.1}) is None
    assert rc.utmos_threshold(plain, {"cwords:plain": 3.2}) == 3.2
    # with_render: the rows of a category in plain mode as marked copies (texts.jsonl never has the mark)
    state = rc.new_state({})
    pool = [row, {"id": "d1", "category": "neutral", "spoken": "Tamam."}]
    assert rc.with_render(pool, state) == pool and rc.mode_of(state, "cwords") == "instructed"
    state["modes"] = {"cwords": {"mode": "plain"}}
    marked = rc.with_render(pool, state)
    assert marked[0] == plain and marked[1] is pool[1] and "render" not in row
    # the records say how a try was rendered
    att = {"t": 2, "model_text": plain["spoken"], "render_text": rc.render_text(plain, 0), "fails": [],
           "duration": 1.5}
    assert rc.attempt_record(att, plain)["render"] == "plain" and "render" not in rc.attempt_record(att, row)
    rec = rc.clip_record(att, plain, 1.5)
    assert rec["render"] == "plain" and rec["train_text"] == "(Astonished.)Vay canına, bitti mi?"
    assert rec["render_text"] == "Vay canına, bitti mi?" and "render" not in rc.clip_record(att, row, 1.5)
    # the option: categories with an instruction and a share in --mix; "none"; not given
    assert rc.parse_plain(None) is None and rc.parse_plain("none") == set()
    assert rc.parse_plain("cwords, emotions") == {"cwords", "emotions"}
    for bad in ("neutral", "", "cwords,long", "none,cwords"):
        with pytest.raises(ValueError):
            rc.parse_plain(bad)
    mix = ["--mix", "neutral=0.5,cwords=0.5"]
    assert rc.parse_args(["--out", "x", *mix, "--plain-render", "cwords"]).plain_render == "cwords"
    assert rc.parse_args(["--out", "x", *mix]).plain_render is None
    assert rc.parse_args(["--out", "x", *mix, "--plain-render", "none"]).plain_render == "none"
    for bad in (["--plain-render", "emotions"], ["--plain-render", "questions"]):  # no share / no instruction
        with pytest.raises(SystemExit):
            rc.parse_args(["--out", "x", *mix, *bad])


def test_a_plain_try_must_show_its_emotion():
    pros = {"f0_median_st": 4.0, "f0_range_st": 6.0, "rate_syl_s": 5.0, "energy_db": -20.0}
    base = rc.baseline([pros] * 25)

    def both(e: str, **d) -> tuple[str, str]:
        p = {**pros, **d}
        return rc.emotion_verdict(e, p, base, G)[0], rc.plain_verdict(e, p, base, G)[0]

    # angry / cheerful / surprised: louder or wider, and all measures together the emotion's way (emo_ref_score)
    assert both("surprised", f0_range_st=9.0, f0_median_st=6.0) == ("expected", "expected")
    assert both("surprised", f0_range_st=5.0) == ("weak", "weak")  # an instructed clip keeps it, a plain one not
    assert both("cheerful", f0_range_st=6.5, f0_median_st=1.0) == ("expected", "weak")  # a hair wider, much lower
    assert both("angry", f0_range_st=3.0) == ("opposite", "opposite")
    # sad / calm: as strict as before
    assert both("sad", rate_syl_s=4.5, f0_median_st=3.0) == ("expected", "expected")
    assert both("calm", rate_syl_s=5.3) == ("weak", "weak")
    # empathetic: no rule for an instructed clip ("recorded", kept); a plain one has to be lower and quieter (slower
    # says nothing: call 1's cwords tries were ~1 syllable / s slower than the neutral clips whatever their emotion)
    assert both("empathetic", f0_median_st=3.0) == ("recorded", "expected")
    assert both("empathetic", f0_median_st=3.0, energy_db=-21.0) == ("recorded", "expected")
    assert both("empathetic", rate_syl_s=3.5) == ("recorded", "weak")
    assert both("empathetic", f0_median_st=3.0, energy_db=-17.0) == ("recorded", "weak")  # lower but much louder
    assert both("empathetic", f0_median_st=5.0, rate_syl_s=4.0, energy_db=-21.0) == ("recorded", "weak")
    assert rc.plain_score("empathetic", {"f0_median_st": -2.0, "rate_syl_s": 9.0, "level_db": 1.0}) == 0.5
    assert rc.plain_score("sad", {"f0_median_st": -2.0}) == rc.emo_ref_score("sad", {"f0_median_st": -2.0})
    assert both("joyful", f0_median_st=3.0) == ("recorded", "recorded")  # no rule at all: fails plain (not "expected")
    assert rc.plain_verdict("surprised", None, base, G)[0] == "unchecked"
    assert rc.plain_verdict("surprised", pros, None, G)[0] == "unchecked"
    # the gate: PLAIN_FAIL for every plain verdict but "expected" (unchecked too); instructed rows as before
    row = {"id": "e", "category": "cwords", "emotion": "surprised"}
    a = {"duration": 3.0, "sim": 0.8, "utmos": 4.0, "cpu_done": True, "onset_final_bad": False,
         "prosody": {**pros, "f0_range_st": 5.0}}
    assert rc.full_fails(dict(a), row, G, {}, base) == []  # a weak surprised clip with the instruction: kept
    p = dict(a)
    assert rc.full_fails(p, {**row, "render": "plain"}, G, {}, base) == [rc.PLAIN_FAIL] == ["emotion_plain"]
    assert p["emotion_verdict"] == "weak" and p["emotion_delta"]["f0_range_st"] == -1.0
    shows = dict(a, prosody={**pros, "f0_range_st": 9.0, "f0_median_st": 6.0})
    assert rc.full_fails(shows, {**row, "render": "plain"}, G, {}, base) == []
    assert shows["emotion_verdict"] == "expected"
    assert rc.full_fails(dict(a), {**row, "render": "plain"}, G, {}, None) == ["emotion_plain"]  # no baseline


def _plain_corpus() -> tuple[list[dict], dict]:
    """A pool and state like /data/f022corpus2 after its call 1: cwords stopped at 0.044 of its tries (c000-c004 kept,
    c005-c054 dropped after 8 tries, c055 open with 3 made and decided, c056-c059 never tried), neutral clips kept
    (the baseline), every cwords emotion with its reference clip."""
    pool = [{"id": f"c{i:03d}", "category": "cwords", "emotion": "surprised", "text": "Cuma günü geldi.",
             "spoken": "Cuma günü geldi.", "instruction": "Astonished.", "source": rc.CWORDS_SOURCE} for i in range(60)]
    pool += [{"id": f"n{i:03d}", "category": "neutral", "spoken": "x" * 80, "text": "", "kinds": []}
             for i in range(50)]
    state = rc.new_state({})
    for i in range(56):
        rid = f"c{i:03d}"
        status, tries = ("kept", 3) if i < 5 else ("dropped", 8) if i < 55 else ("open", 3)
        state["rows"][rid] = {"status": status, "tries": tries, "fails": [["cer"]] * (tries - (status == "kept"))}
        if status == "kept":
            state["kept"][rid] = {"id": rid, "category": "cwords", "wav": f"{rid}.wav", "duration": 2.5,
                                  "render_spoken": "Cuma günü geldi."}
    s = rc.cat_stats(state, "cwords")
    s.update(decided=590, passed=26, decided_g2=590, passed_g2=26, first=56, first_pass=4, kept=5, kept_s=12.5,
             dropped=50, fails={"cer": 396, "sim": 59, "c_dj": 48})
    state["baseline"] = [{"f0_median_st": 4.0, "f0_range_st": 6.0, "rate_syl_s": 5.0, "energy_db": -20.0}] * 25
    state["emotion_refs"] = {"surprised": {"wav": "refs/surprised.wav"}}
    state["calls"] = [{"start": "call 1"}]
    return pool, state


def test_a_mode_switch_lifts_the_stop_and_gives_fresh_tries(tmp_path):
    pool, state = _plain_corpus()
    argv = ["--out", "x", "--hours", "1", "--mix", "neutral=0.94,cwords=0.06", "--min-sim", "0.6"]
    args = rc.parse_args([*argv, "--plain-render", "cwords"])
    assert rc.stopped_cats(state, 0.05) == {"cwords": 0.044} and rc.mode_problem(state, args) == ""
    new, sws = rc.apply_modes(state, pool, args, "now")
    assert sws == [{"category": "cwords", "from": "instructed", "to": "plain", "time": "now", "kept": 5,
                    "reopened": 50, "restarted": 1, "new": 4}]
    # the state it was given is left as it was (estimate plans on a copy)
    assert state["rows"]["c010"]["status"] == "dropped" and "modes" not in state
    assert "modes" not in state["stats"]["cwords"] and rc.stopped_cats(state, 0.05) == {"cwords": 0.044}
    # kept texts stay; the others are new again with a fresh budget from their next try
    assert new["rows"]["c000"] == state["rows"]["c000"]
    assert new["rows"]["c010"] == {**state["rows"]["c010"], "status": "new", "try0": 8,
                                   "mode_tries": {"instructed": 8}}
    assert new["rows"]["c055"]["status"] == "new" and new["rows"]["c055"]["try0"] == 3 and "c056" not in new["rows"]
    assert new["modes"]["cwords"]["mode"] == "plain" and new["modes"]["cwords"]["call"] == 2
    assert rc.mode_of(new, "cwords") == "plain" and rc.mode_of(new, "emotions") == "instructed"
    # the stop lifts: the instructed tries do not count in plain; the guess, a probe's worth of texts in flight
    st = new["stats"]["cwords"]
    assert st["decided"] == 590 and st["modes"]["instructed"]["decided_g2"] == 590
    assert st["modes"]["plain"] == rc.new_counts()
    assert rc.stopped_cats(new, 0.05) == {} and rc.measured_pass(new, "cwords") is None and rc.probing(new, "cwords")
    assert rc.p_try(new, "cwords", 0.6) == rc.PLAIN_PASS
    row, rs = pool[10], new["rows"]["c010"]
    assert rc.try_end(row, rs, 4, 6, 8) == 16 and rc.try_end(row, None, 4, 6, 8) == 8
    assert rc.wave_tries(row, rs, 4, 2, [], 6, 8) == ([8, 9, 10, 11], 4)
    old_seeds = {rc.seed_of(0, row, t) for t in range(8)}
    assert not old_seeds & {rc.seed_of(0, row, t) for t in range(8, 16)}  # no seed made again
    targets = {"cwords": 3600.0}
    rows = rc.wave_rows(pool, new, targets, 4, rc.CHARS_S, cwords_tries=8, min_pass=0.05)
    assert len(rows) == rc.PROBE_TEXTS and all(r["category"] == "cwords" for r in rows)
    assert [r["id"] for r in rc.wave_rows(pool, state, targets, 4, rc.CHARS_S, cwords_tries=8, min_pass=0.05)] == [
        "c055"]  # without the switch: stopped, its open text only
    # a plain row waits for its emotion's reference clip and for the baseline
    assert rc.plain_wait(new, row) == ""
    assert rc.wave_rows(pool, {**new, "emotion_refs": {}}, targets, 4, rc.CHARS_S, cwords_tries=8) == []
    assert "baseline" in rc.plain_wait({**new, "baseline": []}, row)
    # measured in plain: the stop applies to plain's own rate
    pm = st["modes"]["plain"]
    pm.update(decided=30, passed=6, decided_g2=30, passed_g2=6)
    assert rc.measured_pass(new, "cwords") == 0.2 and not rc.probing(new, "cwords")
    pm.update(passed_g2=1)
    assert rc.stopped_cats(new, 0.05) == {"cwords": 0.033}
    # the same flag again: nothing switches; without it: refused; "none" switches back (instructed's own rate again)
    assert rc.apply_modes(new, pool, args) == (new, [])
    no_flag = rc.parse_args(argv)
    why = rc.mode_problem(new, no_flag)
    assert "cwords render plain in this corpus" in why and "--plain-render cwords again" in why
    assert "--plain-render none" in why and "since call 2" in why
    none = rc.parse_args([*argv, "--plain-render", "none"])
    back, sws2 = rc.apply_modes(new, pool, none, "later")
    assert sws2[0]["from"] == "plain" and sws2[0]["to"] == "instructed" and rc.mode_of(back, "cwords") == "instructed"
    assert rc.stopped_cats(back, 0.05) == {"cwords": 0.044}
    assert back["stats"]["cwords"]["modes"]["plain"]["decided"] == 30
    assert [x["to"] for x in back["modes"]["cwords"]["switches"]] == ["plain", "instructed"]
    assert rc.mode_problem(back, no_flag) == ""
    # one budget per mode: back in instructed, a text that used its 8 instructed tries stays dropped, c055 (3 of them)
    # gets the 5 left
    assert {k: sws2[0][k] for k in ("kept", "spent", "new", "back")} == {"kept": 5, "spent": 50, "new": 5, "back": 1}
    assert back["rows"]["c010"]["status"] == "dropped" and back["rows"]["c010"]["mode_tries"] == {
        "instructed": 8, "plain": 0}
    assert back["rows"]["c055"]["status"] == "new" and rc.try_end(pool[55], back["rows"]["c055"], 4, 6, 8) == 8
    assert rc.wave_tries(pool[55], back["rows"]["c055"], 4, 4, [], 6, 8) == ([3, 4, 5, 6], 4)
    line = rc.switch_line(sws2[0], 8)
    assert ("1 of them less the tries they had in instructed before; 50 used up their tries in instructed before: "
            "dropped; 5 kept stay") in line
    # and plain again: no second plain budget (c010 used its 8 plain tries, c011 3 of them: 5 left from try 11)
    mid = {**new, "rows": {**new["rows"], "c010": {**new["rows"]["c010"], "status": "dropped", "tries": 16},
                           "c011": {**new["rows"]["c011"], "status": "open", "tries": 11}}}
    back2 = rc.apply_modes(rc.apply_modes(mid, pool, none)[0], pool, args)[0]
    assert back2["rows"]["c010"]["status"] == "dropped"
    assert back2["rows"]["c010"]["mode_tries"] == {"instructed": 8, "plain": 8}
    assert back2["rows"]["c011"]["status"] == "new" and back2["rows"]["c011"]["try0"] == 8
    assert rc.try_end(pool[11], back2["rows"]["c011"], 4, 6, 8) == 16
    assert rc.wave_tries(pool[11], back2["rows"]["c011"], 4, 4, [], 6, 8) == ([11, 12, 13, 14], 4)
    # a call whose --mix gives cwords no share: no flag needed, nothing switches (cwords keeps plain, renders nothing)
    other = ["--out", "x", "--hours", "1", "--mix", "long=0.5,neutral=0.5"]
    assert rc.mode_problem(new, rc.parse_args(other)) == ""
    assert rc.apply_modes(new, pool, rc.parse_args([*other, "--plain-render", "none"])) == (new, [])
    with pytest.raises(SystemExit):  # plain for a category without a share: refused
        rc.parse_args([*other, "--plain-render", "cwords"])
    # the old mode's undecided tries on disk go when the switch makes their text new (reconcile)
    out = tmp_path / "c"
    for t in (3, 4):
        _wav(rc.attempt_wav(out, "c055", t), 0.5)
        rc.attempt_wav(out, "c055", t).with_suffix(".json").write_text("{}")
    run = rc.Run(args, out, rc.with_render(pool, new), new, rc.Gates(), rc.Cost(0.02), Path("r"), FakeEngine(),
                 FakeScoring)
    run.reconcile()
    assert not (out / "attempts" / "c055").exists()


def test_estimate_plans_the_switch_with_the_guess():
    pool, state = _plain_corpus()
    argv = ["--out", "x", "--hours", "1", "--mix", "neutral=0.94,cwords=0.06", "--min-sim", "0.6"]
    cost = rc.Cost(render_per_audio=1 / 37.5)
    before = rc.estimate(pool, state, rc.parse_args(argv), cost)
    assert before["by"]["cwords"]["stopped"] == 0.044 and before["by"]["cwords"]["pool"] == 1
    assert not before["switches"]
    est = rc.estimate(pool, state, rc.parse_args([*argv, "--plain-render", "cwords"]), cost)
    b = est["by"]["cwords"]
    assert b["render"] == "plain" and b["probe"] == rc.PROBE_TEXTS and "stopped" not in b and b["pool"] == 55
    assert b["p_try"] == rc.PLAIN_PASS and b["p_keep"] == round(1 - (1 - rc.PLAIN_PASS) ** 8, 3)
    assert b["c_checked"] == b["tries"] and est["stopped_left_s"] == 0
    assert est["switches"][0]["budget"] == 8 and est["switches"][0]["reopened"] == 50
    assert "modes" not in state and state["rows"]["c010"]["status"] == "dropped"  # planned on a copy
    text = " ".join(rc.estimate_lines(est))
    assert "render mode cwords: instructed -> plain (55 texts not kept get a fresh budget of 8 tries" in text
    assert "50 of them used up in instructed; 5 kept stay" in text
    assert "(rendered plain: no instruction in the model text" in text and "p(try) guessed 0.15" in text


def test_a_plain_category_stopped_after_its_probe_says_why_it_is_incomplete(tmp_path):
    pool, state = _plain_corpus()
    for i in range(10):  # neutral has its hours
        state["kept"][f"n{i:03d}"] = {"id": f"n{i:03d}", "category": "neutral", "wav": f"n{i:03d}.wav", "duration": 3.0}
        state["rows"][f"n{i:03d}"] = {"status": "kept", "tries": 1, "fails": []}
    for c in state["kept"].values():
        c.update(spoken="x", train_text="x")
    state["emotion_refs"]["surprised"].update(seed=1, sim=0.7, score=1.0, delta={})
    args = rc.parse_args(["--out", "x", "--hours", "0.01", "--mix", "neutral=0.5,cwords=0.5", "--plain-render",
                          "cwords"])
    new = rc.apply_modes(state, pool, args, "now")[0]
    new["stats"]["cwords"]["modes"]["plain"].update(decided=40, passed=1, decided_g2=40, passed_g2=1)
    assert rc.stopped_cats(new, args.min_pass) == {"cwords": 0.025}
    rep = rc.build_report(new, pool, args, [], {}, {})
    assert not rep["complete"] and rep["held_by_stop"] == ["cwords"]  # 55 texts never tried in plain: none will be
    md = rc.render_markdown(rep)
    assert "incomplete: cwords stopped under --min-pass with texts never tried, the same command adds nothing" in md
    assert "corpus-merge --allow-incomplete takes the kept clips" in md
    src = tmp_path / "s"
    src.mkdir()
    (src / rc.MARKER).write_text("x")
    vc.write_json(src / "state.json", {})
    vc.write_json(src / "prep_report.json", rep)
    assert any("cwords stopped under --min-pass: no call adds to it" in b for b in rc.merge_sources([src], False)[1])
    # an open text still to finish: the same command continues
    new["rows"]["c055"] = {**new["rows"]["c055"], "status": "open"}
    rep = rc.build_report(new, pool, args, [], {}, {})
    assert not rep["complete"] and rep["held_by_stop"] == []
    assert "incomplete: the same corpus-render command continues" in rc.render_markdown(rep)


class PlainScoring(FakeScoring):
    """FakeScoring where the instruction turns every c into /k/ (call 1 of /data/f022corpus2: "Cuma" -> "Kuma") and a
    plain try reads it right; a plain cwords try's prosody shows its emotion (higher, wider than the neutral clips)
    except FLAT's first plain try (the neutral clips' prosody: no emotion)."""

    FLAT, FLAT_T = {"e_cheerful_c001"}, 2

    def phon(self, atts, until):
        super().phon(atts, until)
        for a in atts:
            if a["prompt"].startswith("("):
                toks = ["k" if x == "dʒ" else x for x in a["ipa"].split()]
                a["ipa"], (a["c_expected"], a["c_got"]) = " ".join(toks), rc.pc.c_counts(a["model_text"], toks)

    def rest(self, atts, until, cpu):
        super().rest(atts, until, cpu)
        for a, _, pros in cpu:
            if pros and a["category"] == "cwords" and not a["prompt"].startswith("(") and not (
                    a["rid"] in self.FLAT and a["t"] == self.FLAT_T):
                a["prosody"] = {**a["prosody"], "f0_median_st": 7.0, "f0_range_st": 10.0}


def test_run_switches_cwords_to_plain_render_then_refuses_a_call_without_it(tmp_path, audio_libs, monkeypatch,
                                                                             capsys):
    monkeypatch.setattr(rc, "BASELINE_MIN", 3)
    monkeypatch.setattr(rc, "MIN_MEASURED", 5)  # call 1 measures cwords (0 of 6 pass): stopped
    rows = DOMAIN + DOMAIN_C
    dom = (lambda: (rows, {"repo": rc.DOMAIN_REPO, "rows_read": len(rows)}))
    out = tmp_path / "c"
    eng1 = FakeEngine()
    assert rc.run(rc.parse_args(_cw_argv(tmp_path, "c", "--cwords-tries", "2")), render=eng1, scoring=PlainScoring,
                  domain=dom) == 0
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    cw = ["e_angry_c001", "e_cheerful_c001", "e_surprised_c001"]
    assert all(state["rows"][r]["status"] == "dropped" and state["rows"][r]["tries"] == 2 for r in cw)
    assert all(f == ["c_dj"] for r in cw for f in state["rows"][r]["fails"]) and "modes" not in state
    assert json.loads((out / "prep_report.json").read_text(encoding="utf-8"))["stopped"] == {"cwords": 0.0}
    for e in ("surprised", "cheerful", "angry"):  # the emotion references (--refs-from lends them on Modal)
        _wav(out / "refs" / f"{e}.wav", 2.0)
        state["emotion_refs"][e] = {"wav": f"refs/{e}.wav", "seed": 1, "sim": 0.7, "score": 1.0, "delta": {}}
    vc.write_json(out / "state.json", state)
    capsys.readouterr()
    FakeScoring.CALLS.clear()
    # call 2: --plain-render cwords
    eng2 = FakeEngine()
    assert rc.run(rc.parse_args(_cw_argv(tmp_path, "c", "--cwords-tries", "2", "--plain-render", "cwords")),
                  render=eng2, scoring=PlainScoring, domain=dom) == 0
    log_text = capsys.readouterr().out
    assert ("render mode cwords: instructed -> plain (3 texts not kept get a fresh budget of 2 tries from their next "
            "try, 3 of them used up in instructed; 0 kept stay") in log_text
    assert "under --min-pass" not in log_text  # the stop lifted
    pool = {r["id"]: r for r in _rows(out / "texts.jsonl")}
    assert not any("render" in r for r in pool.values())  # the pool file has no mark
    assert eng2.jobs and {j["rid"] for j in eng2.jobs} == set(cw)
    for j in eng2.jobs:  # no instruction, the emotion's clip, tries 2-3 (new seeds)
        assert not j["text"].startswith("(") and j["text"] == pool[j["rid"]]["spoken"] and j["try"] in (2, 3)
        assert j["ref"] == str(out / "refs" / f"{pool[j['rid']]['emotion']}.wav")
    assert not {j["seed"] for j in eng1.jobs} & {j["seed"] for j in eng2.jobs}
    phoned = {rid for call in FakeScoring.CALLS if call[0] == "phon" for rid in call[1]}
    assert phoned == set(cw)  # the c-check ran on the plain tries
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    assert state["modes"]["cwords"]["mode"] == "plain" and state["modes"]["cwords"]["call"] == 2
    assert state["calls"][-1]["plain_render"] == "cwords"
    kept = state["kept"]
    assert set(cw) <= set(kept) and kept["e_cheerful_c001"]["try"] == 3 and kept["e_surprised_c001"]["try"] == 2
    k = kept["e_surprised_c001"]
    assert k["render"] == "plain" and k["emotion_verdict"] == "expected" and k["c_expected"] == k["c_got"] == 1
    assert k["train_text"] == "(Astonished, with rising intonation.)Vay canına, bu kadar hızlı mı bitirdin?"
    assert k["render_text"] == "Vay canına, bu kadar hızlı mı bitirdin?"
    assert k["train_text"] in {r["text"] for r in _rows(out / "voxcpm_train.jsonl")}
    att = [a for a in _rows(out / "attempts.jsonl") if a["id"] == "e_cheerful_c001"]
    assert [(a["t"], a["fails"], a.get("render")) for a in att] == [
        (0, ["c_dj"], None), (1, ["c_dj"], None), (2, ["emotion_plain"], "plain"), (3, [], "plain")]
    s = state["stats"]["cwords"]
    assert (s["decided"], s["passed"], s["fails"]) == (10, 3, {"c_dj": 6, "emotion_plain": 1})
    assert (s["modes"]["instructed"]["decided"], s["modes"]["instructed"]["passed"]) == (6, 0)
    pm = s["modes"]["plain"]
    assert (pm["decided"], pm["passed"], pm["kept"], pm["first"], pm["first_pass"]) == (4, 3, 3, 3, 2)
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    b = rep["by_category"]["cwords"]
    assert b["render"] == "plain" and b["modes"]["plain"]["pass_rate"] == 0.75
    assert b["modes"]["instructed"]["pass_rate"] == 0.0 and b["modes"]["plain"]["fails"] == {"emotion_plain": 1}
    assert rep["stopped"] == {} and rep["render_modes"]["cwords"]["mode"] == "plain"
    cd = rep["distributions"]["c_check"]["by_category"]
    assert cd["cwords"] == {"checked": 6, "failed": 6} and cd["cwords (plain)"] == {"checked": 4, "failed": 0}
    assert rep["distributions"]["emotion_verdicts"]["cheerful (plain)"] == {"weak": 1, "expected": 1}
    md = (out / "PREP_REPORT.md").read_text(encoding="utf-8")
    assert "## Render modes (--plain-render)" in md and "| cwords | plain (now, since call 2) | 4 | 3 |" in md
    assert "| cwords | instructed | 6 | 0 |" in md and "emotion_plain 1" in md
    # a later call without the flag: refused, nothing changed; a dry run of --plain-render none plans the way back
    before = (out / "state.json").read_bytes()
    assert rc.run(rc.parse_args(_cw_argv(tmp_path, "c", "--cwords-tries", "2")), render=FakeEngine(),
                  scoring=PlainScoring, domain=dom) == 2
    assert "refused: cwords render plain in this corpus" in capsys.readouterr().out
    assert rc.run(rc.parse_args(_cw_argv(tmp_path, "c", "--cwords-tries", "2", "--plain-render", "cwords", "--cfg",
                                         "2.5")), render=FakeEngine(), scoring=PlainScoring, domain=dom) == 2
    assert "was rendered otherwise ({'cfg': '2.0 -> 2.5'})" in capsys.readouterr().out  # the flag excuses nothing else
    assert rc.main([*_cw_argv(tmp_path, "c", "--cwords-tries", "2"), "--dry-run"]) == 2
    assert "--plain-render cwords again" in capsys.readouterr().out
    assert rc.main([*_cw_argv(tmp_path, "c", "--cwords-tries", "2", "--plain-render", "none"), "--dry-run"]) == 0
    assert "render mode cwords: plain -> instructed (0 texts not kept" in capsys.readouterr().out
    assert rc.main([*_cw_argv(tmp_path, "c", "--cwords-tries", "2", "--plain-render", "cwords"), "--dry-run"]) == 0
    sample = capsys.readouterr().out  # its cwords line: the training text with the instruction, the render text not
    assert re.search(r"cwords/e_\w+: train '\(\w[^']*'; render '[^(]", sample)
    assert (out / "state.json").read_bytes() == before


def test_modal_app_refuses_a_corpus_render_that_would_drop_plain_render(monkeypatch, capsys):
    from types import SimpleNamespace

    ma, _ = _modal_app()
    state = {**rc.new_state({}), "modes": {"cwords": {"mode": "plain", "call": 3, "since": "2026-10-13T12:00:00Z"}}}
    monkeypatch.setattr(ma, "_rc", lambda: rc)
    monkeypatch.setattr(ma, "_vol_files", lambda rel: {})
    monkeypatch.setattr(ma, "_vol_json", lambda rel: state if rel == "f022corpus2/state.json" else None)
    monkeypatch.setattr(ma, "_vol_rows", lambda rel: [])
    monkeypatch.setattr(ma, "corpus", SimpleNamespace(remote=lambda *a: pytest.fail("no GPU")))
    argv = ["--hours", "1.0", "--mix", "long=0.47,neutral=0.47,cwords=0.06", "--min-sim", "0.6", "--laughter-usd", "0",
            "--dry-run"]
    with pytest.raises(SystemExit, match=r"cwords render plain in this corpus.*nothing started"):
        ma._corpus(argv, 2.0, out="/data/f022corpus2")
    ma._corpus([*argv, "--plain-render", "cwords"], 2.0, out="/data/f022corpus2")
    text = capsys.readouterr().out
    assert "(rendered plain: no instruction in the model text" in text and "--dry-run: nothing started" in text
