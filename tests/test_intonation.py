"""Question intonation: the tr_soru set, score_intonation's item rules, timing and summary, the scorer on synthetic
tones (needs librosa; skipped without it), and synth_eval's soru path (stubbed backends)."""

import json
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VOXCPM = REPO / "tts" / "voxcpm"
sys.path[:0] = [str(VOXCPM), str(REPO / "src")]

import score_intonation as si  # noqa: E402
import synth_eval  # noqa: E402
from score_intonation import (MI, WH, core, infer_items, intonation_summary, judge, load_set,  # noqa: E402
                              whisper_times)
from trnorm import normalize  # noqa: E402

SET = load_set()


def test_import_needs_no_numpy():
    code = (f"import sys; sys.path[:0] = [{str(VOXCPM)!r}, {str(REPO / 'src')!r}]; import score_intonation; "
            "bad = {'numpy', 'librosa', 'scipy', 'soundfile', 'torch'} & set(sys.modules); assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


# ---- the set ----------------------------------------------------------------------------------------------------


def test_set_fields_and_categories():
    assert 30 <= len(SET) <= 40
    assert len({r["id"] for r in SET}) == len(SET)
    for r in SET:
        assert set(r) == {"id", "category", "text", "spoken", "pair_id", "pattern", "peak", "peak_word"}, r["id"]
        assert r["category"] in si.CATEGORIES and r["pattern"] == si.PATTERNS[r["category"]]
        assert r["text"].endswith("." if r["category"] == "duz_eslesik" else "?"), r["id"]
    cats = {c: sum(r["category"] == c for r in SET) for c in si.CATEGORIES}
    assert all(n >= 4 for n in cats.values()), cats
    assert sum(any(ch.isdigit() for ch in r["text"]) for r in SET) >= 8  # numbers, money, time, percent: trnorm


def test_spoken_is_trnorm_and_peak_points_at_peak_word():
    for r in SET:
        assert normalize(r["text"]) == r["spoken"], r["id"]
        words = r["spoken"].split()
        assert core(words[r["peak"]]) == core(r["peak_word"]), r["id"]
        assert si.set_item(r, words)["target"] == r["peak"]


def test_peak_word_matches_its_category():
    for r in SET:
        w = [core(x) for x in r["spoken"].split()]
        k, cat = r["peak"], r["category"]
        if cat in ("mi_soru", "degil_mi"):
            assert MI.match(w[k + 1]), r["id"]  # the syllable right before mI
            assert (w[k] == "değil") == (cat == "degil_mi") and (cat != "degil_mi" or k + 2 == len(w)), r["id"]
        elif cat == "ne_soru":
            assert WH.match(w[k]) and not any(MI.match(x) for x in w[1:]), r["id"]
        elif cat == "yukselen":
            assert k == len(w) - 1 and not any(MI.match(x) or WH.match(x) for x in w), r["id"]


def test_pairs_are_matched():
    groups = {}
    for r in SET:
        if r["pair_id"]:
            groups.setdefault(r["pair_id"], {})[r["category"]] = r
    assert len(groups) >= 8
    for pid, g in groups.items():
        assert "duz_eslesik" in g and "mi_soru" in g, pid  # each yes/no question has its declarative
        decl = [core(x) for x in g["duz_eslesik"]["spoken"].split()]
        q = [core(x) for x in g["mi_soru"]["spoken"].split()]
        assert [x for x in q if not MI.match(x)] == decl or q[:g["mi_soru"]["peak"]] == decl[:g["mi_soru"]["peak"]], pid
        assert core(g["duz_eslesik"]["spoken"].split()[g["duz_eslesik"]["peak"]]).startswith(
            core(g["mi_soru"]["peak_word"])), pid  # the same word: "geliyor(sun)", "otuzda"
        if "yukselen" in g:  # same words, "?" instead of "."
            assert g["yukselen"]["text"][:-1] == g["duz_eslesik"]["text"][:-1], pid
        if "degil_mi" in g:
            assert g["degil_mi"]["text"] == g["duz_eslesik"]["text"][:-1] + ", değil mi?", pid
    assert all(r["pair_id"] for r in SET if r["category"] != "ne_soru")


def test_text_rules_agree_with_the_annotation():
    for r in SET:
        (it,) = infer_items(r["spoken"].split())
        if r["category"] == "duz_eslesik":
            assert (it["category"], it["pattern"]) == ("duz", "fall"), r["id"]
        else:
            assert (it["category"], it["pattern"], it["target"]) == (r["category"], r["pattern"], r["peak"]), r["id"]


def test_load_sets_soru():
    rows = synth_eval.load_sets(None, ["soru"], 0)
    assert len(rows) == len(SET) and {r["set"] for r in rows} == {"soru"}
    for row, r in zip(rows, SET):
        assert row["model_text"] == normalize(r["text"]) and row["refs"] == [r["spoken"]]
        assert (row["pattern"], row["peak"], row["peak_word"], row["pair_id"]) == (
            r["pattern"], r["peak"], r["peak_word"], r["pair_id"])
    firsts = synth_eval.load_sets(None, ["soru"], 5)  # stratified: one per category
    assert sorted(r["category"] for r in firsts) == sorted(si.CATEGORIES)
    assert len(synth_eval.load_sets(None, ["trzor", "soru"], 0)) == 155 + len(SET)


# ---- rules ------------------------------------------------------------------------------------------------------


def test_infer_items_from_a_passage():
    text = ("Toplam tutar iki bin lira; uygun mu? Teslimat adresi hazır. Eksik bir bilgi var mı? Hangi gün gelelim? "
            "Yarın geliyorsun? Bu akşam açıksınız, değil mi? Peki")
    words = text.split()
    got = [(it["category"], it["pattern"], core(words[it["target"]])) for it in infer_items(words)]
    assert got == [("mi_soru", "peak_fall", "uygun"), ("duz", "fall", "adresi"), ("mi_soru", "peak_fall", "var"),
                   ("ne_soru", "wh_peak", "hangi"), ("yukselen", "final_rise", "geliyorsun"),
                   ("degil_mi", "peak_fall", "değil")]  # "Peki" has no clause-final punctuation: no item
    (ne,) = infer_items("Ne?".split())
    assert ne["category"] == "ne_soru" and ne["unscored"] == "one-word wh-question"  # nothing after the wh-word
    assert "unscored" not in infer_items("Peki ne?".split())[0]
    assert infer_items("Geliyor musun ?".split())[0]["target"] == 0  # a spaced "?" still ends the question
    assert infer_items("Kimse gelmedi?".split())[0]["category"] == "yukselen"  # kimse is not a wh-word
    assert infer_items("Kaçırdın?".split())[0]["category"] == "yukselen"  # nor is kaçırmak
    cat = lambda t: [(it["category"], core(t.split()[it["target"]])) for it in infer_items(t.split())]  # noqa: E731
    for t, w in [("Gelecek miydin?", "gelecek"), ("Orada mıydınız?", "orada"), ("Hazır mısınız?", "hazır"),
                 ("Doğru mudur?", "doğru"), ("Yorgun muymuşsun?", "yorgun"), ("Evde miydiler?", "evde")]:
        assert cat(t) == [("mi_soru", w)], t
    for t, w in [("Sorun nedir?", "nedir"), ("Kimsiniz?", "kimsiniz"), ("Bunu kimde bıraktın?", "kimde"),
                 ("Bu kiminle geldi?", "kiminle"), ("Dün neydi o ses?", "neydi")]:
        assert cat(t) == [("ne_soru", w)], t
    for t in ("Mısır pahalı mı?", "Kimya zor mu?", "Neyse geliyor musun?"):  # not particles, not wh-words
        assert cat(t)[0][0] == "mi_soru" and cat(t)[0][1] in ("pahalı", "zor", "geliyor"), t


def test_judge_thresholds():
    assert judge("peak_fall", {"peak_st": 2.0, "drop_st": 3.0, "peak_at": -0.5})
    assert judge("peak_fall", {"peak_st": 2.0, "drop_st": 3.0, "peak_at": -2.0})  # the penult: geLİyor mu
    assert judge("peak_fall", {"peak_st": 2.0, "drop_st": 3.0, "peak_at": 0.3})  # peak delay into mI
    assert judge("peak_fall", {"peak_st": 5, "drop_st": 5, "peak_at": 0.5}) is False  # the peak on mI itself
    assert judge("peak_fall", {"peak_st": 5, "drop_st": 5, "peak_at": -2.5}) is False  # too early
    assert judge("peak_fall", {"peak_st": 5, "drop_st": 5, "peak_at": -1.5}, earliest=-1.0) is False  # 1-syll. word
    assert judge("peak_fall", {"peak_st": 1.9, "drop_st": 8.0, "peak_at": -0.5}) is False
    assert judge("wh_peak", {"peak_st": 5, "drop_st": 2}) is False
    # a measure it needs is missing: not scorable, not a failure
    assert judge("peak_fall", {"peak_st": math.nan, "drop_st": 5.0, "peak_at": -0.5}) is None
    assert judge("peak_fall", {"peak_st": None, "drop_st": None, "peak_at": None}) is None
    assert judge("peak_fall", {"peak_st": 5.0, "drop_st": 5.0}) is None
    assert judge("fall", {"peak_st": None, "end_st": -1.0, "slope_st_s": 0.0})  # fall needs no peak
    assert judge("final_rise", {"end_st": 1.5, "slope_st_s": 5.0})
    assert not judge("final_rise", {"end_st": 3.0, "slope_st_s": -1.0})
    assert judge("fall", {"end_st": -2.0, "slope_st_s": -10.0}) and not judge("fall", {"end_st": 0.5, "slope_st_s": -1})
    with pytest.raises(ValueError):
        judge("wobble", {})


def test_whisper_times_expands_numbers_and_fills_gaps():
    words = "Toplantıya kırk beş kişi hemen katıldı.".split()
    ww = [["Toplantıya", 0.0, 0.6], ["45", 0.6, 1.0], ["kişi", 1.0, 1.3], ["katıldı.", 1.5, 2.1]]
    t = [x for span in whisper_times(words, ww) for x in span]
    # "45" -> "kırk beş" (one syllable each); "hemen" not heard: placed between its neighbours
    assert t == pytest.approx([0.0, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.3, 1.3, 1.5, 1.5, 2.1])
    assert whisper_times(words, [["bambaşka", 0.0, 1.0], ["bir", 1.0, 1.2], ["cümle", 1.2, 2.0]]) is None
    assert whisper_times(words, []) is None


def _item(cat: str, peak, ok: bool | None, pair=None, timing="whisper", end=-1.0) -> dict:
    return {"category": cat, "pattern": si.PATTERNS[cat], "word": "x", "peak_st": peak, "drop_st": None if peak is None
            else peak - end, "end_st": end, "slope_st_s": -5.0, "peak_at": -0.5 if cat == "mi_soru" else None,
            "range_st": 6.0, "pass": ok, "timing": timing, **({"pair_id": pair} if pair else {})}


def test_summary_keys_auc_and_pairs():
    rows = [{"set": "soru", "id": "q1", "intonation": [_item("mi_soru", 5.0, True, "g1")]},
            {"set": "soru", "id": "q2", "intonation": [_item("mi_soru", 1.0, False, "g2")]},
            {"set": "soru", "id": "q3", "intonation": [_item("mi_soru", None, False, "g3")]},
            {"set": "soru", "id": "d1", "intonation": [_item("duz_eslesik", 0.0, True, "g1")]},
            {"set": "soru", "id": "d2", "intonation": [_item("duz_eslesik", 2.0, True, "g2", "syllables")]},
            {"set": "soru", "id": "w1", "intonation": [_item("ne_soru", 3.0, True)]},
            {"set": "soru", "id": "w2", "intonation": [_item("ne_soru", 9.0, None, timing="syllables")]},  # unscored
            {"set": "soru", "id": "bad", "intonation_error": "ValueError: no voiced speech found"},
            {"set": "heldout", "id": "h", "intonation": [_item("mi_soru", 1.0, False), _item("duz", 0.0, True)]},
            {"set": "trzor", "id": "t"}]
    s = intonation_summary(rows)
    assert list(s) == ["heldout", "heldout/duz", "heldout/mi_soru", "soru", "soru/duz_eslesik", "soru/mi_soru",
                       "soru/ne_soru"]
    soru = s["soru"]
    assert set(soru) == {"items", "unscored", "passed", "pass_rate", "peak_st", "drop_st", "end_st", "slope_st_s",
                         "peak_at", "questions_pass_rate", "auc_peak", "pairs", "pair_acc", "timing"}
    assert (soru["items"], soru["unscored"], soru["passed"], soru["pass_rate"]) == (7, 1, 4, 0.667)  # w2 left out
    assert soru["questions_pass_rate"] == 0.5  # 2 of the 4 scored questions
    assert soru["auc_peak"] == 0.75  # mi_soru peaks {5, 1} vs declaratives {0, 2}: 3 of 4 comparisons higher
    assert (soru["pairs"], soru["pair_acc"]) == (2, 0.5)  # g1 won, g2 lost, g3 has no peak
    assert soru["timing"] == {"syllables": 2, "whisper": 5}  # all items, scored or not
    assert s["soru/mi_soru"] == {"items": 3, "unscored": 0, "passed": 1, "pass_rate": 0.333, "peak_st": 3.0,
                                 "drop_st": 4.0, "end_st": -1.0, "slope_st_s": -5.0, "peak_at": -0.5}
    assert s["soru/ne_soru"]["unscored"] == 1 and s["soru/ne_soru"]["peak_st"] == 3.0  # w2's 9.0 not in the mean
    only = intonation_summary([{"set": "x", "id": "u", "intonation": [_item("ne_soru", 1.0, None)]}])["x"]
    assert (only["items"], only["unscored"], only["pass_rate"], only["questions_pass_rate"]) == (1, 1, None, None)
    assert s["heldout"]["auc_peak"] == 1.0 and s["heldout"]["pairs"] == 0 and s["heldout"]["pair_acc"] is None
    assert intonation_summary([]) == {}
    json.dumps(s, allow_nan=False)


def test_fold_octaves():
    np = pytest.importorskip("numpy")
    up = si.fold_octaves(np.r_[np.zeros(30), np.full(20, 12.5)], np.arange(50))  # doubled last syllable
    assert up[30:] == pytest.approx(np.full(20, 0.5)) and not up[:30].any()
    # an octave-sized jump: the piece farther from the median moves, even when it is < 9 st off
    s = np.r_[np.zeros(30), [-4.1, -4.1], np.full(20, 8.5)]
    assert si.fold_octaves(s, np.arange(52))[32:] == pytest.approx(np.full(20, -3.5))
    rise = np.linspace(0.0, 10.0, 40)  # a real rise: no jump between frames, left alone
    assert si.fold_octaves(rise, np.arange(40)) == pytest.approx(rise)
    # far from the median but after a voiceless gap, with no jump: a real peak or final rise, left alone
    for hi in (11.0, 14.0):
        s = np.r_[np.zeros(30), np.linspace(hi - 4, hi, 10), np.zeros(20)]
        idx = np.r_[np.arange(30), np.arange(37, 47), np.arange(53, 73)]
        assert si.fold_octaves(s, idx) == pytest.approx(s)


def test_syllable_times_breaks_and_pauses():
    np = pytest.importorskip("numpy")
    words = "Bir iki üç dört beş altı, uygun mu? Yedi sekiz dokuz on bir.".split()  # 8 + 3 + 8 syllables
    silent = np.zeros(1500, bool)
    silent[500:950] = True  # one long pause, after the first phrase; none between the second and the third
    t = si.syllable_times(words, silent)
    assert t[5][1] == pytest.approx(5.0) and t[6][0] == pytest.approx(9.5)  # "altı," ends where the pause starts
    assert t[7][1] == pytest.approx(11.0)  # "uygun mu?": 3 of the 11 syllables after it, not one frame
    # breaks at two pauses, a hesitation inside a phrase, a comma without a pause; 20 frames (0.2 s) per syllable
    words = "Bir iki üç, dört beş altı yedi sekiz. Dokuz on, bir iki?".split()  # 4 + 8 + 3 + 3 syllables
    silent = np.zeros(520, bool)
    for a, b in ((80, 130), (210, 240), (320, 400)):
        silent[a:b] = True
    t = si.syllable_times(words, silent)
    assert t[2][1] == pytest.approx(0.8) and t[3][0] == pytest.approx(1.3)  # "üç," | pause | "dört"
    assert t[5][1] == pytest.approx(2.1) and t[6][0] == pytest.approx(2.4)  # the hesitation is left out
    assert t[7][1] == pytest.approx(3.2) and t[8][0] == pytest.approx(4.0)
    assert t[9][1] == pytest.approx(4.6) and t[11][1] == pytest.approx(5.2)  # "on," | "bir iki?" by syllables
    assert all(b > a for a, b in t)


def test_measure_leaves_unscorable_items_out():
    np = pytest.importorskip("numpy")
    words = "Yarın sabah geliyor musun?".split()
    times = [(0.0, 0.4), (0.4, 0.8), (0.8, 1.4), (1.4, 1.8)]
    st = np.full(180, np.nan)
    st[:80] = 0.0  # voiced over "Yarın sabah" only: nothing in the target window
    it = si.set_item({"peak": 2, "peak_word": "geliyor", "category": "mi_soru", "pattern": "peak_fall"}, words)
    m = si.measure(st, times, words, it)
    assert m["pass"] is None and m["peak_st"] is None and m["note"].startswith("not scored")
    (ne,) = infer_items(["Ne?"])
    m = si.measure(np.r_[np.linspace(0, 6, 50), np.linspace(6, -2, 50)], [(0.0, 1.0)], ["Ne?"], ne)
    assert m["pass"] is None and m["note"] == "not scored: one-word wh-question" and m["peak_st"] > 2


# ---- the scorer on synthetic tones ------------------------------------------------------------------------------

TONE_SR = 24000
TONES = {  # id: (category, peak, text, F0 contour points (time 0..1, st re 120 Hz))
    "q": ("mi_soru", 2, "Yarın sabah geliyor musun?", [(0, 0), (0.55, 0), (0.72, 6), (1.0, -3)]),
    "d": ("duz_eslesik", 2, "Yarın sabah geliyorsun.", [(0, 1), (0.3, 2), (1.0, -3)]),
    "r": ("yukselen", 2, "Yarın sabah geliyorsun?", [(0, 0), (0.7, -1), (1.0, 6)]),
    "flat": ("mi_soru", 2, "Yarın sabah geliyor musun?", [(0, 0), (1.0, 0)]),
    "early": ("mi_soru", 2, "Yarın sabah geliyor musun?", [(0, 0), (0.15, 7), (0.35, 0), (1.0, -1)]),
}


def _tone(points, dur=1.8, base=120.0, pad=0.25):
    np = pytest.importorskip("numpy")
    t = np.arange(int(dur * TONE_SR)) / TONE_SR
    xs, ys = zip(*points)
    f0 = base * 2 ** (np.interp(t / dur, xs, ys) / 12)
    ph = 2 * np.pi * np.cumsum(f0) / TONE_SR
    y = sum(np.sin(h * ph) / h for h in range(1, 9)) * 0.2  # a harmonic sweep, 8 partials
    z = np.zeros(int(pad * TONE_SR))
    return np.concatenate([z, y, z]).astype(np.float32)


@pytest.fixture
def tone_dir(tmp_path):
    pytest.importorskip("librosa")
    pytest.importorskip("scipy")
    sf = pytest.importorskip("soundfile")
    d = tmp_path / "eval"
    (d / "wav" / "soru").mkdir(parents=True)
    rows = []
    for rid, (cat, peak, text, pts) in TONES.items():
        sf.write(d / "wav" / "soru" / f"{rid}.wav", _tone(pts), TONE_SR)
        rows.append({"set": "soru", "id": rid, "category": cat, "text": text, "model_text": text,
                     "pair_id": "g1" if rid in ("q", "d") else None, "pattern": si.PATTERNS[cat], "peak": peak,
                     "peak_word": text.split()[peak].strip(".?"), "wav": f"/data/exp/x/wav/soru/{rid}.wav"})
    (d / "wav" / "soru" / "broken.wav").write_bytes(b"RIFF")
    rows.append({**rows[0], "id": "broken"})
    rows.append({"set": "trzor", "id": "t1", "category": "sayi", "text": "Bir.", "utmos": 3.0})
    (d / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps({"wer": {"soru": {"wer": 0.0}}, "intonation_error": "old"}),
                                    encoding="utf-8")
    return d


def _read(d: Path) -> tuple[dict, dict]:
    rows = [json.loads(x) for x in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    return {r["id"]: r for r in rows}, json.loads((d / "summary.json").read_text(encoding="utf-8"))


def test_scorer_on_synthetic_tones(tone_dir, capsys):
    before = {p.name: p.read_bytes() for p in tone_dir.iterdir() if p.is_file()}
    assert si.main([str(tone_dir), "--whisper", "none", "--dry-run"]) == 0
    assert {p.name: p.read_bytes() for p in tone_dir.iterdir() if p.is_file()} == before  # dry run
    assert si.main([str(tone_dir), "--whisper", "none"]) == 0
    rows, summary = _read(tone_dir)
    res = {k: r["intonation"][0] for k, r in rows.items() if "intonation" in r}
    assert set(res) == {"q", "d", "r", "flat", "early"}
    assert all(it["timing"] == "syllables" and it["word"] == "geliyor" + ("sun" if k in "dr" else "")
               for k, it in res.items())
    q, d, r = res["q"], res["d"], res["r"]
    assert q["pass"] and q["peak_st"] >= 3.5 and q["drop_st"] >= 5 and q["end_st"] < 0 and q["slope_st_s"] < 0
    assert d["pass"] and d["peak_st"] < 0 and d["end_st"] < -2
    assert r["pass"] and r["end_st"] >= 4 and r["slope_st_s"] > 5
    assert not res["flat"]["pass"] and abs(res["flat"]["peak_st"]) < 0.5 and abs(res["flat"]["drop_st"]) < 0.5
    assert not res["early"]["pass"]  # the peak is on the first word, not before mI
    assert rows["broken"]["intonation_error"] and "intonation" not in rows["broken"]
    assert "intonation" not in rows["t1"] and rows["t1"]["utmos"] == 3.0  # other sets untouched
    s = summary["intonation"]
    assert summary["wer"] == {"soru": {"wer": 0.0}}
    assert summary["intonation_error"].startswith("1 rows not scored") and "soru/broken" in summary["intonation_error"]
    assert set(s) == {"soru", "soru/mi_soru", "soru/duz_eslesik", "soru/yukselen"}
    assert (s["soru"]["items"], s["soru"]["passed"], s["soru"]["auc_peak"], s["soru"]["pair_acc"]) == (5, 3, 1.0, 1.0)
    assert s["soru/mi_soru"]["pass_rate"] == 0.333 and s["soru/yukselen"]["pass_rate"] == 1.0
    out = capsys.readouterr().out
    assert "soru/mi_soru" in out and "auc_peak 1.000" in out


def test_scorer_with_word_timestamps(tone_dir):
    rows = [json.loads(x) for x in (tone_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    q = next(r for r in rows if r["id"] == "q")
    pad = 0.25  # words where the tone has them: 9 syllables over 1.8 s
    q["words"] = [["Yarın", pad, pad + 0.4], ["sabah", pad + 0.4, pad + 0.8], ["geliyor", pad + 0.8, pad + 1.4],
                  ["musun?", pad + 1.4, pad + 1.8]]
    (it,) = si.score_row(q, tone_dir / "wav" / "soru" / "q.wav")
    assert it["timing"] == "whisper" and it["pass"] and it["peak_st"] >= 3.5


def _score_tone(tmp_path, y, cat: str, peak: int, text: str) -> dict:
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    pytest.importorskip("scipy")
    sf.write(tmp_path / "x.wav", y, TONE_SR)
    row = {"set": "soru", "id": "x", "category": cat, "text": text, "model_text": text, "pattern": si.PATTERNS[cat],
           "peak": peak, "peak_word": text.split()[peak].strip(".?")}
    (it,) = si.score_row(row, tmp_path / "x.wav")
    return it


S9 = 1 / 9  # "Yarın sabah geliyor musun?": 9 syllables, 0.2 s each


@pytest.mark.parametrize("pts, ok", [
    ([(0, 0), (6 * S9, 0), (6.5 * S9, 6), (7 * S9, 5), (8 * S9, -1), (1, -3)], True),  # on "yor", before mI
    ([(0, 0), (7 * S9, 0), (7.5 * S9, 6), (8 * S9, 5), (8.5 * S9, -1), (1, -3)], False),  # on the particle "mu"
    ([(0, 0), (4 * S9, 0), (4.5 * S9, 6), (5 * S9, 5), (6 * S9, -1), (1, -3)], False),  # on "ge", too early
])
def test_peak_location_before_mi(tmp_path, pts, ok):
    it = _score_tone(tmp_path, _tone(pts), "mi_soru", 2, "Yarın sabah geliyor musun?")
    assert it["pass"] is ok and it["peak_st"] > 2 and it["drop_st"] > 3  # height and fall alone pass all three
    assert (-2 <= it["peak_at"] <= si.PEAK_LATE) is ok


def _gapped(*parts):
    """Harmonic sweeps [(points, dur)] with 70 ms of noise (a voiceless consonant) between them, padded."""
    np = pytest.importorskip("numpy")
    gap = np.random.default_rng(0).normal(0, 0.01, int(0.07 * TONE_SR)).astype(np.float32)
    ys = [_tone(pts, dur=dur, pad=0.0) for pts, dur in parts]
    pad = np.zeros(int(0.25 * TONE_SR), np.float32)
    return np.concatenate([pad, *[x for y in ys for x in (y, gap)][:-1], pad])


@pytest.mark.parametrize("rise, top", [(11, 11), (14, 13)])  # pass 2 reaches ~ +13.6 st over q65 (PASS2_HI)
def test_large_excursions_after_a_voiceless_gap(tmp_path, rise, top):
    y = _gapped(([(0, 1), (1, -1)], 1.35), ([(0, rise - 6), (1, rise)], 0.28))  # final "-sun?" rising to +rise st
    it = _score_tone(tmp_path, y, "yukselen", 2, "Yarın sabah geliyorsun?")
    assert it["pass"] and it["end_st"] > rise - 3
    y = _gapped(([(0, 0), (1, 0)], 0.75), ([(0, top - 1), (0.4, top), (1, top - 4)], 0.25), ([(0, 0), (1, -3)], 0.6))
    it = _score_tone(tmp_path, y, "mi_soru", 2, "Odada iki kişi mi kalacaksınız?")  # "kişi" peaks after "ş"
    assert it["pass"] and it["peak_st"] > top - 2


def test_cli_refuses_dir_without_wavs(tmp_path):
    (tmp_path / "rows.jsonl").write_text(json.dumps({"set": "soru", "id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        si.main([str(tmp_path)])
    with pytest.raises(SystemExit):
        si.main([str(tmp_path / "nope")])


# ---- synth_eval's soru path (stubbed backends) --------------------------------------------------------------------


class _Killed(BaseException):
    """The budget deadline's SIGKILL: no `except Exception` catches it."""


@pytest.fixture
def fake_backends(monkeypatch):
    """synth_eval.main without numpy/soundfile/torch/voxcpm/faster-whisper/jiwer; the judge records its kwargs."""
    calls = []

    class Tts:
        tts_model = types.SimpleNamespace(sample_rate=10)

        def generate(self, text, **kw):
            return [0.0] * 10

    class Judge:
        def __init__(self, *a, **kw):
            pass

        def transcribe(self, wav, **kw):
            calls.append(kw)
            words = [types.SimpleNamespace(word=" ok", start=0.1234, end=0.5)] if kw.get("word_timestamps") else None
            return iter([types.SimpleNamespace(text=" ok ", words=words)]), None

    def sf_write(path, data, sr):
        Path(path).write_bytes(b"RIFF")

    mods = {"numpy": types.SimpleNamespace(asarray=lambda x, dtype=None: x, float32=None),
            "soundfile": types.SimpleNamespace(write=sf_write),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
            "voxcpm": types.SimpleNamespace(VoxCPM=types.SimpleNamespace(from_pretrained=lambda *a, **kw: Tts())),
            "faster_whisper": types.SimpleNamespace(WhisperModel=Judge)}
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(synth_eval, "wer_parts", lambda ref, hyp: (1, 4))
    return calls


def _run_eval(out: Path, *extra: str) -> int:
    return synth_eval.main(["--model", "m", "--sets", "trzor,soru", "--limit", "5", "--out", str(out), "--no-utmos",
                            *extra])


def _eval_read(d: Path) -> tuple[list[dict], dict]:
    return ([json.loads(x) for x in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines()],
            json.loads((d / "summary.json").read_text(encoding="utf-8")))


def test_eval_soru_scores_intonation(tmp_path, monkeypatch, fake_backends):
    seen = []

    def fake_score_row(row, path, whisper=None):
        seen.append((row["id"], Path(path).relative_to(tmp_path).as_posix(), row["words"]))
        if row["category"] == "ne_soru":
            raise ValueError("no voiced speech found")
        return [_item(row["category"], 3.0, True, row.get("pair_id"))]

    monkeypatch.setattr(si, "score_row", fake_score_row)
    assert _run_eval(tmp_path) == 0
    rows, summary = _eval_read(tmp_path)
    soru = [r for r in rows if r["set"] == "soru"]
    assert len(soru) == 5 and len(rows) == 10
    assert [s[0] for s in seen] == [r["id"] for r in soru]  # only the soru rows, from their WAVs
    assert all(p.startswith("wav/soru/") for _, p, _ in seen)
    assert all(w == [["ok", 0.123, 0.5]] for _, _, w in seen)  # the judge's word timestamps reached the scorer
    assert [c.get("word_timestamps") for c in fake_backends] == [False] * 5 + [True] * 5
    assert all("words" not in r for r in rows if r["set"] == "trzor")
    assert summary["wer"]["soru"]["rows"] == 5 and "soru/mi_soru" in summary["wer"]
    assert summary["intonation"]["soru"]["items"] == 4 and "soru/ne_soru" not in summary["intonation"]
    assert summary["intonation_error"].startswith("1 rows not scored") and "ValueError" in summary["intonation_error"]
    assert [r["id"] for r in soru if "intonation_error" in r] == ["ne_soru_01"]


def test_eval_intonation_failure_keeps_the_rest(tmp_path, monkeypatch, fake_backends):
    def broken(*a, **kw):
        raise ImportError("No module named 'librosa'")

    monkeypatch.setattr(si, "score_row", lambda row, path, whisper=None: [])
    monkeypatch.setattr(si, "intonation_summary", broken)
    assert _run_eval(tmp_path) == 0
    rows, summary = _eval_read(tmp_path)
    assert summary["intonation_error"] == "ImportError: No module named 'librosa'" and "intonation" not in summary
    assert summary["wer"]["soru"]["rows"] == 5 and len(rows) == 10


def test_eval_killed_during_intonation_leaves_wer_on_disk(tmp_path, monkeypatch, fake_backends):
    def killed(*a, **kw):
        raise _Killed

    monkeypatch.setattr(si, "score_row", killed)
    with pytest.raises(_Killed):
        _run_eval(tmp_path)
    rows, summary = _eval_read(tmp_path)
    assert summary["intonation_error"].startswith("not scored: the run ended before it")
    assert summary["wer"]["soru"]["rows"] == 5 and len(rows) == 10


def test_eval_without_soru_or_with_no_intonation(tmp_path, monkeypatch, fake_backends):
    monkeypatch.setattr(si, "score_row", lambda *a, **kw: pytest.fail("intonation ran"))
    assert synth_eval.main(["--model", "m", "--sets", "trzor", "--limit", "3", "--out", str(tmp_path / "a"),
                            "--no-utmos"]) == 0
    assert _run_eval(tmp_path / "b", "--no-intonation") == 0
    for d in ("a", "b"):
        _, summary = _eval_read(tmp_path / d)
        assert "intonation" not in summary and "intonation_error" not in summary
