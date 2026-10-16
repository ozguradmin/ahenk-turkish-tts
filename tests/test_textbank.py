"""tts/voxcpm/build_textbank.py and the text bank in tts/voxcpm/textbank/: schema, spoken forms, question peaks and pairs,
emotion instruction splits, leakage guard, counts, the laughter trial's lines (stdlib + trnorm only)."""

import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VOXCPM = REPO / "tts" / "voxcpm"
sys.path[:0] = [str(VOXCPM), str(REPO / "src")]

import build_textbank as tb  # noqa: E402
from trnorm import normalize  # noqa: E402


@pytest.fixture(scope="module")
def sources():
    return tb.load_sources()


@pytest.fixture(scope="module")
def guard():
    return tb.leak_guard()


@pytest.fixture(scope="module")
def built(sources, guard):
    g, protected = guard
    return tb.build_bank(sources, g, protected)


@pytest.fixture(scope="module")
def rows(built):
    return built["rows"]


def within(n, target, tol=0.15):
    return abs(n - target) <= tol * target


def test_import_needs_no_heavy_deps():
    code = (f"import sys; sys.path[:0] = [{str(VOXCPM)!r}, {str(REPO / 'src')!r}]; import build_textbank; "
            "bad = {'pyarrow', 'numpy', 'soundfile', 'torch', 'torchaudio', 'scipy', 'huggingface_hub', 'modal'} "
            "& set(sys.modules); assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


# --------------------------------------------------------------------------- schema
def test_sources_schema(sources):
    assert tb.schema_errors(sources) == []


def test_ids_unique_and_fields(sources):
    ids = [r["id"] for rs in sources.values() for r in rs]
    assert len(ids) == len(set(ids))
    for cat, rs in sources.items():
        for r in rs:
            assert set(tb.REQUIRED[cat]) <= set(r), (cat, r["id"])


def test_merged_rows_have_one_schema(rows):
    for r in rows:
        assert tuple(r) == tb.FIELDS_OUT, r["id"]
        assert r["category"] in tb.SOURCES


def test_textbank_jsonl_is_up_to_date(rows):
    """The committed textbank.jsonl is the build of the committed sources (rerun build_textbank.py after editing)."""
    on_disk = [json.loads(l) for l in (tb.BANK / tb.OUT_NAME).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert on_disk == json.loads(json.dumps(rows, ensure_ascii=False))


def test_no_row_dropped(built):
    assert built["drops"] == []


# --------------------------------------------------------------------------- spoken forms
def test_spoken_is_trnorm_without_digits(rows):
    for r in rows:
        assert r["spoken"] == " ".join(normalize(r["text"]).split()), r["id"]
        assert not re.search(r"\d", r["spoken"]), (r["id"], r["spoken"])
        assert not re.search(r"[%₺$€£#@]", r["spoken"]), (r["id"], r["spoken"])


def test_spoken_of_rejects_digits_and_failures(monkeypatch):
    monkeypatch.setattr(tb, "normalize", lambda t: t)
    assert tb.spoken_of("Kapı 3 numara.")[1] == "digit left in spoken form"
    assert tb.spoken_of("Yüzde % elli.")[1] == "symbol left in spoken form"

    def boom(t):
        raise RuntimeError("x")
    monkeypatch.setattr(tb, "normalize", boom)
    assert tb.spoken_of("Merhaba.")[1].startswith("normalize failed")


def test_numbers_are_exercised(rows):
    share = tb.stats(rows)["numeric_share"]
    assert share["questions"] >= 0.2 and share["emotions"] >= 0.08 and share["all"] >= 0.12, share
    feats = collections.Counter(f for r in rows for f in r["features"])
    for f in ("time", "money", "phone", "date", "percent", "ordinal", "acronym", "address_email"):
        assert feats[f] >= 3, (f, feats)


# --------------------------------------------------------------------------- questions
def test_peak_word_points_at_the_right_word(rows):
    qs = [r for r in rows if r["category"] == "questions"]
    assert sum(r["peak_word"] is None for r in qs) <= 0.05 * len(qs)
    for r in qs:
        w, k = r["spoken"].split(), r["peak_word"]
        if k is None:
            continue
        assert isinstance(k, int) and 0 <= k < len(w), r["id"]
        if r["pattern"] == "peak_fall":
            assert tb.is_mi(w[k + 1]), (r["id"], w[k], w[k + 1])
        elif r["pattern"] == "wh_peak":
            assert tb.is_wh(w[k]), (r["id"], w[k])
        elif r["pattern"] == "final_rise":
            assert k == len(w) - 1, r["id"]
        if r["kind"] == "degil_mi":
            assert tb.core(w[k]) == "değil"


def test_pairs(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        if r["category"] == "questions" and r["pair_id"]:
            by[r["pair_id"]].setdefault(r["kind"], []).append(r)
    assert within(len(by), 250)
    for pid, kinds in by.items():
        assert len(kinds.get("mi_soru", [])) == 1 and len(kinds.get("duz_eslesik", [])) == 1, pid
        q, d = kinds["mi_soru"][0], kinds["duz_eslesik"][0]
        assert q["text"].endswith("?") and not d["text"].endswith("?")
        assert d["peak_word"] == q["peak_word"], pid
        if q["peak_word"] is not None:
            qw, dw = q["spoken"].split(), d["spoken"].split()
            assert [tb.core(t) for t in qw[:q["peak_word"]]] == [tb.core(t) for t in dw[:q["peak_word"]]], pid
        for k, tail in (("degil_mi", ", değil mi?"), ("yukselen", "?")):
            for r in kinds.get(k, []):
                assert r["text"] == d["text"][:-1] + tail, r["id"]
    for r in rows:
        if r["kind"] in ("mi_soru", "duz_eslesik"):
            assert r["pair_id"] in by


def test_question_shapes(rows):
    for r in (r for r in rows if r["category"] == "questions"):
        w = r["spoken"].split()
        has_mi, has_wh = any(tb.is_mi(t) for t in w), any(tb.is_wh(t) for t in w)
        if r["kind"] in ("mi_soru", "degil_mi"):
            assert has_mi, r["id"]
        if r["kind"] == "duz_eslesik":
            assert not has_mi and r["pattern"] == "fall", r["id"]
        if r["kind"] == "ne_soru":
            assert has_wh and not has_mi, r["id"]
        if r["kind"] == "yukselen":
            assert not has_mi and not has_wh, r["id"]
        assert tb.harmony_issues(r["spoken"]) == [], r["id"]
    forms = {tb.core(t) for r in rows if r["kind"] == "mi_soru" for t in r["spoken"].split() if tb.is_mi(t)}
    groups = {"3sg": r"m[ıiuü]", "2sg": r"m[ıiuü]s[ıiuü]n", "2pl": r"m[ıiuü]s[ıiuü]n[ıiuü]z", "1sg": r"m[ıiuü]y[ıiuü]m",
              "1pl": r"m[ıiuü]y[ıiuü]z", "past": r"m[ıiuü]yd[ıiuü]\w*"}
    for name, rx in groups.items():
        assert any(re.fullmatch(rx, f) for f in forms), (name, forms)
    assert {"mi", "mı", "mu", "mü"} <= forms


def test_mi_and_harmony_helpers():
    assert tb.is_mi("musunuz?") and tb.is_mi("mıydı") and tb.is_mi("miyim,") and not tb.is_mi("müdür")
    assert not tb.is_mi("mısır") and not tb.is_mi("mide")
    assert tb.harmony_issues("Yarın geliyor mi?") == ["geliyor mi?"]
    assert tb.harmony_issues("Saat mi?") == [] and tb.harmony_issues("Kasım'da mı?") == []
    assert tb.expected_peak("mi_soru", "Sen de mi geliyorsun?") is None
    assert tb.expected_peak("hizmet", "Adresinize mi yoksa şubeye mi gönderelim?") is None
    assert tb.expected_peak("ne_soru", "Kargo ne zaman gelir?") == 1
    with pytest.raises(ValueError):
        tb.duz_peak("Yarın geliyor musun?", "Bugün geliyorsun.", 1)


# --------------------------------------------------------------------------- emotions
def test_instruction_splits_disjoint(rows):
    emo = [r for r in rows if r["category"] == "emotions"]
    train = {r["instruction"] for r in emo if r["instruction_split"] == "train"}
    held = {r["instruction"] for r in emo if r["instruction_split"] == "heldout"}
    assert train and held and not train & held
    for e, ps in tb.INSTRUCTIONS.items():
        assert 8 <= len(ps) <= 12
        n_held = sum(p["split"] == "heldout" for p in ps)
        assert n_held == round(0.2 * len(ps)), e
        assert {p["text"] for p in ps if p["split"] == "heldout"} <= held
        for p in ps:
            assert "(" not in p["text"] and ")" not in p["text"]
            assert p["text"][0].isupper() and p["text"].endswith(".")
    for r in emo:
        p = tb.INSTRUCTION_BY_ID[r["instruction_id"]]
        assert (p["text"], p["split"], p["emotion"]) == (r["instruction"], r["instruction_split"], r["emotion"])
    held_rows = sum(r["instruction_split"] == "heldout" for r in emo)
    assert 0.05 * len(emo) <= held_rows <= 0.2 * len(emo)


def test_intensity_words_in_paraphrases():
    for e, ps in tb.INSTRUCTIONS.items():
        words = " ".join(p["text"].lower() for p in ps)
        assert "slightly" in words or "a little" in words or "mildly" in words, e
        assert "very" in words or "deeply" in words, e


def test_emotion_lengths_and_sad_is_quiet(rows):
    emo = [r for r in rows if r["category"] == "emotions"]
    for r in emo:
        assert tb.EMO_MIN_WORDS <= len(r["text"].split()) <= tb.EMO_MAX_WORDS, r["id"]
    sad = [r for r in emo if r["emotion"] == "sad"]
    quiet = [r for r in sad if re.search(r"quiet|slow|soft|low|whisper|weary|tired|pauses", r["instruction"])]
    assert len(quiet) >= 0.6 * len(sad)
    assert sum("..." in r["text"] for r in sad) >= 5


# --------------------------------------------------------------------------- interjections, short
def test_interjections(rows):
    ij = [r for r in rows if r["category"] == "interjections"]
    assert set(tb.INTERJECTIONS) <= {r["interjection"] for r in ij}
    vay = [r for r in ij if r["interjection"] in ("vay", "vay canına")]
    assert vay and all(r["render_variants"] for r in vay)
    for r in ij:
        assert r["span"] in r["text"]
        if r["render_variants"]:
            assert r["render_variants"][0] == r["span"]
            assert r["render_texts"][0] == r["spoken"] and len(r["render_texts"]) == len(r["render_variants"])
            assert len(set(r["render_texts"])) == len(r["render_texts"]), r["id"]  # spellings survive trnorm
        else:
            assert r["render_texts"] is None


def test_short_punctuation_variety(rows):
    marks = collections.Counter(tb.final_mark(r["text"]) for r in rows if r["category"] == "short")
    assert all(marks[m] >= 5 for m in (".", "!", "?", "…")), marks


# --------------------------------------------------------------------------- leakage
def test_no_leakage(rows, guard):
    g, protected = guard
    for r in rows:
        assert not g.hit(tb.eval_key(r["text"]), tb.eval_key(r["spoken"])), r["id"]


def test_no_shared_wording_with_protected_sentences(built):
    """Beyond the guard: no 4-word run with a word other than a number reading is shared with a protected sentence."""
    assert built["content_overlaps"] == []


def test_guard_covers_all_benchmark_sets_and_catches_leaks(guard):
    g, protected = guard
    files = {p.name for p in tb.protected_files()}
    assert {"tr_zor.jsonl", "tr_soru.jsonl", "contest.jsonl"} <= files
    for t in ("Evet.", "Yarın sabah geliyor musun?", "Toplantı saat 14:30'da başlıyor, değil mi?",
              "Kızım dedi ki: Yeter artık! Üç haftadır aynı arıza için arıyorum ve hâlâ kimse gelip bakmadı!"):
        assert g.hit(tb.eval_key(t)), t
    assert not g.hit(tb.eval_key("Evet efendim."))
    if tb.ANTALIA_HELDOUT.exists():
        assert "rows.jsonl" in files


def test_build_drops_leaks_duplicates_and_bad_spoken(guard):
    g, protected = guard
    src = {"questions": [], "emotions": [], "interjections": [], "short": [
        {"id": "a", "kind": "onay", "text": "Olur."},
        {"id": "b", "kind": "onay", "text": "Olur!"},         # same as a (statement marks are one class)
        {"id": "c", "kind": "soru", "text": "Olur mu?"},
        {"id": "d", "kind": "onay", "text": "Evet."},         # TR-Zor sentence
        {"id": "e", "kind": "soru", "text": "Yarın geliyorsun?"},
        {"id": "f", "kind": "onay", "text": "Yarın geliyorsun."},  # other intonation: kept
    ]}
    res = tb.build_bank(src, g, protected)
    assert [r["id"] for r in res["rows"]] == ["a", "c", "e", "f"]
    drops = dict(res["drops"])
    assert set(drops) == {"b", "d"} and drops["b"] == "duplicate of a" and drops["d"].startswith("leakage")


# --------------------------------------------------------------------------- counts
def test_counts_within_15_percent(rows):
    st = tb.stats(rows)
    for cat, target in tb.TARGETS.items():
        assert within(st["by_category"][cat], target), (cat, st["by_category"][cat])
    for kind, target in tb.KIND_TARGETS.items():
        assert within(st["by_kind"]["questions"].get(kind, 0), target), (kind, st["by_kind"]["questions"])
    for e in tb.EMOTIONS:
        assert within(st["by_kind"]["emotions"].get(e, 0), tb.EMOTION_TARGET), (e, st["by_kind"]["emotions"])
    for i in tb.INTERJECTIONS:
        assert st["by_kind"]["interjections"].get(i, 0) >= 8, i


def test_cli_check_writes_nothing(tmp_path, capsys):
    out = tmp_path / "tb.jsonl"
    assert tb.main(["--check", "--out", str(out)]) == 0
    assert not out.exists()
    assert "dropped 0" in capsys.readouterr().out


# --------------------------------------------------------------------------- laughter (render_corpus's trial)
def test_laughter_lines(rows, guard):
    g, _ = guard
    laugh = tb.load_laughter()
    assert 45 <= len(laugh) <= 60 and tb.laughter_errors(laugh, g, rows) == []
    assert not any(r["id"].startswith("l_") for r in rows)  # never merged into textbank.jsonl
    for r in laugh:
        assert r["laugh"] is True and tb.LAUGH_RE.fullmatch(r["laugh_span"]) and r["laugh_span"] in r["text"]
        assert tb.strip_laughter(r["text"]) and not tb.LAUGH_RE.search(tb.strip_laughter(r["text"])), r["id"]
    starts = sum(tb.LAUGH_RE.match(r["text"]) is not None for r in laugh)
    assert 0.5 * len(laugh) <= starts < len(laugh)  # mostly at the start, some in the middle or at the end


def test_laugh_re_and_strip():
    for t in ("Hahaha", "Ha ha ha", "Hehe", "Ahahaha", "Hihihi", "haha"):
        assert tb.LAUGH_RE.fullmatch(t), t
    for t in ("Ha", "Hayır", "Hah", "hadi", "Hey"):
        assert not tb.LAUGH_RE.fullmatch(t), t
    assert tb.strip_laughter("Gülmekten karnım ağrıdı, hahaha.") == "Gülmekten karnım ağrıdı."
    assert tb.strip_laughter("Ay, hahaha, yüzünün hâlini görmeliydin!") == "Ay yüzünün hâlini görmeliydin!"
    assert tb.strip_laughter("Ha ha ha, şaka yapıyorsun herhalde!") == "şaka yapıyorsun herhalde!"
    assert tb.strip_laughter("Hayır, hiç gülmedim.") == "Hayır, hiç gülmedim."


def test_laughter_errors_catch_bad_rows(guard):
    g, _ = guard
    ok = {"id": "l_900", "emotion": "cheerful", "intensity": "mild", "text": "Hahaha, çok tatlısın!", "laugh": True,
          "laugh_span": "Hahaha"}
    assert tb.laughter_errors([ok], g) == []
    bad = [{**ok, "id": "x1"}, {**ok, "id": "l_901", "laugh_span": "Ha"}, {**ok, "id": "l_902", "text": "Hahaha!",
                                                                          "laugh_span": "Hahaha"},
           {**ok, "id": "l_903", "text": "Hahaha, " + "çok " * 14 + "güzel!"}, {**ok, "id": "l_904", "laugh": False},
           {**ok, "id": "l_905"}, {**ok, "id": "l_906", "text": "Hahaha, çok tatlısın!"},
           {k: v for k, v in ok.items() if k != "laugh_span"}]
    errs = tb.laughter_errors(bad, g)
    assert any("x1: want id l_NNN" in e for e in errs) and any("l_901: laugh_span 'Ha'" in e for e in errs)
    assert any("l_902: nothing but laughter" in e for e in errs) and any("l_903: 16 words" in e for e in errs)
    assert any("l_904: want id" in e for e in errs) and any("l_906: duplicate of" in e for e in errs)
    assert any("missing ['laugh_span']" in e for e in errs)
    leak = {**ok, "id": "l_907", "text": "Hahaha, yarın sabah geliyor musun?"}  # a TR-Soru sentence inside
    assert any("leakage" in e for e in tb.laughter_errors([leak], g))
