"""Voice contest data: benchmarks/tts/sets/contest.jsonl (the texts), tts/voxcpm/voice_descriptions.jsonl (VoxCPM2
voice design), tts/voxcpm/fetch_refs.py (registry, candidate selection, measures; stdlib + trnorm, the audio measures
need numpy + soundfile and are skipped without them; no network) and tts/voxcpm/CONTEST_SOURCES.md.

contest.jsonl rows: id (unique), set (trzor | soru | emo), category, text (written form), refs (spoken forms for WER);
soru rows also pair_id, pattern, peak, peak_word (copied from tr_soru.jsonl); emo rows also emotion (= category) and
instruction (English, no parentheses: the model gets "(" + instruction + ")" + trnorm.normalize(text))."""

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VOXCPM = REPO / "tts" / "voxcpm"
sys.path[:0] = [str(VOXCPM), str(REPO / "src")]

import fetch_refs as fr  # noqa: E402
import score_intonation as si  # noqa: E402
import synth_eval  # noqa: E402
from trnorm import normalize  # noqa: E402

SETS = REPO / "benchmarks" / "tts" / "sets"
CONTEST_FILE, VOICES_FILE = SETS / "contest.jsonl", VOXCPM / "voice_descriptions.jsonl"
EMOTIONS = ("angry", "cheerful", "sad", "surprised", "calm", "empathetic")


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


CONTEST, VOICES = load(CONTEST_FILE), load(VOICES_FILE)
TRZOR = {r["id"]: r for r in load(SETS / "tr_zor.jsonl")}
SORU = {r["id"]: r for r in load(SETS / "tr_soru.jsonl")}


def rows(s: str) -> list[dict]:
    return [r for r in CONTEST if r["set"] == s]


@pytest.mark.parametrize("path", [CONTEST_FILE, VOICES_FILE, VOXCPM / "fetch_refs.py", VOXCPM / "CONTEST_SOURCES.md"])
def test_files_are_utf8_lf(path):
    raw = path.read_bytes()
    raw.decode("utf-8")
    assert b"\r" not in raw and raw.endswith(b"\n")


# ---- contest.jsonl ----------------------------------------------------------------------------------------------


def test_contest_size_ids_and_sets():
    assert 60 <= len(CONTEST) <= 80
    assert len({r["id"] for r in CONTEST}) == len(CONTEST)
    assert Counter(r["set"] for r in CONTEST) == {"trzor": 40, "soru": 15, "emo": 15}
    order = [r["set"] for r in CONTEST]
    assert order == sorted(order, key=["trzor", "soru", "emo"].index)  # grouped by set


def test_contest_fields():
    keys = {"trzor": {"id", "set", "category", "text", "refs"},
            "soru": {"id", "set", "category", "text", "refs", "pair_id", "pattern", "peak", "peak_word"},
            "emo": {"id", "set", "category", "emotion", "instruction", "text", "refs"}}
    for r in CONTEST:
        assert set(r) == keys[r["set"]], r["id"]
        assert r["text"] == r["text"].strip() and r["text"], r["id"]
        assert isinstance(r["refs"], list) and r["refs"] and all(x == x.strip() and x for x in r["refs"]), r["id"]


def test_contest_spoken_forms_have_no_digits():
    for r in CONTEST:
        assert not re.search(r"\d", normalize(r["text"])), r["id"]  # what the model reads
        assert not any(re.search(r"\d", x) for x in r["refs"]), r["id"]


def test_trzor_rows_are_the_stratified_40():
    want = synth_eval.stratified(list(TRZOR.values()), 40)  # = synth_eval --limit 40
    assert [r["id"] for r in rows("trzor")] == [r["id"] for r in want]
    # contains the --limit 30 sample, though not as its first 30 rows (voice_contest --limit 30 re-stratifies to it)
    lim30 = {r["id"] for r in synth_eval.stratified(list(TRZOR.values()), 30)}
    assert lim30 <= {r["id"] for r in rows("trzor")}
    assert {r["id"] for r in synth_eval.stratified(rows("trzor"), 30)} == lim30
    for r in rows("trzor"):
        src = TRZOR[r["id"]]
        assert (r["text"], r["category"], r["refs"]) == (src["text"], src["category"], [src["spoken"]] + src["alts"])
    cats = Counter(r["category"] for r in rows("trzor"))
    assert set(cats) == {r["category"] for r in TRZOR.values()} and len(cats) == 13 and min(cats.values()) >= 3


def test_soru_rows_are_copied_and_scorable():
    for r in rows("soru"):
        src = SORU[r["id"]]
        for k in ("category", "text", "pair_id", "pattern", "peak", "peak_word"):
            assert r[k] == src[k], (r["id"], k)
        assert r["refs"] == [src["spoken"]] and r["pattern"] == si.PATTERNS[r["category"]]
        words = normalize(r["text"]).split()  # the model text of a soru row, the words score_intonation indexes
        assert si.set_item({**r}, words)["target"] == r["peak"], r["id"]


def test_soru_rows_cover_patterns_and_pairs():
    soru = rows("soru")
    cats = Counter(r["category"] for r in soru)
    assert set(cats) == set(si.CATEGORIES)  # every question pattern + the declaratives
    assert cats["mi_soru"] >= 4 and cats["duz_eslesik"] >= 4
    by_pair = {}
    for r in soru:
        if r["pair_id"]:
            by_pair.setdefault(r["pair_id"], set()).add(r["category"])
    assert all("duz_eslesik" in c for c in by_pair.values())  # every yes/no question has its declarative
    assert sum({"mi_soru", "duz_eslesik"} <= c for c in by_pair.values()) >= 4


def test_soru_rows_feed_intonation_auc():
    """intonation_summary gives an AUC and pair accuracy on this set: questions peaking 2 st above the declaratives
    score 1.0."""
    contest = []
    for r in rows("soru"):
        q = r["category"] != "duz_eslesik"
        it = {"category": r["category"], "pattern": r["pattern"], "pass": True, "peak_st": 4.0 if q else 2.0,
              "drop_st": 3.0, "end_st": -1.0, "slope_st_s": 0.0, "peak_at": None, "timing": "whisper"}
        if r["pair_id"]:
            it["pair_id"] = r["pair_id"]
        contest.append({"set": "soru", "id": r["id"], "intonation": [it]})
    s = si.intonation_summary(contest)["soru"]
    pairs = len({r["pair_id"] for r in rows("soru") if r["category"] == "mi_soru"})
    assert s["auc_peak"] == 1.0 and s["pair_acc"] == 1.0 and s["pairs"] == pairs >= 4


def test_emo_rows():
    emo = rows("emo")
    cats = Counter(r["emotion"] for r in emo)
    assert set(cats) == set(EMOTIONS) and all(2 <= n <= 3 for n in cats.values())
    others = {r["text"] for r in TRZOR.values()} | {r["text"] for r in SORU.values()}
    n = Counter()
    for r in emo:
        n[r["emotion"]] += 1
        assert r["id"] == f"emo_{r['emotion']}_{n[r['emotion']]:02d}" and r["category"] == r["emotion"]
        ins = r["instruction"]
        assert ins.isascii() and "(" not in ins and ")" not in ins and ins.endswith("."), r["id"]
        assert 3 <= len(ins.split()) <= 14 and ins[0].isupper(), r["id"]
        assert r["text"] not in others and r["text"][-1] in ".!?", r["id"]
        assert 8 <= len(r["text"].split()) <= 20 and "(" not in r["text"], r["id"]
        assert r["refs"] == [normalize(r["text"])], r["id"]
    assert len({r["text"] for r in emo}) == len(emo) and len({r["instruction"] for r in emo}) == len(emo)
    assert all(re.search(r"[çğıöşüÇĞİÖŞÜâ]", r["text"]) for r in emo)  # Turkish sentences


def test_emo_prompt_is_instruction_plus_spoken_form():
    for r in rows("emo"):
        prompt = f"({r['instruction']})" + normalize(r["text"])
        assert prompt.startswith("(") and prompt.count("(") == 1 and prompt.endswith(r["refs"][0])


# ---- voice_descriptions.jsonl -----------------------------------------------------------------------------------

GOOD_RECORDING = ("clean broadcast-quality recording", "natural sound in a small quiet room",
                  "recorded close to the microphone in a quiet studio")
AGE_WORDS = {"20s": "twenties", "30s": "thirties", "40s": "forties", "50s": "fifties"}


def test_voice_descriptions_schema():
    assert 130 <= len(VOICES) <= 170
    assert len({v["id"] for v in VOICES}) == len(VOICES) and len({v["text"] for v in VOICES}) == len(VOICES)
    for v in VOICES:
        assert set(v) == {"id", "gender", "age", "text"}, v
        assert v["gender"] in ("female", "male") and re.fullmatch(rf"{v['gender'][0]}\d{{3}}", v["id"]), v["id"]
        assert v["age"] in AGE_WORDS and AGE_WORDS[v["age"]] in v["text"], v["id"]
        t = v["text"]
        assert t.isascii() and "(" not in t and ")" not in t and t.endswith(".") and 80 <= len(t) <= 300, v["id"]
        female = re.search(r"\b(woman|female|actress|she|her)\b", t, re.I)
        male = re.search(r"\b(man|male|his|he)\b", t, re.I)
        assert bool(female) == (v["gender"] == "female") and bool(male) == (v["gender"] == "male"), v["id"]
    share = sum(v["gender"] == "female" for v in VOICES) / len(VOICES)
    assert 0.65 <= share <= 0.75
    for g in ("female", "male"):
        ages = Counter(v["age"] for v in VOICES if v["gender"] == g)
        assert set(ages) == set(AGE_WORDS) and min(ages.values()) >= 0.15 * sum(ages.values()), (g, ages)


def test_voice_descriptions_follow_the_alania_stats():
    """Attributes whose Alania voices scored lower UTMOS stay out; recording and pace phrases of the better ones."""
    bad = re.compile(r"podcast|brisk|elderly|fast|quick|radio host|nasal|raspy|crisp|authoritative|energetic|"
                     r"enthusiastic|matter-of-fact|sixties|seventies", re.I)
    for v in VOICES:
        assert not bad.search(v["text"]), v["id"]
        assert any(rec in v["text"] for rec in GOOD_RECORDING), v["id"]
        assert re.search(r"\b(slow|measured|moderate|unhurried) pace\b", v["text"]), v["id"]
    slow = sum("slow pace" in v["text"] or "unhurried pace" in v["text"] for v in VOICES) / len(VOICES)
    assert slow >= 0.3
    # identity -> texture -> delivery (VoxCPM cookbook): the voice comes before the pace, the pace before the recording
    for v in VOICES:
        t = v["text"]
        texture = re.search(r"voice|pitched|medium-low|medium-high|deep", t).start()
        assert texture < t.index(" pace") < max(t.index(r) for r in GOOD_RECORDING if r in t), v["id"]


def test_voice_descriptions_no_real_people():
    allowed = {"Istanbul", "Turkish"}
    for v in VOICES:
        t = v["text"]
        caps = {w for i, w in enumerate(re.findall(r"[A-Za-z][\w-]*", t)) if w[0].isupper() and i > 0}
        caps -= {w for w in caps if re.search(rf"[.;]\s+{w}\b", t)}  # sentence starts
        assert caps <= allowed, (v["id"], caps)
        assert not re.search(r"imitat|impersonat|celebrit|famous|sounds like|in the style of", t, re.I), v["id"]


def test_voice_descriptions_mix():
    texts = [v["text"] for v in VOICES]
    assert sum(bool(re.search(r"storytell|narrat|audiobook", t)) for t in texts) >= 8  # a few expressive storytellers
    accent = sum("Istanbul Turkish" in t for t in texts) / len(texts)
    assert 0.2 <= accent <= 0.45  # an A/B of the accent phrase inside the screen
    for rec in GOOD_RECORDING:
        assert sum(rec in t for t in texts) >= 0.2 * len(texts)


# ---- fetch_refs.py ----------------------------------------------------------------------------------------------


def test_fetch_refs_import_needs_no_heavy_deps():
    code = (f"import sys; sys.path[:0] = [{str(VOXCPM)!r}, {str(REPO / 'src')!r}]; import fetch_refs; "
            "bad = {'numpy', 'soundfile', 'pyarrow', 'torch', 'torchaudio', 'huggingface_hub'} & set(sys.modules); "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_fetch_refs_help_and_list_offline():
    env_run = dict(capture_output=True, text=True, encoding="utf-8", check=True,
                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    out = subprocess.run([sys.executable, str(VOXCPM / "fetch_refs.py"), "--help"], **env_run).stdout
    for flag in ("--out", "--only", "--candidates", "--utmos", "--seed", "--force", "--list", "--cache"):
        assert flag in out, flag
    out = subprocess.run([sys.executable, str(VOXCPM / "fetch_refs.py"), "--list"], **env_run).stdout
    for name in fr.ENTRIES:
        assert f"{name}:" in out
    for src in fr.SOURCES.values():
        assert fr.license_url(src) in out


def test_registry():
    assert set(fr.ENTRIES) == {"masha", "dima", "kazakh_f1", "kazakh_m1", "thorsten"}
    for name, e in fr.ENTRIES.items():
        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", name) and name not in ("screen", "refs")
        assert e["source"] in fr.SOURCES and e["gender"] in ("female", "male") and "neutral" in e["clips"]
        assert set(e["clips"]) <= {"neutral", "expressive"}
        lo, hi = e["clips"]["neutral"]["dur"]
        assert (lo, hi) == (6.0, 12.0), name
        rec = fr.entry_record(name)
        for k in ("dataset", "revision", "license", "license_url", "consent", "consent_source", "attribution",
                  "obligations", "speaker", "use"):
            assert rec.get(k), (name, k)
        assert re.fullmatch(r"[0-9a-f]{40}", rec["revision"]) and rec["license_url"].startswith("https://")
        assert "reference_wav_path" in rec["use"] and "never" in rec["use"]
    assert {"masha", "thorsten"} <= {n for n, e in fr.ENTRIES.items() if "expressive" in e["clips"]}


def meta_row(path, spk="M", emo="neutral", dur=8.0, text="Это обычное предложение для проверки отбора клипов."):
    return {"audio_path": path, "speaker_id": spk, "emotion": emo, "duration": str(dur), "text": text}


def test_dialogs_pick_filters_and_orders():
    rows_ = [meta_row("wavs/a.wav"), meta_row("wavs/b.wav", dur=6.5), meta_row("wavs/c.wav", spk="D"),
             meta_row("wavs/d.wav", emo="happy"), meta_row("wavs/e.wav", dur=13.0), meta_row("wavs/f.wav", dur=5.0),
             meta_row("wavs/g.wav", text="Ха-ха, это смешно, правда очень смешно и долго."),
             meta_row("wavs/h.wav", text="Сорок один, 42, сорок три, сорок четыре и сорок пять."),
             meta_row("wavs/i.wav", dur=9.0), meta_row("wavs/missing.wav", dur=9.5), meta_row("wavs/k.wav", dur=11.5)]
    files = {r["audio_path"] for r in rows_} - {"wavs/missing.wav"}
    spec = {"speaker": "M", "style": "neutral", "dur": (6.0, 12.0)}
    got = [r["audio_path"] for r in fr.dialogs_pick(rows_, spec, 10, 0, files)]
    assert set(got) == {"wavs/a.wav", "wavs/b.wav", "wavs/i.wav", "wavs/k.wav"}
    assert set(got[:2]) == {"wavs/a.wav", "wavs/i.wav"}  # 7-11 s first
    assert got == [r["audio_path"] for r in fr.dialogs_pick(rows_, spec, 10, 0, files)]  # deterministic
    assert len(fr.dialogs_pick(rows_, spec, 2, 0, files)) == 2
    assert "wavs/missing.wav" in [r["audio_path"] for r in fr.dialogs_pick(rows_, spec, 10, 0, None)]


def test_parquet_eligible_and_best_row_group():
    kspec = fr.ENTRIES["kazakh_f1"]["clips"]["neutral"]
    k_meta = [{"text": "а" * 120, "audio": {"path": "inform_1_00.wav"}},
              {"text": "а" * 50, "audio": {"path": "inform_1_01.wav"}},  # too short
              {"text": "а" * 120, "audio": {"path": "wiki_1_00.wav"}},  # not news
              {"text": "а" * 100 + " 2014", "audio": {"path": "inform_1_02.wav"}},  # digits
              {"text": "б" * 150, "audio": {"path": "inform_1_03.wav"}}]
    assert fr.parquet_eligible("kazakhtts2", kspec, k_meta) == [0, 4]
    tspec = fr.ENTRIES["thorsten"]["clips"]["neutral"]
    t = {"style": "neutral", "text": "ein ganz normaler satz.", "durationSeconds": 6.5,
         "microphone": "good_rodePodcaster"}
    t_meta = [t, {**t, "durationSeconds": 5.9}, {**t, "microphone": "bad_usbHeadset"},
              {**t, "style": "angry | wütend"}, t]
    assert fr.parquet_eligible("thorsten", tspec, t_meta) == [0, 4]
    found = {("a", 0): [5, 40, 80], ("a", 1): [2, 9], ("b", 0): [1, 3, 30], ("b", 1): [0]}
    assert fr.best_row_group(found, 3) == (("b", 0), [1, 3, 30])  # most rows, then the earliest last one
    assert fr.best_row_group(found, 2) == (("b", 0), [1, 3])
    assert fr.best_row_group({}, 3) is None


def test_issues_and_rank():
    m = {"duration_s": 8.0, "clipped": 0.0, "floor_db": -45.0, "tail_db": -50.0}
    assert fr.issues(m, (6.0, 12.0)) == []
    assert len(fr.issues({**m, "duration_s": 13.0}, (6.0, 12.0))) == 1
    assert "clipped" in fr.issues({**m, "clipped": 0.01}, (6.0, 12.0))[0]
    assert "floor" in fr.issues({**m, "floor_db": -10.0}, (6.0, 12.0))[0]
    assert "mid-word" in fr.issues({**m, "tail_db": -3.0}, (6.0, 12.0))[0]
    a = {**m, "issues": [], "utmos": 3.2}
    b = {**m, "issues": [], "utmos": 3.6, "floor_db": -30.0}
    c = {**m, "issues": ["clipped 1%"], "utmos": 4.2}
    assert max([a, b, c], key=fr.rank_key) is b  # a fault loses to any clean clip, then UTMOS
    d, e = {**m, "issues": [], "utmos": None, "floor_db": -40.0}, {**m, "issues": [], "utmos": None, "floor_db": -60.0}
    assert max([d, e], key=fr.rank_key) is e  # without UTMOS: the lower floor


def test_measure_on_synthetic_audio():
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    import io

    sr = 22050
    t = np.arange(int(0.8 * sr)) / sr
    word = (0.3 * np.sin(2 * np.pi * 180 * t) * np.hanning(len(t))).astype(np.float32)
    pause = np.zeros(int(0.4 * sr), np.float32) + np.float32(1e-4) * np.random.default_rng(0).standard_normal(
        int(0.4 * sr)).astype(np.float32)
    clean = np.concatenate([pause] + [np.concatenate([word, pause]) for _ in range(6)])

    def wav(x):
        b = io.BytesIO()
        sf.write(b, x, sr, format="WAV", subtype="PCM_16")
        return b.getvalue()

    m = fr.measure(wav(clean))
    assert m["sample_rate"] == sr and m["channels"] == 1 and m["subtype"] == "PCM_16"
    assert abs(m["duration_s"] - len(clean) / sr) < 1e-3 and m["clipped"] == 0.0
    assert m["floor_db"] < -40 and m["tail_db"] < -40 and fr.issues(m, (6.0, 12.0)) == []
    cut = fr.measure(wav(np.concatenate([pause, word, word[: len(word) // 2]] * 3)))
    assert any("mid-word" in x for x in fr.issues(cut, (0.0, 12.0)))
    noisy = fr.measure(wav(clean + np.float32(0.05) * np.random.default_rng(1).standard_normal(len(clean)).astype(
        np.float32)))
    assert noisy["floor_db"] > m["floor_db"] + 20


def test_is_current(tmp_path):
    d = tmp_path / "masha"
    d.mkdir()
    (d / "neutral.wav").write_bytes(b"RIFF0")
    rec = {"clips": {"neutral": {"path": "neutral.wav", "sha256": fr.sha256(b"RIFF0")}}, "missing": ["expressive"]}
    (d / "sources.json").write_text(json.dumps(rec), encoding="utf-8")
    assert fr.is_current(d, "masha")  # the optional expressive clip was not found: still current
    (d / "sources.json").write_text(json.dumps({**rec, "missing": []}), encoding="utf-8")
    assert not fr.is_current(d, "masha")
    (d / "sources.json").write_text(json.dumps(rec), encoding="utf-8")
    (d / "neutral.wav").write_bytes(b"RIFF1")
    assert not fr.is_current(d, "masha") and not fr.is_current(tmp_path / "nothing", "masha")


def test_drop_stale(tmp_path):
    for f in ("neutral.wav", "neutral.flac", "expressive.wav", "sources.json"):
        (tmp_path / f).write_bytes(b"x")
    fr.drop_stale(tmp_path, "neutral", tmp_path / "neutral.wav")
    fr.drop_stale(tmp_path, "expressive")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["neutral.wav", "sources.json"]


def _offline(*a, **k):
    raise AssertionError("network")


def test_fetch_entry_is_whole_or_nothing(tmp_path, monkeypatch):
    """Candidates that cannot be downloaded fail the entry (exit 1) and leave its folder as it was, so the next run
    fetches them; only a clip without any candidate in the metadata is recorded as missing. (The clip bytes are not
    audio here: kept anyway as the only candidates, flagged unreadable.)"""
    fetch = {"neutral": lambda: b"RIFF neutral 1", "happy": lambda: b"RIFF happy 1"}
    have = {"neutral": True, "happy": True}

    def fake(spec, k, seed, cache):
        return [{"file": f"wavs/{spec['style']}_{i}.wav", "url": "u", "text": "t", "style": spec["style"],
                 "fetch": fetch[spec["style"]]} for i in range(2 if have[spec["style"]] else 0)]

    def down():
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(fr, "dialogs_candidates", fake)
    monkeypatch.setattr(fr, "_get", _offline)
    d, argv = tmp_path / "masha", ["--out", str(tmp_path), "--only", "masha", "--utmos", "off"]
    snap = lambda: {p.name: p.read_bytes() for p in d.iterdir()}  # noqa: E731

    fetch["happy"] = down  # first run: the expressive clip's downloads fail, nothing is written
    assert fr.main(argv) == 1 and not (d / "sources.json").exists() and not list(d.glob("*.wav"))
    assert not (d / ".candidates").exists()
    fetch["happy"] = lambda: b"RIFF happy 1"  # network back: the next plain run fetches the entry
    assert fr.main(argv) == 0 and fr.is_current(d, "masha")
    rec = json.loads((d / "sources.json").read_text(encoding="utf-8"))
    assert set(rec["clips"]) == {"neutral", "expressive"} and rec["missing"] == []
    assert (d / "expressive.wav").read_bytes() == b"RIFF happy 1"
    assert "unreadable" in rec["clips"]["neutral"]["issues"][0] and "unreadable" in rec["clips"]["neutral"]["selection"]

    old = snap()  # a --force re-fetch that gets a new neutral clip, then fails: old clips and record stay together
    fetch["neutral"], fetch["happy"] = (lambda: b"RIFF neutral 2"), down
    assert fr.main(argv + ["--force"]) == 1 and snap() == old and fr.is_current(d, "masha")

    have["happy"] = False  # the metadata has no expressive candidate: recorded missing, the old clip is dropped
    assert fr.main(argv + ["--force"]) == 0 and fr.is_current(d, "masha")
    rec = json.loads((d / "sources.json").read_text(encoding="utf-8"))
    assert rec["missing"] == ["expressive"] and sorted(snap()) == ["neutral.wav", "sources.json"]
    assert (d / "neutral.wav").read_bytes() == b"RIFF neutral 2"


def test_current_entry_gets_the_registry_fields_without_a_download(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "_get", _offline)
    d = tmp_path / "kazakh_f1"
    d.mkdir()
    (d / "neutral.wav").write_bytes(b"RIFF0")
    clips = {"neutral": {"path": "neutral.wav", "file": "f1/x.parquet#row=0", "duration_s": 8.0,
                         "sha256": fr.sha256(b"RIFF0")}}
    rec = {**fr.entry_record("kazakh_f1"), "attribution": "old text", "clips": clips, "missing": []}
    (d / "sources.json").write_text(json.dumps(rec), encoding="utf-8")
    assert fr.main(["--out", str(tmp_path), "--only", "kazakh_f1", "--utmos", "off"]) == 0
    new = json.loads((d / "sources.json").read_text(encoding="utf-8"))
    assert new["attribution"] == fr.SOURCES["kazakhtts2"]["attribution"] and new["clips"] == clips
    assert "github.com/IS2AI/Kazakh_TTS" in new["attribution"]  # CC BY 4.0 3(a)(1)(A)(v): a link to the material


def test_utmos_auto_never_downloads_the_weights(tmp_path, monkeypatch):
    import score_utmos

    def load(*a, **k):
        raise RuntimeError("would download")

    monkeypatch.setattr(fr, "utmos_weights", lambda: tmp_path / fr.UTMOS_CKPT)
    monkeypatch.setattr(score_utmos, "load_predictor", load)
    u = fr.Utmos("auto")
    assert u.score(tmp_path / "a.wav") is None and "not cached" in u.note and u.score(tmp_path / "a.wav") is None
    with pytest.raises(RuntimeError, match="would download"):
        fr.Utmos("on").score(tmp_path / "a.wav")
    assert fr.Utmos("off").score(tmp_path / "a.wav") is None


def test_token_never_follows_a_redirect_to_another_host():
    h = fr._NoAuthAcrossHosts()
    req = urllib.request.Request("https://huggingface.co/datasets/x/resolve/main/a.parquet",
                                 headers={"Authorization": "Bearer t", "Range": "bytes=0-0"})
    other = h.redirect_request(req, None, 302, "Found", {}, "https://cdn-lfs.example.org/a?sig=1")
    same = h.redirect_request(req, None, 302, "Found", {}, "https://huggingface.co/api/resolve-cache/a")
    assert not other.has_header("Authorization") and other.get_header("Range") == "bytes=0-0"
    assert same.get_header("Authorization") == "Bearer t"


# ---- CONTEST_SOURCES.md -----------------------------------------------------------------------------------------


def test_contest_sources_md_covers_every_entry():
    md = (VOXCPM / "CONTEST_SOURCES.md").read_text(encoding="utf-8")
    for name in fr.ENTRIES:
        assert f"`{name}`" in md, name
    for src in fr.SOURCES.values():
        assert fr.license_url(src) in md and src["revision"][:12] in md
    for word in ("consent", "Attribution", "OpenRAIL", "CC BY 4.0", "CC0", "Apache-2.0", "reference_wav_path"):
        assert word in md, word
