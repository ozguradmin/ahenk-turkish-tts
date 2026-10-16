"""tts/voxcpm/prepare_human.py: text and selection filters, UTMOS batching plan, threshold table, the ISSAI tar reader,
resume / restart / score cache end to end on fake sources (stdlib only), audio processing when numpy is there."""

import io
import json
import os
import subprocess
import sys
import tarfile
import wave
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VOXCPM = REPO / "tts" / "voxcpm"
sys.path[:0] = [str(VOXCPM), str(REPO / "src")]

import prepare_alania as pa  # noqa: E402
import prepare_human as ph  # noqa: E402


def test_import_needs_no_heavy_deps():
    code = (f"import sys; sys.path[:0] = [{str(VOXCPM)!r}, {str(REPO / 'src')!r}]; import prepare_human; "
            "bad = {'pyarrow', 'numpy', 'soundfile', 'soxr', 'torch', 'torchaudio', 'scipy', 'huggingface_hub'} "
            "& set(sys.modules); assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_help_lists_interface():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run([sys.executable, str(VOXCPM / "prepare_human.py"), "--help"], capture_output=True,
                         text=True, encoding="utf-8", env=env, check=True).stdout
    for flag in ("--out", "--hours", "--sources", "--min-utmos", "--gender", "--min-dur", "--max-dur",
                 "--max-per-speaker", "--ref-fraction", "--exclude-manifest", "--cv-split", "--cv-revision",
                 "--issai-archive", "--device", "--batch-seconds", "--workers", "--seed", "--max-files",
                 "--max-issai-chunks", "--row-groups", "--restart"):
        assert flag in out, flag


def test_defaults_and_pins():
    a = ph.parse_args(["--out", "x"])
    assert (a.hours, a.min_utmos, a.gender, a.min_dur, a.max_dur, a.max_per_speaker, a.ref_fraction) == \
        (25.0, 3.3, "any", 2.0, 15.0, 60, 0.3)
    assert a.sources == ("cv",) and a.cv_split == "validated" and a.device == "auto"
    assert len(ph.CV_REVISION) == 40 and len(ph.ISSAI_REVISION) == 40
    assert "CC0" in ph.ATTRIBUTION["cv"] and "MIT" in ph.ATTRIBUTION["issai"]
    assert "Mussakhojayeva" in ph.ATTRIBUTION["issai"]
    assert ph.MARKER != pa.MARKER and "scores.jsonl" in ph.OWN_FILES


@pytest.mark.parametrize("bad", ["", "cv,foo", "mozilla", ","])
def test_sources_parse(bad):
    assert ph.parse_sources("issai, CV") == ("cv", "issai") and ph.parse_sources("issai") == ("issai",)
    with pytest.raises(ValueError):
        ph.parse_sources(bad)
    with pytest.raises(SystemExit):
        ph.parse_args(["--out", "x", "--sources", bad or ","])


def test_spoken_text_keeps_case_and_punctuation_and_rejects_the_rest():
    assert ph.spoken_text("Saat 14:30'da geldi.") == ("Saat on dört otuzda geldi.", "")
    assert ph.spoken_text("“Canavar” dedi.")[1] == "" and ph.spoken_text("erdoğan ın evi")[1] == ""
    for raw, why in (("", "text_empty"), ("...", "text_empty"), (None, "text_empty"), ("Москва", "text_charset"),
                     ("José geldi.", "text_charset"), ("x⁴", "text_digits")):
        assert ph.spoken_text(raw)[1] == why, raw


def test_norm_gender_and_cv_row():
    assert [ph.norm_gender(g) for g in ("male_masculine", "female_feminine", "", None, "intersex")] == \
        ["male", "female", None, None, None]
    r = ph.cv_row({"client_id": "ab" * 64, "path": "/x/tr_validated_0/common_voice_tr_31400937.mp3",
                   "sentence": "Evet.", "up_votes": 2, "down_votes": 0, "gender": "female_feminine", "age": "",
                   "audio": {"bytes": b"m"}})
    assert r["id"] == "common_voice_tr_31400937" and r["speaker"] == "cv_" + "ab" * 8 and r["gender"] == "female"
    assert r["audio"] == b"m" and r["age"] is None and (r["up"], r["down"]) == (2, 0)
    assert ph.cv_row({"client_id": "", "path": None, "audio": {"path": "a.mp3", "bytes": b""}})["speaker"] is None


def test_plan_batches():
    lengths = [100, 99, 98, 50, 49, 10, 100, 1000]
    b = ph.plan_batches(lengths, tol=0.02, max_samples=10_000, max_n=32)
    assert sorted(i for g in b for i in g) == list(range(len(lengths)))
    for g in b:
        assert all(lengths[i] >= 0.98 * lengths[g[0]] for i in g) and lengths[g[0]] == max(lengths[i] for i in g)
    assert [7] in b and sorted(next(g for g in b if 0 in g)) == [0, 1, 2, 6]
    assert all(len(g) <= 2 for g in ph.plan_batches([100] * 9, max_n=2))
    assert all(len(g) * 100 <= 300 for g in ph.plan_batches([100] * 9, max_samples=300))
    assert ph.plan_batches([5000], max_samples=10) == [[0]] and ph.plan_batches([]) == []


def hrow(i, text=None, spk=None, **kw):
    return {"id": f"c{i:04d}", "source": "cv", "speaker": spk if spk is not None else f"cv_s{i % 5}",
            "gender": "female", "text": text or f"Bu {ph.normalize(str(i))} numaralı cümle okundu.", "up": 2,
            "down": 0, "audio": b"", **kw}


def test_prefilter_each_reason():
    guard = pa.build_guard(["Bu cümle eğitimde asla kullanılmamalı."])
    sel = ph.Selector(ph.Filters(gender="female", max_per_speaker=2), guard, 1e9)
    sel._add("c9000", "cv_full", "dolu konuşmacı", 1.0)
    sel._add("c9001", "cv_full", "dolu konuşmacı iki", 1.0)
    sel._add("c9002", "cv_s1", ph.eval_key("Daha önce seçildi."), 1.0)
    rows = [hrow(1), hrow(2, text="Москва"), hrow(3, text="Bu cümle eğitimde asla kullanılmamalı."),
            hrow(4, up=1, down=1), hrow(5, gender=None), hrow(6, spk="cv_full"), hrow(7, text="Daha önce seçildi."),
            hrow(1), hrow(8, text="x⁴"), hrow(9, text="...")]
    rows += [hrow(10 + i, text="Aynı cümle burada.", spk=f"cv_t{i}") for i in range(4)]
    rows += [hrow(20 + i, spk="cv_many") for i in range(8)]  # 2 free slots: 3 x 2 = 6 candidates at most
    cands, drops = sel.prefilter(rows)
    assert drops == Counter({"text_charset": 1, "contamination": 1, "votes": 1, "gender": 1, "speaker_cap": 3,
                             "duplicate_text": 2, "duplicate_id": 1, "text_digits": 1, "text_empty": 1})
    assert sum(c["speaker"] == "cv_many" for c in cands) == 6
    assert sum(c["text"] == "Aynı cümle burada." for c in cands) == 3
    c = cands[0]
    assert c["id"] == "c0001" and c["text_spoken"].startswith("Bu bir numaralı")
    assert c["key"] == ph.eval_key(c["text_spoken"])
    assert c["letters"] == sum(ch.isalpha() for ch in c["text_spoken"])
    issai = ph.Selector(ph.Filters(), pa.Guard(), 1e9)  # no speaker: never capped, no gender needed with "any"
    cands, drops = issai.prefilter([hrow(i, spk="", gender=None, source="issai") for i in range(200)])
    assert len(cands) == 200 and not drops


def scored(i, utmos, dur=5.0, spk=None, text=None):
    r = hrow(i, text=text, spk=spk)
    return {**r, "text_spoken": r["text"], "key": ph.eval_key(r["text"]), "utmos": utmos, "duration": dur}


def test_take_best_first_threshold_dedup_cap_target():
    sel = ph.Selector(ph.Filters(min_utmos=3.0, max_per_speaker=2), pa.Guard(), 20.0)
    rows = [scored(1, 2.9), scored(2, 3.1, spk="a"), scored(3, 3.9, spk="a"), scored(4, 3.5, spk="a"),
            scored(5, 3.6, spk="b", text="Aynı metin."), scored(6, 3.7, spk="c", text="Aynı metin."),
            scored(7, 3.05, spk="d"), scored(8, 3.02, spk="e")]
    t = sel.take(rows)
    assert [r["id"] for r in t.picked] == ["c0003", "c0006", "c0004", "c0007"] and t.truncated
    assert t.drops == Counter({"utmos": 1, "duplicate_text": 1, "speaker_cap": 1, "target_reached": 1})
    assert t.drop_s["utmos"] == 5.0 and sel.total_s == 20.0 and sel.done()
    t = sel.take([scored(9, 4.5)])
    assert not t.picked and t.truncated and t.drops == Counter({"target_reached": 1})
    t = sel.take([scored(10, 1.0)])
    assert not t.truncated and t.drops == Counter({"utmos": 1})  # nothing passing was left


def test_simulate_and_percentiles():
    def line(i, u, spk="s", key=None, src="cv", dur=3600.0):
        return {"id": f"x{i}", "utmos": u, "spk": spk, "key": key or f"k{i}", "src": src, "dur": dur}

    units = [[line(1, 3.4), line(2, 3.1, key="dup"), line(3, 2.0)], [line(4, 3.2, spk="t", key="dup"),
             line(5, 3.6, spk=None, src="issai")]]
    assert ph.simulate(units, 3.0, cap=0) == {"clips": 3, "hours": 3.0, "speakers": 1, "cv_hours": 2.0,
                                              "issai_hours": 1.0}
    one = ph.simulate(units, 3.0, cap=1)  # x2 capped, so x4 (same text) is the first reading of it
    assert (one["clips"], one["speakers"]) == (3, 2) and ph.simulate(units, 3.5, cap=1)["hours"] == 1.0
    p = ph.percentiles([float(i) for i in range(1, 101)])
    assert (p["n"], p["min"], p["p5"], p["median"], p["p95"], p["max"]) == (100, 1.0, 6.0, 51.0, 96.0, 100.0)
    assert ph.percentiles([]) == {}


def test_settings_key_and_audio_key():
    g = pa.Guard({"a b"}, [])
    a = ph.parse_args(["--out", "x"])
    k = ph.settings_key(ph.Filters(), a, g)
    assert k == ph.settings_key(ph.Filters(), ph.parse_args(["--out", "y", "--hours", "9", "--device", "cpu",
                                                             "--max-files", "3", "--workers", "2"]), g)
    assert k != ph.settings_key(ph.Filters(min_utmos=3.0), a, g)
    assert k != ph.settings_key(ph.Filters(), ph.parse_args(["--out", "x", "--sources", "cv,issai"]), g)
    assert k != ph.settings_key(ph.Filters(), ph.parse_args(["--out", "x", "--row-groups", "1"]), g)
    assert ph.audio_key() in k and len(ph.audio_key()) == 10


# --------------------------------------------------------------------------- ISSAI tarball reader
def issai_tar(n=7, far=True) -> bytes:
    """Train pairs 0..n-1, every .txt after ALL .wav files when far; a Test pair that must be ignored."""
    members = [(f"ISSAI_TSC_218/Test/9{i}.wav", b"RIFFtest") for i in range(2)]
    members += [(f"ISSAI_TSC_218/Test/9{i}.txt", b"test metni") for i in range(2)]
    wavs = [(f"ISSAI_TSC_218/Train/{i}.wav", f"RIFF{i}".encode() * 50) for i in range(n)]
    txts = [(f"ISSAI_TSC_218/Train/{i}.txt", f"erdoğan ın {ph.normalize(str(i))} sözü".encode()) for i in range(n)]
    members += wavs + txts[::-1] if far else [m for p in zip(txts, wavs) for m in p]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for d in ("ISSAI_TSC_218", "ISSAI_TSC_218/Train"):
            ti = tarfile.TarInfo(d)
            ti.type = tarfile.DIRTYPE
            tf.addfile(ti)
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_issai_source_pairs_far_apart_in_chunks(tmp_path):
    pytest.importorskip("numpy")  # stt/prepare/common.py (rejoin_split_suffixes) imports it
    raw, opened = issai_tar(), []
    src = ph.IssaiSource(chunk=3, opener=lambda: opened.append(1) or io.BytesIO(raw), spool_dir=tmp_path)
    d0 = src.load("issai/00000")
    assert [r["id"] for r in d0.rows] == ["issai_train_6", "issai_train_5", "issai_train_4"] and not d0.eof
    r = d0.rows[0]
    assert r["audio"] == b"RIFF6" * 50 and r["text"] == "erdoğanın altı sözü" and r["speaker"] is None
    assert d0.nbytes == len(raw) and len(list(tmp_path.glob("issai_spool_*/*"))) == 4  # 0..3 still waiting
    d1, d2 = src.load("issai/00001"), src.load("issai/00002")
    assert [r["id"] for r in d1.rows] == ["issai_train_3", "issai_train_2", "issai_train_1"]
    assert [r["id"] for r in d2.rows] == ["issai_train_0"] and d2.eof and not list(tmp_path.glob("issai_spool_*/*"))
    assert src.load("issai/00003").rows == [] and len(opened) == 1
    again = src.load("issai/00001")  # an earlier unit: the stream is read again, the same pairs come
    assert [r["id"] for r in again.rows] == [r["id"] for r in d1.rows] and len(opened) == 2
    src.close()
    assert not list(tmp_path.iterdir())
    near = ph.IssaiSource(chunk=100, opener=lambda: io.BytesIO(issai_tar(far=False)), spool_dir=tmp_path)
    assert [r["id"] for r in near.load("issai/00000").rows] == [f"issai_train_{i}" for i in range(7)]
    near.close()


# --------------------------------------------------------------------------- end to end on fake sources
N_CV, PER_UNIT, N_ISSAI = 6, 20, 3


class FakeCv:
    def __init__(self, fail_on=""):
        self.fail_on, self.loads = fail_on, []

    def list_files(self):
        return [f"tr/validated/chunk-range-{i:09d}-x/validated-{i:05d}-of-00008.parquet" for i in range(N_CV)]

    def load(self, name):
        if name == self.fail_on:
            raise ConnectionError("simulated kill")
        self.loads.append(name)
        f = self.list_files().index(name)
        rows = []
        for j in range(PER_UNIT):
            k = f * 100 + j
            text = f"Dosya {ph.normalize(str(f))}, satır {ph.normalize(str(j))} okunuyor." if j != 7 else "Ortak cümle."
            rows.append({"id": f"common_voice_tr_{k}", "source": "cv", "speaker": f"cv_{k % 9:04d}",
                         "gender": "female" if j % 3 else "male", "text": text, "up": 2, "down": 0 if j != 4 else 3,
                         "age": "twenties" if j % 2 else None, "audio": f"dur={1.5 + (k % 8)}".encode()})
        return ph.UnitData(rows, 1000)


class FakeIssai:
    def __init__(self):
        self.loads = []

    def load(self, unit):
        self.loads.append(unit)
        k = ph.issai_index(unit)
        if k >= N_ISSAI:
            return ph.UnitData([], 0, True)
        rows = [{"id": f"issai_train_{k * 10 + j}", "source": "issai", "speaker": None, "gender": None,
                 "text": f"bu {ph.normalize(str(k * 10 + j))} numaralı kayıt", "audio": b"dur=4.0"} for j in range(10)]
        return ph.UnitData(rows, 500, k == N_ISSAI - 1)


def fake_process(r):
    d = float(r["audio"].decode().split("=")[1])
    if d < 2.0:
        return None, d, "too_short"
    return b"\0\0" * int(d * ph.RATE), d, ""


class FakeScorer:
    def __init__(self):
        self.calls = []

    def __call__(self, rows):
        self.calls.append([r["id"] for r in rows])
        return [None if r["id"].endswith("_13") else 2.6 + int(r["id"].rsplit("_", 1)[1]) % 13 / 10 for r in rows]


@pytest.fixture(autouse=True)
def no_retry(monkeypatch):
    monkeypatch.setattr(ph, "retry", lambda fn, *a, what="", tries=1: fn(*a))


def run(out, cv=None, issai=None, scorer=None, *extra):
    srcs = {"cv": cv or FakeCv(), **({"issai": issai} if issai else {})}
    return ph.main(["--out", str(out), "--hours", "0.03", "--workers", "3", "--max-per-speaker", "4",
                    "--min-utmos", "3.0", *(["--sources", "cv,issai"] if issai else []), *extra], sources=srcs,
                   scorer=scorer or FakeScorer(), process=fake_process)


def jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def manifest_view(out):
    return [{k: Path(v).name if k in ("audio", "ref_audio") else v for k, v in r.items()}
            for r in jsonl(out / "voxcpm_train.jsonl")]


def test_end_to_end_resume_and_cache(tmp_path):
    full = tmp_path / "full"
    assert run(full) == 0
    clips, state = jsonl(full / "clips.jsonl"), json.loads((full / "state.json").read_text(encoding="utf-8"))
    order = [state["files"][i] for i in pa.spread_order(N_CV)]
    assert [e["file"] for e in state["done"]] == order[:len(state["done"])] and state["kind"] == "human"
    assert sum(c["duration"] for c in clips) >= 108 and state["done"][-1]["truncated"]
    assert all(c["utmos"] >= 3.0 and 2.0 <= c["duration"] for c in clips)
    assert len({ph.eval_key(c["text_spoken"]) for c in clips}) == len(clips)
    assert max(Counter(c["speaker"] for c in clips).values()) <= 4
    assert all(c["license"] == "CC0-1.0" and c["text_spoken"][0].isupper() for c in clips)
    man = jsonl(full / "voxcpm_train.jsonl")
    assert len(man) == len(clips)
    assert all(set(r) <= {"audio", "text", "duration", "ref_audio", "ref_duration"} for r in man)
    by_wav = {c["wav"]: c for c in clips}
    refs = [r for r in man if "ref_audio" in r]
    assert refs and all(by_wav[Path(r["ref_audio"]).name]["speaker"] == by_wav[Path(r["audio"]).name]["speaker"]
                        and r["ref_audio"] != r["audio"] and 3 <= r["ref_duration"] <= 10 for r in refs)
    for p in (Path(r["audio"]) for r in man):
        with wave.open(str(p)) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
    assert (full / "ATTRIBUTION.txt").read_text(encoding="utf-8").splitlines() == [ph.ATTRIBUTION["cv"]]
    rep = json.loads((full / "prep_report.json").read_text(encoding="utf-8"))
    assert rep["reached"] and rep["selected"]["clips"] == len(clips) and rep["drops"]["votes"]["rows"] >= 1
    assert {"utmos", "too_short", "utmos_error"} <= set(rep["drops"])
    u = rep["utmos"]
    assert u["kept"]["n"] == len(clips) and u["candidates"]["n"] > len(clips) and u["kept"]["min"] >= 3.0
    t = {x["min_utmos"]: x for x in u["thresholds"]}
    assert {2.5, 3.0, 3.3, 3.5} <= set(t) and t[2.5]["hours"] >= t[3.0]["hours"] >= t[3.5]["hours"]
    assert t[3.0]["cv_hours_all_files"] > t[3.0]["cv_hours"]
    md = (full / "PREP_REPORT.md").read_text(encoding="utf-8")
    assert "| 3.0 (used) |" in md and "| candidates |" in md and ph.ATTRIBUTION["cv"] in md

    # killed while the 2nd unit was loading: the next call continues and ends with the same clips, scoring each once
    part = tmp_path / "part"
    with pytest.raises(ConnectionError):
        run(part, FakeCv(fail_on=order[1]))
    assert [e["file"] for e in json.loads((part / "state.json").read_text(encoding="utf-8"))["done"]] == order[:1]
    cv, sc = FakeCv(), FakeScorer()
    assert run(part, cv, None, sc) == 0 and order[0] not in cv.loads
    assert jsonl(part / "clips.jsonl") == clips and manifest_view(part) == manifest_view(full)
    scored = [i for c in sc.calls for i in c]
    assert len(scored) == len(set(scored))

    # the same call again reads nothing; more --hours redo the cut unit from the cache
    cv, sc = FakeCv(), FakeScorer()
    assert run(part, cv, None, sc) == 0 and cv.loads == [] and sc.calls == []
    cv, sc = FakeCv(), FakeScorer()
    assert run(part, cv, None, sc, "--hours", "0.06") == 0
    assert state["done"][-1]["file"] in cv.loads
    redone = set(state["done"][-1]["ids"])
    assert not redone & {i for c in sc.calls for i in c}
    more = jsonl(part / "clips.jsonl")
    assert len(more) > len(clips) and more[:len(clips) - len(redone)] == clips[:len(clips) - len(redone)]

    # another threshold needs --restart, which scores nothing again and drops the WAVs no longer selected
    assert run(part, FakeCv(), None, FakeScorer(), "--min-utmos", "3.4") == 2
    cached, sc = {x["id"] for x in jsonl(part / "scores.jsonl")}, FakeScorer()
    assert run(part, FakeCv(), None, sc, "--min-utmos", "3.4", "--restart", "--hours", "0.06") == 0
    assert sc.calls and not cached & {i for c in sc.calls for i in c}
    new = jsonl(part / "clips.jsonl")
    assert new and all(c["utmos"] >= 3.4 for c in new)
    assert {p.name for p in (part / "wav").glob("*.wav")} == {c["wav"] for c in new}
    rep = json.loads((part / "prep_report.json").read_text(encoding="utf-8"))
    assert rep["orphan_wavs_removed"] > 0 and rep["utmos"]["candidates"]["n"] == sum(
        e["candidates"] - e["drops"].get("utmos_error", 0) for e in rep_done(part))


def rep_done(out):
    return json.loads((out / "state.json").read_text(encoding="utf-8"))["done"]


def test_issai_after_cv_no_speaker_no_ref(tmp_path):
    out = tmp_path / "o"
    issai = FakeIssai()
    assert run(out, FakeCv(), issai, None, "--hours", "5", "--max-files", "2") == 0
    clips, state = jsonl(out / "clips.jsonl"), json.loads((out / "state.json").read_text(encoding="utf-8"))
    units = [e["file"] for e in state["done"]]
    assert units[:2] == [state["files"][i] for i in pa.spread_order(N_CV)[:2]]
    assert units[2:] == [ph.issai_unit(k) for k in range(N_ISSAI)] and state["issai_end"] == N_ISSAI
    iss = [c for c in clips if c["source"] == "issai"]
    assert iss and all(c["speaker"] is None and c["license"] == "MIT" and c["text_spoken"].islower() for c in iss)
    assert len(iss) > 4  # no per-speaker cap without speaker ids
    names = {c["wav"] for c in iss}
    assert not any(Path(r["audio"]).name in names for r in jsonl(out / "voxcpm_train.jsonl") if "ref_audio" in r)
    assert (out / "ATTRIBUTION.txt").read_text(encoding="utf-8").splitlines() == [ph.ATTRIBUTION["cv"],
                                                                                  ph.ATTRIBUTION["issai"]]
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    assert not rep["reached"] and rep["selected"]["by_source"]["issai"]["clips"] == len(iss)
    assert rep["selected"]["clips_without_speaker"] == len(iss) and set(rep["dataset"]["sources"]) == {"cv", "issai"}
    issai = FakeIssai()  # done: the end of the archive is remembered, nothing is read again
    assert run(out, FakeCv(), issai, None, "--hours", "5", "--max-files", "2") == 0 and issai.loads == []


def test_missing_wav_is_read_again(tmp_path):
    out = tmp_path / "o"
    assert run(out) == 0
    clips, man = jsonl(out / "clips.jsonl"), manifest_view(out)
    victim = clips[0]
    (out / "wav" / victim["wav"]).unlink()
    cv = FakeCv()
    assert run(out, cv) == 0 and cv.loads == [victim["unit"]]
    assert (out / "wav" / victim["wav"]).is_file() and manifest_view(out) == man and jsonl(out / "clips.jsonl") == clips


def test_foreign_out_and_bad_exclude_manifest(tmp_path):
    alania = tmp_path / "alania16k"  # prepare_alania's output is another corpus for this script
    (alania / "wav").mkdir(parents=True)
    (alania / pa.MARKER).write_text("x", encoding="utf-8")
    (alania / "clips.jsonl").write_text("{}\n", encoding="utf-8")
    (alania / "wav" / "a.wav").write_bytes(b"RIFF")
    for extra in ((), ("--restart",)):
        assert run(alania, None, None, None, *extra) == 2 and (alania / "wav" / "a.wav").exists()
    assert ph.foreign_files(alania) == ["clips.jsonl", "wav/*.wav"]
    assert run(tmp_path / "a", None, None, None, "--exclude-manifest", str(tmp_path / "nope.jsonl")) == 2
    val = tmp_path / "fresh" / "voxcpm_val.jsonl"
    val.parent.mkdir()
    val.write_text('{"audio": "wav/v.wav", "text": "Merhaba dünya nasılsın bugün."}\n', encoding="utf-8")
    assert run(val.parent, None, None, None, "--exclude-manifest", str(val)) == 2
    own = tmp_path / "own"
    assert run(own) == 0 and ph.foreign_files(own) == []
    (own / ph.MARKER).unlink()
    assert ph.foreign_files(own) == []  # its state.json says kind "human"


def test_contamination_guard_drops_protected_sentences(tmp_path):
    val = tmp_path / "val.jsonl"
    val.write_text('{"audio": "x.wav", "text": "Ortak cümle."}\n', encoding="utf-8")
    out = tmp_path / "o"
    assert run(out, None, None, None, "--exclude-manifest", str(val)) == 0
    assert not any(c["text"] == "Ortak cümle." for c in jsonl(out / "clips.jsonl"))
    assert json.loads((out / "prep_report.json").read_text(encoding="utf-8"))["drops"]["contamination"]["rows"] >= 1


# --------------------------------------------------------------------------- audio (numpy, soundfile)
def tone(np, seconds, amp, rate=16000):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def wav_bytes(np, x, rate=16000):
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, x, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_trim_and_level():
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    rng = np.random.default_rng(0)
    x = np.concatenate([1e-4 * rng.standard_normal(16000).astype(np.float32), tone(np, 3.0, 0.05),
                        1e-4 * rng.standard_normal(8000).astype(np.float32)])
    a, b, level = ph.speech_bounds(x)
    assert abs(a / 16000 - (1.0 - ph.TRIM_PAD_S)) < 0.03 and abs(b / 16000 - (4.0 + ph.TRIM_PAD_S)) < 0.03
    assert abs(level - 20 * np.log10(0.05 / np.sqrt(2))) < 0.5
    y, gain = ph.level_normalize(x[a:b], level)
    assert abs(gain - (-20 - level)) < 1e-6 and float(np.abs(y).max()) <= 10 ** (-1 / 20) + 1e-6
    pcm, secs, why = ph.process_audio(wav_bytes(np, x), letters=40, f=ph.Filters())
    assert why == "" and abs(secs - 3.3) < 0.05 and len(pcm) == 2 * round(secs * 16000)
    out = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768
    _, _, out_level = ph.speech_bounds(out)
    assert abs(out_level - (-20.0)) < 0.5
    f = ph.Filters()
    assert ph.process_audio(b"not audio", 10, f)[2] == "decode_error"
    assert ph.process_audio(wav_bytes(np, tone(np, 1.0, 0.3)), 10, f)[2] == "too_short"
    assert ph.process_audio(wav_bytes(np, tone(np, 16.0, 0.3)), 200, f)[2] == "too_long"
    assert ph.process_audio(wav_bytes(np, tone(np, 3.0, 0.3)), 200, f)[2] == "speech_rate"
    assert ph.process_audio(wav_bytes(np, tone(np, 3.0, 1e-3)), 40, f)[2] == "too_quiet"
    assert ph.process_audio(wav_bytes(np, np.zeros(48000, np.float32)), 40, f)[2] == "too_quiet"
    assert ph.process_audio(wav_bytes(np, np.clip(tone(np, 3.0, 3.0), -1, 1)), 40, f)[2] == "clipped"
