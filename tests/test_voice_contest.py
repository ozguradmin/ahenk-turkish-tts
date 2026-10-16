"""tts/voxcpm/voice_contest.py: entry specs, texts, seeds, error rates, cost model, job order, screen ranking,
summaries, and the synthesis pool / render / screen flows with the fake backend (no torch, no voxcpm, no
faster-whisper; the Nano-vLLM pool's tests level clips and prepare references: numpy, soundfile, librosa, else
skipped)."""

import asyncio
import json
import math
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VOXCPM = REPO / "tts" / "voxcpm"
sys.path[:0] = [str(VOXCPM), str(REPO / "src")]

import voice_contest as vc  # noqa: E402
from trnorm import normalize  # noqa: E402


def _wav(p: Path, seconds: float = 0.5) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x01\x00" * int(16000 * seconds))
    return p


class FakeFS:
    """resolve_ref's view of a dict of files (path -> text)."""

    def __init__(self, files: dict[str, str]):
        self.files = files

    def kind(self, p):
        p = p.rstrip("/")
        return "file" if p in self.files else "dir" if any(k.startswith(p + "/") for k in self.files) else None

    def listdir(self, p):
        p = p.rstrip("/") + "/"
        return sorted({k[len(p):].split("/")[0] for k in self.files if k.startswith(p)})

    def read_text(self, p):
        return self.files.get(p)


# ---- import, CLI --------------------------------------------------------------------------------------------------


def test_import_needs_no_heavy_deps():
    code = (f"import sys; sys.path[:0] = [{str(VOXCPM)!r}, {str(REPO / 'src')!r}]; import voice_contest; "
            "bad = {'torch', 'torchaudio', 'numpy', 'voxcpm', 'soundfile', 'librosa', 'faster_whisper', 'speechbrain',"
            " 'modal'} & set(sys.modules); assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_parser_accepts_the_modal_args_and_bounds_workers():
    ns = vc.build_parser().parse_args(["render", "--out", "x", "--timesteps", "16", "--only", "a,b", "--workers", "6"])
    assert (ns.timesteps, ns.only, ns.workers, ns.backend, ns.concurrency) == (16, "a,b", 6, "nanovllm", 64)
    assert vc.build_parser().parse_args(["screen", "--out", "x", "--backend", "voxcpm"]).backend == "voxcpm"
    with pytest.raises(SystemExit):
        vc.build_parser().parse_args(["render", "--out", "x", "--concurrency", "0"])
    ns = vc.build_parser().parse_args(["screen", "--out", "x", "--limit", "10", "--seeds", "0,1,2"])
    assert ns.top == 30 and ns.keep == 5 and ns.need_cer0 == 3 and vc.parse_ints(ns.seeds) == [0, 1, 2]
    with pytest.raises(SystemExit):
        vc.build_parser().parse_args(["render", "--out", "x", "--workers", "9"])
    for opt in ("--out", "--model", "--deadline", "--spec", "--throughput-log", "--backend"):
        assert opt in vc.WRAPPER_OPTS  # modal_app owns these


# ---- specs --------------------------------------------------------------------------------------------------------


def test_parse_entries_valid_and_kinds():
    es = vc.parse_entries({"entries": [{"name": "a", "kind": "ref", "ref": "x.wav", "notes": "n"},
                                       {"name": "b+s1", "kind": "designed", "ref": "y.wav", "lora": "/l"},
                                       {"name": "c", "kind": "lora", "lora": "/l", "source": {"url": "u"}},
                                       {"name": "d", "kind": "ref", "ref": "x.wav", "ref_prep": False}]})
    assert [e["name"] for e in es] == ["a", "b+s1", "c", "d"] and es[2]["source"] == {"url": "u"}
    assert es[3]["ref_prep"] is False and "ref_prep" not in es[0]


@pytest.mark.parametrize("entry, msg", [
    ({"name": "a", "kind": "ref", "ref": "x", "prompt_text": "t"}, "continuation"),
    ({"name": "a", "kind": "ref", "ref": "x", "prompt_wav_path": "t"}, "accent"),
    ({"name": "a", "kind": "ref", "ref": "x", "lora_path": "l"}, "unknown keys"),
    ({"name": "a/b", "kind": "ref", "ref": "x"}, "name must"),
    ({"name": "screen", "kind": "ref", "ref": "x"}, "name must"),
    ({"name": "a", "kind": "voice", "ref": "x"}, "kind must"),
    ({"name": "a", "kind": "ref"}, "clones a reference"),
    ({"name": "a", "kind": "designed", "lora": "/l"}, "clones a reference"),
    ({"name": "a", "kind": "lora"}, "needs lora"),
    ({"name": "a", "kind": "ref", "ref": ""}, "non-empty"),
    ({"name": "a", "kind": "ref", "ref": "x", "ref_prep": "no"}, "ref_prep must be true or false"),
])
def test_parse_entries_refuses(entry, msg):
    with pytest.raises(ValueError, match=msg):
        vc.parse_entries([entry])


def test_parse_entries_duplicate_name():
    with pytest.raises(ValueError, match="duplicate"):
        vc.parse_entries([{"name": "a", "kind": "ref", "ref": "x"}, {"name": "a", "kind": "ref", "ref": "y"}])


SRC = {"license": "CC0 1.0", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
       "consent": "donated by the speaker", "consent_source": "dataset card", "attribution": "Speaker X, dataset Y",
       "use": "reference-only cloning (VoxCPM2 reference_wav_path); never prompt_wav/prompt_text continuation",
       "gender": "female", "clips": {"neutral": {}}}


def _sources(d):
    return SRC


def test_check_sources():
    assert vc.check_sources(SRC) == [] and vc.check_sources(json.loads(json.dumps(SRC))) == []
    assert vc.check_sources({}) == list(vc.SOURCE_KEYS)
    assert vc.check_sources(None) == vc.check_sources([1]) == ["readable JSON object"]
    assert vc.check_sources({**SRC, "consent": " "}) == ["consent"]
    assert vc.check_sources({**SRC, "use": "any cloning"}) == ["use: reference-only cloning"]
    assert set(vc.source_record(SRC)) == set(vc.SOURCE_KEYS)
    for p in sorted((REPO / "voxcpm-refs").glob("*/sources.json")):  # the fetched records (gitignored), when here
        assert vc.check_sources(json.loads(p.read_text(encoding="utf-8"))) == [], p


def test_discover_refs_needs_sources_json_and_names_clips():
    warned = []
    listing = {"masha": ["sources.json", "expressive.wav", "neutral.wav", "notes.txt"], "dima": ["sources.json",
               "clip.flac"], "nolic": ["a.wav"], "empty": ["sources.json"], "bad name!": ["sources.json", "a.wav"],
               "bare": ["sources.json", "a.wav"], "anyuse": ["sources.json", "a.wav"]}
    recs = {"bare": {}, "anyuse": {**SRC, "use": "cloning"}}
    es = vc.discover_refs(listing, "/R", warned.append, lambda d: recs.get(d, SRC))
    assert [(e["name"], e["ref"]) for e in es] == [("dima", "/R/dima/clip.flac"), ("masha", "/R/masha/neutral.wav"),
                                                   ("masha-expressive", "/R/masha/expressive.wav")]
    assert all(e["kind"] == "ref" and e["source"]["license"] == "CC0 1.0" and e["gender"] == "female" for e in es)
    assert len(warned) == 5 and sum("sources.json lacks" in w for w in warned) == 2
    assert "license" in next(w for w in warned if "bare" in w) and "reference-only" in next(
        w for w in warned if "anyuse" in w)
    two = vc.discover_refs({"t": ["sources.json", "a.wav", "b.wav"]}, "/R", read_sources=_sources)  # no neutral
    assert [e["name"] for e in two] == ["t-a", "t-b"]
    assert vc.discover_refs({"t": ["sources.json", "a.wav"]}, "/R") == []  # no reader: nothing passes the check
    assert vc.parse_entries(es)[0]["source"]["attribution"] == "Speaker X, dataset Y"


def test_resolve_spec_refs_screen_and_arms():
    listing = {"masha": ["sources.json", "neutral.wav"], "dima": ["sources.json", "neutral.wav"]}
    screens = {"r1": {"entries": [{"name": "designed-f01-s0", "kind": "designed", "ref": "/c/r1/f01-s0.wav"}]}}
    spec = {"entries": [{"name": "masha", "kind": "ref", "ref": "/mine.wav"}], "refs": "auto",
            "arms": [{"lora": "/s1", "suffix": "+s1", "entries": ["masha", "designed-f01-s0"]}]}
    es = vc.resolve_spec(spec, "r1", lambda: listing, screens.get, "/R", read_sources=_sources)
    by = {e["name"]: e for e in es}
    assert list(by) == ["masha", "dima", "designed-f01-s0", "masha+s1", "designed-f01-s0+s1"]
    assert by["masha"]["ref"] == "/mine.wav"  # explicit wins over the discovered folder
    assert by["masha+s1"]["lora"] == "/s1" and by["masha+s1"]["ref"] == "/mine.wav" and "second arm" in by[
        "masha+s1"]["notes"]
    # the same run's screen is optional by default, named screens must exist
    assert [e["name"] for e in vc.resolve_spec({"entries": [], "refs": False}, "r2", dict, screens.get)] == []
    with pytest.raises(ValueError, match="run contest-screen"):
        vc.resolve_spec({"screen": "r9"}, "r2", dict, screens.get)
    with pytest.raises(ValueError, match="also named"):
        vc.resolve_spec({"entries": [{"name": "designed-f01-s0", "kind": "ref", "ref": "x"}]}, "r1", dict, screens.get)
    with pytest.raises(ValueError, match="no folder"):
        vc.resolve_spec({"refs": ["nope"]}, "r1", lambda: listing, screens.get)
    # render --screen: another run's winners (finals under a new run); a bad run name is refused
    got = vc.resolve_spec(vc.with_screen({"entries": [], "refs": False}, "r1"), "r1-final", dict, screens.get)
    assert [e["name"] for e in got] == ["designed-f01-s0"]
    assert vc.with_screen([], "r1, r2") == {"entries": [], "screen": ["r1", "r2"]} and vc.with_screen(spec, "") is spec
    with pytest.raises(ValueError, match="not a contest run"):
        vc.with_screen({}, "../x")
    with pytest.raises(ValueError, match="run contest-screen"):
        vc.resolve_spec(vc.with_screen({}, "r9"), "r1", dict, screens.get)
    with pytest.raises(ValueError, match="already has a LoRA"):
        vc.resolve_spec({"entries": [{"name": "c", "kind": "lora", "lora": "/l"}],
                         "arms": [{"lora": "/s1", "entries": ["c"]}]}, "", dict, screens.get)
    with pytest.raises(ValueError, match="no entry"):
        vc.resolve_spec({"arms": [{"lora": "/s1", "entries": ["zz"]}]}, "", dict, screens.get)
    with pytest.raises(ValueError, match="unknown spec keys"):
        vc.resolve_spec({"entry": []}, "", dict, screens.get)


def test_default_spec_resolves():
    es = vc.resolve_spec(vc.DEFAULT_SPEC, "r1", lambda: {"masha": ["sources.json", "neutral.wav"]}, lambda r: None,
                         read_sources=_sources)
    assert [e["name"] for e in es] == ["galaktik-s2", "base-antalia", "masha"]
    assert es[0]["lora"] == vc.S2_LORA and es[1]["ref"] == vc.ANTALIA_REF
    # a spec without "entries" (README's s1.json) keeps the controls: finals may name them next to a +s1 arm
    arm = {"refs": "auto", "arms": [{"lora": "/s1", "suffix": "+s1", "entries": ["masha"]}]}
    es = vc.resolve_spec(arm, "r1", lambda: {"masha": ["sources.json", "neutral.wav"]}, lambda r: None,
                         read_sources=_sources)
    assert [e["name"] for e in es] == ["galaktik-s2", "base-antalia", "masha", "masha+s1"]
    assert [e["name"] for e in vc.resolve_spec({"entries": [], "refs": False}, "r1", dict, lambda r: None)] == []


def test_resolve_ref_file_dir_manifest():
    fs = FakeFS({"/R/m/neutral.wav": "", "/R/m/expressive.wav": "", "/R/m/sources.json": "{}", "/R/one/x.flac": "",
                 "/R/two/a.wav": "", "/R/two/b.wav": "", "/d/train.jsonl": '{"audio": "/d/wav/1.wav"}\n{"audio": 2}\n',
                 "/d/wav/1.wav": "", "/R/notes.txt": ""})
    assert vc.resolve_ref("m", fs, "/R") == "/R/m/neutral.wav"  # relative = under the refs root
    assert vc.resolve_ref("/R/one", fs, "/R") == "/R/one/x.flac"
    assert vc.resolve_ref("/R/m/expressive.wav", fs, "/R") == "/R/m/expressive.wav"
    assert vc.resolve_ref("manifest:/d/train.jsonl#0", fs) == "/d/wav/1.wav"
    vol = FakeFS({"/data/c/train.jsonl": '{"audio": "/__modal/volumes/vo-AbC123/c/wav/1.wav"}\n', "/data/c/wav/1.wav": ""})
    assert vc.resolve_ref("manifest:/data/c/train.jsonl#0", vol) == "/data/c/wav/1.wav"  # mount path -> /data
    for bad, msg in (("/R/two", "not exactly one"), ("/R/zz.wav", "not found"), ("/R/notes.txt", "not an audio"),
                     ("manifest:/d/train.jsonl#5", "no row 5"), ("manifest:/d/nope.jsonl#0", "no /d/nope"),
                     ("manifest:/d/train.jsonl", "want manifest")):
        with pytest.raises(ValueError, match=msg):
            vc.resolve_ref(bad, fs, "/R")


def test_resolve_lora_needs_both_files():
    fs = FakeFS({"/data/exp/a/ckpt/s/lora_weights.safetensors": "", "/data/exp/a/ckpt/s/lora_config.json": "",
                 "/x/lora_weights.safetensors": ""})
    assert vc.resolve_lora("a/ckpt/s", fs) == f"{vc.EXP}/a/ckpt/s" == "/data/exp/a/ckpt/s"
    with pytest.raises(ValueError, match="lora_config.json"):
        vc.resolve_lora("/x", fs)
    with pytest.raises(ValueError, match="not found"):
        vc.resolve_lora("/nope", fs)


def test_lora_config_and_entry_key(tmp_path):
    (tmp_path / "lora_config.json").write_text('{"lora_config": {"r": 64, "alpha": 32}}')
    assert vc.lora_config(str(tmp_path)) == {"r": 64, "alpha": 32}
    assert vc.lora_config(str(tmp_path / "no")) == {"auto": True} and vc.lora_config(None) is None
    k = vc.entry_key({"ref": "a", "lora": None}, {"timesteps": 10, "cfg": 2.0})
    assert k == vc.entry_key({"ref": "a", "lora": None, "notes": "x"}, {"cfg": 2.0, "timesteps": 10})
    assert k != vc.entry_key({"ref": "a"}, {"timesteps": 16, "cfg": 2.0})


def test_entry_conflict_and_render_settings():
    ns = vc.build_parser().parse_args(["render", "--out", "x", "--timesteps", "16"])
    s10, s16 = {**vc.render_settings(ns), "timesteps": 10}, vc.render_settings(ns)
    new = {"seed_mode": "per_text", "loudness": "-20dBFS", "ref_prep": "16k/rpad2560/-20dBFS"}  # round 2
    assert s16 == {"timesteps": 16, "cfg": 2.0, "seed": 0, "model": vc.MODEL_DIR, "backend": "nanovllm", **new}
    assert vc.screen_settings(ns) == {"timesteps": 16, "cfg": 2.0, "model": vc.MODEL_DIR, "backend": "nanovllm",
                                      **new}
    raw = vc.render_settings(vc.build_parser().parse_args(["render", "--out", "x", "--no-refprep"]))
    assert raw["ref_prep"] == "raw"
    # a spec entry's ref_prep wins over the call's; its engine too (entry_backend)
    assert vc.entry_settings(s16, {"name": "a", "ref_prep": False}, "voxcpm") == {**s16, "ref_prep": "raw",
                                                                                   "backend": "voxcpm"}
    assert vc.entry_settings(raw, {"name": "a", "ref_prep": True}, "nanovllm")["ref_prep"] == new["ref_prep"]
    assert vc.entry_settings(raw, {"name": "a"}, "nanovllm") == raw
    # round 1's entries (no seed_mode / loudness / ref_prep) are refused: clips made otherwise
    r1 = {k: v for k, v in s16.items() if k not in new}
    old1 = {"ref": "/r.wav", "settings": r1, "key": vc.entry_key({"ref": "/r.wav"}, r1)}
    assert vc.entry_conflict(old1, {"ref": "/r.wav"}, s16) == {k: f"None -> {v}" for k, v in new.items()}
    e = {"name": "a", "ref": "/r.wav"}
    old = {**e, "settings": s10, "key": vc.entry_key(e, s10)}
    assert vc.entry_conflict(None, e, s16) == {} and vc.entry_conflict(old, e, s10) == {}
    assert vc.entry_conflict(old, e, s16) == {"timesteps": "10 -> 16"}
    assert vc.entry_conflict(old, {**e, "lora": "/l"}, s10) == {"lora": "None -> /l"}
    assert "key" in vc.entry_conflict({"key": "zz", "ref": "/r.wav", "settings": s10}, e, {})


def test_pause_clean_settings_only_when_on_and_jobs_carry_it(tmp_path):
    import pause_clean

    on = vc.render_settings(vc.build_parser().parse_args(["render", "--out", "x", "--pause-clean"]))
    off = vc.render_settings(vc.build_parser().parse_args(["render", "--out", "x"]))
    assert on == {**off, "pause_clean": pause_clean.SETTINGS} and "pause_clean" not in off  # old entries keep their key
    assert pause_clean.SETTINGS == "v1/hp80/L-55dB/max25dB"
    e = {"name": "a", "ref": "/r.wav"}
    old = {**e, "settings": off, "key": vc.entry_key(e, off)}
    assert vc.entry_conflict(old, e, off) == {}
    assert vc.entry_conflict(old, e, on) == {"pause_clean": f"None -> {pause_clean.SETTINGS}"}  # never one entry dir
    assert vc.entry_settings(on, e, "voxcpm")["pause_clean"] == pause_clean.SETTINGS
    row = {"set": "s", "id": "a", "prompt": "Yarın", "model_text": "Yarın", "wav": "wav/s/a.wav"}
    assert vc.make_jobs([row], tmp_path, "g", None, None, 10, 2.0, 0, "voxcpm", True, True)[0]["pause_clean"] is True
    assert vc.make_jobs([row], tmp_path, "g", None, None, 10, 2.0, 0)[0]["pause_clean"] is False
    rows = [{"pause_clean": {"pauses": 3, "breaths": 1, "gated_s": 0.1}}, {"pause_clean": {"pauses": 2, "breaths": 2,
                                                                                         "gated_s": 0.25}}, {}]
    assert vc.pause_summary(rows) == {"settings": pause_clean.SETTINGS, "clips": 2, "pauses": 5, "breaths": 3,
                                      "gated_s": 0.35}
    assert vc.pause_summary([{}, {"pause_clean": None}]) is None


def _breathy(sr: int = 16000):
    """Two harmonic tones at -20 dBFS with a 0.45 s pause between them holding a band-noise breath at -32 dB."""
    import numpy as np
    from scipy.signal import butter, sosfilt

    t = np.arange(int(0.5 * sr)) / sr
    tone = sum(a * np.sin(2 * np.pi * 180 * k * t) for k, a in ((1, 1.0), (2, 0.6), (3, 0.4)))
    tone *= 0.1 / np.sqrt(np.mean(tone ** 2)) * np.minimum(1, np.minimum(t, t[::-1]) / 0.01)
    gap = np.zeros(int(0.45 * sr))
    sos = butter(4, (1000, 4000), "bandpass", fs=sr, output="sos")
    v = sosfilt(sos, np.random.default_rng(1).standard_normal(4000))
    gap[1600:5600] = v / np.sqrt(np.mean(v ** 2)) * 0.1 * 10 ** (-32 / 20) * np.hanning(4000)
    x = np.concatenate([np.zeros(3200), tone, gap, tone, np.zeros(3200)])
    return x + np.random.default_rng(2).standard_normal(len(x)) * 1e-5


def test_clean_pauses_turns_a_breath_down_deterministically():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    import pause_clean

    x = _breathy()
    y, rec = vc.clean_pauses(x, 16000)
    y2, rec2 = vc.clean_pauses(x, 16000)
    assert np.array_equal(y, y2) and rec == rec2 and y.dtype == np.float32 and y.shape == x.shape
    assert tuple(rec) == pause_clean.RECORD_KEYS and rec["pauses"] == 1 and rec["breaths"] == 1 and rec["gated_s"] > 0
    mid = slice(3200 + 8000 + 2600, 3200 + 8000 + 4600)  # the breath's middle 125 ms: > 20 dB down
    assert np.sqrt(np.mean(y[mid].astype(float) ** 2)) < 0.1 * np.sqrt(np.mean((x[mid] - x.mean()) ** 2))
    tone = slice(3200, 3200 + 8000)
    assert np.max(np.abs(y[tone] - (x[tone] - x.mean()))) < 1e-6  # the speech: the input (its DC removed)


def test_pool_ram_and_load_parallel_bounds():
    assert vc.pool_ram_gb(4, 2) == 2 * vc.WORKER_RAM_GB + 2 * vc.LOAD_RAM_GB + vc.BASE_RAM_GB
    assert vc.pool_ram_gb(1, 4) == vc.LOAD_RAM_GB + vc.BASE_RAM_GB  # never more loads than workers
    assert vc.pool_ram_gb(8, 2) > vc.pool_ram_gb(8, 1) > vc.pool_ram_gb(4, 1)
    with pytest.raises(SystemExit):
        vc.build_parser().parse_args(["render", "--out", "x", "--load-parallel", "9"])


# ---- texts --------------------------------------------------------------------------------------------------------


def test_load_texts_normalizes_once_and_reads_fields(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [{"id": "z1", "set": "trzor", "category": "sayi", "text": "Toplantıya 45 kişi katıldı.",
             "refs": ["Toplantıya kırk beş kişi katıldı."]},
            {"id": "q1", "set": "soru", "category": "mi_soru", "text": "Bu tren Ankara'ya gidiyor mu?",
             "spoken": "Bu tren Ankara'ya gidiyor mu?", "pattern": "peak_fall", "peak": 3, "peak_word": "gidiyor",
             "pair_id": "g04"},
            {"id": "e1", "set": "emo", "category": "excited", "instruction": "(Cheerful tone.)", "text": "Saat 3'te!"},
            {"id": "e2", "set": "emo", "emotion": "Sadness", "text": "Çok üzgünüm."}]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    t = {r["id"]: r for r in vc.load_texts(p)}
    assert t["z1"]["model_text"] == normalize("Toplantıya 45 kişi katıldı.") == "Toplantıya kırk beş kişi katıldı."
    assert t["z1"]["refs"] == ["Toplantıya kırk beş kişi katıldı."] and "instruction" not in t["z1"]
    assert t["q1"]["refs"] == ["Bu tren Ankara'ya gidiyor mu?"] and t["q1"]["peak"] == 3 and t["q1"]["pair_id"] == "g04"
    assert t["e1"]["instruction"] == "Cheerful tone." and t["e1"]["emotion"] == "cheerful"
    assert t["e1"]["refs"] == [normalize("Saat 3'te!")] and "üç" in t["e1"]["model_text"]
    assert t["e2"]["emotion"] == "sad" and t["e2"]["category"] == "sad"
    assert vc.compose(t["e1"]["instruction"], t["e1"]["model_text"]) == f"(Cheerful tone.){t['e1']['model_text']}"


@pytest.mark.parametrize("row, msg", [
    ({"id": "a", "set": "soru", "text": "Geliyor mu?"}, "soru row needs"),
    ({"id": "a b", "set": "trzor", "text": "x"}, "not unique"),
    ({"id": "a", "set": "emo", "text": "x", "instruction": "Angry (very)."}, "parentheses"),
    ({"id": "a", "set": "trzor", "text": " "}, "empty text"),
    ({"id": "a", "text": "x"}, "want a JSON object"),
])
def test_load_texts_refuses(tmp_path, row, msg):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        vc.load_texts(p)


def test_text_seeds_a_row_seed_ids_and_seed_sweeps(tmp_path):
    a, b = vc.text_seed(0, "soru", "q1"), vc.text_seed(0, "soru", "q2")
    assert a != b and 0 <= a < 2**31 and vc.text_seed(5, "soru", "q1") == a + 5  # --seed shifts every text
    assert a == vc.text_seed(0, "soru", "q1")  # stable across calls and processes (crc32, not hash())
    assert vc.job_seed({"set": "soru", "id": "q1"}, 0) == a and vc.job_seed({"set": "soru", "id": "q1", "seed": 3}, 0) == 3
    p = tmp_path / "t.jsonl"
    rows = [{"id": "q1", "set": "trzor", "text": "Yarın gel."}, {"id": "q2", "set": "trzor", "text": "Bu.", "seed": 7}]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    t = vc.load_texts(p)
    assert "seed" not in t[0] and t[1]["seed"] == 7
    for bad in ("x", -1, True, 1.5):
        q = tmp_path / "bad.jsonl"
        q.write_text(json.dumps({**rows[0], "seed": bad}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="seed must be an integer"):
            vc.load_texts(q)
    sweep = vc.select_texts(t, "q1", "0,3")
    assert [(r["id"], r["seed"]) for r in sweep] == [("q1-s0", 0), ("q1-s3", 3)] and sweep[0]["text"] == "Yarın gel."
    assert vc.select_texts(t) == t and [r["id"] for r in vc.select_texts(t, "q2,q1")] == ["q1", "q2"]
    with pytest.raises(ValueError, match="--ids: no text zz"):
        vc.select_texts(t, "q1,zz")
    with pytest.raises(ValueError, match="--seeds"):
        vc.select_texts(t, "", "-1")
    ns = vc.build_parser().parse_args(["render", "--out", "x", "--texts", str(p), "--ids", "q1", "--seeds", "1,2"])
    assert [r["id"] for r in vc.render_texts(ns)] == ["q1-s1", "q1-s2"]


Y_IDS = "sira_01,soru_01,duz_eslesik_01,mi_soru_01,emo_angry_01,emo_empathetic_01"  # the onset check's texts


def test_verify_onset_spec_and_texts(tmp_path):
    spec = json.loads((VOXCPM / "verify_onset.json").read_text(encoding="utf-8"))
    es = vc.resolve_spec(spec, "onset-v1", lambda: {}, lambda r: None)
    assert [e["name"] for e in es] == ["masha", "dima", "designed-f037-s1", "masha-raw", "dima-raw",
                                       "designed-f037-s1-raw"]
    s = vc.render_settings(vc.build_parser().parse_args(["render", "--out", "x"]))
    prep = {e["name"]: vc.entry_settings(s, e, "nanovllm")["ref_prep"] for e in es}
    assert {n: p != "raw" for n, p in prep.items()} == {e["name"]: not e["name"].endswith("-raw") for e in es}
    row = {"set": "s", "id": "a", "prompt": "Yarın", "model_text": "Yarın", "wav": "wav/s/a.wav"}
    jobs = vc.make_jobs([row], tmp_path, "g", "/r.wav", None, 10, 2.0, 0, "nanovllm", False)
    assert jobs[0]["ref_prep"] is False and jobs[0]["seed"] == vc.text_seed(0, "s", "a")
    if not (REPO / vc.TEXTS).is_file():
        pytest.skip("no contest.jsonl")
    import onset

    ns = vc.build_parser().parse_args(["render", "--out", "x", "--ids", Y_IDS, "--seeds", "0,1,2,3"])
    rows = vc.render_texts(ns)
    assert len(rows) == 24 and {r["seed"] for r in rows} == {0, 1, 2, 3}
    assert all(onset.text_class(_prompt(r)) == "y" for r in rows)


def _prompt(r: dict) -> str:
    return vc.compose(r.get("instruction"), r["model_text"])


def test_contest_set_parses_when_present():
    p = REPO / "benchmarks" / "tts" / "sets" / "contest.jsonl"
    if not p.is_file():
        pytest.skip("no contest.jsonl yet")
    rows = vc.load_texts(p)
    sets = {r["set"] for r in rows}
    assert {"trzor", "soru", "emo"} <= sets
    assert all(r.get("instruction") for r in rows if r["set"] == "emo")
    assert all(r["model_text"] == normalize(r["text"]) for r in rows)


def test_load_descriptions(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"id": "f01", "gender": "Female", "text": "A warm voice; calm."}\n'
                 '{"id": "m01", "description": "(A deep male voice.)"}\n', encoding="utf-8")
    d = vc.load_descriptions(p)
    assert d == [{"id": "f01", "gender": "female", "text": "A warm voice; calm."},
                 {"id": "m01", "gender": "", "text": "A deep male voice."}]
    p.write_text('{"id": "f01", "text": "A voice (soft)."}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="want id and text"):
        vc.load_descriptions(p)


def test_merge_rows_keeps_scores_of_the_same_clip_only(tmp_path):
    _wav(tmp_path / "wav/s/a.wav")
    _wav(tmp_path / "wav/s/b.wav")
    old = [{"set": "s", "id": "a", "prompt": "x", "wav": "wav/s/a.wav", "utmos": 4.0},
           {"set": "s", "id": "b", "prompt": "old", "wav": "wav/s/b.wav", "utmos": 3.0}]
    new = [{"set": "s", "id": "a", "prompt": "x", "wav": "wav/s/a.wav"},
           {"set": "s", "id": "b", "prompt": "new", "wav": "wav/s/b.wav"}]
    out = vc.merge_rows(new, old, tmp_path)
    assert out[0]["utmos"] == 4.0 and "utmos" not in out[1]
    assert (tmp_path / "wav/s/a.wav").exists() and not (tmp_path / "wav/s/b.wav").exists()  # made again


# ---- error rates --------------------------------------------------------------------------------------------------


def test_edit_distance():
    assert vc.edit_distance("kitten", "sitting") == 3
    assert vc.edit_distance([], ["a"]) == 1 and vc.edit_distance(["a", "b"], ["a", "b"]) == 0


def test_wer_cer_closest_ref_and_empty_hyp():
    assert vc.wer_cer(["Bir iki üç."], "bir iki üç") == (0, 3, 0, 10)
    assert vc.wer_cer(["Bin dokuz yüz doksanlı yıllar", "Doksanlı yıllar"], "doksanlı yıllar") == (0, 2, 0, 15)
    assert vc.wer_cer(["a b c d"], "a x c d")[:2] == (1, 4)
    assert vc.wer_cer(["merhaba dünya"], "") == (2, 2, 13, 13)  # all deleted, as synth_eval's "<empty>" counts
    assert vc.wer_cer(["merhaba"], "merhabalar")[2:] == (3, 7)


# ---- cost model and job order -------------------------------------------------------------------------------------


def test_rtf_speedup_and_measured_throughput():
    assert vc.rtf(10) == pytest.approx(vc.RTF_10) and vc.rtf(16) == pytest.approx(vc.RTF_16)
    assert vc.speedup(1) == 1.0 and vc.speedup(4) == pytest.approx(2.2) and vc.speedup(8) == 2.5
    assert all(vc.speedup(w) <= w for w in range(1, 9))
    assert vc.wall_per_audio(10, 4) == pytest.approx(vc.rtf(10) / vc.speedup(4))
    m = [{"timesteps": 10, "workers": 6, "audio_s": 600, "wall_s": 150},
         {"timesteps": 10, "workers": 4, "audio_s": 600, "wall_s": 300},
         {"timesteps": 10, "workers": 4, "audio_s": 10, "wall_s": 100}]  # too little audio: ignored
    assert vc.wall_per_audio(10, 4, m) == pytest.approx(0.5)  # the last usable one of these workers
    assert vc.wall_per_audio(10, 2, m[:1]) == pytest.approx(0.25 * vc.speedup(6) / vc.speedup(2))
    assert vc.wall_per_audio(16, 4, m) == pytest.approx(vc.rtf(16) / vc.speedup(4))


def test_plan_render_fits_in_order_and_counts_done():
    rows = [{"set": "s", "id": str(i), "model_text": "x" * 45} for i in range(10)]  # 3.3 s each
    es = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    per = vc.wall_per_audio(10, 1)
    one = 10 * 3.3 * per + 10 * vc.SCORE_CLIP_S
    budget = vc.SCORE_LOAD_S + vc.SUMMARY_S + vc.WORKERS_UP_S + 2 * one + 1
    plan = vc.plan_render(es, rows, {"b": {"s/0", "s/1"}}, 10, 1, budget)
    assert [p["todo"] for p in plan] == [10, 8, 10]
    assert plan[0]["est_s"] == pytest.approx(one + vc.WORKERS_UP_S, abs=0.1)  # worker start-up once
    assert [p["fits"] for p in plan] == [True, True, False]
    done = {e["name"]: {f"s/{i}" for i in range(10)} for e in es}
    assert all(p["fits"] and p["est_s"] == 0 for p in vc.plan_render(es, rows, done, 10, 1, 0.0 + 150))
    synth = vc.plan_render(es, rows, {}, 10, 1, budget, score=False)  # --no-score: no scorer loads, no scoring
    assert synth[0]["est_s"] == pytest.approx(10 * 3.3 * per + vc.WORKERS_UP_S, abs=0.1) and all(p["fits"]
                                                                                                for p in synth)


def test_plan_screen_and_clone_top():
    a = vc.plan_screen(300, 30, list(vc.CLONE_TEXTS), 10, 4)
    assert a > vc.plan_screen(100, 30, list(vc.CLONE_TEXTS), 10, 4) > vc.plan_screen(100, 30, list(vc.CLONE_TEXTS), 10,
                                                                                     4, clones_done=120)
    assert vc.plan_screen(300, 30, list(vc.CLONE_TEXTS), 16, 4) > a
    assert vc.clone_top(30, 150, 1, 2) == 30 and vc.clone_top(30, 10, 1, 2) == 10 and vc.clone_top(30, 10, 3, 2) == 20


def test_order_jobs_groups_in_order_longest_first():
    jobs = [{"key": "b/1", "group": "b", "est_s": 2.0}, {"key": "a/1", "group": "a", "est_s": 1.0},
            {"key": "b/2", "group": "b", "est_s": 5.0}, {"key": "a/2", "group": "a", "est_s": 3.0},
            {"key": "a/0", "group": "a", "est_s": 3.0}]
    assert [j["key"] for j in vc.order_jobs(jobs)] == ["b/2", "b/1", "a/0", "a/2", "a/1"]


# ---- screen ranking -----------------------------------------------------------------------------------------------


def _cand(i, utmos, desc=None, cer=0, p_tr=0.99, dur=10.0, gender="female"):
    return {"id": f"{desc or 'd' + str(i)}-s{i % 2}", "desc_id": desc or f"d{i}", "seed": i % 2, "gender": gender,
            "description": f"voice {i}", "utmos": utmos, "char_errors": cer, "ref_chars": 100, "p_tr": p_tr,
            "duration": dur}


def test_select_for_clone_filters_and_caps():
    cands = [_cand(0, 4.5, cer=5), _cand(1, 4.4, "dA"), _cand(2, 4.3, "dA"), _cand(3, 4.2, p_tr=0.5),
             _cand(4, 4.1, dur=40.0), _cand(5, 4.0), _cand(6, None), _cand(7, 3.9)]
    got = vc.select_for_clone(cands, 3, per_desc=1, max_cer=0.02, min_p_tr=0.9, dur_range=(4.0, 27.0))
    assert [c["id"] for c in got] == ["dA-s1", "d5-s1", "d7-s1"]  # one of dA; CER, p(tr), length, unscored out
    assert len(vc.select_for_clone(cands, 3, per_desc=2, max_cer=0.02, min_p_tr=0.9, dur_range=(4, 27))) == 3


def _clones(cand, utmos, cer0=4, sim=0.7, p_tr=0.99, n=4):
    return [{"cand": cand, "utmos": u, "char_errors": 0 if k < cer0 else 3, "ref_chars": 60, "sim": sim, "p_tr": p_tr}
            for k, u in enumerate(utmos[:n])]


def test_rank_voices_by_clone_mean_not_the_candidate_clip():
    chosen = [_cand(0, 4.6, "hi"), _cand(1, 4.0, "steady"), _cand(2, 4.2, "fuzzy"), _cand(3, 4.1, "far"),
              _cand(4, 4.3, "short")]
    clones = (_clones(chosen[0]["id"], [3.6, 3.7, 3.5, 3.8]) + _clones(chosen[1]["id"], [4.0, 4.1, 3.9, 4.0])
              + _clones(chosen[2]["id"], [4.4, 4.4, 4.4, 4.4], cer0=2) + _clones(chosen[3]["id"], [4.3] * 4, sim=0.2)
              + _clones(chosen[4]["id"], [4.5] * 4, n=3))
    r = vc.rank_voices(chosen, clones, 4, need_cer0=3, min_sim=0.45, min_p_tr=0.9)
    assert [v["desc_id"] for v in r] == ["steady", "hi", "short", "fuzzy", "far"]  # eligible first, then clone mean
    assert r[0]["utmos_mean"] == pytest.approx(4.0) and r[0]["eligible"] and r[1]["eligible"]
    assert [v["rank"] for v in r] == [1, 2, 3, 4, 5]
    why = {v["desc_id"]: " ".join(v["why_not"]) for v in r}
    assert "CER 0 on 2" in why["fuzzy"] and "similarity" in why["far"] and "3/4 clones" in why["short"]


def test_pick_voices_keeps_a_gender_and_rank_order():
    ranked = [{"cand": f"f{i}", "gender": "female", "eligible": True, "rank": i + 1} for i in range(6)]
    ranked += [{"cand": "m0", "gender": "male", "eligible": True, "rank": 7},
               {"cand": "m1", "gender": "male", "eligible": False, "rank": 8}]
    assert [v["cand"] for v in vc.pick_voices(ranked, 5, 1)] == ["f0", "f1", "f2", "f3", "m0"]
    assert [v["cand"] for v in vc.pick_voices(ranked, 5, 0)] == ["f0", "f1", "f2", "f3", "f4"]
    assert vc.pick_voices([v for v in ranked if not v["eligible"]], 5) == []


def test_designed_entries_are_a_valid_spec():
    v = {"cand": "f01-s0", "gender": "female", "description": "A warm voice.", "rank": 1, "utmos_mean": 4.1,
         "utmos_min": 3.9, "cer0": 4, "clones": 4, "sim_mean": 0.7, "p_tr_mean": 0.99, "cand_utmos": 4.3}
    es = vc.parse_entries(vc.designed_entries([v], lambda c: f"/data/contest/r1/screen/cand/wav/{c}.wav"))
    assert es[0]["name"] == "designed-f01-s0" and es[0]["kind"] == "designed" and es[0]["ref"].endswith("f01-s0.wav")


# ---- summaries ----------------------------------------------------------------------------------------------------


def test_fold_to_moves_whole_octaves_only():
    assert vc.fold_to(15.0, 3.0) == 3.0 and vc.fold_to(-9.5, 3.0) == 2.5
    assert vc.fold_to(11.0, 3.0) == 11.0 and vc.fold_to(-1.0, 3.0) == -1.0  # +8 / -4: real shifts stay
    assert vc.fold_to(None, 3.0) is None and vc.fold_to(4.0, None) == 4.0


def _pros(f0, rng=10.0, rate=6.0, db=-22.0):
    return {"f0_median_st": f0, "f0_range_st": rng, "rate_syl_s": rate, "energy_db": db}


def test_emotion_summary_deltas_octave_fold_and_agreement():
    rows = [{"set": "trzor", "prosody": _pros(f)} for f in (4.0, 5.0, 6.0, 17.0, 5.0)]  # one clip an octave off
    rows += [{"set": "emo", "emotion": "angry", "prosody": _pros(8.0, db=-18.0)},
             {"set": "emo", "emotion": "sad", "prosody": _pros(-7.0 + 12, rate=5.0, db=-25.0)},
             {"set": "emo", "emotion": "calm", "prosody": _pros(5.0, rng=12.0, rate=5.0)},
             {"set": "emo", "emotion": "whisper", "prosody": _pros(5.0)}]
    s = vc.emotion_summary(rows)
    assert s["baseline"]["f0_median_st"] == 5.0 and s["baseline_rows"] == 5
    e = s["by_emotion"]
    assert e["angry"]["f0_median_st"] == pytest.approx(3.0) and e["angry"]["energy_db"] == pytest.approx(4.0)
    assert e["angry"]["agree"] == 1.0 and e["sad"]["agree"] == pytest.approx(2 / 3, abs=1e-3)  # F0 0: not lower
    assert e["calm"]["agree"] == 0.5 and "agree" not in e["whisper"]
    assert s["agree"] == pytest.approx((2 + 2 + 1) / (2 + 3 + 2), abs=1e-3)
    assert vc.emotion_summary([{"set": "trzor", "prosody": _pros(1.0)}]) == {}


def _row(set_, i, errors=0, words=5, utmos=4.0, p_tr=0.99, **kw):
    return {"set": set_, "id": f"{set_}{i}", "category": "c", "wav": "w", "duration": 2.0, "synth_seconds": 2.0,
            "hyp": "h", "errors": errors, "ref_words": words, "char_errors": errors, "ref_chars": 20, "utmos": utmos,
            "p_tr": p_tr, "lang": "tr" if p_tr > 0.5 else "en", "prosody": _pros(5.0), "sim": 0.6, "self_sim": 0.8,
            **kw}


def test_entry_summary_and_leaderboard():
    good = [_row("trzor", i, errors=i % 2) for i in range(4)] + [_row("emo", 0, emotion="angry")]
    s = vc.entry_summary(good, {"name": "good", "kind": "ref", "ref": "/r.wav", "sim_file": "ref.wav"})
    assert s["wer"]["all"] == {"wer": 8.0, "rows": 5, "rows_correct": 3} and s["wer"]["trzor"]["wer"] == 10.0
    assert s["cer"] == 2.0 and s["lang"]["p_tr_mean"] == 0.99 and s["lang"]["share_ok"] == 1.0
    assert s["sim"]["mean"] == 0.6 and s["self_sim"] == 0.8 and s["rtf_job"] == 1.0 and s["complete"]
    accent = [_row("trzor", i, utmos=4.4, p_tr=0.4) for i in range(4)]  # heard as another language
    a = vc.entry_summary(accent, {"name": "accent", "kind": "ref"})
    assert a["lang"]["not_tr"] == 4
    partial = [_row("trzor", 0, utmos=3.0), {"set": "trzor", "id": "x", "category": "c", "wav": "w"}]
    p = vc.entry_summary(partial, {"name": "partial", "kind": "lora", "lora": "/l"})
    assert not p["complete"] and p["synthesized"] == 1
    lines, md = vc.leaderboard([a, p, s])
    assert [h["entry"] for h in lines] == ["good", "partial", "accent"]  # p(tr) < 0.95 last, then by UTMOS
    assert lines[2]["utmos"] == 4.4 and not lines[2]["lang_ok"] and lines[1]["lora"]
    assert "partial (incomplete)" in md and md.count("\n|") >= 4
    assert s["onset"] is None and lines[0]["onset"] is None and "| onset stray |" in md


def test_onset_summary_rates_first_tries_of_covered_texts():
    rows = [_row("soru", 0, onset_bad=True, onset_why=["stray"], onset_retries=1, onset_trimmed=False),
            _row("soru", 1, onset_bad=True, onset_why=["burst", "stray"], onset_retries=3, onset_trimmed=True),
            _row("soru", 2, onset_bad=False, onset_why=[], onset_retries=0, onset_trimmed=False),
            _row("trzor", 0, onset_bad=None, onset_why=[], onset_retries=0, onset_trimmed=False),  # voiceless start
            _row("trzor", 1)]  # never checked
    s = vc.entry_summary(rows, {"name": "e", "kind": "ref"})
    assert s["onset"] == {"covered": 3, "stray": 2, "rate": 0.667, "retried": 2, "trimmed": 1,
                          "why": {"burst": 1, "stray": 2}, "not_covered": 1}
    lines, md = vc.leaderboard([s])
    assert lines[0]["onset"] == 0.667 and "| 0.667 |" in md


def test_cpu_pass_checks_the_onset_of_clips_made_without_it(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    rows = [{"set": "s", "id": "a", "refs": ["bir"], "model_text": "Bu bir.", "prompt": "Bu bir.", "wav": "a.wav",
             "prosody": {}},
            {"set": "s", "id": "b", "refs": ["bir"], "model_text": "Bu bir.", "wav": "b.wav", "prosody": {},
             "onset_bad": False}]
    for r in rows:  # 0.5 s at -20 dBFS
        with wave.open(str(tmp_path / r["wav"]), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes((3277).to_bytes(2, "little", signed=True) * 8000)
    w = lambda r: tmp_path / r["wav"]  # noqa: E731
    items = vc.cpu_items([rows], [w])
    assert [(x[0], x[5]) for x in items] == [((0, 0), True)]  # b was checked at synthesis
    vc.apply_cpu([rows], vc.CpuPass(items, 0).get())
    assert rows[0]["onset_bad"] is False and rows[0]["onset_why"] == [] and "onset_retries" not in rows[0]


# ---- workers, render, screen (fake backend) -----------------------------------------------------------------------


def _job(tmp_path, key, est=0.2, text="x"):
    return {"key": key, "group": key.split("/")[0], "text": text, "ref": None, "lora": None, "cfg": 2.0, "ts": 10,
            "seed": 0, "out": str(tmp_path / f"{key}.wav"), "est_s": est}


def test_pool_runs_jobs_reports_failures_and_stops_feeding_at_the_deadline(tmp_path):
    jobs = [_job(tmp_path, f"g/{i}") for i in range(6)] + [_job(tmp_path, "g/bad", text="FAIL here")]
    seen = []
    pool = vc.SynthPool(2, "unused", backend="fake", log=lambda m: None)
    try:
        st = pool.run(vc.order_jobs(jobs), on_done=lambda j, info: seen.append((j["key"], info["duration"])))
        assert (st["done"], st["failed"], st["left"]) == (6, 1, 0)
        assert st["workers_ready"] in (1, 2)  # spawn: the first worker up may make every fake clip before the second
        assert "g/bad" in st["failures"] and st["audio_s"] == pytest.approx(1.2)
        assert sorted(k for k, _ in seen) == [f"g/{i}" for i in range(6)]
        assert all(Path(j["out"]).stat().st_size > 44 for j in jobs[:6]) and not Path(jobs[6]["out"]).exists()
        late = pool.run([_job(tmp_path, "h/0")], feed_until=0.0)  # could not finish in time: never fed
        assert late["done"] == 0 and late["left"] == 1 and "deadline" in late["stopped"]
    finally:
        pool.close()
    assert all(not p.is_alive() for p in pool.procs.values())


def test_pool_takes_back_the_load_token_of_a_loader_killed_while_loading(tmp_path):
    # the first two of four workers die (exit 137, as the OOM killer) while each holds one of the two load tokens: a
    # shared semaphore would stay taken and the other two would wait till the hard deadline
    jobs = [_job(tmp_path, f"g/{i}") for i in range(6)]
    fails = []
    pool = vc.SynthPool(4, f"die-on-load:2:{tmp_path}", backend="fake", load_parallel=2, log=lambda m: None)
    try:
        st = pool.run(vc.order_jobs(jobs), hard_until=time.time() + 60, on_fail=lambda j, e: fails.append(j["key"]))
    finally:
        pool.close()
    assert (st["done"], st["left"], st["workers_ready"], st["stopped"]) == (6, 0, 2, "")
    assert sorted(st["dead"].values()) == ["exit code 137"] * 2 and not fails and st["load_parallel"] == 2


def test_pool_gives_up_when_no_worker_gets_ready(tmp_path):
    t = time.time()
    pool = vc.SynthPool(2, "hang-on-load", backend="fake", log=lambda m: None, ready_s=3)
    try:
        st = pool.run([_job(tmp_path, "g/0")], hard_until=t + 60)
    finally:
        pool.close(5)
    assert "no worker ready" in st["stopped"] and st["done"] == 0 and time.time() - t < 30
    assert all(not p.is_alive() for p in pool.procs.values())


def test_pool_records_failures_on_rows(tmp_path):
    rows = {"g/ok": {"id": "ok"}, "g/bad": {"id": "bad", "synth_fails": 1}}
    on_done, on_fail = vc.row_callbacks(rows)
    pool = vc.SynthPool(1, "unused", backend="fake", log=lambda m: None)
    try:
        pool.run([_job(tmp_path, "g/ok"), _job(tmp_path, "g/bad", text="FAIL")], on_done=on_done, on_fail=on_fail)
    finally:
        pool.close()
    assert rows["g/ok"]["duration"] > 0 and "synth_error" not in rows["g/ok"]
    assert rows["g/bad"]["synth_fails"] == 2 and "fake failure" in rows["g/bad"]["synth_error"]
    w = lambda r: tmp_path / "x.wav" if r["id"] == "ok" else None  # noqa: E731
    assert vc.unresolved([{**rows["g/ok"], "hyp": "", "utmos_error": "short"}, rows["g/bad"]], w) == []
    assert vc.unresolved([{**rows["g/ok"], "hyp": "x"}, {"id": "new"}], w) == ["ok", "new"]


class NoLoadScorers(vc.Scorers):
    """A scorer must not be loaded: every load fails the test."""

    def whisper(self):
        raise AssertionError("whisper loaded")

    def utmos(self):
        raise AssertionError("utmos loaded")

    def ecapa(self):
        raise AssertionError("ecapa loaded")


def test_scoring_passes_load_no_scorer_without_time(tmp_path):
    rows = [{"set": "s", "id": "a", "refs": ["bir"], "model_text": "bir", "wav": "a.wav"}]
    w = lambda r: _wav(tmp_path / r["wav"])  # noqa: E731
    sc, soon = NoLoadScorers("cpu"), time.time() + 10  # less than any scorer's load time
    assert vc.whisper_pass(sc, rows, w, "t", until=soon) == 1 and "hyp" not in rows[0]
    assert vc.utmos_pass(sc, rows, w, "t", until=soon) == 1
    assert vc.sim_pass(sc, [("t", rows, tmp_path / "a.wav")], w, until=soon) == 1
    items = vc.cpu_items([rows], [w])
    assert vc.CpuPass(items, 0).get(until=time.time() - 1) == []
    late = vc.CpuPass(items, 1)
    assert late.get(until=time.time() - 1) == []  # pool terminated, nothing waited for
    got = vc.CpuPass(items, 1).get(until=time.time() + 120)
    assert len(got) == 1 and got[0][0] == (0, 0)


def _render_setup(tmp_path):
    refs = tmp_path / "refs"
    _wav(refs / "masha" / "neutral.wav")
    (refs / "masha" / "sources.json").write_text(json.dumps(SRC))
    _wav(refs / "nolicence" / "a.wav")
    _wav(refs / "norecord" / "a.wav")
    (refs / "norecord" / "sources.json").write_text('{"licence": "OpenRAIL"}')
    texts = tmp_path / "texts.jsonl"
    rows = [{"id": "z1", "set": "trzor", "category": "sayi", "text": "Bu kitabı 3 kez okudum."},
            {"id": "q1", "set": "soru", "category": "mi_soru", "text": "Bu tren Ankara'ya gidiyor mu?",
             "pattern": "peak_fall", "peak": 3, "peak_word": "gidiyor"},
            {"id": "e1", "set": "emo", "emotion": "angry", "instruction": "Angry tone.", "text": "Yeter artık!"}]
    texts.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"entries": [], "refs": "auto"}))
    return ["render", "--out", str(tmp_path / "run" / "r1"), "--spec", str(spec), "--texts", str(texts),
            "--refs-root", str(refs), "--backend", "fake", "--workers", "2", "--no-score", "--throughput-log", ""]


def test_render_end_to_end_resume_and_settings_guard(tmp_path):
    argv = _render_setup(tmp_path)
    assert vc.main(argv) == 0
    d = tmp_path / "run" / "r1" / "masha"
    rows = vc.read_rows(d / "rows.jsonl")
    assert [r["id"] for r in rows] == ["z1", "q1", "e1"] and all(r["duration"] > 0 for r in rows)
    assert rows[2]["prompt"] == "(Angry tone.)Yeter artık!" and rows[0]["prompt"] == "Bu kitabı üç kez okudum."
    assert (d / "ref.wav").is_file() and json.loads((d / "sources.json").read_text())["license"] == "CC0 1.0"
    meta = json.loads((d / "entry.json").read_text())
    assert meta["settings"]["timesteps"] == 10 and meta["source"]["attribution"] == "Speaker X, dataset Y"
    assert json.loads((d / "summary.json").read_text())["source"]["license"] == "CC0 1.0"
    assert not (tmp_path / "run" / "r1" / "nolicence").exists()  # no sources.json: not an entry
    assert not (tmp_path / "run" / "r1" / "norecord").exists()  # sources.json without licence / consent: not either
    assert "masha: CC0 1.0; Speaker X, dataset Y" in (tmp_path / "run" / "r1" / "leaderboard.md").read_text()
    assert json.loads((tmp_path / "run" / "r1" / "status.json").read_text())["phase"] == "done"
    mtimes = {p: p.stat().st_mtime_ns for p in d.rglob("*.wav")}
    assert vc.main(argv) == 0  # resume: nothing made again
    assert {p: p.stat().st_mtime_ns for p in d.rglob("*.wav")} == mtimes
    assert vc.main([*argv, "--timesteps", "16"]) == 0  # other settings: the entry is refused, its clips stay
    st = json.loads((tmp_path / "run" / "r1" / "status.json").read_text())
    assert "use another --run" in st["errors"]["masha"] and "timesteps" in st["errors"]["masha"]
    assert {p: p.stat().st_mtime_ns for p in d.rglob("*.wav")} == mtimes


def test_render_finals_under_a_new_run_with_screen(tmp_path):
    argv = _render_setup(tmp_path)
    ref = _wav(tmp_path / "run" / "r1" / "screen" / "cand" / "wav" / "f1-s0.wav")
    vc.write_json(tmp_path / "run" / "r1" / "screen" / "entries.json",
                  {"entries": [{"name": "designed-f1-s0", "kind": "designed", "ref": str(ref)}]})
    fin = [*argv[:2], str(tmp_path / "run" / "r1-final"), *argv[3:], "--timesteps", "16", "--screen", "r1",
           "--only", "designed-f1-s0"]
    assert vc.main(fin) == 0
    d = tmp_path / "run" / "r1-final" / "designed-f1-s0"
    assert len(list((d / "wav").rglob("*.wav"))) == 3 and vc.read_json(d / "entry.json")["settings"]["timesteps"] == 16
    assert vc.main([*fin[:-4], "--screen", "r9"]) == 2  # a named screen run must exist


def test_render_past_its_deadline_loads_no_scorer_and_writes_summaries(tmp_path, monkeypatch):
    argv = _render_setup(tmp_path)
    assert vc.main(argv) == 0  # the clips
    monkeypatch.setattr(vc, "Scorers", NoLoadScorers)
    scored = [a for a in argv if a != "--no-score"]
    assert vc.main([*scored, "--deadline", f"{time.time() + 20:.0f}"]) == 0  # less than the scorer loads
    run = tmp_path / "run" / "r1"
    st = json.loads((run / "status.json").read_text())
    assert st["errors"] == {} and st["phase"] == "incomplete" and st["complete"] == {"masha": False}
    s = json.loads((run / "masha" / "summary.json").read_text())
    assert s["synthesized"] == 3 and s["unscored"]["hyp"] == 3 and (run / "leaderboard.md").is_file()


def test_render_dry_run_writes_nothing(tmp_path, capsys):
    argv = _render_setup(tmp_path)
    assert vc.main([*argv, "--dry-run"]) == 0
    assert not (tmp_path / "run").exists()
    out = capsys.readouterr().out
    assert "masha" in out and "emo/e1: (Angry tone.)Yeter artık!" in out
    assert vc.main([*argv, "--dry-run", "--pause-clean"]) == 0 and not (tmp_path / "run").exists()
    assert f"pause_clean {vc.PAUSE_CLEAN} after synthesis" in capsys.readouterr().out


def test_render_pause_clean_records_settings_rows_summary_and_needs_its_own_run(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")  # the fake workers run the real gate on their silent clips
    argv = _render_setup(tmp_path)
    assert vc.main(argv) == 0  # run r1 without it: no settings key, summary None
    d = tmp_path / "run" / "r1" / "masha"
    assert "pause_clean" not in vc.read_json(d / "entry.json")["settings"]
    assert vc.read_json(d / "summary.json")["pause_clean"] is None
    assert not any("pause_clean" in r for r in vc.read_rows(d / "rows.jsonl"))
    mtimes = {p: p.stat().st_mtime_ns for p in d.rglob("*.wav")}
    assert vc.main([*argv, "--pause-clean"]) == 0  # the same run: refused, its clips stay
    st = vc.read_json(tmp_path / "run" / "r1" / "status.json")
    assert "use another --run" in st["errors"]["masha"] and "pause_clean" in st["errors"]["masha"]
    assert {p: p.stat().st_mtime_ns for p in d.rglob("*.wav")} == mtimes
    pc = [*argv, "--pause-clean"]
    pc[2] = str(tmp_path / "run" / "pc1")
    assert vc.main(pc) == 0
    d = tmp_path / "run" / "pc1" / "masha"
    assert vc.read_json(d / "entry.json")["settings"]["pause_clean"] == vc.PAUSE_CLEAN
    rows = vc.read_rows(d / "rows.jsonl")
    assert [r["pause_clean"] for r in rows] == [{"pauses": 0, "breaths": 0, "gated_s": 0.0}] * 3  # silence: no pause
    assert vc.read_json(d / "summary.json")["pause_clean"] == {"settings": vc.PAUSE_CLEAN, "clips": 3, "pauses": 0,
                                                                "breaths": 0, "gated_s": 0.0}
    assert vc.main(pc) == 0  # resume: the rows keep their records
    assert vc.read_json(d / "summary.json")["pause_clean"]["clips"] == 3


UTM = {"f1-s0": 4.5, "f1-s1": 4.0, "f2-s0": 3.9, "f2-s1": 4.2, "m1-s0": 3.8, "m1-s1": 3.0}


def _screen(tmp_path, monkeypatch, descs=(("f1", "female"), ("f2", "female"), ("m1", "male")), no_sim=()):
    """Screen argv (fake backend) with Whisper / UTMOS / similarity stubbed: every clip with a WAV is scored, except no
    similarity for the clones in no_sim."""
    d = tmp_path / "d.jsonl"
    d.write_text("".join(json.dumps({"id": i, "gender": g, "text": f"A {g} voice, {i}."}) + "\n" for i, g in descs),
                 encoding="utf-8")

    def whisper(sc, rows, wav_of, label, rescore=False, until=math.inf):
        for r in rows:
            if wav_of(r) is not None:
                r.update(hyp=r["model_text"], lang="tr", p_tr=0.99)
        vc.errors_from_hyp(rows)

    def utmos(sc, rows, wav_of, label, rescore=False, until=math.inf):
        for r in rows:
            if wav_of(r) is not None:
                r["utmos"] = UTM.get(r["id"], 3.5) if r["set"] == "cand" else (3.7 if r["cand"].startswith("f1")
                                                                               else 4.1)

    def sim(sc, groups, wav_of, rescore=False, until=math.inf, self_sim=True):
        for _, rows, _ in groups:
            for r in rows:
                if wav_of(r) is not None and r["id"] not in no_sim:
                    r["sim"] = 0.8

    monkeypatch.setattr(vc, "whisper_pass", whisper)
    monkeypatch.setattr(vc, "utmos_pass", utmos)
    monkeypatch.setattr(vc, "sim_pass", sim)
    out = tmp_path / "contest" / "r1" / "screen"
    return out, ["screen", "--out", str(out), "--descriptions", str(d), "--top", "3", "--keep", "2", "--backend",
                 "fake", "--workers", "2", "--throughput-log", "", "--seeds", "0,1"]


def test_screen_end_to_end_with_stub_scores(tmp_path, monkeypatch):
    out, argv = _screen(tmp_path, monkeypatch)
    assert vc.main(argv) == 0
    rk = json.loads((out / "ranking.json").read_text())
    assert rk["params"]["clone_test"] == ["f1-s0", "f2-s1", "m1-s0"]  # best per description by candidate UTMOS
    assert [v["cand"] for v in rk["voices"]] == ["f2-s1", "m1-s0", "f1-s0"]  # by the clone mean
    assert rk["picked"] == ["f2-s1", "m1-s0"] and not rk["provisional"]  # keep 2, one of each gender
    es = vc.parse_entries(json.loads((out / "entries.json").read_text()))
    assert [e["name"] for e in es] == ["designed-f2-s1", "designed-m1-s0"] and Path(es[0]["ref"]).is_file()
    assert len(list((out / "clone" / "wav").rglob("*.wav"))) == 3 * len(vc.CLONE_TEXTS)
    assert "f2-s1" in (out / "ranking.md").read_text(encoding="utf-8")
    st = json.loads((out / "status.json").read_text())
    assert st["phase"] == "done" and st["picked"] == ["f2-s1", "m1-s0"]
    # render picks the winners of the same run up by default
    got = vc.resolve_spec({"entries": [], "refs": False}, "r1", dict,
                          lambda r: vc.read_json(tmp_path / "contest" / r / "screen" / "entries.json"))
    assert [e["name"] for e in got] == ["designed-f2-s1", "designed-m1-s0"]


def test_screen_cut_short_runs_no_clone_test_and_drops_old_picks(tmp_path, monkeypatch):
    out, argv = _screen(tmp_path, monkeypatch)
    assert vc.main(argv) == 0 and (out / "entries.json").is_file()
    clones = sorted((out / "clone" / "wav").rglob("*.wav"))
    # a third seed, with a deadline that leaves no time for any new candidate: the pool is partial
    more = [*argv[:-1], "0,1,2", "--deadline", f"{time.time() + 5:.0f}"]
    assert vc.main(more) == 0
    st = json.loads((out / "status.json").read_text())
    assert st["phase"] == "candidates incomplete" and st["unresolved"] == 3 and "picked" not in st
    assert not (out / "entries.json").exists()  # the earlier picks are not this pool's: no render takes them
    assert sorted((out / "clone" / "wav").rglob("*.wav")) == clones  # no clone test paid on a partial pool
    assert vc.resolve_spec({"entries": [], "refs": False}, "r1", dict,
                           lambda r: vc.read_json(tmp_path / "contest" / r / "screen" / "entries.json")) == []
    assert vc.main(more[:-2]) == 0  # the same command without the deadline finishes it
    st = json.loads((out / "status.json").read_text())
    assert st["phase"] == "done" and (out / "entries.json").is_file()
    assert len(list((out / "cand" / "wav").glob("*.wav"))) == 9


def test_screen_clone_gap_is_provisional_and_failing_candidates_get_two_tries(tmp_path, monkeypatch):
    descs = (("f1", "female"), ("f2", "female"), ("m1", "male"), ("x1", "male FAIL"))  # x1's prompts fail
    out, argv = _screen(tmp_path, monkeypatch, descs, no_sim={"f2-s1-c0"})
    assert vc.main(argv) == 0
    st = json.loads((out / "status.json").read_text())
    assert st["phase"] == "candidates incomplete" and st["unresolved"] == 2  # x1: first failure, tried again
    rows = {r["id"]: r for r in vc.read_rows(out / "cand" / "rows.jsonl")}
    assert rows["x1-s0"]["synth_fails"] == 1 and "fake failure" in rows["x1-s0"]["synth_error"]
    assert vc.main(argv) == 0  # second failure: x1 is left out, the clone test runs; one clone lacks similarity
    st = json.loads((out / "status.json").read_text())
    assert st["phase"] == "clone test incomplete" and st["unresolved"] == 1 and "picked" not in st
    rk = json.loads((out / "ranking.json").read_text())
    assert rk["provisional"] and rk["missing"] == ["clone f2-s1-c0"] and not (out / "entries.json").exists()
    assert "PROVISIONAL" in (out / "ranking.md").read_text(encoding="utf-8")
    _, argv = _screen(tmp_path, monkeypatch, descs)  # similarity for every clone now
    assert vc.main(argv) == 0
    assert json.loads((out / "status.json").read_text())["phase"] == "done" and (out / "entries.json").is_file()


# ---- Nano-vLLM backend (a fake server: no GPU, no nanovllm_voxcpm) ------------------------------------------------

# lora_config of /data/exp/voxcpm2-alania-s2/ckpt/step_0000600 (the official trainer's)
S2 = {"enable_lm": True, "enable_dit": True, "enable_proj": False, "r": 64, "alpha": 64, "dropout": 0.0,
      "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
      "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"],
      "target_proj_modules": ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj", "fusion_concat_proj"]}


class FakeNano:
    """Stands in for nanovllm_base.nano_server's async server pool, with the calls NanoPool makes; every call goes to
    LOG. generate yields one 0.16 s chunk (16 kHz) per step, ~15 characters a second as VoxCPM2 speaks; all of
    max_generate_length for a RUNAWAY text with seed 0 (a bad case); FAIL raises; SLOW hangs. fail_lora: register_lora
    refuses."""

    LOG: list = []
    fail_lora = False

    def __init__(self, model, ts, n, lora_kwargs):
        self.ts = ts
        FakeNano.LOG.append(("start", ts, n, lora_kwargs and lora_kwargs["max_lora_rank"]))

    async def wait_for_ready(self):
        pass

    async def register_lora(self, name, path):
        if FakeNano.fail_lora:
            raise ValueError("Unknown LoRA target module 'x'")
        FakeNano.LOG.append(("lora", name, path))

    async def get_model_info(self):
        return {"output_sample_rate": 16000, "feat_dim": 64, "patch_size": 4}

    async def encode_latents(self, wav, fmt):
        FakeNano.LOG.append(("ref", len(wav), fmt))
        return b"\0" * (4 * 64 * 4 * 5)  # 5 patches

    async def generate(self, target_text, max_generate_length, cfg_value, ref_audio_latents=None, lora_name=None,
                       seed=None):
        FakeNano.LOG.append(("gen", target_text, max_generate_length, ref_audio_latents is not None, lora_name, seed))
        if "FAIL" in target_text:
            raise RuntimeError("fake failure")
        if "SLOW" in target_text:
            await asyncio.sleep(60)
        steps = max_generate_length if "RUNAWAY" in target_text and seed == 0 else max(1, round(len(target_text) / 2.4))
        for _ in range(steps):
            await asyncio.sleep(0)
            yield [0.01] * 2560

    async def stop(self):
        FakeNano.LOG.append(("stop", self.ts))


def _words(model):
    return lambda text: len(text.split())


@pytest.fixture
def fake_nano(monkeypatch):
    """vc.nano_server / vc.nano_tokens -> FakeNano / word counts (what NanoPool calls when given none). NanoPool
    levels every clip and prepares every reference (refprep): numpy, soundfile and librosa."""
    for m in ("numpy", "soundfile", "librosa"):
        pytest.importorskip(m)
    FakeNano.LOG.clear()
    FakeNano.fail_lora = False
    monkeypatch.setattr(vc, "nano_server", FakeNano)
    monkeypatch.setattr(vc, "nano_tokens", _words)
    return FakeNano.LOG


def _kinds(log, kind):
    return [x for x in log if x[0] == kind]


def _lora_dir(d: Path, cfg: dict) -> str:
    d.mkdir(parents=True)
    (d / "lora_config.json").write_text(json.dumps({"base_model": "openbmb/VoxCPM2", "lora_config": cfg}))
    (d / "lora_weights.safetensors").write_bytes(b"x")
    return str(d)


def test_nano_groups_by_lora_and_timesteps_in_order():
    jobs = [{"key": "a/1", "lora": "/l1", "ts": 10}, {"key": "b/1", "lora": None, "ts": 10},
            {"key": "a/2", "lora": "/l1", "ts": 10}, {"key": "c/1", "lora": "", "ts": 10},
            {"key": "d/1", "lora": "/l1", "ts": 16}]
    g = vc.nano_groups(jobs)
    assert [k for k, _ in g] == [("/l1", 10), (None, 10), ("/l1", 16)]  # one server each, in order of first appearance
    assert [[j["key"] for j in js] for _, js in g] == [["a/1", "a/2"], ["b/1", "c/1"], ["d/1"]]


def test_nano_estimates_use_the_probe_speeds_and_later_measurements():
    assert vc.nano_wall_per_audio(10, False) == pytest.approx(1 / 53)
    assert vc.nano_wall_per_audio(10, True) == pytest.approx(1 / 49)
    assert vc.nano_wall_per_audio(16, False) == pytest.approx(1 / 38)
    assert vc.nano_wall_per_audio(16, True) == pytest.approx(1 / 36)
    assert 1 / 53 < vc.nano_wall_per_audio(13, False) < 1 / 38
    m = [{"backend": "nanovllm", "timesteps": 10, "ref": False, "jobs": 300, "concurrency": 64, "audio_s": 3000,
          "wall_s": 50},
         {"backend": "nanovllm", "timesteps": 10, "ref": False, "jobs": 70, "concurrency": 64, "audio_s": 500,
          "wall_s": 30},  # one batch: its last requests weigh too much
         {"timesteps": 10, "workers": 4, "audio_s": 600, "wall_s": 600}]  # the official API's pool (no backend key)
    assert vc.nano_wall_per_audio(10, False, m) == pytest.approx(50 / 3000)
    assert vc.nano_wall_per_audio(10, True, m) == pytest.approx(1 / 49)  # no reference measurement yet
    assert vc.wall_per_audio(10, 4, m) == pytest.approx(1.0)  # the voxcpm estimate ignores Nano-vLLM servers
    assert vc.wall_per_audio(10, 4, m[:2]) == pytest.approx(vc.rtf(10) / vc.speedup(4))
    start = vc.NANO_START_S + vc.NANO_STOP_S
    assert start == 70 and vc.NANO_DUR == 1.1
    assert vc.nano_server_s([10.0, 5.0], 10, [False, True]) == pytest.approx(
        start + 11 / 53 + 5.5 / 49 + 11 * 1.05)  # start + stop, its share of the busy server, the last batch
    assert vc.nano_server_s([10.0], 16, [False], start=False) == pytest.approx(11 / 38)
    assert vc.nano_server_s([], 10, []) == 0
    assert "53 audio s per GPU s without a reference, 49 with one, + 70 s per server" in vc.nano_speed_note(10)
    assert "(measured)" in vc.nano_speed_note(10, m)


def test_nano_fit_starts_a_group_only_with_what_fits():
    jobs = [{"est_s": 10.0, "ref": None}] * 100
    full = vc.nano_server_s([10.0] * 100, 10, [False] * 100)
    assert vc.nano_fit(jobs, 10, full) == jobs
    assert len(vc.nano_fit(jobs, 10, full - 0.1)) == 99
    assert vc.nano_fit(jobs, 10, vc.NANO_START_S + vc.NANO_STOP_S + 1) == []  # not even one clip: no server start


def test_plan_render_on_nanovllm_pays_one_server_per_lora():
    rows = [{"set": "s", "id": str(i), "model_text": "x" * 45} for i in range(10)]  # 3.3 s each
    es = [{"name": "s2", "lora": "/l2"}, {"name": "a", "ref": "/a.wav"}, {"name": "b", "ref": "/b.wav"},
          {"name": "s1", "lora": "/l1"}, {"name": "c", "ref": "/c.wav"}]
    eng = {"s1": "voxcpm"}  # a LoRA Nano-vLLM cannot load: the official API's pool
    plan = vc.plan_render(es, rows, {}, 10, 4, math.inf, backend="nanovllm", engines=eng)
    score = 10 * vc.SCORE_CLIP_S
    lora = vc.nano_server_s([3.3] * 10, 10, [False] * 10) + score
    first = vc.nano_server_s([3.3] * 10, 10, [True] * 10) + score  # the server of the entries without a LoRA
    more = vc.nano_server_s([3.3] * 10, 10, [True] * 10, start=False) + score
    pool = 33 * vc.wall_per_audio(10, 4) + vc.WORKERS_UP_S + score
    assert [p["est_s"] for p in plan] == pytest.approx([lora, first, more, pool, more], abs=0.06)
    assert [p["engine"] for p in plan] == ["nanovllm", "nanovllm", "nanovllm", "voxcpm", "nanovllm"]
    # made as synthesize() makes them (Nano-vLLM servers first, then the pool): a tight budget leaves out the last
    budget = vc.SCORE_LOAD_S + vc.SUMMARY_S + lora + first + more + 1
    tight = vc.plan_render(es, rows, {}, 10, 4, budget, backend="nanovllm", engines=eng)
    assert [p["fits"] for p in tight] == [True, True, True, False, False]
    done = {"s2": {f"s/{i}" for i in range(10)}}  # all made: no server for that LoRA
    assert vc.plan_render(es, rows, done, 10, 4, math.inf, backend="nanovllm", engines=eng)[0]["est_s"] == 0


def test_plan_screen_on_nanovllm():
    ct = list(vc.CLONE_TEXTS)
    cand = vc.est_audio_s(normalize(vc.SCREEN_TEXT))
    clones = [vc.est_audio_s(normalize(t)) for t in ct] * 30
    want = (vc.SCORE_LOAD_S + vc.SUMMARY_S + 420 * vc.SCORE_CLIP_S + vc.LOAD_S["whisper"] + vc.LOAD_S["utmos"]
            + vc.nano_server_s([cand] * 300, 10, [False] * 300) + vc.nano_server_s(clones, 10, [True] * 120))
    assert vc.plan_screen(300, 30, ct, 10, 4, backend="nanovllm") == pytest.approx(want)
    assert vc.plan_screen(300, 30, ct, 10, 4, backend="nanovllm") < vc.plan_screen(300, 30, ct, 10, 4) / 2.5
    assert vc.plan_screen(300, 30, ct, 16, 4, backend="nanovllm") > vc.plan_screen(300, 30, ct, 10, 4,
                                                                                    backend="nanovllm")
    assert vc.plan_screen(0, 30, ct, 10, 4, clones_done=120, backend="nanovllm") == vc.SCORE_LOAD_S + vc.SUMMARY_S


def test_nano_lora_support_and_the_fallback_choice():
    assert vc.nano_lora_problem(S2) is None  # our stage-2 LoRA: r 64, LM + DiT q/k/v/o: Nano-vLLM loads it
    assert vc.nano_lora_kwargs(S2)["max_lora_rank"] == 64
    assert "no readable lora_config.json" in vc.nano_lora_problem({"auto": True}) and vc.nano_lora_problem(None)
    assert "alpha" in vc.nano_lora_problem({"r": 64})
    assert "['lm_head']" in vc.nano_lora_problem({**S2, "target_modules_lm": ["q_proj", "lm_head"]})
    assert vc.nano_lora_problem({**S2, "enable_dit": False, "target_modules_dit": ["lm_head"]}) is None  # not built
    assert "target_proj_modules" in vc.nano_lora_problem({**S2, "enable_proj": True, "target_proj_modules": ["x"]})
    e = {"name": "x", "lora": "/l"}
    assert vc.entry_backend("nanovllm", e, S2) == ("nanovllm", None)
    eng, why = vc.entry_backend("nanovllm", e, {"auto": True})
    assert eng == "voxcpm" and why.startswith("Nano-vLLM cannot load LoRA /l: no readable lora_config.json")
    assert vc.entry_backend("nanovllm", {"name": "y"}, None) == ("nanovllm", None)  # no LoRA: never handed over
    assert vc.entry_backend("voxcpm", e, {"auto": True}) == ("voxcpm", None)
    old = {"lora": "/l", "fallback": "refused at register_lora", "settings": {"backend": "voxcpm"}}
    assert vc.entry_backend("nanovllm", e, S2, old) == ("voxcpm", "refused at register_lora")  # stays handed over
    assert vc.entry_backend("nanovllm", {**e, "lora": "/other"}, S2, old) == ("nanovllm", None)


def test_lora_config_reads_another_view_of_the_files():
    files = {"/v/l/lora_config.json": json.dumps({"lora_config": S2}), "/v/bad/lora_config.json": "[]"}
    assert vc.lora_config("/v/l", files.get) == S2
    assert vc.lora_config("/v/bad", files.get) == {"auto": True} == vc.lora_config("/v/none", files.get)


def _nano_job(tmp_path, key, text, ref=None, lora=None, seed=0):
    return {"key": key, "group": key.split("/")[0], "text": text, "ref": ref, "lora": lora, "cfg": 2.0, "ts": 10,
            "seed": seed, "out": str(tmp_path / "out" / f"{key}.wav"), "est_s": 1.0, "backend": "nanovllm"}


def test_nano_pool_groups_seeds_retries_and_stops_every_server(tmp_path, fake_nano):
    ref, lora = _wav(tmp_path / "ref.wav"), _lora_dir(tmp_path / "lora", S2)
    jobs = [_nano_job(tmp_path, "l/0", "bir iki üç dört beş altı yedi", lora=lora),
            _nano_job(tmp_path, "d/s0", "(A calm voice)metin bir", seed=0),
            _nano_job(tmp_path, "d/s1", "(A calm voice)metin bir", seed=1),
            _nano_job(tmp_path, "c/0", "RUNAWAY bir iki", ref=str(ref)),
            _nano_job(tmp_path, "c/1", "üç dört", ref=str(ref)), _nano_job(tmp_path, "c/bad", "FAIL")]
    done, fails = {}, {}
    pool = vc.NanoPool("m", 4, log=lambda m: None)
    st = pool.run(jobs, on_done=lambda j, i: done.update({j["key"]: i}),
                  on_fail=lambda j, e: fails.update({j["key"]: e}))
    assert _kinds(fake_nano, "start") == [("start", 10, 4, 64), ("start", 10, 4, None)]  # LoRA group first, rank 64
    kinds = [x[0] for x in fake_nano]
    assert kinds.count("stop") == 2 and kinds.index("stop") < kinds.index("start", 1)  # stopped before the next
    assert ("lora", "contest", lora) in fake_nano
    import refprep

    # the clip encoded once per server, prepared: 16 kHz float WAV right-padded to whole 160 ms patches
    assert _kinds(fake_nano, "ref") == [("ref", len(refprep.prep_ref(ref)), "wav")]
    gens = _kinds(fake_nano, "gen")
    assert sorted(x[5] for x in gens if x[1].startswith("(A calm")) == [0, 1]  # two seeds: two requests
    assert [(x[2], x[5]) for x in gens if x[1].startswith("RUNAWAY")] == [(28, 0), (28, 1)]  # 3 x 6 + 10; seed + 1
    assert all(x[3] == (x[1] in ("RUNAWAY bir iki", "üç dört")) for x in gens)  # reference latents: clones only
    assert all(x[4] == ("contest" if x[1].startswith("bir iki") else None) for x in gens)
    assert done["c/0"]["tries"] == 2 and "bad_case" not in done["c/0"] and done["l/0"]["engine"] == "nanovllm"
    assert all(i["onset_retries"] == 0 and not i["onset_trimmed"] and not i["onset_bad"] for i in done.values())
    assert set(fails) == {"c/bad"} and "fake failure" in fails["c/bad"]
    assert (st["done"], st["failed"], st["left"], len(st["groups"]), st["stopped"]) == (5, 1, 0, 2, "")
    assert st["groups"][1]["ref_jobs"] == 2 and st["audio_s"] > 0 and st["backend"] == "nanovllm"
    for k, info in done.items():
        with wave.open(str(tmp_path / "out" / f"{k}.wav")) as w:
            assert (w.getframerate(), w.getsampwidth(), w.getnchannels()) == (16000, 2, 1)
            assert w.getnframes() / 16000 == pytest.approx(info["duration"], abs=1e-3)
            assert w.readframes(1) == (3277).to_bytes(2, "little", signed=True)  # 0.01 -> -20 dBFS: 0.1 x 32767
    assert not list((tmp_path / "out").rglob("*.part"))


def test_nano_pool_retries_a_stray_onset_with_the_next_seeds_then_cuts_it(tmp_path, fake_nano):
    calls, logs, done = [], [], {}

    def onset(wav, sr, text):  # "yarın": flagged twice; "bir": always (cut at 0.1 s); "tamam": not covered
        calls.append(text)
        if text.startswith("tamam"):
            return {"covered": False, "bad": False, "why": []}
        bad = text.startswith("bir") or calls.count(text) <= 2
        return {"covered": True, "bad": bad, "why": ["stray"] if bad else [], "cut_s": 0.1}

    jobs = [_nano_job(tmp_path, "a/y", "yarın sabah", seed=7), _nano_job(tmp_path, "a/b", "bir iki üç", seed=9),
            _nano_job(tmp_path, "a/t", "tamam", seed=3), _nano_job(tmp_path, "a/r", "RUNAWAY yarın", seed=0)]
    st = vc.NanoPool("m", 4, log=logs.append, onset=onset).run(jobs, on_done=lambda j, i: done.update({j["key"]: i}))
    assert st["done"] == 4 and st["failed"] == 0
    seeds = {}
    for x in _kinds(fake_nano, "gen"):
        seeds.setdefault(x[1], []).append(x[5])
    assert seeds["yarın sabah"] == [7, 8, 9] and seeds["tamam"] == [3]  # flagged twice: seed + 1, seed + 2
    assert seeds["bir iki üç"] == [9 + k for k in range(vc.NANO_TRIES + 1)]  # NANO_TRIES more tries, then cut
    assert seeds["RUNAWAY yarın"] == [0, 1, 2, 3]  # a bad case (seed 0) is not checked; then two flagged tries
    y, b, t, r = done["a/y"], done["a/b"], done["a/t"], done["a/r"]
    assert (y["onset_bad"], y["onset_retries"], y["onset_trimmed"], y["tries"]) == (True, 2, False, 3)
    assert y["onset_why"] == ["stray"] and (r["onset_retries"], r["tries"]) == (2, 4) and "bad_case" not in r
    assert (b["onset_bad"], b["onset_retries"], b["onset_trimmed"]) == (True, vc.NANO_TRIES, True)
    assert (t["onset_bad"], t["onset_retries"], t["onset_trimmed"], t.get("tries")) == (None, 0, False, None)
    steps = round(len("bir iki üç") / 2.4)  # FakeNano: 0.16 s a step; cut 30 ms before 0.1 s, 10 ms fade-in
    assert b["duration"] == pytest.approx(steps * 0.16 - 0.07, abs=1e-3)
    with wave.open(str(tmp_path / "out" / "a" / "b.wav")) as w:
        first = [int.from_bytes(w.readframes(1), "little", signed=True) for _ in range(161)]
    assert first[0] == 0 and 0 < first[80] < first[159] < 3277 and first[160] == 3277
    assert any("a/b: stray onset (stray) after 3 more tries: cut before 0.07 s" in m for m in logs)


def test_nano_pool_encodes_a_raw_and_a_prepared_reference_apart(tmp_path, fake_nano):
    import refprep

    ref = _wav(tmp_path / "ref.wav", seconds=0.3)
    jobs = [_nano_job(tmp_path, "p/0", "bir", ref=str(ref)), {**_nano_job(tmp_path, "r/0", "iki", ref=str(ref)),
                                                              "ref_prep": False}]
    st = vc.NanoPool("m", 4, log=lambda m: None).run(jobs)
    assert st["done"] == 2
    assert sorted(_kinds(fake_nano, "ref")) == sorted([("ref", len(refprep.prep_ref(ref)), "wav"),
                                                       ("ref", ref.stat().st_size, "wav")])


def test_nano_pool_cleans_pauses_before_the_level_on_jobs_that_ask(tmp_path, fake_nano, monkeypatch):
    import numpy as np

    seen = []

    def fake_clean(x, sr):  # halves the clip: the level after it brings it back to -20 dBFS
        seen.append((round(float(np.abs(x).max()), 4), sr))
        return x * 0.5, {"pauses": 2, "breaths": 1, "gated_s": 0.05}

    monkeypatch.setattr(vc, "clean_pauses", fake_clean)
    jobs = [{**_nano_job(tmp_path, "c/0", "bir iki üç"), "pause_clean": True}, _nano_job(tmp_path, "r/0", "dört beş")]
    done = {}
    st = vc.NanoPool("m", 4, log=lambda m: None).run(jobs, on_done=lambda j, i: done.update({j["key"]: i}))
    assert st["done"] == 2 and seen == [(0.01, 16000)]  # FakeNano's raw 0.01 (levelled: 0.1): before the level
    assert done["c/0"]["pause_clean"] == {"pauses": 2, "breaths": 1, "gated_s": 0.05}
    assert "pause_clean" not in done["r/0"]
    for k in ("c/0", "r/0"):
        with wave.open(str(tmp_path / "out" / f"{k}.wav")) as w:
            assert w.readframes(1) == (3277).to_bytes(2, "little", signed=True)  # both at -20 dBFS
    assert done["c/0"]["level_in"] == pytest.approx(-46.02, abs=0.01)  # the halved clip's level: measured after it


def test_nano_pool_budget_hard_deadline_and_a_refused_lora(tmp_path, fake_nano):
    pool = vc.NanoPool("m", 4, log=lambda m: None)
    jobs = [_nano_job(tmp_path, f"a/{i}", "bir iki") for i in range(3)]
    st = pool.run(jobs, feed_until=time.time() + 30)  # less than a server start: none is started
    assert (st["done"], st["left"]) == (0, 3) and "deadline" in st["stopped"] and not fake_nano
    t = time.time()
    st = pool.run([_nano_job(tmp_path, "s/0", "SLOW"), _nano_job(tmp_path, "s/1", "bir iki")],
                  hard_until=time.time() + 2)
    assert time.time() - t < 30 and "hard deadline" in st["stopped"] and (st["done"], st["left"]) == (1, 1)
    assert _kinds(fake_nano, "stop") in ([], [("stop", 10)])  # hard deadline: killed, no graceful stop
    fake_nano.clear()
    FakeNano.fail_lora = True
    lora = _lora_dir(tmp_path / "lora", S2)
    st = pool.run([_nano_job(tmp_path, "l/0", "bir", lora=lora), _nano_job(tmp_path, "b/0", "iki")])
    assert st["lora_failed"] == {lora: "ValueError: Unknown LoRA target module 'x'"} and st["lora_jobs"] == 1
    assert st["done"] == 1 and st["failed"] == 0 and len(_kinds(fake_nano, "stop")) == 2  # the refusing one too


def test_log_throughput_one_line_per_nano_server(tmp_path):
    p = tmp_path / "t.jsonl"
    groups = [{"lora": None, "timesteps": 10, "jobs": 300, "ref_jobs": 0, "concurrency": 64, "done": 300,
               "ready_s": 50.0, "up_s": 140.0, "audio_s": 3300.0, "wall_s": 70.0, "audio_per_wall": 47.1},
              {"lora": "/l", "timesteps": 10, "jobs": 3, "ref_jobs": 3, "concurrency": 64, "done": 3, "audio_s": 20.0,
               "wall_s": 5.0}]  # too little audio to tell
    vc.log_throughput(str(p), "render", 10, {"backend": "nanovllm", "groups": groups, "gpu": "A100",
                                             "gpu_mem_peak_mib": 69000.0})
    recs = vc.read_rows(p)
    assert len(recs) == 1 and recs[0]["backend"] == "nanovllm" and recs[0]["ref"] is False
    assert (recs[0]["jobs"], recs[0]["concurrency"], recs[0]["gpu_mem_peak_mib"]) == (300, 64, 69000.0)
    assert vc.nano_wall_per_audio(10, False, recs) == pytest.approx(70 / 3300)
    vc.log_throughput(str(p), "render", 10, {"backend": "voxcpm", "workers": 4, "audio_s": 100, "wall_s": 100})
    assert vc.read_rows(p)[-1]["backend"] == "voxcpm" and vc.wall_per_audio(10, 4, vc.read_rows(p)) == 1.0


def _files(d: Path) -> list[str]:
    return sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())


def test_render_on_nanovllm_writes_what_the_other_backends_write(tmp_path, fake_nano):
    argv = _render_setup(tmp_path)  # fake backend -> run/r1
    assert vc.main(argv) == 0
    nano = [*argv, "--backend", "nanovllm"]
    nano[2] = str(tmp_path / "run" / "n1")
    assert vc.main(nano) == 0
    a, b = tmp_path / "run" / "r1", tmp_path / "run" / "n1"
    assert _files(a) == _files(b) and "masha/wav/emo/e1.wav" in _files(b)  # same files: scoring, resume, fetch
    keys = ("set", "id", "prompt", "wav", "ref")
    ra, rb = vc.read_rows(a / "masha" / "rows.jsonl"), vc.read_rows(b / "masha" / "rows.jsonl")
    assert [{k: r.get(k) for k in keys} for r in ra] == [{k: r.get(k) for k in keys} for r in rb]
    assert all(r["duration"] > 0 and r["engine"] == "nanovllm" for r in rb)
    assert vc.read_json(b / "masha" / "entry.json")["settings"]["backend"] == "nanovllm"
    gens = _kinds(fake_nano, "gen")  # reference-only cloning of masha's clip; the emotional row keeps "(instruction)"
    assert sorted(x[1] for x in gens) == sorted(r["prompt"] for r in rb) and all(x[3] for x in gens)
    assert "(Angry tone.)Yeter artık!" in [x[1] for x in gens]
    assert vc.main(nano) == 0 and len(_kinds(fake_nano, "gen")) == len(gens)  # resume: nothing made again
    assert vc.main([*nano[:-1], "voxcpm"]) == 0  # its clips were made by another engine: refused, kept
    st = vc.read_json(b / "status.json")
    assert "use another --run" in st["errors"]["masha"] and "backend" in st["errors"]["masha"]


def test_render_seeds_each_text_alike_in_every_voice_and_records_onsets(tmp_path, fake_nano):
    argv = _render_setup(tmp_path)
    _wav(tmp_path / "refs" / "dima" / "neutral.wav")
    (tmp_path / "refs" / "dima" / "sources.json").write_text(json.dumps(SRC))
    run = [*argv, "--backend", "nanovllm", "--seed", "11"]
    assert vc.main(run) == 0
    gens = _kinds(fake_nano, "gen")
    assert len(gens) == 6  # 2 voices x 3 texts, no stray onset in FakeNano's clips: one request each
    by_text = {}
    for x in gens:
        by_text.setdefault(x[1], []).append(x[5])
    texts = {r["prompt"]: r for r in vc.entry_rows(vc.load_texts(Path(argv[argv.index("--texts") + 1])))}
    assert {t: v for t, v in by_text.items()} == {t: [vc.text_seed(11, r["set"], r["id"])] * 2 for t, r in texts.items()}
    assert len({v[0] for v in by_text.values()}) == 3  # another seed for each sentence
    out = Path(run[2])
    for name in ("masha", "dima"):
        meta = vc.read_json(out / name / "entry.json")
        assert meta["settings"]["seed_mode"] == "per_text" and meta["settings"]["loudness"] == "-20dBFS"
        rows = vc.read_rows(out / name / "rows.jsonl")
        assert all(r["onset_retries"] == 0 and r["onset_trimmed"] is False and "onset_bad" in r for r in rows)
        assert vc.read_json(out / name / "summary.json")["onset"]["covered"] >= 1
    assert "onset stray" in (out / "leaderboard.md").read_text(encoding="utf-8")
    assert vc.main([*run[:-1], "12"]) == 0  # another --seed: other clips, refused in this run
    assert "seed" in vc.read_json(out / "status.json")["errors"]["masha"]


def test_render_hands_lora_entries_nano_vllm_cannot_take_to_the_official_api(tmp_path, fake_nano):
    argv = _render_setup(tmp_path)
    good = _lora_dir(tmp_path / "exp" / "good", S2)
    bad = _lora_dir(tmp_path / "exp" / "bad", {**S2, "target_modules_lm": ["q_proj", "lm_head"]})
    spec = tmp_path / "loras.json"
    spec.write_text(json.dumps({"entries": [{"name": "g", "kind": "lora", "lora": good},
                                            {"name": "b", "kind": "lora", "lora": bad}], "refs": False}))
    run = [*argv, "--backend", "nanovllm", "--spec", str(spec)]
    run[2] = str(tmp_path / "run" / "f1")
    out = Path(run[2])
    assert vc.main(run) == 0
    eb = vc.read_json(out / "b" / "entry.json")  # cannot be loaded (lm_head): the official API's pool takes it
    assert eb["settings"]["backend"] == "voxcpm" and "Nano-vLLM cannot load LoRA" in eb["fallback"]
    st = vc.read_json(out / "status.json")
    assert set(st["fallback"]) == {"b"} and st["synth_render-voxcpm"]["jobs"] == 3  # (no voxcpm here: none made)
    assert vc.read_json(out / "g" / "entry.json")["settings"]["backend"] == "nanovllm"
    assert len(list((out / "g" / "wav").rglob("*.wav"))) == 3 and ("lora", "contest", good) in fake_nano
    assert _kinds(fake_nano, "start") == [("start", 10, 64, 64)]  # one server: b never went to Nano-vLLM
    # a LoRA the server refuses at run time: handed over too, and later calls keep it there
    fake_nano.clear()
    FakeNano.fail_lora = True
    run[2] = str(tmp_path / "run" / "f2")
    out = Path(run[2])
    assert vc.main(run) == 0
    eg = vc.read_json(out / "g" / "entry.json")
    assert eg["settings"]["backend"] == "voxcpm" and "Nano-vLLM could not take LoRA" in eg["fallback"]
    st = vc.read_json(out / "status.json")
    assert good in st["synth_render"]["lora_failed"] and st["synth_render-voxcpm"]["jobs"] == 6  # g and b
    fake_nano.clear()
    FakeNano.fail_lora = False
    assert vc.main(run) == 0 and not _kinds(fake_nano, "start")  # g stays on the official API
    assert vc.read_json(out / "g" / "entry.json")["fallback"] == eg["fallback"]


def test_screen_on_nanovllm_makes_the_same_files_with_a_request_per_seed(tmp_path, monkeypatch, fake_nano):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    out_a, argv_a = _screen(tmp_path / "a", monkeypatch)
    assert vc.main(argv_a) == 0
    out, argv = _screen(tmp_path / "b", monkeypatch)
    assert vc.main([*argv, "--backend", "nanovllm"]) == 0
    assert _files(out_a) == _files(out)
    rk = json.loads((out / "ranking.json").read_text())
    assert rk["picked"] == ["f2-s1", "m1-s0"] and not rk["provisional"]
    assert vc.read_json(out / "settings.json")["backend"] == "nanovllm"
    assert len(_kinds(fake_nano, "start")) == 2 and len(_kinds(fake_nano, "stop")) == 2  # a server per round
    gens = _kinds(fake_nano, "gen")
    cand = sorted((x[1].split(")")[0], x[5]) for x in gens if not x[3])
    # a seed per description, SEED_STRIDE apart per --seeds value: no shared first-step noise, retries never meet
    assert cand == sorted((f"(A {g} voice, {i}.", vc.text_seed(vc.SEED_STRIDE * s, "cand", i))
                          for i, g in (("f1", "female"), ("f2", "female"), ("m1", "male")) for s in (0, 1))
    assert len({x[5] for x in gens if not x[3]}) == 6
    clones = [x for x in gens if x[3]]
    assert len(clones) == 3 * len(vc.CLONE_TEXTS)  # clones: reference-only, no description
    assert all("(" not in x[1] for x in clones) and len(_kinds(fake_nano, "ref")) == 3
    by_text = {}
    for x in clones:  # a clone text's seed: the same in every voice, another for each text
        by_text.setdefault(x[1], set()).add(x[5])
    assert all(len(v) == 1 for v in by_text.values()) and len({min(v) for v in by_text.values()}) == len(by_text)


def test_modal_app_contest_runs_on_the_nano_image_and_passes_the_backend():
    import importlib.util
    from unittest import mock

    spec = importlib.util.spec_from_file_location("modal_app_under_test", VOXCPM / "modal_app.py")
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
    cmd = ma.contest_cmd("render", "r1", ["--limit", "1"], 100.0, "/s.json")
    assert cmd[cmd.index("--backend") + 1] == "nanovllm" and cmd[-2:] == ["--spec", "/s.json"]
    assert ma.contest_cmd("screen", "r1", [], 1.0, backend="voxcpm")[-2:] == ["--backend", "voxcpm"]
    with pytest.raises(ValueError, match="--backend"):
        ma.contest_cmd("screen", "r1", [], 1.0, backend="fake")
    base = mm.Image.debian_slim.return_value.apt_install.return_value.run_commands
    assert any(f"nano-vllm-voxcpm @ git+{ma.nanovllm_base.NANO_REPO}@{ma.nanovllm_base.NANO_COMMIT}" in c
               for call in base.call_args_list for c in call.args)  # nanovllm_app.py's layers
    layer = base.return_value.run_commands.call_args.args[0]
    assert "speechbrain==1.0.3" in layer and ma.nanovllm_base.TORCH_PINS in layer  # torch 2.8.0+cu128 kept
    fn = [c for c in mm.App.return_value.function.call_args_list if c.kwargs.get("memory") == ma.CONTEST_MEM]
    assert len(fn) == 1 and fn[0].kwargs["image"] is ma.contest_image is not ma.image


def test_nano_group_lora_errors_only_from_the_lora_and_no_late_start(tmp_path, fake_nano, monkeypatch):
    lora = _lora_dir(tmp_path / "l", S2)
    job = {"key": "a/1", "text": "bir iki üç", "cfg": 2.0, "seed": 0, "est_s": 1.0, "out": str(tmp_path / "a.wav")}
    pool = vc.NanoPool("m", 4, log=lambda m: None)

    class Oom(FakeNano):
        async def get_model_info(self):
            raise RuntimeError("CUDA out of memory")

    pool.server = Oom
    g = asyncio.run(pool._group(lora, 10, [job], math.inf, math.inf, None, {}))
    assert "lora_error" not in g and "out of memory" in g["error"]  # not the LoRA's fault: no fallback
    FakeNano.fail_lora, pool.server = True, FakeNano
    g = asyncio.run(pool._group(lora, 10, [job], math.inf, math.inf, None, {}))
    assert "Unknown LoRA" in g["lora_error"]
    fake_nano.clear()  # too close to feed_until to start and stop a server: none started, the job left
    g = asyncio.run(pool._group(None, 10, [job], time.time() + vc.NANO_STOP_S / 2, math.inf, None, {}))
    assert g["left"] == 1 and not _kinds(fake_nano, "start")


def test_nano_stop_at_the_hard_deadline_kills_without_a_graceful_stop(fake_nano):
    class Proc:
        killed = joined = 0

        def kill(self):
            Proc.killed += 1

        def join(self, t):
            Proc.joined = t

        def is_alive(self):
            return True

    server = FakeNano("m", 10, 4, None)
    server.process = Proc()
    asyncio.run(vc.NanoPool("m", 4, log=lambda m: None)._stop(server, kill=True))
    assert Proc.killed == 1 and Proc.joined == 5 and not _kinds(fake_nano, "stop")
