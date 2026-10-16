"""tts/voxcpm eval: stratified --limit sampling, UTMOS aggregation and rescoring, synth_eval's write order (stubbed
backends: no torch, no voxcpm, no faster-whisper, no jiwer)."""

import json
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VOXCPM = REPO / "tts" / "voxcpm"
sys.path[:0] = [str(VOXCPM), str(REPO / "src")]

import pause_clean  # noqa: E402  (no numpy at import)
import score_utmos  # noqa: E402
import synth_eval  # noqa: E402
from score_utmos import utmos_summary  # noqa: E402
from synth_eval import load_sets, stratified  # noqa: E402

TRZOR_ORDER = ["sayi", "sira", "tarih", "saat", "para", "yuzde", "birim", "numara", "kisaltma", "akronim", "kisa",
               "soru", "ozel"]


def _rows(cats: str) -> list[dict]:
    return [{"id": f"{c}{i}", "category": c} for i, c in enumerate(cats)]


def test_import_needs_no_torch():
    code = (f"import sys; sys.path[:0] = [{str(VOXCPM)!r}, {str(REPO / 'src')!r}]; import synth_eval, score_utmos; "
            "bad = {'torch', 'torchaudio', 'voxcpm', 'soundfile', 'numpy'} & set(sys.modules); assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_help_lists_new_flags():
    out = subprocess.run([sys.executable, str(VOXCPM / "synth_eval.py"), "--help"], capture_output=True, text=True,
                         check=True).stdout
    assert "--utmos, --no-utmos" in out and "--sample {stratified,head}" in out and "--pause-clean" in out


def test_stratified_round_robin_in_file_order():
    rows = _rows("aaabcc")  # a0 a1 a2 b3 c4 c5
    assert [r["id"] for r in stratified(rows, 3)] == ["a0", "b3", "c4"]
    assert [r["id"] for r in stratified(rows, 4)] == ["a0", "a1", "b3", "c4"]  # 2nd round starts with a
    assert [r["id"] for r in stratified(rows, 5)] == ["a0", "a1", "b3", "c4", "c5"]  # b is exhausted
    assert stratified(rows, 6) == rows and stratified(rows, 100) == rows
    assert stratified(rows, 0) == []


def test_stratified_single_category_is_head():
    rows = _rows("hhhhh")
    assert stratified(rows, 3) == rows[:3]


def test_trzor_limit_30_covers_every_category():
    rows = load_sets(None, ["trzor"], 30)
    assert rows == load_sets(None, ["trzor"], 30)  # deterministic
    cats = Counter(r["category"] for r in rows)
    assert len(rows) == 30 and set(cats) == set(TRZOR_ORDER)
    # 2 rounds over 13 categories = 26, then the 3rd row of the first 4 categories in file order
    assert {c for c, n in cats.items() if n == 3} == set(TRZOR_ORDER[:4])
    assert all(n == 2 for c, n in cats.items() if c not in TRZOR_ORDER[:4])
    # file order kept: categories appear as contiguous runs in the file's order
    assert list(dict.fromkeys(r["category"] for r in rows)) == TRZOR_ORDER


def test_trzor_limit_13_is_first_row_of_each_category():
    full = load_sets(None, ["trzor"], 0)
    firsts = [next(r for r in full if r["category"] == c) for c in TRZOR_ORDER]
    assert load_sets(None, ["trzor"], 13) == firsts


def test_trzor_head_keeps_old_behaviour():
    full = load_sets(None, ["trzor"], 0)
    assert len(full) == 155
    head = load_sets(None, ["trzor"], 30, "head")
    assert head == full[:30] and {r["category"] for r in head} == {"sayi", "sira", "tarih"}


def test_heldout_takes_first_n(tmp_path):
    val = tmp_path / "voxcpm_val.jsonl"
    val.write_text("".join(json.dumps({"audio": f"/d/wav/antalia-{i:04d}.wav", "text": f"cümle {i}"}) + "\n"
                           for i in range(6)), encoding="utf-8")
    for sample in ("stratified", "head"):
        rows = load_sets(val, ["trzor", "heldout"], 4, sample)
        held = [r for r in rows if r["set"] == "heldout"]
        assert [r["id"] for r in held] == [f"heldout_{i:04d}" for i in range(4)]
        assert sum(r["set"] == "trzor" for r in rows) == 4
    assert len([r for r in load_sets(val, ["trzor", "heldout"], 0) if r["set"] == "heldout"]) == 6


def test_utmos_summary_groups_like_wer():
    rows = [{"set": "trzor", "category": "sayi", "utmos": 4.0}, {"set": "trzor", "category": "sayi", "utmos": 3.0},
            {"set": "trzor", "category": "saat", "utmos": 3.1234}, {"set": "trzor", "category": "saat"},
            {"set": "trzor", "category": "para", "utmos": None},
            {"set": "heldout", "category": "heldout", "utmos": 4.2}]
    s = utmos_summary(rows)
    assert list(s) == sorted(s)
    assert s == {"heldout": {"mean": 4.2, "rows": 1}, "heldout/heldout": {"mean": 4.2, "rows": 1},
                 "trzor": {"mean": round((4.0 + 3.0 + 3.1234) / 3, 3), "rows": 3},
                 "trzor/saat": {"mean": 3.123, "rows": 1}, "trzor/sayi": {"mean": 3.5, "rows": 2}}
    assert utmos_summary([]) == {} and utmos_summary(rows[3:5]) == {}


def _eval_dir(d: Path, rows: list[dict], wavs: list[dict], summary: dict) -> None:
    for r in wavs:
        (d / "wav" / r["set"]).mkdir(parents=True, exist_ok=True)
        (d / "wav" / r["set"] / f"{r['id']}.wav").write_bytes(b"")
    d.mkdir(parents=True, exist_ok=True)
    (d / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _read(d: Path) -> tuple[list[dict], dict]:
    return ([json.loads(x) for x in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines()],
            json.loads((d / "summary.json").read_text(encoding="utf-8")))


def test_rescore_dir(tmp_path, monkeypatch):
    d = tmp_path / "step_0000600"
    rows = [{"set": "trzor", "id": "sayi_01", "category": "sayi", "wav": "/data/exp/x/wav/trzor/sayi_01.wav",
             "errors": 0, "utmos": 1.0},
            {"set": "trzor", "id": "saat_01", "category": "saat", "wav": "/data/exp/x/wav/trzor/saat_01.wav"},
            {"set": "heldout", "id": "heldout_ab", "category": "heldout", "wav": "/data/x.wav", "utmos": 4.0}]
    _eval_dir(d, rows, rows[:2], {"rtf": 0.87, "wer": {"trzor": {"wer": 4.2}}, "utmos_error": "HTTPError: 403"})
    seen = []

    def fake_score(predictor, path, device):
        seen.append(Path(path).relative_to(d).as_posix())
        return {"sayi_01.wav": 3.98761, "saat_01.wav": 3.5}[Path(path).name]

    monkeypatch.setattr(score_utmos, "score_wav", fake_score)
    summ = score_utmos.rescore(d, None, "cpu")
    assert seen == ["wav/trzor/sayi_01.wav", "wav/trzor/saat_01.wav"]  # local wav/<set>/<id>.wav, not row["wav"]
    assert b"\r\n" not in (d / "rows.jsonl").read_bytes()
    out, summary = _read(d)
    assert [r.get("utmos") for r in out] == [3.9876, 3.5, 4.0]  # WAV not here: the GPU-computed score is kept
    assert out[0]["errors"] == 0 and out[0]["wav"] == rows[0]["wav"]
    assert summary["wer"] == {"trzor": {"wer": 4.2}} and summary["rtf"] == 0.87
    assert summary["utmos"] == summ == {
        "heldout": {"mean": 4.0, "rows": 1}, "heldout/heldout": {"mean": 4.0, "rows": 1},
        "trzor": {"mean": 3.744, "rows": 2}, "trzor/saat": {"mean": 3.5, "rows": 1},
        "trzor/sayi": {"mean": 3.988, "rows": 1}}
    assert "utmos_error" not in summary  # every row has a score now: the container's failure note is stale
    assert sorted(p.name for p in d.iterdir()) == ["rows.jsonl", "summary.json", "wav"]  # no .tmp left


def test_rescore_bad_clip_and_unscored_rows(tmp_path, monkeypatch):
    d = tmp_path / "e"
    rows = [{"set": "trzor", "id": f"r{i}", "category": "sayi", "utmos": 2.0} for i in range(3)]
    rows.append({"set": "heldout", "id": "h0", "category": "heldout"})  # no WAV, no score
    _eval_dir(d, rows, rows[:3], {"wer": {}})

    def fake_score(predictor, path, device):
        if Path(path).stem == "r1":
            raise RuntimeError("Calculated padded input size per channel: (1). Kernel size: (2)")
        return 3.0

    monkeypatch.setattr(score_utmos, "score_wav", fake_score)
    score_utmos.rescore(d, None, "cpu")
    out, summary = _read(d)
    assert [r.get("utmos") for r in out] == [3.0, None, 3.0, None]  # r1 failed alone, r2 still scored
    assert out[1]["utmos_error"].startswith("RuntimeError: Calculated padded") and "utmos_error" not in out[0]
    assert summary["utmos"]["trzor"] == {"mean": 3.0, "rows": 2} and "heldout" not in summary["utmos"]
    note = summary["utmos_error"]
    assert note.startswith("1 rows not scored") and "trzor/r1: RuntimeError" in note
    assert "1 rows not scored: WAV not under" in note


def test_rescore_without_wavs_changes_nothing(tmp_path, monkeypatch):
    d = tmp_path / "hf-copy"  # the HF eval repo holds *.json/*.jsonl only
    _eval_dir(d, [{"set": "trzor", "id": "a", "category": "sayi", "utmos": 3.9}], [], {"utmos": {"trzor": 3.9}})
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    with pytest.raises(FileNotFoundError):
        score_utmos.rescore(d, None, "cpu")

    def no_load(device):
        raise AssertionError("predictor loaded before the dirs were checked")

    monkeypatch.setattr(score_utmos, "load_predictor", no_load)
    with pytest.raises(SystemExit):
        score_utmos.main([str(d)])
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


def test_cli_rejects_dir_without_rows(tmp_path):
    with pytest.raises(SystemExit):
        score_utmos.main([str(tmp_path)])


class _Killed(BaseException):
    """Stands in for the SIGKILL at modal_app's budget deadline: no `except Exception` catches it."""


@pytest.fixture
def fake_backends(monkeypatch):
    """synth_eval.main without numpy/soundfile/torch/voxcpm/faster-whisper/jiwer: 1 s of 'audio' per row."""
    class Tts:
        tts_model = types.SimpleNamespace(sample_rate=10)

        def generate(self, text, **kw):
            return [0.0] * 10

    class Judge:
        def __init__(self, *a, **kw):
            pass

        def transcribe(self, wav, **kw):
            return [types.SimpleNamespace(text=" ok ")], None

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


def _run_eval(out: Path, *extra: str) -> int:
    return synth_eval.main(["--model", "m", "--sets", "trzor", "--limit", "13", "--out", str(out), *extra])


def test_eval_results_on_disk_before_utmos(tmp_path, monkeypatch, fake_backends):
    def killed(device):  # e.g. the cold hub download is still running at the deadline
        raise _Killed

    monkeypatch.setattr(score_utmos, "load_predictor", killed)
    with pytest.raises(_Killed):
        _run_eval(tmp_path)
    rows, summary = _read(tmp_path)
    assert len(rows) == 13 and all(r["errors"] == 1 and "utmos" not in r for r in rows)
    assert summary["wer"]["trzor"] == {"wer": 25.0, "rows": 13, "rows_correct": 0} and summary["rtf"] >= 0
    assert summary["utmos_error"].startswith("not scored: the run ended before UTMOS") and "utmos" not in summary
    assert (summary["limit"], summary["sample"], summary["seed"]) == (13, "stratified", 0)
    assert not list(tmp_path.glob("*.tmp"))


def test_eval_utmos_load_failure_keeps_wer(tmp_path, monkeypatch, fake_backends):
    def offline(device):
        raise TimeoutError("timed out")

    monkeypatch.setattr(score_utmos, "load_predictor", offline)
    assert _run_eval(tmp_path) == 0
    rows, summary = _read(tmp_path)
    assert summary["utmos"] == {} and summary["utmos_error"] == "TimeoutError: timed out"
    assert summary["wer"]["trzor"]["rows"] == 13


def test_eval_utmos_bad_clip_costs_one_row(tmp_path, monkeypatch, fake_backends):
    bad = next(r["id"] for r in load_sets(None, ["trzor"], 13) if r["category"] == "sayi")
    monkeypatch.setattr(score_utmos, "load_predictor", lambda device: object())

    def fake_score(predictor, path, device):
        if Path(path).stem == bad:
            raise RuntimeError("cannot reshape tensor of 0 elements")
        return 4.0

    monkeypatch.setattr(score_utmos, "score_wav", fake_score)
    assert _run_eval(tmp_path) == 0
    rows, summary = _read(tmp_path)
    assert sum("utmos" in r for r in rows) == 12 and [r["id"] for r in rows if "utmos_error" in r] == [bad]
    assert summary["utmos"]["trzor"] == {"mean": 4.0, "rows": 12} and "trzor/sayi" not in summary["utmos"]
    assert summary["utmos"]["trzor/ozel"] == {"mean": 4.0, "rows": 1}  # the last category was still scored
    assert summary["utmos_error"].startswith("1 rows not scored") and summary["wer"]["trzor"]["rows"] == 13


def test_eval_no_utmos(tmp_path, monkeypatch, fake_backends):
    monkeypatch.setattr(score_utmos, "load_predictor", lambda device: pytest.fail("UTMOS ran with --no-utmos"))
    assert _run_eval(tmp_path, "--no-utmos", "--no-judge") == 0
    rows, summary = _read(tmp_path)
    assert len(rows) == 13 and summary["wer"] == {} and "utmos" not in summary and "utmos_error" not in summary


def test_eval_pause_clean_before_the_write_and_the_judge(tmp_path, monkeypatch, fake_backends):
    calls, writes, judged = [], [], []

    def fake_clean(wav, sr):
        calls.append(sr)
        return ["cleaned"] * len(wav), {"pauses": 2, "breaths": 1, "gated_s": 0.25, "regions": [{"t0": 0.1}]}

    monkeypatch.setattr(pause_clean, "clean", fake_clean)
    monkeypatch.setattr(sys.modules["soundfile"], "write",
                        lambda path, data, sr: (writes.append(list(data)), Path(path).write_bytes(b"RIFF")))
    judge = sys.modules["faster_whisper"].WhisperModel
    monkeypatch.setattr(judge, "transcribe", lambda self, wav, **kw: (judged.append(len(writes)) or
                                                                      ([types.SimpleNamespace(text=" ok ")], None)))
    assert _run_eval(tmp_path / "pc", "--pause-clean", "--no-utmos") == 0
    rows, summary = _read(tmp_path / "pc")
    assert calls == [10] * 13 and writes == [["cleaned"] * 10] * 13  # the cleaned clip is what is written
    assert judged == [13] * 13  # and judged: every WAV was written (cleaned) before the judge ran
    assert all(r["pause_clean"] == {"pauses": 2, "breaths": 1, "gated_s": 0.25} for r in rows)  # no regions in rows
    assert summary["pause_clean"] == {"settings": pause_clean.SETTINGS, "clips": 13, "pauses": 26, "breaths": 13,
                                      "gated_s": 3.25}
    assert all(r["duration"] == 1.0 for r in rows) and summary["wer"]["trzor"]["rows"] == 13
    calls.clear()
    assert _run_eval(tmp_path / "plain", "--no-utmos", "--no-judge") == 0  # default off: no call, no record, None
    rows, summary = _read(tmp_path / "plain")
    assert calls == [] and summary["pause_clean"] is None and not any("pause_clean" in r for r in rows)
