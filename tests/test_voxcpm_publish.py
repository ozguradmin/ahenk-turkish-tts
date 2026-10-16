"""tts/voxcpm/publish_hf.py: model card rendering and upload guards, on fake files (stdlib only, never the network)."""

import importlib.util
import json
import math
import sys
import types
import wave
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("publish_hf", REPO / "tts" / "voxcpm" / "publish_hf.py")
pub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pub)

ATTRIBUTION = ("Antalia (Patientdesk.ai), Antalia Turkish single-speaker scripted speech, CC BY 4.0, "
               "https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus")
QKVO = ["q_proj", "v_proj", "k_proj", "o_proj"]


def write_safetensors(path: Path, tensors: dict) -> None:
    """{name: shape} as float32 zeros, in the safetensors layout (u64 header length, JSON header, data)."""
    header, off = {}, 0
    for name, shape in tensors.items():
        n = 4 * math.prod(shape)
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(len(h).to_bytes(8, "little") + h + bytes(off))


def fake_lora(d: Path, r: int = 32, cfg_r: int | None = None) -> Path:
    d.mkdir(parents=True)
    write_safetensors(d / "lora_weights.safetensors", {"base_lm.layers.0.self_attn.q_proj.lora_A": (r, 16),
                                                       "base_lm.layers.0.self_attn.q_proj.lora_B": (16, r)})
    lc = {"enable_lm": True, "enable_dit": True, "enable_proj": False, "r": cfg_r or r, "alpha": cfg_r or r,
          "dropout": 0.0, "target_modules_lm": QKVO, "target_modules_dit": QKVO,
          "target_proj_modules": ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"]}
    (d / "lora_config.json").write_text(json.dumps({"base_model": "openbmb/VoxCPM2", "lora_config": lc}))
    (d / "training_state.json").write_text('{"step": 600}')
    return d


def write_wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(bytes(2 * int(8000 * seconds)))


def fake_eval(d: Path, summary: dict, rows: list[tuple]) -> Path:
    """rows: (set, id, category, duration, wav present?); rows.jsonl keeps a container path like the real one."""
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(summary))
    with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
        for s, rid, cat, dur, present in rows:
            f.write(json.dumps({"set": s, "id": rid, "category": cat, "text": f"metin {rid} <b>", "duration": dur,
                                "wav": f"/data/exp/x/eval/{d.name}/wav/{s}/{rid}.wav", "errors": 0}) + "\n")
            if present:
                write_wav(d / "wav" / s / f"{rid}.wav", dur)
    return d


OURS = {"model": "/data/models/VoxCPM2", "lora": "/data/exp/voxcpm2-antalia/ckpt/step_0000600", "cfg": 2.0,
        "timesteps": 10, "rtf": 0.8726,
        "wer": {"heldout": {"wer": 1.9, "rows": 20, "rows_correct": 15},
                "trzor": {"wer": 4.2, "rows": 155, "rows_correct": 131},
                "trzor/sayi": {"wer": 4.49, "rows": 15, "rows_correct": 12},
                "trzor/saat": {"wer": 4.76, "rows": 12, "rows_correct": 9}},
        "utmos": {"trzor": {"mean": 3.912, "rows": 155}}}  # held-out UTMOS missing on purpose
REF = {"model": "/data/models/Trendyol-TTS", "lora": "", "cfg": 2.0, "timesteps": 16, "rtf": 1.3745,
       "wer": {"heldout": {"wer": 0.68, "rows": 20}, "trzor": {"wer": 3.98, "rows": 155, "rows_correct": 134},
               "trzor/saat": {"wer": 1.6, "rows": 12, "rows_correct": 11}}}  # no utmos, no held-out rows_correct
ROWS = [("trzor", "sayi_01", "sayi", 1.0, True), ("trzor", "sayi_02", "sayi", 3.0, True),
        ("trzor", "sayi_03", "sayi", 2.5, True), ("trzor", "saat_01", "saat", 2.2, True),
        ("trzor", "saat_02", "saat", 2.1, False), ("trzor", "kisa_01", "kisa", 1.1, True),
        ("heldout", "heldout_aa", "heldout", 14.0, True), ("heldout", "heldout_bb", "heldout", 13.0, True),
        ("heldout", "heldout_cc", "heldout", 15.0, True)]


@pytest.fixture
def staged(tmp_path):
    lora = fake_lora(tmp_path / "voxcpm-lora" / "step_0000600")
    ours = fake_eval(tmp_path / "eval" / "step_0000600", OURS, ROWS)
    ref = fake_eval(tmp_path / "eval" / "trendyol", REF, [])
    out = tmp_path / "hf-publish" / "test-ses"
    assert pub.main(["--name", "Test Ses", "--repo", "ozgurguler/test-ses", "--lora", str(lora), "--evals", str(ours),
                     str(ref), "--samples", "4", "--out", str(out)]) == 0
    return out


def table_row(card: str, start: str) -> list[str]:
    line = next(x for x in card.splitlines() if x.startswith(start))
    return [c.strip() for c in line.strip().strip("|").split("|")]


@pytest.fixture
def no_network(monkeypatch):
    """A huggingface_hub stand-in that records calls; HfApi() raises unless a test configures it."""
    calls, state = [], {"exists": None, "private": True}

    class FakeApi:
        def __init__(self):
            if state["exists"] is None:
                raise AssertionError("HfApi must not be reached")

        def repo_exists(self, repo, repo_type=None):
            return state["exists"]

        def repo_info(self, repo, repo_type=None):
            return types.SimpleNamespace(private=state["private"])

        def __getattr__(self, name):  # create_repo, upload_folder, update_repo_settings
            return lambda *a, **kw: calls.append((name, a, kw))

    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return calls, state


def test_render_fills_every_placeholder(staged):
    card = (staged / "README.md").read_text(encoding="utf-8")
    assert "{{" not in card and "}}" not in card
    assert card.startswith("---\nlanguage:\n  - tr\nlicense: cc-by-4.0\nbase_model: openbmb/VoxCPM2\n")
    assert "# Test Ses — Türkçe ses" in card
    assert ATTRIBUTION in card and pub.attribution() == ATTRIBUTION
    assert 'snapshot_download("ozgurguler/test-ses"' in card
    assert "cfg_value=2.0, inference_timesteps=10, normalize=False" in card
    assert f"VoxCPM@{pub.VOXCPM_COMMIT}" in card
    assert "LoRA r=32, alpha=32" in card and "LM (q_proj, v_proj, k_proj, o_proj) + DiT" in card
    assert "`voxcpm2-antalia/step_0000600` (adım / step 600)" in card
    assert pub.check_staging(staged) == []
    assert sorted(p.name for p in staged.iterdir()) == ["README.md", "lora_config.json", "lora_weights.safetensors",
                                                         "samples"]


def test_numbers_from_summaries_and_dash_for_missing(staged):
    card = (staged / "README.md").read_text(encoding="utf-8")
    ours = table_row(card, "| **Test Ses**")
    assert ours[1:] == ["10 / 2.0", "4.20", "131/155", "1.90", "15/20", "3.91", "—", "0.87"]
    ref = table_row(card, "| [Trendyol-TTS]")
    assert ref[1:] == ["16 / 2.0", "3.98", "134/155", "0.68", "—", "—", "—", "1.37"]
    base = table_row(card, "| VoxCPM2, ince ayarsız")
    assert base[2:5] == ["5.96", "—", "3.80"]
    # category table: in TR-Zor file order, "—" where a reference has no number
    assert table_row(card, "| sayı / numbers")[1:] == ["4.49 (12/15)", "—"]
    assert table_row(card, "| saat / times")[1:] == ["4.76 (9/12)", "1.60 (11/12)"]
    assert card.index("| sayı / numbers") < card.index("| saat / times")
    assert "RTF {{" not in card and "RTF 0.87 (A100 80 GB)" in card


def test_samples_mix_sets_prefer_long_enough_and_skip_missing_wavs(staged):
    names = sorted(p.name for p in (staged / "samples").glob("*.wav"))
    # 4 // 3 = 1 held-out (shortest >= 2 s), then one per TR-Zor category in file order: sayi (2.5 s, not the
    # 1.0 s one), saat (2.2 s; saat_02 has no WAV), kisa (nothing >= 2 s: its shortest)
    assert names == ["heldout_bb.wav", "trzor_kisa_01.wav", "trzor_saat_01.wav", "trzor_sayi_03.wav"]
    card = (staged / "README.md").read_text(encoding="utf-8")
    assert "https://huggingface.co/ozgurguler/test-ses/resolve/main/samples/trzor_sayi_03.wav" in card
    assert "metin sayi_03 &lt;b&gt;" in card  # sample text is HTML-escaped


def test_upload_refuses_without_lora_files(staged, no_network):
    (staged / "lora_weights.safetensors").unlink()
    with pytest.raises(SystemExit, match="lora_weights.safetensors missing"):
        pub.upload(staged, "ozgurguler/test-ses")
    with pytest.raises(SystemExit, match="refusing to upload"):
        pub.main(["--repo", "ozgurguler/test-ses", "--out", str(staged), "--upload"])
    assert no_network[0] == []


def test_upload_refuses_placeholders_and_broken_weights(staged, no_network):
    readme = staged / "README.md"
    text = readme.read_text(encoding="utf-8")
    readme.write_text(text.replace("Test Ses", "{{name}}", 1), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"placeholders: \{\{name\}\}"):
        pub.upload(staged, "ozgurguler/test-ses")
    readme.write_text(text, encoding="utf-8")
    w = staged / "lora_weights.safetensors"
    w.write_bytes(w.read_bytes()[:-4])  # truncated download
    with pytest.raises(SystemExit, match="truncated"):
        pub.upload(staged, "ozgurguler/test-ses")
    assert no_network[0] == []


def test_upload_refuses_card_staged_for_another_repo(staged, no_network):
    with pytest.raises(SystemExit, match="not staged for ozgurguler/other"):
        pub.upload(staged, "ozgurguler/other")
    assert no_network[0] == []


def test_upload_private_by_default_and_only_staged_files(staged, no_network):
    calls, state = no_network
    state["exists"] = False
    (staged / "notes.txt").write_text("not for the Hub")
    pub.upload(staged, "ozgurguler/test-ses")
    (name, a, kw), (name2, _, kw2) = calls
    assert name == "create_repo" and a == ("ozgurguler/test-ses",) and kw["private"] is True
    assert name2 == "upload_folder" and kw2["allow_patterns"] == ["README.md", "lora_weights.safetensors",
                                                                  "lora_config.json", "samples/*.wav"]
    # the Hub's samples/ and package files mirror the staged ones (no package files here: remote ones go)
    assert kw2["delete_patterns"] == ["samples/*.wav", "*.py", "instructions_tr.json"]


def test_upload_never_into_public_repo_without_public(staged, no_network):
    calls, state = no_network
    state.update(exists=True, private=False)
    with pytest.raises(SystemExit, match="PUBLIC"):
        pub.upload(staged, "ozgurguler/test-ses")
    assert calls == []
    state["private"] = True  # existing private repo + --public: upload, then flip visibility
    pub.upload(staged, "ozgurguler/test-ses", public=True)
    assert [c[0] for c in calls] == ["upload_folder", "update_repo_settings"]
    assert calls[1][2]["private"] is False


def test_stage_refuses_eval_of_another_checkpoint_and_rank_mismatch(tmp_path):
    lora = fake_lora(tmp_path / "step_0000600")
    other = fake_eval(tmp_path / "eval" / "step_0001000", {**OURS, "lora": "/data/exp/x/ckpt/step_0001000"}, [])
    with pytest.raises(SystemExit, match="step_0001000"):
        pub.main(["--name", "X", "--repo", "u/x", "--lora", str(lora), "--evals", str(other),
                  "--out", str(tmp_path / "o")])
    bad = fake_lora(tmp_path / "r" / "step_0000600", r=32, cfg_r=64)  # the r=32 run vs today's r=64 default
    with pytest.raises(ValueError, match="r=64 but lora_A tensors have rank"):
        pub.lora_info(bad)


def test_stage_needs_the_experiment_of_the_eval(tmp_path, capsys):
    lora = fake_lora(tmp_path / "voxcpm-lora" / "step_0000600")  # fetch-ckpt names dirs by checkpoint only
    s2 = fake_eval(tmp_path / "eval" / "step_0000600",
                   {**OURS, "lora": "/data/exp/voxcpm2-antalia-s2/ckpt/step_0000600"}, [])
    argv = ["--name", "X", "--repo", "u/x", "--lora", str(lora), "--evals", str(s2), "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="exp/voxcpm2-antalia/ckpt/step_0000600"):
        pub.main(argv)
    assert pub.main(argv + ["--exp-name", "voxcpm2-antalia-s2"]) == 0
    assert "measured with /data/exp/voxcpm2-antalia-s2/ckpt/step_0000600" in capsys.readouterr().out
    assert "`voxcpm2-antalia-s2/step_0000600`" in (tmp_path / "o" / "README.md").read_text(encoding="utf-8")


def test_missing_lora_files_print_a_runnable_fetch_command(tmp_path):
    ev = fake_eval(tmp_path / "eval" / "step_0000600", OURS, [])
    with pytest.raises(SystemExit, match=r"python -m modal run tts/voxcpm/modal_app.py --action fetch-ckpt "
                                         r"--fetch step_0000600$"):
        pub.main(["--name", "X", "--repo", "u/x", "--lora", str(tmp_path / "step_0000600"), "--evals", str(ev),
                  "--out", str(tmp_path / "o")])


def test_failed_restage_leaves_no_card_to_upload(staged, tmp_path, monkeypatch, no_network):
    lora16 = fake_lora(tmp_path / "s2" / "step_0000600", r=16)
    argv = ["--name", "Test Ses", "--repo", "ozgurguler/test-ses", "--lora", str(lora16), "--evals",
            str(tmp_path / "eval" / "step_0000600"), "--out", str(staged)]

    def boom(*a, **kw):
        raise RuntimeError("stopped halfway")
    monkeypatch.setattr(pub, "pick_samples", boom)  # after the new weights are copied, before the card
    with pytest.raises(RuntimeError):
        pub.main(argv)
    assert pub.sha256(staged / "lora_weights.safetensors") == pub.sha256(lora16 / "lora_weights.safetensors")
    assert not (staged / "README.md").exists()
    with pytest.raises(SystemExit, match="README.md missing"):
        pub.upload(staged, "ozgurguler/test-ses")
    assert no_network[0] == []


@pytest.mark.skipif(sys.platform != "win32", reason="open files block deletion only on Windows")
def test_restage_while_a_sample_is_open_stops_without_a_card(staged, tmp_path):
    lora16 = fake_lora(tmp_path / "b" / "step_0000600", r=16)
    argv = ["--name", "Test Ses", "--repo", "ozgurguler/test-ses", "--lora", str(lora16), "--evals",
            str(tmp_path / "eval" / "step_0000600"), "--out", str(staged)]
    with open(next((staged / "samples").glob("*.wav")), "rb"):  # a media player playing it
        with pytest.raises(SystemExit, match="cannot clear"):
            pub.main(argv)
    assert not (staged / "README.md").exists() and pub.check_staging(staged)[0].endswith("missing (stage first)")
    assert pub.main(argv) == 0 and pub.check_staging(staged) == []  # closed: re-stage works
    assert "LoRA r=16, alpha=16" in (staged / "README.md").read_text(encoding="utf-8")


F022 = {**OURS, "lora": "/data/exp/voxcpm2-f022-s2/ckpt/step_0001500", "timesteps": 16, "rtf": 1.2093,
        "wer": {**OURS["wer"], "soru": {"wer": 0.0, "rows": 35, "rows_correct": 35}},
        "intonation": {"soru": {"items": 35, "passed": 20, "questions_pass_rate": 0.44, "auc_peak": 0.963, "pairs": 8,
                                "pair_acc": 1.0}}}


def test_f022_card_renders_fully_with_its_label_and_intonation(tmp_path):
    lora = fake_lora(tmp_path / "voxcpm-lora" / "voxcpm2-f022-s2" / "step_0001500", r=64)
    exp = tmp_path / "eval" / "voxcpm2-f022-s2"
    ours = fake_eval(exp / "step_0001500@ts16-cfg2", F022, ROWS)
    ts10 = fake_eval(exp / "step_0001500@ts10-cfg2", {**F022, "timesteps": 10, "limit": 40, "sample": "stratified"}, [])
    prev = fake_eval(tmp_path / "eval" / "voxcpm2-alania-s2" / "step_0000600@ts16-cfg2",
                     {**OURS, "lora": "/data/exp/voxcpm2-alania-s2/ckpt/step_0000600", "timesteps": 16}, [])
    ref = fake_eval(tmp_path / "eval" / "trendyol", REF, [])
    out = tmp_path / "hf-publish" / "ahenk-tts-tr"
    argv = ["--name", "Ahenk", "--repo", "ozgurguler/ahenk-tts-tr", "--exp-name", "voxcpm2-f022-s2",
            "--lora", str(lora), "--evals", str(ours), f"{ts10}=Ahenk, 10 adım / 10 steps",
            f"{prev}=Önceki sürüm (Antalia sesi) / previous version (Antalia voice)", str(ref), "--samples", "3"]
    template = REPO / "tts" / "voxcpm" / "MODEL_CARD.f022.template.md"
    assert pub.main(argv + ["--template", str(template), "--out", str(out)]) == 0
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "{{" not in card and "}}" not in card
    assert pub.check_staging(out) == []
    header = card.split("\n---\n", 1)[0]
    assert "cloud0day3/alania-domain-text-tr" in header and "antalia" not in header
    assert pub.attribution() not in card  # the Antalia voice is not in this model
    assert "LoRA r=64, alpha=64" in card and "`voxcpm2-f022-s2/step_0001500`" in card
    assert 'snapshot_download("ozgurguler/ahenk-tts-tr"' in card
    assert "cfg_value=2.0, inference_timesteps=16, normalize=False" in card
    assert 'prompt = (f"({instruction})" if instruction else "") + normalize(text)' in card
    assert "`Angry tone, voice getting louder.`" in card and '"Laughing heartily while speaking."' in card
    ours_rows = [x for x in card.splitlines() if x.startswith("| **Ahenk** (")]
    assert len(ours_rows) == 2 and "(VoxCPM2 + f022 LoRA (designed voice), `step_0001500`)" in ours_rows[0]
    cells = [c.strip() for c in ours_rows[0].strip().strip("|").split("|")]
    assert cells[1:] == ["16 / 2.0", "4.20", "131/155", "1.90", "15/20", "3.91", "—", "1.21"]
    cells = [c.strip() for c in ours_rows[1].strip().strip("|").split("|")]  # intonation: pairs, AUC, questions, WER
    assert cells[1:] == ["16 / 2.0", "8/8", "0.96", "0.44", "0.00", "35/35"]
    ts10_rows = [x for x in card.splitlines() if x.startswith("| Ahenk, 10 adım / 10 steps — örneklem")]
    assert len(ts10_rows) == 2 and "| 10 / 2.0 | 8/8 | 0.96 |" in ts10_rows[1]  # results row, then intonation row
    assert sum(x.startswith("| Önceki sürüm") for x in card.splitlines()) == 1  # no intonation: results row only
    # the label is chosen per run: the default template right after the f022 one gets the Antalia label back
    assert pub.main(argv + ["--out", str(tmp_path / "o2")]) == 0
    assert "(VoxCPM2 + Antalia LoRA, `step_0001500`)" in (tmp_path / "o2" / "README.md").read_text(encoding="utf-8")


S3 = {**F022, "lora": "/data/exp/voxcpm2-f022-s3tr/ckpt/step_0000500",
      "pause_clean": {"settings": "v1/hp80/L-55dB/max25dB", "clips": 9, "pauses": 40, "breaths": 17, "gated_s": 2.345}}
VOXCPM_DIR = REPO / "tts" / "voxcpm"
S3_TEMPLATE = VOXCPM_DIR / "MODEL_CARD.f022s3.template.md"
PACKAGE = ["instruction_pick.py", "instructions_tr.json", "onset.py", "pause_clean.py"]


@pytest.fixture
def staged_s3(tmp_path):
    lora = fake_lora(tmp_path / "voxcpm-lora" / "voxcpm2-f022-s3tr" / "step_0000500", r=64)
    ours = fake_eval(tmp_path / "eval" / "voxcpm2-f022-s3tr" / "step_0000500@ts16-cfg2", S3, ROWS)
    prev = fake_eval(tmp_path / "eval" / "voxcpm2-f022-s2" / "step_0001500@ts16-cfg2",
                     {**F022, "lora": "/data/exp/voxcpm2-f022-s2/ckpt/step_0001500"}, [])
    out = tmp_path / "hf-publish" / "ahenk-tts-tr"
    argv = ["--name", "Ahenk", "--repo", "ozgurguler/ahenk-tts-tr", "--exp-name", "voxcpm2-f022-s3tr",
            "--lora", str(lora), "--evals", str(ours), f"{prev}=Önceki sürüm / previous version", "--samples", "3",
            "--template", str(S3_TEMPLATE), "--out", str(out)]
    assert pub.main(argv) == 0
    return out, argv


def _untodo(out: Path) -> None:
    """The review step: every TODO(final) marker filled in (here: just taken out)."""
    readme = out / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8").replace(pub.TODO_MARK + ")", "Not"), encoding="utf-8")


def test_template_key_takes_the_longest_part_and_bundle_follows_local_imports():
    assert pub.template_key(S3_TEMPLATE) == "f022s3" and pub.template_key("x/MODEL_CARD.f022.template.md") == "f022"
    assert pub.template_key(pub.TEMPLATE) is None
    assert pub.template_key("MODEL_CARD.two-stage.template.md") == "two-stage"
    assert pub.TEMPLATE_LABELS["f022s3"] == "VoxCPM2 + f022-s3tr LoRA (designed voice)"
    assert pub.local_imports(VOXCPM_DIR / "pause_clean.py") == {"onset"}  # numpy, scipy, soundfile: not ours
    assert pub.local_imports(VOXCPM_DIR / "onset.py") == set()
    assert pub.local_imports(VOXCPM_DIR / "instruction_pick.py") == set()  # stdlib only
    assert pub.bundle(pub.TEMPLATE_EXTRAS["f022s3"]) == ["instruction_pick.py", "pause_clean.py", "onset.py",
                                                         "instructions_tr.json"]
    with pytest.raises(SystemExit, match="not found"):
        pub.bundle(["nope.py"])


def test_training_instructions_and_their_turkish_table(tmp_path):
    ins = pub.training_instructions()
    assert list(ins) == ["angry", "cheerful", "sad", "surprised", "calm", "empathetic", "laughter"]
    assert all(len(v) == 8 for k, v in ins.items() if k != "laughter") and len(ins["laughter"]) == 3
    assert ins["angry"][0] == "Angry tone, voice getting louder."
    assert "Mildly irritated, speaking curtly." not in ins["angry"]  # a held-out wording: never trained
    table = json.loads((VOXCPM_DIR / "instructions_tr.json").read_text(encoding="utf-8"))
    md = pub.instructions_tr_md()
    rows, note = md.split("\n\n")
    assert rows.count("\n") == 1 + len(ins)  # header, rule, one row per emotion
    trained = sum(len(v) for v in ins.values())  # the other wordings: held-out and contest ones
    assert f"{len(table)} ifadeden eğitimde kullanılan {trained} ifadeyi" in note
    assert f"the other {len(table) - trained} (held-out" in note and "\n" not in note
    assert all(f"`{table[e]}`" in md for v in ins.values() for e in v)
    assert ("| kızgın / angry | `Kızgın bir ses tonuyla, sesi giderek yükselerek.` | "
            "`Angry tone, voice getting louder.` |") in md
    del table["Giggling, amused."]
    (short := tmp_path / "instructions_tr.json").write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"no Turkish wording for \['Giggling, amused.'\]"):
        pub.instructions_tr_md(short)


def test_f022s3_card_ships_the_package_files_and_shows_their_usage(staged_s3, no_network):
    out, _ = staged_s3
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "{{" not in card and "}}" not in card
    for n in PACKAGE:  # copied as they are, each listed with its sha256
        assert (out / n).read_bytes() == (VOXCPM_DIR / n).read_bytes()
        assert f"| `{n}` |" in card and f"`{pub.sha256(out / n)}` |" in card
    assert pub.listed_extras(card) == {n: pub.sha256(out / n) for n in PACKAGE}
    assert "(VoxCPM2 + f022-s3tr LoRA (designed voice), `step_0000500`)" in card
    assert "`voxcpm2-f022-s3tr/step_0000500`" in card and "LoRA r=64, alpha=64" in card
    # usage: trnorm front-end, instruction_pick picks the instruction, pause_clean after generate
    code = card.split("```python\n", 1)[1].split("```", 1)[0]
    assert ('snapshot_download("ozgurguler/ahenk-tts-tr", allow_patterns=["lora_*", "*.py", '
            '"instructions_tr.json"])') in code
    assert "from instruction_pick import compose, pick" in code and "from pause_clean import clean" in code
    assert "from trnorm import normalize" in code
    assert "instruction, model_text, why = pick(emotion, normalize(text), prefer)" in code
    assert "prompt = compose(instruction, model_text)" in code
    assert (code.index("pick(emotion") < code.index("tts.generate(text=prompt")
            < code.index("wav, report = clean(wav, SR)") < code.index("sf.write("))
    assert "cfg_value=2.0, inference_timesteps=16, normalize=False" in code
    # the comments show what pick makes of the examples
    sys.path.insert(0, str(VOXCPM_DIR))
    import instruction_pick as ip
    for emo, text in (("surprised", "Vay canına! Bu manzarayı hiç beklemiyordum!"),
                      ("angry", "Yeter artık! Bu sorun hâlâ çözülmedi!")):
        assert f'say("{text}", "{emo}"' in code and f"#    {ip.compose(*ip.pick(emo, text)[:2])}\n" in code
    acele = "Acele etmenize gerek yok, her şeyi birlikte halledeceğiz."
    assert f'say("{acele}", "calm", "sakin.wav")' in code and ip.pick("calm", acele).instruction.startswith("Calm")
    assert f'say("{acele}", "calm", "sakin_tr.wav", prefer="tr")' in code and '"laughing"' in code
    table = json.loads((VOXCPM_DIR / "instructions_tr.json").read_text(encoding="utf-8"))
    assert '"Biraz üzgün ve yorgun, enerjisiz."' in code and "Biraz üzgün ve yorgun, enerjisiz." in table.values()
    assert pub.instructions_tr_md() in card and '`pick` talimatlı metinde modele "Vaay" yazar' in card
    # the decision and the measured facts, said plainly; synthetic-voice label; licences
    for s in ("vey kenına", "Asele", "Aacele", "Açele", "Cem", "1/14", "13/14", "6/6", "10/10", "2/2", "8/20", "7/20",
              "18'iyle", "18 of the 20", 'prefer="tr"', 'prefer="en"', "kızgın 312, neşeli 284, üzgün 190 Hz",
              "325, 341, 223 Hz", "8'de 7", "birincisi 1/2", "/z/ ya da /tʃ/", "öbürü /aː/", "3 \"vay\", 1 \"vaa\"",
              "2. turda \"üçü de aynı gibi\"", "Speaking softly through tears", "31/108 → 0", "4/4",
              "from one container to the next", "1,015 sa", "yayınlanmadı", "not judged by ear"):
        assert s in card, s
    assert "`instruction_pick.py`, `pause_clean.py`, `onset.py`, `instructions_tr.json` **Apache-2.0**" in card
    assert "**This is a synthetic voice.**" in card and "**Bu bir sentetik (yapay) sestir.**" in card
    assert "**CC-BY-4.0**" in card and "`instructions_tr.json` **Apache-2.0**" in card
    assert "through `pause_clean.py` after synthesis (9 clips, 17 breaths turned down, 2.3 s in all; settings " \
           "`v1/hp80/L-55dB/max25dB`)" in card
    assert "(9 klip, 17 nefes kısıldı, toplam 2,3 sn; ayar" in card  # Turkish decimal comma
    assert pub.attribution() not in card  # no Antalia voice
    # the card is final (no TODO markers); without --extra-samples its listener-picked list says so
    assert pub.TODO_MARK not in card and "Örnek yok / no samples" in card
    assert pub.check_staging(out) == []
    no_network[1]["exists"] = False
    pub.upload(out, "ozgurguler/ahenk-tts-tr")
    (_, _, kw), (_, _, kw2) = no_network[0]
    assert kw2["allow_patterns"] == ["README.md", "lora_weights.safetensors", "lora_config.json", "samples/*.wav",
                                     *PACKAGE]
    assert kw2["delete_patterns"] == ["samples/*.wav", "*.py", "instructions_tr.json"]


def test_extra_samples_are_staged_linked_and_checked(tmp_path):
    import soundfile as sf
    d = tmp_path / "picked"
    d.mkdir()
    for n in ("a_s0.wav", "b_s1.wav"):
        sf.write(d / n, [0.0] * 4800, 48000)
    (d / "picked.jsonl").write_text(
        json.dumps({"title": "Şaşkın / surprised", "instruction": "Şaşkın, yükselen bir tonlamayla.",
                    "text": "Vaay canına!", "wav": "a_s0.wav"}, ensure_ascii=False) + "\n"
        + json.dumps({"title": "Gülerek / laughing", "text": "Hahaha, olur mu öyle!", "wav": "b_s1.wav"},
                     ensure_ascii=False) + "\n", encoding="utf-8")
    rows = pub.load_extra_samples(str(d / "picked.jsonl"))
    assert [Path(r["wav"]).name for r in rows] == ["a_s0.wav", "b_s1.wav"]
    md = pub.extra_samples_md(rows, "ozgurguler/x")
    assert "- **Şaşkın / surprised** — `(Şaşkın, yükselen bir tonlamayla.)`: Vaay canına!" in md
    assert '<audio controls src="https://huggingface.co/ozgurguler/x/resolve/main/samples/b_s1.wav"></audio>' in md
    assert "- **Gülerek / laughing**: Hahaha" in md
    (d / "bad.jsonl").write_text(json.dumps({"title": "x", "text": "y", "wav": "missing.wav"}) + "\n",
                                 encoding="utf-8")
    with pytest.raises(ValueError):
        pub.load_extra_samples(str(d / "bad.jsonl"))
    assert pub.extra_samples_md([], "r").startswith("Örnek yok")


def test_check_staging_ties_the_package_files_to_the_card(staged_s3):
    out, _ = staged_s3
    _untodo(out)
    onset = (out / "onset.py").read_bytes()
    (out / "onset.py").write_bytes(onset + b"\n# edited after staging\n")
    assert pub.check_staging(out) == [f"onset.py differs from the sha256 README.md gives "
                                      f"({pub.sha256(VOXCPM_DIR / 'onset.py')[:16]}...): stage again"]
    (out / "onset.py").unlink()
    problems = pub.check_staging(out)
    assert "README.md lists onset.py, which is not staged: stage again" in problems
    assert "pause_clean.py imports onset, not staged: stage again" in problems
    (out / "onset.py").write_bytes(onset)
    (out / "notes.py").write_text("x = 1\n")  # not in the card's table
    (out / "extra.json").write_text("{")
    problems = " | ".join(pub.check_staging(out))
    assert "does not list (delete them or stage again): extra.json, notes.py" in problems
    assert "extra.json unreadable: JSONDecodeError" in problems
    (out / "notes.py").unlink()
    (out / "extra.json").unlink()
    assert pub.check_staging(out) == []
    table = (out / "instructions_tr.json").read_bytes()
    (out / "instructions_tr.json").write_text('["not", "a table"]')
    problems = pub.check_staging(out)
    assert "instructions_tr.json is not a {english wording: turkish wording} table" in problems
    (out / "instructions_tr.json").write_bytes(table)
    (out / "pause_clean.py").write_text("def broken(:\n")
    assert any(p.startswith("pause_clean.py unreadable: SyntaxError") for p in pub.check_staging(out))


def test_check_staging_ties_instruction_pick_to_the_table(staged_s3):
    out, _ = staged_s3
    _untodo(out)
    src = (out / "instruction_pick.py").read_bytes().decode("utf-8")
    sha = pub.sha256(out / "instruction_pick.py")[:16]
    (out / "instruction_pick.py").write_bytes(src.replace("Kızgın bir ses tonuyla", "Öfkeyle").encode("utf-8"))
    assert pub.check_staging(out) == [
        f"instruction_pick.py differs from the sha256 README.md gives ({sha}...): stage again",
        "instruction_pick.py EMOTIONS angry: the Turkish wording is not instructions_tr.json's for the English one"]
    (out / "instruction_pick.py").write_bytes(src.replace("EMOTIONS: dict", "EMO: dict").encode("utf-8"))
    assert "instruction_pick.py: EMOTIONS not found in instruction_pick.py" in pub.check_staging(out)
    (out / "instruction_pick.py").write_bytes(src.replace('"tr": "Üzgün', '"de": "Üzgün').encode("utf-8"))
    assert any("EMOTIONS is not a" in p for p in pub.check_staging(out))
    (out / "instruction_pick.py").write_bytes(src.encode("utf-8"))
    assert pub.check_staging(out) == []


def test_restage_with_another_template_drops_the_package_files(staged_s3, tmp_path):
    out, argv = staged_s3
    argv = argv[:argv.index("--template")] + ["--template", str(VOXCPM_DIR / "MODEL_CARD.f022.template.md"),
                                              "--out", str(out)]
    assert pub.main(argv) == 0
    assert not pub.staged_extras(out) and pub.check_staging(out) == []
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "(VoxCPM2 + f022 LoRA (designed voice), `step_0000500`)" in card and "pause_clean" not in card
    assert pub.main(argv[:-4] + ["--template", str(S3_TEMPLATE), "--out", str(out)]) == 0  # and back
    assert [p.name for p in pub.staged_extras(out)] == PACKAGE


def test_pause_note_without_pause_clean():
    assert "did not go through `pause_clean.py`" in pub.pause_note(F022)
    assert "went through `pause_clean.py`" in pub.pause_note(S3)


def test_intonation_table_is_a_dash_without_intonation():
    ev = {"dir": Path("trendyol"), "label": "", "summary": REF, "rows": []}
    assert pub.intonation_table([ev], "X", "step_0000600") == pub.DASH


def test_check_staging_ties_the_card_to_the_staged_files(staged, tmp_path):
    other = fake_lora(tmp_path / "x" / "step_0000600", r=16)
    for f in pub.LORA_FILES:  # new weights under the old card (what a half-finished re-stage used to leave)
        (staged / f).write_bytes((other / f).read_bytes())
    problems = " | ".join(pub.check_staging(staged))
    assert "does not name the staged weights" in problems and "README.md says LoRA r/alpha 32/32" in problems
    assert "r=16, alpha=16" in problems
    for f in pub.LORA_FILES:
        (staged / f).write_bytes((tmp_path / "voxcpm-lora" / "step_0000600" / f).read_bytes())
    assert pub.check_staging(staged) == []
    (staged / "samples" / "heldout_bb.wav").unlink()
    write_wav(staged / "samples" / "trzor_old_01.wav", 1.0)  # left over from another stage
    problems = pub.check_staging(staged)
    assert any("not staged: heldout_bb.wav" in p for p in problems)
    assert any("does not link (delete them or stage again): trzor_old_01.wav" in p for p in problems)
