"""tts/small/publish_cep.py (Ahenk Cep model repo staging) and the release pages, on fake files (stdlib only)."""

import ast
import importlib.util
import wave
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("publish_cep", REPO / "tts" / "small" / "publish_cep.py")
cep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cep)


def write_wav(path: Path, seconds: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(bytes(2 * int(8000 * seconds)))


@pytest.fixture
def staged(tmp_path):
    weights = tmp_path / "final"
    weights.mkdir()
    (weights / "student.pt").write_bytes(b"student" * 100)
    (weights / "decoder.pt").write_bytes(b"decoder" * 100)
    for s in cep.SAMPLES:
        write_wav(tmp_path / "samples" / s)
    out = tmp_path / "hf-publish" / "ahenk-cep-tr"
    assert cep.stage(weights, tmp_path / "samples", out) == []
    return out


def test_stage_renders_the_card_with_the_staged_files(staged):
    card = (staged / "README.md").read_text(encoding="utf-8")
    assert "{{" not in card and "—" not in card
    assert f"`{cep.sha256(staged / 'student.pt')}`" in card and f"`{cep.sha256(staged / 'decoder.pt')}`" in card
    assert (staged / cep.CODE).read_bytes() == (REPO / "tts" / "small" / cep.CODE).read_bytes()
    header = card.split("\n---\n", 1)[0]
    for line in ("license: cc-by-4.0", "pipeline_tag: text-to-speech", "base_model: canberkkkkkk/ema-lightning",
                 "  - tr"):
        assert line in header
    assert 'snapshot_download("ozgurguler/ahenk-cep-tr"' in card
    assert "8.6M parametre, 34 MB (fp32)" in card and "öğretmen 8, Ahenk Cep 2" in card
    assert sorted(p.name for p in (staged / "samples").glob("*.wav")) == sorted(cep.SAMPLES)
    assert cep.main(["--check", str(staged)]) == 0


def test_check_staging_catches_changed_and_stray_files(staged):
    (staged / "student.pt").write_bytes(b"other")
    write_wav(staged / "samples" / "old.wav")
    (staged / "notes.txt").write_text("x")
    problems = " | ".join(cep.check_staging(staged))
    assert "sha256 of the staged student.pt" in problems
    assert "does not link: old.wav" in problems and "unexpected files in the staging dir: notes.txt" in problems
    (staged / "README.md").unlink()
    assert cep.check_staging(staged) == [f"{staged / 'README.md'} missing (stage first)"]


def test_stage_refuses_missing_inputs(tmp_path):
    with pytest.raises(SystemExit, match="missing"):
        cep.stage(tmp_path, tmp_path, tmp_path / "out")


def test_space_copies_the_inference_module_and_apps_parse():
    small = REPO / "tts" / "small" / "ahenk_cep.py"
    assert (REPO / "spaces" / "ahenk-cep" / "ahenk_cep.py").read_bytes() == small.read_bytes()
    for app in ("ahenk-cep", "ahenk"):
        ast.parse((REPO / "spaces" / app / "app.py").read_text(encoding="utf-8"))
    yaml = (REPO / "spaces" / "ahenk-cep" / "README.md").read_text(encoding="utf-8").split("\n---\n", 1)[0]
    assert 'python_version: "3.12"' in yaml and "suggested_hardware: cpu-basic" in yaml
    assert "ozgurguler/ahenk-cep-tr" in (REPO / "spaces" / "ahenk-cep" / "app.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("page", ["README.md", "README.tr.md", "tts/voxcpm/MODEL_CARD.ahenk.template.md",
                                  "tts/small/MODEL_CARD.ahenk-cep.template.md", "spaces/ahenk-cep/README.md",
                                  "spaces/ahenk/README.md"])
def test_public_pages_have_no_em_dash(page):
    assert "—" not in (REPO / page).read_text(encoding="utf-8")
