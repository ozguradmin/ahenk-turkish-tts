"""tts/voxcpm/nanovllm_app.py: the Nano-vLLM probe's pure helpers (spec, sentences, wheel URL, LoRA kwargs, stats,
report), run_batch on a fake server, and the worker's resume / budget decisions (modal stubbed: no Modal, torch,
numpy, nanovllm_voxcpm or GPU)."""

import asyncio
import importlib.util
import json
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
VOXCPM = REPO / "tts" / "voxcpm"
sys.path[:0] = [str(VOXCPM), str(REPO / "src")]


def _load():
    """Import nanovllm_app with a stand-in `modal` (its App / Image / Volume calls only build objects)."""
    spec = importlib.util.spec_from_file_location("nanovllm_app", VOXCPM / "nanovllm_app.py")
    mod = importlib.util.module_from_spec(spec)
    had, saved = "modal" in sys.modules, sys.modules.get("modal")
    sys.modules["modal"] = mock.MagicMock()
    try:
        spec.loader.exec_module(mod)
    finally:
        if had:
            sys.modules["modal"] = saved
        else:
            del sys.modules["modal"]
    return mod


na = _load()

# lora_config.json of /data/exp/voxcpm2-alania-s2/ckpt/step_0000600 (the official trainer writes it)
S2_LORA = {"base_model": "openbmb/VoxCPM2", "lora_config": {
    "enable_lm": True, "enable_dit": True, "enable_proj": False, "r": 64, "alpha": 64, "dropout": 0.0,
    "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "target_proj_modules": ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj", "fusion_concat_proj"]}}
TRZOR_CATS = {"sayi", "sira", "tarih", "saat", "para", "yuzde", "birim", "numara", "kisaltma", "akronim", "kisa",
              "soru", "ozel"}


def test_import_needs_no_heavy_deps():
    code = (f"import sys; from unittest import mock; sys.modules['modal'] = mock.MagicMock(); "
            f"sys.path[:0] = [{str(VOXCPM)!r}, {str(REPO / 'src')!r}]; import nanovllm_app; "
            "bad = {'torch', 'numpy', 'soundfile', 'librosa', 'nanovllm_voxcpm', 'voxcpm', 'faster_whisper', 'jiwer', "
            "'synth_eval'} & set(sys.modules); assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_flash_attn_wheel_is_the_release_asset():
    # browser_download_url of the v2.8.3 release asset (GitHub API, 2026-10-11)
    assert na.flash_attn_wheel() == ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
                                     "flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl")
    assert na.flash_attn_wheel("2.8.0+cu128") == na.flash_attn_wheel("2.8.0")
    assert "cu12torch2.7cxx11abiFALSE-cp312-cp312" in na.flash_attn_wheel("2.7.1", cxx11_abi=False, py_tag="cp312")
    assert "cxx11abi$(cat /tmp/abi)-cp311" in na.flash_attn_wheel(cxx11_abi="$(cat /tmp/abi)")


def test_image_pins_torch_in_every_resolver_run():
    cmds = na.modal.Image.debian_slim.return_value.apt_install.return_value.run_commands.call_args.args
    pip = [c for c in cmds if "pip install" in c]
    assert any(f"git checkout {na.VOXCPM_COMMIT}" in c for c in cmds)
    nano = next(c for c in pip if "nano-vllm-voxcpm @ git+" in c)
    assert f"@{na.NANO_COMMIT}'" in nano
    for c in (nano, pip[0]):  # no later resolver run may swap the CUDA build of torch
        assert f"torch=={na.TORCH} torchaudio=={na.TORCH} torchcodec=={na.TORCHCODEC}" in c and "whl/cu128" in c
    fa = next(c for c in cmds if "flash-attention/releases" in c)
    assert "--no-deps" in fa and "_GLIBCXX_USE_CXX11_ABI" in fa and "cxx11abi$(cat /tmp/torch_abi)-cp311" in fa


def test_pins():
    assert len(na.NANO_COMMIT) == 40 and len(na.VOXCPM_COMMIT) == 40
    modal_app = (VOXCPM / "modal_app.py").read_text(encoding="utf-8")
    assert f'VOXCPM_COMMIT = "{na.VOXCPM_COMMIT}"' in modal_app  # the official API = the one synth_eval scored
    assert na.TORCH.startswith("2.8.") and na.TORCHCODEC.startswith("0.7.")  # Nano-vLLM CI's torch / torchcodec pair


def test_parse_ints_and_modes():
    assert na.parse_ints(" 32,16,,16 ", 1, 256, "concurrency") == [16, 32]
    for bad in ("", "0", "8,x", "300"):
        with pytest.raises(ValueError, match="--concurrency"):
            na.parse_ints(bad, 1, 256, "concurrency")
    assert na.parse_modes("ref, noref") == ["noref", "ref"] and na.parse_modes("ref") == ["ref"]
    for bad in ("", "prompt", "noref,continuation"):
        with pytest.raises(ValueError, match="--modes"):
            na.parse_modes(bad)


def test_max_gen_len_is_the_official_first_attempt_cap():
    assert na.max_gen_len(0) == 10 and na.max_gen_len(30) == 190  # int(30 * 6 + 10)
    assert na.max_gen_len(10_000) == na.MAX_GEN_CAP
    assert na.max_gen_len(-5) == 1


def test_percentile():
    assert na.percentile([4, 1, 3, 2], 50) == 2.5
    assert na.percentile([1, 2, 3, 4], 90) == pytest.approx(3.7)
    assert na.percentile([7], 90) == 7 and na.percentile([], 50) is None
    assert na.percentile([1, 2], 150) == 2 and na.percentile([1, 2], -1) == 1


def test_nano_lora_kwargs():
    kw = na.nano_lora_kwargs(S2_LORA)
    assert kw == {"enable_lm": True, "enable_dit": True, "enable_proj": False, "max_loras": 1, "max_lora_rank": 64,
                  "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
                  "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"],
                  "target_proj_modules": ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj",
                                          "fusion_concat_proj"]}
    assert "dropout" not in kw and "r" not in kw  # Nano-vLLM's LoRAConfig forbids extra keys
    assert na.nano_lora_kwargs({"r": 32, "lora_alpha": 16})["max_lora_rank"] == 32  # bare / lora_alpha spelling
    for bad in ({"lora_config": {"r": 64}}, {"lora_config": {"r": 0, "alpha": 1}}, {"lora_config": None}, []):
        with pytest.raises(ValueError, match="lora_config"):
            na.nano_lora_kwargs(bad)


def test_pick_ref():
    rows = [{"audio": "/a", "duration": 3.0}, {"audio": "/b", "duration": 14.0}, {"audio": "/c", "duration": 7.5},
            {"audio": "/d", "duration": 9.0}]
    assert na.pick_ref(rows)["audio"] == "/c"  # the first within 6-12 s, in manifest order
    assert na.pick_ref(rows[:2])["audio"] == "/b"  # none within: closest to 9 s
    assert na.pick_ref([{"duration": 8.0}, {"audio": "", "duration": 8.0}]) is None
    assert na.pick_ref([]) is None


def test_interleave_keeps_every_prefix_mixed():
    out = na.interleave(list("aaaa"), list("bb"))  # keys a: 1/8 3/8 5/8 7/8, b: 1/4 3/4
    assert out == ["a", "b", "a", "a", "b", "a"]
    assert na.interleave([1, 2], []) == [1, 2]


def test_build_items():
    items = na.build_items()
    assert items == na.build_items()  # deterministic
    ids = [it["id"] for it in items]
    assert len(items) == 64 and len(set(ids)) == 64
    assert Counter(it["set"] for it in items) == {"trzor": 42, "soru": 16, "emo": 6}
    assert {it["category"] for it in items if it["set"] == "trzor"} == TRZOR_CATS
    assert {it["category"] for it in items if it["set"] == "soru"} == {"mi_soru", "ne_soru", "degil_mi",
                                                                        "yukselen", "duz_eslesik"}
    assert {it["set"] for it in items[:16]} == {"trzor", "soru", "emo"}  # the c16 batch mixes all three
    from trnorm import normalize

    for (i, instr, s), it in zip(na.EMOTION_LINES, [it for it in items if it["set"] == "emo"]):
        assert it["id"] == i and it["model_text"] == f"({instr}){normalize(s)}" and it["refs"] == [s]
    z = next(it for it in items if it["id"] == "sayi_04")  # trnorm front-end, as synth_eval feeds VoxCPM
    assert z["model_text"] == normalize(z["text"]) and z["model_text"] != z["text"]


def test_load_sentences(tmp_path):
    rows = [{"id": "sayi_01", "set": "trzor", "category": "sayi", "text": "Toplantıya 45 kişi katıldı.",
             "refs": ["Toplantıya kırk beş kişi katıldı."]},
            {"id": "sayi_02", "set": "trzor", "text": "Bu kitabı 3 kez okudum."},
            {"id": "emo_a", "set": "emo", "instruction": "Angry tone.", "text": "Yeter artık, 3 kez söyledim!"},
            {"id": "x", "text": "Merhaba."}]
    f = tmp_path / "set.jsonl"
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n\n" for r in rows), encoding="utf-8")
    items = na.load_sentences(f)
    assert [it["id"] for it in items] == ["sayi_01", "emo_a", "x", "sayi_02"]  # interleaved by set
    by = {it["id"]: it for it in items}
    assert by["sayi_01"]["model_text"] == "Toplantıya kırk beş kişi katıldı." and by["sayi_01"]["category"] == "sayi"
    assert by["emo_a"]["model_text"] == "(Angry tone.)Yeter artık, üç kez söyledim!"
    assert by["emo_a"]["refs"] == ["Yeter artık, 3 kez söyledim!"] and by["x"]["set"] == "file"
    spec = na.build_spec(run="t1", sentences=str(f), concurrency="2")
    assert spec["sentences"] == str(f) and len(spec["items"]) == 4
    assert spec["compare"] == ["sayi_01", "emo_a", "x", "sayi_02"]  # round-robin over sets in the 2 x 2 requests
    for bad in ('{"id": "a"}', '{"text": "b"}', '{"id": "a", "text": "b"}\n{"id": "a", "text": "c"}', ""):
        f.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            na.load_sentences(f)
    with pytest.raises(ValueError, match="--sentences"):
        na.build_spec(sentences=str(tmp_path / "missing.jsonl"))


def test_resume_refuses_an_edited_sentences_file(tmp_path):
    rows = [{"id": "sayi_01", "set": "trzor", "text": "Toplantıya 45 kişi katıldı."},
            {"id": "emo_a", "set": "emo", "instruction": "Angry tone.", "text": "Yeter artık!"}]
    f = tmp_path / "set.jsonl"
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    a = na.build_spec(run="t1", sentences=str(f), concurrency="2", compare=2)
    old = {k: v for k, v in a.items() if k != "items"}  # what results.json keeps
    assert na.resume_conflicts(old, na.build_spec(run="t1", sentences=str(f), concurrency="2", compare=2)) == []
    rows[0]["text"] = "Toplantıya 145 kişi katıldı."  # same ids, same compare pick, other sentence
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    b = na.build_spec(run="t1", sentences=str(f), concurrency="2", compare=2)
    assert b["compare"] == a["compare"]
    assert [c.split(":")[0] for c in na.resume_conflicts(old, b)] == ["items_sha1"]
    assert na.items_sha1(na.build_items()) == na.build_spec(run="t2")["items_sha1"]  # builtin: deterministic


def test_compare_ids_round_robin_over_sets():
    items = na.build_items()
    assert na.compare_ids(items, 4, 32) == ["sayi_01", "duz_eslesik_01", "emo_angry", "sayi_02"]
    assert na.compare_ids(items, 0, 32) == []
    pool = [it["id"] for it in items[:3]]
    got = na.compare_ids(items, 10, 3)
    assert sorted(got) == sorted(pool)  # never outside the lowest concurrency's batch


def test_build_spec():
    s = na.build_spec(run="t1")
    assert s["run"] == "t1" and s["out"] == "/data/nanovllm-probe/t1"
    assert s["timesteps"] == [10, 16] and s["concurrency"] == [16, 32, 64] and s["modes"] == ["noref", "ref"]
    assert s["max_num_seqs"] == 64 and s["seed"] == 0 and s["cfg"] == 2.0 and len(s["compare"]) == 4
    assert s["nano_commit"] == na.NANO_COMMIT and len(s["items"]) == 64
    json.dumps(s)  # it travels as JSON
    assert na.build_spec(lora="/data/exp/x/ckpt/step_1/")["lora"] == "/data/exp/x/ckpt/step_1"
    assert na.build_spec()["run"].startswith("probe-")
    for kw, msg in (({"run": "a/b"}, "--run"), ({"lora": "voxcpm-lora/x"}, "--lora"), ({"ref": "x.wav"}, "--ref"),
                    ({"ref": "/data/x.wav", "modes": "noref"}, "ref mode"), ({"compare": 40, "rounds": 1}, "--compare"),
                    ({"gpu_mem": 0.99}, "--gpu-mem"), ({"rounds": 0}, "--rounds"), ({"cfg": 0.0}, "--cfg"),
                    ({"timesteps": "0"}, "--timesteps")):
        with pytest.raises(ValueError, match=msg):
            na.build_spec(**kw)


def test_resume_conflicts():
    a = na.build_spec(run="t1")
    assert na.resume_conflicts(a, dict(a)) == []
    b = {**a, "lora": "/data/exp/x", "concurrency": [8]}  # concurrency alone is per scenario; the LoRA is not
    assert [c.split(":")[0] for c in na.resume_conflicts(a, b)] == ["lora"]
    assert a["sentences"] == "builtin" and "sentences" in na.RESUME_KEYS
    c = {**a, "gpu": "H100", "torch": "2.9.0", "flash_attn": "2.8.4"}  # other prices / build: never one results.json
    assert [x.split(":")[0] for x in na.resume_conflicts(a, c)] == ["gpu", "torch", "flash_attn"]


def test_default_plan_fits_default_budget():
    s = na.build_spec(run="t1")
    phases = na.estimate(s)
    assert [p for p, _ in phases] == ["env check", "nano-vllm ts10", "nano-vllm ts16", "official API",
                                      "judge (Whisper, UTMOS)"]
    total = sum(x for _, x in phases) + na.START_S + na.TAIL_S + na.IDLE_S
    assert total <= na.usd_to_seconds(na.PROBE_MAX_USD, na.GPU_PRICE["A100-80GB"] + na.SIDE_PRICE)
    assert [p for p, _ in na.estimate(na.build_spec(run="t1", compare=0))][-1] == "nano-vllm ts16"
    assert na.official_rtf(10) == pytest.approx(0.9) and 1.7 <= na.official_rtf(16) <= 2.3


def test_parse_smi_and_window_stats():
    assert na.parse_smi("12345, 87\n") == (12345, 87)
    assert na.parse_smi("[N/A], [N/A]") is None and na.parse_smi("") is None
    samples = [(1.0, 1000, 10), (2.0, 5000, 90), (3.0, 3000, 50), (9.0, 70000, 100)]
    assert na.window_stats(samples, 1.5, 3.0) == {"gpu_mem_peak_mib": 5000, "gpu_util_mean": 70}
    assert na.window_stats(samples, 4.0, 5.0) == {}


def _req(i, t0, first, t1, samples, steps=10, cap=100):
    return {"i": i, "id": f"x{i}", "t0": t0, "first": first, "t1": t1, "samples": samples, "steps": steps, "cap": cap}


def test_batch_stats():
    sr = 1000
    reqs = [_req(0, 0.0, 0.1, 2.1, 4000), _req(1, 0.0, 0.3, 4.3, 8000, steps=100),
            {"i": 2, "id": "x2", "error": "RuntimeError: boom"}]
    s = na.batch_stats(reqs, sr, wall_s=4.0, usd_per_h=3.0)
    assert s["requests"] == 3 and s["ok"] == 2 and s["errors"] == 1 and s["first_error"] == "RuntimeError: boom"
    assert s["audio_s"] == 12.0 and s["audio_s_per_gpu_s"] == 3.0
    # the 8 s of x1 ran to its cap: runaway audio, not usable; $ per usable audio hour
    assert s["usable_audio_s"] == 4.0 and s["usable_audio_s_per_gpu_s"] == 1.0 and s["usd_per_usable_audio_h"] == 3.0
    assert s["hit_cap"] == 1 and s["cap_audio_share"] == pytest.approx(0.667)
    assert s["latency_p50_s"] == pytest.approx(3.2) and s["latency_max_s"] == 4.3
    assert s["ttfb_p50_s"] == pytest.approx(0.2) and s["rtf_req_mean"] == pytest.approx(0.5)  # (2/4 + 4/8) / 2
    assert s["audio_s_mean"] == 6.0
    empty = na.batch_stats([], sr, 0.0)
    assert empty["audio_s_per_gpu_s"] is None and empty["latency_p50_s"] is None and "errors" not in empty
    assert empty["usable_audio_s_per_gpu_s"] is None and empty["cap_audio_share"] is None
    capped = na.batch_stats([_req(0, 0.0, 0.1, 1.0, 4000, steps=100)], sr, 1.0, 3.0)
    assert capped["usable_audio_s_per_gpu_s"] == 0.0 and capped["usd_per_usable_audio_h"] is None


def test_scenario_done_and_server_failure():
    assert na.scenario_done({"ok": 32}) and not na.scenario_done(None) and not na.scenario_done({})
    assert not na.scenario_done({"ok": 30, "errors": 2}) and not na.scenario_done({"ok": 32, "incomplete": True})
    ok = [_req(i, 0.0, 0.1, 1.0, 100) for i in range(4)]
    err = {"i": 9, "id": "x9", "error": "ValueError: bad text"}
    assert na.server_failure(ok) is None and na.server_failure([]) is None
    assert na.server_failure(ok + [err]) is None  # one bad request: the scenario is incomplete, the server goes on
    assert na.server_failure(ok[:2] + [err, err]).startswith("2 of 4 requests failed (ValueError")
    died = {"i": 8, "id": "x8", "error": "WorkerDiedError: worker process died unexpectedly (exitcode=-9)"}
    assert "WorkerDiedError" in na.server_failure(ok + [err, died])  # one is enough


class _FakeServer:
    """Nano-vLLM's AsyncVoxCPM2ServerPool.generate: audio chunks per step; "boom" texts fail mid-stream."""

    def __init__(self):
        self.inflight = self.peak = 0
        self.calls = []

    async def generate(self, target_text, max_generate_length=2000, cfg_value=2.0, ref_audio_latents=None,
                       lora_name=None, seed=None, **kw):
        self.calls.append({"text": target_text, "cap": max_generate_length, "cfg": cfg_value, "ref": ref_audio_latents,
                           "lora": lora_name, "seed": seed})
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            for step in range(3):
                await asyncio.sleep(0.001)
                if target_text == "boom" and step == 1:
                    raise RuntimeError("worker died")
                yield [0.0] * 100
        finally:
            self.inflight -= 1


def test_run_batch_concurrency_keep_and_errors():
    items = [{"id": "a", "model_text": "bir", "max_len": 50}, {"id": "b", "model_text": "boom"},
             {"id": "c", "model_text": "üç", "max_len": 3}]
    srv = _FakeServer()
    reqs, kept, wall = asyncio.run(na.run_batch(srv, items, 2, 7, cfg=2.0, seed=0, ref_latents=b"L", lora_name="probe",
                                                keep=["a", "b"]))
    assert srv.peak == 2 and len(srv.calls) == 7 and wall > 0
    assert [r["id"] for r in reqs] == ["a", "b", "c", "a", "b", "c", "a"]  # request i -> items[i % 3]
    assert all(c["ref"] == b"L" and c["lora"] == "probe" and c["seed"] == 0 for c in srv.calls)
    assert [c["cap"] for c in srv.calls[:3]] == [50, na.MAX_GEN_CAP, 3]
    ok = [r for r in reqs if "error" not in r]
    assert len(ok) == 5 and all(r["samples"] == 300 and r["steps"] == 3 for r in ok)
    assert sum(r["steps"] >= r["cap"] for r in ok) == 2  # "c" ends at its cap
    assert all(r["error"] == "RuntimeError: worker died" for r in reqs if r["id"] == "b")
    assert list(kept) == ["a"] and len(kept["a"]) == 3  # the first "a" only; the failed "b" keeps nothing
    assert [r["i"] for r in ok if r["kept"]] == [0]
    s = na.batch_stats(reqs, 100, wall)
    assert s["ok"] == 5 and s["errors"] == 2 and s["audio_s"] == 15.0 and s["usable_audio_s"] == 9.0


def _row(engine, ts, mode, i, duration, errors=None, words=None, utmos=None, synth_s=None):
    r = {"engine": engine, "ts": ts, "mode": mode, "id": i, "set": "trzor", "wav": f"/x/{i}.wav", "duration": duration}
    if errors is not None:
        r["errors"], r["ref_words"] = errors, words
    if utmos is not None:
        r["utmos"] = utmos
    if synth_s is not None:
        r["synth_s"] = synth_s
    return r


def test_compare_summary():
    rows = [_row("nano", 10, "noref", "a", 4.0, 0, 5, 3.6), _row("nano", 10, "noref", "b", 2.0, 1, 5, 3.4),
            _row("official", 10, "noref", "a", 4.0, 1, 5, 3.5, 3.6), _row("official", 10, "noref", "b", 1.0, 0, 5, 3.3,
                                                                           0.9),
            _row("nano", 16, "ref", "a", 3.0)]  # not judged yet, no official twin
    rows[1]["hit_cap"] = True  # nano "b" ran to its length cap
    s = na.compare_summary({f"{r['engine']}/{r['ts']}/{r['mode']}/{r['id']}": r for r in rows})
    assert s["nano/ts10/noref"] == {"clips": 2, "capped": 1, "duration_s": 3.0, "wer": 10.0, "judged": 2,
                                    "rows_correct": 1, "utmos": 3.5}
    assert s["official/ts10/noref"]["rtf"] == pytest.approx(0.9)  # (3.6 + 0.9) / (4 + 1)
    assert s["nano/ts16/ref"] == {"clips": 1, "duration_s": 3.0}
    assert s["nano_vs_official"] == {"ts10/noref": {"pairs": 2, "nano_capped": 1, "wer_nano": 10.0,
                                                    "wer_official": 10.0, "utmos_nano": 3.5, "utmos_official": 3.4,
                                                    "duration_ratio": 1.5}}  # (4/4 + 2/1) / 2
    assert na.compare_summary({}) == {}


def test_format_report():
    res = {"spec": {"run": "t1"}, "env": {"gpu": "A100", "torch": "2.8.0"},
           "servers": {"ts10": {"ready_s": 61.2, "mem_ready_mib": 70000}},
           "scenarios": {"ts10/noref/c32": {"requests": 64, "audio_s_per_gpu_s": 44.0, "usable_audio_s_per_gpu_s": 40.0,
                                            "usd_per_usable_audio_h": 0.0736, "latency_p50_s": 3.0,
                                            "latency_p90_s": 4.0, "ttfb_p50_s": 0.1, "rtf_req_mean": 0.6,
                                            "gpu_mem_peak_mib": 71000, "gpu_util_mean": 90, "hit_cap": 1,
                                            "cap_audio_share": 0.09},
                         # faster, but 2 requests failed (a dying engine answers fast): never the best
                         "ts10/noref/c64": {"requests": 128, "audio_s_per_gpu_s": 90.0,
                                            "usable_audio_s_per_gpu_s": 90.0, "hit_cap": 0, "errors": 2,
                                            "incomplete": True},
                         "ts10/noref/c16": {"requests": 32, "audio_s_per_gpu_s": 25.0, "hit_cap": 0}},
           "compare_summary": {"official/ts10/noref": {"clips": 4, "rtf": 0.9},
                               "nano_vs_official": {"ts10/noref": {"pairs": 4, "nano_capped": 1, "wer_nano": 0.0,
                                                                   "wer_official": 5.0, "utmos_nano": 3.6,
                                                                   "utmos_official": 3.5, "duration_ratio": 1.01}}},
           "skipped": ["judge (needs ~90 s, 10 s left)"], "errors": ["nano-vllm ts16: RuntimeError: x"]}
    lines = na.format_report(res, 2.944)
    text = "\n".join(lines)
    assert text.index("ts10/noref/c16") < text.index("ts10/noref/c32") < text.index("ts10/noref/c64")  # numeric
    assert "68.4 GiB" in text and "Nano-vLLM best 40.0 usable audio-s/GPU-s (ts10/noref/c32) = 36x" in text
    c32 = next(x for x in lines if x.startswith("ts10/noref/c32")).split()
    assert c32[2:4] == ["44.00", "40.00"] and c32[-3:] == ["1", "9", "0"]  # all / usable; cap, cap% of audio, err
    assert next(x for x in lines if x.startswith("ts10/noref/c64")).endswith("incomplete: runs again")
    assert "WER 0.0 vs 5.0" in text and "(1 nano clips hit the length cap)" in text
    assert "skipped: judge" in text and "error: nano-vllm ts16" in text and "--run t1 to continue" in text
    assert "~$2.65 per audio hour)" in text  # no spec gpu: the caller's price
    res["spec"]["gpu"] = "H100"  # a results.json is priced at the GPU it ran on, whatever TTS_GPU says now
    assert "~$3.95 per audio hour on H100)" in "\n".join(na.format_report(res, 2.944))
    assert na.format_report({}, 2.944) == []


def _probe(tmp_path, deadline_in, **kw):
    spec = {**na.build_spec(run="t1", **kw), "out": str(tmp_path)}
    return na._Probe(spec, time.time() + deadline_in)


def test_probe_resume_keeps_results_and_resets_notes(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({
        "scenarios": {"ts10/noref/c16": {"ok": 32}}, "compare": {"nano/ts10/noref/sayi_01": {"engine": "nano"}},
        "skipped": ["old"], "errors": ["old"], "finished": True, "junk": 1}), encoding="utf-8")
    p = _probe(tmp_path, 600)
    assert p.res["scenarios"] == {"ts10/noref/c16": {"ok": 32}} and "nano/ts10/noref/sayi_01" in p.res["compare"]
    assert p.res["skipped"] == [] and p.res["errors"] == [] and p.res["finished"] is False and "junk" not in p.res
    assert "items" not in p.res["spec"] and p.res["spec"]["run"] == "t1"


def test_probe_skips_by_budget_before_loading_anything(tmp_path):
    p = _probe(tmp_path, 30)  # far less than a server start
    before = set(sys.modules)
    asyncio.run(p.nano(10))
    p.official()
    p.res["compare"]["nano/ts10/noref/sayi_01"] = {"engine": "nano"}
    p.judge()
    assert [s.split(" (")[0] for s in p.res["skipped"]] == ["nano-vllm ts10", "official API", "judge"]
    assert not {"nanovllm_voxcpm", "voxcpm", "torch", "faster_whisper", "synth_eval"} & (set(sys.modules) - before)


def test_probe_skips_finished_work(tmp_path):
    p = _probe(tmp_path, 3600, timesteps="10", modes="noref", concurrency="16", compare=1)
    p.res["scenarios"]["ts10/noref/c16"] = {"ok": 32}
    p.res["compare"]["official/ts10/noref/sayi_01"] = {"engine": "official"}
    asyncio.run(p.nano(10))
    p.official()
    assert p.res["skipped"] == [] and p.res["errors"] == []


def test_probe_records_phase_failures_and_goes_on(tmp_path):
    for mod in ("nanovllm_voxcpm", "voxcpm"):
        if importlib.util.find_spec(mod):
            pytest.skip(f"{mod} is installed here")
    p = _probe(tmp_path, 3600, timesteps="10", modes="noref", concurrency="16", compare=1)
    asyncio.run(p.nano(10))  # no Nano-vLLM here: recorded, the official API phase still runs
    p.official()
    assert [e.split(":")[0] for e in p.res["errors"]] == ["nano-vllm ts10", "official API"]
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["errors"] == p.res["errors"]


class WorkerDiedError(RuntimeError):
    """Named as nanovllm_voxcpm.utils.worker_link's: what every request gets once the engine process is gone."""


class _Engine:
    """A Nano-vLLM server pool for _Probe.nano: generate fails with WorkerDiedError from call die_after + 1 on (the
    engine crashed), and with ValueError for the texts in bad."""

    def __init__(self, die_after: int | None = None, bad: tuple = ()):
        self.die_after, self.bad, self.calls, self.stopped = die_after, bad, 0, False

    async def wait_for_ready(self):
        pass

    async def get_model_info(self):
        return {"output_sample_rate": 1000}

    async def stop(self):
        self.stopped = True

    async def generate(self, target_text, **kw):
        self.calls += 1
        if self.die_after is not None and self.calls > self.die_after:
            raise WorkerDiedError("worker process died unexpectedly (exitcode=-9)")
        if target_text in self.bad:
            raise ValueError("bad text")
        for _ in range(2):
            await asyncio.sleep(0)
            yield [0.0] * 500


def _nano(p, engine):
    """p.nano(10) on engine, no nvidia-smi."""
    fake = types.ModuleType("nanovllm_voxcpm")
    fake.VoxCPM = types.SimpleNamespace(from_pretrained=lambda **kw: engine)
    p.smi = types.SimpleNamespace(samples=[])
    with mock.patch.dict(sys.modules, {"nanovllm_voxcpm": fake}), mock.patch.object(na, "_smi", return_value=None):
        asyncio.run(p.nano(10))


def test_probe_dead_engine_stops_its_server_and_a_resume_redoes_it(tmp_path):
    kw = dict(timesteps="10", modes="noref", concurrency="4,8", compare=0)
    p, eng = _probe(tmp_path, 3600, **kw), _Engine(die_after=10)  # warm-up 8 ok, c4: 2 ok, 6 fail
    _nano(p, eng)
    c4 = p.res["scenarios"]["ts10/noref/c4"]
    assert c4["ok"] == 2 and c4["errors"] == 6 and c4["incomplete"] and c4["gpu"] == "A100-80GB"
    assert "ts10/noref/c8" not in p.res["scenarios"] and eng.calls == 16 and eng.stopped  # c8 never sent to it
    assert len(p.res["errors"]) == 1 and "c4: 6 of 8 requests failed (WorkerDiedError" in p.res["errors"][0]
    assert not p.finished()
    q, eng2 = _probe(tmp_path, 3600, **kw), _Engine()  # the same --run on a healthy engine
    _nano(q, eng2)
    assert eng2.calls == 8 + 8 + 16 and q.res["errors"] == []  # c4 again, then c8
    assert all(na.scenario_done(q.res["scenarios"][k]) for k in ("ts10/noref/c4", "ts10/noref/c8"))
    assert q.finished()
    r = _probe(tmp_path, 3600, **kw)
    _nano(r, _Engine(die_after=0))  # everything done: no server start
    assert r.res["errors"] == [] and r.finished()


def test_probe_dead_engine_in_warm_up_measures_nothing(tmp_path):
    p, eng = _probe(tmp_path, 3600, timesteps="10", modes="noref", concurrency="4,8", compare=0), _Engine(die_after=3)
    _nano(p, eng)
    assert p.res["scenarios"] == {} and eng.calls == 8 and eng.stopped
    assert "warm-up of ts10/noref: 5 of 8 requests failed (WorkerDiedError" in p.res["errors"][0]


def test_probe_a_failed_request_marks_its_scenario_only(tmp_path):
    p = _probe(tmp_path, 3600, timesteps="10", modes="noref", concurrency="4,8", compare=0)
    eng = _Engine(bad=(p.spec["items"][1]["model_text"],))  # item 1: once in c4 (8 requests), once in c8 (16)
    _nano(p, eng)
    sc = p.res["scenarios"]
    assert sc["ts10/noref/c4"]["errors"] == 1 and sc["ts10/noref/c8"]["errors"] == 1  # the server went on
    assert all(s["incomplete"] for s in sc.values()) and len(p.res["errors"]) == 2
    assert p.res["errors"][0].startswith("ts10/noref/c4: 1 of 8 requests failed (ValueError: bad text)")
    assert not p.finished()


def test_cli_plan(capsys):
    assert na.cli(["--run", "t1", "--compare", "3"]) == 0
    out = capsys.readouterr().out
    assert "run t1: 64 sentences of builtin (trzor 42, soru 16, emo 6)" in out and "at most $0.80" in out
    assert "(Angry tone, voice getting louder.)" in out
    with pytest.raises(SystemExit):
        na.cli(["--concurrency", "0"])
    with mock.patch.object(na, "worker_main", return_value=7) as w:
        assert na.cli(["--spec", "x.json", "--deadline", "1"]) == 7
    w.assert_called_once_with(["--spec", "x.json", "--deadline", "1"])
