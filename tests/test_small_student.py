"""tts/small (small on-device model, P1): the trainer's data side (student_data: word bounds and frame timeline as EMA
Lightning's engine builds them, padded batches, length buckets, learning-rate schedule, checkpoint files, the eval
system spec) and the blind listening page (listen_page: BS.1770 loudness, codes and file names, the page, decoding a
pasted result) with numpy / scipy / soundfile only; the model side (student: losses, the samplers, ema.pt equivalence,
a one-batch overfit) only where torch and ema_lightning are installed, else skipped."""

import json
import random
import sys
from itertools import pairwise
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SMALL = REPO / "tts" / "small"
sys.path[:0] = [str(SMALL), str(REPO / "src")]

import listen_page as lp  # noqa: E402
import student_data as D  # noqa: E402

np = pytest.importorskip("numpy")


# ---- reference copies of ema_lightning.engine (Apache-2.0), so the comparison needs no torch ----------------------
def engine_piece(text):
    starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
    bounds = [0] + starts[1:] + [len(text)]
    cw, wstart = [], []
    for w, (a, b) in enumerate(pairwise(bounds)):
        cw += [w] * (b - a)
        wstart += [a] * (b - a)
    return cw, wstart


def engine_plan(n):
    n = np.asarray(n)
    frames = int(n.sum())
    fw = np.repeat(np.arange(n.size), n)[:frames]
    fp = ((np.arange(frames) - (np.cumsum(n) - n)[fw]).astype(np.float64) / n[fw].astype(np.float64)).astype(np.float32)
    return fw, fp


VOCAB = ["<pad>", "<unk>", " ", "!", '"', "%", "&", "'", "(", ")", ",", "-", ".", "/", ":", ";", "?",
         *"abcdefghijklmnopqrstuvwxyz", *"çöüğış"]
STOI = {c: i for i, c in enumerate(VOCAB)}


def row(i, text, word_frames, offset, style=0, split="train", use=True):
    return {"id": f"r{i}", "ema_text": text, "ids": [STOI.get(c, 1) for c in text], "word_frames": word_frames,
            "frames": sum(word_frames), "offset": offset, "style_id": style, "split": split, "use": use}


def rows_and_latents(n=12, seed=0):
    rng = random.Random(seed)
    words = ["bu", "gün", "hava", "çok", "güzel", "mi?", "yarın", "da", "yağmur", "var."]
    rows, off = [], 0
    for i in range(n):
        k = rng.randint(1, 6)
        text = " ".join(rng.choice(words) for _ in range(k))
        wf = [rng.randint(1, 15) for _ in range(k)]
        r = row(i, text, wf, off, style=rng.randint(0, 6), split="val" if i % 5 == 0 else "train")
        rows.append(r)
        off += r["frames"]
    lat = np.random.default_rng(seed).standard_normal((off, D.LATENT_DIM)).astype(np.float16)
    return rows, lat


# ---- student_data ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["merhaba", "bugün hava çok güzel.", "a b c", "evet, hayır? tamam!", " baş boşluk",
                                  "çift  boşluk var", "x"])
def test_word_bounds_match_engine_piece(text):
    assert D.word_bounds(text) == engine_piece(text)


def test_word_bounds_space_belongs_to_preceding_word():
    cw, ws = D.word_bounds("ab cd")
    assert cw == [0, 0, 0, 1, 1] and ws == [0, 0, 0, 3, 3]


@pytest.mark.parametrize("n", [[1], [3, 1, 4], [10, 2, 7, 1, 1], [250]])
def test_timeline_matches_engine_plan(n):
    fw, fp = D.timeline(n)
    efw, efp = engine_plan(n)
    assert fw.tolist() == efw.tolist()
    np.testing.assert_allclose(fp, efp)
    assert len(fw) == sum(n) and fp.min() >= 0 and fp.max() < 1
    for w, k in enumerate(n):  # each word starts at position 0 and steps by 1/k
        np.testing.assert_allclose(fp[fw == w], np.arange(k) / k, rtol=1e-6)


def test_collate_shapes_padding_and_content():
    rows, lat = rows_and_latents()
    b = D.collate(rows[:5], lat)
    B, L, T = 5, max(len(r["ids"]) for r in rows[:5]), max(r["frames"] for r in rows[:5])
    assert b["ids"].shape == (B, L) and b["fw"].shape == (B, T) and b["x1"].shape == (B, T, 64)
    assert b["x1"].dtype == np.float32
    for i, r in enumerate(rows[:5]):
        n, f = len(r["ids"]), r["frames"]
        assert (b["ids"][i, n:] == 0).all() and (b["ids"][i, :n] != 0).all()
        assert (b["cw"][i, n:] == -1).all() and b["cw"][i, :n].tolist() == D.word_bounds(r["ema_text"])[0]
        assert (b["fw"][i, f:] == -1).all() and (b["fw"][i, :f] >= 0).all()
        assert b["nw"][i] == len(r["word_frames"]) and b["wf"][i].sum() == f
        np.testing.assert_array_equal(b["x1"][i, :f], lat[r["offset"]: r["offset"] + f].astype(np.float32))
        assert (b["x1"][i, f:] == 0).all()
        assert b["style"][i] == r["style_id"] and b["frames"][i] == f and b["letters"][i] == n
        # frame word index never exceeds the letter-side word count
        assert b["fw"][i, :f].max() == b["cw"][i, :n].max()


def test_check_row_accepts_good_and_rejects_bad():
    rows, _ = rows_and_latents()
    for r in rows:
        D.check_row(r)
    bad = dict(rows[0], word_frames=rows[0]["word_frames"] + [3])
    with pytest.raises(ValueError, match="word_frames"):
        D.check_row(bad)
    bad = dict(rows[0], frames=rows[0]["frames"] + 1)
    with pytest.raises(ValueError, match="sum"):
        D.check_row(bad)
    bad = dict(rows[0], ids=rows[0]["ids"][:-1])
    with pytest.raises(ValueError, match="ids"):
        D.check_row(bad)
    bad = dict(rows[0], style_id=8)
    with pytest.raises(ValueError, match="style"):
        D.check_row(bad)


def test_read_index_filters(tmp_path):
    rows, _ = rows_and_latents()
    rows[1]["use"] = False
    (tmp_path / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tr = D.read_index(tmp_path, "train")
    assert all(r["split"] == "train" and r["use"] for r in tr) and "r1" not in {r["id"] for r in tr}
    assert len(D.read_index(tmp_path, None, use_only=False)) == len(rows)
    assert all(r["frames"] <= 10 for r in D.read_index(tmp_path, None, max_frames=10))


def test_bucket_batches_cover_every_row_once_within_budget():
    rows = [{"frames": f} for f in random.Random(1).choices(range(5, 300), k=500)] + [{"frames": 900}]
    batches = D.bucket_batches(rows, frame_budget=1200, max_batch=16, rng=random.Random(0))
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(len(rows)))
    for b in batches:
        longest = max(rows[i]["frames"] for i in b)
        assert len(b) <= 16
        assert len(b) * longest <= 1200 or len(b) == 1
    # similar lengths inside a batch (sorted buckets): the spread is small compared with the whole range
    spreads = [max(rows[i]["frames"] for i in b) - min(rows[i]["frames"] for i in b) for b in batches]
    assert np.median(spreads) < 30


def test_bucket_batches_deterministic_per_seed_and_differ_between_epochs():
    rows = [{"frames": f} for f in random.Random(2).choices(range(5, 300), k=300)]
    a = D.bucket_batches(rows, 2000, 64, random.Random(7))
    b = D.bucket_batches(rows, 2000, 64, random.Random(7))
    c = D.bucket_batches(rows, 2000, 64, random.Random(8))
    assert a == b and a != c


def test_lr_schedule_warmup_cosine():
    lr = 1e-4
    assert D.lr_at(0, lr, 100, 1000) == pytest.approx(1e-6)
    assert D.lr_at(99, lr, 100, 1000) == pytest.approx(lr)
    assert D.lr_at(100, lr, 100, 1000) == pytest.approx(lr)
    assert D.lr_at(550, lr, 100, 1000, 0.1) == pytest.approx(lr * (0.1 + 0.9 * 0.5))
    assert D.lr_at(1000, lr, 100, 1000, 0.1) == pytest.approx(lr * 0.1)
    assert D.lr_at(5000, lr, 100, 1000, 0.1) == pytest.approx(lr * 0.1)
    vals = [D.lr_at(s, lr, 100, 1000) for s in range(100, 1001)]
    assert all(x >= y for x, y in pairwise(vals))
    assert D.lr_at(10, lr, 0, 0) == lr


def test_checkpoint_files_latest_and_prune(tmp_path):
    assert D.latest_ckpt(tmp_path / "none") is None
    for s in (500, 1000, 1500, 2000, 2500, 3000):
        (tmp_path / D.ckpt_name(s)).write_bytes(b"x")
    (tmp_path / "ckpt_00003500.pt.tmp").write_bytes(b"partial")  # an interrupted save is never "latest"
    (tmp_path / "log.jsonl").write_text("", encoding="utf-8")
    assert D.ckpt_name(1500) == "ckpt_00001500.pt"
    assert D.latest_ckpt(tmp_path).name == "ckpt_00003000.pt"
    gone = D.prune_ckpts(tmp_path, keep=2, keep_every=1000)
    assert sorted(p.name for p in gone) == ["ckpt_00000500.pt", "ckpt_00001500.pt"]
    assert [s for s, _ in D.list_ckpts(tmp_path)] == [1000, 2000, 2500, 3000]


def test_append_jsonl(tmp_path):
    p = tmp_path / "log.jsonl"
    D.append_jsonl(p, {"a": 1})
    D.append_jsonl(p, {"ş": "ğ"})
    assert [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] == [{"a": 1}, {"ş": "ğ"}]


def test_parse_student_spec():
    assert D.parse_student("student:a=data/x.pt") == {"name": "a", "ckpt": "data/x.pt", "sampler": "euler",
                                                       "steps": 16, "cfg": 1.0, "sway": 0.0, "decoder": None}
    s = D.parse_student("student:b=ema.pt@midpoint,8,2,-1")
    assert (s["ckpt"], s["sampler"], s["steps"], s["cfg"], s["sway"]) == ("ema.pt", "midpoint", 8, 2.0, -1.0)
    assert D.parse_student("student:c=C:/x/ckpt.pt@ema4")["ckpt"] == "C:/x/ckpt.pt"
    assert D.parse_student("student:d=x.pt@euler,16,2,0,runs/dec/decoder_1.pt")["decoder"] == "runs/dec/decoder_1.pt"
    for bad in ("ema", "student:=x.pt", "student:a=x.pt@rk4"):
        with pytest.raises(ValueError):
            D.parse_student(bad)


def test_styles_table():
    assert D.STYLES[0] == "neutral" and D.N_STYLES == 8 and D.STYLES.index("surprised") == 6


# ---- listen_page ----------------------------------------------------------------------------------------------------
def _tone(seconds, amp, f=440.0, sr=48000):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float64)


def test_lufs_reference_tone():
    pytest.importorskip("scipy")
    x = _tone(3.0, 1.0, f=1000.0)  # BS.1770: a full-scale 1 kHz sine reads -3.01 LUFS
    assert lp.lufs(x) == pytest.approx(-3.01, abs=0.1)
    assert lp.lufs(x * 0.1) == pytest.approx(lp.lufs(x) - 20, abs=0.01)
    assert lp.lufs(np.zeros(48000)) == float("-inf")
    assert lp.lufs(np.ones(100)) == float("-inf")


def test_lufs_gating_ignores_silence():
    pytest.importorskip("scipy")
    x = _tone(2.0, 0.1, f=1000.0)
    y = np.concatenate([x, np.zeros(48000 * 4)])
    assert lp.lufs(y) == pytest.approx(lp.lufs(x), abs=0.5)  # ungated it would be 4.8 dB lower (2 s in 6 s)


def test_normalize_hits_target_and_peak_limit():
    pytest.importorskip("scipy")
    y = lp.normalize(_tone(2.0, 0.01, f=1000.0), target=-20.0)
    assert lp.lufs(y) == pytest.approx(-20.0, abs=0.05)
    loud = lp.normalize(_tone(2.0, 0.01, f=1000.0), target=0.0, peak_db=-1.0)
    assert np.abs(loud).max() <= 10 ** (-1 / 20) + 1e-6
    with pytest.raises(ValueError):
        lp.k_weight(np.zeros(10), 16000)


def test_assign_codes_unique_files_random_order():
    sents = [{"key": ["trzor", f"s{i}"], "text": f"cümle {i}"} for i in range(10)]
    systems = ["st", "ema", "antalia", "teacher"]
    items = lp.assign(sents, systems, random.Random(3))
    codes = [c["code"] for it in items for c in it["clips"]]
    files = [c["file"] for it in items for c in it["clips"]]
    assert len(set(codes)) == len(codes) == 40 and len(set(files)) == 40
    assert all(len(c) == 2 and c[0] in lp.CODE_CHARS and c[1] in "23456789" for c in codes)
    assert all(f.startswith("s/") and f.endswith(".wav") and len(f) == 18 for f in files)
    for it in items:
        assert sorted(c["system"] for c in it["clips"]) == sorted(systems)
    orders = {tuple(c["system"] for c in it["clips"]) for it in items}
    assert len(orders) > 1
    assert lp.assign(sents, systems, random.Random(3)) == items


def test_page_hides_system_names_and_decode_roundtrip():
    sents = [{"key": ["trzor", f"s{i}"], "text": f"Cümle {i}?"} for i in range(3)]
    systems = ["st-final", "ema", "antalia", "teacher"]
    items = lp.assign(sents, systems, random.Random(5))
    html = lp.page_html(items, "Deneme")
    for s in systems:
        assert f'"{s}"' not in html and f"'{s}'" not in html
    data = json.loads(html.split("const T=", 1)[1].split(";const NL", 1)[0])
    assert [d["text"] for d in data] == [s["text"] for s in sents]
    assert [[c[0] for c in d["clips"]] for d in data] == [[c["code"] for c in it["clips"]] for it in items]
    # the text the page builds on "done": one line per sentence
    pick = [it["clips"][0]["code"] for it in items]
    pasted = "KUCUK MODEL DINLEME: 3 cümle\n" + "".join(
        f"{j + 1}: {p}  [{','.join(c['code'] for c in it['clips'])}]{'  yorum: robotik' if j == 1 else ''}\n"
        for j, (p, it) in enumerate(zip(pick, items)))
    res = lp.decode(pasted, {"items": items})
    want = [it["clips"][0]["system"] for it in items]
    assert [p["system"] for p in res["picks"]] == want
    assert sum(res["wins"].values()) == 3 and "robotik" in res["picks"][1]["rest"]


def test_listen_page_end_to_end(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("soxr")
    pytest.importorskip("scipy")
    bench = tmp_path / "bench"
    subset = []
    for st, i in lp.SENTENCES:
        subset.append({"set": st, "id": i, "text": f"metin {i}"})
        for k, s in enumerate(("a", "b")):
            p = bench / s / st / f"{i}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sr = 48000 if s == "a" else 24000
            sf.write(str(p), _tone(0.8, 0.05 * (k + 1), sr=sr).astype(np.float32), sr)
    (bench / "subset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in subset), encoding="utf-8")
    out = tmp_path / "bench" / "dinleme"
    assert lp.main(["--bench", str(bench), "--out", str(out), "--systems", "a", "b", "--seed", "1"]) == 0
    html = (out / "index.html").read_text(encoding="utf-8")
    key = json.loads((tmp_path / "bench" / "dinleme-anahtar.json").read_text(encoding="utf-8"))
    assert key["seed"] == 1 and len(key["items"]) == len(lp.SENTENCES)
    wavs = sorted((out / "s").glob("*.wav"))
    assert len(wavs) == 2 * len(lp.SENTENCES)
    for w in wavs[:4]:
        y, sr = sf.read(str(w))
        assert sr == 48000 and lp.lufs(y) == pytest.approx(-20.0, abs=0.2)
    assert '"a"' not in html and "metin ozel_04" in html


# ---- student (torch) -------------------------------------------------------------------------------------------------
TINY = {"d": 32, "n_layers": 2, "n_heads": 4, "latent_dim": 64, "text_conv": 1, "text_attn": 0, "align_heads": 2,
        "pos_scale": 4.0, "lookback": 1, "lookahead": 1, "dur_hidden": 16, "letter_pos": True, "shared_ada": True,
        "distilled": {"times": [0.0, 0.25, 0.5, 0.75]}}  # EMA's architecture, tiny (ema.pt: d 224, 6 layers)


def _torch_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ema_lightning")
    import student as S

    torch.manual_seed(0)
    return torch, S, S.StudentAcoustic(TINY, VOCAB, True)


def test_student_style_embedding_starts_at_zero_and_matches_acoustic():
    torch, S, m = _torch_model()
    from ema_lightning.model import Acoustic

    base = Acoustic(TINY, VOCAB, True)
    missing, unexpected = base.load_state_dict(m.state_dict(), strict=False)
    assert unexpected == ["style_emb.weight"] and not missing
    assert float(m.style_emb.weight.detach().abs().sum()) == 0.0
    rows, lat = rows_and_latents(4)
    bt = S.to_device(D.collate(rows[:3], lat), "cpu")
    mask = bt["ids"] != 0
    m.style = 5
    with torch.no_grad():
        h, d = m.text_stage(bt["ids"], mask)
        hb, db = base.text_stage(bt["ids"], mask)
    torch.testing.assert_close(h, hb)
    torch.testing.assert_close(d, db)


def test_student_ema4_sampler_is_acoustic_sound_stage_and_euler_runs():
    torch, S, m = _torch_model()
    from ema_lightning.model import Acoustic

    m.eval()
    rows, lat = rows_and_latents(4)
    b = S.to_device(D.collate(rows[:3], lat), "cpu")
    mask = b["ids"] != 0
    fmask = b["fw"] >= 0
    noise = torch.randn(3, 4, b["fw"].shape[1], 64)
    with torch.no_grad():
        h, dur = m.text_stage(b["ids"], mask)
        m.set_sampling("ema4")
        a = m.sound_stage(h, dur, mask, b["cw"], b["wstart"], b["fw"], b["fp"], fmask, noise)
        ref = Acoustic.sound_stage(m, h, dur, mask, b["cw"], b["wstart"], b["fw"], b["fp"], fmask, noise)
        torch.testing.assert_close(a, ref)
        for sampler, cfg in (("euler", 1.0), ("euler", 2.0), ("midpoint", 1.5)):
            m.set_sampling(sampler, 4, cfg, -1.0)
            x = m.sound_stage(h, dur, mask, b["cw"], b["wstart"], b["fw"], b["fp"], fmask, noise)
            assert x.shape == (3, b["fw"].shape[1], 64) and torch.isfinite(x).all()
        # CFG batched in one pass equals the two separate passes
        from ema_lightning.model import rope

        cond = m.condition(h, dur, mask, b["cw"], b["wstart"], b["fw"], b["fp"])
        cos, sin = rope(torch.arange(b["fw"].shape[1]), m.dh)
        x = noise[:, 0]
        v = m.velocity(x, cond, 0.3, fmask, cos, sin, 2.0)
        t = torch.full((3,), 0.3)
        vc = m.backbone(x, cond, t, fmask, cos, sin)
        vu = m.backbone(x, torch.zeros_like(cond), t, fmask, cos, sin)
        torch.testing.assert_close(v, vu + 2.0 * (vc - vu), rtol=1e-4, atol=1e-5)


def test_time_grid():
    pytest.importorskip("torch")
    pytest.importorskip("ema_lightning")
    import student as S

    g = S.time_grid(4)
    assert g == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
    s = S.time_grid(8, -1.0)
    assert s[0] == pytest.approx(0) and s[-1] == pytest.approx(1)
    assert all(b > a for a, b in pairwise(s)) and s[1] < 1 / 8  # denser near t = 0


def test_losses_masked_finite_and_padding_invariant():
    torch, S, m = _torch_model()
    rows, lat = rows_and_latents(6)
    one = S.to_device(D.collate(rows[:1], lat), "cpu")
    two = S.to_device(D.collate(rows[:1] + [max(rows, key=lambda r: r["frames"])], lat), "cpu")
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        lf, ld = S.losses(m, one, 0.0, 0.0, torch.tensor([0.4]), gen=g)
        assert torch.isfinite(lf) and torch.isfinite(ld) and lf > 0
        # padding the first clip (batched with a longer one) does not change its own velocity on real frames
        mask = two["ids"] != 0
        from ema_lightning.model import rope

        h = m.features(two["ids"], mask, two["style"])
        dur = m.chardur(h, mask)
        cond = m.condition(h, dur, mask, two["cw"], two["wstart"], two["fw"], two["fp"])
        T = two["fw"].shape[1]
        x = torch.randn(2, T, 64)
        v2 = m.backbone(x, cond, torch.full((2,), 0.4), two["fw"] >= 0, *rope(torch.arange(T), m.dh))
        n, f = len(rows[0]["ids"]), rows[0]["frames"]
        h1 = m.features(two["ids"][:1, :n], mask[:1, :n], two["style"][:1])
        d1 = m.chardur(h1, mask[:1, :n])
        c1 = m.condition(h1, d1, mask[:1, :n], two["cw"][:1, :n], two["wstart"][:1, :n], two["fw"][:1, :f],
                         two["fp"][:1, :f])
        v1 = m.backbone(x[:1, :f], c1, torch.full((1,), 0.4), two["fw"][:1, :f] >= 0, *rope(torch.arange(f), m.dh))
        torch.testing.assert_close(v1, v2[:1, :f], rtol=1e-4, atol=1e-4)


def test_one_batch_overfit():
    torch, S, m = _torch_model()
    rows, lat = rows_and_latents(4)
    bt = S.to_device(D.collate(rows[:4], lat), "cpu")
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    first = None
    for i in range(150):
        g.manual_seed(0)  # the same x0 and t every step: a fixed regression target
        lf, ld = S.losses(m, bt, 0.0, 0.0, torch.full((4,), 0.5), gen=g)
        if first is None:
            first = (float(lf), float(ld))
        opt.zero_grad()
        (lf + ld).backward()
        opt.step()
    assert float(lf) < 0.3 * first[0] and float(ld) < 0.3 * first[1]
