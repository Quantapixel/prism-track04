"""Fast tests that need no model weights or datasets:  python -m pytest -q tests/"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "tools"))

from prism import vocab  # noqa: E402
from prism.actions import detect_training_free, to_json_actions  # noqa: E402
from prism.config import Config  # noqa: E402
from prism.features import FeatureExtractor  # noqa: E402
from prism.head import TemporalHead, build_targets, giou_1d, head_detect, sigmoid_focal  # noqa: E402
from prism.highlights import coverage_highlights, finalize, importance_highlights  # noqa: E402
from prism.schema import validate_obj  # noqa: E402
from prism.video_io import FrameStream, keyframes, probe  # noqa: E402


def _check_highlights(hl, duration, allow_over=False):
    segs = [h["segment"] for h in hl]
    assert segs, "need >= 1 highlight"
    assert any(b - a >= 4.0 for a, b in segs)
    for (a0, b0), (a1, b1) in zip(segs, segs[1:]):
        assert b0 <= a1 and a1 > a0
    for a, b in segs:
        assert 0 <= a < b <= duration + 1e-9
        assert round(a, 2) == a and round(b, 2) == b
    total = sum(b - a for a, b in segs)
    if not allow_over:
        assert total <= 0.15 * duration + 1e-9, (total, 0.15 * duration)
    return total


def _obj(duration, highlights, actions=()):
    return {"team_id": "T", "video_id": "x", "duration_sec": duration,
            "label": {"name": "cooking", "score": 0.9}, "actions": list(actions), "highlights": highlights}


# ---------------------------------------------------------------- vocab
def test_vocab_alignment():
    assert len(vocab.verbs()) == 98 and len(vocab.nouns()) == 301
    assert len(vocab.actions()) == 3806
    v, n = vocab.action_verb_noun_idx()
    assert v.max() < 97 and n.max() < 300


# ---------------------------------------------------------------- highlights
@pytest.mark.parametrize("duration", [11.13, 15.28, 25.16, 29.51, 93.76, 561.53, 1968.6])
def test_coverage_highlights_constraints(duration):
    cfg = Config()
    rng = np.random.default_rng(0)
    dets = []
    for _ in range(400):
        s = rng.uniform(0, max(duration - 2, 0.1))
        dets.append((s, min(duration, s + rng.uniform(0.5, 6)), int(rng.integers(0, 3806)), float(rng.uniform(0.01, 0.9))))
    hl = coverage_highlights(dets, duration, cfg)
    short = 0.15 * duration < 4.0
    total = _check_highlights(hl, duration, allow_over=short)
    if not short:
        assert total >= 0.15 * duration - 4.1  # budget is (nearly) used
    errors, _ = validate_obj(_obj(duration, hl))
    assert not errors, errors


def test_coverage_prefers_distinct_pairs():
    cfg = Config()
    duration = 200.0  # budget 30 s
    dets = [(10 + i * 0.1, 12 + i * 0.1, 0, 0.9) for i in range(30)]  # one pair repeated
    dets += [(100 + 10 * k, 102 + 10 * k, 10 + k, 0.5) for k in range(6)]  # 6 distinct pairs
    hl = coverage_highlights(dets, duration, cfg)
    segs = [h["segment"] for h in hl]
    covered = sum(any(a <= 101 + 10 * k <= b for a, b in segs) for k in range(6))
    assert covered >= 5


def test_importance_highlights_constraints():
    cfg = Config()
    for duration in (37.34, 96.32, 377.07):
        grid = np.arange(0, duration, 0.25) + 0.125
        score = np.sin(grid / 7.0) + 0.1 * np.random.default_rng(1).standard_normal(len(grid))
        hl = importance_highlights(grid, score, duration, cfg)
        total = _check_highlights(hl, duration)
        assert total >= 0.15 * duration * 0.95
        # the chosen time should have higher mean score than the video
        mask = np.zeros(len(grid), bool)
        for h in hl:
            a, b = h["segment"]
            mask |= (grid >= a) & (grid <= b)
        assert score[mask].mean() > score.mean()


def test_finalize_merges_and_rounds():
    out = finalize([(10.004, 14.026), (14.026, 19.0), (30.0, 31.0)], 100.0, 4.02)
    segs = [o["segment"] for o in out]
    assert segs[0][0] >= 10.0 and segs[0][1] <= 19.0 and len(segs) == 2


# ---------------------------------------------------------------- schema
def test_schema_package_examples():
    ex = json.loads((ROOT / "package" / "P02_109.json").read_text())
    errors, _ = validate_obj(ex)
    assert not errors, errors
    ski = json.loads((ROOT / "package" / "skiing_03.json").read_text())
    errors, _ = validate_obj(ski)
    assert any("label.name" in e for e in errors)  # example uses "skiing"; vocab says "ski"


def test_schema_rejects_overlap_and_short():
    errors, _ = validate_obj(_obj(100.0, [{"segment": [0, 5]}, {"segment": [4, 9]}]))
    assert errors
    errors, _ = validate_obj(_obj(100.0, [{"segment": [0, 3]}]))
    assert errors


# ---------------------------------------------------------------- training-free detector
def test_detector_recovers_planted_action():
    cfg = Config()
    n = 200
    centers = (np.arange(n) * 0.5 + 32 / 15 / 2).astype(np.float32)
    logits = np.random.default_rng(0).normal(0, 1, (n, 3806)).astype(np.float32)
    a = vocab.pair_to_action()[(2, 0)]  # wash tap
    logits[80:90, a] += 12
    feats = {"omni_logits": logits.astype(np.float16), "win_center": centers, "duration": np.array(102.0)}
    dets = detect_training_free(feats, cfg)
    assert dets and dets[0][2] == a
    s, e = dets[0][0], dets[0][1]
    assert s < centers[85] < e and e - s < 12
    acts = to_json_actions(dets, 102.0)
    assert acts[0]["verb"] == "wash" and acts[0]["noun"] == "tap"


# ---------------------------------------------------------------- learned head
def test_head_learns_synthetic():
    torch.manual_seed(0)
    T, C = 256, 32
    model = TemporalHead(in_dim=C, d=64, n_layers=1, n_heads=2, n_levels=3)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    n_cls = model.n_verbs + model.n_nouns
    rng = np.random.default_rng(0)
    losses = []
    for it in range(60):
        x = rng.normal(0, 1, (2, T, C)).astype(np.float32)
        gts = []
        for b in range(2):
            g = []
            for _ in range(6):
                s = rng.integers(0, T - 20)
                L = rng.integers(2, 16)
                x[b, s:s + L, 0] += 4.0  # actionness signal
                x[b, s:s + L, 1] += 4.0 * (L > 8)
                g.append((s, s + L, 3, 7))
            gts.append(np.array(g, float))
        tg = [build_targets(T, g, 3, n_cls, model.n_verbs) for g in gts]
        cls_o, reg_o, masks = model(torch.from_numpy(x), torch.ones(2, T, dtype=torch.bool))
        lc = lr = 0
        npos = 0
        for lvl in range(3):
            ct = torch.from_numpy(np.stack([t[lvl][0] for t in tg]))
            rt = torch.from_numpy(np.stack([t[lvl][1] for t in tg]))
            ps = torch.from_numpy(np.stack([t[lvl][2] for t in tg]))
            lc = lc + sigmoid_focal(cls_o[lvl], ct)
            if ps.any():
                lr = lr + giou_1d(reg_o[lvl][ps], rt[ps])
            npos += int(ps.sum())
        loss = (lc + lr) / max(npos, 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert np.mean(losses[-10:]) < 0.6 * np.mean(losses[:10])
    model.eval()
    feats = {"omni_feat": x[0].astype(np.float16), "win_center": (np.arange(T) * 0.5 + 1.07).astype(np.float32),
             "duration": np.array(T * 0.5 + 2)}
    cfg = Config(head_fuse_omni=0.0)
    dets = head_detect(model, feats, cfg, "cpu")
    assert len(dets) > 0
    assert all(0 <= d[0] < d[1] for d in dets)


# ---------------------------------------------------------------- video decode + windowing
@pytest.fixture(scope="module")
def synth_video(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "synth.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=7.3",
                    "-pix_fmt", "yuv420p", "-g", "30", str(p)], check=True)
    return str(p)


def test_probe_and_stream(synth_video):
    info = probe(synth_video)
    assert abs(info.duration - 7.3) < 0.1 and abs(info.fps - 30) < 0.01
    frames = list(FrameStream(synth_video, fps=15, size=224))
    assert abs(len(frames) - 7.3 * 15) <= 2
    assert frames[0][1].shape == (224, 224, 3)
    kf = keyframes(synth_video, max_frames=48)
    assert 5 <= len(kf) <= 9


class _FakeOmni(torch.nn.Module):
    def forward(self, x):
        b = x.shape[0]
        m = x.float().mean(dim=(1, 2, 3, 4))
        return m[:, None].repeat(1, 1024), m[:, None].repeat(1, 3806)


class _FakeClip:
    def encode_images(self, frames):
        e = np.ones((len(frames), 8), np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def test_feature_windows_cover_video(synth_video):
    cfg = Config()
    fx = FeatureExtractor(cfg, "cpu", omnivore=_FakeOmni(), clip=_FakeClip())
    f = fx.extract(synth_video, log=lambda *a: None)
    n = int(f["n_frames"])
    span = float(f["win_span"])
    c = f["win_center"]
    assert np.all(np.diff(c) > 0)
    assert c[0] - span / 2 == pytest.approx(0.0, abs=1e-5)
    assert c[-1] + span / 2 == pytest.approx(n / 15.0, abs=1e-3)  # tail covered
    assert np.allclose(np.diff(c)[:-1], 0.5, atol=0.07)
    assert len(f["clip_emb"]) == len(f["clip_t"]) == int(np.ceil(n / 15))
    assert len(f["motion"]) == n


def test_feature_short_video(tmp_path):
    p = tmp_path / "short.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=1.2",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    fx = FeatureExtractor(Config(), "cpu", omnivore=_FakeOmni(), clip=_FakeClip())
    f = fx.extract(str(p), log=lambda *a: None)
    assert len(f["win_center"]) == 1 and f["omni_logits"].shape == (1, 3806)
