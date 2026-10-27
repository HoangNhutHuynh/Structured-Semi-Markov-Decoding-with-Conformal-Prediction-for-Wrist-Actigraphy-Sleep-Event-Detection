# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Round-trip and physics checks for the pure-Python GENEActiv reader."""
import numpy as np
import pandas as pd
import pytest
from src.data.geneactiv import read_bin, write_synthetic_bin, to_epochs, autocalibrate


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 300 * 40                                   # 40 pages ~ 140 s at 85.7 Hz
    t = np.arange(n)
    xyz = np.column_stack([0.1 * np.sin(t / 500), 0.2 * np.cos(t / 700),
                           np.sqrt(np.clip(1 - 0.01 - 0.04, 0, None)) * np.ones(n)])
    xyz += rng.normal(0, 0.002, xyz.shape)
    p = tmp_path_factory.mktemp("ga") / "s.bin"
    write_synthetic_bin(p, xyz)
    return p, xyz


def test_roundtrip_values_and_time(synth):
    p, xyz = synth
    rec = read_bin(p)
    assert rec.xyz.shape == xyz.shape
    step = 100 / 25548                                        # coarsest axis: 100/gain ~ 3.9 mg
    assert np.abs(rec.xyz - xyz).max() < step / 2 + 1e-6      # quantisation only
    assert abs(rec.sample_rate - 85.7) < 1e-6
    dt = np.diff(rec.time.astype("datetime64[ns]").astype(np.int64)) / 1e9
    assert abs(np.median(dt) - 1 / 85.7) < 1e-4


def test_negative_values_use_twos_complement(tmp_path):
    xyz = np.tile(np.array([[-0.5, 0.25, -1.0]]), (300, 1))
    p = tmp_path / "neg.bin"
    write_synthetic_bin(p, xyz)
    rec = read_bin(p)
    assert np.abs(rec.xyz - xyz).max() < 100 / 25548 / 2 + 1e-6


def test_epochs_match_ggir_definitions(synth):
    p, xyz = synth
    rec = read_bin(p)
    ep = to_epochs(rec, epoch_s=5)
    n = int(round(5 * rec.sample_rate))
    w = xyz[: len(ep) * n].reshape(len(ep), n, 3)
    med = np.median(w, axis=1)
    want_angle = np.degrees(np.arctan2(med[:, 2], np.hypot(med[:, 0], med[:, 1])))
    want_enmo = np.maximum(np.linalg.norm(w, axis=2) - 1, 0).mean(axis=1)
    assert np.abs(ep.anglez.to_numpy() - want_angle).max() < 0.3
    assert np.abs(ep.enmo.to_numpy() - want_enmo).max() < 3e-3
    assert (ep.enmo >= 0).all()


def test_autocalibration_recovers_known_distortion():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(4000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)          # perfectly still, |a| = 1
    true_scale, true_offset = np.array([1.05, 0.96, 1.02]), np.array([0.02, -0.03, 0.01])
    bad = (pts - true_offset) / true_scale
    xyz = np.repeat(bad, 860, axis=0)                          # 10-s static windows at 86 Hz
    scale, offset, n_static, before, after = autocalibrate(xyz, fs=86.0)
    assert n_static > 100 and after < before / 5
    fixed = xyz * scale + offset
    assert abs(np.abs(np.linalg.norm(fixed, axis=1) - 1).mean()) < 0.005


def test_gain_convention_matches_real_header_scale():
    """A still wrist at 1 g must decode to ~1 g with real GENEActiv header gains (~25700)."""
    xyz = np.tile(np.array([[0.0, 0.0, 1.0]]), (600, 1))
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mkdtemp()) / "g.bin"
    write_synthetic_bin(p, xyz)
    rec = read_bin(p)
    assert abs(np.linalg.norm(rec.xyz, axis=1).mean() - 1.0) < 0.005
