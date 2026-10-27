# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
import numpy as np
from src.baselines.hdcza import (WindowInput, rolling_signal, hdcza_window, hdcza_window_grid,
                                 blocks_from_candidates, DEFAULT)

SPM = 12


def _night(seed=0, n=24 * 60 * SPM, sleep=(7 * 60, 15 * 60), nonwear=None):
    """15:00-start window; active (noisy) except a still sleep period [min from 15:00)."""
    rng = np.random.default_rng(seed)
    a = (30 * np.sin(np.arange(n) / 500) + rng.normal(0, 8, n)).clip(-90, 90)
    s, e = sleep[0] * SPM, sleep[1] * SPM
    a[s:e] = a[s] + rng.normal(0, 0.05, e - s)
    for k in range(s + 90 * SPM, e, 120 * SPM):        # posture shifts during sleep
        a[k:e] += 20
    worn = np.ones(n, bool)
    if nonwear:
        worn[nonwear[0] * SPM:nonwear[1] * SPM] = False
        a[nonwear[0] * SPM:nonwear[1] * SPM] = 10.0     # flat (looks like sleep)
    return WindowInput(np.arange(n) + 1000, rolling_signal(a), worn), s, e


def test_detects_sleep_period():
    w, s, e = _night()
    on, wk, score = hdcza_window(w, **DEFAULT)
    assert abs((on - 1000) - s) <= 10 * SPM and abs((wk - 1000) - e) <= 10 * SPM
    assert 0 < score <= 1


def test_nonwear_never_sleep_and_excluded():
    w, s, e = _night(nonwear=(60, 300))    # 4 h flat non-wear in the afternoon
    on, wk, _ = hdcza_window(w, **DEFAULT)
    assert (on - 1000) >= 300 * SPM        # block cannot start inside non-wear


def test_low_wear_window_gives_none():
    w, _, _ = _night(nonwear=(0, 13 * 60))
    assert hdcza_window(w, **DEFAULT) is None


def test_gap_merge_requires_worn_gap():
    cand = np.zeros(200 * SPM, bool); cand[:40 * SPM] = True; cand[60 * SPM:100 * SPM] = True
    worn = np.ones_like(cand)
    assert blocks_from_candidates(cand, worn, 30, 60) == (0, 100 * SPM)
    worn[45 * SPM] = False
    assert blocks_from_candidates(cand, worn, 30, 60) == (0, 40 * SPM)


def test_grid_matches_single_call():
    w, _, _ = _night(seed=3)
    g = hdcza_window_grid(w)
    for combo in [(10, 15, 30, 60), (5, 5, 20, 30), (20, 20, 45, 45)]:
        assert g[combo] == hdcza_window(w, *combo)


def test_threshold_clamped_when_percentile_is_zero():
    w, s, e = _night(seed=5)
    r = w.r.copy(); r[r < np.percentile(r, 30)] = 0.0        # many exact zeros, like CMI stillness
    w0 = WindowInput(w.steps, r, w.worn)
    assert hdcza_window(w0, 10, 15, 30, 60, clamp=False) is None   # unbounded threshold = 0
    assert hdcza_window(w0, 10, 15, 30, 60) is not None            # GGIR-bounded threshold
