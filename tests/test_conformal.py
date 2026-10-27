# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Finite-sample validity of the conformal machinery (independent of the sleep model)."""
import numpy as np
from scripts.run_p3 import conformal_quantile, hpd_score, hpd_width, hpd_interval


def test_split_conformal_marginal_coverage():
    rng = np.random.default_rng(0)
    cov = []
    for _ in range(400):
        cal, test = rng.normal(size=200), rng.normal(size=50)
        q = conformal_quantile(np.abs(cal), 0.1)
        cov.append((np.abs(test) <= q).mean())
    assert 0.885 < float(np.mean(cov)) < 0.915       # nominal 90%


def test_quantile_is_infinite_when_calibration_too_small():
    assert conformal_quantile(np.arange(5.0), 0.05) == float("inf")   # n+1 < 1/alpha
    assert np.isfinite(conformal_quantile(np.arange(100.0), 0.05))


def test_hpd_set_is_nested_and_contains_mode():
    p = np.array([0.4, 0.3, 0.2, 0.08, 0.02])
    assert hpd_score(p, 0) == 0.0                     # mode has nothing above it
    assert abs(hpd_score(p, 2) - 0.7) < 1e-12
    assert hpd_width(p, 0.0) == 1 and hpd_width(p, 0.65) == 2 and hpd_width(p, 0.99) == 5
    assert hpd_width(p, 0.3) <= hpd_width(p, 0.8)     # nested in q
    assert hpd_interval(p, 0.65, 0) == (0, 1)


def test_hpd_coverage_is_calibrated_on_synthetic_posteriors():
    rng = np.random.default_rng(1)
    T = 200

    def draw():
        mu, sig = rng.integers(20, T - 20), rng.uniform(1, 25)
        g = np.exp(-0.5 * ((np.arange(T) - mu) / sig) ** 2); g /= g.sum()
        return g, int(np.clip(rng.normal(mu, sig), 0, T - 1))   # truth ~ the posterior

    cal = [draw() for _ in range(500)]
    s = np.array([hpd_score(m, t) for m, t in cal])
    q = conformal_quantile(s, 0.1)
    test = [draw() for _ in range(2000)]
    assert 0.86 < np.mean([hpd_score(m, t) <= q for m, t in test]) < 0.94


def test_hpd_set_and_score_agree_exactly_under_ties():
    """The conformal guarantee needs set == {u : score(u) <= q}; float16 posteriors tie a lot."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        m = np.round(rng.random(60) * 4) / 4.0          # heavy ties, including exact zeros
        m[rng.random(60) < 0.5] = 0.0
        if m.sum() == 0:
            continue
        for q in (0.0, 0.1, 0.37, 0.8, 1.0):
            direct = {u for u in range(len(m)) if hpd_score(m, u) <= q}
            assert len(direct) == hpd_width(m, q)


def test_hpd_coverage_exact_with_tied_discrete_posteriors():
    rng = np.random.default_rng(3)

    def draw():
        m = np.round(rng.random(40) * 3) / 3.0           # many ties
        m[rng.random(40) < 0.4] = 0.0
        if m.sum() == 0:
            m[0] = 1.0
        p = m / m.sum()
        return m, int(rng.choice(len(p), p=p))

    cal = [draw() for _ in range(1000)]
    q = conformal_quantile(np.array([hpd_score(m, t) for m, t in cal]), 0.1)
    test = [draw() for _ in range(4000)]
    cov = np.mean([hpd_score(m, t) <= q for m, t in test])
    assert cov >= 0.895, cov                              # conformal is conservative, never below
