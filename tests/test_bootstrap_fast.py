# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Exactness of the precomputed/reweighted evaluation vs. explicit rename-and-duplicate."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.metrics.event_ap import PrecomputedEDAP, event_detection_ap, cap_candidates
from src.metrics.clinical import icc2_1
from src.data.clock import series_clock_entry, clock_minutes_from_noon
from src.metrics.clinical import wall_minutes_from_noon


def _toy(seed=0, n_series=6, nights=5):
    rng = np.random.default_rng(seed)
    t, p = [], []
    for i in range(n_series):
        for n in range(nights):
            base = n * 17280
            for e, off in (("onset", 3000), ("wakeup", 9000)):
                gt = base + off + int(rng.integers(-50, 50))
                t.append((f"s{i}", gt, e))
                for _ in range(8):
                    p.append((f"s{i}", gt + int(rng.integers(-400, 400)), e,
                              float(np.round(rng.random(), 1)), f"d{n}"))  # rounding -> ties
    return (pd.DataFrame(t, columns=["series_id", "step", "event"]),
            pd.DataFrame(p, columns=["series_id", "step", "event", "score", "night_date"]))


def _explicit(truth, pred, draw):
    tt, pp = [], []
    for r, sid in enumerate(draw):
        tt.append(truth[truth.series_id == sid].assign(series_id=f"{sid}@{r}"))
        pp.append(pred[pred.series_id == sid].assign(series_id=f"{sid}@{r}"))
    return event_detection_ap(pd.concat(tt), pd.concat(pp))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reweighted_edap_equals_explicit_duplication(seed):
    truth, pred = _toy(seed)
    pre = PrecomputedEDAP(truth, pred)
    rng = np.random.default_rng(seed + 10)
    for _ in range(5):
        draw = rng.choice(sorted(truth.series_id.unique()), size=6, replace=True)
        u, c = np.unique(draw, return_counts=True)
        w = np.array([dict(zip(u, c)).get(s, 0) for s in pre.series], float)
        assert abs(pre.score(w) - _explicit(truth, pred, draw)) < 1e-12


def test_capped_reweighting_equals_explicit():
    truth, pred = _toy(3)
    capped = cap_candidates(pred, 1)
    pre = PrecomputedEDAP(truth, capped)
    draw = np.array(["s0", "s0", "s2", "s5", "s5", "s5"])
    u, c = np.unique(draw, return_counts=True)
    w = np.array([dict(zip(u, c)).get(s, 0) for s in pre.series], float)
    assert abs(pre.score(w) - _explicit(truth, capped, draw)) < 1e-12


def test_icc_closed_form_matches_pingouin():
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(0)
    for n in (5, 50, 200):
        a = rng.normal(520, 100, n); b = a + rng.normal(10, 30, n)
        long = pd.DataFrame({"t": np.repeat(np.arange(n), 2), "r": np.tile(["a", "b"], n),
                             "y": np.column_stack([a, b]).reshape(-1)})
        tab = pg.intraclass_corr(long, targets="t", raters="r", ratings="y").set_index("Type")
        label = "ICC2" if "ICC2" in tab.index else "ICC(A,1)"  # pingouin renamed labels in newer versions
        ref = tab.loc[label, "ICC"]
        assert abs(icc2_1(a, b) - ref) < 1e-8


def test_clock_index_matches_timestamp_text_across_offset_change():
    ts = pd.Series(["2018-11-10T23:59:50-0400", "2018-11-10T23:59:55-0400",
                    "2018-11-11T01:59:55-0400", "2018-11-11T01:00:00-0500",
                    "2018-11-11T07:00:00-0500"])
    steps = np.array([0, 1, 1441, 1442, 5762])
    entry = series_clock_entry(steps, ts)
    got = clock_minutes_from_noon(entry, steps, ["2018-11-10"] * 5)
    ref = [wall_minutes_from_noon(x, "2018-11-10") for x in ts]
    assert np.allclose(got, ref, atol=1e-9)
