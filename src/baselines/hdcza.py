# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""B0 baseline: HDCZA sleep-period-time detection (van Hees et al. 2018, Sci Rep 8:12975).

Paper-faithful chain on the provided 5-s z-angle (anglez):
  d(t) = |anglez[t] - anglez[t-1]|  ->  r(t) = 5-min centred rolling median of d
  per window: thr = P-th percentile of r over WORN samples x M
  candidate = (r < thr) & worn; keep runs > Bmin; merge runs separated by gaps < G whose gap
  samples are all worn; longest block = SPT window (onset = first step, wakeup = last step + 1).
Design decisions: 15:00->15:00 recorded wall-clock windows; non-wear never sleep and excluded
from the percentile; <50% worn samples -> no window; EDAP score = block hours / 24.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
import pandas as pd

SAMPLES_PER_MIN = 12
DEFAULT = {"P": 10, "M": 15, "Bmin": 30, "G": 60}
# GGIR's HDCZA implementation bounds the data-driven threshold to [0.13, 0.50] degrees
# (verify against GGIR HASPT.R before citing). Required on CMI: anglez is often exactly constant
# during stillness, so the 10th percentile of r(t) is 0 in ~50% of labelled nights and an
# unbounded threshold (0 x 15 = 0) yields no candidate at all.
THR_MIN, THR_MAX = 0.13, 0.50


def _threshold(pct_value: float, M: float, clamp: bool = True) -> float:
    thr = pct_value * M
    return float(np.clip(thr, THR_MIN, THR_MAX)) if clamp else float(thr)
GRID = {"P": (5, 10, 15, 20), "M": (5, 10, 15, 20), "Bmin": (20, 30, 45), "G": (30, 45, 60)}


def rolling_signal(anglez: np.ndarray, window: int = 60) -> np.ndarray:
    a = np.asarray(anglez, dtype=float)
    d = np.abs(np.diff(a, prepend=a[:1]))
    if len(d) > 1:
        d[0] = d[1]
    return pd.Series(d).rolling(window, center=True, min_periods=1).median().to_numpy()


def _runs(mask: np.ndarray) -> np.ndarray:
    """(start, end_exclusive) of True runs."""
    if not mask.any():
        return np.zeros((0, 2), dtype=np.int64)
    m = np.r_[False, mask, False].astype(np.int8)
    dd = np.diff(m)
    return np.column_stack([np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)])


def blocks_from_candidates(cand: np.ndarray, worn: np.ndarray, bmin_min: int, gap_min: int,
                           worn_cum: np.ndarray | None = None) -> tuple[int, int] | None:
    """Longest SPT block (start, end_exclusive) in sample indices, or None."""
    runs = _runs(cand)
    if len(runs) == 0:
        return None
    runs = runs[(runs[:, 1] - runs[:, 0]) > bmin_min * SAMPLES_PER_MIN]
    if len(runs) == 0:
        return None
    if worn_cum is None:
        worn_cum = np.r_[0, np.cumsum(worn)]
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        gap = s - pe
        gap_all_worn = (worn_cum[s] - worn_cum[pe]) == gap
        if gap < gap_min * SAMPLES_PER_MIN and gap_all_worn:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    lengths = [e - s for s, e in merged]
    k = int(np.argmax(lengths))  # first longest on ties
    return int(merged[k][0]), int(merged[k][1])


@dataclass
class WindowInput:
    steps: np.ndarray
    r: np.ndarray
    worn: np.ndarray


def hdcza_window(w: WindowInput, P: float, M: float, Bmin: int, G: int,
                 min_worn_fraction: float = 0.5, clamp: bool = True):
    """Return (onset_step, wakeup_step, score) or None for one window."""
    n = len(w.steps)
    if n == 0 or w.worn.mean() < min_worn_fraction:
        return None
    thr = _threshold(np.percentile(w.r[w.worn], P), M, clamp)
    cand = (w.r < thr) & w.worn
    blk = blocks_from_candidates(cand, w.worn, Bmin, G)
    if blk is None:
        return None
    s, e = blk
    return int(w.steps[s]), int(w.steps[e - 1]) + 1, (e - s) / SAMPLES_PER_MIN / 60 / 24


def hdcza_window_grid(w: WindowInput, grid: dict = GRID, min_worn_fraction: float = 0.5,
                      clamp: bool = True) -> dict:
    """All grid combinations for one window, sharing percentile and candidate masks."""
    out = {}
    combos = list(itertools.product(grid["P"], grid["M"], grid["Bmin"], grid["G"]))
    if len(w.steps) == 0 or w.worn.mean() < min_worn_fraction:
        return {c: None for c in combos}
    pct = dict(zip(grid["P"], np.percentile(w.r[w.worn], list(grid["P"]))))
    worn_cum = np.r_[0, np.cumsum(w.worn)]
    for P, M in itertools.product(grid["P"], grid["M"]):
        cand = (w.r < _threshold(pct[P], M, clamp)) & w.worn
        for Bmin, G in itertools.product(grid["Bmin"], grid["G"]):
            blk = blocks_from_candidates(cand, w.worn, Bmin, G, worn_cum)
            out[(P, M, Bmin, G)] = (None if blk is None else
                                    (int(w.steps[blk[0]]), int(w.steps[blk[1] - 1]) + 1,
                                     (blk[1] - blk[0]) / SAMPLES_PER_MIN / 60 / 24))
    return out
