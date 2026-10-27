# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""B1 feature cache and window tensors.

Per series (cached in outputs/b1_cache/<sid>.npz, float16 features):
  raw channels: anglez/45, log1p(enmo), |d anglez|, |d enmo|, local hour (0..24, recorded wall
  clock), nonwear flag (Step A).
Windows: 15:00 -> 15:00 recorded wall clock (same rule as Step A). Each window tensor starts at its
first whole-minute sample (phase = 0..11 samples dropped), so 1-min bin i starts exactly on a minute
boundary; labels (always whole minutes) fall on bin starts and predicted step = steps[start + 12*i].
Model input (7 ch): anglez/45, z(log1p enmo), z(|d anglez|), z(|d enmo|), sin/cos(2*pi*hour/24),
nonwear. z-scores use statistics of the fold's training series only.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

L_MAX = 18000          # 25 h at 5 s
SPM = 12               # samples per minute
M_BINS = L_MAX // SPM  # 1500
BOUNDARY_HOUR = 15


@dataclass
class SeriesData:
    sid: str
    steps: np.ndarray      # int64 (N,)
    feats: np.ndarray      # float16 (N, 6) raw channels
    win_dates: np.ndarray  # str (W,)
    win_start: np.ndarray  # int64, index of first whole-minute sample in window
    win_end: np.ndarray    # int64, exclusive


def build_series(cache: Path, sid: str) -> dict:
    tab = pq.read_table(cache / f"{sid}.parquet",
                        columns=["step", "timestamp", "anglez", "enmo"]).to_pandas()
    tab = tab.sort_values("step", kind="stable").reset_index(drop=True)
    nonwear = np.load(cache / f"{sid}_nonwear.npy").astype(bool)
    if len(nonwear) != len(tab):
        raise ValueError(f"{sid}: nonwear length mismatch")
    ts = tab.timestamp.astype(str)
    local = pd.to_datetime(ts.str[:19], format="%Y-%m-%dT%H:%M:%S")
    sec = local.dt.second.to_numpy()
    hour = (local.dt.hour + local.dt.minute / 60 + sec / 3600).to_numpy()
    nd = (local - pd.Timedelta(hours=BOUNDARY_HOUR)).dt.strftime("%Y-%m-%d").to_numpy().astype(str)
    a = tab.anglez.to_numpy(np.float32)
    e = np.nan_to_num(tab.enmo.to_numpy(np.float32), nan=0.0).clip(min=0)
    a = np.nan_to_num(a, nan=0.0)
    da = np.abs(np.diff(a, prepend=a[:1])); de = np.abs(np.diff(e, prepend=e[:1]))
    feats = np.column_stack([a / 45.0, np.log1p(e), da, de, hour, nonwear]).astype(np.float16)
    change = np.r_[0, np.flatnonzero(nd[1:] != nd[:-1]) + 1, len(nd)]
    dates, starts, ends = [], [], []
    for s, t in zip(change[:-1], change[1:]):
        zero = np.flatnonzero(sec[s:t] == 0)
        first = s + int(zero[0]) if len(zero) else s
        dates.append(nd[s]); starts.append(first); ends.append(min(t, first + L_MAX))
    return {"steps": tab.step.to_numpy(np.int64), "feats": feats, "win_dates": np.array(dates),
            "win_start": np.array(starts, np.int64), "win_end": np.array(ends, np.int64)}


def load_series(cache: Path, b1_cache: Path, sid: str) -> SeriesData:
    f = b1_cache / f"{sid}.npz"
    if not f.exists():
        np.savez(f, **build_series(cache, sid))
    z = np.load(f, allow_pickle=False)
    return SeriesData(sid, z["steps"], z["feats"], z["win_dates"].astype(str),
                      z["win_start"], z["win_end"])


def norm_stats(series: list[SeriesData], max_per_series: int = 50_000, seed: int = 0) -> np.ndarray:
    """mean/std of channels 1..3 over worn samples of the given (training) series."""
    rng = np.random.default_rng(seed)
    chunks = []
    for s in series:
        worn = s.feats[:, 5] < 0.5
        idx = np.flatnonzero(worn)
        if len(idx) > max_per_series:
            idx = rng.choice(idx, max_per_series, replace=False)
        chunks.append(s.feats[idx][:, 1:4].astype(np.float32))
    x = np.concatenate(chunks)
    return np.stack([x.mean(0), x.std(0) + 1e-6])


def window_tensor(s: SeriesData, w: int, stats: np.ndarray):
    """(7, L_MAX) float32 input, (M_BINS,) bool bin mask, start index, n valid samples."""
    a, b = int(s.win_start[w]), int(s.win_end[w])
    raw = s.feats[a:b].astype(np.float32)
    n = len(raw)
    x = np.zeros((7, L_MAX), np.float32)
    x[0, :n] = raw[:, 0]
    x[1:4, :n] = ((raw[:, 1:4] - stats[0]) / stats[1]).T
    x[4, :n] = np.sin(2 * np.pi * raw[:, 4] / 24)
    x[5, :n] = np.cos(2 * np.pi * raw[:, 4] / 24)
    x[6, :n] = raw[:, 5]
    mask = np.zeros(M_BINS, bool)
    mask[: (n + SPM - 1) // SPM] = True
    return x, mask, a, n


def targets(s: SeriesData, w: int, onset_step, wakeup_step, sigma_bins: float = 2.0):
    """(3, M_BINS) float32 targets; None if an event falls outside the window tensor."""
    a = int(s.win_start[w]); n = int(s.win_end[w]) - a
    i_on = int(np.searchsorted(s.steps, onset_step)) - a
    i_wk = int(np.searchsorted(s.steps, wakeup_step)) - a
    if not (0 <= i_on < n and 0 <= i_wk <= n):
        return None
    b_on, b_wk = i_on / SPM, i_wk / SPM
    grid = np.arange(M_BINS, dtype=np.float32)
    y = np.zeros((3, M_BINS), np.float32)
    y[0] = np.exp(-0.5 * ((grid - b_on) / sigma_bins) ** 2)
    y[1] = np.exp(-0.5 * ((grid - b_wk) / sigma_bins) ** 2)
    y[2, int(np.floor(b_on)): int(np.ceil(b_wk))] = 1.0
    return y


def bin_to_step(s: SeriesData, w: int, bins: np.ndarray) -> np.ndarray:
    idx = int(s.win_start[w]) + np.asarray(bins, int) * SPM
    idx = np.clip(idx, 0, len(s.steps) - 1)
    return s.steps[idx]
