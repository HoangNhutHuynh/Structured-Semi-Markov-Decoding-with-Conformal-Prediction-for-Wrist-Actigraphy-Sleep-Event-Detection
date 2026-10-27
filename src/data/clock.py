# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Recorded wall-clock minutes for arbitrary steps, without a 128M-row timestamp table.

Step A verified that UTC (parsed from the recorded offset) advances exactly 5 s per step.
Hence wall_clock(step) = utc0 + 5*(step-step0) + offset(step), where offset changes only at a
few steps per series. Output: minutes from 12:00 of `night_date` on the recorded wall clock,
identical to clinical.wall_minutes_from_noon(timestamp_text, night_date).
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SAMPLE_SECONDS = 5
_EPOCH = datetime(1970, 1, 1)


def _offset_minutes(text: pd.Series) -> np.ndarray:
    o = text.str[-5:]
    sign = np.where(o.str[0] == "-", -1, 1)
    return sign * (o.str[1:3].astype(int).to_numpy() * 60 + o.str[3:5].astype(int).to_numpy())


def series_clock_entry(steps: np.ndarray, timestamps: pd.Series) -> dict:
    order = np.argsort(steps, kind="stable")
    steps = np.asarray(steps)[order]
    ts = timestamps.astype(str).iloc[order].reset_index(drop=True)
    off = _offset_minutes(ts)
    change = np.r_[0, np.flatnonzero(np.diff(off)) + 1]
    first_wall = datetime.strptime(ts.iloc[0][:19], "%Y-%m-%dT%H:%M:%S")
    wall0 = (first_wall - _EPOCH).total_seconds()
    utc0 = wall0 - off[0] * 60
    return {"step0": int(steps[0]), "utc0": float(utc0),
            "change_steps": [int(steps[i]) for i in change],
            "change_offsets_min": [int(off[i]) for i in change]}


def build_clock_index(cache_dir: str | Path, out_file: str | Path | None = None) -> dict:
    """Read each Step A series cache once (step, timestamp only) and store change points."""
    cache = Path(cache_dir)
    index = {}
    for f in sorted(cache.glob("*.parquet")):
        tab = pq.read_table(f, columns=["step", "timestamp"]).to_pandas()
        index[f.stem] = series_clock_entry(tab.step.to_numpy(np.int64), tab.timestamp)
    if out_file is not None:
        Path(out_file).write_text(json.dumps(index), encoding="utf-8")
    return index


def load_clock_index(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clock_minutes_from_noon(entry: dict, steps, night_dates) -> np.ndarray:
    steps = np.asarray(steps, dtype=float)
    out = np.full(steps.shape, np.nan)
    ok = np.isfinite(steps)
    if not ok.any():
        return out
    s = steps[ok]
    cs = np.asarray(entry["change_steps"], dtype=float)
    co = np.asarray(entry["change_offsets_min"], dtype=float)
    k = np.clip(np.searchsorted(cs, s, side="right") - 1, 0, len(cs) - 1)
    wall = entry["utc0"] + SAMPLE_SECONDS * (s - entry["step0"]) + co[k] * 60
    nd = pd.to_datetime(pd.Series(np.asarray(night_dates, dtype=object)[ok]).astype(str).str[:10])
    noon = ((nd - pd.Timestamp("1970-01-01")).dt.total_seconds() + 12 * 3600).to_numpy()
    out[ok] = (wall - noon) / 60.0
    return out


def add_clock_columns(df: pd.DataFrame, index: dict,
                      mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """Add *_clock_min columns (default from onset_step / wakeup_step) for clinical BA."""
    mapping = mapping or {"onset_step": "onset_clock_min", "wakeup_step": "wakeup_clock_min"}
    out = df.copy()
    for src, dst in mapping.items():
        out[dst] = np.nan
    for sid, idx in out.groupby(out.series_id.astype(str)).indices.items():
        if sid not in index:
            raise KeyError(f"series {sid} missing from clock index")
        sub = out.iloc[idx]
        for src, dst in mapping.items():
            out.iloc[idx, out.columns.get_loc(dst)] = clock_minutes_from_noon(
                index[sid], sub[src].to_numpy(dtype=float), sub.night_date.to_numpy())
    return out
