# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Night-level clinical comparisons on fully labelled 15:00-to-15:00 windows.

All minute errors and SPT use actual five-second step distances. Clock-time
Bland--Altman analyses use recorded local wall-clock text, never TZ conversion.
"""
from __future__ import annotations

from datetime import datetime, time
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats

SAMPLE_SECONDS = 5  # fixed by the CMI data protocol


def wall_minutes_from_noon(timestamp: str, night_date: str) -> float:
    """Minutes from 12:00 of starting local calendar date, preserving wall clock.

    The numeric UTC offset is intentionally ignored after extracting the displayed
    civil date and time. DST rules are not inferred from a timezone database.
    """
    if pd.isna(timestamp) or pd.isna(night_date):
        return float("nan")
    displayed = str(timestamp).strip().replace("T", " ")[:19]
    local = datetime.strptime(displayed, "%Y-%m-%d %H:%M:%S")
    noon = datetime.combine(datetime.fromisoformat(str(night_date)[:10]).date(), time(12))
    return (local - noon).total_seconds() / 60.0


def _clock_columns(pairs: pd.DataFrame, step_timestamps: pd.DataFrame | None) -> pd.DataFrame:
    if step_timestamps is None:
        raise ValueError("clock BA requires step_timestamps: series_id, step, timestamp from raw recorded text")
    cols = {"series_id", "step", "timestamp"}
    if not cols.issubset(step_timestamps.columns):
        raise ValueError(f"step_timestamps missing {sorted(cols-step_timestamps.columns)}")
    lut = step_timestamps.loc[:, ["series_id", "step", "timestamp"]]
    if lut.duplicated(["series_id", "step"]).any():
        raise ValueError("duplicate (series_id, step) timestamp lookups")
    for side in ("true", "pred"):
        for event in ("onset", "wakeup"):
            key = f"{event}_{side}_step"
            joined = lut.rename(columns={"step":key, "timestamp":f"{event}_{side}_timestamp"})
            pairs = pairs.merge(joined, how="left", on=["series_id",key], validate="many_to_one")
            pairs[f"{event}_{side}_clock_min"] = [
                wall_minutes_from_noon(ts, date)
                for ts,date in zip(pairs[f"{event}_{side}_timestamp"],pairs.night_date)
            ]
    return pairs


def _ba(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Signed difference = estimate - reference; proportional bias vs mean."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    valid = np.isfinite(reference) & np.isfinite(estimate)
    r = reference[valid]
    e = estimate[valid]
    result = {"n": int(len(r)), "bias": float("nan"), "loa_low": float("nan"),
              "loa_high": float("nan"), "slope": float("nan"), "slope_p": float("nan")}
    if not len(r):
        return result
    difference = e - r
    result["bias"] = float(np.mean(difference))
    if len(r) >= 2:
        spread = float(np.std(difference, ddof=1))
        result["loa_low"] = result["bias"] - 1.96 * spread
        result["loa_high"] = result["bias"] + 1.96 * spread
        midpoint = (e + r) / 2
        if len(r) >= 3 and float(np.ptp(midpoint)) > 0:
            regression = stats.linregress(midpoint, difference)
            result["slope"] = float(regression.slope)
            result["slope_p"] = float(regression.pvalue)
    return result


def icc2_1(truth: np.ndarray, pred: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measure
    (Shrout & Fleiss 1979; identical to pingouin 'ICC2'). Closed form, no pingouin."""
    x = np.column_stack([np.asarray(truth, float), np.asarray(pred, float)])
    x = x[np.isfinite(x).all(axis=1)]
    n, k = x.shape
    if n < 2:
        return float("nan")
    gm = x.mean()
    ss_rows = k * np.sum((x.mean(axis=1) - gm) ** 2)
    ss_cols = n * np.sum((x.mean(axis=0) - gm) ** 2)
    ss_err = np.sum((x - gm) ** 2) - ss_rows - ss_cols
    ms_r = ss_rows / (n - 1)
    ms_c = ss_cols / (k - 1)
    ms_e = ss_err / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom != 0 else float("nan")


def _icc2_spread(pairs: pd.DataFrame) -> float:
    return icc2_1(pairs.spt_true_min.to_numpy(), pairs.spt_pred_min.to_numpy())


def _min_abs_metrics(errors: np.ndarray, prefix: str) -> dict[str, float]:
    if not len(errors):
        return {f"{prefix}_{key}": float("nan") for key in
                ("medae_min", "q1_min", "q3_min", "within15_pct", "within30_pct")}
    a = np.abs(errors)
    return {f"{prefix}_medae_min": float(np.median(a)),
            f"{prefix}_q1_min": float(np.percentile(a, 25)),
            f"{prefix}_q3_min": float(np.percentile(a, 75)),
            f"{prefix}_within15_pct": float(100 * np.mean(a <= 15)),
            f"{prefix}_within30_pct": float(100 * np.mean(a <= 30))}


CLOCK_COLUMNS = ("onset_clock_min", "wakeup_clock_min")


def build_night_table(windows: pd.DataFrame, predicted_windows: pd.DataFrame,
                      step_timestamps: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per observed window with every per-night quantity precomputed.

    windows: series_id, night_date, label_status, onset_step, wakeup_step,
             nonwear_fraction (0..1) [+ optional truth onset_clock_min, wakeup_clock_min].
    predicted_windows: series_id, night_date, onset_step, wakeup_step for predicted windows
             only [+ optional onset_clock_min, wakeup_clock_min].
    Clock minutes come from precomputed columns (src/data/clock.py, preferred) or, for
    small cases/tests only, from a step_timestamps lookup table.
    """
    key = ["series_id", "night_date"]
    must_have = {*key, "label_status", "onset_step", "wakeup_step", "nonwear_fraction"}
    if not must_have.issubset(windows.columns):
        raise ValueError(f"windows missing {sorted(must_have-set(windows.columns))}")
    if not {*key, "onset_step", "wakeup_step"}.issubset(predicted_windows.columns):
        raise ValueError("predicted_windows needs series_id, night_date, onset_step, wakeup_step")
    if windows.duplicated(key).any() or predicted_windows.duplicated(key).any():
        raise ValueError("one row per observed window and at most one predicted pair per window")
    if len(predicted_windows) and not predicted_windows.set_index(key).index.isin(
            windows.set_index(key).index).all():
        raise ValueError("predicted window not found in observed night windows")
    pcols = key + ["onset_step", "wakeup_step"] + [c for c in CLOCK_COLUMNS if c in predicted_windows]
    p = predicted_windows[pcols].rename(columns={
        "onset_step": "onset_pred_step", "wakeup_step": "wakeup_pred_step",
        "onset_clock_min": "onset_pred_clock_min", "wakeup_clock_min": "wakeup_pred_clock_min"})
    w = windows.rename(columns={"onset_clock_min": "onset_true_clock_min",
                                "wakeup_clock_min": "wakeup_true_clock_min"})
    j = w.merge(p, on=key, how="left", validate="one_to_one")
    if (j.onset_pred_step.notna() ^ j.wakeup_pred_step.notna()).any():
        raise ValueError("predicted windows must contain both event steps")
    j["has_window"] = j.onset_pred_step.notna() & j.wakeup_pred_step.notna()
    j["is_labelled"] = j.label_status == "labelled"
    if j.loc[j.is_labelled, ["onset_step", "wakeup_step"]].isna().any().any():
        raise ValueError("labelled windows require both truth events")
    j["is_proxy"] = j.label_status.isin(["unlabelled", "unannotated"]) & (j.nonwear_fraction >= 0.9)
    f = SAMPLE_SECONDS / 60
    j["onset_signed_error_min"] = (j.onset_pred_step - j.onset_step) * f
    j["wakeup_signed_error_min"] = (j.wakeup_pred_step - j.wakeup_step) * f
    j["spt_true_min"] = (j.wakeup_step - j.onset_step) * f
    j["spt_pred_min"] = (j.wakeup_pred_step - j.onset_pred_step) * f
    scored = j.is_labelled & j.has_window
    need_clock = [f"{e}_{s}_clock_min" for e in ("onset", "wakeup") for s in ("true", "pred")]
    if scored.any() and not set(need_clock).issubset(j.columns):
        if step_timestamps is None:
            raise ValueError("clock BA needs precomputed *_clock_min columns (src/data/clock.py) "
                             "or a step_timestamps table")
        pairs = j.loc[scored, key + ["onset_step", "wakeup_step", "onset_pred_step",
                                     "wakeup_pred_step"]].rename(columns={
            "onset_step": "onset_true_step", "wakeup_step": "wakeup_true_step"})
        pairs = _clock_columns(pairs, step_timestamps)
        for c in need_clock:
            j[c] = np.nan
            j.loc[scored, c] = pairs[c].to_numpy()
    for c in need_clock:
        if c not in j:
            j[c] = np.nan
    return j


def clinical_from_night_table(j: pd.DataFrame, weights: Mapping[str, int] | None = None,
                              *, include_icc: bool = True) -> dict[str, float]:
    """Clinical summary; weights = per-series bootstrap multiplicities (None = 1 each)."""
    if weights is not None:
        reps = j.series_id.astype(str).map(weights).fillna(0).astype(int).to_numpy()
        j = j.loc[np.repeat(j.index.to_numpy(), reps)].reset_index(drop=True)
    has_window = j.has_window
    labels = j.is_labelled

    def rate(mask: pd.Series) -> float:
        return float(np.mean(has_window.loc[mask])) if int(mask.sum()) else float("nan")

    result = {"n_labelled": int(labels.sum()),
              "n_labelled_scored": int((labels & has_window).sum()),
              "window_recall_labelled": rate(labels)}
    for status in ("unlabelled", "unannotated"):
        m = j.label_status == status
        proxy = m & j.is_proxy
        result[f"n_{status}"] = int(m.sum())
        result[f"predicted_rate_{status}"] = rate(m)
        result[f"n_proxy_negative_{status}"] = int(proxy.sum())
        result[f"proxy_false_detection_rate_{status}"] = rate(proxy)
    pairs = j.loc[labels & has_window]
    for event in ("onset", "wakeup"):
        result.update(_min_abs_metrics(pairs[f"{event}_signed_error_min"].to_numpy(dtype=float), event))
    ba = _ba(pairs.spt_true_min.to_numpy(dtype=float), pairs.spt_pred_min.to_numpy(dtype=float))
    result.update({f"ba_spt_{k}": v for k, v in ba.items()})
    result["icc2_1_spt"] = (icc2_1(pairs.spt_true_min.to_numpy(), pairs.spt_pred_min.to_numpy())
                            if include_icc else float("nan"))
    for event in ("onset", "wakeup"):
        ba = _ba(pairs[f"{event}_true_clock_min"].to_numpy(dtype=float),
                 pairs[f"{event}_pred_clock_min"].to_numpy(dtype=float))
        result.update({f"ba_{event}_clock_{k}": v for k, v in ba.items()})
    return result


def evaluate_clinical(windows: pd.DataFrame, predicted_windows: pd.DataFrame,
                      step_timestamps: pd.DataFrame | None = None,
                      *, include_icc: bool = True) -> dict[str, float]:
    """Evaluate one optional onset/wake pair per window (see build_night_table).

    Cross-boundary, partial, unlabelled, unannotated windows are never used for
    clinical timing accuracy or ICC. Proxy detection rates are descriptive only.
    """
    table = build_night_table(windows, predicted_windows, step_timestamps)
    return clinical_from_night_table(table, include_icc=include_icc)
