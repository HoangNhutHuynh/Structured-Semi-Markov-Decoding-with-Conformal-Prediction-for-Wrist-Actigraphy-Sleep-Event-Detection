# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Subject-level percentile bootstrap across pooled out-of-fold predictions.

Resampling duplicates whole series. Independent copies are renamed to prevent
cross-copy matching during event AP and to keep nights distinct for ICC.
"""
from __future__ import annotations
from collections.abc import Callable, Mapping
import numpy as np
import pandas as pd

from .clinical import build_night_table, clinical_from_night_table
from .epoch import evaluate_epoch_count_table
from .event_ap import event_detection_ap, precomputed_suite


class OOFEvaluator:
    """Everything resample-invariant is computed once; a bootstrap draw only reweights series.

    Exactness: EDAP matching never crosses series, capping is per window, and every clinical
    quantity is per night, so per-series multiplicities reproduce the earlier
    'rename-and-duplicate' bootstrap exactly (tested in tests/test_bootstrap_fast.py).
    """

    def __init__(self, truth, predictions, windows, predicted_windows, *,
                 step_timestamps=None, epoch_counts=None):
        self.edap = precomputed_suite(truth, predictions)
        self.nights = build_night_table(windows, predicted_windows, step_timestamps)
        self.epoch_counts = epoch_counts
        self.series = sorted({str(x) for x in windows.series_id.unique()})
        for name, pre in self.edap.items():
            extra = set(pre.series) - set(self.series)
            if extra:
                raise ValueError(f"{name}: subjects outside evaluated windows: {sorted(extra)[:5]}")

    def evaluate(self, counts: dict[str, int] | None = None) -> dict[str, float]:
        out = {}
        for name, pre in self.edap.items():
            w = None if counts is None else np.array([counts.get(s, 0) for s in pre.series], float)
            out[name] = pre.score(w)
        out.update(clinical_from_night_table(self.nights, counts))
        if self.epoch_counts is not None:
            ec = self.epoch_counts
            if counts is not None:
                reps = ec.series_id.astype(str).map(counts).fillna(0).astype(int).to_numpy()
                ec = ec.loc[np.repeat(ec.index.to_numpy(), reps)]
            out.update({"epoch_" + k: v for k, v in evaluate_epoch_count_table(ec).items()})
        return out


def evaluate_oof(truth: pd.DataFrame, predictions: pd.DataFrame,
                 windows: pd.DataFrame, predicted_windows: pd.DataFrame,
                 *, step_timestamps: pd.DataFrame | None = None,
                 epoch_counts: pd.DataFrame | None = None) -> dict[str, float]:
    """Primary pooled OOF evaluation; never averages fold-specific AP for primary.

    predictions must have night_date for anti-gaming evaluation. No predictions
    from partial/unlabelled/unannotated windows are dropped for official EDAP.
    """
    return OOFEvaluator(truth, predictions, windows, predicted_windows,
                        step_timestamps=step_timestamps, epoch_counts=epoch_counts).evaluate()


def bootstrap_oof(truth: pd.DataFrame, predictions: pd.DataFrame,
                  windows: pd.DataFrame, predicted_windows: pd.DataFrame,
                  *, step_timestamps: pd.DataFrame | None = None,
                  epoch_counts: pd.DataFrame | None = None,
                  repetitions: int = 1000, seed: int = 0,
                  valid_threshold: int = 950,
                  progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    """95% percentile CI for every scalar statistic (subject-level resampling).

    Return columns metric, estimate, ci_low, ci_high, valid_resamples.
    A resample's undefined scalar metric is dropped *only for that metric*.
    For <valid_threshold valid resamples CI is NaN. Default 1000 draws, seed 0.
    Draws use rng.choice(sorted series ids, n, replace=True) as before.
    """
    if repetitions <= 0 or valid_threshold < 0:
        raise ValueError("invalid repetitions or valid threshold")
    ev = OOFEvaluator(truth, predictions, windows, predicted_windows,
                      step_timestamps=step_timestamps, epoch_counts=epoch_counts)
    ids = ev.series
    if not ids:
        raise ValueError("bootstrap requires one or more evaluated subjects")
    original = ev.evaluate()
    results: dict[str, list[float]] = {k: [] for k in original}
    rng = np.random.default_rng(seed)
    for rep in range(repetitions):
        draw = rng.choice(ids, size=len(ids), replace=True)
        uniq, cnt = np.unique(draw, return_counts=True)
        values = ev.evaluate(dict(zip(map(str, uniq), map(int, cnt))))
        for metric in results:
            results[metric].append(float(values[metric]))
        if progress:
            progress(rep + 1, repetitions)
    rows = []
    for metric, values in results.items():
        sample = np.asarray(values, dtype=float)
        valid = sample[np.isfinite(sample)]
        lo, hi = (np.percentile(valid, [2.5, 97.5]) if len(valid) >= valid_threshold
                  else [float("nan")] * 2)
        rows.append({"metric": metric, "estimate": original[metric],
                     "ci_low": float(lo), "ci_high": float(hi),
                     "valid_resamples": int(len(valid))})
    return pd.DataFrame(rows)


def per_fold_official_edap(truth: pd.DataFrame, predictions: pd.DataFrame,
                           folds: Mapping[str, Mapping[str, list[str]]]) -> pd.DataFrame:
    """Uses held-out series only, never train/val rows."""
    rows=[]
    for fold,splits in sorted(folds.items(),key=lambda pair:int(pair[0])):
        subjects = set(splits['test'])
        t=truth.loc[truth.series_id.isin(subjects)]
        p=predictions.loc[predictions.series_id.isin(subjects)]
        rows.append({"fold":int(fold),"n_subjects":len(subjects),
                     "official_edap":event_detection_ap(t,p)})
    return pd.DataFrame(rows)


def per_fold_edap_summary(per_fold: pd.DataFrame) -> dict[str, float]:
    """Secondary summary; uses sample SD (ddof=1) across the five test folds."""
    if not {"fold","official_edap"}.issubset(per_fold.columns):
        raise ValueError("per_fold table must contain fold and official_edap")
    values=per_fold.official_edap.to_numpy(dtype=float)
    return {"n_folds":len(values),
            "official_edap_mean":float(np.mean(values)) if len(values) else float('nan'),
            "official_edap_sd":float(np.std(values,ddof=1)) if len(values)>1 else float('nan')}
