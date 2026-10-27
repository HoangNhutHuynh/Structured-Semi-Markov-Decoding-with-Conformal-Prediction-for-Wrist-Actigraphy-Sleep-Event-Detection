# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Epoch (5-second sample) classification metrics for B2 reproduction."""
from __future__ import annotations
import numpy as np
import pandas as pd

COUNT_COLUMNS = ("tp", "tn", "fp", "fn")


def confusion_counts(truth: np.ndarray, predicted: np.ndarray,
                     valid: np.ndarray) -> dict[str, int]:
    """Count valid, non-padding binary sleep/wake samples (sleep=1)."""
    a, b, mask = np.asarray(truth), np.asarray(predicted), np.asarray(valid, dtype=bool)
    if a.shape != b.shape or a.shape != mask.shape:
        raise ValueError("truth, predicted and valid must have identical shape")
    if not (np.isin(a[mask], [0, 1]).all() and np.isin(b[mask], [0, 1]).all()):
        raise ValueError("valid predictions/truth must be binary 0/1")
    a, b = a[mask], b[mask]
    return {"tp": int(((a == 1) & (b == 1)).sum()),
            "tn": int(((a == 0) & (b == 0)).sum()),
            "fp": int(((a == 0) & (b == 1)).sum()),
            "fn": int(((a == 1) & (b == 0)).sum())}


def scores_from_counts(counts: dict[str, int] | pd.Series) -> dict[str, float]:
    tp,tn,fp,fn = (int(counts[k]) for k in COUNT_COLUMNS)
    def ratio(x: int, y: int) -> float:
        return float(x / y) if y else float("nan")
    return {"n_valid": tp+tn+fp+fn,
            "accuracy": ratio(tp+tn, tp+tn+fp+fn),
            "sensitivity": ratio(tp, tp+fn),
            "specificity": ratio(tn, tn+fp),
            "precision": ratio(tp, tp+fp),
            "f1": ratio(2*tp, 2*tp+fp+fn)}


def evaluate_epoch_samples(truth: np.ndarray, predicted: np.ndarray,
                           valid: np.ndarray, labelled: np.ndarray) -> dict[str, float]:
    """B2 overall valid-epoch and labelled-night-only metrics."""
    if np.asarray(labelled).shape != np.asarray(valid).shape:
        raise ValueError("labelled mask must have same shape as valid")
    all_count = confusion_counts(truth, predicted, valid)
    labelled_count = confusion_counts(truth, predicted, valid & np.asarray(labelled, dtype=bool))
    return {**{f"all_{k}":v for k,v in scores_from_counts(all_count).items()},
            **{f"labelled_{k}":v for k,v in scores_from_counts(labelled_count).items()}}


def evaluate_epoch_count_table(counts: pd.DataFrame) -> dict[str, float]:
    """Input: per-series sufficient statistics all_tp,.. and labelled_tp,... ."""
    keys = [f"{subset}_{k}" for subset in ("all", "labelled") for k in COUNT_COLUMNS]
    missing = set(keys) - set(counts.columns)
    if missing:
        raise ValueError(f"missing count columns: {sorted(missing)}")
    out = {}
    for subset in ("all", "labelled"):
        sums = counts[[f"{subset}_{k}" for k in COUNT_COLUMNS]].sum(axis=0)
        val = scores_from_counts({k:int(sums[f"{subset}_{k}"]) for k in COUNT_COLUMNS})
        out.update({f"{subset}_{key}":v for key, v in val.items()})
    return out
