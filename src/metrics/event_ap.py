# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Clean-room event detection average precision for CMI sleep states.

Evaluation is global across series within each event type and time tolerance.
No score normalization, snapping, scoring interval, or unlabelled-night filtering.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

EVENTS = ("onset", "wakeup")
TOLERANCES = (12, 36, 60, 90, 120, 150, 180, 240, 300, 360)


def _validate(truth: pd.DataFrame, predictions: pd.DataFrame) -> None:
    required = {"series_id", "step", "event"}
    for name, df, fields in (
        ("truth", truth, required),
        ("predictions", predictions, required | {"score"}),
    ):
        absent = fields.difference(df.columns)
        if absent:
            raise ValueError(f"{name} is missing columns: {sorted(absent)}")
        if df[list(required)].isna().any().any():
            raise ValueError(f"{name}: null series_id, step, or event")
        if not df.event.isin(EVENTS).all():
            raise ValueError(f"{name}: only onset and wakeup events are supported")
        if not np.isfinite(pd.to_numeric(df.step, errors="coerce").to_numpy()).all():
            raise ValueError(f"{name}: nonfinite or nonnumeric step")
    if not np.isfinite(pd.to_numeric(predictions.score, errors="coerce").to_numpy()).all():
        raise ValueError("predictions: scores must be finite")


def _match_group(gt_steps: np.ndarray, det_steps: np.ndarray, det_scores: np.ndarray,
                 tolerances: Sequence[int]) -> np.ndarray:
    """Matched flags (n_det x n_tol) for ONE (series, event) group, in input order.

    Semantics identical to the Kaggle reference: detections in descending score
    order, each takes the nearest still-unmatched GT if distance < tolerance
    (strict). Ties in score keep input order (stable); equidistant GTs -> lowest step.
    """
    n_det = len(det_steps)
    out = np.zeros((n_det, len(tolerances)), dtype=np.bool_)
    if n_det == 0 or len(gt_steps) == 0:
        return out
    # Tie order must replicate the reference exactly: it calls DataFrame.sort_values(score,
    # ascending=False) (pandas default, NOT stable) on detections already sorted by
    # (series, time). Using the identical pandas call reproduces its tie order.
    order = pd.Series(det_scores).sort_values(ascending=False).index.to_numpy()
    err = np.abs(det_steps[order][:, None] - gt_steps[None, :])
    row_min = err.min(axis=1)
    for j, tol in enumerate(tolerances):
        used = np.zeros(len(gt_steps), dtype=np.bool_)
        for r in np.flatnonzero(row_min < tol):
            e = np.where(used, np.inf, err[r])
            k = int(np.argmin(e))
            if e[k] < tol:
                used[k] = True
                out[order[r], j] = True
    return out


class PrecomputedEDAP:
    """Event AP with matches computed ONCE; supports exact subject-bootstrap reweighting.

    Matching never crosses series, so per-prediction matched flags are invariant to
    resampling series. A bootstrap draw is a vector of per-series multiplicities;
    duplicated series form identical tied-score blocks, exactly as renamed copies would.
    """

    def __init__(self, truth: pd.DataFrame, predictions: pd.DataFrame,
                 tolerances: Sequence[int] = TOLERANCES):
        _validate(truth, predictions)
        if not tolerances or any(int(x) != x or x <= 0 for x in tolerances):
            raise ValueError("tolerances must be positive integers")
        self.tolerances = tuple(int(x) for x in tolerances)
        t = truth.assign(series_id=truth.series_id.astype(str))
        p = predictions.assign(series_id=predictions.series_id.astype(str))
        # Kaggle reference: aggregation keys are built ONLY from series that have at least one
        # ground-truth event; detections from series without any GT are silently ignored.
        # (Never triggered on the Kaggle test set; matters for our 8 all-unlabelled series.)
        self.n_dropped_no_gt_series = int((~p.series_id.isin(set(t.series_id))).sum())
        p = p[p.series_id.isin(set(t.series_id))]
        p = p.sort_values(["series_id", "step"], kind="stable").reset_index(drop=True)
        self.series = sorted(set(t.series_id))
        sidx = {sid: i for i, sid in enumerate(self.series)}
        self.events: dict[str, dict] = {}
        for event in EVENTS:
            tc = t.loc[t.event == event]
            if tc.empty:
                continue  # only classes present in truth contribute (reference behaviour)
            gt_count = np.zeros(len(self.series), dtype=np.int64)
            gts = {sid: g.step.to_numpy(dtype=float) for sid, g in tc.groupby("series_id")}
            for sid, arr in gts.items():
                gt_count[sidx[sid]] = len(arr)
                gts[sid] = np.sort(arr)
            pc = p.loc[p.event == event].reset_index(drop=True)
            steps = pc.step.to_numpy(dtype=float)
            scores = pc.score.to_numpy(dtype=float)
            matched = np.zeros((len(pc), len(self.tolerances)), dtype=np.bool_)
            for sid, idx in pc.groupby("series_id", sort=False).indices.items():
                g = gts.get(sid)
                if g is not None:
                    matched[idx] = _match_group(g, steps[idx], scores[idx], self.tolerances)
            order = np.argsort(-scores, kind="stable")
            s_sorted = scores[order]
            ends = (np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
                    if len(s_sorted) else np.zeros(0, dtype=np.int64))
            self.events[event] = {
                "gt_count": gt_count,
                "series_sorted": np.array([sidx[x] for x in pc.series_id.to_numpy()[order]],
                                          dtype=np.int64),
                "matched_sorted": matched[order],
                "ends": ends,
                "n_predictions": len(pc),
            }

    def components(self, weights: np.ndarray | None = None) -> pd.DataFrame:
        w_series = (np.ones(len(self.series)) if weights is None
                    else np.asarray(weights, dtype=float))
        rows = []
        for event, d in self.events.items():
            positives = float(np.dot(w_series, d["gt_count"]))
            for j, tol in enumerate(self.tolerances):
                if positives <= 0:
                    ap = float("nan")
                elif d["n_predictions"] == 0:
                    ap = 0.0
                else:
                    w = w_series[d["series_sorted"]]
                    cum_w = np.cumsum(w)[d["ends"]]
                    tp = np.cumsum(w * d["matched_sorted"][:, j])[d["ends"]]
                    precision = np.divide(tp, cum_w, out=np.zeros_like(tp), where=cum_w > 0)
                    recall = tp / positives
                    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
                rows.append({"event": event, "tolerance": tol, "ap": ap,
                             "n_truth": positives, "n_predictions": d["n_predictions"],
                             "n_matches": int(d["matched_sorted"][:, j].sum())})
        return pd.DataFrame(rows, columns=["event", "tolerance", "ap", "n_truth",
                                           "n_predictions", "n_matches"])

    def score(self, weights: np.ndarray | None = None) -> float:
        comp = self.components(weights)
        return float(comp.groupby("event")["ap"].mean().mean()) if len(comp) else float("nan")


def event_detection_ap(truth: pd.DataFrame, predictions: pd.DataFrame,
                       tolerances: Sequence[int] = TOLERANCES,
                       *, return_components: bool = False) -> float | tuple[float, pd.DataFrame]:
    """CMI EDAP pooled across series per event class (see PrecomputedEDAP)."""
    pre = PrecomputedEDAP(truth, predictions, tolerances)
    comp = pre.components()
    result = float(comp.groupby("event")["ap"].mean().mean()) if len(comp) else float("nan")
    return (result, comp) if return_components else result


def cap_candidates(predictions: pd.DataFrame, k: int, *, window_col: str = "night_date") -> pd.DataFrame:
    """Anti-gaming protocol: retain top k candidates per event type, per window.

    Unlike the official score this requires the observed window identity, which
    must originate from the approved 15:00 local-wall-clock Step A windows.
    No event time or event score is modified.
    """
    if k < 1 or int(k) != k:
        raise ValueError("k must be a positive integer")
    missing = {"series_id", "event", "score", window_col}.difference(predictions.columns)
    if missing:
        raise ValueError(f"capping requires {sorted(missing)}")
    if predictions[window_col].isna().any():
        raise ValueError("every prediction must be assigned to an observed window")
    # The extra step key makes tied-score selection deterministic.
    return (predictions.sort_values(["series_id", window_col, "event", "score", "step"],
                                    ascending=[True, True, True, False, True], kind="stable")
            .groupby(["series_id", window_col, "event"], sort=False, dropna=False)
            .head(int(k)).sort_index().copy())


def anti_gaming_edap(truth: pd.DataFrame, predictions: pd.DataFrame, *, k: int,
                      window_col: str = "night_date") -> float:
    return event_detection_ap(truth, cap_candidates(predictions, k, window_col=window_col))


def precomputed_suite(truth: pd.DataFrame, predictions: pd.DataFrame,
                      window_col: str = "night_date") -> dict[str, PrecomputedEDAP]:
    """Official, K=1 and K=3 anti-gaming EDAP, each matched once (capping is per window)."""
    suite = {"official_edap": PrecomputedEDAP(truth, predictions)}
    if window_col in predictions.columns:
        for k in (1, 3):
            suite[f"anti_edap_k{k}"] = PrecomputedEDAP(
                truth, cap_candidates(predictions, k, window_col=window_col))
    return suite


def truth_from_event_nights(event_nights: pd.DataFrame) -> pd.DataFrame:
    """All non-null event steps, including cross_boundary events.

    Deliberately does NOT filter by label_status. Partial events contribute
    when present in the event file, consistent with event-based scoring.
    """
    require = {"series_id", "onset_step", "wakeup_step"}
    if not require.issubset(event_nights.columns):
        raise ValueError(f"missing event-night columns {sorted(require-event_nights.columns)}")
    rows = []
    for event, col in (("onset", "onset_step"), ("wakeup", "wakeup_step")):
        part = event_nights.loc[event_nights[col].notna(), ["series_id", col]].copy()
        part.columns = ["series_id", "step"]
        part["event"] = event
        rows.append(part)
    return pd.concat(rows, ignore_index=True)
