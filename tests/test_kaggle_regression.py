# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Optional official reference parity, including future B1 pooled OOF prediction file."""
from __future__ import annotations
import os
from pathlib import Path
import warnings
import pandas as pd
import pytest

from src.metrics.event_ap import event_detection_ap, truth_from_event_nights
from src.metrics.kaggle_reference import REFERENCE, official_score


def _reference_or_skip():
    if not REFERENCE.exists():
        message = ("OFFICIAL KAGGLE REGRESSION NOT RUN: "
                   "third_party/kaggle_metric/event_detection_ap.py is absent. "
                   "Clean-room parity is NOT VERIFIED.")
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        pytest.skip(message)


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower()=='.parquet' else pd.read_csv(path)


def test_reference_matches_hand_constructed_cases():
    _reference_or_skip()
    t=pd.DataFrame([('A',100,'onset'),('A',400,'wakeup'),
                    ('B',100,'onset'),('B',400,'wakeup')],
                   columns=['series_id','step','event'])
    rows=[
        [('A',100,'onset',.9),('A',400,'wakeup',.8),
         ('B',100,'onset',.7),('B',400,'wakeup',.6)],
        [('A',100,'onset',.9),('A',401,'wakeup',.8),
         ('B',110,'onset',.7),('B',430,'wakeup',.6)],
        [('A',100,'onset',.9),('A',990,'onset',.7),('A',400,'wakeup',.6),
         ('B',100,'onset',.8),('B',400,'wakeup',.65)],
        [('A',900,'onset',.99),('A',100,'onset',.9),('A',400,'wakeup',.8),
         ('B',100,'onset',.7),('B',400,'wakeup',.6)],
        [('A',112,'onset',.9),('A',400,'wakeup',.8),
         ('B',100,'onset',.7),('B',400,'wakeup',.6)],
        # series C has NO ground truth: the reference ignores its detections entirely
        [('A',100,'onset',.5),('A',400,'wakeup',.5),('C',100,'onset',.99),('C',400,'wakeup',.99),
         ('B',100,'onset',.7),('B',400,'wakeup',.6)],
    ]
    for case,prediction_rows in enumerate(rows):
        p=pd.DataFrame(prediction_rows,columns=['series_id','step','event','score'])
        ours=event_detection_ap(t,p)
        reference=official_score(t,p)
        assert abs(ours-reference)<1e-9, (case,ours,reference)


def test_reference_matches_b1_pooled_oof():
    _reference_or_skip()
    pred_file=Path(os.environ.get('HHN_B1_OOF_PREDICTIONS','outputs/b1_oof_predictions.parquet'))
    ev_file=Path(os.environ.get('HHN_EVENT_NIGHTS','outputs/event_nights.parquet'))
    if not pred_file.is_file() or not ev_file.is_file():
        message=("B1 OOF REGRESSION NOT RUN: expected files "
                 f"{pred_file} and {ev_file}. Run once B1 OOF is available.")
        warnings.warn(message, RuntimeWarning,stacklevel=2)
        pytest.skip(message)
    pred=_load(pred_file)
    event_nights=_load(ev_file)
    truth=truth_from_event_nights(event_nights)
    prediction_events=pred.loc[:,['series_id','step','event','score']]
    assert abs(event_detection_ap(truth,prediction_events)-
               official_score(truth,prediction_events))<1e-9


def test_reference_matches_with_heavy_score_ties():
    """Many tied scores + dense GTs: greedy matching order must follow the reference."""
    _reference_or_skip()
    import numpy as np
    rng = np.random.default_rng(1)
    for _ in range(4):
        t, p = [], []
        for i in range(4):
            for n in range(40):
                for e, off in (("onset", 3000), ("wakeup", 9000)):
                    gt = n * 400 + off
                    t.append((f"s{i}", gt, e))
                    for _k in range(3):
                        p.append((f"s{i}", gt + int(rng.integers(-300, 300)), e,
                                  float(rng.integers(0, 4)) / 4))
        T = pd.DataFrame(t, columns=["series_id", "step", "event"])
        P = pd.DataFrame(p, columns=["series_id", "step", "event", "score"])
        assert abs(event_detection_ap(T, P) - official_score(T, P)) < 1e-9
