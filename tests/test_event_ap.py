# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Reference-oriented hand-computed EDAP tests, independent of model predictions."""
import pandas as pd
import numpy as np
import pytest

from src.metrics.event_ap import (
    event_detection_ap, cap_candidates, anti_gaming_edap,
    truth_from_event_nights,
)


def frame(rows, scored=False):
    return pd.DataFrame(rows, columns=['series_id','step','event','score'] if scored else ['series_id','step','event'])


TRUTH=frame([('A',100,'onset'),('A',400,'wakeup')])


def test_toy_1_perfect():
    pred=frame([('A',100,'onset',0.9),('A',400,'wakeup',0.8)],True)
    assert event_detection_ap(TRUTH,pred)==pytest.approx(1.0)


def test_toy_2_missing_event():
    pred=frame([('A',100,'onset',0.9)],True)
    assert event_detection_ap(TRUTH,pred)==pytest.approx(0.5)


def test_toy_3_high_confidence_false_positive_degrades_ap():
    pred=frame([('A',900,'onset',0.9),('A',100,'onset',0.8),('A',400,'wakeup',0.9)],True)
    assert event_detection_ap(TRUTH,pred)==pytest.approx(0.75)


def test_toy_4_low_confidence_duplicate_does_not_degrade_ap():
    pred=frame([('A',100,'onset',0.9),('A',101,'onset',0.2),('A',400,'wakeup',0.9)],True)
    assert event_detection_ap(TRUTH,pred)==pytest.approx(1.0)


def test_toy_5_cannot_match_across_series():
    pred=frame([('B',100,'onset',0.9),('A',400,'wakeup',0.9)],True)
    assert event_detection_ap(TRUTH,pred)==pytest.approx(0.5)


def test_toy_6_strict_boundary_12_steps():
    pred=frame([('A',112,'onset',0.9),('A',400,'wakeup',0.8)],True)
    # Onset missed at tol=12, detected at other 9; wakeup 10/10.
    assert event_detection_ap(TRUTH,pred)==pytest.approx(0.95)


def test_toy_7_equal_scores_share_one_threshold():
    pred=frame([('A',900,'onset',0.5),('A',100,'onset',0.5),('A',400,'wakeup',0.8)],True)
    assert event_detection_ap(TRUTH,pred)==pytest.approx(0.75)


def test_toy_8_empty_predictions():
    pred=frame([],True)
    assert event_detection_ap(TRUTH,pred)==pytest.approx(0.0)


def test_nearest_unmatched_truth_is_selected():
    truth=frame([('A',100,'onset'),('A',115,'onset'),('A',400,'wakeup')])
    pred=frame([('A',108,'onset',0.9),('A',100,'onset',0.8),('A',400,'wakeup',0.8)],True)
    # First pred at 108 is nearer to 115; second can then match 100.
    score,components=event_detection_ap(truth,pred,tolerances=(12,),return_components=True)
    assert score==pytest.approx(1.0)
    assert components.n_matches.sum()==3


def test_caps_per_event_per_window_and_does_not_shift_step():
    truth=frame([('A',100,'onset'),('A',400,'wakeup')])
    pred=frame([('A',900,'onset',0.99),('A',100,'onset',0.80),
                ('A',400,'wakeup',0.70),('A',420,'wakeup',0.30)],True)
    pred['night_date']='2023-11-04'
    selected=cap_candidates(pred,1)
    assert sorted(selected.step.tolist())==[400,900]
    assert anti_gaming_edap(truth,pred,k=1)==pytest.approx(0.5)
    assert anti_gaming_edap(truth,pred,k=3)==pytest.approx(0.75)


def test_keep_cross_boundary_and_partial_event_steps():
    events=pd.DataFrame([
        {'series_id':'A','label_status':'cross_boundary','onset_step':100,'wakeup_step':400},
        {'series_id':'A','label_status':'partial','onset_step':None,'wakeup_step':900},
        {'series_id':'A','label_status':'unlabelled','onset_step':None,'wakeup_step':None}])
    t=truth_from_event_nights(events)
    assert len(t)==3
    assert set(t.step.astype(int))=={100,400,900}


def test_reject_missing_window_when_capping():
    pred=frame([('A',100,'onset',0.9)],True)
    with pytest.raises(ValueError): cap_candidates(pred,1)
