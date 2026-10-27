# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
from src.metrics.clinical import wall_minutes_from_noon, evaluate_clinical
from src.metrics.epoch import confusion_counts, scores_from_counts, evaluate_epoch_samples


def test_wall_clock_is_not_timezone_converted():
    assert wall_minutes_from_noon('2023-11-05T01:30:00-0400','2023-11-04')==810
    assert wall_minutes_from_noon('2023-11-05T01:30:00-0500','2023-11-04')==810
    assert wall_minutes_from_noon('2023-11-04 15:00:00-0400','2023-11-04')==180
    assert wall_minutes_from_noon('2023-11-05 08:00:00-0500','2023-11-04')==1200


def _samples():
    w=pd.DataFrame([
        ('S','2023-11-04','labelled',100,820,0.0),
        ('S','2023-11-05','labelled',2000,2700,0.0),
        ('S','2023-11-06','cross_boundary',3000,3700,0.0),
        ('S','2023-11-07','unlabelled',None,None,0.98),
        ('S','2023-11-08','unannotated',None,None,0.99),
    ],columns=['series_id','night_date','label_status','onset_step','wakeup_step','nonwear_fraction'])
    p=pd.DataFrame([
        ('S','2023-11-04',112,820),
        ('S','2023-11-06',3200,3900),
        ('S','2023-11-07',4010,4500),
    ],columns=['series_id','night_date','onset_step','wakeup_step'])
    look=pd.DataFrame([(sid,step,ts) for sid,step,ts in [
        ('S',100,'2023-11-04 22:00:00-0400'),
        ('S',112,'2023-11-04 22:01:00-0400'),
        ('S',820,'2023-11-04 23:00:00-0400'),
    ]],columns=['series_id','step','timestamp'])
    return w,p,look


def test_clinical_excludes_cross_boundary_and_computes_rates():
    w,p,look=_samples()
    x=evaluate_clinical(w,p,look,include_icc=False)
    assert x['n_labelled']==2
    assert x['n_labelled_scored']==1
    assert x['window_recall_labelled']==0.5
    assert x['onset_medae_min']==1.0
    assert x['wakeup_medae_min']==0.0
    assert x['onset_within15_pct']==100.0
    assert x['predicted_rate_unlabelled']==1.0
    assert x['predicted_rate_unannotated']==0.0
    assert x['n_proxy_negative_unlabelled']==1
    assert x['n_proxy_negative_unannotated']==1
    assert x['proxy_false_detection_rate_unlabelled']==1.0
    assert x['proxy_false_detection_rate_unannotated']==0.0
    assert x['ba_onset_clock_bias']==1.0
    assert x['ba_wakeup_clock_bias']==0.0
    assert x['ba_spt_bias']==-1.0


def test_icc_2_1_when_pingouin_available():
    pytest.importorskip('pingouin')
    w,p,_=_samples()
    p=pd.concat([p,pd.DataFrame([('S','2023-11-05',2012,2712)],columns=p.columns)],ignore_index=True)
    # Second labelled row must be in timestamp lookups.
    ts=pd.DataFrame([(sid,st,t) for sid,st,t in [
        ('S',100,'2023-11-04 22:00:00-0400'),('S',112,'2023-11-04 22:01:00-0400'),
        ('S',820,'2023-11-04 23:00:00-0400'),('S',2000,'2023-11-05 22:00:00-0500'),
        ('S',2012,'2023-11-05 22:01:00-0500'),('S',2700,'2023-11-06 23:00:00-0500'),
        ('S',2712,'2023-11-06 23:01:00-0500')]],columns=['series_id','step','timestamp'])
    x=evaluate_clinical(w,p,ts)
    assert 'icc2_1_spt' in x
    assert np.isfinite(x['icc2_1_spt']) or np.isnan(x['icc2_1_spt'])


def test_epoch_counts_respect_valid_and_labelled_masks():
    truth=np.array([1,1,0,0,1,0])
    pred=np.array([1,0,0,1,0,0])
    valid=np.array([1,1,1,1,1,0],dtype=bool)
    labelled=np.array([1,1,1,0,0,0],dtype=bool)
    count=confusion_counts(truth,pred,valid)
    assert count=={'tp':1,'tn':1,'fp':1,'fn':2}
    x=evaluate_epoch_samples(truth,pred,valid,labelled)
    assert x['all_accuracy']==pytest.approx(2/5)
    assert x['all_f1']==pytest.approx(2/5)
    assert x['labelled_accuracy']==pytest.approx(2/3)
    assert x['labelled_sensitivity']==pytest.approx(1/2)


def test_reject_prediction_outside_observed_windows():
    w,p,look=_samples()
    p.loc[0,'night_date']='2023-01-01'
    with pytest.raises(ValueError):evaluate_clinical(w,p,look,include_icc=False)
