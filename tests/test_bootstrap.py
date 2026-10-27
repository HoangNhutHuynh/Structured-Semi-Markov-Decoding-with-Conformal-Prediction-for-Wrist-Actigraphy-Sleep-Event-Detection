# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
from __future__ import annotations
import numpy as np
import pandas as pd
from src.metrics.bootstrap import bootstrap_oof, per_fold_official_edap, per_fold_edap_summary


def test_subject_bootstrap_counts_and_per_fold_ap():
    truth=pd.DataFrame([('A',100,'onset'),('A',400,'wakeup'),('B',100,'onset'),('B',400,'wakeup')],
                       columns=['series_id','step','event'])
    pred=truth.copy();pred['score']=.8;pred['night_date']='2023-10-01'
    windows=pd.DataFrame([('A','2023-10-01','labelled',100,400,0),
                          ('B','2023-10-01','labelled',100,400,0)],
                         columns=['series_id','night_date','label_status','onset_step','wakeup_step','nonwear_fraction'])
    # No predicted windows: clinical accuracy undefined, event AP still well-defined.
    pairs=pd.DataFrame(columns=['series_id','night_date','onset_step','wakeup_step'])
    report=bootstrap_oof(truth,pred,windows,pairs,repetitions=12,valid_threshold=10)
    score=report.set_index('metric').loc['official_edap']
    assert score['estimate']==1.0
    assert score['ci_low']==1.0 and score['ci_high']==1.0
    assert score['valid_resamples']==12
    undef=report.set_index('metric').loc['icc2_1_spt']
    assert undef['valid_resamples']==0 and np.isnan(undef['ci_low'])
    folds={'1':{'test':['A']},'2':{'test':['B']}}
    fold_report=per_fold_official_edap(truth,pred,folds)
    assert fold_report.official_edap.tolist()==[1,1]
    summary=per_fold_edap_summary(fold_report)
    assert summary["n_folds"]==2 and summary["official_edap_sd"]==0.0
