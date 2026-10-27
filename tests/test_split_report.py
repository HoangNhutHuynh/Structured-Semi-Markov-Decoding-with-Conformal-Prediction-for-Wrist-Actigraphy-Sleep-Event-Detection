# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
import pandas as pd
from scripts.split_report import split_table


def test_window_counts_and_proxy_split_by_status():
    w=pd.DataFrame([('A','labelled',.0),('A','cross_boundary',.1),
                    ('A','unlabelled',.99),('B','unannotated',.90),
                    ('B','partial',.01),('B','unlabelled',.89)],
                   columns=['series_id','label_status','nonwear_fraction'])
    folds={'1':{'train':['A'],'val':[],'test':['B']},
           '2':{'train':['B'],'val':[],'test':['A']}}
    t=split_table(folds,w)
    assert len(t)==6
    a=t.query('fold==2 and split=="test"').iloc[0]
    assert a['labelled']==1 and a['cross_boundary']==1 and a['proxy_negative_unlabelled']==1
    b=t.query('fold==1 and split=="test"').iloc[0]
    assert b['unannotated']==1 and b['proxy_negative_unannotated']==1
