# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from src.data.pipeline import detect_nonwear,night_date,map_annotations
from scripts.make_splits import generate_splits

def test_noon_assignment_across_dst():
    u=pd.Series(pd.to_datetime(['2021-11-07T15:00:00Z','2021-11-07T18:00:00Z',
                                '2021-11-08T14:00:00Z'],utc=True)).dt.tz_convert('America/New_York')
    assert night_date(u).tolist()==['2021-11-06','2021-11-07','2021-11-07']

def test_repeat_by_step_with_gaps():
    step=np.arange(2200,dtype=np.int64)
    angle=np.arange(2200,dtype=float); enmo=angle.copy()
    angle[1000:1720]=angle[0:720]
    enmo[1000:1720]=enmo[0:720]
    flags=detect_nonwear(step,angle,enmo,shift=1000,min_steps=720)
    assert flags[1000:1720].all() and flags[:720].all()
    assert not flags[1720:].any()

def test_anchor_mismatch_blocks_gap():
    events=pd.DataFrame([{'night':1,'onset_step':10.,'wakeup_step':11.,'label_status':'labelled'},
                         {'night':2,'onset_step':np.nan,'wakeup_step':np.nan,'label_status':'unlabelled'},
                         {'night':3,'onset_step':30.,'wakeup_step':31.,'label_status':'labelled'}])
    mapping, issues=map_annotations('s',events,{10:'2021-01-01',11:'2021-01-01',
                            30:'2021-01-04',31:'2021-01-04'},set(['2021-01-01','2021-01-04']))
    assert len(issues['d'])==1 and issues['d'][0]['night']==2
    assert len(issues['c'])==0 and len(mapping)==2  # (d) nights are not double-counted in (c)

def test_partial_wake_crosscheck():
    events=pd.DataFrame([{'night':1,'onset_step':10.,'wakeup_step':11.,'label_status':'labelled'},
                         {'night':2,'onset_step':np.nan,'wakeup_step':31.,'label_status':'partial'}])
    _,issues=map_annotations('s',events,{10:'2021-01-01',11:'2021-01-01',31:'2021-01-03'},
                            {'2021-01-01','2021-01-02','2021-01-03'})
    assert len(issues['a'])==1

def test_split_disjoint():
    ids=[f's{i:03d}' for i in range(40)]
    events=pd.DataFrame([{'series_id':s,'label_status':'labelled'} for i,s in enumerate(ids) for _ in range(i%9)])
    splits=generate_splits(events,ids)
    assert len(set().union(*(set(v['test']) for v in splits['folds'].values())))==40
