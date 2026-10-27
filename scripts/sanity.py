# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Report event-file and observed-window populations."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.common import load_config

EXPECTED={'labelled':4790,'unlabelled':2459,'partial':5}  # event-file view (window view may show cross_boundary)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/phase1.yaml')
    a=p.parse_args(); out=Path(load_config(a.config)['paths']['output_dir'])
    ev=pd.read_parquet(out/'event_nights.parquet')
    w=pd.read_parquet(out/'night_windows.parquet')
    subjects=pd.read_csv(out/'series_summary.csv')
    print('EVENT-FILE VIEW')
    print('Series:',ev.series_id.nunique(),'nights:',len(ev))
    print(ev.label_status.value_counts().to_string())
    observed=ev.label_status.value_counts().to_dict()
    assert ev.series_id.nunique()==277 and len(ev)==7254, 'Event-file count mismatch'
    assert all(observed.get(k,0)==v for k,v in EXPECTED.items()), 'Event status mismatch'
    print('\nWINDOW VIEW')
    print('Observed windows:',len(w),'series:',w.series_id.nunique())
    print(w.label_status.value_counts().to_string())
    print('Unannotated boundary windows per series:')
    print(subjects.unannotated_boundary_windows.describe().to_string())
    durations=ev.loc[ev.label_status=='labelled'].copy()
    durations['duration_min']=(durations.wakeup_step-durations.onset_step)*5/60
    print('\nEVENT-FILE SLEEP DURATION (minutes):')
    print(durations.duration_min.describe(percentiles=[.25,.5,.75]).to_string())
    print('\nNONWEAR (%) PER SERIES:')
    print((subjects.nonwear_fraction*100).describe(percentiles=[.25,.5,.75]).to_string())
    print('Overall observation-weighted nonwear (%):',
          100*np.average(subjects.nonwear_fraction,weights=subjects.n_steps))
    import json
    checks=json.loads((out/'integrity_report.json').read_text(encoding='utf-8'))['checks']
    print('\nINTEGRITY:',{key:val['count'] for key,val in checks.items()})

if __name__=='__main__': main()
