# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Create deterministic per-series stratified 5-fold splits with inner grouped validation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from src.data.common import load_config, seed_all, write_manifest


def assign_bins(counts: pd.Series) -> np.ndarray:
    """Zero-labelled = 0; positive counts divided into four rank-quantiles."""
    result=np.zeros(len(counts),dtype=np.int64)
    positive=(counts.to_numpy()>0)
    # rank(method='first') makes quartiles deterministic for tied integer counts.
    vals=counts[positive]
    result[positive]=pd.qcut(vals.rank(method='first'),q=4,labels=False).to_numpy(dtype=int)+1
    return result


def generate_splits(event_nights:pd.DataFrame, series_ids:list[str], n_splits:int=5,
                    val_fraction:float=.15, seed:int=42) -> dict:
    counts=event_nights.loc[event_nights.label_status=='labelled'].groupby('series_id').size()
    frame=pd.DataFrame({'series_id':sorted(series_ids)})
    frame['n_labelled']=frame.series_id.map(counts).fillna(0).astype(int)
    frame['stratum']=assign_bins(frame.n_labelled)
    groups=frame.series_id.to_numpy(); y=frame.stratum.to_numpy()
    cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    result={'version':'1.1','seed':seed,'stratification':'0 or positive rank quartile',
            'folds':{}}
    for fold,(trainval_idx,test_idx) in enumerate(cv.split(frame,y,groups),start=1):
        trainval=frame.iloc[trainval_idx]
        inner=GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed)
        ix_train,ix_val=next(inner.split(trainval, groups=trainval.series_id))
        members={
            'train':sorted(trainval.iloc[ix_train].series_id.tolist()),
            'val':sorted(trainval.iloc[ix_val].series_id.tolist()),
            'test':sorted(frame.iloc[test_idx].series_id.tolist())}
        for name,subset in members.items():
            selected=event_nights.loc[event_nights.series_id.isin(subset)]
            counts_by_status=selected.label_status.value_counts()
            print(f'fold={fold} {name}: series={len(subset)} '
                  f'labelled={counts_by_status.get("labelled",0)} '
                  f'unlabelled={counts_by_status.get("unlabelled",0)} '
                  f'partial={counts_by_status.get("partial",0)}')
        result['folds'][str(fold)]=members
    all_ids=set(groups)
    test_parts=[set(v['test']) for v in result['folds'].values()]
    assert set.union(*test_parts)==all_ids
    assert sum(len(x) for x in test_parts)==len(all_ids)
    for v in result['folds'].values():
        assert not(set(v['train'])&set(v['val']) or set(v['train'])&set(v['test']) or set(v['val'])&set(v['test']))
    result['series_bins']=dict(zip(groups.tolist(),y.astype(int).tolist()))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase1.yaml')
    a=p.parse_args(); cfg=load_config(a.config);seed_all(int(cfg['seed']))
    out=Path(cfg['paths']['output_dir'])
    ev=pd.read_parquet(out/'event_nights.parquet')
    summary=pd.read_csv(out/'series_summary.csv',dtype={'series_id':str})
    result=generate_splits(ev,summary.series_id.tolist(),int(cfg['splits']['n_splits']),
                           float(cfg['splits']['val_fraction']),int(cfg['seed']))
    (out/'splits.json').write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8')
    write_manifest(cfg,out/'splits_manifest.json')

if __name__=='__main__':main()
