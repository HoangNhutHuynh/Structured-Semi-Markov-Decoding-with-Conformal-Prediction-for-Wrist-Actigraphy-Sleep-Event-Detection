# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Add-on diagnostic report for EXISTING, patched make_splits.py.

Do not overwrite local Step A source. This reads its on-disk split assignment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml

STATUSES = ("labelled", "cross_boundary", "partial", "unlabelled", "unannotated")


def split_table(folds: dict, windows: pd.DataFrame) -> pd.DataFrame:
    needed = {"series_id", "label_status", "nonwear_fraction"}
    if not needed.issubset(windows.columns):
        raise ValueError(f"night_windows.parquet missing {sorted(needed-set(windows.columns))}")
    if not windows.label_status.isin(STATUSES).all():
        raise ValueError("unexpected window label_status")
    windows=windows.copy()
    windows["series_id"]=windows.series_id.astype(str)
    counts = windows.groupby(["series_id", "label_status"]).size().unstack(fill_value=0)
    for status in STATUSES:
        if status not in counts.columns:
            counts[status]=0
    counts=counts[list(STATUSES)]
    proxy = windows.loc[(windows.label_status.isin(["unlabelled","unannotated"])) &
                        (windows.nonwear_fraction>=0.90)]
    prox_count = proxy.groupby(["series_id", "label_status"]).size().unstack(fill_value=0)
    for status in ("unlabelled", "unannotated"):
        counts[f"proxy_negative_{status}"] = prox_count.get(status,pd.Series(dtype=int)).reindex(counts.index,fill_value=0)
    rows=[]
    universe = set(windows.series_id)
    for fold, groups in sorted(folds.items(), key=lambda kv:int(kv[0])):
        for partition in ("train", "val", "test"):
            ids = list(map(str,groups[partition]))
            missing = set(ids)-universe
            if missing:
                raise ValueError(f"fold {fold}/{partition}: missing series IDs: {sorted(missing)}")
            sums=counts.reindex(ids,fill_value=0).sum()
            row={"fold":int(fold),"split":partition,"n_series":len(ids)}
            for k,v in sums.items():
                row[k]=int(v)
            rows.append(row)
    return pd.DataFrame(rows)


def print_split_report(output_dir: str | Path) -> pd.DataFrame:
    out=Path(output_dir)
    saved=json.loads((out/'splits.json').read_text(encoding='utf8'))
    night_windows=pd.read_parquet(out/'night_windows.parquet')
    table=split_table(saved['folds'],night_windows)
    tested = [sid for fold in saved['folds'].values() for sid in fold['test']]
    expected_subjects = set(night_windows.series_id.astype(str))
    if len(tested) != len(set(tested)) or set(tested) != expected_subjects:
        raise AssertionError("Each observed series_id must appear in exactly one TEST fold")
    print('\nv1.1 SPLITS: observed-window counts; proxy >=90% non-wear')
    print(table.to_string(index=False))
    table.to_csv(out/'split_counts.csv',index=False)
    # Partition totals over test subsets must match each observed window once.
    totals=table.loc[table.split=='test',list(STATUSES)].sum()
    observed=night_windows.label_status.value_counts()
    assert all(int(totals[k]) == int(observed.get(k,0)) for k in STATUSES), (
        'Test folds do not cover observed window population exactly once')
    print('OUTER-TEST COVERAGE VALIDATED: all observed windows included exactly once.')
    return table


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/phase1.yaml')
    args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    print_split_report(config['paths']['output_dir'])


if __name__=='__main__':
    main()
