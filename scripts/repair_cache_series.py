# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Rebuild the Step A cache files for specific series directly from train_series.parquet.

Usage:  python -m scripts.repair_cache_series --config configs/phase1.yaml 062cae666e2a
Creates outputs/cache/<sid>.parquet, <sid>_nonwear.npy, <sid>_sr2025.npy exactly as Step A does
(same columns, step-sorted non-wear flags, sr2025 labels from event_nights.parquet).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from src.data.common import load_config
from src.data.pipeline import load_series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("series", nargs="+")
    a = ap.parse_args()
    cfg = load_config(a.config)
    paths, processing = cfg["paths"], cfg["processing"]
    source = Path(paths["data_dir"]) / paths["series_file"]
    cache = Path(paths["cache_dir"]); out = Path(paths["output_dir"])
    ev = pd.read_parquet(out / "event_nights.parquet")
    dataset = ds.dataset(source, format="parquet")
    for sid in a.series:
        table = dataset.to_table(columns=["series_id", "step", "timestamp", "anglez", "enmo"],
                                 filter=ds.field("series_id") == sid)
        if table.num_rows == 0:
            raise SystemExit(f"{sid}: not found in {source}")
        pq.write_table(table, cache / f"{sid}.parquet", compression="zstd")
        df = load_series(cache, sid, processing)          # same validation + non-wear as Step A
        np.save(cache / f"{sid}_nonwear.npy", df.nonwear.to_numpy(dtype=bool))
        steps = df.step.to_numpy(dtype=np.int64)
        sr = np.zeros(len(df), dtype=np.uint8)
        for _, r in ev[(ev.series_id == sid) & (ev.label_status == "labelled")].iterrows():
            sr[(steps >= int(r.onset_step)) & (steps < int(r.wakeup_step))] = 1
        np.save(cache / f"{sid}_sr2025.npy", sr)
        print(f"{sid}: {table.num_rows:,} rows, non-wear {df.nonwear.mean():.1%}, "
              f"sleep-labelled {sr.mean():.1%} -> {cache}")


if __name__ == "__main__":
    main()
