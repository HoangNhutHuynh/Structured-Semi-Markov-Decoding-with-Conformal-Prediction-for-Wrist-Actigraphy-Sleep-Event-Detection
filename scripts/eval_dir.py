# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Bootstrap + clinical evaluation of any directory holding oof_predictions_kaggle/onepair and
oof_predicted_windows parquet files (same evaluation as B1/P2)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from src.data.common import load_config
from src.data.clock import load_clock_index
from src.metrics.event_ap import truth_from_event_nights
from scripts.run_b1 import evaluate_seed, cell


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--dir", required=True); ap.add_argument("--reps", type=int, default=1000)
    x = ap.parse_args(); out = Path(load_config(x.config)["paths"]["output_dir"])
    win = pd.read_parquet(out / "night_windows.parquet").astype({"series_id": str, "night_date": str})
    truth = truth_from_event_nights(pd.read_parquet(out / "event_nights.parquet")).astype({"series_id": str})
    folds = json.loads((out / "splits.json").read_text(encoding="utf-8"))["folds"]
    b, _ = evaluate_seed(Path(x.dir), truth, win, load_clock_index(out / "clock_index.json"), folds, x.reps, None)
    for m in ["official_edap", "anti_edap_k1", "anti_edap_k3", "onset_medae_min", "wakeup_medae_min",
              "onset_within30_pct", "wakeup_within30_pct", "window_recall_labelled",
              "proxy_false_detection_rate_unlabelled", "proxy_false_detection_rate_unannotated",
              "ba_spt_bias", "icc2_1_spt"]:
        print(f"{m:>40s}  {cell(b, m)}")
    print(f"{'SPT LoA':>40s}  {b.loc['ba_spt_loa_low'].estimate:.1f} to {b.loc['ba_spt_loa_high'].estimate:.1f}")


if __name__ == "__main__":
    main()
