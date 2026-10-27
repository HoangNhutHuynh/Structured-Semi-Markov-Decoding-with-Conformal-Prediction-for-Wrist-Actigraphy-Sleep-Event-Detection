# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
r"""Score decoded/eap_null predictions for seeds 43, 44 (seed 42 already done),
then aggregate all 3 seeds into outputs/p2_main/decoded_seed_summary.csv.

Usage:
    cd <repository root>
    python -m scripts.score_decoded_seeds --config configs/phase1.yaml
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.common import load_config
from src.data.clock import load_clock_index
from src.metrics.event_ap import truth_from_event_nights
from scripts.run_b1 import evaluate_seed


SEEDS = [42, 43, 44]
KEY_METRICS = [
    "official_edap", "anti_edap_k1", "anti_edap_k3",
    "onset_medae_min", "wakeup_medae_min",
    "window_recall_labelled",
    "proxy_false_detection_rate_unlabelled",
    "proxy_false_detection_rate_unannotated",
    "ba_spt_bias", "ba_spt_loa_low", "ba_spt_loa_high",
    "icc2_1_spt", "onepair_official_edap",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--reps", type=int, default=1000,
                    help="Bootstrap repetitions (use 1 for quick point estimates)")
    ap.add_argument("--force", action="store_true",
                    help="Re-score even if metrics.json already exists")
    ap.add_argument("--variant", default="eap_null",
                    help="Decoded variant to score, e.g. eap_null, eap, mass2, "
                         "rule_null, eap_null_t30, eap_null_t70 (default: eap_null)")
    ap.add_argument("--model-dir", default="p2_main",
                    help="Model output folder under outputs/ (e.g. p2_no_null)")
    args = ap.parse_args()
    variant = args.variant
    suffix = "" if variant == "eap_null" else f"_{variant}"

    cfg = load_config(args.config)
    out = Path(cfg["paths"]["output_dir"])
    win = pd.read_parquet(out / "night_windows.parquet").astype(
        {"series_id": str, "night_date": str})
    truth = truth_from_event_nights(
        pd.read_parquet(out / "event_nights.parquet")).astype({"series_id": str})
    folds = json.loads((out / "splits.json").read_text(encoding="utf-8"))["folds"]
    clock = load_clock_index(out / "clock_index.json")

    all_metrics = {}
    for seed in SEEDS:
        d = out / args.model_dir / f"seed{seed}" / "decoded" / variant
        mj = d / "metrics.json"
        if mj.exists() and not args.force:
            print(f"seed{seed}: metrics.json already exists, loading...")
            all_metrics[seed] = json.loads(mj.read_text(encoding="utf-8"))
        else:
            print(f"seed{seed}: scoring decoded/{variant} (reps={args.reps})...")
            _, metrics = evaluate_seed(d, truth, win, clock, folds, args.reps, None)
            all_metrics[seed] = metrics
            print(f"  -> EDAP={metrics['official_edap']:.4f}, "
                  f"K1={metrics['anti_edap_k1']:.4f}, "
                  f"bias={metrics['ba_spt_bias']:.2f}")

    # --- Aggregate into decoded_seed_summary.csv ---
    rows_point = []
    for seed in SEEDS:
        row = {"seed": seed}
        row.update({k: all_metrics[seed].get(k, float("nan")) for k in KEY_METRICS})
        rows_point.append(row)

    df_point = pd.DataFrame(rows_point)
    df_point.to_csv(out / args.model_dir / f"decoded_seed_point_estimates{suffix}.csv", index=False)

    summary = {}
    for k in KEY_METRICS:
        vals = [all_metrics[s][k] for s in SEEDS if k in all_metrics[s]]
        summary[k] = {"mean_over_seeds": np.mean(vals), "sd_over_seeds": np.std(vals, ddof=1)}

    df_summary = pd.DataFrame(summary).T
    df_summary.to_csv(out / args.model_dir / f"decoded_seed_summary{suffix}.csv")

    print(f"\n=== Decoded 3-Seed Summary ({variant}) ===")
    for k in KEY_METRICS:
        m = summary[k]["mean_over_seeds"]
        s = summary[k]["sd_over_seeds"]
        print(f"  {k:>45s}  {m:.4f} ± {s:.4f}")

    print(f"\nSaved: {out / args.model_dir / ('decoded_seed_summary' + suffix + '.csv')}")
    print(f"Saved: {out / args.model_dir / ('decoded_seed_point_estimates' + suffix + '.csv')}")


if __name__ == "__main__":
    main()
