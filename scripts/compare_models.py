# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Where does EDAP differ between two models? Per-tolerance AP, and score ranking quality.

Usage: python -m scripts.compare_models --config configs/phase1.yaml \
          --a outputs/b1/seed42 --b outputs/p2_main/seed42
For each model: per-event/per-tolerance AP for (i) Kaggle-mode candidates and (ii) top-1 per window
(K=1); and, on labelled windows, how well the one-pair score ranks correct vs wrong nights
(AUROC of score for |error| < 1, < 5, < 30 min) — AP at a tolerance is high only if correct
predictions get higher scores than wrong ones across ALL windows.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.common import load_config
from src.metrics.event_ap import truth_from_event_nights, event_detection_ap, cap_candidates

COLS = ["series_id", "step", "event", "score"]


def auroc(pos_scores, neg_scores):
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    s = np.r_[pos_scores, neg_scores]; y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - len(pos_scores) * (len(pos_scores) + 1) / 2) / (len(pos_scores) * len(neg_scores)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    x = ap.parse_args()
    out = Path(load_config(x.config)["paths"]["output_dir"])
    truth = truth_from_event_nights(pd.read_parquet(out / "event_nights.parquet")).astype({"series_id": str})
    win = pd.read_parquet(out / "night_windows.parquet").astype({"series_id": str, "night_date": str})
    lab = win[win.label_status == "labelled"]
    comp_rows, rank_rows = [], []
    for name, d in (("A", Path(x.a)), ("B", Path(x.b))):
        ev = pd.read_parquet(d / "oof_predictions_kaggle.parquet").astype({"series_id": str, "night_date": str})
        for mode, e in (("kaggle", ev), ("K1", cap_candidates(ev, 1))):
            _, comp = event_detection_ap(truth, e[COLS], return_components=True)
            comp["model"], comp["mode"] = f"{name}:{d.parent.name}/{d.name}", mode
            comp_rows.append(comp)
        one = pd.read_parquet(d / "oof_predictions_onepair.parquet").astype({"series_id": str, "night_date": str})
        for event in ("onset", "wakeup"):
            m = one[one.event == event].merge(lab[["series_id", "night_date", f"{event}_step"]],
                                              on=["series_id", "night_date"])
            err = (m.step - m[f"{event}_step"]).abs() * 5 / 60
            row = {"model": f"{name}:{d.parent.name}/{d.name}", "event": event, "n": len(m)}
            for t in (1, 5, 30):
                row[f"AUROC_score_err<{t}min"] = auroc(m.score[err < t].to_numpy(), m.score[err >= t].to_numpy())
                row[f"pct_err<{t}min"] = float(100 * (err < t).mean())
            rank_rows.append(row)
    comp = pd.concat(comp_rows)
    piv = comp.pivot_table(index=["event", "tolerance"], columns=["model", "mode"], values="ap").round(3)
    print("Per-tolerance AP (tolerance in steps; 12 = 1 min):\n" + piv.to_string())
    print("\nMean AP by mode:\n" + comp.groupby(["model", "mode"]).ap.mean().round(4).to_string())
    print("\nOne-pair score ranking on labelled nights:\n" + pd.DataFrame(rank_rows).round(3).to_string(index=False))
    res = out / "compare"; res.mkdir(exist_ok=True)
    piv.to_csv(res / "per_tolerance_ap.csv"); pd.DataFrame(rank_rows).to_csv(res / "score_ranking.csv", index=False)


if __name__ == "__main__":
    main()
