# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Step C diagnostics on B0 (HDCZA default) OOF predictions.

A) Label-provenance check: if CMI labels were produced by an HDCZA/GGIR-like pipeline and then
   minute-rounded, HDCZA errors concentrate at |error| < 1 min. Report the error histogram
   (1-min bins), fractions within 0.5/1/2/5 min, and tails by non-wear tertile and weekday/weekend.
B) EDAP decomposition: official EDAP vs EDAP with predictions restricted to (i) labelled windows
   ('oracle night selection') and (ii) windows that carry any event-file label. The gap measures
   how much EDAP is lost by predicting on nights the annotators left unlabelled.
Usage: python -m scripts.analyze_b0 --config configs/phase1.yaml [--variant b0_default]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.common import load_config
from src.metrics.event_ap import truth_from_event_nights, event_detection_ap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--variant", default="b0_default")
    a = ap.parse_args()
    out = Path(load_config(a.config)["paths"]["output_dir"])
    res = out / "b0_analysis"; res.mkdir(exist_ok=True)
    win = pd.read_parquet(out / "night_windows.parquet").astype({"series_id": str, "night_date": str})
    pw = pd.read_parquet(out / a.variant / "oof_predicted_windows.parquet").astype(
        {"series_id": str, "night_date": str})
    ev = pd.read_parquet(out / a.variant / "oof_predictions.parquet").astype(
        {"series_id": str, "night_date": str})
    truth = truth_from_event_nights(pd.read_parquet(out / "event_nights.parquet"))
    truth["series_id"] = truth.series_id.astype(str)
    summary = {"variant": a.variant}

    # ---- A) error distribution on labelled windows -----------------------------------------
    m = win[win.label_status == "labelled"].merge(pw, on=["series_id", "night_date"],
                                                   suffixes=("_true", "_pred"))
    for e in ("onset", "wakeup"):
        m[f"{e}_err_min"] = (m[f"{e}_step_pred"] - m[f"{e}_step_true"]) * 5 / 60
    for e in ("onset", "wakeup"):
        x = m[f"{e}_err_min"].to_numpy(); ax = np.abs(x)
        summary[e] = {"n": int(len(x)),
                      **{f"pct_abs_lt_{t}min": float(100 * np.mean(ax < t)) for t in (0.5, 1, 2, 5)},
                      "pct_abs_gt_30min": float(100 * np.mean(ax > 30)),
                      "pct_abs_gt_60min": float(100 * np.mean(ax > 60)),
                      "median_signed_min": float(np.median(x)),
                      "mean_signed_min": float(np.mean(x))}
        hist, edges = np.histogram(np.clip(x, -60.5, 60.5), bins=np.arange(-60.5, 61.5, 1))
        pd.DataFrame({"bin_center_min": (edges[:-1] + 0.5).astype(int), "count": hist}).to_csv(
            res / f"{e}_error_hist_1min.csv", index=False)
    # tails by non-wear tertile and weekday/weekend (night_date Fri/Sat = weekend night)
    m["nonwear_tertile"] = pd.qcut(m.nonwear_fraction.rank(method="first"), 3,
                                   labels=["low", "mid", "high"])
    m["weekend_night"] = pd.to_datetime(m.night_date).dt.dayofweek.isin([4, 5])
    m["worst_abs_err"] = m[["onset_err_min", "wakeup_err_min"]].abs().max(axis=1)
    tails = []
    for col in ("nonwear_tertile", "weekend_night"):
        g = m.groupby(col, observed=True).worst_abs_err
        t = pd.DataFrame({"n": g.size(), "median_worst_abs_err_min": g.median(),
                          "pct_worst_gt_30min": g.apply(lambda s: 100 * (s > 30).mean())})
        t.insert(0, "factor", col); tails.append(t.reset_index().rename(columns={col: "level"}))
    tails = pd.concat(tails); tails.to_csv(res / "error_tails.csv", index=False)

    # ---- B) EDAP decomposition ---------------------------------------------------------------
    cols = ["series_id", "step", "event", "score"]
    labelled_keys = set(map(tuple, win.loc[win.label_status == "labelled", ["series_id", "night_date"]].values))
    any_keys = set(map(tuple, win.loc[win.label_status.isin(["labelled", "cross_boundary", "partial"]),
                                      ["series_id", "night_date"]].values))
    key = list(zip(ev.series_id, ev.night_date))
    ev_lab = ev[[k in labelled_keys for k in key]]
    ev_any = ev[[k in any_keys for k in key]]
    status = ev.merge(win[["series_id", "night_date", "label_status"]], on=["series_id", "night_date"])
    summary["edap"] = {
        "official_all_windows": event_detection_ap(truth, ev[cols]),
        "only_labelled_windows": event_detection_ap(truth, ev_lab[cols]),
        "only_windows_with_any_label": event_detection_ap(truth, ev_any[cols]),
        "n_predicted_events_by_status": status.label_status.value_counts().to_dict()}
    (res / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- figure -------------------------------------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True, sharey=True)
        for axx, e in zip(axes, ("onset", "wakeup")):
            h = pd.read_csv(res / f"{e}_error_hist_1min.csv")
            axx.bar(h.bin_center_min, h["count"], width=0.9, color="0.35")
            axx.set_title(f"{e}: HDCZA (default) − label, 1-min bins", fontsize=9)
            axx.set_xlabel("signed error (min); edges clipped at ±60"); axx.set_xlim(-61, 61)
        axes[0].set_ylabel("labelled nights")
        fig.savefig(res / "error_hist.png", dpi=150)
    except Exception as exc:
        print("plot failed:", exc)

    print(json.dumps(summary, indent=2))
    print(tails.to_string(index=False))


if __name__ == "__main__":
    main()
