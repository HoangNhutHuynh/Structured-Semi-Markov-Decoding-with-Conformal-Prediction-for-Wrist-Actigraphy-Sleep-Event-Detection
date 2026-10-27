# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Export data for the qualitative case-study figure (Fig: case study) to CSV
for MATLAB plotting.

Picks from the seed-42 OOF posteriors:
  (a) one NORMAL night   : low p_null, labelled GT inside the window
  (b) one NON-WEAR night : p_null > 0.5 (proxy-negative behaviour)

Writes to outputs/fig_case_study/:
  normal.csv        t_hr, anglez, enmo, m_on, m_wk      (one row per 1-min bin)
  normal_meta.csv   p_null, map_on_hr, map_wk_hr, gt_on_hr, gt_wk_hr,
                    hpd_on_lo_hr, hpd_on_hi_hr, hpd_wk_lo_hr, hpd_wk_hi_hr
  nonwear.csv       same columns as normal.csv
  nonwear_meta.csv  same columns as normal_meta.csv (gt_* = NaN if unlabelled)

Usage:
  python -m scripts.export_fig_case_study --seed-dir outputs/p2_main/seed42
  # pick a different normal night (0 = lowest p_null with GT):
  python -m scripts.export_fig_case_study --normal-rank 3
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.b1_features import load_series, SPM   # SPM = steps per 1-min bin (12)


def hpd_interval(m: np.ndarray, level: float = 0.90):
    """Smallest set of bins (by density) holding `level` of the normalized mass.
    Returns (lo_bin, hi_bin) of the covering interval for display."""
    m = m.astype(np.float64)
    tot = m.sum()
    if tot <= 0:
        return np.nan, np.nan
    p = m / tot
    order = np.argsort(p)[::-1]
    k = int(np.searchsorted(np.cumsum(p[order]), level)) + 1
    sel = np.sort(order[:min(k, len(p))])
    return int(sel[0]), int(sel[-1])


def export_night(z, i, cache_dir: Path, b1_cache: Path, en: pd.DataFrame,
                 out_csv: Path, out_meta: Path):
    sid = str(z["series_id"][i])
    w, n = int(z["w"][i]), int(z["n"][i])
    pn = float(z["p_null"][i])
    s = load_series(cache_dir, b1_cache, sid)
    s0 = int(s.win_start[w])

    # per-bin signal (mean of the SPM 5-s samples in each 1-min bin)
    df = pd.read_parquet(cache_dir / f"{sid}.parquet").set_index("step")
    steps = np.arange(s0, s0 + n * SPM)
    sig = df.reindex(steps)
    ang = np.nanmean(sig["anglez"].to_numpy(np.float64).reshape(n, SPM), axis=1)
    enm = np.nanmean(sig["enmo"].to_numpy(np.float64).reshape(n, SPM), axis=1)

    m_on = z["m_on"][i][:n].astype(np.float64)
    m_wk = z["m_wk"][i][:n].astype(np.float64)
    t_hr = np.arange(n) / 60.0

    pd.DataFrame({"t_hr": t_hr, "anglez": ang, "enmo": enm,
                  "m_on": m_on, "m_wk": m_wk}).to_csv(out_csv, index=False)

    # ground truth (absolute steps -> window bins -> hours)
    gt_on_hr = gt_wk_hr = np.nan
    g = en[(en["series_id"] == sid)
           & (en["onset_step"] >= s0) & (en["onset_step"] < s0 + n * SPM)]
    if len(g):
        gt_on_hr = (float(g["onset_step"].iloc[0]) - s0) / SPM / 60.0
        wk = float(g["wakeup_step"].iloc[0])
        gt_wk_hr = min(max((wk - s0) / SPM, 0), n - 1) / 60.0

    on_lo, on_hi = hpd_interval(m_on)
    wk_lo, wk_hi = hpd_interval(m_wk)
    pd.DataFrame([{
        "series_id": sid, "window": w, "p_null": pn,
        "map_on_hr": int(z["a"][i]) / 60.0, "map_wk_hr": int(z["b"][i]) / 60.0,
        "gt_on_hr": gt_on_hr, "gt_wk_hr": gt_wk_hr,
        "hpd_on_lo_hr": on_lo / 60.0, "hpd_on_hi_hr": on_hi / 60.0,
        "hpd_wk_lo_hr": wk_lo / 60.0, "hpd_wk_hi_hr": wk_hi / 60.0,
    }]).to_csv(out_meta, index=False)
    print(f"  {out_csv.name}: sid={sid} w={w} n={n} p_null={pn:.3f} "
          f"gt_on={gt_on_hr if not np.isnan(gt_on_hr) else float('nan'):.2f}h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", default="outputs/p2_main/seed42")
    ap.add_argument("--out-dir", default="outputs/fig_case_study")
    ap.add_argument("--normal-rank", type=int, default=0,
                    help="0 = lowest-p_null labelled night; try 1,2,... for others")
    args = ap.parse_args()

    sd = Path(args.seed_dir)
    cache_dir = Path("outputs/cache")
    b1_cache = Path("outputs/b1_cache")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    z = dict(np.load(sd / "oof_posteriors.npz"))
    en = pd.read_parquet("outputs/event_nights.parquet")
    en = en[en["label_status"] == "labelled"]

    n_ok = z["n"].astype(int) >= 31
    pn = z["p_null"].astype(float)

    # (a) normal night: low p_null, labelled GT inside window
    cand = []
    for i in np.argsort(pn):
        if not n_ok[i]:
            continue
        sid, w = str(z["series_id"][i]), int(z["w"][i])
        s = load_series(cache_dir, b1_cache, sid)
        s0 = int(s.win_start[w]); n = int(z["n"][i])
        g = en[(en["series_id"] == sid)
               & (en["onset_step"] >= s0) & (en["onset_step"] < s0 + n * SPM)]
        if len(g):
            cand.append(int(i))
        if len(cand) > args.normal_rank:
            break
    if not cand:
        raise SystemExit("no labelled low-p_null night found")
    i_norm = cand[min(args.normal_rank, len(cand) - 1)]

    # (b) non-wear night: highest p_null
    i_null = int(np.argmax(np.where(n_ok, pn, -1.0)))
    if pn[i_null] <= 0.5:
        print(f"WARNING: max p_null = {pn[i_null]:.3f} <= 0.5 "
              "(no abstained window in this subset)")

    print("exporting:")
    export_night(z, i_norm, cache_dir, b1_cache, en,
                 out / "normal.csv", out / "normal_meta.csv")
    export_night(z, i_null, cache_dir, b1_cache, en,
                 out / "nonwear.csv", out / "nonwear_meta.csv")
    print(f"done -> {out}/")


if __name__ == "__main__":
    main()
