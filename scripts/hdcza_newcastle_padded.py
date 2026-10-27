# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
r"""HDCZA on Newcastle WITH vs WITHOUT the 90-min simulated-wake padding of van Hees et al. 2018.

Why: the published HDCZA result on these recordings (MAE onset 30.8 / wake 47.1 min, left wrist)
padded each recording with 90 min of simulated wakefulness at both ends; our Table (Newcastle)
re-implements HDCZA on the raw ~12-h lab recording without padding. This script reruns both the
GGIR-default and the LOSO-tuned HDCZA with padding in {0, 90} min on the SAME 22 labelled
nights, and reports MAE (as in [5]) and MedAE (as in our table), so the manuscript can state
whether padding explains the gap.

Simulated wakefulness = i.i.d. uniform z-angle in [-90, 90] deg per 5-s epoch (fixed seed). Its
rolling median |dz| is ~tens of degrees, far above the [0.13, 0.50] threshold clamp, so padding
can never become a sleep candidate; it only (i) gives the percentile threshold wake context and
(ii) prevents blocks from being cut at the recording edges. NOTE: check GGIR's exact padding
signal if you cite "identical to [5]"; describe it as "90 min of simulated wakefulness" otherwise.

Inputs (already produced by run_newcastle --prep): outputs/newcastle/night_windows.parquet,
outputs/newcastle/cache/<sid>.parquet, outputs/newcastle/feat_cache/<sid>.npz.
Outputs: outputs/newcastle/results/hdcza_padding_summary.csv and hdcza_newcastle_pad{N}[_tuned].csv

Usage (from hhn_phase1_stepA, CPU only, ~1 min):
  python -m scripts.hdcza_newcastle_padded
  python -m scripts.hdcza_newcastle_padded --pads 0 90 120
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.baselines.hdcza import (WindowInput, rolling_signal, hdcza_window, hdcza_window_grid,
                                 DEFAULT, SAMPLES_PER_MIN)
from src.data.b1_features import load_series
from src.metrics.clinical import _ba, icc2_1

EPOCH_S = 5


def make_input(steps: np.ndarray, ang: np.ndarray, pad_min: int, seed: int) -> WindowInput:
    n = int(pad_min * SAMPLES_PER_MIN)
    if n == 0:
        return WindowInput(steps, rolling_signal(ang), np.ones(len(steps), bool))
    rng = np.random.default_rng(seed)
    a = np.r_[rng.uniform(-90, 90, n), ang, rng.uniform(-90, 90, n)]
    st = np.r_[steps[0] - np.arange(n, 0, -1), steps, steps[-1] + np.arange(1, n + 1)]
    return WindowInput(st.astype(np.int64), rolling_signal(a), np.ones(len(a), bool))


def row(sid, res, on_true, wk_true, lo, hi, combo=None):
    on = int(np.clip(res[0], lo, hi)); wk = int(np.clip(res[1], lo, hi))
    d = dict(series_id=sid,
             onset_err_min=(on - on_true) * EPOCH_S / 60,
             wake_err_min=(wk - wk_true) * EPOCH_S / 60,
             spt_pred_min=(wk - on) * EPOCH_S / 60,
             spt_true_min=(wk_true - on_true) * EPOCH_S / 60)
    if combo is not None:
        d["combo"] = str(combo)
    return d


def summarise(df: pd.DataFrame, name: str, pad: int, n_lab: int) -> dict:
    if not len(df):
        return dict(method=name, pad_min=pad, n=0, n_failed=n_lab)
    ba = _ba(df.spt_true_min.to_numpy(), df.spt_pred_min.to_numpy())
    on, wk = df.onset_err_min.abs(), df.wake_err_min.abs()
    return dict(method=name, pad_min=pad, n=len(df), n_failed=n_lab - len(df),
                onset_MAE=on.mean(), onset_MedAE=on.median(), onset_bias=df.onset_err_min.mean(),
                wake_MAE=wk.mean(), wake_MedAE=wk.median(), wake_bias=df.wake_err_min.mean(),
                onset_within30_pct=100 * (on <= 30).mean(), wake_within30_pct=100 * (wk <= 30).mean(),
                spt_bias=ba["bias"], spt_loa_lo=ba["loa_low"], spt_loa_hi=ba["loa_high"],
                spt_icc=icc2_1(df.spt_true_min, df.spt_pred_min))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/newcastle")
    ap.add_argument("--pads", type=int, nargs="+", default=[0, 90])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); res_dir = out / "results"; res_dir.mkdir(exist_ok=True)

    win = pd.read_parquet(out / "night_windows.parquet").astype({"series_id": str, "night_date": str})
    lab = win[win.label_status == "labelled"].reset_index(drop=True)
    data = {s: load_series(out / "cache", out / "feat_cache", s) for s in lab.series_id}
    print(f"{len(lab)} labelled Newcastle nights")

    # the same window slice as run_newcastle.py (night window containing the PSG onset)
    sl = {}
    for r in lab.itertuples(index=False):
        s_ = data[r.series_id]
        w = int(np.flatnonzero(s_.win_dates == r.night_date)[0])
        a0, b0 = int(s_.win_start[w]), int(s_.win_end[w])
        ang = pq.read_table(out / "cache" / f"{r.series_id}.parquet",
                            columns=["anglez"]).to_pandas().anglez.to_numpy()[a0:b0]
        sl[r.series_id] = (s_.steps[a0:b0].astype(np.int64), ang,
                           int(r.onset_step), int(r.wakeup_step))

    summary = []
    for pad in a.pads:
        # ---- GGIR default -------------------------------------------------------------------
        rows = []
        for i, (sid, (steps, ang, on_t, wk_t)) in enumerate(sl.items()):
            res = hdcza_window(make_input(steps, ang, pad, a.seed + i), **DEFAULT)
            if res is not None:
                rows.append(row(sid, res, on_t, wk_t, int(steps[0]), int(steps[-1]) + 1))
        df = pd.DataFrame(rows)
        df.to_csv(res_dir / f"hdcza_newcastle_pad{pad}.csv", index=False)
        summary.append(summarise(df, "HDCZA default", pad, len(sl)))

        # ---- LOSO-tuned (same grid + objective as run_newcastle.py) -------------------------
        grid = {sid: hdcza_window_grid(make_input(steps, ang, pad, a.seed + i))
                for i, (sid, (steps, ang, _, _)) in enumerate(sl.items())}
        combos = list(next(iter(grid.values())))

        def objective(sids, c):
            e = [(abs(grid[s][c][0] - sl[s][2]) * EPOCH_S / 60,
                  abs(grid[s][c][1] - sl[s][3]) * EPOCH_S / 60)
                 for s in sids if grid[s][c] is not None]
            if len(e) < 0.9 * len(sids):
                return np.inf
            x, y = zip(*e)
            return (np.median(x) + np.median(y)) / 2

        rows = []
        for sid, (steps, ang, on_t, wk_t) in sl.items():
            best = min(combos, key=lambda c: objective([s for s in sl if s != sid], c))
            res = grid[sid][best]
            if res is not None:
                rows.append(row(sid, res, on_t, wk_t, int(steps[0]), int(steps[-1]) + 1, best))
        dt = pd.DataFrame(rows)
        dt.to_csv(res_dir / f"hdcza_newcastle_pad{pad}_tuned.csv", index=False)
        summary.append(summarise(dt, "HDCZA LOSO-tuned", pad, len(sl)))
        if len(dt):
            print(f"pad {pad}: tuned combos (P,M,Bmin,G) {dt.combo.value_counts().to_dict()}")

    sm = pd.DataFrame(summary)
    sm.to_csv(res_dir / "hdcza_padding_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== HDCZA on Newcastle, padding comparison (minutes) ===")
    print(sm.round(1).to_string(index=False))
    print("\nPublished (van Hees 2018, left wrist, 90-min padding): onset MAE 30.8, wake MAE 47.1")
    print("Sanity: pad 0 rows must reproduce hdcza_newcastle.csv / hdcza_newcastle_tuned.csv "
          "(default onset MedAE 231.0, tuned 85.0).")


if __name__ == "__main__":
    main()
