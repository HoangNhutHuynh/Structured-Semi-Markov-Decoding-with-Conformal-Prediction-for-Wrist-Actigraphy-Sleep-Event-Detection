# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Synthetic runtime check: ~300k predictions, ~7,673 windows, 277 series."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from src.metrics.bootstrap import OOFEvaluator, bootstrap_oof


def synthetic(n_series=277, nights=28, peaks=20, seed=0):
    rng = np.random.default_rng(seed)
    t, p, w, pw = [], [], [], []
    for i in range(n_series):
        sid = f"s{i:03d}"
        for n in range(nights):
            nd = (pd.Timestamp("2018-01-01") + pd.Timedelta(days=n)).strftime("%Y-%m-%d")
            on, wk = n * 17280 + 7000 + int(rng.integers(-500, 500)), n * 17280 + 13000 + int(rng.integers(-500, 500))
            lab = rng.random() < 0.65
            w.append((sid, nd, "labelled" if lab else "unlabelled", on if lab else np.nan,
                      wk if lab else np.nan, float(rng.random() < 0.1), 600.0, 1100.0))
            if lab:
                t += [(sid, on, "onset"), (sid, wk, "wakeup")]
            po, pk = on + int(rng.integers(-60, 60)), wk + int(rng.integers(-60, 60))
            pw.append((sid, nd, po, pk, 601.0, 1101.0))
            for e, c in (("onset", on), ("wakeup", wk)):
                for k in range(peaks):
                    p.append((sid, c + int(rng.integers(-1500, 1500)), e, float(rng.random()), nd))
    return (pd.DataFrame(t, columns=["series_id", "step", "event"]),
            pd.DataFrame(p, columns=["series_id", "step", "event", "score", "night_date"]),
            pd.DataFrame(w, columns=["series_id", "night_date", "label_status", "onset_step",
                                     "wakeup_step", "nonwear_fraction", "onset_clock_min", "wakeup_clock_min"]),
            pd.DataFrame(pw, columns=["series_id", "night_date", "onset_step", "wakeup_step",
                                      "onset_clock_min", "wakeup_clock_min"]))


if __name__ == "__main__":
    import sys
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    t, p, w, pw = synthetic()
    print(f"predictions={len(p):,} windows={len(w):,} truth={len(t):,}")
    a = time.time(); ev = OOFEvaluator(t, p, w, pw); b = time.time()
    ev.evaluate(); c = time.time()
    print(f"precompute (matching x3): {b-a:.1f} s; one evaluate: {c-b:.2f} s")
    a = time.time(); r = bootstrap_oof(t, p, w, pw, repetitions=reps); b = time.time()
    print(f"bootstrap {reps} resamples total: {b-a:.1f} s")
    print(r.set_index("metric").loc[["official_edap", "anti_edap_k1", "onset_medae_min", "icc2_1_spt"]])
