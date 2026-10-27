# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Offline decision-theoretic decoding of saved Phase 2 posteriors (no retraining).

EDAP scores every candidate with ONE score across 10 tolerances tau (1..30 min, strict <).
Given an event marginal m(t) (sums to 1 - p_null per window), placing a candidate at bin c gives an
expected number of matches at tolerance tau of  H_tau(c) = sum_{|t-c| < tau} m(t).
Expected-AP decoding: G(c) = mean_tau H_tau(c); position = argmax G, score = G(c).
Variants written to <seed_dir>/decoded/<name>/ in the same layout as model outputs:
  mass2     : current rule (peaks of m, score = mass within +-2 bins)           [reference]
  eap       : K=1 at argmax G (score G);  Kaggle-mode = peaks of G (distance 6, top 20)
  eap_null  : as eap, plus one-pair abstention when p_null > 0.5 (Kaggle-mode keeps all windows)
  rule_null : as eap, plus a NON-LEARNED control: abstain when the window's
              non-wear fraction >= 0.9 (same threshold that defines proxy-negatives).
              Intended for the no-null model (outputs/p2_no_null) to test whether the
              learned null state merely replicates its non-wear input feature.
Usage: python -m scripts.decode_posteriors --config configs/phase1.yaml --seed-dir outputs/p2_main/seed42
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from src.data.common import load_config
from src.data.b1_features import load_series, bin_to_step, SPM
from src.metrics.event_ap import truth_from_event_nights, event_detection_ap, cap_candidates

TOL_BINS = np.array([12, 36, 60, 90, 120, 150, 180, 240, 300, 360]) / 12.0   # 1 .. 30 min
COLS = ["series_id", "step", "event", "score"]


def gain_curve(m: np.ndarray) -> np.ndarray:
    """G(c) = mean over tolerances of sum_{|t-c|<tau} m(t)  (strict <, bins)."""
    cs = np.r_[0.0, np.cumsum(m.astype(np.float64))]
    n = len(m); c = np.arange(n); G = np.zeros(n)
    for tau in TOL_BINS:
        r = int(np.ceil(tau) - 1)                     # |t-c| < tau  <=>  |t-c| <= ceil(tau)-1
        lo = np.clip(c - r, 0, n); hi = np.clip(c + r + 1, 0, n)
        G += cs[hi] - cs[lo]
    return G / len(TOL_BINS)


def decode(z, data, variant, null_thr=0.5):
    ev_k, ev_1, pw = [], [], []
    for i in range(len(z["n"])):
        sid, nd, w, n = str(z["series_id"][i]), str(z["night_date"][i]), int(z["w"][i]), int(z["n"][i])
        if n < 31:
            continue
        s = data[sid]; pn = float(z["p_null"][i]); one = {}
        for key, event in (("m_on", "onset"), ("m_wk", "wakeup")):
            m = z[key][i][:n].astype(np.float64)
            if variant == "mass2":
                curve = np.convolve(m, np.ones(5), mode="same")
                peaks, _ = find_peaks(m, height=1e-4, distance=6)
                best = int(z["a"][i] if event == "onset" else z["b"][i])
            else:
                curve = gain_curve(m)
                peaks, _ = find_peaks(curve, height=1e-4, distance=6)
                best = int(np.argmax(curve))
            if len(peaks):
                top = peaks[np.argsort(-curve[peaks])[:20]]
                for st, v in zip(bin_to_step(s, w, top), curve[top]):
                    ev_k.append((sid, nd, event, int(st), float(v)))
            one[event] = (best, float(curve[best]))
        if variant == "eap_null" and pn > null_thr:
            continue
        if variant == "rule_null":
            w0 = int(s.win_start[w])
            seg = s.feats[w0:w0 + n * SPM, 5]
            nw = float(np.mean(seg)) if len(seg) else 1.0
            if nw >= 0.9:
                continue
        (a, sa), (b, sb) = one["onset"], one["wakeup"]
        if b <= a:                                    # keep a valid pair for clinical metrics
            a, b = int(z["a"][i]), int(z["b"][i])
        st_on, st_wk = bin_to_step(s, w, [a, b])
        ev_1 += [(sid, nd, "onset", int(st_on), sa), (sid, nd, "wakeup", int(st_wk), sb)]
        pw.append((sid, nd, int(st_on), int(st_wk)))
    cols = ["series_id", "night_date", "event", "step", "score"]
    return (pd.DataFrame(ev_k, columns=cols), pd.DataFrame(ev_1, columns=cols),
            pd.DataFrame(pw, columns=["series_id", "night_date", "onset_step", "wakeup_step"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--seed-dir", required=True)
    ap.add_argument("--variants", nargs="+", default=["mass2", "eap", "eap_null"])
    ap.add_argument("--null-threshold", type=float, default=0.5,
                    help="Abstention threshold for eap_null (p_null > thr). "
                         "Non-default values write to decoded/eap_null_tNN/")
    x = ap.parse_args()
    cfg = load_config(x.config); out = Path(cfg["paths"]["output_dir"]); sd = Path(x.seed_dir)
    z = dict(np.load(sd / "oof_posteriors.npz"))
    data = {sid: load_series(Path(cfg["paths"]["cache_dir"]), out / "b1_cache", sid)
            for sid in sorted(set(map(str, z["series_id"])))}
    truth = truth_from_event_nights(pd.read_parquet(out / "event_nights.parquet")).astype({"series_id": str})
    rows = []
    for v in x.variants:
        ev_k, ev_1, pw = decode(z, data, v, x.null_threshold)
        vname = v
        if v == "eap_null" and abs(x.null_threshold - 0.5) > 1e-9:
            vname = f"eap_null_t{int(round(x.null_threshold * 100)):02d}"
        d = sd / "decoded" / vname; d.mkdir(parents=True, exist_ok=True)
        ev_k.to_parquet(d / "oof_predictions_kaggle.parquet", index=False)
        ev_1.to_parquet(d / "oof_predictions_onepair.parquet", index=False)
        pw.to_parquet(d / "oof_predicted_windows.parquet", index=False)
        rows.append({"variant": v, "EDAP_kaggle": event_detection_ap(truth, ev_k[COLS]),
                     "EDAP_K1_capped": event_detection_ap(truth, cap_candidates(ev_k, 1)[COLS]),
                     "EDAP_onepair": event_detection_ap(truth, ev_1[COLS]), "n_windows": len(pw)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nFull bootstrap/clinical table for a variant:  python -m scripts.run_b1 style evaluation via\n"
          "  python -m scripts.eval_dir --config configs/phase1.yaml --dir <seed_dir>/decoded/<variant>")


if __name__ == "__main__":
    main()
