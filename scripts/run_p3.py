# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""PHASE 3: per-night conformal intervals + selective prediction, from saved Phase 2 posteriors.

Offline (no GPU, no retraining). Reads <seed_dir>/oof_posteriors.npz, outputs/night_windows.parquet
and outputs/splits.json.

Nonconformity scores (per event, onset and wakeup separately)
  abs : s = |MAP_bin - truth_bin| in minutes -> interval MAP +- q  (fixed width, non-adaptive)
  hpd : s = posterior mass of bins strictly more likely than the truth bin (adaptive prediction
        set). The set at level q is {u : mass of bins more likely than u <= q}; its width adapts
        to how peaked that night's posterior is.

Calibration splits (subject-level exchangeability; never uses the test fold's own subjects)
  val   : the fold's inner-validation series. Those series were used for epoch selection, so this
          is mildly optimistic; reported as the primary analysis with that caveat.
  cross : cross-conformal. For test fold k, calibrate on the OOF predictions of the other folds'
          test series, which their models never saw at all. No selection leakage; the price is
          that calibration and test predictions come from different fold models.

Coverage is reported with subject-level bootstrap CIs (nights within a child are not independent).

Selective prediction sweeps a confidence measure and plots risk vs coverage.
Usage:
  python -m scripts.run_p3 --config configs/phase1.yaml --seed-dir outputs/p2_main/seed42
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.common import load_config

ALPHAS = (0.10, 0.05)
SPM = 12  # 5-s samples per 1-min bin


# ---------------------------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile: ceil((n+1)(1-alpha))/n empirical quantile (finite-sample valid)."""
    n = len(scores)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def _mass_above(m: np.ndarray) -> np.ndarray:
    """mass_above[u] = total mass of bins STRICTLY more likely than u (exact under ties).

    Ties matter: posteriors are stored as float16, so many bins share a value (often 0). The
    conformal guarantee only holds if the prediction set is exactly {u : score(u) <= q}, so the
    set and the score must use identical tie handling.
    """
    p = m / max(m.sum(), 1e-12)
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    cum = np.r_[0.0, np.cumsum(ps)]
    starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(ps)]))
    out = np.empty_like(p)
    out[order] = cum[group_start]
    return out


def hpd_score(m: np.ndarray, t: int) -> float:
    """Mass of bins strictly more likely than bin t (0 = truth is the mode, ~1 = least likely)."""
    return float(_mass_above(m)[t])


def hpd_width(m: np.ndarray, q: float) -> int:
    """Size of the level-q prediction set {u : hpd_score(u) <= q}, in bins (= minutes)."""
    return int((_mass_above(m) <= q).sum())


def hpd_interval(m: np.ndarray, q: float, centre: int) -> tuple[int, int]:
    """Contiguous run of the level-q set around the MAP bin (reported in minutes)."""
    inset = _mass_above(m) <= q
    lo = hi = centre
    while lo > 0 and inset[lo - 1]:
        lo -= 1
    while hi < len(inset) - 1 and inset[hi + 1]:
        hi += 1
    return lo, hi


# ---------------------------------------------------------------------------------------------
def build_table(z: dict, windows: pd.DataFrame) -> pd.DataFrame:
    """One row per labelled window with a posterior: MAP bins, truth bins, scores, confidence."""
    lab = windows[windows.label_status == "labelled"].set_index(["series_id", "night_date"])
    rows = []
    for i in range(len(z["n"])):
        sid, nd = str(z["series_id"][i]), str(z["night_date"][i])
        if (sid, nd) not in lab.index:
            continue
        r = lab.loc[(sid, nd)]
        n = int(z["n"][i])
        if n < 31:
            continue
        m_on = z["m_on"][i][:n].astype(np.float64)
        m_wk = z["m_wk"][i][:n].astype(np.float64)
        rows.append(dict(series_id=sid, night_date=nd, idx=i, n=n,
                         a=int(z["a"][i]), b=int(z["b"][i]), p_null=float(z["p_null"][i]),
                         onset_step=int(r.onset_step), wakeup_step=int(r.wakeup_step),
                         conf_on=float(m_on.max()), conf_wk=float(m_wk.max())))
    return pd.DataFrame(rows)


def add_truth_bins(tab: pd.DataFrame, data) -> pd.DataFrame:
    """Truth bin index inside each window tensor (uses the same mapping as training)."""
    a_bins, b_bins = [], []
    for r in tab.itertuples(index=False):
        s = data[r.series_id]
        w = int(np.flatnonzero(s.win_dates == r.night_date)[0])
        start = int(s.win_start[w])
        a_bins.append((int(np.searchsorted(s.steps, r.onset_step)) - start) // SPM)
        b_bins.append((int(np.searchsorted(s.steps, r.wakeup_step)) - start) // SPM)
    return tab.assign(t_on=a_bins, t_wk=b_bins)


def scores_for(tab: pd.DataFrame, z: dict, kind: str) -> pd.DataFrame:
    on, wk = [], []
    for r in tab.itertuples(index=False):
        m_on = z["m_on"][r.idx][:r.n].astype(np.float64)
        m_wk = z["m_wk"][r.idx][:r.n].astype(np.float64)
        t_on, t_wk = int(np.clip(r.t_on, 0, r.n - 1)), int(np.clip(r.t_wk, 0, r.n - 1))
        if kind == "abs":
            on.append(abs(r.a - t_on)); wk.append(abs(r.b - t_wk))
        else:
            on.append(hpd_score(m_on, t_on)); wk.append(hpd_score(m_wk, t_wk))
    return tab.assign(s_on=on, s_wk=wk)


def subject_bootstrap_ci(values: np.ndarray, subjects: np.ndarray, reps=1000, seed=0):
    ids = np.unique(subjects)
    idx = {s: np.flatnonzero(subjects == s) for s in ids}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        draw = rng.choice(ids, len(ids), replace=True)
        out.append(values[np.concatenate([idx[s] for s in draw])].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--seed-dir", required=True)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--abstain-null", type=float, default=0.5)
    x = ap.parse_args()
    cfg = load_config(x.config); out = Path(cfg["paths"]["output_dir"]); sd = Path(x.seed_dir)
    from src.data.b1_features import load_series
    z = dict(np.load(sd / "oof_posteriors.npz"))
    windows = pd.read_parquet(out / "night_windows.parquet").astype({"series_id": str, "night_date": str})
    folds = json.loads((out / "splits.json").read_text(encoding="utf-8"))["folds"]
    data = {sid: load_series(Path(cfg["paths"]["cache_dir"]), out / "b1_cache", sid)
            for sid in sorted(set(map(str, z["series_id"])))}
    tab = add_truth_bins(build_table(z, windows), data)
    test_fold = {str(s): f for f, p in folds.items() for s in p["test"]}
    val_series = {f: set(map(str, p["val"])) for f, p in folds.items()}
    tab["fold"] = tab.series_id.map(test_fold)
    res = sd / "p3"; res.mkdir(exist_ok=True)
    rows, per_night = [], None

    for kind in ("abs", "hpd"):
        st = scores_for(tab, z, kind)
        for calib in ("val", "cross"):
            for alpha in ALPHAS:
                cov, wid, keep = [], [], []
                for f in sorted(folds, key=int):
                    te = st[st.fold == f]
                    ca = (st[st.series_id.isin(val_series[f])] if calib == "val"
                          else st[st.fold != f])
                    if len(ca) == 0 or len(te) == 0:
                        continue
                    for ev, sc, pred, truth, mkey in (("onset", "s_on", "a", "t_on", "m_on"),
                                                      ("wakeup", "s_wk", "b", "t_wk", "m_wk")):
                        q = conformal_quantile(ca[sc].to_numpy(), alpha)
                        for r in te.itertuples(index=False):
                            s_i = getattr(r, sc); p_i = getattr(r, pred); t_i = getattr(r, truth)
                            if kind == "abs":
                                covered = s_i <= q; w = 2 * q + 1
                            else:
                                m = z[mkey][r.idx][:r.n].astype(np.float64)
                                covered = s_i <= q; w = hpd_width(m, q)
                            cov.append(dict(series_id=r.series_id, night_date=r.night_date, event=ev,
                                            covered=bool(covered), width_min=float(w),
                                            p_null=r.p_null, conf=r.conf_on if ev == "onset" else r.conf_wk,
                                            abs_err_min=abs(p_i - t_i), fold=f, q=q))
                c = pd.DataFrame(cov)
                for ev, g in c.groupby("event"):
                    lo, hi = subject_bootstrap_ci(g.covered.to_numpy(float), g.series_id.to_numpy(),
                                                  x.reps)
                    rows.append(dict(score=kind, calibration=calib, alpha=alpha, event=ev, n=len(g),
                                     target=1 - alpha, coverage=float(g.covered.mean()),
                                     cov_lo=lo, cov_hi=hi,
                                     median_width_min=float(g.width_min.median()),
                                     p90_width_min=float(g.width_min.quantile(0.9))))
                if kind == "hpd" and calib == "cross" and alpha == 0.10:
                    per_night = c
    summary = pd.DataFrame(rows)
    summary.to_csv(res / "conformal_coverage.csv", index=False)
    print("\nConformal coverage (subject-level bootstrap CI):\n" +
          summary.round(3).to_string(index=False))

    # ---- selective prediction: abstain on high p_null, then on wide intervals ------------------
    if per_night is not None:
        per_night.to_csv(res / "per_night_hpd_cross_a10.csv", index=False)
        sel = []
        for name, key, asc in (("p_null", "p_null", True), ("interval width", "width_min", True),
                               ("peak mass", "conf", False)):
            d = per_night.sort_values(key, ascending=asc).reset_index(drop=True)
            for frac in np.arange(0.5, 1.001, 0.05):
                k = int(len(d) * frac)
                g = d.iloc[:k]
                sel.append(dict(criterion=name, coverage_frac=float(frac),
                                median_abs_err_min=float(g.abs_err_min.median()),
                                pct_err_gt30=float(100 * (g.abs_err_min > 30).mean()),
                                conformal_coverage=float(g.covered.mean())))
        s = pd.DataFrame(sel); s.to_csv(res / "risk_coverage.csv", index=False)
        print("\nRisk-coverage (HPD/cross/90%; retained nights sorted by each criterion):\n" +
              s.pivot_table(index="coverage_frac", columns="criterion",
                            values="pct_err_gt30").round(2).to_string())
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
            for name, g in s.groupby("criterion"):
                ax[0].plot(g.coverage_frac * 100, g.pct_err_gt30, marker="o", ms=3, label=name)
            ax[0].set_xlabel("nights retained (%)"); ax[0].set_ylabel("% retained with error > 30 min")
            ax[0].set_title("Risk-coverage (selective prediction)", fontsize=9); ax[0].legend(fontsize=7)
            w = per_night.width_min.clip(upper=np.percentile(per_night.width_min, 99))
            ax[1].hist(w, bins=40, color="0.35")
            ax[1].set_xlabel("90% prediction-set width (min)"); ax[1].set_ylabel("nights")
            ax[1].set_title("Per-night interval width", fontsize=9)
            fig.savefig(res / "risk_coverage.png", dpi=150)
        except Exception as exc:
            print("plot failed:", exc)


if __name__ == "__main__":
    main()
