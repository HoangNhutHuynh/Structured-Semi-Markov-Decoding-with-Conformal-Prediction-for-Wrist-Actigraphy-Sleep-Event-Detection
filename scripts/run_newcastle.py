# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""EXTERNAL VALIDATION on Newcastle PSG + GENEActiv (Zenodo 1160410).

Stage 1 (--prep): decode .bin, autocalibrate, 5-s anglez/ENMO, parse PSG, and write a cache in
exactly the CMI layout (outputs/newcastle/cache/<sid>.parquet + _nonwear.npy, night_windows.parquet)
so the trained models and all metric code run unchanged.

Stage 2 (--eval): run the five CMI fold models of a Phase 2 seed (each with ITS OWN training-fold
normalisation statistics - nothing is refit on Newcastle), and report per-model and ensemble
onset/wake error, Bland-Altman on SPT, ICC, and whether the CMI-calibrated conformal quantiles
still cover on this cohort.

Honest framing: the models were trained on children (CMI, rule-based diary labels); Newcastle is
adult sleep-clinic patients with PSG labels and a different device. Degradation is expected and is
the point of the experiment. Reference = PSG sleep period time (first to last sleep epoch).

Usage:
  python -m scripts.run_newcastle --config configs/phase1.yaml --data "path\\to\\psgnewcastle2015" --prep
  python -m scripts.run_newcastle --config configs/phase1.yaml --seed-dir outputs/p2_main/seed42 --eval
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.data.common import load_config
from src.data.geneactiv import read_bin, autocalibrate, to_epochs
from src.data.newcastle import index_folder, parse_psg
from src.metrics.clinical import icc2_1, _ba

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*generic.*unit.*")
BOUNDARY_HOUR = 15
EPOCH_S = 5
SPM = 12


# ---------------------------------------------------------------------------------- stage 1 ----
def prep(data_dir: Path, out: Path, wrist: str, max_pages: int | None) -> None:
    cache = out / "cache"; cache.mkdir(parents=True, exist_ok=True)
    if not data_dir.is_dir():
        sib = ""
        if data_dir.parent.exists():
            names = sorted(c.name for c in data_dir.parent.iterdir())[:12]
            sib = f"\n  {data_dir.parent} contains: {names}"
        raise SystemExit(f"ERROR: --data is not a folder.\n  received (repr): {str(data_dir)!r}"
                         f"\n  exists={data_dir.exists()} is_dir={data_dir.is_dir()}{sib}"
                         f"\n  Pass the EXTRACTED folder, e.g. --data \"path\\to\\psgnewcastle2015\"")
    n_bin = len(list(data_dir.rglob("*.bin"))); n_txt = len(list(data_dir.rglob("*.txt")))
    pairing = index_folder(data_dir, wrist)
    print(f"scanning {data_dir}: {n_bin} .bin, {n_txt} .txt -> {len(pairing)} participants "
          f"({int(pairing.bin.notna().sum()) if len(pairing) else 0} with a {wrist}-wrist .bin)")
    if not len(pairing):
        raise SystemExit("ERROR: no MECSLEEP<nn> participants matched. Check --data points at the "
                         "extracted folder (it should contain acc/*.bin and psg/*.txt).")
    info_path = next(data_dir.rglob("*participants*info*.csv"), None)
    info = pd.read_csv(info_path) if info_path else pd.DataFrame()
    rows, meta, skipped = [], [], []
    for r in pairing.itertuples(index=False):
        sid = f"mec{r.pid:02d}"
        if r.bin is None:
            skipped.append((sid, "no .bin for this wrist")); continue
        try:
            psg = parse_psg(r.psg)
        except Exception as exc:
            skipped.append((sid, f"PSG: {exc}")); continue
        t0 = time.time()
        rec = read_bin(r.bin, max_pages=max_pages)
        scale, offset, n_static, err_before, err_after = autocalibrate(rec.xyz, rec.sample_rate)
        if n_static < 10:
            scale, offset = np.ones(3), np.zeros(3)
        ep = to_epochs(rec, EPOCH_S, scale, offset)
        ep["step"] = np.arange(len(ep), dtype=np.int64)
        ep["series_id"] = sid
        ts = pd.to_datetime(ep.timestamp)
        ep["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S")
        ep[["series_id", "step", "timestamp", "anglez", "enmo"]].to_parquet(cache / f"{sid}.parquet",
                                                                           index=False)
        np.save(cache / f"{sid}_nonwear.npy", np.zeros(len(ep), bool))   # supervised lab night
        # One row per 15:00->15:00 window, exactly as in CMI (recordings may span 2-3 windows).
        # The PSG night labels whichever window contains the onset; other windows are context.
        nd = (ts - pd.Timedelta(hours=BOUNDARY_HOUR)).dt.strftime("%Y-%m-%d").to_numpy()
        tsv = ts.to_numpy()
        onset_step = int(np.searchsorted(tsv, np.datetime64(psg["onset"])))
        wake_step = int(np.searchsorted(tsv, np.datetime64(psg["wakeup"])))
        onset_in = 0 < onset_step < len(ep)
        # wake is censored when the recording stops before the PSG wake time
        wake_in = onset_in and onset_step < wake_step < len(ep)
        host = nd[onset_step] if onset_in else None
        for date in pd.unique(nd):
            is_host = onset_in and date == host
            status = ("labelled" if (is_host and wake_in) else
                      "partial" if is_host else "unannotated")
            rows.append(dict(series_id=sid, night_date=str(date), label_status=status,
                             onset_step=onset_step if is_host else None,
                             wakeup_step=wake_step if (is_host and wake_in) else None,
                             nonwear_fraction=0.0))
        if not onset_in:
            skipped.append((sid, "PSG onset outside the accelerometer recording"))
        elif not wake_in:
            print(f"    note: {sid} wake censored (recording ends before PSG wake)", flush=True)

        age = sex = diag = None
        if len(info):
            m = info[info.iloc[:, 0] == r.pid]
            if len(m):
                age, sex, diag = m.iloc[0].get("Age"), m.iloc[0].get("Sex"), m.iloc[0].get("Diagnosis")
        meta.append(dict(series_id=sid, pid=r.pid, age=age, sex=sex, diagnosis=diag,
                         psg_spt_min=psg["spt_min"], psg_tst_min=psg["tst_min"],
                         sleep_efficiency=100 * psg["tst_min"] / max(psg["spt_min"], 1e-9),
                         n_epochs=len(ep), hours=len(ep) * EPOCH_S / 3600,
                         calib_static_windows=n_static, calib_err_before=err_before,
                         calib_err_after=err_after, onset_observable=onset_in,
                         wake_observable=wake_in, n_windows=int(len(pd.unique(nd))),
                         bin_file=Path(r.bin).name, seconds=round(time.time() - t0, 1)))
        print(f"  {sid}: {len(ep):,} epochs, {len(ep)*EPOCH_S/3600:.1f} h, "
              f"{len(pd.unique(nd))} window(s), calib {err_before:.4f}->{err_after:.4f} g, "
              f"onset_in={onset_in} wake_in={wake_in} ({time.time()-t0:.0f}s)", flush=True)
    if not rows:
        raise SystemExit("ERROR: every recording was skipped; nothing written. See reasons above.")
    pd.DataFrame(rows).to_parquet(out / "night_windows.parquet", index=False)
    m = pd.DataFrame(meta); m.to_csv(out / "meta.csv", index=False)
    nw = pd.DataFrame(rows)
    print(f"\nprepared {nw.series_id.nunique()} recordings / {len(nw)} windows -> {out}")
    print(f"  windows by status: {nw.label_status.value_counts().to_dict()}")
    if skipped:
        print("skipped:"); [print(f"  {s}: {why}") for s, why in skipped]
    if len(m):
        print(f"\ncalibration error after: median {m.calib_err_after.median():.4f} g, "
              f"max {m.calib_err_after.max():.4f} g")
        print(f"PSG SPT: median {m.psg_spt_min.median():.0f} min; "
              f"sleep efficiency median {m.sleep_efficiency.median():.1f}%")


# ---------------------------------------------------------------------------------- stage 2 ----
def eval_models(out: Path, seed_dir: Path, cmi_out: Path, reps: int) -> None:
    from src.data.b1_features import load_series, window_tensor, bin_to_step
    from src.models.b1_net import B1Net
    from src.models.semimarkov import PairDecoder, FlagshipNet
    from scripts.run_p2 import MAX_DUR, MIN_DUR, gain_curve
    from scripts.run_p3 import conformal_quantile, hpd_score, hpd_width, _mass_above

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    win = pd.read_parquet(out / "night_windows.parquet").astype({"series_id": str, "night_date": str})
    lab = win[win.label_status == "labelled"].reset_index(drop=True)
    n_partial = int((win.label_status == "partial").sum())
    if n_partial:
        print(f"note: {n_partial} recording(s) have a censored wake time (accelerometer stops "
              f"before the PSG wake); they are excluded from this table.")
    meta = pd.read_csv(out / "meta.csv")
    feat = out / "feat_cache"; feat.mkdir(exist_ok=True)
    data = {s: load_series(out / "cache", feat, s) for s in lab.series_id}
    ck_files = sorted(seed_dir.glob("fold*_best.pt"))
    if not ck_files:
        raise SystemExit(f"no fold*_best.pt in {seed_dir}")
    print(f"device {device}; {len(ck_files)} fold models; {len(lab)} labelled recordings")

    post_by_model, rows = {}, []
    for ck_path in ck_files:
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model = FlagshipNet(B1Net(), PairDecoder(MAX_DUR, MIN_DUR)).to(device)
        model.load_state_dict(ck["state_dict"]); model.eval()
        stats = ck["stats"]                       # CMI training-fold statistics, never refit here
        res = {}
        with torch.no_grad():
            for r in lab.itertuples(index=False):
                s = data[r.series_id]
                w = int(np.flatnonzero(s.win_dates == r.night_date)[0])
                x, mask, _, n_samp = window_tensor(s, w, stats)
                X = torch.from_numpy(x[None]).to(device); M = torch.from_numpy(mask[None]).to(device)
                S, s_null, _ = model(X, M)
                p_null, m_on, m_wk, a, d, _ = model.decoder.posterior(S.float(), s_null.float())
                nb = int(mask.sum())
                res[r.series_id] = dict(w=w, n=nb, p_null=float(p_null[0]),
                                        m_on=m_on[0, :nb].cpu().numpy().astype(np.float64),
                                        m_wk=m_wk[0, :nb].cpu().numpy().astype(np.float64),
                                        a=int(a[0]), b=int(a[0] + d[0]))
        post_by_model[ck_path.stem] = res
        rows.append(score_posteriors(ck_path.stem, res, lab, data, gain_curve))
    # ensemble: average the per-bin marginals across the five fold models
    ens = {}
    for sid in lab.series_id:
        base = post_by_model[ck_files[0].stem][sid]
        n = base["n"]
        m_on = np.mean([post_by_model[k][sid]["m_on"][:n] for k in post_by_model], axis=0)
        m_wk = np.mean([post_by_model[k][sid]["m_wk"][:n] for k in post_by_model], axis=0)
        pn = float(np.mean([post_by_model[k][sid]["p_null"] for k in post_by_model]))
        g_on, g_wk = gain_curve(m_on), gain_curve(m_wk)
        a = int(np.argmax(g_on)); b = int(np.argmax(g_wk))
        if b <= a:
            b = min(n - 1, a + 1)
        ens[sid] = dict(w=base["w"], n=n, p_null=pn, m_on=m_on, m_wk=m_wk, a=a, b=b)
    rows.append(score_posteriors("ENSEMBLE", ens, lab, data, gain_curve))
    tab = pd.DataFrame(rows)
    print("\n=== Newcastle external validation (PSG SPT reference) ===")
    print(tab.round(2).to_string(index=False))
    print(f"\nacross the {len(tab)-1} fold models: onset MedAE {tab.iloc[:-1].onset_medae_min.mean():.1f} "
          f"+- {tab.iloc[:-1].onset_medae_min.std(ddof=1):.1f} min; "
          f"wake {tab.iloc[:-1].wake_medae_min.mean():.1f} "
          f"+- {tab.iloc[:-1].wake_medae_min.std(ddof=1):.1f} min")
    res_dir = out / "results"; res_dir.mkdir(exist_ok=True)
    tab.to_csv(res_dir / "models.csv", index=False)

    # ---- conformal transfer: do CMI-calibrated quantiles still cover here? --------------------
    z = dict(np.load(seed_dir / "oof_posteriors.npz"))
    cmi_win = pd.read_parquet(cmi_out / "night_windows.parquet").astype({"series_id": str, "night_date": str})
    from scripts.run_p3 import add_truth_bins, build_table, scores_for
    from src.data.b1_features import load_series as cmi_load
    cmi_data = {sid: cmi_load(cmi_out / "cache", cmi_out / "b1_cache", sid)
                for sid in sorted(set(map(str, z["series_id"])))}
    cmi_tab = scores_for(add_truth_bins(build_table(z, cmi_win), cmi_data), z, "hpd")
    cov_rows = []
    for alpha in (0.10, 0.05):
        q_on = conformal_quantile(cmi_tab.s_on.to_numpy(), alpha)
        q_wk = conformal_quantile(cmi_tab.s_wk.to_numpy(), alpha)
        for ev, q, key, pred in (("onset", q_on, "m_on", "a"), ("wakeup", q_wk, "m_wk", "b")):
            cov, wid = [], []
            for r in lab.itertuples(index=False):
                p = ens[r.series_id]
                s = data[r.series_id]
                truth_step = int(r.onset_step if ev == "onset" else r.wakeup_step)
                start = int(s.win_start[p["w"]])
                t_bin = int(np.clip((int(np.searchsorted(s.steps, truth_step)) - start) // SPM,
                                    0, p["n"] - 1))
                cov.append(hpd_score(p[key], t_bin) <= q)
                wid.append(hpd_width(p[key], q))
            cov_rows.append(dict(alpha=alpha, event=ev, target=1 - alpha, cmi_quantile=q,
                                 newcastle_coverage=float(np.mean(cov)), n=len(cov),
                                 median_width_min=float(np.median(wid))))
    cv = pd.DataFrame(cov_rows)
    print("\n=== conformal transfer (quantiles calibrated on CMI children, applied to Newcastle adults) ===")
    print(cv.round(3).to_string(index=False))
    cv.to_csv(res_dir / "conformal_transfer.csv", index=False)

    # ---- context: the GGIR reference heuristic on the SAME recordings -------------------------
    # Without this the reader cannot tell "our model transfers poorly" from "wrist accelerometry
    # disagrees with PSG by this much, whatever the method".
    from src.baselines.hdcza import WindowInput, rolling_signal, hdcza_window, DEFAULT
    import pyarrow.parquet as pq
    h_rows = []
    for r in lab.itertuples(index=False):
        s_ = data[r.series_id]
        w = int(np.flatnonzero(s_.win_dates == r.night_date)[0])
        a0, b0 = int(s_.win_start[w]), int(s_.win_end[w])
        tab = pq.read_table(out / "cache" / f"{r.series_id}.parquet",
                            columns=["step", "anglez"]).to_pandas()
        ang = tab.anglez.to_numpy()[a0:b0]
        wi = WindowInput(s_.steps[a0:b0], rolling_signal(ang), np.ones(b0 - a0, bool))
        res = hdcza_window(wi, **DEFAULT)
        if res is None:
            continue
        h_rows.append(dict(series_id=r.series_id,
                           onset_err_min=(res[0] - int(r.onset_step)) * EPOCH_S / 60,
                           wake_err_min=(res[1] - int(r.wakeup_step)) * EPOCH_S / 60,
                           spt_pred_min=(res[1] - res[0]) * EPOCH_S / 60,
                           spt_true_min=(int(r.wakeup_step) - int(r.onset_step)) * EPOCH_S / 60))
    # Fairness: HDCZA defaults are designed for multi-day free-living data, where the percentile
    # threshold sees plenty of daytime activity. These windows are mostly night, so we also tune
    # HDCZA ON NEWCASTLE leave-one-subject-out - giving the baseline in-cohort data that our
    # zero-shot model never sees.
    from src.baselines.hdcza import hdcza_window_grid, GRID
    grid_cache, truth_bins = {}, {}
    for r in lab.itertuples(index=False):
        s_ = data[r.series_id]
        w = int(np.flatnonzero(s_.win_dates == r.night_date)[0])
        a0, b0 = int(s_.win_start[w]), int(s_.win_end[w])
        tab = pq.read_table(out / "cache" / f"{r.series_id}.parquet",
                            columns=["step", "anglez"]).to_pandas()
        wi = WindowInput(s_.steps[a0:b0], rolling_signal(tab.anglez.to_numpy()[a0:b0]),
                         np.ones(b0 - a0, bool))
        grid_cache[r.series_id] = hdcza_window_grid(wi)
        truth_bins[r.series_id] = (int(r.onset_step), int(r.wakeup_step))
    combos = list(grid_cache[lab.series_id.iloc[0]])

    def objective(sids, combo):
        e = [(abs(grid_cache[s][combo][0] - truth_bins[s][0]) * EPOCH_S / 60,
              abs(grid_cache[s][combo][1] - truth_bins[s][1]) * EPOCH_S / 60)
             for s in sids if grid_cache[s][combo] is not None]
        if len(e) < 0.9 * len(sids):
            return np.inf
        a_, b_ = zip(*e)
        return (np.median(a_) + np.median(b_)) / 2

    t_rows = []
    for r in lab.itertuples(index=False):
        others = [s for s in lab.series_id if s != r.series_id]
        best = min(combos, key=lambda c: objective(others, c))
        res = grid_cache[r.series_id][best]
        if res is None:
            continue
        t_rows.append(dict(series_id=r.series_id, combo=str(best),
                           onset_err_min=(res[0] - truth_bins[r.series_id][0]) * EPOCH_S / 60,
                           wake_err_min=(res[1] - truth_bins[r.series_id][1]) * EPOCH_S / 60,
                           spt_pred_min=(res[1] - res[0]) * EPOCH_S / 60,
                           spt_true_min=(truth_bins[r.series_id][1] - truth_bins[r.series_id][0]) * EPOCH_S / 60))
    ht = pd.DataFrame(t_rows)
    if len(ht):
        hba2 = _ba(ht.spt_true_min.to_numpy(), ht.spt_pred_min.to_numpy())
        print(f"\n=== HDCZA TUNED on Newcastle (leave-one-subject-out), vs PSG ===")
        print(f"  n={len(ht)} | onset MedAE {ht.onset_err_min.abs().median():.1f} min "
              f"(bias {ht.onset_err_min.mean():+.1f}) | wake MedAE {ht.wake_err_min.abs().median():.1f} min "
              f"(bias {ht.wake_err_min.mean():+.1f})")
        print(f"  within 30 min: onset {100*(ht.onset_err_min.abs()<=30).mean():.0f}%, "
              f"wake {100*(ht.wake_err_min.abs()<=30).mean():.0f}%")
        print(f"  SPT bias {hba2['bias']:+.1f} min, LoA {hba2['loa_low']:.1f} to {hba2['loa_high']:.1f}, "
              f"ICC(2,1) {icc2_1(ht.spt_true_min, ht.spt_pred_min):.3f}")
        print(f"  parameters chosen (P_M_Bmin_G): {pd.Series(ht.combo).value_counts().to_dict()}")
        ht.to_csv(res_dir / "hdcza_newcastle_tuned.csv", index=False)

    hd = pd.DataFrame(h_rows)
    if len(hd):
        hba = _ba(hd.spt_true_min.to_numpy(), hd.spt_pred_min.to_numpy())
        print(f"\n=== HDCZA (GGIR default) on the same Newcastle recordings, vs PSG ===")
        print(f"  n={len(hd)} | onset MedAE {hd.onset_err_min.abs().median():.1f} min "
              f"(bias {hd.onset_err_min.mean():+.1f}) | wake MedAE {hd.wake_err_min.abs().median():.1f} min "
              f"(bias {hd.wake_err_min.mean():+.1f})")
        print(f"  within 30 min: onset {100*(hd.onset_err_min.abs()<=30).mean():.0f}%, "
              f"wake {100*(hd.wake_err_min.abs()<=30).mean():.0f}%")
        print(f"  SPT bias {hba['bias']:+.1f} min, LoA {hba['loa_low']:.1f} to {hba['loa_high']:.1f}, "
              f"ICC(2,1) {icc2_1(hd.spt_true_min, hd.spt_pred_min):.3f}")
        hd.to_csv(res_dir / "hdcza_newcastle.csv", index=False)

    # ---- conformal recalibrated on Newcastle itself (leave-one-subject-out) -------------------
    # The CMI-calibrated quantiles under-cover here; this shows what a handful of in-cohort
    # subjects buys back, which is the practical remedy for population shift.
    loso = []
    for alpha in (0.10, 0.05):
        for ev, key in (("onset", "m_on"), ("wakeup", "m_wk")):
            sc, cov, wid = {}, [], []
            for r in lab.itertuples(index=False):
                p = ens[r.series_id]; s_ = data[r.series_id]
                truth_step = int(r.onset_step if ev == "onset" else r.wakeup_step)
                start = int(s_.win_start[p["w"]])
                t_bin = int(np.clip((int(np.searchsorted(s_.steps, truth_step)) - start) // SPM,
                                    0, p["n"] - 1))
                sc[r.series_id] = (hpd_score(p[key], t_bin), p[key])
            for sid in sc:
                others = np.array([v[0] for k, v in sc.items() if k != sid])
                q = conformal_quantile(others, alpha)
                cov.append(sc[sid][0] <= q)
                wid.append(hpd_width(sc[sid][1], q) if np.isfinite(q) else np.nan)
            loso.append(dict(alpha=alpha, event=ev, target=1 - alpha, n=len(cov),
                             n_finite_q=int(np.sum(np.isfinite(wid))),
                             loso_coverage=float(np.mean(cov)),
                             median_width_min=float(np.nanmedian(wid))))
    lo = pd.DataFrame(loso)
    print(f"\n=== conformal recalibrated on Newcastle (leave-one-subject-out, n={len(lab)}) ===")
    print(lo.round(3).to_string(index=False))
    lo.to_csv(res_dir / "conformal_loso.csv", index=False)

    # ---- per-recording table for the paper ----------------------------------------------------
    per = []
    for r in lab.itertuples(index=False):
        p = ens[r.series_id]; s = data[r.series_id]
        st_on, st_wk = bin_to_step(s, p["w"], [p["a"], p["b"]])
        per.append(dict(series_id=r.series_id,
                        onset_err_min=(int(st_on) - int(r.onset_step)) * EPOCH_S / 60,
                        wake_err_min=(int(st_wk) - int(r.wakeup_step)) * EPOCH_S / 60,
                        spt_pred_min=(int(st_wk) - int(st_on)) * EPOCH_S / 60,
                        spt_true_min=(int(r.wakeup_step) - int(r.onset_step)) * EPOCH_S / 60,
                        p_null=p["p_null"]))
    per = pd.DataFrame(per).merge(meta, on="series_id", how="left")
    per.to_csv(res_dir / "per_recording.csv", index=False)
    ba = _ba(per.spt_true_min.to_numpy(), per.spt_pred_min.to_numpy())
    print(f"\nSPT Bland-Altman (ensemble): bias {ba['bias']:.1f} min, "
          f"LoA {ba['loa_low']:.1f} to {ba['loa_high']:.1f}, slope {ba['slope']:.3f} "
          f"(p={ba['slope_p']:.3g}); ICC(2,1) {icc2_1(per.spt_true_min, per.spt_pred_min):.3f}")
    if "sleep_efficiency" in per:
        lo = per[per.sleep_efficiency < per.sleep_efficiency.median()]
        hi = per[per.sleep_efficiency >= per.sleep_efficiency.median()]
        print(f"onset |err| median: fragmented sleep {lo.onset_err_min.abs().median():.1f} min "
              f"vs consolidated {hi.onset_err_min.abs().median():.1f} min")


def score_posteriors(name, post, lab, data, gain_curve):
    from src.data.b1_features import bin_to_step
    e_on, e_wk = [], []
    for r in lab.itertuples(index=False):
        p = post[r.series_id]; s = data[r.series_id]
        st_on, st_wk = bin_to_step(s, p["w"], [p["a"], p["b"]])
        e_on.append((int(st_on) - int(r.onset_step)) * EPOCH_S / 60)
        e_wk.append((int(st_wk) - int(r.wakeup_step)) * EPOCH_S / 60)
    e_on, e_wk = np.array(e_on), np.array(e_wk)
    return dict(model=name, n=len(e_on),
                onset_medae_min=float(np.median(np.abs(e_on))),
                wake_medae_min=float(np.median(np.abs(e_wk))),
                onset_bias_min=float(np.mean(e_on)), wake_bias_min=float(np.mean(e_wk)),
                onset_within30_pct=float(100 * np.mean(np.abs(e_on) <= 30)),
                wake_within30_pct=float(100 * np.mean(np.abs(e_wk) <= 30)),
                mean_p_null=float(np.mean([post[s]["p_null"] for s in lab.series_id])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--data", help="extracted Newcastle folder (required for --prep)")
    ap.add_argument("--seed-dir", default="outputs/p2_main/seed42")
    ap.add_argument("--wrist", default="left", choices=["left", "right"])
    ap.add_argument("--max-pages", type=int, default=None, help="debug: decode only N pages")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    cmi_out = Path(load_config(a.config)["paths"]["output_dir"])
    out = cmi_out / "newcastle"
    out.mkdir(parents=True, exist_ok=True)
    if a.prep:
        if not a.data:
            raise SystemExit("--prep needs --data pointing at the extracted Newcastle folder")
        prep(Path(a.data), out, a.wrist, a.max_pages)
    if a.eval:
        eval_models(out, Path(a.seed_dir), cmi_out, a.reps)
    if not (a.prep or a.eval):
        raise SystemExit("choose --prep and/or --eval")


if __name__ == "__main__":
    main()
