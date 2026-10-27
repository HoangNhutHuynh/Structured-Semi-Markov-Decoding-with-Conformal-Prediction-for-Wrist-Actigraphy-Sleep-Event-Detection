# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""STEP C: B0 HDCZA (default + per-fold tuned) -> OOF predictions, metrics, bootstrap, plots.

Usage (repo root, after Step A/B):  python -m scripts.run_b0 --config configs/phase1.yaml
Optional: --reps 1000 (bootstrap resamples), --skip-plots
"""
from __future__ import annotations
import argparse
import os
import itertools
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.common import load_config
from src.data.clock import load_clock_index, add_clock_columns
from src.baselines.hdcza import (DEFAULT, GRID, WindowInput, rolling_signal, hdcza_window,
                                 hdcza_window_grid)
from src.metrics.event_ap import truth_from_event_nights, event_detection_ap
from src.metrics.bootstrap import bootstrap_oof, per_fold_official_edap, per_fold_edap_summary

BOUNDARY_HOUR = 15
# numpy>=2 / Python 3.14 emits a harmless 'generic timedelta unit' warning from pandas internals
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*generic.*unit.*")
COMBOS = list(itertools.product(GRID["P"], GRID["M"], GRID["Bmin"], GRID["G"]))


def log(msg: str, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def load_series_arrays(cache: Path, sid: str, b0_cache: Path) -> dict:
    """steps, r(t), worn, night_date codes for one series (cached as npz)."""
    f = b0_cache / f"{sid}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    tab = pq.read_table(cache / f"{sid}.parquet", columns=["step", "timestamp", "anglez"]).to_pandas()
    tab = tab.sort_values("step", kind="stable").reset_index(drop=True)
    nonwear = np.load(cache / f"{sid}_nonwear.npy")
    if len(nonwear) != len(tab):
        raise ValueError(f"{sid}: nonwear length {len(nonwear)} != {len(tab)}")
    local = pd.to_datetime(tab.timestamp.astype(str).str[:19], format="%Y-%m-%dT%H:%M:%S")
    night = (local - pd.Timedelta(hours=BOUNDARY_HOUR)).dt.strftime("%Y-%m-%d").to_numpy()
    out = {"steps": tab.step.to_numpy(np.int64), "r": rolling_signal(tab.anglez.to_numpy()),
           "worn": ~nonwear.astype(bool), "anglez": tab.anglez.to_numpy(np.float32),
           "night_date": night.astype(str)}
    np.savez(f, **out)
    return out


def window_inputs(arr: dict) -> dict[str, tuple[WindowInput, slice]]:
    nd = arr["night_date"]
    change = np.r_[0, np.flatnonzero(nd[1:] != nd[:-1]) + 1, len(nd)]
    res = {}
    for a, b in zip(change[:-1], change[1:]):
        res[str(nd[a])] = (WindowInput(arr["steps"][a:b], arr["r"][a:b], arr["worn"][a:b]), slice(a, b))
    return res


def to_tables(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pw = pd.DataFrame(rows, columns=["series_id", "night_date", "onset_step", "wakeup_step", "score"])
    ev = pd.concat([pw.assign(event="onset", step=pw.onset_step),
                    pw.assign(event="wakeup", step=pw.wakeup_step)], ignore_index=True)
    ev = ev[["series_id", "night_date", "event", "step", "score"]].sort_values(
        ["series_id", "step"]).reset_index(drop=True)
    return pw[["series_id", "night_date", "onset_step", "wakeup_step"]], ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--skip-plots", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(cfg["paths"]["output_dir"]); cache = Path(cfg["paths"]["cache_dir"])
    b0_cache = out / "b0_cache"; b0_cache.mkdir(parents=True, exist_ok=True)
    (out / "b0_default").mkdir(exist_ok=True); (out / "b0_tuned").mkdir(exist_ok=True)
    fh = open(out / "b0_runtime.log", "a", encoding="utf-8")
    t0 = time.time()

    windows = pd.read_parquet(out / "night_windows.parquet")
    windows["series_id"] = windows.series_id.astype(str)
    windows["night_date"] = windows.night_date.astype(str)
    clock = load_clock_index(out / "clock_index.json")
    expected = int(os.environ.get("HHN_EXPECTED_SERIES", 277))  # override only for smoke tests
    assert len(clock) == expected == windows.series_id.nunique(), \
        f"clock index {len(clock)} vs windows {windows.series_id.nunique()} (expected {expected})"
    folds = json.loads((out / "splits.json").read_text(encoding="utf-8"))["folds"]
    event_nights = pd.read_parquet(out / "event_nights.parquet")
    truth = truth_from_event_nights(event_nights)
    truth["series_id"] = truth.series_id.astype(str)

    # ---- 1) HDCZA for every window x every grid combo (+ default) --------------------------
    grid_file = out / "b0_cache" / "grid_predictions.parquet"
    if grid_file.exists():
        grid = pd.read_parquet(grid_file)
        log(f"loaded cached grid predictions ({len(grid):,} rows)", fh)
    else:
        rows = []
        sids = sorted(windows.series_id.unique())
        for i, sid in enumerate(sids, 1):
            arr = load_series_arrays(cache, sid, b0_cache)
            wins = window_inputs(arr)
            expected = set(windows.loc[windows.series_id == sid, "night_date"])
            if set(wins) != expected:
                raise AssertionError(f"{sid}: window dates differ from night_windows.parquet "
                                     f"({len(set(wins) ^ expected)} mismatches)")
            for nd, (w, _) in wins.items():
                res = hdcza_window_grid(w)
                d = hdcza_window(w, **DEFAULT)
                res[("D", "D", "D", "D")] = d
                for combo, val in res.items():
                    if val is not None:
                        rows.append((sid, nd, "_".join(map(str, combo)), *val))
            if i % 25 == 0:
                log(f"HDCZA grid: {i}/{len(sids)} series", fh)
        grid = pd.DataFrame(rows, columns=["series_id", "night_date", "combo", "onset_step",
                                           "wakeup_step", "score"])
        grid.to_parquet(grid_file, index=False)
        log(f"grid done: {len(grid):,} rows", fh)

    # ---- 2) tuning per fold on train+val series ------------------------------------------
    lab = windows.loc[windows.label_status == "labelled",
                      ["series_id", "night_date", "onset_step", "wakeup_step"]]
    g = grid.merge(lab, on=["series_id", "night_date"], how="inner", suffixes=("", "_true"))
    g["onset_err"] = (g.onset_step - g.onset_step_true).abs() * 5 / 60
    g["wakeup_err"] = (g.wakeup_step - g.wakeup_step_true).abs() * 5 / 60
    # Recall denominator = labelled windows that pass the >=50%-worn rule (no combination can
    # predict on the others, so they would cap recall below the eligibility bar for every combo).
    lab_eval = windows.loc[(windows.label_status == "labelled") & (windows.nonwear_fraction <= 0.5)]
    n_lab_by_series = lab_eval.groupby("series_id").size()
    n_lab_all_by_series = lab.groupby("series_id").size()
    chosen, grid_rows = {}, []
    for fold, parts in sorted(folds.items(), key=lambda kv: int(kv[0])):
        tune_ids = set(map(str, parts["train"])) | set(map(str, parts["val"]))
        n_lab = int(n_lab_by_series.reindex(list(tune_ids)).fillna(0).sum())
        gs = g[g.series_id.isin(tune_ids)]
        stats = gs.groupby("combo").agg(n=("onset_err", "size"), on=("onset_err", "median"),
                                        wk=("wakeup_err", "median")).reset_index()
        stats["recall"] = stats.n / n_lab
        stats["objective"] = (stats.on + stats.wk) / 2
        stats["eligible"] = stats.recall >= 0.9
        n_all = int(n_lab_all_by_series.reindex(list(tune_ids)).fillna(0).sum())
        d = stats[stats.combo == "D_D_D_D"]
        log(f"fold {fold}: labelled tuning windows {n_all}, wear-eligible {n_lab}; "
            f"default recall {float(d.recall.iloc[0]) if len(d) else float('nan'):.3f}; "
            f"grid recall min/median/max {stats.recall.min():.3f}/{stats.recall.median():.3f}/"
            f"{stats.recall.max():.3f}", fh)
        stats = stats[stats.combo != "D_D_D_D"]
        stats["fold"] = int(fold)
        el = stats[stats.eligible]
        if el.empty:
            # Documented fallback: combos within 0.02 of the best achievable recall.
            el = stats[stats.recall >= stats.recall.max() - 0.02]
            log(f"fold {fold}: WARNING no combo reaches recall 0.9; fallback to recall >= "
                f"{stats.recall.max() - 0.02:.3f} ({len(el)} combos)", fh)
        best = el[el.objective == el.objective.min()]
        if len(best) > 1:  # tie-break: higher official EDAP on the tuning series
            tt = truth[truth.series_id.isin(tune_ids)]
            edaps = {}
            for c in best.combo:
                _, ev = to_tables(grid[(grid.combo == c) & grid.series_id.isin(tune_ids)]
                                  .drop(columns="combo").to_dict("records"))
                edaps[c] = event_detection_ap(tt, ev[["series_id", "step", "event", "score"]])
            stats["tie_edap"] = stats.combo.map(edaps)
            pick = max(edaps, key=edaps.get)
        else:
            pick = best.combo.iloc[0]
        chosen[fold] = pick
        grid_rows.append(stats)
        log(f"fold {fold}: chosen {pick} (P_M_Bmin_G); objective "
            f"{float(el.loc[el.combo == pick, 'objective'].iloc[0]):.2f} min", fh)
    pd.concat(grid_rows).to_csv(out / "b0_tuned" / "grid_results.csv", index=False)
    params = pd.DataFrame([{"fold": int(f), **dict(zip(["P", "M", "Bmin_min", "G_min"],
                                                        map(int, c.split("_"))))}
                           for f, c in chosen.items()])
    params.to_csv(out / "b0_tuned" / "chosen_params.csv", index=False)

    # ---- 3) OOF tables -------------------------------------------------------------------
    test_fold = {str(s): f for f, p in folds.items() for s in p["test"]}
    variants = {"b0_default": grid[grid.combo == "D_D_D_D"],
                "b0_tuned": grid[grid.combo == grid.series_id.map(lambda s: chosen[test_fold[s]])]}
    win_eval = add_clock_columns(windows[["series_id", "night_date", "label_status", "onset_step",
                                          "wakeup_step", "nonwear_fraction"]], clock)
    summary = {}
    for name, sub in variants.items():
        pw, ev = to_tables(sub.drop(columns="combo").to_dict("records"))
        ev.to_parquet(out / name / "oof_predictions.parquet", index=False)
        pw.to_parquet(out / name / "oof_predicted_windows.parquet", index=False)
        pw_eval = add_clock_columns(pw, clock)
        t1 = time.time()
        boot = bootstrap_oof(truth, ev, win_eval, pw_eval, repetitions=a.reps, seed=0)
        boot.to_csv(out / name / "bootstrap.csv", index=False)
        pf = per_fold_official_edap(truth, ev, folds)
        pf.to_csv(out / name / "per_fold_edap.csv", index=False)
        metrics = dict(zip(boot.metric, boot.estimate))
        metrics.update(per_fold_edap_summary(pf))
        (out / name / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float),
                                                  encoding="utf-8")
        summary[name] = boot.set_index("metric")
        log(f"{name}: official EDAP {metrics['official_edap']:.4f}; bootstrap {time.time()-t1:.0f} s", fh)

    # ---- 4) report table -----------------------------------------------------------------
    def cell(b, m, fmt="{:.3f}"):
        r = b.loc[m]
        ci = "" if np.isnan(r.ci_low) else f" [{fmt.format(r.ci_low)}, {fmt.format(r.ci_high)}]"
        return fmt.format(r.estimate) + ci
    rows = []
    for name, b in summary.items():
        rows.append({
            "method": name,
            "official_EDAP": cell(b, "official_edap"),
            "EDAP_K1": cell(b, "anti_edap_k1"), "EDAP_K3": cell(b, "anti_edap_k3"),
            "onset_MedAE_min": cell(b, "onset_medae_min", "{:.1f}"),
            "onset_IQR": f"{b.loc['onset_q1_min'].estimate:.1f}-{b.loc['onset_q3_min'].estimate:.1f}",
            "wake_MedAE_min": cell(b, "wakeup_medae_min", "{:.1f}"),
            "wake_IQR": f"{b.loc['wakeup_q1_min'].estimate:.1f}-{b.loc['wakeup_q3_min'].estimate:.1f}",
            "onset_pct15/30": f"{b.loc['onset_within15_pct'].estimate:.1f}/{b.loc['onset_within30_pct'].estimate:.1f}",
            "wake_pct15/30": f"{b.loc['wakeup_within15_pct'].estimate:.1f}/{b.loc['wakeup_within30_pct'].estimate:.1f}",
            "recall_labelled": cell(b, "window_recall_labelled"),
            "proxyFD_unlab": cell(b, "proxy_false_detection_rate_unlabelled"),
            "proxyFD_unann": cell(b, "proxy_false_detection_rate_unannotated"),
            "SPT_bias_min": cell(b, "ba_spt_bias", "{:.1f}"),
            "SPT_LoA": f"{b.loc['ba_spt_loa_low'].estimate:.1f} to {b.loc['ba_spt_loa_high'].estimate:.1f}",
            "ICC21_SPT": cell(b, "icc2_1_spt"),
        })
    table = pd.DataFrame(rows)
    table.to_csv(out / "b0_results_table.csv", index=False)
    try:
        md = table.T.to_markdown()
    except ImportError:  # tabulate not installed
        md = table.T.to_string()
    (out / "b0_results_table.md").write_text(md, encoding="utf-8")
    print("\n" + table.T.to_string() + "\n")
    print(params.to_string(index=False))

    # ---- 5) example nights ----------------------------------------------------------------
    if not a.skip_plots:
        try:
            plot_examples(out, cache, b0_cache, windows, variants["b0_tuned"], chosen, test_fold)
        except Exception as exc:  # plots must never invalidate the metrics
            log(f"plotting failed: {exc!r}", fh)
    log(f"STEP C finished in {(time.time()-t0)/60:.1f} min", fh)


def plot_examples(out, cache, b0_cache, windows, tuned, chosen, test_fold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = windows[windows.label_status == "labelled"]
    m = tuned.merge(lab[["series_id", "night_date", "onset_step", "wakeup_step"]],
                    on=["series_id", "night_date"], suffixes=("", "_true"))
    m["err"] = (m.onset_step - m.onset_step_true).abs() * 5 / 60
    m = m.sort_values("err").reset_index(drop=True)
    picks = {"best": m.iloc[0], "median": m.iloc[len(m) // 2], "worst": m.iloc[-1]}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for ax, (tag, row) in zip(axes, picks.items()):
        arr = load_series_arrays(cache, row.series_id, b0_cache)
        sel = arr["night_date"] == row.night_date
        steps, ang, r, worn = arr["steps"][sel], arr["anglez"][sel], arr["r"][sel], arr["worn"][sel]
        hrs = (steps - steps[0]) / 720 + 15
        P, M, Bmin, G = map(int, chosen[test_fold[row.series_id]].split("_"))
        thr = np.percentile(r[worn], P) * M
        ax.plot(hrs, ang, lw=0.4, color="0.6", label="anglez")
        ax2 = ax.twinx(); ax2.plot(hrs, r, lw=0.6, color="tab:blue", label="r(t)")
        ax2.axhline(thr, color="tab:blue", ls=":", lw=1, label="threshold")
        ax2.set_yscale("symlog", linthresh=0.1)
        ax.fill_between(hrs, -90, 90, where=~worn, color="tab:red", alpha=0.12, label="non-wear")
        for s, c, lab_ in ((row.onset_step_true, "k", "truth"), (row.wakeup_step_true, "k", None),
                           (row.onset_step, "tab:orange", "HDCZA"), (row.wakeup_step, "tab:orange", None)):
            ax.axvline((s - steps[0]) / 720 + 15, color=c, ls="--" if c == "k" else "-", lw=1.2, label=lab_)
        ax.set_xlim(15, 39); ax.set_ylim(-90, 90)
        ax.set_title(f"{tag}: {row.series_id} night {row.night_date} — onset error {row.err:.1f} min "
                     f"(P={P}, M={M}, Bmin={Bmin}, G={G})", fontsize=9)
        ax.set_xlabel("hours from 15:00 (recorded wall clock)"); ax.set_ylabel("anglez (deg)")
        ax2.set_ylabel("r(t) (deg)")
        if tag == "best":
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left", ncol=6)
    fig.savefig(out / "b0_tuned" / "example_nights.png", dpi=150)


if __name__ == "__main__":
    main()
