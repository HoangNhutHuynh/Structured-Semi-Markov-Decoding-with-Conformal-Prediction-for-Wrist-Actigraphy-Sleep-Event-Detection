# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""STEP D: B1 Kaggle-style deep baseline (CNN + BiGRU heatmaps), design spec v1.1.

Usage (repo root):
  python -m scripts.run_b1 --config configs/phase1.yaml                 # 3 seeds x 5 folds
  python -m scripts.run_b1 --config configs/phase1.yaml --seeds 42 --folds 1 --epochs 2   # smoke
  python -m scripts.run_b1 --config configs/phase1.yaml --eval-only     # re-evaluate saved OOF
Training uses labelled windows only (cross_boundary/partial/unlabelled/unannotated masked out).
Epoch selection: best inner-validation official EDAP (Kaggle-mode post-processing).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from torch import nn

from src.data.common import load_config
from src.data.clock import load_clock_index, add_clock_columns
from src.data.b1_features import (SeriesData, load_series, norm_stats, window_tensor, targets,
                                  bin_to_step, M_BINS)
from src.models.b1_net import B1Net, n_params
from src.metrics.event_ap import truth_from_event_nights, event_detection_ap
from src.metrics.bootstrap import bootstrap_oof, per_fold_official_edap, per_fold_edap_summary

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*generic.*unit.*")
COLS = ["series_id", "step", "event", "score"]


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def seed_everything(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------------------------
class TrainSet(torch.utils.data.Dataset):
    def __init__(self, items, data: dict[str, SeriesData], stats):
        self.items, self.data, self.stats = items, data, stats

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, w, on, wk = self.items[i]
        s = self.data[sid]
        x, mask, _, _ = window_tensor(s, w, self.stats)
        y = targets(s, w, on, wk)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def labelled_items(series_ids, data, windows):
    lab = windows[windows.label_status == "labelled"]
    lab = lab[lab.series_id.isin(series_ids)]
    items, dropped = [], 0
    for r in lab.itertuples(index=False):
        s = data[r.series_id]
        hit = np.flatnonzero(s.win_dates == r.night_date)
        if len(hit) == 0 or targets(s, int(hit[0]), r.onset_step, r.wakeup_step) is None:
            dropped += 1
            continue
        items.append((r.series_id, int(hit[0]), int(r.onset_step), int(r.wakeup_step)))
    return items, dropped


@torch.no_grad()
def infer(model, series_ids, data, stats, device, batch=32, amp=True):
    """{(sid, night_date): (probs (3,M), mask (M,), window index)} for every window."""
    model.eval()
    out, buf = {}, []

    def flush():
        xs = torch.from_numpy(np.stack([b[2] for b in buf])).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            p = torch.sigmoid(model(xs).float()).cpu().numpy()
        for (sid, w, _, mask), pi in zip(buf, p):
            out[(sid, str(data[sid].win_dates[w]))] = (pi, mask, w)
        buf.clear()

    for sid in series_ids:
        s = data[sid]
        for w in range(len(s.win_dates)):
            x, mask, _, _ = window_tensor(s, w, stats)
            buf.append((sid, w, x, mask))
            if len(buf) == batch:
                flush()
    if buf:
        flush()
    return out


def postprocess(probs: dict, data, mode: str):
    """Kaggle mode -> event rows; one-pair mode -> (event rows, predicted windows)."""
    ev, pw = [], []
    for (sid, nd), (p, mask, w) in probs.items():
        s = data[sid]
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            continue
        last = valid[-1] + 1
        if mode == "kaggle":
            for k, event in ((0, "onset"), (1, "wakeup")):
                peaks, props = find_peaks(p[k, :last], height=0.01, distance=6)
                if len(peaks) == 0:
                    continue
                top = np.argsort(-props["peak_heights"])[:20]
                steps = bin_to_step(s, w, peaks[top])
                for st, sc in zip(steps, props["peak_heights"][top]):
                    ev.append((sid, nd, event, int(st), float(sc)))
        else:
            on = int(np.argmax(p[0, :last]))
            if on + 30 >= last:
                continue
            wk = on + 30 + int(np.argmax(p[1, on + 30:last]))
            sc = float(math.sqrt(p[0, on] * p[1, wk]))
            st_on, st_wk = bin_to_step(s, w, [on, wk])
            ev += [(sid, nd, "onset", int(st_on), sc), (sid, nd, "wakeup", int(st_wk), sc)]
            pw.append((sid, nd, int(st_on), int(st_wk)))
    ev = pd.DataFrame(ev, columns=["series_id", "night_date", "event", "step", "score"])
    pw = pd.DataFrame(pw, columns=["series_id", "night_date", "onset_step", "wakeup_step"])
    return ev, pw


def masked_bce(logits, y, mask):
    m = mask.unsqueeze(1).float()
    bce = nn.functional.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
    w = torch.tensor([1.0, 1.0, 0.5], device=logits.device).view(1, 3, 1)
    return (bce * w * m).sum() / (m.sum() * 3).clamp(min=1)


def train_fold(fold, parts, data, windows, truth, args, device, seed, fh, out_dir):
    seed_everything(seed + int(fold))
    train_ids = [str(x) for x in parts["train"]]
    val_ids = [str(x) for x in parts["val"]]
    stats = norm_stats([data[s] for s in train_ids])
    tr_items, dropped = labelled_items(train_ids, data, windows)
    log(f"seed {seed} fold {fold}: train windows {len(tr_items)} (dropped {dropped}), "
        f"val series {len(val_ids)}", fh)
    g = torch.Generator(); g.manual_seed(seed + int(fold))
    loader = torch.utils.data.DataLoader(TrainSet(tr_items, data, stats), batch_size=args.batch,
                                         shuffle=True, num_workers=args.workers, generator=g,
                                         drop_last=False, pin_memory=device.type == "cuda")
    model = B1Net().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    total = args.epochs * len(loader); warm = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    val_truth = truth[truth.series_id.isin(val_ids)]
    best, best_state, rows = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); losses = []
        for x, y, m in loader:
            x, y, m = x.to(device, non_blocking=True), y.to(device), m.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(x)
            loss = masked_bce(logits, y, m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach()))
        probs = infer(model, val_ids, data, stats, device)
        ev, _ = postprocess(probs, data, "kaggle")
        val_edap = (event_detection_ap(val_truth, ev[COLS]) if len(ev) and len(val_truth) else 0.0)
        if not np.isfinite(val_edap):
            val_edap = 0.0
        rows.append({"fold": int(fold), "epoch": epoch, "train_loss": float(np.mean(losses)),
                     "val_edap": val_edap, "sec": time.time() - t0})
        log(f"  fold {fold} epoch {epoch:02d} loss {np.mean(losses):.4f} val EDAP {val_edap:.4f} "
            f"({time.time()-t0:.0f}s)", fh)
        if val_edap > best or best_state is None:
            best = val_edap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "stats": stats, "epoch": best_epoch, "val_edap": best},
               out_dir / f"fold{fold}_best.pt")
    test_ids = [str(x) for x in parts["test"]]
    probs = infer(model, test_ids, data, stats, device)
    log(f"seed {seed} fold {fold}: best epoch {best_epoch} (val EDAP {best:.4f})", fh)
    return probs, rows, {"fold": int(fold), "best_epoch": best_epoch, "val_edap": best}


# ----------------------------------------------------------------------------------------------
def cell(b, m, fmt="{:.3f}"):
    r = b.loc[m]
    ci = "" if np.isnan(r.ci_low) else f" [{fmt.format(r.ci_low)}, {fmt.format(r.ci_high)}]"
    return fmt.format(r.estimate) + ci


def evaluate_seed(seed_dir, truth, windows, clock, folds, reps, fh):
    ev_k = pd.read_parquet(seed_dir / "oof_predictions_kaggle.parquet")
    ev_1 = pd.read_parquet(seed_dir / "oof_predictions_onepair.parquet")
    pw = pd.read_parquet(seed_dir / "oof_predicted_windows.parquet")
    win_eval = add_clock_columns(windows[["series_id", "night_date", "label_status", "onset_step",
                                          "wakeup_step", "nonwear_fraction"]], clock)
    boot = bootstrap_oof(truth, ev_k, win_eval, add_clock_columns(pw, clock),
                         repetitions=reps, seed=0)
    boot.to_csv(seed_dir / "bootstrap.csv", index=False)
    pf = per_fold_official_edap(truth, ev_k, folds); pf.to_csv(seed_dir / "per_fold_edap.csv", index=False)
    metrics = dict(zip(boot.metric, boot.estimate))
    metrics["onepair_official_edap"] = event_detection_ap(truth, ev_1[COLS])
    metrics.update(per_fold_edap_summary(pf))
    (seed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    log(f"{seed_dir.name}: EDAP kaggle {metrics['official_edap']:.4f}, one-pair "
        f"{metrics['onepair_official_edap']:.4f}", fh)
    return boot.set_index("metric"), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(cfg["paths"]["output_dir"]); cache = Path(cfg["paths"]["cache_dir"])
    b1_cache = out / "b1_cache"; b1_cache.mkdir(parents=True, exist_ok=True)
    root = out / "b1"; root.mkdir(exist_ok=True)
    fh = open(out / "b1_runtime.log", "a", encoding="utf-8")
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          (a.device if a.device != "auto" else "cpu"))
    log(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), fh)

    windows = pd.read_parquet(out / "night_windows.parquet").astype({"series_id": str, "night_date": str})
    clock = load_clock_index(out / "clock_index.json")
    expected = int(os.environ.get("HHN_EXPECTED_SERIES", 277))
    assert len(clock) == expected == windows.series_id.nunique()
    folds = json.loads((out / "splits.json").read_text(encoding="utf-8"))["folds"]
    truth = truth_from_event_nights(pd.read_parquet(out / "event_nights.parquet"))
    truth["series_id"] = truth.series_id.astype(str)
    run_folds = a.folds or sorted(folds, key=int)

    if not a.eval_only:
        t0 = time.time()
        data = {}
        for i, sid in enumerate(sorted(windows.series_id.unique()), 1):
            data[sid] = load_series(cache, b1_cache, sid)
            if i % 50 == 0:
                log(f"features: {i}/{windows.series_id.nunique()} series", fh)
        mism = sum(set(data[s].win_dates) != set(windows.loc[windows.series_id == s, "night_date"])
                   for s in data)
        assert mism == 0, f"{mism} series: window dates differ from night_windows.parquet"
        log(f"features ready ({time.time()-t0:.0f}s); model params {n_params(B1Net()):,}", fh)
        for seed in a.seeds:
            sd = root / f"seed{seed}"; sd.mkdir(exist_ok=True)
            all_probs, logs, best = {}, [], []
            for fold in run_folds:
                probs, rows, b = train_fold(fold, folds[fold], data, windows, truth, a, device,
                                            seed, fh, sd)
                all_probs.update(probs); logs += rows; best.append(b)
            pd.DataFrame(logs).to_csv(sd / "train_log.csv", index=False)
            pd.DataFrame(best).to_csv(sd / "best_epochs.csv", index=False)
            if len(run_folds) < len(folds):
                log(f"seed {seed}: only folds {run_folds} run -> OOF incomplete, skipping evaluation", fh)
                continue
            ev_k, _ = postprocess(all_probs, data, "kaggle")
            ev_1, pw = postprocess(all_probs, data, "onepair")
            ev_k.to_parquet(sd / "oof_predictions_kaggle.parquet", index=False)
            ev_1.to_parquet(sd / "oof_predictions_onepair.parquet", index=False)
            pw.to_parquet(sd / "oof_predicted_windows.parquet", index=False)
            if seed == a.seeds[0]:
                ev_k.to_parquet(out / "b1_oof_predictions.parquet", index=False)
        log(f"training finished in {(time.time()-t0)/60:.1f} min", fh)

    # ---- evaluation: bootstrap for the first seed, point estimates for all seeds ---------------
    results, points = {}, []
    for i, seed in enumerate(a.seeds):
        sd = root / f"seed{seed}"
        if not (sd / "oof_predictions_kaggle.parquet").exists():
            continue
        b, mtr = evaluate_seed(sd, truth, windows, clock, folds, a.reps if i == 0 else 1, fh)
        results[seed] = b; points.append({"seed": seed, **mtr})
    if not points:
        return
    pts = pd.DataFrame(points); pts.to_csv(root / "seed_point_estimates.csv", index=False)
    keys = ["official_edap", "anti_edap_k1", "anti_edap_k3", "onepair_official_edap",
            "onset_medae_min", "wakeup_medae_min", "window_recall_labelled",
            "proxy_false_detection_rate_unlabelled", "proxy_false_detection_rate_unannotated",
            "ba_spt_bias", "icc2_1_spt"]
    seedsum = pd.DataFrame({k: [pts[k].mean(), pts[k].std(ddof=1) if len(pts) > 1 else np.nan]
                            for k in keys}, index=["mean_over_seeds", "sd_over_seeds"]).T
    seedsum.to_csv(root / "seed_summary.csv")
    b = results[a.seeds[0]]
    row = {"method": f"B1 (seed {a.seeds[0]})",
           "official_EDAP": cell(b, "official_edap"), "EDAP_K1": cell(b, "anti_edap_k1"),
           "EDAP_K3": cell(b, "anti_edap_k3"),
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
           "ICC21_SPT": cell(b, "icc2_1_spt")}
    table = pd.DataFrame([row]); table.to_csv(out / "b1_results_table.csv", index=False)
    print("\n" + table.T.to_string() + "\n")
    print("Across seeds (point estimates):\n" + seedsum.round(4).to_string())


if __name__ == "__main__":
    main()
