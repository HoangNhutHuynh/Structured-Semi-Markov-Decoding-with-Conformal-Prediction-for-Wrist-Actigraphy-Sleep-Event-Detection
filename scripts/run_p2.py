# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""PHASE 2: flagship structured decoder (B1 backbone + exact one-pair-per-night decoder + null state).

Usage (repo root):
  python -m scripts.run_p2 --config configs/phase1.yaml --seeds 42 --folds 1 --epochs 2   # smoke
  python -m scripts.run_p2 --config configs/phase1.yaml --seeds 42                        # 1 seed
  python -m scripts.run_p2 --config configs/phase1.yaml --seeds 42 43 44 --eval-only
Options: --learn-kernel (learnable label-noise kernel), --no-null (ablation: no proxy-negative
training and never abstain), --tag NAME (output folder outputs/p2_NAME).
Outputs mirror B1 (outputs/p2_<tag>/seed*/...), so the same evaluation code is used.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from torch import nn

from src.data.common import load_config
from src.data.clock import load_clock_index
from src.data.b1_features import load_series, norm_stats, window_tensor, targets, bin_to_step, SPM
from src.models.b1_net import B1Net, n_params
from src.models.semimarkov import PairDecoder, FlagshipNet
from src.metrics.event_ap import truth_from_event_nights, event_detection_ap
from scripts.run_b1 import log, seed_everything, labelled_items, evaluate_seed, cell, COLS

MAX_DUR, MIN_DUR = 1200, 30


class P2Set(torch.utils.data.Dataset):
    """Labelled windows (a*, b* in bins) and proxy-negative windows (a* = -1)."""

    def __init__(self, items, data, stats):
        self.items, self.data, self.stats = items, data, stats

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, w, on, wk = self.items[i]
        s = self.data[sid]
        x, mask, start, n = window_tensor(s, w, self.stats)
        if on is None:
            a = b = -1; y = np.zeros((3,) + mask.shape, np.float32)
        else:
            a = (int(np.searchsorted(s.steps, on)) - start) // SPM
            b = (int(np.searchsorted(s.steps, wk)) - start) // SPM
            y = targets(s, w, on, wk)
        return (torch.from_numpy(x), torch.from_numpy(mask), torch.tensor(a), torch.tensor(b),
                torch.from_numpy(y))


def proxy_items(series_ids, data, windows):
    px = windows[windows.series_id.isin(series_ids) & windows.label_status.isin(["unlabelled", "unannotated"])
                 & (windows.nonwear_fraction >= 0.9)]
    items = []
    for r in px.itertuples(index=False):
        hit = np.flatnonzero(data[r.series_id].win_dates == r.night_date)
        if len(hit):
            items.append((r.series_id, int(hit[0]), None, None))
    return items


def duration_prior(items, data):
    d = [(int(np.searchsorted(data[s].steps, wk)) - int(np.searchsorted(data[s].steps, on))) // SPM
         for s, w, on, wk in items if on is not None]
    h = np.bincount(np.clip(d, 0, MAX_DUR - 1), minlength=MAX_DUR).astype(float)
    k = np.exp(-0.5 * (np.arange(-40, 41) / 10.0) ** 2)
    h = np.convolve(h, k / k.sum(), mode="same") + 1e-3
    return torch.tensor(np.log(h / h.sum()), dtype=torch.float32)


KERNEL_SIGMA = 1.5


def build_model(init_dur, learn_kernel):
    return FlagshipNet(B1Net(), PairDecoder(MAX_DUR, MIN_DUR, init_dur, KERNEL_SIGMA, learn_kernel))


AUX_HEAT = 0.0


def loss_fn(model, x, mask, a, b, y, use_null):
    S, s_null, s = model(x, mask)
    lab = a >= 0
    loss = torch.zeros((), device=x.device)
    if lab.any():
        loss = loss + model.decoder.nll_labelled(S[lab], s_null[lab], a[lab], b[lab]).mean()
        m = mask[lab].float()
        bce = nn.functional.binary_cross_entropy_with_logits(s[lab], y[lab, 2], reduction="none")
        loss = loss + 0.5 * (bce * m).sum() / m.sum().clamp(min=1)
        if AUX_HEAT > 0:                                  # B1-style per-bin heatmap supervision
            e_on, e_wk = model.last_heads
            h = (nn.functional.binary_cross_entropy_with_logits(e_on[lab], y[lab, 0], reduction="none") +
                 nn.functional.binary_cross_entropy_with_logits(e_wk[lab], y[lab, 1], reduction="none"))
            loss = loss + AUX_HEAT * (h * m).sum() / m.sum().clamp(min=1)
    if use_null and (~lab).any():
        loss = loss + model.decoder.nll_null(S[~lab], s_null[~lab]).mean()
    return loss


@torch.no_grad()
def infer(model, series_ids, data, stats, device, batch=8):
    model.eval()
    out, buf = {}, []

    def flush():
        x = torch.from_numpy(np.stack([b_[2] for b_ in buf])).to(device)
        m = torch.from_numpy(np.stack([b_[3] for b_ in buf])).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            S, s_null, _ = model(x, m)
        p_null, m_on, m_wk, a_map, d_map, p_map = model.decoder.posterior(S.float(), s_null.float())
        for k, (sid, w, _, mk) in enumerate(buf):
            out[(sid, str(data[sid].win_dates[w]))] = dict(
                w=w, n=int(mk.sum()), p_null=float(p_null[k]), m_on=m_on[k].cpu().numpy(),
                m_wk=m_wk[k].cpu().numpy(), a=int(a_map[k]), b=int(a_map[k] + d_map[k]))
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


def gain_curve(m):
    from scripts.decode_posteriors import gain_curve as g
    return g(m)


def mass(m, t, r=2):
    return float(m[max(0, t - r): t + r + 1].sum())


def postprocess(post, data, mode, null_threshold=0.5):
    """kaggle: posterior-marginal peaks (<=20/event/window); onepair: MAP pair with abstention."""
    ev, pw = [], []
    for (sid, nd), r in post.items():
        s, w, n = data[sid], r["w"], r["n"]
        if n < MIN_DUR + 1:
            continue
        if mode == "kaggle":
            for key, event in (("m_on", "onset"), ("m_wk", "wakeup")):
                m = r[key][:n]
                peaks, _ = find_peaks(m, height=1e-4, distance=6)
                if len(peaks) == 0:
                    continue
                sc = np.array([mass(m, p) for p in peaks])
                top = np.argsort(-sc)[:20]
                for st, v in zip(bin_to_step(s, w, peaks[top]), sc[top]):
                    ev.append((sid, nd, event, int(st), float(v)))
        else:
            if null_threshold is not None and r["p_null"] > null_threshold:
                continue
            if mode == "onepair_eap":                     # expected-AP point + score per event
                g_on, g_wk = gain_curve(r["m_on"][:n]), gain_curve(r["m_wk"][:n])
                aa, bb = int(np.argmax(g_on)), int(np.argmax(g_wk))
                if bb <= aa:
                    aa, bb = r["a"], r["b"]
                s_on, s_wk = float(g_on[aa]), float(g_wk[bb])
            else:
                aa, bb = r["a"], r["b"]
                s_on, s_wk = mass(r["m_on"], aa), mass(r["m_wk"], bb)
            st_on, st_wk = bin_to_step(s, w, [aa, bb])
            ev += [(sid, nd, "onset", int(st_on), s_on), (sid, nd, "wakeup", int(st_wk), s_wk)]
            pw.append((sid, nd, int(st_on), int(st_wk)))
    return (pd.DataFrame(ev, columns=["series_id", "night_date", "event", "step", "score"]),
            pd.DataFrame(pw, columns=["series_id", "night_date", "onset_step", "wakeup_step"]))


def train_fold(fold, parts, data, windows, truth, a, device, seed, fh, out_dir):
    seed_everything(seed + int(fold))
    tr_ids = [str(x) for x in parts["train"]]; va_ids = [str(x) for x in parts["val"]]
    stats = norm_stats([data[s] for s in tr_ids])
    lab, _ = labelled_items(tr_ids, data, windows)
    lab = [it for it in lab if 0 < (it[3] - it[2]) // SPM < MAX_DUR]
    px = proxy_items(tr_ids, data, windows) if a.use_null else []
    log(f"seed {seed} fold {fold}: labelled {len(lab)}, proxy-negative {len(px)}", fh)
    g = torch.Generator(); g.manual_seed(seed + int(fold))
    loader = torch.utils.data.DataLoader(P2Set(lab + px, data, stats), batch_size=a.batch, shuffle=True,
                                         num_workers=a.workers, generator=g,
                                         pin_memory=device.type == "cuda")
    model = build_model(duration_prior(lab, data), a.learn_kernel).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    total = a.epochs * len(loader); warm = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / warm if t < warm else
                                              0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    vt = truth[truth.series_id.isin(va_ids)]
    best, best_state, rows, best_epoch = -1.0, None, [], 0
    for epoch in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); losses = []
        for x, m, aa, bb, y in loader:
            x, m, aa, bb, y = (t.to(device, non_blocking=True) for t in (x, m, aa, bb, y))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model, x, m, aa, bb, y, a.use_null)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach()))
        post = infer(model, va_ids, data, stats, device)
        ev, _ = postprocess(post, data, "kaggle")
        ev1, _ = postprocess(post, data, "onepair_eap", None if not a.use_null else 0.5)
        v = event_detection_ap(vt, ev[COLS]) if len(ev) else 0.0
        v1 = event_detection_ap(vt, ev1[COLS]) if len(ev1) else 0.0
        v, v1 = (x_ if np.isfinite(x_) else 0.0 for x_ in (v, v1))
        rows.append({"fold": int(fold), "epoch": epoch, "loss": float(np.mean(losses)),
                     "val_edap": v, "val_edap_k1": v1, "sec": time.time() - t0})
        log(f"  fold {fold} epoch {epoch:02d} loss {np.mean(losses):.4f} val EDAP {v:.4f} "
            f"K1 {v1:.4f} ({time.time()-t0:.0f}s)", fh)
        crit = v1 if a.select == "k1" else v
        if crit > best or best_state is None:
            best, best_epoch = crit, epoch
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "stats": stats, "epoch": best_epoch, "val_edap": best},
               out_dir / f"fold{fold}_best.pt")
    log(f"seed {seed} fold {fold}: best epoch {best_epoch} (val EDAP {best:.4f}); kernel shift "
        f"{model.decoder.k_shift.detach().cpu().numpy().round(2)} sigma "
        f"{model.decoder.k_logsig.exp().detach().cpu().numpy().round(2)}", fh)
    return infer(model, [str(x) for x in parts["test"]], data, stats, device), rows, \
        {"fold": int(fold), "best_epoch": best_epoch, "val_edap": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--learn-kernel", action="store_true")
    ap.add_argument("--no-null", dest="use_null", action="store_false")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--kernel-sigma", type=float, default=1.5, help="label-noise kernel width (bins)")
    ap.add_argument("--aux-heat", type=float, default=0.0, help="weight of B1-style heatmap BCE on e_on/e_wk")
    ap.add_argument("--select", choices=["kaggle", "k1"], default="kaggle",
                    help="epoch selection: val EDAP (Kaggle-mode) or val K1 EDAP (eap one-pair)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--reinfer", action="store_true",
                    help="no training: load fold*_best.pt, re-infer test folds, save OOF + posteriors")
    a = ap.parse_args()
    global KERNEL_SIGMA, AUX_HEAT
    KERNEL_SIGMA, AUX_HEAT = a.kernel_sigma, a.aux_heat
    cfg = load_config(a.config)
    out = Path(cfg["paths"]["output_dir"]); cache = Path(cfg["paths"]["cache_dir"])
    root = out / f"p2_{a.tag}"; root.mkdir(exist_ok=True)
    fh = open(out / f"p2_{a.tag}_runtime.log", "a", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"PHASE 2 [{a.tag}] device {device}; learn_kernel={a.learn_kernel} use_null={a.use_null} "
        f"kernel_sigma={a.kernel_sigma} aux_heat={a.aux_heat} select={a.select}", fh)
    windows = pd.read_parquet(out / "night_windows.parquet").astype({"series_id": str, "night_date": str})
    clock = load_clock_index(out / "clock_index.json")
    assert len(clock) == int(os.environ.get("HHN_EXPECTED_SERIES", 277)) == windows.series_id.nunique()
    folds = json.loads((out / "splits.json").read_text(encoding="utf-8"))["folds"]
    truth = truth_from_event_nights(pd.read_parquet(out / "event_nights.parquet"))
    truth["series_id"] = truth.series_id.astype(str)
    run_folds = a.folds or sorted(folds, key=int)
    if not a.eval_only:
        t0 = time.time()
        data = {sid: load_series(cache, out / "b1_cache", sid) for sid in sorted(windows.series_id.unique())}
        log(f"features ready ({time.time()-t0:.0f}s); params "
            f"{n_params(build_model(None, False)):,}", fh)
        for seed in a.seeds:
            sd = root / f"seed{seed}"; sd.mkdir(exist_ok=True)
            post, logs, best = {}, [], []
            for fold in run_folds:
                if a.reinfer:
                    ck = torch.load(sd / f"fold{fold}_best.pt", map_location=device, weights_only=False)
                    model = build_model(None, a.learn_kernel).to(device)
                    model.load_state_dict(ck["state_dict"])
                    post.update(infer(model, [str(x) for x in folds[fold]["test"]], data, ck["stats"], device))
                    log(f"seed {seed} fold {fold}: re-inferred from checkpoint (epoch {ck['epoch']})", fh)
                    continue
                p, r, b = train_fold(fold, folds[fold], data, windows, truth, a, device, seed, fh, sd)
                post.update(p); logs += r; best.append(b)
            if not a.reinfer:
                pd.DataFrame(logs).to_csv(sd / "train_log.csv", index=False)
                pd.DataFrame(best).to_csv(sd / "best_epochs.csv", index=False)
            if len(run_folds) < len(folds):
                log("partial folds -> skipping evaluation", fh); continue
            ev_k, _ = postprocess(post, data, "kaggle")
            ev_1, pw = postprocess(post, data, "onepair_eap", 0.5 if a.use_null else None)
            ev_k.to_parquet(sd / "oof_predictions_kaggle.parquet", index=False)
            ev_1.to_parquet(sd / "oof_predictions_onepair.parquet", index=False)
            pw.to_parquet(sd / "oof_predicted_windows.parquet", index=False)
            pd.DataFrame([{"series_id": k[0], "night_date": k[1], "p_null": v["p_null"]}
                          for k, v in post.items()]).to_parquet(sd / "oof_p_null.parquet", index=False)
            keys = list(post)
            np.savez_compressed(sd / "oof_posteriors.npz",
                                series_id=np.array([k[0] for k in keys]), night_date=np.array([k[1] for k in keys]),
                                w=np.array([post[k]["w"] for k in keys]), n=np.array([post[k]["n"] for k in keys]),
                                p_null=np.array([post[k]["p_null"] for k in keys], np.float32),
                                a=np.array([post[k]["a"] for k in keys]), b=np.array([post[k]["b"] for k in keys]),
                                m_on=np.stack([post[k]["m_on"] for k in keys]).astype(np.float16),
                                m_wk=np.stack([post[k]["m_wk"] for k in keys]).astype(np.float16))
        log(f"training finished in {(time.time()-t0)/60:.1f} min", fh)
    pts, res = [], {}
    for i, seed in enumerate(a.seeds):
        sd = root / f"seed{seed}"
        if (sd / "oof_predictions_kaggle.parquet").exists():
            b, mtr = evaluate_seed(sd, truth, windows, clock, folds, a.reps if i == 0 else 1, fh)
            res[seed] = b; pts.append({"seed": seed, **mtr})
    if not pts:
        return
    p = pd.DataFrame(pts); p.to_csv(root / "seed_point_estimates.csv", index=False)
    keys = ["official_edap", "anti_edap_k1", "onepair_official_edap", "onset_medae_min",
            "wakeup_medae_min", "onset_within30_pct", "wakeup_within30_pct", "window_recall_labelled",
            "proxy_false_detection_rate_unlabelled", "proxy_false_detection_rate_unannotated",
            "ba_spt_bias", "ba_spt_loa_low", "ba_spt_loa_high", "icc2_1_spt"]
    summ = pd.DataFrame({k: [p[k].mean(), p[k].std(ddof=1) if len(p) > 1 else np.nan] for k in keys},
                        index=["mean", "sd"]).T
    summ.to_csv(root / "seed_summary.csv")
    b = res[a.seeds[0]]
    print("\n" + "\n".join(f"{m:>40s}  {cell(b, m)}" for m in
                           ["official_edap", "anti_edap_k1", "anti_edap_k3", "onset_medae_min",
                            "wakeup_medae_min", "onset_within30_pct", "wakeup_within30_pct",
                            "window_recall_labelled", "proxy_false_detection_rate_unlabelled",
                            "proxy_false_detection_rate_unannotated", "ba_spt_bias", "icc2_1_spt"]))
    print(f"{'SPT LoA':>40s}  {b.loc['ba_spt_loa_low'].estimate:.1f} to {b.loc['ba_spt_loa_high'].estimate:.1f}")
    print("\nAcross seeds:\n" + summ.round(4).to_string())


if __name__ == "__main__":
    main()
