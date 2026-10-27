# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Newcastle PSG + GENEActiv (Zenodo 1160410) -> CMI-shaped cache for external validation.

PSG files are RemLogic event exports: a text header, then a tab-separated table
    Sleep Stage \t Position \t Time [hh:mm:ss] \t Event \t Duration[s]
with one 30-s row per epoch. Times are clock-only, so the date is taken from the header's
"Recording Date" and rolled forward whenever the clock wraps past midnight.

Reference labels (stated explicitly because PSG defines sleep differently from the CMI rule set):
  onset  = start of the first epoch scored N1/N2/N3/R
  wakeup = end   of the last  epoch scored N1/N2/N3/R
i.e. the PSG sleep period time (SPT) window, the closest analogue of the CMI "main sleep window".
TST (total sleep time, excluding wake after onset) is also stored for the sensitivity analysis.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

SLEEP_STAGES = {"N1", "N2", "N3", "N4", "R", "REM", "S1", "S2", "S3", "S4"}
WAKE_STAGES = {"W", "WAKE"}
EPOCH_S = 30


def parse_psg(path: str | Path) -> dict:
    """Return {'epochs': DataFrame(start, stage, is_sleep), 'onset', 'wakeup', 'tst_min', ...}."""
    text = Path(path).read_text(encoding="latin-1", errors="replace").splitlines()
    rec_date = None
    for ln in text[:40]:
        m = re.search(r"Recording Date:\s*(\d{1,2})/(\d{1,2})/(\d{4})", ln)
        if m:
            d, mo, y = map(int, m.groups())          # dd/mm/yyyy in these exports
            rec_date = pd.Timestamp(year=y, month=mo, day=d)
            break
    if rec_date is None:
        raise ValueError(f"{path}: no 'Recording Date' in header")
    hdr = next((i for i, ln in enumerate(text) if ln.startswith("Sleep Stage")), None)
    if hdr is None:
        raise ValueError(f"{path}: no 'Sleep Stage' table header")
    rows = []
    for ln in text[hdr + 1:]:
        parts = [c.strip() for c in ln.split("\t")]
        if len(parts) < 5 or not re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", parts[2]):
            continue
        rows.append((parts[0].upper(), parts[2], float(parts[4])))
    if not rows:
        raise ValueError(f"{path}: no scored epochs")
    stage = [r[0] for r in rows]
    dur = np.array([r[2] for r in rows])
    tod = pd.to_timedelta(pd.Series([r[1] for r in rows]))
    day = (tod.diff().fillna(pd.Timedelta(0)) < pd.Timedelta(0)).cumsum()   # midnight rollovers
    start = rec_date + tod + pd.to_timedelta(day, unit="D")
    ep = pd.DataFrame({"start": start.to_numpy(), "stage": stage, "duration_s": dur})
    ep["is_sleep"] = ep.stage.isin(SLEEP_STAGES)
    if not ep.is_sleep.any():
        raise ValueError(f"{path}: no sleep epochs scored")
    idx = np.flatnonzero(ep.is_sleep.to_numpy())
    onset = pd.Timestamp(ep.start.iloc[idx[0]])
    wakeup = pd.Timestamp(ep.start.iloc[idx[-1]]) + pd.Timedelta(seconds=float(ep.duration_s.iloc[idx[-1]]))
    return {"epochs": ep, "onset": onset, "wakeup": wakeup,
            "tst_min": float(ep.loc[ep.is_sleep, "duration_s"].sum() / 60),
            "spt_min": float((wakeup - onset).total_seconds() / 60),
            "n_epochs": len(ep), "pct_unscored": float(100 * (~ep.is_sleep & ~ep.stage.isin(WAKE_STAGES)).mean()),
            "recording_date": rec_date}


def participant_id(name: str) -> int | None:
    m = re.search(r"MECSLEEP(\d+)", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def index_folder(root: str | Path, wrist: str = "left") -> pd.DataFrame:
    """Match each participant's PSG file with the requested wrist's .bin file."""
    root = Path(root)
    bins, psgs = {}, {}
    for p in root.rglob("*.bin"):
        pid = participant_id(p.name)
        if pid is not None and wrist.lower() in p.name.lower():
            bins[pid] = p
    for p in root.rglob("*.txt"):
        pid = participant_id(p.name)
        if pid is not None:
            psgs[pid] = p
    rows = [{"pid": pid, "bin": bins.get(pid), "psg": psgs[pid]} for pid in sorted(psgs)]
    return pd.DataFrame(rows)
