# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Inspect an unpacked Newcastle (Zenodo 1160410) folder before writing the converter.

Prints the file inventory, the GENEActiv header of one .bin, the first lines of one PSG .txt and
the participant csv, plus decoded epoch statistics for a short slice. Nothing is written.
Usage: python -m scripts.inspect_newcastle --dir "D:\\data\\psgnewcastle2015"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.geneactiv import read_bin, to_epochs, autocalibrate
from src.data.newcastle import parse_psg, index_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pages", type=int, default=2000, help="pages to decode for the sample")
    a = ap.parse_args()
    root = Path(a.dir)
    if not root.exists():
        raise SystemExit(f"ERROR: {root} does not exist. Download dataset_psgnewcastle2015_v1.0.zip "
                         f"from https://zenodo.org/records/1160410 and extract it there.")
    if not root.is_dir():
        raise SystemExit(f"ERROR: {root} is not a folder (point --dir at the EXTRACTED folder, "
                         f"not the .zip).")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"ERROR: no files under {root}. Extract the zip there; the folder should "
                         f"contain 55 .bin files, 28 .txt PSG files and one .csv.")
    by_ext = pd.Series([p.suffix.lower() for p in files]).value_counts()
    print(f"root: {root}\nfiles: {len(files)}\n{by_ext.to_string()}\n")
    for ext in (".csv", ".txt", ".bin"):
        sel = [p for p in files if p.suffix.lower() == ext][:3]
        print(f"--- example {ext} files ---")
        for p in sel:
            print(f"  {p.relative_to(root)}  ({p.stat().st_size/1e6:.1f} MB)")
    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    if csvs:
        print(f"\n=== {csvs[0].name} ===")
        print(pd.read_csv(csvs[0]).head(8).to_string())
    pairing = index_folder(root)
    print("\n=== participant pairing (left wrist) ===")
    print(f"  PSG files matched: {len(pairing)}; with a left-wrist .bin: {pairing.bin.notna().sum()}")
    missing = pairing.loc[pairing.bin.isna(), "pid"].tolist()
    if missing:
        print(f"  no left-wrist .bin for pid: {missing}")
    print("\n=== PSG summaries (all participants) ===")
    srows = []
    for r in pairing.itertuples(index=False):
        try:
            d = parse_psg(r.psg)
            srows.append(dict(pid=r.pid, date=d["recording_date"].date(),
                              onset=d["onset"].strftime("%H:%M"), wake=d["wakeup"].strftime("%H:%M"),
                              spt_min=round(d["spt_min"], 1), tst_min=round(d["tst_min"], 1),
                              epochs=d["n_epochs"], pct_unscored=round(d["pct_unscored"], 1)))
        except Exception as exc:
            srows.append(dict(pid=r.pid, date=f"PARSE FAILED: {exc}"))
    print(pd.DataFrame(srows).to_string(index=False))

    txts = [p for p in files if p.suffix.lower() == ".txt"]
    if txts:
        print(f"\n=== first 25 lines of {txts[0].name} ===")
        with txts[0].open(encoding="latin-1", errors="replace") as f:
            for i, ln in enumerate(f):
                if i >= 25:
                    break
                print(f"  {ln.rstrip()}")
        print(f"  ... total lines: {sum(1 for _ in txts[0].open(encoding='latin-1', errors='replace'))}")
    bins = [p for p in files if p.suffix.lower() == ".bin"]
    if bins:
        print(f"\n=== GENEActiv header of {bins[0].name} ===")
        rec = read_bin(bins[0], max_pages=a.pages)
        for k in ("Device Unique Serial Code", "Measurement Frequency", "Start Time",
                  "x gain", "x offset", "y gain", "y offset", "z gain", "z offset"):
            if k in rec.header:
                print(f"  {k}: {rec.header[k]}")
        print(f"  decoded {len(rec.xyz):,} samples at {rec.sample_rate} Hz; "
              f"first {pd.Timestamp(rec.time[0])}, last {pd.Timestamp(rec.time[-1])}")
        print(f"  |a| mean {np.linalg.norm(rec.xyz, axis=1).mean():.3f} g "
              f"(should be near 1.0 if calibration is sane)")
        sc, off, n_static, before, after = autocalibrate(rec.xyz, rec.sample_rate)
        print(f"  autocalibration: {n_static} static windows, error {before:.4f} -> {after:.4f} g, "
              f"scale {np.round(sc,4)}, offset {np.round(off,4)}")
        ep = to_epochs(rec, 5, sc, off)
        print(f"  5-s epochs: {len(ep):,}; anglez [{ep.anglez.min():.1f}, {ep.anglez.max():.1f}] deg; "
              f"ENMO median {ep.enmo.median():.4f} g, P99 {ep.enmo.quantile(.99):.4f} g")
        print(ep.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
