# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Build outputs/clock_index.json from the Step A series cache (run once after Step A)."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from src.data.clock import build_clock_index


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/phase1.yaml")
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text(encoding="utf-8"))
    out = Path(cfg["paths"]["output_dir"]) / "clock_index.json"
    idx = build_clock_index(cfg["paths"]["cache_dir"], out)
    n_change = sum(len(v["change_steps"]) > 1 for v in idx.values())
    print(f"clock index: {len(idx)} series ({n_change} with offset changes) -> {out}")


if __name__ == "__main__":
    main()
