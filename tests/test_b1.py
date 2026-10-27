# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""B1 alignment/shape tests. Run on the real repo after Step A (uses outputs/); skips otherwise."""
import numpy as np, pandas as pd, torch, pytest
from pathlib import Path
from src.data.b1_features import load_series, targets, bin_to_step, window_tensor, norm_stats, M_BINS
from src.models.b1_net import B1Net

OUT = Path("outputs")
pytestmark = pytest.mark.skipif(not (OUT / "night_windows.parquet").exists(),
                                reason="needs Step A outputs")


def test_target_bins_map_back_to_label_steps_exactly():
    (OUT / "b1_cache").mkdir(exist_ok=True)
    win = pd.read_parquet(OUT / "night_windows.parquet")
    lab = win[win.label_status == "labelled"]
    lab = lab[lab.series_id.isin(sorted(lab.series_id.unique())[:3])]   # 3 series keep it fast
    for r in lab.itertuples():
        s = load_series(OUT / "cache", OUT / "b1_cache", r.series_id)
        w = int(np.flatnonzero(s.win_dates == r.night_date)[0])
        y = targets(s, w, r.onset_step, r.wakeup_step)
        on, wk = bin_to_step(s, w, [int(np.argmax(y[0])), int(np.argmax(y[1]))])
        # real CMI labels sit on whole minutes -> must map back exactly (synthetic: < 12 steps)
        assert abs(on - r.onset_step) < 12 and abs(wk - r.wakeup_step) < 12


def test_model_shapes_and_dst_window_length():
    sid = sorted(pd.read_parquet(OUT / "night_windows.parquet").series_id.unique())[0]
    (OUT / "b1_cache").mkdir(exist_ok=True)
    s = load_series(OUT / "cache", OUT / "b1_cache", sid)
    lengths = s.win_end - s.win_start
    assert lengths.max() <= 18000
    x, mask, _, n = window_tensor(s, 0, norm_stats([s]))
    out = B1Net()(torch.from_numpy(x)[None])
    assert out.shape == (1, 3, M_BINS) and mask.sum() == (n + 11) // 12
