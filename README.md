# Structured Semi-Markov Decoding with Conformal Prediction for Wrist-Actigraphy Sleep Event Detection

**Author:** Hoang Nhut Huynh (Graduate Institute of Biomedical Engineering, National Chung Hsing University, Taiwan)

This is the official code for the manuscript *"Structured Semi-Markov Decoding with Conformal Prediction for Wrist-Actigraphy Sleep Event Detection"* by H. N. Huynh, Q. T. Nguyen Diep, T. V. Huynh, A. T. Tran, C. T. S. Ching, and T. N. Tran. The manuscript is submitted to *IEEE Journal of Biomedical and Health Informatics*.

Sleep onset and wake-up detectors for wrist accelerometry usually finish with heuristic post-processing: peak picking, non-maximum suppression, greedy matching and threshold tuning. This repository replaces that stage with **exact probabilistic inference** over the one-pair-per-night event space. A neural encoder scores onset, wake-up and sleep-state evidence. A **semi-Markov decoder** then computes the exact partition function and posterior over every (onset, duration) pair in $O(T \times D)$ time. A learned **null state** abstains on nights without a valid sleep period. The posterior feeds **conformal prediction intervals**, which give calibrated per-night uncertainty.

---

## Highlights

- **Exact, differentiable one-pair decoder.** The one-pair-per-night constraint reduces the semi-Markov state space to onset–duration pairs. The forward pass and the posterior marginals are therefore exact and end-to-end trainable (`src/models/semimarkov.py`).
- **Null-state abstention.** The model is trained with proxy-negative (non-wear) windows, so the partition function includes a "no sleep period" state. Its proxy false-detection rate is 0.000.
- **Decision-theoretic decoding.** The same posterior can be decoded offline, with no retraining, by its marginal mode (`mass2`), by Expected-AP (`eap`), or with abstention (`eap_null`).
- **Conformal prediction.** The package covers split, cross-conformal and leave-one-subject-out recalibration with absolute and HPD scores. At equal coverage, HPD intervals are 39–41% narrower.
- **Baselines and evaluation.** It includes the GGIR HDCZA heuristic (B0), a Kaggle-style CNN–BiGRU heatmap model (B1), and the official event-detection AP metric. It also provides candidate-capped EDAP (K1/K3), MedAE, Bland–Altman, ICC(2,1) and bootstrap confidence intervals.
- **External validation.** Zero-shot transfer to the Newcastle polysomnography + GENEActiv cohort, with GENEActiv decoding and autocalibration.

---

## Repository structure

```
SemiMarkov-Conformal-Sleep/
├── configs/
│   └── phase1.yaml              # data paths, preprocessing and split settings
├── src/
│   ├── data/                    # CMI streaming pipeline, night windows, clock index,
│   │                            # B1/P2 features, GENEActiv reader, Newcastle PSG parser
│   ├── models/
│   │   ├── b1_net.py            # B1: CNN + BiGRU heatmap encoder
│   │   └── semimarkov.py        # P2: exact one-pair semi-Markov decoder + null state
│   ├── baselines/hdcza.py       # B0: HDCZA (van Hees et al., 2018)
│   └── metrics/                 # EDAP (+ K-capped), epoch metrics, clinical metrics, bootstrap
├── scripts/                     # command-line entry points (see "Reproducing the paper")
├── tests/                       # pytest unit / regression tests
├── matlab_figures/              # MATLAB scripts for the manuscript figures
├── third_party/kaggle_metric/   # official Kaggle event-detection AP metric (unmodified)
├── requirements.txt
└── CITATION.cff
```

---

## Installation

Requirements: Python ≥ 3.10. Training needs a CUDA GPU; the experiments ran on a single NVIDIA RTX 5060. MATLAB R2021a or later is needed only for the figures.

```bash
git clone https://github.com/<your-account>/SemiMarkov-Conformal-Sleep.git
cd SemiMarkov-Conformal-Sleep
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the build for your CUDA
pip install -r requirements.txt
python -m pytest -q
```

---

## Data

No data are distributed with this repository.

| Dataset | Use | Source |
|---|---|---|
| Child Mind Institute – Detect Sleep States (CMI) | training, cross-validation, conformal calibration | [Kaggle competition](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states) |
| Newcastle PSG + GENEActiv (van Hees et al., 2015/2018) | zero-shot external validation | [Zenodo record 1160410](https://zenodo.org/records/1160410) |

1. Put `train_series.parquet` and `train_events.csv` from Kaggle in `data/`, or edit `paths.data_dir` in `configs/phase1.yaml`.
2. Download and extract the Newcastle archive anywhere, then pass that folder to `--data` (step 7 below).

All outputs are written under `outputs/`, which git ignores.

---

## Reproducing the paper

Run every command from the repository root.

### 1. Data pipeline, night windows and splits
```bash
python -m scripts.build_data        --config configs/phase1.yaml   # per-series caches, night windows, non-wear
python -m scripts.sanity            --config configs/phase1.yaml   # integrity checks
python -m scripts.build_clock_index --config configs/phase1.yaml   # per-series clock-offset index
python -m scripts.run_make_splits   --config configs/phase1.yaml   # subject-level 5-fold splits
python -m scripts.split_report      --config configs/phase1.yaml
```

### 2. B0 — HDCZA heuristic baseline
```bash
python -m scripts.run_b0     --config configs/phase1.yaml          # default + per-fold tuned
python -m scripts.analyze_b0 --config configs/phase1.yaml
```

### 3. B1 — Kaggle-style deep baseline
```bash
python -m scripts.run_b1 --config configs/phase1.yaml --seeds 42 43 44
```

### 4. P2 — proposed semi-Markov model and ablations
```bash
# smoke test
python -m scripts.run_p2 --config configs/phase1.yaml --seeds 42 --folds 1 --epochs 2
# main model (3 seeds)
python -m scripts.run_p2 --config configs/phase1.yaml --seeds 42 43 44 --tag main
# ablations
python -m scripts.run_p2 --config configs/phase1.yaml --seeds 42 43 44 --tag no_null --no-null
python -m scripts.run_p2 --config configs/phase1.yaml --seeds 42 43 44 --tag aux --aux-heat 1.0 --select k1
```
Main options: `--kernel-sigma` (label-noise kernel width in bins, default 1.5), `--learn-kernel`, `--select {kaggle,k1}`, `--eval-only`.

### 5. Offline decoding of saved posteriors (no retraining)
```bash
python -m scripts.decode_posteriors   --config configs/phase1.yaml --seed-dir outputs/p2_main/seed42 \
                                      --variants mass2 eap eap_null
python -m scripts.score_decoded_seeds --config configs/phase1.yaml --model-dir p2_main --variant eap_null
python -m scripts.compare_models      --config configs/phase1.yaml --a <dir_a> --b <dir_b>
```

### 6. Conformal prediction and selective prediction
```bash
python -m scripts.run_p3 --config configs/phase1.yaml --seed-dir outputs/p2_main/seed42
```

### 7. External validation (Newcastle PSG)
```bash
python -m scripts.inspect_newcastle --dir  "path/to/psgnewcastle2015"                          # optional
python -m scripts.run_newcastle --config configs/phase1.yaml --data "path/to/psgnewcastle2015" --prep
python -m scripts.run_newcastle --config configs/phase1.yaml --seed-dir outputs/p2_main/seed42 --eval
python -m scripts.hdcza_newcastle_padded                                                       # HDCZA padding check
```

### 8. Figures
```bash
python -m scripts.export_fig_case_study --seed-dir outputs/p2_main/seed42
```
Then, in MATLAB, with `matlab_figures/` as the current folder:
```matlab
run_all_figures
fig_case_study
```

---

## Main results (CMI, out-of-fold, 4,783 labeled windows, 3 seeds)

| Model | EDAP | K1 | FD ↓ | SPT bias (min) | ICC |
|---|---|---|---|---|---|
| B0 HDCZA (default) | 0.514 | 0.514 | 0.000 | +11.1 | 0.829 |
| B1 (Kaggle post-processing) | 0.732 ± 0.006 | 0.680 ± 0.005 | 0.998 | +6.17 ± 0.41 | 0.841 |
| P2 (mass2, no abstention) | 0.721 ± 0.003 | 0.661 ± 0.004 | 1.000 | +1.80 ± 1.31 | 0.833 |
| **P2 (EAP + null, proposed)** | 0.695 ± 0.004 | 0.669 ± 0.003 | **0.000** | +3.75 ± 1.76 | **0.850** |

Conformal intervals at the 90% target (cross-conformal): the absolute score reaches coverage 0.900 with onset/wake-up widths of 60.3/45.0 min. The HPD score reaches coverage 0.897 with widths of 35.7/27.3 min.

These numbers are the values reported in the submitted manuscript and may be updated during peer review.

---

## Tests

```bash
python -m pytest -q -rs
```
The tests cover:
- exactness of the semi-Markov decoder against brute-force enumeration;
- regression of the EDAP implementation against the official Kaggle metric;
- conformal validity, HDCZA, clinical metrics, bootstrap, GENEActiv decoding and the PSG parser.

Some regression tests are skipped until the corresponding outputs exist.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Huynh2026SemiMarkovSleep,
  author  = {Huynh, Hoang Nhut and Nguyen Diep, Quoc Tuan and Huynh, Thanh Ven and
             Tran, Anh Tu and Ching, Congo Tak Shing and Tran, Trung Nghia},
  title   = {Structured Semi-Markov Decoding with Conformal Prediction for
             Wrist-Actigraphy Sleep Event Detection},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License and acknowledgments

Copyright © 2026 Hoang Nhut Huynh. All rights reserved unless a `LICENSE` file states otherwise.

- `third_party/kaggle_metric/event_detection_ap.py` is the official metric of the Kaggle competition. It is included unmodified, and its authorship belongs to Kaggle and the competition hosts.
- HDCZA follows van Hees *et al.*, *Sci. Rep.* 8:12975, 2018.
- GENEActiv autocalibration follows van Hees *et al.*, *J. Appl. Physiol.* 117:738–744, 2014.
- We acknowledge Ho Chi Minh City University of Technology (HCMUT), VNU-HCM, for supporting this study.

## Contact

Hoang Nhut Huynh — d114068001@mail.nchu.edu.tw
