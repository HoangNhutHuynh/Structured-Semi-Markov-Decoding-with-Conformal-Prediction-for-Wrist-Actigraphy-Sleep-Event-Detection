# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Run from repository root: python -m scripts.build_data --config configs/phase1.yaml"""
import argparse
from src.data.common import load_config, seed_all
from src.data.pipeline import run_pipeline

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/phase1.yaml')
    args=p.parse_args(); cfg=load_config(args.config)
    seed_all(int(cfg['seed'])); run_pipeline(cfg)

if __name__=='__main__': main()
