# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Invoke the user's existing, patched v1.1 splitter, then print diagnostic table.

This is a wrapper, not a replacement for the accepted scripts/make_splits.py.
"""
from scripts.make_splits import main as existing_main
from scripts.split_report import main as report_main

if __name__ == '__main__':
    existing_main()
    report_main()
