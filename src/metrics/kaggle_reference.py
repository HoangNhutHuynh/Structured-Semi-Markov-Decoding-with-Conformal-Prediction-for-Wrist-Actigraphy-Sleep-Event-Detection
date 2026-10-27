# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Optional adapter for locally supplied official Kaggle metric module.

Official reference is NOT bundled. Add it under third_party/kaggle_metric/.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pandas as pd

REFERENCE = Path('third_party/kaggle_metric/event_detection_ap.py')


def official_score(truth: pd.DataFrame, predictions: pd.DataFrame,
                   path: str | Path = REFERENCE) -> float:
    filepath = Path(path)
    if not filepath.is_file():
        raise FileNotFoundError(filepath)
    spec = importlib.util.spec_from_file_location('hhn_official_edap',filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load reference: {filepath}')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tolerance={'onset':[12,36,60,90,120,150,180,240,300,360],
               'wakeup':[12,36,60,90,120,150,180,240,300,360]}
    if hasattr(module,'score'):
        return float(module.score(truth.copy(),predictions.copy(),tolerance,
                                  series_id_column_name='series_id',
                                  time_column_name='step',
                                  event_column_name='event',
                                  score_column_name='score',
                                  use_scoring_intervals=False))
    if hasattr(module,'event_detection_ap'):
        # Some Kaggle notebook exports expose global settings set by score().
        for name,value in [('series_id_column_name','series_id'),('time_column_name','step'),
                           ('event_column_name','event'),('score_column_name','score'),
                           ('use_scoring_intervals',False)]:
            if hasattr(module,name):
                setattr(module,name,value)
        return float(module.event_detection_ap(truth.copy(),predictions.copy(),tolerance))
    raise RuntimeError('reference module exposes neither score nor event_detection_ap')
