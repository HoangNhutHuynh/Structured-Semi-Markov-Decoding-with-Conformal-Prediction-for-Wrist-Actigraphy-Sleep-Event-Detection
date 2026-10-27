# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Shared configuration and reproducibility utilities."""
from __future__ import annotations
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a YAML mapping')
    cfg['_config_path'] = str(path)
    root = path.parent.parent
    cfg['_root'] = str(root)
    for key in ('output_dir', 'cache_dir'):
        value = Path(cfg['paths'][key])
        cfg['paths'][key] = str(value if value.is_absolute() else (root / value).resolve())
    return cfg


def seed_all(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def git_hash(root: str) -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def write_manifest(cfg: dict, destination: Path, extra: dict | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
    payload = {'config': safe_cfg, 'git_hash': git_hash(cfg['_root']), 'extra': extra or {}}
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(2**20), b''):
            sha.update(buf)
    return sha.hexdigest()
