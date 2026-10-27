# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Pure-Python GENEActiv .bin reader + GGIR-style anglez / ENMO, for external validation.

No Java and no R: the external-validation pipeline stays reproducible from a plain pip install.
Cross-check against actipy (Java) or GENEAread (R) is available via scripts/check_geneactiv.py.

File format (GENEActiv v2.x binary): an ASCII header, then repeated 300-sample pages. Each page
carries its own start time, temperature and measurement frequency, followed by one hex line of
300 x 12 hex characters: x, y, z as 12-bit signed values (3 hex chars each) and light + button +
reserved in the last 3 hex chars. Calibration from the header: value = (raw - offset) / gain.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

PAGE_SAMPLES = 300
HEX_PER_SAMPLE = 12


@dataclass
class GeneActivRecord:
    time: np.ndarray          # datetime64[ns], local wall clock as recorded
    xyz: np.ndarray           # (N, 3) float32, g units
    temperature: np.ndarray   # (N,) float32, deg C (page-level value repeated)
    sample_rate: float
    header: dict


def _parse_header(lines: list[str]) -> dict:
    h = {}
    for ln in lines:
        if ":" in ln:
            k, _, v = ln.partition(":")
            h.setdefault(k.strip(), v.strip())
    return h


def _calibration(h: dict) -> tuple[np.ndarray, np.ndarray]:
    gain = np.array([float(h[f"{a} gain"]) for a in "xyz"], dtype=np.float64)
    offset = np.array([float(h[f"{a} offset"]) for a in "xyz"], dtype=np.float64)
    return gain, offset


_HEX_LUT = np.full(256, -1, dtype=np.int16)
for _i, _c in enumerate("0123456789abcdef"):
    _HEX_LUT[ord(_c)] = _HEX_LUT[ord(_c.upper())] = _i


def _hex_nibbles(line: str) -> np.ndarray:
    v = _HEX_LUT[np.frombuffer(line.encode("ascii"), dtype=np.uint8)]
    if (v < 0).any():
        raise ValueError("non-hexadecimal character in data page")
    return v.astype(np.int32)


def _hex12_to_signed(block: np.ndarray) -> np.ndarray:
    """(n, 3) unsigned 12-bit -> signed (two's complement)."""
    return np.where(block >= 2048, block - 4096, block).astype(np.int32)


def read_bin(path: str | Path, max_pages: int | None = None) -> GeneActivRecord:
    text = Path(path).read_text(encoding="latin-1", errors="replace").splitlines()
    starts = [i for i, ln in enumerate(text) if ln.startswith("Recorded Data")]
    if not starts:
        raise ValueError(f"{path}: no 'Recorded Data' pages found")
    header = _parse_header(text[: starts[0]])
    gain, offset = _calibration(header)
    if max_pages:
        starts = starts[:max_pages]
    times, data, temps, freqs = [], [], [], []
    for s in starts:
        page = text[s: s + 12]
        meta = _parse_header(page)
        hexline = next((ln.strip() for ln in text[s: s + 14]
                        if len(ln.strip()) == PAGE_SAMPLES * HEX_PER_SAMPLE), None)
        if hexline is None:
            continue
        t0 = meta.get("Page Time")
        freq = float(meta.get("Measurement Frequency", header.get("Measurement Frequency", "nan")))
        nib = _hex_nibbles(hexline).reshape(PAGE_SAMPLES, HEX_PER_SAMPLE)
        xyz12 = (nib[:, :9].reshape(PAGE_SAMPLES, 3, 3) * np.array([256, 16, 1])).sum(axis=2)
        data.append(_hex12_to_signed(xyz12))
        temps.append(float(meta.get("Temperature", "nan")))
        freqs.append(freq)
        times.append(t0)
    if not data:
        raise ValueError(f"{path}: no decodable pages")
    xyz = (np.concatenate(data).astype(np.float64) * 100.0 - offset) / gain
    fs = float(np.nanmedian(freqs))
    # Page times look like "2015-01-01 20:00:00:000": the milliseconds use a colon, not a dot.
    t0s = pd.to_datetime(pd.Series(times).str.strip()
                         .str.replace(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):(\d+)$",
                                      r"\1.\2", regex=True), format="mixed", errors="coerce")
    if t0s.isna().all():
        raise ValueError(f"{path}: could not parse page times (example: {times[0]!r})")
    step = pd.to_timedelta(np.arange(PAGE_SAMPLES) / fs, unit="s")
    time = np.concatenate([(t + step).to_numpy() for t in t0s])
    temperature = np.repeat(np.array(temps, np.float32), PAGE_SAMPLES)
    return GeneActivRecord(time, xyz.astype(np.float32), temperature, fs, header)


# ---------------------------------------------------------------------------------------------
def autocalibrate(xyz: np.ndarray, fs: float, temperature: np.ndarray | None = None,
                  window_s: int = 10, sd_criterion: float = 0.013, iterations: int = 300):
    """van Hees (2014) static-sphere autocalibration: find scale/offset so still windows have |a| = 1.

    Returns (scale, offset, n_static, calibration_error_before, after). If fewer than 10 static
    windows are found the identity calibration is returned (reported, never silently applied).
    """
    n = int(window_s * fs)
    k = len(xyz) // n
    if k < 10:
        return np.ones(3), np.zeros(3), 0, np.nan, np.nan
    w = xyz[: k * n].reshape(k, n, 3)
    sd = w.std(axis=1)
    mean = w.mean(axis=1)
    static = (sd < sd_criterion).all(axis=1)
    pts = mean[static]
    if len(pts) < 10:
        return np.ones(3), np.zeros(3), int(static.sum()), np.nan, np.nan
    before = float(np.abs(np.linalg.norm(pts, axis=1) - 1).mean())
    scale, offset = np.ones(3), np.zeros(3)
    for _ in range(iterations):
        cur = pts * scale + offset
        closest = cur / np.linalg.norm(cur, axis=1, keepdims=True)
        for a in range(3):                      # per-axis linear fit cur[:, a] -> closest[:, a]
            A = np.column_stack([cur[:, a], np.ones(len(cur))])
            sol, *_ = np.linalg.lstsq(A, closest[:, a], rcond=None)
            scale[a] *= sol[0]
            offset[a] = offset[a] * sol[0] + sol[1]
    after = float(np.abs(np.linalg.norm(pts * scale + offset, axis=1) - 1).mean())
    return scale, offset, int(static.sum()), before, after


def to_epochs(rec: GeneActivRecord, epoch_s: int = 5, scale=None, offset=None) -> pd.DataFrame:
    """GGIR-style 5-s epochs: anglez from per-epoch medians, ENMO = mean(max(|a| - 1, 0))."""
    xyz = rec.xyz.astype(np.float64)
    if scale is not None:
        xyz = xyz * scale + offset
    n = int(round(epoch_s * rec.sample_rate))
    k = len(xyz) // n
    if k == 0:
        raise ValueError("recording shorter than one epoch")
    w = xyz[: k * n].reshape(k, n, 3)
    med = np.median(w, axis=1)
    anglez = np.degrees(np.arctan2(med[:, 2], np.sqrt(med[:, 0] ** 2 + med[:, 1] ** 2)))
    enmo = np.maximum(np.linalg.norm(w, axis=2) - 1.0, 0.0).mean(axis=1)
    t = pd.to_datetime(rec.time[: k * n: n])
    return pd.DataFrame({"timestamp": t, "anglez": anglez.astype(np.float32),
                         "enmo": enmo.astype(np.float32)})


# ---------------------------------------------------------------------------------------------
def write_synthetic_bin(path: str | Path, xyz: np.ndarray, fs: float = 85.7,
                        start="2015-01-01 20:00:00:000", gain=(25714., 25696., 25548.),
                        offset=(-16., 858., -927.)) -> None:
    """Write a minimal but format-faithful .bin (used by the unit tests)."""
    n_pages = len(xyz) // PAGE_SAMPLES
    out = ["Device Identity", "Device Unique Serial Code:012345", "Calibration Data"]
    for a, g, o in zip("xyz", gain, offset):
        out += [f"{a} gain:{g}", f"{a} offset:{o}"]
    out += [f"Measurement Frequency:{fs}", "Memory Status", "Number of Pages:%d" % n_pages]
    t0 = pd.Timestamp(start[:19])
    for p in range(n_pages):
        seg = xyz[p * PAGE_SAMPLES:(p + 1) * PAGE_SAMPLES]
        raw = np.rint((seg * np.array(gain) + np.array(offset)) / 100.0).astype(int)
        raw = np.where(raw < 0, raw + 4096, raw) & 0xFFF
        hexes = "".join("".join(f"{v:03x}" for v in row) + "000" for row in raw)
        shift = pd.Timedelta(int(round(p * PAGE_SAMPLES / fs * 1e9)), unit="ns")
        tp = (t0 + shift).strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]
        out += ["Recorded Data", "Device Unique Serial Code:012345", f"Sequence Number:{p}",
                f"Page Time:{tp}", "Unassigned:", "Temperature:25.0", "Battery voltage:4.1",
                "Device Status:Recording", f"Measurement Frequency:{fs}", "", hexes]
    Path(path).write_text("\n".join(out), encoding="latin-1")
