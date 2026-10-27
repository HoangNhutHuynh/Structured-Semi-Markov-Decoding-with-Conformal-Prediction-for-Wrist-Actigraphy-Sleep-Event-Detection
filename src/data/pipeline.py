# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Step A: disk-backed series extraction, local noon windows and annotation mapping."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .common import file_sha256, write_manifest

TZ = 'recorded-offset wall clock (not tz database)'
STATUSES = ('labelled', 'cross_boundary', 'partial', 'unlabelled', 'unannotated')
BLOCKING_CHECKS = ('b', 'c', 'd')  # 'a' is informational (cross_boundary status)
CHECKS = ('a', 'b', 'c', 'd')


def extract_series(source: Path, cache_dir: Path, batch_size: int) -> list[str]:
    """Stream input parquet batches into separate series parquet files; never load all rows."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / 'extraction.json'
    signature = {'source': str(source.resolve()), 'size': source.stat().st_size,
                 'mtime_ns': source.stat().st_mtime_ns, 'batch_size': batch_size}
    if marker.exists():
        old = json.loads(marker.read_text(encoding='utf-8'))
        if old.get('signature') == signature and all((cache_dir / f'{s}.parquet').exists() for s in old['series_ids']):
            return old['series_ids']
    if any(cache_dir.glob('*.parquet')):
        raise RuntimeError('Existing cache is stale; manually remove cache/*.parquet before re-extraction')
    reader = pq.ParquetFile(source)
    writers: dict[str, pq.ParquetWriter] = {}
    try:
        for batch in reader.iter_batches(batch_size=batch_size,
                                         columns=['series_id', 'step', 'timestamp', 'anglez', 'enmo']):
            table = pa.Table.from_batches([batch])
            sid = table.column('series_id').to_numpy(zero_copy_only=False)
            for name in np.unique(sid):
                key = str(name)
                sub = table.filter(pa.array(sid == name))
                if key not in writers:
                    writers[key] = pq.ParquetWriter(cache_dir / f'{key}.parquet', sub.schema,
                                                    compression='zstd')
                writers[key].write_table(sub)
    finally:
        for writer in writers.values():
            writer.close()
    ids = sorted(writers)
    marker.write_text(json.dumps({'signature': signature, 'series_ids': ids}, indent=2), encoding='utf-8')
    return ids


def night_date(local_timestamps: pd.Series, boundary_hour: int = 12) -> pd.Series:
    """Calendar date of the boundary hour that begins a night window; YYYY-MM-DD strings.
    Default 12 (noon) keeps backward compatibility; Phase 1 config uses 15 (see DECISIONS.md)."""
    # Date arithmetic on local wall-clock values avoids DST day-length assumptions.
    dates = local_timestamps.dt.date.astype(str)
    previous = (pd.to_datetime(dates) - pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
    return pd.Series(np.where(local_timestamps.dt.hour < boundary_hour, previous, dates), index=local_timestamps.index)


def detect_nonwear(step: np.ndarray, angle: np.ndarray, enmo: np.ndarray,
                   shift: int = 17280, min_steps: int = 720) -> np.ndarray:
    """Flag invalid readings and runs matching measurements at t±shift STEPS."""
    n = len(step)
    valid = np.isfinite(angle) & np.isfinite(enmo)
    flagged = ~valid
    if n == 0:
        return flagged
    if np.any(np.diff(step) <= 0):
        raise ValueError('steps must be strictly increasing and unique')
    for delta in (-shift, shift):
        targets = step + delta
        idx = np.searchsorted(step, targets)
        in_range = idx < n
        present = np.zeros(n, dtype=bool)
        present[in_range] = step[idx[in_range]] == targets[in_range]
        hit = np.flatnonzero(present)
        matching = np.zeros(n, dtype=bool)
        matching[hit] = (valid[hit] & valid[idx[hit]] &
                         (angle[hit] == angle[idx[hit]]) & (enmo[hit] == enmo[idx[hit]]))
        loc = np.flatnonzero(matching)
        if len(loc) == 0:
            continue
        # Consecutive means consecutive actual step IDs, not merely adjacent parquet rows.
        boundaries = np.r_[0, np.flatnonzero(np.diff(step[loc]) != 1) + 1, len(loc)]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            if b-a >= min_steps:
                flagged[loc[a:b]] = True
    return flagged


def load_series(cache_dir: Path, sid: str, processing: dict) -> pd.DataFrame:
    df = pq.read_table(cache_dir / f'{sid}.parquet').to_pandas()
    df = df.sort_values('step', kind='stable').reset_index(drop=True)
    step = df.step.to_numpy(dtype=np.int64)
    if len(step) > 1 and (np.diff(step) <= 0).any():
        raise ValueError(f'Duplicate or decreasing steps: {sid}')
    # Local wall clock = the clock written in the source timestamp (offset as recorded).
    # Source offsets switch -0400/-0500 about one week away from the official
    # America/New_York DST dates in 56/277 series (e.g. 08db4255286f switches on
    # 2018-11-11 instead of 2018-11-04), so tz-database conversion would shift the
    # wall clock by 1 h on those days. We therefore trust the recorded offsets and only
    # verify that they are internally consistent (UTC advances 5 s per step).
    ts = df.timestamp.astype(str)
    if not ts.str.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4}').all():
        raise ValueError(f'Unexpected timestamp format in series {sid}')
    local = pd.to_datetime(ts.str[:19], format='%Y-%m-%dT%H:%M:%S')
    utc = pd.to_datetime(ts, format='%Y-%m-%dT%H:%M:%S%z', utc=True)
    utc_sec = ((utc - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    if len(step) > 1 and (np.diff(utc_sec) != int(processing['sample_seconds']) * np.diff(step)).any():
        raise ValueError(f'UTC time not continuous with step index in series {sid}')
    df['utc_offset'] = ts.str[-5:]
    df['local_time'] = local
    df['night_date'] = night_date(local, int(processing.get('night_boundary_hour', 12)))
    df['nonwear'] = detect_nonwear(step, df.anglez.to_numpy(dtype=float),
                                   df.enmo.to_numpy(dtype=float),
                                   shift=int(processing['nonwear_repeat_shift']),
                                   min_steps=int(processing['nonwear_min_minutes'] * 60 /
                                                 processing['sample_seconds']))
    return df


def load_event_nights(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path, dtype={'series_id': str, 'night': 'int64', 'event': str})
    required = {'series_id', 'night', 'event', 'step'}
    if not required.issubset(events.columns):
        raise ValueError(f'Missing event columns: {required-set(events.columns)}')
    if not set(events.event.unique()).issubset({'onset', 'wakeup'}):
        raise ValueError('Unknown event types present')
    if events.duplicated(['series_id', 'night', 'event']).any():
        raise ValueError('Duplicate event-file series/night/event entries')
    wide = events.pivot(index=['series_id','night'], columns='event', values='step').reset_index()
    wide.columns.name = None
    for col in ('onset', 'wakeup'):
        if col not in wide:
            wide[col] = np.nan
    wide = wide.rename(columns={'onset':'onset_step', 'wakeup':'wakeup_step'})
    wide['label_status'] = np.select([
        wide.onset_step.notna() & wide.wakeup_step.notna(),
        wide.onset_step.notna() ^ wide.wakeup_step.notna()],
        ['labelled', 'partial'], default='unlabelled')
    return wide.sort_values(['series_id', 'night']).reset_index(drop=True)


def map_annotations(series_id: str, event_rows: pd.DataFrame,
                    step_to_date: dict[int, str], observed_dates: set[str]) -> tuple[dict, dict]:
    """Map annotation indices to noon dates; preserve and report mapping conflicts."""
    problems: dict[str, list[dict]] = {k: [] for k in CHECKS}
    records = event_rows.sort_values('night').to_dict('records')
    anchors = []
    for row in records:
        onset = row['onset_step']; wake = row['wakeup_step']
        if pd.notna(onset):
            # Anchor on the sleep-window midpoint when both events exist, so that
            # early onsets (e.g. 14:39) or late wakeups (e.g. 16:30) still anchor to the
            # night they belong to (design decision 2026-10-19, fixes 3be1545083b7 n14/n15).
            anchor_step = int((int(onset) + int(wake)) // 2) if pd.notna(wake) else int(onset)
            date = step_to_date.get(anchor_step, step_to_date.get(int(onset)))
            if date is not None:
                anchors.append((int(row['night']), date))
    anchors.sort()
    unsafe_indices: set[int] = set()
    for (n1, d1), (n2, d2) in zip(anchors[:-1], anchors[1:]):
        observed_distance = (pd.Timestamp(d2) - pd.Timestamp(d1)).days
        if observed_distance != n2 - n1:
            for row in records:
                night = int(row['night'])
                if n1 < night < n2:
                    unsafe_indices.add(night)
                    problems['d'].append({'series_id':series_id,'night':night,
                                          'anchor_nights':[n1,n2],
                                          'calendar_distance':observed_distance,
                                          'night_index_distance':n2-n1})
    mapped = {}
    observed_sorted = sorted(observed_dates)
    first_obs, last_obs = (observed_sorted[0], observed_sorted[-1]) if observed_sorted else (None, None)
    for row in records:
        night = int(row['night'])
        onset = row['onset_step']; wake = row['wakeup_step']
        if not anchors:
            # Series without any onset anchor (all nights unlabelled): nothing to align.
            problems.setdefault('c_exempt', []).append({'series_id':series_id,'night':night,
                                                        'reason':'no_anchor_in_series'})
            continue
        if night in unsafe_indices:
            continue  # already reported under check (d); left unmapped
        lower = [a for a in anchors if a[0] <= night]
        nearest = lower[-1] if lower else anchors[0]
        intended = (pd.Timestamp(nearest[1]) + pd.Timedelta(days=night-nearest[0])).strftime('%Y-%m-%d')
        onset_date = step_to_date.get(int(onset)) if pd.notna(onset) else None
        wake_date = step_to_date.get(int(wake)) if pd.notna(wake) else None
        row = dict(row)
        if pd.notna(onset) and pd.notna(wake) and onset_date != wake_date:
            # Valid label whose wakeup lies in the next window (e.g. wake after 15:00).
            problems['a'].append({'series_id':series_id, 'night':night,
                                  'onset_date':onset_date, 'wakeup_date':wake_date})
            row['label_status'] = 'cross_boundary'
        elif pd.isna(onset) and pd.notna(wake) and intended != wake_date:
            problems['a'].append({'series_id':series_id, 'night':night,
                                  'index_date':intended, 'wakeup_date':wake_date})
        if intended not in observed_dates:
            if intended < first_obs or intended > last_obs:
                problems.setdefault('c_exempt', []).append({'series_id':series_id,'night':night,
                                                            'mapped_date':intended,
                                                            'reason':'outside_recording'})
            else:
                problems['c'].append({'series_id':series_id,'night':night,'mapped_date':intended})
        else:
            mapped.setdefault(intended, []).append(row)
    for date, rows in mapped.items():
        if len(rows) > 1:
            for row in rows:
                problems['b'].append({'series_id':series_id,'night':int(row['night']),
                                      'mapped_date':date})
    return mapped, problems


def run_pipeline(cfg: dict) -> None:
    paths = cfg['paths']; processing = cfg['processing']
    source = Path(paths['data_dir']) / paths['series_file']
    event_source = Path(paths['data_dir']) / paths['events_file']
    cache = Path(paths['cache_dir']); out = Path(paths['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    ids = extract_series(source, cache, int(processing['batch_size']))
    ev = load_event_nights(event_source)
    night_rows = []; series_rows = []; problems = {k: [] for k in CHECKS}; problems['c_exempt'] = []
    for i, sid in enumerate(ids, 1):
        df = load_series(cache, sid, processing)
        # Cache step-aligned features and flags for later phases.
        np.save(cache / f'{sid}_nonwear.npy', df.nonwear.to_numpy(dtype=bool))
        # Deliberately flawed sr2025 labels: every non-labelled step defaults to wake (0).
        sr2025 = np.zeros(len(df), dtype=np.uint8)
        steps = df.step.to_numpy(dtype=np.int64)
        for _, ev_row in ev.loc[(ev.series_id == sid) & (ev.label_status == 'labelled')].iterrows():
            left = int(ev_row.onset_step); right = int(ev_row.wakeup_step)
            sr2025[(steps >= left) & (steps < right)] = 1
        np.save(cache / f'{sid}_sr2025.npy', sr2025)
        dates = df.night_date.to_numpy()
        step_to_date = dict(zip(df.step.to_numpy(dtype=np.int64), dates))
        mapped, local_problems = map_annotations(sid, ev.loc[ev.series_id == sid],
                                                  step_to_date, set(dates))
        for key in list(CHECKS) + ['c_exempt']:
            problems[key].extend(local_problems.get(key, []))
        unannotated_boundary = 0
        sorted_dates = sorted(set(dates))
        for j, date in enumerate(sorted_dates):
            window = df.loc[df.night_date == date]
            assigned = mapped.get(date, [])
            ann = assigned[0] if len(assigned) == 1 else None
            status = ann['label_status'] if ann is not None else 'unannotated'
            if status == 'unannotated' and j in (0, len(sorted_dates)-1):
                unannotated_boundary += 1
            night_rows.append({'series_id':sid, 'night_date':date,
                               'night':int(ann['night']) if ann is not None else None,
                               'label_status':status,
                               'onset_step':int(ann['onset_step']) if ann is not None and pd.notna(ann['onset_step']) else None,
                               'wakeup_step':int(ann['wakeup_step']) if ann is not None and pd.notna(ann['wakeup_step']) else None,
                               'first_step':int(window.step.iloc[0]),
                               'last_step':int(window.step.iloc[-1]),
                               'n_observations':len(window),
                               'nonwear_fraction':float(window.nonwear.mean()),
                               'unannotated_boundary':status=='unannotated' and j in (0,len(sorted_dates)-1)})
        series_rows.append({'series_id':sid,'n_steps':len(df),
                            'nonwear_fraction':float(df.nonwear.mean()),
                            'unannotated_boundary_windows':unannotated_boundary})
        if i % 25 == 0:
            print(f'Processed {i}/{len(ids)} series', flush=True)
    report = {'checks': {k:{'count':len(v), 'affected':v} for k,v in problems.items()},
              'threshold':int(processing['integrity_max_per_check'])}
    (out/'integrity_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['blocking_checks'] = list(BLOCKING_CHECKS)
    (out/'integrity_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if any(len(problems[k])>int(processing['integrity_max_per_check']) for k in BLOCKING_CHECKS):
        raise RuntimeError('Integrity threshold exceeded; see outputs/integrity_report.json. Metadata not published.')
    pd.DataFrame(night_rows).to_parquet(out/'night_windows.parquet',index=False)
    pd.DataFrame(series_rows).to_csv(out/'series_summary.csv',index=False)
    ev.to_parquet(out/'event_nights.parquet',index=False)
    write_manifest(cfg,out/'step_a_manifest.json',{'event_file_sha256':file_sha256(event_source),
                                                 'series_file_sha256':file_sha256(source)})
    print('Step A pipeline complete; inspect integrity_report.json',flush=True)
