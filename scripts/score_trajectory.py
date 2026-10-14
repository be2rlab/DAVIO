#!/usr/bin/env python3
"""Score a run's trajectories against the reference: the filter's odometry, the live stream as the
map corrected it, and the stream corrected by the final map. Writes RUN/evaluation.json."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from davio.datasets import open_dataset          # noqa: E402
from davio.eval import metrics                   # noqa: E402
from davio.geometry import json_safe             # noqa: E402

STREAMS = dict(odometry='trajectory.tum', live='map_trajectory.tum', final='map_trajectory_final.tum')


def evaluate(run, data_root, reference=None):
    run = Path(run)
    meta = json.loads((run / 'run.json').read_text())
    ds = open_dataset(meta['dataset'], data_root, meta['sequence'], reference)
    start, end = meta['interval_start'], meta['interval_end']
    streams = {name: metrics.read_trajectory(run / file) for name, file in STREAMS.items() if (run / file).is_file()}
    times = np.array([row[0] for row in streams['odometry']])
    windows = [json.loads(s) for s in (run / 'events.jsonl').read_text().splitlines() if '"kind": "window"' in s]
    ages = [w['completed_wall'] - w['arrival_wall'] for w in windows if w.get('report', {}).get('status') == 'mapped']
    gpu = [w['peak_gpu_gb'] for w in windows if w.get('peak_gpu_gb') is not None]
    output = dict(
        dataset=meta['dataset'], sequence=meta['sequence'], state=meta['state'], rate=meta['rate'],
        # A gap in the output is not interpolated away: no pose, no coverage.
        coverage=float(np.minimum(np.diff(times), .1).sum() / (end - start)) if len(times) > 1 else 0.,
        ate={name: metrics.ate(ds, rows, start) for name, rows in streams.items()},
        windows=meta['windows'], loops_closed=meta['loops_closed'],
        pose_age_p50_s=meta['pose_age_p50_s'], pose_age_p95_s=meta['pose_age_p95_s'],
        map_age_p50_s=float(np.median(ages)) if ages else None,      # newest image of a window received -> window in the map
        map_age_p95_s=float(np.percentile(ages, 95)) if ages else None,
        peak_gpu_gb=float(max(gpu)) if gpu else None, groundtruth=ds.reference_provenance())
    (run / 'evaluation.json').write_text(json.dumps(json_safe(output), indent=2, allow_nan=False))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True, help='the directory that holds the sequences')
    p.add_argument('--reference', choices=('dataset', 'openvins'), help='EuRoC: its own reference file, or the corrected '
                   'one OpenVINS distributes (default)')
    a = p.parse_args()
    print(json.dumps(evaluate(a.run, a.data, a.reference)['ate'], indent=2))


if __name__ == '__main__':
    main()
