#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from davio.data import open_dataset                     # noqa: E402
from davio.eval.metrics import matched_points, se3_align  # noqa: E402
from davio_mapper.mapping import world_points           # noqa: E402
from evaluate_run import read_trajectory                # noqa: E402

GT, ODOM, MAP, LOOP = (150, 150, 150), (255, 140, 0), (0, 190, 255), (255, 60, 60)


def align(ds, trajectory, start):
    """(R, t, ate_m) for this stream, or identity when too little matched."""
    est, ref, _o, _t, _n = matched_points(ds, trajectory, start)
    if est is None:
        return np.eye(3), np.zeros(3), None
    rot, trans, rmse = se3_align(est, ref)
    return rot, trans.ravel(), rmse


def polyline(rr, path, times, points, color, radius=.02):
    if len(points) < 2:
        return
    rr.log(path, rr.LineStrips3D([np.asarray(points, np.float32)], colors=[color],
                                 radii=radius), static=True)
    for t, p in zip(times, points):
        rr.set_time('sensor_time', timestamp=float(t))
        rr.log(path + '/head', rr.Points3D([p], colors=[color], radii=radius * 3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, help='.rrd to write (default: <run>/debug.rrd)')
    p.add_argument('--pixel-step', type=int, default=8, help='Submap decimation for viewing')
    p.add_argument('--spawn', action='store_true', help='Open the viewer instead of saving')
    a = p.parse_args()
    import rerun as rr

    meta = json.loads((a.run / 'run.json').read_text())
    ds = open_dataset(meta['dataset'], a.data, seq=meta['sequence'],
                      groundtruth=getattr(a, 'groundtruth', 'dataset'))
    start, end = meta['interval_start'], meta['interval_end']
    gt = ds.groundtruth()
    inside = (gt.t >= start) & (gt.t <= end)

    rr.init(f"davio-{meta['sequence']}-{meta['mode']}", spawn=a.spawn)
    if not a.spawn:
        rr.save(str(a.out or a.run / 'debug.rrd'))
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    report = {}
    map_rot, map_trans = np.eye(3), np.zeros(3)
    for name, color, filename in (('odometry', ODOM, 'trajectory.tum'),
                                  ('map', MAP, 'map_trajectory.tum'),
                                  ('final', (90, 255, 120), 'map_trajectory_final.tum')):
        path = a.run / filename
        if not path.exists():
            continue
        trajectory = read_trajectory(path)
        rot, trans, ate = align(ds, trajectory, start)
        report[name] = ate
        points = [rot @ row[1] + trans for row in trajectory]
        polyline(rr, 'world/' + name, [row[0] for row in trajectory], points, color)
        if name in ('map', 'final'):
            map_rot, map_trans = rot, trans
    polyline(rr, 'world/groundtruth', gt.t[inside], gt.p[inside], GT, radius=.015)

    index_path = a.run / 'map/map_index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for key, entry in index['submaps'].items():
            if 'T_map_submap' not in entry:
                continue
            with np.load(a.run / 'map' / entry['file']) as stored:
                sm = {k: stored[k] for k in stored.files}
            node = np.asarray(entry['T_map_submap'], float)
            correction = np.eye(4)
            correction[:3, :3], correction[:3, 3] = map_rot, map_trans
            xyz, rgb = world_points(sm, entry['center'], correction @ node, a.pixel_step)
            rr.set_time('sensor_time', timestamp=float(entry['timestamp']))
            rr.log(f'world/submaps/{key}', rr.Points3D(xyz.astype(np.float32), colors=rgb,
                                                       radii=.01))

    # Scalars and loop edges, so a bad run is diagnosable without re-reading JSON.
    centres = {}
    for line in (a.run / 'events.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event['kind'] == 'pose':
            rr.set_time('sensor_time', timestamp=float(event['t']))
            if event.get('receipt_to_publish_s') is not None:
                rr.log('plots/publication_age_s', rr.Scalars(event['receipt_to_publish_s']))
            continue
        if event['kind'] != 'vision':
            continue
        result = event['result']
        payload = result.get('payload', {})
        rr.set_time('sensor_time', timestamp=float(result.get('sensor_time', start)))
        rr.log('plots/worker_seconds',
               rr.Scalars(result['completed_wall'] - result.get('started_wall',
                                                                result['completed_wall'])))
        if result.get('kind') == 'assist':
            rr.log('events/proposal', rr.TextLog(f"{payload.get('status')}: "
                                                 f"{payload.get('reason', '')}"))
        if payload.get('status') != 'mapped':
            continue
        rr.log('plots/submap_scale', rr.Scalars(payload['scale']))
        rr.log('plots/alignment_rmse_m', rr.Scalars(payload['alignment_rmse_m']))
        rr.log('plots/graph_nodes', rr.Scalars(float(payload['nodes'])))
        centres[payload['submap']] = payload
        for loop in payload.get('loops', ()):
            ends = [index['submaps'][k]['T_map_submap'] for k in (loop['a'], loop['b'])
                    if k in index['submaps']]
            if len(ends) == 2:
                segment = np.array([map_rot @ np.asarray(e, float)[:3, 3] + map_trans
                                    for e in ends], np.float32)
                rr.log(f"world/loops/{loop['a']}_{loop['b']}",
                       rr.LineStrips3D([segment], colors=[LOOP], radii=.03), static=True)

    print(json.dumps(dict(sequence=meta['sequence'], mode=meta['mode'], state=meta['state'],
                          selected=meta['selected'], ate_m=report,
                          submaps=len(centres)), indent=2))


if __name__ == '__main__':
    main()
