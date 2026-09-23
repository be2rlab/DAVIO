#!/usr/bin/env python3
"""Retrospective body trajectory from final graph center corrections (never online)."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from davio_mapper import sim3
from davio.runtime.geometry import tum_line


def final_trajectory(run):
    index = json.loads((run / 'map/map_index.json').read_text())
    entries = sorted(index['submaps'].values(), key=lambda x: x['timestamp'])
    if not entries or any('T_odom_submap' not in e for e in entries):
        raise ValueError('Map must store original odometry centers; rerun with current version')
    times = np.array([e['timestamp'] for e in entries])
    transforms = np.array([sim3.pose(np.asarray(e['T_map_submap'])) @
                           np.linalg.inv(e['T_odom_submap']) for e in entries])
    rotation = Slerp(times, Rotation.from_matrix(transforms[:,:3,:3])) if len(times)>1 else None
    rows = np.atleast_2d(np.loadtxt(run / 'trajectory.tum'))
    for row in rows:
        t = float(row[0]); clamped = np.clip(t, times[0], times[-1])
        correction = np.eye(4)
        correction[:3,:3] = rotation(clamped).as_matrix() if rotation else transforms[0,:3,:3]
        correction[:3,3] = [np.interp(clamped, times, transforms[:,k,3]) for k in range(3)]
        body = np.eye(4); body[:3,3] = row[1:4]; body[:3,:3] = Rotation.from_quat(row[4:8]).as_matrix()
        yield t, correction @ body


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(); path=a.run/'map_trajectory_final.tum'
    with path.open('x') as f:
        for t,pose in final_trajectory(a.run): f.write(tum_line(t,pose))
    path.with_suffix('.json').write_text(json.dumps(dict(
        kind='retrospective', method='SLERP rotation and linear translation of final center corrections; endpoint clamping',
        source='map/map_index.json', ground_truth_used=False), indent=2))

if __name__=='__main__': main()
