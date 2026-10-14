import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import sim3


def final_trajectory(run):
    run = Path(run)
    index = json.loads((run / 'map/map_index.json').read_text())
    entries = sorted(index['submaps'].values(), key=lambda x: x['timestamp'])
    times = np.array([e['timestamp'] for e in entries])
    transforms = np.array([sim3.pose(np.asarray(e['T_map_submap'])) @
                           np.linalg.inv(e['T_odom_submap']) for e in entries])
    rotation = Slerp(times, Rotation.from_matrix(transforms[:, :3, :3])) if len(times) > 1 else None
    rows = np.atleast_2d(np.loadtxt(run / 'trajectory.tum'))
    for row in rows:
        t = float(row[0])
        clamped = np.clip(t, times[0], times[-1])
        correction = np.eye(4)
        correction[:3, :3] = rotation(clamped).as_matrix() if rotation else transforms[0, :3, :3]
        correction[:3, 3] = [np.interp(clamped, times, transforms[:, k, 3]) for k in range(3)]
        body = np.eye(4)
        body[:3, 3] = row[1:4]
        body[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
        yield t, correction @ body


def write_final_trajectory(run):
    from ..geometry import tum_line
    run = Path(run)
    index = run / 'map/map_index.json'
    if not index.is_file() or not json.loads(index.read_text()).get('submaps'):
        return None
    path = run / 'map_trajectory_final.tum'
    with path.open('w') as stream:
        for t, pose in final_trajectory(run):
            stream.write(tum_line(t, pose))
    return path
