import numpy as np
from scipy.spatial.transform import Rotation
from ..init.jpl import quat_to_rot

def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def camera_pose(state):
    body = np.eye(4)
    body[:3, :3] = quat_to_rot(np.asarray(state['q_GtoI'])).T
    body[:3, 3] = state['p']
    camera_from_body = np.eye(4)
    camera_from_body[:3, :3] = quat_to_rot(np.asarray(state['q_ItoC']))
    camera_from_body[:3, 3] = state['p_IinC']
    return body @ np.linalg.inv(camera_from_body)


def tum_line(t, pose):
    values = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
    return f'{t:.9f} ' + ' '.join(f'{x:.12g}' for x in values) + '\n'

