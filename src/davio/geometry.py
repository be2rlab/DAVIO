"""Poses as 4x4 matrices, body or camera to world; the filter's quaternions are JPL (x, y, z, w)."""
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])


def quat_to_rot(q):
    """Rotation matrix of a JPL quaternion."""
    q = np.asarray(q, dtype=float).reshape(4)
    qv, qw = q[:3], q[3]
    return (2.0 * qw * qw - 1.0) * np.eye(3) - 2.0 * qw * skew(qv) + 2.0 * np.outer(qv, qv)


def rot_to_quat(r):
    """JPL quaternion of a rotation matrix."""
    r = np.asarray(r, dtype=float).reshape(3, 3)
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(1.0 + trace) * 2.0
        q = np.array([(r[1, 2] - r[2, 1]) / s, (r[2, 0] - r[0, 2]) / s, (r[0, 1] - r[1, 0]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2.0
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
        q[3] = (r[j, k] - r[k, j]) / s
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def body_pose(state):
    """Body to world, from the filter's state (q_GtoI, p)."""
    pose = np.eye(4)
    pose[:3, :3] = quat_to_rot(np.asarray(state['q_GtoI'])).T
    pose[:3, 3] = state['p']
    return pose


def camera_pose(state):
    """Camera to world, from the filter's state and its camera-IMU calibration (q_ItoC, p_IinC)."""
    camera_from_body = np.eye(4)
    camera_from_body[:3, :3] = quat_to_rot(np.asarray(state['q_ItoC']))
    camera_from_body[:3, 3] = state['p_IinC']
    return body_pose(state) @ np.linalg.inv(camera_from_body)


def tum_line(t, pose):
    values = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
    return f'{t:.9f} ' + ' '.join(f'{x:.12g}' for x in values) + '\n'


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
