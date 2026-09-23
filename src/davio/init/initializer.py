import numpy as np
from .preintegration import Preintegrated, preintegrate_series

def _identity_preintegration():
    return Preintegrated(dt=0.0, dR=np.eye(3), dv=np.zeros(3), dp=np.zeros(3),
                         dR_dbg=np.zeros((3, 3)))


def _preintegrate_frames(imu, frame_times, bias_gyro, bias_accel=None):
    return [_identity_preintegration()] + preintegrate_series(
        imu, frame_times[0], frame_times[1:], bias_gyro=bias_gyro, bias_accel=bias_accel)


def _handeye_pairs(preints, camera_rotations, frame_times=None):
    pairs = []
    n = len(preints)
    for i in range(n):
        for j in range(i + 1, n):
            r_imu = preints[j].dR @ preints[i].dR.T
            r_cam = np.asarray(camera_rotations[j]) @ np.asarray(camera_rotations[i]).T
            pair = dict(R_cam=r_cam, R_imu=r_imu, weight=1.0)
            if frame_times is not None:
                pair["t_pair"] = (float(frame_times[i]), float(frame_times[j]))
            pairs.append(pair)
    return pairs


def dedupe_pairs_by_timestamp(pairs):
    seen = set()
    out = []
    for p in pairs:
        key = p.get("t_pair")
        if key is None:
            out.append(p)
            continue
        key = tuple(sorted(key))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def n_unique_frames(pairs):
    times = set()
    for p in pairs:
        key = p.get("t_pair")
        if key is not None:
            times.update(key)
    return len(times)

