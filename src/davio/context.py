import numpy as np


def select(poses, first_stamp, centre, cfg):
    newest = first_stamp - int(float(cfg['context_min_age_s']) * 1e9)
    oldest = first_stamp - int(float(cfg['context_max_age_s']) * 1e9)
    near, far = (float(x) for x in cfg['context_range_m'])
    axis, origin = centre[:3, 2], centre[:3, 3]
    candidates = []
    for stamp, pose in poses.items():
        if stamp > newest or stamp < oldest:
            continue
        distance = np.linalg.norm(pose[:3, 3] - origin)
        angle = np.degrees(np.arccos(np.clip(pose[:3, 2] @ axis, -1., 1.)))
        if near <= distance <= far and angle <= float(cfg['context_max_angle_deg']):
            candidates.append((angle, stamp))
    taken = []
    for _, stamp in sorted(candidates):
        if len(taken) == int(cfg['context_frames']):
            break
        if all(np.linalg.norm(poses[stamp][:3, 3] - poses[t][:3, 3]) >= near for t in taken):
            taken.append(stamp)
    return taken


class Head:
    def __init__(self, prediction, n):
        for name in ('depth', 'conf', 'sky', 'extrinsics', 'intrinsics', 'processed_images'):
            value = getattr(prediction, name, None)
            setattr(self, name, None if value is None else np.asarray(value)[:n])
