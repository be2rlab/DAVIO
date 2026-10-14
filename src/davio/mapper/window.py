"""The metric scale of one window."""
import numpy as np
from scipy.spatial.transform import Rotation

from . import sim3


def fit_window(local_poses, metric_poses):
    local, world = np.asarray(local_poses), np.asarray(metric_poses)
    r = Rotation.from_matrix(world[:, :3, :3] @ local[:, :3, :3].transpose(0, 2, 1)).mean().as_matrix()
    x, y = local[:, :3, 3], world[:, :3, 3]
    xc, yc = x - x.mean(0), y - y.mean(0)
    energy = float(np.sum(xc * xc))
    if not np.isfinite(energy) or energy <= 0.:
        return None
    s = float(np.sum((xc @ r.T) * yc) / energy)
    if not np.isfinite(s) or s <= 0.:
        return None
    transform = np.eye(4)
    transform[:3, :3] = s * r
    transform[:3, 3] = y.mean(0) - s * r @ x.mean(0)
    centres = x @ transform[:3, :3].T + transform[:3, 3]
    return sim3.scale(transform), float(np.sqrt(np.mean(np.sum((centres - y) ** 2, axis=1))))
