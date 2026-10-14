import numpy as np
from scipy.spatial import cKDTree

MAX_POINTS = 200_000
GATES_M = (.30, .15)            # the first 10 iterations, then the rest
WIDE_ITERATIONS = 10
MAX_ITERATIONS = 40
TRIM = .10                      # the farthest share of the gated pairs is dropped


def umeyama(source, target, with_scale=True):
    """(s, R, t) minimizing |s R source + t - target|^2; s = 1 when with_scale is False."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    a, b = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(b.T @ a / len(a))
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2] = -1.
    rotation = u @ np.diag(sign) @ vt
    scale = float(np.sum(d * sign) / np.mean(np.sum(a ** 2, axis=1))) if with_scale else 1.
    return scale, rotation, mu_t - scale * rotation @ mu_s


def fit(points, reference, with_scale=True, tree=None, seed=0):
    tree = tree if tree is not None else cKDTree(reference)
    rng = np.random.default_rng(seed)
    sample = points if len(points) <= MAX_POINTS else points[rng.choice(len(points), MAX_POINTS, replace=False)]
    total = np.eye(4)
    current = sample.copy()
    info = dict(iterations=0, pairs=0, converged=False)
    for iteration in range(MAX_ITERATIONS):
        gate = GATES_M[0] if iteration < WIDE_ITERATIONS else GATES_M[1]
        distance, index = tree.query(current, workers=-1)
        keep = np.flatnonzero(distance <= gate)
        if len(keep) < 100:
            break
        keep = keep[distance[keep] <= np.quantile(distance[keep], 1. - TRIM)]
        scale, rotation, translation = umeyama(current[keep], reference[index[keep]], with_scale)
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = scale * rotation, translation
        total = step @ total
        current = current @ step[:3, :3].T + translation
        info.update(iterations=iteration + 1, pairs=int(len(keep)))
        if iteration >= WIDE_ITERATIONS and abs(scale - 1.) < 1e-5 and np.linalg.norm(translation) < 1e-4:
            info['converged'] = True
            break
    linear = total[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(linear / scale) - 1.) / 2., -1., 1.))))
    info.update(scale=scale, rotation_deg=angle, translation_m=float(np.linalg.norm(total[:3, 3])),
                with_scale=bool(with_scale))
    return total, info
