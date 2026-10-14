"""Trajectory scores against the reference. The reference enters the repository here and in the map scorer only."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry import quat_to_rot, rot_to_quat

MATCH_TOLERANCE_S = .05
MIN_MATCHED_POSES = 10


def read_trajectory(path):
    """A TUM file of body poses as (t, p, q_GtoI) rows, the quaternion as the filter writes it (JPL)."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        row = np.array([float(v) for v in line.split()])
        if len(row) != 8 or not np.isfinite(row).all():
            raise ValueError('Invalid trajectory row')
        out.append((row[0], row[1:4], rot_to_quat(Rotation.from_quat(row[4:]).as_matrix().T)))
    return out


def matched_points(ds, trajectory, t_start, tolerance=MATCH_TOLERANCE_S,
                   min_matched=MIN_MATCHED_POSES):
    """Estimated and reference positions on a common index, inside reference support."""
    gt = ds.groundtruth()
    kept = [(t, p, q) for t, p, q in trajectory if t >= t_start - 1e-9]
    if len(kept) < min_matched:
        return None, None, None, None, len(kept)
    pairs = []
    for k, (t, _p, _q) in enumerate(kept):
        j = int(np.argmin(np.abs(gt.t - t)))
        if abs(float(gt.t[j]) - t) <= tolerance:
            pairs.append((k, j))
    if len(pairs) < min_matched:
        return None, None, None, None, len(kept)
    est_p = np.asarray([kept[k][1] for k, _ in pairs]).T
    ref_p = gt.p[[j for _, j in pairs]].T
    est_q = [kept[k][2] for k, _ in pairs]
    ref_r = gt.R[[j for _, j in pairs]]
    times = [kept[k][0] for k, _ in pairs]
    return est_p, ref_p, (est_q, ref_r), times, len(kept)


def conditioning(points):
    centred = np.asarray(points, float) - np.asarray(points, float).mean(1, keepdims=True)
    s = np.linalg.svd(centred, compute_uv=False)
    if s.size < 3 or not np.isfinite(s).all() or s[0] <= 0:
        return 0.
    return float(s[2] / s[0])


def se3_align(est_p, ref_p):
    """Rigid SE(3) alignment with NO fitted scale -> (R, t, rmse_m)."""
    mu_e, mu_r = est_p.mean(1, keepdims=True), ref_p.mean(1, keepdims=True)
    u, _s, vt = np.linalg.svd((est_p - mu_e) @ (ref_p - mu_r).T)
    d = np.diag([1., 1., float(np.sign(np.linalg.det(vt.T @ u.T)))])
    rot = vt.T @ d @ u.T
    trans = (mu_r - rot @ mu_e).reshape(3, 1)
    err = rot @ est_p + trans - ref_p
    return rot, trans, float(np.sqrt((err ** 2).sum(0).mean()))


def ate(ds, trajectory, t_start, tolerance=MATCH_TOLERANCE_S, min_matched=MIN_MATCHED_POSES):
    """Absolute trajectory error under one rigid alignment, orientation under the same fit."""
    est_p, ref_p, orientations, _times, n_states = matched_points(
        ds, trajectory, t_start, tolerance, min_matched)
    if est_p is None:
        return dict(position_m=None, orientation_deg=None, n_matched=0, n_states=n_states,
                    conditioning=None,
                    reason=f'fewer than {min_matched} poses matched the reference within {tolerance:.3f} s')
    rot, _t, rmse = se3_align(est_p, ref_p)
    if not ds.groundtruth().orientation_reliable:
        return dict(position_m=float(rmse), orientation_deg=None, n_matched=est_p.shape[1],
                    n_states=n_states, conditioning=conditioning(ref_p),
                    reason="this reference's orientation is documented as unreliable")
    est_q, ref_r = orientations
    residuals = []
    for q, reference in zip(est_q, ref_r):
        # q_GtoI gives R_GtoI; its transpose is body-to-world, then the alignment maps
        # the estimate's world frame onto the reference's.
        cos = (np.trace((rot @ quat_to_rot(q).T).T @ reference) - 1.) / 2.
        residuals.append(np.degrees(np.arccos(np.clip(cos, -1., 1.))))
    return dict(position_m=float(rmse),
                orientation_deg=float(np.sqrt(np.mean(np.square(residuals)))),
                n_matched=est_p.shape[1], n_states=n_states,
                conditioning=conditioning(ref_p), reason='')
