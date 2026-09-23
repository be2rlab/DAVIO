import numpy as np

from .jpl import quat_L, quat_R, quat_to_rot, rot_to_quat


def _rotation_angle(r):
    cos_theta = (np.trace(np.asarray(r, dtype=float)) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def angle_gate(r_cam, r_imu, min_angle_rad, tol_rad):
    """Rem. 1: admit a pair only if both rotations are informative and agree in angle."""
    a_cam, a_imu = _rotation_angle(r_cam), _rotation_angle(r_imu)
    if a_cam < min_angle_rad or a_imu < min_angle_rad:
        return False
    return abs(a_cam - a_imu) <= tol_rad


def robust_angle_weight(disagree_rad, tol_rad):
    delta = 0.5 * float(tol_rad)
    if not np.isfinite(disagree_rad) or not np.isfinite(delta) or delta <= 0:
        return 1.0
    return 1.0 if disagree_rad <= delta else float(delta / disagree_rad)


def build_handeye_system(pairs, min_angle_rad=0.0, tol_rad=np.inf, robust_weighting=False):
    blocks, used = [], []
    for pair in pairs:
        r_cam = np.asarray(pair["R_cam"], dtype=float)
        r_imu = np.asarray(pair["R_imu"], dtype=float)
        a_cam, a_imu = _rotation_angle(r_cam), _rotation_angle(r_imu)
        if a_cam < min_angle_rad or a_imu < min_angle_rad:
            continue
        disagree = abs(a_cam - a_imu)
        if disagree > tol_rad:
            continue
        weight = float(pair.get("weight", 1.0))
        if robust_weighting:
            weight *= robust_angle_weight(disagree, tol_rad)
        m = quat_L(rot_to_quat(r_imu)) - quat_R(rot_to_quat(r_cam))
        blocks.append(np.sqrt(max(weight, 0.0)) * m)
        used.append(pair)
    if not blocks:
        raise ValueError("no keyframe pairs passed the hand-eye angle gate")
    return np.vstack(blocks), used


def certificate(sigma3, sigma4, n_pairs, config, override=None):
    n = max(int(n_pairs), 0)
    floor_ok = n >= int(getattr(config, "handeye_min_pairs", 0))

    if config.handeye_certificate == "pairs":
        thr = (config.handeye_min_pairs if override is None else float(override))
        return float(n), float(thr), n >= thr

    s3, s4 = float(sigma3), float(sigma4)
    if not (s3 > 0) or not np.isfinite(s4):
        return float("inf"), float("nan"), False
    ratio = s4 / s3
    if config.handeye_certificate == "ratio":
        thr = config.handeye_ratio if override is None else float(override)
        return ratio, thr, (ratio <= thr) and floor_ok
    if config.handeye_certificate == "ratio_n":
        value = ratio / np.sqrt(max(n, 1))
        thr = config.handeye_ratio_n if override is None else float(override)
        return value, thr, (value <= thr) and floor_ok
    raise ValueError("unknown handeye_certificate: %r" % (config.handeye_certificate,))


def solve_handeye(b):
    """Smallest right singular vector of B, with the two smallest singular values."""
    b = np.asarray(b, dtype=float)
    # b is (4n,4): full_matrices=True (the default) computes and discards a (4n,4n) U
    # we never use; the singular values and V^T (both 4-wide) are identical either way.
    _, singular, vt = np.linalg.svd(b, full_matrices=False)
    q = vt[-1, :]
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    sigma4 = float(singular[-1])
    sigma3 = float(singular[-2]) if singular.size >= 2 else 0.0
    return q, sigma3, sigma4


class HandeyeAccumulator:
    def __init__(self):
        self.gram = np.zeros((4, 4))
        self.n_pairs = 0

    def add_pair(self, r_cam, r_imu, weight=1.0, min_angle_rad=0.0, tol_rad=np.inf):
        """Fold one (camera, IMU) relative-rotation pair in. Returns True if admitted."""
        r_cam = np.asarray(r_cam, dtype=float)
        r_imu = np.asarray(r_imu, dtype=float)
        if not angle_gate(r_cam, r_imu, min_angle_rad, tol_rad):
            return False
        m = quat_L(rot_to_quat(r_imu)) - quat_R(rot_to_quat(r_cam))
        self.gram += max(float(weight), 0.0) * (m.T @ m)
        self.n_pairs += 1
        return True

    def add_pairs(self, pairs, min_angle_rad=0.0, tol_rad=np.inf):
        return sum(self.add_pair(p["R_cam"], p["R_imu"], p.get("weight", 1.0),
                                 min_angle_rad, tol_rad) for p in pairs)

    def solve(self):
        if self.n_pairs < 2:
            return None
        evals, evecs = np.linalg.eigh(0.5 * (self.gram + self.gram.T))
        evals = np.clip(evals, 0.0, None)          # symmetric PSD up to round-off
        q = evecs[:, 0] / np.linalg.norm(evecs[:, 0])
        if q[3] < 0:
            q = -q
        return q, float(np.sqrt(evals[1])), float(np.sqrt(evals[0]))

    def copy(self):
        other = HandeyeAccumulator()
        other.gram = self.gram.copy()
        other.n_pairs = self.n_pairs
        return other
