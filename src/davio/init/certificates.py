import numpy as np

from .bias_gyro import log_so3
from .handeye import _rotation_angle
from .jpl import quat_L, quat_R, rot_to_quat


def _pooled_gram(pairs):
    gram = np.zeros((4, 4))
    n = 0
    for pair in pairs:
        r_imu = np.asarray(pair["R_imu"], dtype=float)
        r_cam = np.asarray(pair["R_cam"], dtype=float)
        weight = float(pair.get("weight", 1.0))
        m = quat_L(rot_to_quat(r_imu)) - quat_R(rot_to_quat(r_cam))
        gram += max(weight, 0.0) * (m.T @ m)
        n += 1
    return gram, n


def estimator_covariance_angular_std_deg(pairs):
    gram, n_pairs = _pooled_gram(pairs)
    if n_pairs < 3:
        return float("inf")
    gram = 0.5 * (gram + gram.T)
    eigvals, eigvecs = np.linalg.eigh(gram)          # ascending
    eigvals = np.clip(eigvals, 0.0, None)
    lam4 = eigvals[0]
    gaps = eigvals[1:] - lam4
    if np.any(gaps <= 1e-12):
        return float("inf")
    sigma2 = lam4 / (2.0 * n_pairs)
    cov_trace = sigma2 * float(np.sum(1.0 / gaps))
    if cov_trace < 0 or not np.isfinite(cov_trace):
        return float("inf")
    sigma_theta_rad = 2.0 * np.sqrt(cov_trace)
    return float(np.degrees(sigma_theta_rad))


def rotation_axis_coverage_matrix(pairs):
    sigma = np.zeros((3, 3))
    for pair in pairs:
        r_imu = np.asarray(pair["R_imu"], dtype=float)
        angle = _rotation_angle(r_imu)
        if angle < 1e-9:
            continue
        axis = log_so3(r_imu) / angle
        weight = float(pair.get("weight", 1.0))
        sigma += max(weight, 0.0) * np.outer(axis, axis)
    return sigma


def axis_coverage_ratio(pairs):
    sigma = rotation_axis_coverage_matrix(pairs)
    eigvals = np.sort(np.linalg.eigvalsh(0.5 * (sigma + sigma.T)))[::-1]   # descending
    lam1, lam2 = float(eigvals[0]), float(eigvals[1])
    if lam1 <= 1e-12:
        return 0.0
    return max(lam2, 0.0) / lam1


def _ang_deg(r_a, r_b):
    cos = (np.trace(np.asarray(r_a, dtype=float).T @ np.asarray(r_b, dtype=float)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
