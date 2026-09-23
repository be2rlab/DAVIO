import numpy as np

from .jpl import skew

def log_so3(r):
    r = np.asarray(r, dtype=float).reshape(3, 3)
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    if angle < 1e-9:
        return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) * 0.5
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return angle * axis / (2.0 * np.sin(angle))


def right_jacobian_inv_so3(phi):
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    k = skew(phi)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * k
    return (np.eye(3) + 0.5 * k
            + (1.0 / theta ** 2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))) * (k @ k))


def gyro_bias_rows(camera_rotations, preints, r_ctoi):
    r_ctoi = np.asarray(r_ctoi, dtype=float).reshape(3, 3)
    rows_j, rows_e = [], []
    for i in range(1, len(preints)):
        r_pred = r_ctoi @ np.asarray(camera_rotations[i], dtype=float) @ r_ctoi.T
        dr = preints[i].dR
        e = log_so3(r_pred.T @ dr)
        j = right_jacobian_inv_so3(e) @ dr.T @ preints[i].dR_dbg
        rows_j.append(j)
        rows_e.append(e)
    return rows_j, rows_e


def gyro_bias_update(camera_rotations, preints, r_ctoi):
    rows_j, rows_e = gyro_bias_rows(camera_rotations, preints, r_ctoi)
    jac = np.vstack(rows_j)
    err = np.concatenate(rows_e)
    delta, *_ = np.linalg.lstsq(jac, -err, rcond=None)
    return delta


class GyroBiasAccumulator:

    def __init__(self):
        self.jtj = np.zeros((3, 3))
        self.jtr = np.zeros(3)
        self.n_rows = 0

    def add_window(self, camera_rotations, preints, r_ctoi):
        rows_j, rows_e = gyro_bias_rows(camera_rotations, preints, r_ctoi)
        for j, e in zip(rows_j, rows_e):
            self.jtj += j.T @ j
            self.jtr += j.T @ (-e)
            self.n_rows += 1

    def solve(self):
        if self.n_rows < 1:
            return np.zeros(3)
        if np.linalg.matrix_rank(self.jtj) < 3:
            # Not enough rotational diversity yet to separate all 3 bias components:
            # a regularized (minimum-norm) partial answer, not solve() raising on a
            # singular 3x3.
            delta, *_ = np.linalg.lstsq(self.jtj, self.jtr, rcond=None)
            return delta
        return np.linalg.solve(self.jtj, self.jtr)

    def copy(self):
        other = GyroBiasAccumulator()
        other.jtj, other.jtr, other.n_rows = self.jtj.copy(), self.jtr.copy(), self.n_rows
        return other
