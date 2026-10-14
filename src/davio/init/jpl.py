import numpy as np


def skew(v):
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def exp_so3(phi):
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = float(np.linalg.norm(phi))
    if angle < 1e-12:
        return np.eye(3) + skew(phi)
    axis = phi / angle
    k = skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def quat_to_rot(q):
    q = np.asarray(q, dtype=float).reshape(4)
    qv, qw = q[:3], q[3]
    return (2.0 * qw * qw - 1.0) * np.eye(3) - 2.0 * qw * skew(qv) + 2.0 * np.outer(qv, qv)


def rot_to_quat(r):
    r = np.asarray(r, dtype=float).reshape(3, 3)
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(1.0 + trace) * 2.0
        qw = 0.25 * s
        qx = (r[1, 2] - r[2, 1]) / s
        qy = (r[2, 0] - r[0, 2]) / s
        qz = (r[0, 1] - r[1, 0]) / s
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2.0
        q = np.zeros(3)
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
        qw = (r[j, k] - r[k, j]) / s
        qx, qy, qz = q
    q = np.array([qx, qy, qz, qw])
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def quat_L(q):
    """Left-multiplication matrix: L(a) b = a (x) b  (Eq. 25)."""
    q = np.asarray(q, dtype=float).reshape(4)
    qv, qw = q[:3], q[3]
    m = np.zeros((4, 4))
    m[:3, :3] = qw * np.eye(3) - skew(qv)
    m[:3, 3] = qv
    m[3, :3] = -qv
    m[3, 3] = qw
    return m


def quat_R(q):
    """Right-multiplication matrix: R(b) a = a (x) b  (Eq. 25)."""
    q = np.asarray(q, dtype=float).reshape(4)
    qv, qw = q[:3], q[3]
    m = np.zeros((4, 4))
    m[:3, :3] = qw * np.eye(3) + skew(qv)
    m[:3, 3] = qv
    m[3, :3] = -qv
    m[3, 3] = qw
    return m


def quat_mult(a, b):
    return quat_L(a) @ np.asarray(b, dtype=float).reshape(4)
