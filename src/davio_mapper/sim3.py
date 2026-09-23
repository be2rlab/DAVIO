"""Sim(3), x_global = s R x_local + t; tangent order [v, omega, log_scale]."""
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def phi1(a):
    a = np.asarray(a, dtype=float)
    norm = np.linalg.norm(a, ord=np.inf)
    if not np.isfinite(norm):
        raise ValueError("Non-finite Lie algebra element")
    squarings = max(0, int(np.ceil(np.log2(max(norm, .5) / .5))))
    b = a / (2. ** squarings)
    eye = np.eye(len(a))
    exponential, integral, term = eye.copy(), eye.copy(), eye.copy()
    for n in range(1, 30):
        term = term @ b / n
        exponential += term
        integral += term / (n + 1)
        if np.linalg.norm(term, ord=np.inf) < 1e-16:
            break
    for _ in range(squarings):
        integral = .5 * (eye + exponential) @ integral
        exponential = exponential @ exponential
    return integral


def scale(t):
    determinant = np.linalg.det(t[:3, :3])
    if determinant <= 0 or not np.isfinite(determinant):
        raise ValueError("Sim(3) requires a positive finite scale")
    return float(np.cbrt(determinant))


def validate(t):
    t = np.asarray(t, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError("Expected a finite homogeneous 4x4 similarity matrix")
    r = t[:3, :3] / scale(t)
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5):
        raise ValueError("Affine/projective matrices are not Sim(3)")
    return t.copy()


def exp(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (7,) or not np.isfinite(x).all() or abs(x[6]) > 50:
        raise ValueError("Invalid Sim(3) increment")
    t = np.eye(4)
    t[:3, :3] = np.exp(x[6]) * Rotation.from_rotvec(x[3:6]).as_matrix()
    t[:3, 3] = phi1(skew(x[3:6]) + x[6] * np.eye(3)) @ x[:3]
    return t


def log(t):
    s = scale(t)
    sigma = np.log(s)
    omega = Rotation.from_matrix(t[:3, :3] / s).as_rotvec()
    velocity = np.linalg.solve(phi1(skew(omega) + sigma * np.eye(3)), t[:3, 3])
    return np.r_[velocity, omega, sigma]


def inverse(t):
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T / scale(t) ** 2
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def pose(t):
    out = t.copy()
    out[:3, :3] /= scale(t)
    return out


def adjoint(t):
    r = t[:3, :3] / scale(t)
    out = np.zeros((7, 7))
    out[:3, :3] = t[:3, :3]
    out[:3, 3:6] = skew(t[:3, 3]) @ r
    out[:3, 6] = -t[:3, 3]
    out[3:6, 3:6] = r
    out[6, 6] = 1
    return out


def ad(x):
    out = np.zeros((7, 7))
    out[:3, :3] = skew(x[3:6]) + x[6] * np.eye(3)
    out[:3, 3:6] = skew(x[:3])
    out[:3, 6] = -x[:3]
    out[3:6, 3:6] = skew(x[3:6])
    return out


def log_jacobians(x):
    left = np.linalg.solve(phi1(ad(x)), np.eye(7))
    # J_r^-1(x) = J_l^-1(x) + ad(x).
    return left, left + ad(x)


def projection_jacobian(t):
    # A right Sim(3) translation increment moves a physical camera by s*R*v.
    return np.diag([scale(t)] * 3 + [1.] * 3 + [0.])
