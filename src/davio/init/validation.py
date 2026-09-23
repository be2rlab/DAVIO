import numpy as np
from .types import InitConfig
from .joint_calibration import RANK_RTOL, build_joint_pairs, joint_residual

def rotational_validation(imu, validation_window, r_ctoi, bias_gyro, config=None):
    pairs = build_joint_pairs(imu, [validation_window], bias_gyro, config or InitConfig())
    if not pairs:
        return float("nan"), float("nan"), 0
    norms = np.array([np.linalg.norm(joint_residual(p["R_cam"], p["R_imu"], r_ctoi))
                      for p in pairs])
    return (float(np.degrees(np.sqrt(float(norms @ norms) / len(norms)))),
            float(np.degrees(norms.max())), len(pairs))


def covariance_recoverable(fit):
    # joint_certificate deliberately stops before forming any block when the raw
    # H is non-finite (it cannot equilibrate by a non-finite diagonal), so that
    # case arrives here as "the blocks are missing" and has to be named for what
    # it actually is rather than as a never-computed certificate.
    if fit.h_raw is not None and not np.all(np.isfinite(np.asarray(fit.h_raw, float))):
        return False, "non-finite joint information matrix: covariance not recoverable"
    for name, block in (("H_X|b", fit.h_x_given_b), ("H_bb", fit.h_bb)):
        if block is None:
            return False, "no %s: certificate was never computed" % name
        m = np.asarray(block, dtype=float)
        if not np.all(np.isfinite(m)):
            return False, "non-finite %s: covariance not recoverable" % name
        eig = np.linalg.eigvalsh(0.5 * (m + m.T))
        top = float(eig.max())
        if top <= 0:
            return False, "%s has no positive eigenvalue (max %.3e)" % (name, top)
        # RANK_RTOL is this package's one relative rank tolerance; a negative
        # eigenvalue below -RANK_RTOL*top is structurally negative, not round-off.
        if float(eig.min()) < -RANK_RTOL * top:
            return False, ("%s is indefinite: min eigenvalue %.3e vs max %.3e -- "
                           "not a covariance" % (name, float(eig.min()), top))
        if float(eig.min()) <= RANK_RTOL * top:
            return False, ("%s is singular to the rank tolerance: min eigenvalue %.3e "
                           "vs max %.3e -- covariance not recoverable"
                           % (name, float(eig.min()), top))
    return True, ""

