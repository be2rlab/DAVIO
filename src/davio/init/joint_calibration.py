from dataclasses import dataclass, field

import numpy as np

from .bias_gyro import log_so3, right_jacobian_inv_so3
from .certificates import _ang_deg, axis_coverage_ratio
from .handeye import _rotation_angle, angle_gate, robust_angle_weight
from .initializer import (_handeye_pairs, _preintegrate_frames,
                          dedupe_pairs_by_timestamp, n_unique_frames)
from .jpl import exp_so3
from .types import InitConfig

# Relative rank tolerance for every pseudo-inverse and rank count in this file.
# numpy's own defaults are useless here: np.linalg.pinv's rcond=1e-15 and
# matrix_rank's max(M,N)*eps*sigma_max ~ 1.3e-15 both sit at pure round-off, so
# an accumulated normal matrix -- whose singular values are already squared,
# hence whose round-off floor is ~sqrt of that in the underlying Jacobian --
# essentially never gets declared rank-deficient by them, and H_bb^+ then
# amplifies noise directions by 1e14 instead of truncating them. 1e-9 relative
# to sigma_max sits between the two things it has to separate, with margin on
# both sides (both measured on the synthetic fixtures): a genuinely null
# direction reads ~1e-16 relative (single-axis motion: s(H) = [2.98, 2.02,
# 1.00, 4.1e-16, 1.7e-16, 1.6e-16]), while a real, weakly excited direction
# reads far higher (healthy two-axis motion: cond(H_bb) = 1.01, cond(H) = 4.3e3).
# This is a NUMERICAL tolerance (what counts as a zero singular value), kept
# deliberately separate from the PHYSICAL rejection thresholds in
# joint_calibration_ok (what counts as too ill-conditioned to ship).
RANK_RTOL = 1e-9


# --------------------------------------------------------------------------
# 1. Residual and Jacobian
# --------------------------------------------------------------------------

def joint_residual(r_cam, a_imu, r_ctoi, dtheta=None):
    x = np.asarray(r_ctoi, dtype=float).reshape(3, 3)
    if dtheta is not None:
        x = x @ exp_so3(dtheta)
    b = np.asarray(r_cam, dtype=float).reshape(3, 3)
    a = np.asarray(a_imu, dtype=float).reshape(3, 3)
    return log_so3(a @ x @ b.T @ x.T)


def joint_pair_jacobian(r_cam, a_imu, g_bias, r_ctoi):
    x = np.asarray(r_ctoi, dtype=float).reshape(3, 3)
    b = np.asarray(r_cam, dtype=float).reshape(3, 3)
    a = np.asarray(a_imu, dtype=float).reshape(3, 3)
    g = np.asarray(g_bias, dtype=float).reshape(3, 3)
    r = log_so3(a @ x @ b.T @ x.T)
    jr_inv = right_jacobian_inv_so3(r)
    return r, jr_inv @ x @ (b - np.eye(3)), jr_inv.T @ g


def pair_bias_jacobian(preint_i, preint_j, a_imu):
    return (np.asarray(preint_j.dR_dbg, dtype=float)
            - np.asarray(a_imu, dtype=float) @ np.asarray(preint_i.dR_dbg, dtype=float))


def build_joint_pairs(imu, windows, bias_gyro, config=None):
    config = config or InitConfig()
    bias_gyro = np.asarray(bias_gyro, dtype=float).reshape(3)
    pairs = []
    for win in windows:
        preints = _preintegrate_frames(imu, win.frame_times, bias_gyro)
        win_pairs = _handeye_pairs(preints, win.camera_rotations,
                                   frame_times=win.frame_times)
        n = len(preints)
        index = [(i, j) for i in range(n) for j in range(i + 1, n)]   # _handeye_pairs' own order
        for (i, j), pair in zip(index, win_pairs):
            pair["G_bias"] = pair_bias_jacobian(preints[i], preints[j], pair["R_imu"])
            pairs.append(pair)
    # dedupe_pairs_by_timestamp's own concern: overlapping windows can hand back
    # the literal same two camera frames. Dropping the duplicate
    # is strictly stronger than the correlation down-weighting below, which
    # would only halve it.
    return dedupe_pairs_by_timestamp(pairs)


# --------------------------------------------------------------------------
# 2. Pair correlation
# --------------------------------------------------------------------------

def correlation_scales(pairs):
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    for idx, pair in enumerate(pairs):
        key = pair.get("t_pair")
        ends = (float(key[0]), float(key[1])) if key is not None else (("_", idx, 0), ("_", idx, 1))
        for e in ends:
            parent.setdefault(e, e)
        a, b = find(ends[0]), find(ends[1])
        if a != b:
            parent[a] = b
        edges.append(ends)

    n_frames, n_pairs = {}, {}
    for ends in edges:
        root = find(ends[0])
        n_pairs[root] = n_pairs.get(root, 0) + 1
    for node in parent:
        root = find(node)
        n_frames[root] = n_frames.get(root, 0) + 1

    out = np.empty(len(pairs))
    for idx, ends in enumerate(edges):
        root = find(ends[0])
        out[idx] = max(n_frames[root] - 1, 0) / max(n_pairs[root], 1)
    return out


# --------------------------------------------------------------------------
# 3. Weighted assembly and the robust joint solve
# --------------------------------------------------------------------------

def _admit(pairs, config):
    admitted, weights = [], []
    for pair in pairs:
        r_cam = np.asarray(pair["R_cam"], dtype=float)
        r_imu = np.asarray(pair["R_imu"], dtype=float)
        if not angle_gate(r_cam, r_imu, config.min_pair_angle_rad, config.angle_tol_rad):
            continue
        disagree = abs(_rotation_angle(r_cam) - _rotation_angle(r_imu))
        w = float(pair.get("weight", 1.0))
        w *= robust_angle_weight(disagree, config.angle_tol_rad)
        admitted.append(pair)
        weights.append(max(w, 0.0))
    return admitted, np.asarray(weights, dtype=float)


def assemble_joint_system(pairs, r_ctoi, config=None):
    config = config or InitConfig()
    admitted, weights = _admit(pairs, config)
    if admitted:
        weights = weights * correlation_scales(admitted)
    h = np.zeros((6, 6))
    grad = np.zeros(6)
    cost = 0.0
    for pair, w in zip(admitted, weights):
        r, jx, jb = joint_pair_jacobian(pair["R_cam"], pair["R_imu"], pair["G_bias"], r_ctoi)
        j = np.hstack([jx, jb])
        h += w * (j.T @ j)
        grad -= w * (j.T @ r)
        cost += w * float(r @ r)
    return h, grad, float(cost), admitted, weights


def normalized_cost(cost, weights):
    total = float(np.sum(weights)) if len(weights) else 0.0
    if not np.isfinite(cost) or total <= 0.0:
        return float("inf")
    return float(cost) / total


@dataclass
class JointCalibrationResult:
    r_ctoi: np.ndarray
    bias_gyro: np.ndarray
    converged: bool = False
    n_iterations: int = 0
    step_norm_rot_rad: float = float("nan")
    step_norm_bias_rad_s: float = float("nan")
    termination_reason: str = ""
    cost: float = float("nan")
    cost_per_weight: float = float("nan")   # what the LM accept test minimizes
    n_pairs: int = 0
    n_unique_frames: int = 0
    effective_weight: float = 0.0
    max_pair_angle_rad: float = 0.0
    axis_coverage: float = 0.0
    # --- conditioning certificate ---
    h_raw: np.ndarray = None
    h_full: np.ndarray = None
    h_xx: np.ndarray = None
    h_xb: np.ndarray = None
    h_bb: np.ndarray = None
    h_x_given_b: np.ndarray = None
    column_scale: np.ndarray = None
    singular_values_full: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))
    singular_values_x_given_b: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    singular_values_bb: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    rank_tol: float = RANK_RTOL
    rank_full: int = 0
    rank_bb: int = 0
    rank_x_given_b: int = 0
    cond_full: float = float("inf")
    cond_x_given_b: float = float("inf")
    cond_bb: float = float("inf")
    sigma_max_xx: float = 0.0
    sigma_min_bb: float = 0.0
    sigma_min_x_given_b: float = 0.0
    reason: str = ""


def joint_certificate(h_raw, result):
    h_raw = np.asarray(h_raw, dtype=float).reshape(6, 6)
    h_raw = 0.5 * (h_raw + h_raw.T)
    result.h_raw = h_raw
    diag = np.diag(h_raw).copy()
    if not np.all(np.isfinite(h_raw)):
        result.column_scale = np.ones(6)
        return result
    scale = np.where(diag > 0, 1.0 / np.sqrt(np.where(diag > 0, diag, 1.0)), 1.0)
    h = h_raw * np.outer(scale, scale)
    result.column_scale = scale
    result.h_full = h
    result.h_xx, result.h_xb, result.h_bb = h[:3, :3], h[:3, 3:], h[3:, 3:]

    s_full = np.linalg.svd(h, compute_uv=False)
    result.singular_values_full = s_full
    result.rank_full = int(np.sum(s_full > RANK_RTOL * s_full[0])) if s_full[0] > 0 else 0
    result.cond_full = float(s_full[0] / s_full[-1]) if s_full[-1] > 0 else float("inf")

    result.sigma_max_xx = float(np.linalg.svd(result.h_xx, compute_uv=False)[0])

    s_bb = np.linalg.svd(result.h_bb, compute_uv=False)
    result.singular_values_bb = s_bb
    result.rank_bb = int(np.sum(s_bb > RANK_RTOL * s_bb[0])) if s_bb[0] > 0 else 0
    result.sigma_min_bb = float(s_bb[-1])
    result.cond_bb = float(s_bb[0] / s_bb[-1]) if s_bb[-1] > 0 else float("inf")

    # Schur complement: what the rotation block knows once the bias has been
    # marginalized out, which is the only rotation conditioning number that
    # survives the bias being unknown.
    h_bb_pinv = np.linalg.pinv(result.h_bb, rcond=RANK_RTOL, hermitian=True)
    h_xb = result.h_xb
    schur = result.h_xx - h_xb @ h_bb_pinv @ h_xb.T
    h_x_given_b = 0.5 * (schur + schur.T)          # symmetric up to round-off
    result.h_x_given_b = h_x_given_b
    s_marg = np.linalg.svd(h_x_given_b, compute_uv=False)
    result.singular_values_x_given_b = s_marg
    result.sigma_min_x_given_b = float(s_marg[-1])
    result.rank_x_given_b = (int(np.sum(s_marg > RANK_RTOL * s_marg[0]))
                             if s_marg[0] > 0 else 0)
    # Reported because Sec. 4.2 asks for it, but NOT what joint_calibration_ok
    # rejects on, and the reason is worth stating: when the bias absorbs the
    # rotation entirely -- the exact ambiguity this Schur complement exists to
    # expose -- H_X|b comes out numerically ZERO, and the condition number of a
    # matrix of pure round-off is ~1, i.e. it reports "perfectly conditioned"
    # for the worst possible case. Measured on single-axis synthetic motion:
    # s(H_X|b) = [5.4e-16, 4.8e-16, 2.8e-16], cond = 1.9. The smallest singular
    # value against the equilibrated scale is the statistic that survives this.
    result.cond_x_given_b = float(s_marg[0] / s_marg[-1]) if s_marg[-1] > 0 else float("inf")
    return result


def solve_joint_calibration(imu, windows, config=None, r_init=None, bias_init=None,
                            max_iterations=15, rot_tol_rad=None, alternations=3,
                            alternating_only=False):
    config = config or InitConfig()
    rot_tol = config.bg_convergence_tol_rad_s if rot_tol_rad is None else float(rot_tol_rad)
    bias_tol = float(config.bg_convergence_tol_rad_s)

    init_reason = ""
    # `alternating_only` is the Sec. VII ablation arm: stop at the alternating
    # solution and certify THAT point with the same H_X|b certificate, so the two
    # arms differ only in the solver, never in the acceptance rule or data budget.
    if alternating_only:
        if r_init is not None and bias_init is not None:
            raise ValueError('alternating_only needs the alternating initializer to run')
        max_iterations = 0
    alternating = {}
    if r_init is None or bias_init is None:
        from .multiwindow import global_handeye_auto_bias
        try:
            r_auto, _s3, _s4, _n, b_auto, _nb = global_handeye_auto_bias(
                imu, windows, config, alternations=alternations, diag_out=alternating)
        except (ValueError, np.linalg.LinAlgError) as exc:
            # The alternating solver refuses to start at all when no pair
            # clears the hand-eye angle gate (zero-motion / pure-translation
            # windows). That is a REJECTION to certify, not an exception to
            # propagate: fall back to a neutral linearization point so the
            # certificate below can describe the degenerate geometry, and let
            # joint_calibration_ok name it.
            r_auto, b_auto = np.eye(3), np.zeros(3)
            init_reason = "handeye_init_failed: %s; " % exc
        r_ctoi = np.asarray(r_auto, dtype=float) if r_init is None else np.asarray(r_init, float)
        bias = np.asarray(b_auto, dtype=float) if bias_init is None else np.asarray(bias_init, float)
    else:
        r_ctoi = np.asarray(r_init, dtype=float).reshape(3, 3)
        bias = np.asarray(bias_init, dtype=float).reshape(3)
    r_ctoi, bias = r_ctoi.reshape(3, 3).copy(), bias.reshape(3).copy()

    pairs = build_joint_pairs(imu, windows, bias, config)
    h, grad, cost, admitted, weights = assemble_joint_system(pairs, r_ctoi, config)
    score = normalized_cost(cost, weights)

    lam = 1e-6
    n_iter, converged, reason = 0, False, "max_iterations"
    step_rot = step_bias = float("nan")
    for _ in range(int(max_iterations)):
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(grad)):
            reason = "non_finite_system"
            break
        if not admitted:
            reason = "no_admitted_pairs"
            break
        damping = np.diag(np.maximum(np.diag(h), np.max(np.diag(h)) * 1e-12 + 1e-30))
        delta, *_ = np.linalg.lstsq(h + lam * damping, grad, rcond=RANK_RTOL)
        if not np.all(np.isfinite(delta)):
            reason = "non_finite_update"
            break
        r_try = r_ctoi @ exp_so3(delta[:3])
        b_try = bias + delta[3:]
        pairs_try = build_joint_pairs(imu, windows, b_try, config)
        h_try, grad_try, cost_try, adm_try, w_try = assemble_joint_system(pairs_try, r_try, config)
        n_iter += 1
        # Accept on cost PER UNIT ADMITTED WEIGHT, never on raw cost -- see
        # normalized_cost. The admitted set and the total weight both move
        # between trials, so raw cost is not the same objective twice running.
        score_try = normalized_cost(cost_try, w_try)
        if adm_try and score_try <= score:
            r_ctoi, bias = r_try, b_try
            pairs, h, grad, cost, admitted, weights = (pairs_try, h_try, grad_try,
                                                       cost_try, adm_try, w_try)
            score = score_try
            lam = max(lam * 0.3, 1e-12)
            step_rot = float(np.linalg.norm(delta[:3]))
            step_bias = float(np.linalg.norm(delta[3:]))
            if step_rot < rot_tol and step_bias < bias_tol:
                converged, reason = True, "converged"
                break
        else:
            lam *= 10.0
            if lam > 1e12:
                # Damping this high means the local model cannot find ANY
                # downhill direction: the current point is the best effort, and
                # the certificate below is what says whether it is trustworthy.
                reason = "lm_damping_exhausted"
                break

    if alternating_only:
        converged = bool(alternating.get('converged'))
        reason = 'alternating_' + alternating.get('termination_reason', 'unavailable')
    result = JointCalibrationResult(
        r_ctoi=r_ctoi, bias_gyro=bias, converged=converged, n_iterations=n_iter,
        step_norm_rot_rad=step_rot, step_norm_bias_rad_s=step_bias,
        termination_reason=init_reason + reason, cost=cost, cost_per_weight=score,
        n_pairs=len(admitted),
        n_unique_frames=n_unique_frames(admitted),
        effective_weight=float(np.sum(weights)) if len(weights) else 0.0,
        max_pair_angle_rad=max((_rotation_angle(p["R_imu"]) for p in pairs), default=0.0),
        axis_coverage=axis_coverage_ratio(pairs) if pairs else 0.0)
    return joint_certificate(h, result)


# --------------------------------------------------------------------------
# 4. Acceptance
# --------------------------------------------------------------------------

def joint_calibration_ok(result, config=None):
    """(ok, reason), also stamped onto `result.reason`. See _verdict below."""
    ok, reason = _verdict(result, config)
    result.reason = reason
    return ok, reason


def _verdict(result, config=None):
    config = config or InitConfig()
    min_angle = float(getattr(config, "joint_min_rotation_rad", config.min_pair_angle_rad))
    min_axis = float(getattr(config, "joint_min_axis_coverage", 0.02))
    # Both defaults are measured, not guessed, on the synthetic fixtures in
    # tests/init/test_joint_calibration.py (3 windows x 5 frames, column-
    # equilibrated H so every singular value is dimensionless and O(1)):
    #   healthy two-axis motion   sigma_min(H_X|b) = 1.5e-3, sigma_min(H_bb) = 0.995
    #   single-axis motion        sigma_min(H_X|b) = 5.4e-16 (bias absorbs the
    #                             rotation completely -- H_X|b is round-off)
    # 1e-6 sits ~3 decades below the healthy case and ~10 decades above the
    # degenerate one, i.e. nowhere near either. That asymmetry is deliberate:
    # the cost of a false reject (defer the release one window) is far below
    # the cost of a false accept (ship a rotation the data never determined).
    # Threshold-freeze territory once there are real-sequence numbers to fit.
    min_sigma_marg = float(getattr(config, "joint_min_sigma_x_given_b", 1e-6))
    min_sigma_bb = float(getattr(config, "joint_min_sigma_bb", 1e-6))
    min_pairs = int(getattr(config, "joint_min_pairs", 3))

    fields = [result.r_ctoi, result.bias_gyro, result.h_raw, result.singular_values_full,
              result.singular_values_x_given_b, result.singular_values_bb]
    for value in fields:
        if value is None or not np.all(np.isfinite(np.asarray(value, dtype=float))):
            return False, "non-finite joint calibration (%s)" % result.termination_reason

    # Motion checks first, and computed over ALL pairs rather than the admitted
    # ones: a rig that never rotated admits no pairs at all, and "0 pairs
    # survived the gate" is a true but useless diagnosis of "it did not move".
    # They also come before the conditioning checks because a near-zero H is
    # ill-conditioned for an uninteresting reason.
    if result.max_pair_angle_rad < min_angle:
        return False, ("negligible rotation: largest pair rotation %.4f rad < %.4f"
                       % (result.max_pair_angle_rad, min_angle))
    # Pure-axis motion: reuses certificates.axis_coverage_ratio (lambda2/lambda1
    # of the rotation-axis scatter), the existing measure of exactly this.
    if result.axis_coverage < min_axis:
        return False, ("single-axis motion: axis coverage %.4g < %.4g"
                       % (result.axis_coverage, min_axis))
    if result.n_pairs < min_pairs:
        return False, "only %d admitted pairs (need %d)" % (result.n_pairs, min_pairs)
    if not result.converged:
        return False, "joint solve did not converge (%s)" % result.termination_reason
    if result.sigma_min_bb < min_sigma_bb:
        return False, ("gyro bias ill-conditioned: sigma_min(H_bb) = %.3e < %.3e"
                       " (cond = %.3e)" % (result.sigma_min_bb, min_sigma_bb, result.cond_bb))
    # The one number the alternating solver cannot produce: rotation
    # conditioning AFTER the bias has been marginalized out. A direction of X
    # that only looked observable because the bias was pinned shows up here and
    # nowhere else in this codebase.
    if result.sigma_min_x_given_b < min_sigma_marg:
        return False, ("rotation/bias ambiguity: sigma_min(H_X|b) = %.3e < %.3e"
                       % (result.sigma_min_x_given_b, min_sigma_marg))
    return True, ""


# --------------------------------------------------------------------------
# 5. Joint leave-one-block-out dispersion (auxiliary diagnostic)
# --------------------------------------------------------------------------

def tangent_dispersion_deg(r_pooled, rotations):
    if len(rotations) < 3:
        return float("inf")
    phi = np.array([log_so3(np.asarray(r_pooled).T @ np.asarray(r)) for r in rotations])
    residual = phi - phi.mean(axis=0)
    cov = (residual.T @ residual) / len(rotations)
    return float(np.degrees(np.sqrt(max(np.trace(cov), 0.0))))


def joint_loo_dispersion(imu, windows, config=None, **solve_kwargs):
    config = config or InitConfig()
    pooled = solve_joint_calibration(imu, windows, config, **solve_kwargs)
    rotations, biases = [], []
    for i in range(len(windows)):
        subset = list(windows[:i]) + list(windows[i + 1:])
        if len(subset) < 2:
            continue
        try:
            loo = solve_joint_calibration(imu, subset, config, **solve_kwargs)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(loo.r_ctoi)) or not np.all(np.isfinite(loo.bias_gyro)):
            continue
        rotations.append(loo.r_ctoi)
        biases.append(loo.bias_gyro)

    n = len(rotations)
    out = dict(pooled=pooled, n_loo=n, rotation_rms_deg=float("inf"),
               rotation_dispersion_deg=float("inf"), bias_dispersion_rad_s=float("inf"),
               bias_rms_rad_s=float("inf"))
    if n < 3:
        return out
    deviations = np.array([_ang_deg(r, pooled.r_ctoi) for r in rotations])
    out["rotation_rms_deg"] = float(np.sqrt(np.mean(deviations ** 2)))
    out["rotation_dispersion_deg"] = tangent_dispersion_deg(pooled.r_ctoi, rotations)
    biases = np.asarray(biases, dtype=float)
    out["bias_rms_rad_s"] = float(np.sqrt(np.mean(
        np.sum((biases - pooled.bias_gyro) ** 2, axis=1))))
    residual = biases - biases.mean(axis=0)
    out["bias_dispersion_rad_s"] = float(np.sqrt(max(
        np.trace((residual.T @ residual) / n), 0.0)))
    return out
