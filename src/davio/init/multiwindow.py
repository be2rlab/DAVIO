import numpy as np
from .bias_gyro import gyro_bias_update, log_so3
from .handeye import build_handeye_system, solve_handeye
from .initializer import _preintegrate_frames, _handeye_pairs, dedupe_pairs_by_timestamp
from .jpl import quat_to_rot
from .preintegration import preintegrate
from .types import InitConfig

def bootstrap_gyro_bias(imu, windows, max_cam_angle_deg=3.0, min_dt=0.4):
    thr = np.deg2rad(float(max_cam_angle_deg))
    est, wts = [], []
    for win in windows:
        preints = _preintegrate_frames(imu, win.frame_times, np.zeros(3))
        n = len(preints)
        for i in range(n):
            for j in range(i + 1, n):
                r_cam = (np.asarray(win.camera_rotations[j])
                         @ np.asarray(win.camera_rotations[i]).T)
                ang = np.arccos(np.clip((np.trace(r_cam) - 1.0) / 2.0, -1.0, 1.0))
                dt = preints[j].dt - preints[i].dt
                if ang > thr or dt < min_dt:
                    continue
                d_rel = preints[j].dR @ preints[i].dR.T
                est.append(-log_so3(d_rel) / dt)
                wts.append(dt)
    if not est:
        return np.zeros(3), 0
    est = np.asarray(est)
    wts = np.asarray(wts, dtype=float)
    return (est * wts[:, None]).sum(axis=0) / wts.sum(), len(est)


def global_handeye(imu, windows, bias_gyro, config=None, timer=None):
    from contextlib import nullcontext
    config = config or InitConfig()
    bias_gyro = np.asarray(bias_gyro, dtype=float).reshape(3)

    pairs = []
    for win in windows:
        with (timer.time("imu_preintegration") if timer is not None else nullcontext()):
            preints = _preintegrate_frames(imu, win.frame_times, bias_gyro)
        win.preints = preints                       # cached for the per-window state solve
        with (timer.time("handeye_pair_assembly") if timer is not None else nullcontext()):
            pairs.extend(_handeye_pairs(preints, win.camera_rotations,
                                        frame_times=win.frame_times))

    with (timer.time("handeye_pair_assembly") if timer is not None else nullcontext()):
        # Overlapping windows (offsets closer together than the window duration) can
        # hand back pairs built from the literal same two camera frames;
        # pooling both would double-count one measurement as two.
        pairs = dedupe_pairs_by_timestamp(pairs)
        b_mat, kept = build_handeye_system(pairs, config.min_pair_angle_rad,
                                           config.angle_tol_rad,
                                           robust_weighting=config.handeye_robust_weighting)
        q, sigma3, sigma4 = solve_handeye(b_mat)
    return quat_to_rot(q), float(sigma3), float(sigma4), len(kept)


def global_handeye_auto_bias(imu, windows, config=None, alternations=3,
                             max_cam_angle_deg=3.0, timer=None, use_bootstrap=True,
                             bias_init=None, convergence_tol_rad_s=None, diag_out=None):
    from contextlib import nullcontext
    config = config or InitConfig()
    tol = (config.bg_convergence_tol_rad_s if convergence_tol_rad_s is None
          else convergence_tol_rad_s)
    if use_bootstrap:
        bias, n_boot = bootstrap_gyro_bias(imu, windows, max_cam_angle_deg=max_cam_angle_deg)
    else:
        bias = np.zeros(3) if bias_init is None else np.asarray(bias_init, float).reshape(3)
        n_boot = 0

    r_ctoi, s3, s4, n_pairs = global_handeye(imu, windows, bias, config, timer=timer)
    n_run, converged, step_norm = 0, False, float("nan")
    reason = "no_alternations_requested" if alternations <= 0 else "max_alternations"
    for _ in range(max(0, alternations)):
        deltas, weights = [], []
        for win in windows:
            with (timer.time("imu_preintegration") if timer is not None else nullcontext()):
                preints = _preintegrate_frames(imu, win.frame_times, bias)
            try:
                deltas.append(gyro_bias_update(win.camera_rotations, preints, r_ctoi))
                weights.append(len(preints) - 1)
            except (np.linalg.LinAlgError, ValueError):
                continue
        if not deltas:
            reason = "no_usable_windows"
            break
        w = np.asarray(weights, dtype=float)
        step = (np.asarray(deltas) * w[:, None]).sum(axis=0) / w.sum()
        if not np.all(np.isfinite(step)):
            reason = "non_finite_update"
            break
        bias = bias + step
        n_run += 1
        step_norm = float(np.linalg.norm(step))
        r_ctoi, s3, s4, n_pairs = global_handeye(imu, windows, bias, config, timer=timer)
        if step_norm < tol:
            converged = True
            reason = "converged"
            break
    if diag_out is not None:
        diag_out.update(n_boot=n_boot, n_alternations_run=n_run, converged=converged,
                        final_step_norm_rad_s=step_norm, termination_reason=reason,
                        bootstrap_reliable=bool(use_bootstrap and n_boot > 0))
    return r_ctoi, s3, s4, n_pairs, bias, n_boot

