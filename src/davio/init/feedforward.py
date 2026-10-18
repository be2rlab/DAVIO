from dataclasses import dataclass, field
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .initializer import _preintegrate_frames
from .jpl import exp_so3, rot_to_quat

EPS = 1e-5


def rotvec_to_rot(v):
    return Rotation.from_rotvec(np.asarray(v, float)).as_matrix()


def softplus(x):
    return EPS + np.logaddexp(0., x)


def default_config():
    # Each informative frame contributes three constraints, so two points per frame already
    # form a minimal set; the paper's ten per frame contain an outlier almost surely at a
    # 30 % outlier rate (0.7^40), which is why the minimal set is small and iterations many.
    # ransac_threshold is the FLOOR of the robust inlier band (normalized reprojection error;
    # 0.01 ~ 0.6 deg). max_median_error rejects a window whose best hypothesis still leaves
    # the median point 0.05 (~3 deg) off: DA3 and the IMU chain then disagree too much.
    # bias_accel_prior is tight on purpose: over a <= 1 s window an accelerometer bias is
    # indistinguishable from scale and gravity tilt (measured: freeing it doubled the scale).
    return dict(points_per_frame=100, conf_quantile=0.25, grid=4, ransac_iterations=100,
                ransac_threshold=0.01, min_per_frame=2, min_inlier_fraction=0.4,
                min_frames_covered=3, max_median_error=0.05,
                max_velocity_ms=5.0, gravity_accel_max_deg=30.0, pixel_sigma_norm=0.007,
                lever_arm_prior_m=0.05, bias_gyro_prior=0.05, bias_accel_prior=0.02,
                log_scale_prior=0.5,
                sigma_floor=[0.03, 0.05, 0.10, 0.01, 0.05], max_nfev=60,
                # Equilibrated singular-value ratio below which [s, v, g] is unobservable.
                # Measured on the synthetic fixture: healthy 4-6 frame windows sit at
                # 1e-2..2e-2, three frames at 6e-5 (a slightly wrong rotation manufactures
                # that much), exact degeneracy at 1e-14. Four frames are a hard floor.
                rank_rtol=1e-3, min_frames=4,
                # The lever-arm columns absorb rotation-model error over one window (measured
                # on M1: estimates of 0.2-3.9 m against a true 7 cm), so by default the lever
                # arm is HELD at the prior mean and left to the filter's online calibration.
                estimate_lever_arm=False,
                # Parallax gate: largest DA3 camera-centre displacement over the window as a
                # fraction of the median sampled depth. A static window makes the scale column
                # vanish; column equilibration would hide that and return garbage.
                min_baseline_ratio=0.02)


@dataclass
class FeedForwardResult:
    status: str
    reason: str = ''
    scale: float = float('nan')
    velocity: np.ndarray = None
    gravity: np.ndarray = None
    lever_arm: np.ndarray = None
    bias_gyro: np.ndarray = None
    bias_accel: np.ndarray = None
    sigmas: np.ndarray = None
    state: dict = None
    info: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


# --- sampling and the linear system -------------------------------------------------------

def sample_points(depth, conf, intrinsics, extrinsics, per_frame, conf_quantile, grid, rng):
    depth = np.asarray(depth, float)
    n, h, w = depth.shape
    ext = np.asarray(extrinsics, float)
    if ext.shape == (n, 3, 4):
        full = np.repeat(np.eye(4)[None], n, axis=0)
        full[:, :3] = ext
        ext = full
    to_c0 = [ext[0] @ np.linalg.inv(ext[i]) for i in range(n)]         # T_{C0<-Ci}
    per_cell = int(np.ceil(per_frame / (grid * grid)))
    out = []
    for i in range(n):
        z = depth[i]
        ok = np.isfinite(z) & (z > 0)
        c = None
        if conf is not None:
            c = np.asarray(conf[i], float)
            finite = c[ok & np.isfinite(c)]
            if len(finite):
                ok &= np.isfinite(c) & (c >= np.quantile(finite, conf_quantile))
        rows, cols = np.nonzero(ok)
        if not len(rows):
            continue
        cell = (rows * grid // h) * grid + (cols * grid // w)
        chosen = []
        for cid in np.unique(cell):
            members = np.flatnonzero(cell == cid)
            chosen.extend(rng.choice(members, size=min(per_cell, len(members)), replace=False))
        chosen = np.asarray(chosen, int)
        chosen = chosen[rng.permutation(len(chosen))][:per_frame]
        v, u = rows[chosen], cols[chosen]
        k_inv = np.linalg.inv(np.asarray(intrinsics[i], float))
        uv_norm = np.column_stack((u, v, np.ones(len(u)))) @ k_inv.T
        xyz_ci = uv_norm * z[v, u][:, None]
        xyz_c0 = xyz_ci @ to_c0[i][:3, :3].T + to_c0[i][:3, 3]
        weight = np.ones(len(u)) if c is None else np.sqrt(np.clip(c[v, u], 0., None) + 1e-6)
        out.append(dict(frame=i, uv=uv_norm[:, :2], xyz_c0=xyz_c0, z=z[v, u].astype(float),
                        w=weight))
    return out


def linear_rows(samples, preints, r_ctoi, p_cini=None, return_frames=False):
    r_ctoi = np.asarray(r_ctoi, float)
    r_itoc = r_ctoi.T
    a_rows, b_rows, w_rows, frames = [], [], [], []
    for smp in samples:
        i = smp['frame']
        if i == 0:
            continue                                        # H annihilates its own ray: zero rows
        pre = preints[i]
        r_i, dt, dp = np.asarray(pre.dR, float), float(pre.dt), np.asarray(pre.dp, float)
        rot = r_itoc @ r_i
        lever = r_itoc @ (r_i - np.eye(3))
        for uv, p_bar, z, wt in zip(smp['uv'], smp['xyz_c0'], smp['z'], smp['w']):
            hmat = np.array([[1., 0., -uv[0]], [0., 1., -uv[1]]]) / max(z, 1e-9)
            bmat = hmat @ rot                               # 2x3
            col_s = (bmat @ r_ctoi @ p_bar)[:, None]
            rhs = bmat @ dp
            if p_cini is None:
                row = np.hstack((col_s, -dt * bmat, 0.5 * dt * dt * bmat, hmat @ lever))
            else:
                row = np.hstack((col_s, -dt * bmat, 0.5 * dt * dt * bmat))
                rhs = rhs - hmat @ lever @ np.asarray(p_cini, float)
            a_rows.append(row)
            b_rows.append(rhs)
            w_rows.append(np.full(2, wt))
            frames.append(i)
    if not a_rows:
        raise ValueError('no rows: every sampled point belongs to frame 0')
    A, b, w = np.vstack(a_rows), np.concatenate(b_rows), np.concatenate(w_rows)
    if return_frames:
        return A, b, w, np.asarray(frames)
    return A, b, w


def _tangent_basis(g):
    g = np.asarray(g, float)
    g = g / np.linalg.norm(g)
    helper = np.array([1., 0., 0.]) if abs(g[0]) < 0.9 else np.array([0., 1., 0.])
    b1 = np.cross(g, helper)
    b1 /= np.linalg.norm(b1)
    return np.column_stack((b1, np.cross(g, b1)))


def solve_linear(A, b, w, gravity_mag, lever_prior=None, rounds=2, noise_norm=0.005):
    A, b, w = np.asarray(A, float), np.asarray(b, float), np.asarray(w, float)
    aw, bw = A * w[:, None], b * w
    free_lever = A.shape[1] == 10
    mean, sigma = (np.zeros(3), 0.05) if lever_prior is None else lever_prior
    mean = np.asarray(mean, float)

    def system(s_est):
        if not free_lever:
            return aw, bw
        k = s_est * noise_norm / sigma
        prior = np.zeros((3, 10))
        prior[:, 7:10] = np.eye(3) * k
        return np.vstack((aw, prior)), np.concatenate((bw, mean * k))

    a_sys, b_sys = system(1.)
    x = np.linalg.lstsq(a_sys, b_sys, rcond=None)[0]
    for _ in range(rounds):
        a_sys, b_sys = system(max(abs(float(x[0])), 1e-3))
        g_hat = x[4:7] / max(np.linalg.norm(x[4:7]), 1e-12) * gravity_mag
        basis = _tangent_basis(g_hat)
        a_red = np.hstack((a_sys[:, :4], a_sys[:, 4:7] @ basis * gravity_mag, a_sys[:, 7:]))
        b_red = b_sys - a_sys[:, 4:7] @ g_hat
        y = np.linalg.lstsq(a_red, b_red, rcond=None)[0]
        g_new = g_hat + basis @ y[4:6] * gravity_mag
        g_new *= gravity_mag / np.linalg.norm(g_new)
        x = np.concatenate((y[:4], g_new, y[6:]))
    return x


def rank_deficient(A, w, rtol=1e-3):
    aw = np.asarray(A, float) * np.asarray(w, float)[:, None]
    scale = np.linalg.norm(aw, axis=0)
    scale[scale == 0] = 1.
    s = np.linalg.svd(aw / scale, compute_uv=False)
    return bool(s[-1] < rtol * s[0])


def point_errors(A, b, x):
    """Per-point normalized reprojection error: rows come in (x, y) pairs, divided by s."""
    r = (np.asarray(A) @ x - np.asarray(b)).reshape(-1, 2)
    return np.linalg.norm(r, axis=1) / max(abs(float(x[0])), 1e-9)


def ransac_linear(A, b, w, frame_of_point, gravity_mag, lever_prior, iterations, threshold,
                  min_per_frame, rng, min_frames_covered=3, max_median_error=None):
    frames = np.asarray(frame_of_point)
    by_frame = {f: np.flatnonzero(frames == f) for f in np.unique(frames)}

    def rows(points):
        return np.ravel(np.column_stack((2 * points, 2 * points + 1)))

    def fit(points):
        idx = rows(points)
        return solve_linear(A[idx], b[idx], w[idx], gravity_mag, lever_prior)

    def band(err):
        return max(2.5 * 1.4826 * float(np.median(err)), float(threshold))

    def covered(mask):
        return sum(1 for pts in by_frame.values() if mask[pts].mean() > .5)

    best, best_mask, best_score = None, None, np.inf
    for _ in range(int(iterations)):
        subset = np.concatenate([rng.choice(v, size=min(min_per_frame, len(v)), replace=False)
                                 for v in by_frame.values()])
        x = fit(subset)
        if x[0] <= 0 or not np.isfinite(x).all():
            continue
        err = point_errors(A, b, x)
        mask = err < band(err)
        if covered(mask) < min_frames_covered:
            continue
        score = float(np.median(err))
        if score < best_score:
            best, best_mask, best_score = x, mask, score
    if best is None:
        best = solve_linear(A, b, w, gravity_mag, lever_prior)
        err = point_errors(A, b, best)
        best_mask = err < band(err)
    else:
        best = fit(np.flatnonzero(best_mask))                       # refit on the inlier set
        err = point_errors(A, b, best)
        best_mask = err < band(err)
    err = point_errors(A, b, best)
    info = dict(inlier_fraction=float(best_mask.mean()), inliers=int(best_mask.sum()),
                points=int(len(best_mask)), frames_covered=int(covered(best_mask)),
                median_error=float(np.median(err)),
                median_inlier_error=float(np.median(err[best_mask])) if best_mask.any() else None)
    return best, best_mask, info


# --- feature-free refinement ---------------------------------------------------------------

def _poses_from_state(preints, v, g, bg_delta, ba):
    out = []
    for pre in preints:
        r_i = exp_so3(np.asarray(pre.dR_dbg, float) @ bg_delta) @ np.asarray(pre.dR, float)
        dp = np.asarray(pre.dp, float) + np.asarray(pre.dp_dba, float) @ ba
        dt = float(pre.dt)
        out.append((r_i, v * dt - 0.5 * g * dt * dt + dp, dt))
    return out


def _reprojection(samples, masks, poses, r_ctoi, s, p_cini, sigma):
    r_itoc = r_ctoi.T
    res = []
    for smp, m in zip(samples, masks):
        if smp['frame'] == 0 or not m.any():
            continue
        r_i, p_i, _dt = poses[smp['frame']]
        x_i0 = (smp['xyz_c0'][m] * s) @ r_ctoi.T + p_cini
        x_c = ((x_i0 - p_i) @ r_i.T - p_cini) @ r_itoc.T
        z = np.maximum(x_c[:, 2], 1e-6)
        res.append(((x_c[:, :2] / z[:, None]) - smp['uv'][m]).ravel() / sigma)
    return np.concatenate(res) if res else np.zeros(0)


def _masks_per_sample(samples, inlier_mask):
    masks, offset = [], 0
    for smp in samples:
        n = len(smp['z'])
        if smp['frame'] == 0:
            masks.append(np.zeros(n, bool))
        else:
            masks.append(np.asarray(inlier_mask[offset:offset + n], bool))
            offset += n
    return masks


def refine(samples, inlier_mask, imu, frame_times, x_lin, r_ctoi, p_cini, cfg, gravity_mag,
           bg0=None):
    r_ctoi = np.asarray(r_ctoi, float)
    free_lever = p_cini is None
    bg0 = np.zeros(3) if bg0 is None else np.asarray(bg0, float)
    g0 = np.asarray(x_lin[4:7], float)
    basis = _tangent_basis(g0)
    preints = _preintegrate_frames(imu, frame_times, bg0)
    masks = _masks_per_sample(samples, inlier_mask)
    s_tilde0 = float(np.log(np.expm1(max(float(x_lin[0]) - EPS, 1e-3))))
    p_prior = np.asarray(x_lin[7:10] if free_lever else p_cini, float)
    theta0 = np.concatenate(([s_tilde0], x_lin[1:4], np.zeros(2), np.zeros(3), np.zeros(3),
                             p_prior if free_lever else np.zeros(0)))
    sigma = float(cfg['pixel_sigma_norm'])

    def unpack(theta):
        s = softplus(theta[0])
        v = theta[1:4]
        g = g0 + basis @ theta[4:6] * gravity_mag
        g = g * (gravity_mag / np.linalg.norm(g))
        dbg, ba = theta[6:9], theta[9:12]
        p = theta[12:15] if free_lever else p_prior
        return s, v, g, dbg, ba, p

    def residuals(theta):
        s, v, g, dbg, ba, p = unpack(theta)
        poses = _poses_from_state(preints, v, g, dbg, ba)
        r = _reprojection(samples, masks, poses, r_ctoi, s, p, sigma)
        priors = [dbg / cfg['bias_gyro_prior'], ba / cfg['bias_accel_prior'],
                  np.atleast_1d(np.log(s / max(float(x_lin[0]), 1e-6)) / cfg.get('log_scale_prior', 0.5))]
        if free_lever:
            priors.append((p - p_prior) / cfg['lever_arm_prior_m'])
        return np.concatenate([r] + priors)

    sol = least_squares(residuals, theta0, loss='huber', f_scale=1.0, max_nfev=int(cfg['max_nfev']))
    s, v, g, dbg, ba, p = unpack(sol.x)
    bg = bg0 + dbg
    jtj = sol.jac.T @ sol.jac
    cov = np.linalg.pinv(jtj, rcond=1e-10, hermitian=True)
    n_prior = 10 if free_lever else 7
    rep = sol.fun[:len(sol.fun) - n_prior]
    rms = float(np.sqrt(np.mean(rep ** 2))) * sigma if len(rep) else float('nan')
    info = dict(converged=bool(sol.success), nfev=int(sol.nfev), cost=float(sol.cost),
                theta_sigma=np.sqrt(np.clip(np.diag(cov), 0, None)).tolist())
    return dict(s=float(s), v=v, g=g, bg=bg, ba=ba, p=p), cov, rms, info


# --- bootstrap state -----------------------------------------------------------------------

def bootstrap_state(times, preints_at_bias, x, bg, ba):
    g = np.asarray(x['g'], float)
    z = g / np.linalg.norm(g)
    helper = np.array([1., 0., 0.]) if abs(z[0]) < 0.9 else np.array([0., 1., 0.])
    xax = helper - z * (helper @ z)
    xax /= np.linalg.norm(xax)
    r_g_from_i0 = np.vstack((xax, np.cross(z, xax), z))                  # rows: G axes in I0
    pre = preints_at_bias[-1]
    dt = float(pre.dt)
    r_n = np.asarray(pre.dR, float)                                      # R_{IN<-I0}
    p_n = x['v'] * dt - 0.5 * g * dt * dt + np.asarray(pre.dp, float)
    v_n = x['v'] - g * dt + np.asarray(pre.dv, float)
    r_in_from_g = r_n @ r_g_from_i0.T
    return dict(t=float(times[-1]), q_GtoI=rot_to_quat(r_in_from_g).tolist(),
                p=(r_g_from_i0 @ p_n).tolist(), v=(r_g_from_i0 @ v_n).tolist(),
                bg=np.asarray(bg, float).tolist(), ba=np.asarray(ba, float).tolist(),
                R_GfromI0=r_g_from_i0.tolist())


def _accelerometer_direction(preints, imu, frame_times):
    times = np.array([m.t for m in imu])
    lo, hi = np.searchsorted(times, [frame_times[0], frame_times[-1]])
    vectors = []
    for pre, t in zip(preints, frame_times):
        k = int(np.clip(np.argmin(np.abs(times - t)), 0, len(imu) - 1))
        vectors.append(np.asarray(pre.dR, float).T @ np.asarray(imu[k].accel, float))
    return np.mean(vectors, axis=0)


def gyro_bias_from_window(pred_extrinsics, frame_times, imu, r_ctoi, iterations=2):
    from .bias_gyro import gyro_bias_update
    absolute = np.asarray(pred_extrinsics, float)[:, :3, :3]          # R_{Ci<-W}, W = DA3's reference view
    rotations = absolute @ absolute[0].T                               # R_{Ci<-C0}: what gyro_bias_rows expects
    bg = np.zeros(3)
    for _ in range(int(iterations)):
        preints = _preintegrate_frames(imu, frame_times, bg)
        try:
            step = gyro_bias_update(rotations, preints, np.asarray(r_ctoi, float))
        except (np.linalg.LinAlgError, ValueError):
            break
        if not np.isfinite(step).all():
            break
        bg = bg + step
    return bg


def feedforward_initialize(pred, frame_times, imu, r_ctoi, p_cini, cfg, gravity_mag, rng, bg0=None):
    timings, t0 = {}, time.perf_counter()
    frame_times = np.asarray(frame_times, float)
    lever_held = p_cini is None and not bool(cfg.get('estimate_lever_arm', False))
    if lever_held:
        p_cini = np.zeros(3)
    if len(frame_times) < int(cfg['min_frames']):
        return FeedForwardResult('rejected', f"rank: fewer than {int(cfg['min_frames'])} frames")
    samples = sample_points(pred['depth'], pred.get('conf'), pred['intrinsics'], pred['extrinsics'],
                            int(cfg['points_per_frame']), float(cfg['conf_quantile']),
                            int(cfg['grid']), rng)
    if sum(len(s['z']) for s in samples if s['frame']) < 3 * int(cfg['min_per_frame']):
        return FeedForwardResult('rejected', 'too few confident points')
    ext = np.asarray(pred['extrinsics'], float)
    if ext.shape[1:] == (3, 4):
        full = np.repeat(np.eye(4)[None], len(ext), axis=0)
        full[:, :3] = ext
        ext = full
    centres = np.linalg.inv(ext)[:, :3, 3]
    baseline = float(np.max(np.linalg.norm(centres - centres[0], axis=1)))
    median_depth = float(np.median(np.concatenate([s['z'] for s in samples])))
    parallax = baseline / max(median_depth, 1e-9)
    if parallax < float(cfg['min_baseline_ratio']):
        return FeedForwardResult('rejected', f'insufficient parallax ({parallax:.3f} of depth)',
                                 info=dict(parallax=parallax))
    try:
        if bg0 is None:
            bg0 = gyro_bias_from_window(pred['extrinsics'], frame_times, imu, r_ctoi)
        bg0 = np.asarray(bg0, float)
        preints = _preintegrate_frames(imu, frame_times, bg0)
    except ValueError as exc:
        return FeedForwardResult('rejected', 'imu support: ' + str(exc))
    A, b, w, frames = linear_rows(samples, preints, r_ctoi, p_cini, return_frames=True)
    timings['rows_s'] = time.perf_counter() - t0
    # Observability of [s, v, g]; the lever arm is regularized by its prior, so its columns
    # (rank 2 under a single rotation axis) must not trip the test.
    if rank_deficient(A[:, :7], w, float(cfg['rank_rtol'])):
        return FeedForwardResult('rejected', 'rank: linear system rank deficient (need >= 4 frames with non-constant acceleration)')
    lever_prior = None if p_cini is not None else (np.zeros(3), float(cfg['lever_arm_prior_m']))
    x, mask, info = ransac_linear(A, b, w, frames, gravity_mag, lever_prior,
                                  int(cfg['ransac_iterations']), float(cfg['ransac_threshold']),
                                  int(cfg['min_per_frame']), rng,
                                  min_frames_covered=int(cfg.get('min_frames_covered', 3)))
    timings['linear_s'] = time.perf_counter() - t0
    info.update(linear_scale=float(x[0]), parallax=parallax, bias_gyro_window=bg0.tolist())
    if x[0] <= 0 or not np.isfinite(x).all():
        return FeedForwardResult('rejected', 'nonpositive or non-finite linear solution', info=info)
    if info['frames_covered'] < int(cfg.get('min_frames_covered', 3)):
        return FeedForwardResult('rejected', f"only {info['frames_covered']} frames agree with the IMU chain", info=info)
    if info['median_error'] > float(cfg.get('max_median_error', 0.05)):
        return FeedForwardResult('rejected', f"median error {info['median_error']:.3f} (DA3/IMU disagreement)", info=info)
    if info['inlier_fraction'] < float(cfg['min_inlier_fraction']):
        return FeedForwardResult('rejected', f"inlier fraction {info['inlier_fraction']:.2f}", info=info)
    if np.linalg.norm(x[1:4]) > float(cfg['max_velocity_ms']):
        return FeedForwardResult('rejected', 'implausible velocity', info=info)
    accel = _accelerometer_direction(preints, imu, frame_times)
    cosine = accel @ x[4:7] / (np.linalg.norm(accel) * np.linalg.norm(x[4:7]) + 1e-12)
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))
    info['gravity_vs_accelerometer_deg'] = angle
    if angle > float(cfg['gravity_accel_max_deg']):
        return FeedForwardResult('rejected', f'gravity disagrees with accelerometer by {angle:.1f} deg', info=info)
    x_ref, cov, rms, rinfo = refine(samples, mask, imu, frame_times, x, r_ctoi, p_cini, cfg, gravity_mag, bg0)
    timings['refine_s'] = time.perf_counter() - t0
    info.update(rinfo, refined_rms_norm=rms)
    if not np.all(np.isfinite(cov)) or x_ref['s'] <= 0:
        return FeedForwardResult('rejected', 'covariance not recoverable', info=info)
    try:
        preints_final = _preintegrate_frames(imu, frame_times, x_ref['bg'], x_ref['ba'])
    except ValueError as exc:
        return FeedForwardResult('rejected', 'imu support: ' + str(exc), info=info)
    state = bootstrap_state(frame_times, preints_final, x_ref, x_ref['bg'], x_ref['ba'])
    d = np.sqrt(np.clip(np.diag(cov), 0, None))
    floor = np.asarray(cfg['sigma_floor'], float)
    theta_sigma = max(float(np.linalg.norm(d[4:6])), floor[0])
    sigmas = np.r_[np.full(3, theta_sigma), np.full(3, floor[1]),
                   np.maximum(d[1:4], floor[2]), np.maximum(d[6:9], floor[3]),
                   np.maximum(d[9:12], floor[4])]
    state['sigmas'] = sigmas.tolist()
    info['lever_arm_held_at_prior'] = bool(lever_held)
    return FeedForwardResult('released', '', scale=float(x_ref['s']), velocity=x_ref['v'],
                             gravity=x_ref['g'], lever_arm=np.asarray(x_ref['p'], float),
                             bias_gyro=x_ref['bg'], bias_accel=x_ref['ba'], sigmas=sigmas,
                             state=state, info=info, timings=timings)
