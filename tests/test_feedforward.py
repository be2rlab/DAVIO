from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from davio.init.preintegration import ImuSample                     # noqa: E402
from davio.init.initializer import _preintegrate_frames             # noqa: E402
from davio.init import feedforward as ff                            # noqa: E402

G = 9.81
OMEGA = np.array([0.3, 0.2, -0.1])
ACCEL = np.array([0.4, 0.2, -0.1])
# A constant acceleration would make the camera centres a quadratic in time, which the
# IMU model reproduces with an unknown scale and a flipped gravity: a true degeneracy of the
# feature-free system, not a property of real motion. So the acceleration oscillates.
A1, W = np.array([1.0, -0.6, 0.4]), 2 * np.pi * 1.2
# Likewise a constant rotation rate makes (I - R_i^T) p a quadratic in time, absorbed by v
# and g: the lever arm is only observable when the rate changes, and never along the axis.
RATE_VAR, W2 = 1.5, 2 * np.pi * 1.7
AXIS = OMEGA / np.linalg.norm(OMEGA)


def angle(t, rate_var):
    return np.linalg.norm(OMEGA) * (t + rate_var * (1 - np.cos(W2 * t)) / W2)


def rate(t, rate_var):
    return OMEGA * (1 + rate_var * np.sin(W2 * t))


def accel_i0(t):
    return ACCEL + A1 * np.sin(W * t)


def vel_i0(t, v0):
    return v0 + ACCEL * t + (A1 / W) * (1 - np.cos(W * t))


def pos_i0(t, v0):
    return v0 * t + 0.5 * ACCEL * t * t + (A1 / W) * t - (A1 / W ** 2) * np.sin(W * t)


def synthetic(n_frames=5, dt=0.2, rotate=True, seed=0, scale=3.0, size=(320, 240),
              rate_var=RATE_VAR):
    rng = np.random.default_rng(seed)
    r_ctoi = ff.rotvec_to_rot(np.array([0.1, -1.2, 0.4]))
    p_cini = np.array([0.05, -0.02, 0.03])
    v0 = np.array([0.3, -0.1, 0.05])                                   # I0 frame
    g_i0 = G * np.array([0.1, -0.05, 1.0]) / np.linalg.norm([0.1, -0.05, 1.0])
    times = np.arange(n_frames) * dt

    def rot(t):                                                         # R_{I0<-It}, fixed axis
        return ff.rotvec_to_rot(AXIS * angle(t, rate_var)) if rotate else np.eye(3)

    def pos(t):
        return pos_i0(t, v0)

    imu = [ImuSample(float(t), rate(t, rate_var) if rotate else np.zeros(3),
                     rot(t).T @ (accel_i0(t) + g_i0))
           for t in np.arange(-0.05, times[-1] + 0.06, 0.005)]
    t_i_c = np.eye(4)
    t_i_c[:3, :3], t_i_c[:3, 3] = r_ctoi, p_cini
    cams = []
    for t in times:
        t_i0_i = np.eye(4)
        t_i0_i[:3, :3], t_i0_i[:3, 3] = rot(t), pos(t)
        cams.append(t_i0_i @ t_i_c)                                     # T_{I0<-Ci}
    w, h = size
    k = np.array([[300., 0, w / 2.], [0, 300., h / 2.], [0, 0, 1.]])
    pts_c0 = rng.uniform([-1.5, -1.0, 1.5], [1.5, 1.0, 5.0], size=(600, 3))
    points_i0 = pts_c0 @ cams[0][:3, :3].T + cams[0][:3, 3]
    depth, conf, K, ext = [], [], [], []
    for c in cams:
        pc = (points_i0 - c[:3, 3]) @ c[:3, :3]                         # points in camera i
        d = np.full((h, w), np.nan)
        cf = np.zeros((h, w))
        uv = (pc @ k.T)[:, :2] / pc[:, 2:3]
        for (u, v), z in zip(np.rint(uv).astype(int), pc[:, 2]):
            if 0 <= u < w and 0 <= v < h and z > 0:
                d[v, u] = z / scale
                cf[v, u] = 1.
        depth.append(d)
        conf.append(cf)
        K.append(k)
        w2c = np.linalg.inv(np.linalg.inv(cams[0]) @ c)                # T_{Ci<-C0}
        w2c[:3, 3] /= scale
        ext.append(w2c)
    return dict(depth=np.array(depth), conf=np.array(conf), intrinsics=np.array(K),
                extrinsics=np.array(ext), times=times, imu=imu, r_ctoi=r_ctoi, p_cini=p_cini,
                v0=v0, g=g_i0, scale=scale, cams=cams, rot=rot)


def _samples(s, seed=0):
    return ff.sample_points(s['depth'], s['conf'], s['intrinsics'], s['extrinsics'],
                            per_frame=60, conf_quantile=0., grid=4, rng=np.random.default_rng(seed))


def test_linear_system_recovers_scale_velocity_gravity_with_supplied_lever_arm():
    s = synthetic()
    preints = _preintegrate_frames(s['imu'], s['times'], np.zeros(3))
    A, b, w = ff.linear_rows(_samples(s), preints, s['r_ctoi'], p_cini=s['p_cini'])
    assert A.shape[1] == 7
    x = ff.solve_linear(A, b, w, G)
    assert abs(x[0] - s['scale']) < 0.01 * s['scale']
    np.testing.assert_allclose(x[1:4], s['v0'], atol=0.02)
    np.testing.assert_allclose(x[4:7], s['g'], atol=0.05)
    assert np.linalg.norm(x[4:7]) == pytest.approx(G, abs=1e-9)


def perpendicular(err):
    return err - AXIS * (err @ AXIS)


def test_linear_system_recovers_the_observable_lever_arm_and_holds_prior_otherwise():
    for rotate in (True, False):
        s = synthetic(rotate=rotate)
        preints = _preintegrate_frames(s['imu'], s['times'], np.zeros(3))
        A, b, w = ff.linear_rows(_samples(s, 1), preints, s['r_ctoi'])
        assert A.shape[1] == 10
        x = ff.solve_linear(A, b, w, G, lever_prior=(np.zeros(3), 0.05))
        if rotate:
            # Observable across the rotation axis only; the axis component stays near
            # the prior and is the filter's job (online extrinsic calibration).
            assert np.linalg.norm(perpendicular(x[7:10] - s['p_cini'])) < 0.015
        else:
            # Without rotation the lever-arm columns vanish: it must stay at the prior,
            # never wander to an unobservable value.
            np.testing.assert_allclose(x[7:10], np.zeros(3), atol=1e-6)


def test_two_frames_are_rank_deficient():
    s = synthetic(n_frames=2)
    preints = _preintegrate_frames(s['imu'], s['times'], np.zeros(3))
    A, b, w = ff.linear_rows(_samples(s, 2), preints, s['r_ctoi'], p_cini=s['p_cini'])
    assert ff.rank_deficient(A, w)
    # Each informative frame adds three constraints (its camera centre against the IMU
    # centre), so seven unknowns need three informative frames: four images, not three.
    s3 = synthetic(n_frames=3)
    preints = _preintegrate_frames(s3['imu'], s3['times'], np.zeros(3))
    A, b, w = ff.linear_rows(_samples(s3, 2), preints, s3['r_ctoi'], p_cini=s3['p_cini'])
    assert ff.rank_deficient(A, w)
    s4 = synthetic(n_frames=4)
    preints = _preintegrate_frames(s4['imu'], s4['times'], np.zeros(3))
    A, b, w = ff.linear_rows(_samples(s4, 2), preints, s4['r_ctoi'], p_cini=s4['p_cini'])
    assert not ff.rank_deficient(A, w)


def test_ransac_rejects_outliers():
    s = synthetic()
    samples = _samples(s, 3)
    rng = np.random.default_rng(4)
    for smp in samples:                                                 # corrupt 30 % of the points
        bad = rng.random(len(smp['z'])) < 0.3
        smp['xyz_c0'][bad] *= rng.uniform(1.5, 3., size=(bad.sum(), 1))
    preints = _preintegrate_frames(s['imu'], s['times'], np.zeros(3))
    A, b, w, frame = ff.linear_rows(samples, preints, s['r_ctoi'], p_cini=s['p_cini'],
                                    return_frames=True)
    x, inliers, info = ff.ransac_linear(A, b, w, frame, G, None, iterations=100, threshold=0.01,
                                        min_per_frame=2, rng=np.random.default_rng(5))
    assert abs(x[0] - s['scale']) < 0.02 * s['scale']
    assert 0.55 < info['inlier_fraction'] < 0.8 and info['frames_covered'] == 4


def test_refinement_recovers_biases_and_returns_psd_covariance():
    s = synthetic()
    bg_true, ba_true = np.array([0.006, -0.009, 0.007]), np.array([0.05, -0.03, 0.02])
    imu = [ImuSample(m.t, m.gyro + bg_true, m.accel + ba_true) for m in s['imu']]
    res = ff.feedforward_initialize(s, s['times'], imu, s['r_ctoi'], s['p_cini'],
                                    ff.default_config(), G, np.random.default_rng(0))
    assert res.status == 'released', res.reason
    assert abs(res.scale - s['scale']) < 0.03 * s['scale']
    np.testing.assert_allclose(res.bias_gyro, bg_true, atol=3e-3)
    assert np.all(res.sigmas > 0) and np.all(np.isfinite(res.sigmas)) and len(res.sigmas) == 15
    np.testing.assert_allclose(res.gravity / G, s['g'] / G, atol=0.03)


def test_bootstrap_state_is_gravity_aligned_and_at_the_last_frame():
    from davio.init.jpl import quat_to_rot
    s = synthetic()
    res = ff.feedforward_initialize(s, s['times'], s['imu'], s['r_ctoi'], s['p_cini'],
                                    ff.default_config(), G, np.random.default_rng(0))
    assert res.status == 'released', res.reason
    st = res.state
    assert st['t'] == pytest.approx(s['times'][-1])
    r_i_from_g = quat_to_rot(np.asarray(st['q_GtoI']))                  # R_{I<-G}
    g_body = r_i_from_g @ np.array([0., 0., G])                         # gravity as read in I_N
    r_in_from_i0 = s['rot'](s['times'][-1]).T
    np.testing.assert_allclose(g_body, r_in_from_i0 @ s['g'], atol=0.3)
    v_true = np.linalg.norm(vel_i0(s['times'][-1], s['v0']))
    assert np.linalg.norm(st['v']) == pytest.approx(v_true, abs=0.05)
    assert set(st) >= {'t', 'q_GtoI', 'p', 'v', 'bg', 'ba', 'sigmas'}


def test_free_mode_holds_the_lever_arm_unless_asked_and_rejects_without_frames():
    s = synthetic()
    cfg = ff.default_config()
    held = ff.feedforward_initialize(s, s['times'], s['imu'], s['r_ctoi'], None, cfg, G,
                                     np.random.default_rng(0))
    assert held.status == 'released', held.reason
    np.testing.assert_allclose(held.lever_arm, 0.)                      # default: prior mean
    assert held.info['lever_arm_held_at_prior']
    res = ff.feedforward_initialize(s, s['times'], s['imu'], s['r_ctoi'], None,
                                    dict(cfg, estimate_lever_arm=True), G, np.random.default_rng(0))
    assert res.status == 'released', res.reason
    assert np.linalg.norm(perpendicular(res.lever_arm - s['p_cini'])) < 0.02
    two = synthetic(n_frames=2)
    res2 = ff.feedforward_initialize(two, two['times'], two['imu'], two['r_ctoi'], two['p_cini'],
                                     cfg, G, np.random.default_rng(0))
    assert res2.status == 'rejected' and 'rank' in res2.reason
