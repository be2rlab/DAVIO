import numpy as np
import pytest

from davio.mapper.triangulation import centred_pose_variance, triangulate

K = np.array([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
POINTS = np.random.default_rng(0).uniform([-1, -1, 3], [1, 1, 6], (300, 3))


def camera(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def pixels(pose):
    local = (POINTS - pose[:3, 3]) @ pose[:3, :3]
    return (local @ K.T)[:, :2] / local[:, 2:3], local[:, 2]


A, B = camera(0.), camera(.6)
UA, (UB, ZB) = pixels(A)[0], pixels(B)


def test_exact_matches_give_the_exact_depth_in_the_second_camera():
    out = triangulate(UA, UB, K, K, A, B)
    assert out['ok'].all() and out['reprojection_px'].max() < 1e-6
    np.testing.assert_allclose(out['depth_b'], ZB, atol=1e-9)
    to_a, to_b = POINTS - A[:3, 3], POINTS - B[:3, 3]
    cosine = np.sum(to_a * to_b, axis=1) / np.linalg.norm(to_a, axis=1) / np.linalg.norm(to_b, axis=1)
    np.testing.assert_allclose(out['parallax_deg'], np.degrees(np.arccos(cosine)), atol=1e-6)


def test_the_stated_sigma_is_the_spread_under_pixel_noise_and_under_a_baseline_error():
    rng = np.random.default_rng(1)
    logs = [np.log(triangulate(UA + rng.normal(size=UA.shape), UB + rng.normal(size=UB.shape), K, K, A, B)['depth_b'])
            for _ in range(300)]
    stated = triangulate(UA, UB, K, K, A, B, pixel_sigma=1.)['sigma_log']
    assert np.median(np.std(logs, axis=0) / stated) == pytest.approx(1., abs=.1)
    logs = []
    for _ in range(300):
        moved = B.copy()
        moved[:3, 3] += .02 * rng.normal(size=3)
        logs.append(np.log(triangulate(UA, UB, K, K, A, moved)['depth_b']))
    stated = triangulate(UA, UB, K, K, A, B, pixel_sigma=0., pose_sigma_m=.02)['sigma_log']
    assert np.median(np.std(logs, axis=0) / stated) == pytest.approx(1., abs=.1)


def test_rays_that_are_parallel_or_meet_behind_the_cameras_give_no_depth():
    still = triangulate(UA, UA, K, K, A, A)
    assert not still['ok'].any() and np.isinf(still['sigma_log']).all()
    assert not triangulate(UB, UA, K, K, A, B)['ok'].any()          # the pixels swapped: the rays diverge


def test_the_pose_variance_is_what_a_baseline_error_leaves_once_the_common_shift_is_removed():
    rng = np.random.default_rng(5)
    exact = np.log(ZB)
    left = []
    for _ in range(400):
        moved = B.copy()
        moved[:3, 3] += .01 * rng.normal(size=3)
        r = np.log(triangulate(UA, UB, K, K, A, moved)['depth_b']) - exact
        left.append(r - np.median(r))
    stated = centred_pose_variance(UA, UB, K, K, A, B, .01)
    assert np.median(np.var(left, axis=0) / stated) == pytest.approx(1., abs=.25)
    whole = triangulate(UA, UB, K, K, A, B, pixel_sigma=0., pose_sigma_m=.01)['sigma_log'] ** 2
    assert np.median(whole / stated) > 4.               # most of a baseline error is common to the pair
    assert not centred_pose_variance(UA, UB, K, K, A, B, 0.).any()
