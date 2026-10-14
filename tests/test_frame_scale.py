import numpy as np
import pytest

from davio.mapper.frame_scale import refine_frame_scales

K = np.array([[100., 0., 64.], [0., 100., 48.], [0., 0., 1.]])
ERRORS = np.array([1., 1.15, .9, 1.05, 1.2])


def window(errors, noise=0.):
    """Cameras 10 cm apart that see the same 200 points; the depth of image i is wrong by errors[i]."""
    rng = np.random.default_rng(0)
    points = rng.uniform([-1, -1, 2], [1, 1, 4], size=(200, 3))
    kits, poses = [], []
    for i, error in enumerate(errors):
        pose = np.eye(4)
        pose[0, 3] = .1 * i
        local = points - pose[:3, 3]
        kits.append(dict(pixels=(local @ K.T)[:, :2] / local[:, 2:3], descriptors=np.arange(200.)[:, None],
                         depth=local[:, 2] * error * (1 + noise * rng.normal(size=200)), K=K, shape=(96, 128)))
        poses.append(pose)
    return kits, np.array(poses)


def test_the_images_of_a_window_are_brought_to_one_common_scale(mapping, matcher):
    kits, poses = window(ERRORS)
    scales, info = refine_frame_scales(kits, poses, mapping, matcher)
    np.testing.assert_allclose(scales * ERRORS, np.exp(np.mean(np.log(ERRORS))), atol=2e-3)
    assert np.exp(np.mean(np.log(scales))) == pytest.approx(1.)       # the window's own scale is not touched
    assert info['pairs_used'] == 7 and info['rms_after_rel'] < .1 * info['rms_before_rel']


def test_gross_depth_errors_on_a_fifth_of_the_keypoints_barely_move_the_scales(mapping, matcher):
    kits, poses = window(ERRORS, noise=.02)
    rng = np.random.default_rng(1)
    for kit in kits:
        wrong = rng.random(200) < .2
        kit['depth'][wrong] *= rng.uniform(1.5, 3., size=wrong.sum())
    scales, _ = refine_frame_scales(kits, poses, mapping, matcher)
    np.testing.assert_allclose(scales * ERRORS, np.exp(np.mean(np.log(ERRORS))), atol=.03)


def test_images_without_matches_keep_their_depth(mapping, matcher):
    kits, poses = window(ERRORS)
    for kit in kits:
        kit['descriptors'] = None
    scales, info = refine_frame_scales(kits, poses, mapping, matcher)
    np.testing.assert_array_equal(scales, 1.)
    assert info['pairs_used'] == 0 and info['rms_after_rel'] is None
