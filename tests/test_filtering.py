import numpy as np
import pytest

from davio.mapper.filtering import filter_depth, usable

K = np.array([[30., 0., 16.], [0., 30., 16.], [0., 0., 1.]])


def wall(depth=3., images=2):
    """Images of a wall from one place."""
    return dict(depth=np.full((images, 32, 32), depth, np.float32), conf=np.ones((images, 32, 32), np.float32),
                poses=np.tile(np.eye(4), (images, 1, 1)), intrinsics=np.tile(K, (images, 1, 1)))


def test_usable_depth_is_in_range_with_a_confidence_and_outside_the_sky(mapping):
    depth = np.array([[1., .1, 25., np.nan], [1., 1., 1., 1.]])
    conf = np.array([[1., 1., 1., 1.], [0., np.nan, 1., 1.]])
    sky = np.array([[0, 0, 0, 0], [0, 0, 1, 0]])
    assert usable(depth, mapping, sky, conf).tolist() == [[True, False, False, False], [False, False, False, True]]
    assert usable(depth, mapping).tolist() == [[True, False, False, False], [True, True, True, True]]


def test_depth_that_no_other_image_confirms_is_dropped_with_its_edge(mapping):
    sm = wall()
    assert filter_depth(sm, mapping) == 1. and sm['valid'].all()        # two images that see the wall alike keep it whole
    sm['depth'][0, 10:20, 10:20] = 7.               # a patch that floats behind the wall in one image
    kept = filter_depth(sm, mapping)
    assert 0. < kept < 1.
    assert sm['valid'][0, 5, 5] and not sm['valid'][0, 15, 15]
    assert not sm['valid'][0, 9, 12]                # on the wall, beside the patch: a depth edge
    assert sm['valid'][1, 5, 5] and not sm['valid'][1, 15, 15]      # the other image is not confirmed there either


def test_the_least_confident_pixels_of_each_image_are_dropped(mapping):
    sm = wall()
    sm['conf'][0] = np.linspace(1., 2., 32 * 32).reshape(32, 32)
    filter_depth(sm, mapping)
    assert sm['valid'][0].mean() == pytest.approx(1. - mapping['depth_conf_quantile'], abs=1e-3)
    assert not sm['valid'][0, 0, 0] and sm['valid'][0, -1, -1]


def test_sky_and_depth_out_of_range_are_dropped(mapping):
    sm = wall()
    sm['sky'] = np.zeros((2, 32, 32), bool)
    sm['sky'][:, :8] = True
    sm['depth'][:, 20:] = 50.
    filter_depth(sm, mapping)
    assert not sm['valid'][:, :8].any() and not sm['valid'][:, 20:].any() and sm['valid'][:, 10:18].all()
