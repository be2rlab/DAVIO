"""Reference support must depend on the reference, never on the method being scored."""
import types

import numpy as np

from davio.data.types import GroundTruth
from davio.eval import reference_support as rs

K = np.array([[200., 0., 100.], [0., 200., 100.], [0., 0., 1.]])
SHAPE = (200, 200)


def _dataset(camera_positions, seq='synthetic'):
    """Minimal stand-in: identity camera-to-body, cameras looking down +z at given spots."""
    t = np.arange(len(camera_positions), dtype=float) * .05
    R = np.repeat(np.eye(3)[None], len(camera_positions), axis=0)
    gt = GroundTruth(t=t, p=np.asarray(camera_positions, float), R=R,
                     v=np.zeros((len(t), 3)), valid=np.ones(len(t), bool))
    return types.SimpleNamespace(
        seq=seq, R_CtoI=np.eye(3), p_IC=np.zeros(3),
        groundtruth=lambda: gt,
        groundtruth_provenance=lambda: dict(variant='synthetic', sha256='0' * 64))


def _grid(z, half=.5, step=.02):
    span = np.arange(-half, half + 1e-9, step)
    x, y = np.meshgrid(span, span)
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])


def test_occluding_wall_hides_the_surface_behind_it():
    near_wall, far_wall = _grid(2.), _grid(5.)
    reference = np.vstack([near_wall, far_wall])
    ds = _dataset([[0., 0., 0.]])
    points, crop, visible, prov = rs.build(
        ds, reference, 'refsha', K, SHAPE, near=.1, far=20., period_s=0.,
        voxel=.0, tolerance=.03, dilate=5)
    front = visible[:len(near_wall)]
    behind = visible[len(near_wall):]
    assert front.mean() > .9, 'the wall the camera is pointed at must be visible'
    assert behind.mean() < .1, 'the wall directly behind it must be occluded'
    assert crop.all(), 'everything is within far of the one camera centre'
    assert prov['visible_points'] == int(visible.sum())

    # Why the dilation exists: an undilated buffer lets the far wall through the gaps
    # between the near wall's samples, which at 1 cm spacing and 1 m range are ~4.6 px.
    _p, _c, undilated, _q = rs.build(ds, reference, 'refsha', K, SHAPE, near=.1, far=20.,
                                     period_s=0., voxel=.0, tolerance=.03, dilate=1)
    assert undilated[len(near_wall):].mean() > .3


def test_support_is_independent_of_any_estimate_and_reproducible():
    reference = _grid(3.)
    a = rs.build(_dataset([[0., 0., 0.], [.1, 0., 0.]]), reference, 'sha', K, SHAPE,
                 near=.1, far=20., period_s=0., voxel=.0)
    b = rs.build(_dataset([[0., 0., 0.], [.1, 0., 0.]]), reference, 'sha', K, SHAPE,
                 near=.1, far=20., period_s=0., voxel=.0)
    np.testing.assert_array_equal(a[2], b[2])
    assert a[3]['support_id'] == b[3]['support_id']
    assert a[3]['mask_sha256'] == b[3]['mask_sha256']
    # A different reference trajectory is a different support, and says so.
    c = rs.build(_dataset([[0., 0., 0.]]), reference, 'sha', K, SHAPE,
                 near=.1, far=20., period_s=0., voxel=.0)
    assert c[3]['support_id'] != a[3]['support_id']


def test_far_limit_and_frustum_bound_the_support():
    reference = np.vstack([_grid(3.), _grid(3.) + [10., 0., 0.]])   # second patch off-axis
    ds = _dataset([[0., 0., 0.]])
    _pts, crop, visible, _p = rs.build(ds, reference, 'sha', K, SHAPE,
                                       near=.1, far=4., period_s=0., voxel=.0)
    half = len(reference) // 2
    assert visible[:half].mean() > .9 and not visible[half:].any()
    assert crop[:half].all() and not crop[half:].any(), 'far crop must exclude the distant patch'


def test_voxel_downsample_is_order_stable():
    points = np.array([[0., 0., 0.], [.001, 0., 0.], [1., 1., 1.]])
    out = rs.voxel_downsample(points, .01)
    assert len(out) == 2
    np.testing.assert_allclose(out[0], [0., 0., 0.])
