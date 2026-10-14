import numpy as np
import pytest

from davio_mapper import sim3
from davio_mapper.mapping import world_points


def submap(n_frames=5, baseline=.4, size=24):
    """A synthetic window: frames spread along x, each looking at a slanted wall."""
    rng = np.random.default_rng(0)
    intrinsics = np.tile(np.array([[120., 0., size / 2], [0., 120., size / 2], [0., 0., 1.]]),
                         (n_frames, 1, 1))
    poses = np.tile(np.eye(4), (n_frames, 1, 1))
    centre = (n_frames - 1) // 2
    for i in range(n_frames):
        poses[i, 0, 3] = (i - centre) * baseline       # centre frame stays at the origin
        poses[i, 1, 3] = .05 * (i - centre)
    depth = (2. + rng.uniform(0, .5, (n_frames, size, size))).astype(np.float32)
    return dict(depth=depth, intrinsics=intrinsics, poses=poses,
                rgb=rng.integers(0, 255, (n_frames, size, size, 3), dtype=np.uint8),
                valid=np.ones((n_frames, size, size), bool))


def node(scale, angle=.3, translation=(1.5, -2., .25)):
    """A Sim(3) node pose with a real scale, rotation and translation."""
    from scipy.spatial.transform import Rotation
    out = np.eye(4)
    out[:3, :3] = scale * Rotation.from_rotvec([0., 0., angle]).as_matrix()
    out[:3, 3] = translation
    return out


def viewer_points(sm, index, transform):
    """Exactly what scripts/view_run.py does: scale in the points, pose on the entity."""
    scale_only = np.eye(4) * sim3.scale(transform)
    scale_only[3, 3] = 1.
    local, _rgb = world_points(sm, index, scale_only, 1)
    placement = sim3.pose(transform)
    return local @ placement[:3, :3].T + placement[:3, 3]


def naive_points(sm, index, transform):
    """The trap: no scale in the points, the whole similarity on the entity."""
    local, _rgb = world_points(sm, index, np.eye(4), 1)
    return local @ transform[:3, :3].T + transform[:3, 3]


@pytest.mark.parametrize('scale', [0.85, 0.9933, 1.0, 1.17])
def test_every_frame_lands_where_the_exported_map_puts_it(scale):
    sm, transform = submap(), node(scale)
    for index in range(len(sm['depth'])):
        reference, _rgb = world_points(sm, index, transform, 1)   # what fuse_map writes
        assert np.allclose(viewer_points(sm, index, transform), reference, atol=1e-9), \
            f'frame {index} at scale {scale}'


def test_the_centre_frame_alone_cannot_catch_the_mistake():
    """Why the bug hid: on the centre frame the two paths agree exactly."""
    sm, transform = submap(), node(1.15)
    centre = (len(sm['depth']) - 1) // 2
    assert np.allclose(sm['poses'][centre], np.eye(4)), 'the centre frame is the node origin'
    assert np.allclose(naive_points(sm, centre, transform),
                       viewer_points(sm, centre, transform), atol=1e-9)


def test_the_naive_path_displaces_the_outer_frames_by_the_scale_error():
    """And the size of the error is (s-1) times that frame's baseline."""
    scale, baseline = 1.15, .4
    sm, transform = submap(baseline=baseline), node(scale)
    outer = len(sm['depth']) - 1
    centre = (len(sm['depth']) - 1) // 2
    reference, _rgb = world_points(sm, outer, transform, 1)
    error = np.abs(naive_points(sm, outer, transform) - reference).max()
    expected = abs(scale - 1.) * np.linalg.norm(sm['poses'][outer][:3, 3] - sm['poses'][centre][:3, 3])
    assert error > 1e-3, 'the naive path must actually be wrong, or this test proves nothing'
    assert error == pytest.approx(expected, rel=.3)


def test_a_unit_scale_node_hides_the_difference_entirely():
    """Pose conditioning drives the residual scale to ~1, which is why this went unnoticed."""
    sm, transform = submap(), node(1.0)
    for index in range(len(sm['depth'])):
        assert np.allclose(naive_points(sm, index, transform),
                           viewer_points(sm, index, transform), atol=1e-9)


def test_the_split_preserves_the_node_scale():
    transform = node(0.87)
    scale_only = np.eye(4) * sim3.scale(transform)
    scale_only[3, 3] = 1.
    assert sim3.scale(scale_only) == pytest.approx(0.87)
    assert np.allclose(sim3.pose(scale_only), np.eye(4))
    # pose() keeps the rotation and translation, and drops the scale.
    assert sim3.scale(sim3.pose(transform)) == pytest.approx(1.)
    assert np.allclose(sim3.pose(transform)[:3, 3], transform[:3, 3])
