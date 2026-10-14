import numpy as np
import pytest

from davio.mapper import sim3
from davio.mapper.fusion import fuse_map, read_ply, world_points, write_ply

K = np.array([[400., 0., 64.], [0., 400., 64.], [0., 0., 1.]])


def flat(frame_ids, depth=1., key='0', size=4):
    """A submap whose images all look down +z from the origin at a wall."""
    n = len(frame_ids)
    return dict(id=key, frame_ids=frame_ids, depth=np.full((n, size, size), depth), conf=np.ones((n, size, size)),
                rgb=np.full((n, size, size, 3), 100, np.uint8), intrinsics=np.tile(np.eye(3), (n, 1, 1)),
                poses=np.tile(np.eye(4), (n, 1, 1)))


@pytest.mark.parametrize('scale', (1., 1.12, .85))
def test_the_node_scale_multiplies_the_depth_and_not_the_baseline_between_cameras(scale):
    point, baseline = np.array([.3, -.2, 4.]), np.array([.6, 0., 0.])
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[1, :3, 3] = baseline
    depth = np.full((2, 128, 128), np.nan)
    for i in range(2):          # the point in both images, its depth stored too small by the scale
        local = point - poses[i, :3, 3]
        u, v = np.rint((K @ local)[:2] / local[2]).astype(int)
        depth[i, v, u] = local[2] / scale
    sm = dict(depth=depth, conf=np.ones_like(depth), rgb=np.zeros((2, 128, 128, 3), np.uint8), poses=poses,
              intrinsics=np.stack([K, K]))
    placed = sim3.exp(np.array([.4, -.3, .2, .05, -.02, .03, np.log(scale)]))
    expected = sim3.pose(placed)[:3, :3] @ point + sim3.pose(placed)[:3, 3]
    for i in range(2):
        np.testing.assert_allclose(world_points(sm, i, placed, pixel_step=1)[0], expected[None], atol=1e-6)


def test_pixels_without_depth_or_confidence_or_in_the_sky_give_no_point():
    sm = flat(['a'])
    sm['depth'][0, 0, 0], sm['depth'][0, 0, 1], sm['conf'][0, 0, 2] = np.nan, -1., 0.
    sm['valid'] = np.ones((1, 4, 4), bool)
    sm['valid'][0, 0, 3] = False
    sm['sky'] = np.zeros((1, 4, 4), bool)
    sm['sky'][0, 1] = True
    xyz, rgb = world_points(sm, 0, np.eye(4), pixel_step=1)
    assert len(xyz) == len(rgb) == 8 and xyz[:, 1].min() == 2.      # rows 2 and 3 are left


def test_an_image_two_windows_hold_is_taken_once_from_the_window_it_is_most_central_in():
    first, second = flat(['a', 'b', 'c'], 1., '0'), flat(['b', 'c', 'd'], 2., '1')
    first['depth'][2] = 7.          # c at the edge of the first window
    second['depth'][0] = 5.         # b at the edge of the second
    second['depth'][1] = 3.         # c in its middle
    xyz, _ = fuse_map([first, second], {'0': np.eye(4), '1': np.eye(4)}, voxel_size=.01, pixel_step=1)
    depths, counts = np.unique(np.round(xyz[:, 2], 6), return_counts=True)
    assert depths.tolist() == [1., 2., 3.] and counts.tolist() == [16, 16, 16]


def test_the_points_of_a_voxel_become_their_mean():
    near, far = flat(['a'], 1., '0', size=1), flat(['b'], 1.02, '1', size=1)
    far['rgb'][:] = 200
    xyz, rgb = fuse_map([near, far], {'0': np.eye(4), '1': np.eye(4)}, voxel_size=.1, pixel_step=1)
    np.testing.assert_allclose(xyz, [[0., 0., 1.01]])
    assert rgb.tolist() == [[150, 150, 150]]
    with pytest.raises(ValueError, match='empty'):
        fuse_map([flat(['a'], np.nan)], {'0': np.eye(4)})


def test_a_cloud_written_is_the_cloud_read(tmp_path):
    points = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
    colours = np.zeros((50, 3), np.uint8)
    colours[:, 0] = 255
    write_ply(tmp_path / 'cloud.ply', points, colours)
    np.testing.assert_array_equal(read_ply(tmp_path / 'cloud.ply'), points)
    assert b'property uchar red\nproperty uchar green\nproperty uchar blue' in (tmp_path / 'cloud.ply').read_bytes()
    assert (tmp_path / 'cloud.ply').read_bytes().endswith(b'\xff\x00\x00')       # red comes first
