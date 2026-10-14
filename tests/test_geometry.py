"""Submap geometry conventions: what the residual scale is allowed to touch."""
import numpy as np
import pytest

from davio_mapper.mapping import world_points
from davio_mapper import sim3

K = np.array([[400., 0., 64.], [0., 400., 64.], [0., 0., 1.]])
SIZE = 128


def _two_camera_submap(point, baseline, stored_scale):
    poses = np.stack([np.eye(4), np.eye(4)])
    poses[1, :3, 3] = baseline
    depth = np.full((2, SIZE, SIZE), np.nan, np.float32)
    for i in range(2):
        camera = np.linalg.inv(poses[i])
        local = camera[:3, :3] @ point + camera[:3, 3]
        uv = (K @ local)[:2] / local[2]
        depth[i, int(round(uv[1])), int(round(uv[0]))] = local[2] / stored_scale
    return dict(depth=depth, rgb=np.zeros((2, SIZE, SIZE, 3), np.uint8), poses=poses,
                intrinsics=np.stack([K, K]), id='000000', frame_ids=['0', '1'], center=0)


@pytest.mark.parametrize('scale', (1.0, 1.12, 0.85))
def test_residual_scale_corrects_depth_not_the_metric_baseline(scale):
    point = np.array([.3, -.2, 4.])
    baseline = np.array([.6, 0., 0.])
    sm = _two_camera_submap(point, baseline, scale)
    node = sim3.exp(np.array([.4, -.3, .2, .05, -.02, .03, np.log(scale)]))

    first, _ = world_points(sm, 0, node, pixel_step=1)
    second, _ = world_points(sm, 1, node, pixel_step=1)
    expected = sim3.pose(node)[:3, :3] @ point + sim3.pose(node)[:3, 3]
    np.testing.assert_allclose(first[0], expected, atol=1e-6)
    np.testing.assert_allclose(second[0], expected, atol=1e-6)

    legacy, _ = world_points(sm, 1, node, pixel_step=1, legacy_scaled_baseline=True)
    offset = abs(scale - 1.) * np.linalg.norm(baseline)
    assert np.linalg.norm(legacy - second) == pytest.approx(offset, abs=1e-6)


def test_centre_frame_and_published_pose_agree_under_either_convention():
    """The centre frame is the one place the two conventions must coincide."""
    sm = _two_camera_submap(np.array([.1, .2, 3.]), np.array([.5, .1, 0.]), 1.2)
    node = sim3.exp(np.array([.2, .1, -.4, .02, .03, -.01, np.log(1.2)]))
    current, _ = world_points(sm, 0, node, pixel_step=1)
    legacy, _ = world_points(sm, 0, node, pixel_step=1, legacy_scaled_baseline=True)
    np.testing.assert_allclose(current, legacy, atol=1e-9)


def test_projected_pose_factor_cannot_express_a_depth_scale_disagreement():
    from davio_mapper.graph import Factor, Graph, PointFactor

    wrong = 1.3
    for kind, factor in (('pose', None), ('point', None)):
        graph = Graph()
        graph.add_node('a', np.eye(4), dimensions=0)
        start = np.eye(4)
        start[:3, :3] *= wrong          # node b carries a residual depth-scale error
        graph.add_node('b', start, dimensions=7)
        if kind == 'pose':
            graph.add(Factor('a', 'b', 'adjacent_pose', measurement=np.eye(4),
                             projected=True, sigmas=np.r_[np.full(3, .05),
                                                          np.full(3, np.radians(2.))],
                             huber_delta=2.))
        else:
            points = np.array([[.1, .2, 3.], [-.4, .1, 2.5], [.3, -.3, 4.]])
            graph.add(PointFactor('a', 'b', points, points))
        graph.optimize()
        recovered = sim3.scale(graph.nodes['b'])
        if kind == 'pose':
            assert abs(recovered - wrong) < 1e-6, 'a projected factor must leave scale free'
        else:
            assert abs(recovered - 1.) < 1e-3, 'a point factor must be able to fix the scale'


def test_mapper_dispatches_an_adjacent_pose_edge_as_a_projected_factor(tmp_path):
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
    from test_system import _backend_config
    from davio_mapper.online import OnlineMapper

    # A pose factor can only be exercised on nodes whose pose is free: the joint graph.
    mapper = OnlineMapper(_backend_config(adjacent_factor='pose', graph_nodes='sim3'), tmp_path)
    displaced = np.eye(4)
    displaced[0, 3] = 1.4                     # odometry says 1.0; the match says 1.0 too
    for key, pose, t in (('000000', np.eye(4), 0.), ('000001', displaced, 1.)):
        odom = np.eye(4)
        odom[0, 3] = t
        mapper.nodes[key] = dict(odom=odom, pose=pose.copy(), t=t, path_m=t, features=None)
    measurement = np.eye(4)
    measurement[0, 3] = 1.
    mapper.edges.append(dict(a='000000', b='000001', kind='adjacent_pose',
                             measurement=measurement, inliers=40,
                             sigmas=mapper.loop_sigmas(1.)))
    report = mapper.optimize()
    assert report['final_cost'] <= report['initial_cost']
    np.testing.assert_allclose(sim3.pose(mapper.nodes['000001']['pose'])[:3, 3],
                               [1., 0., 0.], atol=1e-4)


def test_invalid_adjacent_factor_is_not_silently_a_point_factor(tmp_path):
    from davio_mapper.online import OnlineMapper
    with pytest.raises(ValueError, match="adjacent_factor"):
        OnlineMapper({'adjacent_factor': 'poses'}, tmp_path)


def test_ingest_routes_verified_adjacent_matches_to_selected_factor(tmp_path, monkeypatch):
    from test_system import _backend_config, _keyframe
    from davio_mapper.online import OnlineMapper
    rgb = np.full((128, 128, 3), 100, dtype=np.uint8)
    for mode, expected in [('pose', 'adjacent_pose'), ('point', 'sparse')]:
        mapper = OnlineMapper(_backend_config(loops_enabled=False, scale_coupling=False,
                              sparse_alignment=True, sparse_adjacent=True,
                              adjacent_factor=mode), tmp_path / mode)
        monkeypatch.setattr(mapper, 'optimize', lambda: {})
        points = np.array([[0., 0., 3.], [.1, .2, 4.]])
        def verified(*args):
            mapper.last_correspondences = (points, points.copy())
            mapper.last_holdout = None
            measurement = np.eye(4)
            measurement[0, 3] = 1.
            return measurement, 40, 1., {'baseline_m': 1.}
        monkeypatch.setattr(mapper, 'loop', verified)
        for offset in (0., 1.):
            centers = np.array([[offset + i * .1, 0, 0] for i in range(5)])
            metric, pred = _keyframe(rgb, np.full((128, 128), 3.), centers, 1.)
            mapper.add(dict(camera_poses=metric, stamps=list(range(5)),
                            times=list(range(5))), pred)
        geometric = [edge for edge in mapper.edges if edge['kind'] != 'odometry']
        assert len(geometric) == 1
        assert geometric[0]['kind'] == expected
        assert ('points_a' in geometric[0]) == (mode == 'point')
