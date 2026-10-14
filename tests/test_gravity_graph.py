import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy.spatial.transform import Rotation
from davio_mapper.graph import Factor, Graph, PointFactor
from davio_mapper import sim3


def tilted(scale=1.3, roll=.2, pitch=-.15, yaw=.7, t=(1., -.5, .3)):
    T = np.eye(4)
    T[:3, :3] = scale * Rotation.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    T[:3, 3] = t
    return T


def gravity_in_body(T):
    """World z axis seen from the node: unchanged iff roll/pitch are unchanged."""
    return sim3.pose(T)[:3, :3].T @ np.array([0., 0., 1.])


@pytest.mark.parametrize('dims', [5, 1])
def test_chart_gradient_matches_retraction(dims):
    rng = np.random.default_rng(3)
    points = rng.normal(size=(40, 3)) + [0, 0, 3]
    Tb = tilted()
    target = (points - Tb[:3, 3]) @ np.linalg.inv(Tb[:3, :3]).T
    graph = Graph()
    graph.add_node('a', np.eye(4), dimensions=0)
    graph.add_node('b', Tb @ sim3.exp(np.r_[.02, -.01, .03, .01, .02, -.03, .05]), dimensions=dims)
    graph.add(PointFactor('a', 'b', points, target, sigma_radial=.1, sigma_lateral=.01))
    graph.add(Factor('a', 'b', 'loop', measurement=sim3.pose(Tb), projected=True,
                     sigmas=np.r_[np.full(3, .05), np.full(3, .02)]))
    _cost, residual, j, offsets = graph._system(graph.nodes, True)
    gradient = np.asarray(j.T @ residual)[offsets['b']:offsets['b'] + dims]
    for k in range(dims):
        step = np.zeros(dims)
        step[k] = 1e-6
        costs = []
        for sign in (1., -1.):
            trial = dict(graph.nodes)
            trial['b'] = graph.retract('b', graph.nodes['b'], sign * step)
            costs.append(graph._system(trial, False)[0])
        numeric = (costs[0] - costs[1]) / (2 * step[k])
        assert numeric == pytest.approx(gradient[k], rel=1e-5, abs=1e-6)


def test_gravity_node_keeps_vio_roll_and_pitch_under_a_tilting_loop():
    graph = Graph()
    graph.add_node('a', np.eye(4), dimensions=0)
    start = tilted(scale=1.)
    graph.add_node('b', start, dimensions=5)
    # A loop that asks for a different roll/pitch and yaw and a shifted translation.
    wanted = tilted(scale=1., roll=.4, pitch=.1, yaw=.9, t=(1.3, -.4, .5))
    graph.add(Factor('a', 'b', 'loop', measurement=wanted, projected=True,
                     sigmas=np.r_[np.full(3, .05), np.full(3, .02)]))
    info = graph.optimize(max_iterations=30)
    assert info['final_cost'] < info['initial_cost']
    T = graph.nodes['b']
    np.testing.assert_allclose(gravity_in_body(T), gravity_in_body(start), atol=1e-9)
    np.testing.assert_allclose(T[:3, 3], wanted[:3, 3], atol=1e-6)      # translation followed
    yaw = Rotation.from_matrix(sim3.pose(T)[:3, :3] @ sim3.pose(start)[:3, :3].T).as_rotvec()
    assert abs(yaw[2]) > .1 and np.linalg.norm(yaw[:2]) < 1e-9       # rotated about world z only
    assert np.isclose(sim3.scale(T), 1.)                              # projected factor: no scale


def test_depth_perturbation_is_absorbed_by_scale_when_the_pose_is_right():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(50, 3)) + [0, 0, 3]
    Tb = tilted(scale=1.)
    local_b = (points - Tb[:3, 3]) @ np.linalg.inv(Tb[:3, :3]).T
    graph = Graph()
    graph.add_node('a', np.eye(4), dimensions=0)
    graph.add_node('b', Tb, dimensions=5)
    # b's learned depth is 12% too short; its pose is exactly right.
    graph.add(PointFactor('a', 'b', points, local_b / 1.12, sigma_radial=.1, sigma_lateral=.01))
    graph.optimize()
    T = graph.nodes['b']
    assert np.isclose(sim3.scale(T), 1.12, atol=1e-4)
    np.testing.assert_allclose(sim3.pose(T), Tb, atol=1e-5)


def test_scale_only_node_never_moves_its_pose():
    graph = Graph()
    graph.add_node('a', np.eye(4), dimensions=0)
    start = tilted(scale=1.)
    graph.add_node('b', start, dimensions=1)
    wanted = tilted(scale=1., t=(2., 0., 0.))
    graph.add(Factor('a', 'b', 'loop', measurement=wanted, projected=True,
                     sigmas=np.r_[np.full(3, .05), np.full(3, .02)]))
    graph.add(Factor('a', 'b', 'anchor', log_ratio=.2, sigmas=np.array([.1])))
    graph.optimize()
    np.testing.assert_allclose(sim3.pose(graph.nodes['b']), start, atol=1e-12)
    assert np.isclose(np.log(sim3.scale(graph.nodes['b'])), .2, atol=1e-6)


def test_gravity_and_scale_nodes_reject_other_dimensions():
    graph = Graph()
    with pytest.raises(ValueError):
        graph.add_node('x', np.eye(4), dimensions=4)


@pytest.mark.parametrize('nodes', ['gravity', 'scale'])
def test_mapper_node_chart_reaches_the_graph(nodes):
    """C2 keeps every node's VIO roll/pitch through a verified loop; C0 moves no pose."""
    from test_system import _backend_config, _drive, pathlib_tmp
    from davio_mapper.online import OnlineMapper
    rng = np.random.default_rng(3)
    revisit = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    others = [rng.integers(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(5)]
    waypoints = [[0, 0, 0], [.8, 0, 0], [1.6, 0, 0], [2.4, 0, 0], [1.6, .6, 0], [.8, .6, 0], [.4, .2, 0]]
    mapper = OnlineMapper(_backend_config(loop_min_separation_s=2., graph_nodes=nodes), pathlib_tmp())
    final = _drive(mapper, waypoints, [revisit] + others + [revisit])[-1]
    assert final['accepted_loops'] >= 1
    moved = 0
    for node in mapper.nodes.values():
        pose, odom = sim3.pose(node['pose']), node['odom']
        np.testing.assert_allclose(pose[:3, :3].T @ [0, 0, 1.], odom[:3, :3].T @ [0, 0, 1.], atol=1e-12)
        moved += not np.allclose(pose, odom, atol=1e-9)
    if nodes == 'scale':
        assert moved == 0
    else:
        assert moved >= 1 and final['optimizer']['final_cost'] < final['optimizer']['initial_cost']
    index = json.loads((mapper.root / 'map_index.json').read_text())
    assert index['graph_nodes'] == nodes and 'gravity_axis' in index


def test_mapper_rejects_a_node_chart_without_scale_coupling():
    from test_system import _backend_config, pathlib_tmp
    from davio_mapper.online import OnlineMapper
    with pytest.raises(ValueError):
        OnlineMapper(_backend_config(graph_nodes='gravity', scale_coupling=False), pathlib_tmp())
    with pytest.raises(ValueError):
        OnlineMapper(_backend_config(graph_nodes='tilt'), pathlib_tmp())
