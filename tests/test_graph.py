import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from davio.mapper import sim3
from davio.mapper.graph import UP, Graph, PointFactor, PoseFactor, ScaleFactor

SIGMAS = np.r_[np.full(3, .05), np.full(3, .02)]
POINTS = np.random.default_rng(0).normal(size=(30, 3)) + [0, 0, 3]


def node(scale=1., roll=.2, pitch=-.15, yaw=.7, t=(1., -.5, .3)):
    T = np.eye(4)
    T[:3, :3] = scale * Rotation.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    T[:3, 3] = t
    return T


def shift(x):
    T = np.eye(4)
    T[0, 3] = x
    return T


def graph_of(second):
    """A fixed node at the origin and a free one."""
    graph = Graph()
    graph.add_node('a', np.eye(4), free=False)
    graph.add_node('b', second)
    return graph


@pytest.mark.parametrize('factor', [
    PoseFactor('a', 'b', node(t=(1.1, -.4, .2), yaw=.6), SIGMAS, huber_delta=1e6),
    ScaleFactor('a', 'b', .1, .03, huber_delta=1e6),
    PointFactor('a', 'b', POINTS, 1.02 * POINTS + .01, sigma=.1, sigma_radial=.1, sigma_lateral=.01, huber_delta=1e6)])
def test_factor_jacobians_match_finite_differences(factor):
    nodes = {'a': node(1.1, .1, .2, -.3, (0., .2, .1)), 'b': node(1.3)}
    _, _, blocks = factor.linearize(nodes)
    for key in nodes:
        for k in range(7):
            step = np.zeros(7)
            step[k] = 1e-6
            plus, minus = dict(nodes), dict(nodes)
            plus[key], minus[key] = nodes[key] @ sim3.exp(step), nodes[key] @ sim3.exp(-step)
            numeric = (factor.linearize(plus, False)[1] - factor.linearize(minus, False)[1]) / 2e-6
            np.testing.assert_allclose(numeric, blocks[key][:, k], atol=1e-6)


def test_the_chart_is_the_retraction_to_first_order():
    T = node(1.3)
    for k in range(5):
        step = np.zeros(5)
        step[k] = 1e-5
        np.testing.assert_allclose(Graph.retract(T, step), T @ sim3.exp(Graph.chart(T) @ step), atol=1e-9)


def test_a_loop_on_a_drifting_chain_brings_it_back():
    graph = Graph()
    graph.add_node('0', np.eye(4), free=False)
    for i in range(1, 5):       # the odometry says 1.1 m a step; the loop says the end is at 4 m
        graph.add_node(str(i), shift(1.1 * i))
        graph.add(PoseFactor(str(i - 1), str(i), shift(1.1), np.r_[np.full(3, .1), np.full(3, .01)]))
    graph.add(PoseFactor('0', '4', shift(4.), np.r_[np.full(3, 1e-4), np.full(3, .01)]))
    graph.optimize()
    np.testing.assert_allclose([graph.nodes[str(i)][0, 3] for i in range(5)], [0., 1., 2., 3., 4.], atol=1e-4)


def test_a_node_follows_a_loop_in_position_and_yaw_but_keeps_its_roll_and_pitch():
    start = node()
    wanted = node(roll=.4, pitch=.1, yaw=.9, t=(1.3, -.4, .5))
    graph = graph_of(start)
    graph.add(PoseFactor('a', 'b', wanted, SIGMAS))
    graph.optimize(max_iterations=30)
    found = graph.nodes['b']
    np.testing.assert_allclose(found[:3, 3], wanted[:3, 3], atol=1e-6)
    np.testing.assert_allclose(found[:3, :3].T @ UP, start[:3, :3].T @ UP, atol=1e-9)
    turn = Rotation.from_matrix(found[:3, :3] @ start[:3, :3].T).as_rotvec()
    assert abs(turn[2]) > .1 and np.linalg.norm(turn[:2]) < 1e-9
    assert sim3.scale(found) == pytest.approx(1.)


def test_matched_points_put_a_depth_error_on_the_scale_and_leave_the_pose():
    pose = node()
    seen = (POINTS - pose[:3, 3]) @ pose[:3, :3]           # the points in the second node's frame
    graph = graph_of(pose)
    graph.add(PointFactor('a', 'b', POINTS, seen / 1.12, sigma=.1, sigma_radial=.1, sigma_lateral=.01))
    graph.optimize()
    assert sim3.scale(graph.nodes['b']) == pytest.approx(1.12, abs=1e-5)
    np.testing.assert_allclose(sim3.pose(graph.nodes['b']), pose, atol=1e-5)


def test_a_scale_factor_sets_the_ratio_of_two_depth_scales():
    start = shift(1.)
    start[:3, :3] *= 1.4
    graph = graph_of(start)
    graph.add(PoseFactor('a', 'b', shift(1.), SIGMAS))          # satisfied whatever the scales
    graph.add(ScaleFactor('a', 'b', .2, .1))
    graph.optimize()
    assert np.log(sim3.scale(graph.nodes['b'])) == pytest.approx(.2, abs=1e-6)
    np.testing.assert_allclose(sim3.pose(graph.nodes['b']), shift(1.), atol=1e-9)


def test_a_graph_needs_a_fixed_node_and_poses_tied_to_it():
    graph = graph_of(shift(1.))
    graph.add(ScaleFactor('a', 'b', 0., 1.))                # a scale factor ties no pose
    with pytest.raises(ValueError, match='disconnected'):
        graph.optimize()
    free = Graph()
    free.add_node('a', np.eye(4))
    with pytest.raises(ValueError, match='gauge'):
        free.optimize()
