import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from davio_mapper import admission, sim3


def _cfg(**over):
    from test_system import _backend_config
    cfg = _backend_config(sparse_alignment=True, sparse_adjacent=False, graph_nodes='gravity',
                          selective_admission=True, loop_min_separation_s=2.)
    cfg.update(over)
    return cfg


def _edge(points_b, offset=(0., 0., 0.), innovation_m=.02, arc=1.):
    """Group between a (gauge) and b whose points agree with b displaced by `offset`."""
    tb = np.eye(4)
    tb[:3, 3] = [1., 0., 0.]
    true_b = tb.copy()
    true_b[:3, 3] += offset
    world = points_b @ true_b[:3, :3].T + true_b[:3, 3]
    nodes = {'a': dict(pose=np.eye(4), odom=np.eye(4), path_m=0.),
             'b': dict(pose=tb, odom=tb.copy(), path_m=arc)}
    edge = dict(a='a', b='b', kind='sparse', group='loop', points_a=world, points_b=points_b,
                held_out=(world[:5], points_b[:5]), measurement=np.linalg.inv(np.eye(4)) @ true_b,
                info=dict(innovation_m=innovation_m, innovation_deg=.1))
    return edge, nodes


def test_true_pose_perturbation_is_detectable_after_scale_elimination():
    rng = np.random.default_rng(1)
    spread = rng.normal(size=(80, 3)) * [1., 1., .5] + [0., 0., 3.]
    edge, nodes = _edge(spread, offset=(.3, 0., 0.))
    info = admission.pose_information(edge, nodes, .1, .1, .01)
    assert info['lambda_min'] >= 1. and np.isfinite(info['condition'])
    # Depth-only ambiguity: every match on one ray from b -> yaw/translation along the
    # ray are indistinguishable from scale; finite, small information, no exception.
    ray = np.outer(np.linspace(2., 4., 80), [0., 0., 1.])
    edge, nodes = _edge(ray)
    degenerate = admission.pose_information(edge, nodes, .1, .1, .01)
    assert degenerate['lambda_min'] < 1e-6 and np.isfinite(degenerate['lambda_min'])
    state, reason, _ = admission.decide(edge, [edge], nodes, _cfg())
    assert (state, reason) == ('deferred', 'conditioning')


def test_drift_model_rejects_and_missing_confirmation_defers():
    rng = np.random.default_rng(2)
    spread = rng.normal(size=(80, 3)) * [1., 1., .5] + [0., 0., 3.]
    edge, nodes = _edge(spread, innovation_m=.5, arc=1.)      # limit = 3 * max(0.02, 0.02) = 0.06
    assert admission.decide(edge, [edge], nodes, _cfg())[0] == 'rejected'
    edge, nodes = _edge(spread, innovation_m=.02, arc=1.)
    state, reason, checks = admission.decide(edge, [edge], nodes, _cfg())
    assert (state, reason) == ('deferred', 'unconfirmed') and checks['held_out_median_whitened'] < 2.


def _revisit_drive(cfg, waypoints, revisits):
    from test_system import _drive, pathlib_tmp
    from davio_mapper.online import OnlineMapper
    rng = np.random.default_rng(3)
    revisit = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    images = [revisit if i in revisits else rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
              for i in range(len(waypoints))]
    mapper = OnlineMapper(cfg, pathlib_tmp())
    return mapper, _drive(mapper, waypoints, images)


def test_loop_group_moves_a_pose_only_after_a_later_window_confirms_it():
    waypoints = [[0, 0, 0], [.8, 0, 0], [1.6, 0, 0], [2.4, 0, 0], [1.6, .6, 0], [.8, .6, 0],
                 [.2, .1, 0], [.22, .1, 0]]
    mapper, results = _revisit_drive(_cfg(), waypoints, revisits={0, 6, 7})
    assert results[6]['accepted_loops'] >= 1 and results[7]['accepted_loops'] >= 1
    first = results[6]['optimizer']['admission']
    assert first['counts'].get('deferred', 0) >= 1 and not first['counts'].get('admitted')
    groups = [e for e in mapper.edges if e.get('group') == 'loop']
    for e in groups:   # held-out matches are disjoint from the fitted ones, by construction
        fitted = {tuple(np.round(p, 9)) for p in e['points_a']}
        assert e['held_out'] is None or not fitted & {tuple(np.round(p, 9)) for p in e['held_out'][0]}
    final = results[7]['optimizer']['admission']
    assert final['counts'].get('admitted', 0) >= 1, final
    keys = list(mapper.nodes)
    moved = [k for k in keys if not np.allclose(sim3.pose(mapper.nodes[k]['pose']), mapper.nodes[k]['odom'], atol=1e-9)]
    assert keys[6] in moved
    for node in mapper.nodes.values():   # C2 structure underneath: VIO roll/pitch retained
        np.testing.assert_allclose(sim3.pose(node['pose'])[:3, :3].T @ [0, 0, 1.],
                                   node['odom'][:3, :3].T @ [0, 0, 1.], atol=1e-12)


def test_deferred_and_rejected_groups_cannot_move_any_pose():
    waypoints = [[0, 0, 0], [.8, 0, 0], [1.6, 0, 0], [2.4, 0, 0], [1.6, .6, 0], [.8, .6, 0], [.2, .1, 0]]
    mapper, results = _revisit_drive(_cfg(), waypoints, revisits={0, 6})
    assert results[6]['accepted_loops'] >= 1
    assert results[6]['optimizer']['admission']['counts'] == {'deferred': results[6]['accepted_loops']}
    for node in mapper.nodes.values():
        np.testing.assert_allclose(sim3.pose(node['pose']), node['odom'], atol=1e-9)
    # A coherent false loop (exactly self-consistent points, wrong by 0.3 m / 3 deg) meets
    # the same gate: it is rejected by the drift model or left deferred, never admitted.
    mapper, results = _revisit_drive(_cfg(inject_false_loop_every=1, inject_false_loop_translation_m=.3,
                                          inject_false_loop_rotation_deg=3., inject_seed=1),
                                     waypoints[:6], revisits=set())
    offered = [x for r in results for x in r.get('injected_loops', [])]
    assert offered and any(x['admitted'] for x in offered), 'the probe never reached admission'
    states = [e['admission']['state'] for e in mapper.edges if e.get('group') == 'loop']
    assert states and 'admitted' not in states
    for node in mapper.nodes.values():
        np.testing.assert_allclose(sim3.pose(node['pose']), node['odom'], atol=1e-9)
