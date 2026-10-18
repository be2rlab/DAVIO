"""The false-loop probe must meet the real innovation gate, and say what it did."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _mapper(tmp_path, translation_m, rotation_deg):
    from test_system import _backend_config
    from davio_mapper.online import OnlineMapper
    mapper = OnlineMapper(_backend_config(sparse_alignment=True, inject_false_loop_every=1,
                                          inject_false_loop_translation_m=translation_m,
                                          inject_false_loop_rotation_deg=rotation_deg,
                                          inject_seed=4), tmp_path)
    mapper.nodes['000000'] = dict(pose=np.eye(4), odom=np.eye(4), t=0., path_m=0.)
    here = np.eye(4)
    here[0, 3] = 1.
    K = np.array([[300., 0., 160.], [0., 300., 120.], [0., 0., 1.]])
    rng = np.random.default_rng(0)
    kit = dict(pixels=rng.uniform([0, 0], [320, 240], size=(120, 2)).astype(np.float32),
               depth=rng.uniform(1., 4., size=120), K=K, descriptors=None)
    record = dict(pose=here, odom=here.copy(), t=10., path_m=1., features=kit)
    return mapper, record


def test_small_consistent_false_loop_is_admitted_as_exactly_consistent_points(tmp_path):
    mapper, record = _mapper(tmp_path, translation_m=.05, rotation_deg=1.)
    edge, report = mapper.inject_false_loop(record, '000001')
    assert report['admitted'] and edge is not None and edge['kind'] == 'sparse'
    assert abs(report['innovation_m'] - .05) < .02
    assert edge['points_a'].shape == edge['points_b'].shape == (80, 3)
    # The correspondences agree with ONE rigid transform exactly: the aliasing worst case.
    a, b = edge['points_a'], edge['points_b']
    ca, cb = a - a.mean(0), b - b.mean(0)
    u, _s, vt = np.linalg.svd(ca.T @ cb)
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    np.testing.assert_allclose(cb @ rotation.T, ca, atol=1e-9)


def test_large_false_loop_is_refused_by_the_gate_and_still_reported(tmp_path):
    mapper, record = _mapper(tmp_path, translation_m=2., rotation_deg=1.)
    edge, report = mapper.inject_false_loop(record, '000001')
    assert edge is None and report is not None and not report['admitted']
    assert report['innovation_m'] > 1.5
