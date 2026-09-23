import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _mapper(**over):
    from test_system import _backend_config, pathlib_tmp
    from davio_mapper.online import OnlineMapper
    cfg = _backend_config(loops_enabled=False, graph_nodes='scale')
    cfg.update(over)
    return OnlineMapper(cfg, pathlib_tmp())


def _drive_scale_disagreement(mapper):
    from test_system import _keyframe
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    grid = np.linspace(0, 2 * np.pi, 128)
    depth = 2. + .6 * np.outer(np.sin(3 * grid), np.cos(3 * grid))
    for i, fit in enumerate([2., 2., 2.4, 2., 2.]):
        offsets = np.zeros((5, 3))
        offsets[:, 0] = np.linspace(0., .1, 5)
        metric, prediction = _keyframe(rgb, depth, np.array([i * .15, 0., 0.]) + offsets, fit)
        stamps = [int((i + j * .05) * 1e9) for j in range(5)]
        assert mapper.add(dict(stamps=stamps, times=[s * 1e-9 for s in stamps], camera_poses=metric), prediction)['status'] == 'mapped'


def test_no_overlap_scale_factors_leaves_a_wrong_window_fit_uncorrected():
    from davio_mapper import sim3
    mapper = _mapper(overlap_scale_factors=False)
    _drive_scale_disagreement(mapper)
    assert not any(e['kind'] == 'submap_scale' for e in mapper.edges)
    # No relative-scale evidence: every residual scale stays at its window fit (1.0).
    assert all(np.isclose(sim3.scale(n['pose']), 1., atol=1e-9) for n in mapper.nodes.values())


def test_no_scale_anchors_lets_overlap_evidence_act_without_a_prior():
    from davio_mapper import sim3
    anchored, free = _mapper(), _mapper(scale_anchors=False)
    _drive_scale_disagreement(anchored)
    _drive_scale_disagreement(free)
    assert any(e['kind'] == 'submap_scale' for e in free.edges)
    s_anchored = [sim3.scale(n['pose']) for n in anchored.nodes.values()]
    s_free = [sim3.scale(n['pose']) for n in free.nodes.values()]
    # Both pull submap 2 back; without the prior nothing holds the others at 1.0, so the
    # two solutions differ, and the first node still fixes the scale gauge.
    assert s_anchored[2] < .95 and s_free[2] < .95
    assert not np.allclose(s_anchored, s_free, atol=1e-6)
    assert np.isclose(s_free[0], 1.)
    index_free = free._graph(1, free.edges)
    assert not any(getattr(f, 'kind', '') == 'anchor' for f in index_free.factors)
