"""Loop retrieval modes and the XFeat mutual-nearest-neighbour matcher arm."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


class _Scores:
    """Stand-in feature backend: appearance agreement read from a table."""
    backend = 'xfeat_lighterglue'

    def __init__(self, table):
        self.table = table

    def retrieval_score(self, a, b, budget=512, min_cossim=.82):
        return self.table[a['name']]


def _mapper(tmp_path, mode, scores):
    from test_system import _backend_config
    from davio_mapper.online import OnlineMapper
    mapper = OnlineMapper(_backend_config(loop_retrieval=mode, loop_retrieval_min_mnn=25,
                                          max_loop_candidates=4), tmp_path)
    mapper.feature_backend = _Scores(scores)
    for name, t, x in (('far_revisit', 0., 50.), ('near_weak', 1., 1.), ('too_recent', 9., 0.5)):
        pose = np.eye(4)
        pose[0, 3] = x                           # where the DRIFTED chain thinks it is
        mapper.nodes[name] = dict(pose=pose, odom=pose.copy(), t=t, path_m=t,
                                  features=dict(name=name))
    record = dict(pose=np.eye(4), t=10., features=dict(name='query'))
    return mapper, record


def test_spatial_misses_a_drifted_revisit_that_appearance_finds(tmp_path):
    scores = {'far_revisit': 120, 'near_weak': 5, 'too_recent': 300}
    mapper, record = _mapper(tmp_path, 'spatial', scores)
    assert mapper.candidates(record) == ['near_weak']
    mapper.cfg['loop_retrieval'] = 'appearance'
    # too_recent scores highest but violates the time separation; near_weak is below floor.
    assert mapper.candidates(record) == ['far_revisit']
    assert mapper.last_candidate_source == {'far_revisit': 'appearance'}


def test_union_keeps_both_and_labels_the_overlap(tmp_path):
    mapper, record = _mapper(tmp_path, 'union', {'far_revisit': 120, 'near_weak': 40,
                                                 'too_recent': 300})
    assert mapper.candidates(record) == ['near_weak', 'far_revisit']
    assert mapper.last_candidate_source == {'near_weak': 'both', 'far_revisit': 'appearance'}
    mapper.cfg['loop_retrieval'] = 'nearest'
    with pytest.raises(ValueError):
        mapper.candidates(record)


def test_xfeat_mnn_returns_mutual_nearest_neighbours():
    pytest.importorskip('torch')
    if not (ROOT / 'thirdparty/accelerated_features/weights/xfeat.pt').is_file():
        pytest.skip('XFeat weights not present')
    from davio_mapper.features import Features
    features = Features(dict(feature_backend='xfeat_mnn', xfeat_features=64,
                             xfeat_path=str(ROOT / 'thirdparty/accelerated_features')))
    rng = np.random.default_rng(3)
    first = rng.normal(size=(12, 64)).astype(np.float32)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    order = rng.permutation(12)
    second = first[order] + rng.normal(scale=.01, size=(12, 64)).astype(np.float32)
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    kit = lambda d: dict(pixels=np.zeros((len(d), 2), np.float32), descriptors=d, shape=(8, 8))
    pairs = features.match(kit(first), kit(second))
    expected = {(int(order[j]), j) for j in range(12)}
    assert {tuple(p) for p in pairs} == expected
    assert features.retrieval_score(kit(first), kit(second), budget=12) == 12
    # Unrelated descriptors fall under the cosine floor and are not counted as agreement.
    noise = rng.normal(size=(12, 64)).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    assert features.retrieval_score(kit(first), kit(noise), budget=12) == 0
