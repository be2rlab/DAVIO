import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load():
    sys.path.insert(0, str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('replay_map', ROOT / 'scripts/replay_map.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tum(path, n=5):
    rows = np.zeros((n, 8))
    rows[:, 0] = 100. + np.arange(n)
    rows[:, 1] = np.arange(n) * .5
    rows[:, 7] = 1.
    path.write_text('\n'.join(' '.join(f'{v:.6f}' for v in r) for r in rows) + '\n')


def test_a_session_without_a_reference_is_reported_unscored(tmp_path, monkeypatch):
    replay_map = load()
    source, out = tmp_path / 'source', tmp_path / 'out'
    (out / 'map').mkdir(parents=True)
    source.mkdir()
    (source / 'run.json').write_text(json.dumps(dict(
        dataset='phone', data_root=str(tmp_path), sequence='walk', interval_start=100.)))
    tum(out / 'trajectory.tum')
    tum(out / 'map_trajectory_final.tum')
    node = dict(T_map_submap=np.eye(4).tolist(), T_odom_submap=np.eye(4).tolist())
    (out / 'map/map_index.json').write_text(json.dumps(dict(submaps={'000000': node})))

    class NoReference:
        def groundtruth(self):
            raise FileNotFoundError('A phone session ships no reference trajectory')

    import davio.data
    monkeypatch.setattr(davio.data, 'open_dataset', lambda *a, **k: NoReference())
    results = [dict(status='mapped', sparse_edges=2, accepted_loops=1, loops=[],
                    sparse=[], optimizer=None)]
    report = replay_map.score(out, source, results, cfg={}, elapsed=1.0)

    assert report['submaps_mapped'] == 1 and report['accepted_loops'] == 1
    assert report['ate_raw_m'] is None and report['ate_map_final_m'] is None
    assert report['ground_truth_used'] == 'none'
    assert 'no reference' in report['unscored_reason']
    json.dumps(report)                      # and it still serializes as replay.json


def test_depth_overrides_take_effect_only_when_the_replay_refilters(tmp_path, monkeypatch):
    replay_map = load()
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'run.json').write_text(json.dumps(dict(interval_start=0., interval_end=1.)))
    (source / 'trajectory.tum').write_text('')
    seen = []
    monkeypatch.setattr(replay_map, 'filter_depth',
                        lambda sm, cfg: seen.append(cfg['depth_max_m']) or 1.)

    class Mapper:
        def __init__(self, cfg, out):
            (out / 'map').mkdir(parents=True, exist_ok=True)

        def ingest(self, sm, *args, **kwargs):
            return dict(status='mapped')

    monkeypatch.setattr(replay_map, 'OnlineMapper', Mapper)
    monkeypatch.setattr(replay_map, 'final_trajectory', lambda out: [])
    entry = dict(T_odom_submap=np.eye(4).tolist(), timestamp=1., scale=1.)
    monkeypatch.setattr(replay_map, 'load_submaps',
                        lambda source: iter([('000000', entry, {'valid': np.ones(1)})]))
    cfg = dict(depth_filter=True, depth_max_m=12.0)

    replay_map.replay(source, tmp_path / 'recorded', cfg, keep_dense=True)
    assert seen == []                                    # default: the archive as recorded
    replay_map.replay(source, tmp_path / 'refiltered', cfg, keep_dense=True, refilter=True)
    assert seen == [12.0]                                # the override reached the filter
