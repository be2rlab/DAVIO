"""Committed map versions at 25/50/75 % of sensor time are frozen as published."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_mapper_freezes_quartile_snapshots_that_export_from_immutable_archives():
    from test_system import _backend_config, _drive, pathlib_tmp
    from davio_mapper.online import OnlineMapper
    from davio_mapper.mapping import fuse_map
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(5)]
    mapper = OnlineMapper(_backend_config(loops_enabled=False), pathlib_tmp())
    mapper.interval = (0., 4.2)           # keyframes at 0.2 s, 1.2 s, ... -> quartiles at 1.05, 2.1, 3.15 s
    _drive(mapper, [[i * .3, 0, 0] for i in range(5)], images)
    tags = sorted(p.name for p in (mapper.root / 'snapshots').iterdir())
    assert tags == ['q25', 'q50', 'q75']
    for tag, nodes in zip(tags, (2, 3, 4)):
        index = json.loads((mapper.root / 'snapshots' / tag / 'map_index.json').read_text())
        assert index['snapshot']['nodes'] == nodes and len(index['submaps']) == nodes
        assert (mapper.root / 'snapshots' / tag / 'active.ply').is_file()
        submaps, poses = [], {}
        for key, entry in index['submaps'].items():
            path = mapper.root / 'snapshots' / tag / entry['file']
            assert path.resolve().is_file()
            with np.load(path, allow_pickle=False) as z:
                sm = {k: z[k].copy() for k in z.files}
            sm.update(id=key, frame_ids=entry['frame_ids'])
            submaps.append(sm)
            poses['s:' + key] = np.asarray(entry['T_map_submap'])
        assert len(fuse_map(submaps, poses, voxel_size=.05, pixel_step=8)[0]) > 0
    # The final map is the 100 % version; nothing is written for it twice.
    assert mapper.snapshots_done == {'q25', 'q50', 'q75'}


def test_support_until_freezes_cameras_in_time(tmp_path):
    from types import SimpleNamespace
    from davio.eval import reference_support
    t = np.arange(0., 10., .05)
    gt = SimpleNamespace(t=t, R=np.repeat(np.eye(3)[None], len(t), 0), p=np.c_[t, 0 * t, 0 * t], valid=None)
    ds = SimpleNamespace(groundtruth=lambda: gt, R_CtoI=np.eye(3), p_IC=np.zeros(3))
    full, _ = reference_support.camera_track(ds, .1)
    early, times = reference_support.camera_track(ds, .1, until=2.)
    assert len(early) < len(full) and times.max() <= 2. and len(early) == 21
