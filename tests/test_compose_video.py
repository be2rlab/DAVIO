import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def compose():
    sys.path.insert(0, str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('compose_video',
                                                  ROOT / 'scripts/compose_video.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tum(path, t0, t1, n=50):
    rows = np.zeros((n, 8))
    rows[:, 0] = np.linspace(t0, t1, n)
    rows[:, 7] = 1.
    path.write_text('\n'.join(' '.join(f'{v:.6f}' for v in r) for r in rows) + '\n')


def test_schedule_is_render_builds_own(compose):
    """Frame 0 is the first pose, the last rendered frame the last, the hold stays there."""
    line = dict(t0=100., t1=200., fps=30, total_frames=301, hold_frames=60)
    assert compose.sensor_time(0, line) == 100.
    assert compose.sensor_time(150, line) == pytest.approx(150.)
    assert compose.sensor_time(300, line) == 200.
    assert compose.sensor_time(330, line) == 200.          # holding on the finished map
    # render_build's loop, verbatim, for every frame including the hold
    t0, t1, total = 100., 200., 301
    for i in range(total + 60):
        expected = t0 + (t1 - t0) * min(1., i / max(1, total - 1))
        assert compose.sensor_time(i, line) == expected


def test_timeline_prefers_the_sidecar(compose, tmp_path):
    video = tmp_path / 'map_build_fly.mp4'
    video.write_bytes(b'')
    sidecar = dict(t0=1., t1=2., fps=24, total_frames=10, hold_frames=3)
    Path(str(video) + '.timeline.json').write_text(json.dumps(sidecar))
    assert compose.timeline(video, tmp_path, duration=99., hold_s=9., fps=30) == sidecar


def test_timeline_from_arguments_reads_the_renderers_trajectory(compose, tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    tum(run / 'trajectory.tum', 0., 999.)                  # lower priority: must be ignored
    tum(run / 'map_trajectory_final.tum', 50., 180.)
    line = compose.timeline(tmp_path / 'v.mp4', run, duration=24., hold_s=2., fps=30)
    assert (line['t0'], line['t1']) == (50., 180.)
    assert (line['total_frames'], line['hold_frames']) == (720, 60)


def test_a_video_without_a_sidecar_needs_its_duration(compose, tmp_path):
    with pytest.raises(SystemExit):
        compose.timeline(tmp_path / 'v.mp4', tmp_path, duration=None)


def test_frames_are_ordered_by_timestamp_not_by_name(compose, tmp_path):
    for stamp in (9_000_000_000, 10_000_000_000, 100_000_000):
        (tmp_path / f'{stamp}.jpg').write_bytes(b'')
    (tmp_path / 'notes.txt').write_text('')
    stamps, paths = compose.frame_index(tmp_path)
    assert list(stamps) == [0.1, 9.0, 10.0]
    assert [p.stem for p in paths] == ['100000000', '9000000000', '10000000000']


def test_camera_frames_are_found_from_the_runs_container_path(compose, tmp_path, monkeypatch):
    monkeypatch.setattr(compose, 'ROOT', tmp_path)
    (tmp_path / 'data/ori/r01/davio/cam0/data').mkdir(parents=True)
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'run.json').write_text(json.dumps(dict(
        data_root='/workspace/data/ori', sequence='r01')))
    assert compose.image_directory(run) == tmp_path / 'data/ori/r01/davio/cam0/data'
