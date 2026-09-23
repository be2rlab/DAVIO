import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def grid():
    sys.path.insert(0, str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('compose_grid', ROOT / 'scripts/compose_grid.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def line(total, hold, t0=10., t1=150.):
    return dict(t0=t0, t1=t1, fps=30, total_frames=total, hold_frames=hold)


def test_a_shorter_panel_shows_the_frame_for_the_same_sensor_time(grid):
    """Walkthrough 36 s + 2 s hold, overhead 24 s + 2 s: same moment, frame for frame."""
    from compose_video import sensor_time
    master, panel = line(1080, 60), line(720, 60)
    for i in range(0, 1080, 7):
        k = grid.panel_frame(i, master, panel)
        # The panel frame's own sensor time is within half a panel frame of the master's.
        step = (panel['t1'] - panel['t0']) / (panel['total_frames'] - 1)
        assert abs(sensor_time(k, panel) - sensor_time(i, master)) <= step / 2 + 1e-9


def test_the_ends_line_up_and_the_hold_stays_on_the_finished_map(grid):
    master, panel = line(1080, 60), line(720, 60)
    assert grid.panel_frame(0, master, panel) == 0
    assert grid.panel_frame(1079, master, panel) == 719          # last build frame
    assert grid.panel_frame(1080, master, panel) == 720          # into the panel's hold
    assert grid.panel_frame(1139, master, panel) == 779          # and no further
    assert grid.panel_frame(5000, master, panel) == 779


def test_frames_are_requested_in_order_so_the_readers_never_seek(grid):
    master, panel = line(1080, 60), line(720, 60)
    requests = [grid.panel_frame(i, master, panel) for i in range(1140)]
    assert all(b >= a for a, b in zip(requests, requests[1:]))


def test_panels_parse_with_and_without_their_render_timing(grid):
    assert grid.parse_panel('camera') == dict(name='camera')
    fly = grid.parse_panel('fly=renders/x/map_build_fly.mp4:36:2')
    assert (fly['name'], str(fly['path']), fly['duration'], fly['hold']) == \
        ('fly', 'renders/x/map_build_fly.mp4', 36.0, 2.0)
    bare = grid.parse_panel('top=v.mp4')
    assert bare['duration'] is None and bare['hold'] is None
    with pytest.raises(SystemExit):
        grid.parse_panel('fly')


def test_a_panel_is_fitted_into_its_cell_without_distortion(grid):
    wide = np.full((100, 400, 3), 200, np.uint8)                  # 4:1 strip
    cell = grid.fit(wide, 200, 200, (0, 0, 0))
    assert cell.shape == (200, 200, 3)
    rows = np.where(cell.max(axis=(1, 2)) > 0)[0]
    assert len(rows) == 50                                         # 200 wide -> 50 tall
