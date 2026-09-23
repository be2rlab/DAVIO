import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def page():
    sys.path.insert(0, str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('make_page_assets',
                                                  ROOT / 'scripts/make_page_assets.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_ply(path, points, colours):
    header = (f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'end_header\n').encode()
    body = b''.join(struct.pack('<fffBBB', *p, *c) for p, c in zip(points, colours))
    path.write_bytes(header + body)


def test_ply_round_trips_through_the_reader(page, tmp_path):
    points = np.array([[0., 0., 0.], [1.5, -2.25, 3.], [-4., 5., 6.5]])
    colours = np.array([[0, 0, 0], [255, 128, 7], [10, 20, 30]], np.uint8)
    path = tmp_path / 'map.ply'
    write_ply(path, points, colours)
    read_points, read_colours = page.read_ply(path)
    np.testing.assert_allclose(read_points, points)
    np.testing.assert_array_equal(read_colours, colours)


def test_quantization_costs_half_a_step_and_no_more(page, tmp_path):
    rng = np.random.default_rng(0)
    points = rng.normal(size=(5000, 3)) * (20., 8., 3.)
    low = points.min(axis=0)
    span = points.max(axis=0) - low
    scale = np.maximum(span, 1e-6) / 65535.
    quantized = np.clip(np.round((points - low) / scale), 0, 65535).astype('<u2')
    decoded = low + quantized.astype(float) * scale       # exactly what the page does
    assert np.abs(decoded - points).max() <= span.max() / 65535. / 2 * 1.001
    assert np.abs(decoded - points).max() < 1.2e-3


def test_thinning_hits_the_budget_and_keeps_the_extent(page):
    rng = np.random.default_rng(1)
    # A dense blob plus a sparse arm: a plain random sample would eat the arm.
    blob = rng.normal(size=(40000, 3)) * 0.2
    arm = np.stack([np.linspace(0, 30, 2000), np.zeros(2000), np.zeros(2000)], 1)
    points = np.concatenate([blob, arm])
    colours = np.full((len(points), 3), 128, np.uint8)
    thinned, thinned_colours = page.thin(points, colours, 5000)
    assert len(thinned) <= 5000 and len(thinned_colours) == len(thinned)
    assert thinned[:, 0].max() > 29.0      # the far end of the arm survived


def test_thinning_leaves_a_small_cloud_alone(page):
    points = np.zeros((10, 3))
    colours = np.zeros((10, 3), np.uint8)
    thinned, _ = page.thin(points, colours, 5000)
    assert len(thinned) == 10


def test_trajectory_is_thinned_but_keeps_its_ends(page, tmp_path):
    rows = np.zeros((500, 8))
    rows[:, 0] = np.linspace(100., 200., 500)
    rows[:, 1] = np.linspace(0., 10., 500)
    rows[:, 7] = 1.
    (tmp_path / 'map_trajectory.tum').write_text(
        '\n'.join(' '.join(f'{v:.6f}' for v in row) for row in rows))
    out = page.trajectory(tmp_path, step_m=0.5)
    xs = out['points'][0::3]
    assert out['duration_s'] == 100.0
    assert xs[0] == pytest.approx(0.0, abs=1e-6)
    assert xs[-1] == pytest.approx(10.0, abs=1e-6)
    assert len(xs) < 100                      # 0.5 m steps over 10 m, not 500 nodes


def test_numbers_come_from_the_runs_own_artifacts(page, tmp_path):
    run = tmp_path / 'r01_remap'
    run.mkdir()
    (run / 'replay.json').write_text(json.dumps(dict(
        source_run='runs/r01', ate_raw_m=0.207, ate_map_final_m=0.109,
        accepted_loops=10, submaps_mapped=146)))
    (run / 'evaluation.json').write_text(json.dumps(dict(
        ate=dict(position_m=0.207, orientation_deg=1.12, n_states=4008))))
    out = page.numbers(run)
    assert out == dict(ate_online_m=0.207, ate_map_m=0.109, loops=10, submaps=146,
                       orientation_deg=1.12, poses=4008)


def test_a_run_without_scores_reports_nothing_rather_than_guessing(page, tmp_path):
    run = tmp_path / 'bare'
    run.mkdir()
    assert page.numbers(run) == {}
    assert page.trajectory(run) is None
