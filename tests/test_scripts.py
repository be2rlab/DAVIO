import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from davio.geometry import tum_line
from davio.mapper.fusion import read_ply, write_ply

MOVE = np.eye(4)                # the reference's world seen from a run's
MOVE[:3, :3], MOVE[:3, 3] = Rotation.from_rotvec([.05, -.11, .31]).as_matrix(), [1.3, -.7, .25]


# ---- run.py and score_trajectory.py ----

def test_a_run_is_refused_for_an_unknown_setting_or_into_an_existing_directory(script, tmp_path):
    arguments = ['--dataset', 'euroc', '--data', str(tmp_path), '--sequence', 'x']
    with pytest.raises(SystemExit):
        script('run').main([*arguments, '--out', str(tmp_path / 'run'), '--set', 'mapping.window_frame=4'])
    with pytest.raises(SystemExit):
        script('run').main([*arguments, '--out', str(tmp_path)])


def test_a_run_of_the_filter_alone_is_recorded_and_scored_against_the_reference(script, euroc, openvins, tmp_path):
    data, run = euroc(frames=40), tmp_path / 'run'
    done = script('run').main(['--dataset', 'euroc', '--data', str(data), '--sequence', 'V1_02_medium', '--out', str(run),
                               '--no-map', '--rate', '0', '--seed', '3', '--set', 'max_tracking_gap_s=1.5',
                               '--set', 'openvins.init_dyn_min_deg=10'])
    record = json.loads((run / 'run.json').read_text())
    assert done == 0 and record['state'] == 'completed' and record['poses'] == 38
    assert (record['dataset'], record['sequence'], record['seed'], record['rate']) == ('euroc', 'V1_02_medium', 3, 0.)
    assert (record['interval_start'], record['interval_end']) == pytest.approx((1., 2.95))
    assert record['config']['max_tracking_gap_s'] == 1.5 and record['config']['openvins']['init_dyn_min_deg'] == 10
    assert record['config']['mapping']['enabled'] is False and record['overrides'][0] == 'max_tracking_gap_s=1.5'
    assert 'init_dyn_min_deg: 10' in (run / 'filter_config/estimator.yaml').read_text()

    scores = script('score_trajectory').evaluate(run, data, reference='dataset')
    assert set(scores['ate']) == {'odometry', 'live'}           # no map, no final trajectory
    assert scores['ate']['odometry']['position_m'] < 1e-9 and scores['ate']['odometry']['n_matched'] == 38
    assert scores['coverage'] == pytest.approx(37 * .05 / 1.95)       # the two frames before the first pose are not covered
    assert scores['map_age_p50_s'] is None and scores['groundtruth']['variant'] == 'dataset'
    assert json.loads((run / 'evaluation.json').read_text())['ate']['live']['position_m'] < 1e-9


# ---- export_map.py ----

def test_the_exported_map_is_the_fused_submaps_with_each_image_corrected_unless_asked_not_to(
        script, wall, mapping, tmp_path, monkeypatch):
    e = .05 * np.random.default_rng(1).standard_normal(8)
    run, index = tmp_path / 'run', {}
    (run / 'map/submaps').mkdir(parents=True)
    for m, (sm, pose) in enumerate(zip(*wall(e))):
        np.savez(run / f'map/submaps/{m:06d}.npz', **sm)
        index[f'{m:06d}'] = dict(file=f'submaps/{m:06d}.npz', frame_ids=[f'{m}a', f'{m}b'], T_map_submap=pose.tolist())
    (run / 'map/map_index.json').write_text(json.dumps(dict(submaps=index)))
    (run / 'run.json').write_text(json.dumps(dict(config=dict(mapping=mapping))))
    export = script('export_map')
    for arguments in (['--run', str(run)], ['--run', str(run), '--no-correction', '--out', str(run / 'plain.ply')]):
        monkeypatch.setattr(sys, 'argv', ['export_map.py', *arguments])
        export.main()
    corrected, plain = read_ply(run / 'map.ply'), read_ply(run / 'plain.ply')
    assert np.std(corrected[:, 2]) < .3 * np.std(plain[:, 2])
    assert np.median(corrected[:, 2]) == pytest.approx(2. * np.exp(np.mean(e)), abs=.03)        # the wall stays where the mean scale puts it
    record = json.loads((run / 'map.correction.json').read_text())
    assert list(record['log_corrections']) == list(index) and record['images'] == 8
    assert abs(sum(sum(c) for c in record['log_corrections'].values())) < 1e-6
    assert not (run / 'plain.correction.json').exists()
    with pytest.raises(SystemExit):         # a map is never written over
        export.main()


# ---- score_map.py ----

@pytest.fixture
def scored(euroc, tmp_path):
    """A run on a sequence whose scan is a wall 3 m in front of its cameras. The run's world is the
    reference's, moved; its map is the wall, or the wall ``behind`` metres further."""
    data = euroc(frames=40)
    mav = data / 'V1_02_medium/mav0'
    angle = np.linspace(0., 1.2, 40)
    path = np.column_stack((np.cos(angle), np.sin(angle), .3 * angle))
    (mav / 'state_groundtruth_estimate0/data.csv').write_text('#t,p,p,p,qw,qx,qy,qz\n' + ''.join(
        f'{1_000_000_000 + i * 50_000_000},{x},{y},{z},1,0,0,0\n' for i, (x, y, z) in enumerate(path)))
    span = np.arange(-2., 2.01, .05)
    scan = np.column_stack((*(a.ravel() for a in np.meshgrid(span, span)), np.full(len(span) ** 2, 3.)))
    (mav / 'pointcloud0').mkdir()
    write_ply(mav / 'pointcloud0/data.ply', scan, np.zeros((len(scan), 3), np.uint8))

    def make(behind=0.):
        run = tmp_path / f'run_{behind}'
        run.mkdir()
        (run / 'run.json').write_text(json.dumps(dict(dataset='euroc', sequence='V1_02_medium', interval_start=1.)))
        to_run = np.linalg.inv(MOVE)
        with (run / 'map_trajectory_final.tum').open('w') as stream:
            for i, p in enumerate(path):
                body = np.eye(4)
                body[:3, 3] = p
                stream.write(tum_line(1. + .05 * i, to_run @ body))
        wall = (scan + [0., 0., behind]) @ to_run[:3, :3].T + to_run[:3, 3]
        write_ply(run / 'map.ply', wall, np.zeros((len(wall), 3), np.uint8))
        return ['--run', str(run), '--data', str(data), '--reference', 'dataset', '--support-cache', str(tmp_path / 'support')]
    return make


def surface_report(arguments):
    return json.loads((Path(arguments[1]) / 'surface.json').read_text())


def test_a_map_is_placed_on_the_scan_by_its_trajectory_and_scores_one_when_it_is_the_scan(script, scored):
    score = script('score_map')
    arguments = scored()
    assert score.main(arguments) == 0
    report = surface_report(arguments)
    np.testing.assert_allclose(report['alignment']['map_to_reference'], MOVE, atol=1e-6)
    assert report['alignment']['trajectory_rmse_m'] < 1e-6 and report['alignment']['fit'] == 'none'
    assert [report['thresholds'][t]['f1'] for t in ('0.02', '0.03', '0.05', '0.10')] == [1., 1., 1., 1.]
    assert report['estimate_points'] == report['reference_points'] == 81 * 81 and 0 < report['visible_points'] <= report['crop_points']
    assert report['accuracy']['mean_m'] < 1e-5 and report['support']['cameras'] == 20 and report['surface_fit'] is None
    with pytest.raises(SystemExit, match='exists'):         # a report is replaced only when asked
        score.main(arguments)
    assert score.main([*arguments, '--force']) == 0


def test_a_map_four_centimetres_off_the_scan_scores_nothing_below_that_until_it_is_registered(script, scored):
    arguments = scored(behind=.04)
    assert script('score_map').main(arguments) == 0
    report = surface_report(arguments)
    assert [report['thresholds'][t]['f1'] for t in ('0.02', '0.03', '0.05', '0.10')] == [0., 0., 1., 1.]
    assert report['accuracy']['median_m'] == pytest.approx(.04, abs=1e-5) and report['chamfer_mean_m'] == pytest.approx(.04, abs=1e-3)
    assert script('score_map').main([*arguments, '--fit', 'rigid', '--force']) == 0
    report = surface_report(arguments)
    assert report['thresholds']['0.02']['f1'] == 1. and report['surface_fit']['translation_m'] == pytest.approx(.04, abs=1e-3)
    assert report['surface_fit']['scale'] == pytest.approx(1.) and report['alignment']['fit'] == 'rigid'


# ---- the ORI scripts ----

def test_an_ori_scan_becomes_the_reference_cloud_of_its_sequence(script, tmp_path):
    points = np.random.default_rng(0).normal(size=(1000, 3)).astype(np.float32)
    (tmp_path / 'r09/r09_gt').mkdir(parents=True)
    header = ('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 1000\nHEIGHT 1\n'
              'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1000\nDATA binary\n')
    (tmp_path / 'r09/r09_gt/cloud_gt.pcd').write_bytes(header.encode() + points.tobytes())
    out = script('ori_reference').write_reference(tmp_path / 'r09')
    assert out == tmp_path / 'r09/davio/pointcloud0/data.ply'
    np.testing.assert_array_equal(read_ply(out), points)


@pytest.mark.parametrize('names, mode', [(['r01/r01_bag/metadata.yaml'], 0),       # incomplete
                                         (['../escape'], 0), (['/absolute'], 0), (['r01/link'], stat.S_IFLNK | 0o777)])
def test_an_incomplete_or_unsafe_ori_archive_leaves_nothing_behind(script, tmp_path, monkeypatch, names, mode):
    with zipfile.ZipFile(tmp_path / 'r01.zip', 'w') as archive:
        for name in names:
            member = zipfile.ZipInfo(name)
            member.external_attr = mode << 16
            archive.writestr(member, 'content')
    fetch = script('fetch_ori')
    monkeypatch.setattr(fetch, 'require_space', lambda path, size: None)        # whatever the disk of this machine holds
    with pytest.raises(ValueError):
        fetch.extract(tmp_path / 'r01.zip', tmp_path / 'r01', 'r01', {})
    assert not (tmp_path / 'r01').exists()


# ---- the launcher ----

def test_the_launcher_gives_each_script_the_arguments_it_takes(root, tmp_path):
    (tmp_path / 'scripts').mkdir()
    shutil.copy(root / 'davio', tmp_path / 'davio')
    (tmp_path / 'scripts/docker.sh').write_text('#!/usr/bin/env bash\necho "${DAVIO_GPUS:-all}: $*" >> calls.txt\n')
    (tmp_path / 'scripts/docker.sh').chmod(0o755)
    for run, dataset in (('x', 'ori'), ('V1_01_easy', 'euroc')):
        (tmp_path / 'runs' / run).mkdir(parents=True)
        (tmp_path / 'runs' / run / 'run.json').write_text(json.dumps(dict(dataset=dataset)))

    def davio(*arguments):
        (tmp_path / 'calls.txt').write_text('')
        done = subprocess.run(['bash', str(tmp_path / 'davio'), *arguments], capture_output=True, text=True)
        return done.returncode, (tmp_path / 'calls.txt').read_text().splitlines()
    assert davio('run', 'r01', '--dataset', 'ori', '--out', 'runs/x', '--rate', '0.3') == (0, [
        'all: python3 scripts/run.py --dataset ori --data data/ori --sequence r01 --out runs/x --rate 0.3',
        'none: python3 scripts/export_map.py --run runs/x',
        'none: python3 scripts/score_trajectory.py --run runs/x --data data/ori',
        'none: python3 scripts/score_map.py --run runs/x --data data/ori --force'])
    assert davio('run', 'V1_01_easy') == (0, [
        'all: python3 scripts/run.py --dataset euroc --data data/euroc --sequence V1_01_easy --out runs/V1_01_easy',
        'none: python3 scripts/export_map.py --run runs/V1_01_easy',
        'none: python3 scripts/score_trajectory.py --run runs/V1_01_easy --data data/euroc',
        'none: python3 scripts/score_map.py --run runs/V1_01_easy --data data/euroc --force'])
    assert davio('map', 'runs/x', '--no-correction', '--out', 'runs/x/fused.ply') == (0, [
        'none: python3 scripts/export_map.py --run runs/x --no-correction --out runs/x/fused.ply'])
    assert davio('test', '-k', 'graph') == (0, ['none: python3 -m pytest -q -k graph'])
    assert davio() == (1, []) and davio('fly') == (1, [])
