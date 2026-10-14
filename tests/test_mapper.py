"""The mapper on a wall of numbered dots 2 m in front of cameras that move sideways."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from davio.mapper import mapper as mapper_module, sim3
from davio.mapper.fusion import read_ply
from davio.mapper.mapper import Mapper, arrays

K = np.array([[100., 0., 64.], [0., 100., 48.], [0., 0., 1.]])
SHAPE = (96, 128)
CELL = .02                      # what one pixel covers on the wall
DOTS = np.random.default_rng(0).choice(400 * 96, 1300, replace=False)
DOTS = np.column_stack((DOTS % 400 - 100, DOTS // 400))             # column and row of each dot on the wall


def window(x, t, depth=1., drift=(0., 0., 0.), unit=1.):
    """Five cameras 4 cm and half a second apart, the centre one at wall column ``x`` and time
    ``t`` + 1 s. ``depth``: the factor by which the predicted depth is wrong, for all or per
    image; ``drift``: the error of the filter's poses; ``unit``: metres per unit of the prediction."""
    poses, images = np.tile(np.eye(4), (5, 1, 1)), np.zeros((5, *SHAPE, 3), np.uint8)
    for i in range(5):
        column = x + 2 * (i - 2)
        poses[i, 0, 3] = column * CELL
        u = DOTS[:, 0] - column + 64
        seen = np.flatnonzero((u >= 0) & (u < SHAPE[1]))
        images[i, DOTS[seen, 1], u[seen]] = np.column_stack((seen >> 8, seen & 255, np.full(len(seen), 255)))
    poses[:, :3, 3] += drift
    predicted = poses.copy()
    predicted[:, :3, 3] /= unit
    wrong = np.reshape(depth * np.ones(5), (5, 1, 1))
    prediction = SimpleNamespace(depth=(2. / unit * wrong * np.ones((5, *SHAPE))).astype(np.float32), conf=np.ones((5, *SHAPE), np.float32),
                                 processed_images=images, extrinsics=np.linalg.inv(predicted), intrinsics=np.tile(K, (5, 1, 1)))
    stamps = [int((t + .5 * i) * 1e9) for i in range(5)]
    return dict(stamps=stamps, times=[s * 1e-9 for s in stamps], camera_poses=poses), prediction


@pytest.fixture
def mapper(mapping, matcher, tmp_path, monkeypatch):
    monkeypatch.setattr(mapper_module, 'Features', lambda settings: matcher)
    cv2.setRNGSeed(0)
    return Mapper(mapping, tmp_path)


def stored(mapper, key):
    mapper.flush()
    return np.load(mapper.root / f'submaps/{key}.npz')


def test_a_window_is_stored_in_metres_on_the_cameras_of_the_filter(mapper):
    task, prediction = window(10, 0., unit=1.25)
    report = mapper.add(task, prediction)
    assert report['status'] == 'mapped' and report['submap'] == '000000'
    assert report['scale'] == pytest.approx(1.25) and report['alignment_rmse_m'] < 1e-9
    archive = stored(mapper, '000000')
    np.testing.assert_allclose(archive['depth'], 2., rtol=1e-6)
    np.testing.assert_allclose(archive['poses'][:, 0, 3], [-.08, -.04, 0., .04, .08], atol=1e-12)
    entry = json.loads((mapper.root / 'map_index.json').read_text())['submaps']['000000']
    assert entry['frame_ids'] == [str(s) for s in task['stamps']] and entry['center'] == 2 and entry['timestamp'] == 1.
    np.testing.assert_allclose(entry['T_odom_submap'], task['camera_poses'][2])
    cloud = read_ply(mapper.root / 'active.ply')
    assert len(cloud) > 500 and np.allclose(cloud[:, 2], 2., atol=1e-5)


def test_a_window_that_does_not_move_or_does_not_follow_its_poses_is_not_mapped(mapper):
    task, prediction = window(0, 0.)
    still = dict(task, camera_poses=np.tile(np.eye(4), (5, 1, 1)))
    assert mapper.add(still, prediction)['status'] == 'deferred'
    backwards = dict(task, camera_poses=np.linalg.inv(task['camera_poses']))
    assert mapper.add(backwards, prediction) == dict(status='rejected', reason='no positive metric scale fit')
    bent = task['camera_poses'].copy()
    bent[::2, 1, 3] += .5
    assert 'alignment' in mapper.add(dict(task, camera_poses=bent), prediction)['reason']
    report = mapper.add(*window(0, 0., unit=2.))
    assert report['status'] == 'rejected' and 'scale 2.00' in report['reason']
    assert not mapper.nodes and not list(mapper.root.glob('submaps/*'))


def test_each_image_comes_to_metric_depth_by_its_window_and_by_triangulation_against_an_older_one(mapper):
    wrong = np.array([1.1, 1., 1.05, .95, 1.])
    assert mapper.add(*window(0, 0.))['frame_scale_anchor'] is None             # nothing older to triangulate against
    report = mapper.add(*window(20, 3., depth=wrong))
    relative = np.array(report['frame_scales']['scales'])
    np.testing.assert_allclose(relative * wrong, np.exp(np.mean(np.log(wrong))), rtol=1e-3)     # one scale for the window
    assert report['frame_scale_anchor']['keypoints'] > 100
    assert report['frame_scale_anchor']['log_offset'] == pytest.approx(np.mean(np.log(wrong)), abs=1e-3)
    np.testing.assert_allclose(stored(mapper, '000001')['depth'].mean(axis=(1, 2)), 2., rtol=1e-3)


def test_keypoints_on_depth_the_filter_dropped_do_not_anchor(mapper):
    mapper.add(*window(0, 0.))
    task, prediction = window(20, 3., depth=1.1)
    prediction.depth[2, :, :64] *= 1.4          # half of the centre image is far off: no other image agrees with it
    report = mapper.add(task, prediction)
    assert report['frame_scale_anchor']['log_offset'] == pytest.approx(np.log(1.1), abs=2e-3)
    kept = stored(mapper, '000001')
    assert not kept['valid'][2, :, :60].any() and kept['valid'][2, :, 70:].all()
    np.testing.assert_allclose(kept['depth'][2, :, 70:], 2., rtol=1e-3)


def test_windows_that_agree_with_the_odometry_leave_the_map_on_it(mapper, mapping):
    mapping['max_active_submaps'] = 2
    reports = [mapper.add(*window(6 * k, 1.5 * k)) for k in range(4)]            # each shares two images with the next
    assert [r['status'] for r in reports] == ['mapped'] * 4 and reports[-1]['nodes'] == 4
    assert list(mapper.geometry) == ['000002', '000003']        # the dense arrays are bounded, the graph keeps every window
    assert [(e['a'], e['b'], e['kind']) for e in mapper.edges[:3]] == [
        ('000000', '000001', 'odometry'), ('000000', '000001', 'points'), ('000000', '000001', 'scale')]
    assert sum(e['kind'] == 'odometry' for e in mapper.edges) == 3
    for edge in mapper.edges:
        if edge['kind'] == 'odometry':      # the filter's own relative pose
            np.testing.assert_allclose(edge['measurement'][:3, 3], [.12, 0., 0.], atol=1e-12)
        if edge['kind'] == 'scale':
            assert edge['log_ratio'] == pytest.approx(0., abs=1e-6)
    np.testing.assert_allclose(reports[-1]['T_map_odom'], np.eye(4), atol=1e-6)
    for node in mapper.nodes.values():
        np.testing.assert_allclose(node['pose'], node['odom'], atol=1e-6)
    index = json.loads((mapper.root / 'map_index.json').read_text())
    assert list(index['submaps']) == list(mapper.nodes) and index['last_sensor_time'] == 6.5
    np.testing.assert_allclose(index['submaps']['000003']['T_map_submap'], mapper.nodes['000003']['pose'])


def test_windows_that_share_images_measure_the_ratio_of_their_depths(mapper, mapping):
    for k, wrong in enumerate([1., 1.2, 1., 1.]):       # the second has no older window far enough to triangulate against
        report = mapper.add(*window(6 * k, 1.5 * k, depth=wrong))
    ratios = {(e['a'], e['b']): (e['log_ratio'], e['sigma_log']) for e in mapper.edges if e['kind'] == 'scale'}
    assert ratios['000000', '000001'] == pytest.approx((-np.log(1.2), mapping['scale_sigma_log']), abs=1e-3)
    assert ratios['000001', '000002'] == pytest.approx((np.log(1.2), mapping['scale_sigma_log']), abs=1e-3)
    scales = [sim3.scale(node['pose']) for node in mapper.nodes.values()]
    assert scales[1] < .99 and all(abs(s - 1.) < .01 for s in scales[:1] + scales[2:])      # towards its neighbours
    for node in mapper.nodes.values():
        np.testing.assert_allclose(sim3.pose(node['pose']), node['odom'], atol=.01)
    assert sim3.scale(np.asarray(report['T_map_odom'])) == pytest.approx(1.)     # the live stream gets a pose, never a scale


def test_a_revisit_after_a_short_path_gives_its_depth_scale_and_moves_no_pose(mapper, mapping):
    for k, x in enumerate((0, 20, 40)):
        mapper.add(*window(x, 2. * k))
    report = mapper.add(*window(2, 6.))
    assert report['accepted_loops'] == 0 and len(report['loops_deferred_by_drift']) == 1
    loop = report['loops_deferred_by_drift'][0]
    assert (loop['a'], loop['b'], loop['pose_admitted']) == ('000000', '000003', False)
    assert loop['baseline_m'] == pytest.approx(.04, abs=1e-6) and loop['depth_ratio'] == pytest.approx(1., abs=1e-6)
    # over 1.56 m the odometry may have drifted by its floor; the loop's sigma grows with its length from a larger one
    assert loop['drift_over_arc_m'] == mapping['odometry_sigma_floor_m'] < mapping['loop_sigma_m']
    assert loop['loop_sigma_m'] == pytest.approx(np.hypot(mapping['loop_sigma_m'], mapping['loop_scale_relative_error'] * .04))
    between = [e['kind'] for e in mapper.edges if (e['a'], e['b']) == ('000000', '000003')]
    assert between == ['scale']


def test_a_loop_brings_back_a_window_the_odometry_has_let_drift(mapper, mapping):
    mapping['odometry_drift_m_per_m'] = .2          # an odometry worse than the loop over this path
    for k, x in enumerate((0, 20, 40, 60)):
        mapper.add(*window(x, 2. * k))
    report = mapper.add(*window(2, 8., drift=(0., .06, 0.)))
    assert report['accepted_loops'] == 2 and [loop['a'] for loop in report['loops']] == ['000000', '000001']
    assert report['loops'][0]['innovation_m'] == pytest.approx(.06, abs=1e-6)
    assert report['T_map_odom'][1][3] == pytest.approx(-.06, abs=.01)
    np.testing.assert_allclose(mapper.nodes['000004']['pose'][:3, 3], [.04, 0., 0.], atol=.01)
    np.testing.assert_array_equal(mapper.nodes['000000']['pose'], np.eye(4))


def test_a_match_far_from_where_the_odometry_puts_it_is_no_loop(mapper):
    for k, x in enumerate((0, 20, 40)):
        mapper.add(*window(x, 2. * k))
    report = mapper.add(*window(2, 6., drift=(0., 1., 0.)))
    assert report['accepted_loops'] == 0 and report['loops_deferred_by_drift'] == [] and report['sparse_edges'] == 0


def test_a_window_takes_one_offset_the_robust_mean_of_the_ratios_of_all_its_images_within_a_bound(mapper, mapping):
    rng = np.random.default_rng(0)
    residuals = {i: [(.05 + rng.normal(0, .02, 200), np.ones(200))] for i in range(3)}
    residuals[1][0][0][:20] = 2.                # mismatches
    residuals[4] = [(np.full(5, .2), np.ones(5))]       # few of its own: they count with the window's
    sm = dict(depth=np.ones((6, 4, 4), np.float32))
    applied = mapper.anchor(sm, residuals)
    assert applied['log_offset'] == pytest.approx(.05, abs=.01) and applied['keypoints'] == 605
    np.testing.assert_allclose(sm['depth'], np.exp(-applied['log_offset']), rtol=1e-6)       # every image alike
    np.testing.assert_allclose(sm['frame_anchor_log'], applied['log_offset'])
    far = mapper.anchor(dict(depth=np.ones((6, 4, 4), np.float32)), {3: [(np.full(50, 1.), np.ones(50))]})
    assert far['log_offset'] == mapping['anchor_max_log']
    assert mapper.anchor(sm, {0: residuals[4]}) is None         # no image has enough of them


def test_a_prediction_becomes_cameras_about_the_centre_image_and_colour_on_the_depth_grid():
    _, prediction = window(0, 0.)
    sm = arrays(prediction)
    np.testing.assert_allclose(sm['poses'][:, 0, 3], [-.08, -.04, 0., .04, .08], atol=1e-12)
    assert sm['center'] == 2 and sm['rgb'] is not prediction.processed_images and (sm['rgb'] == prediction.processed_images).all()
    prediction.extrinsics = prediction.extrinsics[:, :3]           # the model may leave out the last row
    np.testing.assert_array_equal(arrays(prediction)['poses'], sm['poses'])
    red = np.zeros((48, 64, 4), np.uint8)       # half the size of the depth, with an alpha channel
    red[..., 0], red[..., 3] = 200, 255
    coloured = arrays(prediction, [red] * 4 + [np.full((48, 64), 90, np.uint8)])
    assert coloured['rgb'].shape == (5, *SHAPE, 3) and coloured['rgb'][0, 0, 0].tolist() == [200, 0, 0]
    assert coloured['rgb'][4, 0, 0].tolist() == [90, 90, 90]
    np.testing.assert_array_equal(coloured['depth'], sm['depth'])
