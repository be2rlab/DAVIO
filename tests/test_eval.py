from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from davio.datasets import Reference
from davio.eval import metrics, reference_support, surface_fit
from davio.geometry import quat_to_rot, rot_to_quat, tum_line

K = np.array([[200., 0., 100.], [0., 200., 100.], [0., 0., 1.]])
SHAPE = (200, 200)
MOVE = np.eye(4)
MOVE[:3, :3], MOVE[:3, 3] = Rotation.from_rotvec([.05, -.11, .31]).as_matrix(), [1.3, -.7, .25]


def dataset(positions, rotations=None, reliable=True):
    """A sequence that is nothing but its reference: body poses 50 ms apart, the camera on the body."""
    n = len(positions)
    R = np.tile(np.eye(3), (n, 1, 1)) if rotations is None else rotations
    reference = Reference(t=.05 * np.arange(n), p=np.asarray(positions, float), R=R, orientation_reliable=reliable)
    return SimpleNamespace(seq='synthetic', R_CtoI=np.eye(3), p_IC=np.zeros(3), groundtruth=lambda: reference,
                           reference_provenance=lambda: dict(variant='synthetic', sha256='0' * 64))


def curve(n=60, reliable=True):
    angle = np.linspace(0., 1.2, n)
    return dataset(np.column_stack((np.cos(angle), np.sin(angle), .3 * angle)),
                   Rotation.from_euler('z', angle).as_matrix(), reliable)


def estimate(ds, frame=np.eye(4), scale=1., turn=np.eye(3)):
    """The reference as a trajectory, seen from another world frame, scaled, the body turned."""
    reference = ds.groundtruth()
    return [(t, scale * (frame[:3, :3] @ p + frame[:3, 3]), rot_to_quat((frame[:3, :3] @ R @ turn).T))
            for t, p, R in zip(reference.t, reference.p, reference.R)]


def grid(z, half=.5, step=.02):
    span = np.arange(-half, half + 1e-9, step)
    x, y = np.meshgrid(span, span)
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def room(seed, n):
    """Points on the faces of a 4 x 3 x 2.5 m box."""
    rng = np.random.default_rng(seed)
    size = np.array([4., 3., 2.5])
    points, face = rng.uniform(0., 1., (n, 3)) * size, rng.integers(0, 6, n)
    for axis in range(3):
        points[face == 2 * axis, axis] = 0.
        points[face == 2 * axis + 1, axis] = size[axis]
    return points


def moved(points, scale, angle_deg, shift):
    centre = points.mean(0)
    return (points - centre) @ (scale * Rotation.from_euler('z', angle_deg, degrees=True).as_matrix()).T + centre + shift


def support(ds, scan, **settings):
    return reference_support.build(ds, scan, 'sha', K, SHAPE, **dict(dict(near=.1, far=20., period_s=0., voxel=0.), **settings))


# ---- trajectories ----

def test_a_trajectory_file_is_read_back_as_time_position_and_the_filter_quaternion(tmp_path):
    pose = MOVE.copy()
    (tmp_path / 'trajectory.tum').write_text('# t x y z qx qy qz qw\n\n' + tum_line(1.5, pose) + tum_line(2., np.eye(4)))
    (t, p, q), (_, _, identity) = metrics.read_trajectory(tmp_path / 'trajectory.tum')
    assert t == 1.5 and p.tolist() == pose[:3, 3].tolist() and identity.tolist() == [0., 0., 0., 1.]
    np.testing.assert_allclose(quat_to_rot(q).T, pose[:3, :3], atol=1e-10)
    (tmp_path / 'short.tum').write_text('1.0 0 0 0\n')
    with pytest.raises(ValueError):
        metrics.read_trajectory(tmp_path / 'short.tum')


def test_the_alignment_is_rigid_and_fits_no_scale():
    points = curve().groundtruth().p.T
    rot, trans, rmse = metrics.se3_align(np.linalg.inv(MOVE)[:3, :3] @ points + np.linalg.inv(MOVE)[:3, 3:], points)
    np.testing.assert_allclose(np.c_[rot, trans], MOVE[:3], atol=1e-9)
    assert rmse < 1e-9
    rot, _, rmse = metrics.se3_align(1.25 * points, points)
    assert np.linalg.det(rot) == pytest.approx(1.) and rmse > .05


def test_the_trajectory_error_is_zero_in_any_world_frame_and_sees_a_turned_body():
    ds = curve()
    exact = metrics.ate(ds, estimate(ds, np.linalg.inv(MOVE)), 0.)
    assert exact['position_m'] < 1e-9 and exact['orientation_deg'] < 1e-5 and exact['n_matched'] == 60 and exact['reason'] == ''
    line = dataset(np.column_stack((np.arange(60.), np.zeros(60), np.zeros(60))))
    assert exact['conditioning'] > 1e-3 > metrics.ate(line, estimate(line), 0.)['conditioning']       # a straight reference is flagged
    turned = metrics.ate(ds, estimate(ds, turn=Rotation.from_euler('x', 5, degrees=True).as_matrix()), 0.)
    assert turned['position_m'] < 1e-9 and turned['orientation_deg'] == pytest.approx(5.)
    assert metrics.ate(ds, estimate(ds, scale=1.1), 0.)['position_m'] > .02


def test_poses_before_the_start_or_outside_the_reference_are_not_scored():
    ds = curve()
    late = [(t + 1.02, p, q) for t, p, q in estimate(ds)]       # the last second has no reference
    out = metrics.ate(ds, late, 1.5)
    assert out['n_states'] == 50 and out['n_matched'] == 30
    few = metrics.ate(ds, estimate(ds)[:9], 0.)
    assert few['position_m'] is None and few['n_matched'] == 0 and 'fewer than 10' in few['reason']


def test_a_reference_whose_orientation_is_unreliable_gives_no_orientation_error():
    ds = curve(reliable=False)
    out = metrics.ate(ds, estimate(ds), 0.)
    assert out['position_m'] < 1e-9 and out['orientation_deg'] is None and 'unreliable' in out['reason']


# ---- the part of the scan a map is scored on ----

def test_a_wall_hides_the_surface_behind_it():
    near, far = grid(2.), grid(5.)
    _, crop, visible, record = support(dataset([[0., 0., 0.]]), np.vstack((near, far)))
    assert visible[:len(near)].mean() > .9 and visible[len(near):].mean() < .1 and crop.all()
    assert record['visible_points'] == visible.sum() and record['crop_points'] == len(crop)
    # without widening, the far wall shows through the gaps between the near wall's points
    _, _, undilated, _ = support(dataset([[0., 0., 0.]]), np.vstack((near, far)), dilate=1)
    assert undilated[len(near):].mean() > .3


def test_only_what_is_in_view_and_within_range_of_a_reference_camera_counts():
    scan = np.vstack((grid(3.), grid(3.) + [10., 0., 0.]))
    _, crop, visible, _ = support(dataset([[0., 0., 0.]]), scan, far=4.)
    half = len(scan) // 2
    assert visible[:half].mean() > .9 and not visible[half:].any() and crop[:half].all() and not crop[half:].any()


def test_reference_cameras_are_taken_by_time_and_the_scan_thinned_to_one_point_a_voxel():
    cameras, times = reference_support.camera_track(curve(), .1)
    np.testing.assert_allclose(times, .1 * np.arange(30))
    np.testing.assert_allclose(cameras[1, :3, 3], curve().groundtruth().p[2])
    thinned = reference_support.voxel_downsample(np.array([[0., 0., 0.], [.001, 0., 0.], [1., 1., 1.]]), .01)
    assert thinned.tolist() == [[0., 0., 0.], [1., 1., 1.]]


# ---- a map registered to the scan ----

def test_the_closed_form_fit_recovers_a_similarity_or_the_rigid_part_alone():
    source = room(0, 500)
    target = moved(source, 1.2, 30., np.array([.5, -.2, .1]))
    scale, rotation, translation = surface_fit.umeyama(source, target)
    np.testing.assert_allclose(scale * source @ rotation.T + translation, target, atol=1e-9)
    assert scale == pytest.approx(1.2) and surface_fit.umeyama(source, target, with_scale=False)[0] == 1.


def test_a_map_three_percent_small_and_displaced_is_registered_with_its_scale_or_without():
    scan = room(0, 60_000)
    tree = cKDTree(scan)
    cloud = moved(room(1, 30_000), 1 / 1.03, 1., np.array([.04, -.03, .02]))
    similar, info = surface_fit.fit(cloud, scan, tree=tree)
    assert info['scale'] == pytest.approx(1.03, abs=.004) and info['with_scale']
    assert np.cbrt(np.linalg.det(similar[:3, :3])) == pytest.approx(info['scale'])
    rigid, info_rigid = surface_fit.fit(cloud, scan, with_scale=False, tree=tree)
    assert info_rigid['scale'] == pytest.approx(1., abs=1e-9)
    left_similar = tree.query(cloud @ similar[:3, :3].T + similar[:3, 3])[0]
    left_rigid = tree.query(cloud @ rigid[:3, :3].T + rigid[:3, 3])[0]
    assert np.median(left_similar) < .015 and np.mean(left_rigid) > 2 * np.mean(left_similar)
