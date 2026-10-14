import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from davio.mapper.window import fit_window


def cameras(centres, turn=(0., 0., 0.)):
    poses = np.tile(np.eye(4), (len(centres), 1, 1))
    poses[:, :3, :3] = Rotation.from_rotvec(turn).as_matrix()
    poses[:, :3, 3] = centres
    return poses


CENTRES = np.array([[0., 0., 0.], [.1, 0., .02], [.2, .05, 0.], [.3, 0., -.03], [.4, .02, 0.]])


def test_the_scale_between_predicted_and_metric_cameras_is_found_in_any_frame():
    turn = Rotation.from_rotvec([.3, -.5, .8]).as_matrix()
    metric = cameras(2.5 * CENTRES @ turn.T + [1., 2., 3.], [.3, -.5, .8])
    scale, rmse = fit_window(cameras(CENTRES), metric)
    assert scale == pytest.approx(2.5) and rmse < 1e-9


def test_the_distance_left_between_the_cameras_is_reported_in_metres():
    metric = cameras(CENTRES)
    metric[::2, 1, 3] += .1
    metric[1::2, 1, 3] -= .1
    _, rmse = fit_window(cameras(CENTRES), metric)
    assert .08 < rmse < .1


def test_cameras_that_run_the_other_way_give_no_scale():
    assert fit_window(cameras(CENTRES), cameras(-CENTRES)) is None


def test_predicted_cameras_in_one_place_give_no_scale():
    assert fit_window(cameras(np.zeros((5, 3))), cameras(CENTRES)) is None
