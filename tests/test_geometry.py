import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from davio.geometry import body_pose, camera_pose, quat_to_rot, rot_to_quat, tum_line


@pytest.mark.parametrize('rotvec', [[.3, -.2, .5], [3., .2, .1], [.1, 3., .2], [.2, .1, 3.], [0., 0., 0.]])
def test_a_rotation_survives_its_quaternion(rotvec):
    R = Rotation.from_rotvec(rotvec).as_matrix()
    q = rot_to_quat(R)
    np.testing.assert_allclose(quat_to_rot(q), R, atol=1e-12)
    assert np.linalg.norm(q) == pytest.approx(1.) and q[3] >= 0


def test_the_quaternion_is_jpl_the_transpose_of_the_hamilton_one():
    q = Rotation.from_rotvec([.3, -.2, .5]).as_quat()
    np.testing.assert_allclose(quat_to_rot(q), Rotation.from_quat(q).as_matrix().T, atol=1e-12)


def test_the_body_pose_is_the_inverse_of_the_filter_rotation_and_the_camera_adds_its_mounting():
    R = Rotation.from_rotvec([.3, -.2, .5]).as_matrix()            # body to world
    pose = body_pose(dict(q_GtoI=rot_to_quat(R.T), p=[1., 2., 3.]))
    np.testing.assert_allclose(pose[:3, :3], R, atol=1e-12)
    np.testing.assert_allclose(pose[:, 3], [1., 2., 3., 1.])
    state = dict(q_GtoI=[0, 0, 0, 1], p=[1, 2, 3], q_ItoC=[0, 0, 0, 1], p_IinC=[-.2, 0, 0])
    np.testing.assert_allclose(camera_pose(state)[:3, 3], [1.2, 2, 3])
    mount = Rotation.from_euler('y', 90, degrees=True).as_matrix()          # camera to body
    state.update(q_ItoC=rot_to_quat(mount.T), p_IinC=-mount.T @ [.1, 0., 0.])
    np.testing.assert_allclose(camera_pose(state)[:3, :3], mount, atol=1e-12)
    np.testing.assert_allclose(camera_pose(state)[:3, 3], [1.1, 2, 3], atol=1e-12)


def test_a_trajectory_line_is_time_position_and_a_hamilton_quaternion():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([.3, -.2, .5]).as_matrix()
    pose[:3, 3] = [1., -2., .5]
    line = tum_line(1.5, pose)
    assert line.startswith('1.500000000 ') and line.endswith('\n')
    row = np.array(line.split(), float)
    np.testing.assert_allclose(row[1:4], [1., -2., .5])
    np.testing.assert_allclose(Rotation.from_quat(row[4:]).as_matrix(), pose[:3, :3], atol=1e-10)
