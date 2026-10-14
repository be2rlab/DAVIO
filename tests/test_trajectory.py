import json

import numpy as np
from scipy.spatial.transform import Rotation

from davio.mapper.trajectory import final_trajectory, write_final_trajectory


def run_with_map(run, corrections):
    """A run whose filter moved 1 m/s along x, and whose map moved the keyframe at time t by
    corrections[t] along y."""
    (run / 'map').mkdir(parents=True)
    submaps = {}
    for i, (t, shift) in enumerate(corrections.items()):
        odom, placed = np.eye(4), np.eye(4)
        odom[0, 3] = placed[0, 3] = t
        placed[1, 3] = shift
        placed[:3, :3] *= 1.1               # the depth scale of a node is no part of the correction
        submaps[f'{i:06d}'] = dict(timestamp=t, T_odom_submap=odom.tolist(), T_map_submap=placed.tolist())
    (run / 'map/map_index.json').write_text(json.dumps(dict(submaps=submaps)))
    (run / 'trajectory.tum').write_text(''.join(f'{t} {t} 0 0 0 0 0 1\n' for t in (0., .5, 1., 1.5, 2., 3.)))
    return run


def test_every_pose_takes_the_correction_of_the_keyframes_around_it(tmp_path):
    poses = dict(final_trajectory(run_with_map(tmp_path, {1.: 0., 2.: .4})))
    # before the first keyframe and after the last, their correction; between two, in proportion
    np.testing.assert_allclose([poses[t][1, 3] for t in (0., .5, 1., 1.5, 2., 3.)], [0., 0., 0., .2, .4, .4], atol=1e-12)
    np.testing.assert_allclose([poses[t][0, 3] for t in (0., 1.5, 3.)], [0., 1.5, 3.])
    np.testing.assert_allclose(poses[1.5][:3, :3], np.eye(3), atol=1e-12)


def test_a_rotation_of_the_map_turns_the_trajectory_about_the_keyframe(tmp_path):
    run = run_with_map(tmp_path, {1.: 0.})
    index = json.loads((run / 'map/map_index.json').read_text())
    turned = np.eye(4)
    turned[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    turned[:3, 3] = [1., 0., 0.]            # the keyframe stays where it was and turns
    index['submaps']['000000']['T_map_submap'] = turned.tolist()
    (run / 'map/map_index.json').write_text(json.dumps(index))
    poses = dict(final_trajectory(run))
    np.testing.assert_allclose(poses[2.][:3, 3], [1., 1., 0.], atol=1e-12)


def test_the_final_trajectory_is_written_beside_the_others(tmp_path):
    path = write_final_trajectory(run_with_map(tmp_path, {1.: .5}))
    assert path == tmp_path / 'map_trajectory_final.tum'
    rows = np.loadtxt(path)
    np.testing.assert_allclose(rows[:, :3], [[t, t, .5] for t in (0., .5, 1., 1.5, 2., 3.)])


def test_a_run_without_a_map_has_no_final_trajectory(tmp_path):
    (tmp_path / 'trajectory.tum').write_text('1.0 0 0 0 0 0 0 1\n')
    assert write_final_trajectory(tmp_path) is None
    (tmp_path / 'map').mkdir()
    (tmp_path / 'map/map_index.json').write_text(json.dumps(dict(submaps={})))
    assert write_final_trajectory(tmp_path) is None and not (tmp_path / 'map_trajectory_final.tum').exists()
