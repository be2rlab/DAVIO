import json
import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_surface import world_alignment      # noqa: E402

from davio.data.types import GroundTruth          # noqa: E402
from davio.runtime.geometry import tum_line       # noqa: E402


def _reference(n=60):
    t = np.arange(n) * .05
    angle = np.linspace(0, 1.2, n)
    p = np.column_stack([np.cos(angle), np.sin(angle), .3 * angle])
    R = Rotation.from_rotvec(np.column_stack([np.zeros(n), np.zeros(n), angle])).as_matrix()
    return GroundTruth(t=t, p=p, R=R, v=np.zeros((n, 3)), valid=np.ones(n, bool))


def _dataset(gt):
    return types.SimpleNamespace(seq='synthetic', R_CtoI=np.eye(3), p_IC=np.zeros(3),
                                 groundtruth=lambda: gt)


def test_recovers_a_known_rigid_placement(tmp_path):
    gt = _reference()
    # The estimate is the reference expressed in another frame: map -> world is `truth`.
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_rotvec([.05, -.11, .31]).as_matrix()
    truth[:3, 3] = [1.3, -.7, .25]
    inverse = np.linalg.inv(truth)

    (tmp_path / 'run.json').write_text(json.dumps(dict(interval_start=float(gt.t[0]))))
    with (tmp_path / 'map_trajectory_final.tum').open('w') as stream:
        for k in range(len(gt.t)):
            body = np.eye(4)
            body[:3, :3], body[:3, 3] = gt.R[k], gt.p[k]
            stream.write(tum_line(float(gt.t[k]), inverse @ body))

    recovered, rmse = world_alignment(tmp_path, _dataset(gt), 'map_trajectory_final.tum')
    np.testing.assert_allclose(recovered, truth, atol=1e-8)
    assert rmse < 1e-9

    # A map expressed in the estimate's frame lands exactly on the reference geometry.
    local = np.array([[.2, .1, .4], [-.5, .9, 0.]])
    world = local @ recovered[:3, :3].T + recovered[:3, 3]
    np.testing.assert_allclose(world, local @ truth[:3, :3].T + truth[:3, 3], atol=1e-9)


def test_alignment_is_rigid_and_never_fits_scale(tmp_path):
    gt = _reference()
    (tmp_path / 'run.json').write_text(json.dumps(dict(interval_start=float(gt.t[0]))))
    with (tmp_path / 'map_trajectory_final.tum').open('w') as stream:
        for k in range(len(gt.t)):
            body = np.eye(4)
            body[:3, :3], body[:3, 3] = gt.R[k], gt.p[k] * 1.25   # a 25% scale error
            stream.write(tum_line(float(gt.t[k]), body))
    recovered, rmse = world_alignment(tmp_path, _dataset(gt), 'map_trajectory_final.tum')
    determinant = np.linalg.det(recovered[:3, :3])
    assert abs(determinant - 1.) < 1e-9, 'placement must stay SE(3), never absorb scale'
    assert rmse > .05, 'a scale error must show up as residual, not be fitted away'
