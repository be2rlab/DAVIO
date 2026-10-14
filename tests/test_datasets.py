import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from davio import datasets
from davio.datasets import Euroc, Ori, open_dataset
from davio.filter import read_camera

TURN = Rotation.from_euler('xyz', [.1, -.2, .3])


@pytest.fixture
def ori(tmp_path):
    """A converted ORI sequence of four red frames; returns the data root."""
    seq = tmp_path / 'ori/r99'
    for name in ('davio/cam0/data', 'davio/imu0', 'r99_gt'):
        (seq / name).mkdir(parents=True)
    stamps = [1_700_000_000_000_000_000 + i * 33_333_333 for i in range(4)]
    red = np.zeros((8, 8, 3), np.uint8)
    red[:, :, 2] = 255                      # OpenCV writes blue first
    for stamp in stamps:
        cv2.imwrite(str(seq / f'davio/cam0/data/{stamp}.jpg'), red)
    (seq / 'davio/imu0/data.csv').write_text('#timestamp [ns],w,w,w,a,a,a\n' + ''.join(
        f'{stamps[0] + i * 1_000_000},0.01,0,0,0,0,9.80766\n' for i in range(140)))
    (seq / 'davio/conversion.json').write_text(json.dumps(dict(images=4)))
    q = TURN.as_quat()
    (seq / 'r99_gt/poses_gt.txt').write_text(''.join(
        f'{s * 1e-9:.9f} {1. + .01 * i} 2.0 0.5 {q[0]} {q[1]} {q[2]} {q[3]}\n' for i, s in enumerate(stamps)))
    return tmp_path / 'ori'


def test_a_euroc_sequence_gives_its_frames_its_imu_in_time_order_and_its_camera(euroc):
    ds = Euroc(euroc(frames=5), 'V1_02_medium')
    assert ds.image_stamps().tolist() == [1_000_000_000 + i * 50_000_000 for i in range(5)]
    image = ds.load_filter_image(ds.image_stamps()[3])
    assert image.shape == (6, 8) and image.dtype == np.uint8 and (image == 3).all()
    imu = ds.imu()                          # the file lists the samples backwards
    assert len(imu) == 60 and [s.t for s in imu] == sorted(s.t for s in imu)
    assert imu[2].t == pytest.approx(1.) and imu[2].gyro.tolist() == [0., 0., 2.] and imu[2].accel[2] == 9.81
    assert ds.camera.model == 'radtan' and ds.camera.resolution == (752, 480) and ds.camera.K[0, 0] == 458.654
    np.testing.assert_allclose(ds.R_CtoI @ ds.R_CtoI.T, np.eye(3), atol=1e-6)


def test_the_euroc_reference_is_read_as_body_poses_with_the_scalar_first(euroc):
    root = euroc(frames=5)
    x, y, z, w = TURN.as_quat()
    (root / 'V1_02_medium/mav0/state_groundtruth_estimate0/data.csv').write_text(
        f'#timestamp,p,p,p,qw,qx,qy,qz\n1000000000,1,2,3,{w},{x},{y},{z},0,0,0\n1050000000,1,2,4,{2 * w},{2 * x},{2 * y},{2 * z}\n')
    reference = Euroc(root, 'V1_02_medium').groundtruth()
    np.testing.assert_allclose(reference.t, [1., 1.05])
    np.testing.assert_allclose(reference.p, [[1., 2., 3.], [1., 2., 4.]])
    np.testing.assert_allclose(reference.R, [TURN.as_matrix()] * 2, atol=1e-12)


def test_the_euroc_reference_is_the_corrected_file_unless_the_sequences_own_is_asked_for(euroc, tmp_path, monkeypatch):
    root = euroc(seq='V1_01_easy', frames=5)
    corrected = tmp_path / 'thirdparty/open_vins/ov_data/euroc_mav/V1_01_easy.csv'
    corrected.parent.mkdir(parents=True)
    corrected.write_text('#t,p,p,p,qw,qx,qy,qz\n1000000000,7,0,0,1,0,0,0\n')
    monkeypatch.setattr(datasets, 'ROOT', tmp_path)
    default = open_dataset('euroc', root, 'V1_01_easy')
    assert default.reference_path == corrected and default.groundtruth().p[0, 0] == 7.
    assert default.groundtruth().orientation_reliable
    own = open_dataset('euroc', root, 'V1_01_easy', 'dataset')
    assert own.reference_path == root / 'V1_01_easy/mav0/state_groundtruth_estimate0/data.csv'
    assert not own.groundtruth().orientation_reliable       # this sequence's own orientation is documented as inaccurate
    record = own.reference_provenance()
    assert record['variant'] == 'dataset' and record['samples'] == 5 and record['orientation_reliable'] is False
    assert record['sha256'] == datasets.sha256(own.reference_path) and len(record['sha256']) == 64


def test_an_ori_sequence_takes_its_camera_from_the_rig_file(ori, root):
    ds = open_dataset('ori', ori, 'r99')
    cam = read_camera(root / 'config/ori/estimator_config.yaml')
    assert ds.camera.model == 'equidistant' and ds.camera.resolution == (1472, 1440)
    assert [ds.camera.K[0, 0], ds.camera.K[1, 1], ds.camera.K[0, 2], ds.camera.K[1, 2]] == cam['intrinsics']
    assert ds.camera.D.tolist() == cam['distortion_coeffs']
    np.testing.assert_allclose(np.c_[ds.R_CtoI, ds.p_IC], np.asarray(cam['T_imu_cam'])[:3])
    assert len(ds.image_stamps()) == 4 and len(ds.imu()) == 140 and ds.imu()[0].accel[2] == 9.80766


def test_an_ori_frame_is_grey_for_the_filter_and_red_first_for_the_map(ori):
    ds = Ori(ori, 'r99')
    stamp = ds.image_stamps()[0]
    assert ds.load_filter_image(stamp).shape == (8, 8)
    colour = ds.load_color_image(stamp)
    assert colour.shape == (8, 8, 3) and colour[0, 0, 0] > 200 and colour[0, 0, 2] < 50


def test_the_ori_reference_is_its_camera_poses_carried_to_the_body(ori):
    ds = Ori(ori, 'r99')
    reference = ds.groundtruth()
    body, mount = np.eye(4), np.eye(4)
    body[:3, :3], body[:3, 3] = reference.R[1], reference.p[1]
    mount[:3, :3], mount[:3, 3] = ds.R_CtoI, ds.p_IC
    camera = body @ mount
    np.testing.assert_allclose(camera[:3, 3], [1.01, 2., .5], atol=1e-9)
    np.testing.assert_allclose(camera[:3, :3], TURN.as_matrix(), atol=1e-9)
    assert reference.orientation_reliable and ds.reference_provenance()['samples'] == 4
    assert reference.t[0] == pytest.approx(1_700_000_000.)
