"""ORI adapter: converted layout, release calibration, camera->body ground truth, rectifier."""
import json
import numpy as np
import pytest
from pathlib import Path
from scipy.spatial.transform import Rotation


def _synthetic(root):
    import cv2
    seq = root / 'r99'
    (seq / 'davio/cam0/data').mkdir(parents=True)
    (seq / 'davio/imu0').mkdir()
    (seq / 'r99_gt').mkdir()
    rng = np.random.default_rng(0)
    stamps = [1_700_000_000_000_000_000 + i * 33_333_333 for i in range(4)]
    for s in stamps:
        cv2.imwrite(str(seq / 'davio/cam0/data' / f'{s}.jpg'), rng.integers(0, 255, (1440, 1472, 3), np.uint8))
    rows = ['#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z']
    for i in range(140):
        rows.append(f'{stamps[0] - 10_000_000 + i * 1_000_000},0.01,0.0,0.0,0.0,0.0,9.80766')
    (seq / 'davio/imu0/data.csv').write_text('\n'.join(rows) + '\n')
    (seq / 'davio/conversion.json').write_text(json.dumps(dict(images=4)))
    q = Rotation.from_euler('xyz', [.1, -.2, .3]).as_quat()
    (seq / 'r99_gt/poses_gt.txt').write_text('\n'.join(
        f'{s * 1e-9:.9f} {1.0 + i * .01} 2.0 0.5 {q[0]} {q[1]} {q[2]} {q[3]}' for i, s in enumerate(stamps)) + '\n')
    return seq


def test_ori_adapter_reads_converted_layout_and_release_calibration(tmp_path):
    from davio.data.ori import OriDataset
    seq = _synthetic(tmp_path)
    ds = OriDataset(seq, seq='r99')
    assert ds.camera.model == 'equidistant' and ds.camera.resolution == (1472, 1440)
    assert ds.camera.K[0, 0] == pytest.approx(463.9994465216521)
    # Release extrinsics: cam0 sits ~2-3 cm from the IMU, rotation is proper.
    assert 0.02 < np.linalg.norm(ds.p_IC) < 0.05 and np.isclose(np.linalg.det(ds.R_CtoI), 1.)
    assert len(ds.image_stamps()) == 4 and ds.cam_period == pytest.approx(1 / 30., rel=1e-3)
    assert len(ds.imu()) == 140 and ds.imu()[0].accel[2] == pytest.approx(9.80766)
    assert ds.load_filter_image(ds.image_stamps()[0]).shape == (1440, 1472)
    out = ds.load_backbone_images(ds.image_stamps()[:2])
    assert len(out) == 2 and out[0].shape[:2] == (512, 512) and ds.rectifier.output_resolution == (512, 512)
    gt = ds.groundtruth()
    # Body pose = camera pose composed with inv(T_imu_cam): re-deriving the camera pose
    # from the body pose must give the file back.
    T_ic = np.eye(4); T_ic[:3, :3], T_ic[:3, 3] = ds.R_CtoI, ds.p_IC
    T_wi = np.eye(4); T_wi[:3, :3], T_wi[:3, 3] = gt.R[0], gt.p[0]
    T_wc = T_wi @ T_ic
    np.testing.assert_allclose(T_wc[:3, 3], [1.0, 2.0, 0.5], atol=1e-9)
    np.testing.assert_allclose(T_wc[:3, :3], Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix(), atol=1e-9)
    assert gt.orientation_reliable and ds.groundtruth_provenance()['samples'] == 4
    with pytest.raises(ValueError):
        OriDataset(seq, seq='r99', groundtruth='openvins')
