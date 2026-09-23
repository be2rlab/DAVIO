import csv
import json

import cv2
import numpy as np
import pytest

from davio.data import open_dataset
from davio.data.vcu_rvi import VcuRviDataset


def build(root, n=6, period_ns=33_000_000, first_ns=1_000_000_000, grayscale=False,
          gt_origin=214.031736808, gt_rows=40):
    """A minimal converted sequence, the shape scripts/convert_vcu_rvi.py writes."""
    seq = root / 'hall3'
    davio = seq / 'davio'
    (davio / 'cam0/data').mkdir(parents=True)
    (davio / 'imu0').mkdir(parents=True)
    stamps = [first_ns + i * period_ns for i in range(n)]
    with (davio / 'cam0/data.csv').open('w') as f:
        f.write('#timestamp [ns],filename\n')
        for i, s in enumerate(stamps):
            frame = np.zeros((480, 640, 3), np.uint8)
            frame[:, :, 0] = 10 + i      # a distinct colour per frame, B != G != R
            frame[:, :, 1] = 90 + i
            frame[:, :, 2] = 200 + i
            cv2.imwrite(str(davio / 'cam0/data' / f'{s}.png'),
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if grayscale else frame)
            f.write(f'{s},{s}.png\n')
    with (davio / 'imu0/data.csv').open('w') as f:
        f.write('#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n')
        t = first_ns - 5_000_000
        while t <= stamps[-1] + 5_000_000:
            f.write(f'{t},0.001,0.002,0.003,0.0,0.0,9.81\n')
            t += 5_000_000
    (davio / 'conversion.json').write_text(json.dumps(dict(
        first_image_ns=stamps[0], last_image_ns=stamps[-1], grayscale=grayscale,
        counts=dict(color=n, imu=0, depth=0), topics=dict(color='/cam0/color', imu='/imu'))))
    with (seq / 'hall3_gt.csv').open('w') as f:
        for i in range(gt_rows):
            f.write(f'{gt_origin + i * 0.008333:.9f} {i * 0.01:.6f} 0.0 0.0 0.0 0.0 0.0 1.0\n')
    return seq, stamps


def test_calibration_comes_from_the_release_not_another_rig(tmp_path):
    seq, _s = build(tmp_path)
    ds = VcuRviDataset(seq)
    assert ds.camera.resolution == (640, 480)
    assert ds.camera.K[0, 0] == pytest.approx(459.357)
    assert ds.camera.K[1, 1] == pytest.approx(459.764)
    assert ds.camera.K[0, 2] == pytest.approx(332.695)
    assert np.allclose(ds.camera.D, 0), 'the release ships rectified imagery'
    # T_imu_cam must be a proper rotation or OpenVINS will reject it.
    assert np.allclose(ds.R_CtoI.T @ ds.R_CtoI, np.eye(3), atol=1e-9)
    assert np.linalg.det(ds.R_CtoI) == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.norm(ds.p_IC) < .1


def test_frames_and_imu_load_on_the_stamp_grid(tmp_path):
    seq, stamps = build(tmp_path)
    ds = VcuRviDataset(seq)
    assert list(ds.image_stamps()) == stamps
    assert ds.cam_period == pytest.approx(.033, abs=1e-3)
    assert ds.pick_stamps([stamps[2] + 1000]) == [stamps[2]]
    assert len(ds.imu()) > len(stamps)


def test_the_filter_gets_grey_and_the_map_gets_colour(tmp_path):
    seq, stamps = build(tmp_path)
    ds = VcuRviDataset(seq)
    grey = ds.load_filter_image(stamps[0])
    color = ds.load_color_image(stamps[0])
    assert grey.ndim == 2, 'OpenVINS tracks on one channel'
    assert color.shape == (480, 640, 3)
    assert not (color[..., 0] == color[..., 2]).all(), 'the map must get real colour'


def test_a_grayscale_conversion_reports_no_colour(tmp_path):
    seq, stamps = build(tmp_path, grayscale=True)
    ds = VcuRviDataset(seq)
    assert ds.load_color_image(stamps[0]) is None
    assert ds.load_filter_image(stamps[0]).ndim == 2


def test_backbone_images_keep_the_sensor_resolution(tmp_path):
    seq, stamps = build(tmp_path)
    ds = VcuRviDataset(seq)
    out = ds.load_backbone_images(stamps[:2])
    assert len(out) == 2
    assert out[0].shape[:2] == (480, 640)
    assert ds.rectifier.output_resolution == (640, 480)


def test_the_reference_is_left_on_the_recording_clock(tmp_path):
    seq, stamps = build(tmp_path, gt_origin=0.9)
    ds = VcuRviDataset(seq)
    gt = ds.groundtruth()
    assert gt.t[0] == pytest.approx(0.9), 'the reference keeps its own timestamps'
    assert gt.t[0] < stamps[0] * 1e-9, 'the mocap started before the camera, as recorded'
    provenance = ds.groundtruth_provenance()
    assert 'clock_offset_s' not in provenance, 'no offset is applied, so none is reported'
    assert provenance['sensor_first_time'] == pytest.approx(stamps[0] * 1e-9)


def test_partial_mocap_coverage_is_reported(tmp_path):
    """The volume is one room; a sequence walks out of it and the reference simply stops."""
    seq, _s = build(tmp_path, gt_rows=40)
    covered = VcuRviDataset(seq).groundtruth_provenance()
    assert covered['n_gaps'] == 0 and covered['covered_s'] > 0
    # Now a reference with a hole in the middle.
    rows = (seq / 'hall3_gt.csv').read_text().splitlines()
    kept = rows[:10] + [r.replace(r.split()[0], f'{float(r.split()[0]) + 30:.9f}', 1)
                        for r in rows[10:]]
    (seq / 'hall3_gt.csv').write_text('\n'.join(kept) + '\n')
    gapped = VcuRviDataset(seq).groundtruth_provenance()
    assert gapped['n_gaps'] == 1
    assert gapped['covered_s'] < covered['covered_s'] + 1


def test_only_the_shipped_reference_is_accepted(tmp_path):
    seq, _s = build(tmp_path)
    with pytest.raises(ValueError, match='one reference'):
        VcuRviDataset(seq, groundtruth='openvins')


def test_an_unconverted_sequence_says_what_to_run(tmp_path):
    (tmp_path / 'hall3').mkdir()
    with pytest.raises(FileNotFoundError, match='convert_vcu_rvi'):
        VcuRviDataset(tmp_path / 'hall3')


def test_open_dataset_reaches_the_adapter(tmp_path):
    seq, _s = build(tmp_path)
    ds = open_dataset('vcu_rvi', tmp_path, seq='hall3')
    assert ds.name == 'vcu_rvi' and ds.seq == 'hall3'


# ----------------------------------------------------------- converter unit handling
def converter():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('conv', root / 'scripts/convert_vcu_rvi.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_accelerometer_in_g_is_scaled_to_si():
    conv = converter()
    scale, why = conv.accel_scale(np.full(500, 1.016))
    assert scale == pytest.approx(9.80665)
    assert 'units are g' in why


def test_accelerometer_already_in_si_is_left_alone():
    conv = converter()
    scale, why = conv.accel_scale(np.full(500, 9.81))
    assert scale == 1.0 and 'already' in why


def test_units_that_are_neither_are_refused_rather_than_guessed():
    conv = converter()
    with pytest.raises(SystemExit, match='neither'):
        conv.accel_scale(np.full(500, 3.1))       # the shape of a corrupted stream


def test_the_release_sign_is_negative_and_named():
    """Not a silent default: the constant is what makes the choice auditable."""
    assert converter().ACCEL_SIGN == -1.0


def test_image_layout_comes_from_the_payload_not_the_encoding_name():
    """This release labels colour as 8UC3, which no ROS encoding table contains."""
    conv = converter()

    class Message:
        def __init__(self, encoding, h, w, data):
            self.encoding, self.height, self.width, self.data = encoding, h, w, data

    colour = Message('8UC3', 4, 5, np.arange(4 * 5 * 3, dtype=np.uint8).tobytes())
    assert conv.image_array(colour).shape == (4, 5, 3)
    depth = Message('16UC1', 4, 5, np.arange(4 * 5, dtype=np.uint16).tobytes())
    out = conv.image_array(depth)
    assert out.shape == (4, 5) and out.dtype == np.uint16


def test_an_rgb_encoding_is_swapped_but_8uc3_is_not():
    conv = converter()

    class Message:
        def __init__(self, encoding, h, w, data):
            self.encoding, self.height, self.width, self.data = encoding, h, w, data

    pixels = np.array([[[1, 2, 3]]], np.uint8)
    assert list(conv.image_array(Message('rgb8', 1, 1, pixels.tobytes()))[0, 0]) == [3, 2, 1]
    assert list(conv.image_array(Message('8UC3', 1, 1, pixels.tobytes()))[0, 0]) == [1, 2, 3]


def test_a_payload_that_does_not_fill_the_frame_is_an_error():
    conv = converter()

    class Message:
        encoding, height, width = '8UC3', 4, 5
        data = np.zeros(4 * 5 * 3 - 1, np.uint8).tobytes()

    with pytest.raises(ValueError, match='do not fill'):
        conv.image_array(Message())
