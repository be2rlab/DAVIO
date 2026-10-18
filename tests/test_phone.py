import json

import numpy as np
import pytest

from davio.data import open_dataset
from davio.data.phone import PhoneDataset, square_pixels


def session(root, n=8, period_ns=120_000_000, first_ns=413_130_000_000_000,
            resolution=(1280, 720), array=(4000, 3000), intrinsics=(950., 712.5, 640., 360.),
            shuffle_csv=True):
    """A minimal vi-recorder session."""
    import cv2
    seq = root / '20260917_014153'
    (seq / 'mav0/cam0/data').mkdir(parents=True)
    (seq / 'mav0/imu0').mkdir(parents=True)
    stamps = [first_ns + i * period_ns for i in range(n)]
    for s in stamps:
        cv2.imwrite(str(seq / 'mav0/cam0/data' / f'{s}.png'),
                    np.full((resolution[1], resolution[0]), 120, np.uint8))
    order = list(reversed(stamps)) if shuffle_csv else stamps
    with (seq / 'mav0/cam0/data.csv').open('w') as f:
        f.write('#timestamp [ns],filename\n')
        for s in order:
            f.write(f'{s},{s}.png\n')
    with (seq / 'mav0/imu0/data.csv').open('w') as f:
        f.write('#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n')
        t = first_ns - 5_000_000
        while t <= stamps[-1] + 5_000_000:
            f.write(f'{t},0.001,0.002,0.003,9.7,0.2,1.4\n')
            t += 5_000_000
    (seq / 'mav0/cam0/sensor.yaml').write_text(
        'sensor_type: camera\n'
        'T_BS:\n  cols: 4\n  rows: 4\n  data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,'
        ' 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]\n'
        f'resolution: [{resolution[0]}, {resolution[1]}]\n'
        'camera_model: pinhole\n'
        f'intrinsics: [{intrinsics[0]}, {intrinsics[1]}, {intrinsics[2]}, {intrinsics[3]}]\n'
        'distortion_model: radial-tangential\n'
        'distortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n')
    (seq / 'mav0/imu0/sensor.yaml').write_text(
        'sensor_type: imu\nrate_hz: 200\ngyroscope_noise_density: 1.6968e-04\n'
        'gyroscope_random_walk: 1.9393e-05\naccelerometer_noise_density: 2.0e-3\n'
        'accelerometer_random_walk: 3.0e-3\n')
    (seq / 'metadata.json').write_text(json.dumps(dict(
        camera=dict(active_array_width=array[0], active_array_height=array[1],
                    configured_width=resolution[0], configured_height=resolution[1],
                    requested_fps=30, rolling_shutter_skew_ns=27_335_699),
        session=dict(frames_written=n, frames_dropped=3 * n))))
    return seq, stamps


# ------------------------------------------------------------------ the focal repair
def test_fy_is_repaired_when_the_output_crops_the_short_axis():
    """1280x720 out of a 4:3 array is a CROP, so the pixels stay square and fy == fx."""
    fx, fy, cx, cy = square_pixels(
        (950., 712.5, 640., 360.),
        dict(camera=dict(active_array_width=4000, active_array_height=3000,
                         configured_width=1280, configured_height=720)))
    assert (fx, fy) == (950., 950.)
    assert (cx, cy) == (640., 360.)


def test_a_matching_aspect_is_left_alone():
    """No crop, so the reported pair is already consistent and nothing is assumed."""
    out = square_pixels((950., 712.5, 640., 480.),
                        dict(camera=dict(active_array_width=4000, active_array_height=3000,
                                         configured_width=1280, configured_height=960)))
    assert out[1] == 712.5, 'a 4:3 output must not be second-guessed'


def test_missing_metadata_changes_nothing():
    assert square_pixels((950., 712.5, 640., 360.), {})[1] == 712.5
    assert square_pixels((950., 712.5, 640., 360.), None)[1] == 712.5


def test_the_repair_reaches_the_camera_model(tmp_path):
    seq, _s = session(tmp_path)
    ds = PhoneDataset(seq)
    assert ds.camera.K[0, 0] == pytest.approx(950.)
    assert ds.camera.K[1, 1] == pytest.approx(950.), 'fy must be repaired, not 712.5'


# ------------------------------------------------------------------ frame ordering
def test_frames_are_ordered_by_filename_not_by_data_csv(tmp_path):
    seq, stamps = session(tmp_path, shuffle_csv=True)
    ds = PhoneDataset(seq)
    assert list(ds.image_stamps()) == sorted(stamps)
    assert (np.diff(ds.image_stamps()) > 0).all()


# ------------------------------------------------------------------ health reporting
def test_health_reports_what_will_hurt_before_a_run_does(tmp_path):
    seq, _s = session(tmp_path)
    health = PhoneDataset(seq).health()
    assert health['kept_fraction'] == pytest.approx(0.25)
    assert health['effective_hz'] == pytest.approx(1 / .12, rel=.01)
    assert health['requested_hz'] == 30
    assert health['rolling_shutter_skew_ms'] == pytest.approx(27.34, abs=.01)
    assert health['extrinsic_is_identity'] is True
    assert health['intrinsics_reported'][1] == 712.5
    assert health['intrinsics_used'][1] == 950.


def test_the_extrinsic_prior_follows_the_capture_orientation():
    """Landscape turns the image 90 deg about the optical axis; the IMU frame does not turn."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('rig', root / 'scripts/phone_rig.py')
    rig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rig)
    landscape, why = rig.orientation_prior((1280, 720))
    assert 'landscape' in why
    assert np.allclose(landscape, [[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    portrait, why = rig.orientation_prior((720, 1280))
    assert 'portrait' in why and np.allclose(portrait, np.diag([1., -1, -1]))
    for R in (np.asarray(landscape), np.asarray(portrait)):
        assert np.linalg.det(R) == pytest.approx(1.0), 'must be a proper rotation'


# ------------------------------------------------------------------ no reference
def test_there_is_no_reference_to_score_against(tmp_path):
    seq, _s = session(tmp_path)
    with pytest.raises(FileNotFoundError, match='no reference'):
        PhoneDataset(seq).groundtruth()
    assert PhoneDataset(seq).groundtruth_provenance()['variant'] == 'none'


def test_a_directory_that_is_not_a_session_says_so(tmp_path):
    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError, match='not an ASL session'):
        PhoneDataset(tmp_path / 'empty')


def test_open_dataset_reaches_the_adapter(tmp_path):
    seq, _s = session(tmp_path)
    ds = open_dataset('phone', tmp_path, seq=seq.name)
    assert ds.name == 'phone' and ds.seq == seq.name


# ------------------------------------------------------------------ Kalibr sessions
def kalibr_session(root):
    """A session that ships calibration/*-camchain-imucam.yaml, as the Redmi one does."""
    seq, stamps = session(root, resolution=(1080, 720), intrinsics=(721.18, 638.49, 540., 360.))
    (seq / 'calibration').mkdir()
    (seq / 'calibration/seq1-camchain-imucam.yaml').write_text(
        'cam0:\n'
        '  T_cam_imu:\n'
        '  - [0.0013056718943138884, -0.9999961477594319, 0.0024494258915203834, 0.06739012093128824]\n'
        '  - [-0.9999974189969871, -0.0013011113477710762, 0.001862554327293603, -0.040504720947204115]\n'
        '  - [-0.001859360176463284, -0.002451851454381561, -0.9999952655908811, 0.16622843709138746]\n'
        '  - [0.0, 0.0, 0.0, 1.0]\n'
        '  cam_overlaps: []\n  camera_model: pinhole\n'
        '  distortion_coeffs: [0.15, -0.341, -0.00043, 0.0004]\n'
        '  distortion_model: radtan\n'
        '  intrinsics: [731.078, 735.195, 540.163, 356.181]\n'
        '  resolution: [1080, 720]\n  rostopic: /cam0/image_raw\n'
        '  timeshift_cam_imu: -0.004348572073192924\n')
    return seq, stamps


def test_a_kalibr_calibration_wins_over_the_recorders_guess(tmp_path):
    from davio.data.phone import read_calibration
    seq, _s = kalibr_session(tmp_path)
    cal = read_calibration(seq)
    assert cal['source'] == 'kalibr'
    assert cal['used'] == pytest.approx([731.078, 735.195, 540.163, 356.181])
    assert cal['reported'][1] == pytest.approx(638.49), 'the recorder value is kept for the report'
    assert list(cal['camera'].D) == pytest.approx([0.15, -0.341, -0.00043, 0.0004])
    assert cal['timeshift_cam_imu'] == pytest.approx(-0.0043486, abs=1e-6)


def test_kalibr_T_cam_imu_is_inverted_to_T_imu_cam(tmp_path):
    from davio.data.phone import read_calibration
    seq, _s = kalibr_session(tmp_path)
    T = read_calibration(seq)['T_imu_cam']
    assert T[:3, 3] == pytest.approx([-0.04028353, 0.06774473, 0.16613803], abs=1e-6)
    assert T[0, :3] == pytest.approx([0.00130567, -0.99999742, -0.00185936], abs=1e-6)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-9)


def test_the_adapter_uses_the_kalibr_extrinsic(tmp_path):
    seq, _s = kalibr_session(tmp_path)
    ds = PhoneDataset(seq)
    health = ds.health()
    assert health['calibration_source'] == 'kalibr'
    assert health['extrinsic_is_identity'] is False
    assert health['lever_arm_m'] == pytest.approx(0.1839, abs=1e-3)


def test_a_session_without_kalibr_falls_back_to_the_recorder(tmp_path):
    from davio.data.phone import read_calibration
    seq, _s = session(tmp_path)
    cal = read_calibration(seq)
    assert cal['source'] == 'recorder' and cal['chain'] is None
    assert cal['timeshift_cam_imu'] == 0.0


def test_an_equidistant_kalibr_model_is_carried_through(tmp_path):
    """OpenVINS and the rectifier both read equidistant, so it is accepted, not refused."""
    from davio.data.phone import read_calibration
    seq, _s = kalibr_session(tmp_path)
    chain = seq / 'calibration/seq1-camchain-imucam.yaml'
    chain.write_text(chain.read_text().replace('distortion_model: radtan', 'distortion_model: equidistant'))
    assert read_calibration(seq)['camera'].model == 'equidistant'


def test_an_unsupported_kalibr_model_is_refused(tmp_path):
    from davio.data.phone import read_calibration
    seq, _s = kalibr_session(tmp_path)
    chain = seq / 'calibration/seq1-camchain-imucam.yaml'
    chain.write_text(chain.read_text().replace('distortion_model: radtan', 'distortion_model: fov'))
    with pytest.raises(ValueError, match='radtan or equidistant'):
        read_calibration(seq)


def test_grey_frames_offer_no_colour_and_colour_frames_do(tmp_path):
    import cv2
    seq, stamps = session(tmp_path)
    assert PhoneDataset(seq).load_color_image(stamps[0]) is None, 'grey session, no colour'
    for s in stamps:
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[..., 0], frame[..., 2] = 30, 200
        cv2.imwrite(str(seq / 'mav0/cam0/data' / f'{s}.png'), frame)
    colour = PhoneDataset(seq).load_color_image(stamps[0])
    assert colour is not None and colour.shape[2] == 3


def test_jpeg_sessions_are_read_as_well_as_png(tmp_path):
    import cv2
    seq, stamps = session(tmp_path)
    for s in stamps:
        (seq / 'mav0/cam0/data' / f'{s}.png').unlink()
        cv2.imwrite(str(seq / 'mav0/cam0/data' / f'{s}.jpg'), np.full((720, 1280), 90, np.uint8))
    ds = PhoneDataset(seq)
    assert ds.suffix == '.jpg' and len(ds.image_stamps()) == len(stamps)
    assert ds.load_filter_image(stamps[0]).shape == (720, 1280)


# ------------------------------------------------------------ fy repair without metadata
def test_fy_is_repaired_from_the_recorders_own_numbers_when_there_is_no_metadata():
    fx, fy, cx, cy = square_pixels((721.180374, 638.490592, 540., 360.), None, (1080, 720))
    assert fy == pytest.approx(721.180374)


def test_the_metadata_free_repair_agrees_with_the_metadata_one():
    """Both routes read the same sensor aspect; they must give the same answer."""
    with_meta = square_pixels((950., 712.5, 640., 360.),
                              dict(camera=dict(active_array_width=4000, active_array_height=3000,
                                               configured_width=1280, configured_height=720)))
    without = square_pixels((950., 712.5, 640., 360.), None, (1280, 720))
    assert with_meta == pytest.approx(without)


def test_a_consistent_4_3_session_is_still_left_alone_without_metadata():
    assert square_pixels((475., 474.99998, 320., 240.), None, (640, 480))[1] == pytest.approx(474.99998)


def test_the_recorder_calibration_can_be_chosen_over_a_shipped_kalibr_one(tmp_path):
    from davio.data.phone import read_calibration
    seq, _s = kalibr_session(tmp_path)
    auto, forced = read_calibration(seq), read_calibration(seq, 'recorder')
    assert auto['source'] == 'kalibr' and forced['source'] == 'recorder'
    assert forced['timeshift_cam_imu'] == 0.0
    assert forced['used'][1] == pytest.approx(721.18), 'recorder fy, repaired to square pixels'


def test_asking_for_kalibr_where_there_is_none_is_an_error(tmp_path):
    from davio.data.phone import read_calibration
    seq, _s = session(tmp_path)
    with pytest.raises(FileNotFoundError, match='camchain'):
        read_calibration(seq, 'kalibr')
