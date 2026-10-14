"""The phone (custom) and TUM-VI sequences, as the engine and the scorers read them."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from davio.datasets import Custom, TumVi, open_dataset
from davio.filter import read_camera

STAMPS = [1_520_530_308_200_000_000 + i * 50_000_000 for i in range(4)]


def imu(path, start):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#timestamp [ns],w,w,w,a,a,a\n' + ''.join(f'{start + i * 5_000_000},0,0,0,0,0,9.81\n' for i in range(60)))


@pytest.fixture
def phone(tmp_path):
    mav = tmp_path / 'custom/walk/mav0'
    (mav / 'cam0/data').mkdir(parents=True)
    orange = np.zeros((6, 8, 3), np.uint8)
    orange[:, :, 2], orange[:, :, 1] = 250, 120          # OpenCV writes blue first
    for stamp in STAMPS:
        cv2.imwrite(str(mav / f'cam0/data/{stamp}.jpg'), orange, [cv2.IMWRITE_JPEG_QUALITY, 100])
    imu(mav / 'imu0/data.csv', STAMPS[0] - 10_000_000)
    return tmp_path / 'custom'


@pytest.fixture
def tumvi(tmp_path):
    mav = tmp_path / 'tumvi/room9/mav0'
    (mav / 'cam0/data').mkdir(parents=True)
    for i, stamp in enumerate(STAMPS):
        cv2.imwrite(str(mav / f'cam0/data/{stamp}.png'), np.full((6, 8), 256 * (40 + i) + 255, np.uint16))
    imu(mav / 'imu0/data.csv', STAMPS[0] - 10_000_000)
    (mav / 'mocap0').mkdir()
    turn = Rotation.from_euler('z', .3).as_quat()        # x y z w
    (mav / 'mocap0/data.csv').write_text('#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z\n' + ''.join(
        f'{s},{.1 * i},2.0,1.25,{turn[3]},{turn[0]},{turn[1]},{turn[2]}\n' for i, s in enumerate(STAMPS)))
    return tmp_path / 'tumvi'


def test_a_phone_frame_is_grey_for_the_filter_and_red_first_for_the_map(phone):
    ds = open_dataset('custom', phone, 'walk')
    assert isinstance(ds, Custom) and ds.image_stamps().tolist() == STAMPS
    grey, colour = ds.load_filter_image(STAMPS[1]), ds.load_color_image(STAMPS[1])
    assert grey.shape == (6, 8) and grey.dtype == np.uint8 and abs(int(grey.mean()) - 145) <= 3
    assert colour.shape == (6, 8, 3) and colour[..., 0].mean() > 240 and colour[..., 2].mean() < 10
    assert len(ds.imu()) == 60


def test_a_phone_session_has_no_reference_and_its_camera_comes_from_its_rig(phone, root):
    ds = Custom(phone, 'walk')
    assert ds.reference_path is None
    with pytest.raises(LookupError):
        ds.groundtruth()
    rig = read_camera(root / 'config/phone/estimator_config.yaml')
    assert ds.camera.resolution == tuple(rig['resolution']) == (640, 480) and ds.camera.model == 'radtan'
    np.testing.assert_allclose(ds.R_CtoI, np.asarray(rig['T_imu_cam'])[:3, :3])


def test_a_tumvi_frame_is_its_sixteen_bits_scaled_to_eight(tumvi):
    ds = open_dataset('tumvi', tumvi, 'room9')
    assert isinstance(ds, TumVi) and ds.image_stamps().tolist() == STAMPS
    image = ds.load_filter_image(STAMPS[2])
    assert image.dtype == np.uint8 and image.shape == (6, 8) and (image == 42).all()
    assert ds.camera.model == 'equidistant' and ds.camera.resolution == (512, 512)


def test_the_tumvi_reference_is_the_motion_capture_of_the_imu_with_the_scalar_first(tumvi):
    reference = TumVi(tumvi, 'room9').groundtruth()
    assert reference.t[1] == pytest.approx(STAMPS[1] * 1e-9) and reference.p[2].tolist() == pytest.approx([.2, 2., 1.25])
    np.testing.assert_allclose(reference.R[0], Rotation.from_euler('z', .3).as_matrix(), atol=1e-9)


def test_every_dataset_has_a_rig_and_the_runner_accepts_it(root, cfg, script, capsys):
    for name in ('euroc', 'ori', 'custom', 'tumvi'):
        assert (root / cfg['rigs'][name]).is_file()
    with pytest.raises(SystemExit):         # refused, but only for its existing --out: tumvi is a dataset it takes
        script('run').main(['--dataset', 'tumvi', '--data', str(root), '--sequence', 'x', '--out', str(root)])
    err = capsys.readouterr().err
    assert 'invalid choice' not in err and '--out must not exist' in err


def test_a_run_without_a_reference_is_described_but_not_scored(phone, script, tmp_path, capsys):
    import json
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'run.json').write_text(json.dumps(dict(
        dataset='custom', sequence='walk', state='completed', rate=1., interval_start=STAMPS[0] * 1e-9,
        interval_end=STAMPS[-1] * 1e-9, windows=dict(mapped=1), loops_closed=0, pose_age_p50_s=.01, pose_age_p95_s=.02)))
    (run / 'trajectory.tum').write_text(''.join(f'{s * 1e-9:.9f} 0 0 0 0 0 0 1\n' for s in STAMPS))
    (run / 'events.jsonl').write_text('')
    out = script('score_trajectory').evaluate(run, phone)
    assert out['ate'] == {} and out['groundtruth'] is None and out['coverage'] > 0
    assert script('score_map').main(['--run', str(run), '--data', str(phone)]) == 0
    assert 'no reference surface' in capsys.readouterr().out and not (run / 'surface.json').exists()
