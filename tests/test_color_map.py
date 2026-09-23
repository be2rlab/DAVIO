import numpy as np
import pytest

from davio.data.images import to_u8, to_u8_color, build_rectifier
from davio.data.types import CameraModel
from davio_mapper.online import arrays, resample_colors


def camera(width=64, height=48):
    K = np.array([[60., 0., width / 2], [0., 60., height / 2], [0., 0., 1.]])
    return CameraModel(K=K, D=np.zeros(4), model='radtan', resolution=(width, height))


class FakePrediction:
    """What DA3 hands back for a grey window: processed_images is grey, replicated."""

    def __init__(self, n=3, size=16):
        rng = np.random.default_rng(0)
        self.depth = (1. + rng.uniform(0, .5, (n, size, size))).astype(np.float32)
        grey = rng.integers(0, 255, (n, size, size), dtype=np.uint8)
        self.processed_images = np.repeat(grey[..., None], 3, axis=3)
        self.extrinsics = np.tile(np.eye(4), (n, 1, 1))
        self.intrinsics = np.tile(np.array([[30., 0., 8.], [0., 30., 8.], [0., 0., 1.]]), (n, 1, 1))


def test_to_u8_color_keeps_channels_that_to_u8_collapses():
    image = np.dstack([np.full((4, 4), 10, np.uint8), np.full((4, 4), 80, np.uint8),
                       np.full((4, 4), 200, np.uint8)])
    assert to_u8([image])[0].shape == (4, 4)          # the filter's path: one channel
    assert np.array_equal(to_u8_color([image])[0][0, 0], [10, 80, 200])


def test_to_u8_color_promotes_a_mono_frame_rather_than_failing():
    out = to_u8_color([np.full((4, 4), 7, np.uint8)])[0]
    assert out.shape == (4, 4, 3) and (out == 7).all()


def test_the_window_stretch_is_pooled_so_it_cannot_white_balance():
    """Stretching each channel separately would change the colour it is recording."""
    image = np.zeros((8, 8, 3), np.uint16)
    image[..., 0] = 1000          # a strongly red-biased frame
    image[..., 1] = 200
    image[..., 2] = 100
    out = to_u8_color([image])[0].astype(int)
    assert out[..., 0].mean() > out[..., 1].mean() > out[..., 2].mean(), 'bias must survive'


def test_rectifier_color_matches_the_grey_geometry():
    """Colour goes through the same maps, so it lands on the identical grid."""
    rectifier = build_rectifier(camera(), tone='window_stretch')
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    grey = rectifier([frame])[0]
    color = rectifier.color([frame])[0]
    assert color.shape == (*grey.shape, 3)


def test_resample_colors_lands_on_the_depth_grid():
    colors = [np.full((32, 32, 3), 100, np.uint8) for _ in range(3)]
    out = resample_colors(colors, (16, 16))
    assert out.shape == (3, 16, 16, 3) and out.dtype == np.uint8


def test_arrays_without_colors_still_uses_what_da3_was_fed():
    prediction = FakePrediction()
    sm = arrays(prediction)
    assert np.array_equal(sm['rgb'], prediction.processed_images)
    channels = sm['rgb'].astype(int)
    assert (channels[..., 0] == channels[..., 1]).all(), 'a grey window stays grey'


def test_arrays_with_colors_paints_the_submap_in_colour():
    prediction = FakePrediction()
    n, size = len(prediction.depth), prediction.depth.shape[1]
    colors = [np.dstack([np.full((size, size), 10, np.uint8),
                         np.full((size, size), 90, np.uint8),
                         np.full((size, size), 220, np.uint8)]) for _ in range(n)]
    sm = arrays(prediction, colors)
    assert np.array_equal(sm['rgb'][0][0, 0], [10, 90, 220])
    channels = sm['rgb'].astype(int)
    assert not (channels[..., 0] == channels[..., 2]).all(), 'the map must be real colour'


def test_colour_never_moves_the_geometry():
    """Painting a point is not allowed to change where it is."""
    prediction = FakePrediction()
    n, size = len(prediction.depth), prediction.depth.shape[1]
    colors = [np.full((size, size, 3), 200, np.uint8) for _ in range(n)]
    grey_sm = arrays(prediction)
    color_sm = arrays(prediction, colors)
    for key in ('poses', 'depth', 'intrinsics'):
        assert np.array_equal(grey_sm[key], color_sm[key]), key
    assert grey_sm['center'] == color_sm['center']


def test_a_mismatched_colour_size_is_resampled_not_rejected():
    prediction = FakePrediction()
    n, size = len(prediction.depth), prediction.depth.shape[1]
    colors = [np.full((size * 3, size * 3, 3), 77, np.uint8) for _ in range(n)]
    sm = arrays(prediction, colors)
    assert sm['rgb'].shape == (n, size, size, 3)
    assert (sm['rgb'] == 77).all()


def test_an_rgba_colour_frame_loses_only_its_alpha():
    prediction = FakePrediction()
    n, size = len(prediction.depth), prediction.depth.shape[1]
    colors = [np.dstack([np.full((size, size), 5, np.uint8)] * 3
                        + [np.full((size, size), 255, np.uint8)]) for _ in range(n)]
    sm = arrays(prediction, colors)
    assert sm['rgb'].shape[-1] == 3 and (sm['rgb'] == 5).all()


def test_ori_offers_colour_because_its_fisheye_is_a_colour_camera(tmp_path, monkeypatch):
    """ORI maps were grey only because the colour was never plumbed, not for want of it."""
    from davio.data.ori import OriDataset
    assert hasattr(OriDataset, 'load_color_image')
    # The equidistant rectifier must take the colour path too, not only the radtan one.
    from davio.data.images import build_rectifier
    from davio.data.types import CameraModel
    # A genuinely wide fisheye, as ORI's is: f=60 over 128 px reaches ~65 deg, past the 45
    # a 90 deg square needs, so this exercises the square path.
    camera = CameraModel(K=np.array([[60., 0., 128.], [0., 60., 128.], [0., 0., 1.]]),
                         D=np.zeros(4), model='equidistant', resolution=(256, 256))
    rectifier = build_rectifier(camera, fov_deg=90., out_size=128, tone='window_stretch')
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    grey = rectifier([frame])[0]
    color = rectifier.color([frame])[0]
    assert grey.shape == (128, 128) and color.shape == (128, 128, 3)


def test_every_colour_dataset_exposes_the_same_hook():
    """replay.py looks up load_color_image BY NAME, so a typo silently falls back to grey."""
    from davio.data.ori import OriDataset
    from davio.data.vcu_rvi import VcuRviDataset
    from davio.data.euroc import EurocDataset
    from davio.data.realsense import RealSenseDataset
    for cls in (OriDataset, VcuRviDataset):
        assert callable(getattr(cls, 'load_color_image', None)), cls.__name__
    # EuRoC is monochrome, and a recorded RealSense session is grey on disk because the
    # capture path converts colour frames before writing them. Neither may pretend
    # otherwise: an empty hook would cost a colour rectification per frame for nothing.
    for cls in (EurocDataset, RealSenseDataset):
        assert not hasattr(cls, 'load_color_image'), cls.__name__


# ------------------------------------------------------------------- channel order
def test_read_color_u8_returns_rgb_not_opencv_bgr(tmp_path):
    import cv2
    from davio.data.images import read_color_u8
    wanted = np.zeros((3, 3, 3), np.uint8)
    wanted[:, :, 0] = 200        # a pure-red frame, in the FILE
    wanted[:, :, 2] = 30
    cv2.imwrite(str(tmp_path / 'f.png'), cv2.cvtColor(wanted, cv2.COLOR_RGB2BGR))
    out = read_color_u8(tmp_path / 'f.png')
    assert list(out[0, 0]) == [200, 0, 30], 'channel 0 must be RED'


def test_a_red_frame_stays_red_all_the_way_into_the_submap(tmp_path):
    """End to end: file -> read -> rectify -> resample -> submap rgb, red-first throughout."""
    import cv2
    from davio.data.images import read_color_u8, build_rectifier

    frame = np.zeros((48, 64, 3), np.uint8)
    frame[:, :, 0] = 220         # red in the file
    cv2.imwrite(str(tmp_path / 'f.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    loaded = read_color_u8(tmp_path / 'f.png')
    rectified = build_rectifier(camera(), tone='window_stretch').color([loaded])[0]
    prediction = FakePrediction(n=1, size=16)
    sm = arrays(prediction, [rectified])
    centre = sm['rgb'][0][8, 8].astype(int)
    assert centre[0] > 150 and centre[2] < 80, f'red must survive, got {centre}'


def test_the_ply_export_column_order_is_the_reason_this_matters():
    """write_ply's first colour column is `red`; that fixes the archive's convention."""
    import inspect
    from davio_mapper.mapping import write_ply
    source = inspect.getsource(write_ply)
    assert '("red", "green", "blue")' in source


# ------------------------------------------------------- narrow Kannala-Brandt cameras
def test_a_narrow_kb4_camera_is_undistorted_to_a_same_size_pinhole():
    from davio.data.images import build_rectifier
    from davio.data.types import CameraModel
    camera = CameraModel(K=np.array([[436.08, 0., 319.37], [0., 437.99, 239.34], [0., 0., 1.]]),
                         D=np.array([0.5833, -0.9065, 1.7077, -0.7382]), model='equidistant',
                         resolution=(640, 480))
    rectifier = build_rectifier(camera, fov_deg=90., out_size=512, tone='window_stretch')
    assert rectifier.output_resolution == (640, 480), 'same size, not a 512 square'
    frame = np.full((480, 640, 3), 128, np.uint8)
    out = rectifier.color([frame])[0]
    assert (out.max(axis=2) == 0).mean() < 0.01, 'balance 0: no invalid border survives'


def test_a_wide_fisheye_keeps_the_square_view_it_always_had():
    """ORI reaches ~88 deg on every side; its rectifier must not change."""
    from davio.data.images import build_rectifier, _half_fov_deg
    from davio.data.types import CameraModel
    K = np.array([[464., 0., 735.], [0., 463., 720.], [0., 0., 1.]])
    D = np.array([0.0315, -0.0117, -0.0022, 0.0002])
    assert min(_half_fov_deg(K, D, (1472, 1440))) > 45
    camera = CameraModel(K=K, D=D, model='equidistant', resolution=(1472, 1440))
    assert build_rectifier(camera, fov_deg=90., out_size=512).output_resolution == (512, 512)
