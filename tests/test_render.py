import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.path.insert(0, str(ROOT / 'src'))
    spec = importlib.util.spec_from_file_location(name, ROOT / f'scripts/{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def render_map():
    return load('render_map')


@pytest.fixture(scope='module')
def render_build():
    load('render_map')
    return load('render_build')


def test_fitted_camera_holds_every_point_inside_the_frame(render_map):
    """The whole cloud projects inside the image at the distance the fit returns."""
    points = np.random.default_rng(0).normal(size=(4000, 3)) * (8., 2., 1.)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.
    direction = render_map.orbit_eye(np.zeros(3), 1., 35., 20.)
    up, fov, aspect = np.array([0., 0., 1.]), 55., 16 / 9
    distance = render_map.fit_distance(points, centre, direction, up, fov, aspect,
                                       margin=1.0, quantile=1.0)
    world_to_camera = render_map.extrinsic(centre, centre + direction * distance, up)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    tan_v = math.tan(math.radians(fov) / 2.)
    assert (camera[:, 2] > 0).all()
    assert (np.abs(camera[:, 1]) <= camera[:, 2] * tan_v + 1e-9).all()
    assert (np.abs(camera[:, 0]) <= camera[:, 2] * tan_v * aspect + 1e-9).all()


def test_a_closer_fit_would_cut_the_cloud_off(render_map):
    """The fit is tight, not merely sufficient: 10% nearer and something leaves the frame."""
    points = np.random.default_rng(1).normal(size=(2000, 3)) * (6., 3., 1.)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.
    direction = render_map.orbit_eye(np.zeros(3), 1., 40., 0.)
    up, fov, aspect = np.array([0., 0., 1.]), 55., 16 / 9
    distance = render_map.fit_distance(points, centre, direction, up, fov, aspect,
                                       margin=1.0, quantile=1.0)
    world_to_camera = render_map.extrinsic(centre, centre + direction * distance * .9, up)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    tan_v = math.tan(math.radians(fov) / 2.)
    outside = ((np.abs(camera[:, 1]) > camera[:, 2] * tan_v)
               | (np.abs(camera[:, 0]) > camera[:, 2] * tan_v * aspect))
    assert outside.any()


def test_overhead_framing_lays_the_long_axis_across_the_image(render_map):
    points = np.random.default_rng(2).normal(size=(3000, 3)) * (10., 1., .5)
    rotation = math.radians(37.)
    turn = np.array([[math.cos(rotation), -math.sin(rotation), 0],
                     [math.sin(rotation), math.cos(rotation), 0], [0, 0, 1.]])
    # An axis has no sign, so the answer is only defined modulo 180 degrees.
    assert render_map.principal_azimuth(points @ turn.T) % 180. == pytest.approx(37., abs=1.)


def test_cropping_keeps_the_content_and_drops_the_border(render_map):
    image = np.full((100, 200, 3), 255, np.uint8)
    image[40:60, 80:120] = (10, 20, 30)
    cropped = render_map.crop_background(image, (1., 1., 1.), margin_px=5)
    assert cropped.shape[:2] == (30, 50)


def test_an_empty_frame_does_not_widen_the_crop(render_map):
    """A build animation opens on nothing; that must not veto the crop for every frame."""
    blank = np.full((100, 200, 3), 255, np.uint8)
    image = blank.copy()
    image[40:60, 80:120] = (10, 20, 30)
    assert render_map.content_box(blank, (1., 1., 1.)) is None
    box = render_map.union(None, render_map.content_box(blank, (1., 1., 1.)))
    box = render_map.union(box, render_map.content_box(image, (1., 1., 1.), margin_px=5))
    assert box == (35, 65, 75, 125)


def test_trajectory_is_resampled_to_the_tube_it_will_be_drawn_as(render_map):
    positions = np.stack([np.linspace(0, 10, 1000), np.zeros(1000), np.zeros(1000)], 1)
    index = render_map.resample(positions, .5)
    kept = positions[index]
    assert index[0] == 0 and index[-1] == len(positions) - 1
    assert (np.linalg.norm(np.diff(kept, axis=0), axis=1) >= .49).all()


def test_smoothing_holds_the_ends_where_the_walk_starts_and_stops(render_build):
    """Padding with the end value, not zeros: the camera must not lunge at either end."""
    values = np.stack([np.linspace(0, 1, 50)] * 3, 1)
    step = 1 / 49.
    smoothed = render_build.smooth(values, window=9)
    assert smoothed.shape == values.shape
    assert np.abs(smoothed[0] - values[0]).max() < 2 * step
    assert np.abs(smoothed[-1] - values[-1]).max() < 2 * step
    # And it is a smoothing: the jitter of a noisy walk comes down.
    noisy = values + np.random.default_rng(3).normal(scale=.05, size=values.shape)
    jitter = lambda x: np.abs(np.diff(x, 2, axis=0)).mean()
    assert jitter(render_build.smooth(noisy, 9)) < jitter(noisy) / 2


def test_chase_camera_sits_behind_and_above_the_walk_looking_ahead(render_build):
    times = np.linspace(0, 10, 101)
    positions = np.stack([times * 1., np.zeros_like(times), np.zeros_like(times)], 1)
    eye, target = render_build.chase_camera(times, positions, 5., back_m=2., height_m=1.,
                                            lead_s=1.)
    assert eye[0] == pytest.approx(3., abs=1e-6)        # 2 m behind x = 5
    assert eye[2] == pytest.approx(1., abs=1e-6)
    assert target[0] > eye[0]


def test_looking_down_lowers_the_aim_and_nothing_else(render_build):
    times = np.linspace(0, 10, 101)
    positions = np.stack([times * 1., np.zeros_like(times), np.zeros_like(times)], 1)
    level = render_build.chase_camera(times, positions, 5., 2., 1., 1.)
    down = render_build.chase_camera(times, positions, 5., 2., 1., 1., look_down_m=1.2)
    np.testing.assert_allclose(down[0], level[0])                  # same eye
    np.testing.assert_allclose(down[1], level[1] - [0., 0., 1.2])  # aim 1.2 m lower


def test_a_stationary_moment_does_not_spin_the_chase_camera(render_build):
    times = np.linspace(0, 10, 101)
    positions = np.zeros((101, 3))
    positions[:, 0] = np.where(times < 5, times, 5.)     # walks, then stops
    render_build.chase_camera(times, positions, 3., 2., 1., 1.)
    eye, _target = render_build.chase_camera(times, positions, 9., 2., 1., 1.)
    assert eye[0] == pytest.approx(3., abs=1e-6)         # still behind, not beside


def submap_archive(directory, key, frame_ids, timestamp):
    n = len(frame_ids)
    np.savez(directory / f'{key}.npz', depth=np.full((n, 4, 4), 2.),
             rgb=np.zeros((n, 4, 4, 3), np.uint8),
             intrinsics=np.repeat(np.eye(3)[None], n, axis=0),
             poses=np.repeat(np.eye(4)[None], n, axis=0),
             valid=np.ones((n, 4, 4), bool), conf=np.ones((n, 4, 4)))
    return dict(file=f'{key}.npz', frame_ids=frame_ids, center=0, timestamp=timestamp,
                scale=1., T_odom_submap=np.eye(4).tolist(), alignment_rmse_m=0.,
                T_map_submap=np.eye(4).tolist())


def test_a_frame_two_submaps_share_is_drawn_once(render_build, tmp_path):
    """Overlapping submaps must not each contribute the same image to the animation."""
    maps = tmp_path / 'map'
    maps.mkdir()
    index = dict(schema_version=3, submaps={
        '000000': submap_archive(maps, '000000', ['1', '2'], 10.),
        '000001': submap_archive(maps, '000001', ['2', '3'], 11.)})
    (maps / 'map_index.json').write_text(json.dumps(index))

    class Cloud:
        def __init__(self, points):
            self.points = points
            self.colors = None

        def voxel_down_sample(self, _size):
            return self

    o3d = types.SimpleNamespace(
        geometry=types.SimpleNamespace(PointCloud=Cloud),
        utility=types.SimpleNamespace(Vector3dVector=lambda x: np.asarray(x)))
    submaps = render_build.load_submaps(o3d, tmp_path, pixel_step=1, voxel_m=0.)
    assert [t for t, _cloud in submaps] == [10., 11.]
    # Two frames of 16 pixels in the first submap, one in the second: frame '2' went to
    # the submap that holds it most centrally, not to both.
    assert [len(cloud.points) for _t, cloud in submaps] == [32, 16]
