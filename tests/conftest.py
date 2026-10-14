"""What several test files share: the configuration, and stand-ins for the filter, the matcher and the data."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / 'config/davio.yaml').read_text())


@pytest.fixture
def root():
    return ROOT


@pytest.fixture
def cfg():
    return dict(copy.deepcopy(CONFIG), seed=0)


@pytest.fixture
def mapping(cfg):
    return cfg['mapping']


class Dots:
    """In place of XFeat: a keypoint is a pixel with blue in it, and its red and green name it."""

    def __init__(self, cfg=None):
        pass

    def extract(self, rgb):
        rows, cols = np.nonzero(rgb[:, :, 2])
        if not len(rows):
            return np.zeros((0, 2), np.float32), None
        names = rgb[rows, cols, 0].astype(np.float32) * 256 + rgb[rows, cols, 1]
        return np.column_stack((cols, rows)).astype(np.float32), names[:, None]

    def match(self, a, b):
        if a['descriptors'] is None or b['descriptors'] is None:
            return np.empty((0, 2), int)
        _, ia, ib = np.intersect1d(a['descriptors'][:, 0], b['descriptors'][:, 0], return_indices=True)
        return np.column_stack((ia, ib))


@pytest.fixture
def matcher():
    return Dots()


class FakeVio:
    """In place of the binding's VioManager: initialized at the third frame, then ``step`` metres
    along x per frame, the camera on the body."""
    step = .1
    silent_after = None         # frames after which no new state comes

    def __init__(self, config_path):
        self.config_path, self.t, self.frames, self.imu = config_path, None, 0, []

    def feed_imu(self, t, gyro, accel):
        self.imu.append(t)

    def feed_camera(self, t, image, camera):
        if self.silent_after is None or self.frames < self.silent_after:
            self.t = t
        self.frames += 1

    def initialized(self):
        return self.frames > 2

    def get_state(self):
        return dict(t=self.t, q_GtoI=[0., 0., 0., 1.], p=[self.frames * self.step, 0., 0.], v=[0.] * 3, bg=[0.] * 3,
                    ba=[0.] * 3, q_ItoC=[0., 0., 0., 1.], p_IinC=[0., 0., 0.])


@pytest.fixture
def openvins(monkeypatch):
    """The binding, faked; the class is returned so that a test can change how it behaves."""
    vio = type('Vio', (FakeVio,), {})
    monkeypatch.setitem(sys.modules, 'openvins_ext', SimpleNamespace(VioManager=vio))
    return vio


class Recording:
    """A sequence in memory: a 20 Hz camera from t = 1 s and a 200 Hz IMU around it."""

    def __init__(self, frames=10, shape=(480, 752)):
        self.stamps = np.arange(frames) * 50_000_000 + 1_000_000_000
        self.samples = [SimpleNamespace(t=(990 + 5 * i) / 1000, gyro=np.zeros(3), accel=np.array([0., 0., 9.81]))
                        for i in range(frames * 10 + 10)]
        self.shape, self.loaded = shape, []

    def imu(self):
        return self.samples

    def image_stamps(self):
        return self.stamps

    def load_filter_image(self, stamp):
        self.loaded.append(int(stamp))
        return np.full(self.shape, int(stamp) // 50_000_000 % 251, np.uint8)


@pytest.fixture
def recording():
    return Recording


@pytest.fixture
def euroc(tmp_path):
    """Writes a small EuRoC sequence whose reference moves as FakeVio does; returns the data root."""
    import cv2

    def write(seq='V1_02_medium', frames=40, step=FakeVio.step):
        mav = tmp_path / 'euroc' / seq / 'mav0'
        for name in ('cam0/data', 'imu0', 'state_groundtruth_estimate0'):
            (mav / name).mkdir(parents=True)
        for i in range(frames):
            cv2.imwrite(str(mav / f'cam0/data/{1_000_000_000 + i * 50_000_000}.png'), np.full((6, 8), i, np.uint8))
        imu = [f'{990_000_000 + i * 5_000_000},0,0,{i},0,0,9.81' for i in range(frames * 10 + 10)]
        (mav / 'imu0/data.csv').write_text('#timestamp [ns],w,w,w,a,a,a\n' + '\n'.join(reversed(imu)) + '\n')
        rows = [f'{1_000_000_000 + i * 50_000_000},{(i + 1) * step},0,0,1,0,0,0,0,0,0,0,0,0,0,0,0' for i in range(frames)]
        (mav / 'state_groundtruth_estimate0/data.csv').write_text('#timestamp,p,p,p,qw,qx,qy,qz\n' + '\n'.join(rows) + '\n')
        return tmp_path / 'euroc'
    return write


@pytest.fixture
def wall():
    """Makes windows of two images side by side in front of a wall 2 m away, each image's depth off
    by its own log-scale error; returns the submaps and where each stands."""
    K = np.array([[250., 0., 252.], [0., 250., 161.], [0., 0., 1.]])

    def make(errors, spacing=.1, shape=(322, 504)):
        submaps, poses = [], []
        for m, pair in enumerate(np.asarray(errors, float).reshape(-1, 2)):
            local = np.tile(np.eye(4), (2, 1, 1))
            local[1, 0, 3] = spacing
            submaps.append(dict(depth=np.stack([np.full(shape, 2. * np.exp(e), np.float32) for e in pair]),
                                conf=np.ones((2, *shape), np.float32), rgb=np.full((2, *shape, 3), 120, np.uint8),
                                intrinsics=np.tile(K, (2, 1, 1)), poses=local))
            poses.append(np.eye(4))
            poses[-1][0, 3] = 2 * spacing * m
        return submaps, poses
    return make


@pytest.fixture
def script(root):
    """A script of scripts/ as a module, by its name."""
    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / f'scripts/{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return load
