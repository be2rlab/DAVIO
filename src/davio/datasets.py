"""EuRoC and ORI sequences: camera frames, IMU, calibration, and the reference trajectory.
The reference is read by the scorers only; nothing online touches it.
"""
from collections import namedtuple
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[2]
ImuSample = namedtuple('ImuSample', 't gyro accel')         # s, rad/s, m/s^2


@dataclass(frozen=True)
class Camera:
    K: np.ndarray
    D: np.ndarray
    model: str              # radtan | equidistant
    resolution: tuple       # width, height


@dataclass
class Reference:
    """Body poses in the reference's world frame."""
    t: np.ndarray
    p: np.ndarray
    R: np.ndarray
    orientation_reliable: bool = True


def load_imu(path):
    """An ASL imu0/data.csv, sorted by time."""
    samples = []
    with open(path, encoding='utf-8') as handle:
        for row in csv.reader(handle):
            if row and not row[0].startswith('#'):
                samples.append(ImuSample(int(row[0]) * 1e-9, np.array([float(x) for x in row[1:4]]),
                                         np.array([float(x) for x in row[4:7]])))
    samples.sort(key=lambda s: s.t)
    return samples


def read_image(path, color=False):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f'could not read image: {path}')
    if color:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image[:, :, 0] if image.ndim == 3 else image


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Sequence:
    extension = '.png'

    def open(self, root, frames, imu):
        self.root, self.frames = Path(root), Path(frames)
        self.stamps = np.array(sorted(int(p.stem) for p in self.frames.glob('*' + self.extension)))
        if self.stamps.size < 2:
            raise FileNotFoundError(f'no frames under {self.frames}')
        self.imu_path, self._imu, self._reference = Path(imu), None, None

    def imu(self):
        if self._imu is None:
            self._imu = load_imu(self.imu_path)
        return self._imu

    def image_stamps(self):
        return self.stamps

    def load_filter_image(self, stamp):
        return read_image(self.frames / f'{int(stamp)}{self.extension}')

    def reference_provenance(self):
        """Which reference file the scores rest on."""
        reference = self.groundtruth()
        return dict(variant=self.reference_variant, path=str(self.reference_path), sha256=sha256(self.reference_path),
                    samples=int(len(reference.t)), orientation_reliable=bool(reference.orientation_reliable))


class Euroc(Sequence):
    """EuRoC MAV: 752x480 at 20 Hz, radial-tangential distortion; Vicon or Leica reference."""
    name = 'euroc'
    # cam0 of mav0/cam0/sensor.yaml: camera to body.
    T_BS = np.array([[0.0148655, -0.99988093, 0.0041403, -0.02164015],
                     [0.99955725, 0.01496721, 0.02571553, -0.06467699],
                     [-0.02577444, 0.00375619, 0.99966073, 0.00981073],
                     [0.0, 0.0, 0.0, 1.0]])
    # The orientation of these sequences' own reference is documented as inaccurate; OpenVINS distributes a corrected file.
    ORIENTATION_SUSPECT = ('V1_01_easy',)

    def __init__(self, root, seq, reference='dataset'):
        root = Path(root) / seq / 'mav0'
        self.open(root, root / 'cam0/data', root / 'imu0/data.csv')
        self.seq = seq
        self.camera = Camera(K=np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]]),
                             D=np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]),
                             model='radtan', resolution=(752, 480))
        self.R_CtoI, self.p_IC = self.T_BS[:3, :3], self.T_BS[:3, 3]
        if reference not in ('dataset', 'openvins'):
            raise ValueError('reference must be dataset or openvins')
        self.reference_variant = reference
        self.reference_path = (ROOT / f'thirdparty/open_vins/ov_data/euroc_mav/{seq}.csv' if reference == 'openvins'
                               else root / 'state_groundtruth_estimate0/data.csv')

    def groundtruth(self):
        if self._reference is None:
            rows = []
            with self.reference_path.open() as handle:
                for row in csv.reader(handle):
                    if row and not row[0].lstrip().startswith('#'):
                        rows.append([float(x) for x in row[:8]])
            table = np.asarray(rows, float)

            def rotation(q):        # Hamilton, w first
                w, x, y, z = q / np.linalg.norm(q)
                return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            R = np.asarray([rotation(q) for q in table[:, 4:8]])
            reliable = not (self.reference_variant == 'dataset' and self.seq in self.ORIENTATION_SUSPECT)
            self._reference = Reference(t=table[:, 0] * 1e-9, p=table[:, 1:4], R=R, orientation_reliable=reliable)
        return self._reference


class Ori(Sequence):
    name = 'ori'
    extension = '.jpg'

    def __init__(self, root, seq):
        self.seq_root, self.seq = Path(root) / seq, seq
        root = self.seq_root / 'davio'
        if not (root / 'conversion.json').is_file():
            raise FileNotFoundError(f'{root} lacks conversion.json; run scripts/convert_ori.py first')
        self.open(root, root / 'cam0/data', root / 'imu0/data.csv')
        cam = yaml.safe_load((ROOT / 'config/ori/kalibr_imucam_chain.yaml').read_text().replace('%YAML:1.0', ''))['cam0']
        fx, fy, cx, cy = cam['intrinsics']
        self.camera = Camera(K=np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]), D=np.asarray(cam['distortion_coeffs'], float),
                             model=cam['distortion_model'], resolution=tuple(cam['resolution']))
        T_ic = np.asarray(cam['T_imu_cam'], float)
        self.R_CtoI, self.p_IC = T_ic[:3, :3], T_ic[:3, 3]
        self.reference_variant = 'camera poses registered to the scan, carried to the body by the calibration'
        self.reference_path = next(self.seq_root.glob('*_gt')) / 'poses_gt.txt'

    def load_color_image(self, stamp):
        return read_image(self.frames / f'{int(stamp)}.jpg', color=True)

    def groundtruth(self):
        if self._reference is None:
            table = np.loadtxt(self.reference_path)         # t x y z qx qy qz qw, camera to world
            R_wc = Rotation.from_quat(table[:, 4:8]).as_matrix()
            R_ci, p_ci = self.R_CtoI.T, -self.R_CtoI.T @ self.p_IC
            self._reference = Reference(t=table[:, 0], p=table[:, 1:4] + np.einsum('nij,j->ni', R_wc, p_ci), R=R_wc @ R_ci)
        return self._reference


def open_dataset(name, root, seq, reference=None):
    """``reference`` (EuRoC only): 'dataset', the sequence's own file, or 'openvins', the corrected
    file OpenVINS distributes, which is the default."""
    if name == 'euroc':
        return Euroc(root, seq, reference or 'openvins')
    if name == 'ori':
        return Ori(root, seq)
    raise ValueError('dataset must be euroc or ori')
