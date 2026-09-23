import pathlib
import numpy as np
from scipy.spatial.transform import Rotation
from .types import CameraModel, GroundTruth

GRAVITY_MAG = 9.80766


def _load_calibration(repo_root):
    import yaml
    text = (pathlib.Path(repo_root) / 'config/ori/kalibr_imucam_chain.yaml').read_text()
    return yaml.safe_load(text.replace('%YAML:1.0', ''))['cam0']


class OriDataset:
    """ORI Insta360 front fisheye (1472x1440 equidistant, 30 Hz) + 1 kHz IMU."""

    name = 'ori'

    def __init__(self, seq_dir, seq=None, groundtruth='dataset', repo_root=None):
        from .images import build_rectifier
        self.seq_root = pathlib.Path(seq_dir)
        self.root = self.seq_root / 'davio'
        self.seq = seq
        self._gt_choice = groundtruth
        self._repo_root = pathlib.Path(repo_root if repo_root is not None
                                       else pathlib.Path(__file__).resolve().parents[3])
        if groundtruth not in ('dataset', 'auto'):
            raise ValueError('ORI ships one reference (poses_gt.txt); use groundtruth=dataset')
        if not (self.root / 'conversion.json').is_file():
            raise FileNotFoundError(f'{self.root} lacks conversion.json; run scripts/convert_ori.py first')
        cam = _load_calibration(self._repo_root)
        fx, fy, cx, cy = cam['intrinsics']
        self.camera = CameraModel(K=np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]),
                                  D=np.asarray(cam['distortion_coeffs'], float),
                                  model=cam['distortion_model'], resolution=tuple(cam['resolution']))
        T_ic = np.asarray(cam['T_imu_cam'], float)
        self.R_CtoI, self.p_IC = T_ic[:3, :3], T_ic[:3, 3]
        self.gravity_mag = GRAVITY_MAG
        self.rectifier = build_rectifier(self.camera, fov_deg=90., out_size=512, tone='window_stretch')
        self._cam_dir = self.root / 'cam0/data'
        self._stamps = np.array(sorted(int(p.stem) for p in self._cam_dir.glob('*.jpg')))
        if self._stamps.size < 2:
            raise FileNotFoundError(f'no cam0 frames under {self._cam_dir}')
        self.cam_period = float(np.median(np.diff(self._stamps)) * 1e-9)
        self.srvins_config = None
        self._imu = None
        self._gt = None
        self._gt_provenance = None

    def imu(self):
        from ..init.preintegration import load_asl_imu
        if self._imu is None:
            self._imu = load_asl_imu(self.root / 'imu0/data.csv')
        return self._imu

    def image_stamps(self):
        return self._stamps

    def image_times(self):
        return self._stamps * 1e-9

    def pick_stamps(self, times_ns):
        return [int(self._stamps[int(np.argmin(np.abs(self._stamps - t)))]) for t in times_ns]

    def load_backbone_images(self, times_ns, timer=None):
        import cv2
        raw = [cv2.imread(str(self._cam_dir / f'{s}.jpg'), cv2.IMREAD_UNCHANGED) for s in self.pick_stamps(times_ns)]
        return self.rectifier(raw)

    def load_filter_image(self, stamp_ns):
        from .images import read_gray_u8
        return read_gray_u8(self._cam_dir / f'{int(stamp_ns)}.jpg')

    def load_color_image(self, stamp_ns):
        from .images import read_color_u8
        return read_color_u8(self._cam_dir / f'{int(stamp_ns)}.jpg')

    def _gt_path(self):
        return next(self.seq_root.glob('*_gt')) / 'poses_gt.txt'

    def groundtruth(self):
        if self._gt is not None:
            return self._gt
        from . import groundtruth as gt_select
        path = self._gt_path()
        table = np.loadtxt(path)
        if table.ndim != 2 or table.shape[1] != 8:
            raise ValueError(f'{path}: expected TUM rows t x y z qx qy qz qw')
        t = table[:, 0]
        R_wc = Rotation.from_quat(table[:, 4:8]).as_matrix()           # xyzw, world <- camera
        p_wc = table[:, 1:4]
        # Body (IMU) pose from the camera pose through the release extrinsics:
        # T_wi = T_wc @ inv(T_ic).
        R_ci, p_ci = self.R_CtoI.T, -self.R_CtoI.T @ self.p_IC
        R_wi = R_wc @ R_ci
        p_wi = p_wc + np.einsum('nij,j->ni', R_wc, p_ci)
        self._gt_provenance = gt_select.provenance(path, 'ori-lidar-registered camera poses -> body via config/ori extrinsics', True, t)
        self._gt = GroundTruth(t=t, p=p_wi, R=R_wi, v=np.zeros_like(p_wi),
                               bg=np.zeros_like(p_wi), ba=np.zeros_like(p_wi),
                               valid=np.ones(len(t), bool), v_valid=np.zeros(len(t), bool),
                               orientation_reliable=True)
        return self._gt

    def groundtruth_provenance(self):
        self.groundtruth()
        return dict(self._gt_provenance)

    def reference_biases(self):
        return None, None    # no bias reference in this release

    def reference_time_offset(self):
        return 0.0
