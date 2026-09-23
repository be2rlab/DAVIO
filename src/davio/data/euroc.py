import pathlib

import numpy as np

from .types import CameraModel, GroundTruth

# cam0 T_BS from mav0/cam0/sensor.yaml: maps CAMERA -> BODY(IMU).
T_BS = np.array([[0.0148655, -0.99988093, 0.0041403, -0.02164015],
                 [0.99955725, 0.01496721, 0.02571553, -0.06467699],
                 [-0.02577444, 0.00375619, 0.99966073, 0.00981073],
                 [0.0, 0.0, 0.0, 1.0]])
K_CAM0 = np.array([[458.654, 0.0, 367.215],
                   [0.0, 457.296, 248.375],
                   [0.0, 0.0, 1.0]])
D_CAM0 = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
GRAVITY_MAG = 9.81


def _quat_to_rot(q):
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])


class EurocDataset:
    """EuRoC MAV sequence: 752x480 radtan pinhole at 20 Hz, ADIS16448, Vicon/Leica GT."""

    name = "euroc"

    def __init__(self, mav0_dir, seq=None, groundtruth='dataset', repo_root=None):
        from .images import build_rectifier

        self.root = pathlib.Path(mav0_dir)
        self.seq = seq
        # Which reference this dataset scores against, decided here and recorded with
        # every number it produces. See data/groundtruth.py for why it is explicit.
        self._gt_choice = groundtruth
        self._repo_root = pathlib.Path(
            repo_root if repo_root is not None
            else pathlib.Path(__file__).resolve().parents[3])
        self._gt_provenance = None
        self.gravity_mag = GRAVITY_MAG
        self.camera = CameraModel(K=K_CAM0, D=D_CAM0, model="radtan",
                                  resolution=(752, 480))
        # T_BS maps camera -> body, so it IS (R_CtoI, I_p_C) directly -- the
        # opposite convention from a camera-from-IMU (T_cam_imu) descriptor,
        # which is why neither is hardcoded anywhere outside its own descriptor.
        self.R_CtoI = T_BS[:3, :3]
        self.p_IC = T_BS[:3, 3]
        self.rectifier = build_rectifier(self.camera, tone="window_stretch")
        self._cam_dir = self.root / "cam0" / "data"
        self._stamps = np.array(sorted(int(p.stem) for p in self._cam_dir.glob("*.png")))
        if self._stamps.size < 2:
            raise FileNotFoundError("no cam0 frames under %s" % (self._cam_dir,))
        self.cam_period = float(np.median(np.diff(self._stamps)) * 1e-9)
        self.srvins_config = None
        self._imu = None
        self._gt = None

    def imu(self):
        from ..init.preintegration import load_asl_imu

        if self._imu is None:
            self._imu = load_asl_imu(self.root / "imu0" / "data.csv")
        return self._imu

    def image_stamps(self):
        return self._stamps

    def image_times(self):
        return self._stamps * 1e-9

    def pick_stamps(self, times_ns):
        return [int(self._stamps[int(np.argmin(np.abs(self._stamps - t)))]) for t in times_ns]

    def load_backbone_images(self, times_ns, timer=None):
        import cv2

        from contextlib import nullcontext
        with (timer.time("image_load") if timer is not None else nullcontext()):
            raw = [cv2.imread(str(self._cam_dir / ("%d.png" % s)), cv2.IMREAD_UNCHANGED)
                  for s in self.pick_stamps(times_ns)]
        with (timer.time("rectification") if timer is not None else nullcontext()):
            out = self.rectifier(raw)
        return out

    def load_filter_image(self, stamp_ns):
        from .images import read_gray_u8

        return read_gray_u8(self._cam_dir / ("%d.png" % int(stamp_ns)))

    def groundtruth(self):
        if self._gt is not None:
            return self._gt
        from . import groundtruth as gt_select

        path, variant, reliable = gt_select.resolve(
            "euroc", self.seq, self.root, self._repo_root, self._gt_choice)
        t, p, q, v, bg, ba = gt_select.read_asl_state_csv(path)
        R = np.asarray([_quat_to_rot(row) for row in q])
        self._gt_provenance = gt_select.provenance(path, variant, reliable, t)
        self._gt = GroundTruth(t=t, p=p, R=R, v=v, bg=bg, ba=ba,
                               valid=np.ones(len(t), bool),
                               v_valid=np.ones(len(t), bool),
                               orientation_reliable=reliable)
        return self._gt

    def groundtruth_provenance(self):
        """Identity of the reference actually used; safe to embed in any result file."""
        self.groundtruth()
        return dict(self._gt_provenance)

    def reference_biases(self):
        gt = self.groundtruth()
        return gt.bg[0].copy(), gt.ba[0].copy()

    def reference_time_offset(self):
        return 0.0
