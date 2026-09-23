import csv
import pathlib

import numpy as np

from .types import CameraModel, GroundTruth

GRAVITY_MAG = 9.81
ROOM_SEQUENCES = ("room1", "room2", "room3", "room4", "room5", "room6")

# Sec. 5.1 requires a rectified pinhole crop out of the 195-degree fisheye before the backbone.
# No value for it is recoverable from the reference, so this matches the UZH-FPV adapter's
# already-declared choice rather than inventing a second, differently-cropped convention.
DEFAULT_FOV_DEG = 90.0
DEFAULT_OUT_SIZE = 512

# Mocap gaps in these sequences run to ~1 s (measured on room1), far above the 120 Hz period;
# 0.1 s separates "a dropped sample" from "the subject left the volume".
GT_GAP_S = 0.1


def _quat_to_rot(q):
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])


def load_groundtruth(mocap_csv, gap_s=GT_GAP_S):
    """`#timestamp [ns], p_RS_R_{x,y,z}, q_RS_{w,x,y,z}` -- EuRoC's own GT column order."""
    from .gt_velocity import differentiate_positions

    t, p, q = [], [], []
    with pathlib.Path(mocap_csv).open() as handle:
        for row in csv.reader(handle):
            if not row or row[0].lstrip().startswith("#"):
                continue
            t.append(int(row[0]) * 1e-9)
            p.append([float(x) for x in row[1:4]])
            q.append([float(x) for x in row[4:8]])
    t = np.asarray(t)
    p = np.asarray(p)
    R = np.asarray([_quat_to_rot(np.asarray(row)) for row in q])
    v, v_valid = differentiate_positions(t, p, gap_s)
    return GroundTruth(t=t, p=p, R=R, v=v, bg=None, ba=None,
                       valid=np.ones(len(t), bool), v_valid=v_valid)


class TumViDataset:
    """TUM-VI room sequence: 512x512 equidistant fisheye at 20 Hz, BMI160 at 200 Hz, mocap GT."""

    name = "tumvi"

    def __init__(self, root, seq=None, fov_deg=DEFAULT_FOV_DEG, out_size=DEFAULT_OUT_SIZE):
        import yaml

        from .images import build_rectifier

        self.root = pathlib.Path(root)
        self.seq = seq
        self.gravity_mag = GRAVITY_MAG
        self.mav0 = self.root / "mav0"

        chain = yaml.safe_load((self.root / "dso" / "camchain.yaml").read_text())
        cam0 = chain["cam0"]
        if cam0["distortion_model"] != "equidistant":
            raise ValueError("expected an equidistant cam0, got %r"
                             % (cam0["distortion_model"],))
        fx, fy, cx, cy = cam0["intrinsics"]
        self.camera = CameraModel(
            K=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
            D=np.asarray(cam0["distortion_coeffs"], float),
            model="equidistant",
            resolution=(int(cam0["resolution"][0]), int(cam0["resolution"][1])))
        # Kalibr's T_cam_imu maps IMU -> CAM, the opposite of EuRoC's T_BS. Inverted here once,
        # in the descriptor that knows its own direction, exactly as `uzhfpv` does.
        t_cam_imu = np.asarray(cam0["T_cam_imu"], float)
        self.R_CtoI = t_cam_imu[:3, :3].T
        self.p_IC = -t_cam_imu[:3, :3].T @ t_cam_imu[:3, 3]

        self.rectifier = build_rectifier(self.camera, fov_deg=fov_deg, out_size=out_size,
                                         tone="window_stretch")
        self._cam_dir = self.mav0 / "cam0" / "data"
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
            self._imu = load_asl_imu(self.mav0 / "imu0" / "data.csv")
        return self._imu

    def image_stamps(self):
        return self._stamps

    def image_times(self):
        return self._stamps * 1e-9

    def pick_stamps(self, times_ns):
        return [int(self._stamps[int(np.argmin(np.abs(self._stamps - t)))]) for t in times_ns]

    def load_backbone_images(self, times_ns, timer=None):
        from contextlib import nullcontext

        import cv2

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
        if self._gt is None:
            self._gt = load_groundtruth(self.mav0 / "mocap0" / "data.csv")
        return self._gt

    def reference_biases(self):
        raise NotImplementedError(
            "TUM-VI ships no reference IMU biases (unlike EuRoC's "
            "state_groundtruth_estimate0) -- a baseline needing a bias prior must supply "
            "its own rather than read a zero out of this adapter")

    def reference_time_offset(self):
        return 0.0
