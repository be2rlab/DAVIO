import json
import pathlib
import threading
import time

import numpy as np

from .types import CameraModel

# D400 units report a gravity magnitude of one g; the local value belongs in the rig file.
GRAVITY_MAG = 9.81
DEFAULTS = dict(width=848, height=480, fps=30, gyro_hz=200, accel_hz=250)


def _config_dir(repo_root=None):
    root = pathlib.Path(repo_root) if repo_root else pathlib.Path(__file__).resolve().parents[3]
    return root / 'config/realsense'


def load_calibration(repo_root=None):
    """cam0 block of the generated rig, as a dict. Raises if the rig was never generated."""
    import yaml
    path = _config_dir(repo_root) / 'kalibr_imucam_chain.yaml'
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} does not exist. Plug the camera in and run '
            '`python3 scripts/realsense_calibrate.py` to write it from the device.')
    return yaml.safe_load(path.read_text().replace('%YAML:1.0', ''))['cam0']


def camera_model(cam):
    fx, fy, cx, cy = cam['intrinsics']
    return CameraModel(K=np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]),
                       D=np.asarray(cam['distortion_coeffs'], float),
                       model=cam['distortion_model'], resolution=tuple(cam['resolution']))


def _interpolate_accel(t, samples):
    """Accelerometer value at time ``t`` from ``[(time, xyz), ...]``, or None if outside."""
    if len(samples) < 2 or not samples[0][0] <= t <= samples[-1][0]:
        return None
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if samples[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, a0 = samples[lo]
    t1, a1 = samples[hi]
    if t1 <= t0:
        return a0
    w = (t - t0) / (t1 - t0)
    return (1. - w) * a0 + w * a1


class RealSenseCamera:

    name = 'realsense'

    def __init__(self, serial=None, stream='infrared', width=DEFAULTS['width'],
                 height=DEFAULTS['height'], fps=DEFAULTS['fps'], gyro_hz=DEFAULTS['gyro_hz'],
                 accel_hz=DEFAULTS['accel_hz'], emitter=False, repo_root=None):
        self.serial, self.stream_name = serial, stream
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.gyro_hz, self.accel_hz, self.emitter = int(gyro_hz), int(accel_hz), bool(emitter)
        self._repo_root = repo_root
        self.gravity_mag = GRAVITY_MAG
        self._pipeline = self._rs = None
        self._lock = threading.Lock()
        self._frames, self._gyro, self._accel = [], [], []
        self._stopped = threading.Event()
        self.profile_info = None

    # ------------------------------------------------------------------ device
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def start(self):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                'pyrealsense2 is not installed. Build the camera image with '
                '`make image-realsense`, or `pip install pyrealsense2` on the host.') from exc
        self._rs = rs
        config = rs.config()
        if self.serial:
            config.enable_device(str(self.serial))
        if self.stream_name == 'infrared':
            # Index 1 is the LEFT imager, which is the depth module's reference frame and
            # the one the factory extrinsics are expressed against.
            config.enable_stream(rs.stream.infrared, 1, self.width, self.height,
                                 rs.format.y8, self.fps)
        elif self.stream_name == 'color':
            config.enable_stream(rs.stream.color, self.width, self.height,
                                 rs.format.bgr8, self.fps)
        else:
            raise ValueError("stream must be 'infrared' or 'color'")
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, self.gyro_hz)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, self.accel_hz)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config, self._on_frame)
        device = profile.get_device()
        for sensor in device.query_sensors():
            # One clock for every stream, or the camera and IMU timestamps are incomparable.
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1)
            if sensor.supports(rs.option.emitter_enabled):
                sensor.set_option(rs.option.emitter_enabled, 1 if self.emitter else 0)
        self.profile_info = dict(
            name=device.get_info(rs.camera_info.name),
            serial=device.get_info(rs.camera_info.serial_number),
            firmware=device.get_info(rs.camera_info.firmware_version),
            stream=self.stream_name, width=self.width, height=self.height, fps=self.fps,
            gyro_hz=self.gyro_hz, accel_hz=self.accel_hz, emitter=self.emitter)
        return self.profile_info

    def close(self):
        self._stopped.set()
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
            self._pipeline = None

    def stop(self):
        """Ask packets() to finish: it drains what the device already delivered, then returns."""
        self._stopped.set()

    # ------------------------------------------------------------------ capture
    def _on_frame(self, frame):
        """librealsense callback thread: timestamp, convert and buffer. No blocking work."""
        rs = self._rs
        now = time.monotonic()
        motion = frame.as_motion_frame()
        if motion:
            t = motion.get_timestamp() * 1e-3        # device milliseconds -> seconds
            value = motion.get_motion_data()
            xyz = np.array([value.x, value.y, value.z], float)
            with self._lock:
                if motion.get_profile().stream_type() == rs.stream.gyro:
                    self._gyro.append((t, xyz))
                else:
                    self._accel.append((t, xyz))
            return
        video = frame.as_video_frame()
        if not video:
            return
        image = np.asanyarray(video.get_data())
        if image.ndim == 3:                          # colour: the tracker wants one channel
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        with self._lock:
            self._frames.append((video.get_timestamp() * 1e-3, image.copy(), now))

    def _take(self):
        """Move buffered frames out and take a snapshot of the inertial buffers."""
        with self._lock:
            frames, self._frames = self._frames, []
            return frames, list(self._gyro), list(self._accel)

    def _trim(self, before):
        """Drop inertial samples the engine already has, keeping enough to interpolate."""
        with self._lock:
            self._gyro = [s for s in self._gyro if s[0] > before]
            older = [i for i, s in enumerate(self._accel) if s[0] <= before]
            if older:
                self._accel = self._accel[older[-1]:]

    def packets(self, seconds=None, warmup_s=1.0):
        if self._pipeline is None:
            raise RuntimeError('call start() (or use the context manager) first')
        pending = []
        origin = None
        emitted = -np.inf      # newest camera time handed to the engine
        consumed = -np.inf     # newest inertial time handed to the engine
        first = True
        final = False
        while True:
            # stop() is honoured by making the NEXT pass the last one rather than
            # returning straight away: frames the device delivered while this generator
            # was suspended at a yield still reach the engine.
            final = final or self._stopped.is_set()
            frames, gyro, accel = self._take()
            # By time only: two frames can carry the same device timestamp, and tuple
            # ordering would then fall through to comparing the image arrays.
            pending = sorted(pending + frames, key=lambda frame: frame[0])
            # Inertial stream at the gyroscope's own times, accelerometer interpolated onto
            # them. A gyro sample outside the accelerometer's span yields nothing.
            fused = [(t, g, a) for t, g in gyro
                     if (a := _interpolate_accel(t, accel)) is not None]
            progressed = False
            while pending:
                t, image, arrival = pending[0]
                after = next((s for s in fused if s[0] > t), None)
                if after is None:
                    break                       # no bracketing sample yet: wait for one
                pending.pop(0)
                progressed = True
                if origin is None:
                    origin = t
                if t - origin < warmup_s or t <= emitted:
                    continue
                window = [s for s in fused if consumed < s[0] <= t]
                if first:
                    # One sample before the first frame, for boundary interpolation.
                    window = [s for s in fused if s[0] <= t][-2:]
                if not window:
                    continue
                samples = [(float(s[0]), s[1], s[2]) for s in window + [after]]
                consumed, emitted, first = after[0], t, False
                self._trim(consumed)
                yield dict(t=float(t), image=image, imu=samples, arrival_wall=arrival)
                if seconds is not None and t - origin >= seconds + warmup_s:
                    return
            if final:
                return
            if not progressed:
                time.sleep(.002)


class RealSenseDataset:
    """A session recorded by scripts/record_realsense.py, replayed like any other dataset."""

    name = 'realsense'

    def __init__(self, seq_dir, seq=None, groundtruth='dataset', repo_root=None):
        from .images import build_rectifier
        self.root = pathlib.Path(seq_dir)
        self.seq = seq or self.root.name
        if groundtruth not in ('dataset', 'auto'):
            raise ValueError('A recorded RealSense session has no reference trajectory')
        session = self.root / 'session.json'
        self.session = json.loads(session.read_text()) if session.is_file() else {}
        cam = self.session.get('cam0') or load_calibration(repo_root)
        self.camera = camera_model(cam)
        T = np.asarray(cam['T_imu_cam'], float)
        self.R_CtoI, self.p_IC = T[:3, :3], T[:3, 3]
        self.gravity_mag = float(self.session.get('gravity_mag', GRAVITY_MAG))
        self.rectifier = build_rectifier(self.camera, fov_deg=90., out_size=512,
                                         tone='window_stretch')
        self._cam_dir = self.root / 'cam0/data'
        self._stamps = np.array(sorted(int(p.stem) for p in self._cam_dir.glob('*.png')))
        if self._stamps.size < 2:
            raise FileNotFoundError(f'no cam0 frames under {self._cam_dir}')
        self.cam_period = float(np.median(np.diff(self._stamps)) * 1e-9)
        self.srvins_config = None
        self._imu = None

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
        raw = [cv2.imread(str(self._cam_dir / f'{s}.png'), cv2.IMREAD_UNCHANGED)
               for s in self.pick_stamps(times_ns)]
        return self.rectifier(raw)

    def load_filter_image(self, stamp_ns):
        from .images import read_gray_u8
        return read_gray_u8(self._cam_dir / f'{int(stamp_ns)}.png')

    def groundtruth(self):
        raise FileNotFoundError(
            'A recorded RealSense session ships no reference trajectory; there is nothing '
            'to score ATE against. Use scripts/view_run.py to inspect the result instead.')

    def groundtruth_provenance(self):
        return dict(variant='none', path=None, orientation_reliable=False,
                    note='handheld capture; no reference exists')

    def reference_biases(self):
        return None, None

    def reference_time_offset(self):
        return 0.0
