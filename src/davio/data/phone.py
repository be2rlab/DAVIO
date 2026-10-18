import json
import pathlib

import numpy as np

from .types import CameraModel, GroundTruth

GRAVITY_MAG = 9.81


def square_pixels(intrinsics, metadata=None, resolution=None):
    fx, fy, cx, cy = (float(v) for v in intrinsics)
    camera = (metadata or {}).get('camera') or {}
    array = (camera.get('active_array_width'), camera.get('active_array_height'))
    out = (camera.get('configured_width'), camera.get('configured_height'))
    if all(array) and all(out):
        sensor_aspect, output_aspect = array[0] / array[1], out[0] / out[1]
    elif resolution is not None and fx > 0 and fy > 0:
        output_aspect = resolution[0] / resolution[1]
        sensor_aspect = output_aspect * fy / fx
    else:
        return fx, fy, cx, cy
    if abs(sensor_aspect - output_aspect) < 1e-2 * output_aspect:
        return fx, fy, cx, cy         # same aspect: the reported pair is consistent
    return fx, fx, cx, cy             # cropped short axis: square pixels, fy == fx


def kalibr_camchain(session):
    """The session's Kalibr camera-IMU chain, if it ships one, else None."""
    folder = pathlib.Path(session) / 'calibration'
    if not folder.is_dir():
        return None
    chains = sorted(folder.glob('*camchain-imucam.yaml'))
    return chains[0] if chains else None


# Distortion models DAVIO can carry end to end: OpenVINS reads both, and data.images builds
# a rectifier for both. Basalt's kb4 is the model Kalibr and OpenVINS call `equidistant`
# and OpenCV calls `fisheye` --- the same four-coefficient Kannala-Brandt polynomial.
MODELS = {'radtan': 'radtan', 'equidistant': 'equidistant', 'kb4': 'equidistant'}


def basalt_calibration(session):
    """The session's Basalt calibration.json, if it ships one, else None."""
    for folder in ('calib', 'calibration'):
        path = pathlib.Path(session) / folder / 'calibration.json'
        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except ValueError:
                continue
            if 'value0' in data and 'intrinsics' in data['value0']:
                return path
    return None


def read_basalt(path):
    """Camera, T_imu_cam, time offset and IMU noise from a Basalt calibration.json."""
    from scipy.spatial.transform import Rotation
    value = json.loads(pathlib.Path(path).read_text())['value0']
    entry = value['intrinsics'][0]
    kind = entry['camera_type']
    if kind not in MODELS:
        raise ValueError(f'{path}: DAVIO reads radtan or kb4/equidistant, not {kind}')
    k = entry['intrinsics']
    fx, fy, cx, cy = (float(k[n]) for n in ('fx', 'fy', 'cx', 'cy'))
    coeffs = ([float(k[n]) for n in ('k1', 'k2', 'k3', 'k4')] if kind == 'kb4'
              else [float(k.get(n, 0.)) for n in ('k1', 'k2', 'p1', 'p2')])
    width, height = (int(v) for v in value['resolution'][0])
    camera = CameraModel(K=np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]),
                         D=np.asarray(coeffs, float), model=MODELS[kind],
                         resolution=(width, height))
    # Basalt stores the camera's pose in the IMU frame --- DAVIO's T_imu_cam directly.
    q = value['T_imu_cam'][0]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q['qx'], q['qy'], q['qz'], q['qw']]).as_matrix()
    T[:3, 3] = [q['px'], q['py'], q['pz']]
    # Basalt's accel/gyro_noise_std are continuous-time densities, the same quantity
    # Kalibr calls noise_density; the bias terms are its random walks.
    noise = dict(
        accelerometer_noise_density=float(np.mean(value.get('accel_noise_std', [2e-3]))),
        gyroscope_noise_density=float(np.mean(value.get('gyro_noise_std', [1.6968e-4]))),
        accelerometer_random_walk=float(np.mean(value.get('accel_bias_std', [3e-3]))),
        gyroscope_random_walk=float(np.mean(value.get('gyro_bias_std', [1.9393e-5]))))
    # Basalt and Kalibr share the convention t_imu = t_cam + offset.
    return camera, T, float(value.get('cam_time_offset_ns', 0)) * 1e-9, noise


def read_calibration(session, source='auto'):
    import yaml
    session = pathlib.Path(session)
    sensor = yaml.safe_load((session / 'mav0/cam0/sensor.yaml').read_text())
    meta_path = session / 'metadata.json'
    metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if source not in ('auto', 'kalibr', 'basalt', 'recorder'):
        raise ValueError("source must be auto, kalibr, basalt or recorder")
    chain = kalibr_camchain(session) if source in ('auto', 'kalibr') else None
    if source == 'kalibr' and chain is None:
        raise FileNotFoundError(f'{session} ships no calibration/*camchain-imucam.yaml')
    basalt = basalt_calibration(session) if (source == 'basalt' or
                                             (source == 'auto' and chain is None)) else None
    if source == 'basalt' and basalt is None:
        raise FileNotFoundError(f'{session} ships no calib/calibration.json (Basalt)')
    if basalt is not None:
        camera, T_ic, shift, noise = read_basalt(basalt)
        K = camera.K
        return dict(camera=camera, metadata=metadata, T_imu_cam=T_ic,
                    timeshift_cam_imu=shift, source='basalt', chain=str(basalt),
                    imu_noise=noise,
                    reported=[float(v) for v in sensor['intrinsics']],
                    used=[float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])])
    if chain is not None:
        cam = yaml.safe_load(chain.read_text())['cam0']
        fx, fy, cx, cy = (float(v) for v in cam['intrinsics'])
        # Kalibr publishes T_cam_imu (IMU -> camera). DAVIO and OpenVINS's kalibr reader
        # both want the camera's pose in the IMU frame, T_imu_cam, which is its inverse.
        T_ci = np.asarray(cam['T_cam_imu'], float)
        T_ic = np.eye(4)
        T_ic[:3, :3] = T_ci[:3, :3].T
        T_ic[:3, 3] = -T_ci[:3, :3].T @ T_ci[:3, 3]
        kind = cam.get('distortion_model', 'radtan')
        if kind not in MODELS:
            raise ValueError(f'{chain}: DAVIO reads radtan or equidistant distortion, not {kind}')
        model = CameraModel(
            K=np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]),
            D=np.asarray(cam['distortion_coeffs'], float), model=MODELS[kind],
            resolution=tuple(int(v) for v in cam['resolution']))
        return dict(camera=model, metadata=metadata, T_imu_cam=T_ic,
                    timeshift_cam_imu=float(cam.get('timeshift_cam_imu', 0.)),
                    source='kalibr', chain=str(chain),
                    reported=[float(v) for v in sensor['intrinsics']],
                    used=[fx, fy, cx, cy])
    fx, fy, cx, cy = square_pixels(sensor['intrinsics'], metadata, sensor['resolution'])
    model = CameraModel(
        K=np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]),
        D=np.asarray(sensor.get('distortion_coefficients') or [0., 0., 0., 0.], float),
        model='radtan', resolution=tuple(int(v) for v in sensor['resolution']))
    return dict(camera=model, metadata=metadata,
                T_imu_cam=np.asarray(sensor['T_BS']['data'], float).reshape(4, 4),
                timeshift_cam_imu=0., source='recorder', chain=None,
                reported=[float(v) for v in sensor['intrinsics']], used=[fx, fy, cx, cy])


class PhoneDataset:
    """One vi-recorder session: monocular rolling-shutter camera plus the handset IMU."""

    name = 'phone'

    def __init__(self, seq_dir, seq=None, groundtruth='dataset', repo_root=None):
        from .images import build_rectifier
        self.seq_root = pathlib.Path(seq_dir)
        self.root = self.seq_root / 'mav0'
        self.seq = seq or self.seq_root.name
        if groundtruth not in ('dataset', 'auto'):
            raise ValueError('A phone session ships no reference trajectory')
        if not (self.root / 'cam0/data').is_dir():
            raise FileNotFoundError(f'{self.root} is not an ASL session (no cam0/data)')
        self.calibration = read_calibration(self.seq_root)
        self.camera = self.calibration['camera']
        self.metadata = self.calibration['metadata']
        # From Kalibr when the session ships it; otherwise the recorder's identity, which is
        # kept as the placeholder it declares itself to be rather than replaced with a guess.
        T = self.calibration['T_imu_cam']
        self.R_CtoI, self.p_IC = T[:3, :3], T[:3, 3]
        self.gravity_mag = GRAVITY_MAG
        # The same arguments the engine uses, so a kb4/equidistant camera is rectified here
        # exactly as the online run rectifies it (radtan ignores them).
        self.rectifier = build_rectifier(self.camera, fov_deg=90., out_size=512,
                                         tone='window_stretch')
        self._cam_dir = self.root / 'cam0/data'
        # The recorder has written PNG in one session and JPEG in another, so the format is
        # discovered rather than assumed.
        frames = sorted((p for p in self._cam_dir.iterdir()
                         if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
                        key=lambda p: p.name) if self._cam_dir.is_dir() else []
        if len(frames) < 2:
            raise FileNotFoundError(f'no cam0 frames under {self._cam_dir}')
        self.suffix = frames[0].suffix
        # The stems, not data.csv: one session's index is written in completion order, which
        # is NOT chronological (its first two rows go backwards in time), and the engine
        # requires strictly increasing camera times.
        self._stamps = np.array(sorted(int(p.stem) for p in frames))
        self.cam_period = float(np.median(np.diff(self._stamps)) * 1e-9)
        self.srvins_config = None
        self._imu = None
        self._has_color = None

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
        raw = [cv2.imread(str(self._cam_dir / f'{s}{self.suffix}'), cv2.IMREAD_UNCHANGED)
               for s in self.pick_stamps(times_ns)]
        return self.rectifier(raw)

    def load_filter_image(self, stamp_ns):
        from .images import read_gray_u8
        return read_gray_u8(self._cam_dir / f'{int(stamp_ns)}{self.suffix}')

    @property
    def has_color(self):
        """Whether the frames carry colour. Some recorder builds write grey into RGBA."""
        if self._has_color is None:
            import cv2
            frame = cv2.imread(str(self._cam_dir / f'{int(self._stamps[len(self._stamps) // 2])}'
                                   f'{self.suffix}'), cv2.IMREAD_COLOR)
            spread = np.abs(frame[:, :, 0].astype(int) - frame[:, :, 2].astype(int)).max()
            self._has_color = bool(spread > 20)
        return self._has_color

    def load_color_image(self, stamp_ns):
        """The colour frame the dense map is painted with, or None for a grey session."""
        from .images import read_color_u8
        if not self.has_color:
            return None
        return read_color_u8(self._cam_dir / f'{int(stamp_ns)}{self.suffix}')

    def health(self):
        times = self.image_times()
        gaps = np.diff(times)
        imu = self.imu()
        imu_t = np.array([s.t for s in imu])
        accel = np.linalg.norm(np.array([s.accel for s in imu]), axis=1)
        session = (self.metadata or {}).get('session') or {}
        camera = (self.metadata or {}).get('camera') or {}
        written, dropped = session.get('frames_written'), session.get('frames_dropped')
        return dict(
            frames=len(times), duration_s=float(times[-1] - times[0]),
            effective_hz=float((len(times) - 1) / (times[-1] - times[0])),
            requested_hz=camera.get('requested_fps'),
            frames_dropped=dropped,
            kept_fraction=(None if not (written and dropped is not None)
                           else written / (written + dropped)),
            gap_median_ms=float(np.median(gaps) * 1000), gap_max_ms=float(gaps.max() * 1000),
            imu_hz=float((len(imu_t) - 1) / (imu_t[-1] - imu_t[0])),
            imu_covers_camera=bool(imu_t[0] <= times[0] and imu_t[-1] >= times[-1]),
            accel_mean_ms2=float(accel.mean()),
            rolling_shutter_skew_ms=(None if camera.get('rolling_shutter_skew_ns') is None
                                     else camera['rolling_shutter_skew_ns'] / 1e6),
            calibration_source=self.calibration['source'],
            intrinsics_reported=self.calibration['reported'],
            intrinsics_used=self.calibration['used'],
            distortion=[float(x) for x in self.camera.D],
            timeshift_cam_imu_s=self.calibration['timeshift_cam_imu'],
            lever_arm_m=float(np.linalg.norm(self.p_IC)),
            extrinsic_is_identity=bool(np.allclose(self.R_CtoI, np.eye(3))
                                       and np.allclose(self.p_IC, 0)))

    def groundtruth(self):
        raise FileNotFoundError(
            'A phone session ships no reference trajectory; there is nothing to score ATE '
            'against. Open the run in scripts/view_run.py instead.')

    def groundtruth_provenance(self):
        return dict(variant='none', path=None, orientation_reliable=False,
                    note='handheld phone capture; no reference exists')

    def reference_biases(self):
        return None, None

    def reference_time_offset(self):
        return 0.0
