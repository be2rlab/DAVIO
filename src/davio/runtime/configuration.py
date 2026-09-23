"""Materialize independent native OpenVINS configs; never modify a running filter."""
from pathlib import Path
import numpy as np
import yaml


def read_yaml(path):
    return yaml.safe_load('\n'.join(s for s in Path(path).read_text().splitlines()
                                    if not s.startswith('%YAML')))


class _OpenCvDumper(yaml.SafeDumper):

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def read_relative(estimator_path, key):
    """Read a sub-config the way OpenVINS does: relative to the estimator config's folder."""
    estimator_path = Path(estimator_path)
    return read_yaml(estimator_path.parent / read_yaml(estimator_path)[key])


def write_yaml(path, data):
    """Write a config OpenVINS (via cv::FileStorage) can actually parse."""
    Path(path).write_text('%YAML:1.0\n' + yaml.dump(data, Dumper=_OpenCvDumper,
                                                    sort_keys=False))
    return Path(path)


def materialize(source, directory, overrides, candidate=None, mode='supplied', priors=None):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    if mode not in ('supplied', 'extrinsics_free', 'free'):
        raise ValueError('calibration mode must be supplied, extrinsics_free or free')
    if mode != 'supplied' and candidate is None:
        raise ValueError(f'{mode} needs a feed-forward candidate; no native instance can be built')
    if mode != 'supplied' and priors is None:
        raise ValueError('calibration priors are required for a free mode')
    if mode == 'free' and 'K_raw' not in candidate:
        raise ValueError('free mode needs candidate intrinsics K_raw')
    directory.mkdir(parents=True, exist_ok=False)
    cfg = read_yaml(source)
    cfg.update(overrides)
    if int(cfg.get('max_cameras', 1)) != 1:
        raise ValueError('This driver supplies cam0 only; a stereo config needs a stereo driver')
    # Both the filter and ov_init read these files. Changing only the state after
    # VioManager construction would leave ov_init's calibration inconsistent.
    for key in ('relative_config_imu', 'relative_config_imucam'):
        src = (source.parent / cfg[key]).resolve()
        data = read_yaml(src)
        if candidate is not None and key == 'relative_config_imucam':
            r = np.asarray(candidate['R_CtoI'], float)
            b = np.asarray(candidate['bg'], float)
            if r.shape != (3, 3) or b.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(b).all():
                raise ValueError('Invalid startup calibration candidate')
            if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or np.linalg.det(r) < .999:
                raise ValueError('Candidate must be a proper rotation')
            cam = data['cam0']
            transform = np.asarray(cam['T_imu_cam'], float)
            transform[:3, :3] = r
            if mode != 'supplied':
                # Nothing of the rig's extrinsics survives: translation from the candidate
                # (prior-held when unobservable), time offset from zero.
                p = np.asarray(candidate.get('p_CinI', [0., 0., 0.]), float)
                if p.shape != (3,) or not np.isfinite(p).all():
                    raise ValueError('Invalid lever arm')
                transform[:3, 3] = p
                cam['timeshift_cam_imu'] = 0.0
                cam['cam_overlaps'] = []
                if mode == 'free':
                    cam['intrinsics'] = [float(x) for x in candidate['K_raw']]
                    cam['distortion_coeffs'] = [float(x) for x in candidate.get('D', [0., 0., 0., 0.])]
                    cam['distortion_model'] = 'radtan'
                    cam['camera_model'] = 'pinhole'
                data = {'cam0': cam}
            cam['T_imu_cam'] = transform.tolist()   # supplied: camera origin preserved
            cfg['init_dyn_bias_g'] = b.tolist()  # Optimizer initial guess, no prior information.
            if candidate.get('bootstrap') is not None:
                # The injected state must never race the filter's own initializer. ov_init's
                # parser exits on a non-positive threshold or disparity when the dynamic path
                # is off, and its static path fires whenever the disparity gate says "still"
                # (wait_for_jerk is false with ZUPT on). A tiny positive disparity limit makes
                # every window count as moving, so neither path can ever initialize.
                cfg.update(init_dyn_use=False, init_imu_thresh=1e9, init_max_disparity=1e-6)
            if mode != 'supplied':
                cfg.update(calib_cam_extrinsics=True, calib_cam_timeoffset=True,
                           calib_cam_intrinsics=(mode == 'free'),
                           init_prior_qc=float(np.radians(float(priors['rotation_deg']))),
                           init_prior_pc=float(priors['translation_m']),
                           init_prior_fc=float(priors['focal_px']),
                           init_prior_dc1=float(priors['distortion']),
                           init_prior_dc2=float(priors['distortion']),
                           init_prior_t=float(priors['time_offset_s']))
        dest = write_yaml(directory / (key + '.yaml'), data)
        cfg[key] = dest.name
    # Disable shared /tmp diagnostic writers in competing native instances.
    cfg.update(record_timing_information=False, save_total_state=False,
               record_init_pose=False, record_init_timing=False)
    return write_yaml(directory / 'estimator.yaml', cfg)


def fingerprint(root):
    """Content hash of every source, config and build file that can change a result."""
    import hashlib
    root = Path(root).resolve()
    paths = sorted(p for folder in ('src', 'config', 'scripts', 'native', 'docker', 'thirdparty/accelerated_features/modules')
                   for p in (root / folder).rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.cpp', '.yaml', '.sh', '.txt'))
    digest = hashlib.sha256()
    for p in paths:
        digest.update(str(p.relative_to(root)).encode() + b'\0' + p.read_bytes())
    for p in sorted((root/'thirdparty/accelerated_features/weights').glob('*.pt')):
        digest.update(p.name.encode()+p.read_bytes())
    return digest.hexdigest()


def perturb_rig(source, directory, degrees, seed):
    from scipy.spatial.transform import Rotation
    source, directory = Path(source).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    cfg = read_yaml(source)
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    delta = Rotation.from_rotvec(np.radians(float(degrees)) * axis).as_matrix()
    for key in ('relative_config_imu', 'relative_config_imucam'):
        data = read_yaml((source.parent / cfg[key]).resolve())
        if key == 'relative_config_imucam':
            for entry in data.values():
                transform = np.asarray(entry['T_imu_cam'], float)
                transform[:3, :3] = delta @ transform[:3, :3]
                entry['T_imu_cam'] = transform.tolist()
        dest = write_yaml(directory / (key + '.yaml'), data)
        cfg[key] = dest.name
    return write_yaml(directory / 'estimator.yaml', cfg)
