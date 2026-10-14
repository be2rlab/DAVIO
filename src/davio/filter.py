"""The visual-inertial filter: OpenVINS, unmodified, through its binding (native/openvins_binding.cpp)."""
from pathlib import Path

import numpy as np
import yaml


class Filter:
    def __init__(self, config_path):
        import openvins_ext
        self.vio = openvins_ext.VioManager(str(config_path))

    def feed(self, packet):
        """One camera frame and the IMU samples up to it. Returns the state at that frame (t, q_GtoI,
        p, v, bg, ba, q_ItoC, p_IinC, ...) or None while the filter is not initialized."""
        for t, gyro, accel in packet['imu']:
            self.vio.feed_imu(t, np.asarray(gyro), np.asarray(accel))
        self.vio.feed_camera(packet['t'], packet['image'], 0)
        if not self.vio.initialized():
            return None
        state = self.vio.get_state()
        if not all(np.isfinite(np.asarray(state[k], float)).all()
                   for k in ('t', 'q_GtoI', 'p', 'v', 'bg', 'ba', 'q_ItoC', 'p_IinC')):
            raise ValueError('OpenVINS emitted a non-finite state')
        if abs(float(state['t']) - packet['t']) > .002:      # a state of an earlier frame, not an update
            return None
        return state


def read_yaml(path):
    return yaml.safe_load('\n'.join(s for s in Path(path).read_text().splitlines() if not s.startswith('%YAML')))


def read_camera(rig):
    """The cam0 entry of a rig's camera-IMU file (intrinsics, distortion, T_imu_cam, resolution)."""
    rig = Path(rig)
    return read_yaml(rig.parent / read_yaml(rig)['relative_config_imucam'])['cam0']


class OpenCvDumper(yaml.SafeDumper):
    """cv::FileStorage rejects PyYAML's indentless sequences."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def write_yaml(path, data):
    Path(path).write_text('%YAML:1.0\n' + yaml.dump(data, Dumper=OpenCvDumper, sort_keys=False))
    return Path(path)


def materialize(rig, directory, overrides):
    """Write the filter's configuration for one run: the rig's estimator file with ``overrides``,
    and its IMU and camera files beside it. Returns the estimator file."""
    rig, directory = Path(rig).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    cfg = read_yaml(rig)
    cfg.update(overrides)
    if int(cfg.get('max_cameras', 1)) != 1:
        raise ValueError('One camera only')
    for key in ('relative_config_imu', 'relative_config_imucam'):
        cfg[key] = write_yaml(directory / (key + '.yaml'), read_yaml((rig.parent / cfg[key]).resolve())).name
    cfg.update(record_timing_information=False, save_total_state=False, record_init_pose=False, record_init_timing=False)
    return write_yaml(directory / 'estimator.yaml', cfg)
