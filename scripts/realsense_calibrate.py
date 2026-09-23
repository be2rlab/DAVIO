#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--serial', help='Which unit, if more than one is attached')
    p.add_argument('--stream', choices=('infrared', 'color'), default='infrared')
    p.add_argument('--width', type=int, default=848)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--gyro-hz', type=int, default=200)
    p.add_argument('--accel-hz', type=int, default=250)
    p.add_argument('--out', type=Path, default=ROOT / 'config/realsense/kalibr_imucam_chain.yaml')
    p.add_argument('--print-only', action='store_true', help='Show the values, write nothing')
    a = p.parse_args(argv)

    import numpy as np
    try:
        import pyrealsense2 as rs
    except ImportError:
        p.error('pyrealsense2 is not installed. Build the camera image with '
                '`make image-realsense`, or `pip install pyrealsense2` on the host.')
    from davio.runtime.configuration import write_yaml

    config = rs.config()
    if a.serial:
        config.enable_device(a.serial)
    if a.stream == 'infrared':
        config.enable_stream(rs.stream.infrared, 1, a.width, a.height, rs.format.y8, a.fps)
    else:
        config.enable_stream(rs.stream.color, a.width, a.height, rs.format.bgr8, a.fps)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, a.gyro_hz)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, a.accel_hz)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    try:
        device = profile.get_device()
        stream = (profile.get_stream(rs.stream.infrared, 1) if a.stream == 'infrared'
                  else profile.get_stream(rs.stream.color))
        video = stream.as_video_stream_profile()
        intrinsics = video.get_intrinsics()
        gyro = profile.get_stream(rs.stream.gyro)
        extrinsics = stream.get_extrinsics_to(gyro)
        info = dict(name=device.get_info(rs.camera_info.name),
                    serial=device.get_info(rs.camera_info.serial_number),
                    firmware=device.get_info(rs.camera_info.firmware_version))
    finally:
        pipeline.stop()

    # librealsense hands back a column-major 3x3 and a translation that together take a
    # point in the CAMERA frame to the IMU frame: exactly T_imu_cam = (R_CtoI, p_CinI).
    R = np.asarray(extrinsics.rotation, float).reshape(3, 3).T
    t = np.asarray(extrinsics.translation, float)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t

    # A D400 reports plus_modified_brown_conrady (or none, for the rectified IR imagers);
    # OpenVINS calls the same four-coefficient model radtan.
    model = str(intrinsics.model).rsplit('.', 1)[-1]
    coefficients = [float(c) for c in intrinsics.coeffs[:4]]
    if model not in ('brown_conrady', 'modified_brown_conrady', 'inverse_brown_conrady', 'none'):
        print(f'warning: distortion model {model!r} is not a radtan model; '
              f'DAVIO will treat the four coefficients as radtan', file=sys.stderr)

    cam = dict(
        T_imu_cam=T.tolist(),
        cam_overlaps=[],
        camera_model='pinhole',
        distortion_coeffs=coefficients,
        distortion_model='radtan',
        intrinsics=[float(intrinsics.fx), float(intrinsics.fy),
                    float(intrinsics.ppx), float(intrinsics.ppy)],
        resolution=[int(intrinsics.width), int(intrinsics.height)],
        rostopic=f'/realsense/{a.stream}/image_raw',
        timeshift_cam_imu=0.0)

    print(json.dumps(dict(device=info, stream=a.stream, librealsense_distortion=model,
                          cam0=cam), indent=1))
    if a.print_only:
        return 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(a.out, dict(cam0=cam))
    (a.out.parent / 'device.json').write_text(json.dumps(
        dict(device=info, stream=a.stream, width=a.width, height=a.height, fps=a.fps,
             gyro_hz=a.gyro_hz, accel_hz=a.accel_hz,
             librealsense_distortion=model), indent=1))
    print(f'\nwrote {a.out.relative_to(ROOT)} and {(a.out.parent / "device.json").relative_to(ROOT)}')
    print('Run DAVIO on the camera with:  python3 scripts/run_realsense.py --out runs/live')
    return 0


if __name__ == '__main__':
    sys.exit(main())
