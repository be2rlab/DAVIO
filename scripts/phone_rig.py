#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

# Android SENSOR frame: x right, y up, z out of the screen toward the user.
# Camera OPTICAL frame: x right in the image, y down in the image, z along the view
# direction, out of the back. A rear camera therefore opposes the sensor frame's y and z:
#   portrait   R_CtoI = diag(1, -1, -1)
# A LANDSCAPE capture (the output is wider than it is tall) additionally turns the image
# 90 degrees about the optical axis relative to the device, whose sensor frame does not
# rotate with the capture orientation:
#   landscape  R_CtoI = diag(1, -1, -1) @ Rz(90 deg) = [[0,-1,0], [-1,0,0], [0,0,-1]]
#
# This is a prior, not a calibration, but it is the right ORDER of prior: on the session in
# data/phone, DAVIO's feed-forward initializer --- which estimates the rotation from the
# imagery and the IMU with no prior at all --- converged to within 2.3 degrees of the
# landscape matrix, against 91 degrees for the portrait one and 179 degrees for the
# identity the recorder writes.
PORTRAIT_R_CtoI = [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
LANDSCAPE_R_CtoI = [[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]]


def orientation_prior(resolution):
    """(R_CtoI, description) from the capture aspect."""
    width, height = int(resolution[0]), int(resolution[1])
    if width >= height:
        return LANDSCAPE_R_CtoI, 'android landscape prior diag(1,-1,-1) @ Rz(90)'
    return PORTRAIT_R_CtoI, 'android portrait prior diag(1,-1,-1)'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'config/phone',
                   help='Rig directory; a different handset belongs in its own, since the '
                        'rig is per device')
    p.add_argument('--check', action='store_true', help='Report the session and write nothing')
    p.add_argument('--identity-extrinsic', action='store_true',
                   help="Write the recorder's identity instead of the Android prior "
                        '(ignored when the session ships a Kalibr calibration)')
    p.add_argument('--calibration', choices=('auto', 'kalibr', 'basalt', 'recorder'), default='auto',
                   help="auto prefers a shipped Kalibr calibration; recorder ignores it and uses "
                        "the recorder's own values with the Android prior, and lets the filter "
                        'calibrate intrinsics and time offset online instead')
    p.add_argument('--raw-imu-noise', action='store_true',
                   help='Use the noise densities exactly as calibrated instead of inflating '
                        'them (white noise x2, random walk x10) the way OpenVINS does')
    p.add_argument('--lever-arm', choices=('kalibr', 'zero'), default='kalibr',
                   help='With a Kalibr calibration: keep its camera-IMU translation, or zero '
                        'it and let the filter estimate it. See the report this prints')
    a = p.parse_args(argv)
    import numpy as np

    import shutil
    import yaml
    from davio.data.phone import read_calibration, PhoneDataset
    from davio.runtime.configuration import write_yaml

    calibration = read_calibration(a.session, a.calibration)
    camera = calibration['camera']
    health = PhoneDataset(a.session).health()
    # The dataset reports the calibration it would use by default; the rig follows --calibration.
    health.update(calibration_source=calibration['source'],
                  intrinsics_reported=calibration['reported'], intrinsics_used=calibration['used'],
                  distortion=[float(x) for x in camera.D],
                  timeshift_cam_imu_s=calibration['timeshift_cam_imu'],
                  lever_arm_m=float(np.linalg.norm(calibration['T_imu_cam'][:3, 3])),
                  extrinsic_is_identity=bool(np.allclose(calibration['T_imu_cam'], np.eye(4))))
    print(json.dumps(health, indent=1))
    warn = []
    if health['kept_fraction'] and health['kept_fraction'] < .9:
        warn.append(f"the recorder dropped {100 * (1 - health['kept_fraction']):.0f}% of frames: "
                    f"{health['effective_hz']:.1f} Hz reaches disk, not "
                    f"{health['requested_hz']}. Gaps up to {health['gap_max_ms']:.0f} ms")
    if health['rolling_shutter_skew_ms'] and health['rolling_shutter_skew_ms'] > 5:
        warn.append(f"rolling shutter reads out over {health['rolling_shutter_skew_ms']:.0f} ms "
                    'and nothing in this pipeline models it')
    if calibration['source'] in ('kalibr', 'basalt'):
        print(f"  calibration: {calibration['source']} ({calibration['chain']}), "
              f"{camera.model} model")
        if health['lever_arm_m'] > .05:
            warn.append(f"Kalibr's camera-IMU translation is {100 * health['lever_arm_m']:.0f} cm. "
                        'A phone camera sits within a few cm of its IMU; a lever arm this long '
                        'is the usual sign of weakly excited translation or a wrong tag size')
    else:
        if health['extrinsic_is_identity']:
            warn.append('the camera-IMU extrinsic is uncalibrated (identity in the session)')
        if health['intrinsics_reported'][:2] != health['intrinsics_used'][:2]:
            warn.append(f"fy repaired {health['intrinsics_reported'][1]:.1f} -> "
                        f"{health['intrinsics_used'][1]:.1f} (square pixels; see phone.square_pixels)")
    if not health['imu_covers_camera']:
        warn.append('the IMU does not span the whole camera stream; the replay interval '
                    'is clipped to the overlap')
    for w in warn:
        print(f'  ! {w}')
    if a.check:
        return 0

    if calibration['source'] in ('kalibr', 'basalt'):
        T = np.asarray(calibration['T_imu_cam'], float).copy()
        description = f"{calibration['source']} ({Path(calibration['chain']).name})"
        if a.lever_arm == 'zero':
            T[:3, 3] = 0.
            description += ', translation zeroed'
        timeshift = calibration['timeshift_cam_imu']
    else:
        prior, description = orientation_prior(camera.resolution)
        if a.identity_extrinsic:
            prior, description = np.eye(3).tolist(), 'identity (recorder default)'
        T = np.eye(4)
        T[:3, :3] = np.asarray(prior, float)
        timeshift = 0.
    print(f'  extrinsic: {description}; timeshift_cam_imu {timeshift * 1000:+.2f} ms')

    a.out.mkdir(parents=True, exist_ok=True)
    template = ROOT / 'config/phone/estimator_config.yaml'
    if a.out.resolve() != template.parent.resolve():
        shutil.copy(template, a.out / 'estimator_config.yaml')
    write_yaml(a.out / 'kalibr_imucam_chain.yaml', dict(cam0=dict(
        T_imu_cam=T.tolist(), cam_overlaps=[], camera_model='pinhole',
        distortion_coeffs=[float(x) for x in camera.D[:4]],
        # The model the coefficients belong to. Writing radtan for kb4 coefficients would be
        # read by OpenVINS as a different lens without any error.
        distortion_model=camera.model,
        intrinsics=[float(camera.K[0, 0]), float(camera.K[1, 1]),
                    float(camera.K[0, 2]), float(camera.K[1, 2])],
        resolution=[int(camera.resolution[0]), int(camera.resolution[1])],
        rostopic='/cam0/image_raw', timeshift_cam_imu=float(timeshift))))

    # IMU noise: a Kalibr imu0.yaml beside the camchain if there is one (measured, at least
    # for the white-noise terms), else the recorder's placeholders.
    imu_source = a.session / 'calibration/imu0.yaml'
    if calibration['source'] == 'basalt':
        imu_source = Path(calibration['chain'])
        imu_sensor = dict(calibration['imu_noise'])
    else:
        if calibration['source'] != 'kalibr' or not imu_source.is_file():
            imu_source = a.session / 'mav0/imu0/sensor.yaml'
        imu_sensor = yaml.safe_load(imu_source.read_text())
    imu = dict(T_i_b=np.eye(4).tolist())
    for key, fallback in (('accelerometer_noise_density', 2e-3), ('accelerometer_random_walk', 3e-3),
                          ('gyroscope_noise_density', 1.6968e-4), ('gyroscope_random_walk', 1.9393e-5)):
        imu[key] = float(imu_sensor.get(key, fallback))
    # A calibrated density is the sensor alone; the filter also has to absorb everything it
    # does not model --- here a rolling shutter, vibration, the resampling the recorder
    # applies. OpenVINS inflates for exactly this (config/rs_d455: white x2, random walk
    # x10), and on the Redmi session raw Kalibr noise let the filter over-trust the IMU:
    # 165 m of drift in 60 s, against 68 m inflated, everything else equal.
    inflated = (calibration['source'] == 'kalibr' and imu_source.parent.name == 'calibration'
                and not a.raw_imu_noise)
    if inflated:
        for key, factor in (('accelerometer_noise_density', 2.), ('gyroscope_noise_density', 2.),
                            ('accelerometer_random_walk', 10.), ('gyroscope_random_walk', 10.)):
            imu[key] *= factor
    imu.update(rostopic='/imu0', time_offset=0.0,
               update_rate=float(round(health['imu_hz'])), model='kalibr',
               Tw=np.eye(3).tolist(), R_IMUtoGYRO=np.eye(3).tolist(), Ta=np.eye(3).tolist(),
               R_IMUtoACC=np.eye(3).tolist(), Tg=np.zeros((3, 3)).tolist())
    write_yaml(a.out / 'kalibr_imu_chain.yaml', dict(imu0=imu))
    (a.out / 'session.json').write_text(json.dumps(
        dict(session=str(a.session), device=(calibration['metadata'] or {}).get('device'),
             calibration=calibration['source'], imu_noise_from=str(imu_source),
             health=health, extrinsic=description, warnings=warn), indent=1))
    print(f'  imu noise: {imu_source}' + (' (inflated x2 white / x10 random walk)' if inflated else ''))
    print(f'\nwrote {a.out}/ (estimator_config, kalibr chains, session.json)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
