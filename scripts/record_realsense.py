#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class Recorder:
    """Writes engine packets to an ASL-layout directory as they stream past."""

    def __init__(self, out, device, rig=None):
        import yaml
        self.root = Path(out)
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f'{self.root} exists and is not empty')
        (self.root / 'cam0/data').mkdir(parents=True, exist_ok=True)
        (self.root / 'imu0').mkdir(parents=True, exist_ok=True)
        self.frames = (self.root / 'cam0/data.csv').open('w', buffering=1)
        self.frames.write('#timestamp [ns],filename\n')
        self.imu = (self.root / 'imu0/data.csv').open('w', buffering=1)
        self.imu.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],'
                       'w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],'
                       'a_RS_S_z [m s^-2]\n')
        cam = None
        if rig is not None:
            chain = Path(rig).parent / 'kalibr_imucam_chain.yaml'
            if chain.is_file():
                cam = yaml.safe_load(chain.read_text().replace('%YAML:1.0', ''))['cam0']
        self.session = dict(device=device, cam0=cam, gravity_mag=9.81,
                            recorded=time.strftime('%Y-%m-%dT%H:%M:%S'))
        (self.root / 'session.json').write_text(json.dumps(self.session, indent=1))
        self.n_frames, self.n_imu, self.last_imu = 0, 0, -1.

    def write(self, packet):
        import cv2
        stamp = round(float(packet['t']) * 1e9)
        name = f'{stamp}.png'
        cv2.imwrite(str(self.root / 'cam0/data' / name), packet['image'])
        self.frames.write(f'{stamp},{name}\n')
        self.n_frames += 1
        for t, gyro, accel in packet['imu']:
            if t <= self.last_imu:
                continue                     # packets overlap by one bracketing sample
            self.last_imu = t
            self.imu.write(f'{round(t * 1e9)},{gyro[0]!r},{gyro[1]!r},{gyro[2]!r},'
                           f'{accel[0]!r},{accel[1]!r},{accel[2]!r}\n')
            self.n_imu += 1

    def close(self):
        self.frames.close()
        self.imu.close()
        self.session.update(n_frames=self.n_frames, n_imu=self.n_imu)
        (self.root / 'session.json').write_text(json.dumps(self.session, indent=1))
        return self.session


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True, help='New, empty session directory')
    p.add_argument('--seconds', type=float, help='Stop after this much sensor time')
    p.add_argument('--serial')
    p.add_argument('--stream', choices=('infrared', 'color'), default='infrared')
    p.add_argument('--width', type=int, default=848)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--gyro-hz', type=int, default=200)
    p.add_argument('--accel-hz', type=int, default=250)
    p.add_argument('--emitter', action='store_true', help='Leave the IR projector on')
    p.add_argument('--warmup', type=float, default=1.0)
    p.add_argument('--rig', type=Path, default=ROOT / 'config/realsense/estimator_config.yaml')
    a = p.parse_args(argv)

    import signal
    from davio.data.realsense import RealSenseCamera
    camera = RealSenseCamera(serial=a.serial, stream=a.stream, width=a.width, height=a.height,
                             fps=a.fps, gyro_hz=a.gyro_hz, accel_hz=a.accel_hz, emitter=a.emitter)
    device = camera.start()
    print(f"RealSense {device['name']} serial {device['serial']}: recording to {a.out}")
    print('  Ctrl-C to stop.')
    recorder = Recorder(a.out, device, a.rig)
    signal.signal(signal.SIGINT, lambda *_: camera.stop())
    started = time.monotonic()
    try:
        for packet in camera.packets(seconds=a.seconds, warmup_s=a.warmup):
            recorder.write(packet)
            if recorder.n_frames % (a.fps * 2) == 0:
                print(f'  {recorder.n_frames:6d} frames  {recorder.n_imu:7d} IMU  '
                      f'{time.monotonic() - started:6.1f} s', flush=True)
    finally:
        camera.close()
        session = recorder.close()
    print(f"\n{a.out}: {session['n_frames']} frames, {session['n_imu']} IMU samples")
    print(f'Replay it with:  python3 scripts/run_davio.py --dataset realsense '
          f'--data {a.out.parent} --sequence {a.out.name} --rate 1 --out runs/{a.out.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
