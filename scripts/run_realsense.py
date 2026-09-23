#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    import yaml
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=ROOT / 'config/system.yaml')
    p.add_argument('--rig', type=Path, default=ROOT / 'config/realsense/estimator_config.yaml')
    p.add_argument('--out', type=Path, required=True, help='New run directory, never overwritten')
    p.add_argument('--serial', help='Which unit, if more than one is attached')
    p.add_argument('--stream', choices=('infrared', 'color'), default='infrared',
                   help='infrared is global-shutter and factory-rectified; colour is rolling-shutter')
    p.add_argument('--width', type=int, default=848)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--gyro-hz', type=int, default=200)
    p.add_argument('--accel-hz', type=int, default=250)
    p.add_argument('--emitter', action='store_true',
                   help='Leave the IR projector ON. Its dot pattern is a field of fake corners '
                        'that moves with the camera; only useful to see what it does to tracking')
    p.add_argument('--mode', choices=('assist', 'native'), default='assist')
    p.add_argument('--no-map', action='store_true')
    p.add_argument('--weights', help='Local DA3 checkpoint directory')
    p.add_argument('--seconds', type=float, help='Stop after this much sensor time')
    p.add_argument('--warmup', type=float, default=1.0,
                   help='Seconds of frames to drop while auto-exposure settles')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--record', type=Path,
                   help='Also write the session here in ASL layout, replayable with '
                        '--dataset realsense')
    p.add_argument('--set', action='append', default=[], metavar='a.b=VALUE',
                   help='Override one config key, YAML-parsed. Repeatable')
    a = p.parse_args(argv)
    if a.out.exists():
        p.error('--out must not exist')
    if not a.rig.is_file():
        p.error(f'{a.rig} does not exist; run scripts/realsense_calibrate.py first')
    chain = a.rig.parent / 'kalibr_imucam_chain.yaml'
    if not chain.is_file():
        p.error(f'{chain} does not exist. Plug the camera in and run '
                'scripts/realsense_calibrate.py to write it from the device.')

    from davio.data.realsense import RealSenseCamera
    from davio.runtime.configuration import fingerprint
    from davio.runtime.engine import Engine

    cfg = yaml.safe_load(a.config.read_text())
    cfg['seed'] = a.seed
    if a.mode == 'native':
        cfg['assistance']['enabled'] = False
    if a.no_map:
        cfg['mapping']['enabled'] = False
    if a.weights:
        cfg['model'] = str(Path(a.weights).resolve())
    for item in a.set:
        key, _, raw = item.partition('=')
        parts = key.split('.')
        target = cfg
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                p.error(f'unknown config section {part!r} in {key!r}')
            target = target[part]
        if not isinstance(target, dict) or parts[-1] not in target:
            p.error(f'unknown config key {key!r}')
        target[parts[-1]] = yaml.safe_load(raw)

    camera = RealSenseCamera(serial=a.serial, stream=a.stream, width=a.width, height=a.height,
                             fps=a.fps, gyro_hz=a.gyro_hz, accel_hz=a.accel_hz,
                             emitter=a.emitter)
    device = camera.start()
    print(f"RealSense {device['name']} serial {device['serial']} firmware {device['firmware']}")
    print(f"  {device['stream']} {device['width']}x{device['height']}@{device['fps']} Hz, "
          f"gyro {device['gyro_hz']} Hz, accel {device['accel_hz']} Hz, "
          f"IR projector {'ON' if device['emitter'] else 'off'}")

    recorder = None
    if a.record:
        from record_realsense import Recorder
        recorder = Recorder(a.record, device, a.rig)

    meta = dict(source_hash=fingerprint(ROOT), dataset='realsense', sequence=device['serial'],
                data_root=None, rig=str(a.rig.resolve()), mode=a.mode, seed=a.seed, rate=1.,
                interval_start=None, interval_end=None, overrides=list(a.set), perturb_deg=0.,
                clock='live camera', device=device,
                performance_validation='live capture; no reference trajectory exists',
                faults=dict(camera_blackout=None, delivery_stall=None))
    engine = Engine(cfg, a.rig, a.out, meta=meta)
    stop = {'now': False}

    def interrupt(_signum, _frame):
        if stop['now']:
            raise KeyboardInterrupt
        stop['now'] = True
        camera.stop()
        print('\nstopping: draining workers, Ctrl-C again to abort', flush=True)

    previous = signal.signal(signal.SIGINT, interrupt)
    started, frames = time.monotonic(), 0
    try:
        for packet in camera.packets(seconds=a.seconds, warmup_s=a.warmup):
            engine.step(packet)
            if recorder is not None:
                recorder.write(packet)
            frames += 1
            if frames % (a.fps * 2) == 0:
                print(f'  {frames:6d} frames  {time.monotonic() - started:6.1f} s  '
                      f'selected={engine.selected}  poses={engine.count}  '
                      f'map windows={engine.submitted} dropped={engine.dropped}', flush=True)
    except BaseException as exc:
        signal.signal(signal.SIGINT, previous)
        camera.close()
        if recorder is not None:
            recorder.close()
        engine.close(error=f'{type(exc).__name__}: {exc}')
        raise
    signal.signal(signal.SIGINT, previous)
    camera.close()
    if recorder is not None:
        recorder.close()
    engine.close()
    meta = json.loads((a.out / 'run.json').read_text())
    print(f"\n{a.out}: {meta['state']}, {meta['n_states']} poses, "
          f"selected={meta['selected']}, dense map={'yes' if meta['integrated_mapping_success'] else 'no'}")
    print(f'View it with:  python3 scripts/view_run.py --run {a.out}')
    return 0 if engine.status == 'completed' else 1


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f'DAVIO stopped: {exc}', file=sys.stderr)
        raise SystemExit(2)
