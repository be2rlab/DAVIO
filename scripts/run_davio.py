#!/usr/bin/env python3
"""One online run: OpenVINS, optional DAVIO calibration assistance, async metric mapping."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'config/system.yaml')
    p.add_argument('--dataset', required=True,
                   choices=('euroc', 'ori', 'custom', 'tumvi', 'vcu_rvi', 'realsense', 'phone'))
    p.add_argument('--data', type=Path, required=True, help='Parent of sequence directories')
    p.add_argument('--sequence', required=True)
    p.add_argument('--rig', type=Path, help='Override the dataset OpenVINS calibration/config')
    p.add_argument('--mode', choices=('assist', 'native'), default='assist')
    p.add_argument('--no-map', action='store_true')
    p.add_argument('--weights', help='Local DA3 checkpoint directory (recommended for transfer)')
    p.add_argument('--rate', type=float, default=1., help='Pacing factor; 1=sensor rate, 0=unpaced')
    p.add_argument('--seconds', type=float)
    p.add_argument('--start-offset', type=float, default=0.,
                   help='Seconds after the sensor origin; every invocation resets all state')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--perturb-deg', type=float, default=0.,
                   help='Rotate the SUPPLIED camera-IMU rotation prior by this angle about a '
                        'seeded random axis before the run; all arms must share the same value')
    p.add_argument('--set', action='append', default=[], metavar='a.b=VALUE',
                   help='Override one config key, YAML-parsed. Repeatable; this is how '
                        'ablation arms are declared, never by editing the frozen config')
    p.add_argument('--camera-blackout', metavar='OFFSET:DURATION',
                   help='Fault injection: withhold camera frames for DURATION s starting OFFSET s '
                        'after the interval start; IMU keeps flowing')
    p.add_argument('--delivery-stall', metavar='OFFSET:DURATION',
                   help='Fault injection: deliver nothing for DURATION s, then the backlog in one '
                        'burst; paced replay only')
    p.add_argument('--out', type=Path, required=True, help='New run directory, never overwritten')
    a = p.parse_args(argv)

    def window(text, name):
        if text is None:
            return None
        try:
            offset, duration = (float(x) for x in text.split(':'))
        except ValueError:
            p.error(f'{name} must be OFFSET:DURATION in seconds')
        if offset < 0 or duration <= 0:
            p.error(f'{name} needs a nonnegative offset and positive duration')
        return offset, duration
    blackout = window(a.camera_blackout, '--camera-blackout')
    stall = window(a.delivery_stall, '--delivery-stall')
    if stall is not None and a.rate <= 0:
        p.error('--delivery-stall only has meaning for paced replay (rate > 0)')
    if a.rate < 0 or a.start_offset < 0 or (a.seconds is not None and a.seconds <= 0):
        p.error('rate must be nonnegative and seconds positive')
    rig_dir = a.out.with_name(a.out.name + '.rig')
    if a.out.exists() or (a.perturb_deg and rig_dir.exists()):
        p.error('--out must not exist')

    from davio.data import open_dataset
    from davio.data.replay import interval, replay
    from davio.runtime.configuration import fingerprint, perturb_rig
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
        # Overrides may only retarget an existing key, so a typo is an error rather
        # than a silently inert setting that the recorded config would still show.
        if not isinstance(target, dict) or parts[-1] not in target:
            p.error(f'unknown config key {key!r}')
        target[parts[-1]] = yaml.safe_load(raw)

    ds = open_dataset(a.dataset, a.data, seq=a.sequence)
    start, end = interval(ds, a.start_offset, a.seconds)
    rig = a.rig or ROOT / cfg['openvins']['configs'][a.dataset]
    if a.perturb_deg:
        # materialize() copies whatever this points at into the run's own config dir,
        # so the run stays self-describing wherever the perturbed rig was written.
        rig = perturb_rig(rig, rig_dir, a.perturb_deg, a.seed)
    meta = dict(source_hash=fingerprint(ROOT), dataset=a.dataset, sequence=a.sequence,
                data_root=str(a.data.resolve()), rig=str(Path(rig).resolve()), mode=a.mode,
                seed=a.seed, rate=a.rate, interval_start=start, interval_end=end,
                overrides=list(a.set), perturb_deg=a.perturb_deg,
                clock='paced dataset replay' if a.rate > 0 else 'unpaced dataset replay',
                performance_validation='not established by source preparation',
                faults=dict(camera_blackout=blackout, delivery_stall=stall))
    engine = Engine(cfg, rig, a.out, meta=meta)
    try:
        replay(engine, ds, start, end, a.rate, blackout=blackout, stall=stall)
    except BaseException as exc:
        engine.close(error=f'{type(exc).__name__}: {exc}')
        raise
    engine.close()
    return 0 if engine.status == 'completed' else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f'DAVIO stopped: {exc}', file=sys.stderr)
        raise SystemExit(2)
