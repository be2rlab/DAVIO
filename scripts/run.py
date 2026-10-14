#!/usr/bin/env python3
"""One online run on a recorded sequence: the filter at sensor rate, the map built beside it."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=('euroc', 'ori', 'custom', 'tumvi'))
    p.add_argument('--data', type=Path, required=True, help='the directory that holds the sequences')
    p.add_argument('--sequence', required=True)
    p.add_argument('--out', type=Path, required=True, help='a new run directory')
    p.add_argument('--config', type=Path, default=ROOT / 'config/davio.yaml')
    p.add_argument('--weights', help='a Depth Anything 3 checkpoint directory, in place of the configured model')
    p.add_argument('--rate', type=float, default=1., help='1: sensor rate; 0.3: slowed down; 0: as fast as it goes')
    p.add_argument('--seconds', type=float, help='stop after this much of the sequence')
    p.add_argument('--start-offset', type=float, default=0.)
    p.add_argument('--no-map', action='store_true', help='the filter alone')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--set', action='append', default=[], metavar='a.b=VALUE', help='change one setting; repeatable')
    a = p.parse_args(argv)
    if a.out.exists():
        p.error('--out must not exist')

    from davio.datasets import open_dataset
    from davio.engine import Engine
    from davio.replay import interval, replay

    cfg = yaml.safe_load(a.config.read_text())
    cfg['seed'] = a.seed
    if a.no_map:
        cfg['mapping']['enabled'] = False
    if a.weights:
        cfg['model'] = str(Path(a.weights).resolve())
    for item in a.set:
        key, _, raw = item.partition('=')
        *sections, name = key.split('.')
        target = cfg
        for section in sections:
            target = target.get(section) if isinstance(target, dict) else None
        if not isinstance(target, dict) or name not in target:
            p.error(f'unknown setting {key!r}')
        target[name] = yaml.safe_load(raw)

    ds = open_dataset(a.dataset, a.data, a.sequence)
    start, end = interval(ds, a.start_offset, a.seconds)
    meta = dict(dataset=a.dataset, sequence=a.sequence, data_root=str(a.data.resolve()), seed=a.seed, rate=a.rate,
                interval_start=start, interval_end=end, overrides=list(a.set))
    engine = Engine(cfg, ROOT / cfg['rigs'][a.dataset], a.out, meta=meta)
    try:
        replay(engine, ds, start, end, a.rate)
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
