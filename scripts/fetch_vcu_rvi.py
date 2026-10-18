#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'config/vcu_rvi/downloads.json'
RESERVE = 4 * 1024**3


def gdown(file_id, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        print(f'  have {destination.relative_to(ROOT)}')
        return True
    code = subprocess.call([sys.executable, '-m', 'gdown', file_id, '-O', str(destination)])
    if code != 0 or not destination.is_file():
        print(f'  FAILED {file_id} -> {destination.name}', file=sys.stderr)
        return False
    return True


def main(argv=None):
    plan = json.loads(PLAN.read_text())
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sequences', nargs='*', default=[], metavar='SEQ')
    p.add_argument('--data', type=Path, default=ROOT / 'data/vcu_rvi')
    p.add_argument('--list', action='store_true', help='Show what is available and exit')
    p.add_argument('--convert', action='store_true',
                   help='Run scripts/convert_vcu_rvi.py after each bag, then delete the bag')
    p.add_argument('--keep-bag', action='store_true', help='With --convert, keep the bag')
    a = p.parse_args(argv)

    if a.list or not a.sequences:
        print(f"source: {plan['source_folder']}\nnote:   {plan['note']}\n")
        for name, entry in plan['sequences'].items():
            size = entry.get('bag_bytes')
            print(f"  {name:11s} {entry['page_name']:24s}"
                  + (f'  ~{size/1e9:.1f} GB' if size else ''))
        return 0

    unknown = [s for s in a.sequences if s not in plan['sequences']]
    if unknown:
        p.error(f'unknown sequence(s) {unknown}; --list shows the names')

    calib = a.data / '_calib'
    for name, file_id in plan['calibration'].items():
        gdown(file_id, calib / name)

    failed = []
    for name in a.sequences:
        entry = plan['sequences'][name]
        target = a.data / name
        print(f'\n=== {name} ({entry["page_name"]})')
        size = entry.get('bag_bytes') or 0
        free = shutil.disk_usage(a.data.parent if a.data.exists() else ROOT).free
        if size and free < size + RESERVE:
            print(f'  SKIP: needs {(size + RESERVE)/1e9:.1f} GB free, have {free/1e9:.1f} GB',
                  file=sys.stderr)
            failed.append(name)
            continue
        if not gdown(entry['groundtruth'], target / f'{name}_gt.csv'):
            failed.append(name)
            continue
        bag = target / f'{name}.bag'
        if not gdown(entry['bag'], bag):
            failed.append(name)
            continue
        if a.convert:
            # The converter needs rosbags and OpenCV, which live in the image; the host
            # generally has neither, so dispatch there unless this IS the image.
            argv = ['python3', 'scripts/convert_vcu_rvi.py', '--bag',
                    str(bag.relative_to(ROOT) if bag.is_relative_to(ROOT) else bag)]
            if not (Path('/.dockerenv').exists() or os.environ.get('DAVIO_IN_CONTAINER') == '1'):
                argv = [str(ROOT / 'scripts/docker.sh')] + argv
            code = subprocess.call(argv, cwd=ROOT)
            if code != 0:
                failed.append(name)
                continue
            if not a.keep_bag:
                bag.unlink()
                print(f'  removed {bag.name} (converted)')
    if failed:
        print(f'\nnot completed: {failed}', file=sys.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
