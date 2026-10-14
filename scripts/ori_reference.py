#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from davio.mapper.fusion import write_ply          # noqa: E402


def read_pcd(path):
    """Binary PCD with float x y z fields only (what the ORI release ships)."""
    with Path(path).open('rb') as stream:
        header = {}
        while True:
            line = stream.readline().decode('ascii', 'replace').strip()
            if not line or line.startswith('#'):
                continue
            key, _, value = line.partition(' ')
            header[key] = value
            if key == 'DATA':
                break
        if header.get('DATA') != 'binary' or header.get('FIELDS') != 'x y z' or header.get('TYPE') != 'F F F':
            raise ValueError(f'unsupported PCD layout: {header}')
        count = int(header['POINTS'])
        return np.frombuffer(stream.read(count * 12), dtype='<f4', count=count * 3).reshape(count, 3).copy()


def write_reference(sequence):
    sequence = Path(sequence)
    points = read_pcd(sequence / f'{sequence.name}_gt/cloud_gt.pcd')
    out = sequence / 'davio/pointcloud0/data.ply'
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out, points, np.full((len(points), 3), 180, np.uint8))
    return out


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        print(write_reference(arg))
