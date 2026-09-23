#!/usr/bin/env python3
"""Explicit offline export of all persisted metric submaps; no model or GT needed."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from davio_mapper.mapping import evaluate_cloud, fuse_map, read_ply, write_ply


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map', type=Path, required=True, help='Run directory/map')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--voxel-m', type=float, default=.04)
    p.add_argument('--pixel-step', type=int, default=4)
    p.add_argument('--reference-ply', type=Path,
                   help='Registered reference surface in the same metric world frame')
    p.add_argument('--threshold-m', type=float, default=.05)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output exists')
    index = json.loads((a.map / 'map_index.json').read_text())
    submaps, nodes = [], {}
    for key, entry in index['submaps'].items():
        with np.load(a.map / entry['file'], allow_pickle=False) as z:
            sm = {k: z[k].copy() for k in z.files}
        sm.update(id=key, frame_ids=entry['frame_ids'])
        submaps.append(sm)
        nodes['s:' + key] = np.asarray(entry['T_map_submap'])
    xyz, rgb, _ = fuse_map(submaps, nodes, voxel_size=a.voxel_m, pixel_step=a.pixel_step)
    write_ply(a.out, xyz, rgb)
    if a.reference_ply:
        # No alignment is fitted: an unregistered reference would score the registration.
        reference = read_ply(a.reference_ply)
        metrics = evaluate_cloud(xyz, reference, threshold=a.threshold_m)
        a.out.with_suffix('.surface.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
