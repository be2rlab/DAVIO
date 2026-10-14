#!/usr/bin/env python3
"""Fuse the submaps of a run into one point cloud, each image's scale corrected by its agreement with the other windows that see the same surface (the corrections go beside it, as <name>.correction.json)."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from davio.mapper import revisit, sim3                     # noqa: E402
from davio.mapper.fusion import fuse_map, write_ply       # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, help='the .ply to write (default RUN/map.ply)')
    p.add_argument('--no-correction', action='store_true',
                   help='the map as fused, without the scale per image from the agreement between windows')
    a = p.parse_args()
    out = a.out or a.run / 'map.ply'
    if out.exists():
        p.error(f'{out} exists')
    cfg = json.loads((a.run / 'run.json').read_text())['config']['mapping']
    index = json.loads((a.run / 'map/map_index.json').read_text())
    submaps, nodes = [], {}
    for key, entry in index['submaps'].items():
        with np.load(a.run / 'map' / entry['file'], allow_pickle=False) as z:
            sm = {k: z[k].copy() for k in z.files}
        sm.update(id=key, frame_ids=entry['frame_ids'])
        submaps.append(sm)
        nodes[key] = np.asarray(entry['T_map_submap'])
    if not a.no_correction:
        placed = [nodes[sm['id']] for sm in submaps]
        applied, info = revisit.corrections(submaps, [sim3.pose(T) for T in placed], [sim3.scale(T) for T in placed], cfg)
        for sm, c in zip(submaps, applied):
            sm['depth'] = sm['depth'] * np.exp(-c)[:, None, None].astype(sm['depth'].dtype)
        out.with_suffix('.correction.json').write_text(json.dumps(dict(
            info, log_corrections={sm['id']: c.tolist() for sm, c in zip(submaps, applied)}), indent=1) + '\n')
        print(json.dumps(info))
    xyz, rgb = fuse_map(submaps, nodes, voxel_size=cfg['voxel_m'], pixel_step=cfg['pixel_step'])
    write_ply(out, xyz, rgb)


if __name__ == '__main__':
    main()
