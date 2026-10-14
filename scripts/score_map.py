#!/usr/bin/env python3
"""Score a run's map against the dataset's laser scan.

The map is brought into the scan's frame by the rigid fit of the run's trajectory to the reference
trajectory: nothing is fitted to the surface, and no scale. Accuracy is the distance from each map
point to the scan around the reference cameras; completeness the distance to the map from each scan
point those cameras see (davio/eval/reference_support.py). The F-score is given at 2, 3, 5 and 10 cm.
With --fit, the map is afterwards registered to the scan by a rigid transform or a similarity: the
map's shape scored apart from where the trajectory puts it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from davio.datasets import open_dataset                              # noqa: E402
from davio.eval import metrics, reference_support, surface_fit       # noqa: E402
from davio.images import depth_view                                  # noqa: E402
from davio.mapper.fusion import read_ply                             # noqa: E402

THRESHOLDS_M = (.02, .03, .05, .10)
NEAR_M, FAR_M = .15, 20.            # the range of a reference camera
REFERENCE_VOXEL_M = .01             # the scan is thinned to one point per voxel
SUPPORT_PERIOD_S = .1               # one reference camera per this much time
OCCLUSION_TOLERANCE_M = .03


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def world_alignment(run, ds, trajectory_file):
    """Map to reference frame: the rigid fit of the run's trajectory to the reference trajectory."""
    meta = json.loads((run / 'run.json').read_text())
    est_p, ref_p, _, _, _ = metrics.matched_points(ds, metrics.read_trajectory(run / trajectory_file), meta['interval_start'])
    if est_p is None:
        raise ValueError('No reference support for this trajectory; cannot place the map')
    rot, translation, rmse = metrics.se3_align(est_p, ref_p)
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rot, translation.reshape(3)
    return out, float(rmse)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True, help='the directory that holds the sequences')
    p.add_argument('--map', type=Path, help='the exported cloud (default RUN/map.ply)')
    p.add_argument('--trajectory', default='map_trajectory_final.tum', help='the stream that places the map')
    p.add_argument('--reference', choices=('dataset', 'openvins'), help='EuRoC: its own reference file, or the corrected '
                   'one OpenVINS distributes (default)')
    p.add_argument('--fit', choices=('none', 'rigid', 'similarity'), default='none')
    p.add_argument('--support-cache', type=Path, default=ROOT / 'runs/_reference_support')
    p.add_argument('--out', type=Path, help='the report (default RUN/surface.json)')
    p.add_argument('--force', action='store_true', help='replace an existing report')
    a = p.parse_args(argv)
    meta = json.loads((a.run / 'run.json').read_text())
    ds = open_dataset(meta['dataset'], a.data, meta['sequence'], a.reference)
    cloud = a.map or (a.run / 'map.ply')
    reference_path = ds.root / 'pointcloud0/data.ply'
    if not reference_path.is_file():
        print(f'{meta["sequence"]} has no reference surface ({reference_path}); the map is not scored')
        return 0
    out = a.out or (a.run / 'surface.json')
    if out.exists() and not a.force:
        raise SystemExit(f'{out} exists; pass --force to replace it')

    transform, fit_rmse = world_alignment(a.run, ds, a.trajectory)
    estimate = read_ply(cloud) @ transform[:3, :3].T + transform[:3, 3]
    reference_sha = sha256_file(reference_path)
    view = depth_view(ds.camera)
    width, height = view.resolution
    reference, crop, seen, support = reference_support.load_or_build(
        a.support_cache, ds, read_ply(reference_path), reference_sha, view.K, (height, width), near=NEAR_M, far=FAR_M,
        period_s=SUPPORT_PERIOD_S, voxel=REFERENCE_VOXEL_M, tolerance=OCCLUSION_TOLERANCE_M)
    if not seen.any():
        raise SystemExit('No reference point is supported')

    surface_fit_info = None
    if a.fit != 'none':
        fitted, surface_fit_info = surface_fit.fit(estimate, reference[crop], with_scale=a.fit == 'similarity')
        estimate = estimate @ fitted[:3, :3].T + fitted[:3, 3]
        surface_fit_info['transform'] = fitted.tolist()

    accuracy = cKDTree(reference[crop]).query(estimate)[0]
    completeness = cKDTree(estimate).query(reference[seen])[0]

    def summary(d):
        return dict(mean_m=float(d.mean()), median_m=float(np.median(d)), p95_m=float(np.percentile(d, 95)))
    scores = {}
    for t in THRESHOLDS_M:
        precision, recall = float(np.mean(accuracy <= t)), float(np.mean(completeness <= t))
        scores[f'{t:.2f}'] = dict(precision=precision, recall=recall, f1=2 * precision * recall / max(precision + recall, 1e-30))
    report = dict(
        run=str(a.run), sequence=meta['sequence'], map_ply=str(cloud), map_sha256=sha256_file(cloud),
        reference_ply=str(reference_path), reference_sha256=reference_sha, support=support,
        alignment=dict(source=a.trajectory, fit=a.fit, trajectory_rmse_m=fit_rmse, map_to_reference=transform.tolist()),
        surface_fit=surface_fit_info, groundtruth=ds.reference_provenance(),
        reference_points=int(len(reference)), crop_points=int(crop.sum()), visible_points=int(seen.sum()),
        estimate_points=int(len(estimate)), accuracy=summary(accuracy), completeness=summary(completeness),
        chamfer_mean_m=float((accuracy.mean() + completeness.mean()) / 2), thresholds=scores)
    out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ('sequence', 'estimate_points', 'accuracy', 'completeness', 'thresholds')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
