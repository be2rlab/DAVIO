#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from davio.data import open_dataset                                  # noqa: E402
from davio.eval import metrics                                       # noqa: E402
from davio_mapper import sim3                                        # noqa: E402
from davio_mapper.mapping import read_ply                             # noqa: E402
from davio.eval import reference_support                             # noqa: E402
from evaluate_run import read_trajectory                             # noqa: E402

THRESHOLDS_M = (.02, .03, .05, .10)    # 0.03 added 2026-09-13 (alignment gate, ScaRF parity)


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def world_alignment(run, ds, trajectory_file, until=None):
    meta = json.loads((run / 'run.json').read_text())
    trajectory = [row for row in read_trajectory(run / trajectory_file)
                  if until is None or row[0] <= until + 1e-9]
    est_p, ref_p, _o, _t, _n = metrics.matched_points(ds, trajectory, meta['interval_start'])
    if est_p is None:
        raise ValueError('No reference support for this trajectory; cannot place the map')
    rot, translation, rmse = metrics.se3_align(est_p, ref_p)
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rot, translation.reshape(3)
    return out, float(rmse)


def legacy_arm_frustum(reference, run, transform, near, far):
    index = json.loads((run / 'map/map_index.json').read_text())['submaps']
    mask = np.zeros(len(reference), bool)
    for entry in index.values():
        archive = run / 'map' / entry['file']
        if not archive.is_file():
            raise FileNotFoundError(f'{archive} is required for the visibility mask')
        with np.load(archive) as data:
            k = np.asarray(data['intrinsics'][int(entry['center'])], float)
            height, width = data['depth'].shape[1:]
        node = transform @ sim3.pose(np.asarray(entry['T_map_submap'], float))
        camera = np.linalg.inv(node)
        points = reference[~mask] @ camera[:3, :3].T + camera[:3, 3]
        ahead = (points[:, 2] > near) & (points[:, 2] < far)
        uv = (points @ k.T)[:, :2] / np.maximum(points[:, 2:3], 1e-6)
        inside = ahead & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        mask[np.flatnonzero(~mask)[inside]] = True
    return mask


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--map', type=Path, help='Exported PLY (default RUN/map.ply)')
    p.add_argument('--trajectory', default='map_trajectory_final.tum',
                   help='Which stream fits the rigid map placement')
    p.add_argument('--near', type=float, default=.15)
    p.add_argument('--far', type=float, default=20.)
    p.add_argument('--groundtruth', default='dataset',
                   choices=('dataset', 'openvins', 'openvins_original', 'auto'),
                   help='Reference variant for the map placement alignment and the support')
    p.add_argument('--reference-voxel-m', type=float, default=.01,
                   help='Downsample the Leica cloud so density is a stated quantity')
    p.add_argument('--support', default='groundtruth',
                   choices=('groundtruth', 'legacy-arm-frustum'),
                   help="'groundtruth' is frozen per sequence and identical for every arm. "
                        "'legacy-arm-frustum' reproduces pre-2026-09-13 scores and is not "
                        "comparable across arms.")
    p.add_argument('--support-period-s', type=float, default=.1,
                   help='One support camera per this many seconds of reference trajectory. '
                        'Time-based so references of different rates stay comparable.')
    p.add_argument('--occlusion-tolerance-m', type=float, default=.03)
    p.add_argument('--support-cache', type=Path, default=ROOT / 'runs/_reference_support')
    p.add_argument('--until', type=float, help='Sensor time of a committed map snapshot: the '
                   'support uses reference cameras up to it and the alignment poses up to it')
    p.add_argument('--out', type=Path, help='Report path (default RUN/surface.json)')
    p.add_argument('--force', action='store_true', help='Replace an existing report')
    a = p.parse_args(argv)
    meta = json.loads((a.run / 'run.json').read_text())
    ds = open_dataset(meta['dataset'], a.data, seq=meta['sequence'],
                      groundtruth=a.groundtruth)
    cloud = a.map or (a.run / 'map.ply')
    reference_path = Path(ds.root) / 'pointcloud0/data.ply'
    if not reference_path.is_file():
        raise SystemExit(f'{meta["sequence"]} ships no reference surface at {reference_path}')
    out = a.out or (a.run / ('surface.json' if a.support == 'groundtruth'
                             else 'surface_legacy.json'))
    if out.exists() and not a.force:
        raise SystemExit(f'{out} exists; pass --force to replace it')

    estimate_raw = read_ply(cloud)
    transform, fit_rmse = world_alignment(a.run, ds, a.trajectory, a.until)
    estimate = estimate_raw @ transform[:3, :3].T + transform[:3, 3]
    reference_sha = sha256_file(reference_path)

    if a.support == 'groundtruth':
        # Defined once per sequence from the reference trajectory and the sensor's own
        # camera model; no quantity produced by this run enters it.
        width, height = ds.rectifier.output_resolution
        reference, crop, seen, support = reference_support.load_or_build(
            a.support_cache, ds, read_ply(reference_path), reference_sha,
            ds.rectifier.K_new, (height, width), near=a.near, far=a.far,
            period_s=a.support_period_s, voxel=a.reference_voxel_m,
            tolerance=a.occlusion_tolerance_m, until=a.until)
    else:
        reference = reference_support.voxel_downsample(
            read_ply(reference_path), a.reference_voxel_m)
        seen = legacy_arm_frustum(reference, a.run, transform, a.near, a.far)
        crop = seen
        support = dict(kind='LEGACY arm-dependent frustum; not comparable across arms',
                       reference_sha256=reference_sha, reference_points=int(len(reference)),
                       visible_points=int(seen.sum()), crop_points=int(seen.sum()))
    if not seen.any():
        raise SystemExit('No reference point is supported; check the support parameters')

    # Accuracy against the whole cropped surface: a map point sitting on a real wall we
    # excluded from the visible set is not an error. Completeness only over what the
    # camera could have seen.
    accuracy = cKDTree(reference[crop]).query(estimate)[0]
    completeness = cKDTree(estimate).query(reference[seen])[0]

    def summary(d):
        return dict(mean_m=float(d.mean()), median_m=float(np.median(d)),
                    p95_m=float(np.percentile(d, 95)))
    scores = {}
    for t in THRESHOLDS_M:
        precision = float(np.mean(accuracy <= t))
        recall = float(np.mean(completeness <= t))
        scores[f'{t:.2f}'] = dict(precision=precision, recall=recall,
                                  f1=2 * precision * recall / max(precision + recall, 1e-30))
    report = dict(
        schema_version=2, run=str(a.run), sequence=meta['sequence'],
        map_ply=str(cloud), map_sha256=sha256_file(cloud),
        map_index_geometry=json.loads((a.run / 'map/map_index.json').read_text()).get(
            'geometry_convention', 'pre-schema-3 (node similarity also scaled the metric '
            'in-window baseline; see docs/GEOMETRY.md)')
        if (a.run / 'map/map_index.json').is_file() else None,
        reference_ply=str(reference_path), reference_sha256=reference_sha,
        support=support,
        alignment=dict(source=a.trajectory, kind='rigid SE(3) from the trajectory score; '
                       'no surface ICP and no fitted scale', trajectory_rmse_m=fit_rmse,
                       until=a.until,
                       map_to_reference=transform.tolist()),
        groundtruth=ds.groundtruth_provenance() if hasattr(ds, 'groundtruth_provenance') else None,
        accuracy_support='reference points within far of a reference camera centre',
        completeness_support='reference points visible to a reference camera (occlusion '
                             'tested against the reference cloud)',
        reference_points=int(len(reference)), crop_points=int(crop.sum()),
        visible_points=int(seen.sum()), estimate_points=int(len(estimate)),
        accuracy=summary(accuracy), completeness=summary(completeness),
        chamfer_mean_m=float((accuracy.mean() + completeness.mean()) / 2),
        thresholds=scores)
    out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ('sequence', 'estimate_points', 'crop_points',
                                             'visible_points', 'accuracy', 'completeness',
                                             'chamfer_mean_m', 'thresholds')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
