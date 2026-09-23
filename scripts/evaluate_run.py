#!/usr/bin/env python3
"""Offline evaluation of one run directory. Never imported by the online runtime."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from davio.data import open_dataset                          # noqa: E402
from davio.eval import metrics                               # noqa: E402
from davio.init.jpl import quat_to_rot, rot_to_quat          # noqa: E402
from davio.runtime.geometry import json_safe                 # noqa: E402


def read_trajectory(path):
    """TUM body-in-world (Hamilton) back to the (t, p, q_GtoI) JPL form the metrics use."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        row = np.array([float(v) for v in line.split()])
        if len(row) != 8 or not np.isfinite(row).all():
            raise ValueError('Invalid trajectory row')
        out.append((row[0], row[1:4], rot_to_quat(Rotation.from_quat(row[4:]).as_matrix().T)))
    return out


def calibration_convergence(events, ref, start, checkpoints=(0., 5., 15., 30., 60.)):
    u, _s, vt = np.linalg.svd(np.asarray(ref['R_CtoI'], float))
    r_ref = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    rows = []
    for e in events:
        st = e.get('state') or {}
        if e.get('kind') != 'pose' or 'q_ItoC' not in st:
            continue
        r_est = quat_to_rot(np.asarray(st['q_ItoC'], float)).T          # R_CtoI
        row = dict(t=float(e['t']) - start,
                   rotation_deg=float(np.degrees(Rotation.from_matrix(r_ref.T @ r_est).magnitude())),
                   translation_m=float(np.linalg.norm(-r_est @ np.asarray(st['p_IinC'], float)
                                                      - np.asarray(ref['p_IC'], float))))
        if 'cam_k' in st and ref.get('K') is not None:
            k = np.asarray(st['cam_k'], float)
            K = np.asarray(ref['K'], float)
            row.update(focal_px=float(np.hypot(k[0] - K[0, 0], k[1] - K[1, 1])),
                       principal_px=float(np.hypot(k[2] - K[0, 2], k[3] - K[1, 2])))
            if ref.get('D') is not None and k.shape[0] >= 8:
                row['distortion_l2'] = float(np.linalg.norm(k[4:8] - np.asarray(ref['D'], float)[:4]))
        if 'dt' in st:
            row['time_offset_ms'] = float(abs(float(st['dt']) - float(ref.get('td', 0.))) * 1e3)
        rows.append(row)
    if not rows:
        return None
    at = {}
    for c in checkpoints:
        later = [r for r in rows if r['t'] >= c]
        if later:
            at[str(int(c))] = {k: v for k, v in later[0].items() if k != 't'}
    step = max(1, len(rows) // 200)
    return dict(at=at, final={k: v for k, v in rows[-1].items() if k != 't'},
                series=[r for i, r in enumerate(rows) if i % step == 0])


def evaluate(run, data_root, groundtruth='dataset', force=False):
    run = Path(run)
    meta = json.loads((run / 'run.json').read_text())
    ds = open_dataset(meta['dataset'], data_root, seq=meta['sequence'],
                      groundtruth=groundtruth)
    trajectory = read_trajectory(run / 'trajectory.tum')
    # The back-end's own stream, scored the same way. Both are rigidly aligned, so a
    # constant frame offset is absorbed and only the shape of the path is compared.
    corrected = read_trajectory(run / 'map_trajectory.tum')
    start, end = meta['interval_start'], meta['interval_end']
    events = [json.loads(s) for s in (run / 'events.jsonl').read_text().splitlines() if s.strip()]
    poses = [e for e in events if e['kind'] == 'pose']
    vision = [e['result'] for e in events if e['kind'] == 'vision']

    # Reference camera-IMU rotation, orthonormalized before it is used as a reference.
    u, _s, vt = np.linalg.svd(np.asarray(ds.R_CtoI, float))
    reference = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    angles = [float(np.degrees(Rotation.from_matrix(
        reference.T @ quat_to_rot(np.asarray(e['state']['q_ItoC'])).T).magnitude()))
        for e in poses if 'q_ItoC' in e.get('state', {})]

    times = np.array([row[0] for row in trajectory])
    # Output gaps are not interpolated away: no output means zero coverage.
    coverage = float(np.minimum(np.diff(times), .1).sum() / (end - start)) if len(times) > 1 else 0.
    ate = metrics.ate(ds, trajectory, start)
    position = ate['position_m']
    ages = [e['receipt_to_publish_s'] for e in poses if e.get('receipt_to_publish_s') is not None]
    mapped = [v['payload'] for v in vision
              if v.get('kind') == 'map' and v.get('payload', {}).get('status') == 'mapped']
    # How stale the map is: newest contributing frame received -> map committed.
    map_ages = [v['completed_wall'] - v['arrival_wall'] for v in vision
                if v.get('kind') == 'map' and v.get('arrival_wall') is not None]
    gpu = [v['peak_gpu_gb'] for v in vision if v.get('peak_gpu_gb') is not None]
    proposals = [v['payload'] for v in vision if v.get('kind') == 'assist']
    ref = dict(R_CtoI=ds.R_CtoI, p_IC=ds.p_IC, K=getattr(ds.camera, 'K', None),
               D=getattr(ds.camera, 'D', None), td=ds.reference_time_offset())
    candidate = json.loads((run / 'candidate.json').read_text()) if (run / 'candidate.json').exists() else None
    output = dict(
        schema_version=4, source_hash=meta['source_hash'], state=meta['state'],
        error=meta.get('error'), dataset=meta['dataset'], sequence=meta['sequence'],
        mode=meta['mode'], seed=meta['seed'], rate=meta['rate'],
        interval_start=start, interval_end=end,
        initialized=bool(trajectory), selected=meta['selected'],
        integrated_mapping_success=meta.get('integrated_mapping_success', False),
        vision_failed=meta.get('vision_failed', False),
        first_output_sensor_span_s=float(times[0] - start) if len(times) else None,
        first_output_wall_s=meta['first_publication_wall_s'],
        tracking_coverage=coverage, n_states=len(times),
        ate=ate, rpe=metrics.rpe(ds, trajectory, start),
        ate_map=metrics.ate(ds, corrected, start),
        ate_map_final=(metrics.ate(ds, read_trajectory(run/'map_trajectory_final.tum'), start)
                       if (run/'map_trajectory_final.tum').exists() else None),
        rpe_map=metrics.rpe(ds, corrected, start),
        legacy_threshold_success=position is not None and position < .5,
        completed_success=(meta['state'] == 'completed' and position is not None
                           and position < .5 and coverage >= .8),
        success_definition='completed + ATE<0.5 m + time coverage>=80%; the ATE-only '
                           'threshold is reported separately and must not be conflated',
        initial_rotation_error_deg=angles[0] if angles else None,
        final_rotation_error_deg=angles[-1] if angles else None,
        calibration_mode=(meta.get('config') or {}).get('calibration', {}).get('mode', 'supplied'),
        calibration=calibration_convergence(events, ref, start),
        candidate_attempts=sum(e['kind'] == 'candidate_started' for e in events),
        candidate_ff=(candidate or {}).get('ff'),
        bootstrap_used=bool((candidate or {}).get('bootstrap')) and meta.get('selected') == 'assisted',
        proposals=len(proposals),
        proposals_released=sum(p.get('status') == 'released' for p in proposals),
        proposals_rejected=sum(p.get('status') == 'rejected' for p in proposals),
        publication_age_p50_s=float(np.median(ages)) if ages else None,
        publication_age_p95_s=float(np.percentile(ages, 95)) if ages else None,
        map_updates=len(mapped),
        map_submap_scale_median=float(np.median([m['scale'] for m in mapped])) if mapped else None,
        map_alignment_rmse_p95_m=float(np.percentile([m['alignment_rmse_m'] for m in mapped], 95))
        if mapped else None,
        map_accepted_loops=sum(m['accepted_loops'] for m in mapped),
        map_nodes=mapped[-1]['nodes'] if mapped else 0,
        map_edges=mapped[-1]['edges'] if mapped else 0,
        # How far the back-end moved the current pose away from raw odometry. Zero
        # means the graph never disagreed with OpenVINS, not that it never ran.
        map_correction_m=float(np.linalg.norm(
            np.asarray(mapped[-1]['T_map_odom'], float)[:3, 3])) if mapped else None,
        map_completion_age_p50_s=float(np.median(map_ages)) if map_ages else None,
        map_completion_age_p95_s=float(np.percentile(map_ages, 95)) if map_ages else None,
        map_updates_per_s=len(mapped) / (end - start) if mapped else None,
        peak_gpu_gb=float(max(gpu)) if gpu else None,
        map_rejected=sum(v.get('payload', {}).get('status') in ('rejected', 'deferred')
                         for v in vision if v.get('kind') == 'map'),
        dropped_optional_tasks=meta['dropped_optional_tasks'],
        submitted_optional_tasks=meta['submitted_optional_tasks'],
        dense_surface_metrics=None,
        dense_surface_reason='requires a separately registered reference surface; '
                             'trajectory ground truth is not a surface (see scripts/export_map.py)',
        # Which reference produced every number above. A score without this is not
        # comparable with one computed against a different reference file.
        groundtruth=getattr(ds, 'groundtruth_provenance', lambda: None)())
    # A non-default reference gets its own file, and an existing score is never
    # silently replaced: archived results are evidence, not scratch space.
    name = 'evaluation.json' if groundtruth == 'dataset' else f'evaluation_{groundtruth}.json'
    path = run / name
    if path.exists() and not force:
        raise FileExistsError(f'{path} exists; pass --force to replace it')
    path.write_text(json.dumps(json_safe(output), indent=2, allow_nan=False))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--groundtruth', default='dataset',
                   choices=('dataset', 'openvins', 'openvins_original', 'auto'),
                   help="Reference variant. 'dataset' is the sequence's own file and the "
                        "default so scores stay comparable with earlier runs; 'openvins' "
                        "is the corrected file where one is distributed.")
    p.add_argument('--force', action='store_true',
                   help='Replace an existing score file for this reference variant')
    a = p.parse_args()
    print(json.dumps(evaluate(a.run, a.data, a.groundtruth, a.force)['ate'], indent=2))


if __name__ == '__main__':
    main()
