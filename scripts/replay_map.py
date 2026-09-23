#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import hashlib
import shutil
import sys
import time
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from davio_mapper.online import OnlineMapper                  # noqa: E402
from davio_mapper.filtering import filter_depth               # noqa: E402
from davio_mapper import sim3                                 # noqa: E402
from davio.runtime.geometry import json_safe, tum_line        # noqa: E402
from export_trajectory import final_trajectory                # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint(config_path):
    """Everything whose content changes a replay's numbers."""
    files = sorted((ROOT / 'src/davio_mapper').glob('*.py'))
    files += [ROOT / 'scripts/replay_map.py', Path(config_path)]
    return {str(f.resolve().relative_to(ROOT)) if f.resolve().is_relative_to(ROOT) else str(f):
            sha256_file(f) for f in files if f.is_file()}


def write_descriptor(source, out, cfg, overrides, config_path, groundtruth, index,
                     refilter=False):
    meta = json.loads((source / 'run.json').read_text())
    descriptor = dict(
        schema_version=1, kind='cached-submap replay',
        state='completed', error=None,
        produced_by='scripts/replay_map.py',
        note='Not a live run. Archived submaps were pushed through OnlineMapper.ingest; '
             'no sensor, estimator or scheduler was involved, so latency and queue '
             'behaviour from the source run do not carry over.',
        dataset=meta['dataset'], sequence=meta['sequence'], data_root=meta['data_root'],
        interval_start=meta['interval_start'], interval_end=meta['interval_end'],
        inherited_fields=['dataset', 'sequence', 'data_root', 'interval_start',
                          'interval_end'],
        source_run=dict(path=str(source), run_json_sha256=sha256_file(source / 'run.json'),
                        map_index_sha256=sha256_file(source / 'map/map_index.json'),
                        seed=meta.get('seed'), mode=meta.get('mode'),
                        rate=meta.get('rate'), source_hash=meta.get('source_hash')),
        replay=dict(mapping_config=cfg, overrides=list(overrides), refilter=bool(refilter),
                    config_file=str(config_path),
                    config_sha256=sha256_file(config_path),
                    groundtruth_variant=groundtruth),
        code_sha256=code_fingerprint(config_path),
        input_archives={k: sha256_file(source / 'map' / e['file'])
                        for k, e in sorted(index.items())})
    (out / 'run.json').write_text(json.dumps(json_safe(descriptor), indent=2, allow_nan=False))
    return descriptor


def load_submaps(source):
    index = json.loads((source / 'map/map_index.json').read_text())['submaps']
    for key, entry in sorted(index.items(), key=lambda kv: kv[1]['timestamp']):
        data = np.load(source / 'map' / entry['file'])
        sm = {k: data[k] for k in data.files}
        sm['center'] = int(entry['center'])
        sm['frame_ids'] = list(entry['frame_ids'])
        yield key, entry, sm


def replay(source, out, cfg, keep_dense=False, refilter=False):
    mapper = OnlineMapper(cfg, out)
    meta = json.loads((source / 'run.json').read_text())
    mapper.interval = (float(meta['interval_start']), float(meta['interval_end']))
    results = []
    for key, entry, sm in load_submaps(source):
        if not cfg.get('depth_filter', True):
            sm.pop('valid', None)      # the archived mask is the filter's own output
        elif refilter:
            # The archive keeps the raw (frame-scaled) depth and DA3's confidence, which is
            # everything the online filter read. Re-running it here with the replay's
            # config is exactly what the online path would have kept under that config;
            # without this, depth-filter overrides silently do nothing.
            filter_depth(sm, cfg)
        payload = mapper.ingest(sm, np.asarray(entry['T_odom_submap'], float),
                                float(entry['timestamp']), scale=float(entry['scale']),
                                alignment_rmse=float(entry.get('alignment_rmse_m', 0.)))
        payload['source_submap'] = key
        results.append(payload)
    shutil.copyfile(source / 'trajectory.tum', out / 'trajectory.tum')
    with (out / 'map_trajectory_final.tum').open('w') as stream:
        for t, pose in final_trajectory(out):
            stream.write(tum_line(t, pose))
    if not keep_dense:
        shutil.rmtree(out / 'map/submaps', ignore_errors=True)
        (out / 'map/active.ply').unlink(missing_ok=True)
    return results, mapper


def held_out_alignment(mapper, nodes):
    if mapper is None:
        return dict(reason='no mapper handle')
    before, after = [], []
    for edge in mapper.edges:
        pair = edge.get('held_out')
        if edge['kind'] != 'sparse' or pair is None or not len(pair[0]):
            continue
        pa, pb = pair
        for poses, sink in ((('T_odom_submap',), before), (('T_map_submap',), after)):
            a = np.asarray(nodes[edge['a']][poses[0]], float)
            b = np.asarray(nodes[edge['b']][poses[0]], float)
            sink.extend(np.linalg.norm(pa @ a[:3, :3].T + a[:3, 3]
                                       - pb @ b[:3, :3].T - b[:3, 3], axis=1))
    if not before:
        return dict(reason='every edge kept all of its correspondences; none held out')
    before, after = np.asarray(before), np.asarray(after)
    return dict(points=int(len(before)), edges=int(sum(
                    1 for e in mapper.edges
                    if e['kind'] == 'sparse' and e.get('held_out') is not None
                    and len(e['held_out'][0]))),
                odometry_median_m=float(np.median(before)),
                optimized_median_m=float(np.median(after)),
                odometry_p95_m=float(np.percentile(before, 95)),
                optimized_p95_m=float(np.percentile(after, 95)),
                improvement_median=float(np.median(before) - np.median(after)))


def loop_truth(ds, nodes, loops, tolerance=.05):
    gt = ds.groundtruth()
    # Camera->body from the dataset's own descriptor, not a hardcoded EuRoC constant.
    body_camera = np.eye(4)
    body_camera[:3, :3], body_camera[:3, 3] = np.asarray(ds.R_CtoI), np.asarray(ds.p_IC)

    def camera(t):
        j = int(np.argmin(np.abs(gt.t - t)))
        if abs(float(gt.t[j]) - t) > tolerance or (gt.valid is not None and not bool(gt.valid[j])):
            return None
        world = np.eye(4)
        world[:3, :3], world[:3, 3] = gt.R[j], gt.p[j]
        return world @ body_camera

    errors, angles, spans, sources = [], [], [], []
    for loop in loops:
        if 'measurement' not in loop:
            continue
        a, b = camera(nodes[loop['a']]['timestamp']), camera(nodes[loop['b']]['timestamp'])
        if a is None or b is None:
            continue
        reference = np.linalg.inv(a) @ b
        measured = np.asarray(loop['measurement'], float)
        errors.append(float(np.linalg.norm(measured[:3, 3] - reference[:3, 3])))
        cos = (np.trace(measured[:3, :3].T @ reference[:3, :3]) - 1.) / 2.
        angles.append(float(np.degrees(np.arccos(np.clip(cos, -1., 1.)))))
        spans.append(abs(nodes[loop['b']]['timestamp'] - nodes[loop['a']]['timestamp']))
        sources.append(loop.get('retrieval', 'spatial'))
    if not errors:
        return dict(loop_truth='no reference support for any accepted loop')
    errors, angles = np.asarray(errors), np.asarray(angles)
    # Which retrieval proposed each accepted loop, scored separately: an appearance
    # candidate that verification accepts is only worth having if it is also right.
    by_retrieval = {}
    for source in sorted(set(sources)):
        mask = np.asarray([x == source for x in sources])
        by_retrieval[source] = dict(
            accepted=int(mask.sum()),
            error_m_median=float(np.median(errors[mask])),
            accurate_under_0p3m=int(np.sum(errors[mask] <= .3)),
            false_over_0p5m=int(np.sum(errors[mask] > .5)))
    return dict(loop_scored=len(errors),
                loop_true_error_m_median=float(np.median(errors)),
                loop_true_error_m_p95=float(np.percentile(errors, 95)),
                loop_true_error_deg_median=float(np.median(angles)),
                loop_true_error_deg_p95=float(np.percentile(angles, 95)),
                # A loop whose measured revisit is metres from the reference one is not
                # a weak constraint, it is a wrong one.
                loop_false_fraction_over_0p5m=float(np.mean(errors > .5)),
                loop_separation_s_median=float(np.median(spans)),
                loop_separation_s_max=float(np.max(spans)),
                loop_accurate_under_0p3m=int(np.sum(errors <= .3)),
                loop_false_over_0p5m=int(np.sum(errors > .5)),
                loop_by_retrieval=by_retrieval)


def score(out, source, results, cfg, elapsed, mapper=None, groundtruth='dataset'):
    from davio.data import open_dataset
    from davio.eval import metrics
    from evaluate_run import read_trajectory
    meta = json.loads((source / 'run.json').read_text())
    ds = open_dataset(meta['dataset'], Path(meta['data_root']), seq=meta['sequence'],
                      groundtruth=groundtruth)
    start = meta['interval_start']
    unscored = None
    try:
        raw = metrics.ate(ds, read_trajectory(out / 'trajectory.tum'), start)
        final = metrics.ate(ds, read_trajectory(out / 'map_trajectory_final.tum'), start)
    except FileNotFoundError as missing:
        # A phone or live session ships no reference. The replay is exactly as valid; it
        # just has nothing to be scored against, and says so instead of dying after the work.
        unscored = str(missing)
        raw = final = dict(position_m=None, orientation_deg=None)
    mapped = [r for r in results if r['status'] == 'mapped']
    solves = [r['optimizer'] for r in mapped if r.get('optimizer')]
    nodes = json.loads((out / 'map/map_index.json').read_text())['submaps']
    shift = [float(np.linalg.norm(np.asarray(e['T_map_submap'])[:3, 3]
                                  - np.asarray(e['T_odom_submap'])[:3, 3]))
             for e in nodes.values()]
    scales = [float(sim3.scale(np.asarray(e['T_map_submap']))) for e in nodes.values()]
    loops = [l for r in mapped for l in r.get('loops', [])]
    truth = loop_truth(ds, nodes, loops) if unscored is None else dict(loop_truth=unscored)
    sparse_info = [x for r in mapped for x in r.get('sparse', []) if x]

    def spread(values, name):
        v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
        if not len(v):
            return {}
        return {name + '_median': float(np.median(v)), name + '_p95': float(np.percentile(v, 95))}

    diagnostics = dict(n_loops=len(loops), **truth)
    for field in ('baseline_m', 'innovation_m', 'innovation_deg', 'depth_ratio', 'depth_median_m'):
        diagnostics.update(spread([l.get(field) for l in loops], 'loop_' + field))
    for field in ('innovation_m', 'depth_ratio'):
        diagnostics.update(spread([x.get(field) for x in sparse_info], 'sparse_' + field))
    ratios = [l['innovation_m'] / max(l['baseline_m'], 1e-6) for l in loops
              if l.get('baseline_m') and l.get('innovation_m') is not None]
    diagnostics.update(spread(ratios, 'loop_innovation_over_baseline'))
    injections = [x for res in mapped for x in res.get('injected_loops', [])]
    groups = [e for e in (mapper.edges if mapper is not None else [])
              if e['kind'] == 'sparse' and e.get('group') == 'loop' and e.get('admission')]
    admission = None
    if groups:
        admission = dict(groups=len(groups), final_states={}, reasons={}, admitted_at=[])
        for e in groups:
            st = e['admission']
            admission['final_states'][st['state']] = admission['final_states'].get(st['state'], 0) + 1
            admission['reasons'][st['reason']] = admission['reasons'].get(st['reason'], 0) + 1
            if st['state'] == 'admitted':
                admission['admitted_at'].append(dict(a=e['a'], b=e['b'], decided_at=st.get('decided_at')))
    return dict(
        schema_version=2, kind='cached-submap replay (no DA3, no scheduling)',
        false_loop_injection=dict(
            offered=len(injections), admitted=int(sum(x['admitted'] for x in injections)),
            rotation_deg=cfg.get('inject_false_loop_rotation_deg'),
            translation_m=cfg.get('inject_false_loop_translation_m'),
            every=cfg.get('inject_false_loop_every', 0),
            admitted_innovation_m=[x['innovation_m'] for x in injections if x['admitted']])
        if injections else None,
        descriptor='run.json in this directory describes the replay, not the source run',
        loop_diagnostics=diagnostics, held_out_alignment=held_out_alignment(mapper, nodes),
        admission=admission,
        source_run=str(source), sequence=meta['sequence'], mapping_config=cfg,
        submaps_in=len(results), submaps_mapped=len(mapped),
        sparse_edges=int(sum(r['sparse_edges'] for r in mapped)),
        accepted_loops=int(sum(r['accepted_loops'] for r in mapped)),
        ate_raw_m=raw['position_m'], ate_map_final_m=final['position_m'],
        ate_raw_orientation_deg=raw['orientation_deg'],
        ate_map_final_orientation_deg=final['orientation_deg'],
        node_shift_median_m=float(np.median(shift)) if shift else None,
        node_shift_max_m=float(np.max(shift)) if shift else None,
        residual_scale_median=float(np.median(scales)) if scales else None,
        residual_scale_p05_p95=[float(np.percentile(scales, 5)),
                                float(np.percentile(scales, 95))] if scales else None,
        solver_converged=int(sum(bool(s.get('converged')) for s in solves)), solves=len(solves),
        final_cost_last=float(solves[-1]['final_cost']) if solves else None,
        wall_s=elapsed, ground_truth_used='scoring only' if unscored is None else 'none',
        unscored_reason=unscored,
        groundtruth=ds.groundtruth_provenance()
        if unscored is None and hasattr(ds, 'groundtruth_provenance') else None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Completed run with map/submaps')
    p.add_argument('--out', type=Path, required=True, help='New directory, never overwritten')
    p.add_argument('--config', type=Path, default=ROOT / 'config/system.yaml')
    p.add_argument('--set', action='append', default=[], metavar='key=VALUE',
                   help='Override one mapping key, YAML-parsed. Repeatable.')
    p.add_argument('--keep-dense', action='store_true', help='Keep re-archived submaps')
    p.add_argument('--refilter', action='store_true',
                   help='Re-run the depth filter on each archived submap with this config, '
                        'so depth_* overrides take effect (the archived mask is otherwise '
                        'used as recorded)')
    p.add_argument('--groundtruth', default='dataset',
                   choices=('dataset', 'openvins', 'openvins_original', 'auto'),
                   help='Reference variant for scoring only; the back-end never sees it')
    a = p.parse_args(argv)
    if a.out.exists():
        p.error('--out must not exist')
    cfg = yaml.safe_load(a.config.read_text())['mapping']
    for item in a.set:
        key, _, raw = item.partition('=')
        if key not in cfg:
            p.error(f'unknown mapping key {key!r}')
        cfg[key] = yaml.safe_load(raw)
    a.out.mkdir(parents=True)
    # Written first: a replay that dies still says what it was.
    source_index = json.loads((a.source / 'map/map_index.json').read_text())['submaps']
    depth_keys = [item.partition('=')[0] for item in a.set
                  if item.partition('=')[0].startswith('depth_')]
    if depth_keys and not a.refilter:
        print(f'note: {", ".join(depth_keys)} only changes the result with --refilter; '
              'the archived depth masks are used as recorded', file=sys.stderr)
    write_descriptor(a.source, a.out, cfg, a.set, a.config, a.groundtruth, source_index,
                     a.refilter)
    started = time.monotonic()
    results, mapper = replay(a.source, a.out, cfg, a.keep_dense, a.refilter)
    report = score(a.out, a.source, results, cfg, time.monotonic() - started, mapper,
                   a.groundtruth)
    (a.out / 'replay.json').write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in (
        'submaps_mapped', 'sparse_edges', 'accepted_loops', 'ate_raw_m', 'ate_map_final_m',
        'node_shift_median_m', 'node_shift_max_m', 'residual_scale_median', 'wall_s')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
