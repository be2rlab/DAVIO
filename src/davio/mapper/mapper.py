from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import cv2
import numpy as np

from . import sim3
from .features import Features
from .filtering import filter_depth, usable
from .frame_scale import refine_frame_scales
from .graph import Graph, PointFactor, PoseFactor, ScaleFactor
from .fusion import fuse_map, world_points, write_ply
from .window import fit_window
from .triangulation import centred_pose_variance, triangulate


def resample_colors(colors, shape):
    """The camera's colour frames on the depth grid."""
    height, width = shape
    out = []
    for image in colors:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        out.append(image[:, :, :3])
    return np.stack(out).astype(np.uint8)


def arrays(prediction, colors=None):
    """A window's arrays from the depth model's prediction: depth, confidence, intrinsics, colour,
    and the predicted cameras relative to the centre image."""
    depth = np.asarray(prediction.depth)
    rgb = resample_colors(colors, depth.shape[1:]) if colors is not None else np.asarray(prediction.processed_images)
    extrinsics = np.asarray(prediction.extrinsics, float)
    intrinsics = np.asarray(prediction.intrinsics, float)
    n = len(depth)
    if depth.ndim != 3 or rgb.shape != (*depth.shape, 3):
        raise ValueError('The depth model must return RGB on the depth grid')
    if extrinsics.shape == (n, 3, 4):
        full = np.repeat(np.eye(4)[None], n, axis=0)
        full[:, :3] = extrinsics
        extrinsics = full
    if extrinsics.shape != (n, 4, 4) or intrinsics.shape != (n, 3, 3):
        raise ValueError('Invalid camera arrays from the depth model')
    if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all():
        raise ValueError('Non-finite cameras from the depth model')
    poses = np.linalg.inv(extrinsics)
    center = n // 2
    poses = np.linalg.inv(poses[center]) @ poses
    for pose in poses:
        sim3.validate(pose)
    sm = dict(poses=poses, depth=depth.copy(), rgb=np.clip(rgb, 0, 255).astype(np.uint8),
              intrinsics=intrinsics, center=center, conf=np.asarray(prediction.conf))
    if getattr(prediction, 'sky', None) is not None:
        sm['sky'] = np.asarray(prediction.sky)
    if any(sm[name].shape != depth.shape for name in ('conf', 'sky') if name in sm):
        raise ValueError('Confidence or sky on another grid than the depth')
    return sm


def robust_mean(r, w, rounds=5):
    """Weighted Huber mean: the weighted mean first, then reweighted by the distance from it."""
    mean, weights = 0., np.asarray(w, float).copy()
    for _ in range(rounds):
        mean = float(np.sum(weights * r) / np.sum(weights))
        d = np.abs(r - mean)
        scale = max(1.4826 * np.median(d), 1e-4)
        weights = w * np.minimum(1., 1.5 * scale / np.maximum(d, 1e-12))
    return mean


class Mapper:
    def __init__(self, settings, output):
        self.cfg = settings
        self.root = Path(output) / 'map'
        (self.root / 'submaps').mkdir(parents=True, exist_ok=True)
        self.nodes = OrderedDict()       # every window: its pose in the odometry and in the map, its keypoints
        self.geometry = OrderedDict()    # the dense arrays of the most recent windows
        self.index = OrderedDict()       # what map_index.json lists
        self.edges = []
        self.counter = 0
        self.matcher = Features(settings)
        self.extracted = {}              # keypoints of the window in hand, by image
        # Archives and the live map are written by one thread, in order, off the task; flush() waits.
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='davio-map-io')
        self.writes, self.export, self.export_stale = [], None, False

    # ---- one window ----

    def add(self, task, prediction):
        """Map one window: the depth model's prediction for its images, and the filter's camera
        poses of them. Returns a report whose status is mapped, deferred or rejected."""
        cfg, started = self.cfg, time.monotonic()
        self.extracted = {}
        sm = arrays(prediction, task.get('colors'))
        metric = np.asarray(task['camera_poses'], float)
        if metric.shape != sm['poses'].shape:
            raise ValueError('Every image needs its own camera pose')
        baseline = np.max(np.linalg.norm(metric[:, :3, 3] - metric[0, :3, 3], axis=1))
        if baseline < cfg['min_baseline_m']:
            return dict(status='deferred', reason='insufficient metric camera translation')
        fit = fit_window(sm['poses'], metric)
        if fit is None:
            return dict(status='rejected', reason='no positive metric scale fit')
        scale, rmse = fit
        if rmse > cfg['max_alignment_rmse_m']:
            return dict(status='rejected', reason='visual/VIO submap alignment disagreement')
        if abs(np.log(scale)) > cfg['max_log_scale']:
            # The model was given these poses: a prediction that still needs this much rescaling did not follow them.
            return dict(status='rejected', reason=f'conditioned prediction disagrees with metric poses (scale {scale:.2f})')
        sm['depth'] *= scale
        sm['poses'] = np.linalg.inv(metric[sm['center']]) @ metric      # the images stand on the filter's cameras
        kits = [self.keypoints(sm, i) for i in range(len(sm['depth']))]
        scales, info = refine_frame_scales(kits, sm['poses'], cfg, self.matcher)
        sm['depth'] *= scales[:, None, None]
        sm['frame_scales'] = scales
        sm['frame_ids'] = [str(s) for s in task['stamps']]
        retained = filter_depth(sm, cfg)
        if retained == 0:
            return dict(status='rejected', reason='no multi-view supported depth')
        report = self.insert(sm, metric[sm['center']].copy(), float(task['times'][sm['center']]), scale, rmse,
                             float(task['times'][-1]))
        return dict(report, frame_scales=dict(info, scales=scales.tolist()), depth_retained_fraction=retained,
                    mapping_s=time.monotonic() - started)

    def keypoints(self, sm, i, filtered=True):
        depth = sm['depth'][i]
        if i not in self.extracted:
            self.extracted[i] = self.matcher.extract(sm['rgb'][i])
        pixels, desc = self.extracted[i]
        kit = dict(pixels=pixels[:0], descriptors=None, depth=np.empty(0), kept=np.empty(0, bool),
                   K=np.asarray(sm['intrinsics'][i], float), shape=tuple(depth.shape))
        if desc is None or not len(pixels):
            return kit
        p = np.rint(pixels).astype(int)
        inside = (p[:, 0] >= 0) & (p[:, 0] < depth.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < depth.shape[0])
        rows, cols = p[inside, 1], p[inside, 0]
        z, conf, kept = np.full(len(pixels), np.nan), np.full(len(pixels), np.nan), np.zeros(len(pixels), bool)
        z[inside] = depth[rows, cols]
        conf[inside] = np.asarray(sm['conf'][i], float)[rows, cols]
        if 'valid' in sm:
            kept[inside] = sm['valid'][i][rows, cols]
        good = inside & np.isfinite(z) & (z > 0)
        if not filtered:
            good[inside] &= usable(depth, self.cfg, sm['sky'][i] if 'sky' in sm else None, sm['conf'][i])[rows, cols]
        elif 'valid' in sm:
            good[inside] &= sm['valid'][i][rows, cols]
        with np.errstate(invalid='ignore'):
            good &= np.isfinite(conf) & (conf > 0)
        kit.update(pixels=pixels[good], descriptors=desc[good], depth=z[good].astype(float), conf=conf[good], kept=kept[good])
        return kit

    # ---- the metric scale of the window ----

    def triangulated(self, record, views):
        cfg = self.cfg
        centre = record['odom'][:3, 3]
        near = [key for key, node in reversed(self.nodes.items())
                if cfg['tri_min_baseline_m'] <= float(np.linalg.norm(node['odom'][:3, 3] - centre)) <= cfg['tri_max_baseline_m']]
        residuals = {}
        for frame, kit, here in views:
            for key in near[:int(cfg['tri_nodes'])]:
                node = self.nodes[key]
                old = node['tri_kit']
                if old['descriptors'] is None or not len(old['pixels']):
                    continue
                pairs = np.asarray(self.matcher.match(old, kit), int).reshape(-1, 2)
                if len(pairs) < 20:
                    continue
                q, t = pairs.T
                tri = triangulate(old['pixels'][q], kit['pixels'][t], old['K'], kit['K'], node['odom'], here,
                                  float(cfg['tri_pixel_sigma']))
                ok = (tri['ok'] & (tri['parallax_deg'] >= cfg['tri_min_parallax_deg'])
                      & (tri['reprojection_px'] <= cfg['tri_max_reprojection_px'] * float(cfg['tri_pixel_sigma']))
                      & (tri['sigma_log'] <= cfg['tri_max_sigma_log']))
                z, conf = kit['depth'][t][ok], kit['conf'][t][ok]
                good = np.isfinite(z) & (z > 0) & np.isfinite(conf) & (conf > 0)
                if good.sum() < 20:
                    continue
                chosen = np.flatnonzero(ok)[good]
                # The filter's error between the two cameras, grown with the path between them.
                pose_sigma = max(cfg['tri_pose_floor_m'], cfg['odometry_drift_m_per_m'] * abs(record['path_m'] - node['path_m']))
                pose_var = centred_pose_variance(old['pixels'][q][chosen], kit['pixels'][t][chosen], old['K'], kit['K'],
                                                 node['odom'], here, pose_sigma)
                r = np.log(z[good]) - np.log(tri['depth_b'][ok][good])
                weight = 1. / (tri['sigma_log'][ok][good] ** 2 + pose_var + float(cfg['anchor_sigma0']) ** 2)
                kept = kit['kept'][t][ok][good]
                residuals.setdefault(frame, []).append((r[kept], weight[kept]))
        return residuals

    def anchor(self, sm, residuals):
        parts = [part for image in residuals.values() for part in image]
        if not any(sum(len(r) for r, _ in image) >= int(self.cfg['anchor_min_targets']) for image in residuals.values()):
            return None
        r, w = np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])
        limit = float(self.cfg['anchor_max_log'])
        applied = np.full(len(sm['depth']), np.clip(robust_mean(r, w), -limit, limit))
        sm['depth'] = sm['depth'] * np.exp(-applied)[:, None, None].astype(sm['depth'].dtype)
        sm['frame_anchor_log'] = applied
        return dict(log_offset=float(applied[0]), keypoints=int(len(r)))

    # ---- edges between windows ----

    def match_windows(self, old, current, depth_tolerance):
        cfg = self.cfg
        a, b = old['features'], current['features']
        pairs = self.matcher.match(a, b)
        if len(pairs) < cfg['loop_min_matches']:
            return None
        query, train = pairs.T
        uv0, uv1 = a['pixels'][query], b['pixels'][train]
        z0, z1 = a['depth'][query], b['depth'][train]
        xyz = np.column_stack((uv0, np.ones(len(z0)))) @ np.linalg.inv(a['K']).T * z0[:, None]
        ok, rvec, trans, inliers = cv2.solvePnPRansac(
            xyz, uv1, b['K'], None, iterationsCount=100, reprojectionError=cfg['loop_reprojection_px'],
            confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None:
            return None
        good = inliers.ravel()
        if len(good) < cfg['loop_min_inliers'] or len(good) / len(xyz) < cfg['loop_min_inlier_ratio']:
            return None
        for uv, shape in ((uv0[good], a['shape']), (uv1[good], b['shape'])):      # inliers spread over the image
            tiles = np.floor(uv / np.array([shape[1], shape[0]]) * 4).astype(int)
            if len(np.unique(tiles, axis=0)) < 4:
                return None
        rvec, trans = cv2.solvePnPRefineLM(xyz[good], uv1[good], b['K'], None, rvec, trans)
        rotation = cv2.Rodrigues(rvec)[0]
        transformed = xyz[good] @ rotation.T + trans.reshape(3)
        depth_error = np.median(np.abs(transformed[:, 2] - z1[good]) / z1[good])
        if np.any(transformed[:, 2] <= 0) or depth_error > depth_tolerance:
            return None
        ratio = float(np.median(z1[good] / transformed[:, 2]))
        if not np.isfinite(ratio) or ratio <= 0:
            return None
        current_from_old = np.eye(4)
        current_from_old[:3, :3], current_from_old[:3, 3] = rotation, trans.ravel()
        measurement = np.linalg.inv(current_from_old)
        admitted, innovation = self.agrees_with_odometry(old, current, measurement)
        if not admitted:
            return None
        xyz1 = np.column_stack((uv1, np.ones(len(z1)))) @ np.linalg.inv(b['K']).T * z1[:, None]
        support = good[np.abs(transformed[:, 2] - z1[good]) / z1[good] < depth_tolerance]
        if len(support) < cfg['loop_min_inliers']:
            return None
        support = support[np.linspace(0, len(support) - 1, min(len(support), int(cfg['sparse_max_points']))).astype(int)]
        info = dict(baseline_m=float(np.linalg.norm(measurement[:3, 3])),
                    innovation_m=float(np.linalg.norm(innovation[:3])),
                    innovation_deg=float(np.degrees(np.linalg.norm(innovation[3:6]))),
                    depth_ratio=ratio, inliers=int(len(good)), depth_median_m=float(np.median(z1[good])),
                    measurement=measurement.tolist())
        return measurement, info, (xyz[support], xyz1[support])

    def agrees_with_odometry(self, old, current, measurement):
        cfg = self.cfg
        expected = np.linalg.inv(sim3.pose(old['pose'])) @ sim3.pose(current['pose'])
        innovation = sim3.log(np.linalg.inv(expected) @ measurement)
        arc = abs(current['path_m'] - old['path_m'])
        drift = max(cfg['odometry_sigma_floor_m'], cfg['odometry_drift_m_per_m'] * arc)
        drift_deg = max(cfg['odometry_sigma_floor_deg'], cfg['odometry_drift_deg_per_m'] * arc)
        sigmas = self.loop_sigmas(float(np.linalg.norm(measurement[:3, 3])))
        k = float(cfg['loop_max_innovation_sigma'])
        rotation, translation = np.linalg.norm(innovation[3:6]), np.linalg.norm(innovation[:3])
        if rotation > np.radians(cfg['loop_max_rotation_deg']) or rotation > k * np.hypot(sigmas[3], np.radians(drift_deg)):
            return False, innovation
        if translation > cfg['loop_max_translation_m'] or translation > k * np.hypot(sigmas[0], drift):
            return False, innovation
        return True, innovation

    def loop_sigmas(self, baseline_m):
        cfg = self.cfg
        return np.r_[np.full(3, float(np.hypot(cfg['loop_sigma_m'], float(cfg['loop_scale_relative_error']) * float(baseline_m)))),
                     np.full(3, np.radians(cfg['loop_sigma_deg']))]

    def loop_useful(self, old, current, baseline_m):
        cfg = self.cfg
        drift = max(cfg['odometry_sigma_floor_m'], cfg['odometry_drift_m_per_m'] * abs(current['path_m'] - old['path_m']))
        sigma = float(self.loop_sigmas(baseline_m)[0])
        return bool(drift >= float(cfg['loop_min_drift_ratio']) * sigma), float(drift), sigma

    def loop_candidates(self, record):
        here, near = record['pose'][:3, 3], []
        for key, node in self.nodes.items():
            if record['t'] - node['t'] < self.cfg['loop_min_separation_s']:
                continue
            distance = float(np.linalg.norm(node['pose'][:3, 3] - here))
            if distance <= self.cfg['loop_search_radius_m']:
                near.append((distance, key))
        near.sort()
        return [key for _, key in near[:int(self.cfg['max_loop_candidates'])]]

    def odometry_sigmas(self, relative):
        cfg, span = self.cfg, float(np.linalg.norm(relative[:3, 3]))
        return np.r_[np.full(3, max(cfg['odometry_sigma_floor_m'], cfg['odometry_drift_m_per_m'] * span)),
                     np.full(3, np.radians(max(cfg['odometry_sigma_floor_deg'], cfg['odometry_drift_deg_per_m'] * span)))]

    def shared_frame_ratio(self, older, sm):
        logs = []
        for i, fid in enumerate(older['frame_ids']):
            for j, fid2 in enumerate(sm['frame_ids']):
                a, b = np.asarray(older['depth'][i], float), np.asarray(sm['depth'][j], float)
                if fid != fid2 or a.shape != b.shape:
                    continue
                ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0) & older['valid'][i] & sm['valid'][j]
                if ok.sum() >= int(self.cfg['scale_overlap_min_points']):
                    logs.append(float(np.median(np.log(b[ok] / a[ok]))))
        if not logs:
            return None, None
        spread = (max(logs) - min(logs)) / 2. if len(logs) > 1 else 0.
        return float(np.exp(np.mean(logs))), max(float(self.cfg['scale_sigma_log']), spread)

    def projected_ratio(self, source, source_pose, target, target_pose):
        step = max(1, 2 * int(self.cfg['pixel_step']))
        xyz = np.concatenate([world_points(source, i, source_pose, step)[0] for i in range(len(source['depth']))])
        centre = target['center']
        camera = np.linalg.inv(target_pose @ target['poses'][centre])
        points = xyz @ camera[:3, :3].T + camera[:3, 3]
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-3)]
        if not len(points):
            return None
        depth = target['depth'][centre]
        uv = np.rint((points @ np.asarray(target['intrinsics'][centre], float).T)[:, :2] / points[:, 2:3]).astype(int)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
        uv, z = uv[inside], points[inside, 2]
        if not len(z):
            return None
        ratio = depth[uv[:, 1], uv[:, 0]] / z
        ratio = ratio[np.isfinite(ratio) & (abs(ratio - 1.) < float(self.cfg['scale_max_relative_error']))]
        if len(ratio) < int(self.cfg['scale_overlap_min_points']):
            return None
        return float(np.median(ratio))

    def overlap_ratio(self, old_key, record, sm):
        older = self.geometry.get(old_key)
        if older is None:
            return None
        old_pose, new_pose = sim3.pose(self.nodes[old_key]['pose']), sim3.pose(record['pose'])
        forward = self.projected_ratio(older, old_pose, sm, new_pose)
        backward = self.projected_ratio(sm, new_pose, older, old_pose)
        if not forward or not backward:
            return None
        return float(np.sqrt(forward / backward))

    # ---- the graph ----

    def insert(self, sm, odom, t, scale, rmse, last_time):
        cfg = self.cfg
        key = f'{self.counter:06d}'
        sm['id'] = key
        centre = sm['center']
        record = dict(odom=odom, pose=odom.copy(), t=t, path_m=0.)
        if self.nodes:
            last_key = next(reversed(self.nodes))
            last = self.nodes[last_key]
            relative = np.linalg.inv(last['odom']) @ odom
            record['pose'] = sim3.pose(last['pose']) @ relative      # the chain is metric: the neighbour's scale stays out
            record['path_m'] = last['path_m'] + float(np.linalg.norm(relative[:3, 3]))
        record['tri_kit'] = self.keypoints(sm, centre, filtered=False)
        anchor = None
        if record['tri_kit']['descriptors'] is not None:
            views = [(centre, record['tri_kit'], odom)] + [(i, self.keypoints(sm, i, filtered=False), odom @ sm['poses'][i])
                                                           for i in range(len(sm['depth'])) if i != centre]
            anchor = self.anchor(sm, self.triangulated(record, views))
        record['features'] = self.keypoints(sm, centre)
        sparse, scales, loops, deferred = [], [], [], []
        if self.nodes:
            self.edges.append(dict(a=last_key, b=key, kind='odometry', measurement=relative, sigmas=self.odometry_sigmas(relative)))
            adjacent = self.match_windows(last, record, cfg['scale_max_relative_error'])
            if adjacent is not None:
                sparse.append(dict(a=last_key, b=key, kind='points', points_a=adjacent[2][0], points_b=adjacent[2][1], info=adjacent[1]))
            older = self.geometry.get(last_key)
            ratio, sigma = self.shared_frame_ratio(older, sm) if older is not None else (None, None)
            if ratio is None:
                ratio, sigma = self.overlap_ratio(last_key, record, sm), float(cfg['scale_sigma_log'])
            if ratio is not None:
                scales.append(dict(a=last_key, b=key, kind='scale', log_ratio=-np.log(ratio), sigma_log=sigma))
        if cfg['loops_enabled']:
            for old_key in self.loop_candidates(record):
                old = self.nodes[old_key]
                found = self.match_windows(old, record, cfg['loop_depth_relative_error'])
                if found is None:
                    continue
                measurement, info, points = found
                useful, drift, sigma = self.loop_useful(old, record, info['baseline_m'])
                info.update(drift_over_arc_m=drift, loop_sigma_m=sigma, pose_admitted=useful)
                if useful:
                    sparse.append(dict(a=old_key, b=key, kind='points', points_a=points[0], points_b=points[1], info=info))
                    loops.append(dict(info, a=old_key, b=key))
                else:       # the odometry is better than this loop: only its depth-scale observation is kept
                    deferred.append(dict(info, a=old_key, b=key))
                scales.append(dict(a=old_key, b=key, kind='scale', log_ratio=-np.log(info['depth_ratio']),
                                   sigma_log=float(cfg['scale_sigma_log'])))
        self.nodes[key] = record
        self.geometry[key] = sm
        self.edges.extend(sparse)
        self.edges.extend(scales)
        self.evict()
        if sparse or scales:        # the odometry chain alone is consistent: nothing to solve
            self.optimize()

        path = self.root / 'submaps' / (key + '.npz')
        archive = {k: v for k, v in sm.items() if k not in ('id', 'frame_ids', 'center')}
        self.writes = [f for f in self.writes if not f.done() or f.result() is not None]
        self.writes.append(self.writer.submit(np.savez_compressed, path, **archive))
        self.index[key] = dict(file=str(path.relative_to(self.root)), frame_ids=sm['frame_ids'], center=centre,
                               timestamp=record['t'], scale=scale, T_odom_submap=odom.tolist(), alignment_rmse_m=rmse)
        for k, node in self.nodes.items():
            self.index[k]['T_map_submap'] = node['pose'].tolist()
        self.counter += 1
        if self.export is None or self.export.done():
            if self.export is not None:
                self.export.result()        # an error of the last export surfaces here
            self.export, self.export_stale = self.writer.submit(self.write_live_map, *self.live_map_inputs()), False
        else:
            self.export_stale = True        # the live file lags; flush() brings it up to date
        temp = self.root / 'map_index.tmp.json'
        temp.write_text(json.dumps(dict(submaps=self.index, last_sensor_time=last_time), indent=2, allow_nan=False))
        temp.replace(self.root / 'map_index.json')
        # The correction of the odometry by the map, for the live pose stream. A pose, not a similarity:
        # the node's scale belongs to its depth.
        correction = sim3.pose(record['pose']) @ np.linalg.inv(odom)
        return dict(status='mapped', submap=key, scale=scale, alignment_rmse_m=rmse, frame_scale_anchor=anchor,
                    scale_correction=float(sim3.scale(record['pose'])), accepted_loops=len(loops), loops=loops,
                    loops_deferred_by_drift=deferred, sparse_edges=len(sparse), nodes=len(self.nodes),
                    edges=len(self.edges), T_map_odom=correction.tolist())

    def evict(self):
        while len(self.geometry) > self.cfg['max_active_submaps']:
            self.geometry.popitem(last=False)
        while len(self.nodes) > self.cfg['max_graph_nodes']:
            gone, _ = self.nodes.popitem(last=False)
            self.edges = [e for e in self.edges if gone not in (e['a'], e['b'])]
            self.geometry.pop(gone, None)

    def optimize(self):
        cfg = self.cfg
        graph = Graph()
        keys = list(self.nodes)
        for i, key in enumerate(keys):
            graph.add_node(key, self.nodes[key]['pose'], free=i > 0)
        for edge in self.edges:
            if edge['kind'] == 'points':
                graph.add(PointFactor(edge['a'], edge['b'], edge['points_a'], edge['points_b'], sigma=cfg['sparse_sigma_m'],
                                      sigma_radial=cfg['sparse_sigma_radial'], sigma_lateral=cfg['sparse_sigma_lateral']))
            elif edge['kind'] == 'scale':
                graph.add(ScaleFactor(edge['a'], edge['b'], edge['log_ratio'], edge['sigma_log']))
            else:
                graph.add(PoseFactor(edge['a'], edge['b'], edge['measurement'], edge['sigmas']))
        # The anchor made each window's depth metric, up to its own error; the same prior holds a window
        # the anchor could not measure.
        for key in keys[1:]:
            graph.add(ScaleFactor(keys[0], key, 0., float(cfg['anchor_graph_sigma'])))
        graph.optimize(max_iterations=cfg['graph_iterations'])
        for key in self.nodes:
            self.nodes[key]['pose'] = graph.nodes[key]

    # ---- files ----

    def live_map_inputs(self):
        return list(self.geometry.values()), {k: self.nodes[k]['pose'].copy() for k in self.geometry}

    def write_live_map(self, geometry, nodes):
        xyz, rgb = fuse_map(geometry, nodes, voxel_size=self.cfg['voxel_m'], pixel_step=self.cfg['pixel_step'])
        temp = self.root / 'active.tmp.ply'
        write_ply(temp, xyz, rgb)
        temp.replace(self.root / 'active.ply')

    def flush(self):
        """Wait until every submap archive and the live map are on disk."""
        for future in self.writes:
            future.result()
        self.writes = []
        if self.export is not None:
            self.export.result()
            self.export = None
        if self.export_stale:
            self.export_stale = False
            self.write_live_map(*self.live_map_inputs())
