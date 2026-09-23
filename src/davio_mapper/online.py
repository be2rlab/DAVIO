"""Pose-graph back-end over OpenVINS odometry. Corrections never feed the VIO filter."""
from collections import OrderedDict
import json
import shutil
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from . import sim3
from .metric import initialize_metric_submap
from .graph import Factor, Graph, PointFactor
from . import admission
from .features import Features
from .filtering import filter_depth
from .frame_scale import refine_frame_scales
from .mapping import fuse_map, world_points, write_ply


def resample_colors(colors, shape):
    """Rectified colour frames onto DA3's depth grid, which is a different resolution."""
    import cv2
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
    depth = np.asarray(prediction.depth)
    rgb = (resample_colors(colors, depth.shape[1:]) if colors is not None
           else np.asarray(prediction.processed_images))
    extrinsics = np.asarray(prediction.extrinsics, float)
    intrinsics = np.asarray(prediction.intrinsics, float)
    n = len(depth)
    if depth.ndim != 3 or rgb.shape != (*depth.shape, 3):
        raise ValueError('DA3 must return processed RGB on the depth grid')
    if extrinsics.shape == (n, 3, 4):
        full = np.repeat(np.eye(4)[None], n, axis=0)
        full[:, :3] = extrinsics
        extrinsics = full
    if extrinsics.shape != (n, 4, 4) or intrinsics.shape != (n, 3, 3):
        raise ValueError('Invalid DA3 camera arrays')
    if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all():
        raise ValueError('Non-finite DA3 geometry')
    poses = np.linalg.inv(extrinsics)
    center = n // 2
    poses = np.linalg.inv(poses[center]) @ poses
    for pose in poses:
        sim3.validate(pose)
    result = dict(poses=poses, depth=depth.copy(), rgb=np.clip(rgb, 0, 255).astype(np.uint8),
                  intrinsics=intrinsics, center=center)
    for name in ('conf', 'sky'):
        value = getattr(prediction, name, None)
        if value is not None:
            if np.asarray(value).shape != depth.shape:
                raise ValueError('DA3 confidence/sky grid mismatch')
            result[name] = np.asarray(value)
    return result


class OnlineMapper:

    def __init__(self, settings, output):
        if settings.get("adjacent_factor", "point") not in ("point", "pose"):
            raise ValueError("adjacent_factor must be point or pose")
        if settings.get('graph_nodes', 'sim3') not in ('sim3', 'gravity', 'scale'):
            raise ValueError('graph_nodes must be sim3, gravity or scale')
        if settings.get('graph_nodes', 'sim3') == 'gravity' and not settings.get('scale_coupling', False):
            raise ValueError('gravity node charts carry a depth-scale variable: enable scale_coupling')
        if settings.get('selective_admission', False) and (
                settings.get('graph_nodes') != 'gravity' or not settings.get('sparse_alignment', False)):
            raise ValueError('selective_admission is defined for graph_nodes=gravity with sparse_alignment')
        self.cfg = settings
        self.root = Path(output) / 'map'
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'submaps').mkdir(exist_ok=True)
        self.nodes = OrderedDict()       # every keyframe: odometry pose, map pose, loop kit
        self.geometry = OrderedDict()    # bounded: the dense arrays still in memory
        self.index = OrderedDict()
        self.edges = []
        self.counter = 0
        self.feature_backend = Features(settings)
        self.last_candidate_source = {}
        self.last_correspondences = None
        self.last_holdout = None
        # (start, end) sensor time of the run; committed map versions are frozen at
        # 25/50/75 % of it (the final map is the 100 % version). Set from the first map
        # task or by an offline driver; None disables snapshots.
        self.interval = None
        self.snapshots_done = set()
        self._last_anchor = None         # (kits, world poses) of the last refined window

    def features(self, sm, i=None):
        i = sm['center'] if i is None else int(i)
        depth = sm['depth'][i]
        pixels, desc = self.feature_backend.extract(sm['rgb'][i])
        kit = dict(pixels=pixels[:0], descriptors=None, depth=np.empty(0),
                   K=np.asarray(sm['intrinsics'][i], float), shape=tuple(depth.shape))
        if desc is None or not len(pixels):
            return kit
        p = np.rint(pixels).astype(int)
        inside = ((p[:, 0] >= 0) & (p[:, 0] < depth.shape[1])
                  & (p[:, 1] >= 0) & (p[:, 1] < depth.shape[0]))
        z = np.full(len(pixels), np.nan)
        rows, cols = p[inside, 1], p[inside, 0]
        z[inside] = depth[rows, cols]
        # A keypoint without valid metric depth cannot take part in a metric PnP,
        # so it is dropped here rather than filtered again at every candidate.
        good = inside & np.isfinite(z) & (z > 0)
        if 'valid' in sm:
            good[inside] &= sm['valid'][i][rows, cols]
        kit.update(pixels=pixels[good], descriptors=desc[good], depth=z[good].astype(float))
        return kit

    def loop(self, old, current, depth_tolerance=None):
        a, b = old['features'], current['features']
        self.last_correspondences = self.last_holdout = None
        pairs = self.feature_backend.match(a, b)
        if len(pairs) < self.cfg['loop_min_matches']:
            return None
        query, train = pairs.T
        uv0, uv1 = a['pixels'][query], b['pixels'][train]
        z0, z1 = a['depth'][query], b['depth'][train]
        xyz = np.column_stack((uv0, np.ones(len(z0)))) @ np.linalg.inv(a['K']).T * z0[:, None]
        ok, rvec, trans, inliers = cv2.solvePnPRansac(
            xyz, uv1, b['K'], None, iterationsCount=100,
            reprojectionError=self.cfg['loop_reprojection_px'], confidence=.999,
            flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None:
            return None
        good = inliers.ravel()
        if len(good) < self.cfg['loop_min_inliers'] or len(good) / len(xyz) < self.cfg.get('loop_min_inlier_ratio', .5):
            return None
        for uv, shape in ((uv0[good], a['shape']), (uv1[good], b['shape'])):
            tiles = np.floor(uv / np.array([shape[1], shape[0]]) * 4).astype(int)
            if len(np.unique(tiles, axis=0)) < 4:
                return None
        rvec, trans = cv2.solvePnPRefineLM(xyz[good], uv1[good], b['K'], None, rvec, trans)
        rotation = cv2.Rodrigues(rvec)[0]
        transformed = xyz[good] @ rotation.T + trans.reshape(3)
        depth_error = np.median(np.abs(transformed[:, 2] - z1[good]) / z1[good])
        tolerance = (self.cfg['loop_depth_relative_error'] if depth_tolerance is None
                     else depth_tolerance)
        if np.any(transformed[:, 2] <= 0) or depth_error > tolerance:
            return None
        # xyz carries OLD's stored scale and z1 CURRENT's, so this ratio is precisely
        # the disagreement between the two window fits -- not a depth error.
        scale_ratio = float(np.median(z1[good] / transformed[:, 2]))
        if not np.isfinite(scale_ratio) or scale_ratio <= 0:
            return None
        current_from_old = np.eye(4)
        current_from_old[:3, :3], current_from_old[:3, 3] = rotation, trans.ravel()
        measurement = np.linalg.inv(current_from_old)
        admitted, innovation = self.innovation_gate(old, current, measurement)
        if not admitted:
            return None
        # Centre cameras are the submap origins. Keep BOTH depth observations;
        # a rigid PnP pose alone loses the information needed to refine depth scale.
        xyz1 = np.column_stack((uv1, np.ones(len(z1)))) @ np.linalg.inv(b['K']).T * z1[:, None]
        consistent = np.abs(transformed[:, 2] - z1[good]) / z1[good] < tolerance
        support = good[consistent]
        if len(support) < self.cfg['loop_min_inliers']:
            return None
        selected = support[np.linspace(0, len(support)-1, min(len(support),
                           int(self.cfg.get('sparse_max_points', 80)))).astype(int)]
        # Whatever the subsample left over never reaches the optimizer, so its residual
        # afterwards measures alignment rather than how well the solver fitted itself.
        held_out = np.setdiff1d(support, selected)
        support = selected
        self.last_correspondences = (xyz[support], xyz1[support])
        self.last_holdout = (xyz[held_out], xyz1[held_out])
        # A loop's metric translation inherits the submaps' depth scale error, so the
        # constraint's own baseline is what its uncertainty has to be charged against.
        info = dict(baseline_m=float(np.linalg.norm(measurement[:3, 3])),
                    innovation_m=float(np.linalg.norm(innovation[:3])),
                    innovation_deg=float(np.degrees(np.linalg.norm(innovation[3:6]))),
                    depth_ratio=scale_ratio, inliers=int(len(good)),
                    depth_median_m=float(np.median(z1[good])),
                    # The measurement itself, so an offline scorer can compare it with a
                    # reference revisit instead of only with the odometry it argues against.
                    measurement=measurement.tolist())
        return measurement, len(good), scale_ratio, info

    def _projected_ratio(self, source, source_pose, target, target_pose):
        """Median (target depth / projected source depth) where source lands in target."""
        step = max(1, 2 * int(self.cfg['pixel_step']))
        xyz = np.concatenate([world_points(source, i, source_pose, step)[0]
                              for i in range(len(source['depth']))])
        centre = target['center']
        camera = np.linalg.inv(target_pose @ target['poses'][centre])
        points = xyz @ camera[:3, :3].T + camera[:3, 3]
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-3)]
        if not len(points):
            return None
        depth = target['depth'][centre]
        uv = np.rint((points @ np.asarray(target['intrinsics'][centre], float).T)[:, :2]
                     / points[:, 2:3]).astype(int)
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1])
                  & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0]))
        uv, z = uv[inside], points[inside, 2]
        if not len(z):
            return None
        ratio = depth[uv[:, 1], uv[:, 0]] / z
        # A wide band drops occlusions and rays that hit a different surface. It is far
        # wider than the ~10% error being measured, so it does not bias the median.
        ratio = ratio[np.isfinite(ratio) & (abs(ratio - 1.) < float(self.cfg['scale_max_relative_error']))]
        if len(ratio) < int(self.cfg['scale_overlap_min_points']):
            return None
        return float(np.median(ratio))

    def overlap_scale(self, old_key, record, sm):
        older = self.geometry.get(old_key)
        if older is None:
            return None
        old_pose = sim3.pose(self.nodes[old_key]['pose'])
        new_pose = sim3.pose(record['pose'])
        forward = self._projected_ratio(older, old_pose, sm, new_pose)
        backward = self._projected_ratio(sm, new_pose, older, old_pose)
        if not forward or not backward:
            return None
        # Whatever a surface hidden behind another does to one direction it does to the
        # other, so the geometric mean cancels it. A one-sided ratio drifts with it.
        return float(np.sqrt(forward / backward))

    def candidates(self, record):
        mode = self.cfg.get('loop_retrieval', 'spatial')
        if mode not in ('spatial', 'appearance', 'union'):
            raise ValueError('loop_retrieval must be spatial, appearance or union')
        cap = int(self.cfg['max_loop_candidates'])
        spatial = self.spatial_candidates(record) if mode != 'appearance' else []
        appearance = self.appearance_candidates(record) if mode != 'spatial' else []
        source = {}
        for key in spatial:
            source[key] = 'spatial'
        for key in appearance:
            source[key] = 'both' if key in source else 'appearance'
        ordered = list(dict.fromkeys(spatial + appearance))[:cap * (2 if mode == 'union' else 1)]
        self.last_candidate_source = {key: source[key] for key in ordered}
        return ordered

    def appearance_candidates(self, record):
        cap = int(self.cfg['max_loop_candidates'])
        budget = int(self.cfg.get('loop_retrieval_budget', 512))
        floor = int(self.cfg.get('loop_retrieval_min_mnn', 25))
        cosine = float(self.cfg.get('xfeat_mnn_min_cossim', .82))
        scored = []
        for key, node in self.nodes.items():
            if record['t'] - node['t'] < self.cfg['loop_min_separation_s']:
                continue
            if not node.get('features'):
                continue
            score = self.feature_backend.retrieval_score(node['features'], record['features'],
                                                        budget, cosine)
            if score >= floor:
                scored.append((-score, key))
        scored.sort()
        return [key for _score, key in scored[:cap]]

    def spatial_candidates(self, record):
        here = record['pose'][:3, 3]
        near = []
        for key, node in self.nodes.items():
            if record['t'] - node['t'] < self.cfg['loop_min_separation_s']:
                continue
            distance = float(np.linalg.norm(node['pose'][:3, 3] - here))
            if distance <= self.cfg['loop_search_radius_m']:
                near.append((distance, key))
        near.sort()
        return [key for _d, key in near[:int(self.cfg['max_loop_candidates'])]]

    def odometry_sigmas(self, relative):
        span = float(np.linalg.norm(relative[:3, 3]))
        return np.r_[
            np.full(3, max(self.cfg['odometry_sigma_floor_m'],
                           self.cfg['odometry_drift_m_per_m'] * span)),
            np.full(3, np.radians(max(self.cfg['odometry_sigma_floor_deg'],
                                      self.cfg['odometry_drift_deg_per_m'] * span)))]

    def innovation_gate(self, old, current, measurement):
        expected = np.linalg.inv(sim3.pose(old['pose'])) @ sim3.pose(current['pose'])
        innovation = sim3.log(np.linalg.inv(expected) @ measurement)
        arc = abs(current.get('path_m', 0.) - old.get('path_m', 0.))
        drift = max(self.cfg['odometry_sigma_floor_m'],
                    self.cfg['odometry_drift_m_per_m'] * arc)
        drift_deg = max(self.cfg['odometry_sigma_floor_deg'],
                        self.cfg['odometry_drift_deg_per_m'] * arc)
        sigmas = self.loop_sigmas(float(np.linalg.norm(measurement[:3, 3])))
        k = float(self.cfg.get('loop_max_innovation_sigma', np.inf))
        if (np.linalg.norm(innovation[3:6]) > np.radians(self.cfg['loop_max_rotation_deg'])
                or np.linalg.norm(innovation[3:6]) > k * np.hypot(sigmas[3],
                                                                  np.radians(drift_deg))):
            return False, innovation
        if (np.linalg.norm(innovation[:3]) > self.cfg['loop_max_translation_m']
                or np.linalg.norm(innovation[:3]) > k * np.hypot(sigmas[0], drift)):
            return False, innovation
        return True, innovation

    def inject_false_loop(self, record, key):
        kit = record.get('features') or {}
        if kit.get('pixels') is None or not len(kit.get('pixels', [])):
            return None, None
        rng = np.random.default_rng(int(self.cfg.get('inject_seed', 0)) * 100003 + self.counter)
        pool = [k for k, n in self.nodes.items()
                if record['t'] - n['t'] >= self.cfg['loop_min_separation_s']]
        if not pool:
            return None, None
        old_key = pool[int(rng.integers(len(pool)))]
        old = self.nodes[old_key]
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        error = np.eye(4)
        error[:3, :3] = sim3.exp(np.r_[0., 0., 0., axis * np.radians(
            float(self.cfg.get('inject_false_loop_rotation_deg', 10.))), 0.])[:3, :3]
        error[:3, 3] = direction * float(self.cfg.get('inject_false_loop_translation_m', 1.))
        expected = np.linalg.inv(sim3.pose(old['pose'])) @ sim3.pose(record['pose'])
        wrong = expected @ error
        admitted, innovation = self.innovation_gate(old, record, wrong)
        report = dict(a=old_key, b=key, admitted=bool(admitted),
                      innovation_m=float(np.linalg.norm(innovation[:3])),
                      innovation_deg=float(np.degrees(np.linalg.norm(innovation[3:6]))))
        if not admitted:
            return None, report
        if self.cfg.get('sparse_alignment', False):
            count = min(len(kit['pixels']), int(self.cfg.get('sparse_max_points', 80)))
            chosen = np.linspace(0, len(kit['pixels']) - 1, count).astype(int)
            uv, z = kit['pixels'][chosen], kit['depth'][chosen]
            points_b = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(kit['K']).T * z[:, None]
            points_a = points_b @ wrong[:3, :3].T + wrong[:3, 3]
            edge = dict(a=old_key, b=key, kind='sparse', points_a=points_a, points_b=points_b,
                        info=dict(report, injected=True), group='loop', measurement=wrong,
                        held_out=None)
        else:
            edge = dict(a=old_key, b=key, kind='loop', measurement=wrong, inliers=0,
                        info=dict(report, injected=True), sigmas=self.loop_sigmas(
                            float(np.linalg.norm(wrong[:3, 3]))))
        return edge, report

    def loop_useful(self, old, current, baseline_m):
        ratio = float(self.cfg.get('loop_min_drift_ratio', 0.))
        arc = abs(current.get('path_m', 0.) - old.get('path_m', 0.))
        drift = max(self.cfg['odometry_sigma_floor_m'], self.cfg['odometry_drift_m_per_m'] * arc)
        sigma = float(self.loop_sigmas(baseline_m)[0])
        return bool(ratio <= 0. or drift >= ratio * sigma), float(drift), sigma

    def loop_sigmas(self, baseline_m):
        relative = float(self.cfg.get('loop_scale_relative_error', 0.))
        return np.r_[np.full(3, float(np.hypot(self.cfg['loop_sigma_m'],
                                               relative * float(baseline_m)))),
                     np.full(3, np.radians(self.cfg['loop_sigma_deg']))]

    def correction(self, key):
        node = self.nodes[key]
        window = int(self.cfg.get('correction_nodes', 1))
        recent = [self.nodes[k] for k in list(self.nodes)[-window:]] if window > 1 else []
        if len(recent) < 2:
            return sim3.pose(node['pose']) @ np.linalg.inv(node['odom'])
        rotation = Rotation.from_matrix(np.array(
            [sim3.pose(n['pose'])[:3, :3] @ n['odom'][:3, :3].T for n in recent])).mean().as_matrix()
        out = np.eye(4)
        out[:3, :3] = rotation
        out[:3, 3] = np.mean([sim3.pose(n['pose'])[:3, 3] - rotation @ n['odom'][:3, 3]
                              for n in recent], axis=0)
        return out

    def snapshot(self, observation_time):
        if self.interval is None:
            return
        start, end = self.interval
        fraction = (observation_time - start) / max(end - start, 1e-9)
        for q in (.25, .5, .75):
            tag = f'q{int(round(q * 100))}'
            if fraction < q or tag in self.snapshots_done:
                continue
            self.snapshots_done.add(tag)
            folder = self.root / 'snapshots' / tag
            folder.mkdir(parents=True, exist_ok=True)
            index = json.loads((self.root / 'map_index.json').read_text())
            for entry in index['submaps'].values():
                entry['file'] = '../../' + entry['file']
            index['snapshot'] = dict(quantile=q, observation_time=observation_time,
                                     sensor_fraction=fraction, publication_wall=time.monotonic(),
                                     nodes=len(self.nodes), resident_submaps=len(self.geometry),
                                     note='committed map version as published; not reconstructed')
            (folder / 'map_index.json').write_text(json.dumps(index, indent=2, allow_nan=False))
            if (self.root / 'active.ply').is_file():
                shutil.copyfile(self.root / 'active.ply', folder / 'active.ply')

    def evict(self):
        """Bound the dense geometry; keep the pose nodes, which are what loops need."""
        while len(self.geometry) > self.cfg['max_active_submaps']:
            self.geometry.popitem(last=False)
        # ponytail: a hard node cap so a multi-hour run cannot grow without bound.
        # Raise it, or marginalize into a prior, if loops must reach beyond it.
        while len(self.nodes) > self.cfg['max_graph_nodes']:
            gone, _ = self.nodes.popitem(last=False)
            self.edges = [e for e in self.edges if gone not in (e['a'], e['b'])]
            self.geometry.pop(gone, None)

    def _graph(self, free, edges):
        coupled = bool(self.cfg['scale_coupling'])
        graph = Graph()
        keys = list(self.nodes)
        for i, key in enumerate(keys):
            graph.add_node(key, self.nodes[key]['pose'], dimensions=0 if i == 0 else free)
        for edge in edges:
            if edge['kind'] == 'sparse':
                graph.add(PointFactor(edge['a'], edge['b'], edge['points_a'], edge['points_b'],
                                      sigma=self.cfg.get('sparse_sigma_m', .10),
                                      sigma_radial=self.cfg.get('sparse_sigma_radial', 0.),
                                      sigma_lateral=self.cfg.get('sparse_sigma_lateral', 0.)))
            elif edge['kind'] == 'submap_scale':
                graph.add(Factor(edge['a'], edge['b'], 'submap_scale',
                                 log_ratio=edge['log_ratio'], huber_delta=2.,
                                 sigmas=np.array([edge.get('sigma_log', self.cfg['scale_sigma_log'])])))
            else:
                graph.add(Factor(edge['a'], edge['b'], edge['kind'], measurement=edge['measurement'],
                                 projected=True, sigmas=edge['sigmas'], huber_delta=2.))
        # Every submap keeps its own window fit unless an overlap disagrees with it.
        # Without a prior per node the relative-scale chain is an unanchored random
        # walk, and the projected pose factors leave the 7th DoF otherwise free.
        if coupled and self.cfg.get('scale_anchors', True):
            for key in keys[1:]:
                graph.add(Factor(keys[0], key, 'anchor', log_ratio=0., huber_delta=2.,
                                 sigmas=np.array([self.cfg['scale_prior_sigma_log']])))
        return graph

    def _solve(self, graph):
        info = graph.optimize(max_iterations=self.cfg['graph_iterations'])
        for key in self.nodes:
            self.nodes[key]['pose'] = graph.nodes[key]
        return info

    def optimize(self):
        coupled = bool(self.cfg['scale_coupling'])
        # sim3: 7 (or 6 without scale); gravity: 5, VIO roll/pitch retained; scale: 1, poses
        # fixed (0 when scale coupling is off too: every node fixed, the graph is inert).
        free = {'sim3': 7 if coupled else 6, 'gravity': 5, 'scale': 1 if coupled else 0}[self.cfg.get('graph_nodes', 'sim3')]
        if self.cfg.get('selective_admission', False):
            return self._optimize_selective(free)
        return self._solve(self._graph(free, self.edges))

    def _optimize_selective(self, free):
        groups = [e for e in self.edges if e['kind'] == 'sparse' and e.get('group') == 'loop']
        for e in groups:
            e.setdefault('admission', dict(state='deferred', reason='new', checks={}))

        def state(e):
            key = e.get('loop_group') or ((e['a'], e['b']) if e.get('group') == 'loop' else None)
            if key is None:
                return 'background'
            match = [g for g in groups if (g['a'], g['b']) == tuple(key)]
            return match[0]['admission']['state'] if match else 'background'

        usable = [e for e in self.edges if state(e) != 'rejected']
        stage_a = self._solve(self._graph(1, usable))
        counts, reasons = {}, {}
        for e in groups:
            st = e['admission']
            if st['state'] == 'deferred':
                new_state, reason, checks = admission.decide(e, groups, self.nodes, self.cfg)
                st.update(state=new_state, reason=reason, checks=checks, decided_at=self.counter)
            counts[st['state']] = counts.get(st['state'], 0) + 1
            reasons[st['reason'].split(' ')[0] if st['state'] != 'admitted' else 'admitted'] = \
                reasons.get(st['reason'].split(' ')[0] if st['state'] != 'admitted' else 'admitted', 0) + 1
        usable = [e for e in self.edges if state(e) != 'rejected'
                  and not (e.get('group') == 'loop' and e['admission']['state'] != 'admitted')]
        info = self._solve(self._graph(free, usable))
        info['admission'] = dict(counts=counts, reasons=reasons, stage_a_cost=stage_a.get('final_cost'))
        return info

    def add(self, task, prediction):
        """DA3 prediction -> metric submap -> graph. The metric fit lives only here."""
        started = time.monotonic()
        if self.interval is None and task.get('interval') is not None:
            self.interval = tuple(float(x) for x in task['interval'])
        sm = arrays(prediction, task.get('colors'))
        metric = np.asarray(task['camera_poses'], float)
        if metric.shape != sm['poses'].shape:
            raise ValueError('Every DA3 frame needs its own OpenVINS camera pose')
        baseline = np.max(np.linalg.norm(metric[:, :3, 3] - metric[0, :3, 3], axis=1))
        if baseline < self.cfg['min_baseline_m']:
            return dict(status='deferred', reason='insufficient metric camera translation')
        transform, observable, fit = initialize_metric_submap(sm['poses'], metric, return_info=True)
        # The offline builder permits a positive technical seed; online mapping
        # must not turn that fallback into a claimed metric depth observation.
        if not observable or not fit['positive_interior_fit']:
            return dict(status='rejected', reason='no positive metric scale fit')
        scale = sim3.scale(transform)
        predicted_centers = sm['poses'][:, :3, 3] @ transform[:3, :3].T + transform[:3, 3]
        fit_rmse = float(np.sqrt(np.mean(np.sum((predicted_centers - metric[:, :3, 3]) ** 2, axis=1))))
        if fit_rmse > self.cfg['max_alignment_rmse_m']:
            return dict(status='rejected', reason='visual/VIO submap alignment disagreement')
        conditioned = bool(task.get('conditioned', False))
        if conditioned and abs(np.log(scale)) > float(self.cfg.get('da3_conditioned_max_log_scale', .3)):
            # Pose-conditioned DA3 already aligned its output to the metric poses; a window
            # whose centre fit still needs a large rescaling did not follow its conditioning.
            return dict(status='rejected',
                        reason=f'conditioned prediction disagrees with metric poses (scale {scale:.2f})')
        sm['depth'] *= scale
        # Place every frame at OpenVINS's own camera pose. DA3's in-window relative
        # translation disagrees with the filter's by ~10% of the baseline, and the
        # filter's is already supplied here, so the visual poses fix the scale above
        # and never position dense geometry.
        sm['poses'] = np.linalg.inv(metric[sm['center']]) @ metric
        frame_scales = None
        if self.cfg.get('frame_scale_refinement', False):
            # ScaRF III-B: per-frame RELATIVE depth scales from matches with the VIO poses
            # fixed; the window's absolute scale stays the fit above. With
            # frame_scale_absolute the previous window's refined frames anchor the absolute
            # scale (their baselines to this window are several times the in-window ones).
            kits = [self.features(sm, i) for i in range(len(sm['depth']))]
            anchors = None
            absolute = bool(self.cfg.get('frame_scale_absolute', False))
            if absolute and self._last_anchor is not None:
                prev_kits, prev_world = self._last_anchor
                centre_inv = np.linalg.inv(metric[sm['center']])
                anchors = [(k, centre_inv @ T) for k, T in zip(prev_kits, prev_world)]
            scales, frame_scales = refine_frame_scales(kits, sm['poses'], self.cfg, self.feature_backend,
                                                       anchors=anchors)
            sm['depth'] *= scales[:, None, None]
            sm['frame_scales'] = scales
            frame_scales = dict(frame_scales, scales=scales.tolist())
            if absolute:
                self._last_anchor = ([self.features(sm, i) for i in range(len(sm['depth']))], [m.copy() for m in metric])
        retained = filter_depth(sm, self.cfg) if self.cfg.get('depth_filter', False) else 1.
        if retained == 0:
            return dict(status='rejected', reason='no multi-view supported depth')
        sm['frame_ids'] = [str(s) for s in task['stamps']]
        return self.ingest(sm, metric[sm['center']].copy(), float(task['times'][sm['center']]),
                           scale=scale, alignment_rmse=fit_rmse, retained=retained,
                           last_sensor_time=float(task['times'][-1]), started=started,
                           extra=dict(conditioned=conditioned, frame_scales=frame_scales))

    def shared_frame_ratio(self, older, sm):
        pairs = [(i, j) for i, fid in enumerate(older['frame_ids'])
                 for j, fid2 in enumerate(sm['frame_ids']) if fid == fid2]
        logs = []
        for i, j in pairs:
            a, b = np.asarray(older['depth'][i], float), np.asarray(sm['depth'][j], float)
            if a.shape != b.shape:
                continue
            ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
            if 'valid' in older:
                ok &= older['valid'][i]
            if 'valid' in sm:
                ok &= sm['valid'][j]
            if ok.sum() < int(self.cfg['scale_overlap_min_points']):
                continue
            logs.append(float(np.median(np.log(b[ok] / a[ok]))))
        if not logs:
            return None, None, 0
        spread = (max(logs) - min(logs)) / 2. if len(logs) > 1 else 0.
        return float(np.exp(np.mean(logs))), max(float(self.cfg['scale_sigma_log']), spread), len(logs)

    def ingest(self, sm, odom, t, *, scale=1., alignment_rmse=0., retained=1.,
               last_sensor_time=None, started=None, extra=None):
        started = time.monotonic() if started is None else started
        key = f'{self.counter:06d}'
        sm['id'] = key
        sm.setdefault('frame_ids', [f'{key}_{i}' for i in range(len(sm['depth']))])

        # The graph's chain is OpenVINS's OWN camera pose at this keyframe, not the
        # DA3 alignment: the fit residual belongs to the submap, not to the odometry.
        record = dict(odom=odom, pose=odom.copy(), t=float(t), path_m=0.,
                      features=self.features(sm))
        scales = []
        sparse = []
        if self.nodes:
            last_key = next(reversed(self.nodes))
            last = self.nodes[last_key]
            relative = np.linalg.inv(last['odom']) @ odom
            # sim3.pose(): a neighbour's scale correction must not enter the chain, which
            # is metric already. Only this submap's own dense geometry is rescaled.
            record['pose'] = sim3.pose(last['pose']) @ relative
            record['path_m'] = last['path_m'] + float(np.linalg.norm(relative[:3, 3]))
            self.edges.append(dict(a=last_key, b=key, kind='odometry', measurement=relative,
                                   sigmas=self.odometry_sigmas(relative)))
            if self.cfg.get('sparse_alignment', False) and self.cfg.get('sparse_adjacent', True):
                adjacent = self.loop(last, record, self.cfg['scale_max_relative_error'])
                if adjacent is not None:
                    measurement, inliers, _ratio, info = adjacent
                    if self.cfg.get('adjacent_factor', 'point') == 'pose':
                        # The rigid control for the point factors: the same verified
                        # matches, summarised as one SE(3) measurement instead of N 3D
                        # residuals. Projected, so it says nothing about depth scale --
                        # which is exactly the comparison worth making, since a rigid
                        # factor cannot express a depth-scale disagreement at all.
                        sparse.append(dict(a=last_key, b=key, kind='adjacent_pose',
                                           measurement=measurement, inliers=int(inliers),
                                           info=info,
                                           sigmas=self.loop_sigmas(info['baseline_m'])))
                    else:
                        pa, pb = self.last_correspondences
                        sparse.append(dict(a=last_key, b=key, kind='sparse',
                                           points_a=pa, points_b=pb,
                                           held_out=self.last_holdout, info=info))
            if self.cfg['scale_coupling']:
                # The previous submap overlaps this one heavily, and that overlap is
                # where a bad window fit becomes visible. Images both submaps contain give
                # the ratio pixel-aligned; re-projection is the fallback after queue drops.
                older = self.geometry.get(last_key)
                ratio, sigma, shared = (self.shared_frame_ratio(older, sm)
                                        if older is not None and self.cfg.get('shared_frame_scale', True)
                                        else (None, None, 0))
                if ratio is None:
                    ratio, sigma = self.overlap_scale(last_key, record, sm), float(self.cfg['scale_sigma_log'])
                if ratio is not None:
                    scales.append(dict(a=last_key, b=key, kind='submap_scale',
                                       log_ratio=-np.log(ratio), sigma_log=sigma, shared_frames=shared))
        loops = []
        deferred_loops = []
        if self.cfg['loops_enabled']:
            for old_key in self.candidates(record):
                found = self.loop(self.nodes[old_key], record)
                if found is not None:
                    measurement, count, ratio, info = found
                    info['retrieval'] = getattr(self, 'last_candidate_source', {}).get(
                        old_key, 'spatial')
                    useful, drift, sigma = self.loop_useful(self.nodes[old_key], record, info['baseline_m'])
                    info.update(drift_over_arc_m=drift, loop_sigma_m=sigma, pose_admitted=useful)
                    if not useful:
                        # Verified, but the odometry is still better than this loop: keep
                        # its depth-scale observation only.
                        deferred_loops.append(dict(info, a=old_key, b=key))
                        if self.cfg['scale_coupling']:
                            scales.append(dict(a=old_key, b=key, kind='submap_scale',
                                               log_ratio=-np.log(ratio), loop_group=(old_key, key)))
                        continue
                    if self.cfg.get('sparse_alignment', False):
                        pa, pb = self.last_correspondences
                        sparse.append(dict(a=old_key, b=key, kind='sparse', points_a=pa,
                                           points_b=pb, held_out=self.last_holdout, info=info,
                                           group='loop', measurement=measurement))
                    loops.append(dict(a=old_key, b=key, kind='loop', measurement=measurement,
                                      inliers=count, info=info,
                                      sigmas=self.loop_sigmas(info['baseline_m'])))
                    if self.cfg['scale_coupling']:
                        scales.append(dict(a=old_key, b=key, kind='submap_scale',
                                           log_ratio=-np.log(ratio), loop_group=(old_key, key)))
        injected = []
        every = int(self.cfg.get('inject_false_loop_every', 0))
        if every > 0 and self.counter >= every and self.counter % every == 0:
            edge, report = self.inject_false_loop(record, key)
            if report is not None:
                injected.append(report)
            if edge is not None:
                (sparse if edge['kind'] == 'sparse' else loops).append(edge)
        if not self.cfg.get('overlap_scale_factors', True):
            scales = []      # E4 ablation: no relative depth-scale observations at all
        self.nodes[key] = record
        self.geometry[key] = sm
        # Sparse factors replace PnP pose factors to avoid counting the same matches twice.
        if not self.cfg.get('sparse_alignment', False):
            self.edges.extend(loops)
        self.edges.extend(sparse)
        self.edges.extend(scales)
        self.evict()
        # An odometry-only chain is exactly consistent: its cost is identically zero
        # and a solve would do nothing. Optimize when a loop, or an overlapping submap's
        # scale, disagrees with it.
        solved = self.optimize() if (loops or scales or sparse) else {}

        subpath = self.root / 'submaps' / (key + '.npz')
        np.savez_compressed(subpath, **{k: v for k, v in sm.items()
                                        if k not in ('id', 'frame_ids', 'center')})
        self.index[key] = dict(file=str(subpath.relative_to(self.root)), frame_ids=sm['frame_ids'],
                               center=sm['center'], timestamp=record['t'], scale=scale,
                               T_odom_submap=odom.tolist(), alignment_rmse_m=alignment_rmse)
        for k, node in self.nodes.items():
            self.index[k]['T_map_submap'] = node['pose'].tolist()
        self.counter += 1
        if self.counter % self.cfg['export_every_submaps'] == 0 and self.geometry:
            xyz, rgb, _ = fuse_map(list(self.geometry.values()),
                                   {'s:' + k: self.nodes[k]['pose'] for k in self.geometry},
                                   voxel_size=self.cfg['voxel_m'],
                                   pixel_step=self.cfg['pixel_step'])
            temp = self.root / 'active.tmp.ply'
            write_ply(temp, xyz, rgb)
            temp.replace(self.root / 'active.ply')
        index = dict(schema_version=3,
                     # How a reader must turn these archives into points. Schema 2 and
                     # earlier were written while world_points applied the node
                     # similarity to the metric in-window baseline as well as to depth;
                     # see docs/GEOMETRY.md. The archives themselves are unchanged.
                     geometry_convention='depth_only_scale',
                     graph_nodes=self.cfg.get('graph_nodes', 'sim3'),
                     selective_admission=bool(self.cfg.get('selective_admission', False)),
                     gravity_axis='odometry world z; assumed gravity-aligned by OpenVINS '
                                  'initialization, not verified per run (relevant to gravity nodes)',
                     map_frame='corrected; related to OpenVINS odometry by T_map_odom',
                     scope='all pose nodes optimized; dense geometry bounded in memory, '
                           'archived on disk; nodes past max_graph_nodes are frozen',
                     submaps=self.index,
                     last_sensor_time=float(record['t'] if last_sensor_time is None
                                            else last_sensor_time))
        temp = self.root / 'map_index.tmp.json'
        temp.write_text(json.dumps(index, indent=2, allow_nan=False))
        temp.replace(self.root / 'map_index.json')
        self.snapshot(float(record['t'] if last_sensor_time is None else last_sensor_time))
        # The odometry->map transform, so the caller can correct its pose stream at
        # odometry rate instead of waiting for the next submap.
        # sim3.pose(): the scale correction belongs to this submap's dense geometry, not
        # to the trajectory. A Sim(3) here would rescale every published pose.
        correction = self.correction(key)
        return dict(status='mapped', submap=key, scale=scale, alignment_rmse_m=alignment_rmse,
                    **(extra or {}),
                    scale_correction=float(sim3.scale(record['pose'])),
                    accepted_loops=len(loops), sparse_edges=len(sparse),
                    feature_backend=self.feature_backend.backend,
                    nodes=len(self.nodes), edges=len(self.edges),
                    # Endpoints, not just a count: a false loop is only diagnosable if
                    # you can see which two keyframes it joined.
                    # dict(info, a=.., b=..): an injected probe's info already names its
                    # endpoints, and a keyword collision here crashed the E7 path.
                    loops=[dict(e['info'], a=e['a'], b=e['b']) for e in loops],
                    loops_deferred_by_drift=deferred_loops,
                    sparse=[dict(e.get('info', {}), a=e['a'], b=e['b']) for e in sparse],
                    resident_submaps=len(self.geometry), T_map_odom=correction.tolist(),
                    depth_retained_fraction=retained, mapping_s=time.monotonic() - started,
                    optimizer=solved, injected_loops=injected)
