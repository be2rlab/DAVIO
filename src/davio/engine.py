from collections import OrderedDict, deque
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import context
from .datasets import Camera
from .filter import Filter, materialize, read_camera
from .geometry import body_pose, camera_pose, json_safe, tum_line
from .images import depth_view
from .mapper.trajectory import write_final_trajectory
from .worker import Worker, vision_loop


def blend(start, target, fraction):
    """Between two poses: fraction 0 is ``start``, 1 is ``target``."""
    f = float(np.clip(fraction, 0., 1.))
    out = np.eye(4)
    out[:3, :3] = Slerp([0., 1.], Rotation.from_matrix(np.array([start[:3, :3], target[:3, :3]])))(f).as_matrix()
    out[:3, 3] = (1. - f) * start[:3, 3] + f * target[:3, 3]
    return out


class Engine:
    def __init__(self, cfg, rig, out, meta=None):
        self.cfg, self.out, self.meta = cfg, Path(out).resolve(), dict(meta or {})
        self.out.mkdir(parents=True, exist_ok=False)
        self.status, self.error, self.opened = 'starting', None, time.monotonic()
        self.start, self.last_t, self.last_pose, self.last_keyframe = None, -np.inf, -np.inf, -np.inf
        self.poses, self.ages = 0, []
        self.window = deque(maxlen=cfg['mapping']['window_frames'])      # keyframes: (stamp, image, camera pose, colour)
        self.pool = OrderedDict()                                        # recent keyframes, for context: stamp -> (image, pose)
        self.windows = dict(submitted=0, dropped=0, mapped=0, deferred=0, rejected=0)
        self.pending, self.loops, self.vision_failed = 0, 0, False
        # Odometry to map, as last reported by the mapper; a step beyond the jump is slid in over the ramp.
        self.correction, self.correction_from, self.correction_start = np.eye(4), np.eye(4), None
        self.events = (self.out / 'events.jsonl').open('w', buffering=1)
        self.streams = {name: (self.out / f'{name}.tum').open('w', buffering=1)
                        for name in ('trajectory', 'camera_trajectory', 'map_trajectory')}
        cam = read_camera(rig)
        fx, fy, cx, cy = cam['intrinsics']
        self.view = depth_view(Camera(K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), D=np.array(cam['distortion_coeffs']),
                                      model=cam['distortion_model'], resolution=tuple(int(x) for x in cam['resolution'])))
        self.filter = Filter(materialize(rig, self.out / 'filter_config', cfg['openvins']))
        cv2.setRNGSeed(int(cfg['seed']))
        self.vision = Worker(vision_loop, (cfg, str(self.out)), capacity=1) if cfg['mapping']['enabled'] else None
        if self.vision is not None:     # no frame is taken while the depth model loads: its first windows would be lost
            first = self.vision.ready()
            if 'error' in first:
                self.take(first)
        self.status = 'running'
        self.write_status()

    def step(self, packet):
        """One camera frame: t (s), image, the IMU samples since the last frame (the last one at or
        after t), the wall time it arrived, and optionally a colour image for the map."""
        t = float(packet['t'])
        if t <= self.last_t:
            raise ValueError('Camera times must increase')
        if not packet['imu'] or packet['imu'][-1][0] < t:
            raise ValueError('IMU must bracket the camera')
        if self.start is None:
            self.start = t
        self.last_t = t
        state = self.filter.feed(packet)
        if state is not None and float(state['t']) > self.last_pose:
            self.publish(state, packet)
        self.poll()
        if self.poses and t - self.last_pose > self.cfg['max_tracking_gap_s']:
            raise RuntimeError('Tracking gap')

    def publish(self, state, packet):
        t = float(state['t'])
        self.last_pose = t
        self.poses += 1
        if self.poses % 20 == 0:
            self.write_status()
        body, camera = body_pose(state), camera_pose(state)
        self.streams['trajectory'].write(tum_line(t, body))
        self.streams['camera_trajectory'].write(tum_line(t, camera))
        self.streams['map_trajectory'].write(tum_line(t, self.active_correction(t) @ body))
        self.ages.append(time.monotonic() - packet['arrival_wall'])
        mapping, stamp = self.cfg['mapping'], round(t * 1e9)
        if self.vision is None or self.vision_failed or stamp != round(packet['t'] * 1e9):
            return
        if t - self.last_keyframe < mapping['keyframe_period_s']:
            return
        self.last_keyframe = t
        # The depth model sees the frame the filter sees (grey); the colour frame, when there is one, paints the map.
        color = self.view(packet['color']) if packet.get('color') is not None else None
        self.window.append((stamp, self.view(packet['image']), camera, color))
        if len(self.window) < self.window.maxlen:
            return
        stamps, images, poses, colors = zip(*self.window)
        task = dict(stamps=list(stamps), times=[s * 1e-9 for s in stamps], images=list(images),
                    colors=list(colors) if all(c is not None for c in colors) else None, K=self.view.K,
                    camera_poses=np.asarray(poses), context=self.context_for(stamps, images, poses),
                    arrival_wall=packet['arrival_wall'])
        if self.vision.submit(task):
            self.windows['submitted'] += 1
            self.pending += 1
            while len(self.window) > int(mapping['overlap_frames']):
                self.window.popleft()
        else:               # the worker is busy: the window slides on by one keyframe
            self.windows['dropped'] += 1
            self.event('dropped', t=t)

    def context_for(self, stamps, images, poses):
        """Older keyframes to show the depth model beside this window, or None. The window's own
        keyframes then join the pool, and what is too old for a later window leaves it."""
        mapping = self.cfg['mapping']
        picked = context.select({s: pose for s, (_, pose) in self.pool.items()}, stamps[0], np.asarray(poses[len(poses) // 2]), mapping)
        extra = dict(stamps=picked, images=[self.pool[s][0] for s in picked],
                     camera_poses=np.asarray([self.pool[s][1] for s in picked])) if picked else None
        for stamp, image, pose in zip(stamps, images, poses):
            self.pool.setdefault(stamp, (image, np.asarray(pose)))
        while self.pool and next(iter(self.pool)) < stamps[0] - int(float(mapping['context_max_age_s']) * 1e9):
            self.pool.popitem(last=False)
        return extra

    def active_correction(self, t):
        if self.correction_start is None:
            return self.correction
        f = (t - self.correction_start) / max(float(self.cfg['mapping']['correction_ramp_s']), 1e-9)
        if f >= 1.:
            self.correction_start = None
            return self.correction
        return blend(self.correction_from, self.correction, f)

    def poll(self):
        """Take what the vision process has finished."""
        if self.vision is None:
            return
        for item in self.vision.drain():
            self.take(item)
        if not self.vision.process.is_alive():
            self.vision_failed = True

    def take(self, item):
        """One window back from the vision process: count it, and take over the map's correction."""
        self.event('window', **item)
        if 'error' in item:
            self.vision_failed = True
            return
        self.pending -= 1
        report = item['report']
        self.windows[report['status']] += 1
        self.loops += report.get('accepted_loops', 0)
        if report.get('T_map_odom') is None:
            return
        mapping, new, active = self.cfg['mapping'], np.asarray(report['T_map_odom'], float), self.active_correction(self.last_t)
        delta = np.linalg.inv(active) @ new
        jump_m = float(np.linalg.norm(delta[:3, 3]))
        jump_deg = float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1.) / 2., -1., 1.))))
        ramped = mapping['correction_ramp_s'] > 0 and (jump_m > mapping['correction_jump_m'] or jump_deg > mapping['correction_jump_deg'])
        self.correction_from, self.correction_start = (active, self.last_t) if ramped else (new, None)
        self.correction = new
        self.event('correction', t=self.last_t, submap=report.get('submap'), jump_m=jump_m, jump_deg=jump_deg, ramped=ramped)

    def event(self, kind, **fields):
        self.events.write(json.dumps(json_safe(dict(kind=kind, wall=time.monotonic(), **fields)), allow_nan=False) + '\n')

    def write_status(self):
        value = dict(self.meta, state=self.status, error=self.error, poses=self.poses, first_sensor_time=self.start,
                     last_sensor_time=self.last_t if np.isfinite(self.last_t) else None,
                     elapsed_wall_s=time.monotonic() - self.opened, windows=self.windows, loops_closed=self.loops,
                     vision_failed=self.vision_failed,
                     pose_age_p50_s=float(np.median(self.ages)) if self.ages else None,
                     pose_age_p95_s=float(np.percentile(self.ages, 95)) if self.ages else None, config=self.cfg)
        temp = self.out / 'run.tmp.json'
        temp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False))
        temp.replace(self.out / 'run.json')

    def close(self, error=None):
        if self.status in ('completed', 'failed'):
            return
        if self.vision is not None:
            # The windows already given to the worker are finished; nothing new is asked of it.
            until = time.monotonic() + self.cfg['shutdown_drain_s']
            while self.pending > 0 and not self.vision_failed and time.monotonic() < until:
                self.poll()
                time.sleep(.02)
            for item in self.vision.close():
                self.take(item)
        for stream in self.streams.values():
            stream.close()
        try:
            write_final_trajectory(self.out)
        except (OSError, ValueError) as exc:
            self.event('final_trajectory_failed', reason=str(exc))
        self.status = 'failed' if error or not self.poses else 'completed'
        self.error = error or (None if self.poses else 'no initialization')
        self.write_status()
        self.events.close()
