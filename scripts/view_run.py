#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from davio.runtime.geometry import camera_pose            # noqa: E402
from davio_mapper.mapping import world_points             # noqa: E402
from davio_mapper import sim3                             # noqa: E402

# Two palettes: the bright background needs darker lines than the dark one, or the map is
# the only thing on screen you can read.
PALETTE = dict(
    dark=dict(odom=(255, 140, 0), map=(0, 190, 255), keyframe=(90, 255, 120),
              loop=(255, 70, 70), gt=(150, 150, 150)),
    light=dict(odom=(214, 95, 0), map=(0, 110, 200), keyframe=(20, 140, 60),
               loop=(200, 30, 30), gt=(95, 95, 95)))
BACKGROUNDS = ('light', 'white', 'dark')


def run_provenance(run):
    path = Path(run) / 'run.json'
    if not path.is_file():
        return 'AWAITING', 'No run.json yet: either a run about to start, or a wrong path.'
    try:
        meta = json.loads(path.read_text())
    except (ValueError, OSError):
        return 'UNREADABLE', 'run.json could not be parsed.'
    kind, state = meta.get('kind'), meta.get('state')
    if kind == 'cached-submap replay':
        return 'CACHED REPLAY', (
            'Archived submaps pushed through the back-end. No sensor, estimator or '
            'scheduler took part, so latency and queue behaviour shown here are not those '
            f'of a live run. Source: {(meta.get("source_run") or {}).get("path")}')
    if state in ('starting', 'running'):
        return 'LIVE', f'Run in progress ({meta.get("sequence")}), written as it is read.'
    return 'RECORDED', (
        f'Completed run replayed from disk ({meta.get("sequence")}, state={state}). '
        'Timing shown is the timing it was recorded with, not a measurement being taken now.')


def reference_in_run_frame(run, data):
    from davio.data import open_dataset
    from davio.eval.metrics import matched_points, se3_align
    from evaluate_run import read_trajectory
    meta = json.loads((run / 'run.json').read_text())
    trajectory_path = next((run / n for n in ('map_trajectory_final.tum', 'map_trajectory.tum',
                                              'trajectory.tum') if (run / n).is_file()), None)
    if trajectory_path is None:
        return None, 'the run has no trajectory to align against'
    try:
        ds = open_dataset(meta['dataset'], data, seq=meta['sequence'])
        gt = ds.groundtruth()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return None, f'no reference for this run: {exc}'
    trajectory = read_trajectory(trajectory_path)
    est, ref, _o, _t, _n = matched_points(ds, trajectory, meta.get('interval_start'))
    if est is None:
        return None, 'too few matched poses to align the reference'
    rot, trans, rmse = se3_align(est, ref)          # estimate -> reference; est is 3xN
    inverse_rot, inverse_trans = rot.T, -rot.T @ np.asarray(trans).reshape(3)
    start, end = meta.get('interval_start'), meta.get('interval_end')
    inside = np.ones(len(gt.t), bool)
    if start is not None:
        inside &= gt.t >= start
    if end is not None:
        inside &= gt.t <= end
    moved = gt.p[inside] @ inverse_rot.T + inverse_trans
    return (gt.t[inside], moved), f'aligned on {est.shape[1]} poses, ATE {rmse:.4f} m'


class Stream:
    """Incremental reader of one run directory: events, map index and dense archives."""

    def __init__(self, run, pixel_step, colors, frames='all', radius=.004):
        self.run, self.pixel_step = Path(run), pixel_step
        self.colors, self.frames, self.radius = colors, frames, radius
        self.offset, self.partial, self.version = 0, '', None
        self.loaded, self.missing, self.loop_edges = set(), set(), set()
        # Depth scale each submap's points were drawn at, so a loop closure that moves a
        # node's scale re-draws it instead of leaving stale geometry behind.
        self.drawn_scale = {}
        # Which camera frames already have a cloud. Overlapping submaps share frames
        # (mapping.overlap_frames), and drawing one twice just doubles the point count.
        self.drawn_frames = set()
        self.points = 0
        self.odom, self.mapped = [], []
        self.correction = np.eye(4)
        self.legacy_geometry = False
        self.index = {}

    def events(self):
        """Yield every event appended since the last call."""
        path = self.run / 'events.jsonl'
        if not path.exists():
            return
        with path.open() as f:
            f.seek(self.offset)
            chunk = f.read()
            self.offset = f.tell()
        lines = (self.partial + chunk).split('\n')
        self.partial = lines.pop()
        for line in lines:
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError:
                    continue            # a half-written line from a run still being written

    def map_changed(self):
        path = self.run / 'map/map_index.json'
        if not path.exists():
            return False
        stamp = path.stat().st_mtime_ns
        if stamp == self.version:
            return False
        try:
            self.index = json.loads(path.read_text())
        except (ValueError, OSError):
            return False                # written atomically, but read while replacing
        self.version = stamp
        self.legacy_geometry = self.index.get('geometry_convention') != 'depth_only_scale'
        return True


def log_event(rr, stream, event):
    """One event onto the timeline. Returns its sensor time, or None."""
    if event['kind'] == 'pose':
        t = float(event['t'])
        rr.set_time('sensor_time', timestamp=t)
        camera = camera_pose(event['state'])
        corrected = stream.correction @ camera
        stream.odom.append(camera[:3, 3])
        stream.mapped.append(corrected[:3, 3])
        # Incremental segments: re-logging the whole polyline each pose is quadratic.
        if len(stream.odom) > 1:
            segment = len(stream.odom) - 1
            rr.log(f'world/odometry/{segment}',
                   rr.LineStrips3D([stream.odom[-2:]], colors=[stream.colors['odom']]))
            rr.log(f'world/online_map/{segment}',
                   rr.LineStrips3D([stream.mapped[-2:]], colors=[stream.colors['map']]))
        rr.log('world/camera', rr.Transform3D(translation=corrected[:3, 3],
                                              mat3x3=corrected[:3, :3]))
        rr.log('world/camera/frustum', rr.Pinhole(focal_length=250., width=512, height=512,
                                                  image_plane_distance=.25))
        if event.get('receipt_to_publish_s') is not None:
            rr.log('plots/pose_age_s', rr.Scalars(event['receipt_to_publish_s']))
        return t
    if event['kind'] == 'vision':
        result = event['result']
        payload = result.get('payload', {})
        t = float(result.get('sensor_time', 0.) or 0.)
        rr.set_time('sensor_time', timestamp=t)
        rr.log('events/status', rr.TextLog(str(payload.get('status', result.get('error', '')))))
        stream.loop_edges.update((e['a'], e['b']) for e in payload.get('loops', []))
        # How far the worst accepted loop disagreed with the odometry chain. This is the
        # quantity that decides whether the graph helps or hurts, so it belongs on screen
        # while the run is happening.
        if payload.get('loops'):
            rr.log('plots/loop_innovation_m', rr.Scalars(
                max(float(e.get('innovation_m', 0.)) for e in payload['loops'])))
        if payload.get('T_map_odom') is not None:
            stream.correction = np.asarray(payload['T_map_odom'])
        for key in ('scale', 'scale_correction', 'alignment_rmse_m', 'accepted_loops',
                    'sparse_edges', 'mapping_s'):
            if key in payload:
                rr.log('plots/' + key, rr.Scalars(payload[key]))
        return t
    if event['kind'] == 'correction_activated':
        rr.set_time('sensor_time', timestamp=float(event['t']))
        rr.log('plots/correction_jump_m', rr.Scalars(event.get('jump_m', 0.)))
        return float(event['t'])
    return None


def submap_clouds(rr, stream, key, entry):
    archive = stream.run / 'map' / entry['file']
    try:
        with np.load(archive) as stored:
            sm = {k: stored[k] for k in stored.files}
    except (OSError, ValueError) as exc:
        stream.missing.add(key)
        return f'submap {key}: unreadable ({exc})'
    node = np.asarray(entry['T_map_submap'], float)
    # A similarity carrying the node's scale and nothing else: world_points multiplies the
    # depth by scale() and places the point with pose() @ the frame's own local pose.
    scale_only = np.eye(4) * sim3.scale(node)
    scale_only[3, 3] = 1.
    ids = entry.get('frame_ids') or []
    centre = entry['center']
    fallback = float(entry.get('timestamp', 0.))
    wanted = (range(len(sm['depth'])) if stream.frames in ('all', 'none') else [centre])
    drawn = 0
    for i in wanted:
        frame_id = ids[i] if i < len(ids) else f'{key}:{i}'
        # Frame ids are nanosecond sensor stamps; fall back to the submap's own time.
        try:
            when = int(ids[i]) * 1e-9
        except (IndexError, TypeError, ValueError):
            when = fallback
        rr.set_time('sensor_time', timestamp=when)
        if stream.frames == 'none':
            pass                        # the fused map already carries the geometry
        elif stream.frames != 'all' or frame_id not in stream.drawn_frames:
            xyz, rgb = world_points(sm, i, scale_only, stream.pixel_step,
                                    legacy_scaled_baseline=stream.legacy_geometry)
            if len(xyz):
                rr.log(f'world/submaps/{key}/cloud/{i}',
                       rr.Points3D(xyz.astype(np.float32), colors=rgb, radii=stream.radius))
                stream.drawn_frames.add(frame_id)
                stream.points += len(xyz)
                drawn += 1
        # The image views get every frame, at the rate the camera ran, not one per submap.
        rr.log('latest/rgb', rr.Image(sm['rgb'][i]))
        depth = sm['depth'][i].copy() * sim3.scale(node)
        if 'valid' in sm:
            depth[~sm['valid'][i]] = 0
        rr.log('latest/depth', rr.DepthImage(depth, meter=1.))
    stream.drawn_scale[key] = sim3.scale(node)
    stream.loaded.add(key)
    return None if drawn else f'submap {key}: no valid points'


def log_map(rr, stream):
    """Refresh every submap pose, and load the geometry of new or rescaled ones."""
    index = stream.index
    now = float(index['last_sensor_time'])
    centres = []
    for key, entry in index['submaps'].items():
        node = np.asarray(entry['T_map_submap'], float)
        centres.append(node[:3, 3])
        # Rotation and translation only; the scale lives in the points (see submap_clouds).
        placement = sim3.pose(node)
        transform = rr.Transform3D(translation=placement[:3, 3], mat3x3=placement[:3, :3])
        # A submap's points are stamped when they were SEEN, so its first pose has to be
        # stamped then too --- Rerun resolves a transform from the newest value at or
        # before the query time, and a pose logged only at `now` would leave every earlier
        # point of that submap sitting at the origin when the timeline is scrubbed back.
        if key not in stream.loaded and key not in stream.missing:
            rr.set_time('sensor_time', timestamp=float(entry.get('timestamp', now)))
            rr.log('world/submaps/' + key, transform)
        # Then again at the current time, which is how a loop closure's correction shows up
        # as the geometry moving rather than as it having always been there.
        rr.set_time('sensor_time', timestamp=now)
        rr.log('world/submaps/' + key, transform)
        if key in stream.missing:
            continue
        # A loop closure can move a node's depth scale; points drawn at the old one are
        # stale, so redraw when it has moved by more than half a percent.
        rescaled = key in stream.drawn_scale and abs(
            np.log(sim3.scale(node) / stream.drawn_scale[key])) > .005
        if key in stream.loaded and not rescaled:
            continue
        archive = stream.run / 'map' / entry['file']
        # An ablation run keeps its index and poses but may have had its dense archives
        # pruned to bound disk. Show the graph and say the geometry is gone; do not take
        # the viewer down over it.
        if not archive.is_file():
            stream.missing.add(key)
            rr.log('events/status', rr.TextLog(f'submap {key}: dense archive pruned, poses only'))
            continue
        if rescaled:
            stream.drawn_frames -= {f for f in (entry.get('frame_ids') or [])}
        problem = submap_clouds(rr, stream, key, entry)
        if problem:
            rr.log('events/status', rr.TextLog(problem))
    rr.set_time('sensor_time', timestamp=now)
    segments = [[index['submaps'][k]['T_map_submap'][i][3] for i in range(3)]
                for edge in sorted(stream.loop_edges) for k in edge if k in index['submaps']]
    if len(segments) >= 2:
        rr.log('world/loops', rr.LineStrips3D(np.asarray(segments).reshape(-1, 2, 3),
                                              colors=[stream.colors['loop']], radii=.01))
    if len(centres) > 1:
        rr.log('world/keyframes', rr.LineStrips3D([centres],
                                                  colors=[stream.colors['keyframe']], radii=.008))


def log_fused(rr, run, voxel, pixel_step, radius):
    from davio_mapper.mapping import fuse_map
    index = json.loads((Path(run) / 'map/map_index.json').read_text())
    submaps, nodes = [], {}
    for key, entry in index['submaps'].items():
        archive = Path(run) / 'map' / entry['file']
        if not archive.is_file():
            continue
        with np.load(archive, allow_pickle=False) as z:
            sm = {k: z[k].copy() for k in z.files}
        sm.update(id=key, frame_ids=entry['frame_ids'])
        submaps.append(sm)
        nodes['s:' + key] = np.asarray(entry['T_map_submap'])
    if not submaps:
        return 0
    xyz, rgb, _ = fuse_map(submaps, nodes, voxel_size=voxel, pixel_step=max(1, pixel_step))
    rr.set_time('sensor_time', timestamp=float(index['last_sensor_time']))
    rr.log('world/fused_map', rr.Points3D(xyz.astype(np.float32), colors=rgb, radii=radius))
    return len(xyz)


def check_ports(parser, *ports):
    import socket
    import subprocess
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                holder = ''
                try:
                    out = subprocess.run(['ss', '-lntp', f'sport = :{port}'],
                                         capture_output=True, text=True, timeout=5).stdout
                    users = [line for line in out.splitlines() if 'users:' in line]
                    if users:
                        holder = f'\n  held by: {users[0].split("users:")[1].strip()}'
                except (OSError, subprocess.SubprocessError):
                    pass
                parser.error(
                    f'port {port} is already in use, most likely by another viewer.{holder}\n'
                    f'  Use different ports:  --web-port {port + 100} '
                    f'--grpc-port {port + 110}\n'
                    f'  Or write a file instead:  --save tour.rrd')


def free_port(start):
    """First free localhost port at or after ``start``."""
    import socket
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f'no free port in [{start}, {start + 200})')


def open_viewer(url, web_url):
    import shutil
    import subprocess
    binary = shutil.which('rerun')
    if binary:
        port = free_port(9900)
        try:
            subprocess.Popen([binary, '--port', str(port), url],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            return f'opened the desktop viewer (its own server on port {port})'
        except OSError as exc:
            print(f'  could not start {binary}: {exc}', flush=True)
    import webbrowser
    if webbrowser.open(web_url):
        return 'opened the browser viewer'
    return ('could not open a viewer automatically; `pip install rerun-sdk==0.23.1` for the '
            'desktop app, or open the URL above by hand')


def blueprint(rr, rrb, background='light'):
    kinds = dict(light=rrb.components.BackgroundKind.GradientBright,
                 white=[250, 250, 250],
                 dark=rrb.components.BackgroundKind.GradientDark)
    return rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin='world', name='Metric map and trajectory',
                          background=kinds[background]),
        rrb.Vertical(
            rrb.Horizontal(rrb.Spatial2DView(origin='latest/rgb', name='RGB'),
                           rrb.Spatial2DView(origin='latest/depth', name='Metric depth')),
            rrb.TimeSeriesView(origin='plots', name='Runtime and alignment'),
            rrb.TextDocumentView(origin='provenance', name='What is being shown'),
            row_shares=[3, 3, 2]),
        column_shares=[3, 2]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--follow', action='store_true',
                   help='Keep watching for new events; the default for a run in progress')
    p.add_argument('--rate', type=float, default=0.,
                   help='Play a recorded run back at this multiple of sensor time (0 = at once)')
    p.add_argument('--groundtruth', action='store_true',
                   help='Overlay the dataset reference; completed runs only')
    p.add_argument('--data', type=Path, help='Dataset root, for --groundtruth')
    p.add_argument('--spawn', action='store_true', help='Open the desktop viewer')
    p.add_argument('--open', action='store_true',
                   help='Launch a viewer on the served stream, desktop app if there is one')
    p.add_argument('--save', type=Path, help='Write a .rrd instead of serving')
    p.add_argument('--web-port', type=int, default=9090)
    p.add_argument('--grpc-port', type=int, default=9876)
    p.add_argument('--pixel-step', type=int, default=4,
                   help='Submap decimation. 4 is the density the mapper itself exports at; '
                        'raise it for a lighter view, lower it for a denser one')
    p.add_argument('--frames', choices=('all', 'centre'), default='all',
                   help="'all' draws every frame of every submap, as the exported map "
                        "fuses them; 'centre' draws one frame per submap and is ~5x lighter")
    p.add_argument('--point-radius', type=float, default=.004, help='Submap point size, metres')
    p.add_argument('--fused', action='store_true',
                   help='Show the voxel-fused map (the one export_map.py writes) instead of '
                        'every frame of every submap. Overlapping submaps each carry small '
                        'pose and scale errors, and drawing them all stacks the same surface '
                        'several times; fusion averages them into one. Completed runs only')
    p.add_argument('--fused-voxel', type=float, default=.02,
                   help='Voxel size for --fused, metres')
    p.add_argument('--background', choices=BACKGROUNDS, default='light',
                   help='Viewer background; light and white suit a dense RGB map')
    p.add_argument('--poll-s', type=float, default=.2)
    p.add_argument('--memory-limit', default='2GB',
                   help='Server memory budget. A dense map is millions of points, and the '
                        'server drops the oldest data when it runs out')
    a = p.parse_args(argv)
    if not a.run.is_dir():
        p.error(f'{a.run} is not a directory')
    import rerun as rr
    import rerun.blueprint as rrb

    label, detail = run_provenance(a.run)
    colors = PALETTE['dark' if a.background == 'dark' else 'light']
    live = label in ('LIVE', 'AWAITING')
    follow = a.follow or live
    if a.groundtruth and live:
        p.error('--groundtruth reads the reference trajectory and is refused while a run is '
                'still being written: a live view must not quietly become an evaluation.')
    if a.groundtruth and a.data is None:
        p.error('--groundtruth needs --data')

    if not a.save and not a.spawn:
        check_ports(p, a.grpc_port, a.web_port)
    rr.init(f'DAVIO [{label}] {a.run.name}')
    if a.save:
        rr.save(str(a.save))
    elif a.spawn:
        rr.spawn()
    else:
        url = rr.serve_grpc(grpc_port=a.grpc_port, server_memory_limit=a.memory_limit)
        rr.serve_web_viewer(web_port=a.web_port, open_browser=False, connect_to=url)
        web_url = f'http://localhost:{a.web_port}/?url={url}'
        print(f'Rerun [{label}]: {web_url}', flush=True)
        if a.open:
            print(f'  {open_viewer(url, web_url)}', flush=True)
    print(f'  {detail}', flush=True)
    rr.send_blueprint(blueprint(rr, rrb, a.background))
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    note = ''
    if a.groundtruth:
        reference, info = reference_in_run_frame(a.run, a.data)
        if reference is None:
            print(f'  no reference overlay: {info}', flush=True)
            note = f'\n\nReference overlay unavailable: {info}'
        else:
            times, points = reference
            rr.log('world/groundtruth', rr.LineStrips3D([points.astype(np.float32)],
                                                        colors=[colors['gt']], radii=.006),
                   static=True)
            print(f'  reference overlay: {info}', flush=True)
            note = (f'\n\nGrey line: dataset reference, rigidly moved into this run\'s frame '
                    f'({info}). The map is shown where the back-end put it.')

    def provenance(current, text):
        rr.log('provenance', rr.TextDocument(f'# {current}\n\n{text}\n\nRun: `{a.run}`{note}',
                                             media_type=rr.MediaType.MARKDOWN), static=True)
    provenance(label, detail)

    stream = Stream(a.run, a.pixel_step, colors, a.frames, a.point_radius)
    if a.fused:
        if live:
            p.error('--fused needs a completed run: fusion reads every submap at once')
        fused = log_fused(rr, a.run, a.fused_voxel, a.pixel_step, a.point_radius)
        print(f'  fused map: {fused:,} points at {100 * a.fused_voxel:.0f} cm', flush=True)
        stream.frames = 'none'        # per-frame clouds would stack on top of the fused one
    shown, origin_sensor, origin_wall = label, None, None
    while True:
        for event in stream.events():
            t = log_event(rr, stream, event)
            # Paced playback of a recording: hold each event until its own sensor time.
            if t is not None and a.rate > 0 and not live:
                if origin_sensor is None:
                    origin_sensor, origin_wall = t, time.monotonic()
                wait = (t - origin_sensor) / a.rate - (time.monotonic() - origin_wall)
                if wait > 0:
                    time.sleep(min(wait, 5.))
        if stream.map_changed():
            log_map(rr, stream)
        current, detail = run_provenance(a.run)
        if current != shown:
            shown = current
            provenance(current, detail)
            print(f'Rerun provenance now [{current}]: {detail}', flush=True)
        if not follow:
            break
        time.sleep(a.poll_s)

    print(f'  {stream.points:,} map points from {len(stream.loaded)} submaps '
          f'(--frames {a.frames}, --pixel-step {a.pixel_step})', flush=True)
    if a.save:
        print(f'wrote {a.save}  --- open it with: rerun {a.save}', flush=True)
        return 0
    print('Serving; interrupt to stop.', flush=True)
    while not a.spawn:
        time.sleep(1.)
    return 0


if __name__ == '__main__':
    sys.exit(main())
