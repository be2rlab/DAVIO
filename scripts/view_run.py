#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import signal
import socket
import sys
import time
from urllib.parse import quote
import zipfile
import zlib

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from davio.mapper import sim3                       # noqa: E402
from davio.mapper.fusion import world_points        # noqa: E402

TIMELINE = 'sensor_time'
TRAJECTORIES = ('trajectory', 'camera_trajectory', 'map_trajectory')
CHUNK = 500             
STEP = 10               
RESCALE = .005          
FLOOR_PERCENTILE = 2    
FLOOR_LAYER_M = .05     
FLOOR_BELOW_CAMERA_M = 2.5
HEIGHT_BAND_M = 2.8     
GRID_MARGIN_M = 3       
DEPTH_RANGE_M = (0., 8.)
# Floor to ceiling: deep blue, teal, green, amber.
RAMP = np.array([[60, 120, 235], [35, 190, 215], [125, 215, 105], [250, 210, 75]], float)
PALETTE = dict(
    dark=dict(background=(13, 16, 21), map_trajectory=(0, 210, 255), trajectory=(255, 150, 60, 150), keyframe=(240, 244, 248),
              graph=(105, 115, 130), loop=(255, 70, 200), grid=(44, 50, 60)),
    light=dict(background=(247, 248, 250), map_trajectory=(0, 115, 215), trajectory=(230, 110, 20, 170), keyframe=(35, 40, 50),
               graph=(150, 156, 168), loop=(210, 30, 150), grid=(218, 222, 228)))
# Each plotted quantity: its name in the legend and its colour.
SERIES = {'plots/latency/window': ('last image to map (s)', (0, 210, 255)),
          'plots/scale/window': ('window fit', (250, 205, 75)),
          'plots/scale/graph': ('after the graph', (255, 70, 200)),
          'plots/corrections/jump': ('correction jump (cm)', (255, 150, 60)),
          'plots/corrections/alignment': ('alignment RMSE (cm)', (110, 200, 105))}


def ui(points):
    return -float(points)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def provenance(run):
    meta = read_json(Path(run) / 'run.json')
    if not meta:
        return 'AWAITING', 'No run.json yet: a run about to start, or a wrong path.'
    sequence, state = meta.get('sequence'), meta.get('state')
    if state in ('starting', 'running'):
        return 'LIVE', f'{sequence}, being written as it is read.'
    if state == 'failed':
        return 'FAILED', f'{sequence} failed: {meta.get("error")}'
    failed = ' Its mapping process failed, so there is no dense map: see the window events.' if meta.get('vision_failed') else ''
    return 'RECORDED', f'{sequence}, completed and read from disk.{failed}'


def run_card(run):
    run = Path(run)
    label, detail = provenance(run)
    meta = read_json(run / 'run.json')
    if not meta:
        return f'## {label}\n\n{detail}'
    rows = []
    start, end, last = meta.get('interval_start'), meta.get('interval_end'), meta.get('last_sensor_time')
    if start is not None and last is not None:
        rate = meta.get('rate')
        rows.append(('Sensor time', f'{last - start:.0f} s' + (f' of {end - start:.0f} s' if end is not None else '')
                     + (f' · at {rate:g}×' if rate else ' · unpaced')))
    rows.append(('Poses', f'{meta.get("poses", 0):,}'))
    windows = meta.get('windows') or {}
    if windows:
        rows.append(('Windows mapped', ' · '.join([f'{windows.get("mapped", 0)} of {windows.get("submitted", 0)}']
                                                  + [f'{windows[k]} {k}' for k in ('deferred', 'rejected', 'dropped') if windows.get(k)])))
    rows.append(('Loops closed', str(meta.get('loops_closed', 0))))
    if meta.get('pose_age_p50_s') is not None and meta.get('pose_age_p95_s') is not None:
        rows.append(('Pose latency', f'{1e3 * meta["pose_age_p50_s"]:.0f} ms median · {1e3 * meta["pose_age_p95_s"]:.0f} ms p95'))
    ate = read_json(run / 'evaluation.json').get('ate') or {}
    errors = [f'{name} {100 * ate[name]["position_m"]:.1f} cm' for name in ('odometry', 'live', 'final')
              if (ate.get(name) or {}).get('position_m') is not None]
    if errors:
        rows.append(('**Trajectory error (ATE)**', ' · '.join(errors)))
    surface = read_json(run / 'surface.json')
    f1 = ((surface.get('thresholds') or {}).get('0.05') or {}).get('f1')
    if f1 is not None:
        chamfer = surface.get('chamfer_mean_m')
        rows.append(('**Surface F-score at 5 cm**', f'{f1:.2f}' + (f' · chamfer {100 * chamfer:.1f} cm' if chamfer is not None else '')))
    table = '\n'.join(f'| {name} | {value} |' for name, value in rows)
    warning = f'{detail}\n\n' if label == 'FAILED' or meta.get('vision_failed') else ''
    return f'## {label} · {meta.get("sequence")}\n\n{warning}| | |\n|---|---|\n{table}\n'


class Tail:
    """The lines appended to a file since the last look; one still being written waits for its end."""

    def __init__(self, path):
        self.path, self.offset, self.partial = Path(path), 0, ''

    def lines(self):
        try:
            with self.path.open() as f:
                f.seek(self.offset)
                chunk = f.read()
                self.offset = f.tell()
        except FileNotFoundError:
            return []
        lines = (self.partial + chunk).split('\n')
        self.partial = lines.pop()
        return [line for line in lines if line.strip()]


def read_pose(line):
    """A TUM line: its time and its pose."""
    values = np.array(line.split(), float)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = Rotation.from_quat(values[4:8]).as_matrix(), values[1:4]
    return values[0], pose


def placement(node):
    return sim3.pose(node)


def image_points(sm, i, node, pixel_step):
    s = sim3.scale(node)
    return world_points(sm, i, np.diag([s, s, s, 1.]), pixel_step)


def is_grey(rgb):
    rgb = np.asarray(rgb, np.int16)
    return int(np.abs(rgb[..., 0] - rgb[..., 1]).max()) <= 2 and int(np.abs(rgb[..., 1] - rgb[..., 2]).max()) <= 2


def floor_height(z, camera_z=None):
    """The floor: the densest FLOOR_LAYER_M layer of the points within FLOOR_BELOW_CAMERA_M below the camera, or below
    their median height when the camera's is not known. A low percentile would take what a glossy floor reflects, a
    room's worth of points below it, for the floor, and its mirrored ceiling is denser than the floor itself."""
    z = np.asarray(z, float)
    if not len(z):
        return 0.
    low = z[z <= np.median(z)]
    if camera_z is not None:
        near = z[(z >= camera_z - FLOOR_BELOW_CAMERA_M) & (z <= camera_z + FLOOR_LAYER_M)]
        low = near if len(near) else low
    edges = np.arange(low.min(), low.max() + 2 * FLOOR_LAYER_M, FLOOR_LAYER_M)
    counts, edges = np.histogram(low, edges)
    k = int(np.argmax(counts))
    return float(np.median(low[(low >= edges[k]) & (low <= edges[k + 1])]))


def height_colors(z, floor, intensity):
    f = np.clip((np.asarray(z, float) - floor) / HEIGHT_BAND_M, 0., 1.) * (len(RAMP) - 1)
    lo = np.minimum(f.astype(int), len(RAMP) - 2)
    w = (f - lo)[:, None]
    shade = .55 + .45 * np.asarray(intensity, float)[:, None] / 255.
    return np.clip((RAMP[lo] * (1. - w) + RAMP[lo + 1] * w) * shade, 0, 255).astype(np.uint8)


def metric_depth(sm, i, node):
    depth = sm['depth'][i] * sim3.scale(node)
    keep = np.isfinite(depth) & (sm['conf'][i] > 0)
    if 'valid' in sm:
        keep &= sm['valid'][i]
    if 'sky' in sm:
        keep &= ~sm['sky'][i].astype(bool)
    return np.where(keep, depth, 0.).astype(np.float32)


def upright(image, rotate):
    """An image turned ``rotate`` degrees clockwise, for display: a phone held in portrait records sideways frames."""
    return image if not rotate else np.ascontiguousarray(np.rot90(image, -int(rotate) // 90))


def camera_image(rr, rgb, light):
    """The camera's image as logged: as it is, or as a JPEG, a tenth of the size, in a recording for the web."""
    image = rr.Image(rgb)
    return image.compress(jpeg_quality=80) if light else image


def depth_for_viewer(depth, light):
    """The depth as logged: every pixel, or every other one in a recording for the web."""
    return depth[::2, ::2] if light else depth


class Viewer:
    def __init__(self, rr, run, pixel_step=4, frames='all', radius=.012, color='auto', palette=PALETTE['dark'],
                 light=False, rotate=0, floor_cut=None, ceiling_cut=None, floor=None):
        self.rr, self.run, self.light, self.rotate = rr, Path(run), light, rotate
        self.floor_cut, self.ceiling_cut = floor_cut, ceiling_cut
        self.pixel_step, self.frames, self.radius, self.color, self.palette = pixel_step, frames, radius, color, palette
        self.tails = {name: Tail(self.run / f'{name}.tum') for name in TRAJECTORIES}
        self.pending = {name: [] for name in TRAJECTORIES}
        self.events = Tail(self.run / 'events.jsonl')
        self.odometry, self.corrected, self.now = [], [], None
        self.index, self.version = None, None
        self.mode, self.floor = None, floor   # 'height' or 'rgb', and the floor's height: both from the first submap
        self.grid = None            # the floor grid's corners (x, y), lowest and highest
        self.drawn = {}             # submap -> the scale its points were drawn at
        self.owner = {}             # image -> the submap whose copy of it is drawn
        self.counts = {}            # (submap, image) -> points
        self.waiting = set()        # submaps listed whose archive could not be read yet
        self.loops, self.loops_drawn = set(), 0
        rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for entity, (name, colour) in SERIES.items():
            rr.log(entity, rr.SeriesLines(names=[name], colors=[colour], widths=[2.]), static=True)

    @property
    def poses(self):
        return len(self.odometry)

    @property
    def points(self):
        return sum(self.counts.values())

    def update(self):
        self.trajectories()
        for line in self.events.lines():
            self.event(json.loads(line))
        self.map()

    def trajectories(self):
        for name, tail in self.tails.items():
            self.pending[name] += [read_pose(line) for line in tail.lines()]
        n = min(len(rows) for rows in self.pending.values())
        rows = list(zip(*(self.pending[name][:n] for name in TRAJECTORIES)))
        for name in TRAJECTORIES:
            del self.pending[name][:n]
        for k, ((t, body), (_, camera), (_, mapped)) in enumerate(rows):
            self.odometry.append(body[:3, 3])
            self.corrected.append(mapped[:3, 3])
            if self.poses % STEP == 0 or k == n - 1:
                # The camera where the map has it: the map's correction of the body, on the camera.
                self.draw_trajectories(t, mapped @ np.linalg.inv(body) @ camera)

    def draw_trajectories(self, t, camera):
        rr, n = self.rr, self.poses
        rr.set_time(TIMELINE, timestamp=t)
        chunk = (n - 1) // CHUNK
        start = max(chunk * CHUNK - 1, 0)       # from the last pose of the strip before, so the line is unbroken
        for name, points, width in (('trajectory', self.odometry, 1.25), ('map_trajectory', self.corrected, 2.5)):
            rr.log(f'world/{name}/{chunk}', rr.LineStrips3D([np.asarray(points[start:n])], colors=[self.palette[name]], radii=ui(width)))
        rr.log('world/camera', rr.Transform3D(translation=camera[:3, 3], mat3x3=camera[:3, :3]))
        self.now = t

    def event(self, e):
        rr, kind = self.rr, e.get('kind')
        t = e.get('sensor_time', e.get('t', self.now))
        if t is not None:
            rr.set_time(TIMELINE, timestamp=float(t))
        if kind == 'window' and 'error' in e:
            last = (e['error'].strip().splitlines() or [''])[-1]
            rr.log('events', rr.TextLog(f'the mapping process failed: {last}', level=rr.TextLogLevel.ERROR))
        elif kind == 'window':
            report = e['report']
            parts = [report['status']]
            if report.get('submap'):
                parts.append(f'submap {report["submap"]}')
            if report.get('scale') is not None:
                parts.append(f'scale {report["scale"]:.3f}')
            if report.get('accepted_loops'):
                parts.append(f'{report["accepted_loops"]} loop' + ('s' if report['accepted_loops'] > 1 else ''))
            text = ' · '.join(parts) + (f': {report["reason"]}' if report.get('reason') else '')
            rr.log('events', rr.TextLog(text, level=rr.TextLogLevel.INFO if report['status'] == 'mapped' else rr.TextLogLevel.WARN))
            if e.get('wall') is not None and e.get('arrival_wall') is not None:
                # From the window's last image arriving to its map: whether the mapper keeps up.
                rr.log('plots/latency/window', rr.Scalars(float(e['wall']) - float(e['arrival_wall'])))
            for entity, key, unit in (('plots/scale/window', 'scale', 1.), ('plots/scale/graph', 'scale_correction', 1.),
                                      ('plots/corrections/alignment', 'alignment_rmse_m', 100.)):
                if report.get(key) is not None:
                    rr.log(entity, rr.Scalars(float(report[key]) * unit))
            self.loops.update((loop['a'], loop['b']) for loop in report.get('loops') or [])
        elif kind == 'correction':
            rr.log('plots/corrections/jump', rr.Scalars(100. * float(e['jump_m'])))
        elif kind == 'dropped':
            rr.log('events', rr.TextLog('window dropped: the mapper was still busy', level=rr.TextLogLevel.WARN))
        elif kind == 'final_trajectory_failed':
            rr.log('events', rr.TextLog(f'no final trajectory: {e.get("reason")}', level=rr.TextLogLevel.ERROR))

    def map(self):
        path = self.run / 'map/map_index.json'
        try:
            stamp = path.stat().st_mtime_ns
            changed = stamp != self.version
            if changed:
                self.index, self.version = json.loads(path.read_text()), stamp
        except (OSError, ValueError):       # not written yet, or read while it was replaced
            return
        if not changed and not self.waiting and len(self.loops) == self.loops_drawn:
            return
        rr, now = self.rr, float(self.index['last_sensor_time'])
        submaps = {key: entry for key, entry in self.index['submaps'].items() if 'T_map_submap' in entry}
        for key, entry in submaps.items():
            node = np.asarray(entry['T_map_submap'], float)
            placed = placement(node)
            transform = rr.Transform3D(translation=placed[:3, 3], mat3x3=placed[:3, :3])
            if key not in self.drawn:
                # Placed when it was seen as well: the transform shown is the newest at or before the time
                # on the timeline, and one logged only now would leave the submap's earlier points at the origin.
                rr.set_time(TIMELINE, timestamp=float(entry['timestamp']))
                rr.log(f'world/submaps/{key}', transform)
            if changed:         # and now, so that a loop closure shows as the geometry moving
                rr.set_time(TIMELINE, timestamp=now)
                rr.log(f'world/submaps/{key}', transform)
            if key not in self.drawn:
                self.draw(key, entry, node)
            elif abs(np.log(sim3.scale(node) / self.drawn[key])) > RESCALE:
                self.draw(key, entry, node, when=now)
        rr.set_time(TIMELINE, timestamp=now)
        centres = {key: np.asarray(entry['T_map_submap'])[:3, 3] for key, entry in submaps.items()}
        if centres:
            nodes = np.array(list(centres.values()))
            rr.log('world/keyframes', rr.Points3D(nodes, colors=[self.palette['keyframe']], radii=ui(3.5)))
            if len(nodes) > 1:
                rr.log('world/graph', rr.LineStrips3D([nodes], colors=[self.palette['graph']], radii=ui(1.)))
            if self.floor is not None:
                self.draw_grid(nodes)
        loops = [[centres[a], centres[b]] for a, b in sorted(self.loops) if a in centres and b in centres]
        if loops:
            rr.log('world/loops', rr.LineStrips3D(np.array(loops), colors=[self.palette['loop']], radii=ui(2.5)))
        self.loops_drawn = len(self.loops)

    def draw(self, key, entry, node, when=None):
        """A submap's images as points in its own frame, each at the time it was taken (``when``, for a redraw)."""
        rr = self.rr
        try:
            with np.load(self.run / 'map' / entry['file']) as stored:
                sm = {name: stored[name] for name in stored.files}
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error):
            self.waiting.add(key)           # the mapper lists a submap before its archive is written
            return
        self.waiting.discard(key)
        if self.mode is None:
            self.mode = 'rgb' if self.color == 'rgb' or (self.color == 'auto' and not is_grey(sm['rgb'])) else 'height'
            height, width = sm['depth'].shape[1:]
            rr.log('world/camera/frustum', rr.Pinhole(image_from_camera=sm['intrinsics'][0], resolution=[width, height],
                                                      image_plane_distance=.3), static=True)
        ids, placed = entry['frame_ids'], placement(node)
        images = []
        for i in range(len(sm['depth'])) if self.frames == 'all' else [entry['center']]:
            if self.frames == 'all' and self.owner.setdefault(ids[i], key) != key:
                continue                    # an image that two windows hold is drawn once
            xyz, rgb = image_points(sm, i, node, self.pixel_step)
            images.append((i, xyz, rgb, xyz @ placed[2, :3] + placed[2, 3]))
        if self.floor is None and any(len(z) for *_, z in images):
            self.floor = float(np.percentile(np.concatenate([z for *_, z in images]), FLOOR_PERCENTILE))
        if self.floor is not None and (self.floor_cut is not None or self.ceiling_cut is not None):    # for display
            low = -np.inf if self.floor_cut is None else self.floor - self.floor_cut
            high = np.inf if self.ceiling_cut is None else self.floor + self.ceiling_cut
            images = [(i, xyz[keep], rgb[keep], z[keep]) for i, xyz, rgb, z in images for keep in [(z >= low) & (z <= high)]]
        for i, xyz, rgb, z in images:
            if self.mode == 'height':
                rgb = height_colors(z, self.floor, rgb.mean(axis=1))
            rr.set_time(TIMELINE, timestamp=when if when is not None else int(ids[i]) * 1e-9)      # ids: stamps in ns, as text
            rr.log(f'world/submaps/{key}/images/{i}', rr.Points3D(xyz.astype(np.float32), colors=rgb, radii=self.radius))
            self.counts[key, i] = len(xyz)
            if when is None:
                rr.log('latest/image', camera_image(rr, upright(sm['rgb'][i], self.rotate), self.light))
                depth = upright(depth_for_viewer(metric_depth(sm, i, node), self.light), self.rotate)
                rr.log('latest/depth', rr.DepthImage(depth, meter=1.,
                                                     colormap='turbo', depth_range=DEPTH_RANGE_M))
        self.drawn[key] = sim3.scale(node)

    def draw_grid(self, centres):
        """Square metres on the floor under the keyframes and a little beyond; it grows with the map, never shrinks."""
        lo, hi = np.floor(centres[:, :2].min(axis=0) - GRID_MARGIN_M), np.ceil(centres[:, :2].max(axis=0) + GRID_MARGIN_M)
        if self.grid is not None:
            if (lo >= self.grid[0]).all() and (hi <= self.grid[1]).all():
                return
            lo, hi = np.minimum(lo, self.grid[0]), np.maximum(hi, self.grid[1])
        self.grid, z = (lo, hi), self.floor
        lines = ([[[x, lo[1], z], [x, hi[1], z]] for x in np.arange(lo[0], hi[0] + 1.)]
                 + [[[lo[0], y, z], [hi[0], y, z]] for y in np.arange(lo[1], hi[1] + 1.)])
        self.rr.log('world/grid', self.rr.LineStrips3D(np.array(lines), colors=[self.palette['grid']], radii=ui(.75)), static=True)


def blueprint(rrb, palette, layout='inspect'):
    """'inspect': the map beside the camera, its depth, the plots and the run's card; 'video': the map beside the
    camera, its depth and the scale, nothing else on screen."""
    corner = rrb.components.Corner2D
    if layout == 'video':
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin='world', name='Map', background=palette['background'],
                                  line_grid=rrb.archetypes.LineGrid3D(visible=False)),
                rrb.Vertical(rrb.Spatial2DView(origin='latest/image', name='Camera'),
                             rrb.Spatial2DView(origin='latest/depth', name='Metric depth'),
                             rrb.TimeSeriesView(origin='plots/scale', name='Depth scale', plot_legend=corner.RightBottom),
                             row_shares=[3., 3., 2.]),
                column_shares=[2, 1]),
            rrb.BlueprintPanel(state='hidden'), rrb.SelectionPanel(state='hidden'), rrb.TimePanel(state='hidden'))
    if layout != 'inspect':
        raise ValueError(f'unknown layout {layout!r}')
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin='world', name='Map', background=palette['background'],
                              line_grid=rrb.archetypes.LineGrid3D(visible=False)),      # its own, at z = 0, would cut the room
            rrb.Vertical(
                rrb.Horizontal(rrb.Spatial2DView(origin='latest/image', name='Camera'),
                               rrb.Spatial2DView(origin='latest/depth', name='Metric depth')),
                rrb.TimeSeriesView(origin='plots/latency', name='Mapping latency (s)', plot_legend=rrb.PlotLegend(visible=False)),
                rrb.TimeSeriesView(origin='plots/scale', name='Depth scale', plot_legend=corner.RightBottom),
                rrb.TimeSeriesView(origin='plots/corrections', name='Map corrections (cm)', plot_legend=corner.LeftTop),
                rrb.Tabs(rrb.TextDocumentView(origin='run', name='Run'), rrb.TextLogView(origin='events', name='Windows')),
                row_shares=[3., 1.5, 1.5, 1.5, 4.5]),
            column_shares=[7, 3]),
        rrb.BlueprintPanel(state='collapsed'), rrb.SelectionPanel(state='collapsed'), rrb.TimePanel(state='collapsed'))


def viewer_url(web_port, server):
    return f'http://localhost:{web_port}/?url={quote(server, safe="")}'


def port_free(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--follow', action='store_true',
                   help='keep drawing as the run is written, until it ends; the default for a run in progress, and '
                        'it waits for a run directory that is not there yet')
    p.add_argument('--save', type=Path, help='write a .rrd file (open it with `rerun FILE`) instead of serving')
    p.add_argument('--web-port', type=int, default=9090)
    p.add_argument('--grpc-port', type=int, default=9876)
    p.add_argument('--pixel-step', type=int, default=4, help='every this many pixels of an image; 4 is what the exported map fuses')
    p.add_argument('--frames', choices=('all', 'centre'), default='all',
                   help="'centre': one image per submap, about five times lighter (a long run, a remote viewer)")
    p.add_argument('--color', choices=('auto', 'height', 'rgb'), default='auto',
                   help="the map's colour: by height, shaded by the image, or the camera's own; 'auto': height for a grey camera")
    p.add_argument('--point-radius', type=float, default=.012, help='metres')
    p.add_argument('--background', choices=tuple(PALETTE), default='dark')
    p.add_argument('--rotate', type=int, choices=(0, 90, 180, 270), default=0,
                   help='degrees clockwise to turn the camera and depth images on screen (90: a phone held in portrait)')
    p.add_argument('--floor', type=float, default=None,
                   help="the floor's height in the map frame; by default the first submap's lowest points")
    p.add_argument('--floor-cut', type=float, default=None,
                   help='hide points more than this many metres below the floor: what a glossy floor reflects')
    p.add_argument('--ceiling-cut', type=float, default=None,
                   help='hide points more than this many metres above the floor, to see into rooms from above')
    p.add_argument('--light', action='store_true',
                   help='camera images as JPEG and depth at half resolution: a recording small enough for a web page')
    p.add_argument('--layout', choices=('inspect', 'video'), default='inspect',
                   help="'video': the map, the camera, its depth and the scale only, for recording (record_rerun.py)")
    p.add_argument('--memory-limit', default='2GB', help='what the server keeps; beyond it, it drops the oldest data')
    p.add_argument('--poll-s', type=float, default=.2)
    a = p.parse_args(argv)
    # Stopped by ./davio when the run it watches is over; in the image it is process 1, which ignores
    # SIGTERM unless something handles it.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if not a.follow and not a.run.is_dir():
        p.error(f'{a.run} is not a run directory')
    follow = a.follow or provenance(a.run)[0] == 'LIVE'
    if not a.save:
        for port in (a.web_port, a.grpc_port):
            if not port_free(port):
                p.error(f'port {port} is taken, most likely by another viewer: --web-port and --grpc-port '
                        '(DAVIO_WEB_PORT and DAVIO_GRPC_PORT for ./davio), or --save')
    import rerun as rr
    import rerun.blueprint as rrb

    rr.init(f'DAVIO · {a.run.name}')
    if a.save:
        rr.save(str(a.save))
    else:
        url = rr.serve_grpc(grpc_port=a.grpc_port, server_memory_limit=a.memory_limit)
        rr.serve_web_viewer(web_port=a.web_port, open_browser=False, connect_to=url)
        print(f'Rerun: {viewer_url(a.web_port, url)}', flush=True)
    palette = PALETTE[a.background]
    rr.send_blueprint(blueprint(rrb, palette, a.layout))
    viewer = Viewer(rr, a.run, a.pixel_step, a.frames, a.point_radius, a.color, palette, a.light, a.rotate, a.floor_cut,
                    a.ceiling_cut, a.floor)
    shown = dict(label=None, card=None)

    def refresh():
        """The run card, and a line on the terminal when what is shown changes."""
        label, detail = provenance(a.run)
        if label != shown['label']:
            print(f'  [{label}] {detail}', flush=True)
        card = run_card(a.run)
        if card != shown['card']:
            rr.log('run', rr.TextDocument(card, media_type=rr.MediaType.MARKDOWN), static=True)
        shown.update(label=label, card=card)
        return label

    while True:
        label = refresh()
        viewer.update()
        if not follow or label in ('RECORDED', 'FAILED'):
            break
        time.sleep(a.poll_s)
    unread = f'; {len(viewer.waiting)} submap archives unreadable' if viewer.waiting else ''
    print(f'  {viewer.poses} poses, {viewer.points:,} points from {len(viewer.drawn)} submaps{unread}', flush=True)
    if a.save:
        rr.disconnect()
        print(f'wrote {a.save}', flush=True)
        return 0
    print('Serving; interrupt to stop.', flush=True)
    try:
        while True:         # the scores come after the run: the card shows them once they are written
            time.sleep(2.)
            refresh()
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    sys.exit(main())
