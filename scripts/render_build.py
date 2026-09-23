#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from render_map import (BACKGROUNDS, Stage, content_box, fit_distance,  # noqa: E402
                        orbit_eye, principal_azimuth, ramp, read_tum, resample, tube,
                        union, write_frame)


def load_submaps(o3d, run, pixel_step, voxel_m):
    from davio_mapper.mapping import world_points
    index = json.loads((run / 'map/map_index.json').read_text())['submaps']
    owners = {}
    for key, entry in index.items():
        for i, frame in enumerate(entry['frame_ids']):
            candidate = (abs(i - (len(entry['frame_ids']) - 1) / 2), entry['timestamp'], key, i)
            if frame not in owners or candidate < owners[frame]:
                owners[frame] = candidate
    mine = {}
    for _centrality, _t, key, i in owners.values():
        mine.setdefault(key, []).append(i)

    out = []
    for key, entry in sorted(index.items(), key=lambda kv: kv[1]['timestamp']):
        if key not in mine:
            continue
        with np.load(run / 'map' / entry['file'], allow_pickle=False) as z:
            sm = {k: z[k] for k in z.files}
        transform = np.asarray(entry['T_map_submap'], float)
        xyz, rgb = [], []
        for i in sorted(mine[key]):
            points, colours = world_points(sm, i, transform, pixel_step=pixel_step)
            good = np.isfinite(points).all(axis=1)
            xyz.append(points[good]); rgb.append(colours[good])
        xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        cloud.colors = o3d.utility.Vector3dVector(rgb.astype(float) / 255.)
        if voxel_m:
            cloud = cloud.voxel_down_sample(voxel_m)
        out.append((float(entry['timestamp']), cloud))
    return out


def smooth(values, window):
    """Moving average over `window` samples, with the ends held rather than tapered."""
    window = max(1, int(window) | 1)
    if window == 1 or len(values) < window:
        return values
    pad = window // 2
    padded = np.concatenate([np.repeat(values[:1], pad, axis=0), values,
                             np.repeat(values[-1:], pad, axis=0)])
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, k], kernel, mode='valid')
                     for k in range(values.shape[1])], axis=1)


def chase_camera(times, positions, t, back_m, height_m, lead_s, look_down_m=0.):
    here = np.array([np.interp(t, times, positions[:, k]) for k in range(3)])
    ahead = np.array([np.interp(min(t + lead_s, times[-1]), times, positions[:, k])
                      for k in range(3)])
    behind = np.array([np.interp(max(t - lead_s, times[0]), times, positions[:, k])
                       for k in range(3)])
    direction = ahead - behind
    direction[2] = 0.
    norm = np.linalg.norm(direction)
    # A stationary moment has no heading of its own; keep the last one rather than let the
    # camera swing to whatever numerical noise says.
    direction = direction / norm if norm > 1e-3 else chase_camera.last
    chase_camera.last = direction
    eye = here - direction * back_m + np.array([0., 0., height_m])
    return eye, ahead - np.array([0., 0., look_down_m])


chase_camera.last = np.array([1., 0., 0.])


def encode(frames, video, fps, box=None, crf=23):
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps),
               '-i', str(frames / '%05d.jpg')]
    if box is not None:
        height = (box[1] - box[0]) // 2 * 2          # h264 needs even dimensions
        width = (box[3] - box[2]) // 2 * 2
        command += ['-vf', f'crop={width}:{height}:{box[2]}:{box[0]}']
    command += ['-c:v', 'libx264', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                '-crf', str(crf), str(video)]
    return subprocess.run(command).returncode == 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True, help='Run directory with map/submaps')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--camera', choices=('fly', 'top', 'side', 'hero', 'orbit'), default='fly')
    p.add_argument('--duration', type=float, default=24., help='Video seconds for the whole run')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--hold-s', type=float, default=1.5,
                   help='Seconds to sit on the finished map at the end')
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--background', choices=tuple(BACKGROUNDS), default='white')
    p.add_argument('--point-size', type=float, default=2.5)
    p.add_argument('--pixel-step', type=int, default=2, help='Depth pixels sampled per submap')
    p.add_argument('--voxel-m', type=float, default=0.03)
    p.add_argument('--outlier-neighbours', type=int, default=16,
                   help='Statistical outlier removal, per submap; 0 disables')
    p.add_argument('--outlier-sigma', type=float, default=2.0)
    p.add_argument('--drop-quantile', type=float, default=0.005,
                   help='Discard this fraction of the points furthest from the centre')
    p.add_argument('--cut-ceiling', type=float, default=0.,
                   help='Drop points above this height quantile (0.88 opens up the rooms; '
                        'the fly camera ignores it unless asked, since it flies inside)')
    p.add_argument('--tube-radius', type=float, default=0.,
                   help='0 = 1/500 of the scene, or 1/1500 flying: the same tube a metre '
                        'from the lens would fill half the frame')
    p.add_argument('--fov', type=float, default=0., help='0 = 70 flying, 55 fixed')
    p.add_argument('--margin', type=float, default=1.06)
    p.add_argument('--elevation', type=float, default=38.)
    p.add_argument('--azimuth', default='auto')
    p.add_argument('--chase-back', type=float, default=4.5, help='Metres behind the pose')
    p.add_argument('--chase-height', type=float, default=2.2, help='Metres above it')
    p.add_argument('--chase-lead', type=float, default=1.5,
                   help='Seconds ahead the camera aims, which also sets its heading')
    p.add_argument('--chase-look-down', type=float, default=0.,
                   help='Metres below the path to aim; ~1 frames the ground outdoors')
    p.add_argument('--crf', type=int, default=23, help='h264 quality; lower is bigger')
    p.add_argument('--keep-frames', action='store_true')
    a = p.parse_args(argv)

    if not (a.run / 'map/map_index.json').is_file():
        p.error(f'{a.run}/map/map_index.json not found: this needs a run with dense archives '
                '(`./davio map` keeps them; scripts/replay_map.py needs --keep-dense)')
    try:
        import open3d as o3d
    except ImportError:
        p.error('Open3D is not installed for this interpreter. It runs on the host, not in '
                'the DAVIO image: `pip install open3d`.')

    a.out.mkdir(parents=True, exist_ok=True)
    fov = a.fov or (70. if a.camera == 'fly' else 55.)
    print(f'{a.run}, {a.camera} camera:')
    submaps = load_submaps(o3d, a.run, a.pixel_step, a.voxel_m)
    if not submaps:
        p.error('no submaps in this run')
    total = sum(len(s.points) for _t, s in submaps)
    print(f'  {len(submaps)} submaps, {total:,} points at {a.voxel_m} m voxels')

    if a.outlier_neighbours:
        submaps = [(t, s.remove_statistical_outlier(nb_neighbors=a.outlier_neighbours,
                                                    std_ratio=a.outlier_sigma)[0])
                   for t, s in submaps]
    if a.drop_quantile:
        # A depth outlier lands far behind the room it came from. Left in, it never shows
        # as anything but a speck, yet it sets the scene bounds -- so the fixed cameras
        # frame mostly empty space and the video crops to nothing.
        everything = np.concatenate([np.asarray(s.points) for _t, s in submaps])
        middle = np.median(everything, axis=0)
        limit = float(np.quantile(np.linalg.norm(everything - middle, axis=1),
                                  1 - a.drop_quantile))
        submaps = [(t, s.select_by_index(np.where(
            np.linalg.norm(np.asarray(s.points) - middle, axis=1) <= limit)[0]))
            for t, s in submaps]
    if a.outlier_neighbours or a.drop_quantile:
        kept = sum(len(s.points) for _t, s in submaps)
        print(f'  {kept:,} points after cleaning ({100 * kept / max(1, total):.1f}%)')

    if a.cut_ceiling:
        everything = np.concatenate([np.asarray(s.points) for _t, s in submaps])
        ceiling = float(np.quantile(everything[:, 2], a.cut_ceiling))
        submaps = [(t, s.select_by_index(
            np.where(np.asarray(s.points)[:, 2] <= ceiling)[0])) for t, s in submaps]
        print(f'  cutaway: ceiling above z = {ceiling:.2f} m removed')

    path = next((a.run / n for n in ('map_trajectory_final.tum', 'map_trajectory.tum',
                                     'trajectory.tum') if (a.run / n).is_file()), None)
    if path is None:
        p.error('no trajectory file in the run')
    times, positions = read_tum(path)
    rate = len(times) / max(1e-9, times[-1] - times[0])
    flight = smooth(positions, window=rate * 1.2)
    print(f'  trajectory: {len(times):,} poses over {times[-1] - times[0]:.0f} s from {path}')

    points = np.concatenate([np.asarray(s.points) for _t, s in submaps])
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.
    extent = points.max(axis=0) - points.min(axis=0)
    tube_radius = a.tube_radius or (float(np.linalg.norm(extent))
                                    / (1500. if a.camera == 'fly' else 500.))
    axis_deg = principal_azimuth(points)
    azimuth = axis_deg + 90. if a.azimuth == 'auto' else float(a.azimuth)
    aspect = a.width / a.height
    up = np.array([0., 0., 1.])

    # The trajectory is laid down in chunks as the walk reaches them, so the tube grows
    # with the map instead of being there from the first frame.
    index = resample(positions, tube_radius)
    nodes, node_times = positions[index], times[index]
    fraction = (node_times - node_times[0]) / max(1e-9, node_times[-1] - node_times[0])
    colours = ramp(fraction)
    pieces = []
    step = max(2, len(nodes) // 120)
    for start in range(0, len(nodes) - 1, step - 1):
        stop = min(start + step, len(nodes))
        if stop - start >= 2:
            pieces.append((float(node_times[stop - 1]),
                           tube(o3d, nodes[start:stop], tube_radius, colours[start:stop])))

    stage = Stage(o3d, a.width, a.height, a.background, a.point_size)
    # The visualizer needs geometry before it has a view at all, and at frame 0 the map is
    # empty. Eight background-coloured corners of the finished map give it the scene's real
    # extent from the start -- invisible, and the camera is set explicitly every frame.
    corners = np.array([[x, y, z] for x in (points[:, 0].min(), points[:, 0].max())
                        for y in (points[:, 1].min(), points[:, 1].max())
                        for z in (points[:, 2].min(), points[:, 2].max())])
    anchor = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(corners))
    anchor.colors = o3d.utility.Vector3dVector(np.repeat(
        np.asarray(BACKGROUNDS[a.background])[None], len(corners), axis=0))
    stage.add(anchor)
    control = stage.vis.get_view_control()
    # Clipping planes are derived from whatever geometry is present, which at frame 0 is
    # one submap: without pinning them, the far plane would cut the building in half as
    # the map outgrows its first piece.
    control.set_constant_z_near(max(0.05, tube_radius))
    control.set_constant_z_far(float(np.linalg.norm(extent)) * 6.)

    def framed(elevation_deg, azimuth_deg, camera_up=up):
        direction = orbit_eye(np.zeros(3), 1., elevation_deg, azimuth_deg)
        return centre + direction * fit_distance(points, centre, direction, camera_up,
                                                 fov, aspect, a.margin), camera_up

    overhead_up = np.array([-math.sin(math.radians(axis_deg)),
                            math.cos(math.radians(axis_deg)), 0.])
    fixed = {'top': framed(89.5, azimuth, overhead_up),
             'side': framed(6., azimuth + 90.),
             'hero': framed(a.elevation, azimuth)}.get(a.camera)
    orbit_radius = (max(fit_distance(points, centre, orbit_eye(np.zeros(3), 1., a.elevation,
                                                               angle), up, fov, aspect, a.margin)
                        for angle in range(0, 360, 15)) if a.camera == 'orbit' else 0.)

    frames_dir = a.out / f'build_{a.camera}'
    frames_dir.mkdir(exist_ok=True)
    total_frames = int(a.duration * a.fps)
    hold_frames = int(a.hold_s * a.fps)
    t0, t1 = float(times[0]), float(times[-1])
    pending = sorted([(t, ('cloud', g)) for t, g in submaps]
                     + [(t, ('tube', g)) for t, g in pieces], key=lambda item: item[0])
    box = None
    for i in range(total_frames + hold_frames):
        t = t0 + (t1 - t0) * min(1., i / max(1, total_frames - 1))
        while pending and pending[0][0] <= t:
            stage.vis.add_geometry(pending.pop(0)[1][1], reset_bounding_box=False)
        if a.camera == 'fly':
            eye, target = chase_camera(times, flight, t, a.chase_back, a.chase_height,
                                       a.chase_lead, a.chase_look_down)
            image = stage.shot(target, eye, up, fov)
        else:
            if a.camera == 'orbit':
                eye, camera_up = orbit_eye(centre, orbit_radius, a.elevation,
                                           azimuth + 360. * i / max(1, total_frames)), up
            else:
                eye, camera_up = fixed
            image = stage.shot(centre, eye, camera_up, fov)
            # Every camera but the flying one holds still or circles at a fixed distance,
            # so the union of what the frames cover is a stable crop for the whole video.
            box = union(box, content_box(image, BACKGROUNDS[a.background]))
        write_frame(frames_dir, i, image)
    stage.close()

    video = a.out / f'map_build_{a.camera}.mp4'
    if encode(frames_dir, video, a.fps, box, a.crf):
        seconds = (total_frames + hold_frames) / a.fps
        print(f'  wrote {video} ({seconds:.0f} s at {a.fps} fps)')
        # The frame -> sensor-time schedule, so anything laid beside this video (the camera
        # frames, in scripts/compose_video.py) can line up with it exactly.
        Path(str(video) + '.timeline.json').write_text(json.dumps(dict(
            t0=t0, t1=t1, fps=a.fps, total_frames=total_frames, hold_frames=hold_frames,
            run=str(a.run), camera=a.camera)))
        if not a.keep_frames:
            for frame in frames_dir.glob('*.*'):
                frame.unlink()
            frames_dir.rmdir()
    else:
        print(f'  ffmpeg failed; the frames are in {frames_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
