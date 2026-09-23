#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

# Matplotlib's turbo, sampled every 16th entry: a time ramp that stays legible projected
# and in print, without taking a matplotlib dependency into a rendering script.
TURBO = np.array([
    (0.190, 0.072, 0.232), (0.244, 0.288, 0.750), (0.275, 0.474, 0.950), (0.265, 0.640, 0.989),
    (0.196, 0.786, 0.867), (0.150, 0.893, 0.710), (0.238, 0.967, 0.510), (0.425, 0.999, 0.329),
    (0.609, 0.973, 0.222), (0.758, 0.891, 0.208), (0.878, 0.786, 0.222), (0.961, 0.652, 0.191),
    (0.996, 0.489, 0.125), (0.968, 0.325, 0.056), (0.887, 0.196, 0.018), (0.762, 0.096, 0.011),
    (0.610, 0.030, 0.007), (0.480, 0.016, 0.011)])

# 'sky' is for outdoor scenes: the walkway reads as outdoors, and the few sky pixels the
# depth filter lets through land on the colour they came from instead of floating on white.
BACKGROUNDS = dict(white=(1., 1., 1.), light=(0.94, 0.945, 0.96), sky=(0.84, 0.91, 0.98),
                   dark=(0.09, 0.10, 0.12), black=(0., 0., 0.))


def ramp(fraction):
    """TURBO sampled at `fraction` in [0, 1], linearly interpolated."""
    x = np.clip(np.asarray(fraction, float), 0., 1.) * (len(TURBO) - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(TURBO) - 1)
    w = (x - lo)[..., None]
    return TURBO[lo] * (1 - w) + TURBO[hi] * w


def read_tum(path):
    rows = np.loadtxt(path)
    rows = rows.reshape(1, -1) if rows.ndim == 1 else rows
    return rows[:, 0], rows[:, 1:4]


def resample(positions, step_m):
    kept = [0]
    for i in range(1, len(positions)):
        if np.linalg.norm(positions[i] - positions[kept[-1]]) >= step_m:
            kept.append(i)
    if kept[-1] != len(positions) - 1:
        kept.append(len(positions) - 1)
    return np.asarray(kept)


def tube(o3d, positions, radius, colors, sides=14):
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], 1)
    vertices, colours, triangles = [], [], []
    tangents = np.gradient(positions, axis=0)
    for i, (point, tangent) in enumerate(zip(positions, tangents)):
        norm = np.linalg.norm(tangent)
        tangent = tangent / norm if norm > 1e-9 else np.array([0., 0., 1.])
        helper = np.array([0., 0., 1.]) if abs(tangent[2]) < .9 else np.array([1., 0., 0.])
        u = np.cross(tangent, helper); u /= np.linalg.norm(u)
        v = np.cross(tangent, u)
        vertices.append(point + radius * (ring[:, :1] * u + ring[:, 1:] * v))
        colours.append(np.repeat(np.asarray(colors[i])[None], sides, axis=0))
        if i:
            base, previous = i * sides, (i - 1) * sides
            for k in range(sides):
                nxt = (k + 1) % sides
                triangles.append((previous + k, base + k, base + nxt))
                triangles.append((previous + k, base + nxt, previous + nxt))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.concatenate(vertices)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(colours))
    mesh.compute_vertex_normals()
    return mesh


def load_cloud(o3d, path, voxel_m, outlier_neighbours, outlier_sigma, drop_quantile):
    cloud = o3d.io.read_point_cloud(str(path))
    if not len(cloud.points):
        raise SystemExit(f'{path} holds no points')
    reported = [f'{len(cloud.points):,} points']
    if voxel_m:
        cloud = cloud.voxel_down_sample(voxel_m)
        reported.append(f'{len(cloud.points):,} after {voxel_m} m voxels')
    if drop_quantile:
        # Depth-network outliers land far behind the room they came from, so a few hundred
        # of them set the scene bounds and the camera then frames mostly empty space.
        points = np.asarray(cloud.points)
        radius = np.linalg.norm(points - np.median(points, axis=0), axis=1)
        keep = np.where(radius <= np.quantile(radius, 1 - drop_quantile))[0]
        cloud = cloud.select_by_index(keep)
        reported.append(f'{len(cloud.points):,} inside the {1 - drop_quantile:.4g} radius quantile')
    if outlier_neighbours:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=outlier_neighbours,
                                                   std_ratio=outlier_sigma)
        reported.append(f'{len(cloud.points):,} after outlier removal')
    print('  cloud: ' + ' -> '.join(reported))
    return cloud


def extrinsic(centre, eye, up):
    """World -> camera for a camera at `eye` looking at `centre` (OpenGL/Open3D: +z forward)."""
    forward = np.asarray(centre, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    if np.linalg.norm(right) < 1e-9:              # looking straight along `up`
        right = np.cross(forward, np.array([1., 0., 0.]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([right, down, forward])
    matrix[:3, 3] = -matrix[:3, :3] @ np.asarray(eye, float)
    return matrix


class Stage:
    """One offscreen window, reused for every shot so the geometry is uploaded once."""

    def __init__(self, o3d, width, height, background, point_size):
        self.o3d = o3d
        self.vis = o3d.visualization.Visualizer()
        if not self.vis.create_window(width=width, height=height, visible=False):
            raise SystemExit('Open3D could not create a window. Is a display available '
                             '(DISPLAY set, or an X server / Xvfb running)?')
        option = self.vis.get_render_option()
        option.background_color = np.asarray(BACKGROUNDS[background])
        option.point_size = point_size
        option.light_on = True
        self.width, self.height = width, height

    def add(self, geometry):
        self.vis.add_geometry(geometry, reset_bounding_box=True)

    def shot(self, centre, eye, up, fov):
        control = self.vis.get_view_control()
        parameters = control.convert_to_pinhole_camera_parameters()
        focal = (self.height / 2.) / math.tan(math.radians(fov) / 2.)
        intrinsic = self.o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, focal, focal,
            self.width / 2. - .5, self.height / 2. - .5)
        parameters.intrinsic = intrinsic
        parameters.extrinsic = extrinsic(centre, eye, up)
        control.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
        self.vis.poll_events()
        self.vis.update_renderer()
        image = np.asarray(self.vis.capture_screen_float_buffer(do_render=True))
        return (np.clip(image, 0, 1) * 255 + .5).astype(np.uint8)

    def close(self):
        self.vis.destroy_window()


def principal_azimuth(points):
    flat = points[:, :2] - points[:, :2].mean(axis=0)
    _values, vectors = np.linalg.eigh(np.cov(flat.T))
    axis = vectors[:, -1]
    return math.degrees(math.atan2(axis[1], axis[0]))


def content_box(image, background, margin_px=24):
    reference = (np.asarray(background) * 255 + .5).astype(np.uint8)
    content = np.abs(image.astype(np.int16) - reference.astype(np.int16)).max(axis=2) > 6
    if not content.any():
        return None
    rows, columns = np.where(content.any(axis=1))[0], np.where(content.any(axis=0))[0]
    return (max(0, rows[0] - margin_px), min(image.shape[0], rows[-1] + 1 + margin_px),
            max(0, columns[0] - margin_px), min(image.shape[1], columns[-1] + 1 + margin_px))


def union(box, seen):
    """The smallest box holding both, ignoring either that is None."""
    if seen is None:
        return box
    if box is None:
        return seen
    return (min(box[0], seen[0]), max(box[1], seen[1]),
            min(box[2], seen[2]), max(box[3], seen[3]))


def crop_background(image, background, margin_px=24):
    """Trim the uniform border so the slide is the map, not the padding around it."""
    box = content_box(image, background, margin_px)
    if box is None:
        return image
    top, bottom, left, right = box
    return image[top:bottom, left:right]


def fit_distance(points, centre, eye_direction, up, fov_deg, aspect, margin, quantile=0.999):
    forward = np.asarray(eye_direction, float)
    forward = -forward / np.linalg.norm(forward)          # camera looks back at the centre
    right = np.cross(forward, np.asarray(up, float))
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(forward, np.array([1., 0., 0.]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    offset = points - centre
    tan_v = math.tan(math.radians(fov_deg) / 2.)
    tan_h = tan_v * aspect
    needed = np.maximum(np.abs(offset @ right) / tan_h, np.abs(offset @ down) / tan_v)
    return float(np.quantile(needed - offset @ forward, quantile)) * margin


def orbit_eye(centre, radius, elevation_deg, azimuth_deg):
    """A camera position on a circle about `centre`, in the world's horizontal plane."""
    elevation, azimuth = math.radians(elevation_deg), math.radians(azimuth_deg)
    flat = radius * math.cos(elevation)
    return np.asarray(centre) + np.array([flat * math.cos(azimuth), flat * math.sin(azimuth),
                                          radius * math.sin(elevation)])


def write_png(path, image):
    import cv2
    cv2.imwrite(str(path), image[:, :, ::-1])


def write_frame(directory, number, image, suffix='jpg', quality=95):
    import cv2
    path = directory / f'{number:05d}.{suffix}'
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if suffix in ('jpg', 'jpeg') else []
    cv2.imwrite(str(path), image[:, :, ::-1], params)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, help='Run directory; --ply and --trajectory default into it')
    p.add_argument('--ply', type=Path, help='Fused cloud (default <run>/map.ply)')
    p.add_argument('--trajectory', type=Path, help="TUM file (default the run's map trajectory)")
    p.add_argument('--odometry', choices=('map', 'raw', 'both', 'none'), default='map',
                   help="Which pose stream to draw: the back-end's corrected one, the "
                        "filter's raw one, both, or neither")
    p.add_argument('--out', type=Path, required=True, help='Directory for the renders')
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--background', choices=tuple(BACKGROUNDS), default='white')
    p.add_argument('--point-size', type=float, default=2.0)
    p.add_argument('--voxel-m', type=float, default=0., help='Downsample before rendering')
    p.add_argument('--outlier-neighbours', type=int, default=20,
                   help='Statistical outlier removal; 0 disables')
    p.add_argument('--outlier-sigma', type=float, default=2.0)
    p.add_argument('--drop-quantile', type=float, default=0.005,
                   help='Discard this fraction of the points furthest from the centre')
    p.add_argument('--cut-ceiling', type=float, default=0.,
                   help='Drop points above this height quantile (0.9 is a good dollhouse '
                        'cutaway; 0 keeps the ceiling)')
    p.add_argument('--tube-radius', type=float, default=0., help='0 = 1/500 of the scene')
    p.add_argument('--elevation', type=float, default=38., help='Hero and orbit elevation, degrees')
    p.add_argument('--azimuth', default='auto',
                   help="Hero azimuth in degrees, or 'auto' to look across the building's "
                        'own longest horizontal axis')
    p.add_argument('--no-crop', action='store_true',
                   help='Keep the full frame instead of trimming the background border')
    p.add_argument('--fov', type=float, default=55.)
    p.add_argument('--margin', type=float, default=1.06,
                   help='How much room to leave around the cloud, 1.0 = exactly framed')
    p.add_argument('--orbit-frames', type=int, default=360, help='0 skips the turntable')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--keep-frames', action='store_true',
                   help='Keep the orbit PNGs after encoding (hundreds of MB; they are only '
                        'needed to re-encode differently, or to build a GIF)')
    a = p.parse_args(argv)

    ply = a.ply or (a.run / 'map.ply' if a.run else None)
    if ply is None or not Path(ply).is_file():
        p.error('no fused cloud: pass --ply, or export one with `./davio export <run>`')
    try:
        import open3d as o3d
    except ImportError:
        p.error('Open3D is not installed for this interpreter. It runs on the host, not in '
                'the DAVIO image: `pip install open3d`.')

    a.out.mkdir(parents=True, exist_ok=True)
    print(f'{ply}:')
    cloud = load_cloud(o3d, ply, a.voxel_m, a.outlier_neighbours, a.outlier_sigma,
                       a.drop_quantile)

    if a.cut_ceiling:
        points = np.asarray(cloud.points)
        ceiling = float(np.quantile(points[:, 2], a.cut_ceiling))
        cloud = cloud.select_by_index(np.where(points[:, 2] <= ceiling)[0])
        print(f'  cutaway: ceiling above z = {ceiling:.2f} m removed, '
              f'{len(cloud.points):,} points left')

    points = np.asarray(cloud.points)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.
    extent = points.max(axis=0) - points.min(axis=0)
    tube_radius = a.tube_radius or float(np.linalg.norm(extent)) / 500.
    axis_deg = principal_azimuth(points)
    azimuth = axis_deg + 90. if a.azimuth == 'auto' else float(a.azimuth)
    if a.azimuth == 'auto':
        print(f'  framing: long axis at {axis_deg:.0f} deg, viewed from {azimuth:.0f} deg')

    stage = Stage(o3d, a.width, a.height, a.background, a.point_size)
    stage.add(cloud)

    streams = []
    if a.odometry in ('map', 'both'):
        streams.append(('map', a.trajectory or (a.run and next(
            (a.run / n for n in ('map_trajectory.tum', 'map_trajectory_final.tum',
                                 'trajectory.tum') if (a.run / n).is_file()), None))))
    if a.odometry in ('raw', 'both'):
        streams.append(('raw', (a.run / 'trajectory.tum') if a.run else None))
    for name, path in streams:
        if path is None or not Path(path).is_file():
            print(f'  no {name} trajectory; skipped')
            continue
        times, positions = read_tum(path)
        index = resample(positions, tube_radius)
        positions, times = positions[index], times[index]
        fraction = ((times - times[0]) / (times[-1] - times[0])
                    if times[-1] > times[0] else np.zeros(len(times)))
        colours = ramp(fraction) if name == 'map' else np.full((len(times), 3), .40)
        stage.add(tube(o3d, positions, tube_radius * (1. if name == 'map' else .6), colours))
        print(f'  {name} trajectory: {len(times):,} nodes from {path}')

    up = np.array([0., 0., 1.])
    aspect = a.width / a.height

    def framed(elevation_deg, azimuth_deg, camera_up=up):
        """Eye position for this direction, pushed back just far enough to hold the cloud."""
        direction = orbit_eye(np.zeros(3), 1., elevation_deg, azimuth_deg)
        distance = fit_distance(points, centre, direction, camera_up, a.fov, aspect, a.margin)
        return centre + direction * distance, camera_up, distance

    # Straight down, with the long axis laid along the image's width.
    overhead_up = np.array([-math.sin(math.radians(axis_deg)),
                            math.cos(math.radians(axis_deg)), 0.])
    shots = {'top': framed(89.5, azimuth, overhead_up),
             'hero': framed(a.elevation, azimuth),
             'side': framed(6., azimuth + 90.)}
    for name, (eye, camera_up, _distance) in shots.items():
        image = stage.shot(centre, eye, camera_up, a.fov)
        if not a.no_crop:
            image = crop_background(image, BACKGROUNDS[a.background])
        write_png(a.out / f'map_{name}.png', image)
        print(f'  wrote {a.out / f"map_{name}.png"}  {image.shape[1]}x{image.shape[0]}')

    if a.orbit_frames:
        frames = a.out / 'orbit'
        frames.mkdir(exist_ok=True)
        # One distance for the whole orbit, the widest any azimuth needs, so the building
        # does not breathe in and out as the camera goes round.
        radius = max(fit_distance(points, centre,
                                  orbit_eye(np.zeros(3), 1., a.elevation, angle), up,
                                  a.fov, aspect, a.margin)
                     for angle in range(0, 360, 15))
        box = None
        for i in range(a.orbit_frames):
            eye = orbit_eye(centre, radius, a.elevation,
                            azimuth + 360. * i / a.orbit_frames)
            image = stage.shot(centre, eye, up, a.fov)
            write_frame(frames, i, image)
            # Every frame keeps the full canvas, but the video is cropped to the union of
            # what the orbit ever covers: a fixed frame, without the dead margin that the
            # widest azimuth forces on all the others.
            box = union(box, content_box(image, BACKGROUNDS[a.background]))
        video = a.out / 'map_orbit.mp4'
        height = (box[1] - box[0]) // 2 * 2          # h264 needs even dimensions
        width = (box[3] - box[2]) // 2 * 2
        command = ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(a.fps),
                   '-i', str(frames / '%05d.jpg'),
                   '-vf', f'crop={width}:{height}:{box[2]}:{box[0]}',
                   '-c:v', 'libx264', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                   '-crf', '20', str(video)]
        if subprocess.run(command).returncode == 0:
            print(f'  wrote {video} ({a.orbit_frames} frames at {a.fps} fps, {width}x{height})')
            if not a.keep_frames:
                for frame in frames.glob('*.*'):
                    frame.unlink()
                frames.rmdir()
        else:
            print(f'  ffmpeg failed; the frames are in {frames}')

    stage.close()
    (a.out / 'render.json').write_text(json.dumps(dict(
        ply=str(ply), points=len(cloud.points), extent_m=extent.tolist(),
        centre_m=centre.tolist(),
        camera_distance_m={k: v[2] for k, v in shots.items()},
        cut_ceiling=a.cut_ceiling,
        resolution=[a.width, a.height], background=a.background, point_size=a.point_size,
        fov_deg=a.fov, elevation_deg=a.elevation, azimuth_deg=azimuth,
        long_axis_deg=axis_deg,
        orbit_frames=a.orbit_frames), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
