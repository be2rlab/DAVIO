#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def read_ply(path):
    """Points and colours of a binary little-endian PLY, as written by export_map."""
    with Path(path).open('rb') as stream:
        header = b''
        while not header.endswith(b'end_header\n'):
            line = stream.readline()
            if not line:
                raise SystemExit(f'{path}: truncated PLY header')
            header += line
        text = header.decode()
        if 'binary_little_endian' not in text:
            raise SystemExit(f'{path}: only binary_little_endian PLY is supported')
        count = int(next(l for l in text.splitlines()
                         if l.startswith('element vertex')).split()[-1])
        kinds = {'float': '<f4', 'double': '<f8', 'uchar': 'u1', 'uint8': 'u1'}
        fields = [l.split() for l in text.splitlines() if l.startswith('property')]
        dtype = np.dtype([(f[2], kinds[f[1]]) for f in fields])
        rows = np.frombuffer(stream.read(count * dtype.itemsize), dtype)
    points = np.stack([rows['x'], rows['y'], rows['z']], 1).astype(np.float64)
    colours = (np.stack([rows['red'], rows['green'], rows['blue']], 1)
               if 'red' in rows.dtype.names else np.full((count, 3), 200, np.uint8))
    return points, colours.astype(np.uint8)


def thin(points, colours, target, seed=0):
    if len(points) <= target:
        return points, colours
    rng = np.random.default_rng(seed)
    extent = points.max(axis=0) - points.min(axis=0)
    # Voxel size that lands near the budget: cube root of volume per point, then one
    # pass to confirm rather than trusting the estimate on a non-cubic scene.
    volume = max(np.prod(np.maximum(extent, 1e-3)), 1e-9)
    size = float((volume / target) ** (1 / 3))
    for _ in range(8):
        keys = np.floor((points - points.min(axis=0)) / size).astype(np.int64)
        flat = (keys[:, 0] << 42) ^ (keys[:, 1] << 21) ^ keys[:, 2]
        order = np.argsort(flat, kind='stable')
        first = order[np.r_[True, flat[order][1:] != flat[order][:-1]]]
        if len(first) <= target:
            break
        size *= 1.25
    keep = first if len(first) <= target else rng.choice(first, target, replace=False)
    if len(keep) < target:
        rest = np.setdiff1d(np.arange(len(points)), keep, assume_unique=False)
        extra = rng.choice(rest, min(target - len(keep), len(rest)), replace=False)
        keep = np.concatenate([keep, extra])
    return points[keep], colours[keep]


def trajectory(run, step_m=0.05):
    """The map-frame path, thinned to `step_m`, with each node's time fraction."""
    path = next((run / n for n in ('map_trajectory_final.tum', 'map_trajectory.tum',
                                   'trajectory.tum') if (run / n).is_file()), None)
    if path is None:
        return None
    rows = np.loadtxt(path)
    times, positions = rows[:, 0], rows[:, 1:4]
    kept = [0]
    for i in range(1, len(positions)):
        if np.linalg.norm(positions[i] - positions[kept[-1]]) >= step_m:
            kept.append(i)
    kept.append(len(positions) - 1)
    kept = np.unique(kept)
    span = max(times[-1] - times[0], 1e-9)
    return dict(source=path.name,
                points=np.round(positions[kept], 4).ravel().tolist(),
                fraction=np.round((times[kept] - times[0]) / span, 4).tolist(),
                duration_s=round(float(span), 1))


def numbers(run):
    """Headline numbers the page prints under the viewer, straight from the artifacts."""
    out = {}
    evaluation = run / 'evaluation.json'
    replay = run / 'replay.json'
    source = None
    if replay.is_file():
        report = json.loads(replay.read_text())
        source = Path(report['source_run'])
        out.update(ate_online_m=report.get('ate_raw_m'), ate_map_m=report.get('ate_map_final_m'),
                   loops=report.get('accepted_loops'), submaps=report.get('submaps_mapped'))
    if not evaluation.is_file() and source is not None:
        evaluation = ROOT / source / 'evaluation.json'
    if evaluation.is_file():
        scores = json.loads(evaluation.read_text())
        out.setdefault('ate_online_m', scores['ate']['position_m'])
        out['orientation_deg'] = scores['ate'].get('orientation_deg')
        out['poses'] = scores['ate'].get('n_states')
    return {k: v for k, v in out.items() if v is not None}


def video(source, destination, width, crf, denoise=True):
    """A web-sized copy of one render; returns False when ffmpeg or the source is missing."""
    if not Path(source).is_file():
        return False
    filters = (['hqdn3d=2:2:5:5'] if denoise else []) + [f'scale={width}:-2']
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(source),
               '-vf', ','.join(filters), '-c:v', 'libx264', '-preset', 'slow',
               '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(destination)]
    return subprocess.run(command).returncode == 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True, help='Run directory holding map.ply')
    p.add_argument('--name', required=True, help='Asset name, e.g. ori or euroc')
    p.add_argument('--title', required=True, help='What the page calls it')
    p.add_argument('--note', default='', help='One line under the title')
    p.add_argument('--ply', type=Path, help='Default <run>/map.ply')
    p.add_argument('--points', type=int, default=600000, help='Budget for the browser')
    p.add_argument('--out', type=Path, default=ROOT / 'website/assets/maps')
    p.add_argument('--videos', type=Path, metavar='DIR',
                   help='Also transcode the build videos in DIR (./davio render --build) '
                        'down to web sizes')
    a = p.parse_args(argv)

    ply = a.ply or (a.run / 'map.ply')
    if not ply.is_file():
        p.error(f'{ply} not found; run `./davio export {a.run}` first')
    a.out.mkdir(parents=True, exist_ok=True)

    points, colours = read_ply(ply)
    print(f'{ply}: {len(points):,} points')
    points, colours = thin(points, colours, a.points)
    low, high = points.min(axis=0), points.max(axis=0)
    scale = np.maximum(high - low, 1e-6) / 65535.
    quantized = np.clip(np.round((points - low) / scale), 0, 65535).astype('<u2')

    binary = a.out / f'{a.name}.bin'
    with binary.open('wb') as stream:
        stream.write(quantized.tobytes())
        stream.write(colours.astype(np.uint8).tobytes())
    meta = dict(
        name=a.name, title=a.title, note=a.note,
        count=int(len(points)), bin=f'{a.name}.bin',
        # The page multiplies each uint16 by scale and adds offset; nothing else to parse.
        offset=[round(float(v), 6) for v in low], scale=[float(v) for v in scale],
        bounds=dict(min=[round(float(v), 3) for v in low],
                    max=[round(float(v), 3) for v in high]),
        trajectory=trajectory(a.run), numbers=numbers(a.run), source_run=str(a.run))
    (a.out / f'{a.name}.json').write_text(json.dumps(meta))
    print(f'  wrote {binary} ({binary.stat().st_size / 1e6:.1f} MB, '
          f'{len(points):,} points) and {a.out / f"{a.name}.json"}')

    if a.videos:
        a.video_dir = a.videos
        clips = a.out / 'video'
        clips.mkdir(exist_ok=True)
        for source, name, width, crf in (
                ('map_build_fly.mp4', 'walkthrough.mp4', 960, 30),
                ('map_build_top.mp4', 'build_top.mp4', 960, 26),
                ('map_build_orbit.mp4', 'build_orbit.mp4', 900, 28)):
            if video(a.video_dir / source, clips / name, width, crf):
                print(f'  wrote {clips / name} '
                      f'({(clips / name).stat().st_size / 1e6:.1f} MB)')
            else:
                print(f'  skipped {source}: not found, or ffmpeg failed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
