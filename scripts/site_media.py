#!/usr/bin/env python3
"""A finished run as the project page's media: the Open3D grid, the Rerun video, posters, a slim Rerun recording
and the run's facts. Open3D and Chrome run on the host (pip install open3d playwright rerun-sdk==0.23.1; DISPLAY set):

    python3 scripts/site_media.py --run runs/web/ori --name ori --work renders --out renders/site
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Flying is close work: a finer voxel, every pixel and larger points than a wide shot needs. A long outdoor walk
# cannot afford that density; it is seen from further away anyway.
INDOOR_FLY = dict(voxel=.012, step=1, size=5., back=4.5, height=2.2)
# In corridors and small rooms the camera stays close behind the walker, or walls come between them.
CLOSE_FLY = dict(INDOOR_FLY, back=1.5, height=.6)
PRESETS = {
    'euroc': dict(title='EuRoC V1_01', note='Vicon room · drone · grey camera', ceiling=.85, rotate=0,
                  background='white', fly=INDOOR_FLY, zoom=-60.),
    'ori': dict(title='ORI r01', note='building interior · handheld · colour fisheye', ceiling=.88, rotate=0,
                background='white', fly=CLOSE_FLY, zoom=-60., turn=40.),
    'beach': dict(title='Phone walk', note='beach promenade · phone, uncalibrated', ceiling=0., rotate=90,
                  background='sky', look_down=1., fly=dict(voxel=.025, step=2, size=4., back=4.5, height=2.2), zoom=0.),
    'tumvi': dict(title='TUM-VI room1', note='motion-capture room · handheld · grey fisheye', ceiling=.85, rotate=0,
                  background='white', fly=CLOSE_FLY, zoom=-60., floor_cut=.3, crane=.78, ceiling_cut=2.7),
}
DURATION_S, HOLD_S, FPS = 40., 5., 30
RRD_TARGET_MB, RRD_LIMIT_MB, VIDEO_LIMIT_MB = 40, 95, 12


def check_run(run):
    meta = json.loads((Path(run) / 'run.json').read_text())
    if meta.get('state') != 'completed':
        raise SystemExit(f'{run}: the run {meta.get("state")}{": " + meta["error"] if meta.get("error") else ""}; '
                         'its video would show part of the sequence')
    if meta.get('vision_failed'):
        raise SystemExit(f'{run}: its mapping process failed, so there is no dense map to show')
    return meta


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def facts(run, preset):
    """What the page says about a run, from the run's own files; a number it does not have is left out."""
    run = Path(run)
    meta = read_json(run / 'run.json')
    out = dict(title=preset['title'], note=preset['note'], duration_s=round(meta['interval_end'] - meta['interval_start']))
    path = next((run / n for n in ('map_trajectory_final.tum', 'map_trajectory.tum') if (run / n).is_file()), None)
    if path is not None:
        p = np.loadtxt(path, ndmin=2)[:, 1:4]
        out['path_m'] = round(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), 1)
    submaps = read_json(run / 'map/map_index.json').get('submaps')
    if submaps:
        out['submaps'] = len(submaps)
    out['loops'] = int(meta.get('loops_closed', 0))
    ate = ((read_json(run / 'evaluation.json').get('ate') or {}).get('final') or {}).get('position_m')
    if ate is not None:
        out['ate_final_m'] = round(float(ate), 4)
    f1 = ((read_json(run / 'surface.json').get('thresholds') or {}).get('0.05') or {}).get('f1')
    if f1 is not None:
        out['f1_5cm'] = round(float(f1), 3)
    return out


def map_floor(run):
    """The floor under the run's cameras, from its whole exported map: what the viewer would take from its first
    submap alone can be a reflection (view_run.floor_height)."""
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
    from davio.mapper.fusion import read_ply
    from view_run import floor_height
    run = Path(run)
    path = next(run / n for n in ('map_trajectory_final.tum', 'map_trajectory.tum') if (run / n).is_file())
    return floor_height(np.asarray(read_ply(run / 'map.ply'))[:, 2], camera_z=float(np.median(np.loadtxt(path, ndmin=2)[:, 3])))


def check_size(path, limit_mb):
    size = Path(path).stat().st_size / 1e6
    if size > limit_mb:
        raise SystemExit(f'{path} is {size:.0f} MB, over the {limit_mb} MB the page allows')
    return size


def publish(made, page_path, limit_mb):
    """``made`` moved to the page's assets once it fits them; one that does not stays where it was made."""
    check_size(made, limit_mb)
    Path(page_path).parent.mkdir(parents=True, exist_ok=True)
    Path(made).replace(page_path)


def sh(*command):
    print('$ ' + ' '.join(map(str, command)), flush=True)
    subprocess.run([str(c) for c in command], check=True)


def web(source, video, poster, at_s, qualities=(28, 30, 32)):
    """The page's encode of a full-size video, at the best of ``qualities`` (h264 CRF) that fits the page, and a
    poster frame from it; encoded beside ``source`` and moved to ``video`` only once it fits."""
    made = Path(source).with_name(Path(video).stem + '.web.mp4')
    for crf in qualities:
        sh('ffmpeg', '-y', '-loglevel', 'error', '-i', source, '-vf', 'scale=1280:-2', '-c:v', 'libx264', '-preset',
           'slow', '-crf', crf, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', made)
        if made.stat().st_size / 1e6 <= VIDEO_LIMIT_MB:
            break
    publish(made, video, VIDEO_LIMIT_MB)
    sh('ffmpeg', '-y', '-loglevel', 'error', '-ss', at_s, '-i', video, '-frames:v', '1', '-q:v', '3', poster)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--name', required=True, choices=tuple(PRESETS))
    p.add_argument('--work', type=Path, required=True, help='full-size renders and recordings go in WORK/NAME')
    p.add_argument('--out', type=Path, required=True, help="the page's assets")
    p.add_argument('--skip', action='append', default=[], choices=('open3d', 'rerun', 'fly', 'hero', 'top', 'grid'),
                   help='a step already done; repeatable')
    a = p.parse_args(argv)
    check_run(a.run)
    preset, py = PRESETS[a.name], sys.executable
    work, out = a.work / a.name, a.out
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if 'open3d' not in a.skip:
        common = ['--run', a.run, '--out', work, '--duration', DURATION_S, '--hold-s', HOLD_S, '--fps', FPS]
        if preset.get('floor_cut') is not None:
            common += ['--floor-cut', preset['floor_cut']]
        fly = preset['fly']
        if 'fly' not in a.skip:
            sh(py, ROOT / 'scripts/render_build.py', *common, '--camera', 'fly', '--background', preset['background'],
               '--voxel-m', fly['voxel'], '--pixel-step', fly['step'], '--point-size', fly['size'],
               '--chase-look-down', preset.get('look_down', 0.), '--crane-cut', preset.get('crane', preset['ceiling']),
               '--chase-back', fly['back'], '--chase-height', fly['height'])
        for camera in ('hero', 'top'):
            if camera not in a.skip:
                sh(py, ROOT / 'scripts/render_build.py', *common, '--camera', camera, '--cut-ceiling', preset['ceiling'],
                   '--hold-turn', preset.get('turn', 120.))
        if 'grid' not in a.skip:
            sh(py, ROOT / 'scripts/compose_grid.py', '--run', a.run, '--out', work / 'grid.mp4', '--rotate', preset['rotate'],
               '--panel', 'camera', '--panel', f'fly={work / "map_build_fly.mp4"}',
               '--panel', f'hero={work / "map_build_hero.mp4"}', '--panel', f'top={work / "map_build_top.mp4"}')
        web(work / 'grid.mp4', out / f'{a.name}_open3d.mp4', out / f'{a.name}_open3d.jpg', DURATION_S * .8)
    if 'rerun' not in a.skip:
        shown = ['--rotate', preset['rotate'], '--floor', f'{map_floor(a.run):.3f}'] + [
            x for key in ('floor_cut', 'ceiling_cut') if key in preset for x in ('--' + key.replace('_', '-'), preset[key])]
        sh(py, ROOT / 'scripts/view_run.py', '--run', a.run, '--save', work / 'video.rrd', '--layout', 'video', *shown)
        sh(py, ROOT / 'scripts/record_rerun.py', '--rrd', work / 'video.rrd', '--out', work / 'rerun.mp4',
           '--duration', DURATION_S, '--hold-s', HOLD_S, '--fps', FPS, '--zoom', preset['zoom'])
        web(work / 'rerun.mp4', out / f'{a.name}_rerun.mp4', out / f'{a.name}_rerun.jpg', DURATION_S * .8)
        slim = work / f'{a.name}.rrd'
        for step in (8, 12, 16):
            slim.unlink(missing_ok=True)
            sh(py, ROOT / 'scripts/view_run.py', '--run', a.run, '--save', slim, '--frames', 'centre', '--pixel-step', step,
               '--light', *shown)
            if slim.stat().st_size / 1e6 <= RRD_TARGET_MB:
                break
        publish(slim, out / f'{a.name}.rrd', RRD_LIMIT_MB)
    (out / f'{a.name}.json').write_text(json.dumps(facts(a.run, preset), indent=1, ensure_ascii=False) + '\n')
    print(f'{a.name}: assets in {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
