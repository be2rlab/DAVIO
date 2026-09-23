#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ROTATIONS = {0: None, 90: 'ROTATE_90_CLOCKWISE', 180: 'ROTATE_180',
             270: 'ROTATE_90_COUNTERCLOCKWISE'}


def trajectory_span(run):
    """(t0, t1) of the trajectory file render_build reads, in the order it looks for them."""
    path = next((run / n for n in ('map_trajectory_final.tum', 'map_trajectory.tum',
                                   'trajectory.tum') if (run / n).is_file()), None)
    if path is None:
        raise SystemExit(f'{run}: no trajectory file to take the timeline from')
    times = np.loadtxt(path, usecols=0, ndmin=1)
    return float(times[0]), float(times[-1])


def timeline(video, run, duration=None, hold_s=None, fps=None):
    """The render's frame -> sensor time mapping, from its sidecar or from its arguments."""
    sidecar = Path(str(video) + '.timeline.json')
    if sidecar.is_file():
        return json.loads(sidecar.read_text())
    if duration is None:
        raise SystemExit(f'{sidecar} not found: pass the --duration (and --hold-s) the video '
                         'was rendered with')
    t0, t1 = trajectory_span(run)
    return dict(t0=t0, t1=t1, fps=fps, total_frames=int(duration * fps),
                hold_frames=int((hold_s or 0.) * fps))


def sensor_time(i, line):
    """Exactly render_build's schedule: linear over the run, then held at the end."""
    return line['t0'] + (line['t1'] - line['t0']) * min(1., i / max(1, line['total_frames'] - 1))


def image_directory(run):
    """Where the run's camera frames live, from its own descriptor."""
    meta = json.loads((run / 'run.json').read_text())
    root = Path(meta['data_root'])
    if str(root).startswith('/workspace'):          # recorded inside the container
        root = ROOT / root.relative_to('/workspace')
    sequence = root / meta['sequence']
    for candidate in (sequence / 'davio/cam0/data', sequence / 'mav0/cam0/data',
                      sequence / 'cam0/data'):
        if candidate.is_dir():
            return candidate
    raise SystemExit(f'no camera frames found under {sequence}; pass --images')


def frame_index(directory):
    """(sorted timestamps in seconds, paths), from filenames -- the adapters' own order."""
    paths = [p for p in directory.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    stamps = np.array([int(p.stem) for p in paths], dtype=np.int64)
    order = np.argsort(stamps)
    return stamps[order] * 1e-9, [paths[i] for i in order]


def label(image, text, corner, scale):
    """Small caption on a translucent chip, so it reads on light and dark frames alike."""
    import cv2
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(scale * 2)))
    (w, h), base = cv2.getTextSize(text, font, scale, thickness)
    pad = int(8 * scale / 0.6)
    x, y = corner
    chip = image[y:y + h + base + 2 * pad, x:x + w + 2 * pad]
    chip[:] = (chip * 0.45 + np.array([20, 20, 20]) * 0.55).astype(np.uint8)
    cv2.putText(image, text, (x + pad, y + pad + h), font, scale, (245, 245, 245),
                thickness, cv2.LINE_AA)


def compose(video, run, out, line, images=None, rotate=0, min_height=720, crf=23,
            camera_label='camera', map_label='DAVIO map'):
    import cv2
    stamps, paths = frame_index(images or image_directory(run))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f'{video}: cannot be read')
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    expected = line['total_frames'] + line['hold_frames']
    if abs(count - expected) > 1:
        print(f'  warning: {video.name} has {count} frames, the timeline expects {expected}; '
              'check --duration/--hold-s', file=sys.stderr)

    # A long, thin render (a promenade from the side) would shrink the camera panel to a
    # thumbnail; pad the map to a readable height instead, in its own background colour.
    canvas_h = max(height, min_height)
    ok, first = capture.read()
    if not ok:
        raise SystemExit(f'{video}: no frames')
    background = first[2, 2].tolist()
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    probe = cv2.imread(str(paths[0]))
    if ROTATIONS[rotate]:
        probe = cv2.rotate(probe, getattr(cv2, ROTATIONS[rotate]))
    panel_w = int(round(probe.shape[1] * canvas_h / probe.shape[0]))
    total_w = (panel_w + 4 + width) // 2 * 2
    canvas_h = canvas_h // 2 * 2
    scale = 0.55 * canvas_h / 720

    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{total_w}x{canvas_h}', '-r', str(line['fps']), '-i', '-',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(out)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    cache, i = {}, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        t = sensor_time(i, line)
        j = int(np.clip(np.searchsorted(stamps, t), 1, len(stamps) - 1))
        j = j - 1 if abs(stamps[j - 1] - t) <= abs(stamps[j] - t) else j
        if j not in cache:
            cache.clear()
            camera = cv2.imread(str(paths[j]))
            if ROTATIONS[rotate]:
                camera = cv2.rotate(camera, getattr(cv2, ROTATIONS[rotate]))
            cache[j] = cv2.resize(camera, (panel_w, canvas_h), interpolation=cv2.INTER_AREA)
        canvas = np.empty((canvas_h, total_w, 3), np.uint8)
        canvas[:] = background
        canvas[:, :panel_w] = cache[j]
        canvas[:, panel_w:panel_w + 4] = (40, 40, 40)
        top = (canvas_h - height) // 2
        right = min(width, total_w - panel_w - 4)
        canvas[top:top + height, panel_w + 4:panel_w + 4 + right] = frame[:, :right]
        label(canvas, f'{camera_label}  t = {t - line["t0"]:5.1f} s', (12, 12), scale)
        label(canvas, map_label, (panel_w + 16, 12), scale)
        encoder.stdin.write(canvas.tobytes())
        i += 1
    capture.release()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit('ffmpeg failed')
    return dict(frames=i, size=[total_w, canvas_h], images=len(paths))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', type=Path, required=True, help='A render_build video')
    p.add_argument('--run', type=Path, required=True, help='The run it was rendered from')
    p.add_argument('--out', type=Path, help='Default <video>_with_camera.mp4')
    p.add_argument('--images', type=Path, help="Camera frame directory (default: the run's)")
    p.add_argument('--rotate', type=int, default=0, choices=sorted(ROTATIONS),
                   help='Degrees clockwise; 90 stands up a phone held in portrait')
    p.add_argument('--duration', type=float, help='The render\'s --duration, if no sidecar')
    p.add_argument('--hold-s', type=float, default=None, help='The render\'s --hold-s')
    p.add_argument('--min-height', type=int, default=720)
    p.add_argument('--crf', type=int, default=23)
    a = p.parse_args(argv)
    try:
        import cv2
    except ImportError:
        p.error('needs OpenCV (pip install opencv-python)')
    capture = cv2.VideoCapture(str(a.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.
    capture.release()
    line = timeline(a.video, a.run, a.duration, a.hold_s, round(fps))
    out = a.out or a.video.with_name(a.video.stem + '_with_camera.mp4')
    report = compose(a.video, a.run, out, line, a.images, a.rotate, a.min_height, a.crf)
    print(f'  wrote {out}  {report["size"][0]}x{report["size"][1]}, {report["frames"]} frames '
          f'from {report["images"]} camera images')
    return 0


if __name__ == '__main__':
    sys.exit(main())
