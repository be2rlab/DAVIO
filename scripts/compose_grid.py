#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from compose_video import (ROTATIONS, frame_index, image_directory, label,  # noqa: E402
                           sensor_time, timeline)

TITLES = dict(camera='camera', fly='walkthrough', hero='three-quarter', top='overhead',
              side='side', orbit='orbit')


def panel_frame(i, master, panel):
    if i < master['total_frames']:
        fraction = i / max(1, master['total_frames'] - 1)
        return int(round(fraction * (panel['total_frames'] - 1)))
    last = panel['total_frames'] + panel['hold_frames'] - 1
    return min(panel['total_frames'] - 1 + (i - master['total_frames'] + 1), last)


class Reader:

    def __init__(self, path):
        import cv2
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise SystemExit(f'{path}: cannot be read')
        self.index, self.frame = -1, None

    def at(self, k):
        while self.index < k:
            ok, frame = self.capture.read()
            if not ok:
                break
            self.index, self.frame = self.index + 1, frame
        return self.frame


def fit(image, width, height, background):
    """`image` scaled to fit a width x height cell, centred on `background`."""
    import cv2
    cell = np.empty((height, width, 3), np.uint8)
    cell[:] = background
    scale = min(width / image.shape[1], height / image.shape[0])
    w, h = max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    y, x = (height - h) // 2, (width - w) // 2
    cell[y:y + h, x:x + w] = resized
    return cell


def parse_panel(text):
    """'camera' | 'name=path[:duration:hold]' -> dict."""
    if text == 'camera':
        return dict(name='camera')
    name, _, rest = text.partition('=')
    if not rest:
        raise SystemExit(f'--panel {text!r}: expected NAME=PATH[:DURATION:HOLD] or camera')
    parts = rest.split(':')
    path = Path(parts[0])
    duration = float(parts[1]) if len(parts) > 1 else None
    hold = float(parts[2]) if len(parts) > 2 else None
    return dict(name=name, path=path, duration=duration, hold=hold)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--panel', action='append', required=True,
                   help="'camera', or NAME=PATH[:DURATION:HOLD]; row by row")
    p.add_argument('--cols', type=int, default=2)
    p.add_argument('--cell', default='958x538',
                   help='Size of one panel, WxH; the default makes a 2x2 grid exactly 1920x1080')
    p.add_argument('--rotate', type=int, default=0, choices=sorted(ROTATIONS),
                   help='Degrees clockwise for the camera panel')
    p.add_argument('--images', type=Path, help="Camera frame directory (default: the run's)")
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--crf', type=int, default=23)
    a = p.parse_args(argv)
    import cv2

    cell_w, cell_h = (int(v) for v in a.cell.lower().split('x'))
    panels = [parse_panel(t) for t in a.panel]
    videos = [q for q in panels if q['name'] != 'camera']
    if not videos:
        p.error('at least one video panel is needed to set the timeline')
    for q in videos:
        capture = cv2.VideoCapture(str(q['path']))
        fps = round(capture.get(cv2.CAP_PROP_FPS) or 30)
        ok, first = capture.read()
        capture.release()
        if not ok:
            p.error(f"{q['path']}: no frames")
        q['line'] = timeline(q['path'], a.run, q['duration'], q['hold'], fps)
        q['background'] = first[2, 2].tolist()
        q['reader'] = Reader(q['path'])
    master = max((q['line'] for q in videos),
                 key=lambda line: line['total_frames'] + line['hold_frames'])
    if len({round(q['line']['t0'], 3) for q in videos}) > 1:
        p.error('the panels come from different runs (their timelines start at different times)')

    stamps, paths = (frame_index(a.images or image_directory(a.run))
                     if any(q['name'] == 'camera' for q in panels) else (None, None))
    rows = -(-len(panels) // a.cols)
    gap = 4
    width = (a.cols * cell_w + (a.cols - 1) * gap) // 2 * 2
    height = (rows * cell_h + (rows - 1) * gap) // 2 * 2
    scale = 0.5 * cell_h / 540

    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{width}x{height}', '-r', str(master['fps']), '-i', '-',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', str(a.crf), '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(a.out)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    frames = master['total_frames'] + master['hold_frames']
    camera_cache = {}
    for i in range(frames):
        t = sensor_time(i, master)
        canvas = np.empty((height, width, 3), np.uint8)
        canvas[:] = (60, 60, 60)
        for n, q in enumerate(panels):
            r, c = divmod(n, a.cols)
            x, y = c * (cell_w + gap), r * (cell_h + gap)
            if q['name'] == 'camera':
                j = int(np.clip(np.searchsorted(stamps, t), 1, len(stamps) - 1))
                j = j - 1 if abs(stamps[j - 1] - t) <= abs(stamps[j] - t) else j
                if j not in camera_cache:
                    camera_cache.clear()
                    image = cv2.imread(str(paths[j]))
                    if ROTATIONS[a.rotate]:
                        image = cv2.rotate(image, getattr(cv2, ROTATIONS[a.rotate]))
                    camera_cache[j] = fit(image, cell_w, cell_h, (18, 18, 18))
                cell = camera_cache[j].copy()
                label(cell, f'camera  t = {t - master["t0"]:5.1f} s', (10, 10), scale)
            else:
                frame = q['reader'].at(panel_frame(i, master, q['line']))
                cell = fit(frame, cell_w, cell_h, q['background'])
                label(cell, TITLES.get(q['name'], q['name']), (10, 10), scale)
            canvas[y:y + cell_h, x:x + cell_w] = cell[:min(cell_h, height - y),
                                                      :min(cell_w, width - x)]
        encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit('ffmpeg failed')
    print(f'  wrote {a.out}  {width}x{height}, {frames} frames at {master["fps"]} fps '
          f'({frames / master["fps"]:.0f} s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
