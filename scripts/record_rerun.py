#!/usr/bin/env python3
"""A Rerun recording as a video: Rerun's own web viewer in headless Chrome, its timeline set frame by frame and each
frame captured, so the video plays smoothly whatever the renderer's speed. Runs on the host
(pip install playwright; it drives the system Chrome)."""
import argparse
import functools
import http.server
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np

VIEWER = 'https://unpkg.com/@rerun-io/web-viewer@0.23.1/index.js'      # imports ./re_viewer, which only unpkg resolves
PAGE = """<!doctype html><meta charset="utf-8"><style>html,body{margin:0;height:100%%;background:#0d1015;overflow:hidden}</style>
<body><script type="module">
import { WebViewer } from '%(viewer)s';
const viewer = new WebViewer();
await viewer.start('recording.rrd', document.body, { hide_welcome_screen: true, width: '100%%', height: '100%%',
                                                     render_backend: '%(backend)s' });
for (const panel of ['top', 'blueprint', 'selection', 'time']) viewer.override_panel_state(panel, 'hidden');
window.viewer = viewer;
</script>"""
GPU_FLAGS = {
    # Headless Chrome on Linux gets no WebGL 2 through ANGLE's Vulkan; through desktop OpenGL it gets the GPU.
    'gpu': ['--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=gl'],
    'software': ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
}


def frame_times(lo, hi, total, hold):
    """``total`` times from ``lo`` to ``hi``, evenly, then ``hold`` more at ``hi``."""
    return [lo + (hi - lo) * i / max(1, total - 1) for i in range(total)] + [hi] * hold


def blank(image, tolerance=4):
    """Whether a frame is one colour: what a viewer that could not draw leaves."""
    image = np.asarray(image, np.int16)
    return int(np.abs(image - image.reshape(-1, image.shape[-1])[0]).max()) <= tolerance


def step_to(page, timeline, t, settle_ms):
    """The viewer at time ``t``, drawn: two animation frames for it to draw, then ``settle_ms`` more."""
    page.evaluate(f"""async () => {{ const v = window.viewer, id = v.get_active_recording_id();
        v.set_current_time(id, '{timeline}', {t!r});
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))); }}""")
    page.wait_for_timeout(settle_ms)


def serve(directory):
    handler = functools.partial(QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rrd', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timeline', default='sensor_time')
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--duration', type=float, default=40., help='video seconds for the whole recording')
    p.add_argument('--hold-s', type=float, default=3.)
    p.add_argument('--backend', choices=('webgl', 'webgpu'), default='webgl')
    p.add_argument('--renderer', choices=tuple(GPU_FLAGS), default='gpu', help="Chrome's: the GPU, or SwiftShader")
    p.add_argument('--chrome', default=shutil.which('google-chrome') or 'google-chrome')
    p.add_argument('--settle-ms', type=int, default=60, help='after each time step, before the capture')
    p.add_argument('--crf', type=int, default=18)
    # Rerun 0.23's blueprint cannot place the 3D eye, so it is moved the way a person would: with the mouse. One drag,
    # started on the finished map and held for the whole video: down by --tilt to look into the rooms, then sideways by
    # --turn, a little every frame, so the map turns slowly while it is built.
    p.add_argument('--tilt', type=float, default=.17, help='downward drag before the first frame, in viewport heights')
    p.add_argument('--turn', type=float, default=.35, help='sideways drag over the whole video, in viewport widths')
    p.add_argument('--zoom', type=float, default=0., help='mouse wheel before the drag; negative zooms in')
    a = p.parse_args(argv)
    import cv2
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as site:
        site = Path(site)
        (site / 'recording.rrd').symlink_to(a.rrd.resolve())
        (site / 'index.html').write_text(PAGE % dict(viewer=VIEWER, backend=a.backend))
        server = serve(site)
        url = f'http://127.0.0.1:{server.server_address[1]}/index.html'
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=a.chrome, headless=True, args=GPU_FLAGS[a.renderer])
            page = browser.new_page(viewport=dict(width=a.width, height=a.height))
            page.goto(url)
            page.wait_for_function('window.viewer && window.viewer.get_active_recording_id()', timeout=120_000)
            # The file streams in: wait until the timeline stops growing.
            page.wait_for_function(f"""() => {{
                const v = window.viewer, id = v.get_active_recording_id(), r = v.get_time_range(id, '{a.timeline}');
                if (!r) return false;
                const now = performance.now();
                if (window._last !== r.max) {{ window._last = r.max; window._since = now; return false; }}
                return now - window._since > 3000; }}""", timeout=600_000, polling=250)
            lo, hi = page.evaluate(f"""() => {{ const v = window.viewer, id = v.get_active_recording_id();
                v.set_playing(id, false); const r = v.get_time_range(id, '{a.timeline}'); return [r.min, r.max]; }}""")
            times = frame_times(lo, hi, int(a.duration * a.fps), int(a.hold_s * a.fps))
            # The eye is framed on the scene as it is when first moved: on the finished map, not on its first submap.
            step_to(page, a.timeline, hi, a.settle_ms + 500)
            x0, y0 = a.width / 3., a.height / 2.            # the middle of the map view
            page.mouse.move(x0, y0)
            if a.zoom:
                page.mouse.wheel(0, a.zoom)
            page.mouse.down()
            page.mouse.move(x0, y0 + a.tilt * a.height, steps=12)
            encoder = subprocess.Popen(
                ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{a.width}x{a.height}',
                 '-r', str(a.fps), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(a.crf), '-pix_fmt', 'yuv420p',
                 '-movflags', '+faststart', str(a.out)], stdin=subprocess.PIPE)
            for i, t in enumerate(times):
                page.mouse.move(x0 + a.turn * a.width * i / max(1, len(times) - 1), y0 + a.tilt * a.height)
                step_to(page, a.timeline, t, a.settle_ms)
                image = cv2.imdecode(np.frombuffer(page.screenshot(type='png'), np.uint8), cv2.IMREAD_COLOR)
                if i == len(times) // 2 and blank(image):
                    encoder.kill()
                    raise SystemExit('the viewer drew nothing: no GPU context in headless Chrome? try --backend webgpu '
                                     'or --renderer software')
                encoder.stdin.write(image.tobytes())
                if i % 150 == 0:
                    print(f'  frame {i} of {len(times)}', flush=True)
            page.mouse.up()
            encoder.stdin.close()
            browser.close()
        server.shutdown()
    if encoder.wait() != 0:
        raise SystemExit('ffmpeg failed')
    print(f'wrote {a.out}: {len(times)} frames at {a.fps} fps')
    return 0


if __name__ == '__main__':
    sys.exit(main())
