import multiprocessing as mp
from queue import Empty, Full
import time
import traceback

import numpy as np

from . import context


class Worker:
    def __init__(self, target, args=(), capacity=1):
        ctx = mp.get_context('spawn')
        self.input = ctx.Queue(maxsize=capacity)
        self.output = ctx.Queue(maxsize=capacity + 8)
        self.process = ctx.Process(target=target, args=(self.input, self.output, *args), daemon=True)
        self.process.start()

    def ready(self, timeout_s=600.):
        until = time.monotonic() + timeout_s
        while self.process.is_alive() and time.monotonic() < until:
            try:
                return self.output.get(timeout=.1)
            except Empty:
                pass
        return (self.drain() or [dict(error='the vision process did not start')])[0]

    def submit(self, value):
        try:
            self.input.put_nowait(value)
            return True
        except Full:
            return False

    def drain(self):
        items = []
        while True:
            try:
                items.append(self.output.get_nowait())
            except Empty:
                return items

    def close(self, grace_s=20.):
        try:
            self.input.put(None, timeout=grace_s)
        except Full:
            pass
        self.process.join(timeout=grace_s)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.)
        left = self.drain()
        for queue in (self.input, self.output):
            queue.cancel_join_thread()
            queue.close()
        return left


def as_rgb(image):
    return np.repeat(image[..., None], 3, axis=2) if image.ndim == 2 else image


def map_window(model, mapper, task, cfg):
    """Depth for a window's images, given the filter's poses of them, then the mapper. Returns its report."""
    import torch
    mapping = cfg['mapping']
    if time.monotonic() - task['arrival_wall'] > mapping['max_task_age_s']:
        return dict(status='deferred', reason='stale mapping task')
    poses = np.asarray(task['camera_poses'], float)
    if float(np.max(np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1))) < float(mapping['min_baseline_m']):
        return dict(status='deferred', reason='insufficient metric camera translation')
    images, extra = [as_rgb(im) for im in task['images']], task.get('context')
    own = len(images)
    if extra:
        images = images + [as_rgb(im) for im in extra['images']]
        poses = np.concatenate((poses, np.asarray(extra['camera_poses'], float)))
    with torch.inference_mode():
        prediction = model.inference(image=images, extrinsics=np.linalg.inv(poses),
                                     intrinsics=np.repeat(np.asarray(task['K'], float)[None], len(images), axis=0),
                                     process_res=int(cfg['process_res']), process_res_method='upper_bound_resize')
    if len(images) > own:
        prediction = context.Head(prediction, own)
    if cfg['device'].startswith('cuda'):
        torch.cuda.synchronize()
    return dict(mapper.add(task, prediction), context_frames=len(images) - own)


def vision_loop(inbox, outbox, cfg, output_dir):
    try:
        import cv2
        import torch
        from depth_anything_3.api import DepthAnything3
        from .mapper.mapper import Mapper
        torch.set_num_threads(int(cfg['torch_threads']))
        torch.manual_seed(int(cfg['seed']))
        np.random.seed(int(cfg['seed']))
        cv2.setRNGSeed(int(cfg['seed']))
        torch.backends.cudnn.benchmark = False
        model = DepthAnything3.from_pretrained(cfg['model']).to(cfg['device']).eval()
        mapper = Mapper(cfg['mapping'], output_dir)
        outbox.put(dict(ready=True))
        while True:
            task = inbox.get()
            if task is None:
                mapper.flush()
                break
            started = time.monotonic()
            report = map_window(model, mapper, task, cfg)
            outbox.put(dict(report=report, sensor_time=task['times'][-1], arrival_wall=task['arrival_wall'],
                            started_wall=started, completed_wall=time.monotonic(),
                            peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if cfg['device'].startswith('cuda') else None))
    except Exception:       # a failed window ends the mapping, never the filter: the map is not continued from a half-made state
        outbox.put(dict(error=traceback.format_exc()))
