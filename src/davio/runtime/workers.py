"""Process isolation and bounded queues. Optional work never blocks the VIO queue."""
import multiprocessing as mp
from queue import Empty, Full
import time
import traceback
import numpy as np


class Worker:
    def __init__(self, target, args=(), capacity=1):
        ctx = mp.get_context('spawn')
        self.input = ctx.Queue(maxsize=capacity)
        self.output = ctx.Queue(maxsize=capacity + 8)
        self.process = ctx.Process(target=target, args=(self.input, self.output, *args), daemon=True)
        self.process.start()

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

    # A worker that has already left its loop still has CUDA and interpreter teardown to
    # do, and measurably needs more than a fifth of a second for it. Terminating mid-
    # finalize made pycolmap's signal handler print a SIGTERM stack trace on every run,
    # which reads as a crash and is not one. Still bounded: terminate follows regardless.
    GRACE_S = 2.

    def close(self):
        # Shutdown is bounded even if CUDA/native code is stuck. Completed maps
        # are already persisted; no mandatory final GPU job delays the caller.
        self.submit(None)
        self.process.join(timeout=self.GRACE_S)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.)
        for queue in (self.input, self.output):
            queue.cancel_join_thread()
            queue.close()


def feed_packet(backend, packet, bootstrap=None):
    for t, gyro, accel in packet['imu']:
        backend.feed_imu(t, np.asarray(gyro), np.asarray(accel))
    if bootstrap is not None:
        backend.initialize(bootstrap)
    backend.feed_camera(packet['t'], packet['image'])
    if not backend.initialized():
        return None
    state = backend.state()
    if not all(np.isfinite(np.asarray(state[k], float)).all()
               for k in ('t', 'q_GtoI', 'p', 'v', 'bg', 'ba', 'q_ItoC', 'p_IinC')):
        raise ValueError('OpenVINS emitted a non-finite state')
    # A repeated/stale state is not a fresh camera update or evidence of stability.
    if abs(float(state['t']) - packet['t']) > .002:
        return None
    return state


def shadow_loop(inbox, outbox, config, history, seed, bootstrap=None):
    from ..backends.openvins import OpenVinsBackend
    backend = None
    pending = dict(bootstrap) if bootstrap else None

    def feed(packet):
        nonlocal pending
        boot = None
        if pending is not None and packet['t'] >= float(pending['t']) - 1e-9:
            boot, pending = pending, None
        return feed_packet(backend, packet, boot)

    try:
        import cv2
        cv2.setRNGSeed(seed)
        backend = OpenVinsBackend(config)
        cv2.setRNGSeed(seed)
        warm = 0
        for packet in history:
            state = feed(packet)
            warm = warm + 1 if state is not None else 0
        while True:
            packet = inbox.get()
            if packet is None:
                break
            state = feed(packet)
            warm = warm + 1 if state is not None else 0
            outbox.put(dict(kind='state', state=state, warm_frames=warm,
                            timestamp=packet['t'], completed_wall=time.monotonic()))
    except Exception:
        outbox.put(dict(kind='error', error=traceback.format_exc()))
    finally:
        if backend is not None:
            backend.close()


def vision_loop(inbox, outbox, settings, output_dir):
    """One GPU model, startup first and dense mapping after estimator selection."""
    try:
        import torch
        from depth_anything_3.api import DepthAnything3
        from .assistance import CalibrationAssistant
        from davio_mapper.online import OnlineMapper
        torch.set_num_threads(int(settings['torch_threads']))
        torch.manual_seed(int(settings['seed']))
        np.random.seed(int(settings['seed']))
        import cv2
        cv2.setRNGSeed(int(settings['seed']))
        torch.backends.cudnn.benchmark = False
        model = DepthAnything3.from_pretrained(settings['model']).to(settings['device']).eval()
        assistant = CalibrationAssistant(settings['assistance'], settings.get('calibration_runtime'))
        mapper = None
        while True:
            task = inbox.get()
            if task is None:
                break
            started = time.monotonic()
            if task['kind'] == 'map' and started-task['arrival_wall'] > settings['mapping'].get('max_task_age_s', float('inf')):
                outbox.put(dict(kind='map', payload=dict(status='deferred', reason='stale mapping task'),
                                sensor_time=task['times'][-1], started_wall=started,
                                completed_wall=started, arrival_wall=task['arrival_wall']))
                continue
            try:
                with torch.inference_mode():
                    images = [np.repeat(im[..., None], 3, axis=2) if im.ndim == 2 else im for im in task['images']]
                    # K is None in the free calibration mode until the filter's own intrinsics
                    # exist; DA3 then predicts the intrinsics itself.
                    kw = {} if task['K'] is None else dict(
                        intrinsics=np.repeat(np.asarray(task['K'], float)[None], len(task['images']), axis=0))
                    conditioned = False
                    if task['kind'] == 'map' and settings['mapping'].get('da3_pose_conditioning', False) and task['K'] is not None:
                        # ScaRF-SLAM: condition DA3 on the VIO camera poses (world-to-camera,
                        # metric odometry frame) and intrinsics; DA3 aligns its output to them,
                        # so the returned depth is metric. A window without baseline cannot
                        # anchor the alignment and is inferred unconditioned instead.
                        poses = np.asarray(task['camera_poses'], float)
                        baseline = float(np.max(np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1)))
                        if np.isfinite(poses).all() and baseline >= float(settings['mapping']['min_baseline_m']):
                            kw['extrinsics'] = np.linalg.inv(poses)
                            conditioned = True
                    task['conditioned'] = conditioned
                    prediction = model.inference(image=images, process_res=int(settings['process_res']),
                                                 process_res_method='upper_bound_resize', **kw)
                if settings['device'].startswith('cuda'):
                    torch.cuda.synchronize()
                if task['kind'] == 'assist':
                    payload = assistant.propose(task, prediction)
                else:
                    if mapper is None:
                        mapper = OnlineMapper(settings['mapping'], output_dir)
                    payload = mapper.add(task, prediction)
                peak = (torch.cuda.max_memory_allocated() / 2 ** 30
                        if settings['device'].startswith('cuda') else None)
                outbox.put(dict(kind=task['kind'], payload=payload, started_wall=started,
                                completed_wall=time.monotonic(), sensor_time=task['times'][-1],
                                # Charged from receipt of the newest contributing frame, so
                                # the reported age includes time spent queued.
                                arrival_wall=task['arrival_wall'], peak_gpu_gb=peak))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                if task['kind'] == 'map':
                    # Do not continue from partially mutated graph/archive state.
                    # The parent retains the VIO trajectory and last committed map index.
                    raise RuntimeError('Mapping worker stopped: ' + str(exc)) from exc
                outbox.put(dict(kind=task['kind'], payload=dict(status='rejected', reason=str(exc)),
                                completed_wall=time.monotonic(), sensor_time=task['times'][-1],
                                arrival_wall=task['arrival_wall']))
    except Exception:
        outbox.put(dict(kind='error', error=traceback.format_exc()))
