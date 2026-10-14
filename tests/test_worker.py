import contextlib
import queue
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from davio.worker import Worker, map_window, vision_loop

K = np.array([[100., 0., 4.], [0., 100., 3.], [0., 0., 1.]])


class Model:
    """In place of Depth Anything 3: keeps what it was called with."""

    def inference(self, **call):
        self.call = call
        n = len(call['image'])
        return SimpleNamespace(depth=np.ones((n, 6, 8)), conf=np.ones((n, 6, 8)), extrinsics=call['extrinsics'],
                               intrinsics=call['intrinsics'], processed_images=np.zeros((n, 6, 8, 3), np.uint8))


class Mapper:
    def add(self, task, prediction):
        self.prediction = prediction
        return dict(status='mapped')


@pytest.fixture
def torch(monkeypatch, cfg):
    """torch, faked: nothing of it is needed but its inference context."""
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=contextlib.nullcontext))
    cfg['device'] = 'cpu'


def window(span=.4, context=0):
    """A window of five grey images, and all the camera poses the model is to be given."""
    poses = np.tile(np.eye(4), (5 + context, 1, 1))
    poses[:5, 0, 3] = np.linspace(0., span, 5)
    poses[5:, 0, 3] = -1.
    task = dict(images=[np.full((6, 8), i, np.uint8) for i in range(5)], camera_poses=poses[:5], K=K,
                arrival_wall=time.monotonic(), context=None)
    if context:
        task['context'] = dict(images=[np.full((6, 8), 9, np.uint8)] * context, camera_poses=poses[5:])
    return task, poses


def test_the_model_sees_the_window_with_its_context_and_the_mapper_the_window_alone(torch, cfg):
    task, poses = window(context=2)
    model, mapper = Model(), Mapper()
    assert map_window(model, mapper, task, cfg) == dict(status='mapped', context_frames=2)
    assert [im.shape for im in model.call['image']] == [(6, 8, 3)] * 7 and [im[0, 0, 0] for im in model.call['image']] == [0, 1, 2, 3, 4, 9, 9]
    np.testing.assert_allclose(model.call['extrinsics'], np.linalg.inv(poses))       # world to camera
    np.testing.assert_array_equal(model.call['intrinsics'], np.tile(K, (7, 1, 1)))
    assert model.call['process_res'] == cfg['process_res'] and model.call['process_res_method'] == 'upper_bound_resize'
    assert mapper.prediction.depth.shape == (5, 6, 8) and mapper.prediction.extrinsics.shape == (5, 4, 4)
    assert map_window(model, mapper, window()[0], cfg)['context_frames'] == 0 and len(model.call['image']) == 5


def test_a_window_that_waited_too_long_or_barely_moves_never_reaches_the_model(torch, cfg):
    old, _ = window()
    old['arrival_wall'] -= cfg['mapping']['max_task_age_s'] + 1.
    still, _ = window(span=.01)
    assert map_window(None, None, old, cfg) == dict(status='deferred', reason='stale mapping task')
    assert map_window(None, None, still, cfg) == dict(status='deferred', reason='insufficient metric camera translation')


def late(inbox, outbox):
    time.sleep(.3)
    outbox.put(dict(ready=True))
    inbox.get()


def dying(inbox, outbox):
    outbox.put(dict(error='no model'))


def silent(inbox, outbox):
    pass


def test_a_worker_is_ready_when_its_process_says_so():
    worker, asked = Worker(late), time.monotonic()
    assert worker.ready() == dict(ready=True) and time.monotonic() - asked > .2
    worker.close()


def test_a_worker_whose_process_ends_first_gives_its_error_or_says_that_it_did_not_start():
    for target, expected in ((dying, 'no model'), (silent, 'the vision process did not start')):
        worker = Worker(target)
        assert worker.ready() == dict(error=expected)
        worker.close(grace_s=1.)


def test_the_vision_process_is_ready_once_the_model_is_loaded_and_before_it_takes_a_window(monkeypatch, cfg, tmp_path):
    loaded = []

    class DepthAnything3:
        @classmethod
        def from_pretrained(cls, name):
            loaded.append(name)
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        set_num_threads=lambda n: None, manual_seed=lambda n: None, backends=SimpleNamespace(cudnn=SimpleNamespace())))
    monkeypatch.setitem(sys.modules, 'depth_anything_3', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'depth_anything_3.api', SimpleNamespace(DepthAnything3=DepthAnything3))
    monkeypatch.setattr('davio.mapper.mapper.Mapper', lambda mapping, out: SimpleNamespace(flush=lambda: loaded.append('flushed')))
    inbox, outbox = queue.Queue(), queue.Queue()
    inbox.put(None)
    vision_loop(inbox, outbox, cfg, tmp_path)
    assert outbox.get_nowait() == dict(ready=True) and outbox.empty() and loaded == [cfg['model'], 'flushed']
