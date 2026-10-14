import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from davio import engine as engine_module
from davio.engine import Engine, blend


class Vision:
    """In place of the vision process: keeps the windows it is offered and hands back what is put in ``done``."""

    first = dict(ready=True)

    def __init__(self, target=None, args=(), capacity=1):
        self.tasks, self.done, self.busy, self.asked = [], [], False, 0
        self.process = SimpleNamespace(is_alive=lambda: True)

    def ready(self):
        self.asked += 1
        return self.first

    def submit(self, task):
        if not self.busy:
            self.tasks.append(task)
        return not self.busy

    def drain(self):
        done, self.done = self.done, []
        return done

    def close(self):
        return self.drain()


def answering(inbox, outbox, cfg, out):
    """A vision process that maps every window."""
    outbox.put(dict(ready=True))
    while inbox.get() is not None:
        outbox.put(mapped(np.eye(4)))


def mapped(correction):
    return dict(report=dict(status='mapped', submap='000000', accepted_loops=1, T_map_odom=np.asarray(correction).tolist()))


def shift(x):
    T = np.eye(4)
    T[0, 3] = x
    return T


@pytest.fixture
def start(cfg, root, openvins, tmp_path, monkeypatch):
    """Makes an engine on the EuRoC rig, the filter and the vision process faked."""
    monkeypatch.setattr(engine_module, 'Worker', Vision)
    cfg['shutdown_drain_s'] = 0.

    def make(**mapping):
        cfg['mapping'].update(mapping)
        return Engine(cfg, root / cfg['rigs']['euroc'], tmp_path / 'run', meta=dict(sequence='fake'))
    return make


def drive(engine, frames, first=0, color=False):
    """Frames at 20 Hz from t = 1 s; the filter gives a pose from the third on."""
    for i in range(first, first + frames):
        t = 1. + .05 * i
        engine.step(dict(t=t, image=np.zeros((480, 752), np.uint8), imu=[(t + .001, np.zeros(3), np.zeros(3))],
                         color=np.zeros((480, 752, 3), np.uint8) if color else None, arrival_wall=time.monotonic()))


def stream(engine, name):
    return np.loadtxt(engine.out / f'{name}.tum')


def events(engine, kind):
    return [e for e in map(json.loads, (engine.out / 'events.jsonl').read_text().splitlines()) if e['kind'] == kind]


def status(engine):
    return json.loads((engine.out / 'run.json').read_text())


def test_a_blend_goes_from_one_pose_to_the_other():
    target = shift(.3)
    target[:3, :3] = Rotation.from_euler('z', 10, degrees=True).as_matrix()
    np.testing.assert_allclose(blend(np.eye(4), target, 0.), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(blend(np.eye(4), target, 1.), target, atol=1e-12)
    np.testing.assert_allclose(blend(np.eye(4), target, 7.), target, atol=1e-12)
    half = blend(np.eye(4), target, .5)
    assert half[0, 3] == pytest.approx(.15) and Rotation.from_matrix(half[:3, :3]).magnitude() == pytest.approx(np.radians(5.))


def test_a_run_without_a_map_writes_every_pose_of_the_filter_and_its_status(start):
    engine = start(enabled=False)
    drive(engine, 8)
    engine.close()
    poses = stream(engine, 'trajectory')
    assert poses.shape == (6, 8)                # no pose for the two frames before the filter starts
    np.testing.assert_allclose(poses[:, 0], 1.1 + .05 * np.arange(6))
    np.testing.assert_allclose(poses[:, 1], .1 * np.arange(3, 9))
    np.testing.assert_array_equal(stream(engine, 'map_trajectory'), poses)
    np.testing.assert_array_equal(stream(engine, 'camera_trajectory'), poses)       # the fake camera sits on the body
    run = status(engine)
    assert run['state'] == 'completed' and run['error'] is None and run['poses'] == 6 and run['sequence'] == 'fake'
    assert run['first_sensor_time'] == 1. and run['last_sensor_time'] == pytest.approx(1.35)
    assert run['pose_age_p50_s'] >= 0. and run['config']['mapping']['enabled'] is False
    assert (engine.out / 'filter_config/estimator.yaml').is_file() and engine.vision is None


def test_a_run_in_which_the_filter_never_starts_has_failed(start):
    engine = start(enabled=False)
    drive(engine, 2)
    with pytest.raises(ValueError, match='increase'):       # a frame that is not newer is refused
        drive(engine, 1, first=1)
    engine.close()
    assert status(engine)['state'] == 'failed' and status(engine)['error'] == 'no initialization'


def test_a_filter_that_gives_no_pose_for_too_long_stops_the_run(start, openvins, cfg):
    openvins.silent_after = 5
    engine = start(enabled=False)
    with pytest.raises(RuntimeError, match='Tracking gap'):
        drive(engine, 100)
    assert engine.last_t - engine.last_pose > cfg['max_tracking_gap_s']
    engine.close(error='Tracking gap')
    assert status(engine)['state'] == 'failed' and status(engine)['poses'] == 3


def test_keyframes_half_a_second_apart_make_windows_of_five_that_share_two(start):
    engine = start()
    drive(engine, 75, color=True)
    first, second = engine.vision.tasks
    assert first['stamps'] == [1_100_000_000, 1_600_000_000, 2_100_000_000, 2_600_000_000, 3_100_000_000]
    assert second['stamps'][:2] == first['stamps'][-2:] and len(second['stamps']) == 5
    gaps = np.diff(second['stamps'])
    assert (gaps >= 500_000_000).all() and (gaps <= 550_000_000).all()      # the first frame after half a second
    np.testing.assert_allclose(first['times'], [1.1, 1.6, 2.1, 2.6, 3.1])
    np.testing.assert_allclose(first['camera_poses'][:, 0, 3], [.3, 1.3, 2.3, 3.3, 4.3])     # the filter's, at 2 m/s
    np.testing.assert_array_equal(first['K'], engine.view.K)
    assert [im.shape for im in first['images']] == [(480, 752)] * 5 and [im.shape for im in first['colors']] == [(480, 752, 3)] * 5
    assert first['context'] is None and engine.windows['submitted'] == 2 and engine.pending == 2


def test_a_window_the_worker_has_no_room_for_is_dropped_and_the_next_starts_a_keyframe_later(start):
    engine = start()
    engine.vision.busy = True
    drive(engine, 55)                           # six keyframes: two windows offered
    assert engine.vision.tasks == [] and engine.windows['dropped'] == 2 and len(events(engine, 'dropped')) == 2
    engine.vision.busy = False
    drive(engine, 10, first=55)
    assert engine.vision.tasks[0]['stamps'][0] == 2_100_000_000 and engine.windows['submitted'] == 1


def test_older_keyframes_that_see_the_same_place_join_a_window(start, openvins):
    openvins.step = .02                         # keyframes 20 cm apart
    engine = start()
    drive(engine, 110)
    assert [task['context'] is None for task in engine.vision.tasks] == [True, True, False]
    third = engine.vision.tasks[2]
    keyframes = sorted({s for task in engine.vision.tasks for s in task['stamps']})
    # at least 2 s older than the window, and not within 30 cm of one another
    assert third['context']['stamps'] == [keyframes[0], keyframes[2]]
    assert third['context']['camera_poses'].shape == (2, 4, 4) and len(third['context']['images']) == 2
    assert third['stamps'][0] - keyframes[2] >= 2_000_000_000


def test_a_small_correction_by_the_map_moves_the_live_poses_at_once(start):
    engine = start()
    drive(engine, 10)
    engine.vision.done.append(mapped(shift(.03)))
    drive(engine, 10, first=10)
    engine.close()
    moved = stream(engine, 'map_trajectory')[:, 1] - stream(engine, 'trajectory')[:, 1]
    np.testing.assert_allclose(moved, [0.] * 9 + [.03] * 9, atol=1e-9)      # from the frame after it arrived
    (event,) = events(engine, 'correction')
    assert event['ramped'] is False and event['jump_m'] == pytest.approx(.03) and event['submap'] == '000000'
    assert engine.loops == 1 and engine.windows['mapped'] == 1


def test_a_large_correction_is_slid_in_over_the_ramp(start, mapping):
    engine = start()
    drive(engine, 10)
    engine.vision.done.append(mapped(shift(.3)))        # arrives at t = 1.5 s
    drive(engine, 30, first=10)
    engine.close()
    odometry, live = stream(engine, 'trajectory'), stream(engine, 'map_trajectory')
    since = np.clip((live[:, 0] - 1.5) / mapping['correction_ramp_s'], 0., 1.)
    np.testing.assert_allclose(live[:, 1] - odometry[:, 1], .3 * since, atol=1e-9)
    assert events(engine, 'correction')[0]['ramped'] is True
    np.testing.assert_allclose(odometry[:, 1], .1 * np.arange(3, 41))          # the filter's own poses are not touched


def test_no_frame_is_taken_before_the_vision_process_is_ready(start):
    engine = start()
    assert engine.vision.asked == 1 and engine.poses == 0 and not engine.vision_failed
    assert events(engine, 'window') == []


def test_a_vision_process_that_cannot_start_leaves_a_run_without_a_map(start, monkeypatch):
    monkeypatch.setattr(Vision, 'first', dict(error='Traceback ...'))
    engine = start()
    drive(engine, 100)
    engine.close()
    assert engine.vision.tasks == [] and status(engine)['vision_failed'] is True and status(engine)['poses'] == 98
    assert events(engine, 'window')[0]['error'] == 'Traceback ...'


def test_a_failure_of_the_vision_process_ends_the_mapping_and_not_the_run(start):
    engine = start()
    drive(engine, 10)
    engine.vision.done.append(dict(error='Traceback ...'))
    drive(engine, 90, first=10)
    engine.close()
    assert engine.vision.tasks == [] and engine.vision_failed
    assert status(engine)['state'] == 'completed' and status(engine)['vision_failed'] is True and status(engine)['poses'] == 98
    assert events(engine, 'window')[0]['error'] == 'Traceback ...'


def test_at_the_end_the_window_in_the_vision_process_is_awaited_and_the_final_trajectory_written(
        cfg, root, openvins, tmp_path, monkeypatch):
    monkeypatch.setattr(engine_module, 'vision_loop', answering)        # in a real process
    engine = Engine(cfg, root / cfg['rigs']['euroc'], tmp_path / 'run')
    drive(engine, 50)
    (engine.out / 'map').mkdir()
    (engine.out / 'map/map_index.json').write_text(json.dumps(dict(submaps={'000000': dict(
        timestamp=2.1, T_odom_submap=np.eye(4).tolist(), T_map_submap=shift(.5).tolist())})))
    engine.close()
    run = status(engine)
    assert run['state'] == 'completed' and run['windows'] == dict(submitted=1, dropped=0, mapped=1, deferred=0, rejected=0)
    assert run['loops_closed'] == 1 and not engine.vision.process.is_alive()
    np.testing.assert_allclose(stream(engine, 'map_trajectory_final')[:, 1] - stream(engine, 'trajectory')[:, 1], .5)
