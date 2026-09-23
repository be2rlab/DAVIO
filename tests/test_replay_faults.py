"""Injected replay faults must be what they claim: no lost IMU, and honest arrival times."""
from types import SimpleNamespace

import numpy as np

from davio.data.replay import replay


def _dataset(seconds=4., camera_hz=20., imu_hz=200.):
    stamps = (np.arange(0., seconds, 1. / camera_hz) * 1e9).astype(np.int64) + 10**9
    imu = [SimpleNamespace(t=1. + k / imu_hz, gyro=np.zeros(3), accel=np.zeros(3))
           for k in range(int((seconds + .2) * imu_hz))]
    return SimpleNamespace(imu=lambda: imu, image_stamps=lambda: stamps,
                           load_filter_image=lambda stamp: np.zeros((2, 2), np.uint8))


class _Engine:
    def __init__(self):
        self.packets = []

    def step(self, packet):
        self.packets.append(packet)


def _run(**faults):
    ds, engine = _dataset(), _Engine()
    replay(engine, ds, 1., 4.5, rate=0, **faults)
    return ds, engine.packets


def test_camera_blackout_withholds_frames_but_not_one_imu_sample():
    ds, clean = _run()
    _ds, dark = _run(blackout=(1.0, 0.5))
    times = np.array([p['t'] for p in dark])
    assert not np.any((times - 1. >= 1.0) & (times - 1. < 1.5)), 'a withheld frame was delivered'
    assert len(clean) - len(dark) == 10                       # 0.5 s at 20 Hz
    flatten = lambda packets: [s[0] for p in packets for s in p['imu']]
    assert flatten(dark) == flatten(clean), 'blackout must not lose or duplicate IMU'
    assert all(p['imu'][-1][0] > p['t'] for p in dark), 'every frame stays bracketed'
    after = next(p for p in dark if p['t'] - 1. >= 1.5)
    assert len(after['imu']) > 100 * .5, 'the gap IMU rides with the first frame after it'


def test_delivery_stall_charges_arrival_at_the_burst():
    ds, engine = _dataset(), _Engine()
    replay(engine, ds, 1., 3., rate=1000., stall=(.5, .5))
    stalled = [p for p in engine.packets if .5 <= p['t'] - 1. < 1.]
    free = [p for p in engine.packets if p['t'] - 1. < .5]
    assert stalled and free
    bursts = {round(p['arrival_wall'], 9) for p in stalled}
    assert len(bursts) == 1, 'every stalled packet arrives in the same burst'
    assert min(p['arrival_wall'] for p in stalled) > max(p['arrival_wall'] for p in free)
