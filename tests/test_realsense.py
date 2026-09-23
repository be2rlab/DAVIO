import numpy as np
import pytest

from davio.data.realsense import RealSenseCamera, _interpolate_accel


def feed(camera, *, frames=(), gyro=(), accel=()):
    camera._frames.extend((t, np.full((4, 4), i, np.uint8), 100. + t)
                          for i, t in enumerate(frames))
    camera._gyro.extend((t, np.array([t, 0., 0.])) for t in gyro)
    camera._accel.extend((t, np.array([0., t, 9.81])) for t in accel)


def offline_camera():
    """A camera whose pipeline is never started; only the buffers and the assembly run."""
    camera = RealSenseCamera(fps=30)
    camera._pipeline = 'fake'
    return camera


def test_interpolate_accel_is_linear_and_refuses_extrapolation():
    samples = [(0., np.zeros(3)), (1., np.ones(3)), (2., 2 * np.ones(3))]
    assert _interpolate_accel(.25, samples) == pytest.approx([.25] * 3)
    assert _interpolate_accel(1.5, samples) == pytest.approx([1.5] * 3)
    assert _interpolate_accel(-.1, samples) is None      # before the first sample
    assert _interpolate_accel(2.1, samples) is None      # after the last


def test_packets_bracket_every_camera_time():
    """The invariant Engine.step checks: imu[-1][0] > t for every packet."""
    camera = offline_camera()
    ticks = [i / 200. for i in range(200)]               # 1 s of 200 Hz inertial data
    feed(camera, frames=[.1, .2, .3, .4], gyro=ticks, accel=ticks)
    camera.stop()
    packets = list(camera.packets(warmup_s=0.))
    assert [p['t'] for p in packets] == [.1, .2, .3, .4]
    for packet in packets:
        assert packet['imu'], 'a packet must carry inertial data'
        assert packet['imu'][-1][0] > packet['t'], 'inertial data must bracket the frame'
        assert all(np.isfinite(g).all() and np.isfinite(a).all() for _t, g, a in packet['imu'])


def test_a_duplicate_or_reordered_frame_never_breaks_the_clock():
    """Engine.step raises on a camera time that does not increase, so this must not emit one."""
    camera = offline_camera()
    ticks = [i / 200. for i in range(200)]
    # A duplicate timestamp and one frame delivered out of order, as a device stream does.
    feed(camera, frames=[.1, .2, .2, .15, .3], gyro=ticks, accel=ticks)
    camera.stop()
    times = [p['t'] for p in camera.packets(warmup_s=0.)]
    assert times == sorted(times) and len(times) == len(set(times))
    assert set(times) <= {.1, .15, .2, .3}
    # Frames that arrived together are reordered rather than dropped.
    assert times == [.1, .15, .2, .3]


def test_a_frame_that_arrives_after_a_newer_one_is_dropped():
    """Once a time has been published the engine cannot accept an older one; drop it."""
    camera = offline_camera()
    ticks = [i / 200. for i in range(200)]
    feed(camera, frames=[.2], gyro=ticks, accel=ticks)
    stream = camera.packets(warmup_s=0.)
    assert next(stream)['t'] == .2
    feed(camera, frames=[.15, .3])          # .15 is late; .3 is fine
    camera.stop()
    assert [p['t'] for p in stream] == [.3]


def test_inertial_samples_are_handed_over_once():
    """Consecutive packets overlap by exactly the one bracketing sample, never more."""
    camera = offline_camera()
    ticks = [i / 200. for i in range(200)]
    feed(camera, frames=[.1, .2, .3], gyro=ticks, accel=ticks)
    camera.stop()
    packets = list(camera.packets(warmup_s=0.))
    for before, after in zip(packets, packets[1:]):
        shared = {t for t, _g, _a in before['imu']} & {t for t, _g, _a in after['imu']}
        assert len(shared) <= 1
        assert after['imu'][0][0] >= before['imu'][-1][0]


def test_warmup_drops_the_settling_frames():
    camera = offline_camera()
    ticks = [i / 200. for i in range(400)]
    feed(camera, frames=[.1, .5, 1.1, 1.5], gyro=ticks, accel=ticks)
    camera.stop()
    times = [p['t'] for p in camera.packets(warmup_s=1.0)]
    assert times == [1.1, 1.5], 'frames within warmup_s of the first frame are dropped'


def test_a_frame_without_a_later_inertial_sample_is_held_back():
    """No bracketing sample yet means wait, not emit: the engine would raise otherwise."""
    camera = offline_camera()
    ticks = [i / 200. for i in range(40)]               # inertial data stops at 0.195 s
    feed(camera, frames=[.1, .5], gyro=ticks, accel=ticks)
    camera.stop()
    times = [p['t'] for p in camera.packets(warmup_s=0.)]
    assert times == [.1], 'the 0.5 s frame has nothing after it and must not be emitted'


def test_a_frame_outside_the_accelerometer_span_is_held_back():
    """Gyroscope and accelerometer start at different moments on a real unit."""
    camera = offline_camera()
    gyro = [i / 200. for i in range(200)]
    accel = [i / 100. for i in range(20, 100)]          # accelerometer starts at 0.2 s
    feed(camera, frames=[.1, .5], gyro=gyro, accel=accel)
    camera.stop()
    times = [p['t'] for p in camera.packets(warmup_s=0.)]
    assert times == [.5], 'the 0.1 s frame has no interpolatable accelerometer data'


def test_packets_refuses_to_run_before_the_device_is_open():
    with pytest.raises(RuntimeError, match='start'):
        next(RealSenseCamera().packets())


def test_start_explains_the_missing_dependency():
    import builtins
    real_import = builtins.__import__

    def without_pyrealsense2(name, *args, **kw):
        if name == 'pyrealsense2':
            raise ImportError('no module named pyrealsense2')
        return real_import(name, *args, **kw)

    builtins.__import__ = without_pyrealsense2
    try:
        with pytest.raises(ImportError, match='make image-realsense'):
            RealSenseCamera().start()
    finally:
        builtins.__import__ = real_import
