from types import SimpleNamespace

import numpy as np
import pytest

from davio.replay import frames, interval, replay


def packets(ds, start=None, end=None, rate=0.):
    out = []
    span = interval(ds)
    replay(SimpleNamespace(step=out.append), ds, span[0] if start is None else start, span[1] if end is None else end, rate)
    return out


def test_the_interval_is_what_camera_and_imu_both_cover(recording):
    ds = recording(frames=10)
    assert interval(ds) == pytest.approx((1., 1.45))
    assert interval(ds, start_offset=.2, seconds=.1) == pytest.approx((1.2, 1.3))
    assert interval(ds, seconds=60.) == pytest.approx((1., 1.45))
    ds.samples = ds.samples[:50]            # the IMU stops at 1.235 s
    assert interval(ds) == pytest.approx((1., 1.235))
    with pytest.raises(ValueError, match='no shared'):
        interval(ds, start_offset=1.)


def test_frames_come_once_and_in_order_with_their_colour_when_there_is_one(recording):
    ds = recording(frames=12)
    out = list(frames(ds, ds.image_stamps(), ahead=3))
    assert [s for s, _, _ in out] == ds.image_stamps().tolist() and sorted(ds.loaded) == ds.image_stamps().tolist()
    assert all(color is None for _, _, color in out)
    assert [int(image[0, 0]) for _, image, _ in out] == [20 + i for i in range(12)]
    ds.load_color_image = lambda stamp: np.full((2, 2, 3), 7, np.uint8)
    assert all((color == 7).all() for _, _, color in frames(ds, ds.image_stamps()))


def test_every_frame_arrives_with_the_imu_up_to_the_first_sample_after_it(recording):
    ds = recording(frames=10)
    out = packets(ds)
    assert [p['t'] for p in out] == pytest.approx([1. + .05 * i for i in range(10)])
    for packet in out:
        times = [t for t, _, _ in packet['imu']]
        assert times[-1] > packet['t'] and all(t <= packet['t'] for t in times[:-1])
        assert packet['image'].shape == (480, 752) and packet['color'] is None
    fed = [t for p in out for t, _, _ in p['imu']]
    assert fed == [s.t for s in ds.samples[1:1 + len(fed)]]     # no sample twice, none skipped
    assert fed[0] == .995                                       # the last sample before the first frame
    assert [p['t'] for p in packets(ds, start=1.2, end=1.3)] == pytest.approx([1.2, 1.25, 1.3])


@pytest.mark.parametrize('rate, period', [(1., .05), (2., .025)])
def test_frames_arrive_at_the_sensor_rate_times_the_given_rate(recording, rate, period):
    arrivals = [p['arrival_wall'] for p in packets(recording(frames=5), rate=rate)]
    np.testing.assert_allclose(np.diff(arrivals), period, atol=1e-6)


def test_the_replay_ends_where_the_imu_ends(recording):
    ds = recording(frames=10)
    start, end = interval(ds)
    ds.samples = ds.samples[:50]
    assert [p['t'] for p in packets(ds, start, end)] == pytest.approx([1., 1.05, 1.1, 1.15, 1.2])
