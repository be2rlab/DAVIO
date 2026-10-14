"""A recorded sequence fed to the engine as its sensors would deliver it."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time


def interval(ds, start_offset=0., seconds=None):
    """The time span both the camera and the IMU cover, from ``start_offset`` in, for ``seconds``."""
    imu, stamps = ds.imu(), ds.image_stamps()
    if not len(imu) or not len(stamps):
        raise ValueError('Sequence has no IMU or no images')
    start = max(float(stamps[0]) * 1e-9, float(imu[0].t)) + float(start_offset)
    end = min(float(stamps[-1]) * 1e-9, float(imu[-1].t))
    if seconds is not None:
        end = min(end, start + float(seconds))
    if end <= start:
        raise ValueError('Requested interval has no shared sensor support')
    return start, end


def frames(ds, stamps, ahead=4, workers=2):
    color = getattr(ds, 'load_color_image', None)

    def load(stamp):
        return ds.load_filter_image(stamp), (color(stamp) if color is not None else None)
    pending, source = deque(), iter(stamps)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            while True:
                while len(pending) <= ahead:
                    stamp = next(source, None)
                    if stamp is None:
                        break
                    pending.append((stamp, pool.submit(load, stamp)))
                if not pending:
                    return
                stamp, future = pending.popleft()
                yield (stamp, *future.result())
        finally:
            for _, future in pending:
                future.cancel()


def replay(engine, ds, start, end, rate=1.):
    """Feed the sequence between ``start`` and ``end`` at ``rate`` times sensor rate (0: as fast as
    the engine takes it)."""
    imu = ds.imu()
    index = 0
    while index < len(imu) and imu[index].t < start:
        index += 1
    index = max(0, index - 1)       # one sample before the start, for the filter's interpolation
    origin = time.monotonic()
    stamps = [s for s in ds.image_stamps() if start <= float(s) * 1e-9 <= end]
    for stamp, image, color in frames(ds, stamps):
        t = float(stamp) * 1e-9
        samples = []
        while index < len(imu):
            sample = imu[index]
            samples.append((float(sample.t), sample.gyro, sample.accel))
            index += 1
            if sample.t > t:
                break
        if not samples or samples[-1][0] <= t:      # the IMU ended before this frame
            return
        arrival = origin + (samples[-1][0] - start) / rate if rate > 0 else time.monotonic()
        if rate > 0:
            time.sleep(max(0., arrival - time.monotonic()))
        engine.step(dict(t=t, image=image, color=color, imu=samples, arrival_wall=arrival))
