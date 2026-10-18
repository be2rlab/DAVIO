import time


def interval(ds, start_offset=0., seconds=None):
    """Shared camera/IMU support, shifted by ``start_offset``. No ground truth is read."""
    imu = ds.imu()
    stamps = ds.image_stamps()
    if not len(imu) or not len(stamps):
        raise ValueError('Sequence has no IMU or no images')
    start = max(float(stamps[0]) * 1e-9, float(imu[0].t)) + float(start_offset)
    end = min(float(stamps[-1]) * 1e-9, float(imu[-1].t))
    if seconds is not None:
        end = min(end, start + float(seconds))
    if end <= start:
        raise ValueError('Requested interval has no shared sensor support')
    return start, end


def replay(engine, ds, start, end, rate=1., blackout=None, stall=None):
    imu = ds.imu()
    color = getattr(ds, 'load_color_image', None)
    index = 0
    while index < len(imu) and imu[index].t < start:
        index += 1
    index = max(0, index - 1)  # One preceding sample for boundary interpolation.
    origin = time.monotonic()
    for stamp in ds.image_stamps():
        t = float(stamp) * 1e-9
        if t < start:
            continue
        if t > end:
            break
        if blackout is not None and blackout[0] <= t - start < blackout[0] + blackout[1]:
            continue          # IMU stays unconsumed and rides along with the next frame
        samples = []
        while index < len(imu):
            sample = imu[index]
            samples.append((float(sample.t), sample.gyro, sample.accel))
            index += 1
            if sample.t > t:
                break
        # The engine requires a bracketing sample; a truncated IMU tail ends the replay
        # rather than silently feeding an image the filter cannot propagate onto.
        if not samples or samples[-1][0] <= t:
            return
        arrival = origin + (samples[-1][0] - start) / rate if rate > 0 else time.monotonic()
        if rate > 0 and stall is not None and stall[0] <= t - start < stall[0] + stall[1]:
            arrival = max(arrival, origin + (stall[0] + stall[1]) / rate)
        if rate > 0:
            time.sleep(max(0., arrival - time.monotonic()))
        packet = dict(t=t, image=ds.load_filter_image(stamp), imu=samples,
                      arrival_wall=arrival)
        # A colour camera's frames are worth keeping for the MAP; the filter is handed the
        # grey image either way. Datasets without colour simply do not define this.
        if color is not None:
            packet['color'] = color(stamp)
        engine.step(packet)
