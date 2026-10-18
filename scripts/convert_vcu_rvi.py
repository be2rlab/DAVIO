#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# The release's own topic names (calibration/struct_core_v2.yaml).
COLOR_TOPICS = ('/cam0/color', '/camera/color/image_raw', '/cam0/image_raw')
DEPTH_TOPICS = ('/cam0/depth', '/camera/depth/image_rect_raw')
IMU_TOPICS = ('/imu', '/imu0', '/camera/imu')


def digest(path, limit=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        read = 0
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
            read += len(block)
            if limit and read >= limit:
                break
    return h.hexdigest()


STANDARD_GRAVITY = 9.80665


# This release publishes the accelerometer with the opposite sign to the ROS convention:
# sensor_msgs/Imu.linear_acceleration is SPECIFIC FORCE, which at rest points UP, away from
# gravity. Here it points down --- at rest hall3 reads (-9.36, -0.05, 2.37) whose vertical
# component opposes the IMU's own stated up axis. A filter that trusts the ROS convention
# therefore initialises gravity upside down; measured on hall3, 60 s: 159 deg of orientation
# error and 167 m of ATE as published, 36 deg and 0.95 m once negated. It is a property of
# the release, not something to detect per file, so it is a named default and is recorded
# in conversion.json rather than applied silently.
ACCEL_SIGN = -1.0


def accel_scale(magnitudes):
    mean = float(np.mean(magnitudes))
    if 0.5 <= mean <= 2.0:
        return STANDARD_GRAVITY, f'magnitude averages {mean:.3f}: units are g, scaled to m/s^2'
    if 5.0 <= mean <= 15.0:
        return 1.0, f'magnitude averages {mean:.3f}: already m/s^2, unscaled'
    raise SystemExit(
        f'accelerometer magnitude averages {mean:.3f}, which is neither g (~1) nor '
        f'm/s^2 (~9.81); refusing to guess the units')


def pick(available, preferred, kind):
    for name in preferred:
        if name in available:
            return name
    raise SystemExit(f'no {kind} topic in the bag; it has: {sorted(available)}')


def image_array(message):
    import numpy as np
    encoding = message.encoding.lower()
    dtype = np.uint16 if ('16' in encoding or encoding in ('mono16',)) else np.uint8
    data = np.frombuffer(message.data, dtype=dtype)
    pixels = int(message.height) * int(message.width)
    if pixels <= 0 or data.size % pixels:
        raise ValueError(f'{message.encoding}: {data.size} values do not fill '
                         f'{message.height}x{message.width}')
    channels = data.size // pixels
    frame = data.reshape(int(message.height), int(message.width), channels)
    if channels == 1:
        frame = frame[:, :, 0]
    elif encoding.startswith('rgb'):
        frame = frame[:, :, [2, 1, 0]]          # OpenCV writes BGR
    elif channels == 4:
        frame = frame[:, :, :3]
    return np.ascontiguousarray(frame)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bag', type=Path, required=True)
    p.add_argument('--out', type=Path, help='Default: <bag directory>/davio')
    p.add_argument('--depth', action='store_true',
                   help="Also extract the bag's depth stream. DAVIO never reads it; this "
                        'is for inspection only and roughly doubles the output size')
    p.add_argument('--accel-sign', type=float, default=ACCEL_SIGN, choices=(1.0, -1.0),
                   help='Sign applied to the accelerometer. The release publishes the '
                        'opposite of the ROS specific-force convention, so this defaults '
                        'to -1; pass 1 to keep the bag values as they are')
    p.add_argument('--grayscale', action='store_true',
                   help='Store single-channel frames. Smaller, but the dense map then has '
                        'no colour')
    a = p.parse_args(argv)
    if not a.bag.is_file():
        p.error(f'{a.bag} does not exist')
    out = a.out or a.bag.parent / 'davio'
    if (out / 'conversion.json').is_file():
        print(f'{out} already converted; delete it to redo')
        return 0

    import cv2
    import numpy as np
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.ROS1_NOETIC)

    staging = Path(tempfile.mkdtemp(dir=out.parent, prefix='.convert-'))
    try:
        (staging / 'cam0/data').mkdir(parents=True)
        (staging / 'imu0').mkdir(parents=True)
        if a.depth:
            (staging / 'depth0/data').mkdir(parents=True)
        frames = (staging / 'cam0/data.csv').open('w')
        frames.write('#timestamp [ns],filename\n')
        # The IMU is buffered rather than streamed: its units cannot be decided until the
        # whole stream has been seen (see accel_scale).
        samples = []
        counts = dict(color=0, depth=0, imu=0)
        span = {}
        with Reader(a.bag) as reader:
            available = {c.topic for c in reader.connections}
            color_topic = pick(available, COLOR_TOPICS, 'colour image')
            imu_topic = pick(available, IMU_TOPICS, 'IMU')
            depth_topic = None
            if a.depth:
                depth_topic = pick(available, DEPTH_TOPICS, 'depth image')
            wanted = {color_topic, imu_topic} | ({depth_topic} if depth_topic else set())
            connections = [c for c in reader.connections if c.topic in wanted]
            print(f'topics: colour={color_topic} imu={imu_topic}'
                  + (f' depth={depth_topic}' if depth_topic else ''), flush=True)
            for connection, _receipt, raw in reader.messages(connections=connections):
                message = typestore.deserialize_ros1(raw, connection.msgtype)
                # The message's OWN stamp, not the bag's receipt time.
                stamp = int(message.header.stamp.sec) * 10**9 + int(message.header.stamp.nanosec)
                key = 'imu' if connection.topic == imu_topic else (
                    'depth' if connection.topic == depth_topic else 'color')
                lo, hi = span.get(key, (stamp, stamp))
                span[key] = (min(lo, stamp), max(hi, stamp))
                counts[key] += 1
                if key == 'imu':
                    g, b = message.angular_velocity, message.linear_acceleration
                    samples.append((stamp, g.x, g.y, g.z, b.x, b.y, b.z))
                    continue
                frame = image_array(message)
                if key == 'color':
                    if a.grayscale and frame.ndim == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(str(staging / 'cam0/data' / f'{stamp}.png'), frame)
                    frames.write(f'{stamp},{stamp}.png\n')
                else:
                    cv2.imwrite(str(staging / 'depth0/data' / f'{stamp}.png'), frame)
                if sum(counts.values()) % 2000 == 0:
                    print(f'  {counts}', flush=True)
        frames.close()
        if counts['color'] < 2 or counts['imu'] < 2:
            raise SystemExit(f'bag yielded too little data: {counts}')
        table = np.asarray(samples, float)
        scale, units = accel_scale(np.linalg.norm(table[:, 4:7], axis=1))
        scale *= a.accel_sign
        print(f'accelerometer: {units}; sign {a.accel_sign:+.0f}', flush=True)
        with (staging / 'imu0/data.csv').open('w') as imu:
            imu.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],'
                      'w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],'
                      'a_RS_S_z [m s^-2]\n')
            for row in samples:
                imu.write(f'{int(row[0])},{row[1]!r},{row[2]!r},{row[3]!r},'
                          f'{row[4] * scale!r},{row[5] * scale!r},{row[6] * scale!r}\n')

        def rate(key):
            lo, hi = span[key]
            return (counts[key] - 1) / ((hi - lo) * 1e-9) if hi > lo else None
        record = dict(
            bag=str(a.bag.relative_to(ROOT)) if a.bag.is_relative_to(ROOT) else str(a.bag),
            bag_bytes=a.bag.stat().st_size,
            bag_sha256_first_64mb=digest(a.bag, 64 << 20),
            topics=dict(color=color_topic, imu=imu_topic, depth=depth_topic),
            counts=counts,
            color_hz=rate('color'), imu_hz=rate('imu'),
            first_image_ns=span['color'][0], last_image_ns=span['color'][1],
            first_imu_ns=span['imu'][0], last_imu_ns=span['imu'][1],
            grayscale=a.grayscale,
            accel_scale=scale, accel_units=units, accel_sign=a.accel_sign,
            note='verbatim pixels; no resampling, no rectification, no ground truth. '
                 'Timestamps are message header stamps, not bag receipt times.')
        (staging / 'conversion.json').write_text(json.dumps(record, indent=1))
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(out, ignore_errors=True)
        staging.rename(out)
        staging = None
        print(json.dumps(record, indent=1))
        print(f'\nwrote {out}')
        return 0
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
