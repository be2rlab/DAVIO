#!/usr/bin/env python3
import csv
import hashlib
import json
import sys
from pathlib import Path
from rosbags.highlevel import AnyReader

seq = Path(sys.argv[1])
bag = next(seq.glob('*_bag'))
out = seq / 'davio'
if out.exists():
    raise SystemExit(f'{out} exists; conversion is never repeated in place')
(out / 'cam0/data').mkdir(parents=True)
(out / 'imu0').mkdir()
digest = hashlib.sha256()
for part in sorted(bag.glob('*.mcap')):
    with part.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            digest.update(chunk)
n_img, n_imu, stamps = 0, 0, []
with (out / 'imu0/data.csv').open('w', newline='') as f, AnyReader([bag]) as reader:
    w = csv.writer(f)
    w.writerow(['#timestamp [ns]', 'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]',
                'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]'])
    conns = [c for c in reader.connections if c.topic in ('/insta/cam0/image_raw/compressed', '/insta/imu/data_raw')]
    for conn, ts, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        stamp = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        if conn.topic.endswith('compressed'):
            if msg.format != 'jpeg':
                raise SystemExit(f'unexpected image format {msg.format}')
            (out / 'cam0/data' / f'{stamp}.jpg').write_bytes(bytes(msg.data))
            n_img += 1
            stamps.append(stamp)
        else:
            a, g = msg.linear_acceleration, msg.angular_velocity
            w.writerow([stamp, repr(g.x), repr(g.y), repr(g.z), repr(a.x), repr(a.y), repr(a.z)])
            n_imu += 1
info = dict(bag=str(bag), bag_sha256=digest.hexdigest(), images=n_img, imu_samples=n_imu,
            first_image_ns=min(stamps), last_image_ns=max(stamps),
            image_topic='/insta/cam0/image_raw/compressed (jpeg, header stamp)',
            imu_topic='/insta/imu/data_raw (SI units, header stamp)',
            note='verbatim payloads; no resampling, no rectification, no ground truth')
(out / 'conversion.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
