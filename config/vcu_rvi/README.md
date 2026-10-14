# VCU-RVI benchmark configuration

Handheld sequences from a Structure Core v2 (640x480 rectified colour camera + IMU).
Source: <https://vcu-rvi-dataset.github.io>.

- `estimator_config.yaml`, `kalibr_imu_chain.yaml`, `kalibr_imucam_chain.yaml`: the
  release's **own** calibration (`calibration/struct_core_v2.yaml`), transcribed into the
  files OpenVINS reads. Intrinsics `fx 459.357, fy 459.764, cx 332.695, cy 258.998`,
  distortion all zero, `T_imu_cam` from the release's `body_T_cam0`, IMU noise from its
  `acc_n/gyr_n/acc_w/gyr_w`, gravity from its `g_norm`. Nothing is estimated here.
- `downloads.json`: Google Drive ids that actually resolve.

## The download links on the dataset page are dead

Every per-file link printed on the release's download page returns **404** on every Drive
endpoint (`/file/d/<id>/view`, `uc?export=download`, `drive.usercontent.google.com`),
checked 2026-09-16. The shared folder itself still returns 200: the release was
reorganised into per-sequence subfolders and the old share ids were not carried over.

`downloads.json` records the ids read by traversing the live folder.

```
python3 scripts/fetch_vcu_rvi.py --list
python3 scripts/fetch_vcu_rvi.py --sequences hall3 --convert
```

The bags are large — `hall3` is 16 GB, `corridor3` is 10 GB — so `--convert` extracts and
then deletes each bag rather than requiring both to fit at once.

## Conversion

```
python3 scripts/convert_vcu_rvi.py --bag data/vcu_rvi/hall3/hall3.bag
```

Writes `data/vcu_rvi/<seq>/davio/{cam0,imu0}` in ASL layout. Only the colour stream and
the IMU are extracted: DAVIO's depth comes from DA3, so the bag's own depth stream is
skipped unless `--depth` is given. Timestamps are each message's **header** stamp, never
bag receipt time, which would put the camera and IMU on different clocks.

## Three things the release does not do the ROS way

All three were found by running it, and all three are handled in
`scripts/convert_vcu_rvi.py` and recorded in each sequence's `conversion.json`.

| | What the release publishes | What it must be |
|---|---|---|
| Accelerometer units | **g** — hall3 averages 1.016 | m/s², scaled by 9.80665 |
| Accelerometer sign | inverted — at rest hall3 reads `(-9.36, -0.05, 2.37)`, pointing *along* gravity | `sensor_msgs/Imu.linear_acceleration` is **specific force**, which at rest points *up* |
| Image encodings | OpenCV type strings `8UC3`, `16UC1` | not the ROS colour encodings, so the channel count is derived from the payload |

Measured effect on hall3, first 60 s, odometry only:

| | ATE | Orientation |
|---|---|---|
| as published | 167 m | 159° |
| units fixed (g → m/s²) | 975 m* | — |
| units **and** sign fixed | **0.66 m** | **12.2°** |

\* fixing the units alone makes it worse: it scales up an accelerometer that is still
pointing the wrong way. Both are needed, which is why neither is a silent default —
`--accel-sign` names the choice and `conversion.json` records it.

The units are decided from the data (`accel_scale`), because a release that got this wrong
once may not be consistent about it, and scaling already-SI data by 9.8 would be equally
broken. The sign is a property of the release, so it is a named constant rather than a
guess.

## Ground truth

`<seq>/<seq>_gt.csv`, TUM-format body poses at 120 Hz from the release's motion capture.

It is **already on the recording clock** and nothing is shifted: hall3's mocap starts at
214.03 s and its first camera frame is at 215.64 s, which is what starting the mocap 1.6 s
before the camera looks like, not an offset to correct. An offset fitted here would absorb
a real timing error, which is the one failure a reference must not hide.

Coverage is **partial by design**. The mocap volume is a single room and a sequence walks
out of it and back: hall3 carries 7812 samples — 71 s of tracking across 6 gaps in 356 s of
recording — in a 5.3 m box, along a 45 m path that returns to its start. Scoring drops an
estimate pose with no reference sample within its match tolerance rather than matching it
to a distant one, and `groundtruth_provenance()` reports `covered_s` and `n_gaps`. A
full-sequence ATE against this reference is therefore not comparable with a EuRoC ATE; the
release's own protocol scores start-to-end drift.

## Colour

This is the first dataset in this repository with a colour camera. The filter is still
handed grey (OpenVINS tracks on one channel), but the dense map is painted from the colour
frames; see `docs/VISUALIZE.md`. Before this, every map was grey.
