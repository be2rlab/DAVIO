# Running DAVIO on your own recording

```bash
./davio rig data/custom/my_walk          # once: the OpenVINS rig from the calibration
./davio run my_walk --dataset custom     # then any number of runs
./davio view runs/my_walk
```

## 1. Put the recording in EuRoC/ASL layout, under `data/`

```
data/custom/my_walk/
  mav0/cam0/data/<timestamp_ns>.png|jpg   one monocular camera, colour or grey
  mav0/cam0/sensor.yaml                   optional: intrinsics if nothing better exists
  mav0/imu0/data.csv                      timestamp_ns, gyro xyz [rad/s], accel xyz [m/s^2]
  mav0/imu0/sensor.yaml                   optional: IMU noise densities
  calibration/                            the calibration, if you have one (below)
```

It has to live under the checkout: the container mounts this folder and nothing else.
Frames are ordered by the timestamp in their **filename**, so a `data.csv` written out of
order does no harm. Colour frames give a colour map; the filter only ever sees grey.

A phone recorder such as `vi-recorder` writes this layout directly. A ROS bag needs one
conversion (see `scripts/convert_vcu_rvi.py` for a worked example).

## 2. Calibration

`./davio rig` takes the best calibration it finds, in this order, and prints what it used:

| source | files |
|---|---|
| **Kalibr** | `calibration/*camchain-imucam.yaml` — intrinsics, distortion, camera–IMU extrinsic and time offset; `calibration/imu0.yaml` for the IMU noise |
| **Basalt** | `calibration/calibration.json` (or `calib/calibration.json`) — the same, with Kannala-Brandt fisheye (`kb4`) supported |
| **the recorder** | `mav0/cam0/sensor.yaml` and `mav0/imu0/sensor.yaml` — usually intrinsics only, with an identity extrinsic |

Force one with `--calibration kalibr|basalt|recorder`. Kalibr's IMU noise densities are
inflated the way OpenVINS recommends (white noise ×2, random walk ×10) to cover what a
phone does not model; `--raw-imu-noise` keeps them as calibrated.

Pinhole-radtan and equidistant (fisheye) models both work end to end. A fisheye is
undistorted to a pinhole view for the depth network automatically.

Before a run, `./davio rig data/custom/my_walk --check` prints the recording's health —
frame rate, drops, gaps, IMU coverage, what is and is not calibrated — and writes nothing.

## 3. Without a calibration

It can still work. When the recording only carries a placeholder extrinsic, `./davio rig`
writes the right *prior* for a phone's rear camera from Android's frame conventions
(landscape capture: `R_CtoI = [[0,-1,0],[-1,0,0],[0,0,-1]]`), and two things refine it
while the run goes:

* DAVIO's **feed-forward initializer** estimates the camera–IMU rotation from the first
  seconds of imagery and IMU, with no prior at all — within 2.3° of the true value on a
  test phone, against 91° for the portrait matrix and 179° for identity;
* OpenVINS's **online calibration** refines intrinsics and the time offset. Turn it on
  for an uncalibrated camera:

```bash
./davio run my_walk --dataset custom \
    --set openvins.overrides.calib_cam_intrinsics=true \
    --set openvins.overrides.calib_cam_timeoffset=true
```

On a 212 s phone walk with nothing calibrated, that recovered the focal length (475 →
482 px) and a 15 ms time offset, and tracked 186 m with no jumps.

## 4. What decides whether it works

**Dropped frames, before anything else.** Two recordings from the same phone: at 8 Hz with
73% of frames dropped the trajectory wandered off by 120 m in 40 s; at 30 Hz with no drops
it tracked cleanly. The tracker cannot bridge a third of a second of handheld motion.
If a recording diverges, check the drop rate (`--check`) before changing settings.

Then, in order: calibrate with Kalibr; prefer a capture mode with a short rolling-shutter
readout (it is not modelled, and shows up as a ~20–30 ms time offset); avoid long stretches
of blank wall or sky while turning.

## Outdoors: the sky

The depth network gives sky pixels a finite depth, and a permissive filter lets them into
the map as pale-blue sheets. Sky is low-confidence and far, so these two settings remove
most of it while keeping most of the real geometry:

```bash
./davio run my_walk --dataset custom \
    --set mapping.depth_conf_quantile=0.3 --set mapping.depth_max_m=12.0
# or, on a run already made, without re-running the network:
./davio map runs/my_walk --refilter --set depth_conf_quantile=0.3 --set depth_max_m=12.0
```

Expect a strip along the path rather than a panorama: past about 12 m almost nothing passes
the mapper's multi-view consistency check, because a one-second window's baseline cannot
verify depth that far out.

## What a recording without a reference cannot give you

A number. There is no reference trajectory, so `./davio evaluate` says so instead of
inventing one. Look at the run with `./davio view`, or render it (`./davio render`).

## A live camera

For an Intel RealSense D435i/D455, `./davio calibrate` writes the rig from the device's own
factory calibration and `./davio camera` runs live — see [REALSENSE.md](REALSENSE.md).
