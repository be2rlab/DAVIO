# Running DAVIO on an Intel RealSense

DAVIO's engine is transport-independent: it consumes packets of
`(camera time, image, bracketing IMU samples, receipt time)`, not files. A RealSense is
therefore just another transport, and a live run produces exactly the same run directory as
a dataset run — same `run.json`, same `events.jsonl`, same `map/`, same viewer.

Supported: the **D400 series with a built-in IMU** — D435i, D455, D456. Anything else with a
global-shutter camera and a gyroscope/accelerometer pair will work if you can write a rig
file for it.

> None of the numbers in the paper come from a camera. This path exists so the system can be
> used, not to produce results; a handheld capture has no reference trajectory, so nothing
> is scored.

## 1. Build the camera image

```bash
make image-realsense          # davio/dense-gpu:latest + libusb + pyrealsense2
```

It is a separate image because `pyrealsense2` is only useful with a camera attached. The
container also has to be able to see the USB device; `scripts/docker.sh` passes
`--device /dev/bus/usb` when `DAVIO_REALSENSE=1`, which `./davio camera`, `./davio record`
and `./davio calibrate` all set for you.

Check the host sees the camera first — `rs-enumerate-devices`, or just `lsusb | grep Intel`.
If the host cannot see it, neither can the container.

## 2. Generate the rig from your own unit

```bash
./davio calibrate
```

This reads the camera's **own factory calibration** out of firmware and writes
`config/realsense/kalibr_imucam_chain.yaml` plus `config/realsense/device.json`. Every unit
is different, which is why that file is generated rather than checked in. What it reads:

| | |
|---|---|
| intrinsics, distortion | the chosen stream's video profile |
| `T_imu_cam` | extrinsics from the camera stream to the gyroscope stream: rotation camera→IMU and the camera origin in the IMU frame |

The factory extrinsics are a good **prior**, not a calibration — millimetre-accurate on
translation, roughly a degree on rotation. That is enough here: OpenVINS's online extrinsic
calibration is left on, and DAVIO's feed-forward initializer re-estimates the rotation from
scratch anyway. If you have a Kalibr result for the unit, put it in this file instead.

The IMU noise model in `config/realsense/kalibr_imu_chain.yaml` **is** checked in, and is
OpenVINS's own `rs_d455` values (Allan plots from the RPNG AR table dataset, white noise
×2 and bias random walk ×10). It is a good default, not a measurement of your unit. If
tracking is jittery, measure your own with `allan_ros2` or `imu_utils` over a few hours of
a stationary recording and replace the four numbers.

## 3. Run

```bash
./davio camera                          # writes runs/live
./davio camera --out runs/kitchen --seconds 120
./davio view runs/live --follow         # in a second terminal
```

Ctrl-C stops it: the engine drains its workers and writes a complete run directory.

Three device details are handled for you rather than left as a trap:

* **The IR projector is turned off.** Its dot pattern is what makes the depth sensor work,
  and it is also a field of fake corners that moves with the camera — the worst possible
  input to a KLT tracker. DAVIO never uses the RealSense depth stream (depth comes from
  DA3), so the emitter has no purpose here. `--emitter` turns it back on if you want to see
  what it does.
* **The left infrared imager is the default camera.** It is global-shutter and already
  factory-rectified. The RGB imager is rolling-shutter, which no filter in this repository
  models; `--stream color` selects it anyway.
* **Gyroscope and accelerometer arrive as separate streams at different rates.** A filter
  needs both at one instant, so the accelerometer is linearly interpolated onto gyroscope
  timestamps. Samples outside the accelerometer's span are dropped rather than
  extrapolated, which is also why the first frames of a session are discarded
  (`--warmup`, default 1 s — auto-exposure has not settled either).

`global_time_enabled` is set on every sensor, so the video and inertial streams share one
clock. `calib_cam_timeoffset` is consequently left off; if you see a systematic lag, turn
it on with `--set openvins.overrides.calib_cam_timeoffset=true`.

## 4. Record, then replay

A live run cannot be repeated: the same hand motion never happens twice, and the scheduler
drops a different set of dense windows every time. To compare two settings, record once and
replay:

```bash
./davio record data/realsense/desk --seconds 60
./davio run desk --dataset realsense --out runs/desk
./davio run desk --dataset realsense --out runs/desk_noloops --set mapping.loops_enabled=false
./davio view runs/desk
```

`./davio record` writes the ASL layout `davio.data.realsense.RealSenseDataset` reads:

```
cam0/data/<timestamp_ns>.png    frames, 8-bit grayscale
cam0/data.csv                   timestamp index
imu0/data.csv                   timestamp, gyro xyz (rad/s), accel xyz (m/s^2)
session.json                    device, stream settings, and the cam0 calibration block
```

`session.json` carries the calibration, so a recorded session stays interpretable even if
`config/realsense/` is later regenerated for a different unit.

`./davio camera --record data/realsense/desk` does both at once.

## Options worth knowing

| Flag | Default | |
|---|---|---|
| `--stream` | `infrared` | `color` uses the RGB imager (rolling shutter) |
| `--width --height --fps` | 848×480@30 | must be a mode the unit supports |
| `--gyro-hz --accel-hz` | 200 / 250 | D435i also offers 400 Hz gyro |
| `--serial` | first device | which unit, when more than one is attached |
| `--mode native` | `assist` | skip the feed-forward initializer |
| `--no-map` | off | odometry only; no GPU needed |
| `--emitter` | off | leave the IR projector on |

Options `./davio camera` does not define are passed straight through to
`scripts/run_realsense.py`; `python3 scripts/run_realsense.py --help` has the rest.

## When it does not work

| Symptom | Cause |
|---|---|
| `no device connected` inside the container | the USB bus was not passed through — use `./davio camera`, not a bare `scripts/docker.sh` |
| `pyrealsense2 is not installed` | you are in `davio/dense-gpu`, not `davio/realsense`; run `make image-realsense` |
| `kalibr_imucam_chain.yaml does not exist` | run `./davio calibrate` first |
| The filter never initializes | the dynamic initializer needs motion and parallax. Translate, do not only rotate, for the first few seconds |
| Tracking is jumpy, features sit on a dot grid | the IR projector is on. Do not pass `--emitter` |
| Poses drift badly but the images look fine | the IMU noise model is a D455 default; measure your unit's |
| The map is sparse and `dropped_optional_tasks` is high | the GPU cannot keep up with the window rate. Raise `--set mapping.keyframe_period_s=1.0` |

## Testing without a camera

`tests/test_realsense.py` drives the whole packet-assembly path with synthetic device
frames — bracketing, duplicate and out-of-order timestamps, the warm-up window, the
gyroscope/accelerometer span mismatch and clean shutdown — so the part of the driver that
has logic is covered with no hardware attached. `./davio test` runs it.
