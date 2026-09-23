# How it is put together

Three processes, two queues, and one rule: **nothing optional may block the pose stream.**

```
                 packets: (t, image, bracketing IMU, receipt wall time)
                                    |
  transport ----------------------->|
  data/replay.py    (dataset)       |
  data/realsense.py (camera)        v
                        +-----------------------------+
                        |  runtime/engine.py          |   owner process, single thread
                        |  Engine.step()              |
                        +--+-----------------------+--+
                           |                       |
        native OpenVINS    |                       |  capacity-1 queue, drop on full
        (backends/         v                       v
         openvins.py)  shadow worker         vision worker
                       an assisted           DA3: startup proposals
                       OpenVINS instance     and dense map windows
                       started from a        (davio_mapper/online.py)
                       bootstrap state
                           |                       |
                           +----------+------------+
                                      v
                          runs/<name>/  events.jsonl, *.tum, run.json, map/
```

## The engine — `src/davio/runtime/engine.py`

`Engine.step(packet)` is the whole online loop, and only one thread ever calls it. It:

1. rectifies the frame once for DA3 and keeps a bounded history,
2. feeds the native OpenVINS instance and publishes its state,
3. hands windows to the vision worker **if there is room**, and counts a drop if not,
4. applies the newest odometry→map correction the mapper has published,
5. writes `trajectory.tum` (odometry), `map_trajectory.tum` (corrected) and `events.jsonl`.

The engine is transport-independent: it consumes packets, not files. That is the whole
reason a RealSense needed 300 lines rather than a fork.

**The correction ramp.** A new odometry→map correction further than
`mapping.correction_jump_m` / `_deg` from the active one is slid in over
`correction_ramp_s` of sensor time, so the live stream never jumps. The loop-closed
keyframe trajectory carries the full step.

**Drops are recorded, not hidden.** `dropped_optional_tasks` and
`submitted_optional_tasks` in `run.json` say how many dense windows the GPU could not keep
up with. A slower machine drops more, and gets a different map.

The filter is unaffected: `step` feeds every packet to the backend whether or not the
driver slept first, so pacing changes when a packet arrives, not which ones or in what
order. Two runs of the same sequence that select the same estimator write byte-identical
`trajectory.tum`, however differently the mapper behaved.

## Workers — `src/davio/runtime/workers.py`

Separate `spawn` processes with bounded queues. `submit()` returns `False` rather than
blocking when the queue is full, which is how "optional work never blocks the VIO queue"
is actually enforced rather than merely intended. A worker that dies is reported, and a
candidate estimator that dies is a rejected candidate, not a dead run.

## Start-up assistance — `src/davio/init/`, `src/davio/runtime/assistance.py`

On a five-image DA3 window plus preintegrated IMU:

| | |
|---|---|
| `feedforward.py` | the feature-free linear system and its robust refinement; emits a full bootstrap state (rotation, gravity, velocity, biases) |
| `joint_calibration.py` | joint hand-eye + gyro-bias solve with a Schur certificate, for when the camera–IMU rotation is not supplied |
| `validation.py` | the later-window prediction test a candidate must pass |
| `preintegration.py`, `jpl.py` | IMU preintegration and the JPL quaternion conventions OpenVINS uses |

An accepted candidate materializes a **new** OpenVINS config
(`runtime/configuration.py::materialize`) and starts a fresh instance from the bootstrap
state, replaying buffered history to catch up. If it warms up in time it is selected and
the native instance is closed. **Nothing here ever writes into a running filter** — the
assisted estimator is a different instance, which is what keeps the comparison honest.

`calibration.mode` decides what the filter is told: `supplied` (the rig as shipped),
`extrinsics_free` (extrinsics from the initializer), `free` (intrinsics too).

## The dense back-end — `src/davio_mapper/`

| | |
|---|---|
| `online.py` | the mapper: window admission, DA3 inference, submap construction, the map index, snapshots |
| `metric.py` | placing a window metrically — residual scale acts on **depth only**, so metric camera baselines are never rescaled |
| `frame_scale.py` | per-frame depth scales from XFeat matches with the VIO poses fixed |
| `features.py` | XFeat + LighterGlue matching, and the appearance retrieval index |
| `graph.py` | the sparse factor graph and its solver |
| `sim3.py` | the Sim(3) and gravity-preserving node charts |
| `filtering.py` | depth confidence, edge and consistency filters |
| `admission.py` | the selective geometry-to-pose admission control (a recorded negative result; off in the frozen system) |
| `mapping.py` | turning an archived submap back into world points — the geometry convention lives here |

**The node chart is a setting.** `mapping.graph_nodes` selects `sim3` (7 variables),
`gravity` (5: world xyz, yaw about gravity, log depth scale — the frozen default) or
`scale` (1: poses fixed, depth scale only). A `gravity` chart carries a depth-scale
variable and therefore requires `scale_coupling`.

**A loop moves poses only if it earns it.** A verified loop closure changes map-frame poses
only when the declared odometry drift over the arc between its nodes is at least
`loop_min_drift_ratio` times the loop's own translation sigma. Below that it keeps only its
depth-scale observation. DA3-depth loops are wrong by roughly 10 % of their baseline, and
without this gate a 0.21 m constraint outvoted 0.06 m of odometry.

## Configuration

`config/system.yaml` holds every setting. **Change one per run with `--set`, not by editing
the file** — and `--set` may only retarget a key that already exists, so a typo is an error
rather than a silently inert setting that the recorded config would still show. Each run
records its full resolved config in `run.json`, so a run always says what it was.

`config/{euroc,ori,tumvi,vcu_rvi,realsense}/` hold per-rig OpenVINS calibration:
`estimator_config.yaml` plus the two Kalibr chains it points at. `config/phone/` is the
template `./davio rig` starts from; the rigs it writes go to `config/custom/`.

## A run directory

```
run.json                 the descriptor: config, state, timing, drops, provenance
events.jsonl             one line per pose, vision result, correction and diagnostic
trajectory.tum           odometry poses (body frame)
camera_trajectory.tum    the same, camera frame
map_trajectory.tum       poses through the live odometry->map correction
map_trajectory_final.tum the loop-closed keyframe trajectory
native_config/           the exact OpenVINS config the native instance ran
assisted_config/         ... and the assisted one, if a candidate was accepted
map/map_index.json       every submap: pose, scale, member frames, the graph's state
map/submaps/*.npz        the dense archives (regenerable input to a replay; prunable)
map/snapshots/           committed map versions at 25/50/75 % of the interval
map/active.ply           the resident geometry, rewritten as the map grows
map.ply                  the exported merged cloud (scripts/export_map.py)
evaluation*.json         scores, written by scripts/evaluate_run.py
surface*.json            surface scores against the Leica reference
replay.json              present only on a cached-submap replay
```

## Where ground truth enters

`src/davio/eval/` — and nowhere else in the runtime, the assistance or the mapper.
`data/groundtruth.py` decides *which* reference a dataset scores against and records that
choice with every number it produces. Two other places read it, both offline and both
explicit: `scripts/rerun_export.py` and `scripts/view_run.py --groundtruth`, which refuses
to run on a run still being written.

`run.json` carries `ground_truth_used: false` for every online run.
