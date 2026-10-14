# Evaluating a run

```bash
./davio run V1_01_easy                          # scored automatically when it finishes
./davio evaluate runs/V1_01_easy                # score again, e.g. against another reference
./davio map runs/V1_01_easy                     # re-map offline, with loop closure
./davio evaluate runs/V1_01_easy_remap          # the re-mapped trajectory's ATE
./davio evaluate runs/V1_01_easy_remap --surface   # the map itself against a reference scan
```

Ground truth is read only here, in `src/davio/eval/` and the scoring scripts. The runtime,
the initializer and the mapper never see it.

## Trajectory: `evaluation.json`

| field | what it is |
|---|---|
| `ate.position_m` | absolute trajectory error of the **live filter stream** (`trajectory.tum`), after a rigid SE(3) alignment — no scale is fitted, because the output is metric |
| `ate.orientation_deg` | orientation error under the same alignment; `null` when the reference's orientation is documented as unreliable |
| `ate_map`, `ate_map_final` | the same for the back-end's map-frame trajectory, causal and after the last correction |
| `rpe`, `rpe_map` | relative pose error over fixed distances |
| `first_output_sensor_span_s` | how long after the first frame the first pose was published — the initializer's job |
| `tracking_coverage` | fraction of the sequence with a published pose |
| `publication_age_p50_s`, `_p95_s` | time from a frame's arrival to its pose being published |
| `map_completion_age_p95_s` | how stale the map is: newest contributing frame to committed map |
| `dropped_optional_tasks` | mapping windows skipped because the GPU was behind; never a pose |
| `map_accepted_loops`, `map_correction_m` | loop closures, and how far they moved the map frame |
| `groundtruth` | which reference file produced these numbers, with its SHA-256 |

## After an offline re-map: `replay.json`

`./davio map` pushes the windows a run archived back through the back-end with no clock,
so nothing is dropped for being late and loop closures are free to act. Its report carries
`ate_raw_m` (the live stream, unchanged) and `ate_map_final_m` (after the loops), plus
`accepted_loops`, `node_shift_*` and the `residual_scale_*` of the depth. On ORI r01, for
example, the offline re-map closes 10 loops and takes the error from 0.207 m to 0.109 m.

Depth-filter settings (`depth_*`) only change a re-map with `--refilter`, which re-runs the
filter on the archived raw depth: `./davio map runs/X --refilter --set depth_max_m=12`.

## Surface: `surface.json`

`--surface` exports the map if it has not been, places it in the reference frame through
the **same rigid alignment** the trajectory score fitted — no surface ICP, no fitted scale —
and scores it against the sequence's reference scan:

| field | |
|---|---|
| accuracy | distance from each map point to the reference surface |
| completeness | distance from each *visible* reference point to the map |
| F-score @ 2 / 3 / 5 / 10 cm | the usual precision/recall combination at each threshold |

A reference scan ships with the EuRoC Vicon-room sequences (V1_*, V2_*) and with ORI.
Completeness is only measured over reference points some camera actually looked at: a
surface nobody pointed a camera at is not a hole in the reconstruction. That visible set is
built once per sequence from the reference trajectory, and its id is recorded in every
report, because two surface scores are only comparable on the same one.

## Which reference

EuRoC ships more than one reference and they disagree, so the choice is explicit and
recorded with every number (`--groundtruth` on both commands):

| variant | |
|---|---|
| `dataset` (default) | `mav0/state_groundtruth_estimate0/data.csv` as shipped |
| `openvins` | the corrected file OpenVINS distributes (`ov_data/euroc_mav/`); the paper's EuRoC rows use this |
| `auto` | the corrected file only where the shipped orientation is documented as wrong |

The difference is not academic: V1_01_easy's shipped orientation is off by a near-constant
5.5°, the same size as the error it would be used to measure.

**ORI** has one reference, `rXX_gt/poses_gt.txt`: LiDAR-registered camera poses, converted
to the body frame through the release's own camera–IMU extrinsics. **TUM-VI** uses the
shipped reference. **Your own recordings** and **RealSense** have none, and the scorer says
so rather than inventing one — look at them with `./davio view` instead.

## Why did it do that?

```bash
./davio shell python3 scripts/diagnose.py runs/V1_01_easy
```

tallies every decision a run made — initializer attempts, rejected windows, loop candidates
— grouped by the gate that stopped it, with a worked example of each.
