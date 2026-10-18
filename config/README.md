# Configuration

| | |
|---|---|
| `system.yaml` | every DAVIO setting: initialization, mapping, loop closure, OpenVINS overrides. Change one per run with `--set section.key=value`; a key that does not exist is an error |
| `euroc/`, `ori/`, `tumvi/`, `vcu_rvi/` | the OpenVINS rig for each dataset: `estimator_config.yaml` and the two Kalibr chains it reads |
| `realsense/` | written by `./davio calibrate` from the attached camera |
| `phone/` | the template `./davio rig` starts from for your own recordings |
| `custom/` | the rigs `./davio rig` writes, one folder per recording (git-ignored) |

Each run copies the rig and the fully resolved `system.yaml` into its own directory, so a
run always records exactly what it ran with.
