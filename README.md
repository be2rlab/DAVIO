# DAVIO

Dense metric mapping from one camera and an IMU, online, with loop closure.

## Install

Linux, Docker with the NVIDIA container toolkit, an NVIDIA GPU (6 GB for the base model) and about 30 GB of disk.

```bash
git clone --recursive <this repository> && cd DAVIO
make setup                    # image, OpenVINS, its binding, model weights, XFeat (about an hour, once)
make euroc                    # V1_01_easy and V1_02_medium into data/euroc
make ori                      # r01 into data/ori
```

## Run

```bash
./davio run V1_01_easy                          # online at sensor rate, then exported and scored
./davio run r01 --dataset ori --rate 0.3        # ORI, slowed down
./davio run V1_01_easy --weights thirdparty/da3_weights/giant    # another checkpoint (make weights MODEL=giant)
```

A run directory holds the trajectories (`trajectory.tum` from the filter, `map_trajectory.tum` as corrected by the map while it ran, `map_trajectory_final.tum` by the final map), the submaps
(`map/`), the fused cloud (`map.ply`) and the scores (`evaluation.json`, `surface.json`).

```bash
./davio map RUN --no-correction --out RUN/map_fused.ply     # the map without the export-time correction
./davio score RUN                                           # score again
./davio test
```

## See it

```bash
./davio run V1_01_easy --viz                    # the run with a viewer beside it
./davio view runs/V1_01_easy                    # a run, finished or still being written
./davio view runs/V1_01_easy --save tour.rrd    # a file, to open with `rerun tour.rrd`
```

Open http://localhost:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy (the viewer prints it). Without the `url`
part, or with its `+` unescaped, the page shows Rerun's welcome screen and no run. The viewer is [Rerun](https://rerun.io) 0.23: the host's when it has `rerun-sdk` (`pip install rerun-sdk==0.23.1`), the
image's otherwise. It reads only what the run has written, so watching cannot change a run. On screen: the map's trajectory (cyan)
over the filter's (orange), the camera, the dense submaps as the map places them (a loop closure moves them) on a 1 m floor
grid, keyframes (white), loop closures (magenta), the newest image and its metric depth, the mapper's latency, scale and
corrections per window, and a card with the run's progress and, once scored, its trajectory error and surface F-score. The map
is coloured by height above the floor, shaded by the image, when the camera is grey, and in its own colours when it is not
(`--color height|rgb`); `--background light` for a bright one. It is the map before the export-time correction
(`./davio map RUN --no-correction`). `--frames centre` or a larger `--pixel-step` makes it
lighter. From another machine, forward both ports: `ssh -L 9090:localhost:9090 -L 9876:localhost:9876 HOST`;
`DAVIO_WEB_PORT` and `DAVIO_GRPC_PORT` move them.

## Scores

`scripts/score_trajectory.py`: absolute trajectory error after a rigid fit, no scale fitted.
`scripts/score_map.py`: the map is placed in the laser scan's frame by that same fit of the
trajectory, nothing is fitted to the surface; accuracy, completeness and F-score at 2, 3, 5 and
10 cm (`--fit rigid|similarity` registers the map to the scan first, as a second protocol).

## Licence

GPL-3.0, as OpenVINS. The Depth Anything 3 weights have their own: base is Apache-2.0; large and giant are CC BY-NC 4.0.
