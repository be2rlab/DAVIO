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

## Scores

`scripts/score_trajectory.py`: absolute trajectory error after a rigid fit, no scale fitted.
`scripts/score_map.py`: the map is placed in the laser scan's frame by that same fit of the
trajectory, nothing is fitted to the surface; accuracy, completeness and F-score at 2, 3, 5 and
10 cm (`--fit rigid|similarity` registers the map to the scan first, as a second protocol).

## Licence

GPL-3.0, as OpenVINS. The Depth Anything 3 weights have their own: base is Apache-2.0; large and giant are CC BY-NC 4.0.
