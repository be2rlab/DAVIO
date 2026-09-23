# Seeing what it is doing

```bash
./davio view runs/ori_live --open                   # dense RGB map, bright background
./davio view runs/V1_01_easy                        # a finished run
./davio view runs/live --follow                     # a run being written right now
./davio view runs/V1_01_easy --rate 1               # play it back at sensor rate
./davio view runs/V1_01_easy --groundtruth --data data/euroc
./davio view runs/V1_01_easy --save tour.rrd        # a file to send someone
```

The viewer is [Rerun](https://rerun.io). By default it serves a browser viewer at
`http://localhost:9090` and keeps serving until interrupted; `--spawn` opens the desktop
viewer instead.

`./davio view` prefers the host when `rerun-sdk` is importable there — the browser and the
desktop viewer then work without port plumbing — and falls back to the container otherwise.
`pip install rerun-sdk==0.23.1` on the host is worth it.

## What is on screen

| Entity | |
|---|---|
| `world/odometry` | the OpenVINS pose stream, uncorrected — orange |
| `world/online_map` | the same poses through the live odometry→map correction — blue |
| `world/submaps/*` | dense metric RGB geometry: every frame of every submap, at the same density the exported map fuses at. Each submap carries its own `Transform3D`, so a loop closure visibly moves the geometry it corrects |
| `world/keyframes` | the optimized submap centres — green |
| `world/loops` | accepted loop closures — red |
| `world/camera` | the current pose and its frustum |
| `latest/rgb`, `latest/depth` | the newest mapped window and its metric depth |
| `plots/*` | publication age, submap scale, alignment RMSE, accepted loops, sparse edges, loop innovation, correction jump |
| `provenance` | what you are actually looking at |

Everything is in the run's **own map frame**. Nothing is transformed to make it look
better.

## Density and colour

Colour comes from the submap archives, which store full-resolution RGB alongside the depth.

Whether that RGB is real colour depends on the camera. EuRoC and ORI are monochrome, and
DA3 returns whatever it was fed, so their maps are grey (the stored channels differ by at
most one level, which is rounding). A colour camera --- VCU-RVI's Structure Core, a
RealSense colour stream --- gets its frames rectified through the same maps and resampled
onto DA3's depth grid, and the map is painted with those instead. The filter is handed grey
either way: OpenVINS tracks on one channel, and colour never touches the geometry.

By default the viewer draws **every frame of every submap** at `--pixel-step 4`, the density
`scripts/export_map.py` fuses the exported cloud at. A frame that two overlapping submaps
share is drawn once. On the 77-submap ORI r01 map that is 2.8 million points:

| | points |
|---|---|
| `--frames all --pixel-step 4` (default) | 2,813,146 |
| `--frames centre --pixel-step 8` | 225,295 |

`--frames centre` draws one frame per submap and is about five times lighter; raising
`--pixel-step` decimates further. Both are worth reaching for on a long sequence or over a
remote connection. The viewer prints what it drew when it finishes loading.

A submap's residual scale multiplies **depth only**, so the viewer puts the scale in the
points and only the rotation and translation on the entity transform. Handing the whole
similarity to the transform instead displaces every non-centre frame by `(s-1)` times its
baseline — 3.4 cm on ORI r01 — which is why this is pinned by
`tests/test_viewer_geometry.py` against `fuse_map`'s own reader. When a loop closure moves
a node's scale by more than half a percent, its clouds are redrawn rather than left stale.

## Background

`--background light` (the default) is a bright gradient; `--background white` is a flat
near-white, and `--background dark` is rerun's usual dark gradient. The trajectory, keyframe
and loop colours switch with it, so the lines stay readable either way.

## Following a live run

`--follow` tails `events.jsonl` and `map/map_index.json` and keeps drawing as they grow. It
is the default when the run's `run.json` says `state: running`, so

```bash
./davio run V1_01_easy &
./davio view runs/V1_01_easy --follow
```

works without thinking about it. The viewer reads only files the engine has already
committed; it never touches the estimator, and starting or stopping it cannot change a run.

Watch `plots/pose_age_s`: that is the time from a frame's receipt to its pose being
published, and it is the quantity that says whether the system is keeping up. Watch
`plots/loop_innovation_m` too — how far the worst accepted loop disagreed with the odometry
chain is what decides whether the graph is helping or hurting.

## The provenance panel

A viewer that titles everything "live" invites an archived run, tailed after the fact, to
be read as a new experiment. This one reads the label off the run's own descriptor:

| Label | |
|---|---|
| `LIVE` | the run is in progress, being written as it is read |
| `RECORDED` | a completed run replayed from disk — the timing shown is the timing it was recorded with, not a measurement being taken now |
| `CACHED REPLAY` | archived submaps pushed through the back-end. No sensor, estimator or scheduler took part, so latency and queue behaviour shown are **not** those of a live run |
| `AWAITING` / `UNREADABLE` | no `run.json` yet, or it does not parse |

If a run completes while you are watching, the label changes and the panel says so.
`tests/test_viewer_provenance.py` pins this behaviour.

## The reference overlay

`--groundtruth` needs `--data` and is **refused on a run that is still being written**: a
live view must not quietly become an evaluation. It is also the only place besides
`scripts/rerun_export.py` and `src/davio/eval/` where this repository reads a reference
trajectory at all.

The reference is rigidly aligned **into the run's frame**, not the other way round, so
every submap stays exactly where the back-end put it. The alignment and its RMSE are
printed and written into the provenance panel:

```
reference overlay: aligned on 1604 poses, ATE 0.0489 m
```

For the per-stream view — odometry and map each at *their own* best gauge, which is how ATE
is defined — `./davio shell python3 scripts/rerun_export.py --run runs/X --data data/euroc
--out runs/X/debug.rrd` writes a `.rrd` of it.

## Sharing one

```bash
./davio view runs/V1_01_easy --save tour.rrd
rerun tour.rrd
```

A `.rrd` is self-contained — geometry, timeline, plots and the provenance panel — and opens
in the desktop viewer or at <https://rerun.io/viewer>. They are large, and the default
density makes them larger: use `--frames centre` and a higher `--pixel-step` for one you
intend to send someone.

## Slides and talks

The viewer is for inspecting a run. For a figure or a talk, `./davio render` puts the same
map through Open3D offscreen and writes stills and a turntable instead:

```bash
./davio export runs/ori_r01                    # map.ply, once
./davio render runs/ori_r01 --cut-ceiling 0.88 --width 3840 --height 2160
```

It writes `map_top.png` (a floor plan, the building's long axis laid along the width),
`map_hero.png`, `map_side.png` and `map_orbit.mp4`, all cropped to their own content, with
the trajectory drawn as a tube coloured by time. `--cut-ceiling Q` drops everything above
the Q-th height quantile, which is what turns a roof into a dollhouse you can see the rooms
in; `--odometry both` adds the uncorrected filter stream in grey beside the corrected one.

### The map being built

`--build CAMERA` animates the map arriving instead of showing it finished: each submap
enters the scene at the sensor time it was committed, and the trajectory grows with it.

```bash
./davio render runs/ori_r01 --build fly     # walks the trajectory as the map appears
./davio render runs/ori_r01 --build top --cut-ceiling 0.88     # fixed overhead, filling in
```

The cameras are `fly` (behind and above the current pose, looking ahead), `top`, `side`,
`hero` (all fixed, framed on the finished map) and `orbit` (circles while it builds).
`--duration` sets how many video seconds the whole run is compressed into, `--hold-s` how
long to sit on the finished map at the end.

To show what the map was made from, set the camera beside it:

```bash
./davio compose renders/ori_r01/map_build_fly.mp4 --run runs/ori_r01
./davio compose renders/beach/map_build_fly.mp4 --run runs/beach --rotate 90
```

Each video frame gets the camera image recorded at that frame's sensor time, from the
`.timeline.json` the build render writes beside its video, so the two halves show the same
moment. `--rotate 90` stands up a phone held in portrait. A long, thin render is padded to a
readable height rather than shrinking the camera to a thumbnail. For a video rendered before
the sidecar existed, pass its `--duration` and `--hold-s`.

Or put several views and the camera in one grid, all on one clock:

```bash
./davio grid --run runs/ori_r01 --out map_grid.mp4 \
    --panel camera --panel fly=map_build_fly.mp4 \
    --panel hero=map_build_hero.mp4 --panel top=map_build_top.mp4
```

The views can be rendered at different lengths: the longest one sets the timeline, and every
other cell shows its own frame for the same sensor time. Panels fill the grid row by row
(`--cols 2`); the default cell size makes a 2×2 grid exactly 1920×1080.

This reads the submap archives rather than `map.ply`, so it needs no export — but it does
need a run that kept its dense geometry (`./davio map` keeps it; `scripts/replay_map.py`
needs `--keep-dense`). Flying is close work: at a metre from the lens, a 2 cm voxel cloud
is visibly stippled, so use a finer `--voxel-m` (0.012), `--pixel-step 1` and a larger
`--point-size` (5) than a wide shot needs.

Open3D runs on the **host**, not in the image, which carries no graphics stack:
`pip install open3d` and make sure `DISPLAY` points at a screen or an Xvfb. Rendering goes
through Open3D's OpenGL visualizer rather than its newer Filament one, which silently
returns a black frame past ~65k points on some drivers.

## Options

| Flag | Default | |
|---|---|---|
| `--follow` | on for a live run | keep watching for new events |
| `--rate` | `0` | play a recording back at this multiple of sensor time; `0` loads it at once |
| `--frames` | `all` | `centre` draws one frame per submap, about 5x lighter |
| `--pixel-step` | `4` | submap decimation; the density the exported map is fused at |
| `--point-radius` | `0.004` | submap point size, in metres |
| `--background` | `light` | `light`, `white` or `dark` |
| `--open` | off | launch a viewer on the served stream, desktop app if there is one |
| `--memory-limit` | `2GB` | server budget; a dense map is millions of points |
| `--spawn` | off | desktop viewer instead of the browser |
| `--save FILE.rrd` | off | write a file instead of serving |
| `--web-port --grpc-port` | 9090 / 9876 | change both to view two runs side by side |
| `--poll-s` | `0.2` | how often `--follow` re-reads |

## A run whose dense archives were deleted

A run keeps its poses, graph and `map.ply` even if `map/submaps/` is deleted to save disk.
The viewer then still shows the trajectory, the keyframes and the loops; it logs
`submap NNN: dense archive pruned, poses only` for each missing cloud and carries on rather
than failing.
