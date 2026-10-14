# Project page

One static `index.html` and its assets; no build step. GitHub Actions publishes this folder
to GitHub Pages on every push to `main` (`.github/workflows/pages.yml`) — enable it once
under *Settings → Pages → Source: GitHub Actions*.

```
index.html               the page
assets/maps/<name>.bin   a map for the browser: count x 3 uint16 positions, then count x 3 uint8 colours
assets/maps/<name>.json  its scale/offset, trajectory and scores
assets/videos/<name>_open3d.mp4   the 2x2 grid of a run (camera | walkthrough / three-quarter | overhead)
assets/videos/<name>_rerun.mp4    the same run replayed in Rerun (map, camera, metric depth, scale)
assets/videos/<name>.jpg|json     posters, and the run's own numbers the page shows beside them
assets/rerun/<name>.rrd           a light Rerun recording, opened on the page on request
assets/teaser.jpg        also the README's image
assets/pipeline.png      the method figure
```

Look at it locally (the viewer fetches binary files, so `file://` will not do):

```bash
python3 -m http.server 8000 --directory website     # then http://localhost:8000
```

## Rebuilding the assets

A map, from any run that has been exported (`./davio export`):

```bash
python3 scripts/make_page_assets.py --run runs/r01_remap --name ori \
    --title "ORI r01" --note "building interior · 140 s handheld · colour" --points 600000
```

It thins the cloud to the budget, quantizes positions to 16 bits over the scene's own box
(sub-millimetre at building scale) and carries the trajectory and the run's scores into the
JSON, which is where the page's facts and results table come from — nothing is typed twice.
To add a map, write it with a new `--name` and add one line to `DATASETS` in the page.

The videos, their posters and the Rerun recordings come from a finished run, one command per run, on the
`feature/davio++` branch (host: `pip install open3d playwright rerun-sdk==0.23.1`, Chrome, `DISPLAY` set):

```bash
./davio run V1_01_easy --dataset euroc --out runs/web/euroc
python3 scripts/site_media.py --run runs/web/euroc --name euroc --work renders --out renders/site
cp renders/site/*.mp4 renders/site/*.jpg renders/site/*.json website/assets/videos/
cp renders/site/*.rrd website/assets/rerun/
```

The names are `euroc` (EuRoC V1_01_easy), `ori` (ORI r01, `--rate 0.3`), `beach` (`data/custom/umbrella_beach`,
`--dataset custom --set openvins.calib_cam_intrinsics=true --set openvins.calib_cam_timeoffset=true`) and `tumvi`
(TUM-VI room1, `scripts/fetch_tumvi.sh data room1` first). The Open3D views are rendered offscreen; the Rerun
video is Rerun's own web viewer in headless Chrome, stepped frame by frame (`scripts/record_rerun.py`). `*.rrd`
is ignored repository-wide; `.gitignore` lets `website/assets/rerun/*.rrd` through.

Three.js is loaded from jsDelivr, and Rerun's web viewer from unpkg when a visitor asks for it; nothing else leaves the page.
