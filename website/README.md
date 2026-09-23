# Project page

One static `index.html` and its assets; no build step. GitHub Actions publishes this folder
to GitHub Pages on every push to `main` (`.github/workflows/pages.yml`) — enable it once
under *Settings → Pages → Source: GitHub Actions*.

```
index.html               the page
assets/maps/<name>.bin   a map for the browser: count x 3 uint16 positions, then count x 3 uint8 colours
assets/maps/<name>.json  its scale/offset, trajectory and scores
assets/grid_*.mp4        the 2x2 videos (camera | walkthrough / three-quarter | overhead)
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

The videos come from `./davio render RUN --build ...` and `./davio grid` (see
[docs/VISUALIZE.md](../docs/VISUALIZE.md)), then a web encode:

```bash
ffmpeg -i map_grid.mp4 -vf scale=1280:-2 -c:v libx264 -preset slow -crf 30 \
    -pix_fmt yuv420p -movflags +faststart -an website/assets/grid_ori.mp4
```

Three.js is loaded from jsDelivr; nothing else leaves the page.
