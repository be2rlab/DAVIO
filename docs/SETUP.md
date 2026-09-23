# Setup

```bash
git clone --recursive https://github.com/be2rlab/DAVIO.git && cd DAVIO
make setup        # everything below, in order
./davio doctor    # what is present, and what each missing piece blocks
```

A ready checkout looks like this:

```
  ok    docker                        /usr/bin/docker
  ok    image davio/dense-gpu:latest  present
  ok    gpu                           NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB
  ok    openvins_ext binding          openvins_ext.cpython-310-x86_64-linux-gnu.so
  ok    DA3 checkpoint                thirdparty/da3_weights/base
  ok    XFeat weights                 present
  ok    EuRoC data                    V1_01_easy
  ok    ORI data                      r01
  --    RealSense rig                 not generated: plug a camera in and run `./davio calibrate`
```

## What the host needs

Linux, Docker, the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
and an NVIDIA driver new enough for CUDA 12.6. About 30 GB of disk for the images, the
checkpoint and a couple of sequences. Nothing else is installed on the host: CUDA, ROS 2
Humble, Torch, OpenCV and OpenVINS all live in the image, and `./davio` dispatches there.

Two things run on the host instead, because they open windows:

```bash
pip install rerun-sdk==0.23.1     # ./davio view: opens the viewer without port plumbing
pip install open3d                # ./davio render: stills and videos
```

`./davio view` falls back to the container if rerun is not installed on the host.

## The steps `make setup` runs

| step | what it does | time |
|---|---|---|
| `make image-base` | `davio/standalone-gpu`: CUDA 12.6, ROS 2 Humble, Torch cu126 — about 6 GB of downloads | long |
| `make image` | `davio/dense-gpu`: the base plus kornia; what every command runs in | minutes |
| `make openvins` | fetches the pinned OpenVINS submodule and builds it with colcon, inside the image | ~15 min |
| `make shim` | builds the pybind11 binding (`native/`) into `src/openvins_ext*.so` | ~1 min |
| `make weights` | the pinned Depth Anything 3 checkpoint (DA3-BASE, Apache-2.0) | ~1 GB |
| `make xfeat` | XFeat + LighterGlue, for loop closure and per-frame depth scales | small |

OpenVINS is used **unmodified**, pinned by the submodule. `make weights` pins the checkpoint
by revision; `scripts/fetch_da3_weights.sh large` fetches DA3-Large instead (CC BY-NC 4.0).

Any command can use another image with `DAVIO_IMAGE=...`.

## Data

Everything goes under `data/` — the container mounts this checkout and nothing else.

```bash
make euroc                                  # V1_01_easy, MH_01_easy (about 3 GB each)
make euroc SEQS="V2_01_easy MH_04_difficult"
make ori                                    # r01; SEQ=r02 ... r05 for the others
make tumvi
```

`make ori` needs `pip install gdown` on the host for the download; the conversion from the
release's MCAP bag runs in the image. Your own recordings go under `data/custom/` — see
[CUSTOM_DATA.md](CUSTOM_DATA.md).

## Checking it works

```bash
./davio test                              # ~250 tests in the image, about 20 s
./davio run V1_01_easy --seconds 30       # a short real run, scored at the end
```

The tests fake everything that would touch the binding or the GPU. They also run on a plain
host, before any of the long builds: `pip install -r requirements.txt && python3 -m pytest`
(the one mapper test that needs torch and XFeat skips itself there).

## When something is wrong

| symptom | fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | the NVIDIA container toolkit is not installed, or Docker was not restarted after installing it |
| `openvins_ext` missing after `make shim` | `make openvins` did not finish: it has to build before the binding can link against it |
| `... is outside this checkout` | data has to live under `data/` (a symlink to elsewhere is not visible inside the container) |
| a run stops with `no rig for ...` | a custom recording needs `./davio rig data/custom/<name>` once |
| the viewer does not open | `pip install rerun-sdk==0.23.1`, or pass `--save run.rrd` and open it with `rerun run.rrd` |
| out of GPU memory | nothing else should hold the GPU; 6 GB is enough for DA3-BASE at the default settings |
