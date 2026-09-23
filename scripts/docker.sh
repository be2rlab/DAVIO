#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TTY=(); [ -t 0 ] && TTY=(-it)
# A RealSense is a USB device; the container only sees it when it is passed through.
USB=(); [ -n "${DAVIO_REALSENSE:-}" ] && USB=(--device /dev/bus/usb -v /dev:/dev:ro)
exec docker run --rm "${TTY[@]}" --gpus all --ipc host "${USB[@]}" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONUNBUFFERED=1 \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=2 \
  -e PYTHONPATH=/workspace/src:/workspace/thirdparty/Depth-Anything-3/src \
  -v "${ROOT}:/workspace" -w /workspace \
  "${DAVIO_IMAGE:-davio/dense-gpu:latest}" \
  bash -c 'source /opt/ros/humble/setup.bash; exec "$@"' -- "${@:-bash}"
