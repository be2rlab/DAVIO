#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TTY=(); [ -t 0 ] && TTY=(-it)
if [ "${DAVIO_GPUS:-all}" = none ]; then
  GPU=(--runtime runc -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=)
else
  GPU=(--gpus "${DAVIO_GPUS:-all}")
fi
MOUNTS=(-v "$ROOT:/workspace")
[ -n "${DAVIO_DATA:-}" ] && MOUNTS+=(-v "$DAVIO_DATA:/workspace/data")
[ -n "${DAVIO_THIRDPARTY:-}" ] && MOUNTS+=(-v "$DAVIO_THIRDPARTY:/workspace/thirdparty")
# shellcheck disable=SC2206
EXTRA=(${DAVIO_DOCKER_ARGS:-})
exec docker run --rm "${TTY[@]}" "${GPU[@]}" "${EXTRA[@]}" --ipc host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONUNBUFFERED=1 \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=2 \
  -e PYTHONPATH=/workspace/src:/workspace/thirdparty/Depth-Anything-3/src \
  "${MOUNTS[@]}" -w /workspace "${DAVIO_IMAGE:-davio/dense-gpu:latest}" \
  bash -c 'source /opt/ros/humble/setup.bash; exec "$@"' -- "${@:-bash}"
