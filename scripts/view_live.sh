#!/usr/bin/env bash
# Local browser visualization; no host networking and no GPU allocation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: scripts/view_live.sh runs/RUN [web-port] [grpc-port]}"
WEB="${2:-9090}"
GRPC="${3:-9876}"
exec docker run --rm --name "davio-rerun-${WEB}" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONPATH=/workspace/src -e OPENBLAS_NUM_THREADS=1 \
  -p "127.0.0.1:${WEB}:${WEB}" -p "127.0.0.1:${GRPC}:${GRPC}" \
  -v "${ROOT}:/workspace:ro" -w /workspace \
  "${DAVIO_IMAGE:-davio/dense-gpu:latest}" \
  python3 scripts/view_run.py --run "$RUN" --web-port "$WEB" --grpc-port "$GRPC"
