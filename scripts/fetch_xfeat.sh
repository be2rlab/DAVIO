#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d thirdparty/accelerated_features ]; then
  git clone https://github.com/verlab/accelerated_features.git thirdparty/accelerated_features
fi
git -C thirdparty/accelerated_features checkout e92685f57f8318b18725c5c8c0bd28c7fe188d9a
git -C thirdparty/accelerated_features rev-parse HEAD
# kornia (the LighterGlue backend) is already in the image and in requirements.txt.
