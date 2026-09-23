#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "${HERE}/../src" && pwd)"
# The libraries are a ROS 2 build (ROS_AVAILABLE==2), so rclcpp must be on the CMake path.
set +u; source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"; set -u
cmake -S "${HERE}" -B "${HERE}/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${OUT}"
cmake --build "${HERE}/build" -j"$(nproc)"
echo "built: $(ls "${OUT}"/openvins_ext*.so)"
