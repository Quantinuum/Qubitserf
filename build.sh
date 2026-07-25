#!/usr/bin/env bash
# Configure + build BOTH native libraries (libdistfind and libcodeaut) and drop each into its
# Python subpackage's _lib/ so they import without an install.  Use for local development /
# running the tests in-tree (PYTHONPATH=python).  For an installed wheel, `pip install .` runs
# the same CMake build via scikit-build-core.
#
# On macOS the Homebrew clang on PATH lacks the macOS SDK/frameworks needed for the Metal
# backends, so we pin xcrun's clang for both C++ and Objective-C++.
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="${BUILD_DIR:-build}"

CMAKE_ARGS=(-S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release)
if [[ "$(uname -s)" == "Darwin" ]]; then
  CXX_BIN="$(xcrun -f clang++)"
  CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="$CXX_BIN" -DCMAKE_OBJCXX_COMPILER="$CXX_BIN")
fi

cmake "${CMAKE_ARGS[@]}" "$@"

JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
cmake --build "$BUILD_DIR" -j"$JOBS"

echo
echo "Built. Shared libraries:"
for d in distfind codeaut; do
  for f in python/qubitserf/"$d"/_lib/*; do
    [ -e "$f" ] && echo "  $d/_lib/$(basename "$f")"
  done
done
