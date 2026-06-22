#!/usr/bin/env bash
# Configure + build qminweight with the Apple toolchain (the homebrew clang on PATH
# lacks the macOS SDK/frameworks, so we pin xcrun's clang).
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="${BUILD_DIR:-build}"
CXX_BIN="$(xcrun -f clang++)"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX_BIN" \
  -DCMAKE_OBJCXX_COMPILER="$CXX_BIN" \
  "$@"

JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
cmake --build "$BUILD_DIR" -j"$JOBS"

echo
echo "Built. Shared library:"
ls -1 python/qminweight/_lib/ 2>/dev/null || true
