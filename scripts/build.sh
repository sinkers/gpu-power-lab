#!/usr/bin/env bash
# scripts/build.sh — build the C runner in release mode.
# Runs from anywhere; resolves paths relative to the script.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"

cmake -S "$root/runner" -B "$root/runner/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/runner/build" -j "$(nproc)"

echo
echo "Built: $root/runner/build/gpu-power-runner"
"$root/runner/build/gpu-power-runner" --help
