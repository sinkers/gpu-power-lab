#!/usr/bin/env bash
# scripts/smoke.sh — run one tiny rung to verify the binary works on this host.
# Requires: gpu-power-runner already built, a visible NVIDIA GPU.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
bin="$root/runner/build/gpu-power-runner"
out="${1:-/tmp/gpu-power-smoke}"

if [[ ! -x "$bin" ]]; then
    echo "runner not built. Run scripts/build.sh first." >&2
    exit 1
fi

mkdir -p "$out"

# Small, fast rung: fp32 4096-square GEMM, 3s warmup, 5s steady.
"$bin" \
    --op sgemm --precision fp32 --size 4096 --streams 1 \
    --warmup-sec 3 --steady-sec 5 --sample-hz 50 \
    --device 0 \
    --rung-id smoke-$(date -u +%Y%m%dT%H%M%SZ) \
    --out-metrics "$out/metrics.ndjson" \
    --out-summary "$out/summary.json"

echo
echo "=== summary ==="
cat "$out/summary.json"
echo
echo "=== first 3 metric samples ==="
head -3 "$out/metrics.ndjson"
echo
echo "=== last 3 metric samples ==="
tail -3 "$out/metrics.ndjson"
echo
echo "Output directory: $out"
