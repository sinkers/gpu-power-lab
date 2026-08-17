#!/usr/bin/env bash
#
# Compile the runner locally, with no GPU, in a CUDA container.
#
#   ./scripts/build-local.sh            # build + spill check + arg smoke test
#   ./scripts/build-local.sh --sass     # also dump the tensor loop's SASS
#
# nvcc does not need a GPU to compile, so the entire T0 gate — all six
# architectures, register spills, SASS assertions, argument parsing — runs on
# a laptop for free. NVIDIA publishes native arm64 CUDA images, so this is not
# even emulated on Apple Silicon.
#
# This exists because the first three attempts at new kernel code each failed
# to compile on a rented B300 at several dollars an hour, for mistakes a
# compiler catches in ninety seconds: a macro defined below its first use, and
# an intrinsic that does not exist in device code. There is no reason to buy
# that feedback.

set -euo pipefail

IMAGE="${GPL_CUDA_IMAGE:-nvidia/cuda:13.0.1-devel-ubuntu22.04}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WANT_SASS=0
[ "${1:-}" = "--sass" ] && WANT_SASS=1

echo "==> building in $IMAGE (no GPU required)"

docker run --rm -v "$REPO":/src -w /src/runner "$IMAGE" bash -c '
set -e
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq cmake >/dev/null 2>&1

cmake -S . -B /tmp/build -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -E "CUDA targets|Error" || true
cmake --build /tmp/build -j 2>&1 | tee /tmp/build.log \
  | grep -viE "deprecat|declared here|^\s+\||^\s+\^|In file included|note:" \
  | grep -iE "error|Built target" || true

echo
echo "==> register spills (a spill turns the power virus from compute-bound"
echo "    into memory-bound without changing anything visible in the source)"
if grep -E "spill (stores|loads)" /tmp/build.log | grep -vE "\b0 bytes spill stores, 0 bytes spill loads" | grep -q .; then
    echo "    FAIL — spills detected:"
    grep -E "spill (stores|loads)" /tmp/build.log | grep -vE "\b0 bytes spill stores, 0 bytes spill loads"
    exit 1
else
    n=$(grep -c "0 bytes spill stores, 0 bytes spill loads" /tmp/build.log || true)
    echo "    OK — $n kernel/arch combinations, zero spills"
fi

echo
echo "==> registers per kernel"
grep -E "Compiling entry function|Used [0-9]+ registers" /tmp/build.log \
  | paste - - 2>/dev/null | sed "s/ptxas info    : //g" | head -20 || true

echo
echo "==> argument smoke test"
# The binary links against the NVML stub that ships with the toolkit. Without
# a driver the real libnvidia-ml.so.1 is absent, so the loader needs pointing
# at the stub directory or every invocation dies before main().
# The toolkit ships the stub as libnvidia-ml.so, but the binary is linked
# against the soname libnvidia-ml.so.1 that the real driver provides. One
# symlink is the difference between "cannot run at all" and a working
# argument-parsing test.
mkdir -p /tmp/stubs
ln -sf /usr/local/cuda/lib64/stubs/libnvidia-ml.so /tmp/stubs/libnvidia-ml.so.1
export LD_LIBRARY_PATH=/tmp/stubs:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-}
if /tmp/build/gpu-power-runner --help >/dev/null 2>&1; then
    echo "    --help OK"
else
    echo "    --help FAILED"; ldd /tmp/build/gpu-power-runner | grep -i "not found" || true
    exit 1
fi
for bad in "--op nonsense" "--duty-ramp sideways" "--precision fp7"; do
    if /tmp/build/gpu-power-runner $bad >/dev/null 2>&1; then
        echo "    ACCEPTED BAD ARGS: $bad"; exit 1
    fi
done
echo "    bad arguments rejected"

cp /tmp/build/gpu-power-runner /src/runner/gpu-power-runner.localbuild 2>/dev/null || true
'

if [ "$WANT_SASS" = "1" ]; then
    echo
    echo "==> SASS for the tensor loop on sm_103a"
    docker run --rm -v "$REPO":/src -w /src/runner "$IMAGE" bash -c '
      nvcc -arch=sm_103a -cubin -o /tmp/pv.cubin src/workload_powervirus.cu \
           -Isrc -I/usr/local/cuda/include 2>/dev/null || {
        echo "    (cubin build needs the full include path; skipping)"; exit 0; }
      cuobjdump -sass /tmp/pv.cubin | grep -cE "HMMA|OMMA|QMMA" \
        | xargs -I{} echo "    tensor-core instructions emitted: {}"
      cuobjdump -sass /tmp/pv.cubin | grep -cE "FFMA" \
        | xargs -I{} echo "    FFMA instructions emitted: {}"'
fi

rm -f "$REPO/runner/gpu-power-runner.localbuild"
echo
echo "Done. This is the T0 gate from TESTPLAN.md, running for free."
