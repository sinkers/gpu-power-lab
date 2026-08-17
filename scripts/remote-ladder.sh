#!/usr/bin/env bash
#
# Run the probe and the O1 attribution ladder on a remote GPU box.
#
#   ./scripts/remote-ladder.sh user@host [ssh-args...]
#
# Copies the source, builds if the box has a usable toolkit, then runs the
# capability probe followed by the mix ladder. Everything is read-only on
# the platform except the probe's no-op limit write, so this is safe to run
# on a box you do not own.
#
# The probe runs FIRST and its verdict is printed before any measurement,
# because a container platform will happily produce a full sweep of numbers
# that mean nothing. See TESTPLAN.md, "Platform constraint".
#
# If the remote toolkit is too old for Blackwell (needs CUDA >= 12.9 for
# sm_103a), build elsewhere and ship runner/build/gpu-power-runner instead —
# the binary is fat across sm_86 / 90a / 120a / 100f / 100a / 103a.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 user@host [extra ssh args...]" >&2
    exit 2
fi

HOST="$1"; shift
SSH=(ssh -o StrictHostKeyChecking=accept-new "$@" "$HOST")
REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> syncing to $HOST"
rsync -az --exclude .git --exclude build --exclude results \
      -e "ssh -o StrictHostKeyChecking=accept-new $*" \
      "$REPO/" "$HOST:~/gpu-power-lab/"

echo "==> remote environment"
"${SSH[@]}" 'nvidia-smi --query-gpu=name,driver_version,power.limit,power.max_limit --format=csv || true
             nvcc --version 2>/dev/null | tail -1 || echo "no nvcc"'

echo "==> building"
"${SSH[@]}" 'cd ~/gpu-power-lab/runner &&
             cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -E "CUDA targets|Error" &&
             cmake --build build -j 2>&1 | grep -E "error|spill (stores|loads)[^0]|Built target" | tail -5'

echo
echo "==> capability probe (does this platform permit real measurement?)"
# Try unprivileged first: on a container there is no sudo and no point.
"${SSH[@]}" 'cd ~/gpu-power-lab/runner &&
             (sudo -n ./build/gpu-power-runner --probe 2>&1 || ./build/gpu-power-runner --probe 2>&1)' \
    || echo "   (probe exit != 0 — O1 is not measurable here; ladder below is indicative only)"

echo
echo "==> O1 attribution ladder"
# No --raise-power-limit: the ladder is a relative comparison at whatever
# limit is in force, so it stays valid without root.
"${SSH[@]}" 'cd ~/gpu-power-lab/runner
cat > /tmp/ladder.sh <<'"'"'INNER'"'"'
run() {  # mixT mixF mixD backend label
  ./build/gpu-power-runner --op powervirus \
    --mix-tensor $1 --mix-fma $2 --mix-dram $3 --tensor-backend $4 \
    --precision fp16 --size 4096 \
    --warmup-sec 4 --steady-sec 15 --sample-hz 100 \
    --out-summary "/tmp/rung-$5.json" >/dev/null 2>&1
  python3 -c "
import json;d=json.load(open('/tmp/rung-$5.json'));p=d['power'];t=d['transient']
print('%-26s avg %6.1fW  %%TDP %5.1f  clk %4.0fMHz  gap %5.1f%%  hz %5.1f  throttle=%s'%(
 '$5',p['avg_w'],p['pct_of_tdp'],d['clocks']['sm_avg_mhz'],
 t['energy_gap_pct'],t['sample_hz_achieved'],
 ','.join(d['throttle']['reasons']) or '-'))"
}

# Single-unit baselines.
run 0 1 0 wmma   fma-only
run 0 0 1 wmma   dram-only
run 1 0 0 wmma   tensor-only-wmma
run 1 0 0 cublas tensor-only-cublas

# THE DECISIVE COMPARISON.
# wmma is warp-synchronous and must displace FMA warps to run. cuBLAS on
# Blackwell reaches tcgen05, which is async and single-thread-issued, so it
# need not displace anything.
#   If cublas-tensor+fma EXCEEDS fma-only  -> the units overlap; the
#      mixed-unit power virus is real on this architecture.
#   If it lands BELOW fma-only             -> still competing, exactly as
#      Ampere behaved, and powervirus is an FFMA saturator.
run 1 1 0 wmma   mix-tensor-fma-wmma
run 1 1 0 cublas mix-tensor-fma-cublas
run 2 1 0 cublas mix-tensor2-fma1-cublas
run 1 2 0 cublas mix-tensor1-fma2-cublas
run 1 1 1 cublas mix-all-three-cublas
INNER
bash /tmp/ladder.sh'

echo
echo "==> O2 spot check: does the averaged power path follow a 200ms square wave?"
"${SSH[@]}" 'cd ~/gpu-power-lab/runner &&
  ./build/gpu-power-runner --op powervirus --mix-tensor 0 --mix-fma 1 --mix-dram 0 \
    --duty-on-ms 200 --duty-off-ms 200 --warmup-sec 3 --steady-sec 12 --sample-hz 100 \
    --out-summary /tmp/d.json --out-metrics /tmp/d.ndjson >/dev/null 2>&1 &&
  python3 -c "
import json
rows=[json.loads(l) for l in open(\"/tmp/d.ndjson\") if \"steady\" in l]
pw=[r[\"power_w\"] for r in rows]
pi=[r[\"power_instant_w\"] for r in rows if r.get(\"power_instant_w\",-1)>=0]
print(\"  usage  : min %.1f max %.1f  swing %.1f\"%(min(pw),max(pw),max(pw)-min(pw)))
if pi: print(\"  instant: min %.1f max %.1f  swing %.1f\"%(min(pi),max(pi),max(pi)-min(pi)))
else:  print(\"  instant: unavailable on this part\")"'

echo
echo "==> done. Pull the JSON with:"
echo "    scp $HOST:/tmp/{r,d}.json results/"
