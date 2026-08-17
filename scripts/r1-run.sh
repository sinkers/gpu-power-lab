#!/usr/bin/env bash
#
# R1 — run the training waterfall with power sampled alongside.
#
#   ./scripts/r1-run.sh ubuntu@<host> [-i key]
#
# For each phase: start the workload, wait for it to finish warming up, then
# sample with `--op observe` for the measured window. The sampler is a
# separate process because the workload is PyTorch - that is exactly what
# --op observe was built for.
set -euo pipefail
HOST="${1:?usage: $0 user@host [ssh args]}"; shift
SSHA=("$@")
REPO="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az -e "ssh -o StrictHostKeyChecking=accept-new ${SSHA[*]}" \
      "$REPO/scripts/r1_training.py" "$HOST:~/gpu-power-lab/scripts/"

ssh -o StrictHostKeyChecking=accept-new "${SSHA[@]}" "$HOST" 'bash -s' <<'REMOTE'
set -u
export PATH=/usr/local/cuda/bin:$PATH
cd ~/gpu-power-lab
mkdir -p /tmp/r1
R=./runner/build/gpu-power-runner
SECS=40

for PHASE in gemm forward fwdbwd optim full; do
  echo "[$(date -u +%H:%M:%S)] phase: $PHASE"
  rm -f /tmp/r1/$PHASE.ready
  python3 scripts/r1_training.py --phase $PHASE --seconds $((SECS+15)) --warmup 25 \
      --batch 2 --seqlen 4096 --ready-file /tmp/r1/$PHASE.ready \
      --out /tmp/r1/work-$PHASE.json > /tmp/r1/$PHASE.stdout 2>/tmp/r1/$PHASE.stderr &
  WPID=$!

  # Wait for warmup to finish, or give up if the phase died.
  for i in $(seq 1 120); do
    [ -f /tmp/r1/$PHASE.ready ] && break
    kill -0 $WPID 2>/dev/null || break
    sleep 1
  done
  if [ ! -f /tmp/r1/$PHASE.ready ]; then
    echo "  $PHASE failed to start:"; tail -3 /tmp/r1/$PHASE.stderr; wait $WPID 2>/dev/null; continue
  fi

  sudo $R --op observe --steady-sec $SECS --sample-hz 100 \
      --out-summary /tmp/r1/rung-train-$PHASE.json \
      --out-metrics /tmp/r1/rung-train-$PHASE.ndjson >/dev/null 2>&1
  wait $WPID 2>/dev/null

  python3 -c "
import json
p=json.load(open('/tmp/r1/rung-train-$PHASE.json')); w={}
try: w=json.load(open('/tmp/r1/work-$PHASE.json'))
except Exception: pass
pw=p['power']
print('  %-9s %7.1fW  %%lim %5.1f  %3.0fC  %6.1f TFLOPs  %8.0f tok/s  %4.1f GB'%(
 '$PHASE', pw['avg_w'], pw['pct_of_enforced_limit'], p['thermal']['peak_c'],
 w.get('model_tflops',0), w.get('tokens_per_s',0), w.get('peak_mem_gb',0)))" 2>/dev/null \
  || echo "  $PHASE: no summary"
done
echo "R1 DONE"
REMOTE
