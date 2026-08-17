#!/usr/bin/env bash
#
# B300 batch 2 — everything the first B300 run left undone.
#
#   ./scripts/b300-batch2.sh ubuntu@<ip> [-i ~/.ssh/gpu-power-lab]
#
# Runs in strict priority order and writes each result the moment it lands, so
# killing the box early still leaves everything up to that point usable. Total
# ~75 min if it runs to completion; the first 40 min carry most of the value.
#
#   1. build + probe                    ~5 min   (gate: stop if probe fails)
#   2. thermal soak, 30 min             ~32 min  R0 — the headline gap
#   3. ladder with traces retained      ~12 min  real NDJSON for the charts
#   4. duty period sweep 1ms → 5s       ~15 min  O2, only 200 ms done so far
#   5. idle floor                       ~2 min   the other end of every swing
#
# EVERY rung writes --out-metrics. The first B300 campaign kept only summaries
# and the instance is gone, so the highest-resolution data we ever collected no
# longer exists. Not again.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 user@host [ssh args...]       # run the batch" >&2
    echo "       $0 --collect user@host [ssh...]  # pull results back" >&2
    exit 2
fi

COLLECT=0
if [ "$1" = "--collect" ]; then COLLECT=1; shift; fi
HOST="$1"; shift
SSHA=("$@")
SSH=(ssh -o StrictHostKeyChecking=accept-new "${SSHA[@]}" "$HOST")
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/results/t2-b300-batch2"

if [ "$COLLECT" = "1" ]; then
    # Pull summaries AND traces. The traces are the point: a summary cannot
    # show what the power did between the mean and the peak.
    mkdir -p "$OUT"
    echo "==> collecting into $OUT"
    scp -q "${SSHA[@]}" "$HOST:/tmp/batch2/*.json"   "$OUT/" 2>/dev/null || true
    scp -q "${SSHA[@]}" "$HOST:/tmp/batch2/*.ndjson" "$OUT/" 2>/dev/null || true
    scp -q "${SSHA[@]}" "$HOST:/tmp/batch2.log"      "$OUT/" 2>/dev/null || true
    n_json=$(ls "$OUT"/rung-*.json 2>/dev/null | wc -l | tr -d " ")
    n_tr=$(ls "$OUT"/*.ndjson 2>/dev/null | wc -l | tr -d " ")
    echo "    $n_json summaries, $n_tr traces, $(du -sh "$OUT" | cut -f1)"
    if [ "$n_tr" = "0" ]; then
        echo "    WARNING: no traces came back. Do not terminate the box yet." >&2
    fi
    echo
    echo "Render with:  python3 scripts/generate_report.py $OUT"
    exit 0
fi

echo "==> syncing"
rsync -az --exclude .git --exclude build --exclude results \
      -e "ssh -o StrictHostKeyChecking=accept-new ${SSHA[*]}" \
      "$REPO/" "$HOST:~/gpu-power-lab/"

echo "==> environment"
"${SSH[@]}" 'nvidia-smi --query-gpu=name,driver_version,power.limit,power.max_limit,temperature.gpu --format=csv
             nvcc --version 2>/dev/null | tail -1 || echo "NO NVCC — install CUDA first"'

echo "==> build"
"${SSH[@]}" 'export PATH=/usr/local/cuda/bin:$PATH
             cd ~/gpu-power-lab/runner &&
             cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -E "CUDA targets|Error"
             cmake --build build -j 2>&1 | grep -E "error|Built target" | tail -3'

echo "==> probe (gate)"
"${SSH[@]}" 'cd ~/gpu-power-lab/runner && sudo ./build/gpu-power-runner --probe 2>&1 | head -6'

# ---------------------------------------------------------------------------
# Remote batch. Written as one script so the run survives an SSH drop, which
# a 30-minute soak over a home connection will otherwise eventually hit.
# ---------------------------------------------------------------------------
cat > /tmp/gpl-batch2.sh <<'REMOTE'
set -u
export PATH=/usr/local/cuda/bin:$PATH
cd ~/gpu-power-lab/runner
mkdir -p /tmp/batch2
R=./build/gpu-power-runner

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

report() {  # $1 = name
  python3 -c "
import json,sys
d=json.load(open('/tmp/batch2/rung-$1.json'))
p=d['power']; e=d.get('efficiency',{}); t=d.get('thermal',{}); tr=d.get('transient',{})
print('  %-22s avg %7.1fW peak %7.1fW  %%lim %5.1f  %3.0f/%3.0fC  edp %.4g  edpp %.4g  hz %.0f  %s'%(
 '$1', p['avg_w'], p['peak_w'], p['pct_of_enforced_limit'],
 t.get('avg_c',0), t.get('peak_c',0),
 e.get('edp_j_s',0), e.get('edpp_j_s2',0),
 tr.get('sample_hz_achieved',0),
 ','.join(d['throttle']['reasons']) or 'no-throttle'))" 2>/dev/null || echo "  $1: no summary"
}

# ---- 1. idle floor -------------------------------------------------------
# The bottom of every swing measurement. Cheap, and we have never recorded it.
say "idle floor (60s, nothing running)"
sudo $R --op powervirus --mix-tensor 0 --mix-fma 0 --mix-dram 1 \
  --warmup-sec 0.1 --steady-sec 1 --sample-hz 50 \
  --out-summary /tmp/batch2/rung-warm.json >/dev/null 2>&1 || true
sleep 20
sudo $R --op observe --steady-sec 60 --sample-hz 100 \
  --out-summary /tmp/batch2/rung-idle.json \
  --out-metrics /tmp/batch2/rung-idle.ndjson >/dev/null 2>&1 \
  || say "  (--op observe not built yet — skipping idle rung)"
[ -f /tmp/batch2/rung-idle.json ] && report idle

# ---- 2. thermal soak, 30 minutes ----------------------------------------
# R0. Every rung in batch 1 was 12s, which ranks workloads and says nothing
# about temperature. Fixed-work mode so EDP/EDPp mean what they claim.
say "THERMAL SOAK — 30 min, tensor-only bf16, fixed work. Go and do something else."
sudo $R --op powervirus --mix-tensor 1 --mix-fma 0 --mix-dram 0 \
  --tensor-backend cublas --precision bf16 --size 8192 \
  --warmup-sec 10 --steady-sec 1800 --sample-hz 100 --raise-power-limit \
  --out-summary /tmp/batch2/rung-soak-30min.json \
  --out-metrics /tmp/batch2/rung-soak-30min.ndjson >/dev/null 2>&1
report soak-30min
say "soak done"

# ---- 3. the ladder again, this time keeping the traces ------------------
# 20s rather than 12s: still short, but enough samples that the envelope
# chart has something to show.
lad() {  # T F D backend precision label
  sudo $R --op powervirus --mix-tensor $1 --mix-fma $2 --mix-dram $3 \
    --tensor-backend $4 --precision $5 --size 8192 \
    --warmup-sec 5 --steady-sec 20 --sample-hz 100 --raise-power-limit \
    --out-summary "/tmp/batch2/rung-$6.json" \
    --out-metrics "/tmp/batch2/rung-$6.ndjson" >/dev/null 2>&1
  report "$6"
}
say "ladder, with traces"
lad 1 0 0 cublas bf16 tensor-only-bf16
lad 1 0 0 cublas fp16 tensor-only-fp16
lad 0 0 1 wmma   fp16 dram-only
lad 0 1 0 wmma   fp16 fma-only
lad 1 0 0 wmma   fp16 tensor-only-wmma
lad 1 1 1 cublas bf16 mix-all-cublas

# ---- 4. duty period sweep ------------------------------------------------
# Batch 1 only did 200 ms. The design predicts different regimes across
# 1 ms -> 10 s; below the controller's response time is where an over-limit
# excursion is most likely, and we have never looked there.
duty() {  # on_ms off_ms label
  sudo $R --op powervirus --mix-tensor 1 --mix-fma 0 --mix-dram 0 \
    --tensor-backend cublas --precision bf16 --size 8192 \
    --duty-on-ms $1 --duty-off-ms $2 \
    --warmup-sec 5 --steady-sec 40 --sample-hz 100 --raise-power-limit \
    --out-summary "/tmp/batch2/rung-$3.json" \
    --out-metrics "/tmp/batch2/rung-$3.ndjson" >/dev/null 2>&1
  python3 -c "
import json
rows=[json.loads(l) for l in open('/tmp/batch2/rung-$3.ndjson') if '\"steady\"' in l]
pw=[r['power_w'] for r in rows]
pi=[r['power_instant_w'] for r in rows if r.get('power_instant_w',-1)>=0]
d=json.load(open('/tmp/batch2/rung-$3.json'))
print('  %-16s usage %6.1f-%7.1f (swing %6.1f)  instant %6.1f-%7.1f (swing %6.1f)  peak/cap %5.1f%%'%(
 '$3', min(pw),max(pw),max(pw)-min(pw),
 (min(pi) if pi else 0),(max(pi) if pi else 0),((max(pi)-min(pi)) if pi else 0),
 100*max(max(pw), max(pi) if pi else 0)/d['power']['enforced_limit_w']))" 2>/dev/null || echo "  $3: failed"
}
say "duty period sweep"
duty 1    1    duty-1ms
duty 5    5    duty-5ms
duty 20   20   duty-20ms
duty 100  100  duty-100ms
duty 500  500  duty-500ms
duty 2000 2000 duty-2s

say "ALL DONE"
ls -la /tmp/batch2/ | tail -30
du -sh /tmp/batch2
REMOTE

echo "==> launching remote batch (detached — survives an SSH drop)"
scp -q -o StrictHostKeyChecking=accept-new "${SSHA[@]}" /tmp/gpl-batch2.sh "$HOST:/tmp/"
"${SSH[@]}" 'nohup bash /tmp/gpl-batch2.sh > /tmp/batch2.log 2>&1 & echo "started pid $!"'

echo
echo "Follow it with:   ssh ${SSHA[*]} $HOST 'tail -f /tmp/batch2.log'"
echo "Collect with:     $0 --collect $HOST ${SSHA[*]}"
echo
echo "Results will land in $OUT"
