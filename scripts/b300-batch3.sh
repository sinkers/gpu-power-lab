#!/usr/bin/env bash
#
# B300 batch 3 — everything that was outstanding after batch 2.
#
#   ./scripts/b300-batch3.sh ubuntu@<ip> [-i ~/.ssh/gpu-power-lab]
#   ./scripts/b300-batch3.sh --collect ubuntu@<ip> [-i ...]
#
# Five groups, ~30 min total, in value order so an early stop still leaves
# something worth having:
#
#   1. Cap sweep, fixed-work        ~7 min  EDP/EDPp vs cap - the curve we
#                                           designed and never produced, and
#                                           the first rungs where delay is
#                                           actually measured rather than
#                                           imposed.
#   2. Precision ladder             ~5 min  fp16/bf16/tf32/fp32/fp64/int8,
#                                           plus fp8 and fp4 via cuBLASLt.
#   3. Attribution ladder           ~8 min  eight single-unit rungs and the
#                                           pairs, for the interaction terms.
#   4. Overshoot at a lowered cap   ~4 min  bursting against 500 W should make
#                                           an over-limit excursion far easier
#                                           to catch than at 1100 W.
#   5. Ramp shapes + locked clocks  ~5 min  how much of the transient hazard is
#                                           inherent vs an implementation
#                                           choice, and a reproducibility run.
#
# Every rung keeps its NDJSON.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 [--collect] user@host [ssh args...]" >&2; exit 2
fi
COLLECT=0
if [ "$1" = "--collect" ]; then COLLECT=1; shift; fi
HOST="$1"; shift
SSHA=("$@")
SSH=(ssh -o StrictHostKeyChecking=accept-new "${SSHA[@]}" "$HOST")
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/results/t2-b300-batch3"

if [ "$COLLECT" = "1" ]; then
    mkdir -p "$OUT"
    echo "==> collecting into $OUT"
    scp -q "${SSHA[@]}" "$HOST:/tmp/batch3/*.json"   "$OUT/" 2>/dev/null || true
    scp -q "${SSHA[@]}" "$HOST:/tmp/batch3/*.ndjson" "$OUT/" 2>/dev/null || true
    scp -q "${SSHA[@]}" "$HOST:/tmp/batch3.log"      "$OUT/" 2>/dev/null || true
    echo "    $(ls "$OUT"/rung-*.json 2>/dev/null | wc -l | tr -d ' ') summaries, $(ls "$OUT"/*.ndjson 2>/dev/null | wc -l | tr -d ' ') traces, $(du -sh "$OUT" | cut -f1)"
    echo "Render with:  python3 scripts/generate_report.py $OUT"
    exit 0
fi

echo "==> syncing"
rsync -az --exclude .git --exclude build --exclude results \
      -e "ssh -o StrictHostKeyChecking=accept-new ${SSHA[*]}" \
      "$REPO/" "$HOST:~/gpu-power-lab/"

echo "==> build (this batch adds cuBLASLt, five new kernel roles, ramps)"
"${SSH[@]}" 'export PATH=/usr/local/cuda/bin:$PATH
             cd ~/gpu-power-lab/runner &&
             cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -E "CUDA targets|Error" &&
             cmake --build build -j 2>&1 | grep -E "error|spill (stores|loads)" | head -20
             cmake --build build -j 2>&1 | grep -cE "Built target" >/dev/null && echo BUILD_OK'

cat > /tmp/gpl-batch3.sh <<'REMOTE'
set -u
export PATH=/usr/local/cuda/bin:$PATH
cd ~/gpu-power-lab/runner
mkdir -p /tmp/batch3
R=./build/gpu-power-runner
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

rep() {
  python3 -c "
import json
try: d=json.load(open('/tmp/batch3/rung-$1.json'))
except Exception as e: print('  %-24s FAILED (%s)'%('$1',e)); raise SystemExit
p=d['power']; e=d.get('efficiency',{}); t=d.get('thermal',{})
print('  %-24s avg %7.1fW  %%lim %5.1f  %3.0fC  clk %4.0f  edp %.4g  edpp %.4g  %s'%(
 '$1',p['avg_w'],p['pct_of_enforced_limit'],t.get('peak_c',0),
 d['clocks']['sm_avg_mhz'],e.get('edp_j_s',0),e.get('edpp_j_s2',0),
 ','.join(d['throttle']['reasons']) or '-'))" 2>/dev/null
}

# ---- 1. CAP SWEEP, fixed work -------------------------------------------
# The curve we designed and never ran. --iters makes delay a measurement
# rather than a setting, which is what EDP and EDPp need to mean anything;
# every previous rung was fixed-time, so their EDP was really just energy.
say "cap sweep (fixed work, 4000 iters each)"
for W in 1100 900 700 550 450; do
  sudo $R --op powervirus --mix-tensor 1 --mix-fma 0 --mix-dram 0 \
    --tensor-backend cublas --precision bf16 --size 8192 \
    --power-limit $W --iters 4000 \
    --warmup-sec 5 --steady-sec 600 --sample-hz 50 \
    --out-summary /tmp/batch3/rung-cap-${W}w.json \
    --out-metrics /tmp/batch3/rung-cap-${W}w.ndjson >/dev/null 2>&1
  rep cap-${W}w
done

# ---- 2. PRECISION LADDER -------------------------------------------------
# Which format actually draws the most? The fastest is not necessarily the
# hungriest: a format that consumes operands sooner spends the difference
# waiting on memory.
say "precision ladder"
for P in fp16 bf16 tf32 fp32 fp64 int8 fp8 fp4; do
  sudo $R --op powervirus --mix-tensor 1 --mix-fma 0 --mix-dram 0 \
    --tensor-backend cublas --precision $P --size 8192 \
    --warmup-sec 4 --steady-sec 20 --sample-hz 50 --raise-power-limit \
    --out-summary /tmp/batch3/rung-prec-$P.json \
    --out-metrics /tmp/batch3/rung-prec-$P.ndjson >/dev/null 2>&1 \
    || say "  $P: unsupported on this part (expected for some)"
  rep prec-$P
done

# ---- 3. ATTRIBUTION LADDER ----------------------------------------------
# Single units first, then pairs. The pair minus the sum of its parts is the
# interaction term, which is the number that says whether two units overlap
# or merely compete.
say "attribution: single units"
single() { # flag label
  sudo $R --op powervirus --mix-tensor 0 --mix-fma 0 --mix-dram 0 $1 \
    --precision fp16 --size 8192 \
    --warmup-sec 4 --steady-sec 20 --sample-hz 50 --raise-power-limit \
    --out-summary /tmp/batch3/rung-unit-$2.json \
    --out-metrics /tmp/batch3/rung-unit-$2.ndjson >/dev/null 2>&1
  rep unit-$2
}
single "--mix-fma 1"                       fma
single "--mix-dram 1"                      dram
single "--mix-l2 1"                        l2
single "--mix-sfu 1"                       sfu
single "--mix-int32 1"                     int32
single "--mix-smem 1"                      smem
single "--mix-atomic 1"                    atomic
single "--mix-tensor 1"                    tensor-wmma

say "attribution: pairs (for the interaction terms)"
single "--mix-fma 1 --mix-int32 1"         fma+int32
single "--mix-fma 1 --mix-sfu 1"           fma+sfu
single "--mix-fma 1 --mix-dram 1"          fma+dram
single "--mix-dram 1 --mix-l2 1"           dram+l2
single "--mix-fma 1 --mix-smem 1"          fma+smem
single "--mix-fma 1 --mix-int32 1 --mix-sfu 1 --mix-smem 1" four-way

# ---- 4. OVERSHOOT AT A LOWERED CAP --------------------------------------
# At 1100 W the peak reached 98.2% and no violation was recorded. Against a
# 500 W cap the controller has to fight much harder, so an over-limit
# excursion should be far easier to observe - if one is observable at all.
say "overshoot against a lowered cap"
over() { # cap on_ms off_ms label
  sudo $R --op powervirus --mix-tensor 1 --mix-fma 0 --mix-dram 0 \
    --tensor-backend cublas --precision bf16 --size 8192 \
    --power-limit $1 --duty-on-ms $2 --duty-off-ms $3 \
    --warmup-sec 4 --steady-sec 40 --sample-hz 100 \
    --out-summary /tmp/batch3/rung-$4.json \
    --out-metrics /tmp/batch3/rung-$4.ndjson >/dev/null 2>&1
  python3 -c "
import json
rows=[json.loads(l) for l in open('/tmp/batch3/rung-$4.ndjson') if '\"steady\"' in l]
pw=[r['power_w'] for r in rows]
pi=[r['power_instant_w'] for r in rows if r.get('power_instant_w',-1)>=0]
peak=max(max(pw), max(pi) if pi else 0)
print('  %-20s cap %4d W  peak %7.1f W  = %6.1f%% of cap  %s'%(
 '$4', $1, peak, 100*peak/$1, 'OVER LIMIT' if peak > $1 else ''))" 2>/dev/null
}
over 500 5    5    over-500w-5ms
over 500 50   50   over-500w-50ms
over 500 500  500  over-500w-500ms
over 700 20   20   over-700w-20ms

# ---- 5. RAMP SHAPES + LOCKED CLOCKS -------------------------------------
# How much of the transient is inherent, and how much is just the square
# edge? And a locked-clock run so there is one rung with DVFS removed.
say "ramp shapes"
for SHAPE in none linear exp; do
  sudo $R --op powervirus --mix-tensor 1 --mix-fma 0 --mix-dram 0 \
    --tensor-backend cublas --precision bf16 --size 8192 \
    --duty-on-ms 200 --duty-off-ms 200 --duty-ramp $SHAPE \
    --warmup-sec 4 --steady-sec 30 --sample-hz 100 --raise-power-limit \
    --out-summary /tmp/batch3/rung-ramp-$SHAPE.json \
    --out-metrics /tmp/batch3/rung-ramp-$SHAPE.ndjson >/dev/null 2>&1
  python3 -c "
import json
rows=[json.loads(l) for l in open('/tmp/batch3/rung-ramp-$SHAPE.ndjson') if '\"steady\"' in l]
pi=[r['power_instant_w'] for r in rows if r.get('power_instant_w',-1)>=0] or [r['power_w'] for r in rows]
# crude slew: largest sample-to-sample delta, at 100 Hz
sl=max(abs(pi[i+1]-pi[i]) for i in range(len(pi)-1))
print('  ramp %-7s swing %6.1f W  max step %6.1f W/sample'%('$SHAPE', max(pi)-min(pi), sl))" 2>/dev/null
done

say "locked clocks (boost-off, reproducibility)"
sudo $R --op powervirus --mix-tensor 1 --mix-fma 0 --mix-dram 0 \
  --tensor-backend cublas --precision bf16 --size 8192 --lock-clocks \
  --warmup-sec 4 --steady-sec 20 --sample-hz 50 --raise-power-limit \
  --out-summary /tmp/batch3/rung-locked-clocks.json \
  --out-metrics /tmp/batch3/rung-locked-clocks.ndjson >/dev/null 2>&1
rep locked-clocks

say "ALL DONE"
du -sh /tmp/batch3
REMOTE

echo "==> launching (detached)"
scp -q -o StrictHostKeyChecking=accept-new "${SSHA[@]}" /tmp/gpl-batch3.sh "$HOST:/tmp/"
"${SSH[@]}" 'nohup bash /tmp/gpl-batch3.sh > /tmp/batch3.log 2>&1 & echo "started pid $!"'
echo
echo "Follow:   ssh ${SSHA[*]} $HOST 'tail -f /tmp/batch3.log'"
echo "Collect:  $0 --collect $HOST ${SSHA[*]}"
