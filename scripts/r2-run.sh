#!/usr/bin/env bash
#
# R2 — inference power measurement against a vLLM server.
#
#   ./scripts/r2-run.sh ubuntu@<host> [-i key]
#
# Requires r2-setup.sh to have completed. Starts vLLM, waits for it to serve,
# then runs three load phases with `--op observe` sampling power alongside
# each, and finally an idle-with-model-resident reading.
#
# THE BOX MUST BE OTHERWISE QUIESCENT. The load generator runs on the same
# host, so competing CPU work distorts both the achieved request rate and the
# sampler thread's scheduling. Anything else running invalidates the run.
#
# --gpu-memory-utilization 0.95 is deliberate: vLLM expands the KV cache to
# fill whatever fraction it is given, so this puts the memory system in the
# state a saturated server would actually be in rather than leaving most of
# the card allocated but unused.

set -euo pipefail
HOST="${1:?usage: $0 user@host [ssh args...]}"; shift
SSHA=("$@")
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${GPL_MODEL_DIR:-/tmp/qwen72b}"

rsync -az -e "ssh -o StrictHostKeyChecking=accept-new ${SSHA[*]}" \
      "$REPO/scripts/r2_inference.py" "$HOST:~/gpu-power-lab/scripts/"

ssh -o StrictHostKeyChecking=accept-new "${SSHA[@]}" "$HOST" \
    "MODEL_DIR='$MODEL_DIR' nohup bash -s > /tmp/r2.log 2>&1 &" <<'REMOTE'
set -u
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
cd ~/gpu-power-lab
mkdir -p /tmp/r2
R=./runner/build/gpu-power-runner
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

say "checking the box is quiet before starting"
LOAD=$(cut -d' ' -f1 /proc/loadavg)
say "  1-minute load average: $LOAD (competing CPU work will distort this run)"

say "starting vLLM on $MODEL_DIR"
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name qwen72b \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16384 \
    --disable-log-requests \
    > /tmp/r2/vllm-server.log 2>&1 &
VPID=$!

say "waiting for the server (model load takes several minutes at this size)"
for i in $(seq 1 120); do
    if curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        say "  serving after ${i}0s"; break
    fi
    kill -0 $VPID 2>/dev/null || { say "  server died:"; tail -20 /tmp/r2/vllm-server.log; exit 1; }
    sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 || {
    say "  server never became ready"; tail -20 /tmp/r2/vllm-server.log; exit 1; }

nvidia-smi --query-gpu=memory.used,memory.total,power.draw --format=csv,noheader \
    | sed 's/^/  KV+weights resident: /'

# Idle with the model loaded: the correct baseline for a serving deployment,
# and distinct from the 183.5 W empty-device idle measured earlier.
say "idle, model resident"
sudo $R --op observe --steady-sec 45 --sample-hz 100 \
    --out-summary /tmp/r2/rung-infer-idle.json \
    --out-metrics /tmp/r2/rung-infer-idle.ndjson >/dev/null 2>&1

phase() {  # name concurrency in out seconds
    say "phase: $1 (concurrency $2, ${3} in / ${4} out)"
    rm -f /tmp/r2/$1.ready
    python3 scripts/r2_inference.py --phase "$1" --concurrency "$2" \
        --input-tokens "$3" --output-tokens "$4" --seconds "$5" --warmup 25 \
        --ready-file /tmp/r2/$1.ready --out /tmp/r2/work-$1.json \
        > /tmp/r2/$1.stdout 2>/tmp/r2/$1.stderr &
    WPID=$!
    for i in $(seq 1 180); do
        [ -f /tmp/r2/$1.ready ] && break
        kill -0 $WPID 2>/dev/null || break
        sleep 1
    done
    if [ ! -f /tmp/r2/$1.ready ]; then
        say "  $1 failed to reach steady state:"; tail -3 /tmp/r2/$1.stderr
        wait $WPID 2>/dev/null; return
    fi
    sudo $R --op observe --steady-sec "$5" --sample-hz 100 \
        --out-summary /tmp/r2/rung-infer-$1.json \
        --out-metrics /tmp/r2/rung-infer-$1.ndjson >/dev/null 2>&1
    wait $WPID 2>/dev/null
    python3 -c "
import json
p=json.load(open('/tmp/r2/rung-infer-$1.json')); pw=p['power']
try: w=json.load(open('/tmp/r2/work-$1.json'))
except Exception: w={}
print('  %-9s %7.1fW  %%lim %5.1f  %3.0fC  %6.2f req/s  %8.1f out tok/s  %8.1f total tok/s  p95 %.2fs'%(
 '$1', pw['avg_w'], pw['pct_of_enforced_limit'], p['thermal']['peak_c'],
 w.get('req_per_s',0), w.get('output_tok_per_s',0), w.get('total_tok_per_s',0),
 w.get('latency_p95_s') or 0))" 2>/dev/null || say "  $1: no summary"
}

# Prefill-dominated: long prompts, minimal generation. Compute-bound.
phase prefill  32  8192    8  60
# Decode-dominated: short prompts, long generation. Memory-bound; this is the
# phase a large model is required to represent honestly.
phase decode   96   128 2048  60
# A plausible serving mix, for reference.
phase balanced 64  1024  256  60

say "stopping vLLM"
kill $VPID 2>/dev/null || true
sleep 10
say "R2 DONE"
REMOTE

echo
echo "Follow:   ssh ${SSHA[*]} $HOST 'tail -f /tmp/r2.log'"
echo "Collect:  scp ${SSHA[*]} '$HOST:/tmp/r2/*.json' '$HOST:/tmp/r2/*.ndjson' results/t2-b300-r2/"
