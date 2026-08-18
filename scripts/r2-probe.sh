#!/usr/bin/env bash
#
# R2 probe — find a vLLM configuration that fills the card, and record it.
#
#   ./scripts/r2-probe.sh ubuntu@<host> [-i key]
#
# THIS IS CONFIGURATION DISCOVERY, NOT MEASUREMENT.
#
# It starts vLLM under several configurations, records what each one actually
# allocates, and stops. Any power figure taken during a probe is worthless:
# the box may be running other work, the server is starting and stopping, and
# the load is not controlled. The measurement run (r2-run.sh) happens
# afterwards on a quiescent machine.
#
# Safe to run alongside CPU-bound work. It touches the GPU but only to
# allocate and report, and nothing here depends on timing.
#
# What we are trying to establish:
#   1. The largest --max-model-len the model and card will accept.
#   2. How much KV cache that leaves, in tokens and in concurrent sequences.
#   3. Whether fp8 KV cache is supported, which would roughly double capacity.
#   4. A configuration that leaves the memory system genuinely loaded rather
#      than mostly allocated-but-idle.

set -uo pipefail
HOST="${1:?usage: $0 user@host [ssh args...]}"; shift
SSHA=("$@")
MODEL_DIR="${GPL_MODEL_DIR:-/tmp/qwen72b}"

ssh -o StrictHostKeyChecking=accept-new "${SSHA[@]}" "$HOST" \
    "MODEL_DIR='$MODEL_DIR' bash -s" <<'REMOTE'
set -u
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
# vLLM runs under a Python 3.12 venv; see r2-setup.sh for why.
VPY="${GPL_VLLM_PYTHON:-$HOME/vllm-venv/bin/python}"
mkdir -p /tmp/r2probe
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

say "model config"
python3 - <<'PY'
import json, os
d = os.environ.get("MODEL_DIR", "/tmp/qwen72b")
c = json.load(open(f"{d}/config.json"))
print(f"  architecture      : {c.get('architectures')}")
print(f"  hidden / layers   : {c.get('hidden_size')} / {c.get('num_hidden_layers')}")
print(f"  attn heads / kv   : {c.get('num_attention_heads')} / {c.get('num_key_value_heads')}")
print(f"  native ctx        : {c.get('max_position_embeddings')}")
print(f"  rope scaling      : {c.get('rope_scaling')}")
print(f"  dtype             : {c.get('torch_dtype')}")
# KV bytes per token = 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes
hd = c['hidden_size'] // c['num_attention_heads']
for name, b in (("bf16", 2), ("fp8", 1)):
    per_tok = 2 * c['num_hidden_layers'] * c['num_key_value_heads'] * hd * b
    print(f"  KV per token {name:4}: {per_tok/1024:.1f} KiB  "
          f"({per_tok/1e6:.3f} MB) -> 100 GB holds {100e9/per_tok/1e6:.2f} M tokens")
PY

probe() {  # label extra-args...
    local label="$1"; shift
    say "probe: $label"
    timeout 900 "$VPY" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" --served-model-name probe \
        "$@" \
        > "/tmp/r2probe/$label.log" 2>&1 &
    local pid=$!
    local ok=0
    for i in $(seq 1 90); do
        if curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then ok=1; break; fi
        kill -0 $pid 2>/dev/null || break
        sleep 10
    done
    if [ "$ok" = "1" ]; then
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader \
            | sed 's/^/    GPU memory: /'
        grep -oE "GPU KV cache size: [0-9,]+ tokens|# GPU blocks: [0-9]+|Maximum concurrency for [0-9]+ tokens per request: [0-9.]+x" \
            "/tmp/r2probe/$label.log" | sort -u | sed 's/^/    /'
    else
        echo "    FAILED to start. Cause:"
        grep -iE "error|ValueError|out of memory|no available memory|must be|does not support" \
            "/tmp/r2probe/$label.log" | tail -3 | sed 's/^/      /'
    fi
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    sleep 15   # let memory actually free before the next probe
}

# Each probe reloads 136 GB of weights, so the set is kept to the configs
# that answer a distinct question. Dropped: low-allocation and short-context
# baselines, which only confirm that a smaller ask also fits.

# Native context at high allocation — the reference configuration.
probe ctx32k          --gpu-memory-utilization 0.95 --max-model-len 32768

# fp8 KV: halves bytes per token, so roughly doubles cache capacity. The
# single largest lever on how much of the card the KV cache occupies.
probe fp8kv-ctx32k    --gpu-memory-utilization 0.95 --max-model-len 32768 \
                      --kv-cache-dtype fp8

# Beyond native context requires rope scaling; expected to fail without it,
# and the failure message is the useful output.
probe ctx131k         --gpu-memory-utilization 0.95 --max-model-len 131072 \
                      --kv-cache-dtype fp8

# High concurrency with chunked prefill: the configuration a saturated
# server would actually run.
probe conc-chunked    --gpu-memory-utilization 0.95 --max-model-len 16384 \
                      --kv-cache-dtype fp8 --max-num-seqs 256 \
                      --enable-chunked-prefill --max-num-batched-tokens 8192

say "PROBE DONE — logs in /tmp/r2probe/"
REMOTE
