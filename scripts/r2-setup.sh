#!/usr/bin/env bash
#
# R2 setup — everything needed before an inference measurement can run.
#
#   ./scripts/r2-setup.sh ubuntu@<host> [-i key]           # default model
#   GPL_MODEL=Qwen/Qwen2.5-32B-Instruct ./scripts/r2-setup.sh ubuntu@<host> -i key
#
# Idempotent: skips anything already present, so it is safe to re-run after a
# spot reclamation or a failed attempt. Everything runs detached, because the
# model download is long enough that an SSH drop would otherwise kill it.
#
# Model choice matters for this measurement in a way it does not for training.
# Decode is memory-bound: every generated token requires reading the entire
# weight set from HBM, so power during decode scales with parameter count.
# A small model on a 275 GB card leaves the memory system idle regardless of
# how much KV cache vLLM allocates, and would understate decode power.
# Default is a 72B in bf16, roughly 145 GB of weights.

set -euo pipefail

HOST="${1:?usage: $0 user@host [ssh args...]}"; shift
SSHA=("$@")
SSH=(ssh -o StrictHostKeyChecking=accept-new "${SSHA[@]}" "$HOST")
MODEL="${GPL_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
MODEL_DIR="${GPL_MODEL_DIR:-/tmp/qwen72b}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> syncing scripts"
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new ${SSHA[*]}" \
      "$REPO/scripts/r2_inference.py" "$HOST:~/gpu-power-lab/scripts/"

echo "==> setup (detached; follow /tmp/r2-setup.log)"
"${SSH[@]}" "MODEL='$MODEL' MODEL_DIR='$MODEL_DIR' nohup bash -s > /tmp/r2-setup.log 2>&1 &" <<'REMOTE'
set -u
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

say "python deps"
python3 -m pip install --quiet --upgrade pip 2>&1 | tail -1
# hf_transfer is worth the extra dependency: it roughly doubles throughput on
# a multi-file download of this size.
python3 -m pip install --quiet vllm aiohttp huggingface_hub hf_transfer 2>&1 | tail -2
python3 -c "import vllm, aiohttp; print('vllm', vllm.__version__)"

say "model: $MODEL -> $MODEL_DIR"
if [ -f "$MODEL_DIR/config.json" ] && [ "$(du -s "$MODEL_DIR" | cut -f1)" -gt 1000000 ]; then
    say "  already present ($(du -sh "$MODEL_DIR" | cut -f1)), skipping"
else
    export HF_HUB_ENABLE_HF_TRANSFER=1
    hf download "$MODEL" --local-dir "$MODEL_DIR" 2>&1 | tail -2
    say "  downloaded $(du -sh "$MODEL_DIR" | cut -f1)"
fi

df -h / | tail -1
nvidia-smi --query-gpu=memory.total,memory.used,power.draw --format=csv,noheader
say "R2_SETUP_READY"
REMOTE

echo
echo "Follow:  ssh ${SSHA[*]} $HOST 'tail -f /tmp/r2-setup.log'"
echo "Then:    ./scripts/r2-run.sh $HOST ${SSHA[*]}"
