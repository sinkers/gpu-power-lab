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
# Write the remote body to a file and nohup THAT. Feeding it to
# `nohup bash -s` over stdin looks equivalent but is not: when the SSH
# channel closes, bash -s stops reading and the job dies, leaving an
# empty log and no process. nohup protects against SIGHUP, not against
# stdin disappearing.
cat > /tmp/gpl-r2-setup.sh <<'REMOTE'
set -u
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# vLLM 0.27 pulls in flashinfer, which uses `array.array[int]` subscript
# syntax that Ubuntu 22.04's system Python 3.10 cannot parse:
#   TypeError: 'type' object is not subscriptable
# The engine dies during initialisation with a traceback that points at
# vllm/compilation, not at the Python version, so it is worth stating the
# cause here. Fix is a newer interpreter, not a vLLM downgrade.
#
# uv fetches a standalone CPython rather than requiring a PPA, which keeps
# this reproducible on a fresh box with no extra apt sources.
say "python 3.12 toolchain"
if [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -1
fi
export PATH=$HOME/.local/bin:$PATH

VENV="$HOME/vllm-venv"
if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python 3.12 "$VENV" 2>&1 | tail -2
fi
"$VENV/bin/python" --version

say "vllm into $VENV"
if ! "$VENV/bin/python" -c "import vllm" 2>/dev/null; then
    uv pip install --python "$VENV/bin/python" --quiet \
        vllm aiohttp huggingface_hub hf_transfer 2>&1 | tail -3
fi
"$VENV/bin/python" -c "import vllm, aiohttp; print('vllm', vllm.__version__)"

# The model download only needs huggingface_hub, so keep it on system python
# to avoid re-downloading if the venv is ever rebuilt.
python3 -m pip install --quiet huggingface_hub hf_transfer 2>&1 | tail -1

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

scp -q -o StrictHostKeyChecking=accept-new "${SSHA[@]}" /tmp/gpl-r2-setup.sh "$HOST:/tmp/"
"${SSH[@]}" "MODEL='$MODEL' MODEL_DIR='$MODEL_DIR' nohup bash /tmp/gpl-r2-setup.sh > /tmp/r2-setup.log 2>&1 & echo started pid \$!"

echo
echo "Follow:  ssh ${SSHA[*]} $HOST 'tail -f /tmp/r2-setup.log'"
echo "Then:    ./scripts/r2-run.sh $HOST ${SSHA[*]}"
