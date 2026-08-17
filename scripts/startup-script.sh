#!/usr/bin/env bash
#
# gpu-power-lab startup script.
#
# Paste this into the provider's "startup script" / user-data / cloud-init
# runcmd field when launching a box. It installs the driver and toolchain
# during provisioning, so the instance is ready to measure the moment SSH
# answers instead of burning 8-10 minutes of paid GPU time on an apt download
# while you watch.
#
# Idempotent: safe to re-run, skips anything already present.
#
# When it finishes it touches /var/log/gpu-power-lab-ready. Poll for that
# rather than for SSH — SSH answers well before the driver is usable, and a
# batch that starts too early fails on the first nvidia-smi.

set -x
exec > >(tee -a /var/log/gpu-power-lab-setup.log) 2>&1
echo "gpu-power-lab setup starting: $(date -u)"

export DEBIAN_FRONTEND=noninteractive

# --- toolchain ------------------------------------------------------------
apt-get update -qq
apt-get install -y -qq wget curl git cmake build-essential python3 python3-pip rsync

# --- NVIDIA driver + CUDA -------------------------------------------------
# Datacenter Blackwell needs the open kernel modules; nvidia-open provides
# them. CUDA 13.x is required for the sm_103a target — a 12.x toolkit
# compiles happily and silently omits the tcgen05 path the power virus
# depends on, which is a very expensive thing to discover on a rented B300.
if ! command -v nvidia-smi >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
    cd /tmp
    . /etc/os-release
    case "$VERSION_ID" in
        24.04) REPO=ubuntu2404 ;;
        22.04) REPO=ubuntu2204 ;;
        *)     REPO=ubuntu2204 ;;
    esac
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y -qq nvidia-open cuda-toolkit-13-0
fi

echo 'export PATH=/usr/local/cuda/bin:$PATH' > /etc/profile.d/cuda.sh
chmod 644 /etc/profile.d/cuda.sh

# --- measurement hygiene --------------------------------------------------
# Persistence mode: without it the driver tears down state between NVML
# clients, adding ~100 ms to every nvmlInit and letting clocks settle
# differently between rungs.
nvidia-smi -pm 1 || true

# Record the as-found state before anything of ours runs. If a later result
# looks odd, this says what the box looked like untouched.
nvidia-smi -q > /var/log/gpu-power-lab-nvidia-asfound.txt 2>&1 || true
nvcc --version > /var/log/gpu-power-lab-nvcc.txt 2>&1 || \
  /usr/local/cuda/bin/nvcc --version > /var/log/gpu-power-lab-nvcc.txt 2>&1 || true

# --- readiness marker -----------------------------------------------------
# Only set once the GPU actually answers. SSH being up proves nothing.
if nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi -L > /var/log/gpu-power-lab-ready
    echo "READY: $(date -u)" >> /var/log/gpu-power-lab-ready
else
    echo "FAILED: driver not responding after install" > /var/log/gpu-power-lab-failed
fi

echo "gpu-power-lab setup finished: $(date -u)"
